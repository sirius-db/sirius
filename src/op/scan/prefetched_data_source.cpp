/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op/scan/prefetched_data_source.hpp"

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <driver_types.h>

#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace sirius::op::scan {

prefetched_data_source::prefetched_data_source(
  std::unique_ptr<cache_ranges> ranges, std::shared_ptr<cudf::io::datasource> fallback_source)
  : ranges_(std::move(ranges)), fallback_(std::move(fallback_source))
{
}

prefetched_data_source::~prefetched_data_source() = default;

std::unique_ptr<cudf::io::datasource::buffer> prefetched_data_source::host_read(size_t offset,
                                                                                size_t size)
{
  std::vector<uint8_t> buffer(size);
  host_read(offset, size, buffer.data());
  return cudf::io::datasource::buffer::create(std::move(buffer));
}

size_t prefetched_data_source::host_read(size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) return 0;

  try {
    auto spans          = ranges_->get_ranges(offset, size);
    size_t bytes_copied = 0;
    for (auto const& span : spans) {
      std::memcpy(dst + bytes_copied, span.data(), span.size());
      bytes_copied += span.size();
    }
    total_bytes_read_from_cache_.fetch_add(size, std::memory_order_relaxed);
    return bytes_copied;
  } catch (std::out_of_range const&) {
    if (!fallback_) throw;
    size_t bytes_read = fallback_->host_read(offset, size, dst);
    total_bytes_read_from_fallback_.fetch_add(bytes_read, std::memory_order_relaxed);
    return bytes_read;
  }
}

std::unique_ptr<cudf::io::datasource::buffer> prefetched_data_source::device_read(
  size_t offset, size_t size, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buffer(size, stream);
  device_read(offset, size, static_cast<uint8_t*>(buffer.data()), stream);
  return cudf::io::datasource::buffer::create(std::move(buffer));
}

size_t prefetched_data_source::device_read(size_t offset,
                                           size_t size,
                                           uint8_t* dst,
                                           rmm::cuda_stream_view stream)
{
  if (size == 0) return 0;

  std::vector<std::span<const std::byte>> spans;
  try {
    spans = ranges_->get_ranges(offset, size);
  } catch (std::out_of_range const&) {
    if (!fallback_) throw;

    size_t bytes_read = 0;
    // Fallback: let the underlying datasource handle the device read (or host read + copy).
    if (fallback_->supports_device_read()) {
      bytes_read = fallback_->device_read(offset, size, dst, stream);
    } else {
      // Fallback does not support device read; read to host then copy to device.
      auto host_buf = fallback_->host_read(offset, size);
      RMM_CUDA_TRY(::cudaMemcpyAsync(
        dst, host_buf->data(), host_buf->size(), cudaMemcpyHostToDevice, stream.value()));
      bytes_read = host_buf->size();
    }
    total_bytes_read_from_fallback_.fetch_add(bytes_read, std::memory_order_relaxed);
    return bytes_read;
  }

  std::vector<void*> src_ptrs;
  std::vector<void*> dst_ptrs;
  std::vector<size_t> counts;
  src_ptrs.reserve(spans.size());
  dst_ptrs.reserve(spans.size());
  counts.reserve(spans.size());

  size_t bytes_queued = 0;
  for (auto const& span : spans) {
    src_ptrs.push_back(const_cast<void*>(static_cast<void const*>(span.data())));
    dst_ptrs.push_back(static_cast<void*>(dst + bytes_queued));
    counts.push_back(span.size());
    bytes_queued += span.size();
  }

// Batch copy if cudaMemcpyBatchAsync is available (CUDA 13+), otherwise fall back to per-span
// async copies.
#if CUDART_VERSION >= 13000
  bool user_stream           = stream.value() != nullptr && stream.value() != cudaStreamLegacy;
  cudaStream_t stream_handle = user_stream ? stream.value() : cudaStreamPerThread;
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.srcLocHint = {cudaMemLocationTypeHost, (ranges_->numa_id() >= 0) ? ranges_->numa_id() : 0};
  attr.dstLocHint = {cudaMemLocationTypeDevice,
                     (ranges_->device_id() >= 0) ? ranges_->device_id() : 0};
  attr.flags      = cudaMemcpyFlagDefault;
  RMM_CUDA_TRY(::cudaMemcpyBatchAsync(
    dst_ptrs.data(), src_ptrs.data(), counts.data(), counts.size(), attr, stream_handle));
  if (!user_stream) { RMM_CUDA_TRY(::cudaStreamSynchronize(stream_handle)); }
#else
  for (size_t i = 0; i < dst_ptrs.size(); ++i) {
    RMM_CUDA_TRY(::cudaMemcpyAsync(
      dst_ptrs[i], src_ptrs[i], counts[i], cudaMemcpyHostToDevice, stream.value()));
  }
#endif

  total_bytes_read_from_cache_.fetch_add(bytes_queued, std::memory_order_relaxed);
  return bytes_queued;
}

std::future<size_t> prefetched_data_source::device_read_async(size_t offset,
                                                              size_t size,
                                                              uint8_t* dst,
                                                              rmm::cuda_stream_view stream)
{
  return std::async(std::launch::async, [this, offset, size, dst, stream]() {
    return device_read(offset, size, dst, stream);
  });
}

size_t prefetched_data_source::size() const
{
  // If a fallback datasource is present, report its size (it covers the full file).
  if (fallback_) return fallback_->size();
  return ranges_->max_offset();
}

}  // namespace sirius::op::scan
