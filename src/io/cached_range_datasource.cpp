/*
 * Copyright 2026, Sirius Contributors.
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

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <io/cached_range_datasource.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sirius::io {

cached_range_datasource::cached_range_datasource(std::shared_ptr<op::scan::cache_ranges> ranges,
                                                 std::shared_ptr<sirius_datasource> fallback)
  : _ranges(std::move(ranges)), _fallback(std::move(fallback))
{
}

size_t cached_range_datasource::size() const { return _fallback->size(); }

size_t cached_range_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  // Serve from pinned memory when the whole read is cached; otherwise the range
  // was never pinned (footer, page index, or an unpinned column) — read it from
  // the file through the fallback datasource.
  if (auto spans = _ranges->get_ranges(offset, size)) {
    size_t written = 0;
    for (auto const& span : *spans) {
      std::memcpy(dst + written, span.data(), span.size());
      written += span.size();
    }
    return written;
  }
  SIRIUS_LOG_WARN(
    "[cached_range_datasource] host_read FALLBACK to disk off={} size={}", offset, size);
  return _fallback->host_read(offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> cached_range_datasource::host_read(size_t offset,
                                                                                 size_t size)
{
  std::vector<uint8_t> buf(size);
  auto const n = host_read(offset, size, buf.data());
  buf.resize(n);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

size_t cached_range_datasource::device_read(size_t offset,
                                            size_t size,
                                            uint8_t* dst,
                                            rmm::cuda_stream_view stream)
{
  auto spans = _ranges->get_ranges(offset, size);
  if (!spans) {
    SIRIUS_LOG_WARN(
      "[cached_range_datasource] device_read FALLBACK to disk off={} size={}", offset, size);
    return _fallback->device_read(offset, size, dst, stream);
  }
  // The cached spans live in CUDA-pinned host memory, so the H2D copies are
  // truly async on the stream. Synchronize before returning: the sync device_read
  // contract is that the bytes are resident in dst on return.
  size_t written = 0;
  for (auto const& span : *spans) {
    cudaMemcpyAsync(
      dst + written, span.data(), span.size(), cudaMemcpyHostToDevice, stream.value());
    written += span.size();
  }
  stream.synchronize();
  return written;
}

std::unique_ptr<cudf::io::datasource::buffer> cached_range_datasource::device_read(
  size_t offset, size_t size, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buf(size, stream);
  auto n = device_read(offset, size, reinterpret_cast<uint8_t*>(buf.data()), stream);
  n      = std::min(n, size);
  buf.resize(n, stream);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> cached_range_datasource::device_read_async(size_t offset,
                                                               size_t size,
                                                               uint8_t* dst,
                                                               rmm::cuda_stream_view stream)
{
  auto spans = _ranges->get_ranges(offset, size);
  if (!spans) {
    SIRIUS_LOG_WARN(
      "[cached_range_datasource] device_read_async FALLBACK to disk off={} size={}", offset, size);
    return _fallback->device_read_async(offset, size, dst, stream);
  }
  // Enqueue the H2D copies on the stream and return the count immediately; the
  // copies are stream-ordered against the decode that consumes them, so no host
  // sync is needed here (matching a device-read datasource's async contract).
  size_t written = 0;
  for (auto const& span : *spans) {
    cudaMemcpyAsync(
      dst + written, span.data(), span.size(), cudaMemcpyHostToDevice, stream.value());
    written += span.size();
  }
  std::promise<size_t> p;
  p.set_value(written);
  return p.get_future();
}

}  // namespace sirius::io
