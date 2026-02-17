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

// sirius
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/span.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>

// cuda runtime
#include <cuda_runtime_api.h>

#include <driver_types.h>

namespace sirius {

namespace detail {

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_gpu(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // Source stuff
  auto& host_src          = source.cast<host_parquet_representation>();
  auto const& byte_ranges = host_src.get_column_chunk_byte_ranges();
  auto& reader            = host_src.get_parquet_reader();

  // Target stuff
  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  auto const& allocation = host_src.get_column_chunks();

  // Allocate individual device buffers per column chunk byte range
  std::vector<rmm::device_buffer> column_chunk_buffers;
  column_chunk_buffers.reserve(byte_ranges.size());
  for (auto const& byte_range : byte_ranges) {
    column_chunk_buffers.emplace_back(byte_range.size(), stream, mr_ref);
  }

  // Build a mapping from global byte offset to (buffer_index, local_offset) for the copy.
  // Each column_chunk_buffer corresponds to one byte_range, laid out contiguously in the host
  // allocation. We iterate over host allocation blocks and map into the correct device buffer.
  std::vector<size_t> range_offsets;  // cumulative start offset per byte range
  range_offsets.reserve(byte_ranges.size() + 1);
  range_offsets.push_back(0);
  for (auto const& br : byte_ranges) {
    range_offsets.push_back(range_offsets.back() + br.size());
  }

  // Copy HOST data to GPU with a single async batch copy.
  size_t bytes_copied = 0;
  std::vector<void*> dst_ptrs;
  std::vector<void*> src_ptrs;
  std::vector<size_t> counts;
  while (bytes_copied < host_src.get_size_in_bytes()) {
    auto const& block        = allocation->at(bytes_copied / allocation->block_size());
    auto const block_offset  = bytes_copied % allocation->block_size();
    auto const bytes_to_copy = std::min(allocation->block_size() - block_offset,
                                        host_src.get_size_in_bytes() - bytes_copied);

    // Find which device buffer this global offset falls into
    // Upper bound gives the first range whose start is > bytes_copied; subtract 1 for the range.
    auto it        = std::upper_bound(range_offsets.begin(), range_offsets.end(), bytes_copied);
    auto buf_idx   = static_cast<size_t>(std::distance(range_offsets.begin(), it) - 1);
    auto local_off = bytes_copied - range_offsets[buf_idx];
    dst_ptrs.push_back(
      static_cast<void*>(static_cast<uint8_t*>(column_chunk_buffers[buf_idx].data()) + local_off));
    src_ptrs.push_back(const_cast<void*>(
      static_cast<void const*>(reinterpret_cast<uint8_t const*>(block.data() + block_offset))));
    counts.push_back(bytes_to_copy);
    bytes_copied += bytes_to_copy;
  }

// Try to do batch copy if cudaMemcpyBatchAsync is possible, otherwise fall back to individual
// async copies
#if CUDART_VERSION >= 13000
  cudaStream_t stream_handle = (stream.value() != nullptr && stream.value() != cudaStreamLegacy)
                                 ? stream.value()
                                 : cudaStreamPerThread;
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.srcLocHint     = {cudaMemLocationTypeHost,
                         host_src.get_device_id()};  // this is numa node id for pinned host
  attr.dstLocHint     = {cudaMemLocationTypeDevice, target_memory_space->get_device_id()};
  attr.flags          = cudaMemcpyFlagDefault;
  RMM_CUDA_TRY(::cudaMemcpyBatchAsync(
    dst_ptrs.data(), src_ptrs.data(), counts.data(), counts.size(), attr, stream_handle));
  RMM_CUDA_TRY(::cudaStreamSynchronize(stream_handle));
#else
  for (size_t i = 0; i < dst_ptrs.size(); ++i) {
    RMM_CUDA_TRY(::cudaMemcpyAsync(
      dst_ptrs[i], src_ptrs[i], counts[i], cudaMemcpyHostToDevice, stream.value()));
  }
#endif

  // Invoke the Parquet reader to materialize the table on GPU
  auto result = reader.materialize_all_columns(
    host_src.get_rg_span(), std::move(column_chunk_buffers), host_src.get_reader_options(), stream);
  auto new_table = std::move(result.tbl);  // Discard metadata
  stream.synchronize();

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(new_table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_parquet_representation to host_parquet_representation (cross-host copy)
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_host_parquet(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /* stream */)
{
  auto& host_src       = source.cast<host_parquet_representation>();
  auto const data_size = host_src.get_size_in_bytes();

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto* mr = target_memory_space
               ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }

  auto const& src_allocation  = host_src.get_column_chunks();
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_block_index      = 0;
  size_t dst_block_offset     = 0;
  size_t const src_block_size = src_allocation->block_size();
  size_t const dst_block_size = dst_allocation->block_size();
  size_t copied               = 0;
  while (copied < data_size) {
    size_t remaining     = data_size - copied;
    size_t src_avail     = src_block_size - src_block_offset;
    size_t dst_avail     = dst_block_size - dst_block_offset;
    size_t bytes_to_copy = std::min({remaining, src_avail, dst_avail});
    auto* src_ptr        = src_allocation->at(src_block_index).data() + src_block_offset;
    auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
    std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    copied += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    dst_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
    if (dst_block_offset == dst_block_size) {
      dst_block_index++;
      dst_block_offset = 0;
    }
  }
  return std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(host_src.move_parquet_reader()),
    host_src.get_reader_options(),
    std::move(host_src.get_row_group_indices()),
    std::move(host_src.get_column_chunk_byte_ranges()),
    data_size,
    host_src.get_uncompressed_size_in_bytes());
}

}  // namespace detail

void register_parquet_converters(cucascade::representation_converter_registry& registry)
{
  // HOST Parquet -> GPU
  if (!registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_parquet_representation, cucascade::gpu_table_representation>(
      detail::convert_host_parquet_to_gpu);
  }

  // HOST Parquet -> HOST Parquet (cross-host copy)
  if (!registry.has_converter<host_parquet_representation, host_parquet_representation>()) {
    registry.register_converter<host_parquet_representation, host_parquet_representation>(
      detail::convert_host_parquet_to_host_parquet);
  }
}

}  // namespace sirius
