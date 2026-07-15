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
#include <data/cached_data_representation.hpp>
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>
#include <log/logging.hpp>
#include <op/scan/cached_ranges.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/utilities/span.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// cuda runtime (FIX-02 carryover: cudaGetLastError sticky-state consume)
#include <cuda_runtime_api.h>

// standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace detail {

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 *
 * Multi-GPU stream-correctness pattern: the prior implementation set
 *   rmm::cuda_set_device_raii target_device_raii(target_device_id)
 * but then called `cudf::io::read_parquet(opts, stream, mr_ref)` using the
 * CALLER-supplied `stream`. Under `num_gpus == 2`, the caller's stream may be
 * bound to a non-target device (e.g. the pipeline-executor's GPU-0 stream
 * while `target_device_id == 1`), producing `cudaErrorInvalidValue` inside
 * cudf's internal H2D path. Same root cause as cucascade's built-in
 * convert_host_fast_to_gpu / convert_gpu_to_gpu before they were fixed.
 *
 * Fix pattern:
 *   1. Sync caller's stream so any upstream work on it is flushed.
 *   2. Enter `rmm::cuda_set_device_raii` for the target device.
 *   3. Acquire a target-bound stream from `target_memory_space->acquire_stream()`.
 *   4. Use the TARGET-bound stream for read_parquet + apply_partition_inject +
 *      final sync (never the caller's stream).
 *   5. Consume sticky cuda errors before returning.
 */
std::unique_ptr<cucascade::idata_representation>
convert_host_parquet_to_gpu_with_prefetched_data_source(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // Convert host_parquet_representation to gpu_table_representation.
  // The prior implementation used a deleted prefetched_data_source adapter.
  // Reimplemented: use cudf::io::read_parquet directly from the host buffer
  // via a host_span datasource, then copy the result to the target GPU device.
  auto& host_src       = source.cast<host_parquet_representation>();
  auto const data_size = host_src.get_size_in_bytes();

  // Get the target GPU device and set context
  int target_device_id = target_memory_space ? target_memory_space->get_device_id() : 0;
  rmm::cuda_set_device_raii target_device_raii{rmm::cuda_device_id{target_device_id}};

  // Sync caller's stream so any upstream work is flushed
  stream.synchronize();

  // Get the target memory resource for allocation
  auto* mr = target_memory_space
               ? target_memory_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>()
               : nullptr;
  rmm::device_async_resource_ref alloc = mr ? static_cast<rmm::device_async_resource_ref>(*mr)
                                            : rmm::mr::get_current_device_resource();

  // Get the host buffer as a span and create a datasource
  auto const& host_allocation = host_src.get_column_chunks();
  // Use cudf::io::datasource::from_host_span if available, otherwise from_pointer
  // The host_allocation contains the raw Parquet bytes in pinned host memory
  auto const& blocks = host_allocation->get_blocks();
  if (blocks.empty()) {
    throw std::runtime_error("convert_host_parquet_to_gpu: empty host allocation");
  }

  // Create a contiguous host buffer from the blocks (they may be non-contiguous)
  std::vector<uint8_t> contiguous_host(data_size);
  size_t copied = 0;
  for (auto const& block : blocks) {
    if (copied >= data_size) break;
    size_t to_copy = std::min(static_cast<size_t>(block.size()), data_size - copied);
    std::memcpy(contiguous_host.data() + copied, block.data(), to_copy);
    copied += to_copy;
  }

  // Create a datasource from the contiguous host buffer
  auto datasource = cudf::io::datasource::from_host_span(
    cudf::host_span<uint8_t const>{contiguous_host.data(), data_size});

  // Read the Parquet data from the host buffer into a GPU table
  auto reader_options = host_src.get_reader_options();
  reader_options.set_source(datasource.get());

  // Use the target-bound stream for the read
  auto target_stream = target_memory_space ? target_memory_space->acquire_stream() : stream;

  auto result = cudf::io::read_parquet(reader_options, target_stream, alloc);

  // read_parquet does not throw on all CUDA errors; check for a sticky error
  // and fail loudly rather than constructing a table from a partially-failed read.
  auto const read_err = cudaGetLastError();
  if (read_err != cudaSuccess) {
    throw std::runtime_error(
      std::string("convert_host_parquet_to_gpu: read_parquet failed: ") +
      cudaGetErrorString(read_err));
  }

  auto gpu_table = std::make_unique<cudf::table>(std::move(result.tbl));
  // gpu_table_representation binds a non-const memory_space reference for later
  // stream/memory-resource acquisition. The source space is not mutated through
  // it; the const_cast bridges cuCascade's non-const API to this converter's
  // const-qualified input contract.
  auto dst = std::make_unique<cucascade::gpu_table_representation>(
    std::move(gpu_table),
    const_cast<cucascade::memory::memory_space&>(*target_memory_space),
    target_stream);

  // Partition injection: the host source may carry a partition-inject function
  // that must be preserved across the host→GPU boundary. The GPU table
  // representation does not currently carry partition metadata, so fail loudly
  // rather than silently dropping partitions.
  if (host_src.has_partition_inject_fn()) {
    throw std::runtime_error(
      "convert_host_parquet_to_gpu: source has a partition-inject function but "
      "gpu_table_representation does not carry partition metadata; partitioned "
      "cached-host-parquet GPU conversion is not supported");
  }

  return dst;
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

  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;
  auto cloned_reader       = std::make_unique<hybrid_scan_reader>(
    host_src.get_parquet_reader()->parquet_metadata(), host_src.get_reader_options());
  auto dst = std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(cloned_reader),
    host_src.get_reader_options(),
    host_src.get_row_group_indices(),
    host_src.get_column_chunk_byte_ranges(),
    data_size,
    host_src.get_uncompressed_data_size_in_bytes(),
    host_src.get_file_size(),
    host_src.get_fallback_datasource(),
    host_src.get_filter_expression_by_device(),
    host_src.get_post_filter_projection_ids());
  if (host_src.has_partition_inject_fn()) {
    dst->set_partition_inject_fn(host_src.get_partition_inject_fn());
    dst->set_partition_values(host_src.get_partition_values());
  }
  if (!host_src.get_data_file_path().empty()) {
    dst->set_data_file_path(host_src.get_data_file_path());
  }
  return dst;
}

}  // namespace detail

void register_parquet_converters(cucascade::representation_converter_registry& registry)
{
  // HOST Parquet -> GPU
  if (!registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_parquet_representation, cucascade::gpu_table_representation>(
      detail::convert_host_parquet_to_gpu_with_prefetched_data_source);
  }

  // HOST Parquet -> HOST Parquet (cross-host copy)
  if (!registry.has_converter<host_parquet_representation, host_parquet_representation>()) {
    registry.register_converter<host_parquet_representation, host_parquet_representation>(
      detail::convert_host_parquet_to_host_parquet);
  }

  if (!registry
         .has_converter<cached_host_data_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<cached_host_data_representation, cucascade::gpu_table_representation>(
        [&registry](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
          auto r = source.cast<cached_host_data_representation>().get_representation();
          return registry.convert<cucascade::gpu_table_representation>(
            *r, target_memory_space, stream);
        });
  }

  if (!registry.has_converter<cached_host_parquet_representation,
                              cucascade::gpu_table_representation>()) {
    registry
      .register_converter<cached_host_parquet_representation, cucascade::gpu_table_representation>(
        [&registry](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
          auto r = source.cast<cached_host_parquet_representation>().get_representation();
          return registry.convert<cucascade::gpu_table_representation>(
            *r, target_memory_space, stream);
        });
  }
}

}  // namespace sirius
