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
#include <cucascade/data/gpu_data_representation.hpp>
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
 *   4. Use the TARGET-bound stream for read_parquet + apply_post_convert +
 *      apply_partition_inject + final sync (never the caller's stream).
 *   5. Consume sticky cuda errors before returning.
 */
std::unique_ptr<cucascade::idata_representation>
convert_host_parquet_to_gpu_with_prefetched_data_source(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // STRUCTURAL GAP (io rebase): sirius::op::scan::prefetched_data_source — the
  // in-memory cudf::io::datasource adapter that wrapped a cache_ranges and fed
  // cudf::io::read_parquet from prefetched host blocks — was removed
  // (prefetched_data_source.cpp deleted, the type is no longer declared) with no
  // replacement. The cached-host-parquet -> GPU conversion below cannot be
  // reconstructed until that adapter is reimplemented, so this path is stubbed
  // to throw and the dependent body is disabled.
  (void)source;
  (void)target_memory_space;
  (void)stream;
  throw std::runtime_error(
    "convert_host_parquet_to_gpu_with_prefetched_data_source: prefetched_data_source "
    "adapter was removed in the io rebase; cached-host-parquet GPU conversion is "
    "unavailable until it is reimplemented.");
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
  if (host_src.has_post_convert_fn()) { dst->set_post_convert_fn(host_src.get_post_convert_fn()); }
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
