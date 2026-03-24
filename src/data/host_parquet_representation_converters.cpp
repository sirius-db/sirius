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
#include <op/scan/cached_ranges.hpp>
#include <op/scan/prefetched_data_source.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include "cudf/cudf_utils.hpp"

#include <cudf/utilities/span.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace detail {

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 */
std::unique_ptr<cucascade::idata_representation>
convert_host_parquet_to_gpu_with_prefetched_data_source(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_src = source.cast<host_parquet_representation>();

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  // Build a cache_ranges from the packed host blocks and the column-chunk byte-range descriptors.
  // The block pointers are raw std::byte* owned by the allocation; cache_ranges does not take
  // ownership of them (the allocation still lives for the duration of this call).
  auto const& allocation = host_src.get_column_chunks();
  auto block_ptrs =
    std::vector<std::byte*>(allocation->get_blocks().begin(), allocation->get_blocks().end());

  auto ranges =
    std::make_unique<sirius::op::scan::cache_ranges>(host_src.get_column_chunk_byte_ranges(),
                                                     std::move(block_ptrs),
                                                     allocation->block_size(),
                                                     target_memory_space->get_device_id(),
                                                     host_src.get_device_id());  // NUMA node id

  auto data_source = std::make_unique<sirius::op::scan::prefetched_data_source>(
    std::move(ranges), host_src.get_file_size(), host_src.get_fallback_datasource());

  // Point the reader options at our in-memory datasource and call cudf::io::read_parquet.
  auto opts = host_src.get_reader_options();
  opts.set_source(cudf::io::source_info{data_source.get()});
  // set_row_groups expects one inner vector per source; we have a single source.
  opts.set_row_groups({std::vector<cudf::size_type>(host_src.get_row_group_indices().begin(),
                                                    host_src.get_row_group_indices().end())});

  auto result = cudf::io::read_parquet(opts, stream, mr_ref);
  stream.synchronize();

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(result.tbl), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
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
  return std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(cloned_reader),
    host_src.get_reader_options(),
    std::move(host_src.get_row_group_indices()),
    std::move(host_src.get_column_chunk_byte_ranges()),
    data_size,
    host_src.get_uncompressed_size_in_bytes(),
    host_src.get_file_size(),
    host_src.get_fallback_datasource());
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
