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
#include <cudf/cudf_utils.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
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
#include <tuple>

// cuda runtime
#include <cuda_runtime_api.h>

#include <driver_types.h>

namespace sirius {

namespace detail {

std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table)
{
  auto filter_columns  = filter_table->release();
  auto payload_columns = payload_table->release();

  auto all_columns = std::vector<std::unique_ptr<cudf::column>>{};
  all_columns.reserve(filter_columns.size() + payload_columns.size());
  std::move(filter_columns.begin(), filter_columns.end(), std::back_inserter(all_columns));
  std::move(payload_columns.begin(), payload_columns.end(), std::back_inserter(all_columns));
  return std::make_unique<cudf::table>(std::move(all_columns));
}

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_gpu(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  using cudf::io::parquet::experimental::use_data_page_mask;

  // Source stuff
  auto& host_src = source.cast<host_parquet_representation>();
  auto& reader   = host_src.get_parquet_reader();

  // Target stuff
  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  // Allocate a single device buffer and partition it according to the byte ranges
  std::vector<cudf::device_span<uint8_t const>> column_chunk_spans_d;
  std::vector<cudf::device_span<uint8_t const>> filter_column_chunk_spans_d;
  std::vector<cudf::device_span<uint8_t const>> payload_column_chunk_spans_d;
  rmm::device_buffer device_buffer(host_src.get_size_in_bytes(), stream, mr_ref);
  auto buffer_data = static_cast<uint8_t*>(device_buffer.data());
  if (host_src.get_page_index_buffer()) {
    auto const& filter_byte_ranges  = host_src.get_filter_column_chunk_byte_ranges();
    auto const& payload_byte_ranges = host_src.get_payload_column_chunk_byte_ranges();
    std::ignore                     = std::accumulate(
      filter_byte_ranges.begin(),
      filter_byte_ranges.end(),
      size_t{0},
      [&filter_column_chunk_spans_d, buffer_data](auto sum, auto const& byte_range) {
        filter_column_chunk_spans_d.emplace_back(buffer_data + sum, byte_range.size());
        return sum + byte_range.size();
      });
    auto const payload_offset = filter_byte_ranges.empty() ? 0
                                                           : filter_byte_ranges.back().offset() +
                                                               filter_byte_ranges.back().size();
    std::ignore               = std::accumulate(
      payload_byte_ranges.begin(),
      payload_byte_ranges.end(),
      payload_offset,
      [&payload_column_chunk_spans_d, buffer_data](auto sum, auto const& byte_range) {
        payload_column_chunk_spans_d.emplace_back(buffer_data + sum, byte_range.size());
        return sum + byte_range.size();
      });
  } else {
    auto const& byte_ranges = host_src.get_all_column_chunk_byte_ranges();
    std::ignore =
      std::accumulate(byte_ranges.begin(),
                      byte_ranges.end(),
                      size_t{0},
                      [&column_chunk_spans_d, buffer_data](auto sum, auto const& byte_range) {
                        column_chunk_spans_d.emplace_back(buffer_data + sum, byte_range.size());
                        return sum + byte_range.size();
                      });
  }

  // Copy HOST data to GPU with a single async batch copy.
  auto const& allocation = host_src.get_column_chunks();
  size_t bytes_copied    = 0;
  std::vector<void*> dst_ptrs;
  std::vector<void*> src_ptrs;
  std::vector<size_t> counts;
  while (bytes_copied < host_src.get_size_in_bytes()) {
    auto const& block        = allocation->at(bytes_copied / allocation->block_size());
    auto const block_offset  = bytes_copied % allocation->block_size();
    auto const bytes_to_copy = std::min(allocation->block_size() - block_offset,
                                        host_src.get_size_in_bytes() - bytes_copied);
    dst_ptrs.push_back(static_cast<void*>(buffer_data + bytes_copied));
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
  std::unique_ptr<cudf::table> result_table;
#if CUDF_VERSION_NUM >= 2604
  if (host_src.get_page_index_buffer()) {
    auto filter_chunk_spans_h = cudf::host_span<const cudf::device_span<uint8_t const>>(
      filter_column_chunk_spans_d.data(), filter_column_chunk_spans_d.size());
    auto payload_chunk_spans_h = cudf::host_span<const cudf::device_span<uint8_t const>>(
      payload_column_chunk_spans_d.data(), payload_column_chunk_spans_d.size());

    auto page_index_buffer = host_src.get_page_index_buffer();

    // Build initial row mask
    auto row_mask = reader.build_row_mask_with_page_index_stats(
      host_src.get_rg_span(), host_src.get_reader_options(), stream, mr_ref);

    // Materialize filter columns
    auto row_mask_mutable_view = row_mask->mutable_view();
    auto [filter_table, filter_metadata] =
      reader.materialize_filter_columns(host_src.get_rg_span(),
                                        filter_chunk_spans_h,
                                        row_mask_mutable_view,
                                        use_data_page_mask::YES,
                                        host_src.get_reader_options(),
                                        stream);

    // Materialize payload columns
    auto [payload_table, payload_metadata] =
      reader.materialize_payload_columns(host_src.get_rg_span(),
                                         payload_chunk_spans_h,
                                         row_mask->view(),
                                         use_data_page_mask::YES,
                                         host_src.get_reader_options(),
                                         stream);

    // Combine filter and payload tables into a single table
    result_table = combine_tables(std::move(filter_table), std::move(payload_table));
  } else {
    auto column_chunk_spans_h = cudf::host_span<const cudf::device_span<uint8_t const>>(
      column_chunk_spans_d.data(), column_chunk_spans_d.size());
    auto [table, _] = reader.materialize_all_columns(
      host_src.get_rg_span(), column_chunk_spans_h, host_src.get_reader_options(), stream, mr_ref);
    result_table = std::move(table);
  }
#else
  // cudf 26.02 takes std::vector<rmm::device_buffer>&& instead of spans
  if (host_src.get_page_index_buffer()) {
    std::vector<rmm::device_buffer> filter_chunk_buffers;
    std::vector<rmm::device_buffer> payload_chunk_buffers;
    filter_chunk_buffers.reserve(filter_column_chunk_spans_d.size());
    payload_chunk_buffers.reserve(payload_column_chunk_spans_d.size());
    for (auto const& span : filter_column_chunk_spans_d) {
      filter_chunk_buffers.emplace_back(span.data(), span.size(), stream, mr_ref);
    }
    for (auto const& span : payload_column_chunk_spans_d) {
      payload_chunk_buffers.emplace_back(span.data(), span.size(), stream, mr_ref);
    }

    auto page_index_buffer = host_src.get_page_index_buffer();

    // Build initial row mask
    auto row_mask = reader.build_row_mask_with_page_index_stats(
      host_src.get_rg_span(), host_src.get_reader_options(), stream, mr_ref);

    // Materialize filter columns
    auto row_mask_mutable_view = row_mask->mutable_view();
    auto [filter_table, filter_metadata] =
      reader.materialize_filter_columns(host_src.get_rg_span(),
                                        std::move(filter_chunk_buffers),
                                        row_mask_mutable_view,
                                        use_data_page_mask::YES,
                                        host_src.get_reader_options(),
                                        stream);

    // Materialize payload columns
    auto [payload_table, payload_metadata] =
      reader.materialize_payload_columns(host_src.get_rg_span(),
                                         std::move(payload_chunk_buffers),
                                         row_mask->view(),
                                         use_data_page_mask::YES,
                                         host_src.get_reader_options(),
                                         stream);

    // Combine filter and payload tables into a single table
    result_table = combine_tables(std::move(filter_table), std::move(payload_table));
  } else {
    std::vector<rmm::device_buffer> column_chunk_buffers;
    for (auto const& span : column_chunk_spans_d) {
      column_chunk_buffers.emplace_back(span.data(), span.size(), stream, mr_ref);
    }
    auto [table, _] = reader.materialize_all_columns(host_src.get_rg_span(),
                                                     std::move(column_chunk_buffers),
                                                     host_src.get_reader_options(),
                                                     stream);
    result_table    = std::move(table);
  }
#endif

  auto new_table = std::move(result_table);
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
  auto& host_src             = source.cast<host_parquet_representation>();
  auto const data_size       = host_src.get_size_in_bytes();
  auto const& reader_options = host_src.get_reader_options();
  auto page_index_buffer     = host_src.get_page_index_buffer();
  auto cloned_reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    host_src.get_parquet_reader().parquet_metadata(), reader_options);

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

  if (page_index_buffer) {
    // Setup the page index in the cloned reader
    cloned_reader->setup_page_index(
      cudf::host_span<uint8_t const>(page_index_buffer->data(), page_index_buffer->size()));
    return std::make_unique<host_parquet_representation>(
      const_cast<cucascade::memory::memory_space*>(target_memory_space),
      std::move(dst_allocation),
      std::move(cloned_reader),
      reader_options,
      host_src.get_row_group_indices(),
      host_src.get_filter_column_chunk_byte_ranges(),
      host_src.get_payload_column_chunk_byte_ranges(),
      data_size,
      host_src.get_uncompressed_size_in_bytes(),
      std::move(page_index_buffer));
  }
  return std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(cloned_reader),
    reader_options,
    host_src.get_row_group_indices(),
    host_src.get_all_column_chunk_byte_ranges(),
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
