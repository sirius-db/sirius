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
#include "op/sirius_physical_parquet_scan.hpp"

#include <data/data_batch_utils.hpp>
#include <data/host_parquet_representation.hpp>
#include <op/scan/parquet_scan_task.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/config.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

namespace detail {

bool projected_columns_are_flat(cudf::io::parquet::FileMetaData const& meta,
                                std::vector<size_t> const& selected_column_indices)
{
  // Empty files are effectively "flat" for our purposes here.
  if (meta.row_groups.empty()) { return true; }
  auto const& cols = meta.row_groups.front().columns;

  // Flat leaf column => path length == 1.
  // For projections, we only need this property to hold for the projected (selected) leaf columns.
  return std::all_of(
    selected_column_indices.begin(), selected_column_indices.end(), [&cols](auto col_idx) {
      return col_idx < cols.size() && cols[col_idx].meta_data.path_in_schema.size() == 1;
    });
}

std::vector<size_t> make_selected_column_indices(sirius_physical_parquet_scan const& scan_op)
{
  // Deduplication set
  std::unordered_set<size_t> seen;
  std::vector<size_t> selected_column_indices;

  // In case there are duplicate columns in the projection list, we deduplicate, in order
  auto push_unique = [&selected_column_indices, &seen](auto col_idx) {
    if (col_idx == duckdb::DConstants::INVALID_INDEX) { return; }
    if (seen.insert(col_idx).second) {
      // Insert successful (not yet seen)
      selected_column_indices.push_back(col_idx);
    }
  };

  if (scan_op.projection_ids.empty()) {
    //===----------No Projection: Select All Columns----------===//
    std::for_each(scan_op.column_ids.begin(),
                  scan_op.column_ids.end(),
                  [&push_unique](duckdb::ColumnIndex const& column_id) {
                    push_unique(column_id.GetPrimaryIndex());
                  });
    return selected_column_indices;
  }

  //===----------Projection Applied: Select Projected Columns Only----------===//
  std::for_each(scan_op.projection_ids.begin(),
                scan_op.projection_ids.end(),
                [&scan_op, &push_unique](duckdb::idx_t projection_id) {
                  push_unique(scan_op.column_ids[projection_id].GetPrimaryIndex());
                });
  return selected_column_indices;
}

static std::unique_ptr<cudf::io::datasource::buffer> read_parquet_footer(cudf::io::datasource& src)
{
  auto constexpr file_tail_size    = sizeof(cudf::io::parquet::file_ender_s);
  auto constexpr footer_magic_size = sizeof(cudf::io::parquet::file_header_s);
  auto constexpr footer_size_size  = sizeof(uint32_t);
  std::string_view magic           = "PAR1";

  auto const file_size = src.size();
  if (file_size < file_tail_size) {
    throw std::runtime_error(
      "[parquet_scan_task] Parquet file is too small to contain valid footer");
  }

  // Read the file tail
  auto tail = src.host_read(file_size - file_tail_size, file_tail_size);

  // Verify the magic bytes
  if (std::memcmp(tail->data() + footer_size_size, magic.data(), footer_magic_size) != 0) {
    throw std::runtime_error(
      "[parquet_scan_task] Parquet file footer magic does not match expected value");
  }

  // Extract the footer size (little endian)
  uint32_t footer_size;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));

  // Return the footer buffer
  return src.host_read(file_size - file_tail_size - footer_size, footer_size);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Parquet Scan Task Global State
//===----------------------------------------------------------------------===//
parquet_scan_task_global_state::parquet_scan_task_global_state(
  sirius_physical_parquet_scan* scan_op, size_t approximate_batch_size)
  : _scan_op(scan_op),
    _approximate_batch_size(approximate_batch_size),
    _is_projected(!scan_op->projection_ids.empty()),
    _selected_column_indices(detail::make_selected_column_indices(*scan_op))
{
  if (scan_op->function.in_out_function) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] In-out table functions are not supported in sirius parquet "
      "scans.");
  }

  // Filter pushdown is not supported
  if (scan_op->dynamic_filters) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] Dynamic table filters are not supported in sirius parquet "
      "scans.");
  }

  // Expect parquet_scan to be bound through the multi-file reader
  auto& bind_data = scan_op->bind_data->Cast<duckdb::MultiFileBindData>();
  if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
    throw std::runtime_error("[parquet_scan_task_global_state] No input files to scan");
  }
  if (bind_data.file_list->GetTotalFileCount() != 1) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] parquet_scan_task currently supports a single parquet "
      "file per scan");
  }

  // Construct the io_source and read the footer
  auto file          = bind_data.file_list->GetFirstFile();
  _file_path         = file.path;
  auto datasource    = cudf::io::datasource::create(_file_path);
  auto footer_buffer = detail::read_parquet_footer(*datasource);

  // Initialize reader options for applying projections (FUTURE: filters)
  _reader_options = cudf::io::parquet_reader_options::builder().build();

  // Construct the file reader and read the metadata
  auto reader = hybrid_scan_reader(
    cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()), _reader_options);
  _file_metadata = reader.parquet_metadata();

  // Apply projections by column name using DuckDB's bound column names.
  if (_is_projected) {
    if (scan_op->names.empty()) {
      throw std::runtime_error(
        "[parquet_scan_task_global_state] Cannot apply projection: scan has no column names");
    }

    // We currently only support flat schemas for parquet scans with projections
    if (!detail::projected_columns_are_flat(_file_metadata, _selected_column_indices)) {
      throw std::runtime_error(
        "[parquet_scan_task_global_state] Parquet scans with projections currently only support "
        "flat projected columns");
    }

    std::vector<std::string> projected_columns;
    projected_columns.reserve(_selected_column_indices.size());
    std::for_each(_selected_column_indices.begin(),
                  _selected_column_indices.end(),
                  [&scan_op, &projected_columns](size_t col_idx) {
                    projected_columns.emplace_back(scan_op->names[col_idx]);
                  });

    _reader_options.set_column_names(std::move(projected_columns));
  }

  // Compute the byte counts per row group for the selected columns for task partitioning
  accumulate_row_group_byte_sizes();

  // Partition the row groups into ranges for each scan task
  partition_row_groups();
}

void parquet_scan_task_global_state::accumulate_row_group_byte_sizes()
{
  auto const total_rg = _file_metadata.row_groups.size();
  _row_group_uncompressed_bytes.reserve(total_rg);
  _row_group_compressed_bytes.reserve(total_rg);

  auto add_rg_bytes = [this](cudf::io::parquet::RowGroup const& rg) {
    size_t uncompressed_bytes = 0;
    size_t compressed_bytes   = 0;

    std::for_each(_selected_column_indices.begin(),
                  _selected_column_indices.end(),
                  [&uncompressed_bytes, &compressed_bytes, &rg](size_t col_idx) {
                    auto const& column_metadata = rg.columns[col_idx].meta_data;
                    if (column_metadata.total_uncompressed_size > 0) {
                      uncompressed_bytes += column_metadata.total_uncompressed_size;
                    }
                    if (column_metadata.total_compressed_size > 0) {
                      compressed_bytes += column_metadata.total_compressed_size;
                    }
                  });

    _row_group_uncompressed_bytes.push_back(uncompressed_bytes);
    _row_group_compressed_bytes.push_back(compressed_bytes);
  };

  std::for_each(_file_metadata.row_groups.begin(), _file_metadata.row_groups.end(), add_rg_bytes);
}

void parquet_scan_task_global_state::partition_row_groups()
{
  size_t partition_uncompressed_bytes = 0;
  size_t partition_compressed_bytes   = 0;
  size_t rg_start                     = 0;
  size_t rg_count                     = 0;
  for (size_t rg_idx = 0; rg_idx < _file_metadata.row_groups.size(); ++rg_idx) {
    partition_uncompressed_bytes += static_cast<size_t>(_row_group_uncompressed_bytes[rg_idx]);
    partition_compressed_bytes += static_cast<size_t>(_row_group_compressed_bytes[rg_idx]);
    ++rg_count;

    if (partition_uncompressed_bytes >= _approximate_batch_size) {
      _row_group_partitions.emplace_back(
        rg_start, rg_count, partition_uncompressed_bytes, partition_compressed_bytes);
      partition_uncompressed_bytes = 0;
      partition_compressed_bytes   = 0;
      rg_start                     = rg_idx + 1;
      rg_count                     = 0;
    }
  }
  // We may have a final partition that doesn't amount to the target batch size
  if (rg_count > 0) {
    _row_group_partitions.emplace_back(
      rg_start, rg_count, partition_uncompressed_bytes, partition_compressed_bytes);
  }
}

//===----------------------------------------------------------------------===//
// Parquet Scan Task Local State
//===----------------------------------------------------------------------===//
parquet_scan_task_local_state::parquet_scan_task_local_state(
  parquet_scan_task_global_state& g_state)
{
  // Get the next row-group partition
  auto const partition_idx = g_state.get_next_rg_partition_idx();
  if (partition_idx >= g_state.get_num_row_group_partitions()) {
    // Too many tasks have been created for this table scan!
    throw std::runtime_error(
      "[parquet_scan_task_local_state] No more row group partitions available for reservation.");
  }
  auto const& partition = g_state.get_row_group_partition(partition_idx);

  _rg_indices.resize(partition.row_group_count);
  std::iota(_rg_indices.begin(), _rg_indices.end(), partition.start_row_group);
  _reserved_uncompressed_bytes = partition.reserved_uncompressed_bytes;
  _reserved_compressed_bytes   = partition.reserved_compressed_bytes;
}

std::unique_ptr<parquet_scan_task_local_state::multiple_blocks_allocation>
parquet_scan_task_local_state::make_allocation()
{
  auto& mem_space = _reservation->get_memory_space();
  auto* allocator =
    mem_space.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (allocator == nullptr) {
    throw std::runtime_error(
      "[parquet_scan_task_local_state] Failed to get fixed_size_host_memory_resource allocator "
      "for HOST memory space");
  }
  return allocator->allocate_multiple_blocks(_reserved_compressed_bytes, _reservation.get());
}

//===----------------------------------------------------------------------===//
// Parquet Scan Task
//===----------------------------------------------------------------------===//
std::vector<std::shared_ptr<cucascade::data_batch>> parquet_scan_task::compute_task(
  rmm::cuda_stream_view /* stream */)
{
  auto& l_state = this->_local_state->cast<parquet_scan_task_local_state>();
  auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
  auto reader   = g_state.make_reader();

  // Make the allocation and accessor
  auto allocation = l_state.make_allocation();
  memory::multiple_blocks_allocation_accessor<uint8_t> data_accessor;
  data_accessor.initialize(0, allocation);

  // Get the byte ranges for the range of row groups assigned to this task
  auto byte_ranges =
    reader->all_column_chunks_byte_ranges(l_state.get_rg_span(), g_state.get_options());

  // Read each byte range into the allocation asynchronously
  std::vector<cudf::io::text::byte_range_info> new_byte_ranges;
  new_byte_ranges.reserve(byte_ranges.size());
  int64_t new_offset = 0;
  std::vector<std::future<std::size_t>> read_futures;
  for (auto const& range : byte_ranges) {
    read_range_into_allocation(
      range.offset(), range.size(), data_accessor, allocation, read_futures);
    new_byte_ranges.emplace_back(new_offset, range.size());
    new_offset += range.size();
  }
  std::for_each(read_futures.begin(), read_futures.end(), [](auto& future) { future.wait(); });
  assert(new_offset == l_state.get_reserved_compressed_bytes());

  // Create a data batch with the column chunks
  auto parquet_representation =
    std::make_unique<host_parquet_representation>(l_state.get_memory_space(),
                                                  std::move(allocation),
                                                  std::move(reader),
                                                  g_state.get_options(),
                                                  std::move(l_state.move_rg_indices()),
                                                  std::move(new_byte_ranges),
                                                  l_state.get_reserved_compressed_bytes(),
                                                  l_state.get_reserved_uncompressed_bytes());
  auto data_batch =
    std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(parquet_representation));
  return {data_batch};
}

void parquet_scan_task::publish_output(
  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches,
  rmm::cuda_stream_view /* stream */)
{
  for (auto& batch : output_batches) {
    _data_repo->add_data_batch(std::move(batch));
  }
}

void parquet_scan_task::read_range_into_allocation(
  size_t file_offset,
  size_t n_bytes,
  multiple_blocks_allocation_accessor& data_blocks_accessor,
  std::unique_ptr<multiple_blocks_allocation>& allocation,
  std::vector<std::future<std::size_t>>& read_futures)
{
  auto remaining_bytes = n_bytes;
  auto current_offset  = file_offset;

  while (remaining_bytes > 0) {
    auto const bytes_to_read = std::min(
      remaining_bytes, data_blocks_accessor.block_size - data_blocks_accessor.offset_in_block);
    auto buffer_ptr =
      reinterpret_cast<uint8_t*>(allocation->get_blocks()[data_blocks_accessor.block_index]) +
      data_blocks_accessor.offset_in_block;
    read_futures.push_back(_datasource->host_read_async(current_offset, bytes_to_read, buffer_ptr));
    remaining_bytes -= bytes_to_read;
    current_offset += bytes_to_read;
    data_blocks_accessor.offset_in_block += bytes_to_read;
    // Do we need to advance to the next block?
    if (data_blocks_accessor.offset_in_block == data_blocks_accessor.block_size) {
      ++data_blocks_accessor.block_index;
      data_blocks_accessor.offset_in_block = 0;
    }
  }
}

}  // namespace sirius::op::scan
