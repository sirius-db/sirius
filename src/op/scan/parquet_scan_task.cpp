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
#include <data/data_batch_utils.hpp>
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>
#include <data/sirius_converter_registry.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_scan_task.hpp>
#include <op/sirius_physical_parquet_scan.hpp>
#include <pipeline/sirius_pipeline.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/multi_file/multi_file_states.hpp>

// cudf
#include "cucascade/data/cpu_data_representation.hpp"
#include "cucascade/data/gpu_data_representation.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/cached_data_representation.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#if CUDF_VERSION_NUM >= 2604
#include <cudf/io/parquet_io_utils.hpp>
#endif

// standard library
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

#if CUDF_VERSION_NUM < 2604
namespace {
// Fallback for cudf < 26.04 which lacks cudf::io::parquet::fetch_footer_to_host.
// Reads the Parquet footer: last 8 bytes = [4-byte footer_len LE][4-byte "PAR1"],
// then reads footer_len bytes before that.
std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host_fallback(
  cudf::io::datasource& datasource)
{
  constexpr size_t PARQUET_MAGIC_SIZE = 4;
  constexpr size_t FOOTER_LEN_SIZE    = 4;
  constexpr size_t TAIL_SIZE          = PARQUET_MAGIC_SIZE + FOOTER_LEN_SIZE;

  auto const file_size = datasource.size();
  if (file_size < TAIL_SIZE + PARQUET_MAGIC_SIZE) {
    throw std::runtime_error("File too small to be a valid Parquet file");
  }

  // Read the last 8 bytes to get footer length
  auto tail_buf    = datasource.host_read(file_size - TAIL_SIZE, TAIL_SIZE);
  auto const* tail = tail_buf->data();

  // Footer length is a little-endian uint32 at offset 0
  uint32_t footer_len = tail[0] | (tail[1] << 8) | (tail[2] << 16) | (tail[3] << 24);

  // Read the footer bytes
  auto const footer_offset = file_size - TAIL_SIZE - footer_len;
  return datasource.host_read(footer_offset, footer_len);
}
}  // namespace
#endif

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
    if (duckdb::IsVirtualColumn(col_idx)) { return; }
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
  // Collect the set of column_ids indices that are referenced by projection_ids,
  // then iterate in column_ids order (not projection_ids order).
  // This ensures the parquet reader produces columns in the same order that
  // the TABLE_SCAN filter expects (column_ids order), since the filter's
  // BoundReferenceExpression indices are offsets into column_ids.
  std::unordered_set<duckdb::idx_t> projected_set(scan_op.projection_ids.begin(),
                                                  scan_op.projection_ids.end());
  for (duckdb::idx_t i = 0; i < scan_op.column_ids.size(); i++) {
    if (projected_set.count(i)) { push_unique(scan_op.column_ids[i].GetPrimaryIndex()); }
  }
  return selected_column_indices;
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Parquet Scan Task Global State
//===----------------------------------------------------------------------===//
parquet_scan_task_global_state::parquet_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_parquet_scan* scan_op,
  size_t approximate_batch_size)
  : pipeline::sirius_pipeline_task_global_state(pipeline),
    _scan_op(scan_op),
    _approximate_batch_size(approximate_batch_size),
    _is_projected(!scan_op->projection_ids.empty()),
    _selected_column_indices(detail::make_selected_column_indices(*scan_op))
{
  if (scan_op->function.in_out_function) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] In-out table functions are not supported in sirius "
      "parquet scans.");
  }

  // Filter pushdown is not supported
  if (scan_op->dynamic_filters) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] Dynamic table filters are not supported in sirius "
      "parquet scans.");
  }

  // Expect parquet_scan to be bound through the multi-file reader
  auto& bind_data = scan_op->bind_data->Cast<duckdb::MultiFileBindData>();
  if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
    throw std::runtime_error("[parquet_scan_task_global_state] No input files to scan");
  }

  auto files = bind_data.file_list->GetAllFiles();
  _file_paths.reserve(files.size());
  std::for_each(
    files.begin(), files.end(), [this](auto const& file) { _file_paths.push_back(file.path); });

  // Construct the io_sources and read the footers
  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  datasources.reserve(files.size());
  footer_buffers.reserve(files.size());
  std::for_each(
    _file_paths.begin(), _file_paths.end(), [&datasources, &footer_buffers](auto const& file_path) {
      auto datasource = cudf::io::datasource::create(file_path);
      datasources.push_back(std::move(datasource));
#if CUDF_VERSION_NUM >= 2604
      footer_buffers.push_back(cudf::io::parquet::fetch_footer_to_host(*datasources.back()));
#else
      footer_buffers.push_back(fetch_footer_to_host_fallback(*datasources.back()));
#endif
    });

  // Initialize reader options for applying projections (FUTURE: filters)
  _reader_options = cudf::io::parquet_reader_options::builder().build();

  // Construct the file readers and parse the metadata
  std::vector<std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader>> readers;
  _file_metadatas.reserve(files.size());
  readers.reserve(files.size());
  std::for_each(
    footer_buffers.begin(), footer_buffers.end(), [&readers, this](auto& footer_buffer) {
      auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
        cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
        _reader_options);
      _file_metadatas.push_back(reader->parquet_metadata());
      readers.push_back(std::move(reader));
    });

  // Apply projections by column name using DuckDB's bound column names.
  if (_is_projected) {
    if (scan_op->names.empty()) {
      throw std::runtime_error(
        "[parquet_scan_task_global_state] Cannot apply projection: scan has no column names");
    }

    // We currently only support flat schemas for parquet scans with projections
    /// TODO: Support nested schemas for projected scans
    for (auto const& meta : _file_metadatas) {
      if (!detail::projected_columns_are_flat(meta, _selected_column_indices)) {
        throw std::runtime_error(
          "[parquet_scan_task_global_state] Parquet scans with projections currently only support "
          "flat projected columns");
      }
    }

    std::vector<std::string> projected_columns;
    projected_columns.reserve(_selected_column_indices.size());
    std::for_each(_selected_column_indices.begin(),
                  _selected_column_indices.end(),
                  [&scan_op, &projected_columns](size_t col_idx) {
                    projected_columns.emplace_back(scan_op->names[col_idx]);
                  });

#if CUDF_VERSION_NUM >= 2604
    _reader_options.set_column_names(std::move(projected_columns));
#else
    _reader_options.set_columns(std::move(projected_columns));
#endif
  }

  // Compute the byte counts per row group for the selected columns for task partitioning
  accumulate_row_group_byte_sizes();

  // Partition the row groups into ranges for each scan task
  partition_row_groups();
}

void parquet_scan_task_global_state::accumulate_row_group_byte_sizes()
{
  _row_group_compressed_bytes.resize(_file_metadatas.size());
  _row_group_uncompressed_bytes.resize(_file_metadatas.size());
  for (size_t file_idx = 0; file_idx < _file_metadatas.size(); ++file_idx) {
    auto const& meta    = _file_metadatas[file_idx];
    auto const total_rg = meta.row_groups.size();
    _row_group_uncompressed_bytes[file_idx].reserve(total_rg);
    _row_group_compressed_bytes[file_idx].reserve(total_rg);

    auto add_rg_bytes = [this, file_idx](cudf::io::parquet::RowGroup const& rg) {
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

      _row_group_uncompressed_bytes[file_idx].push_back(uncompressed_bytes);
      _row_group_compressed_bytes[file_idx].push_back(compressed_bytes);
    };

    std::for_each(meta.row_groups.begin(), meta.row_groups.end(), add_rg_bytes);
  }
}

void parquet_scan_task_global_state::partition_row_groups()
{
  for (size_t file_idx = 0; file_idx < _file_metadatas.size(); ++file_idx) {
    auto const& meta = _file_metadatas[file_idx];

    size_t partition_uncompressed_bytes = 0;
    size_t partition_compressed_bytes   = 0;
    size_t rg_start                     = 0;
    size_t rg_count                     = 0;
    for (size_t rg_idx = 0; rg_idx < meta.row_groups.size(); ++rg_idx) {
      partition_uncompressed_bytes +=
        static_cast<size_t>(_row_group_uncompressed_bytes[file_idx][rg_idx]);
      partition_compressed_bytes +=
        static_cast<size_t>(_row_group_compressed_bytes[file_idx][rg_idx]);
      ++rg_count;

      if (partition_uncompressed_bytes >= _approximate_batch_size) {
        _row_group_partitions.emplace_back(
          file_idx, rg_start, rg_count, partition_uncompressed_bytes, partition_compressed_bytes);
        partition_uncompressed_bytes = 0;
        partition_compressed_bytes   = 0;
        rg_start                     = rg_idx + 1;
        rg_count                     = 0;
      }
    }
    // We may have a final partition that doesn't amount to the target batch size
    if (rg_count > 0) {
      _row_group_partitions.emplace_back(
        file_idx, rg_start, rg_count, partition_uncompressed_bytes, partition_compressed_bytes);
    }
  }
}

//===----------------------------------------------------------------------===//
// Parquet Scan Task Local State
//===----------------------------------------------------------------------===//
parquet_scan_task_local_state::parquet_scan_task_local_state(
  parquet_scan_task_global_state& g_state, size_t partition_idx)
{
  auto const& partition = g_state.get_row_group_partition(partition_idx);

  _file_idx = partition.file_idx;
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

parquet_scan_task::~parquet_scan_task()
{
  if (_global_state != nullptr) {
    auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
    if (auto pipeline = g_state.get_operator().get_pipeline()) { pipeline->mark_task_completed(); }
  }
}

//===----------------------------------------------------------------------===//
// Parquet Scan Task
//===----------------------------------------------------------------------===//
std::unique_ptr<op::operator_data> parquet_scan_task::compute_task(
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
  auto& l_state = this->_local_state->cast<parquet_scan_task_local_state>();
  auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
  auto reader   = g_state.make_reader(l_state.get_file_idx());

  auto& scan_op      = g_state.get_operator();
  auto const num_rgs = l_state.get_rg_span().size();
  SIRIUS_LOG_TRACE(
    "Pipeline {}: operator {} (id={}) executing on {} batches with num row: {}",
    scan_op.get_pipeline().get() != nullptr ? scan_op.get_pipeline()->get_pipeline_id() : 0,
    scan_op.get_name(),
    scan_op.get_operator_id(),
    0,
    "");
  auto const task_start = std::chrono::high_resolution_clock::now();

  // Make the allocation and accessor
  auto allocation = l_state.make_allocation();
  memory::multiple_blocks_allocation_accessor<uint8_t> data_accessor;
  data_accessor.initialize(0, allocation);

  // Get the byte ranges for the range of row groups assigned to this task
  auto byte_ranges =
    reader->all_column_chunks_byte_ranges(l_state.get_rg_span(), g_state.get_options());

  // Read each byte range into the allocation asynchronously
  int64_t bytes_read = 0;
  std::vector<std::future<std::size_t>> read_futures;
  for (auto const& range : byte_ranges) {
    read_range_into_allocation(
      range.offset(), range.size(), data_accessor, allocation, read_futures);
    bytes_read += range.size();
  }
  std::for_each(read_futures.begin(), read_futures.end(), [](auto& future) { future.get(); });

  if (bytes_read != l_state.get_reserved_compressed_bytes()) {
    // Metadata / file data mismatch
    throw std::runtime_error(
      "[parquet_scan_task] Error in reading byte ranges: total bytes read does not match reserved "
      "compressed bytes");
  }

  // Create a data batch with the column chunks
  auto parquet_representation =
    std::make_unique<host_parquet_representation>(l_state.get_memory_space(),
                                                  std::move(allocation),
                                                  std::move(reader),
                                                  g_state.get_options(),
                                                  std::move(l_state.get_rg_indices()),
                                                  std::move(byte_ranges),
                                                  l_state.get_reserved_compressed_bytes(),
                                                  l_state.get_reserved_uncompressed_bytes(),
                                                  _datasource);

  std::shared_ptr<cucascade::data_batch> batch;
  if (_materialized_columns) {
    auto& registry          = sirius::converter_registry::get();
    auto materialized_table = registry.convert<cucascade::gpu_table_representation>(
      *parquet_representation, _gpu_memory_space, stream);
    stream.synchronize();
    parquet_representation.reset();
    auto host_table = registry.convert<cucascade::host_data_representation>(
      *materialized_table, l_state.get_memory_space(), stream);
    if (_wrap_in_cache) {
      batch = std::make_shared<cucascade::data_batch>(
        get_next_batch_id(),
        std::make_unique<cached_host_data_representation>(std::move(host_table)));
    } else {
      batch = std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(host_table));
    }
  } else {
    if (_wrap_in_cache) {
      batch = std::make_shared<cucascade::data_batch>(
        get_next_batch_id(),
        std::make_unique<cached_host_parquet_representation>(std::move(parquet_representation)));
    } else {
      batch = std::make_shared<cucascade::data_batch>(get_next_batch_id(),
                                                      std::move(parquet_representation));
    }
  }
  auto result = std::make_unique<op::operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});

  auto const task_end = std::chrono::high_resolution_clock::now();
  auto const task_duration =
    std::chrono::duration_cast<std::chrono::microseconds>(task_end - task_start);
  SIRIUS_LOG_TRACE(
    "Pipeline {}: operator {} (id={}) produced {} batches with num rows: {}, execution time: "
    "{:.2f} ms",
    scan_op.get_pipeline().get() != nullptr ? scan_op.get_pipeline()->get_pipeline_id() : 0,
    scan_op.get_name(),
    scan_op.get_operator_id(),
    result->get_data_batches().size(),
    num_rgs,
    task_duration.count() / 1000.0);
  return result;
}

void parquet_scan_task::publish_output(op::operator_data& output_data,
                                       rmm::cuda_stream_view /* stream */)
{
  for (auto& batch : output_data.get_data_batches()) {
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
