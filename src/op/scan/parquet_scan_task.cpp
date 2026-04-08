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
#include <data/data_batch_utils.hpp>
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression_executor/gpu_expression_translator.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_scan_task.hpp>
#include <op/sirius_physical_parquet_scan.hpp>
#include <pipeline/sirius_pipeline.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/multi_file/multi_file_states.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/cudf_utils.hpp>
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

std::vector<byte_range_info> merge_byte_ranges(std::vector<byte_range_info> const& byte_ranges)
{
  if (byte_ranges.empty()) { return {}; }

  std::vector<byte_range_info> merged;
  merged.reserve(byte_ranges.size());

  auto current_start = byte_ranges[0].offset();
  auto current_end   = current_start + byte_ranges[0].size();

  for (auto const& range : byte_ranges) {
    auto const range_start = range.offset();
    auto const range_end   = range_start + range.size();

    if (range_start <= current_end) {
      // Ranges are contiguous, extend the current range
      current_end = std::max(current_end, range_end);
    } else {
      // No overlap, push the current range and start a new one
      merged.emplace_back(current_start, current_end - current_start);
      current_start = range_start;
      current_end   = range_end;
    }
  }
  // Push the final range
  merged.emplace_back(current_start, current_end - current_start);

  return merged;
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Parquet Scan Task Global State
//===----------------------------------------------------------------------===//
parquet_scan_task_global_state::parquet_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_parquet_scan* scan_op,
  std::size_t approximate_batch_size)
  : pipeline::sirius_pipeline_task_global_state(pipeline),
    _scan_op(scan_op),
    _approximate_batch_size(approximate_batch_size)
{
  if (scan_op->function.in_out_function) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] In-out table functions are not supported in sirius "
      "parquet scans.");
  }
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

  initialize_from_files();
}

// Protected constructor: caller supplies pre-resolved file paths and column indices.
// Skips MultiFileBindData extraction; everything else is identical to the public
// constructor (footer reads, metadata parsing, row-group partitioning).
parquet_scan_task_global_state::parquet_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_parquet_scan* scan_op,
  std::vector<std::string> file_paths,
  std::vector<size_t> const& selected_column_indices,
  std::size_t approximate_batch_size)
  : pipeline::sirius_pipeline_task_global_state(pipeline),
    _scan_op(scan_op),
    _approximate_batch_size(approximate_batch_size),
    _file_paths(std::move(file_paths))
{
  if (_file_paths.empty()) {
    throw std::runtime_error("[parquet_scan_task_global_state] No input files to scan");
  }
  if (scan_op->function.in_out_function) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] In-out table functions are not supported in sirius "
      "parquet scans.");
  }
  if (scan_op->dynamic_filters) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] Dynamic table filters are not supported in sirius "
      "parquet scans.");
  }

  initialize_from_files();
}

void parquet_scan_task_global_state::initialize_from_files()
{
  // Construct the io_sources and read the footers.
  // Also record each file's total size and footer offset so that scan tasks
  // can cache the parquet header+footer alongside the column-chunk data,
  // eliminating all file I/O during subsequent (preload) iterations.
  constexpr size_t PARQUET_MAGIC_SIZE = 4;
  constexpr size_t FOOTER_TAIL_SIZE   = 8;  // 4-byte footer_len + 4-byte magic

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  datasources.reserve(_file_paths.size());
  footer_buffers.reserve(_file_paths.size());
  _file_sizes.reserve(_file_paths.size());
  _metadata_byte_sizes.reserve(_file_paths.size());
  _footer_offsets.reserve(_file_paths.size());

  for (auto const& file_path : _file_paths) {
    auto datasource      = cudf::io::datasource::create(file_path);
    auto const file_size = datasource->size();
    datasources.push_back(std::move(datasource));

#if CUDF_VERSION_NUM >= 2604
    footer_buffers.push_back(cudf::io::parquet::fetch_footer_to_host(*datasources.back()));
    auto const footer_len = footer_buffers.back()->size();
#else
    footer_buffers.push_back(fetch_footer_to_host_fallback(*datasources.back()));
    auto const footer_len = footer_buffers.back()->size();
#endif

    auto const footer_offset  = file_size - FOOTER_TAIL_SIZE - footer_len;
    auto const metadata_bytes = PARQUET_MAGIC_SIZE + footer_len + FOOTER_TAIL_SIZE;

    _file_sizes.push_back(file_size);
    _footer_offsets.push_back(footer_offset);
    _metadata_byte_sizes.push_back(metadata_bytes);
  }

  // Initialize reader options for applying projections and/or filters
  _reader_options = cudf::io::parquet_reader_options::builder().build();

  // If filtering or projecting, we need column names
  bool const do_filter    = _scan_op->translated_filter.has_value();
  bool const is_projected = !_scan_op->projection_ids.empty();
  if (do_filter || is_projected) {
    if (_scan_op->names.empty()) {
      throw std::runtime_error(
        "[parquet_scan_task_global_state] Cannot apply filter or projection: scan has no column "
        "names");
    }
  }

  //===----------Projections----------===//
  auto projected_column_indices = detail::make_selected_column_indices(*_scan_op);
  std::unordered_set<std::size_t> pure_filter_column_indices;
  if (is_projected) {
    std::vector<std::string> projected_column_names;
    std::for_each(projected_column_indices.begin(),
                  projected_column_indices.end(),
                  [this, &projected_column_names](std::size_t col_idx) {
                    projected_column_names.push_back(_scan_op->names[col_idx]);
                  });
#if CUDF_VERSION_NUM >= 2604
    _reader_options.set_column_names(std::move(projected_column_names));
#else
    _reader_options.set_columns(std::move(projected_column_names));
#endif
    // We only prune the pure filter columns from the projected set when the reader performs the
    // filter. Otherwise, the expression executor will not find the filter columns.
    if (do_filter) {
      _post_filter_projection_ids.reserve(_scan_op->types.size());
      for (std::size_t i = 0; i < _scan_op->projection_ids.size(); i++) {
        if (i < _scan_op->types.size()) {
          _post_filter_projection_ids.push_back(_scan_op->projection_ids[i]);
        } else {
          // This is a pure filter column that is not among the expected output columns.
          auto const projection_id = _scan_op->projection_ids[i];
          pure_filter_column_indices.insert(_scan_op->column_ids[projection_id].GetPrimaryIndex());
        }
      }
    }
  }

  //===----------Filters----------===//
  if (do_filter) {
    // The filter was attempted in the physical operator constructor.
    // If translation failed, the table scan operator will execute the filter, otherwise the table
    // scan operator will be a no-op passthrough.
    _translated_filter = std::make_shared<gpu_expression_translator::translated_expression>(
      std::move(*_scan_op->translated_filter));
    _reader_options.set_filter(_translated_filter->back());
  }

  // Construct the file readers and parse the metadata
  std::vector<std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader>> readers;
  _file_metadatas.reserve(_file_paths.size());
  readers.reserve(_file_paths.size());
  std::for_each(footer_buffers.begin(),
                footer_buffers.end(),
                [&readers, is_projected, &projected_column_indices, this](auto& footer_buffer) {
                  auto reader =
                    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
                      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
                      _reader_options);
                  auto meta = reader->parquet_metadata();
                  if (is_projected) {
                    // We currently only support flat schemas for parquet scans with projections.
                    // This is only because we need the set of needed primary indices for
                    // partitioning the row groups, and determining the full set of primary indices
                    // for a nested type is more complex.
                    /// TODO: Support nested schemas for projected scans
                    if (!detail::projected_columns_are_flat(meta, projected_column_indices)) {
                      throw std::runtime_error(
                        "[parquet_scan_task_global_state] Parquet scans with projections currently "
                        "only support flat projected columns");
                    }
                  }
                  _file_metadatas.push_back(std::move(meta));
                  readers.push_back(std::move(reader));
                });

  //===----------Row Group Partitioning for Task Generation----------===//
  for (std::size_t file_idx = 0; file_idx < _file_paths.size(); ++file_idx) {
    auto row_group_indices = readers[file_idx]->all_row_groups(_reader_options);
    if (_translated_filter) {
      auto const row_groups_before_pruning = row_group_indices.size();
      // clang-format off
      SIRIUS_LOG_INFO("[parquet_scan_task_global_state] Row group pruning: file: {}\n" \
                      "                                                         before: {}",
                      _file_paths[file_idx],
                      row_groups_before_pruning);
      // clang-format on
      // Prune row groups with filter pushdown using metadata statistics.
      row_group_indices = readers[file_idx]->filter_row_groups_with_stats(
        row_group_indices, _reader_options, rmm::cuda_stream_default);
      auto const row_groups_after_pruning = row_group_indices.size();
      auto const pruned_row_groups        = row_groups_before_pruning - row_groups_after_pruning;
      // clang-format off
      SIRIUS_LOG_INFO("[parquet_scan_task_global_state]                    after: {} (pruned {})",
                      row_groups_after_pruning,
                      pruned_row_groups);
      // clang-format on
    }
    auto const& file_metadata = _file_metadatas[file_idx];

    std::size_t partition_uncompressed_bytes = 0;
    std::size_t partition_compressed_bytes   = 0;
    std::vector<cudf::size_type> partition_rg_indices;
    partition_rg_indices.reserve(row_group_indices.size());

    auto flush_partition = [&]() {
      if (partition_rg_indices.empty()) { return; }
      _row_group_partitions.emplace_back(file_idx,
                                         std::move(partition_rg_indices),
                                         partition_uncompressed_bytes,
                                         partition_compressed_bytes);
      partition_rg_indices.clear();
      partition_uncompressed_bytes = 0;
      partition_compressed_bytes   = 0;
    };

    for (auto const rg_idx : row_group_indices) {
      auto const& row_group = file_metadata.row_groups[rg_idx];
      partition_rg_indices.push_back(rg_idx);

      for (auto const col_idx : projected_column_indices) {
        auto const& column_metadata = row_group.columns[col_idx].meta_data;
        // To reflect the fact that pure filter columns are not part of the table scan result,
        // we omit them from the uncompressed byte count.
        if (column_metadata.total_uncompressed_size > 0 &&
            !pure_filter_column_indices.contains(col_idx)) {
          partition_uncompressed_bytes +=
            static_cast<std::size_t>(column_metadata.total_uncompressed_size);
        }
        if (column_metadata.total_compressed_size > 0) {
          partition_compressed_bytes +=
            static_cast<std::size_t>(column_metadata.total_compressed_size);
        }
      }

      if (partition_uncompressed_bytes >= _approximate_batch_size) { flush_partition(); }
    }

    // Emit any trailing partition smaller than the target size.
    flush_partition();
  }
}

//===----------------------------------------------------------------------===//
// Parquet Scan Task Local State
//===----------------------------------------------------------------------===//
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
  return allocator->allocate_multiple_blocks(get_reserved_compressed_bytes(), _reservation.get());
}

//===----------------------------------------------------------------------===//
// Parquet Scan Task
//===----------------------------------------------------------------------===//
parquet_scan_task::~parquet_scan_task()
{
  if (_global_state != nullptr) {
    auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
    if (auto pipeline = g_state.get_operator().get_pipeline()) { pipeline->mark_task_completed(); }
  }
}

void parquet_scan_task::execute(rmm::cuda_stream_view stream)
{
  auto& l_state        = this->_local_state->cast<parquet_scan_task_local_state>();
  auto estimated_bytes = l_state.get_reserved_compressed_bytes();

  // Record memory metrics for future reservation estimates.
  // Parquet scan tasks don't have peak memory tracking, so use output size as proxy.
  if (auto output_data = compute_task(stream); output_data) {
    std::size_t output_bytes = 0;
    for (const auto& batch : output_data->get_data_batches()) {
      if (batch && batch->get_data()) { output_bytes += batch->get_data()->get_size_in_bytes(); }
    }
    auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
    g_state.get_memory_history().record({estimated_bytes, output_bytes, output_bytes});

    publish_output(*output_data, stream);
  }
}

std::unique_ptr<op::operator_data> parquet_scan_task::compute_task(
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
  auto& l_state = this->_local_state->cast<parquet_scan_task_local_state>();
  auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();

  if (!_datasource) {
    _datasource = cudf::io::datasource::create(g_state.get_file_path(l_state.get_file_idx()));
  }

  auto reader = g_state.make_reader(l_state.get_file_idx());

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

  // Get the byte ranges for the range of row groups assigned to this task.
  // Prepend the parquet header (4-byte magic at offset 0) and append the
  // footer + trailer so that the cache covers ALL bytes cuDF needs to open
  // the file, enabling zero file I/O during preload iterations.
  auto const file_idx    = l_state.get_file_idx();
  auto const file_size   = g_state.get_file_size(file_idx);
  auto const footer_off  = g_state.get_footer_offset(file_idx);
  auto const footer_size = file_size - footer_off;

  using range_t = cudf::io::text::byte_range_info;

  auto column_chunk_ranges =
    reader->all_column_chunks_byte_ranges(l_state.get_rg_span(), g_state.get_options());
  auto merged_column_chunk_ranges = detail::merge_byte_ranges(column_chunk_ranges);

  std::vector<range_t> byte_ranges;
  byte_ranges.reserve(merged_column_chunk_ranges.size() + 2);
  byte_ranges.emplace_back(0, 4);  // PAR1 header
  byte_ranges.insert(
    byte_ranges.end(), merged_column_chunk_ranges.begin(), merged_column_chunk_ranges.end());
  byte_ranges.emplace_back(footer_off, footer_size);  // footer + trailer

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
                                                  file_size,
                                                  _datasource,
                                                  g_state.get_filter_expression(),
                                                  g_state.get_post_filter_projection_ids());

  // Propagate the post-convert hook and data-file path (non-null only for iceberg V2 scans).
  if (g_state.has_post_convert_fn()) {
    parquet_representation->set_post_convert_fn(g_state.get_post_convert_fn());
    parquet_representation->set_data_file_path(g_state.get_file_path(l_state.get_file_idx()));
  }

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

size_t parquet_scan_task::get_estimated_reservation_size() const
{
  auto current_estimate =
    this->_local_state->cast<parquet_scan_task_local_state>().get_task_consumption_basis();
  auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
  auto refined  = g_state.get_memory_history().estimate_peak_memory(current_estimate);
  if (refined) { return *refined; }
  return current_estimate;
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
