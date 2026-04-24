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
#include <helper/type_conversions.hpp>
#include <io/cucascade_datasource.hpp>
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
#include <duckdb/common/hive_partitioning.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#if CUDF_VERSION_NUM >= 2604
#include <cudf/io/parquet_io_utils.hpp>
#endif

// rmm
#include <rmm/cuda_stream.hpp>

// standard library
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
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
                                std::vector<std::string> const& projected_column_names)
{
  // Empty files are effectively "flat" for our purposes here.
  if (meta.row_groups.empty()) { return true; }
  auto const& cols = meta.row_groups.front().columns;

  // Build name → column index map for the parquet file.
  std::unordered_map<std::string, size_t> name_to_idx;
  for (size_t i = 0; i < cols.size(); ++i) {
    if (!cols[i].meta_data.path_in_schema.empty()) {
      name_to_idx[cols[i].meta_data.path_in_schema[0]] = i;
    }
  }

  // Flat leaf column => path length == 1.
  return std::all_of(projected_column_names.begin(),
                     projected_column_names.end(),
                     [&cols, &name_to_idx](auto const& col_name) {
                       auto it = name_to_idx.find(col_name);
                       return it != name_to_idx.end() &&
                              cols[it->second].meta_data.path_in_schema.size() == 1;
                     });
}

std::vector<size_t> make_selected_column_indices(
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids)
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

  if (projection_ids.empty()) {
    //===----------No Projection: Select All Columns----------===//
    std::for_each(
      column_ids.begin(), column_ids.end(), [&push_unique](duckdb::ColumnIndex const& column_id) {
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
  std::unordered_set<std::size_t> projected_set(projection_ids.begin(), projection_ids.end());
  for (std::size_t i = 0; i < column_ids.size(); i++) {
    if (projected_set.count(i)) { push_unique(column_ids[i].GetPrimaryIndex()); }
  }
  return selected_column_indices;
}

std::vector<byte_range_info> merge_byte_ranges(std::vector<byte_range_info> const& byte_ranges)
{
  if (byte_ranges.empty()) { return {}; }

  // The merge walk requires ranges sorted by offset. Callers may pass ranges in
  // projection order (e.g. reader->all_column_chunks_byte_ranges returns them in
  // the order of set_column_names), which can differ from file-offset order when
  // the user selects columns out of parquet-file order. Sort defensively.
  std::vector<byte_range_info> sorted(byte_ranges.begin(), byte_ranges.end());
  std::sort(sorted.begin(), sorted.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<byte_range_info> merged;
  merged.reserve(sorted.size());

  auto current_start = sorted[0].offset();
  auto current_end   = current_start + sorted[0].size();

  for (auto const& range : sorted) {
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
  std::size_t approximate_batch_size,
  std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends)
  : pipeline::sirius_pipeline_task_global_state(pipeline),
    _approximate_batch_size(approximate_batch_size),
    _scan_op(scan_op),
    _gpu_io_backends(std::move(gpu_io_backends))
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

  // Detect hive partition columns — these exist in the DuckDB schema but not in parquet files.
  // Their values come from directory paths (e.g., partition_col=42/).
  for (auto const& hpi : bind_data.reader_bind.hive_partitioning_indexes) {
    _hive_partition_index_set.insert(hpi.index);
    _hive_partition_columns.push_back(hive_partition_column{hpi.value, hpi.index});
  }

  // Build selected column indices, then drop any hive partition columns (they are injected
  // post-read from the directory path, not read from the parquet file itself).
  _selected_column_indices =
    detail::make_selected_column_indices(scan_op->column_ids, scan_op->projection_ids);
  if (!_hive_partition_index_set.empty()) {
    _selected_column_indices.erase(
      std::remove_if(_selected_column_indices.begin(),
                     _selected_column_indices.end(),
                     [this](size_t idx) { return _hive_partition_index_set.count(idx) > 0; }),
      _selected_column_indices.end());
  }

  auto files = bind_data.file_list->GetAllFiles();
  _file_paths.reserve(files.size());
  std::for_each(
    files.begin(), files.end(), [this](auto const& file) { _file_paths.push_back(file.path); });

  initialize_from_files();

  // Build partition injection function if this scan has partition columns.
  init_hive_partitions(bind_data, scan_op);
}

// Protected constructor: caller supplies pre-resolved file paths and column indices.
// Skips MultiFileBindData extraction; everything else is identical to the public
// constructor (footer reads, metadata parsing, row-group partitioning).
parquet_scan_task_global_state::parquet_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_parquet_scan* scan_op,
  std::vector<std::string> file_paths,
  std::vector<size_t> selected_column_indices,
  std::size_t approximate_batch_size,
  std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends)
  : pipeline::sirius_pipeline_task_global_state(pipeline),
    _approximate_batch_size(approximate_batch_size),
    _scan_op(scan_op),
    _selected_column_indices(std::move(selected_column_indices)),
    _file_paths(std::move(file_paths)),
    _gpu_io_backends(std::move(gpu_io_backends))
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

  // IO-05: use cucascade-backed datasource instead of kvikio file_source.
  // Planning-time reads — pick the first available GPU backend deterministically;
  // the reads are small (footer only) and don't populate per-GPU row-group
  // allocations, so context mismatch is correctness-neutral (research Pitfall 6).
  auto const planning_backend_it = _gpu_io_backends.begin();
  if (planning_backend_it == _gpu_io_backends.end()) {
    throw std::runtime_error(
      "[parquet_scan_task_global_state] No GPU io_backends configured — "
      "SiriusContext::initialize() must have populated at least one "
      "(Approach C seeding via task_creator required).");
  }

  for (auto const& file_path : _file_paths) {
    // cucascade::idisk_io_backend has no size() API; use std::filesystem
    // (research Open Q3). The adapter caches the file_size for size() calls.
    auto const file_size = std::filesystem::file_size(file_path);
    auto datasource      = std::make_unique<sirius::io::cucascade_datasource>(
      planning_backend_it->second, std::filesystem::path{file_path}, file_size);
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
  bool const do_filter    = !_scan_op->translated_filter_by_device.empty();
  bool const is_projected = !_scan_op->projection_ids.empty();
  if (do_filter || is_projected) {
    if (_scan_op->names.empty()) {
      throw std::runtime_error(
        "[parquet_scan_task_global_state] Cannot apply filter or projection: scan has no column "
        "names");
    }
  }

  // Parse file metadata first so we can detect partitions that weren't advertised
  // via bind_data.reader_bind.hive_partitioning_indexes (iceberg path).
  std::vector<std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader>> readers;
  _file_metadatas.reserve(_file_paths.size());
  readers.reserve(_file_paths.size());
  for (auto& footer_buffer : footer_buffers) {
    auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      _reader_options);
    _file_metadatas.push_back(reader->parquet_metadata());
    readers.push_back(std::move(reader));
  }

  // -----------------------------------------------------------------------
  // Detect hive partition columns from the parquet schema.
  //
  // Columns that appear in the DuckDB schema (scan_op->names) but NOT in the
  // parquet file's leaf columns are treated as hive partition columns. This
  // catches the case where bind_data did not advertise them (e.g. iceberg).
  // Detected partition columns are removed from _selected_column_indices and
  // recorded in _hive_partition_columns for injection after GPU read.
  // -----------------------------------------------------------------------
  if (!_file_metadatas.empty() && !_selected_column_indices.empty() && !_scan_op->names.empty()) {
    auto const& first_meta = _file_metadatas[0];
    std::unordered_set<std::string> parquet_col_names;
    for (size_t i = 1; i < first_meta.schema.size(); ++i) {
      if (first_meta.schema[i].num_children == 0) {
        parquet_col_names.insert(first_meta.schema[i].name);
      }
    }

    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(_selected_column_indices.size());
    for (auto idx : _selected_column_indices) {
      if (idx < _scan_op->names.size() && parquet_col_names.count(_scan_op->names[idx])) {
        filtered_indices.push_back(idx);
      } else if (idx < _scan_op->names.size()) {
        _hive_partition_index_set.insert(idx);
        _hive_partition_columns.push_back(hive_partition_column{_scan_op->names[idx], idx});
        SIRIUS_LOG_DEBUG(
          "[parquet_scan] Column '{}' (idx={}) not in parquet schema — "
          "treating as partition column.",
          _scan_op->names[idx],
          idx);
      }
    }

    if (filtered_indices.size() != _selected_column_indices.size()) {
      _selected_column_indices = std::move(filtered_indices);
    }
  }

  //===----------Projections----------===//
  std::unordered_set<std::size_t> pure_filter_column_indices;
  std::vector<std::string> projected_column_names;
  if (is_projected) {
    projected_column_names.reserve(_selected_column_indices.size());
    std::for_each(_selected_column_indices.begin(),
                  _selected_column_indices.end(),
                  [this, &projected_column_names](std::size_t col_idx) {
                    projected_column_names.push_back(_scan_op->names[col_idx]);
                  });
#if CUDF_VERSION_NUM >= 2604
    _reader_options.set_column_names(projected_column_names);
#else
    _reader_options.set_columns(projected_column_names);
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
    // Per-GPU filter expressions were built by the physical operator constructor
    // (one per configured GPU). We move the whole map into shared ownership so
    // host_parquet_representation can keep it alive across all tasks. The filter is
    // NOT set on _reader_options here because a single _reader_options instance is
    // shared by all tasks regardless of target GPU; set_filter with a device-
    // specific tree here would bind everyone to one device. Instead, each converter
    // call selects the right per-device tree and calls set_filter on its own opts
    // copy under target_device_raii.
    _translated_filter_by_device = std::make_shared<
      std::unordered_map<int, gpu_expression_translator::translated_expression>>(
      std::move(_scan_op->translated_filter_by_device));
  }

  // Verify projected columns are flat (we don't support nested projections yet).
  if (is_projected) {
    for (auto const& meta : _file_metadatas) {
      if (!detail::projected_columns_are_flat(meta, projected_column_names)) {
        throw std::runtime_error(
          "[parquet_scan_task_global_state] Parquet scans with projections currently only support "
          "flat projected columns");
      }
    }
  }

  //===----------Row Group Partitioning for Task Generation----------===//
  //
  // Per-row-group byte-size accumulation uses name-based lookup to map each
  // _selected_column_indices entry (DuckDB primary index) to the parquet
  // column position. This is necessary because after hive partition removal
  // the DuckDB indices no longer coincide with parquet column positions.
  //
  // HYG-01: explicit stream for the planning-time filter_row_groups_with_stats
  // call below. A throwaway local stream is sufficient here — this is
  // scan-plan time, called once per file, and the filter call is
  // self-contained (no other work queued on this stream). User rule
  // forbids the default-stream sentinel everywhere in Sirius.
  rmm::cuda_stream planning_stream;
  // Pick the per-device filter entry that matches the current device for
  // planning-time row-group pruning. Tasks will later pick their own entry at
  // converter time; this planning-time set_filter is just for the metadata
  // stats evaluation on this thread.
  cudf::io::parquet_reader_options planning_options = _reader_options;
  if (_translated_filter_by_device && !_translated_filter_by_device->empty()) {
    int planning_device = 0;
    (void)::cudaGetDevice(&planning_device);
    auto it = _translated_filter_by_device->find(planning_device);
    if (it == _translated_filter_by_device->end()) {
      it = _translated_filter_by_device->begin();  // fallback to any device
    }
    planning_options.set_filter(it->second.back());
  }
  for (std::size_t file_idx = 0; file_idx < _file_paths.size(); ++file_idx) {
    auto row_group_indices = readers[file_idx]->all_row_groups(planning_options);
    if (_translated_filter_by_device && !_translated_filter_by_device->empty()) {
      auto const row_groups_before_pruning = row_group_indices.size();
      // clang-format off
      SIRIUS_LOG_INFO("[parquet_scan_task_global_state] Row group pruning: file: {}\n" \
                      "                                                         before: {}",
                      _file_paths[file_idx],
                      row_groups_before_pruning);
      // clang-format on
      // Prune row groups with filter pushdown using metadata statistics.
      row_group_indices = readers[file_idx]->filter_row_groups_with_stats(
        row_group_indices, planning_options, planning_stream.view());
      auto const row_groups_after_pruning = row_group_indices.size();
      auto const pruned_row_groups        = row_groups_before_pruning - row_groups_after_pruning;
      // clang-format off
      SIRIUS_LOG_INFO("[parquet_scan_task_global_state]                    after: {} (pruned {})",
                      row_groups_after_pruning,
                      pruned_row_groups);
      // clang-format on
    }
    auto const& file_metadata = _file_metadatas[file_idx];

    // Build DuckDB index → parquet column position map for this file by name.
    std::vector<size_t> parquet_col_indices;
    parquet_col_indices.reserve(_selected_column_indices.size());
    std::unordered_set<size_t> pure_filter_parquet_indices;
    if (!file_metadata.row_groups.empty()) {
      auto const& cols = file_metadata.row_groups.front().columns;
      std::unordered_map<std::string, size_t> name_to_pq_idx;
      for (size_t i = 0; i < cols.size(); ++i) {
        if (!cols[i].meta_data.path_in_schema.empty()) {
          name_to_pq_idx[cols[i].meta_data.path_in_schema[0]] = i;
        }
      }
      for (auto duckdb_idx : _selected_column_indices) {
        if (duckdb_idx < _scan_op->names.size()) {
          auto it = name_to_pq_idx.find(_scan_op->names[duckdb_idx]);
          if (it != name_to_pq_idx.end()) {
            parquet_col_indices.push_back(it->second);
            if (pure_filter_column_indices.count(duckdb_idx)) {
              pure_filter_parquet_indices.insert(it->second);
            }
          }
        }
      }
    }

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

      for (auto const col_idx : parquet_col_indices) {
        auto const& column_metadata = row_group.columns[col_idx].meta_data;
        // To reflect the fact that pure filter columns are not part of the table scan result,
        // we omit them from the uncompressed byte count.
        if (column_metadata.total_uncompressed_size > 0 &&
            !pure_filter_parquet_indices.contains(col_idx)) {
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

void parquet_scan_task_global_state::init_hive_partitions(
  duckdb::MultiFileBindData const& bind_data, sirius_physical_parquet_scan* scan_op)
{
  // Populate metadata from DuckDB's hive_partitioning_indexes if not already
  // populated (e.g. by the public constructor or by schema-based detection).
  if (_hive_partition_columns.empty()) {
    for (auto const& hpi : bind_data.reader_bind.hive_partitioning_indexes) {
      _hive_partition_index_set.insert(hpi.index);
      _hive_partition_columns.push_back(hive_partition_column{hpi.value, hpi.index});
    }
  }

  if (_hive_partition_columns.empty()) return;

  // Build the output column map in the order the pipeline expects.
  //
  // cuDF returns data columns in _selected_column_indices order (which
  // follows column_ids order). We build a DuckDB-index → cuDF-position
  // map, then iterate column_ids to produce the output in the order
  // DuckDB's pipeline operators expect.
  struct col_source {
    bool is_partition;
    size_t data_col_idx;
    std::string partition_name;
    sirius::logical_type type;
  };

  // Map DuckDB primary index → cuDF column position.
  std::unordered_map<size_t, size_t> duckdb_to_cudf;
  for (size_t i = 0; i < _selected_column_indices.size(); ++i) {
    duckdb_to_cudf[_selected_column_indices[i]] = i;
  }

  // Build output_map in column_ids order (the order the pipeline expects).
  std::vector<col_source> output_map;
  std::unordered_set<size_t> seen;
  for (auto const& col_id : scan_op->column_ids) {
    auto primary_idx = col_id.GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) continue;
    if (!seen.insert(primary_idx).second) continue;

    if (_hive_partition_index_set.count(primary_idx)) {
      output_map.push_back(col_source{/* is_partition */ true,
                                      /* data_col_idx */ 0,
                                      scan_op->names[primary_idx],
                                      scan_op->returned_types[primary_idx]});
    } else {
      auto it = duckdb_to_cudf.find(primary_idx);
      if (it != duckdb_to_cudf.end()) {
        output_map.push_back(col_source{/* is_partition */ false,
                                        /* data_col_idx */ it->second,
                                        /* partition_name */ {},
                                        /* type */ {}});
      }
    }
  }

  SIRIUS_LOG_INFO(
    "[parquet_scan] Hive partitions detected: {} partition col(s), {} data col(s), "
    "{} output col(s).",
    _hive_partition_columns.size(),
    duckdb_to_cudf.size(),
    output_map.size());

  _partition_inject_fn = [output_map = std::move(output_map)](
                           std::unique_ptr<cudf::table> tbl,
                           std::string const& file_path,
                           rmm::cuda_stream_view stream) -> std::unique_ptr<cudf::table> {
    if (!tbl || tbl->num_rows() == 0) return tbl;

    auto partitions     = duckdb::HivePartitioning::Parse(file_path);
    auto const num_rows = tbl->num_rows();
    auto data_columns   = tbl->release();  // move columns out, no GPU copy

    std::vector<std::unique_ptr<cudf::column>> output_columns;
    output_columns.reserve(output_map.size());

    for (auto const& src : output_map) {
      if (!src.is_partition) {
        output_columns.push_back(std::move(data_columns[src.data_col_idx]));
      } else {
        auto it = partitions.find(src.partition_name);
        if (it == partitions.end()) {
          throw std::runtime_error("[parquet_scan] Missing hive partition key '" +
                                   src.partition_name + "' in file path: " + file_path);
        }
        // DefaultCastAs requires a DuckDB type; the scalar factory takes the sirius type.
        auto duckdb_val = duckdb::Value(it->second).DefaultCastAs(sirius::to_duckdb(src.type));
        auto scalar     = sirius::value_to_cudf_scalar(duckdb_val, src.type, stream);
        output_columns.push_back(cudf::make_column_from_scalar(*scalar, num_rows, stream));
      }
    }

    return std::make_unique<cudf::table>(std::move(output_columns));
  };
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
    auto& pipelineable_output_data = dynamic_cast<op::pipelineable_operator_data&>(*output_data);
    std::size_t output_bytes       = 0;
    for (const auto& batch : pipelineable_output_data.get_data_batches()) {
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

  // [mgpu-probe] entry instrumentation (08-07 gap-closure).
  // Captures the device/stream context AT THE UPSTREAM H2D frame boundary,
  // before the read_range_into_allocation -> prefetched_data_source H2D chain
  // runs. If current_device != preferred_device_id at this point, the
  // hazard is hypothesis A (upstream is wrong-device) and the subsequent
  // converter entry will observe the same mismatch. If current_device matches
  // here but mismatches at the converter entry breadcrumb in
  // host_parquet_representation_converters.cpp:~89, a frame between
  // compute_task and lock_or_prepare_batch is switching device context.
  {
    int current_device = -1;
    (void)cudaGetDevice(&current_device);
    // Phase 9 FIX-A: two-tier preferred_device_id lookup (local-wins-over-global).
    // Mirrors gpu_pipeline_task::get_preferred_device_id (gpu_pipeline_task.hpp:188-194).
    // Probe reports the EFFECTIVE value that _datasource construction below will see.
    auto const local_preferred_probe = l_state.get_preferred_device_id();
    auto const preferred_probe       = local_preferred_probe.has_value()
      ? local_preferred_probe
      : g_state.get_preferred_device_id();
    auto* memspace_probe       = l_state.get_memory_space();
    SIRIUS_LOG_INFO(
      "[mgpu-probe] parquet_scan_task::compute_task entry current_device={} stream={} "
      "preferred_device_id={} memspace_device_id={}",
      current_device,
      static_cast<void*>(stream.value()),
      preferred_probe.value_or(-1),
      memspace_probe != nullptr ? memspace_probe->get_device_id() : -1);
  }

  if (!_datasource) {
    // IO-05 + IO-04: route the per-task datasource construction to the
    // per-GPU cucascade backend selected by preferred_device_id.
    //
    // parquet_scan_task is a sirius_pipeline_itask (NOT a gpu_pipeline_task),
    // so the two-tier local_state/global_state get_preferred_device_id() helper
    // from gpu_pipeline_task is not directly available. We consult the
    // global_state's pipeline-level preferred device (set on
    // sirius_pipeline_task_global_state base) when present. Today, the
    // pipeline_executor routes non-gpu_pipeline_task instances to the first GPU
    // executor by default (pipeline_executor.cpp:237-244), so parquet_scan_task
    // effectively runs on the first GPU when no explicit preference is set —
    // we mirror that behavior here by falling back to the first configured
    // backend. This keeps the adapter construction aligned with the actual
    // executor-routing decision and avoids silent context mismatch.
    auto const& backends = g_state.get_gpu_io_backends();
    if (backends.empty()) {
      throw std::runtime_error(
        "[parquet_scan_task::compute_task] no GPU io_backends configured — "
        "SiriusContext::initialize() must have populated at least one "
        "(Approach C seeding via task_creator required)");
    }
    // Phase 9 FIX-A: two-tier lookup (local-wins-over-global). See also the
    // same idiom in the [mgpu-probe] entry breadcrumb above — both must
    // produce the SAME value for the probe log to match the actual routing.
    auto const local_preferred = l_state.get_preferred_device_id();
    auto const preferred       = local_preferred.has_value()
      ? local_preferred
      : g_state.get_preferred_device_id();
    auto backend_it =
      preferred.has_value() ? backends.find(*preferred) : backends.begin();
    if (backend_it == backends.end()) {
      throw std::out_of_range(
        "[parquet_scan_task::compute_task] no io_backend for device_id=" +
        std::to_string(preferred.value_or(-1)));
    }
    auto const& file_path = g_state.get_file_path(l_state.get_file_idx());
    auto const file_size  = g_state.get_file_size(l_state.get_file_idx());
    _datasource           = std::make_shared<sirius::io::cucascade_datasource>(
      backend_it->second, std::filesystem::path{file_path}, file_size);
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
                                                  g_state.get_filter_expression_by_device(),
                                                  g_state.get_post_filter_projection_ids());

  // Propagate hooks and data-file path to the converter.
  if (g_state.has_post_convert_fn()) {
    parquet_representation->set_post_convert_fn(g_state.get_post_convert_fn());
  }
  if (g_state.has_hive_partitions()) {
    parquet_representation->set_partition_inject_fn(g_state.get_partition_inject_fn());
  }
  if (g_state.has_post_convert_fn() || g_state.has_hive_partitions()) {
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
  auto result = std::make_unique<op::pipelineable_operator_data>(
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
  auto& pipelineable_output = dynamic_cast<op::pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.release_data_batches()) {
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
