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
#include <log/logging.hpp>
#include <op/scan/hive_partition.hpp>  // build_partition_inject_fn
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/parquet_scan_task.hpp>  // detail::make_selected_column_indices, detail::projected_columns_are_flat
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/scan/sirius_parquet_metadata_scan_operator.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>

// standard library
#include <algorithm>
#include <stdexcept>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//
sirius_parquet_metadata_scan_operator::sirius_parquet_metadata_scan_operator(
  sirius_gpu_parquet_scan_operator* gpu_scan,
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<sirius::logical_type> const& returned_types,
  duckdb::idx_t estimated_cardinality,
  std::vector<std::string> const& file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
  duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
  std::size_t approximate_batch_size,
  std::size_t max_file_processed)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARQUET_METADATA_SCAN, std::move(types), estimated_cardinality),
    _file_paths(file_paths),
    _is_projected(!projection_ids.empty()),
    _approximate_batch_size(approximate_batch_size),
    _max_file_processed(max_file_processed),
    _total_files(file_paths.size()),
    _gpu_scan(gpu_scan)
{
  _selected_column_indices = detail::make_selected_column_indices(column_ids, projection_ids);

  // Detect hive partition columns — these exist in the DuckDB schema but not in parquet files.
  // Their values come from directory paths (e.g., partition_col=42/).
  // We need to drop hive partition columns from the selected column indices, since these are are
  // injected post-read from the directory path, not read from the parquet file itself.
  for (auto const& hp_index : partition_indices) {
    _hive_partition_index_set.insert(hp_index.index);
    _hive_partition_columns.push_back(hive_partition_column{hp_index.value, hp_index.index});
  }
  if (!_hive_partition_index_set.empty()) {
    _selected_column_indices.erase(
      std::remove_if(_selected_column_indices.begin(),
                     _selected_column_indices.end(),
                     [this](std::size_t idx) { return _hive_partition_index_set.count(idx) > 0; }),
      _selected_column_indices.end());

    // Build and install the post-read partition injection closure on the paired GPU scan
    // operator. The closure interleaves partition-column values parsed from the file path into
    // the cudf table in the order DuckDB expects.
    _gpu_scan->set_hive_partition_inject_fn(build_partition_inject_fn(column_ids,
                                                                      names,
                                                                      returned_types,
                                                                      _selected_column_indices,
                                                                      _hive_partition_columns,
                                                                      _hive_partition_index_set));
  }

  // Convert the table filter set into a DuckDB expression. AST translation is deferred to
  // execute() so that a task-local CUDA stream can be used.
  _has_filter = false;
  if (table_filter_set && !table_filter_set->filters.empty()) {
    auto batch_column_map  = build_batch_column_map(projection_ids, column_ids.size());
    auto duckdb_expression = convert_table_filters_to_expression(
      *table_filter_set, column_ids, returned_types, batch_column_map);
    if (duckdb_expression) {
      _has_filter               = true;
      _duckdb_filter_expression = std::move(duckdb_expression);
      // Pre-compute the ref_index -> column_name mapping for AST translation in execute().
      // If names are empty, AST translation will be skipped and the DuckDB filter used instead.
      if (!names.empty()) {
        _column_name_by_ref.resize(column_ids.size());
        for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
          _column_name_by_ref[i] = names[column_ids[i].GetPrimaryIndex()];
        }
      }
    }
  }

  // Projections require column names to map indices to parquet column names.
  if (_is_projected && names.empty()) {
    throw std::runtime_error(
      "[sirius_parquet_metadata_scan_operator] Projection requires column names to be provided.");
  }

  // Construct a) the post_filter_projection_ids list of projection ids corresponding to columns
  //              that remain after pruning pure filter columns, and
  //           b) the set of column indices corresponding to pure filter columns that will be pruned
  //              after filtering.
  if (_is_projected) {
    for (auto idx : _selected_column_indices) {
      _projected_column_names.push_back(names[idx]);
    }
    if (_has_filter) {
      std::vector<std::size_t> candidate_post_filter_ids;
      for (std::size_t i = 0; i < projection_ids.size(); i++) {
        auto const projection_id = projection_ids[i];
        if (i < this->types.size()) {
          candidate_post_filter_ids.push_back(projection_id);
        } else {
          // This is a pure filter column that is not among the expected output columns.
          auto const column_index = column_ids[projection_id].GetPrimaryIndex();
          _pure_filter_column_indices.insert(column_index);
        }
      }
      // Only set post_filter_projection_ids when there are pure filter columns to prune.
      if (!_pure_filter_column_indices.empty()) {
        _post_filter_projection_ids = std::move(candidate_post_filter_ids);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_parquet_metadata_scan_operator::get_next_task_hint()
{
  if (_next_file_idx.load(std::memory_order_relaxed) < _total_files) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}

bool sirius_parquet_metadata_scan_operator::all_ports_empty()
{ return _next_file_idx.load(std::memory_order_relaxed) >= _total_files; }

std::unique_ptr<operator_data> sirius_parquet_metadata_scan_operator::get_next_task_input_data()
{
  auto const start = _next_file_idx.fetch_add(_max_file_processed, std::memory_order_relaxed);
  if (start >= _total_files) { return nullptr; }

  auto const end = std::min(start + _max_file_processed, _total_files);
  std::vector<std::string> batch_files(_file_paths.begin() + static_cast<ptrdiff_t>(start),
                                       _file_paths.begin() + static_cast<ptrdiff_t>(end));

  return std::make_unique<parquet_metadata_input>(std::move(batch_files), _approximate_batch_size);
}

//===----------------------------------------------------------------------===//
// execute() — metadata parsing
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_parquet_metadata_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  auto const* input_ptr = dynamic_cast<const parquet_metadata_input*>(&input_data);
  if (!input_ptr) {
    throw std::runtime_error(
      "[sirius_parquet_metadata_scan_operator] execute() called with unexpected operator_data "
      "type; expected parquet_metadata_input.");
  }
  auto const& input = *input_ptr;

  auto result        = std::make_unique<partitioned_parquet_metadata>();
  result->file_paths = input.file_paths;

  //===----------Build reader options----------===//
  result->reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());

  // Projections
  if (_is_projected) { result->reader_options->set_column_names(_projected_column_names); }

  // Filter
  std::shared_ptr<translated_expression> ast_filter;
  if (_has_filter) {
    if (!_column_name_by_ref.empty()) {
      gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
      auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
        if (ref_index >= _column_name_by_ref.size()) {
          throw std::runtime_error(
            "[sirius_parquet_metadata_scan_operator] Filter expression contains reference to "
            "column index " +
            std::to_string(ref_index) +
            " which is out of bounds for the provided column metadata.");
        }
        return _column_name_by_ref[ref_index];
      };
      auto optional_filter =
        translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
      if (optional_filter) {
        ast_filter = std::make_shared<translated_expression>(std::move(*optional_filter));
        result->reader_options->set_filter(ast_filter->back());
        result->filter_expression = ast_filter;
        SIRIUS_LOG_DEBUG(
          "[sirius_parquet_metadata_scan_operator] Successfully translated filter expression for "
          "pushdown.");
      } else {
        result->filter_expression = _duckdb_filter_expression;
        SIRIUS_LOG_DEBUG(
          "[sirius_parquet_metadata_scan_operator] Failed to translate filter expression for "
          "pushdown. Filter will be applied in the GPU scan operator.");
      }
    } else {
      // Column names were not provided; AST translation requires column name references.
      result->filter_expression = _duckdb_filter_expression;
      SIRIUS_LOG_DEBUG(
        "[sirius_parquet_metadata_scan_operator] Column names not available for AST filter "
        "translation. Filter will be applied in the GPU scan operator.");
    }
    result->post_filter_projection_ids = _post_filter_projection_ids;
  }

  // Loop over files to read footers, parse metadata, and compute row-group partitions.
  result->datasources.reserve(input.file_paths.size());
  std::size_t file_idx = 0;
  for (auto const& file_path : input.file_paths) {
    //===----------Read metadata footers----------===//
    result->datasources.push_back(cudf::io::datasource::create(file_path));

    std::unique_ptr<cudf::io::datasource::buffer> footer_buffer;
    footer_buffer = cudf::io::parquet::fetch_footer_to_host(*result->datasources.back());

    //===----------Parse metadata----------===//
    hybrid_scan_reader reader(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      *result->reader_options);
    auto metadata = reader.parquet_metadata();
    if (_is_projected && !detail::projected_columns_are_flat(metadata, _projected_column_names)) {
      /// TODO: Support nested column schemas with projection.
      throw std::runtime_error(
        "[sirius_parquet_metadata_scan_operator] Parquet scans with projections currently only "
        "support flat projected columns.");
    }

    //===----------Row Group Partitioning----------===//
    auto row_group_indices = reader.all_row_groups(*result->reader_options);
    // Row group pruning with filter pushdown using metadata statistics.
    if (ast_filter) {
      auto const row_groups_before_pruning = row_group_indices.size();
      // clang-format off
      SIRIUS_LOG_DEBUG("[sirius_parquet_metadata_scan_operator] Row group pruning: file: {}\n" \
                       "                                                         before: {}",
                       file_path,
                       row_groups_before_pruning);
      // clang-format on
      // Prune row groups with filter pushdown using metadata statistics.
      row_group_indices =
        reader.filter_row_groups_with_stats(row_group_indices, *result->reader_options, stream);
      auto const row_groups_after_pruning = row_group_indices.size();
      auto const pruned_row_groups        = row_groups_before_pruning - row_groups_after_pruning;
      // clang-format off
      SIRIUS_LOG_DEBUG("[sirius_parquet_metadata_scan_operator]                    after: {} (pruned {})",
                       row_groups_after_pruning,
                       pruned_row_groups);
      // clang-format on
    }

    std::size_t partition_uncompressed_bytes = 0;
    std::size_t partition_compressed_bytes   = 0;
    std::vector<cudf::size_type> partition_rg_indices;
    partition_rg_indices.reserve(row_group_indices.size());

    auto flush_partition = [&result,
                            &partition_rg_indices,
                            &partition_uncompressed_bytes,
                            &partition_compressed_bytes,
                            &file_idx]() {
      if (partition_rg_indices.empty()) { return; }
      result->row_group_partitions.emplace_back(file_idx,
                                                std::move(partition_rg_indices),
                                                partition_uncompressed_bytes,
                                                partition_compressed_bytes);
      partition_uncompressed_bytes = 0;
      partition_compressed_bytes   = 0;
    };

    for (auto const rg_idx : row_group_indices) {
      auto const& row_group = metadata.row_groups[rg_idx];
      partition_rg_indices.push_back(rg_idx);

      for (auto const col_idx : _selected_column_indices) {
        auto const& column_metadata = row_group.columns[col_idx].meta_data;
        // To reflect the fact that pure filter columns are not part of the table scan result,
        // we omit them from the uncompressed byte count.
        if (column_metadata.total_uncompressed_size > 0 &&
            !_pure_filter_column_indices.contains(col_idx)) {
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

    ++file_idx;
  }

  SIRIUS_LOG_DEBUG(
    "[sirius_parquet_metadata_scan_operator] Parsed {} files, produced {} row-group partitions",
    input.file_paths.size(),
    result->row_group_partitions.size());

  return result;
}

//===----------------------------------------------------------------------===//
// Sink interface — forward accumulated metadata to the paired GPU scan
//===----------------------------------------------------------------------===//
void sirius_parquet_metadata_scan_operator::sink(const operator_data& input_data,
                                                 rmm::cuda_stream_view /*stream*/)
{
  auto const* metadata = dynamic_cast<const partitioned_parquet_metadata*>(&input_data);
  if (!metadata) {
    throw std::runtime_error(
      "[sirius_parquet_metadata_scan_operator] sink() received unexpected operator_data type; "
      "expected partitioned_parquet_metadata.");
  }
  _gpu_scan->accumulate_metadata(*metadata);
}

void sirius_parquet_metadata_scan_operator::finalize_operator()
{ _gpu_scan->finalize_partitions(); }

}  // namespace sirius::op::scan
