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
#include <op/scan/parquet_schema_mapping.hpp>  // detail::leaf_indices_for_column
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/scan/sirius_parquet_metadata_scan_operator.hpp>
#include <sirius/exception.hpp>

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
    _approximate_batch_size(approximate_batch_size),
    _max_file_processed(max_file_processed),
    _total_files(file_paths.size()),
    _gpu_scan(gpu_scan)
{
  // Any non-trivial scan shape — reader-side projection, filter pushdown, or hive-partition
  // injection — needs column names for reader set_column_names / AST name resolution /
  // HivePartitioning::Parse lookups. The plain "read everything and emit naturally" shape
  // does not, so we only reject empty names when the scan actually needs them.
  bool const needs_names = !projection_ids.empty() ||
                           (table_filter_set && !table_filter_set->filters.empty()) ||
                           !partition_indices.empty();
  if (needs_names && names.empty()) {
    throw sirius::internal_exception(
      "[sirius_parquet_metadata_scan_operator] Projection, filter pushdown, or hive partitions "
      "require column names to be provided.");
  }

  // One canonical plan: data columns (D-order), hive partitions, output layout, C→D map.
  // Everything downstream (reader projection, filter expression, post-read injection, row-group
  // byte accounting) reads from this single structure.
  _plan = build_scan_plan(
    column_ids, projection_ids, names, returned_types, this->types.size(), partition_indices);

  // Install the post-read assembly closure. The closure handles both hive-partition injection
  // and pure-filter-column pruning by consuming scan_plan::output_layout; it returns a nullptr
  // closure when the plan is an identity (no partitions and no pure-filter columns), in which
  // case the GPU scan operator skips the assembly step altogether.
  if (auto inject_fn = _plan.build_inject_fn()) {
    _gpu_scan->set_hive_partition_inject_fn(std::move(inject_fn));
  }

  // Build the DuckDB filter expression. AST translation is deferred to execute() so that a
  // task-local CUDA stream can be used. Filters on hive-partition columns are dropped because
  // those columns aren't in the parquet file (DuckDB prunes them at the file-list level).
  if (table_filter_set && !table_filter_set->filters.empty()) {
    auto batch_column_map  = _plan.make_batch_column_map();
    auto duckdb_expression = convert_table_filters_to_expression(*table_filter_set,
                                                                 column_ids,
                                                                 returned_types,
                                                                 batch_column_map,
                                                                 _plan.partition_primary_indices);
    if (duckdb_expression) { _duckdb_filter_expression = std::move(duckdb_expression); }
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
{
  return _next_file_idx.load(std::memory_order_relaxed) >= _total_files;
}

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
  auto const data_column_names = _plan.data_column_names();
  result->reader_options       = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());

  // Tell the parquet reader which columns to produce. Required whenever the scan
  // is projected / has hive partitions to remove.
  if (_plan.is_projected()) { result->reader_options->set_column_names(data_column_names); }

  // Translate the filter to a cudf AST for reader-side pushdown, falling back to a post-read
  // DuckDB-expression evaluation when translation isn't possible. Partition-column filters
  // have already been dropped at construction; anything remaining references data columns.
  std::shared_ptr<translated_expression> ast_filter;
  if (_duckdb_filter_expression) {
    // Resolver maps the BoundReferenceExpression's batch position (D) to the corresponding
    // parquet column name. scan_plan::batch_column_name is the single source of truth for
    // this D→name mapping; the previously-cached _column_name_by_ref was C-indexed and
    // silently wrong whenever projection reordered or dropped columns.

    /// KEVIN: There is a stream/concurrency bug in the AST filter at large scale factors. Looking
    /// into it.

    // gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    // auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
    //   return _plan.batch_column_name(ref_index);
    // };
    // auto optional_filter =
    //   translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
    // stream.synchronize();
    // if (optional_filter) {
    //   ast_filter = std::make_shared<translated_expression>(std::move(*optional_filter));
    //   result->reader_options->set_filter(ast_filter->back());
    //   result->filter_expression = ast_filter;
    //   SIRIUS_LOG_DEBUG(
    //     "[sirius_parquet_metadata_scan_operator] Translated filter expression for pushdown.");
    // } else {
    //   result->filter_expression = _duckdb_filter_expression;
    //   SIRIUS_LOG_DEBUG(
    //     "[sirius_parquet_metadata_scan_operator] AST translation failed; filter will be applied "
    //     "post-read by the GPU scan operator.");
    // }
    result->filter_expression = _duckdb_filter_expression;
  }
  // post_filter_projection_ids is intentionally left empty: the scan_plan-backed inject
  // closure handles output assembly (data + partition + pure-filter drop) in one pass, so
  // the GPU scan operator's separate pruning step is a no-op for this path.

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
    if (_plan.is_projected() && !detail::projected_columns_are_flat(metadata, data_column_names)) {
      /// TODO: Support nested column schemas with projection.
      throw std::runtime_error(
        "[sirius_parquet_metadata_scan_operator] Parquet scans with projections currently only "
        "support flat projected columns.");
    }

    //===----------Resolve selected DuckDB columns to parquet column chunk indices----------===//
    // row_group.columns is indexed in parquet schema-leaf order (preorder), which can differ from
    // DuckDB's logical column order. Resolve by name per file (chunk order is consistent across row
    // groups in a single file, but can vary across files).
    std::vector<std::size_t> selected_chunk_indices;
    std::unordered_set<std::size_t> pure_filter_chunk_indices;
    if (_plan.is_projected()) {
      auto const pure_filter_positions = _plan.pure_filter_batch_positions();
      selected_chunk_indices.reserve(data_column_names.size());
      for (std::size_t k = 0; k < data_column_names.size(); ++k) {
        auto leaves = detail::leaf_indices_for_column(metadata, data_column_names[k]);
        // projected_columns_are_flat (checked above) guarantees exactly one leaf per name.
        if (leaves.size() != 1) {
          throw std::runtime_error(
            "[sirius_parquet_metadata_scan_operator] Projected column '" + data_column_names[k] +
            "' did not resolve to exactly one parquet leaf in file: " + file_path);
        }
        selected_chunk_indices.push_back(leaves.front());
        if (pure_filter_positions.count(k)) { pure_filter_chunk_indices.insert(leaves.front()); }
      }
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

    auto accumulate_chunk = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
      auto const& column_metadata = chunk.meta_data;
      // Pure filter columns are not part of the scan result, so we omit them from the
      // uncompressed byte count used for sizing partitions.
      if (column_metadata.total_uncompressed_size > 0 && !is_pure_filter) {
        partition_uncompressed_bytes +=
          static_cast<std::size_t>(column_metadata.total_uncompressed_size);
      }
      if (column_metadata.total_compressed_size > 0) {
        partition_compressed_bytes +=
          static_cast<std::size_t>(column_metadata.total_compressed_size);
      }
    };

    for (auto const rg_idx : row_group_indices) {
      auto const& row_group = metadata.row_groups[rg_idx];
      partition_rg_indices.push_back(rg_idx);

      if (_plan.is_projected()) {
        for (auto const chunk_idx : selected_chunk_indices) {
          accumulate_chunk(row_group.columns[chunk_idx],
                           pure_filter_chunk_indices.contains(chunk_idx));
        }
      } else {
        // Non-projected: all chunks contribute, no pure-filter pruning.
        for (auto const& chunk : row_group.columns) {
          accumulate_chunk(chunk, false);
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
{
  _gpu_scan->finalize_partitions();
}

}  // namespace sirius::op::scan
