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

#include "scan_manager/parquet_split_provider.hpp"

#include "exec/thread_pool.hpp"
#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/scan/parquet_schema_mapping.hpp"
#include "op/scan/scan_utils.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

parquet_split_provider::parquet_split_provider(
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<std::string> const& file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names,
  std::size_t scan_output_arity,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
  duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
  std::size_t approximate_batch_size,
  std::size_t max_file_processed)
  : _file_paths(file_paths),
    _approximate_batch_size(approximate_batch_size),
    _max_file_processed(max_file_processed),
    _total_files(file_paths.size())
{
  // Any non-trivial scan shape — reader-side projection, filter pushdown, or hive-partition
  // injection — needs column names for reader set_column_names / AST name resolution /
  // HivePartitioning::Parse lookups.
  bool const needs_names = !projection_ids.empty() ||
                           (table_filter_set && !table_filter_set->filters.empty()) ||
                           !partition_indices.empty();
  if (needs_names && names.empty()) {
    throw sirius::internal_exception(
      "[parquet_split_provider] Projection, filter pushdown, or hive partitions "
      "require column names to be provided.");
  }

  // One canonical plan: data columns (D-order), hive partitions, output layout, C→D map.
  // Everything downstream (reader projection, filter expression, post-read injection, row-group
  // byte accounting) reads from this single structure. Held by shared_ptr so each
  // parquet_scan_data can carry it to the GPU scan operator without copying.
  _plan = std::make_shared<op::scan::scan_plan const>(op::scan::build_scan_plan(
    column_ids, projection_ids, names, returned_types, scan_output_arity, partition_indices));

  // Install the post-read assembly closure. The closure handles both hive-partition injection
  // and pure-filter-column pruning by consuming scan_plan::output_layout; it returns a nullptr
  // closure when the plan is an identity (no partitions and no pure-filter columns), in which
  // case the GPU scan operator skips the assembly step altogether.
  if (auto inject_fn = _plan->build_inject_fn()) { _partition_inject_fn = std::move(inject_fn); }

  // Build the DuckDB filter expression. AST translation is deferred to execute() so that a
  // task-local CUDA stream can be used. Filters on hive-partition columns are dropped because
  // those columns aren't in the parquet file (DuckDB prunes them at the file-list level).
  if (table_filter_set && !table_filter_set->filters.empty()) {
    auto batch_column_map = _plan->make_batch_column_map();
    auto duckdb_expression =
      op::convert_table_filters_to_expression(*table_filter_set,
                                              column_ids,
                                              returned_types,
                                              batch_column_map,
                                              _plan->partition_primary_indices);
    if (duckdb_expression) { _duckdb_filter_expression = std::move(duckdb_expression); }
  }
}

parquet_split_provider::~parquet_split_provider() = default;

op::scan::partition_inject_fn_t parquet_split_provider::take_partition_inject_fn()
{
  return std::move(_partition_inject_fn);
}

std::optional<parquet_split_provider::file_batch> parquet_split_provider::next_task_input()
{
  if (_next_file_idx >= _total_files) { return std::nullopt; }
  auto const start = _next_file_idx;
  auto const end   = std::min(start + _max_file_processed, _total_files);
  _next_file_idx   = end;

  file_batch batch;
  batch.file_paths.assign(_file_paths.begin() + static_cast<std::ptrdiff_t>(start),
                          _file_paths.begin() + static_cast<std::ptrdiff_t>(end));
  return batch;
}

std::future<void> parquet_split_provider::start(exec::thread_pool& pool, split_connector& connector)
{
  // Drain all batches up-front so we can size the remaining-task counter
  // precisely; the connector closes when the last batch lands.
  std::vector<file_batch> batches;
  while (auto next = next_task_input()) {
    batches.push_back(std::move(*next));
  }

  auto promise = std::make_shared<std::promise<void>>();
  auto future  = promise->get_future();

  if (batches.empty()) {
    connector.close();
    promise->set_value();
    return future;
  }

  auto remaining   = std::make_shared<std::atomic<std::size_t>>(batches.size());
  auto first_error = std::make_shared<std::atomic<bool>>(false);
  auto error_ptr   = std::make_shared<std::exception_ptr>();
  auto error_mutex = std::make_shared<std::mutex>();

  for (auto& batch : batches) {
    pool.schedule([this,
                   batch = std::move(batch),
                   &connector,
                   remaining,
                   promise,
                   first_error,
                   error_ptr,
                   error_mutex]() {
      try {
        run_batch(batch, connector);
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("[parquet_split_provider] metadata scan task failed: {}", e.what());
        bool expected = false;
        if (first_error->compare_exchange_strong(expected, true)) {
          std::lock_guard<std::mutex> lock(*error_mutex);
          *error_ptr = std::current_exception();
        }
      } catch (...) {
        SIRIUS_LOG_ERROR("[parquet_split_provider] metadata scan task failed (unknown)");
        bool expected = false;
        if (first_error->compare_exchange_strong(expected, true)) {
          std::lock_guard<std::mutex> lock(*error_mutex);
          *error_ptr = std::current_exception();
        }
      }
      if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        connector.close();
        if (first_error->load(std::memory_order_acquire)) {
          std::lock_guard<std::mutex> lock(*error_mutex);
          promise->set_exception(*error_ptr);
        } else {
          promise->set_value();
        }
      }
    });
  }
  return future;
}

void parquet_split_provider::run_batch(file_batch const& batch, split_connector& connector)
{
  auto stream = cudf::get_default_stream();

  //===----------Build reader options----------===//
  auto const data_column_names = _plan->data_column_names();
  auto reader_options          = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());

  // Tell the parquet reader which columns to produce. Required whenever the scan
  // is projected / has hive partitions to remove.
  if (_plan->is_projected()) { reader_options->set_column_names(data_column_names); }

  // Translate the filter to a cudf AST for reader-side pushdown, falling back to a post-read
  // DuckDB-expression evaluation when translation isn't possible. Partition-column filters
  // have already been dropped at construction; anything remaining references data columns.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_duckdb_filter_expression) {
    // Resolver maps the BoundReferenceExpression's batch position (D) to the corresponding
    // parquet column name. scan_plan::batch_column_name is the single source of truth for
    // this D→name mapping.
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression =
      translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
    if (ast_expression) {
      reader_options->set_filter(ast_expression->back());
      SIRIUS_LOG_DEBUG(
        "[parquet_split_provider] Translated filter expression for row group pruning.");
    } else {
      SIRIUS_LOG_DEBUG("[parquet_split_provider] AST translation failed for row group pruning.");
    }
  }

  // Loop over files to read footers, parse metadata, and compute row-group partitions.
  std::size_t file_idx = 0;
  for (auto const& file_path : batch.file_paths) {
    //===----------Read metadata footers----------===//
    auto datasource = cudf::io::datasource::create(file_path);

    std::unique_ptr<cudf::io::datasource::buffer> footer_buffer;
    footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

    //===----------Parse metadata----------===//
    op::scan::hybrid_scan_reader reader(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      *reader_options);
    auto metadata = reader.parquet_metadata();

    //===----------Resolve selected DuckDB columns to parquet column chunk indices----------===//
    // row_group.columns is indexed in parquet schema-leaf order (preorder), which can differ from
    // DuckDB's logical column order. Resolve by name per file (chunk order is consistent across row
    // groups in a single file, but can vary across files).
    std::vector<std::size_t> selected_chunk_indices;
    std::unordered_set<std::size_t> pure_filter_chunk_indices;
    if (_plan->is_projected()) {
      auto const pure_filter_positions = _plan->pure_filter_batch_positions();
      selected_chunk_indices.reserve(data_column_names.size());
      for (std::size_t k = 0; k < data_column_names.size(); ++k) {
        auto leaves = op::scan::detail::leaf_indices_for_column(metadata, data_column_names[k]);
        if (leaves.empty()) {
          throw std::runtime_error("[parquet_split_provider] Projected column '" +
                                   data_column_names[k] +
                                   "' not found in parquet file: " + file_path);
        }
        bool const is_pure_filter = pure_filter_positions.count(k);
        for (auto const leaf : leaves) {
          selected_chunk_indices.push_back(leaf);
          if (is_pure_filter) { pure_filter_chunk_indices.insert(leaf); }
        }
      }
    }

    //===----------Row Group Partitioning----------===//
    auto row_group_indices = reader.all_row_groups(*reader_options);
    // Row group pruning with filter pushdown using metadata statistics.
    if (ast_expression) {
      auto const row_groups_before_pruning = row_group_indices.size();
      // clang-format off
      SIRIUS_LOG_DEBUG("[parquet_split_provider] Row group pruning: file: {}\n" \
                       "                                                  before: {}",
                       file_path,
                       row_groups_before_pruning);
      // clang-format on
      // Prune row groups with filter pushdown using metadata statistics.
      row_group_indices =
        reader.filter_row_groups_with_stats(row_group_indices, *reader_options, stream);
      auto const row_groups_after_pruning = row_group_indices.size();
      auto const pruned_row_groups        = row_groups_before_pruning - row_groups_after_pruning;
      // clang-format off
      SIRIUS_LOG_DEBUG("[parquet_split_provider]                     after: {} (pruned {})",
                       row_groups_after_pruning,
                       pruned_row_groups);
      // clang-format on
    }

    std::size_t partition_uncompressed_bytes = 0;
    std::size_t partition_compressed_bytes   = 0;
    std::vector<cudf::size_type> partition_rg_indices;
    partition_rg_indices.reserve(row_group_indices.size());

    auto datasource_shared = std::shared_ptr<cudf::io::datasource>(std::move(datasource));

    auto flush_partition = [&]() {
      if (partition_rg_indices.empty()) { return; }
      op::scan::row_group_range rg{file_idx,
                                   std::move(partition_rg_indices),
                                   partition_uncompressed_bytes,
                                   partition_compressed_bytes};
      partition_rg_indices         = {};
      partition_uncompressed_bytes = 0;
      partition_compressed_bytes   = 0;
      auto split                   = std::make_unique<op::scan::parquet_scan_data>(file_path,
                                                                 std::move(rg),
                                                                 reader_options,
                                                                 _duckdb_filter_expression,
                                                                 datasource_shared,
                                                                 _plan);
      connector.push_split(std::move(split));
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

      if (_plan->is_projected()) {
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
}

}  // namespace sirius::scan_manager
