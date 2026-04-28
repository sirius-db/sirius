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
#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/scan/parquet_schema_mapping.hpp"
#include "op/scan/scan_utils.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/utilities/default_stream.hpp>

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
    _is_projected(!projection_ids.empty()),
    _approximate_batch_size(approximate_batch_size),
    _max_file_processed(max_file_processed),
    _total_files(file_paths.size())
{
  _selected_column_indices =
    op::scan::detail::make_selected_column_indices(column_ids, projection_ids);

  // Hive partition columns live in the DuckDB schema but not in parquet files.
  // Drop them from the selected indices and stash them so the gpu scan op can
  // inject their values from the file path.
  for (auto const& hp_index : partition_indices) {
    _hive_partition_index_set.insert(hp_index.index);
    _hive_partition_columns.push_back(
      op::scan::hive_partition_column{hp_index.value, hp_index.index});
  }
  if (!_hive_partition_index_set.empty()) {
    _selected_column_indices.erase(
      std::remove_if(_selected_column_indices.begin(),
                     _selected_column_indices.end(),
                     [this](std::size_t idx) { return _hive_partition_index_set.count(idx) > 0; }),
      _selected_column_indices.end());

    _hive_partition_inject_fn = op::scan::build_partition_inject_fn(column_ids,
                                                                    names,
                                                                    returned_types,
                                                                    _selected_column_indices,
                                                                    _hive_partition_columns,
                                                                    _hive_partition_index_set);
  }

  // Convert the table filter set into a DuckDB expression. AST translation is
  // deferred to the metadata-scan task so a per-task CUDA stream can be used.
  _has_filter = false;
  if (table_filter_set && !table_filter_set->filters.empty()) {
    auto batch_column_map  = op::build_batch_column_map(projection_ids, column_ids.size());
    auto duckdb_expression = op::convert_table_filters_to_expression(
      *table_filter_set, column_ids, returned_types, batch_column_map, _hive_partition_index_set);
    if (duckdb_expression) {
      _has_filter               = true;
      _duckdb_filter_expression = std::move(duckdb_expression);
      if (!names.empty()) {
        _column_name_by_ref.resize(column_ids.size());
        for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
          _column_name_by_ref[i] = names[column_ids[i].GetPrimaryIndex()];
        }
      }
    }
  }

  if (_is_projected && names.empty()) {
    throw std::runtime_error(
      "[parquet_split_provider] Projection requires column names to be provided.");
  }

  if (_is_projected) {
    for (auto idx : _selected_column_indices) {
      _projected_column_names.push_back(names[idx]);
    }
    if (_has_filter) {
      std::vector<std::size_t> candidate_post_filter_ids;
      for (std::size_t i = 0; i < projection_ids.size(); i++) {
        auto const projection_id = projection_ids[i];
        if (i < scan_output_arity) {
          candidate_post_filter_ids.push_back(projection_id);
        } else {
          // Pure filter column not in the expected output set.
          auto const column_index = column_ids[projection_id].GetPrimaryIndex();
          _pure_filter_column_indices.insert(column_index);
        }
      }
      if (!_pure_filter_column_indices.empty()) {
        _post_filter_projection_ids = std::move(candidate_post_filter_ids);
      }
    }
  }
}

parquet_split_provider::~parquet_split_provider() = default;

op::scan::partition_inject_fn_t parquet_split_provider::take_hive_partition_inject_fn()
{
  return std::move(_hive_partition_inject_fn);
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

  auto reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  if (_is_projected) { reader_options->set_column_names(_projected_column_names); }

  std::variant<std::shared_ptr<op::scan::parquet_scan_data::translated_expression>,
               std::shared_ptr<duckdb::Expression>>
    filter_expression;
  if (_has_filter) { filter_expression = _duckdb_filter_expression; }

  std::size_t file_idx = 0;
  for (auto const& file_path : batch.file_paths) {
    auto datasource = cudf::io::datasource::create(file_path);

    auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

    op::scan::hybrid_scan_reader reader(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      *reader_options);
    auto metadata = reader.parquet_metadata();
    if (_is_projected &&
        !op::scan::detail::projected_columns_are_flat(metadata, _projected_column_names)) {
      throw std::runtime_error(
        "[parquet_split_provider] Parquet scans with projections currently only support flat "
        "projected columns.");
    }

    std::vector<std::size_t> selected_chunk_indices;
    std::unordered_set<std::size_t> pure_filter_chunk_indices;
    if (_is_projected) {
      selected_chunk_indices.reserve(_projected_column_names.size());
      for (std::size_t k = 0; k < _projected_column_names.size(); ++k) {
        auto leaves =
          op::scan::detail::leaf_indices_for_column(metadata, _projected_column_names[k]);
        if (leaves.size() != 1) {
          throw std::runtime_error(
            "[parquet_split_provider] Projected column '" + _projected_column_names[k] +
            "' did not resolve to exactly one parquet leaf in file: " + file_path);
        }
        selected_chunk_indices.push_back(leaves.front());
        if (_pure_filter_column_indices.contains(_selected_column_indices[k])) {
          pure_filter_chunk_indices.insert(leaves.front());
        }
      }
    }

    auto row_group_indices = reader.all_row_groups(*reader_options);

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
                                                                 filter_expression,
                                                                 _post_filter_projection_ids,
                                                                 datasource_shared);
      connector.push_split(std::move(split));
    };

    auto accumulate_chunk = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
      auto const& column_metadata = chunk.meta_data;
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

      if (_is_projected) {
        for (auto const chunk_idx : selected_chunk_indices) {
          accumulate_chunk(row_group.columns[chunk_idx],
                           pure_filter_chunk_indices.contains(chunk_idx));
        }
      } else {
        for (auto const& chunk : row_group.columns) {
          accumulate_chunk(chunk, false);
        }
      }

      if (partition_uncompressed_bytes >= _approximate_batch_size) { flush_partition(); }
    }

    flush_partition();
    ++file_idx;
  }

  (void)stream;  // currently unused; preserved for future AST translation.
}

}  // namespace sirius::scan_manager
