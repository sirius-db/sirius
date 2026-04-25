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

#pragma once

#include "helper/logical_type.hpp"
#include "op/scan/hive_partition.hpp"
#include "scan_manager/split_provider.hpp"
#include "sirius_config.hpp"

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace duckdb {
class Expression;
class TableFilterSet;
}  // namespace duckdb

namespace sirius::scan_manager {

/**
 * @brief Split provider that parses parquet metadata and emits one
 *        @c parquet_scan_data per row-group partition.
 *
 * This provider replaces the previous sirius_parquet_metadata_scan_operator:
 * its constructor performs the same up-front filter / projection / hive
 * partition setup, and start() drives the metadata-scan iteration on the
 * provided thread pool, pushing parquet_scan_data into the connector.
 */
class parquet_split_provider : public split_provider {
 public:
  /// Default number of files processed per metadata-scan task.
  static constexpr std::size_t DEFAULT_MAX_FILE_PROCESSED = 8;

  /**
   * @param returned_types          Types of all columns in the source schema.
   * @param file_paths              Parquet files to scan.
   * @param column_ids              Column ids exposed by the table function.
   * @param projection_ids          Indices into @p column_ids that the planner
   *                                projected (empty = read all columns).
   * @param names                   Column names in schema order.
   * @param scan_output_arity       Number of output columns the gpu scan
   *                                operator will return (== types.size() in
   *                                the original operator). Used to derive the
   *                                post-filter projection ids.
   * @param table_filter_set        Filter set for row-group pruning / pushdown.
   * @param partition_indices       Hive partition indices, if any.
   * @param approximate_batch_size  Target uncompressed bytes per row-group
   *                                partition.
   * @param max_file_processed      Maximum number of files handled by one
   *                                metadata task.
   */
  parquet_split_provider(
    duckdb::vector<sirius::logical_type> const& returned_types,
    std::vector<std::string> const& file_paths,
    duckdb::vector<duckdb::ColumnIndex> const& column_ids,
    duckdb::vector<duckdb::idx_t> const& projection_ids,
    duckdb::vector<std::string> const& names,
    std::size_t scan_output_arity,
    duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set            = nullptr,
    duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices = {},
    std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    std::size_t max_file_processed     = DEFAULT_MAX_FILE_PROCESSED);

  ~parquet_split_provider() override;

  parquet_split_provider(const parquet_split_provider&)            = delete;
  parquet_split_provider& operator=(const parquet_split_provider&) = delete;
  parquet_split_provider(parquet_split_provider&&)                 = delete;
  parquet_split_provider& operator=(parquet_split_provider&&)      = delete;

  /// \brief Take ownership of the hive-partition injection function. The
  ///        caller (typically the pipeline converter) installs it on the gpu
  ///        scan operator. Returns an empty function when the scan has no
  ///        hive partition columns.
  op::scan::partition_inject_fn_t take_hive_partition_inject_fn();

  std::future<void> start(exec::thread_pool& pool, split_connector& connector) override;

 private:
  struct file_batch {
    std::vector<std::string> file_paths;
  };

  /// \brief Pop the next file batch (up to @c _max_file_processed files).
  ///        Returns nullopt when all files have been dispatched.
  std::optional<file_batch> next_task_input();

  /// \brief Run the metadata-scan logic for one batch and push parquet_scan_data
  ///        per partition into @p connector.
  void run_batch(file_batch const& batch, split_connector& connector);

  std::vector<std::string> _file_paths;
  std::vector<std::size_t> _selected_column_indices;
  bool _is_projected;
  std::vector<std::string> _projected_column_names;
  bool _has_filter;
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;
  std::vector<std::string> _column_name_by_ref;
  std::vector<std::size_t> _post_filter_projection_ids;
  std::unordered_set<std::size_t> _pure_filter_column_indices;
  std::unordered_set<std::size_t> _hive_partition_index_set;
  std::vector<op::scan::hive_partition_column> _hive_partition_columns;
  op::scan::partition_inject_fn_t _hive_partition_inject_fn;

  std::size_t _approximate_batch_size;
  std::size_t _max_file_processed;
  std::size_t _total_files;
  std::size_t _next_file_idx{0};
};

}  // namespace sirius::scan_manager
