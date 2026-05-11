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
#include "op/scan/scan_plan.hpp"
#include "scan_manager/split_provider.hpp"
#include "sirius_config.hpp"

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {
class Expression;
class TableFilterSet;
}  // namespace duckdb

namespace sirius::io {
class sirius_ioctx;
}  // namespace sirius::io

namespace sirius::scan_manager {

class sirius_scan_manager;

/**
 * @brief Split provider that parses parquet metadata and emits one
 *        @c parquet_scan_data per row-group partition.
 *
 * The constructor performs up-front filter / projection / hive partition setup
 * and pre-decomposes the file list into immutable per-task @c file_batch
 * entries. @ref next_split_provider() atomically claims the next batch index
 * and returns a callable that runs the metadata scan when invoked — splitting
 * the claim from the work lets the driver enqueue all batches and have the
 * worker pool process them in parallel. When the index has overshot the batch
 * list, the returned callable yields an empty vector.
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
   *                                the original operator). Used to split
   *                                @p projection_ids into output vs pure-filter
   *                                columns when building the scan_plan.
   * @param table_filter_set        Filter set for row-group pruning / pushdown.
   * @param partition_indices       Hive partition indices, if any.
   * @param approximate_batch_size  Target uncompressed bytes per row-group
   *                                partition.
   * @param max_file_processed      Maximum number of files handled by one
   *                                metadata task.
   */
  /// Legacy overload: no @c scan_manager. Used by pre-existing #740-era
  /// tests that construct the provider with no ioctx and expect the
  /// @c cudf::io::datasource::create fallback for every file. @c _scan_manager
  /// stays nullptr; @c run_batch falls through to cudf's own datasource.
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

  /// @param scan_manager  Reference to the engine's @c sirius_scan_manager.
  ///                      Used for per-path ioctx dispatch via
  ///                      @c sirius_scan_manager::io_ctx_shared_for(path):
  ///                      each file in @p file_paths is independently routed
  ///                      to its supporting backend at @c run_batch time.
  ///                      Files whose path is claimed by no backend cause
  ///                      @c run_batch to throw @c std::runtime_error
  ///                      ("no backend supports path: ...").
  parquet_split_provider(duckdb::vector<sirius::logical_type> const& returned_types,
                         std::vector<std::string> const& file_paths,
                         duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                         duckdb::vector<duckdb::idx_t> const& projection_ids,
                         duckdb::vector<std::string> const& names,
                         std::size_t scan_output_arity,
                         duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
                         duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
                         std::size_t approximate_batch_size,
                         std::size_t max_file_processed,
                         sirius_scan_manager& scan_manager);

  ~parquet_split_provider() override;

  parquet_split_provider(const parquet_split_provider&)            = delete;
  parquet_split_provider& operator=(const parquet_split_provider&) = delete;
  parquet_split_provider(parquet_split_provider&&)                 = delete;
  parquet_split_provider& operator=(parquet_split_provider&&)      = delete;

  [[nodiscard]] bool has_more_splits() const override;

  /// \brief Atomically claim the next batch index and return a callable that
  ///        runs the metadata scan for it. Once every batch has been claimed,
  ///        @c next_split_provider returns @c nullptr — base @c run() skips
  ///        the empty slot rather than enqueueing a no-op task.
  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

 private:
  /// Internal delegating ctor that both public ctors forward to. Carrying
  /// scan_manager as a pointer lets the legacy (no-scan_manager) public
  /// ctor delegate with nullptr, falling through to
  /// @c cudf::io::datasource::create in @c run_batch.
  parquet_split_provider(duckdb::vector<sirius::logical_type> const& returned_types,
                         std::vector<std::string> const& file_paths,
                         duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                         duckdb::vector<duckdb::idx_t> const& projection_ids,
                         duckdb::vector<std::string> const& names,
                         std::size_t scan_output_arity,
                         duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
                         duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
                         std::size_t approximate_batch_size,
                         std::size_t max_file_processed,
                         sirius_scan_manager* scan_manager);

 private:
  struct file_batch {
    std::vector<std::string> file_paths;
  };

  /// \brief Run the metadata-scan logic for one batch, appending one
  ///        @c parquet_scan_data per emitted partition to @p out.
  void run_batch(file_batch const& batch, std::vector<std::unique_ptr<op::operator_data>>& out);

  std::vector<std::string> _file_paths;
  /// Canonical scan plan — data columns (D order), partition columns, output layout,
  /// and C→D filter map. Held as a shared_ptr<const> so each parquet_scan_data can
  /// carry it to the GPU scan operator's per-task AST translation without copying.
  std::shared_ptr<op::scan::scan_plan const> _plan;
  /// The coalesced DuckDB filter expression (AST translation attempted in run_batch()).
  /// Empty when no filters were translatable (after skipping partition-column filters).
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;

  std::size_t _approximate_batch_size;
  std::size_t _max_file_processed;
  std::size_t _total_files;
  /// Scan manager reference for per-path ioctx dispatch. Resolved per-file
  /// in @c run_batch via @c _scan_manager->io_ctx_shared_for(file_path) so
  /// mixed-backend batches (e.g. local file:// + s3://) each read through
  /// their supporting backend. May be nullptr only for test fixtures that
  /// build the provider without any scan_manager — production paths
  /// (@c create_provider_for) always set this to a live manager.
  sirius_scan_manager* _scan_manager{nullptr};

  /// Pre-decomposed file batches built once in the constructor; immutable
  /// thereafter. Each callable returned by next_split_provider() processes
  /// one entry.
  std::vector<file_batch> _batches;
  /// Atomically incremented to claim the next batch index. Lets multiple
  /// workers process distinct batches in parallel with no mutex.
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::scan_manager
