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
#include "sirius_config.hpp"

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
}  // namespace sirius::io

namespace sirius::scan_manager {
class split_provider;
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

/**
 * @brief Polymorphic base for per-format scan bind-data.
 *
 * Carries the fields common to every file-format scan. Subclasses add
 * format-specific fields and implement @ref make_provider. Populated by
 * the pipeline converter; consumed once via the operator's
 * take_scan_info().
 */
struct scan_info {
  /// Logical types of the scan source's columns, in source schema order.
  duckdb::vector<sirius::logical_type> returned_types;
  std::vector<std::string> file_paths;
  /// Column ids exposed by the table function.
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  /// Indices into @ref column_ids that the planner projected (empty = read all columns).
  duckdb::vector<duckdb::idx_t> projection_ids;
  /// Column names in source schema order.
  duckdb::vector<std::string> names;
  /// Filter set for row-group pruning / pushdown. May be null.
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  duckdb::vector<duckdb::HivePartitioningIndex> partition_indices;
  /// Target uncompressed bytes per row-group partition.
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  /// Number of output columns the gpu scan operator returns.
  /// Set by the pipeline converter from the scan operator's `types.size()`.
  std::size_t scan_output_arity = 0;

  scan_info()                            = default;
  virtual ~scan_info()                   = default;
  scan_info(scan_info const&)            = delete;
  scan_info& operator=(scan_info const&) = delete;
  scan_info(scan_info&&)                 = default;
  scan_info& operator=(scan_info&&)      = default;

  /**
   * @brief Build the format-specific split_provider this scan_info dispatches to.
   *
   * Called once by the scan_manager after the cache short-circuit misses.
   * May move fields off @c *this into the provider.
   *
   * @param manager     Owning scan_manager, forwarded so the format provider can
   *                    route each path to its backend (s3:// -> s3_ioctx via
   *                    @c io_ctx_shared_for, local -> per-GPU uring). Combined-
   *                    runtime S3 integration layered on upstream #749.
   * @param gpu_ioctxs  Per-GPU sirius_ioctx map forwarded to the underlying
   *                    split_provider for multi-GPU IO routing. An empty map is
   *                    accepted only at the provider / test-fallback layer (the
   *                    provider then routes local paths through cudf's bundled
   *                    datasource factory); the live GPU scan path requires a
   *                    non-empty map and throws otherwise
   *                    (sirius_gpu_parquet_scan_operator).
   */
  virtual std::unique_ptr<scan_manager::split_provider> make_provider(
    scan_manager::sirius_scan_manager& manager,
    std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs) = 0;
};

}  // namespace sirius::op::scan
