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

#include <op/scan/parquet_scan_task.hpp>
#include <op/sirius_physical_iceberg_scan.hpp>

#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Iceberg Delete State
//===----------------------------------------------------------------------===//

/**
 * @brief Delete state for an Iceberg V2 scan.
 *
 * Constructed once at query-start time in iceberg_scan_task_global_state.
 * Shared (via shared_ptr captured in the post-convert lambda) across all
 * parquet_scan_tasks that process data files from the same Iceberg table.
 */
struct iceberg_delete_state {
  /// Per-data-file sorted positional deletes.
  /// Key   = canonical data-file path.
  /// Value = sorted list of 0-based absolute row positions to delete.
  std::unordered_map<std::string, std::vector<int64_t>> positional_deletes;
};

/**
 * @brief State for Iceberg V2 equality deletes.
 *
 * Holds the deduplicated equality-delete key table on the GPU and a pre-built
 * `cudf::distinct_hash_join` for fast probe operations.  Shared across all
 * scan tasks for the same table (captured by shared_ptr in the hook lambda).
 *
 * For each data-chunk the post-convert hook probes the hash join with the
 * chunk's key columns, then downloads the per-row match result (build_indices)
 * to the host, builds a boolean keep-mask, and applies
 * `cudf::apply_boolean_mask`.  The sync cost is small because build_indices is
 * at most one INT32 per data row (~400 KB for a 100 K-row batch).
 */
struct iceberg_equality_delete_state {
  /// De-duplicated equality-delete rows projected to key columns only.
  /// Kept alive while `hash_join` is in use (hash_join holds a table_view ref).
  std::unique_ptr<cudf::table> delete_key_table;

  /// Pre-built distinct hash join (build side = delete_key_table).
  std::unique_ptr<cudf::distinct_hash_join> hash_join;

  /// Indices into the data-chunk that correspond to the equality key columns.
  /// Parallel to the columns of delete_key_table.
  std::vector<cudf::size_type> data_key_indices;
};

//===----------------------------------------------------------------------===//
// Iceberg Scan Task Global State
//===----------------------------------------------------------------------===//

/**
 * @brief Global state for an Iceberg scan task.
 *
 * Inherits the parquet-scan task global-state machinery (footer reads,
 * row-group partitioning, reader options) via the protected constructor that
 * accepts pre-resolved file paths and column indices.
 *
 * Additionally:
 *  - For V1 tables (no delete files on sirius_physical_iceberg_scan) the
 *    behaviour is identical to plain parquet_scan_task_global_state.
 *  - For V2 tables with positional deletes, loads the delete file data into
 *    a host-side map and installs a post-convert hook that filters the deleted
 *    rows from each row-group batch on the GPU before it is pushed down the
 *    pipeline.
 *  - V2 equality deletes: reads delete files, builds a distinct_hash_join,
 *    and installs a hook that applies an anti-join per row-group batch.
 */
class iceberg_scan_task_global_state : public parquet_scan_task_global_state {
 public:
  /**
   * @brief Construct the global state for an iceberg scan.
   *
   * Extracts data file paths from bind_data, computes column projection, calls
   * the protected parquet_scan_task_global_state constructor, then builds the
   * delete state (if any) from the delete-file lists on @p scan_op.
   *
   * @param pipeline             The pipeline for this scan.
   * @param scan_op              The physical iceberg scan operator.
   * @param approximate_batch_size  Target uncompressed batch size.
   */
  iceberg_scan_task_global_state(
    duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
    sirius_physical_iceberg_scan* scan_op,
    size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE);

 private:
  // -------------------------------------------------------------------------
  // Two-stage construction helpers
  // -------------------------------------------------------------------------

  /// Pre-computed init data extracted before base-class initialisation.
  struct init_data {
    std::vector<std::string> file_paths;
    std::vector<size_t> selected_column_indices;
  };

  /// Extract file paths and column indices from the scan operator.
  static init_data prepare(sirius_physical_iceberg_scan* scan_op);

  /// Private delegating constructor — receives pre-computed @p init so that
  /// the base constructor can be called in the member initialiser list.
  iceberg_scan_task_global_state(
    duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
    sirius_physical_iceberg_scan* scan_op,
    init_data init,
    size_t approximate_batch_size);

  // -------------------------------------------------------------------------
  // Delete-state construction
  // -------------------------------------------------------------------------

  /**
   * @brief Read the delete files listed on @p scan_op and install the
   * post-convert hook if there are any deletes to apply.
   */
  void build_delete_state(sirius_physical_iceberg_scan* scan_op);

  // -------------------------------------------------------------------------
  // Fields
  // -------------------------------------------------------------------------
  std::shared_ptr<iceberg_delete_state>          _delete_state;
  std::shared_ptr<iceberg_equality_delete_state> _equality_delete_state;
};

}  // namespace sirius::op::scan
