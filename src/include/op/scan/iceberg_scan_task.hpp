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

#include <io/types.hpp>
#include <op/scan/iceberg_delete_filter.hpp>
#include <op/scan/parquet_scan_task.hpp>
#include <op/sirius_physical_iceberg_scan.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

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
 * Delete handling is delegated to an iceberg_delete_pipeline which composes
 * one or more iceberg_delete_filter stages:
 *  - V1 tables (no delete files): no filters, behaves as plain parquet scan.
 *  - V2 positional deletes: positional_delete_filter.
 *  - V2 equality deletes: equality_delete_filter.
 *  - Combined: both filters chained in the pipeline.
 */
class iceberg_scan_task_global_state : public parquet_scan_task_global_state {
 public:
  /**
   * @brief Construct the global state for an iceberg scan.
   *
   * Extracts data file paths from bind_data, computes column projection, calls
   * the protected parquet_scan_task_global_state constructor, then builds the
   * delete pipeline (if any) from the delete-file lists on @p scan_op.
   *
   * @param pipeline             The pipeline for this scan.
   * @param scan_op              The physical iceberg scan operator.
   * @param approximate_batch_size  Target uncompressed batch size.
   * @param ioctx                Shared sirius_ioctx for I/O.
   */
  iceberg_scan_task_global_state(
    duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
    sirius_physical_iceberg_scan* scan_op,
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
    size_t approximate_batch_size                   = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    std::shared_ptr<sirius::io::sirius_ioctx> ioctx = {});

 private:
  // -------------------------------------------------------------------------
  // Two-stage construction helpers
  // -------------------------------------------------------------------------

  /// Pre-computed init data extracted before base-class initialisation.
  struct init_data {
    std::vector<std::string> file_paths;
    std::vector<size_t> selected_column_indices;
    size_t extra_eq_delete_columns = 0;  ///< columns force-added for eq delete key matching
  };

  /// Extract file paths and column indices from the scan operator.
  static init_data prepare(sirius_physical_iceberg_scan* scan_op);

  /// Private delegating constructor — receives pre-computed @p init so that
  /// the base constructor can be called in the member initialiser list.
  iceberg_scan_task_global_state(
    duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
    sirius_physical_iceberg_scan* scan_op,
    init_data init,
    size_t approximate_batch_size,
    std::shared_ptr<sirius::io::sirius_ioctx> ioctx,
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context);

  // -------------------------------------------------------------------------
  // Delete pipeline construction
  // -------------------------------------------------------------------------

  /**
   * @brief Read the delete files listed on @p scan_op, build filters, and
   * install the pipeline's post-convert hook if there are any deletes to apply.
   */
  void build_delete_pipeline(sirius_physical_iceberg_scan* scan_op,
                             size_t extra_eq_delete_columns = 0);

  // -------------------------------------------------------------------------
  // Fields
  // -------------------------------------------------------------------------

  /// The composable delete pipeline (empty for V1 tables).
  iceberg_delete_pipeline _delete_pipeline;
};

}  // namespace sirius::op::scan
