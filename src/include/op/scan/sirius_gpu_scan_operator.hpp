/*
 * Copyright 2026, Sirius Contributors.
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

// sirius
#include <op/scan/gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>

// cudf
#include <cudf/types.hpp>

// standard library
#include <memory>
#include <optional>
#include <vector>

namespace sirius::scan_manager {
class split_connector;
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
/**
 * @brief Unified GPU scan source operator.
 *
 * Replaces the per-format @c sirius_gpu_parquet_scan_operator and
 * @c sirius_gpu_duckdb_native_scan_operator. The operator carries no
 * format-specific code: it pulls @c op::operator_data splits from its
 * bound @c split_connector, delegates per-split materialize/post-process
 * work to an installed @c gpu_ingestible, and wraps the result for the
 * downstream pipeline.
 *
 * Lifecycle:
 *   1. The pipeline converter constructs the operator with its
 *      @c gpu_ingestible (parquet or duckdb-native today).
 *   2. @c sirius_scan_manager::prepare_for_query matches the ingestible's
 *      @c table_info against pinned-cache entries to decide cached-vs-fresh
 *      serving, installs a fresh @c split_connector, and drives a
 *      @c scan_manager::split_provider composing the same ingestible — the
 *      provider populates the connector with splits that the operator
 *      pulls via @ref get_next_task_input_data.
 *   3. @ref execute dispatches each split through the ingestible's
 *      @c materialize_table and (conditionally) @c post_filter_and_project.
 */
class sirius_gpu_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_SCAN;

  /**
   * @param types                  Output column types in plan order.
   * @param estimated_cardinality  Planner-estimated row count.
   * @param ingestible             Source this scan materializes through; its
   *                               @c table_info identifies the scan to the
   *                               scan_manager.
   * @param compressed_materialization_observer  Plan-time counter sink for
   *                               narrowing observability; may be null.
   */
  sirius_gpu_scan_operator(duckdb::vector<sirius::logical_type> types,
                           duckdb::idx_t estimated_cardinality,
                           std::shared_ptr<gpu_ingestible> ingestible,
                           duckdb::SiriusContext* compressed_materialization_observer = nullptr);

  ~sirius_gpu_scan_operator() override;

  // -----------------------------
  // Source interface
  // -----------------------------
  bool is_source() const override { return true; }

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<op::operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /**
   * @brief Produce a data batch for one split.
   *
   * Two input shapes are supported:
   *   - @c scan_operator_input — fresh read; calls
   *     @c gpu_ingestible::materialize_table and (when filter info is
   *     present) @c gpu_ingestible::post_filter_and_project.
   *   - @c scan_operator_with_pinned_table_input — pinned-cache hit;
   *     forwards the batch when no filter info is set, otherwise calls
   *     @c gpu_ingestible::post_filter_and_project on the cached view.
   *
   * Throws when the input is neither type (programmer error: the only
   * operator_data types pushed into the operator's connector are the two
   * above).
   */
  std::unique_ptr<op::operator_data> execute(const op::operator_data& input_data,
                                             rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

  [[nodiscard]] gpu_ingestible& get_ingestible() const;

  scan_manager::split_connector& get_split_connector();

 private:
  std::shared_ptr<gpu_ingestible> _ingestible;
  std::shared_ptr<scan_manager::split_connector> _split_connector;
  /// Non-owning observer. The registered-state shared_ptr owns the context for
  /// at least as long as the query plan; unit-test operators may leave it null.
  duckdb::SiriusContext* _compressed_materialization_observer;
  /// Native cuDF mapping of `types`, computed once at construction; empty when any output type
  /// has no cuDF mapping. Immutable afterwards, so concurrent execute() calls read it safely.
  std::vector<cudf::data_type> _native_physical_types;
};

}  // namespace sirius::op::scan
