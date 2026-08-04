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

// standard library
#include <atomic>
#include <memory>
#include <optional>

namespace sirius::scan_manager {
class split_connector;
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::late_mat {
struct deferred_scan_output;  // late_mat/defer_directive.hpp
struct planned_deferral;      // late_mat/plan_deferral.hpp
struct planned_fd_graph;      // late_mat/plan_deferral.hpp
}  // namespace sirius::late_mat

namespace sirius::op {
class sirius_dynamic_filter_set;  // membership channel (op/sirius_dynamic_filter.hpp)
}

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
 * 1. The plan generator creates the ingestible and the scan operator.
 * 2. prepare_for_query checks whether a pinned entry can serve the scan.
 * 3. A cache miss uses split_provider. A cache hit uses cached_databatch_provider.
 * 4. Both providers send inputs through the operator's split connector.
 * 5. execute() runs each split through materialize_table, then
 *    post_filter_and_project unless the result is already
 *    row-filtered and projected.
 *
 * The operator retains the ingestible created by the plan generator.
 */
class sirius_gpu_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_SCAN;

  /**
   * @param types                  Output column types in plan order.
   * @param estimated_cardinality  Planner-estimated row count.
   * @param ingestible             Per-table source built by the plan generator.
   */
  sirius_gpu_scan_operator(duckdb::vector<sirius::logical_type> types,
                           duckdb::idx_t estimated_cardinality,
                           std::shared_ptr<gpu_ingestible> ingestible);

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
   * The input is a @c scan_operator_input holding either scan metadata
   * (fresh read) or a resident batch (pinned-cache hit). Both go through
   * @c gpu_ingestible::materialize_table; @c post_filter_and_project runs
   * afterwards unless materialize returned
   * @c filter_state::ROW_FILTERED_AND_PROJECTED.
   *
   * Throws on any other operator_data type (programmer error: the connector
   * carries only @c scan_operator_input).
   */
  std::unique_ptr<op::operator_data> execute(const op::operator_data& input_data,
                                             rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

  /**
   * @brief Const accessor for the ingestible's table_info. Called by the
   *        pipeline converter's scan-identity dump.
   */
  [[nodiscard]] const ingestible_table_info& peek_table_info() const;

  [[nodiscard]] gpu_ingestible& get_ingestible() const;

  scan_manager::split_connector& get_split_connector();

  /// Shared handle to the connector, for components (e.g. the memory
  /// prefetcher) that must outlive-safely reference it from a background
  /// thread.
  [[nodiscard]] std::shared_ptr<scan_manager::split_connector> get_shared_split_connector() const
  {
    return _split_connector;
  }

  /// Late-mat deferral directive (SIRIUS_EXP_LATE_MAT): installed by the
  /// defer policy at query prepare, always in a pair with the consuming
  /// operator's late_mat_port_directive. execute() substitutes the listed
  /// output positions with a UINT64 pin-order rowid column (first position)
  /// and INT8 placeholders (the rest). Empty when the gate is off.
  std::shared_ptr<const late_mat::deferred_scan_output> late_mat_defer;

  /// v2 planner annotation (SIRIUS_EXP_LATE_MAT_V2): per-output-column
  /// lifetime facts from the plan-time pass. Read by the lowering backend at
  /// query prepare; empty when the sub-gate is off.
  std::shared_ptr<const late_mat::planned_deferral> late_mat_plan;

  /// v3 raw FD graph (SIRIUS_EXP_LATE_MAT_V3): ONE query-wide graph shared by
  /// every scan (equality edges + aggregate key provenances). The lowering
  /// runs the determination closure against the pinned entries' uniqueness
  /// facts. Empty below the v3 gate.
  std::shared_ptr<const late_mat::planned_fd_graph> late_mat_fd_graph;

 private:
  std::shared_ptr<gpu_ingestible> _ingestible;
  std::shared_ptr<scan_manager::split_connector> _split_connector;
  /// RULE-2 bail latch for the fused scan-filter pipeline, shared with every
  /// split this operator hands out (see scan_operator_input::fused_bail_flag).
  /// Per-operator so another query's scan of the same pinned entry decides
  /// fresh.
  std::shared_ptr<std::atomic<bool>> _fused_rule2_bailed =
    std::make_shared<std::atomic<bool>>(false);
  /// The scan's dynamic-filter channel (null for non-parquet ingestibles),
  /// resolved once at construction and stamped onto every split so
  /// prepare_for_processing can snapshot membership filters at DECODE time
  /// (see scan_operator_input::dynamic_filters).
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> _dynamic_filters_channel;
};

}  // namespace sirius::op::scan
