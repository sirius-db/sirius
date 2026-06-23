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
#include <io/gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>

// standard library
#include <memory>
#include <mutex>
#include <optional>

namespace sirius::scan_manager {
class split_connector;
class sirius_scan_manager;
}  // namespace sirius::scan_manager

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
 * work to an installed @c io::gpu_ingestible, and wraps the result for the
 * downstream pipeline.
 *
 * Lifecycle:
 *   1. Pipeline converter constructs the operator with the per-table
 *      @c ingestible_table_info (parquet or duckdb-native today).
 *   2. @c sirius_scan_manager::prepare_for_query peeks the table_info to
 *      match pinned-cache entries. On match: builds the cached ingestible
 *      from the stolen table_info. On miss: calls
 *      @c io::make_gpu_ingestible(take_table_info(), mgr, gpu_ioctxs).
 *      Either way, scan_manager calls @ref install_ingestible to hand
 *      the operator its source.
 *   3. The scan_manager also installs a fresh @c split_connector and
 *      drives a @c scan_manager::split_provider that composes the same
 *      ingestible — the provider populates the connector with splits
 *      that the operator pulls via @ref get_next_task_input_data.
 *   4. @ref execute dispatches each split through the installed
 *      ingestible's @c materialize_table and (conditionally)
 *      @c post_filter_and_project.
 */
class sirius_gpu_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_SCAN;

  /**
   * @param types                  Output column types in plan order.
   * @param estimated_cardinality  Planner-estimated row count.
   * @param table_info             Per-table bind data; consumed by
   *                               @c sirius_scan_manager during
   *                               @c prepare_for_query.
   */
  sirius_gpu_scan_operator(duckdb::vector<sirius::logical_type> types,
                           duckdb::idx_t estimated_cardinality,
                           std::unique_ptr<io::ingestible_table_info> table_info);

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

  // -----------------------------
  // scan_manager wiring (public for simplicity; scan_manager is the only caller)
  // -----------------------------
  /**
   * @brief Take ownership of the per-table bind data. Called by
   *        @c sirius_scan_manager::prepare_for_query when the pinned-cache
   *        short-circuit either misses (and the ingestible is built via
   *        @c io::make_gpu_ingestible) or wins (the cached ingestible
   *        keeps the table_info alive for parquet-bind-data accessors).
   *        After the call returns, @c peek_table_info would dereference
   *        null — callers MUST peek BEFORE taking.
   */
  std::unique_ptr<io::ingestible_table_info> take_table_info();

  /**
   * @brief Const accessor for the parked table_info. Used by
   *        @c sirius_scan_manager::try_make_cached_ingestible to match
   *        the operator's file paths against pinned entries.
   *
   * @pre @c take_table_info has not been called yet.
   */
  [[nodiscard]] io::ingestible_table_info const& peek_table_info() const noexcept
  {
    return *_table_info;
  }

  /**
   * @brief Install the ingestible that will drive both the operator's
   *        per-split execute and the @c split_provider's metadata
   *        emission. Called by @c sirius_scan_manager exactly once per
   *        query.
   */
  void install_ingestible(std::shared_ptr<io::gpu_ingestible> ingestible);

  [[nodiscard]] io::gpu_ingestible& get_ingestible() const;

  void set_split_connector(std::unique_ptr<scan_manager::split_connector> c);
  [[nodiscard]] scan_manager::split_connector* get_split_connector() noexcept
  {
    return _split_connector.get();
  }

 private:
  std::unique_ptr<io::ingestible_table_info> _table_info;
  std::shared_ptr<io::gpu_ingestible> _ingestible;
  std::unique_ptr<scan_manager::split_connector> _split_connector;
  // Serializes split-pull and pre_assigned_batch_id assignment so that batch IDs
  // reflect split-pull order (scan order) rather than task-completion order.
  std::mutex _scan_order_mutex_;
};

}  // namespace sirius::op::scan
