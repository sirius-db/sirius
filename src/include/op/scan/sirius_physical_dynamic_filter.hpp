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

#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/read_time_filter_bypass.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <memory>

namespace sirius::op::scan {

/// @brief Why the planner installed this operator -- exposed for plan-shape tests.
enum class dynamic_filter_endpoint_provenance { scan_route, join_edge, top_n_endpoint };

//===----------------------------------------------------------------------===//
// sirius_physical_dynamic_filter
//===----------------------------------------------------------------------===//
/// @brief Applies dynamic filters to the batches flowing through one point in the plan.
///
/// The planner installs this operator in three roles; @ref provenance reports which.
///
/// On a scan route it sits directly above `sirius_gpu_scan_operator`. Parquet has already applied
/// AST-capable filters through the reader, so the endpoint uses @c membership_masks_only. A
/// DuckDB-native scan has no reader filter and uses @c include_ast_row_masks. The mode chosen at
/// plan time is a premise about the serve path; when `prepare_for_query` serves the scan from the
/// pinned cache no reader runs, the shared `read_time_filter_bypass` is marked before execution,
/// and `effective_mode()` promotes membership_masks_only to include_ast_row_masks for the whole
/// execution.
///
/// On a direct route, `planner::place_endpoint()` inserts it in the producing join's probe
/// subtree. A @c dynamic_filter_route_class::direct target accepts membership filters, and the
/// operator uses @c membership_masks_only.
///
/// As a Top-N sited endpoint it sits at a trace stop point that material work separates from the
/// Top-N sink. Top-N boundary filters apply through @ref sirius::op::sirius_compaction_applicable
/// (one fused kernel pass); join zone maps keep the AST row-mask path.
///
/// Each execution snapshots the currently visible filters. The batch passes through unchanged when
/// the channel has no applicable filter, the current device has no replica, or
/// @ref dynamic_filter_gate declines the work. `on_finalize_operator()` closes the channel after
/// the endpoint drains.
class sirius_physical_dynamic_filter : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DYNAMIC_FILTER;

  sirius_physical_dynamic_filter(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<sirius::op::sirius_dynamic_filter_set> filters,
    double gate_keep_threshold     = dynamic_filter_gate::k_default_keep_threshold,
    dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::membership_masks_only,
    dynamic_filter_endpoint_provenance provenance = dynamic_filter_endpoint_provenance::scan_route,
    sirius::op::dynamic_filter_stats* stats       = nullptr,
    std::shared_ptr<read_time_filter_bypass> read_bypass = nullptr);

  /// Why the planner installed this operator. A test-observability seam: plan-shape tests are
  /// the only readers -- no runtime behavior consults it.
  [[nodiscard]] dynamic_filter_endpoint_provenance provenance() const noexcept
  {
    return _provenance;
  }

  /// The mode this execution actually applies: the plan-time mode, promoted to
  /// include_ast_row_masks when the scan's serve path bypassed read-time filtering. Promotion is
  /// monotone (never demotes) and settled before the first batch.
  [[nodiscard]] dynamic_filter_apply_mode effective_mode() const noexcept;

  /// The serve-path latch shared with the wrapped scan; null for join_edge / top_n_endpoint
  /// provenance. Exposed for plan-shape tests.
  [[nodiscard]] std::shared_ptr<read_time_filter_bypass const> read_bypass() const noexcept
  {
    return _read_bypass;
  }

  /// The channel this operator consumes, for plan-shape assertions on registered routing.
  [[nodiscard]] std::shared_ptr<sirius::op::sirius_dynamic_filter_set const> filters()
    const noexcept
  {
    return _filters;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void on_finalize_operator() override;

  /// Returns the input footprint. Filtering passes rows through or removes them and never expands
  /// the input.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override
  {
    return stats.bytes;
  }

 private:
  /// The append-only publication channel; co-owned with the producing hash-join build.
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> _filters;
  /// Per-scan selectivity + per-filter marginal-keep gate, shared across this scan's split tasks.
  dynamic_filter_gate _gate;
  /// Which filter capabilities apply post-decode: membership only when AST filters already ran at
  /// read time (parquet), AST row masks too when the scan has no read-time dynamic phase
  /// (duckdb-native).
  dynamic_filter_apply_mode _mode;
  /// Which planner role installed this operator.
  dynamic_filter_endpoint_provenance _provenance;
  /// Non-owning counter sink with `SiriusContext` lifetime -- the hash join's stats contract.
  sirius::op::dynamic_filter_stats* _stats;
  /// Serve-path latch shared with the wrapped scan operator; null off the scan route.
  std::shared_ptr<read_time_filter_bypass> _read_bypass;
};

}  // namespace sirius::op::scan
