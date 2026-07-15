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

#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/sirius_dynamic_filter.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <memory>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// sirius_physical_dynamic_filter
//===----------------------------------------------------------------------===//
/// @brief Applies dynamic filters to a GPU scan's decoded output.
///
/// Sits directly above the scan in its pipeline: the scan reads, decodes, and assembles each
/// batch, then this operator filters it. The apply mode matches the scan format's read-time
/// capabilities: a parquet scan already ran AST-capable filters (zone maps) through the reader's
/// @c set_filter, so its operator applies membership masks only; a duckdb-native scan has no
/// read-time dynamic phase, so its operator also evaluates AST-capable filters row-wise
/// (@c include_ast_row_masks).
///
/// Filters arrive on a @ref sirius_dynamic_filter_set the producing hash-join build publishes into.
/// The @ref dynamic_filter_gate decides per scan whether filtering earns its cost, and a batch
/// passes through unchanged when the publication attempt emitted no filter applicable under the
/// mode, no device-local replica exists, or the gate declines. A producing join's immediate probe
/// edge is ordered after build-port publication; this operator can run earlier when its scan is a
/// transitive target below an intervening join, so each execution uses the filters then visible.
class sirius_physical_dynamic_filter : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DYNAMIC_FILTER;

  sirius_physical_dynamic_filter(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<sirius::op::sirius_dynamic_filter_set> filters,
    double gate_keep_threshold     = dynamic_filter_gate::k_default_keep_threshold,
    dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::membership_masks_only);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void on_finalize_operator() override;

  /// Filtering only shrinks or passes through its input, never expands it — reserve at most the
  /// input footprint rather than the base 2× expansion default.
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
};

}  // namespace sirius::op::scan
