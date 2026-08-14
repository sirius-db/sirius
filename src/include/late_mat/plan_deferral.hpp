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

// Planner-side deferral annotations (SIRIUS_EXP_LATE_MAT_V2).
//
// Produced by the plan-time lifetime pass (planner/late_mat_plan_pass) over
// the finished physical operator TREE: for each pinned-candidate scan output
// column, the operator that FIRST READS ITS CONTENT (every earlier ancestor
// only moves it positionally). The pass only records plan FACTS — value
// thresholds, arbitration and every runtime guard stay in the lowering
// backend (scan_manager/late_mat_defer_policy), which also RE-walks the
// pipeline graph and refuses fail-closed on any plan/walk disagreement.
//
// Stamped on the producing scan as one empty-by-default shared_ptr; never
// constructed unless the v2 sub-gate is on.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::late_mat {

/// One scan output column's lifetime fact.
struct planned_column_deferral {
  std::size_t scan_output_position{0};
  /// First operator whose expressions/keys read the column's content; the
  /// column is a pure positional pass-through at every ancestor before it —
  /// EXCEPT group-key reads, which the march rides through (see group_key_at):
  /// a group key's position survives the aggregate, so its content-consumer
  /// may lie beyond it.
  op::sirius_physical_operator* consumer{nullptr};
  /// The column's input position AT the consumer (post every remap) — the
  /// lowering uses it directly for ports past pass-modeled pipelines.
  std::size_t final_position{0};
  /// Aggregates (HASH_GROUP_BY / MERGE_GROUP_BY) where this column's position
  /// is a PLANNED GROUP KEY, in ride order. Set for deferred riders AND for
  /// real-riding columns (e.g. a join key that is also a group key) — the
  /// group-by-rowid uniqueness admission needs the latter. The §4-addendum
  /// bijection is what makes riding a placeholder through these sound.
  std::vector<op::sirius_physical_operator*> group_key_at;
  /// Whether the consumer is an aggregate whose ONLY reads of this column are
  /// COUNT_VALID aggregates (the count-on-deferred input: count(col) ==
  /// count(rowid) for a non-null source column, so the ride never
  /// materializes at all).
  bool consumed_as_count_only{false};
  /// Whether an outer join on the ride may have NULL-extended this column's
  /// position (LEFT/RIGHT/FULL pass-through). SOUNDNESS: the join's NULLIFY
  /// gather nullifies WHATEVER occupies the position — a rowid placeholder is
  /// nullified exactly as the original column would be, positionally — so
  /// COUNT_VALID over the ride is preserved and count-on-deferred remains
  /// valid. Every OTHER consumption is disqualified by this flag: a
  /// materialize-at-port bundle cannot gather NULL rowids (the port
  /// materializer refuses them by design).
  bool nullified_on_ride{false};
  /// Pipeline-breaking ancestors (joins/aggregates) crossed before the
  /// consumer — informational; the lowering recomputes port boundaries itself.
  std::size_t crossings{0};
};

/// Per-scan analysis result (empty-by-default annotation on the scan op).
struct planned_deferral {
  std::vector<planned_column_deferral> columns;
};

/// v3 raw FD material (SIRIUS_EXP_LATE_MAT_V3), stamped as ONE shared graph
/// on every GPU_SCAN of the query. The pass records only PLAN facts —
/// INNER-join bare-reference equality edges between scan-column provenances,
/// and each aggregate's group-key provenances. The determination closure and
/// nomination run in the LOWERING, which is where the pin-time uniqueness
/// facts live (the pass runs before cache matching and cannot see entries).
struct planned_fd_graph {
  /// (scan_a, pos_a) ≡ (scan_b, pos_b), contributed by INNER join `join`
  /// (kept for the census chain trace).
  struct equality_edge {
    op::sirius_physical_operator* scan_a{nullptr};
    std::size_t pos_a{0};
    op::sirius_physical_operator* scan_b{nullptr};
    std::size_t pos_b{0};
    op::sirius_physical_operator* join{nullptr};
  };
  std::vector<equality_edge> edges;
  /// One aggregate group key whose input is a pure pass-through of
  /// (scan, scan_pos).
  struct key_provenance {
    op::sirius_physical_operator* aggregate{nullptr};
    std::size_t input_pos{0};
    op::sirius_physical_operator* scan{nullptr};
    std::size_t scan_pos{0};
  };
  std::vector<key_provenance> key_provenances;
};

/// v3 resolved FD proof — BUILT BY THE LOWERING (closure + nomination over
/// the graph above plus the pinned entries' uniqueness facts); the census
/// prints `chain_trace` verbatim so adversarial review can audit every link.
struct planned_group_by_fd {
  /// Aggregates the substituted keys ride through, in ride order.
  std::vector<op::sirius_physical_operator*> aggregate_chain;
  /// The nominated origin's pin-unique key (scan output position) — stays a
  /// REAL group key; this is the closure's (⇒) direction.
  std::size_t nominated_unique_key_position{0};
  struct origin_bundle {
    op::sirius_physical_operator* origin_scan{nullptr};
    /// Dropped group-key columns, as that scan's output positions (ascending;
    /// the first carries this origin's rowid/rider column).
    std::vector<std::size_t> dropped_scan_positions;
    bool is_nominated{false};  ///< false = rider origin
  };
  std::vector<origin_bundle> bundles;
  /// Human-readable derivation-order proof chain for the census.
  std::string chain_trace;
};

}  // namespace sirius::late_mat
