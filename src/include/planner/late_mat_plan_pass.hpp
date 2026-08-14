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

// How long a scanned column's VALUES are needed (env gate: SIRIUS_EXP_LATE_MAT).
//
// A column can be deferred for as long as nobody looks at what is in it.
// Operators between the scan and its first real reader only move it — a filter
// keeps or drops whole rows, a projection re-orders, a join copies the payload
// beside its key — and none of them care whether the column holds values or a
// rowid standing in for them. So the question this answers, per scan output
// column, is: which operator first reads its CONTENT, and how many operator
// boundaries did it cross to get there?
//
// FAIL-CLOSED BY CONSTRUCTION. Any operator shape not modelled here is treated
// as reading everything it receives. That can only make a lifetime shorter than
// it truly is, never longer — so an unmodelled shape costs a deferral that was
// possible, and can never permit one that was not. Adding a shape to the
// modelled set is therefore always safe to get wrong in the conservative
// direction, which is the property that makes this extensible.
//
// This pass decides NOTHING. It reports lifetimes; whether a bundle is worth
// deferring is late_mat/defer_policy.hpp, and what a deferral does to a plan is
// late_mat/defer_directive.hpp. Keeping the analysis free of the economics is
// what lets the thresholds move on a measurement without touching the walk.

#include "helper/logical_type.hpp"
#include "late_mat/defer_directive.hpp"
#include "late_mat/defer_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::planner {

/// Where one scan output column stops being merely carried.
struct column_lifetime {
  std::size_t scan_output_position = 0;
  /// The operator that first reads the column's content, or nullptr when
  /// nothing does — a column that reaches the plan's root unread is one the
  /// query only ever projects.
  op::sirius_physical_operator const* first_reader = nullptr;
  /// Operator boundaries crossed before that reader. This is what the defer
  /// policy weighs the ride against; a column read by the scan's own parent
  /// crossed one.
  int boundaries = 0;
  /// The column's position in `first_reader`'s input, which is where a
  /// materialization would have to put it back.
  std::size_t position_at_reader = 0;
  /// Whether some join on the ride could leave this column's row unmatched.
  /// The rowid is then null for those rows and the column materializes as
  /// null — so a deferral is still sound, but only for a consumer that accepts
  /// nulls. Refusing outer joins outright would be simpler and would cost
  /// every outer-shaped query.
  bool nullified_on_ride = false;
};

/// Where a column entering an INNER hash join from one side lands in its
/// output, or nullopt when that side does not project it out.
///
/// Exposed because this is the one piece of the walk whose failure is silent:
/// a wrong offset here does not refuse a deferral, it materializes a column
/// into a position holding something else. The join's output is its lhs
/// projection followed by its rhs projection, so an rhs column carries the
/// lhs's emitted width as an offset.
[[nodiscard]] std::optional<std::size_t> join_output_position(
  bool from_lhs,
  std::vector<int> const& lhs_projection,
  std::vector<int> const& rhs_projection,
  std::size_t in_position);

/// Per-row bytes a scanned column would stop carrying if it were deferred.
///
/// Variable-width columns have no width in their type, and the ride's value is
/// exactly their width, so a fixed estimate stands in. It is deliberately
/// conservative: understating a string's width can only refuse a bundle the
/// policy would have taken, while overstating it would install one that never
/// repays the rowid.
[[nodiscard]] std::int64_t estimated_value_bytes(sirius::logical_type const& type);

/// Bundle lifetimes into candidates, one per operator that first reads them.
///
/// Columns that stop at the same operator ride together and materialize
/// together, so they compete for that slot as a unit — which is also why the
/// policy arbitrates whole bundles rather than columns.
///
/// Columns nothing reads are excluded: with no consumer there is nowhere to
/// install the materializing half of the pair, and half a deferral loses the
/// data outright.
///
/// @p readers, when given, comes back parallel to the returned candidates: entry
/// i is the operator candidate i would materialize at. The slot labels encode
/// the same order, but a label is a string for the census and an installer needs
/// the operator itself.
[[nodiscard]] std::vector<late_mat::defer_candidate> build_defer_candidates(
  op::sirius_physical_operator const& scan,
  std::vector<column_lifetime> const& lifetimes,
  std::vector<op::sirius_physical_operator const*>* readers = nullptr);

/// Lifetimes of every output column of @p scan, in scan output order.
///
/// @p scan must be within the tree whose parents were stamped by
/// set_parent_ops; the walk goes upward from it.
[[nodiscard]] std::vector<column_lifetime> analyze_column_lifetimes(
  op::sirius_physical_operator const& scan);

/// The one deferral a scan would install, and the census of everything weighed.
///
/// ONE BUNDLE PER SCAN. The substituted output carries a single rowid, so two
/// bundles landing at two different consumers are not representable; the widest
/// wins and the rest are refused as @ref late_mat::defer_refusal::second_bundle
/// rather than dropped, because a bundle that quietly vanished looks exactly
/// like one that never qualified.
struct planned_deferral {
  /// The operator that would materialize, or nullptr when nothing installs.
  op::sirius_physical_operator* port = nullptr;
  /// Scan output positions to defer, ascending. The first carries the rowid.
  std::vector<std::size_t> positions;
  /// The columns' types as the port must restore them, parallel to positions.
  std::vector<sirius::logical_type> restored_types;
  std::int64_t net_value_bytes = 0;
  int boundaries               = 0;
  /// Every candidate weighed, in the order the walk found them.
  std::vector<late_mat::defer_outcome> census;
  /// Columns excluded before the weighing because an outer join on their ride
  /// could null them. Sound to defer — a null rowid must materialize a null —
  /// but the materializer does not produce nulls yet, so v1 refuses them here
  /// rather than at the far end where the answer would already be wrong.
  std::size_t nullable_columns_skipped = 0;

  [[nodiscard]] bool installable() const noexcept { return port != nullptr && !positions.empty(); }
};

/// Weigh @p scan's columns and report the deferral to install, if any.
///
/// Decides nothing about origins or execution: the result says WHICH positions
/// and WHERE they land, and the caller — which is the only code that knows
/// whether this scan's rows are addressable at all — completes the pair.
[[nodiscard]] planned_deferral plan_deferral(op::sirius_physical_operator& scan,
                                             late_mat::defer_policy const& policy = {});

/// Stamp both halves of @p pair, or neither.
///
/// Fails (changing nothing) on an invalid pair, or when either operator already
/// carries a half — a second deferral through the same port would materialize
/// against a schema the first one has already rewritten.
bool install_deferral(op::sirius_physical_operator& scan,
                      op::sirius_physical_operator& port,
                      late_mat::defer_pair pair);

}  // namespace sirius::planner
