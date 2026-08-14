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

#include <cstddef>
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

/// Lifetimes of every output column of @p scan, in scan output order.
///
/// @p scan must be within the tree whose parents were stamped by
/// set_parent_ops; the walk goes upward from it.
[[nodiscard]] std::vector<column_lifetime> analyze_column_lifetimes(
  op::sirius_physical_operator const& scan);

}  // namespace sirius::planner
