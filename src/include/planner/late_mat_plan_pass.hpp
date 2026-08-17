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

/// The ride a column could take PAST the group-bys it is a key of.
///
/// A group-by reads its keys, so the sound stop for a deferred key is the
/// aggregate's input — and that is what @ref column_lifetime still reports.
/// But if the deferred keys are functionally determined by a key that is unique
/// over the pinned table, grouping by the ROWID yields exactly the groups
/// grouping by the values would: every row of a group carries one and the same
/// rowid. The ride may then continue past the aggregate and materialize at its
/// far end — one row per group instead of one per join match, which on q10 is
/// ~150k rows instead of tens of millions.
///
/// THIS STRUCT ASSERTS NOTHING ABOUT THAT BIJECTION. The walk cannot see the
/// pinned entry's uniqueness facts, so it only reports where the ride WOULD end
/// and which aggregates it passed; whoever holds the facts admits or refuses it
/// (and on a refusal the conservative stop is still there, unchanged).
struct group_key_ride {
  /// The group-bys ridden, in ride order. A ride is only admissible if the
  /// proven-unique key is a planned group key at every one of them.
  std::vector<op::sirius_physical_operator const*> group_bys;
  /// Where the ride actually ends: the first operator to read the column's
  /// content past the aggregates, or nullptr when it reaches the plan's top.
  op::sirius_physical_operator const* reader       = nullptr;
  std::size_t position_at_reader                   = 0;
  op::sirius_physical_operator const* reader_input = nullptr;
  int port_crossings                               = 0;
  bool nullified_on_ride                           = false;
  /// Something past the aggregates compares this column as a join key — the
  /// extension is void (the conservative stop still stands).
  bool read_as_join_key = false;
};

/// Where one scan output column stops being merely carried.
struct column_lifetime {
  std::size_t scan_output_position = 0;
  /// The operator that first reads the column's content, or nullptr when
  /// nothing does — a column that reaches the plan's root unread is one the
  /// query only ever projects.
  op::sirius_physical_operator const* first_reader = nullptr;
  /// PORT crossings before that reader — the times the column's bytes were
  /// written to a repository and read back, which is the only carrying that
  /// costs anything and so the only unit the defer policy's thresholds are
  /// meaningful in. A crossing is counted when the operator being LEFT is a
  /// pipeline sink; a filter or a projection in the same pipeline hands its
  /// columns on without materializing them and costs nothing to ride past.
  int port_crossings = 0;
  /// The column's position in `first_reader`'s input, which is where a
  /// materialization would have to put it back. NOT the scan position: a join
  /// on the ride widens the table and reorders it.
  std::size_t position_at_reader = 0;
  /// The operator `first_reader` received this column from — the last step of
  /// the ride, and the one whose output schema a batch at the port has.
  op::sirius_physical_operator const* reader_input = nullptr;
  /// Whether some join on the ride could leave this column's row unmatched.
  /// The rowid is then null for those rows and the column materializes as
  /// null — so a deferral is still sound, but only for a consumer that accepts
  /// nulls. Refusing outer joins outright would be simpler and would cost
  /// every outer-shaped query.
  bool nullified_on_ride = false;
  /// Whether something on the ride compares this column as a KEY — a join's
  /// condition, or the partition that feeds one. Such a column may never be
  /// deferred, and the binding reason is the partition rather than the join: it
  /// hashes the key to place a row, and a rowid hashes differently from the
  /// value it stands for, so equal keys would land in different partitions and
  /// the join would simply miss matches. Stopping the ride at the join is not
  /// enough — the port materializes at the join's input, which is after the
  /// partition has already hashed. This is the walk's one silent-wrong-answer
  /// shape; every other refusal merely costs a deferral.
  bool read_as_join_key = false;
  /// Every group-by this column is a planned KEY of, in ride order — recorded
  /// whatever else reads it, and so present even for a column that stopped long
  /// before (a join key, say). That is the point: a column riding REAL is the
  /// one whose pin-time uniqueness can admit ANOTHER column's ride, and the
  /// proof has to hold at every aggregate that ride crosses.
  std::vector<op::sirius_physical_operator const*> group_key_at;
  /// Present when the column rode at least one group-by AS A KEY: the fields
  /// above then describe the sound stop at the first such aggregate, and this
  /// describes the longer ride that the pin-uniqueness bijection would unlock.
  /// See @ref group_key_ride — reported, never assumed.
  std::optional<group_key_ride> group_ride;
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

/// The longer ride the installed bundle could take, past the group-bys it is a
/// key of — reported alongside the sound plan, never in place of it.
///
/// Emitted only when EVERY column of the bundle rode as a group key through the
/// SAME chain of aggregates to the SAME reader, none of them nullified and none
/// read as a join key. Whoever holds the pin-time uniqueness facts then admits
/// it (by proving one of @ref unique_key_candidates distinct over the pinned
/// table) or refuses it — and on a refusal the bundle installs at
/// @ref planned_deferral::port exactly as it would have.
struct group_key_extension {
  /// The aggregates ridden, in ride order. The proof has to hold at every one:
  /// a key that stops being a group key half way up stops determining the rows.
  std::vector<op::sirius_physical_operator const*> group_bys;
  /// Where the bundle would materialize instead — one row per group.
  op::sirius_physical_operator* port = nullptr;
  /// Where each deferred column arrives there, parallel to
  /// @ref planned_deferral::positions.
  std::vector<std::size_t> port_positions;
  op::sirius_physical_operator const* port_input = nullptr;
  int boundaries                                 = 0;
  /// Scan output positions that ride REAL (are not in the bundle) and are group
  /// keys at every aggregate in @ref group_bys. One of these being unique over
  /// the pinned table is what makes grouping by the rowid the same grouping —
  /// the pass cannot know which, so it reports all of them.
  std::vector<std::size_t> unique_key_candidates;
};

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
  /// Where each of those columns ARRIVES at the port, parallel to `positions`.
  /// A join between the two ends widens and reorders the table, so these are
  /// not the scan's positions and the port half is built from them.
  std::vector<std::size_t> port_positions;
  /// The operator whose output the port reads — the last step of the ride. Its
  /// schema is the one a batch at the port has, which is what the port's
  /// whole-schema match is built from.
  op::sirius_physical_operator const* port_input = nullptr;
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
  /// Columns excluded because something on the ride keys on them. Not an
  /// economic refusal: a partition hashes its consumer's key to place rows, so
  /// a key riding as a rowid would scatter equal values across partitions and
  /// the join would miss matches. See column_lifetime::read_as_join_key.
  std::size_t join_keys_skipped = 0;
  /// The group-by-rowid extension of this bundle, when the walk found one. See
  /// @ref group_key_extension: reported, not admitted.
  std::optional<group_key_extension> group_extension;

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
/// Attach a RIDER to a port that already carries a deferral.
///
/// The rider is a second pinned table's bundle materializing at the same
/// consumer — q10's `nation` beside `customer`. Only the group-by-rowid ride
/// makes this admissible, and only the caller can know that, so this checks
/// only what it can see: the port must already hold a primary, the rider's
/// positions must not collide with anything already claimed, and the merged
/// directive must still be valid. On any of those it changes NOTHING — a
/// half-attached rider is a scan that threw its values away with no consumer
/// to put them back.
///
/// @p pair is built exactly as for a primary install (the rider's own scan
/// schema, its own port positions), against a port schema that already carries
/// the primary's substitutions.
bool install_rider(op::sirius_physical_operator& scan,
                   op::sirius_physical_operator& port,
                   late_mat::defer_pair pair);

bool install_deferral(op::sirius_physical_operator& scan,
                      op::sirius_physical_operator& port,
                      late_mat::defer_pair pair);

}  // namespace sirius::planner
