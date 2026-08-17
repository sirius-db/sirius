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

// What a deferral IS, as a pair of instructions (env gate: SIRIUS_EXP_LATE_MAT).
//
// A deferral is two edits to a plan, and they are only ever meaningful
// together: the producing scan stops emitting some columns' values, and the
// consuming operator puts them back. Installed apart, either one is a bug — a
// scan that substitutes with nobody to materialize loses the data, and a
// consumer that materializes what was never substituted corrupts a batch that
// was already correct. So they are built as a pair and installed atomically.
//
// BETWEEN THE PAIR, NOTHING CHANGES. The deferred columns ride as ordinary
// data: a UINT64 pin-order rowid at the FIRST deferred position, and 1-byte
// INT8 placeholders at the rest. Arity and positions are preserved, so every
// operator between the two ends — filters, joins, partitions, exchanges — sees
// a table of the shape it expected and needs no knowledge of any of this. That
// is what makes deferral transparent rather than a rewrite of the pipeline.
//
// Why the rowid sits at the first deferred position rather than being appended:
// appending changes arity, and arity is what every operator between the pair
// would then have to agree about.
//
// Measured on TPC-H q10 at sf1000 (gpu pin, medians): carrying the five wide
// customer columns through the joins costs 247 ms of a 531 ms query, and
// removing them as GROUP BY keys costs a further 69 ms — 316 ms total, 60% of
// the query. The ride is where most of it is, which is what this pair removes.

#include "late_mat/column_origin.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius::late_mat {

/// The column type riding at each deferred position: pin-order ids are global,
/// so they are 64-bit (lineitem at sf1000 is past 2^32 rows), and the
/// placeholders are the narrowest thing that keeps a position occupied.
inline constexpr cudf::type_id kRowidType = cudf::type_id::UINT64;
/// The same id, half the ride, for a pinned table whose rows fit 32 bits — which
/// is every TPC-H table but lineitem at SF1000. The width is decided per
/// deferral (the pin's row count is known where a pair is built) and carried on
/// BOTH halves, because the scan side writes it and the port side reads it, and
/// a disagreement is a batch nobody materializes.
inline constexpr cudf::type_id kNarrowRowidType = cudf::type_id::UINT32;

/// Stamped on the producing scan: emit these output positions as a rowid and
/// placeholders instead of their values.
///
/// Positions are scan-OUTPUT positions, and ascending. The first carries the
/// rowid; the rest carry placeholders, and exist only so the arity and the
/// positions after them do not move.
struct deferred_scan_output {
  std::vector<std::size_t> output_positions;
  /// What the rowid rides as; see kNarrowRowidType.
  cudf::type_id rowid_type = kRowidType;

  [[nodiscard]] bool empty() const noexcept { return output_positions.empty(); }
  /// Where the rowid rides. Only meaningful when non-empty.
  [[nodiscard]] std::size_t rowid_position() const { return output_positions.front(); }
  [[nodiscard]] bool defers(std::size_t position) const noexcept;
};

inline constexpr cudf::type_id kPlaceholderType = cudf::type_id::INT8;

/// Stamped on the consuming operator: turn the rowid back into columns.
///
/// A batch is matched by its full expected schema rather than by position
/// alone. A matcher that only checked "is there a UINT64 here" would fire on an
/// unrelated batch that happens to have one, and materializing against the
/// wrong batch reads arbitrary rows of the pinned table — so the check is the
/// whole shape, and a mismatch declines rather than guesses.
/// THE PORT'S POSITIONS ARE NOT THE SCAN'S. Between the two ends a join widens
/// the table and reorders it, so a column deferred at scan position 1 may arrive
/// at the port as column 7 of a table twice as wide. Both halves therefore carry
/// their own schema and their own positions; only the origins are shared. Using
/// the scan's schema at the port matches no batch at all if you are lucky, and
/// matches the wrong one if you are not.
/// One additional pinned origin materializing at the same port as the primary
/// bundle — a RIDER.
///
/// Two scans can ride to one consumer when the consumer is a group-by whose
/// groups are already pinned down by a proven-unique key (see
/// planner::group_key_extension). q10 is the shape: `customer`'s five wide
/// columns ride as one rowid, and `nation`'s `n_name` — a group key of the same
/// aggregates — rides as a second. Each rider carries its own rowid column at
/// its own port position and its own origins, because the two bundles come from
/// DIFFERENT pinned tables and a rowid means nothing outside the table it
/// indexes.
struct rider_bundle {
  /// Positions to restore, ascending, in the PORT's coordinates.
  std::vector<std::size_t> output_positions;
  /// Where this rider's rowid rides at the port; one of output_positions.
  std::size_t rowid_at = 0;
  /// Where each restored column comes from, parallel to output_positions.
  std::vector<column_origin> origins;
  /// The dtype each restored column must come back as, parallel likewise.
  std::vector<cudf::data_type> restored_types;
  /// What this rider's rowid rides as — its own pinned table's row count
  /// decides it, so it need not match the primary bundle's.
  cudf::type_id rowid_type = kRowidType;
};

struct port_materialize_directive {
  /// The schema this directive expects — the batch AT THE PORT, with the
  /// deferred positions already swapped to rowid/placeholder types. Covers the
  /// primary bundle AND every rider, since all of them are substituted in the
  /// batch that arrives.
  std::vector<cudf::data_type> expected_schema;
  /// Positions to restore, ascending, in the PORT's coordinates.
  std::vector<std::size_t> output_positions;
  /// Where the rowid rides at the port. One of output_positions, but not
  /// necessarily the first: which column carries it is decided on the scan side,
  /// and the ride may reorder the bundle before it arrives.
  std::size_t rowid_at = 0;
  /// Where each restored column comes from, parallel to output_positions.
  std::vector<column_origin> origins;
  /// The dtype each restored column must come back as, parallel likewise.
  std::vector<cudf::data_type> restored_types;
  /// What the rowid rides as; see kNarrowRowidType. Must match what the scan
  /// half writes — @ref valid checks the expected schema against it.
  cudf::type_id rowid_type = kRowidType;
  /// Bundles from OTHER pinned tables materializing at this same port; see
  /// @ref rider_bundle. Empty in the ordinary single-origin case.
  std::vector<rider_bundle> riders;

  [[nodiscard]] bool empty() const noexcept { return output_positions.empty(); }
  [[nodiscard]] std::size_t rowid_position() const { return rowid_at; }

  /// Whether `schema` is the batch this directive was installed for.
  [[nodiscard]] bool matches(std::vector<cudf::data_type> const& schema) const;

  /// Self-consistency: the parallel vectors agree, the positions are ascending
  /// and in range, and the schema carries a rowid and placeholders exactly
  /// where the positions say. Checked at install, so a malformed pair cannot
  /// reach execution.
  [[nodiscard]] bool valid() const;
};

/// How many deferrals have installed in this process, ever.
///
/// A query that ran with the gate on and deferred nothing looks exactly like one
/// that deferred and gained nothing, and the difference is the whole question
/// when a measurement disappoints — or when a test claims to exercise this path.
/// Monotonic and process-wide; a test reads it before and after.
[[nodiscard]] std::uint64_t deferrals_installed() noexcept;

/// Called by planner::install_deferral, and by nothing else.
void note_deferral_installed() noexcept;

/// The two halves, which only mean anything together.
struct defer_pair {
  deferred_scan_output scan;
  port_materialize_directive port;

  [[nodiscard]] bool valid() const;
};

/// Build the pair for one bundle, from the two ends' own coordinates.
///
/// @p scan_schema / @p scan_positions describe the producing scan's output;
/// @p port_schema / @p port_positions the batch as it arrives at the consumer,
/// with `port_positions[i]` the place `scan_positions[i]` travelled to. The
/// three per-column vectors (scan positions, port positions, origins) are
/// parallel and in scan order; the port half is sorted into its own ascending
/// order internally, so a ride that reorders the bundle is handled here rather
/// than by every caller.
///
/// Returns a pair whose valid() is false if the request is not installable —
/// no positions, positions out of range or repeated, mismatched counts, or a
/// port column whose type disagrees with what the scan gave up. The caller
/// installs both halves or neither.
[[nodiscard]] defer_pair make_defer_pair(std::vector<cudf::data_type> const& scan_schema,
                                         std::vector<std::size_t> const& scan_positions,
                                         std::vector<cudf::data_type> const& port_schema,
                                         std::vector<std::size_t> const& port_positions,
                                         std::vector<column_origin> const& origins,
                                         cudf::type_id rowid_type = kRowidType);

}  // namespace sirius::late_mat
