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

/// Stamped on the producing scan: emit these output positions as a rowid and
/// placeholders instead of their values.
///
/// Positions are scan-OUTPUT positions, and ascending. The first carries the
/// rowid; the rest carry placeholders, and exist only so the arity and the
/// positions after them do not move.
struct deferred_scan_output {
  std::vector<std::size_t> output_positions;

  [[nodiscard]] bool empty() const noexcept { return output_positions.empty(); }
  /// Where the rowid rides. Only meaningful when non-empty.
  [[nodiscard]] std::size_t rowid_position() const { return output_positions.front(); }
  [[nodiscard]] bool defers(std::size_t position) const noexcept;
};

/// The column type riding at each deferred position: pin-order ids are global,
/// so they are 64-bit (lineitem at sf1000 is past 2^32 rows), and the
/// placeholders are the narrowest thing that keeps a position occupied.
inline constexpr cudf::type_id kRowidType       = cudf::type_id::UINT64;
inline constexpr cudf::type_id kPlaceholderType = cudf::type_id::INT8;

/// Stamped on the consuming operator: turn the rowid back into columns.
///
/// A batch is matched by its full expected schema rather than by position
/// alone. A matcher that only checked "is there a UINT64 here" would fire on an
/// unrelated batch that happens to have one, and materializing against the
/// wrong batch reads arbitrary rows of the pinned table — so the check is the
/// whole shape, and a mismatch declines rather than guesses.
struct port_materialize_directive {
  /// The schema this directive expects, with the deferred positions already
  /// swapped to rowid/placeholder types.
  std::vector<cudf::data_type> expected_schema;
  /// Positions to restore, ascending and identical to the scan side's.
  std::vector<std::size_t> output_positions;
  /// Where each restored column comes from, parallel to output_positions.
  std::vector<column_origin> origins;
  /// The dtype each restored column must come back as, parallel likewise.
  std::vector<cudf::data_type> restored_types;

  [[nodiscard]] bool empty() const noexcept { return output_positions.empty(); }
  [[nodiscard]] std::size_t rowid_position() const { return output_positions.front(); }

  /// Whether `schema` is the batch this directive was installed for.
  [[nodiscard]] bool matches(std::vector<cudf::data_type> const& schema) const;

  /// Self-consistency: the parallel vectors agree, the positions are ascending
  /// and in range, and the schema carries a rowid and placeholders exactly
  /// where the positions say. Checked at install, so a malformed pair cannot
  /// reach execution.
  [[nodiscard]] bool valid() const;
};

/// The two halves, which only mean anything together.
struct defer_pair {
  deferred_scan_output scan;
  port_materialize_directive port;

  [[nodiscard]] bool valid() const;
};

/// Build the pair for one bundle: `schema` is the producer's planned output
/// types, `positions` the columns to defer, `origins` where each comes from.
///
/// Returns a pair whose valid() is false if the request is not installable —
/// no positions, positions out of range or unordered, or a mismatched origin
/// count. The caller installs both halves or neither.
[[nodiscard]] defer_pair make_defer_pair(std::vector<cudf::data_type> const& schema,
                                         std::vector<std::size_t> const& positions,
                                         std::vector<column_origin> const& origins);

}  // namespace sirius::late_mat
