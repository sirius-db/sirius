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

// Whether a set of columns is worth deferring (env gate: SIRIUS_EXP_LATE_MAT).
//
// Deferring replaces a column's values with an 8-byte rowid for the stretch of
// pipeline between the scan and whatever consumes it, then materializes at the
// far end. That trade is not always good, and the ways it goes bad are
// measured rather than guessed:
//
//  * A NARROW bundle loses. The ride saves (values - rowid) bytes per row per
//    boundary, but materializing costs a canonicalization on the port side,
//    which broke even near 60 B/row on the sort path. Dimension columns of
//    11-25 B were measured COSTING +61 ms on a 800M-row port, while a 154.6 B
//    bundle and a 50 B pair both won. Hence a floor on deferred value, not a
//    column count.
//  * A SHORT ride loses. Saving bytes across one or two port crossings does
//    not repay the materialization; the rides that paid crossed 6 and 8.
//
// ARBITRATION IS WIDEST-WINS, and that is not a refinement. Taking the first
// candidate to arrive let a 25-row dimension ride occupy the consumer slot and
// lock out the bundle that actually mattered. So a wider bundle evicts a
// narrower one already holding the slot, and eviction is atomic — a bundle
// installs whole or not at all, since half a bundle pays the materialization
// without saving the traffic.
//
// Every refusal is reported rather than dropped: a deferral that silently did
// not happen looks exactly like one that did nothing, and the difference is
// the whole question when a measurement disappoints.

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <vector>

namespace sirius::late_mat {

/// Why a candidate did not install.
enum class defer_refusal : std::uint8_t {
  installed = 0,     ///< not a refusal
  too_little_value,  ///< the bundle is narrower than the rowid earns back
  too_short_a_ride,  ///< too few port crossings to repay materializing
  no_columns,        ///< nothing to defer
  evicted,           ///< a wider bundle took the slot
  /// A bundle from the same scan already rides. The substituted scan output
  /// carries ONE rowid, so a second bundle — landing at a different consumer,
  /// and therefore needing its own — is not representable. A representational
  /// limit, not an economic one, and the widest bundle is the one kept.
  second_bundle,
};

[[nodiscard]] char const* describe(defer_refusal r) noexcept;

/// One column a candidate would defer.
struct defer_column {
  std::uint32_t column_pos = 0;  ///< position within the origin entry
  std::int64_t value_bytes = 0;  ///< per-row width as it would materialize
};

/// One bundle of columns from one origin, riding to one consumer slot.
///
/// `slot` identifies where the columns would be materialized; candidates
/// sharing a slot compete, since only one bundle can occupy it.
struct defer_candidate {
  std::string slot;
  std::vector<defer_column> columns;
  /// Port crossings the ride avoids carrying values across.
  int boundaries = 0;

  /// Bytes per row the ride actually saves: the values it stops carrying, less
  /// the rowid it carries instead.
  [[nodiscard]] std::int64_t net_value_bytes(std::int64_t rowid_bytes) const noexcept;
};

/// Port-crossing floor, env-overridable for measurement (default 4).
inline int default_min_boundaries()
{
  static int const value = [] {
    char const* v = std::getenv("SIRIUS_LATE_MAT_MIN_BOUNDARIES");
    if (v == nullptr || v[0] == '\0') { return 4; }
    int parsed      = 0;
    auto const* end = v + std::strlen(v);
    auto const rc   = std::from_chars(v, end, parsed);
    return (rc.ec == std::errc{} && rc.ptr == end && parsed >= 0) ? parsed : 4;
  }();
  return value;
}

/// The thresholds, in one place so a measurement can move them together.
struct defer_policy {
  /// Deferred value must exceed this, per row, after the rowid is paid for.
  std::int64_t min_value_bytes = 32;
  /// Port crossings the ride must save. Overridable with
  /// SIRIUS_LATE_MAT_MIN_BOUNDARIES so a measurement can separate "the ride is
  /// not worth taking" from "the crossing count under-reports it".
  int min_boundaries = default_min_boundaries();
  /// What riding costs per row.
  std::int64_t rowid_bytes = 8;
};

/// What became of one candidate.
struct defer_outcome {
  std::string slot;
  defer_refusal refusal        = defer_refusal::installed;
  std::int64_t net_value_bytes = 0;
  int boundaries               = 0;

  [[nodiscard]] bool installed() const noexcept { return refusal == defer_refusal::installed; }
};

/// Decide which candidates install, one per slot.
///
/// Returns an outcome for EVERY candidate, in the order given — the refusals
/// are the census, and a candidate that quietly vanished would be
/// indistinguishable from one that never existed.
[[nodiscard]] std::vector<defer_outcome> choose_deferrals(
  std::vector<defer_candidate> const& candidates, defer_policy const& policy = {});

}  // namespace sirius::late_mat
