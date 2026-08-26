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
//  * A NARROW bundle loses. The ride saves (values - carrier) bytes per row per
//    boundary — the carrier being the rowid plus one placeholder for every
//    column past the first — but materializing costs a canonicalization on the
//    port side,
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

#include <algorithm>
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
  /// Real but does not repay: 16 B/row over 9 crossings pays where 8 B/row
  /// over 9 does not, so what is weighed is the product.
  below_value_x_boundaries,
};

[[nodiscard]] char const* describe(defer_refusal r) noexcept;

/// Per-row width of one placeholder, matching `kPlaceholderType` (INT8) in
/// `defer_directive.hpp`. A bundle rides as ONE rowid plus one placeholder per
/// further column, so an n-column bundle costs `rowid_bytes + (n - 1)` per row,
/// not `rowid_bytes`.
inline constexpr std::int64_t kPlaceholderBytes = 1;

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
  /// what rides in their place.
  [[nodiscard]] std::int64_t net_value_bytes(std::int64_t rowid_bytes) const noexcept;

  /// Bytes per row the substituted columns cost: the rowid at the first
  /// deferred position plus a placeholder at each of the others.
  [[nodiscard]] std::int64_t carrier_bytes(std::int64_t rowid_bytes) const noexcept;
};

/// Port-crossing floor, env-overridable for measurement (default 4). The
/// product below relaxes the VALUE floor for long rides, never this one.
inline int default_min_boundaries()
{
  static int const value = [] {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_MIN_BOUNDARIES");
    if (v == nullptr || v[0] == '\0') { return 4; }
    int parsed      = 0;
    auto const* end = v + std::strlen(v);
    auto const rc   = std::from_chars(v, end, parsed);
    return (rc.ec == std::errc{} && rc.ptr == end && parsed >= 0) ? parsed : 4;
  }();
  return value;
}

/// Value x crossings floor (default 128 = the 32 B/row x 4 crossings the two
/// independent floors used to imply). Value and crossings TRADE OFF.
inline std::int64_t default_min_value_x_boundaries()
{
  static std::int64_t const value = [] {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_MIN_VALUE_X_BOUNDARIES");
    if (v == nullptr || v[0] == '\0') { return std::int64_t{128}; }
    std::int64_t parsed = 0;
    auto const* end     = v + std::strlen(v);
    auto const rc       = std::from_chars(v, end, parsed);
    return (rc.ec == std::errc{} && rc.ptr == end && parsed >= 0) ? parsed : std::int64_t{128};
  }();
  return value;
}

/// Count-on-deferred admit switch (SIRIUS_EXP_LATE_MAT_COUNT_DEFER, default OFF).
///
/// A column read only by COUNTs can ride and never come back: the aggregate
/// needs the row, not the value. It is dark by default because the shapes that
/// fire it on TPC-H save ~4 B/row, which no A/B run could separate from noise —
/// and because a mis-marked column would lose values nothing restores. A
/// workload that counts WIDE columns over long rides can turn it on.
inline bool count_on_deferred_enabled()
{
  static bool const enabled = [] {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_COUNT_DEFER");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

/// Deferred-value floor for a bundle whose origin is COMPRESSED in the pin
/// (SIRIUS_EXP_LATE_MAT_MIN_VALUE_COMPRESSED, default: the ordinary floor).
///
/// Such a bundle saves more than the ride: the scan can skip decompressing what
/// it is about to replace with a rowid. Until that decode-skip exists the two
/// floors are the same number, and this knob is how the difference gets
/// measured when it does.
inline std::int64_t min_value_bytes_compressed(std::int64_t ordinary)
{
  static std::int64_t const value = []() -> std::int64_t {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_MIN_VALUE_COMPRESSED");
    if (v == nullptr || v[0] == '\0') { return -1; }
    std::int64_t parsed = 0;
    auto const* end     = v + std::strlen(v);
    auto const rc       = std::from_chars(v, end, parsed);
    return (rc.ec == std::errc{} && rc.ptr == end && parsed >= 0) ? parsed : -1;
  }();
  return value < 0 ? ordinary : value;
}

/// How much dearer a materialization is when the pin lives on the HOST tier
/// (SIRIUS_EXP_LATE_MAT_HOST_COST_MULTIPLIER, default 12).
///
/// A host-tier ride is gathered out of pinned host blocks over the CPU-GPU link
/// rather than out of device memory, while the value floor below is calibrated
/// against a device gather. The floor and the gather scale with the same row
/// count, so the correction is a plain multiplier on the product floor.
///
/// 12 is the worst measured ratio of a blocked host gather to the equivalent
/// device gather on GB300 (a 150M x 8 B pinned column, random ids, selectivities
/// from 0.01% to 100%: 2.0x, 8.1x, 10.6x, 12.2x, 11.0x, 6.6x). That bounds ONE
/// operation rather than calibrating a query, which is why it is a knob.
inline std::int64_t host_tier_cost_multiplier()
{
  static std::int64_t const value = [] {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_HOST_COST_MULTIPLIER");
    if (v == nullptr || v[0] == '\0') { return std::int64_t{12}; }
    std::int64_t parsed = 0;
    auto const* end     = v + std::strlen(v);
    auto const rc       = std::from_chars(v, end, parsed);
    return (rc.ec == std::errc{} && rc.ptr == end && parsed >= 1) ? parsed : std::int64_t{12};
  }();
  return value;
}

/// The thresholds, in one place so a measurement can move them together.
struct defer_policy {
  /// A ride must save SOMETHING per row after the rowid; whether it repays is
  /// @ref min_value_x_boundaries.
  std::int64_t min_value_bytes = 1;
  /// Net value per row TIMES crossings saved — what the floor is really about.
  std::int64_t min_value_x_boundaries = default_min_value_x_boundaries();
  /// Port crossings the ride must save. Overridable with
  /// SIRIUS_EXP_LATE_MAT_MIN_BOUNDARIES so a measurement can separate "the ride is
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
