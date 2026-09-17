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

// Host-only (no CUDA) classification of membership dynamic-filter keys. The three membership
// filters (small IN-list, hash IN-list, Bloom), the publisher, the publish-plan validator, and the
// planner's direct-route gate consult this header instead of spelling out type lists.
//
// Two closed axes describe every supported key:
//   * the key *rep*: the device element type a set / Bloom / needle buffer is instantiated over.
//     Every supported build type maps onto one of four reps, which bounds storage-variant
//     alternatives and kernel instantiations;
//   * the key *family*: which probe adapter (see cuda/dynamic_filter_probe.cuh) converts a probe
//     column at its own carrier into the rep. A family is the correctness surface: it names the
//     probe carriers whose values are comparable to the stored keys.

// cudf
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <optional>

namespace sirius::op {

/// Device element type of a membership set. Every supported key family maps onto one of these.
enum class membership_key_rep : std::uint8_t { i32, i64, u32, u64 };

/// Probe-adapter selector. Each value names one device adapter (cuda/dynamic_filter_probe.cuh)
/// and one arm each of `classify_membership_key` and `membership_probe_compatible`; a new key
/// family adds a value here and those three arms.
///
/// Temporal keys are integers underneath (cudf stores `TIMESTAMP_DAYS` as int32 epoch days and
/// the other timestamp units as int64 ticks), so they share the signed integral adapter over the
/// column's storage type; the family only decides which probe types are comparable:
///   * `date_days`: a `DATE` key. Probes are `TIMESTAMP_DAYS` or the `INT8`/`INT16`/`INT32`
///     carriers compressed materialization stores a `DATE` in;
///   * `timestamp`: a sub-day timestamp key. Probes must carry the same unit; a different unit
///     is a planner cast and never reaches a membership probe, so it declines rather than
///     converting.
enum class membership_key_family : std::uint8_t { signed_int, unsigned_int, date_days, timestamp };

struct membership_key_domain {
  membership_key_rep rep{membership_key_rep::i32};
  membership_key_family family{membership_key_family::signed_int};
  /// Build column type the filter was constructed from (the carrier the set was published for).
  cudf::data_type native{cudf::type_id::EMPTY};

  [[nodiscard]] bool operator==(membership_key_domain const&) const = default;
};

/**
 * @brief Build-side classification of a key column type
 *
 * The rep is the narrowest listed rep that holds every value of @p build_type, so a build column
 * arriving at a narrowed carrier (compressed materialization) yields a carrier-sized set and wider
 * probes range-check down into it. Returns nullopt for a type no membership filter supports.
 */
[[nodiscard]] std::optional<membership_key_domain> classify_membership_key(
  cudf::data_type build_type) noexcept;

/**
 * @brief Single source of truth for the membership filters' `supports()` type gate
 *
 * True iff `classify_membership_key(t)` has a value. Filters add their own non-type gates (the
 * small IN-list size cap) on top of this; nullable build columns are accepted by all three, with
 * null keys compacted out.
 */
[[nodiscard]] bool membership_key_supported(cudf::data_type t) noexcept;

/**
 * @brief True when a probe column of type @p probe can be adapted to @p domain
 *
 * Host mirror of the device-side adapter dispatch: a probe type this rejects is one every filter's
 * `compute_mask` declines with a null result. Signed and unsigned carriers never mix.
 */
[[nodiscard]] bool membership_probe_compatible(membership_key_domain const& domain,
                                               cudf::data_type probe) noexcept;

/// cudf type of the device element a rep is instantiated over (INT32/INT64/UINT32/UINT64).
[[nodiscard]] cudf::data_type membership_rep_type(membership_key_rep rep) noexcept;

/// Byte width of a rep's device element.
[[nodiscard]] std::size_t membership_rep_bytes(membership_key_rep rep) noexcept;

/**
 * @brief cudf type whose buffer layout @p t shares
 *
 * Temporal columns store plain integers: `TIMESTAMP_DAYS` maps to `INT32`, every other timestamp
 * unit to `INT64`. Any other type is its own storage type. The probe adapters and build-side
 * iterators read a column through this type, so a temporal key is bit-identical to an integer
 * one on the device and adds no kernel instantiations.
 */
[[nodiscard]] cudf::data_type membership_storage_type(cudf::data_type t) noexcept;

}  // namespace sirius::op
