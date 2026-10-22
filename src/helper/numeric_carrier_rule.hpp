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

// The one place the numeric-carrier rules are written down, for both host and device code.
//
// Compressed materialization (src/helper/numeric_narrowing.hpp) decides on the host whether a
// column's values fit a narrower carrier; the membership dynamic filters
// (src/cuda/dynamic_filter_probe.cuh) make the same decision per element in a kernel when a probe
// arrives at a carrier wider than the key rep. Both call `value_fits` below, so the host predicate
// and the device conversion cannot drift. Likewise `integer_storage_type` is the single answer to
// "which cudf types are plain integers underneath": the narrowing helper and the membership
// filters both read a temporal column through it.
//
// This header is deliberately light (cudf/types.hpp for the type ids and the CUDF_HOST_DEVICE
// macro, libcu++ for limits and traits) so a CUDA translation unit can include it without pulling
// in host-only cudf APIs.

#include <cudf/types.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdint>

namespace sirius {

/**
 * @brief cudf type whose buffer layout @p t shares
 *
 * Temporal columns store plain integers: `TIMESTAMP_DAYS` is int32 epoch days and every other
 * timestamp unit is int64 ticks, so those map to `INT32` / `INT64`. Every other type, including
 * fixed-point (whose scale is part of the type), is its own storage type.
 *
 * Two consumers apply this differently, on purpose:
 *   * the membership filters read *every* temporal column through it, because a device kernel
 *     compares the integer bits and the key family (`membership_key_family`) decides which probe
 *     types are comparable;
 *   * carrier narrowing (`narrowing_rep_type`) applies it only to `TIMESTAMP_DAYS`, the one
 *     temporal type with a narrowing domain (`narrow_domain_of`); sub-day timestamps keep their
 *     own type there so a plain INT32/INT64 carrier can never pass the validators that reject a
 *     carrier contradicting a column's declared timestamp type.
 */
[[nodiscard]] constexpr cudf::data_type integer_storage_type(cudf::data_type t) noexcept
{
  switch (t.id()) {
    case cudf::type_id::TIMESTAMP_DAYS: return cudf::data_type{cudf::type_id::INT32};
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return cudf::data_type{cudf::type_id::INT64};
    default: return t;
  }
}

/**
 * @brief True when @p value is exactly representable in the integer type @p To
 *
 * The lossless-conversion rule shared by the host narrowing predicates (`numeric_range_fits`) and
 * the device probe adapters (`probe_key_convert`). Widening within a signedness always fits; a
 * negative value never fits an unsigned type; otherwise the value must lie within `To`'s range.
 * @p From may be any integer type including `__int128_t`; only `To`'s limits are consulted, so
 * `To` must be a type `::cuda::std::numeric_limits` specializes (the eight fixed-width integers).
 */
template <class To, class From>
[[nodiscard]] CUDF_HOST_DEVICE constexpr bool value_fits(From value) noexcept
{
  static_assert(::cuda::std::is_integral_v<To> || ::cuda::std::is_same_v<To, __int128_t>,
                "value_fits targets an integer type");
  static_assert(::cuda::std::is_integral_v<From> || ::cuda::std::is_same_v<From, __int128_t>,
                "value_fits converts from an integer type");
  using limits                 = ::cuda::std::numeric_limits<To>;
  constexpr bool from_signed   = ::cuda::std::is_signed_v<From>;
  constexpr bool to_signed     = ::cuda::std::is_signed_v<To>;
  constexpr bool from_is_wider = sizeof(From) > sizeof(To);
  if constexpr (from_signed == to_signed) {
    if constexpr (!from_is_wider) {
      return true;
    } else {
      return value >= static_cast<From>(limits::min()) && value <= static_cast<From>(limits::max());
    }
  } else if constexpr (from_signed) {
    // signed -> unsigned: non-negative, and within To's maximum when From can exceed it.
    if (value < From{0}) { return false; }
    if constexpr (sizeof(From) <= sizeof(To)) {
      return true;
    } else {
      return value <= static_cast<From>(limits::max());
    }
  } else {
    // unsigned -> signed: fits when From is strictly narrower, else must not exceed To's maximum.
    if constexpr (sizeof(From) < sizeof(To)) {
      return true;
    } else {
      return value <= static_cast<From>(limits::max());
    }
  }
}

/**
 * @brief True when the inclusive range [@p minimum, @p maximum] lies within the integer type @p To
 *
 * The host-side form of `value_fits` over exact column bounds (see `compute_exact_numeric_range`).
 */
template <class To>
[[nodiscard]] CUDF_HOST_DEVICE constexpr bool range_fits(__int128_t minimum,
                                                         __int128_t maximum) noexcept
{
  return value_fits<To>(minimum) && value_fits<To>(maximum);
}

/**
 * @brief Value a membership hash set reserves as its empty slot, which therefore cannot be stored
 *
 * Signed reps use the minimum; unsigned reps use the maximum because 0 is a common real key. The
 * hash IN-list keeps a probe equal to it conservatively (it may have been a build key cuco could
 * not insert); tests reach the same value through this definition rather than restating it.
 */
template <class KeyT>
struct set_sentinel {
  static constexpr KeyT value = ::cuda::std::is_signed_v<KeyT>
                                  ? ::cuda::std::numeric_limits<KeyT>::min()
                                  : ::cuda::std::numeric_limits<KeyT>::max();
};

}  // namespace sirius
