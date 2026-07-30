/*
 * Copyright 2025, Sirius Contributors.
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

#include "helper/logical_type.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <optional>

namespace sirius {

enum class numeric_range_domain : uint8_t { SIGNED_INTEGER, UNSIGNED_INTEGER, DECIMAL };

/// Exact source bounds. DECIMAL bounds are the raw unscaled carriers; decimal_scale is the SQL
/// scale and must agree with the logical type before narrowing is allowed.
struct numeric_range {
  numeric_range_domain domain;
  __int128_t minimum;
  __int128_t maximum;
  int32_t decimal_scale{0};
};

[[nodiscard]] constexpr numeric_range signed_integer_range(int64_t minimum, int64_t maximum)
{
  return {numeric_range_domain::SIGNED_INTEGER, minimum, maximum, 0};
}

[[nodiscard]] constexpr numeric_range unsigned_integer_range(uint64_t minimum, uint64_t maximum)
{
  return {numeric_range_domain::UNSIGNED_INTEGER,
          static_cast<__int128_t>(minimum),
          static_cast<__int128_t>(maximum),
          0};
}

[[nodiscard]] constexpr numeric_range decimal_range(__int128_t minimum,
                                                    __int128_t maximum,
                                                    uint8_t scale)
{
  return {numeric_range_domain::DECIMAL, minimum, maximum, scale};
}

[[nodiscard]] bool is_narrowable_numeric_type(const logical_type& type) noexcept;

// A carrier family is a set of supported numeric carriers whose conversions preserve integral
// signedness or fixed-point cuDF scale; narrowing and restoring never cross families.

/// Return true when converting @p source to @p target is a strict same-family width reduction.
[[nodiscard]] bool can_narrow_to(cudf::data_type source, cudf::data_type target);

/// Return true when converting @p source to @p target is a strict same-family width restoration.
[[nodiscard]] bool can_restore_to(cudf::data_type source, cudf::data_type target);

/// Return whether the exact bounds fit in @p target without changing numeric family or decimal
/// scale. This is the single fitting authority: `choose_narrow_physical_type` selects carriers
/// with it and the scan verifies planned downcasts against materialized data with it.
[[nodiscard]] bool numeric_range_fits(cudf::data_type target, const numeric_range& range) noexcept;

/// Return the narrowest exact cuDF carrier that is strictly smaller than the logical type's native
/// carrier. A missing result means unknown/incompatible bounds or no profitable width reduction.
[[nodiscard]] std::optional<cudf::data_type> choose_narrow_physical_type(
  const logical_type& type, const numeric_range& range);

/// Reduce a materialized numeric column to exact host-visible bounds. DECIMAL bounds use the raw
/// unscaled carrier and retain the logical SQL scale. Empty/all-null columns and carrier/logical
/// mismatches return no range.
[[nodiscard]] std::optional<numeric_range> compute_exact_numeric_range(
  cudf::column_view const& column,
  logical_type const& logical,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/// Reduce a materialized supported numeric carrier to exact host-visible bounds, deriving the
/// range domain (and DECIMAL scale) from the carrier itself. Empty/all-null columns return no
/// range.
[[nodiscard]] std::optional<numeric_range> compute_exact_numeric_range(
  cudf::column_view const& column, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

}  // namespace sirius
