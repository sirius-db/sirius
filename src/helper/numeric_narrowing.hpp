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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>

namespace sirius {

/**
 * @brief Numeric interpretation of exact range bounds
 *
 * DATE epoch-day bounds use `SIGNED_INTEGER`; `narrow_domain` tracks their distinct semantic
 * domain separately.
 */
enum class numeric_range_domain : uint8_t {
  SIGNED_INTEGER,    ///< Signed integral values
  UNSIGNED_INTEGER,  ///< Unsigned integral values
  DECIMAL            ///< Raw fixed-point values at one SQL scale
};

/**
 * @brief Inclusive host-visible bounds used to select and validate physical carriers
 *
 * DECIMAL bounds contain raw unscaled values, and `decimal_scale` contains their SQL scale.
 * `decimal_scale` is ignored for integral domains.
 */
struct numeric_range {
  numeric_range_domain domain;  ///< Interpretation of the bounds
  __int128_t minimum;           ///< Inclusive minimum
  __int128_t maximum;           ///< Inclusive maximum
  int32_t decimal_scale{0};     ///< SQL scale for DECIMAL bounds
};

/**
 * @brief Constructs exact signed-integer bounds
 *
 * @param minimum Inclusive minimum
 * @param maximum Inclusive maximum
 * @return Bounds in the `SIGNED_INTEGER` domain
 */
[[nodiscard]] constexpr numeric_range signed_integer_range(int64_t minimum, int64_t maximum)
{
  return {numeric_range_domain::SIGNED_INTEGER, minimum, maximum, 0};
}

/**
 * @brief Constructs exact unsigned-integer bounds
 *
 * @param minimum Inclusive minimum
 * @param maximum Inclusive maximum
 * @return Bounds in the `UNSIGNED_INTEGER` domain
 */
[[nodiscard]] constexpr numeric_range unsigned_integer_range(uint64_t minimum, uint64_t maximum)
{
  return {numeric_range_domain::UNSIGNED_INTEGER,
          static_cast<__int128_t>(minimum),
          static_cast<__int128_t>(maximum),
          0};
}

/**
 * @brief Constructs exact fixed-point bounds from raw unscaled values
 *
 * @param minimum Inclusive raw minimum
 * @param maximum Inclusive raw maximum
 * @param scale SQL decimal scale
 * @return Bounds in the `DECIMAL` domain at @p scale
 */
[[nodiscard]] constexpr numeric_range decimal_range(__int128_t minimum,
                                                    __int128_t maximum,
                                                    uint8_t scale)
{
  return {numeric_range_domain::DECIMAL, minimum, maximum, scale};
}

/**
 * @brief Semantic domain preserved by physical carrier narrowing.
 *
 * A carrier does not identify what its bits mean. Two values may be compared at a shared narrow
 * width only when their domains agree. `NONE` denotes types outside the supported narrowing
 * domains.
 */
enum class narrow_domain : uint8_t { NONE, SIGNED_INTEGER, UNSIGNED_INTEGER, DECIMAL, DATE };

/**
 * @brief Returns the semantic narrowing domain of a logical type.
 *
 * Already-minimal integer types still have a domain even though they cannot themselves narrow.
 *
 * @param type Logical type to classify.
 * @return Narrowing domain of @p type, or `narrow_domain::NONE` when unsupported.
 */
[[nodiscard]] narrow_domain narrow_domain_of(const logical_type& type) noexcept;

/**
 * @brief Returns whether a logical type has a narrower exact physical carrier.
 *
 * @param type Logical type to inspect.
 * @return `true` when at least one supported carrier is narrower than the native mapping.
 */
[[nodiscard]] bool is_narrowable_numeric_type(const logical_type& type) noexcept;

/**
 * @brief Returns the physical representation used for carrier narrowing.
 *
 * Maps `TIMESTAMP_DAYS` to `INT32` and returns every other type unchanged.
 *
 * @param type Type to map.
 * @return Narrowing representation of @p type.
 */
[[nodiscard]] cudf::data_type narrowing_rep_type(cudf::data_type type) noexcept;

/**
 * @brief Returns a zero-copy view using `narrowing_rep_type`.
 *
 * The returned view aliases @p column; the input buffers must outlive every use of the result.
 *
 * @param column Non-owning input column view.
 * @return Aliasing view with the narrowing representation type.
 */
[[nodiscard]] cudf::column_view narrowing_rep_view(cudf::column_view const& column);

/**
 * @brief Converts a column, tunneling supported DATE carrier conversions.
 *
 * A conversion between `TIMESTAMP_DAYS` and a strictly narrower signed-integral carrier is
 * treated as physical narrowing or restoration and tunneled through the DATE column's `INT32`
 * representation. All other pairs delegate to `cudf::cast`.
 *
 * This function cannot distinguish that physical conversion from a same-shaped SQL
 * temporal-numeric cast. Callers invoking it for such a pair must establish carrier provenance.
 * Expression evaluation does so with `sirius::ast::cast_kind`; DuckDB-translated semantic
 * temporal-numeric casts are rejected before other AST consumers receive them.
 *
 * The function does not validate bounds before a narrowing conversion. Callers narrowing DATE
 * values must first prove that the values fit @p target.
 *
 * @throw internal_exception If a tunneled result cannot be bit-cast to @p target.
 * @param column Non-owning input view whose buffers remain valid for work on @p stream.
 * @param target Output cuDF type.
 * @param stream CUDA stream used for conversion.
 * @param mr Memory resource used for output allocation.
 * @return Owning converted column.
 */
[[nodiscard]] std::unique_ptr<cudf::column> cast_through_rep(
  cudf::column_view const& column,
  cudf::data_type target,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Checks whether a conversion is a strict carrier-width reduction
 *
 * Integral carriers must preserve signedness, and fixed-point carriers must preserve scale. Family
 * comparison uses `narrowing_rep_type`, so `TIMESTAMP_DAYS` may narrow only to signed-integral
 * carriers smaller than its `INT32` representation.
 *
 * @param source Source cuDF type
 * @param target Candidate narrower cuDF type
 * @return `true` if @p source and @p target share a carrier family and @p target is strictly
 * smaller
 */
[[nodiscard]] bool can_narrow_to(cudf::data_type source, cudf::data_type target);

/**
 * @brief Checks whether a conversion is a strict carrier-width restoration
 *
 * Applies the carrier-family constraints of `can_narrow_to` in the widening direction.
 * `TIMESTAMP_DAYS` may be restored only from a signed-integral carrier smaller than its `INT32`
 * representation.
 *
 * @param source Source cuDF type
 * @param target Candidate wider cuDF type
 * @return `true` if @p source and @p target share a carrier family and @p target is strictly larger
 */
[[nodiscard]] bool can_restore_to(cudf::data_type source, cudf::data_type target);

/**
 * @brief Checks whether exact bounds are representable by a numeric carrier
 *
 * The bounds must be ordered, match @p target's signed, unsigned, or fixed-point domain, and fit
 * its storage width. Fixed-point bounds must also have the same scale as @p target. Unsupported
 * target types do not fit.
 *
 * @param target Candidate numeric carrier
 * @param range Exact bounds to validate
 * @return `true` if @p range can be represented by @p target without changing its values or scale
 */
[[nodiscard]] bool numeric_range_fits(cudf::data_type target, const numeric_range& range) noexcept;

/**
 * @brief Selects the narrowest exact carrier for a logical type and range
 *
 * The result is strictly smaller than the logical type's native representation. DATE candidates
 * are selected against its `INT32` epoch-day representation.
 *
 * @param type Logical type to narrow
 * @param range Exact bounds in the type's carrier domain; DATE uses signed epoch-day bounds
 * @return Narrowest fitting carrier, or `std::nullopt` for unsupported types, incompatible bounds,
 * or no strict width reduction
 */
[[nodiscard]] std::optional<cudf::data_type> choose_narrow_physical_type(
  const logical_type& type, const numeric_range& range);

/**
 * @brief Computes exact bounds for a column matching its declared logical type
 *
 * The logical type must be supported by `is_narrowable_numeric_type`, and @p column must use its
 * native cuDF mapping. Nulls are ignored. DECIMAL bounds contain raw unscaled values and the SQL
 * scale; DATE bounds contain signed epoch days obtained through its `INT32` representation.
 *
 * @param column Non-owning column view to reduce
 * @param logical Declared logical type of @p column
 * @param stream CUDA stream used for the reduction and scalar reads
 * @param mr Memory resource used for reduction allocations
 * @return Exact non-null bounds, or `std::nullopt` when the column is empty, all-null, unsupported,
 * or inconsistent with @p logical
 */
[[nodiscard]] std::optional<numeric_range> compute_exact_numeric_range(
  cudf::column_view const& column,
  logical_type const& logical,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes exact bounds directly from a supported physical carrier
 *
 * Supports signed and unsigned integral carriers, fixed-point carriers with a representable SQL
 * scale, and `TIMESTAMP_DAYS` through its `INT32` representation. Nulls are ignored, and DECIMAL
 * bounds contain raw unscaled values.
 *
 * @param column Non-owning column view to reduce
 * @param stream CUDA stream used for the reduction and scalar reads
 * @param mr Memory resource used for reduction allocations
 * @return Exact non-null bounds, or `std::nullopt` when the column is empty, all-null, unsupported,
 * or has an unsupported fixed-point scale
 */
[[nodiscard]] std::optional<numeric_range> compute_exact_numeric_range(
  cudf::column_view const& column, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

}  // namespace sirius
