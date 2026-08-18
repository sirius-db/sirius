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
#include "helper/numeric_narrowing.hpp"

#include "cudf/cudf_utils.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace sirius {
namespace {

template <typename T>
bool fits(__int128_t minimum, __int128_t maximum)
{
  return minimum >= static_cast<__int128_t>(std::numeric_limits<T>::lowest()) &&
         maximum <= static_cast<__int128_t>(std::numeric_limits<T>::max());
}

// True when `source` and `target` are supported numeric carriers in the same family: integral
// carriers preserving signedness, or fixed-point carriers preserving their cuDF scale. Both sides
// are taken under narrowing_rep_type, so DATE participates as its int32 representation.
bool same_numeric_carrier_family(cudf::data_type source, cudf::data_type target)
{
  auto const source_rep = narrowing_rep_type(source);
  auto const target_rep = narrowing_rep_type(target);

  auto const same_family = [&] {
    if (cudf::is_integral_not_bool(source_rep) && cudf::is_integral_not_bool(target_rep)) {
      return (cudf::is_signed(source_rep) && cudf::is_signed(target_rep)) ||
             (cudf::is_unsigned(source_rep) && cudf::is_unsigned(target_rep));
    }
    return cudf::is_fixed_point(source_rep) && cudf::is_fixed_point(target_rep) &&
           source_rep.scale() == target_rep.scale();
  }();
  if (!same_family) { return false; }

  // A representation only stands in for its type where that type is the one being narrowed or
  // restored. So when exactly one side is representation-backed, the other side must be strictly
  // narrower than the representation. Without that condition INT64 and TIMESTAMP_DAYS would be
  // same-family, and a plain 64-bit integer would pass the validators that exist to reject a
  // carrier contradicting the column's declared type.
  auto const source_is_rep_backed = source_rep != source;
  if (source_is_rep_backed == (target_rep != target)) { return true; }
  auto const representation = source_is_rep_backed ? source_rep : target_rep;
  auto const carrier        = source_is_rep_backed ? target_rep : source_rep;
  return cudf::size_of(carrier) < cudf::size_of(representation);
}

template <typename T>
std::pair<T, T> numeric_bounds(cudf::scalar const& minimum,
                               cudf::scalar const& maximum,
                               rmm::cuda_stream_view stream)
{
  return {static_cast<cudf::numeric_scalar<T> const&>(minimum).value(stream),
          static_cast<cudf::numeric_scalar<T> const&>(maximum).value(stream)};
}

template <typename Decimal>
std::pair<__int128_t, __int128_t> decimal_bounds(cudf::scalar const& minimum,
                                                 cudf::scalar const& maximum,
                                                 rmm::cuda_stream_view stream)
{
  using scalar_type = cudf::fixed_point_scalar<Decimal>;
  return {static_cast<__int128_t>(static_cast<scalar_type const&>(minimum).value(stream)),
          static_cast<__int128_t>(static_cast<scalar_type const&>(maximum).value(stream))};
}

std::optional<numeric_range> range_from_scalars(cudf::scalar const& minimum,
                                                cudf::scalar const& maximum,
                                                uint8_t decimal_scale,
                                                rmm::cuda_stream_view stream)
{
  switch (minimum.type().id()) {
    case cudf::type_id::INT8: {
      auto const [lo, hi] = numeric_bounds<int8_t>(minimum, maximum, stream);
      return signed_integer_range(lo, hi);
    }
    case cudf::type_id::INT16: {
      auto const [lo, hi] = numeric_bounds<int16_t>(minimum, maximum, stream);
      return signed_integer_range(lo, hi);
    }
    case cudf::type_id::INT32: {
      auto const [lo, hi] = numeric_bounds<int32_t>(minimum, maximum, stream);
      return signed_integer_range(lo, hi);
    }
    case cudf::type_id::INT64: {
      auto const [lo, hi] = numeric_bounds<int64_t>(minimum, maximum, stream);
      return signed_integer_range(lo, hi);
    }
    case cudf::type_id::UINT8: {
      auto const [lo, hi] = numeric_bounds<uint8_t>(minimum, maximum, stream);
      return unsigned_integer_range(lo, hi);
    }
    case cudf::type_id::UINT16: {
      auto const [lo, hi] = numeric_bounds<uint16_t>(minimum, maximum, stream);
      return unsigned_integer_range(lo, hi);
    }
    case cudf::type_id::UINT32: {
      auto const [lo, hi] = numeric_bounds<uint32_t>(minimum, maximum, stream);
      return unsigned_integer_range(lo, hi);
    }
    case cudf::type_id::UINT64: {
      auto const [lo, hi] = numeric_bounds<uint64_t>(minimum, maximum, stream);
      return unsigned_integer_range(lo, hi);
    }
    case cudf::type_id::DECIMAL32: {
      auto const [lo, hi] = decimal_bounds<numeric::decimal32>(minimum, maximum, stream);
      return decimal_range(lo, hi, decimal_scale);
    }
    case cudf::type_id::DECIMAL64: {
      auto const [lo, hi] = decimal_bounds<numeric::decimal64>(minimum, maximum, stream);
      return decimal_range(lo, hi, decimal_scale);
    }
    case cudf::type_id::DECIMAL128: {
      auto const [lo, hi] = decimal_bounds<numeric::decimal128>(minimum, maximum, stream);
      return decimal_range(lo, hi, decimal_scale);
    }
    default: return std::nullopt;
  }
}

bool is_supported_numeric_carrier(cudf::data_type type)
{
  auto const rep = narrowing_rep_type(type);
  return cudf::is_integral_not_bool(rep) || cudf::is_fixed_point(rep);
}

}  // namespace

cudf::data_type narrowing_rep_type(cudf::data_type type) noexcept
{
  return type.id() == cudf::type_id::TIMESTAMP_DAYS ? cudf::data_type{cudf::type_id::INT32} : type;
}

cudf::column_view narrowing_rep_view(cudf::column_view const& column)
{
  auto const rep = narrowing_rep_type(column.type());
  return rep == column.type() ? column : cudf::bit_cast(column, rep);
}

std::unique_ptr<cudf::column> cast_through_rep(cudf::column_view const& column,
                                               cudf::data_type target,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  // Given caller-established carrier provenance, tunnel only when exactly one side is a
  // representation-backed DATE and the other is a strict narrow carrier. Every other pair,
  // including DATE<->TIMESTAMP, follows cuDF's conversion rules.
  auto const source     = column.type();
  auto const rep_target = narrowing_rep_type(target);
  auto const narrows_carrier =
    narrowing_rep_type(source) != source && can_narrow_to(source, target);
  auto const restores_carrier = rep_target != target && can_restore_to(source, target);
  if (!narrows_carrier && !restores_carrier) { return cudf::cast(column, target, stream, mr); }

  auto result = cudf::cast(narrowing_rep_view(column), rep_target, stream, mr);
  if (rep_target == target) { return result; }

  // Re-tag the int32 result as the temporal target. bit_cast only yields a view, and the caller
  // needs ownership, so hand the buffers to a new column instead of copying them. Bit-castability
  // is what makes the move sound, and it also rules out the compound types whose children
  // `release` would drop.
  if (!cudf::is_bit_castable(rep_target, target)) {
    throw internal_exception("narrowing representation is not bit-castable to its carrier");
  }
  auto const size       = result->size();
  auto const null_count = result->null_count();
  auto contents         = result->release();
  return std::make_unique<cudf::column>(
    target, size, std::move(*contents.data), std::move(*contents.null_mask), null_count);
}

narrow_domain narrow_domain_of(const logical_type& type) noexcept
{
  switch (type.id()) {
    case type_id::TINYINT:
    case type_id::SMALLINT:
    case type_id::INTEGER:
    case type_id::BIGINT: return narrow_domain::SIGNED_INTEGER;
    case type_id::UTINYINT:
    case type_id::USMALLINT:
    case type_id::UINTEGER:
    case type_id::UBIGINT: return narrow_domain::UNSIGNED_INTEGER;
    case type_id::DECIMAL: return narrow_domain::DECIMAL;
    // DATE is int32 days from epoch, narrowed through narrowing_rep_type. Narrowing is absolute,
    // with no frame of reference subtracted, so what has to fit is the distance from the 1970
    // epoch: int16 days reach from 1880 to 2059. Other temporal types get no domain because their
    // sub-day units provide no useful span in a 16-bit carrier.
    case type_id::DATE: return narrow_domain::DATE;
    default: return narrow_domain::NONE;
  }
}

bool is_narrowable_numeric_type(const logical_type& type) noexcept
{
  // Having a domain is necessary but not sufficient: the already-minimal carriers mean something
  // narrowable and still have nowhere narrower to go. No `default`, so a new domain has to be
  // ruled on here rather than silently inheriting one of these answers.
  switch (narrow_domain_of(type)) {
    case narrow_domain::NONE: return false;
    case narrow_domain::SIGNED_INTEGER: return type.id() != type_id::TINYINT;
    case narrow_domain::UNSIGNED_INTEGER: return type.id() != type_id::UTINYINT;
    // DuckDB's DECIMAL16 carrier has no cuDF equivalent; DECIMAL32 is already the narrowest
    // fixed-point carrier supported by cuDF.
    case narrow_domain::DECIMAL:
      return type.decimal_precision() > logical_type::decimal_max_precision_int32;
    case narrow_domain::DATE: return true;
  }
  return false;
}

bool can_narrow_to(cudf::data_type source, cudf::data_type target)
{
  return same_numeric_carrier_family(source, target) &&
         cudf::size_of(source) > cudf::size_of(target);
}

bool can_restore_to(cudf::data_type source, cudf::data_type target)
{
  return same_numeric_carrier_family(source, target) &&
         cudf::size_of(source) < cudf::size_of(target);
}

bool numeric_range_fits(cudf::data_type target, const numeric_range& range) noexcept
{
  if (range.minimum > range.maximum) { return false; }

  switch (target.id()) {
    case cudf::type_id::INT8:
      return range.domain == numeric_range_domain::SIGNED_INTEGER &&
             fits<int8_t>(range.minimum, range.maximum);
    case cudf::type_id::INT16:
      return range.domain == numeric_range_domain::SIGNED_INTEGER &&
             fits<int16_t>(range.minimum, range.maximum);
    case cudf::type_id::INT32:
      return range.domain == numeric_range_domain::SIGNED_INTEGER &&
             fits<int32_t>(range.minimum, range.maximum);
    case cudf::type_id::INT64:
      return range.domain == numeric_range_domain::SIGNED_INTEGER &&
             fits<int64_t>(range.minimum, range.maximum);
    case cudf::type_id::UINT8:
      return range.domain == numeric_range_domain::UNSIGNED_INTEGER && range.minimum >= 0 &&
             fits<uint8_t>(range.minimum, range.maximum);
    case cudf::type_id::UINT16:
      return range.domain == numeric_range_domain::UNSIGNED_INTEGER && range.minimum >= 0 &&
             fits<uint16_t>(range.minimum, range.maximum);
    case cudf::type_id::UINT32:
      return range.domain == numeric_range_domain::UNSIGNED_INTEGER && range.minimum >= 0 &&
             fits<uint32_t>(range.minimum, range.maximum);
    case cudf::type_id::UINT64:
      return range.domain == numeric_range_domain::UNSIGNED_INTEGER && range.minimum >= 0 &&
             fits<uint64_t>(range.minimum, range.maximum);
    case cudf::type_id::DECIMAL32:
      return range.domain == numeric_range_domain::DECIMAL &&
             static_cast<int32_t>(range.decimal_scale) == -target.scale() &&
             fits<int32_t>(range.minimum, range.maximum);
    case cudf::type_id::DECIMAL64:
      return range.domain == numeric_range_domain::DECIMAL &&
             static_cast<int32_t>(range.decimal_scale) == -target.scale() &&
             fits<int64_t>(range.minimum, range.maximum);
    case cudf::type_id::DECIMAL128:
      return range.domain == numeric_range_domain::DECIMAL &&
             static_cast<int32_t>(range.decimal_scale) == -target.scale();
    default: return false;
  }
}

std::optional<cudf::data_type> choose_narrow_physical_type(const logical_type& type,
                                                           const numeric_range& range)
{
  if (!is_narrowable_numeric_type(type)) { return std::nullopt; }
  auto const declared = try_get_cudf_type(type);
  if (!declared) { return std::nullopt; }
  // Candidates are chosen against the representation, so DATE picks from the signed integer set.
  auto const native = narrowing_rep_type(*declared);

  // Narrowest-first candidate carriers of the native carrier's family; `can_narrow_to` keeps only
  // the strictly narrower ones and `numeric_range_fits` picks the first that holds the bounds.
  std::array<cudf::data_type, 3> candidates;
  std::size_t candidate_count = 0;
  if (cudf::is_fixed_point(native)) {
    candidates      = {cudf::data_type{cudf::type_id::DECIMAL32, native.scale()},
                       cudf::data_type{cudf::type_id::DECIMAL64, native.scale()}};
    candidate_count = 2;
  } else if (cudf::is_signed(native)) {
    candidates      = {cudf::data_type{cudf::type_id::INT8},
                       cudf::data_type{cudf::type_id::INT16},
                       cudf::data_type{cudf::type_id::INT32}};
    candidate_count = 3;
  } else {
    candidates      = {cudf::data_type{cudf::type_id::UINT8},
                       cudf::data_type{cudf::type_id::UINT16},
                       cudf::data_type{cudf::type_id::UINT32}};
    candidate_count = 3;
  }

  for (std::size_t i = 0; i < candidate_count; ++i) {
    if (can_narrow_to(native, candidates[i]) && numeric_range_fits(candidates[i], range)) {
      return candidates[i];
    }
  }
  return std::nullopt;
}

std::optional<numeric_range> compute_exact_numeric_range(cudf::column_view const& column,
                                                         logical_type const& logical,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (!is_narrowable_numeric_type(logical) || column.size() == 0 ||
      column.null_count() == column.size()) {
    return std::nullopt;
  }
  auto const native = get_cudf_type(logical);
  if (column.type() != native) { return std::nullopt; }

  return compute_exact_numeric_range(column, stream, mr);
}

std::optional<numeric_range> compute_exact_numeric_range(cudf::column_view const& column,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (!is_supported_numeric_carrier(column.type()) || column.size() == 0 ||
      column.null_count() == column.size()) {
    return std::nullopt;
  }

  // Reduce over the representation: cudf::minmax on a TIMESTAMP_DAYS column yields
  // timestamp_scalars, which range_from_scalars has no arm for.
  auto const rep_column = narrowing_rep_view(column);

  uint8_t decimal_scale = 0;
  if (cudf::is_fixed_point(column.type())) {
    auto const cudf_scale = column.type().scale();
    if (cudf_scale > 0 || cudf_scale < -static_cast<int32_t>(std::numeric_limits<uint8_t>::max())) {
      return std::nullopt;
    }
    decimal_scale = static_cast<uint8_t>(-cudf_scale);
  }

  auto [minimum, maximum] = cudf::minmax(rep_column, stream, mr);
  if (!minimum || !maximum || minimum->type() != maximum->type()) { return std::nullopt; }
  if (!minimum->is_valid(stream) || !maximum->is_valid(stream)) { return std::nullopt; }
  return range_from_scalars(*minimum, *maximum, decimal_scale, stream);
}

}  // namespace sirius
