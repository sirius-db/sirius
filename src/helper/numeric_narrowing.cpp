/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "helper/numeric_narrowing.hpp"

#include "cudf/cudf_utils.hpp"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

#include <limits>

namespace sirius {
namespace {

template <typename T>
bool fits(__int128_t minimum, __int128_t maximum)
{
  return minimum >= static_cast<__int128_t>(std::numeric_limits<T>::lowest()) &&
         maximum <= static_cast<__int128_t>(std::numeric_limits<T>::max());
}

std::optional<cudf::data_type> choose_signed(type_id id, const numeric_range& range)
{
  if (range.domain != numeric_range_domain::SIGNED_INTEGER || range.minimum > range.maximum) {
    return std::nullopt;
  }
  if (fits<int8_t>(range.minimum, range.maximum)) { return cudf::data_type{cudf::type_id::INT8}; }
  if ((id == type_id::INTEGER || id == type_id::BIGINT) &&
      fits<int16_t>(range.minimum, range.maximum)) {
    return cudf::data_type{cudf::type_id::INT16};
  }
  if (id == type_id::BIGINT && fits<int32_t>(range.minimum, range.maximum)) {
    return cudf::data_type{cudf::type_id::INT32};
  }
  return std::nullopt;
}

std::optional<cudf::data_type> choose_unsigned(type_id id, const numeric_range& range)
{
  if (range.domain != numeric_range_domain::UNSIGNED_INTEGER || range.minimum < 0 ||
      range.minimum > range.maximum) {
    return std::nullopt;
  }
  if (fits<uint8_t>(range.minimum, range.maximum)) { return cudf::data_type{cudf::type_id::UINT8}; }
  if ((id == type_id::UINTEGER || id == type_id::UBIGINT) &&
      fits<uint16_t>(range.minimum, range.maximum)) {
    return cudf::data_type{cudf::type_id::UINT16};
  }
  if (id == type_id::UBIGINT && fits<uint32_t>(range.minimum, range.maximum)) {
    return cudf::data_type{cudf::type_id::UINT32};
  }
  return std::nullopt;
}

std::optional<cudf::data_type> choose_decimal(const logical_type& type, const numeric_range& range)
{
  if (range.domain != numeric_range_domain::DECIMAL || range.minimum > range.maximum ||
      range.decimal_scale != type.decimal_scale()) {
    return std::nullopt;
  }

  auto const precision = type.decimal_precision();
  auto const scale     = -static_cast<int32_t>(type.decimal_scale());
  if (precision > logical_type::decimal_max_precision_int32 &&
      fits<int32_t>(range.minimum, range.maximum)) {
    return cudf::data_type{cudf::type_id::DECIMAL32, scale};
  }
  if (precision > logical_type::decimal_max_precision_int64 &&
      fits<int64_t>(range.minimum, range.maximum)) {
    return cudf::data_type{cudf::type_id::DECIMAL64, scale};
  }
  return std::nullopt;
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
  return cudf::is_integral_not_bool(type) || cudf::is_fixed_point(type);
}

}  // namespace

bool is_narrowable_numeric_type(const logical_type& type) noexcept
{
  switch (type.id()) {
    case type_id::SMALLINT:
    case type_id::INTEGER:
    case type_id::BIGINT:
    case type_id::USMALLINT:
    case type_id::UINTEGER:
    case type_id::UBIGINT: return true;
    case type_id::DECIMAL:
      // DuckDB's DECIMAL16 carrier has no cuDF equivalent; DECIMAL32 is already the narrowest
      // fixed-point carrier supported by cuDF.
      return type.decimal_precision() > logical_type::decimal_max_precision_int32;
    default: return false;
  }
}

bool same_numeric_carrier_family(cudf::data_type source, cudf::data_type target)
{
  if (cudf::is_integral_not_bool(source) && cudf::is_integral_not_bool(target)) {
    return (cudf::is_signed(source) && cudf::is_signed(target)) ||
           (cudf::is_unsigned(source) && cudf::is_unsigned(target));
  }
  return cudf::is_fixed_point(source) && cudf::is_fixed_point(target) &&
         source.scale() == target.scale();
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
                                                           const numeric_range& range) noexcept
{
  if (!is_narrowable_numeric_type(type)) { return std::nullopt; }
  switch (type.id()) {
    case type_id::SMALLINT:
    case type_id::INTEGER:
    case type_id::BIGINT: return choose_signed(type.id(), range);
    case type_id::USMALLINT:
    case type_id::UINTEGER:
    case type_id::UBIGINT: return choose_unsigned(type.id(), range);
    case type_id::DECIMAL: return choose_decimal(type, range);
    default: return std::nullopt;
  }
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

  uint8_t decimal_scale = 0;
  if (cudf::is_fixed_point(column.type())) {
    auto const cudf_scale = column.type().scale();
    if (cudf_scale > 0 || cudf_scale < -static_cast<int32_t>(std::numeric_limits<uint8_t>::max())) {
      return std::nullopt;
    }
    decimal_scale = static_cast<uint8_t>(-cudf_scale);
  }

  auto [minimum, maximum] = cudf::minmax(column, stream, mr);
  if (!minimum || !maximum || minimum->type() != maximum->type()) { return std::nullopt; }
  return range_from_scalars(*minimum, *maximum, decimal_scale, stream);
}

}  // namespace sirius
