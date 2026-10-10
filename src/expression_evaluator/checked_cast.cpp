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

// sirius
#include <expression_evaluator/checked_cast.hpp>
#include <expression_evaluator/round_floating_point.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/types/cast_helpers.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>

// standard library
#include <string>
#include <utility>

namespace sirius {
namespace {

constexpr cudf::data_type bool_type{cudf::type_id::BOOL8};

using int128 = __int128_t;

int128 power_of_ten(int32_t digits)
{
  int128 result = 1;
  for (int32_t i = 0; i < digits; ++i) {
    result *= 10;
  }
  return result;
}

std::unique_ptr<cudf::column> compare(cudf::column_view const& values,
                                      cudf::scalar const& bound,
                                      cudf::binary_operator op,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return cudf::binary_operation(values, bound, op, bool_type, stream, mr);
}

/// AND of two BOOL8 columns, either of which may be nullptr for "all true".
std::unique_ptr<cudf::column> both(std::unique_ptr<cudf::column> lhs,
                                   std::unique_ptr<cudf::column> rhs,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  if (!lhs) { return rhs; }
  if (!rhs) { return lhs; }
  return cudf::binary_operation(
    lhs->view(), rhs->view(), cudf::binary_operator::LOGICAL_AND, bool_type, stream, mr);
}

/// BOOL8 column, true where `lower < value < upper`. NULL stays NULL and NaN is false.
std::unique_ptr<cudf::column> strictly_between(cudf::column_view const& values,
                                               cudf::scalar const& lower,
                                               cudf::scalar const& upper,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  return both(compare(values, lower, cudf::binary_operator::GREATER, stream, mr),
              compare(values, upper, cudf::binary_operator::LESS, stream, mr),
              stream,
              mr);
}

/// Checks integer and DECIMAL columns; for DECIMAL the bounds apply to the unscaled value.
struct within_fn {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values,
                                           int128 lower,
                                           int128 upper,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if constexpr (cudf::is_integral_not_bool<T>() || cudf::is_fixed_point<T>()) {
      using rep  = cudf::device_storage_type_t<T>;
      auto bound = [&](int128 value) -> std::unique_ptr<cudf::scalar> {
        if constexpr (cudf::is_fixed_point<T>()) {
          return std::make_unique<cudf::fixed_point_scalar<T>>(
            static_cast<rep>(value), numeric::scale_type{values.type().scale()}, true, stream, mr);
        } else {
          return std::make_unique<cudf::numeric_scalar<T>>(static_cast<T>(value), true, stream, mr);
        }
      };
      // A bound beyond the storage type's range holds for every value.
      auto above =
        lower > static_cast<int128>(cuda::std::numeric_limits<rep>::min())
          ? compare(values, *bound(lower), cudf::binary_operator::GREATER_EQUAL, stream, mr)
          : nullptr;
      auto below = upper < static_cast<int128>(cuda::std::numeric_limits<rep>::max())
                     ? compare(values, *bound(upper), cudf::binary_operator::LESS_EQUAL, stream, mr)
                     : nullptr;
      return both(std::move(above), std::move(below), stream, mr);
    } else {
      throw invalid_input_exception("[checked_cast] expected an integer or DECIMAL column, got {}",
                                    cudf::type_to_name(values.type()));
    }
  }
};

/// BOOL8 column, true where `lower <= value <= upper`, for @p lower <= 0 <= @p upper. NULL stays
/// NULL. Returns nullptr when the type cannot hold a value outside the bounds.
std::unique_ptr<cudf::column> within(cudf::column_view const& values,
                                     int128 lower,
                                     int128 upper,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(values.type(), within_fn{}, values, lower, upper, stream, mr);
}

/// BOOL8 column, true where the unscaled DECIMAL or integer value has at most @p digits digits.
/// Returns nullptr when the type cannot hold a longer value.
std::unique_ptr<cudf::column> digits_fit(cudf::column_view const& values,
                                         int32_t digits,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto const limit = power_of_ten(digits) - 1;
  return within(values, -limit, limit, stream, mr);
}

/// Nulls the rows of @p result where @p fits is false. A false row that was not NULL in the input
/// is a failed cast: it throws unless @p try_cast. A nullptr @p fits means every row fits.
/// Synchronizes @p stream.
std::unique_ptr<cudf::column> apply_fits(std::unique_ptr<cudf::column> result,
                                         std::unique_ptr<cudf::column> fits,
                                         cudf::size_type input_null_count,
                                         bool try_cast,
                                         std::string const& target_name,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  if (!fits) { return result; }
  auto [mask, null_count] = cudf::bools_to_mask(fits->view(), stream, mr);
  if (!try_cast && null_count > input_null_count) {
    throw invalid_input_exception("Could not cast {} value(s) to {}: out of range, NaN or infinite",
                                  null_count - input_null_count,
                                  target_name);
  }
  result->set_null_mask(std::move(*mask), null_count);
  return result;
}

std::string decimal_name(cudf::data_type target, uint8_t precision)
{
  return "DECIMAL(" + std::to_string(precision) + "," + std::to_string(-target.scale()) + ")";
}

/// Inclusive value range of an integer target type.
struct integer_range {
  int128 lower;
  int128 upper;
};

struct integer_range_fn {
  template <typename T>
  integer_range operator()() const
  {
    if constexpr (cudf::is_integral_not_bool<T>()) {
      return {cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::max()};
    } else {
      throw invalid_input_exception("[cast_to_integer] expected an integer target type");
    }
  }
};

std::unique_ptr<cudf::column> cast_floating_to_decimal(cudf::column_view const& input,
                                                       cudf::data_type target,
                                                       uint8_t precision,
                                                       bool try_cast,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  // Mirrors DuckDB's DoubleToDecimalCast step by step. Each step is one correctly rounded IEEE
  // operation, or an exact one, on the same operands, so the GPU result is bit-identical.
  auto const fp64     = cudf::data_type{cudf::type_id::FLOAT64};
  auto const is_float = input.type().id() == cudf::type_id::FLOAT32;
  // DuckDB scales FLOAT in DOUBLE; widening is exact.
  auto widened       = is_float ? cudf::cast(input, fp64, stream, mr) : nullptr;
  auto const doubles = widened ? widened->view() : input;
  // The constants DuckDB scales and range-checks with.
  auto const* powers = duckdb::NumericHelper::DOUBLE_POWERS_OF_TEN;
  cudf::numeric_scalar<double> factor(powers[-target.scale()], true, stream, mr);
  auto scaled =
    cudf::binary_operation(doubles, factor, cudf::binary_operator::MUL, fp64, stream, mr);
  // round(), ties away from zero.
  auto rounded = round_floating_point(scaled->view(), 0, stream, mr);
  cudf::numeric_scalar<double> lower(-powers[precision], true, stream, mr);
  cudf::numeric_scalar<double> upper(powers[precision], true, stream, mr);
  auto fits = strictly_between(rounded->view(), lower, upper, stream, mr);

  // Only finite whole numbers below 10^38 reach the conversion; failed and NULL rows become 0.
  cudf::numeric_scalar<double> zero(0.0, true, stream, mr);
  auto whole = cudf::copy_if_else(rounded->view(), zero, fits->view(), stream, mr);
  // DuckDB stores static_cast<float>(rounded) for FLOAT input.
  if (is_float) {
    whole = cudf::cast(whole->view(), cudf::data_type{cudf::type_id::FLOAT32}, stream, mr);
  }
  // A whole number converts to scale 0 exactly; it already is the unscaled target value.
  auto unscaled =
    cudf::cast(whole->view(), cudf::data_type{target.id(), numeric::scale_type{0}}, stream, mr);
  auto const size = unscaled->size();
  auto contents   = unscaled->release();
  auto result     = std::make_unique<cudf::column>(
    target, size, std::move(*contents.data), rmm::device_buffer{}, 0);
  return apply_fits(std::move(result),
                    std::move(fits),
                    input.null_count(),
                    try_cast,
                    decimal_name(target, precision),
                    stream,
                    mr);
}

std::unique_ptr<cudf::column> cast_decimal_to_decimal(cudf::column_view const& input,
                                                      cudf::data_type target,
                                                      uint8_t precision,
                                                      bool try_cast,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const source_scale = -input.type().scale();
  auto const target_scale = -target.scale();
  if (target_scale < source_scale) {
    // HALF_UP rounds ties away from zero, like DuckDB's DecimalScaleDownOperator; cudf::cast
    // would truncate.
    auto rounded =
      cudf::round_decimal(input, target_scale, cudf::rounding_method::HALF_UP, stream, mr);
    auto fits = digits_fit(rounded->view(), precision, stream, mr);
    return apply_fits(cudf::cast(rounded->view(), target, stream, mr),
                      std::move(fits),
                      input.null_count(),
                      try_cast,
                      decimal_name(target, precision),
                      stream,
                      mr);
  }
  // Scaling up is exact. Check the source, so the multiplication cannot overflow.
  auto fits = digits_fit(input, precision - (target_scale - source_scale), stream, mr);
  return apply_fits(cudf::cast(input, target, stream, mr),
                    std::move(fits),
                    input.null_count(),
                    try_cast,
                    decimal_name(target, precision),
                    stream,
                    mr);
}

std::unique_ptr<cudf::column> cast_integer_to_decimal(cudf::column_view const& input,
                                                      cudf::data_type target,
                                                      uint8_t precision,
                                                      bool try_cast,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  // DuckDB's StandardNumericToDecimalCast: |x| < 10^(precision - scale). Check the source, so the
  // multiplication by 10^scale cannot overflow.
  auto fits = digits_fit(input, precision + target.scale(), stream, mr);
  return apply_fits(cudf::cast(input, target, stream, mr),
                    std::move(fits),
                    input.null_count(),
                    try_cast,
                    decimal_name(target, precision),
                    stream,
                    mr);
}

/// FLOAT32 or FLOAT64 scalar of @p type holding @p value.
std::unique_ptr<cudf::scalar> floating_scalar(cudf::data_type type,
                                              double value,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  if (type.id() == cudf::type_id::FLOAT32) {
    return cudf::make_fixed_width_scalar(static_cast<float>(value), stream, mr);
  }
  return cudf::make_fixed_width_scalar(value, stream, mr);
}

std::unique_ptr<cudf::column> cast_floating_to_integer(cudf::column_view const& input,
                                                       cudf::data_type target,
                                                       bool try_cast,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  // DuckDB's TryCastWithOverflowCheckFloat: `min <= x < max + 1` in the input's type, then
  // std::nearbyint, ties to even. The bounds are 0 or powers of two, exact in FLOAT and DOUBLE.
  auto const range = cudf::type_dispatcher(target, integer_range_fn{});
  auto lower       = floating_scalar(input.type(), static_cast<double>(range.lower), stream, mr);
  auto upper   = floating_scalar(input.type(), static_cast<double>(range.upper + 1), stream, mr);
  auto zero    = floating_scalar(input.type(), 0.0, stream, mr);
  auto rounded = cudf::unary_operation(input, cudf::unary_operator::RINT, stream, mr);
  // Checking the rounded value against the upper bound also rejects values that round up to it,
  // which overflow in DuckDB.
  auto fits = both(compare(input, *lower, cudf::binary_operator::GREATER_EQUAL, stream, mr),
                   compare(rounded->view(), *upper, cudf::binary_operator::LESS, stream, mr),
                   stream,
                   mr);
  // Only whole numbers in range reach the conversion; failed and NULL rows become 0.
  auto whole = cudf::copy_if_else(rounded->view(), *zero, fits->view(), stream, mr);
  return apply_fits(cudf::cast(whole->view(), target, stream, mr),
                    std::move(fits),
                    input.null_count(),
                    try_cast,
                    cudf::type_to_name(target),
                    stream,
                    mr);
}

}  // namespace

std::unique_ptr<cudf::column> cast_to_decimal(cudf::column_view const& input,
                                              cudf::data_type target,
                                              uint8_t precision,
                                              bool try_cast,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  if (!cudf::is_fixed_point(target) || precision < 1 || precision > 38) {
    throw invalid_input_exception(
      "[cast_to_decimal] unsupported target type id={} with precision {}",
      static_cast<int>(target.id()),
      precision);
  }
  if (cudf::is_floating_point(input.type())) {
    return cast_floating_to_decimal(input, target, precision, try_cast, stream, mr);
  }
  if (cudf::is_fixed_point(input.type())) {
    return cast_decimal_to_decimal(input, target, precision, try_cast, stream, mr);
  }
  if (cudf::is_integral_not_bool(input.type())) {
    return cast_integer_to_decimal(input, target, precision, try_cast, stream, mr);
  }
  throw invalid_input_exception("[cast_to_decimal] unsupported source type id={}",
                                static_cast<int>(input.type().id()));
}

std::unique_ptr<cudf::column> cast_to_integer(cudf::column_view const& input,
                                              cudf::data_type target,
                                              bool try_cast,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  if (!cudf::is_integral_not_bool(target)) {
    throw invalid_input_exception("[cast_to_integer] unsupported target type {}",
                                  cudf::type_to_name(target));
  }
  if (cudf::is_floating_point(input.type())) {
    return cast_floating_to_integer(input, target, try_cast, stream, mr);
  }
  auto const range = cudf::type_dispatcher(target, integer_range_fn{});
  if (cudf::is_fixed_point(input.type())) {
    // DuckDB's TryCastDecimalToNumeric rounds half away from zero to a whole number first.
    auto rounded = cudf::round_decimal(input, 0, cudf::rounding_method::HALF_UP, stream, mr);
    auto fits    = within(rounded->view(), range.lower, range.upper, stream, mr);
    return apply_fits(cudf::cast(rounded->view(), target, stream, mr),
                      std::move(fits),
                      input.null_count(),
                      try_cast,
                      cudf::type_to_name(target),
                      stream,
                      mr);
  }
  if (cudf::is_integral_not_bool(input.type())) {
    auto fits = within(input, range.lower, range.upper, stream, mr);
    return apply_fits(cudf::cast(input, target, stream, mr),
                      std::move(fits),
                      input.null_count(),
                      try_cast,
                      cudf::type_to_name(target),
                      stream,
                      mr);
  }
  throw invalid_input_exception("[cast_to_integer] unsupported source type {}",
                                cudf::type_to_name(input.type()));
}

}  // namespace sirius
