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
#include <expression_evaluator/cast_to_decimal.hpp>
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
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

// standard library
#include <utility>

namespace sirius {
namespace {

constexpr cudf::data_type bool_type{cudf::type_id::BOOL8};

/// BOOL8 column, true where `lower < value < upper`. NULL stays NULL and NaN is false.
std::unique_ptr<cudf::column> strictly_between(cudf::column_view const& values,
                                               cudf::scalar const& lower,
                                               cudf::scalar const& upper,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto above =
    cudf::binary_operation(values, lower, cudf::binary_operator::GREATER, bool_type, stream, mr);
  auto below =
    cudf::binary_operation(values, upper, cudf::binary_operator::LESS, bool_type, stream, mr);
  return cudf::binary_operation(
    above->view(), below->view(), cudf::binary_operator::LOGICAL_AND, bool_type, stream, mr);
}

template <typename Rep>
std::unique_ptr<cudf::column> decimal_within(cudf::column_view const& values,
                                             int32_t digits,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  using decimal = numeric::fixed_point<Rep, numeric::Radix::BASE_10>;
  Rep limit     = 1;
  for (int32_t i = 0; i < digits; ++i) {
    limit *= 10;
  }
  auto const scale = numeric::scale_type{values.type().scale()};
  cudf::fixed_point_scalar<decimal> lower(-limit, scale, true, stream, mr);
  cudf::fixed_point_scalar<decimal> upper(limit, scale, true, stream, mr);
  return strictly_between(values, lower, upper, stream, mr);
}

/// BOOL8 column, true where the unscaled DECIMAL value has fewer than @p digits digits. Returns
/// nullptr when the storage type cannot hold a value that long, so every value fits.
std::unique_ptr<cudf::column> decimal_fits(cudf::column_view const& values,
                                           int32_t digits,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  switch (values.type().id()) {
    case cudf::type_id::DECIMAL32:
      return digits >= 9 ? nullptr : decimal_within<int32_t>(values, digits, stream, mr);
    case cudf::type_id::DECIMAL64:
      return digits >= 18 ? nullptr : decimal_within<int64_t>(values, digits, stream, mr);
    case cudf::type_id::DECIMAL128:
      return digits >= 38 ? nullptr : decimal_within<__int128_t>(values, digits, stream, mr);
    default:
      throw invalid_input_exception("[cast_to_decimal] expected a DECIMAL column, got type id={}",
                                    static_cast<int>(values.type().id()));
  }
}

/// Nulls the rows of @p result where @p fits is false. A false row that was not NULL in the input
/// is a failed cast: it throws unless @p try_cast. Synchronizes @p stream.
std::unique_ptr<cudf::column> apply_fits(std::unique_ptr<cudf::column> result,
                                         cudf::column_view const& fits,
                                         cudf::size_type input_null_count,
                                         bool try_cast,
                                         uint8_t precision,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto [mask, null_count] = cudf::bools_to_mask(fits, stream, mr);
  if (!try_cast && null_count > input_null_count) {
    throw invalid_input_exception(
      "Could not cast {} value(s) to DECIMAL({},{}): out of range, NaN or infinite",
      null_count - input_null_count,
      precision,
      -result->type().scale());
  }
  result->set_null_mask(std::move(*mask), null_count);
  return result;
}

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
  return apply_fits(
    std::move(result), fits->view(), input.null_count(), try_cast, precision, stream, mr);
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
    auto fits   = decimal_fits(rounded->view(), precision, stream, mr);
    auto result = cudf::cast(rounded->view(), target, stream, mr);
    if (!fits) { return result; }
    return apply_fits(
      std::move(result), fits->view(), input.null_count(), try_cast, precision, stream, mr);
  }
  // Scaling up is exact. Check the source, so the multiplication cannot overflow.
  auto fits   = decimal_fits(input, precision - (target_scale - source_scale), stream, mr);
  auto result = cudf::cast(input, target, stream, mr);
  if (!fits) { return result; }
  return apply_fits(
    std::move(result), fits->view(), input.null_count(), try_cast, precision, stream, mr);
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
  throw invalid_input_exception("[cast_to_decimal] unsupported source type id={}",
                                static_cast<int>(input.type().id()));
}

}  // namespace sirius
