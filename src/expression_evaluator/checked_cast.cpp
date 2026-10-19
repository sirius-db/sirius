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

using int128 = __int128_t;

/// The cast result before failed rows are nulled, and a BOOL8 column that is false where the cast
/// failed. A nullptr `fits` means no row can fail.
using unchecked_cast = std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>;

/// Inclusive range of accepted values.
struct range {
  int128 lower;
  int128 upper;
};

struct integer_range_fn {
  template <typename T>
  range operator()() const
  {
    if constexpr (cudf::is_integral_not_bool<T>()) {
      return {cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::max()};
    } else {
      throw invalid_input_exception("[checked_cast] expected an integer type");
    }
  }
};

range integer_range(cudf::data_type type)
{
  return cudf::type_dispatcher(type, integer_range_fn{});
}

/// Values with at most @p digits digits.
range digits_range(int32_t digits)
{
  int128 limit = 1;
  for (int32_t i = 0; i < digits; ++i) {
    limit *= 10;
  }
  return {1 - limit, limit - 1};
}

std::unique_ptr<cudf::column> compare(cudf::column_view const& values,
                                      cudf::scalar const& bound,
                                      cudf::binary_operator op,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return cudf::binary_operation(
    values, bound, op, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
}

/// AND of two BOOL8 columns, either of which may be nullptr for "all true".
std::unique_ptr<cudf::column> both(std::unique_ptr<cudf::column> lhs,
                                   std::unique_ptr<cudf::column> rhs,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  if (!lhs) { return rhs; }
  if (!rhs) { return lhs; }
  return cudf::binary_operation(lhs->view(),
                                rhs->view(),
                                cudf::binary_operator::LOGICAL_AND,
                                cudf::data_type{cudf::type_id::BOOL8},
                                stream,
                                mr);
}

/// BOOL8 column, true where the integer or unscaled DECIMAL value lies in @p bounds, which holds
/// 0. NULL stays NULL. Returns nullptr when the type cannot hold a value outside the bounds.
struct within_fn {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values,
                                           range bounds,
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
        bounds.lower > static_cast<int128>(cuda::std::numeric_limits<rep>::min())
          ? compare(values, *bound(bounds.lower), cudf::binary_operator::GREATER_EQUAL, stream, mr)
          : nullptr;
      auto below =
        bounds.upper < static_cast<int128>(cuda::std::numeric_limits<rep>::max())
          ? compare(values, *bound(bounds.upper), cudf::binary_operator::LESS_EQUAL, stream, mr)
          : nullptr;
      return both(std::move(above), std::move(below), stream, mr);
    } else {
      throw invalid_input_exception("[checked_cast] expected an integer or DECIMAL column, got {}",
                                    cudf::type_to_name(values.type()));
    }
  }
};

/// Integer or DECIMAL input. Dropping fractional digits rounds half away from zero, as in DuckDB's
/// DecimalScaleDownOperator and TryCastDecimalToNumeric, where cudf::cast would truncate. The rest
/// is exact once the value fits; the check runs on the source, so scaling up cannot overflow.
unchecked_cast cast_integer_or_decimal(cudf::column_view const& input,
                                       cudf::data_type target,
                                       uint8_t precision,
                                       ::cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto rounded =
    target.scale() > input.type().scale()
      ? cudf::round_decimal(input, -target.scale(), cudf::rounding_method::HALF_UP, stream, mr)
      : nullptr;
  auto const values = rounded ? rounded->view() : input;
  // A DECIMAL target holds |x| < 10^(precision - scale), as in DuckDB's CanScaleDownDecimal and
  // StandardNumericToDecimalCast.
  auto const bounds = cudf::is_fixed_point(target)
                        ? digits_range(precision + target.scale() - values.type().scale())
                        : integer_range(target);
  return {cudf::cast(values, target, stream, mr),
          cudf::type_dispatcher(values.type(), within_fn{}, values, bounds, stream, mr)};
}

/// Mirrors DuckDB's DoubleToDecimalCast step by step. Each step is one correctly rounded IEEE
/// operation, or an exact one, on the same operands, so the GPU result is bit-identical.
unchecked_cast cast_floating_to_decimal(cudf::column_view const& input,
                                        cudf::data_type target,
                                        uint8_t precision,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto const fp64 = cudf::data_type{cudf::type_id::FLOAT64};
  // DuckDB scales FLOAT in DOUBLE; widening is exact.
  auto widened =
    input.type().id() == cudf::type_id::FLOAT32 ? cudf::cast(input, fp64, stream, mr) : nullptr;
  auto const* powers = duckdb::NumericHelper::DOUBLE_POWERS_OF_TEN;
  cudf::numeric_scalar<double> factor(powers[-target.scale()], true, stream, mr);
  auto scaled = cudf::binary_operation(
    widened ? widened->view() : input, factor, cudf::binary_operator::MUL, fp64, stream, mr);
  // round(), ties away from zero, must stay below 10^precision in magnitude. NaN and infinities
  // fail.
  auto rounded   = round_floating_point(scaled->view(), 0, stream, mr);
  auto magnitude = cudf::unary_operation(rounded->view(), cudf::unary_operator::ABS, stream, mr);
  cudf::numeric_scalar<double> limit(powers[precision], true, stream, mr);
  auto fits = compare(magnitude->view(), limit, cudf::binary_operator::LESS, stream, mr);

  // Failed and NULL rows become 0, so only whole numbers below 10^38 reach the conversion. A whole
  // number converts to scale 0 exactly and already is the unscaled target value. FLOAT input is not
  // narrowed back to FLOAT, as in DuckDB v2: that would drop digits beyond 2^24.
  cudf::numeric_scalar<double> zero(0.0, true, stream, mr);
  auto whole = cudf::copy_if_else(rounded->view(), zero, fits->view(), stream, mr);
  auto unscaled =
    cudf::cast(whole->view(), cudf::data_type{target.id(), numeric::scale_type{0}}, stream, mr)
      ->release();
  return {std::make_unique<cudf::column>(
            target, input.size(), std::move(*unscaled.data), rmm::device_buffer{}, 0),
          std::move(fits)};
}

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

/// DuckDB v2's TryCastWithOverflowCheckFloat: std::nearbyint, ties to even, then
/// `min <= rounded < max + 1` in the input's type. The bounds are 0 or powers of two, exact in
/// FLOAT and DOUBLE. NaN fails both comparisons; -0.4 rounds to -0.0, which fits unsigned types.
unchecked_cast cast_floating_to_integer(cudf::column_view const& input,
                                        cudf::data_type target,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto const bounds = integer_range(target);
  auto lower        = floating_scalar(input.type(), static_cast<double>(bounds.lower), stream, mr);
  auto upper   = floating_scalar(input.type(), static_cast<double>(bounds.upper + 1), stream, mr);
  auto zero    = floating_scalar(input.type(), 0.0, stream, mr);
  auto rounded = cudf::unary_operation(input, cudf::unary_operator::RINT, stream, mr);
  auto fits =
    both(compare(rounded->view(), *lower, cudf::binary_operator::GREATER_EQUAL, stream, mr),
         compare(rounded->view(), *upper, cudf::binary_operator::LESS, stream, mr),
         stream,
         mr);
  // Only whole numbers in range reach the conversion; failed and NULL rows become 0.
  auto whole = cudf::copy_if_else(rounded->view(), *zero, fits->view(), stream, mr);
  return {cudf::cast(whole->view(), target, stream, mr), std::move(fits)};
}

}  // namespace

std::unique_ptr<cudf::column> checked_cast(cudf::column_view const& input,
                                           cudf::data_type target,
                                           uint8_t precision,
                                           bool try_cast,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const to_decimal = cudf::is_fixed_point(target);
  auto [result, fits]   = !cudf::is_floating_point(input.type())
                            ? cast_integer_or_decimal(input, target, precision, stream, mr)
                          : to_decimal
                            ? cast_floating_to_decimal(input, target, precision, stream, mr)
                            : cast_floating_to_integer(input, target, stream, mr);
  if (!fits) { return std::move(result); }

  // A row that fails but was not NULL in the input is a failed cast.
  auto [mask, null_count] = cudf::bools_to_mask(fits->view(), stream, mr);
  if (!try_cast && null_count > input.null_count()) {
    auto const target_name = to_decimal ? "DECIMAL(" + std::to_string(precision) + "," +
                                            std::to_string(-target.scale()) + ")"
                                        : cudf::type_to_name(target);
    throw invalid_input_exception("Could not cast {} value(s) to {}: out of range, NaN or infinite",
                                  null_count - input.null_count(),
                                  target_name);
  }
  result->set_null_mask(std::move(*mask), null_count);
  return std::move(result);
}

}  // namespace sirius
