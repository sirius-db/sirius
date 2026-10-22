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

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <helper/timestamp_semantics.hpp>
#include <sirius/exception.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <variant>

namespace sirius::temporal {

std::unique_ptr<cudf::column> finite_mask(cudf::column_view const& input,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  if (!cudf::is_timestamp(input.type())) {
    throw invalid_input_exception("Temporal finite mask requires a date or timestamp column");
  }
  auto const days = input.type().id() == cudf::type_id::TIMESTAMP_DAYS;
  auto const ticks =
    cudf::bit_cast(input, cudf::data_type{days ? cudf::type_id::INT32 : cudf::type_id::INT64});
  int64_t const limit =
    days ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int64_t>::max();
  cudf::numeric_scalar<int64_t> lower(-limit, true, stream, mr);
  cudf::numeric_scalar<int64_t> upper(limit, true, stream, mr);
  auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
  auto not_negative_infinity =
    cudf::binary_operation(ticks, lower, cudf::binary_operator::NOT_EQUAL, bool_type, stream, mr);
  auto not_positive_infinity =
    cudf::binary_operation(ticks, upper, cudf::binary_operator::NOT_EQUAL, bool_type, stream, mr);
  return cudf::binary_operation(not_negative_infinity->view(),
                                not_positive_infinity->view(),
                                cudf::binary_operator::LOGICAL_AND,
                                bool_type,
                                stream,
                                mr);
}

std::unique_ptr<cudf::column> cast_to_microseconds_checked(cudf::column_view const& input,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  auto const source_type = input.type().id();
  if (source_type != cudf::type_id::TIMESTAMP_SECONDS &&
      source_type != cudf::type_id::TIMESTAMP_MILLISECONDS &&
      source_type != cudf::type_id::TIMESTAMP_NANOSECONDS) {
    throw invalid_input_exception(
      "Checked microsecond conversion requires second, millisecond, or nanosecond timestamps");
  }
  auto const return_type = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  auto const int_type    = cudf::data_type{cudf::type_id::INT64};
  auto const bool_type   = cudf::data_type{cudf::type_id::BOOL8};
  auto ticks             = cudf::bit_cast(input, int_type);
  auto const nanos       = source_type == cudf::type_id::TIMESTAMP_NANOSECONDS;
  auto finite            = finite_mask(input, stream, mr);
  // Do not multiply infinity sentinels, even in rows overwritten below.
  cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
  auto finite_ticks   = cudf::copy_if_else(ticks, zero, finite->view(), stream, mr);
  int64_t const scale = source_type == cudf::type_id::TIMESTAMP_SECONDS ? 1000000 : 1000;
  if (!nanos) {
    // Integer division truncates toward zero, giving inclusive safe bounds.
    cudf::numeric_scalar<int64_t> lower(
      std::numeric_limits<int64_t>::min() / scale, true, stream, mr);
    cudf::numeric_scalar<int64_t> upper(
      std::numeric_limits<int64_t>::max() / scale, true, stream, mr);
    auto below = cudf::binary_operation(
      finite_ticks->view(), lower, cudf::binary_operator::LESS, bool_type, stream, mr);
    auto above = cudf::binary_operation(
      finite_ticks->view(), upper, cudf::binary_operator::GREATER, bool_type, stream, mr);
    auto overflow = cudf::binary_operation(
      below->view(), above->view(), cudf::binary_operator::LOGICAL_OR, bool_type, stream, mr);
    auto any_overflow   = cudf::reduce(overflow->view(),
                                     *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                     bool_type,
                                     stream,
                                     mr);
    auto const& invalid = static_cast<cudf::numeric_scalar<bool> const&>(*any_overflow);
    if (invalid.is_valid(stream) && invalid.value(stream)) {
      throw invalid_input_exception("Could not convert Timestamp({}) to Timestamp(US)",
                                    source_type == cudf::type_id::TIMESTAMP_SECONDS ? "S" : "MS");
    }
  }
  cudf::numeric_scalar<int64_t> factor(nanos ? scale / 2 : scale, true, stream, mr);
  auto micros =
    cudf::binary_operation(finite_ticks->view(),
                           factor,
                           nanos ? cudf::binary_operator::DIV : cudf::binary_operator::MUL,
                           int_type,
                           stream,
                           mr);
  if (nanos) {
    // Round to the nearest microsecond, with ties away from the epoch. Scale
    // before adding the half-unit so the full finite int64 domain stays safe.
    auto negative = cudf::binary_operation(
      micros->view(), zero, cudf::binary_operator::LESS, bool_type, stream, mr);
    cudf::numeric_scalar<int64_t> plus_one(1, true, stream, mr);
    cudf::numeric_scalar<int64_t> minus_one(-1, true, stream, mr);
    auto increment = cudf::copy_if_else(minus_one, plus_one, negative->view(), stream, mr);
    auto rounded   = cudf::binary_operation(
      micros->view(), increment->view(), cudf::binary_operator::ADD, int_type, stream, mr);
    cudf::numeric_scalar<int64_t> two(2, true, stream, mr);
    micros = cudf::binary_operation(
      rounded->view(), two, cudf::binary_operator::DIV, int_type, stream, mr);
  }
  return cudf::copy_if_else(cudf::bit_cast(micros->view(), return_type),
                            cudf::bit_cast(ticks, return_type),
                            finite->view(),
                            stream,
                            mr);
}

std::unique_ptr<cudf::column> date_arithmetic_checked(cudf::transform_input const& left,
                                                      cudf::transform_input const& right,
                                                      date_arithmetic_operation operation,
                                                      cudf::size_type row_count,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  using recipe          = date_arithmetic_operation;
  auto const date_type  = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto const int32_type = cudf::data_type{cudf::type_id::INT32};
  auto const int64_type = cudf::data_type{cudf::type_id::INT64};
  auto const reversed   = operation == recipe::days_add_date;
  auto const difference = operation == recipe::date_difference;
  auto day_counts       = [&](cudf::transform_input const& input, cudf::data_type expected_type) {
    return std::visit(
      [&](auto const& view) -> cudf::transform_input {
        using view_type       = std::decay_t<decltype(view)>;
        constexpr bool scalar = std::is_same_v<view_type, cudf::scalar_column_view>;
        if (view.type() != expected_type || (!scalar && view.size() != row_count)) {
          throw internal_exception("[date_arithmetic] source type or shape mismatch");
        }
        if constexpr (scalar) {
          return cudf::scalar_column_view{cudf::bit_cast(view.as_column_view(), int32_type)};
        } else {
          return cudf::bit_cast(view, int32_type);
        }
      },
      input);
  };
  auto date_ticks = day_counts(reversed ? right : left, date_type);
  auto days_ticks = day_counts(reversed ? left : right, difference ? date_type : int32_type);
  // transform_extended skips rows with either input NULL and propagates their
  // validity. Infinity must be handled before invoking checked arithmetic.
  static std::string const offset_prefix  = R"(
#include <cuda/numeric>

__device__ cudf::errc date_offset(int32_t* out, int32_t date, int32_t days)
{
  constexpr int32_t infinity = 2147483647;
  if (date == infinity || date == -infinity) {
    *out = date;
    return cudf::errc::SUCCESS;
  }
  auto result = cuda::)";
  static std::string const offset_suffix  = R"((date, days);
  if (result.overflow || result.value == infinity || result.value == -infinity) {
    return cudf::errc::OVERFLOW;
  }
  *out = result.value;
  return cudf::errc::SUCCESS;
}
)";
  static std::string const add_udf        = offset_prefix + "add_overflow" + offset_suffix;
  static std::string const sub_udf        = offset_prefix + "sub_overflow" + offset_suffix;
  static std::string const difference_udf = R"(
__device__ void date_difference(int64_t* out, int32_t left, int32_t right)
{
  *out = static_cast<int64_t>(left) - static_cast<int64_t>(right);
}
)";
  auto const& udf =
    difference ? difference_udf : (operation == recipe::date_sub_days ? sub_udf : add_udf);
  std::array<cudf::transform_input, 2> inputs{date_ticks, days_ticks};
  std::unique_ptr<cudf::column> result;
  try {
    result = cudf::transform_extended(inputs,
                                      udf,
                                      difference ? int64_type : int32_type,
                                      cudf::udf_source_type::CUDA,
                                      std::nullopt,
                                      cudf::null_aware::NO,
                                      row_count,
                                      cudf::output_nullability::PRESERVE,
                                      stream,
                                      mr);
  } catch (cudf::evaluation_error const& error) {
    if (error.error_code() != cudf::errc::OVERFLOW) { throw; }
    throw invalid_input_exception("Date out of range");
  }
  if (difference) { return result; }
  // Publish the INT32 day buffer as DATE without copying or converting values.
  auto const size       = result->size();
  auto const null_count = result->null_count();
  auto contents         = result->release();
  return std::make_unique<cudf::column>(
    date_type, size, std::move(*contents.data), std::move(*contents.null_mask), null_count);
}

}  // namespace sirius::temporal
