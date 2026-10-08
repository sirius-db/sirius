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
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <helper/timestamp_semantics.hpp>
#include <sirius/exception.hpp>

#include <cstdint>
#include <limits>

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
  cudf::numeric_scalar<int64_t> factor(scale, true, stream, mr);
  auto micros =
    cudf::binary_operation(finite_ticks->view(),
                           factor,
                           nanos ? cudf::binary_operator::DIV : cudf::binary_operator::MUL,
                           int_type,
                           stream,
                           mr);
  return cudf::copy_if_else(cudf::bit_cast(micros->view(), return_type),
                            cudf::bit_cast(ticks, return_type),
                            finite->view(),
                            stream,
                            mr);
}

}  // namespace sirius::temporal
