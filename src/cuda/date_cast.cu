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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/bit.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <expression_evaluator/date_cast.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace sirius {
namespace {

constexpr auto date_infinity      = std::numeric_limits<int32_t>::max();
constexpr auto timestamp_infinity = std::numeric_limits<int64_t>::max();
constexpr int64_t micros_per_day  = 86'400'000'000;

int64_t ticks_per_day(cudf::type_id target)
{
  switch (target) {
    case cudf::type_id::TIMESTAMP_SECONDS: return 86'400;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 86'400'000;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return micros_per_day;
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return 86'400'000'000'000;
    default: throw internal_exception("Expected a sub-day timestamp cast target");
  }
}

}  // namespace

std::unique_ptr<cudf::column> cast_date_to_timestamp(cudf::column_view const& input,
                                                     cudf::data_type target,
                                                     bool try_cast,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  if (input.type().id() != cudf::type_id::TIMESTAMP_DAYS) {
    throw internal_exception("Expected a DATE cast input");
  }
  auto const scale = ticks_per_day(target.id());
  // DuckDB converts DATE through microseconds even for second/millisecond targets.
  auto const max_days     = timestamp_infinity / std::max(scale, micros_per_day);
  auto const* days        = input.data<int32_t>();
  auto const* input_mask  = input.null_mask();
  auto const offset       = input.offset();
  auto const begin        = thrust::make_counting_iterator<cudf::size_type>(0);
  auto [mask, null_count] = cudf::detail::valid_if(
    begin,
    begin + input.size(),
    [=] __device__(cudf::size_type i) {
      if (input_mask && !cudf::bit_is_set(input_mask, offset + i)) { return false; }
      auto const day = days[i];
      return day == date_infinity || day == -date_infinity || (day >= -max_days && day <= max_days);
    },
    stream,
    mr);
  if (!try_cast && null_count > input.null_count()) {
    throw invalid_input_exception("DATE value is out of range for TIMESTAMP");
  }
  auto result =
    cudf::make_timestamp_column(target, input.size(), std::move(mask), null_count, stream, mr);
  auto const* output_mask = result->view().null_mask();
  thrust::transform(rmm::exec_policy(stream, mr),
                    begin,
                    begin + input.size(),
                    result->mutable_view().data<int64_t>(),
                    [=] __device__(cudf::size_type i) -> int64_t {
                      if (!cudf::bit_is_set(output_mask, i)) { return 0; }
                      auto const day = days[i];
                      if (day == date_infinity) { return timestamp_infinity; }
                      if (day == -date_infinity) { return -timestamp_infinity; }
                      return static_cast<int64_t>(day) * scale;
                    });
  return result;
}

}  // namespace sirius
