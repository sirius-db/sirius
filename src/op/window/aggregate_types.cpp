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

#include "op/window/aggregate_types.hpp"

#include "op/aggregate/wide_integer_sum.hpp"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <stdexcept>

namespace sirius::op {

cudf::data_type window_sum_type(cudf::data_type input)
{
  using cudf::type_id;
  switch (input.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64: return wide_integer_sum_type();
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: return cudf::data_type{type_id::DECIMAL128, input.scale()};
    case type_id::FLOAT32:
    case type_id::FLOAT64: return cudf::data_type{type_id::FLOAT64};
    default: throw std::invalid_argument("Unsupported window sum argument type");
  }
}

cudf::data_type window_result_type(aggregate_id function, cudf::data_type input)
{
  switch (function) {
    case aggregate_id::sum: return window_sum_type(input);
    case aggregate_id::avg: return cudf::data_type{cudf::type_id::FLOAT64};
    case aggregate_id::count:
    case aggregate_id::count_star: return cudf::data_type{cudf::type_id::INT64};
    case aggregate_id::min:
    case aggregate_id::max: return input;
    default: throw std::invalid_argument("Unsupported window function");
  }
}

void check_decimal_sum_fits(cudf::column_view sums,
                            ::cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  constexpr auto limit = int128_pow10(38);
  auto bounds          = cudf::minmax(sums, stream, mr);
  for (auto const* bound : {bounds.first.get(), bounds.second.get()}) {
    if (bound->is_valid(stream)) {
      auto const value =
        static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(*bound).value(stream);
      if (value <= -limit || value >= limit) {
        throw std::overflow_error("Window sum exceeds DECIMAL(38)");
      }
    }
  }
}

}  // namespace sirius::op
