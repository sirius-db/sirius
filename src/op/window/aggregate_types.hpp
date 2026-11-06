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

#pragma once

#include "expression/aggregate_id.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

namespace sirius::op {

/// 10^@p digits as a signed 128 bit integer.
constexpr __int128_t int128_pow10(int digits)
{
  __int128_t value = 1;
  for (int i = 0; i < digits; ++i) {
    value *= 10;
  }
  return value;
}

/**
 * @brief Return the type a window `sum` over an argument of type @p input accumulates in and
 * returns.
 *
 * Signed integers give DECIMAL128 with scale 0, the exact carrier of DuckDB's HUGEINT. Decimals
 * give DECIMAL128 at the input scale, and floating point gives FLOAT64.
 *
 * @throws std::invalid_argument for any other type.
 */
cudf::data_type window_sum_type(cudf::data_type input);

/**
 * @brief Return the type of the result column of a window aggregate over an argument of type
 * @p input.
 *
 * `sum` gives `window_sum_type(input)`, `avg` FLOAT64, `count` and `count_star` INT64, and `min`
 * and `max` the argument type.
 *
 * @throws std::invalid_argument for any other function, or a `sum` over an unsupported type.
 */
cudf::data_type window_result_type(aggregate_id function, cudf::data_type input);

/**
 * @brief Check that every value of a DECIMAL128 sum column is below 10^38 in magnitude.
 *
 * @throws std::overflow_error if a value reaches the DECIMAL(38) bound.
 */
void check_decimal_sum_fits(cudf::column_view sums,
                            ::cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr);

}  // namespace sirius::op
