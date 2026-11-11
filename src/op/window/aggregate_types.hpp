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

#include <vector>

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
 * @brief One aggregate of a window: the function and the column it reads.
 */
struct window_aggregate {
  /// One of `sum` (`sum_no_overflow` is treated as `sum`), `avg`, `count`, `count_star`, `min` and
  /// `max`. Each evaluator documents the subset it accepts and rejects any other id.
  sirius::aggregate_id function;
  /// Index of the argument column in the input table. `count_star` counts rows including NULLs, so
  /// any column serves as its argument.
  cudf::size_type argument_column;
};

/**
 * @brief The ORDER BY of a window, in cuDF terms.
 *
 * `directions` and `nulls` have one entry for each of `columns`.
 */
struct window_order {
  /// Indices of the order key columns in the input table.
  std::vector<cudf::size_type> columns;
  /// Direction of each order key.
  std::vector<cudf::order> directions;
  /// Where NULLs sort for each order key, in the cuDF sense: applied in the ascending frame and
  /// reversed for a descending direction. `to_cudf_null_order` converts SQL `NULLS FIRST` and
  /// `NULLS LAST`.
  std::vector<cudf::null_order> nulls;
};

/**
 * @brief Whether @p type is an 8 to 32 bit integer.
 *
 * Such a value has a magnitude of at most 2^31 and a column holds fewer than 2^31 rows, so any sum
 * over one column, including every running sum, fits INT64. cuDF sums these types into INT64.
 */
bool is_narrow_integer(cudf::data_type type);

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
 * @brief @p function as the window evaluators compute it: `sum_no_overflow` becomes `sum`.
 *
 * @throws std::invalid_argument for any id other than `sum`, `sum_no_overflow`, `avg`, `count`,
 * `count_star`, `min` and `max`.
 */
aggregate_id normalize_window_aggregate(aggregate_id function);

/**
 * @brief Check that every value of a DECIMAL128 sum column is below 10^38 in magnitude.
 *
 * @throws std::overflow_error if a value reaches the DECIMAL(38) bound.
 */
void check_decimal_sum_fits(cudf::column_view sums,
                            ::cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr);

}  // namespace sirius::op
