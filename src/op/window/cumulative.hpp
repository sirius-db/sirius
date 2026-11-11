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
#include "op/window/ranking.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief An aggregate over the frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
 */
struct cumulative_aggregate {
  /// One of `sum` (`sum_no_overflow` is treated as `sum`), `count`, `count_star`, `min` and `max`.
  /// Any other id is rejected.
  sirius::aggregate_id function;
  /// Index of the argument column in the input table. `count_star` counts rows including NULLs, so
  /// any column serves as its argument.
  cudf::size_type argument_column;
};

/**
 * @brief Evaluate running aggregates over each partition of the input.
 *
 * The rows of each partition are ordered by @p order. The result for a row aggregates the rows of
 * its partition from the first one up to and including the row itself, which is the frame `ROWS
 * BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Rows with equal order keys are not peers, so each
 * gets its own result. The sort is stable, so equal rows keep their input order.
 *
 * The input is one complete hash bucket, which can hold several `PARTITION BY` groups. It is
 * evaluated as one cuDF table, so it must fit in GPU memory and hold fewer than 2^31 rows. A
 * partition is never split across calls. Rows with equal partition column values form one
 * partition, and NULL partition values are equal to each other. The result columns are scattered
 * back to the row order of @p input, so the input columns are neither copied nor reordered and the
 * caller appends the result columns to them.
 *
 * Partition columns and order keys can be in an exact narrower carrier of their logical type.
 * Aggregate arguments must have their logical types, including decimal scales.
 *
 * Result types:
 * - `sum` over signed integers is DECIMAL128 with scale 0, the exact carrier of DuckDB's HUGEINT.
 *   Over decimals it is DECIMAL128 at the input scale and over floating point it is FLOAT64.
 * - `count` and `count_star` are INT64. `count` counts the non-NULL arguments so far and
 * `count_star` the rows so far.
 * - `min` and `max` keep the argument type. Floating point arguments are not supported.
 *
 * `sum`, `min` and `max` are NULL until the first non-NULL argument of the partition and then
 * keep the running value, so a NULL argument does not reset or hide it.
 *
 * @param input Table holding the partition columns, the order keys and the aggregate arguments.
 * @param partition_columns Indices of the partition columns in @p input. Empty for a window without
 * `PARTITION BY`.
 * @param order The window ORDER BY.
 * @param aggregates The aggregates to evaluate. Must not be empty.
 * @param stream CUDA stream used for all device work.
 * @param mr Device memory resource used to allocate the result columns and the intermediate
 * columns created by the evaluator.
 *
 * @return One column per aggregate, in the order of @p aggregates, each with `input.num_rows()`
 * rows in the row order of @p input. An empty input gives empty typed columns.
 *
 * @throws std::invalid_argument if @p aggregates or `order.columns` is empty, a column index is
 * outside @p input, or an aggregate id or argument type is unsupported.
 * @throws std::overflow_error if a `sum` over a DECIMAL128 argument reaches 10^38 in magnitude.
 */
std::unique_ptr<cudf::table> evaluate_cumulative_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  ranking_order const& order,
  std::vector<cumulative_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
