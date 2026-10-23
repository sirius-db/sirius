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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <memory>
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
 * @brief One aggregate to evaluate over the whole partition.
 */
struct whole_partition_aggregate {
  /// One of `sum` (`sum_no_overflow` is treated as `sum`), `avg`, `count`, `count_star`, `min` and
  /// `max`. Any other id is rejected.
  sirius::aggregate_id function;
  /// Index of the argument column in the input table. `count_star` counts rows including NULLs, so
  /// any column serves as its argument.
  cudf::size_type argument_column;
};

/**
 * @brief Evaluate aggregates over the whole partition of every row.
 *
 * Every row receives the aggregate of all rows with the same partition key values. Rows with NULL
 * keys form one partition, and without partition columns the whole table is one partition. The
 * aggregates are computed with a cuDF group by, or a reduction when there are no partition
 * columns, and joined back to the input rows with a distinct hash join. No sort is performed.
 *
 * The input is one complete hash bucket, which can hold several SQL partitions. Partition columns
 * and aggregate arguments refer to columns of @p input and must already have their logical types,
 * including decimal scales. Compressed physical carriers are not interpreted. Partition columns
 * can be booleans, 8 to 64 bit integers, decimals, timestamps or strings.
 *
 * Result types:
 * - `sum` over signed integers is DECIMAL128 with scale 0, the exact carrier of DuckDB's HUGEINT.
 *   Over decimals it is DECIMAL128 at the input scale and over floating point it is FLOAT64.
 *   8 to 32 bit integers are summed in INT64. 64 bit integers and decimals are widened before
 *   the sum.
 * - `avg` is FLOAT64.
 * - `count` and `count_star` are INT64. `count` counts non-NULL arguments and is 0 for a partition
 *   of NULLs. `count_star` counts rows including NULLs.
 * - `min` and `max` keep the argument type. NaN orders above every other value.
 *
 * `sum`, `avg`, `min` and `max` are NULL for a partition whose arguments are all NULL.
 *
 * @param input Table holding the partition columns and the aggregate arguments.
 * @param partition_columns Indices of the partition columns in @p input. Empty for a window without
 * `PARTITION BY`.
 * @param aggregates The aggregates to evaluate. Must not be empty.
 * @param stream CUDA stream used for all device work.
 * @param mr Device memory resource used to allocate the result columns and the intermediate columns
 * created by the evaluator.
 *
 * @return One column per aggregate, in the order of @p aggregates, each with `input.num_rows()`
 * rows in the row order of @p input. The input columns are neither copied nor reordered, so the
 * caller appends the result columns to them. An empty input gives empty typed columns.
 *
 * @throws std::invalid_argument if @p aggregates is empty, a column index is outside @p input, or
 * an aggregate id, partition column type or argument type is unsupported.
 * @throws std::overflow_error if a `sum` exceeds the 128 bit accumulator, or if a `sum` over a
 * DECIMAL128 argument reaches 10^38. A DECIMAL128 argument is always treated as DECIMAL(38, scale),
 * so a `HUGEINT` column passed as DECIMAL128 with scale 0 is held to that bound and not to the
 * wider `HUGEINT` range.
 *
 * @note Temporary allocations made inside cuDF, including the hash table of the distinct hash join,
 * use cuDF's current device resource.
 */
std::unique_ptr<cudf::table> evaluate_whole_partition_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<whole_partition_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
