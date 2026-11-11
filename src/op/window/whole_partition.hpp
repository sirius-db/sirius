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

#include "op/window/aggregate_plan.hpp"
#include "op/window/aggregate_types.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief How the evaluation of aggregates over the whole partition runs, decided from types alone.
 *
 * plan_whole_partition_window() builds it from the input schema and the requested aggregates,
 * without touching device memory, so a planner can build it once, decline the window when building
 * throws, and hand the plan to the operator. evaluate_whole_partition_window() then executes it on
 * each table. The requests all run in one cuDF group by over `group_columns`, or as reductions when
 * there are no group columns.
 *
 * Whether a 64 bit integer, DECIMAL32 or DECIMAL64 sum is widened to DECIMAL128 before summing
 * (`preparation::widen_if_needed`) is the only decision taken for each table, from the values of
 * the table.
 */
struct whole_partition_plan : window_aggregate_plan {
  std::vector<cudf::size_type> group_columns;
};

/// The input columns evaluate_whole_partition_window() needs in their logical types, each once:
/// the group columns and the aggregate arguments.
std::vector<cudf::size_type> logical_type_columns(whole_partition_plan const& plan);

/**
 * @brief Plan the evaluation of aggregates over the whole partition over tables of the schema
 * @p input_types.
 *
 * Every row receives the aggregate of all rows with the same partition key values. Rows with NULL
 * keys form one partition, and without partition columns the whole table is one partition.
 *
 * Partition columns and aggregate arguments must have their logical types, including decimal
 * scales. Compressed physical carriers are not interpreted. Partition columns can be booleans, 8 to
 * 64 bit integers, decimals, timestamps or strings.
 *
 * Result types:
 * - `sum` over signed integers is DECIMAL128 with scale 0, the exact carrier of DuckDB's HUGEINT.
 *   Over decimals it is DECIMAL128 at the input scale and over floating point it is FLOAT64.
 *   8 to 32 bit integers are summed in INT64. 64 bit integers, DECIMAL32 and DECIMAL64 are
 *   widened before the sum only when their values could overflow the column's own type.
 * - `avg` is FLOAT64.
 * - `count` and `count_star` are INT64. `count` counts non-NULL arguments and is 0 for a partition
 *   of NULLs. `count_star` counts rows including NULLs.
 * - `min` and `max` keep the argument type. NaN orders above every other value.
 *
 * `sum`, `avg`, `min` and `max` are NULL for a partition whose arguments are all NULL.
 *
 * @param input_types The column types of the tables the plan is evaluated on.
 * @param partition_columns Indices of the partition columns. Empty for a window without
 * `PARTITION BY`.
 * @param aggregates The aggregates to evaluate. Must not be empty.
 *
 * @throws std::invalid_argument if @p aggregates is empty, a column index is outside
 * @p input_types, or an aggregate id, partition column type or argument type is unsupported.
 */
whole_partition_plan plan_whole_partition_window(
  std::vector<cudf::data_type> const& input_types,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<window_aggregate> const& aggregates);

/**
 * @brief Evaluate a whole-partition window plan over one table.
 *
 * The aggregates are computed with a cuDF group by, or a reduction when there are no partition
 * columns, and joined back to the input rows with a distinct hash join. No sort is performed.
 *
 * The input is one complete hash bucket, which can hold several SQL partitions. It is evaluated as
 * one cuDF table, so it must fit in GPU memory and hold fewer than 2^31 rows. A partition is never
 * split across calls.
 *
 * @param plan A plan from plan_whole_partition_window() for the schema of @p input.
 * @param input The table to evaluate.
 * @param stream CUDA stream used for all device work.
 * @param mr Device memory resource used to allocate the result columns and the intermediate columns
 * created by the evaluator.
 *
 * @return One column per aggregate the plan was built for, in request order, each with
 * `input.num_rows()` rows in the row order of @p input. The input columns are neither copied nor
 * reordered, so the caller appends the result columns to them. An empty input gives empty typed
 * columns.
 *
 * @throws std::invalid_argument if a column the plan reads is missing from @p input or has another
 * type than the plan was built for.
 * @throws std::overflow_error if a `sum` exceeds the 128 bit accumulator, or if a `sum` over a
 * DECIMAL128 argument reaches 10^38. A DECIMAL128 argument is always treated as DECIMAL(38, scale),
 * so a `HUGEINT` column passed as DECIMAL128 with scale 0 is held to that bound and not to the
 * wider `HUGEINT` range.
 *
 * @note Temporary allocations made inside cuDF, including the hash table of the distinct hash join,
 * use cuDF's current device resource.
 */
std::unique_ptr<cudf::table> evaluate_whole_partition_window(whole_partition_plan const& plan,
                                                             cudf::table_view input,
                                                             ::cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr);

/**
 * @brief Plan and evaluate aggregates over the whole partition of @p input in one call.
 *
 * Equivalent to evaluate_whole_partition_window(plan_whole_partition_window(types of @p input,
 * ...), ...). See plan_whole_partition_window() for the arguments and the semantics.
 */
std::unique_ptr<cudf::table> evaluate_whole_partition_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<window_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
