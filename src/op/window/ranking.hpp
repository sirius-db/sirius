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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief A ranking function of a window with an ORDER BY.
 */
enum class ranking_function {
  /// 1 plus the number of rows that sort strictly before the row, so peers share a rank and gaps
  /// follow ties.
  rank,
  /// 1 plus the number of distinct order key values that sort before the row. No gaps.
  dense_rank,
  /// The 1-based position of the row in its partition. Peers are numbered in their input order.
  row_number,
};

/**
 * @brief The ORDER BY of a ranking window.
 */
struct ranking_order {
  /// Indices of the order key columns in the input table. Must not be empty.
  std::vector<cudf::size_type> columns;
  /// Direction shared by all order keys.
  cudf::order direction = cudf::order::ASCENDING;
  /// Where NULLs sort, in the cuDF sense: applied in the ascending frame and reversed for a
  /// descending direction. `to_cudf_null_order` converts SQL `NULLS FIRST` and `NULLS LAST`.
  cudf::null_order nulls = cudf::null_order::BEFORE;
};

/**
 * @brief Evaluate ranking functions over each partition of the input.
 *
 * The rows are ordered by the partition columns and then by the order keys. Rows with equal
 * partition column values form one partition, and NULL partition values are equal to each other.
 * The input is one complete hash bucket, which can hold several SQL partitions. It is evaluated as
 * one cuDF table, so it must fit in GPU memory and hold fewer than 2^31 rows. A partition is never
 * split across calls. The result columns are scattered back to the row order of @p input, so the
 * input columns are neither copied nor reordered and the caller appends the result columns to
 * them.
 *
 * Partition columns and order keys refer to columns of @p input and must already have their
 * logical types, including decimal scales. Compressed physical carriers are not interpreted.
 * NaN orders above every other value and equals NaN, and `-0.0` equals `0.0`. Strings order by
 * their bytes.
 *
 * @param input Table holding the partition columns and the order keys.
 * @param partition_columns Indices of the partition columns in @p input. Must not be empty.
 * @param order The window ORDER BY.
 * @param functions The ranking functions to evaluate. Must not be empty.
 * @param stream CUDA stream used for all device work.
 * @param mr Device memory resource used to allocate the result columns and the intermediate
 * columns created by the evaluator.
 *
 * @return One INT64 column per function, in the order of @p functions, each with
 * `input.num_rows()` rows in the row order of @p input and without NULLs. An empty input gives
 * empty columns.
 *
 * @throws std::invalid_argument if @p functions, @p partition_columns or `order.columns` is empty
 * or a column index is outside @p input.
 *
 * @note Temporary allocations made inside cuDF, including the sort and the group by, use cuDF's
 * current device resource.
 */
std::unique_ptr<cudf::table> evaluate_ranking_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  ranking_order const& order,
  std::vector<ranking_function> const& functions,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
