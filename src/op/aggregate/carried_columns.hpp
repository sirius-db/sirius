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

/**
 * @file carried_columns.hpp
 * @brief The carried columns of a grouped aggregate, shared by
 * `gpu_aggregate_impl::local_grouped_aggregate` and `gpu_merge_impl::merge_grouped_aggregate`.
 *
 * A carried column is the input of a FIRST: each group's output row copies it from one input row of
 * that group, and every carried column of a group comes from the same input row. Each kernel
 * decides which of its columns are carried and where they land in its output.
 */

#include <cudf/column/column.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::op::detail {

/**
 * @brief Groups a table whose only aggregates are carried columns
 *
 * Runs `cudf::distinct` on the leading @p num_keys columns instead of a groupby, since there is
 * nothing to aggregate. NULL keys compare equal, and all NaN values compare equal.
 *
 * @param input Key columns followed by the carried columns
 * @param num_keys Number of leading key columns; must be positive, since `cudf::distinct` over no
 * key returns no rows
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for the returned table
 * @return One row per distinct key, in the column layout of @p input
 */
[[nodiscard]] std::unique_ptr<cudf::table> aggregate_carried_columns_only(
  cudf::table_view input,
  cudf::size_type num_keys,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Owned row positions and the identity of their appended aggregation request
 *
 * Keep this value alive until the groupby has consumed the request's borrowed values. Sorting
 * request values preserves the original row positions stored in the column.
 */
struct carried_row_selector {
  std::unique_ptr<cudf::column> row_positions;
  std::size_t request_idx;
};

/**
 * @brief Appends the request that picks each group's carried row to @p requests
 *
 * The request is a MIN over the row positions `0 .. num_rows - 1`, so its result is the lowest
 * input row of each group. It stays an input position when the caller sorts every request's values
 * together with the keys before the groupby. Subsequent requests do not change its identity.
 *
 * @param requests The groupby's requests
 * @param num_rows Number of rows in the groupby input
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for the returned column
 * @return The owned row positions and the appended request index
 */
[[nodiscard]] carried_row_selector append_carried_row_selector(
  std::vector<cudf::groupby::aggregation_request>& requests,
  cudf::size_type num_rows,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Gathers the carried columns of each group's row
 *
 * @param carried The carried columns, in the row order of the groupby input
 * @param row_positions Result of the request identified by `carried_row_selector::request_idx`
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for the returned columns
 * @return One column per column of @p carried, with one row per group in the groupby's group order
 */
[[nodiscard]] std::vector<std::unique_ptr<cudf::column>> gather_carried_columns(
  cudf::table_view carried,
  cudf::column_view row_positions,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op::detail
