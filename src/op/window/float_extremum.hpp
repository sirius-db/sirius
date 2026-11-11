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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <memory>

namespace sirius::op {

/**
 * @brief NaN handling for a window `min` or `max` over floating point.
 *
 * DuckDB orders NaN above every other value, but cuDF's `MIN` and `MAX` do not, see
 * https://github.com/NVIDIA/cudf/issues/20246. An evaluator takes the extremum of a floating point
 * argument like this:
 * 1. `make_nan_free_argument` checks the argument for NaN, once per argument column. Without NaN
 *    it returns nothing, and the extremum runs over the argument as it is.
 * 2. With NaN, the extremum runs over the NaN free copy, and the maximum of the NaN flags runs over
 *    the same rows (per group, or running).
 * 3. `restore_nan` puts NaN back into the result from that maximum.
 */

/**
 * @brief A floating point argument of a `min` or `max`, prepared for NaN.
 *
 * Both members are nullptr when the argument holds no NaN.
 */
struct nan_free_argument {
  /// 1 for a NaN row and 0 for every other row, including NULL rows, as INT8.
  std::unique_ptr<cudf::column> flags;
  /// The argument with NaN replaced by NULL.
  std::unique_ptr<cudf::column> finite;
};

/**
 * @brief Check the floating point column @p values for NaN and, when it holds any, return its NaN
 * flags and its NaN free copy.
 *
 * Together with `restore_nan` it works around https://github.com/NVIDIA/cudf/issues/20246. Once
 * cuDF can order NaN above every other value in `MIN` and `MAX`, both go away.
 */
nan_free_argument make_nan_free_argument(cudf::column_view values,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Put NaN back into a `min` or `max` taken over the NaN free copy of its argument.
 *
 * A `max` is NaN where any of its rows is NaN. A `min` is NaN where any of its rows is NaN and
 * no other non-NULL value is, which is where @p result is NULL.
 *
 * @param result The `min` or `max` over the NaN free copy.
 * @param any_nan The maximum of the NaN flags over the same rows as each value of @p result.
 * @param function `min` or `max`.
 */
std::unique_ptr<cudf::column> restore_nan(std::unique_ptr<cudf::column> result,
                                          cudf::column_view any_nan,
                                          aggregate_id function,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace sirius::op
