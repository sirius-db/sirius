/*
 * Copyright 2025, Sirius Contributors.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <expression/function_id.hpp>

#include <memory>

namespace sirius::temporal {

/**
 * Sirius temporal semantics currently match DuckDB: signed epoch ticks, exactly
 * +/-MAX reserved for infinity, MIN finite, and NULL propagated. Frontends must
 * normalize to this representation before using these helpers. No frontend types
 * or frontend identity are needed by the GPU evaluator.
 */

/**
 * Returns a nullable BOOL8 mask: true for finite values, false for +/-infinity,
 * and NULL for NULL input. Accepts DATE and all timestamp precisions.
 */
[[nodiscard]] std::unique_ptr<cudf::column> finite_mask(cudf::column_view const& input,
                                                        ::cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr);

/**
 * Converts second, millisecond, or nanosecond timestamps to microseconds.
 * Preserves infinity and NULL; rounds nanoseconds to nearest with ties away from the epoch; throws
 * sirius::invalid_input_exception on finite overflow or unsupported input type.
 * The overflow check synchronizes the supplied stream to read its result.
 */
[[nodiscard]] std::unique_ptr<cudf::column> cast_to_microseconds_checked(
  cudf::column_view const& input, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr);

// Applies a bound DATE/day signature. Offsets preserve NULL and infinity and
// throw invalid_input_exception on finite overflow. Differences subtract raw day
// counts in INT64. Column inputs must have row_count rows; scalar inputs broadcast
// their single row. The caller retains input ownership; overflow checks synchronize
// the supplied stream.
[[nodiscard]] std::unique_ptr<cudf::column> date_arithmetic_checked(
  cudf::transform_input const& left,
  cudf::transform_input const& right,
  date_arithmetic_operation operation,
  cudf::size_type row_count,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::temporal
