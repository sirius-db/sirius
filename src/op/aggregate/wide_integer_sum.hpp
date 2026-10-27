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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace sirius::op {

/// The type a wide integer SUM (see is_wide_integer_sum) is accumulated in when INT64 could
/// overflow: DECIMAL128 with scale 0, an exact 128-bit integer.
inline cudf::data_type wide_integer_sum_type()
{
  return cudf::data_type{cudf::type_id::DECIMAL128, 0};
}

/**
 * @brief The exact sum of the valid values of @p input, accumulated in 128 bits in one pass.
 *
 * Returns a one-row DECIMAL128 column with scale 0 (wide_integer_sum_type()), NULL when no value
 * is valid. @p input is INT32 or INT64 (a BIGINT or HUGEINT carrier). The sum cannot overflow: a
 * column holds fewer than 2^31 rows of magnitude at most 2^63, so the total stays below 2^94. It
 * reads the input once like an INT64 reduction, without the per-batch magnitude scan or the
 * widening copy a DECIMAL128 reduction would need. Throws sirius::internal_exception for any
 * other type.
 */
std::unique_ptr<cudf::column> wide_integer_sum(cudf::column_view const& input,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

/**
 * @brief sum(|value|) over the valid values of the INT64 column @p input, accumulated in double in
 * one pass.
 *
 * 0 when no value is valid. Synchronizes @p stream once to return the total. Throws
 * sirius::internal_exception for any other type.
 */
double sum_of_magnitudes(cudf::column_view const& input,
                         ::cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr);

}  // namespace sirius::op
