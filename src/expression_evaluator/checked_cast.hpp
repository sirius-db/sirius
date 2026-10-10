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

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

// standard library
#include <cstdint>
#include <memory>

namespace sirius {

/**
 * @brief Casts a numeric column to DECIMAL(@p precision, -target.scale()) the way DuckDB does.
 *
 * `cudf::cast` truncates toward zero when it drops digits, where DuckDB rounds half away from zero,
 * and wraps values that do not fit.
 *
 * - FLOAT and DOUBLE: DuckDB computes `round(double(x) * 10^scale)` in FP64 with the same
 *   power-of-ten constants used here, so each step is the same correctly rounded IEEE operation on
 *   both sides. For FLOAT input the rounded value is narrowed back to FLOAT before it is stored, as
 *   DuckDB does.
 * - DECIMAL with a larger scale: rounds half away from zero to the target scale.
 * - DECIMAL with a smaller or equal scale, and integers: rescales exactly.
 *
 * A non-NULL value whose result needs more than @p precision digits, NaN and +/-infinity fail the
 * cast: CAST throws and TRY_CAST yields NULL. NULLs propagate.
 *
 * @param input     FLOAT32, FLOAT64, DECIMAL32, DECIMAL64, DECIMAL128 or 8- to 64-bit integer
 *                  column.
 * @param target    DECIMAL32, DECIMAL64 or DECIMAL128 result type; its scale is the negated
 *                  number of fractional digits.
 * @param precision Total number of digits of the target DECIMAL, 1 to 38.
 * @param try_cast  Yield NULL instead of throwing for values that do not fit.
 * @param stream    CUDA stream to run on. Checking for failed rows synchronizes it.
 * @param mr        Memory resource for the result.
 * @throws sirius::invalid_input_exception if a non-NULL value does not fit and @p try_cast is
 *         false, or for unsupported input or target types.
 */
std::unique_ptr<cudf::column> cast_to_decimal(cudf::column_view const& input,
                                              cudf::data_type target,
                                              uint8_t precision,
                                              bool try_cast,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);

/**
 * @brief Casts a numeric column to an 8- to 64-bit integer type the way DuckDB does.
 *
 * `cudf::cast` truncates toward zero and wraps values that do not fit.
 *
 * - FLOAT and DOUBLE round to the nearest integer, ties to even (`std::nearbyint` in DuckDB). A
 *   value fits when it is at least the target's minimum and, once rounded, below its maximum plus
 *   one, compared in the input's type with DuckDB's bounds. DuckDB only checks the unrounded value,
 *   so a value within 0.5 below the maximum plus one overflows there; here it fails.
 * - DECIMAL rounds half away from zero to a whole number, which must fit the target.
 * - Integers must fit the target.
 *
 * A non-NULL value that does not fit, NaN and +/-infinity fail the cast: CAST throws and TRY_CAST
 * yields NULL. NULLs propagate.
 *
 * @param input    FLOAT32, FLOAT64, DECIMAL32, DECIMAL64, DECIMAL128 or 8- to 64-bit integer
 *                 column.
 * @param target   INT8 to INT64 or UINT8 to UINT64 result type.
 * @param try_cast Yield NULL instead of throwing for values that do not fit.
 * @param stream   CUDA stream to run on. Checking for failed rows synchronizes it.
 * @param mr       Memory resource for the result.
 * @throws sirius::invalid_input_exception if a non-NULL value does not fit and @p try_cast is
 *         false, or for unsupported input or target types.
 */
std::unique_ptr<cudf::column> cast_to_integer(cudf::column_view const& input,
                                              cudf::data_type target,
                                              bool try_cast,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);

}  // namespace sirius
