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
 * @brief Casts a numeric column to an integer or DECIMAL type the way DuckDB does.
 *
 * `cudf::cast` truncates toward zero and wraps values that do not fit, where DuckDB rounds and
 * range-checks:
 *
 * - FLOAT and DOUBLE to DECIMAL compute `round(double(x) * 10^scale)` in FP64 with DuckDB's
 *   power-of-ten constants, so each step is the same correctly rounded IEEE operation as on the
 *   CPU. The rounded DOUBLE converts exactly, also for FLOAT input.
 * - FLOAT and DOUBLE to an integer round to nearest, ties to even (`std::nearbyint`). As in DuckDB
 *   v2, the rounded value must be at least the target's minimum and below its maximum plus one,
 *   compared in the input's type: 127.6 fails for TINYINT, -128.4 casts to -128 and -0.4 to
 *   UTINYINT 0.
 * - DECIMAL to fewer fractional digits, including to an integer, rounds half away from zero. Other
 *   integer and DECIMAL conversions are exact.
 *
 * A non-NULL value that does not fit, NaN and +/-infinity fail the cast. NULLs propagate.
 *
 * @param input     8- to 64-bit integer, FLOAT32, FLOAT64 or DECIMAL column.
 * @param target    8- to 64-bit integer or DECIMAL result type.
 * @param precision Total number of digits of a DECIMAL target, 1 to 38. Ignored for integers.
 * @param try_cast  Yield NULL instead of throwing for values that do not fit.
 * @param stream    CUDA stream to run on. Checking for failed rows synchronizes it.
 * @param mr        Memory resource for the result.
 * @throws sirius::invalid_input_exception if a non-NULL value does not fit and @p try_cast is
 *         false.
 */
std::unique_ptr<cudf::column> checked_cast(cudf::column_view const& input,
                                           cudf::data_type target,
                                           uint8_t precision,
                                           bool try_cast,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

}  // namespace sirius
