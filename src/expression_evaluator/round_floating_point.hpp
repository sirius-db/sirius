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

// rmm
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

// standard library
#include <cstdint>
#include <memory>

namespace sirius {

/**
 * @brief Rounds a floating point column to @p precision decimal places.
 *
 * Matches DuckDB's `round(x, precision)`. Each value is rounded in double precision as
 * `round(x * 10^precision) / 10^precision`, or as `round(x / 10^-precision) * 10^-precision` for a
 * negative @p precision, with ties rounded away from zero. For a @p precision of 1 to 22, a product
 * `x * 10^precision` that lands on a midpoint only through its own rounding is rounded to the side
 * the exact product lies on, so 2.675, stored just below the midpoint, rounds to 2.67 at 2 places.
 * A non-finite result yields the input value; a negative @p precision whose `10^-precision`
 * overflows rounds finite values to 0. NULLs propagate. `round(x)` is `round(x, 0)`.
 *
 * @param input     FLOAT32 or FLOAT64 column.
 * @param precision Number of decimal places. Negative values round to the left of the decimal
 *                  point.
 * @param stream    CUDA stream to run on.
 * @param mr        Memory resource for the result.
 * @return A column of the input's type.
 * @throws sirius::invalid_input_exception if @p input is not FLOAT32 or FLOAT64.
 */
std::unique_ptr<cudf::column> round_floating_point(cudf::column_view const& input,
                                                   int32_t precision,
                                                   ::cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr);

}  // namespace sirius
