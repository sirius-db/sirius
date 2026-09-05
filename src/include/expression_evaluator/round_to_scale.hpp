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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>

namespace sirius {

/// Rounds a FLOAT32/FLOAT64 column to `decimal_places` fractional digits the way DuckDB's
/// floating -> DECIMAL cast does: `round(x * 10^decimal_places) / 10^decimal_places` with
/// `round` half away from zero, on the whole value in one step.
///
/// This is deliberately not `cudf::round`: cuDF splits the value into integer and fractional
/// parts before scaling (`modf`), so 2.675 (stored as 2.67499999999999982) scales its fraction
/// to 67.49999999999999 and rounds down to 2.67, while DuckDB scales the whole value to 267.5
/// (the nearest double) and rounds up to 2.68. A cast that must agree with DuckDB's answer to
/// the last digit has to reproduce its arithmetic, not just its rounding mode.
///
/// Nulls are preserved; NaN and infinities pass through unchanged (they are out of range for
/// any decimal and the cast that follows is what reports them).
/// @throws sirius::invalid_input_exception when `input` is not a floating column.
std::unique_ptr<cudf::column> round_to_scale_like_duckdb(cudf::column_view const& input,
                                                         std::int32_t decimal_places,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);

}  // namespace sirius
