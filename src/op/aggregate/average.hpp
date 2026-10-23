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

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace sirius::op {

/**
 * @brief AVG as FLOAT64 from its merged SUM column @p sum and COUNT column @p count.
 *
 * A DECIMAL sum is divided like DuckDB divides it: the unscaled integer sum over
 * `count * P`, where P is 10^scale rounded to DOUBLE the way DuckDB rounds it (exact up to scale
 * 22). Rounding the scaled sum, `count * P` or the sum itself to DOUBLE first would make equal
 * averages over different counts differ in the last bit. Instead `count * P` is kept exact as two
 * DOUBLEs, the sum is split into two DOUBLEs (exact below 2^106) and the quotient is formed with
 * about 106 bits and rounded once. The result is therefore the correctly rounded value of
 * `sum / (count * P)` except within about 2^-104 relative of a rounding tie, for any count below
 * 2^53. DuckDB divides in `long double` (113-bit on aarch64, 64-bit on x86-64), or in DOUBLE for
 * DECIMAL inputs of width 9 or less, so the two can still differ in the last bit where DuckDB's
 * own quotient is not correctly rounded, e.g. such a narrow input whose unscaled sum exceeds
 * 2^53. Any other sum is cast to FLOAT64 and divided. A row is NULL when @p sum or @p count is.
 */
std::unique_ptr<cudf::column> average_as_double(cudf::column_view const& sum,
                                                cudf::column_view const& count,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace sirius::op
