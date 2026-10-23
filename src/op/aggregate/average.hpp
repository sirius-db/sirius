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
 * `count * 10^scale`. Converting the scaled sum to FLOAT64 first would round twice, so equal
 * averages over different counts could differ in the last bit. The result is exact-rounded while
 * the unscaled sum and `count * 10^scale` stay below 2^53; past that DuckDB's `long double` can
 * still differ in the last bit. Any other sum is cast to FLOAT64 and divided. A row is NULL when
 * @p sum or @p count is.
 */
std::unique_ptr<cudf::column> average_as_double(cudf::column_view const& sum,
                                                cudf::column_view const& count,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace sirius::op
