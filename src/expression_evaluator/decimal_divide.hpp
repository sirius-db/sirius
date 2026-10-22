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
 * @brief GPU sirius_decimal_divide(): the exact quotient of two columns as DECIMAL(38, scale).
 *
 * Row by row, `dividend / divisor` computed exactly and rounded half away from zero to
 * @p result_scale fractional digits (see `decimal_divide_round` in src/helper/decimal_divide.hpp,
 * which the CPU function shares). `cudf::binary_operation` DIV on fixed-point truncates, so it is
 * not used.
 *
 * A row is NULL when either input is NULL or the divisor is 0.
 *
 * @param dividend     INT32, INT64, DECIMAL32, DECIMAL64 or DECIMAL128 column. Integers have
 *                     scale 0.
 * @param divisor      Same types as @p dividend, with the same number of rows.
 * @param result_scale Fractional digits of the result, 0 to 38.
 * @param stream       CUDA stream to run on. The call synchronizes it, to read the null and
 *                     overflow counts.
 * @param mr           Memory resource for the result.
 * @return A DECIMAL128 column with scale -@p result_scale.
 * @throws sirius::invalid_input_exception if a non-NULL quotient needs more than 38 digits, or for
 *         unsupported input types.
 */
std::unique_ptr<cudf::column> decimal_divide(cudf::column_view const& dividend,
                                             cudf::column_view const& divisor,
                                             int32_t result_scale,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace sirius
