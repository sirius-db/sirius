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

// duckdb
#include <duckdb/function/scalar_function.hpp>

namespace duckdb {
class DatabaseInstance;
}

namespace sirius {

/**
 * @brief The DuckDB scalar function `sirius_decimal_divide(dividend, divisor, result_scale)`.
 *
 * DuckDB divides decimals in DOUBLE. This function returns the exact quotient as
 * `DECIMAL(38, result_scale)`, rounded half away from zero, which is how StarRocks computes `/`
 * and AVG on DECIMALV3 (with the result scale its FE chose):
 *
 * - `dividend` and `divisor` are DECIMAL or integer values. The bind casts integers to INTEGER,
 *   BIGINT or, for wider ones, DECIMAL(p, 0), and a DECIMAL of precision 4 or less to precision 9,
 *   so the GPU sees 32-, 64- or 128-bit operands only.
 * - `result_scale` is an integer constant from 0 to 38, read at bind time. The return type carries
 *   it; execution, on the CPU and the GPU, reads it from there and ignores the third argument.
 * - The result is NULL when either operand is NULL or the divisor is 0.
 * - A quotient that needs more than 38 digits raises an out-of-range error.
 *
 * The GPU evaluates the same function by name (`function_id::decimal_divide`), with the same
 * arithmetic (src/helper/decimal_divide.hpp).
 */
duckdb::ScalarFunction get_decimal_divide_function();

/// Registers sirius_decimal_divide() in @p instance's system catalog. Idempotent.
void register_decimal_divide_function(duckdb::DatabaseInstance& instance);

}  // namespace sirius
