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

// sirius
#include "helper/logical_type.hpp"  // sirius::logical_type

// standard library
#include <cstdint>
#include <string>
#include <variant>

namespace duckdb {
class Value;
}  // namespace duckdb

namespace sirius {

/**
 * @brief Carrier for a typed SQL NULL literal.
 *
 * Empty by design — the SQL type of a NULL is recovered from the
 * sibling sirius::logical_type that travels with the value (see
 * sirius::ast::constant::return_type).
 */
struct null_value {};

/**
 * @brief Carrier for a SQL DATE literal (32-bit days from epoch).
 *
 * Wrapped in a named struct so std::visit dispatches DATE separately
 * from int32_t (INTEGER). Mirrors duckdb::date_t.days.
 */
struct date_value {
  int32_t days{};
};

/**
 * @brief Carrier for a SQL TIMESTAMP literal in seconds resolution.
 *
 * 64-bit signed seconds from epoch. Mirrors duckdb::timestamp_sec_t.value.
 */
struct timestamp_sec_value {
  int64_t value{};
};

/**
 * @brief Carrier for a SQL TIMESTAMP literal in milliseconds resolution.
 *
 * 64-bit signed milliseconds from epoch. Mirrors duckdb::timestamp_ms_t.value.
 */
struct timestamp_ms_value {
  int64_t value{};
};

/**
 * @brief Carrier for a SQL TIMESTAMP literal in microseconds resolution
 *        (the SQL default TIMESTAMP precision).
 *
 * 64-bit signed microseconds from epoch. Mirrors duckdb::timestamp_tz_t.value
 * (the accessor used by constant.cpp for microsecond TIMESTAMP).
 */
struct timestamp_us_value {
  int64_t value{};
};

/**
 * @brief Carrier for a SQL TIMESTAMP literal in nanoseconds resolution.
 *
 * 64-bit signed nanoseconds from epoch. Mirrors duckdb::timestamp_ns_t.value.
 */
struct timestamp_ns_value {
  int64_t value{};
};

/**
 * @brief Carrier for a fixed-point DECIMAL literal whose unscaled value
 *        fits in int32_t (precision <= 9).
 *
 * Precision lives in the sibling sirius::logical_type. cuDF expects a
 * *negative* scale (numeric::scale_type{-scale}); the negation happens at
 * the cuDF boundary, NOT in this struct.
 */
struct decimal32 {
  int32_t value{};
  uint8_t scale{};
};

/**
 * @brief Carrier for a fixed-point DECIMAL literal whose unscaled value
 *        fits in int64_t (precision 10..18).
 */
struct decimal64 {
  int64_t value{};
  uint8_t scale{};
};

/**
 * @brief Carrier for a fixed-point DECIMAL literal whose unscaled value
 *        requires __int128_t (precision 19..38).
 */
struct decimal128 {
  __int128_t value{};
  uint8_t scale{};
};

/**
 * @brief Sum type over every supported SQL literal kind.
 *
 * The alternative order is part of the public ABI — std::variant indexes by
 * position, so std::visit dispatch depends on this ordering. NULL is
 * alternative 0 so value{} default-constructs to NULL.
 *
 * HUGEINT and UHUGEINT do not have distinct alternatives; they reuse int64_t
 * and uint64_t respectively (narrowing — matches the executor's current cuDF
 * mapping).
 */
using value = std::variant<null_value,
                           bool,
                           int8_t,
                           int16_t,
                           int32_t,
                           int64_t,  // BIGINT and HUGEINT (narrowed)
                           uint8_t,
                           uint16_t,
                           uint32_t,
                           uint64_t,  // UBIGINT and UHUGEINT (narrowed)
                           float,
                           double,
                           date_value,
                           timestamp_sec_value,
                           timestamp_ms_value,
                           timestamp_us_value,
                           timestamp_ns_value,
                           decimal32,
                           decimal64,
                           decimal128,
                           std::string>;

/**
 * @brief Convert a duckdb::Value to a sirius::value.
 *
 * The supplied logical_type drives DECIMAL precision recovery and typed-NULL
 * fidelity. For HUGEINT / UHUGEINT, the value is narrowed to int64_t /
 * uint64_t — the same narrowing the executor performs today via
 * duckdb::GetCudfType.
 *
 * @throws sirius::not_implemented_exception if the type is unsupported.
 */
value from_duckdb(const duckdb::Value& v, const logical_type& type);

/**
 * @brief Convert a sirius::value back to a duckdb::Value.
 *
 * The supplied logical_type drives DECIMAL precision width selection
 * (Value::DECIMAL(int16_t,...) vs int32 vs int64 vs hugeint_t) and
 * recovers the SQL type of typed NULLs.
 *
 * @throws sirius::not_implemented_exception if the type is unsupported.
 */
duckdb::Value to_duckdb(const value& v, const logical_type& type);

}  // namespace sirius
