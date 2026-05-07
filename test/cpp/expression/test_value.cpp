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

// Tests for sirius::value — the typed-payload variant + DuckDB boundary mappers.
//
// Round-trips every supported sirius::type_id through from_duckdb / to_duckdb,
// verifies typed-NULL fidelity (D-05), DECIMAL across all three width classes
// (D-02 / D-06), HUGEINT narrowing (D-03), and the unsupported-type throw
// path (D-07).

#include "catch.hpp"
#include "expression/value.hpp"
#include "helper/logical_type.hpp"
#include "sirius/exception.hpp"

#include <duckdb/common/types/value.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

using sirius::date_value;
using sirius::decimal128;
using sirius::decimal32;
using sirius::decimal64;
using sirius::logical_type;
using sirius::not_implemented_exception;
using sirius::null_value;
using sirius::timestamp_ms_value;
using sirius::timestamp_ns_value;
using sirius::timestamp_sec_value;
using sirius::timestamp_us_value;
using sirius::type_id;
using sirius::value;

// ============================================================================
// Compile-time invariants
// ============================================================================

static_assert(std::is_default_constructible_v<value>,
              "sirius::value must default-construct (NULL is the natural default).");
static_assert(std::variant_size_v<value> == 21,
              "sirius::value has exactly 21 alternatives (D-01 — locked ABI).");

// ============================================================================
// Round-trip every supported type_id (CONTEXT.md test plan item 1)
// ============================================================================

TEST_CASE("ast_value - BOOLEAN round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::BOOLEAN);
  auto const dv = duckdb::Value::BOOLEAN(true);
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<bool>(sv));
  REQUIRE(std::get<bool>(sv) == true);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - TINYINT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::TINYINT);
  auto const dv = duckdb::Value::TINYINT(static_cast<int8_t>(-7));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<int8_t>(sv));
  REQUIRE(std::get<int8_t>(sv) == -7);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - SMALLINT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::SMALLINT);
  auto const dv = duckdb::Value::SMALLINT(static_cast<int16_t>(-12345));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<int16_t>(sv));
  REQUIRE(std::get<int16_t>(sv) == -12345);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - INTEGER round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::INTEGER);
  auto const dv = duckdb::Value::INTEGER(42);
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<int32_t>(sv));
  REQUIRE(std::get<int32_t>(sv) == 42);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - BIGINT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::BIGINT);
  auto const dv = duckdb::Value::BIGINT(static_cast<int64_t>(9'000'000'000));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<int64_t>(sv));
  REQUIRE(std::get<int64_t>(sv) == 9'000'000'000);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - HUGEINT narrows to int64_t (D-03) and round-trips", "[ast_value]")
{
  // Use a value that fits cleanly in int64 so the narrowing is lossless.
  // Out-of-int64-range HUGEINT values are a Phase 5 concern.
  auto const t  = logical_type::make(type_id::HUGEINT);
  auto const dv = duckdb::Value::HUGEINT(duckdb::hugeint_t{0, 12345});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<int64_t>(sv));
  REQUIRE(std::get<int64_t>(sv) == 12345);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - UTINYINT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::UTINYINT);
  auto const dv = duckdb::Value::UTINYINT(static_cast<uint8_t>(255));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<uint8_t>(sv));
  REQUIRE(std::get<uint8_t>(sv) == 255);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - USMALLINT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::USMALLINT);
  auto const dv = duckdb::Value::USMALLINT(static_cast<uint16_t>(60000));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<uint16_t>(sv));
  REQUIRE(std::get<uint16_t>(sv) == 60000);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - UINTEGER round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::UINTEGER);
  auto const dv = duckdb::Value::UINTEGER(static_cast<uint32_t>(4'000'000'000));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<uint32_t>(sv));
  REQUIRE(std::get<uint32_t>(sv) == 4'000'000'000);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - UBIGINT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::UBIGINT);
  auto const dv = duckdb::Value::UBIGINT(static_cast<uint64_t>(18'000'000'000ULL));
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<uint64_t>(sv));
  REQUIRE(std::get<uint64_t>(sv) == 18'000'000'000ULL);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - UHUGEINT narrows to uint64_t (D-03) and round-trips", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::UHUGEINT);
  auto const dv = duckdb::Value::UHUGEINT(duckdb::uhugeint_t{0, 67890});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<uint64_t>(sv));
  REQUIRE(std::get<uint64_t>(sv) == 67890);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - FLOAT round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::FLOAT);
  auto const dv = duckdb::Value::FLOAT(3.14f);
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<float>(sv));
  REQUIRE(std::get<float>(sv) == 3.14f);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - DOUBLE round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::DOUBLE);
  auto const dv = duckdb::Value::DOUBLE(1.5);
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<double>(sv));
  REQUIRE(std::get<double>(sv) == 1.5);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - DATE round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::DATE);
  auto const dv = duckdb::Value::DATE(duckdb::date_t{19000});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<date_value>(sv));
  REQUIRE(std::get<date_value>(sv).days == 19000);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - TIMESTAMP_SEC round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::TIMESTAMP_SEC);
  auto const dv = duckdb::Value::TIMESTAMPSEC(duckdb::timestamp_sec_t{1700000000});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<timestamp_sec_value>(sv));
  REQUIRE(std::get<timestamp_sec_value>(sv).value == 1700000000);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - TIMESTAMP_MS round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::TIMESTAMP_MS);
  auto const dv = duckdb::Value::TIMESTAMPMS(duckdb::timestamp_ms_t{1700000000123});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<timestamp_ms_value>(sv));
  REQUIRE(std::get<timestamp_ms_value>(sv).value == 1700000000123);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - TIMESTAMP (microsecond) round-trips through from_duckdb / to_duckdb",
          "[ast_value]")
{
  auto const t  = logical_type::make(type_id::TIMESTAMP);
  auto const dv = duckdb::Value::TIMESTAMP(duckdb::timestamp_t{1700000000123456});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<timestamp_us_value>(sv));
  REQUIRE(std::get<timestamp_us_value>(sv).value == 1700000000123456);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - TIMESTAMP_NS round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::TIMESTAMP_NS);
  auto const dv = duckdb::Value::TIMESTAMPNS(duckdb::timestamp_ns_t{1700000000123456789});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<timestamp_ns_value>(sv));
  REQUIRE(std::get<timestamp_ns_value>(sv).value == 1700000000123456789);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - VARCHAR round-trips through from_duckdb / to_duckdb", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::VARCHAR);
  auto const dv = duckdb::Value("hello");
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<std::string>(sv));
  REQUIRE(std::get<std::string>(sv) == "hello");
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - DECIMAL(4,2) maps to decimal32 (precision <= 4 width class)", "[ast_value]")
{
  // precision <= 4 → DuckDB int16 storage; sirius alt is decimal32.
  auto const t  = logical_type::make_decimal(4, 2);
  auto const dv = duckdb::Value::DECIMAL(static_cast<int16_t>(1234), uint8_t{4}, uint8_t{2});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<decimal32>(sv));
  REQUIRE(std::get<decimal32>(sv).value == 1234);
  REQUIRE(std::get<decimal32>(sv).scale == 2);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - DECIMAL(18,4) maps to decimal64 (precision 10..18 width class)",
          "[ast_value]")
{
  auto const t  = logical_type::make_decimal(18, 4);
  auto const dv = duckdb::Value::DECIMAL(int64_t{1234567890}, uint8_t{18}, uint8_t{4});
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<decimal64>(sv));
  REQUIRE(std::get<decimal64>(sv).value == 1234567890);
  REQUIRE(std::get<decimal64>(sv).scale == 4);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

TEST_CASE("ast_value - DECIMAL(38,10) maps to decimal128 (precision 19..38 width class)",
          "[ast_value]")
{
  auto const t           = logical_type::make_decimal(38, 10);
  auto const hi_in       = duckdb::hugeint_t{0, 1'000'000'000'000ULL};
  auto const dv          = duckdb::Value::DECIMAL(hi_in, uint8_t{38}, uint8_t{10});
  auto const sv          = sirius::from_duckdb(dv, t);
  auto const expected128 = static_cast<__int128_t>(1'000'000'000'000LL);
  REQUIRE(std::holds_alternative<decimal128>(sv));
  REQUIRE(std::get<decimal128>(sv).value == expected128);
  REQUIRE(std::get<decimal128>(sv).scale == 10);
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back == dv);
}

// ============================================================================
// Typed NULL fidelity (D-05; CONTEXT.md test plan item 2)
// ============================================================================

TEST_CASE("ast_value - typed NULL round-trips for INTEGER", "[ast_value]")
{
  auto const t = logical_type::make(type_id::INTEGER);
  // duckdb::Value(LogicalType) constructs a typed NULL.
  auto const dv = duckdb::Value(duckdb::LogicalType::INTEGER);
  REQUIRE(dv.IsNull());
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<null_value>(sv));
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back.IsNull());
  REQUIRE(back.type() == duckdb::LogicalType::INTEGER);
}

TEST_CASE("ast_value - typed NULL round-trips for DECIMAL(10,2)", "[ast_value]")
{
  auto const t  = logical_type::make_decimal(10, 2);
  auto const dv = duckdb::Value(duckdb::LogicalType::DECIMAL(10, 2));
  REQUIRE(dv.IsNull());
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<null_value>(sv));
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back.IsNull());
  REQUIRE(duckdb::DecimalType::GetWidth(back.type()) == 10);
  REQUIRE(duckdb::DecimalType::GetScale(back.type()) == 2);
}

TEST_CASE("ast_value - typed NULL round-trips for VARCHAR", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::VARCHAR);
  auto const dv = duckdb::Value(duckdb::LogicalType::VARCHAR);
  REQUIRE(dv.IsNull());
  auto const sv = sirius::from_duckdb(dv, t);
  REQUIRE(std::holds_alternative<null_value>(sv));
  auto const back = sirius::to_duckdb(sv, t);
  REQUIRE(back.IsNull());
  REQUIRE(back.type() == duckdb::LogicalType::VARCHAR);
}

// ============================================================================
// Unsupported types throw (D-07; CONTEXT.md test plan item 4)
// ============================================================================

TEST_CASE("ast_value - from_duckdb throws on unsupported logical_type", "[ast_value]")
{
  // INTERVAL has no sirius::type_id — type_id::INVALID is used as the carrier
  // for "unsupported by Sirius". The supplied logical_type drives the throw,
  // independent of the duckdb::Value's own type.
  auto const t  = logical_type::make(type_id::INVALID);
  auto const dv = duckdb::Value::INTERVAL(0, 0, 1);  // 1-microsecond interval — non-NULL
  REQUIRE_THROWS_AS(sirius::from_duckdb(dv, t), not_implemented_exception);
}

TEST_CASE("ast_value - to_duckdb throws on unsupported logical_type", "[ast_value]")
{
  auto const t  = logical_type::make(type_id::INVALID);
  auto const sv = value{int32_t{0}};  // any non-null payload — the throw is type-driven
  REQUIRE_THROWS_AS(sirius::to_duckdb(sv, t), not_implemented_exception);
}
