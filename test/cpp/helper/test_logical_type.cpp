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

// Tests for sirius::logical_type and the DuckDB boundary conversions.
//
// These tests verify that sirius::logical_type correctly emulates the behaviour
// of duckdb::LogicalType for every type that Sirius supports, and that the
// round-trip through from_duckdb() / to_duckdb() is lossless.

#include "catch.hpp"
#include "helper/logical_type.hpp"
#include "helper/type_conversions.hpp"

#include <duckdb/common/optional_idx.hpp>
#include <duckdb/common/types.hpp>

#include <stdexcept>

using sirius::logical_type;
using sirius::type_id;

// ============================================================================
// Helpers — all type_id values except DECIMAL (precision/scale tested
// separately via make_decimal()). Includes INVALID, SQLNULL, LIST, and STRUCT
// so predicate edge-cases (e.g. is_fixed_width()) are exercised for them too.
// ============================================================================

static const std::vector<type_id> k_all_type_ids = {
  type_id::BOOLEAN,   type_id::TINYINT,      type_id::SMALLINT,      type_id::INTEGER,
  type_id::BIGINT,    type_id::HUGEINT,      type_id::UTINYINT,      type_id::USMALLINT,
  type_id::UINTEGER,  type_id::UBIGINT,      type_id::UHUGEINT,      type_id::FLOAT,
  type_id::DOUBLE,    type_id::DATE,         type_id::TIMESTAMP_SEC, type_id::TIMESTAMP_MS,
  type_id::TIMESTAMP, type_id::TIMESTAMP_NS, type_id::VARCHAR,       type_id::STRUCT,
  type_id::LIST,      type_id::ARRAY,        type_id::SQLNULL,       type_id::INVALID,
  // DECIMAL tested separately via make_decimal()
};

// ============================================================================
// Construction and factory
// ============================================================================

TEST_CASE("logical_type - default constructor yields SQLNULL", "[logical_type]")
{
  logical_type t;
  REQUIRE(t.id() == type_id::SQLNULL);
}

TEST_CASE("logical_type - make() preserves type_id for every non-decimal type", "[logical_type]")
{
  for (auto id : k_all_type_ids) {
    REQUIRE(logical_type::make(id).id() == id);
  }
}

TEST_CASE("logical_type - make_decimal() stores precision and scale", "[logical_type]")
{
  auto t = logical_type::make_decimal(18, 6);
  REQUIRE(t.id() == type_id::DECIMAL);
  REQUIRE(t.decimal_precision() == 18);
  REQUIRE(t.decimal_scale() == 6);
}

TEST_CASE("logical_type - make_decimal precision and scale boundary values", "[logical_type]")
{
  // Minimum
  auto t_min = logical_type::make_decimal(1, 0);
  REQUIRE(t_min.decimal_precision() == 1);
  REQUIRE(t_min.decimal_scale() == 0);

  // Max INT32 (precision <= 9)
  auto t32 = logical_type::make_decimal(9, 4);
  REQUIRE(t32.decimal_precision() == 9);
  REQUIRE(t32.decimal_scale() == 4);

  // Max INT64 (precision <= 18)
  auto t64 = logical_type::make_decimal(18, 9);
  REQUIRE(t64.decimal_precision() == 18);
  REQUIRE(t64.decimal_scale() == 9);

  // INT128 (precision > 18)
  auto t128 = logical_type::make_decimal(38, 10);
  REQUIRE(t128.decimal_precision() == 38);
  REQUIRE(t128.decimal_scale() == 10);
}

// ============================================================================
// Predicate methods
// ============================================================================

TEST_CASE("logical_type - is_decimal()", "[logical_type]")
{
  REQUIRE(logical_type::make_decimal(10, 2).is_decimal());
  for (auto id : k_all_type_ids) {
    REQUIRE_FALSE(logical_type::make(id).is_decimal());
  }
}

TEST_CASE("logical_type - is_varchar()", "[logical_type]")
{
  REQUIRE(logical_type::make(type_id::VARCHAR).is_varchar());
  for (auto id : k_all_type_ids) {
    if (id == type_id::VARCHAR) continue;
    REQUIRE_FALSE(logical_type::make(id).is_varchar());
  }
  REQUIRE_FALSE(logical_type::make_decimal(10, 2).is_varchar());
}

TEST_CASE("logical_type - is_integer()", "[logical_type]")
{
  const std::vector<type_id> integer_ids = {
    type_id::TINYINT,
    type_id::SMALLINT,
    type_id::INTEGER,
    type_id::BIGINT,
    type_id::HUGEINT,
    type_id::UTINYINT,
    type_id::USMALLINT,
    type_id::UINTEGER,
    type_id::UBIGINT,
    type_id::UHUGEINT,
  };
  for (auto id : integer_ids) {
    REQUIRE(logical_type::make(id).is_integer());
  }
  // Non-integer types
  for (auto id : {type_id::FLOAT,
                  type_id::DOUBLE,
                  type_id::BOOLEAN,
                  type_id::DATE,
                  type_id::VARCHAR,
                  type_id::DECIMAL,
                  type_id::SQLNULL}) {
    if (id == type_id::DECIMAL) {
      REQUIRE_FALSE(logical_type::make_decimal(10, 2).is_integer());
    } else {
      REQUIRE_FALSE(logical_type::make(id).is_integer());
    }
  }
}

TEST_CASE("logical_type - is_numeric()", "[logical_type]")
{
  // All integer types are numeric
  for (auto id : {type_id::TINYINT,
                  type_id::SMALLINT,
                  type_id::INTEGER,
                  type_id::BIGINT,
                  type_id::HUGEINT,
                  type_id::UTINYINT,
                  type_id::USMALLINT,
                  type_id::UINTEGER,
                  type_id::UBIGINT,
                  type_id::UHUGEINT,
                  type_id::FLOAT,
                  type_id::DOUBLE}) {
    REQUIRE(logical_type::make(id).is_numeric());
  }
  REQUIRE(logical_type::make_decimal(10, 2).is_numeric());

  // Non-numeric types
  for (auto id : {type_id::BOOLEAN,
                  type_id::DATE,
                  type_id::TIMESTAMP,
                  type_id::VARCHAR,
                  type_id::STRUCT,
                  type_id::LIST,
                  type_id::SQLNULL}) {
    REQUIRE_FALSE(logical_type::make(id).is_numeric());
  }
}

TEST_CASE("logical_type - is_temporal()", "[logical_type]")
{
  for (auto id : {type_id::DATE,
                  type_id::TIMESTAMP_SEC,
                  type_id::TIMESTAMP_MS,
                  type_id::TIMESTAMP,
                  type_id::TIMESTAMP_NS}) {
    REQUIRE(logical_type::make(id).is_temporal());
  }
  for (auto id :
       {type_id::INTEGER, type_id::FLOAT, type_id::VARCHAR, type_id::BOOLEAN, type_id::SQLNULL}) {
    REQUIRE_FALSE(logical_type::make(id).is_temporal());
  }
  REQUIRE_FALSE(logical_type::make_decimal(10, 2).is_temporal());
}

TEST_CASE("logical_type - is_fixed_width()", "[logical_type]")
{
  // Variable-length / unsupported types are NOT fixed-width
  for (auto id :
       {type_id::VARCHAR, type_id::STRUCT, type_id::LIST, type_id::SQLNULL, type_id::INVALID}) {
    REQUIRE_FALSE(logical_type::make(id).is_fixed_width());
  }
  // All remaining types are fixed-width
  for (auto id : {type_id::BOOLEAN,
                  type_id::TINYINT,
                  type_id::SMALLINT,
                  type_id::INTEGER,
                  type_id::BIGINT,
                  type_id::HUGEINT,
                  type_id::UTINYINT,
                  type_id::USMALLINT,
                  type_id::UINTEGER,
                  type_id::UBIGINT,
                  type_id::UHUGEINT,
                  type_id::FLOAT,
                  type_id::DOUBLE,
                  type_id::DATE,
                  type_id::TIMESTAMP_SEC,
                  type_id::TIMESTAMP_MS,
                  type_id::TIMESTAMP,
                  type_id::TIMESTAMP_NS}) {
    REQUIRE(logical_type::make(id).is_fixed_width());
  }
  REQUIRE(logical_type::make_decimal(10, 2).is_fixed_width());
}

// ============================================================================
// fixed_width_byte_size
// ============================================================================

TEST_CASE("logical_type - fixed_width_byte_size() returns correct sizes", "[logical_type]")
{
  REQUIRE(logical_type::make(type_id::BOOLEAN).fixed_width_byte_size() == 1);
  REQUIRE(logical_type::make(type_id::TINYINT).fixed_width_byte_size() == 1);
  REQUIRE(logical_type::make(type_id::UTINYINT).fixed_width_byte_size() == 1);

  REQUIRE(logical_type::make(type_id::SMALLINT).fixed_width_byte_size() == 2);
  REQUIRE(logical_type::make(type_id::USMALLINT).fixed_width_byte_size() == 2);

  REQUIRE(logical_type::make(type_id::INTEGER).fixed_width_byte_size() == 4);
  REQUIRE(logical_type::make(type_id::UINTEGER).fixed_width_byte_size() == 4);
  REQUIRE(logical_type::make(type_id::FLOAT).fixed_width_byte_size() == 4);
  REQUIRE(logical_type::make(type_id::DATE).fixed_width_byte_size() == 4);

  REQUIRE(logical_type::make(type_id::BIGINT).fixed_width_byte_size() == 8);
  REQUIRE(logical_type::make(type_id::UBIGINT).fixed_width_byte_size() == 8);
  REQUIRE(logical_type::make(type_id::HUGEINT).fixed_width_byte_size() == 8);   // mapped to INT64
  REQUIRE(logical_type::make(type_id::UHUGEINT).fixed_width_byte_size() == 8);  // mapped to UINT64
  REQUIRE(logical_type::make(type_id::DOUBLE).fixed_width_byte_size() == 8);
  REQUIRE(logical_type::make(type_id::TIMESTAMP_SEC).fixed_width_byte_size() == 8);
  REQUIRE(logical_type::make(type_id::TIMESTAMP_MS).fixed_width_byte_size() == 8);
  REQUIRE(logical_type::make(type_id::TIMESTAMP).fixed_width_byte_size() == 8);
  REQUIRE(logical_type::make(type_id::TIMESTAMP_NS).fixed_width_byte_size() == 8);

  // DECIMAL storage depends on precision
  REQUIRE(logical_type::make_decimal(9, 2).fixed_width_byte_size() == 4);     // DECIMAL32
  REQUIRE(logical_type::make_decimal(18, 4).fixed_width_byte_size() == 8);    // DECIMAL64
  REQUIRE(logical_type::make_decimal(19, 4).fixed_width_byte_size() == 16);   // DECIMAL128
  REQUIRE(logical_type::make_decimal(38, 10).fixed_width_byte_size() == 16);  // DECIMAL128

  // VARCHAR is variable-length: sentinel value 0
  REQUIRE(logical_type::make(type_id::VARCHAR).fixed_width_byte_size() == 0);
}

TEST_CASE("logical_type - fixed_width_byte_size() throws for non-scalar types", "[logical_type]")
{
  REQUIRE_THROWS(logical_type::make(type_id::STRUCT).fixed_width_byte_size());
  REQUIRE_THROWS(logical_type::make(type_id::LIST).fixed_width_byte_size());
  REQUIRE_THROWS(logical_type::make(type_id::SQLNULL).fixed_width_byte_size());
  REQUIRE_THROWS(logical_type::make(type_id::INVALID).fixed_width_byte_size());
}

// ============================================================================
// to_string
// ============================================================================

TEST_CASE("logical_type - to_string() returns expected names", "[logical_type]")
{
  REQUIRE(logical_type::make(type_id::INVALID).to_string() == "INVALID");
  REQUIRE(logical_type::make(type_id::SQLNULL).to_string() == "NULL");
  REQUIRE(logical_type::make(type_id::BOOLEAN).to_string() == "BOOLEAN");
  REQUIRE(logical_type::make(type_id::TINYINT).to_string() == "TINYINT");
  REQUIRE(logical_type::make(type_id::SMALLINT).to_string() == "SMALLINT");
  REQUIRE(logical_type::make(type_id::INTEGER).to_string() == "INTEGER");
  REQUIRE(logical_type::make(type_id::BIGINT).to_string() == "BIGINT");
  REQUIRE(logical_type::make(type_id::HUGEINT).to_string() == "HUGEINT");
  REQUIRE(logical_type::make(type_id::UTINYINT).to_string() == "UTINYINT");
  REQUIRE(logical_type::make(type_id::USMALLINT).to_string() == "USMALLINT");
  REQUIRE(logical_type::make(type_id::UINTEGER).to_string() == "UINTEGER");
  REQUIRE(logical_type::make(type_id::UBIGINT).to_string() == "UBIGINT");
  REQUIRE(logical_type::make(type_id::UHUGEINT).to_string() == "UHUGEINT");
  REQUIRE(logical_type::make(type_id::FLOAT).to_string() == "FLOAT");
  REQUIRE(logical_type::make(type_id::DOUBLE).to_string() == "DOUBLE");
  REQUIRE(logical_type::make(type_id::DATE).to_string() == "DATE");
  REQUIRE(logical_type::make(type_id::TIMESTAMP_SEC).to_string() == "TIMESTAMP_S");
  REQUIRE(logical_type::make(type_id::TIMESTAMP_MS).to_string() == "TIMESTAMP_MS");
  REQUIRE(logical_type::make(type_id::TIMESTAMP).to_string() == "TIMESTAMP");
  REQUIRE(logical_type::make(type_id::TIMESTAMP_NS).to_string() == "TIMESTAMP_NS");
  REQUIRE(logical_type::make(type_id::VARCHAR).to_string() == "VARCHAR");
  REQUIRE(logical_type::make(type_id::STRUCT).to_string() == "STRUCT");
  REQUIRE(logical_type::make(type_id::LIST).to_string() == "LIST");
  REQUIRE(logical_type::make_decimal(18, 6).to_string() == "DECIMAL(18,6)");
  REQUIRE(logical_type::make_decimal(9, 0).to_string() == "DECIMAL(9,0)");
}

// ============================================================================
// Equality operators
// ============================================================================

TEST_CASE("logical_type - operator== and operator!=", "[logical_type]")
{
  // Same type compares equal
  REQUIRE(logical_type::make(type_id::INTEGER) == logical_type::make(type_id::INTEGER));
  REQUIRE(logical_type::make(type_id::VARCHAR) == logical_type::make(type_id::VARCHAR));
  REQUIRE(logical_type::make(type_id::INVALID) == logical_type::make(type_id::INVALID));

  // Different types compare unequal
  REQUIRE(logical_type::make(type_id::INTEGER) != logical_type::make(type_id::BIGINT));
  REQUIRE(logical_type::make(type_id::FLOAT) != logical_type::make(type_id::DOUBLE));

  // DECIMAL: same precision + scale compare equal
  REQUIRE(logical_type::make_decimal(10, 3) == logical_type::make_decimal(10, 3));

  // DECIMAL: different precision compare unequal
  REQUIRE(logical_type::make_decimal(10, 3) != logical_type::make_decimal(11, 3));

  // DECIMAL: different scale compare unequal
  REQUIRE(logical_type::make_decimal(10, 3) != logical_type::make_decimal(10, 4));

  // DECIMAL vs non-DECIMAL
  REQUIRE(logical_type::make_decimal(10, 0) != logical_type::make(type_id::INTEGER));

  // Default (SQLNULL) equals explicit SQLNULL
  REQUIRE(logical_type() == logical_type::make(type_id::SQLNULL));
}

// ============================================================================
// ARRAY type (fixed-size list)
// ============================================================================

TEST_CASE("logical_type - make_array() stores child type and size", "[logical_type]")
{
  auto arr = logical_type::make_array(logical_type::make(type_id::INTEGER), 3);
  REQUIRE(arr.id() == type_id::ARRAY);
  REQUIRE(arr.is_array());
  REQUIRE(arr.has_child());
  REQUIRE(arr.array_size() == 3);
  REQUIRE(arr.array_child() == logical_type::make(type_id::INTEGER));
}

TEST_CASE("logical_type - make_array() with size 0 means any size", "[logical_type]")
{
  // size 0 is the "any size" sentinel and must be stored verbatim, not rejected.
  auto arr = logical_type::make_array(logical_type::make(type_id::DOUBLE), 0);
  REQUIRE(arr.is_array());
  REQUIRE(arr.array_size() == 0);
  REQUIRE(arr.array_child() == logical_type::make(type_id::DOUBLE));
}

TEST_CASE("logical_type - make_array() supports nesting", "[logical_type]")
{
  // ARRAY of ARRAYs of INTEGER: the child is itself an array.
  auto inner = logical_type::make_array(logical_type::make(type_id::INTEGER), 2);
  auto outer = logical_type::make_array(inner, 4);
  REQUIRE(outer.is_array());
  REQUIRE(outer.array_size() == 4);
  REQUIRE(outer.array_child().is_array());
  REQUIRE(outer.array_child().array_size() == 2);
  REQUIRE(outer.array_child().array_child() == logical_type::make(type_id::INTEGER));
}

TEST_CASE("logical_type - is_array() is false for every non-array type", "[logical_type]")
{
  for (auto id : k_all_type_ids) {
    if (id == type_id::ARRAY) continue;  // Skip ARRAY itself
    REQUIRE_FALSE(logical_type::make(id).is_array());
  }
  REQUIRE_FALSE(logical_type::make_decimal(10, 2).is_array());
}

TEST_CASE("logical_type - ARRAY is not fixed-width", "[logical_type]")
{
  REQUIRE_FALSE(logical_type::make_array(logical_type::make(type_id::INTEGER), 3).is_fixed_width());
  REQUIRE_FALSE(logical_type::make(type_id::ARRAY).is_fixed_width());
}

TEST_CASE("logical_type - fixed_width_byte_size() throws for ARRAY", "[logical_type]")
{
  REQUIRE_THROWS(
    logical_type::make_array(logical_type::make(type_id::INTEGER), 3).fixed_width_byte_size());
}

TEST_CASE("logical_type - array_child() throws when ARRAY has no child metadata", "[logical_type]")
{
  // An ARRAY built without make_array() (e.g. via make()) has no child and must throw
  REQUIRE_FALSE(logical_type::make(type_id::ARRAY).has_child());
  REQUIRE_THROWS(logical_type::make(type_id::ARRAY).array_child());
}

TEST_CASE("logical_type - to_string() formats ARRAY with size and child", "[logical_type]")
{
  REQUIRE(logical_type::make_array(logical_type::make(type_id::INTEGER), 3).to_string() ==
          "INTEGER[3]");
  // size 0 is DuckDB's unsized array form
  REQUIRE(logical_type::make_array(logical_type::make(type_id::DOUBLE), 0).to_string() ==
          "DOUBLE[ANY]");
  // Nested arrays recurse through the child's to_string()
  auto nested =
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::BIGINT), 2), 4);
  REQUIRE(nested.to_string() == "BIGINT[2][4]");
}

TEST_CASE("logical_type - ARRAY equality compares size and child recursively", "[logical_type]")
{
  auto int3 = logical_type::make_array(logical_type::make(type_id::INTEGER), 3);

  // Identical arrays compare equal
  REQUIRE(int3 == logical_type::make_array(logical_type::make(type_id::INTEGER), 3));

  // Different size compares unequal
  REQUIRE(int3 != logical_type::make_array(logical_type::make(type_id::INTEGER), 4));

  // Different child type compares unequal
  REQUIRE(int3 != logical_type::make_array(logical_type::make(type_id::BIGINT), 3));

  // Nested arrays compare equal element-by-element
  auto nested_a =
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::INTEGER), 2), 4);
  auto nested_b =
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::INTEGER), 2), 4);
  REQUIRE(nested_a == nested_b);

  // Mismatch in the nested child size propagates to inequality.
  auto nested_c =
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::INTEGER), 5), 4);
  REQUIRE(nested_a != nested_c);

  // One side has child metadata, the other does not.
  REQUIRE(int3 != logical_type::make(type_id::ARRAY));
  REQUIRE(logical_type::make(type_id::ARRAY) != int3);

  // ARRAY vs unrelated type.
  REQUIRE(int3 != logical_type::make(type_id::INTEGER));
}

// ============================================================================
// DuckDB boundary conversions
// ============================================================================

TEST_CASE("type_conversions - from_duckdb covers all supported DuckDB types", "[logical_type]")
{
  using duckdb::LogicalType;
  using duckdb::LogicalTypeId;

  // Scalar types map to the matching sirius type_id
  REQUIRE(sirius::from_duckdb(LogicalType::BOOLEAN).id() == type_id::BOOLEAN);
  REQUIRE(sirius::from_duckdb(LogicalType::TINYINT).id() == type_id::TINYINT);
  REQUIRE(sirius::from_duckdb(LogicalType::SMALLINT).id() == type_id::SMALLINT);
  REQUIRE(sirius::from_duckdb(LogicalType::INTEGER).id() == type_id::INTEGER);
  REQUIRE(sirius::from_duckdb(LogicalType::BIGINT).id() == type_id::BIGINT);
  REQUIRE(sirius::from_duckdb(LogicalType::HUGEINT).id() == type_id::HUGEINT);
  REQUIRE(sirius::from_duckdb(LogicalType::UTINYINT).id() == type_id::UTINYINT);
  REQUIRE(sirius::from_duckdb(LogicalType::USMALLINT).id() == type_id::USMALLINT);
  REQUIRE(sirius::from_duckdb(LogicalType::UINTEGER).id() == type_id::UINTEGER);
  REQUIRE(sirius::from_duckdb(LogicalType::UBIGINT).id() == type_id::UBIGINT);
  REQUIRE(sirius::from_duckdb(LogicalType::UHUGEINT).id() == type_id::UHUGEINT);
  REQUIRE(sirius::from_duckdb(LogicalType::FLOAT).id() == type_id::FLOAT);
  REQUIRE(sirius::from_duckdb(LogicalType::DOUBLE).id() == type_id::DOUBLE);
  REQUIRE(sirius::from_duckdb(LogicalType::DATE).id() == type_id::DATE);
  REQUIRE(sirius::from_duckdb(LogicalType::TIMESTAMP_S).id() == type_id::TIMESTAMP_SEC);
  REQUIRE(sirius::from_duckdb(LogicalType::TIMESTAMP_MS).id() == type_id::TIMESTAMP_MS);
  REQUIRE(sirius::from_duckdb(LogicalType::TIMESTAMP).id() == type_id::TIMESTAMP);
  REQUIRE(sirius::from_duckdb(LogicalType::TIMESTAMP_NS).id() == type_id::TIMESTAMP_NS);
  REQUIRE(sirius::from_duckdb(LogicalType::VARCHAR).id() == type_id::VARCHAR);
  REQUIRE(sirius::from_duckdb(LogicalType::STRUCT({})).id() == type_id::STRUCT);
  REQUIRE(sirius::from_duckdb(LogicalType::LIST(LogicalType::INTEGER)).id() == type_id::LIST);

  // DECIMAL: precision and scale are preserved
  auto dec = sirius::from_duckdb(LogicalType::DECIMAL(18, 6));
  REQUIRE(dec.id() == type_id::DECIMAL);
  REQUIRE(dec.decimal_precision() == 18);
  REQUIRE(dec.decimal_scale() == 6);
}

TEST_CASE("type_conversions - from_duckdb throws for unsupported DuckDB types", "[logical_type]")
{
  REQUIRE_THROWS(sirius::from_duckdb(duckdb::LogicalType::INTERVAL));
  REQUIRE_THROWS(sirius::from_duckdb(duckdb::LogicalType::BLOB));
  // JSON is a VARCHAR alias in DuckDB (LogicalTypeId::VARCHAR), so it maps to type_id::VARCHAR.
  REQUIRE(sirius::from_duckdb(duckdb::LogicalType::JSON()).id() == sirius::type_id::VARCHAR);
}

TEST_CASE("type_conversions - to_duckdb round-trip for all GPU-processable types", "[logical_type]")
{
  // For every type that can go through to_duckdb → from_duckdb, the round-trip
  // must recover the original sirius::logical_type.
  const std::vector<logical_type> round_trip_types = {
    logical_type::make(type_id::BOOLEAN),       logical_type::make(type_id::TINYINT),
    logical_type::make(type_id::SMALLINT),      logical_type::make(type_id::INTEGER),
    logical_type::make(type_id::BIGINT),        logical_type::make(type_id::HUGEINT),
    logical_type::make(type_id::UTINYINT),      logical_type::make(type_id::USMALLINT),
    logical_type::make(type_id::UINTEGER),      logical_type::make(type_id::UBIGINT),
    logical_type::make(type_id::UHUGEINT),      logical_type::make(type_id::FLOAT),
    logical_type::make(type_id::DOUBLE),        logical_type::make(type_id::DATE),
    logical_type::make(type_id::TIMESTAMP_SEC), logical_type::make(type_id::TIMESTAMP_MS),
    logical_type::make(type_id::TIMESTAMP),     logical_type::make(type_id::TIMESTAMP_NS),
    logical_type::make(type_id::VARCHAR),       logical_type::make_decimal(9, 2),
    logical_type::make_decimal(18, 9),          logical_type::make_decimal(38, 10),
  };

  for (const auto& t : round_trip_types) {
    auto duckdb_type   = sirius::to_duckdb(t);
    auto round_tripped = sirius::from_duckdb(duckdb_type);
    REQUIRE(round_tripped == t);
  }
}

TEST_CASE("type_conversions - to_duckdb throws for non-GPU types", "[logical_type]")
{
  REQUIRE_THROWS(sirius::to_duckdb(logical_type::make(type_id::INVALID)));
  REQUIRE_THROWS(sirius::to_duckdb(logical_type::make(type_id::LIST)));
  // STRUCT maps to LogicalType::STRUCT({}) — cuDF supports STRUCT columns
  REQUIRE_NOTHROW(sirius::to_duckdb(logical_type::make(type_id::STRUCT)));
}

TEST_CASE("type_conversions - from_duckdb_vec / to_duckdb_vec round-trip", "[logical_type]")
{
  duckdb::vector<duckdb::LogicalType> duckdb_types = {
    duckdb::LogicalType::INTEGER,
    duckdb::LogicalType::VARCHAR,
    duckdb::LogicalType::DECIMAL(15, 3),
    duckdb::LogicalType::TIMESTAMP,
  };

  auto sirius_types    = sirius::from_duckdb_vec(duckdb_types);
  auto recovered_types = sirius::to_duckdb_vec(sirius_types);

  REQUIRE(sirius_types.size() == duckdb_types.size());
  REQUIRE(recovered_types.size() == duckdb_types.size());

  REQUIRE(sirius_types[0] == logical_type::make(type_id::INTEGER));
  REQUIRE(sirius_types[1] == logical_type::make(type_id::VARCHAR));
  REQUIRE(sirius_types[2] == logical_type::make_decimal(15, 3));
  REQUIRE(sirius_types[3] == logical_type::make(type_id::TIMESTAMP));

  REQUIRE(recovered_types[0] == duckdb::LogicalType::INTEGER);
  REQUIRE(recovered_types[3] == duckdb::LogicalType::TIMESTAMP);
}

TEST_CASE("type_conversions - from_duckdb maps ARRAY to sirius ARRAY", "[logical_type]")
{
  using duckdb::LogicalType;

  auto arr = sirius::from_duckdb(LogicalType::ARRAY(LogicalType::INTEGER, duckdb::optional_idx(3)));
  REQUIRE(arr.id() == type_id::ARRAY);
  REQUIRE(arr.array_size() == 3);
  REQUIRE(arr.array_child().id() == type_id::INTEGER);
}

TEST_CASE("type_conversions - ARRAY round-trips through to_duckdb / from_duckdb", "[logical_type]")
{
  const std::vector<logical_type> array_types = {
    logical_type::make_array(logical_type::make(type_id::INTEGER), 3),
    logical_type::make_array(logical_type::make(type_id::DOUBLE), 1),
    logical_type::make_array(logical_type::make(type_id::BIGINT), 16),
    // Nested ARRAY of ARRAY.
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::INTEGER), 2), 4),
  };

  for (const auto& t : array_types) {
    auto duckdb_type   = sirius::to_duckdb(t);
    auto round_tripped = sirius::from_duckdb(duckdb_type);
    REQUIRE(round_tripped == t);
  }
}

TEST_CASE("type_conversions - to_duckdb throws for ARRAY without child metadata", "[logical_type]")
{
  // An ARRAY missing its child type cannot be lowered back to DuckDB.
  REQUIRE_THROWS(sirius::to_duckdb(logical_type::make(type_id::ARRAY)));
}
