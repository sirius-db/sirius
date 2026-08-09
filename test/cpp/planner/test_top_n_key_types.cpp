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

/**
 * @file test_top_n_key_types.cpp
 * @brief The per-key type allowlist, asserted directly at every band boundary.
 *
 * Reachability differs by scan format, so plan-shape tests alone cannot pin every band: `p >= 19`
 * is refused by the duckdb-native scan but reaches producer admission through parquet (pinned
 * there by the plan-shape suite), while `p <= 4` is genuinely unreachable because
 * `sirius::get_cudf_type` cannot map it at all. Asserting the rule directly covers every band
 * without depending on which scan a test builds.
 */

#include <catch.hpp>
#include <planner/top_n_key_types.hpp>

using sirius::planner::admitted_top_n_key_storage_type;

TEST_CASE("the top-n key allowlist admits exactly the exactly-representable types",
          "[dynamic_filter][top_n][placement]")
{
  SECTION("the integer and date types map to their exact cuDF representation")
  {
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::TINYINT) ==
            cudf::data_type{cudf::type_id::INT8});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::SMALLINT) ==
            cudf::data_type{cudf::type_id::INT16});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::INTEGER) ==
            cudf::data_type{cudf::type_id::INT32});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::BIGINT) ==
            cudf::data_type{cudf::type_id::INT64});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DATE) ==
            cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  }

  SECTION("types whose ordering has no exactness proof are refused")
  {
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::VARCHAR) == std::nullopt);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DOUBLE) == std::nullopt);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::FLOAT) == std::nullopt);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::HUGEINT) == std::nullopt);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::UBIGINT) == std::nullopt);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::TIMESTAMP) == std::nullopt);
  }

  SECTION("decimal precision selects the storage type, and the scale is carried negated")
  {
    // cuDF's scale convention is the negation of DuckDB's, and the scale is part of the type:
    // it is what makes the raw-integer comparison downstream mean the right thing.
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(9, 2)) ==
            cudf::data_type{cudf::type_id::DECIMAL32, -2});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(5, 0)) ==
            cudf::data_type{cudf::type_id::DECIMAL32, 0});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(18, 4)) ==
            cudf::data_type{cudf::type_id::DECIMAL64, -4});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(10, 10)) ==
            cudf::data_type{cudf::type_id::DECIMAL64, -10});
  }

  SECTION("the band boundaries fall exactly where the physical width changes")
  {
    // p=4 is INT16 in DuckDB, which cuDF fixed point has no counterpart for.
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(4, 2)) == std::nullopt);
    // p=5..9 is INT32, p=10..18 is INT64: the two bands the boundary path holds exactly.
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(5, 2))->id() ==
            cudf::type_id::DECIMAL32);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(9, 2))->id() ==
            cudf::type_id::DECIMAL32);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(10, 2))->id() ==
            cudf::type_id::DECIMAL64);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(18, 2))->id() ==
            cudf::type_id::DECIMAL64);
    // p>=19 is INT128. The comparison kernel reads components by width 1/2/4/8 and the components
    // themselves widen to int64, so a 16-byte scaled integer would be truncated, not rejected.
    // Admitting it needs the params widened first; until then this refusal is the whole defense.
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(19, 2)) == std::nullopt);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(38, 4)) == std::nullopt);
  }
}
