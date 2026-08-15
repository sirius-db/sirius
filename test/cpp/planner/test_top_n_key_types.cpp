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
 * Reachability differs by scan format, so plan-shape tests alone cannot pin every band: a
 * `p >= 19` *column* is refused by the duckdb-native scan's decode gate but reaches producer
 * admission through parquet (pinned there by the plan-shape suite), an aggregate-output
 * DECIMAL128 key admits on every format, and `p <= 4` is genuinely unreachable because
 * `sirius::get_cudf_type` cannot map it at all. Asserting the rule directly covers every band
 * without depending on which scan a test builds.
 */

#include <cudf/cudf_utils.hpp>

#include <catch.hpp>
#include <helper/logical_type.hpp>
#include <planner/top_n_key_types.hpp>

#include <algorithm>
#include <cstdint>

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
    // p=5..9 is INT32, p=10..18 is INT64, p=19..38 is INT128: the three bands the boundary path
    // holds exactly since the __int128_t widening and the kernel's width-16 load landed.
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(5, 2))->id() ==
            cudf::type_id::DECIMAL32);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(9, 2))->id() ==
            cudf::type_id::DECIMAL32);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(10, 2))->id() ==
            cudf::type_id::DECIMAL64);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(18, 2))->id() ==
            cudf::type_id::DECIMAL64);
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(19, 2)) ==
            cudf::data_type{cudf::type_id::DECIMAL128, -2});
    REQUIRE(admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(38, 4)) ==
            cudf::data_type{cudf::type_id::DECIMAL128, -4});
  }
}

TEST_CASE("decimal admission delegates to the single cudf banding derivation",
          "[dynamic_filter][top_n][placement]")
{
  // The single-source property: admission's decimal verdict is definitionally the banding
  // `sirius::cudf_decimal_type` derives, and `sirius::get_cudf_type` executes with the same
  // derivation -- so the planner cannot admit a mapping the engine does not use.
  for (std::uint8_t p = 1; p <= 38; ++p) {
    auto const s = std::min<std::uint8_t>(p, 4);
    CAPTURE(static_cast<int>(p), static_cast<int>(s));

    auto const admitted = admitted_top_n_key_storage_type(duckdb::LogicalType::DECIMAL(p, s));
    auto const banded   = sirius::cudf_decimal_type(p, s);
    REQUIRE(admitted.has_value() == (p > 4));
    REQUIRE(admitted == banded);

    auto const t = sirius::logical_type::make_decimal(p, s);
    if (p <= 4) {
      // The p <= 4 refusal keeps its byte-identical exception message.
      REQUIRE_THROWS_WITH(
        sirius::get_cudf_type(t),
        Catch::Contains("stored as INT16 in DuckDB") && Catch::Contains("CPU fallback"));
    } else {
      REQUIRE(sirius::get_cudf_type(t) == *banded);
    }
  }
}
