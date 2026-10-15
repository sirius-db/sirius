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
 * @file test_gpu_execution_negate.cpp
 * @brief GPU unary minus matches DuckDB.
 *
 * DuckDB binds `-x` as a one-argument "-" function. The GPU used to read a second argument that
 * does not exist and crashed the process. These cases run unary minus on every type the GPU
 * negates (signed integers, FLOAT, DOUBLE, DECIMAL of every width), with NULLs, in projections,
 * filters, aggregate arguments and nested expressions, and check that each query ran on the GPU.
 * Negating a signed integer's minimum raises like DuckDB, and types the GPU does not negate fall
 * back at plan time.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <filesystem>
#include <string>

using NegateFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_negate_table(NegateFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE neg_t AS SELECT id, t::TINYINT AS t, s::SMALLINT AS s, i::INTEGER AS i,"
    " b::BIGINT AS b, f::FLOAT AS f, d::DOUBLE AS d,"
    " d9::DECIMAL(9, 2) AS d9, d18::DECIMAL(18, 4) AS d18, k::INTEGER AS k FROM ("
    " SELECT * FROM (VALUES"
    " (1, 5, 300, 7, -5, 2.5, 2.5, 1234.56, 12345678901234.5678, 3),"
    " (2, -7, -300, -3, 9, '-0.0', '-0.0', -0.01, -0.0001, -4),"
    " (3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),"
    " (4, 0, 0, 0, 0, 0.0, 0.0, 0.00, 0.0000, 2),"
    " (5, 127, 32767, 2147483647, 9223372036854775807, 3.4e38, 1e300, 9999999.99,"
    "  99999999999999.9999, -1),"
    " (6, -127, -32767, -2147483647, -9223372036854775807, 'inf', '-inf', -9999999.99,"
    "  -99999999999999.9999, 1)"
    ") v(id, t, s, i, b, f, d, d9, d18, k)"
    " UNION ALL SELECT 100 + n, n % 100 - 50, n % 20000 - 10000, n * 7919 % 100003 - 50000,"
    " n * 104729 - 2000000, n / 7.0 - 300, n / 3.0 - 1000,"
    " (n * 13 % 200001 - 100000) * 0.01, (n * 104729 - 2000000) * 0.0001, n % 7 - 3"
    " FROM range(5000) r(n));");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(NegateFixture,
                 "gpu_execution unary minus in a projection matches DuckDB",
                 "[integration][gpu_execution][projection][negate]")
{
  create_negate_table(*this);
  for (auto const* column : {"t", "s", "i", "b", "f", "d", "d9", "d18"}) {
    CAPTURE(column);
    compare_gpu_vs_cpu(std::string("SELECT id, -") + column + " AS n FROM neg_t");
  }
}

TEST_CASE_METHOD(NegateFixture,
                 "gpu_execution unary minus on DECIMAL128 matches DuckDB",
                 "[integration][gpu_execution][projection][negate]")
{
  create_negate_table(*this);
  // The native scan does not decode DECIMAL128 storage, so these rows come from parquet.
  auto const path = temp_db_path + ".negate.parquet";
  run_ok("COPY (SELECT id, (d18 * 1000000000000)::DECIMAL(38, 9) AS d38 FROM neg_t) TO '" + path +
         "' (FORMAT PARQUET);");
  compare_gpu_vs_cpu("SELECT id, -d38 AS n FROM read_parquet('" + path + "')");
  compare_gpu_vs_cpu("SELECT id FROM read_parquet('" + path + "') WHERE -d38 > 0");
  std::filesystem::remove(path);
}

TEST_CASE_METHOD(NegateFixture,
                 "gpu_execution unary minus in filters, aggregates and nested expressions matches "
                 "DuckDB",
                 "[integration][gpu_execution][projection][negate]")
{
  create_negate_table(*this);
  for (auto const* query : {
         // DuckDB v2 moves a unary minus compared with a constant onto the constant, adding an
         // isnan guard for floating point that the GPU does not run. Comparing a negation with a
         // column keeps the negation in the filter.
         "SELECT id FROM neg_t WHERE -i > 0",
         "SELECT id FROM neg_t WHERE -i > k",
         "SELECT id FROM neg_t WHERE -d18 < -1 AND -d >= k - 500",
         "SELECT id FROM neg_t WHERE -t = 7 OR -d9 = 0.01",
         // The extreme rows would overflow a BIGINT sum, so the aggregates read the generated rows.
         "SELECT sum(-i), sum(-b), sum(-d18), min(-d), max(-s), count(-t) FROM neg_t WHERE id >= "
         "100",
         "SELECT k, sum(-i), sum(-d9) FROM neg_t WHERE id >= 100 GROUP BY k",
         "SELECT id, -(i * k) AS a, - -i AS b, -(-d18) AS c, -d9 + d9 AS z FROM neg_t",
         "SELECT id, -(d / 2) AS a, -(i + k) * 2 AS b FROM neg_t WHERE id >= 100",
         "SELECT id, +i AS a, +d18 AS b FROM neg_t",
       }) {
    CAPTURE(query);
    compare_gpu_vs_cpu(query);
  }
}

TEST_CASE_METHOD(NegateFixture,
                 "gpu_execution unary minus of a signed integer minimum raises like DuckDB",
                 "[integration][gpu_execution][projection][negate]")
{
  run_ok(
    "CREATE TABLE neg_min AS SELECT * FROM (VALUES (1, 1::INTEGER, 1::BIGINT),"
    " (2, (-2147483648)::INTEGER, (-9223372036854775808)::BIGINT)) v(id, i, b);");
  run_ok("CHECKPOINT;");
  for (auto const* column : {"i", "b"}) {
    CAPTURE(column);
    auto const query = std::string("SELECT id, -") + column + " FROM neg_min";
    run_ok("SET gpu_execution = false;");
    auto cpu = con->Query(query);
    REQUIRE(cpu->HasError());
    REQUIRE_THAT(cpu->GetError(), Catch::Matchers::ContainsSubstring("Overflow in negation"));

    run_ok("SET gpu_execution = true;");
    run_ok("SET enable_duckdb_fallback = false;");
    auto gpu = con->Query(query);
    run_ok("SET enable_duckdb_fallback = true;");
    REQUIRE(gpu->HasError());
    INFO(gpu->GetError());
    REQUIRE_THAT(gpu->GetError(), Catch::Matchers::ContainsSubstring("Overflow in negation"));
  }
}

TEST_CASE_METHOD(NegateFixture,
                 "gpu_execution unary minus on a type the GPU does not negate falls back at plan "
                 "time",
                 "[integration][gpu_execution][projection][negate]")
{
  create_negate_table(*this);
  // HUGEINT rides an INT64 carrier on the GPU, and DuckDB rejects negating a nonzero unsigned
  // value. The operands themselves run on the GPU, so the negation is what falls back.
  compare_gpu_vs_cpu(
    "SELECT id, i::HUGEINT AS h, (k + 3)::UTINYINT AS u FROM neg_t WHERE id >= 100");
  expect_plan_fallback_matches_cpu("SELECT id, -(i::HUGEINT) AS n FROM neg_t WHERE id >= 100");
  // k + 3 is zero only where k = -3, and DuckDB negates an unsigned zero to zero.
  expect_plan_fallback_matches_cpu(
    "SELECT id, -((k + 3)::UTINYINT) AS n FROM neg_t WHERE id >= 100 AND k = -3");

  // Other unsigned values raise on the CPU after the plan-time fallback, as without the GPU.
  auto const query = std::string("SELECT id, -((k + 3)::UTINYINT) AS n FROM neg_t WHERE id >= 100");
  run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto gpu          = con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(gpu->HasError());
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);
  run_ok("SET gpu_execution = false;");
  auto cpu = con->Query(query);
  run_ok("SET gpu_execution = true;");
  REQUIRE(cpu->HasError());
  REQUIRE_THAT(cpu->GetError(), Catch::Matchers::ContainsSubstring("Overflow in negation"));
  REQUIRE(gpu->GetError() == cpu->GetError());
}
