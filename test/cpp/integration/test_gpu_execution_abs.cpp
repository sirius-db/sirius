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
 * @file test_gpu_execution_abs.cpp
 * @brief GPU abs() on FLOAT, DOUBLE, DECIMAL and signed integers matches DuckDB.
 *
 * Covers negative and positive values, signed zeros, infinities, NaN, NULL and the TPC-DS shape
 * of a relative difference in a filter, DECIMAL32, DECIMAL64 and DECIMAL128 values, and the signed
 * integer types including the overflow of the smallest value, and unsigned integers. HUGEINT falls
 * back during planning.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

using AbsFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_abs_table(AbsFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE abs_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f, CAST(id AS INTEGER) - 20 AS i,"
    " CAST((id % 200) - 100 AS TINYINT) AS i8, CAST((id * 31) % 60000 - 30000 AS SMALLINT) AS i16,"
    " CAST(id * 1000003 - 1000000000 AS BIGINT) AS i64, CAST(id % 256 AS UTINYINT) AS u8,"
    " CAST((id * 31) % 65536 AS USMALLINT) AS u16, CAST(id * 100003 AS UINTEGER) AS u32,"
    " CAST(id AS UBIGINT) * 8000000000000000 AS u64,"
    " CAST(((id * 7919) % 200000 - 100000) / 100.0 AS DECIMAL(9, 2)) AS dec9,"
    " CAST(((id * 7919) % 200000 - 100000) / 100.0 AS DECIMAL(18, 4)) AS dec18 FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " (2.675::DOUBLE), (-2.675), (1.005), (-0.285), (0.0), (-0.0), (123456.789), (-98765.4321),"
    " (1e300), (-1e300), (5e-324), (-5e-324), ('inf'::DOUBLE), ('-inf'::DOUBLE), ('nan'::DOUBLE),"
    " (NULL)) v(d)"
    " UNION ALL SELECT 100 + i, ((i * 7919) % 100003 - 50000) / ((i % 977) + 3.0)"
    " FROM range(2000) t(i));");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs of DOUBLE and FLOAT matches DuckDB",
                 "[integration][gpu_execution][projection][abs]")
{
  create_abs_table(*this);
  for (auto const* column : {"d", "f"}) {
    CAPTURE(column);
    compare_gpu_vs_cpu(std::string("SELECT id, abs(") + column + ") AS a FROM abs_t");
    compare_gpu_vs_cpu(std::string("SELECT id, abs(") + column + " - 1.5) AS a FROM abs_t");
  }
}

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs composes in filters and quotients",
                 "[integration][gpu_execution][projection][abs]")
{
  create_abs_table(*this);
  compare_gpu_vs_cpu("SELECT id FROM abs_t WHERE id >= 100 AND abs(d) > 3");
  // The TPC-DS Q47 shape: a relative difference to a running average.
  compare_gpu_vs_cpu(
    "SELECT id FROM abs_t WHERE id >= 100 AND CASE WHEN d > 0 THEN abs(d - 2.0) / d ELSE NULL END"
    " > 0.1");
}

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs of DECIMAL matches DuckDB",
                 "[integration][gpu_execution][projection][abs]")
{
  create_abs_table(*this);
  // DECIMAL128 columns of a native table do not scan on the GPU, so that width comes from a cast.
  for (auto const* column : {"dec9", "dec18", "CAST(dec18 AS DECIMAL(38, 4))"}) {
    CAPTURE(column);
    compare_gpu_vs_cpu(std::string("SELECT id, abs(") + column + ") AS a FROM abs_t");
    compare_gpu_vs_cpu(std::string("SELECT id FROM abs_t WHERE abs(") + column + ") > 100");
  }
  compare_gpu_vs_cpu("SELECT id, abs(dec18 - dec9) AS a FROM abs_t");
}

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs of signed integers matches DuckDB",
                 "[integration][gpu_execution][projection][abs]")
{
  create_abs_table(*this);
  for (auto const* column : {"i8", "i16", "i", "i64"}) {
    CAPTURE(column);
    compare_gpu_vs_cpu(std::string("SELECT id, abs(") + column + ") AS a FROM abs_t");
    compare_gpu_vs_cpu(std::string("SELECT id FROM abs_t WHERE abs(") + column + ") > 50");
  }
}

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs overflow of the smallest integer raises DuckDB's error",
                 "[integration][gpu_execution][projection][abs]")
{
  run_ok(
    "CREATE TABLE abs_min AS SELECT * FROM (VALUES (CAST(-128 AS TINYINT), CAST(-32768 AS "
    "SMALLINT),"
    " CAST(-2147483648 AS INTEGER), CAST(-9223372036854775808 AS BIGINT)),"
    " (CAST(5 AS TINYINT), CAST(7 AS SMALLINT), CAST(9 AS INTEGER), CAST(11 AS BIGINT)),"
    " (NULL, NULL, NULL, NULL)) v(i8, i16, i32, i64);");
  run_ok("CHECKPOINT;");
  for (auto const* column : {"i8", "i16", "i32", "i64"}) {
    CAPTURE(column);
    auto const result = con->Query(std::string("SELECT abs(") + column + ") FROM abs_min;");
    REQUIRE(result);
    REQUIRE(result->HasError());
    REQUIRE(result->GetError().find("Overflow on abs") != std::string::npos);
    // Rows that are filtered out do not raise, and NULL is not an overflow.
    compare_gpu_vs_cpu(std::string("SELECT abs(") + column + ") AS a FROM abs_min WHERE " + column +
                       " IS NULL OR " + column + " > 0");
  }
}

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs of unsigned integers returns the input",
                 "[integration][gpu_execution][projection][abs]")
{
  create_abs_table(*this);
  for (auto const* column : {"u8", "u16", "u32", "u64"}) {
    CAPTURE(column);
    compare_gpu_vs_cpu(std::string("SELECT id, abs(") + column + ") AS a FROM abs_t");
  }
  run_ok(
    "CREATE TABLE abs_max AS SELECT * FROM (VALUES (CAST(255 AS UTINYINT), CAST(65535 AS "
    "USMALLINT),"
    " CAST(4294967295 AS UINTEGER), CAST(18446744073709551615 AS UBIGINT)),"
    " (CAST(0 AS UTINYINT), CAST(0 AS USMALLINT), CAST(0 AS UINTEGER), CAST(0 AS UBIGINT)),"
    " (NULL, NULL, NULL, NULL)) v(u8, u16, u32, u64);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT abs(u8), abs(u16), abs(u32), abs(u64) FROM abs_max");
}

TEST_CASE_METHOD(AbsFixture,
                 "gpu_execution abs of HUGEINT falls back during planning",
                 "[integration][gpu_execution][projection][abs]")
{
  create_abs_table(*this);
  expect_plan_fallback_matches_cpu("SELECT id, abs(CAST(i AS HUGEINT)) AS a FROM abs_t");
}
