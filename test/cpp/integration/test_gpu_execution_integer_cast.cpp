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
 * @file test_gpu_execution_integer_cast.cpp
 * @brief GPU CAST to integer types, and from integers to DECIMAL, matches DuckDB.
 *
 * cudf::cast and the cuDF AST CAST_TO_INT64 truncate toward zero and wrap values that do not fit.
 * DuckDB rounds FLOAT and DOUBLE to the nearest integer with ties to even (0.93 -> 1, 2.5 -> 2,
 * -2.675 -> -3), rounds DECIMAL half away from zero (2.5000 -> 3), and fails values that do not fit
 * the target: CAST raises an error and TRY_CAST yields NULL. Covers ties, signed values, values at
 * and beyond each target's range, NaN, infinities and NULL, through the materialized and the AST
 * evaluation paths and over parquet.
 *
 * DuckDB only range-checks the unrounded FLOAT or DOUBLE, so a value within 0.5 below the target's
 * maximum plus one rounds out of range and overflows there (127.6 casts to TINYINT -128); the GPU
 * fails it. Such values are not tested.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using IntegerCastFixture = sirius::test::GpuExecutionFixture;

namespace {

constexpr char const* kIntegerTargets[] = {
  "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"};

/// DECIMAL targets for integer sources. DECIMAL with a precision of 4 or less has no cuDF type.
constexpr char const* kDecimalTargets[] = {"DECIMAL(9,0)",
                                           "DECIMAL(9,2)",
                                           "DECIMAL(18,0)",
                                           "DECIMAL(18,8)",
                                           "DECIMAL(38,0)",
                                           "DECIMAL(38,30)"};

/// `vals_t` holds DOUBLE `d`, FLOAT `f` and DECIMAL(18,4) `dec`: ties, values next to each integer
/// type's bounds, NaN, infinities and NULL. `ints_t` holds BIGINT `b` and its INTEGER, SMALLINT,
/// TINYINT and UBIGINT casts at and beyond each integer type's bounds.
void create_tables(IntegerCastFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE vals_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f,"
    " TRY_CAST(d AS DECIMAL(18,4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " ('0.93'::DOUBLE), ('-0.93'::DOUBLE), ('2.5'::DOUBLE), ('-2.5'::DOUBLE), ('1.5'::DOUBLE),"
    " ('0.5'::DOUBLE), ('-0.5'::DOUBLE), ('3.5'::DOUBLE), ('2.675'::DOUBLE), ('-2.675'::DOUBLE),"
    " ('0.49999999999999994'::DOUBLE), ('-0.4'::DOUBLE), ('0.0'::DOUBLE), ('-0.0'::DOUBLE),"
    " ('5e-324'::DOUBLE), ('-5e-324'::DOUBLE), ('127.4'::DOUBLE), ('126.5'::DOUBLE),"
    " ('-128.4'::DOUBLE), ('-128.5'::DOUBLE), ('-128.6'::DOUBLE), ('128.0'::DOUBLE),"
    " ('254.5'::DOUBLE), ('255.0'::DOUBLE), ('256.0'::DOUBLE), ('32767.0'::DOUBLE),"
    " ('-32768.5'::DOUBLE), ('32768.0'::DOUBLE), ('65535.25'::DOUBLE), ('65536.0'::DOUBLE),"
    " ('2147483646.5'::DOUBLE), ('-2147483648.0'::DOUBLE), ('-2147483648.4'::DOUBLE),"
    " ('-2147483647.5'::DOUBLE), ('2147483648.0'::DOUBLE), ('4294967294.5'::DOUBLE),"
    " ('4294967296.0'::DOUBLE), ('1e10'::DOUBLE), ('-1e10'::DOUBLE), ('9.2e18'::DOUBLE),"
    " ('-9.2e18'::DOUBLE), ('9223372036854775808.0'::DOUBLE), ('-9223372036854775808.0'::DOUBLE),"
    " ('1.8e19'::DOUBLE), ('18446744073709551616.0'::DOUBLE), ('1e30'::DOUBLE), ('-1e30'::DOUBLE),"
    " ('nan'::DOUBLE), ('inf'::DOUBLE), ('-inf'::DOUBLE), (NULL)) v(d)"
    " UNION ALL SELECT 100 + i, i / 4.0 - 100 FROM range(800) t(i)"
    " UNION ALL SELECT * FROM (SELECT 5000 + i AS id,"
    " (i * 7919 % 100003) / ((i % 977) + 3.0) - 1000"
    " AS d FROM range(3000) t(i)) WHERE NOT (d >= 127.5 AND d < 128 OR d >= 255.5 AND d < 256));");
  fx.run_ok(
    "CREATE TABLE ints_t AS SELECT id, b, TRY_CAST(b AS INTEGER) AS i,"
    " TRY_CAST(b AS SMALLINT) AS s, TRY_CAST(b AS TINYINT) AS t, TRY_CAST(b AS UBIGINT) AS u FROM ("
    " SELECT row_number() OVER () AS id, b FROM (VALUES"
    " (0::BIGINT), (1), (-1), (127), (128), (-128), (-129), (255), (256), (32767), (32768),"
    " (-32768), (-32769), (65535), (65536), (2147483647), (2147483648), (-2147483648),"
    " (-2147483649), (4294967295), (4294967296), (9999999), (10000000), (999999999),"
    " (1000000000), (-1000000000), (10000000000), (99999999999999999), (100000000000000000),"
    " (999999999999999999), (1000000000000000000), (9223372036854775807),"
    " (-9223372036854775808), (NULL)) v(b)"
    " UNION ALL SELECT 100 + i, (i - 1000) * (i - 1000) * (i - 1000) FROM range(2000) t(i));");
  fx.run_ok("CHECKPOINT;");
}

/// Runs @p query on the CPU and on the GPU: both must fail with the same error, the GPU through a
/// runtime fallback that replays the query on the CPU.
void expect_same_error(IntegerCastFixture& fx, std::string const& query)
{
  CAPTURE(query);
  fx.run_ok("SET gpu_execution = false;");
  auto cpu_result = fx.con->Query(query);
  REQUIRE(cpu_result);
  REQUIRE(cpu_result->HasError());

  fx.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto gpu_result   = fx.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
  REQUIRE(gpu_result);
  if (!gpu_result->HasError()) { UNSCOPED_INFO("GPU returned: " << gpu_result->ToString()); }
  REQUIRE(gpu_result->HasError());
  REQUIRE(gpu_result->GetError() == cpu_result->GetError());
  REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks + 1);
}

}  // namespace

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution CAST of DOUBLE, FLOAT and DECIMAL to integers rounds like DuckDB",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_tables(*this);
  // Each range keeps the values every target of its signedness accepts.
  for (auto const* column : {"d", "f", "dec"}) {
    for (auto const* target : {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, CAST(") + column + " AS " + target +
                         ") AS c FROM vals_t WHERE d BETWEEN -128 AND 127");
    }
    for (auto const* target : {"UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, CAST(") + column + " AS " + target +
                         ") AS c FROM vals_t WHERE d BETWEEN 0 AND 255");
    }
  }
}

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution TRY_CAST to integers gives NULL where the value does not fit",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_tables(*this);
  for (auto const* target : kIntegerTargets) {
    for (auto const* column : {"d", "f", "dec", "CAST(dec AS DECIMAL(38,4))"}) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, TRY_CAST(") + column + " AS " + target +
                         ") AS c FROM vals_t");
    }
    for (auto const* column : {"b", "i", "s", "t"}) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, TRY_CAST(") + column + " AS " + target +
                         ") AS c FROM ints_t");
    }
  }
  // Unsigned sources reach the GPU only for targets that hold their whole range.
  for (auto const* target : {"UBIGINT", "UINTEGER"}) {
    compare_gpu_vs_cpu(std::string("SELECT id, TRY_CAST(CAST(t AS UTINYINT) AS ") + target +
                       ") AS c FROM ints_t WHERE t >= 0");
  }
}

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution CAST of integers to DECIMAL checks the precision like DuckDB",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_tables(*this);
  for (auto const* target : kDecimalTargets) {
    for (auto const* column : {"b", "i", "s", "t", "u"}) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, TRY_CAST(") + column + " AS " + target +
                         ") AS c FROM ints_t");
      compare_gpu_vs_cpu(std::string("SELECT id, CAST(") + column + " AS " + target +
                         ") AS c FROM ints_t WHERE b BETWEEN -9999999 AND 9999999");
    }
  }
}

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution CAST to integers and DECIMAL fails like DuckDB where it overflows",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_tables(*this);
  // Each query holds one value that fails the cast. The GPU must not return a wrapped or truncated
  // value: it fails, the query replays on the CPU and raises DuckDB's conversion error.
  for (auto const* query : {
         "SELECT CAST(b AS DECIMAL(9,0)) FROM ints_t WHERE b IN (0, 10000000000)",
         "SELECT CAST(b AS DECIMAL(9,0)) FROM ints_t WHERE b IN (0, 1000000000)",
         "SELECT CAST(b AS DECIMAL(18,8)) FROM ints_t WHERE b IN (0, 10000000000)",
         "SELECT CAST(u AS DECIMAL(18,0)) FROM ints_t WHERE b IN (0, 1000000000000000000)",
         "SELECT CAST(b AS INTEGER) FROM ints_t WHERE b IN (0, 2147483648)",
         "SELECT CAST(b AS UINTEGER) FROM ints_t WHERE b IN (0, -1)",
         "SELECT CAST(i AS SMALLINT) FROM ints_t WHERE b IN (0, -32769)",
         "SELECT CAST(t AS UBIGINT) FROM ints_t WHERE b IN (0, -1)",
         "SELECT CAST(d AS INTEGER) FROM vals_t WHERE d IN (0.93, 2147483648)",
         "SELECT CAST(d AS INTEGER) FROM vals_t WHERE d IN (0.93, -2147483648.4)",
         "SELECT CAST(d AS BIGINT) FROM vals_t WHERE d IN (0.93, 9223372036854775808.0)",
         "SELECT CAST(d AS BIGINT) FROM vals_t WHERE d IN (0.93, 'inf'::DOUBLE)",
         "SELECT CAST(d AS UTINYINT) FROM vals_t WHERE d IN (0.93, -0.4)",
         "SELECT CAST(f AS SMALLINT) FROM vals_t WHERE d IN (0.93, 32768)",
         "SELECT CAST(dec AS TINYINT) FROM vals_t WHERE d IN (0.93, -128.5)",
         "SELECT CAST(dec AS INTEGER) FROM vals_t WHERE d IN (0.93, 1e10)",
       }) {
    expect_same_error(*this, query);
  }
}

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution CAST to BIGINT inside cuDF AST expressions rounds like DuckDB",
                 "[integration][gpu_execution][projection][filter][integer_cast]")
{
  create_tables(*this);
  // Enough operators for the AST path: a cast from DOUBLE or DECIMAL must not lower to
  // CAST_TO_INT64, which truncates.
  compare_gpu_vs_cpu(
    "SELECT id, CAST(d AS BIGINT) * 2 + id AS c, TRY_CAST(f AS BIGINT) - id AS t FROM vals_t"
    " WHERE d BETWEEN -1e6 AND 1e6");
  compare_gpu_vs_cpu(
    "SELECT id, CAST(dec AS BIGINT) * 2 + id AS c FROM vals_t WHERE d BETWEEN -1e6 AND 1e6");
  compare_gpu_vs_cpu("SELECT id FROM vals_t WHERE CAST(dec AS BIGINT) = -3");
  compare_gpu_vs_cpu("SELECT id FROM vals_t WHERE TRY_CAST(d AS BIGINT) + 1 = 2");
  // Exact integer casts keep the AST path.
  compare_gpu_vs_cpu("SELECT id, CAST(i AS BIGINT) * 2 + CAST(t AS BIGINT) AS c FROM ints_t");
  compare_gpu_vs_cpu(
    "SELECT id, CAST(i AS DOUBLE) * 2 + d AS c FROM ints_t JOIN vals_t USING (id)");
}

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution CAST of DOUBLE to integers over parquet rounds like DuckDB",
                 "[integration][gpu_execution][parquet][integer_cast]")
{
  create_tables(*this);
  sirius::test::scratch_dir dir("integer_cast");
  auto const file = dir.file("vals.parquet");
  run_ok("SET gpu_execution = false;");
  run_ok("COPY vals_t TO " + sirius::test::sql_literal(file) + " (FORMAT parquet);");
  run_ok("SET gpu_execution = true;");
  auto const scan = "read_parquet(" + sirius::test::sql_literal(file) + ")";
  compare_gpu_vs_cpu(
    "SELECT id, CAST(d AS BIGINT) AS b, CAST(d AS INTEGER) AS i,"
    " CAST(dec AS INTEGER) AS di FROM " +
    scan + " WHERE d BETWEEN -1e6 AND 1e6");
  compare_gpu_vs_cpu("SELECT id, TRY_CAST(d AS INTEGER) AS i FROM " + scan);
  compare_gpu_vs_cpu("SELECT id FROM " + scan + " WHERE CAST(dec AS BIGINT) = 1");
  compare_gpu_vs_cpu("SELECT id FROM " + scan + " WHERE TRY_CAST(d AS BIGINT) = -3");
  compare_gpu_vs_cpu("SELECT count(*) FROM " + scan + " WHERE TRY_CAST(d AS SMALLINT) IS NULL");
}

TEST_CASE_METHOD(IntegerCastFixture,
                 "gpu_execution TRY_CAST to HUGEINT replays on the CPU beyond BIGINT",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_tables(*this);
  // HUGEINT runs as INT64 on the GPU.
  compare_gpu_vs_cpu(
    "SELECT id, TRY_CAST(d AS HUGEINT) AS h FROM vals_t WHERE d BETWEEN -1e6 AND 1e6");
  auto const beyond = std::string("SELECT TRY_CAST(d AS HUGEINT) AS h FROM vals_t WHERE d = 1e30");
  expect_gpu_fallback(beyond);
  run_ok("SET gpu_execution = false;");
  auto cpu_result = con->Query(beyond);
  run_ok("SET gpu_execution = true;");
  auto gpu_result = con->Query(beyond);
  REQUIRE_FALSE(cpu_result->HasError());
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->ToString() == cpu_result->ToString());
}
