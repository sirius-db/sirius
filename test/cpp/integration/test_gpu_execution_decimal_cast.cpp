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
 * @file test_gpu_execution_decimal_cast.cpp
 * @brief GPU CAST from DOUBLE, FLOAT and DECIMAL to DECIMAL matches DuckDB.
 *
 * DuckDB rounds half away from zero when a cast drops digits; cudf::cast truncates. 1 - 0.07 in
 * DOUBLE is 0.9299999999999999, which DuckDB casts to 0.93 in DECIMAL(16,2). Covers values just
 * below a tie, ties that are exact in binary (0.125), signed values, values at the precision limit,
 * NaN, infinities and NULL, for 32-, 64- and 128-bit DECIMAL targets. Values that do not fit fail
 * CAST and give NULL from TRY_CAST.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using DecimalCastFixture = sirius::test::GpuExecutionFixture;

namespace {

/// DECIMAL targets with at least 7 integer digits, so every value of `fits_t` fits. DECIMAL with a
/// precision of 4 or less has no cuDF type and is not tested here.
constexpr char const* kTargets[] = {"DECIMAL(9,2)",
                                    "DECIMAL(9,0)",
                                    "DECIMAL(12,3)",
                                    "DECIMAL(16,2)",
                                    "DECIMAL(18,0)",
                                    "DECIMAL(18,6)",
                                    "DECIMAL(38,2)",
                                    "DECIMAL(38,10)"};

/// `fits_t` holds values below 10^7 in magnitude, also as FLOAT, and `limits_t` values at and
/// beyond the precision limits plus NaN and infinities. Both hold DOUBLE `d`, FLOAT `f` and
/// DECIMAL(18,4) `dec`. The scan does not decode DECIMAL(38,x) storage, so 128-bit sources are cast
/// in the query.
void create_tables(DecimalCastFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE fits_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f,"
    " TRY_CAST(d AS DECIMAL(18,4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " (1::DOUBLE - 0.07::DOUBLE), (1::DOUBLE - 0.29::DOUBLE), ('0.9299999999999999'::DOUBLE),"
    " ('1.005'::DOUBLE), ('-1.005'::DOUBLE), ('2.675'::DOUBLE), ('-2.675'::DOUBLE),"
    " ('0.125'::DOUBLE), ('-0.125'::DOUBLE), ('0.5'::DOUBLE), ('-0.5'::DOUBLE), ('1.5'::DOUBLE),"
    " ('2.5'::DOUBLE), ('-2.5'::DOUBLE), ('0.005'::DOUBLE), ('-0.005'::DOUBLE),"
    " ('0.00499999999999'::DOUBLE), ('0.49999999999999994'::DOUBLE), ('0.0'::DOUBLE),"
    " ('-0.0'::DOUBLE), ('5e-324'::DOUBLE), ('1234567.125'::DOUBLE), ('-1234567.125'::DOUBLE),"
    " ('9999999.49'::DOUBLE), ('-9999999.49'::DOUBLE), ('9999999.4949'::DOUBLE), (NULL)) v(d)"
    " UNION ALL SELECT 100 + i, (i * 7919 % 100003) / ((i % 977) + 3.0) - 1000 FROM range(3000) "
    "t(i)"
    " UNION ALL SELECT 4000 + i, 1::DOUBLE - (i % 11) / 100.0 FROM range(1000) t(i));");
  fx.run_ok(
    "CREATE TABLE limits_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f,"
    " TRY_CAST(d AS DECIMAL(18,4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " ('9999999.994999'::DOUBLE), ('9999999.996'::DOUBLE), ('-9999999.996'::DOUBLE),"
    " ('99999999.995'::DOUBLE), ('999999999.4'::DOUBLE), ('999999999.5'::DOUBLE),"
    " ('99999999999.9995'::DOUBLE), ('99999999999999.99'::DOUBLE), ('1e16'::DOUBLE),"
    " ('999999999999999872'::DOUBLE), ('1e18'::DOUBLE), ('-1e18'::DOUBLE), ('1e27'::DOUBLE),"
    " ('9.999999999999999e27'::DOUBLE), ('1e36'::DOUBLE), ('9.99999999999999e37'::DOUBLE),"
    " ('1e38'::DOUBLE), ('-1e38'::DOUBLE), ('1.7976931348623157e308'::DOUBLE),"
    " ('inf'::DOUBLE), ('-inf'::DOUBLE), ('nan'::DOUBLE), ('0.125'::DOUBLE), (NULL),"
    " ('9999999.99'::DOUBLE), ('9999999.995'::DOUBLE)) v(d));");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(DecimalCastFixture,
                 "gpu_execution CAST of DOUBLE and FLOAT to DECIMAL rounds like DuckDB",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_tables(*this);
  for (auto const* column : {"d", "f"}) {
    for (auto const* target : kTargets) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, CAST(") + column + " AS " + target +
                         ") AS c FROM fits_t");
    }
  }
}

TEST_CASE_METHOD(DecimalCastFixture,
                 "gpu_execution CAST of DECIMAL to a smaller scale rounds like DuckDB",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_tables(*this);
  for (auto const* column :
       {"dec", "CAST(dec AS DECIMAL(38,4))", "TRY_CAST(dec AS DECIMAL(9,4))"}) {
    for (auto const* target : kTargets) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, CAST(") + column + " AS " + target +
                         ") AS c FROM fits_t");
    }
  }
}

TEST_CASE_METHOD(DecimalCastFixture,
                 "gpu_execution TRY_CAST to DECIMAL gives NULL where the value does not fit",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_tables(*this);
  for (auto const* column : {"d", "f", "dec", "TRY_CAST(d AS DECIMAL(38,4))"}) {
    for (auto const* target : kTargets) {
      CAPTURE(column, target);
      compare_gpu_vs_cpu(std::string("SELECT id, TRY_CAST(") + column + " AS " + target +
                         ") AS c FROM limits_t");
    }
  }
}

TEST_CASE_METHOD(DecimalCastFixture,
                 "gpu_execution CAST to DECIMAL fails like DuckDB where the value does not fit",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_tables(*this);
  // Each query holds one value that fails the cast. The GPU must not return a wrapped value: it
  // fails, the query replays on the CPU and raises DuckDB's conversion error.
  for (auto const* query :
       {"SELECT CAST(d AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)",
        "SELECT CAST(d AS DECIMAL(18,0)) FROM limits_t WHERE id IN (10, 11)",
        "SELECT CAST(d AS DECIMAL(38,0)) FROM limits_t WHERE id IN (16, 17)",
        "SELECT CAST(f AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)",
        "SELECT CAST(d AS DECIMAL(16,2)) FROM limits_t WHERE id IN (20, 23, 24)",
        "SELECT CAST(d AS DECIMAL(16,2)) FROM limits_t WHERE id IN (22, 23, 24)",
        "SELECT CAST(dec AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)",
        "SELECT CAST(dec AS DECIMAL(12,0)) FROM limits_t WHERE id IN (6, 8)",
        "SELECT CAST(dec AS DECIMAL(12,6)) FROM limits_t WHERE id IN (5, 23)",
        "SELECT CAST(CAST(d AS DECIMAL(38,4)) AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)"}) {
    CAPTURE(query);
    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    REQUIRE(cpu_result);
    REQUIRE(cpu_result->HasError());

    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu_result   = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(gpu_result);
    if (!gpu_result->HasError()) { UNSCOPED_INFO("GPU returned: " << gpu_result->ToString()); }
    REQUIRE(gpu_result->HasError());
    REQUIRE(gpu_result->GetError() == cpu_result->GetError());
    REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  }
}

TEST_CASE_METHOD(DecimalCastFixture,
                 "gpu_execution TPC-H revenue factor CAST(1 - discount AS DECIMAL) matches DuckDB",
                 "[integration][gpu_execution][aggregate][decimal_cast]")
{
  // The StarRocks plans evaluate 1 - l_discount in DOUBLE and cast it back to DECIMAL(16,2).
  run_ok(
    "CREATE TABLE lineitem_t AS SELECT (i % 11 / 100)::DECIMAL(15,2) AS l_discount,"
    " (900 + i % 9000 / 7)::DECIMAL(15,2) AS l_extendedprice FROM range(20000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(
    "SELECT CAST((1::DOUBLE - l_discount::DOUBLE) AS DECIMAL(16,2)) AS c, count(*) AS n"
    " FROM lineitem_t GROUP BY 1");
  compare_gpu_vs_cpu(
    "SELECT l_discount, sum(l_extendedprice * CAST((1::DOUBLE - l_discount::DOUBLE) AS"
    " DECIMAL(16,2))) AS revenue FROM lineitem_t GROUP BY 1");
}
