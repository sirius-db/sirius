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
 * @file test_gpu_execution_round.cpp
 * @brief GPU round() on FLOAT and DOUBLE matches DuckDB bit for bit.
 *
 * Covers ties written as decimals (2.675), values next to a tie, signed zeros, infinities, NaN,
 * NULL, values whose scaled form overflows, and precisions from -400 to 400. Results are compared
 * as DuckDB's shortest round-trip strings, so a one-ulp difference fails. DECIMAL and integer
 * inputs and a non-constant precision fall back during planning.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

using RoundFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_round_table(RoundFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE round_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f, CAST(id AS BIGINT) AS b,"
    " TRY_CAST(d AS DECIMAL(18, 4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " (2.675::DOUBLE), (-2.675), (1.005), (0.285), (0.125), (-0.125), (0.5), (-0.5), (1.5), (2.5),"
    " (0.49999999999999994), (-0.49999999999999994), (0.0), (-0.0), (123456.789), (-98765.4321),"
    " (4503599627370497.0), (1e300), (-1e300), (1.7976931348623157e308), (5e-324),"
    " ('inf'::DOUBLE), ('-inf'::DOUBLE), ('nan'::DOUBLE), (NULL)) v(d)"
    " UNION ALL SELECT 100 + i, (i * 7919 % 100003) / ((i % 977) + 3.0) FROM range(5000) t(i));");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(RoundFixture,
                 "gpu_execution round of DOUBLE and FLOAT matches DuckDB",
                 "[integration][gpu_execution][projection][round]")
{
  create_round_table(*this);
  for (auto const* column : {"d", "f"}) {
    CAPTURE(column);
    compare_gpu_vs_cpu(std::string("SELECT id, round(") + column + ") AS r FROM round_t");
    for (auto const* precision : {"0", "2", "15", "300", "400", "-1", "-300", "-400"}) {
      CAPTURE(precision);
      compare_gpu_vs_cpu(std::string("SELECT id, round(") + column + ", " + precision +
                         ") AS r FROM round_t");
    }
  }
}

TEST_CASE_METHOD(RoundFixture,
                 "gpu_execution round composes in filters, ordering and quotients",
                 "[integration][gpu_execution][projection][round]")
{
  create_round_table(*this);
  // The GPU comparison treats NaN as unordered and DuckDB as larger than any number, with or
  // without round, so the filter skips the rows that hold special values.
  compare_gpu_vs_cpu("SELECT id FROM round_t WHERE id >= 100 AND round(d, 1) > 3");
  compare_gpu_vs_cpu_ordered(
    "SELECT id, round(d, 2) AS r FROM round_t WHERE id >= 100 ORDER BY r, id LIMIT 50");
  // The TPC-DS Q2 and Q78 shape: a rounded quotient of casts.
  compare_gpu_vs_cpu(
    "SELECT id, round(CAST(id AS DOUBLE) * 1.00 / (CAST(b AS DOUBLE) + 7), 2) AS r FROM round_t");
}

TEST_CASE_METHOD(RoundFixture,
                 "gpu_execution round of a constant that DuckDB does not fold matches DuckDB",
                 "[integration][gpu_execution][projection][round]")
{
  create_round_table(*this);
  sirius::test::disabled_optimizers_guard rewriter_off(*con, "expression_rewriter");
  compare_gpu_vs_cpu("SELECT id, round(2675e-3, 2) AS r FROM round_t");
}

TEST_CASE_METHOD(RoundFixture,
                 "round of DECIMAL or integer input or with a column precision falls back at plan "
                 "time",
                 "[integration][gpu_execution][projection][round]")
{
  create_round_table(*this);
  for (auto const* expression :
       {"round(dec, 2)", "round(dec)", "round(b)", "round(d, CAST(id % 3 AS INTEGER))"}) {
    CAPTURE(expression);
    expect_plan_fallback_matches_cpu(std::string("SELECT id, ") + expression +
                                     " AS r FROM round_t");
  }
}
