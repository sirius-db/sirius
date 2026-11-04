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

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using ExpressionFallbackFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_expression_table(ExpressionFallbackFixture& fx)
{
  fx.run_ok("CREATE TABLE expr_t(s VARCHAR, d DATE, ts TIMESTAMP, unit VARCHAR);");
  fx.run_ok(
    "INSERT INTO expr_t VALUES"
    " ('42', DATE '2024-06-15', TIMESTAMP '2024-06-15 12:34:56.123456', 'day'),"
    " ('invalid', DATE '2023-01-02', TIMESTAMP '2023-01-02 01:02:03.456789', 'week'),"
    " (NULL, NULL, NULL, NULL);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(ExpressionFallbackFixture,
                 "TRY falls back at plan time in projections and filters",
                 "[integration][gpu_execution][expression_plan_fallback]")
{
  create_expression_table(*this);
  for (auto const* query : {"SELECT TRY(s) FROM expr_t;",
                            "SELECT COALESCE(TRY(CAST(s AS INTEGER)), -1) FROM expr_t;",
                            "SELECT s FROM expr_t WHERE TRY(s IS NOT NULL);",
                            "SELECT s FROM expr_t WHERE TRY(CAST(s AS INTEGER) > 0);"}) {
    CAPTURE(query);
    expect_plan_fallback_matches_cpu(query);
  }
}

TEST_CASE_METHOD(ExpressionFallbackFixture,
                 "unsupported date_trunc frequencies fall back at plan time",
                 "[integration][gpu_execution][expression_plan_fallback]")
{
  create_expression_table(*this);
  for (auto const* unit : {"year", "month", "week", "quarter"}) {
    CAPTURE(unit);
    for (auto const* column : {"d", "ts"}) {
      CAPTURE(column);
      auto const expr = std::string("date_trunc('") + unit + "', " + column + ")";
      expect_plan_fallback_matches_cpu("SELECT " + expr + " FROM expr_t;");
      expect_plan_fallback_matches_cpu("SELECT s FROM expr_t WHERE " + expr + " IS NOT NULL;");
    }
  }
  expect_plan_fallback_matches_cpu("SELECT date_trunc(unit, ts) FROM expr_t;");
}

TEST_CASE_METHOD(ExpressionFallbackFixture,
                 "unsupported expressions fail during planning when fallback is disabled",
                 "[integration][gpu_execution][expression_plan_fallback]")
{
  create_expression_table(*this);
  run_ok("SET enable_duckdb_fallback = false;");
  for (auto const* query : {"SELECT TRY(s) FROM expr_t;",
                            "SELECT s FROM expr_t WHERE TRY(s IS NOT NULL);",
                            "SELECT date_trunc('week', ts) FROM expr_t;"}) {
    CAPTURE(query);
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    REQUIRE(result->HasError());
    REQUIRE(result->GetError().find("GPU plan generation failed") != std::string::npos);
    REQUIRE(result->GetError().find("Unsupported") != std::string::npos);
    REQUIRE(after.executions == before.executions);
    REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks);
  }
  run_ok("SET enable_duckdb_fallback = true;");
}

TEST_CASE_METHOD(ExpressionFallbackFixture,
                 "supported date_trunc frequencies still execute on the GPU",
                 "[integration][gpu_execution][expression_plan_fallback]")
{
  create_expression_table(*this);
  for (auto const* unit : {"day", "hour", "minute", "second", "millisecond", "microsecond"}) {
    CAPTURE(unit);
    compare_gpu_vs_cpu(std::string("SELECT date_trunc('") + unit + "', ts) FROM expr_t;");
    compare_gpu_vs_cpu(std::string("SELECT date_trunc('") + unit + "', d) FROM expr_t;");
  }
}

TEST_CASE_METHOD(ExpressionFallbackFixture,
                 "date_trunc on DATE gives parent expressions its declared TIMESTAMP type",
                 "[integration][gpu_execution][expression_plan_fallback]")
{
  run_ok("CREATE TABLE trunc_t(k INTEGER, d DATE, ts TIMESTAMP, ts_ms TIMESTAMP_MS);");
  run_ok(
    "INSERT INTO trunc_t VALUES"
    " (1, DATE '2024-06-15', TIMESTAMP '2024-06-15 12:34:56.123456',"
    "  TIMESTAMP_MS '2024-06-16 01:02:03.456'),"
    " (2, DATE '1969-12-31', TIMESTAMP '1969-12-31 23:59:59.999999',"
    "  TIMESTAMP_MS '1970-01-01 00:00:00.001'),"
    " (3, DATE 'infinity', TIMESTAMP '2000-01-01 00:00:00', NULL),"
    " (4, DATE '-infinity', NULL, TIMESTAMP_MS '1991-09-07 09:07:46.337'),"
    " (5, NULL, TIMESTAMP '2020-01-01 12:30:45', TIMESTAMP_MS '2020-01-01 12:30:45.5');");
  run_ok("CHECKPOINT;");
  for (auto const* unit : {"day", "hour", "minute", "second", "millisecond", "microsecond"}) {
    CAPTURE(unit);
    auto const trunc = std::string("date_trunc('") + unit + "', d)";
    compare_gpu_vs_cpu("SELECT k, CASE WHEN k % 2 = 0 THEN " + trunc +
                       " ELSE ts END FROM trunc_t;");
    compare_gpu_vs_cpu("SELECT k, CASE WHEN k > 2 THEN " + trunc + " ELSE ts_ms END FROM trunc_t;");
    compare_gpu_vs_cpu("SELECT k, coalesce(" + trunc + ", ts) FROM trunc_t;");
    compare_gpu_vs_cpu("SELECT k FROM trunc_t WHERE " + trunc + " <= ts;");
  }
  // Fuzz finding 022-gpu_internal_error-e13f6ef8: a DATE THEN beside TIMESTAMP_MS branches.
  compare_gpu_vs_cpu(
    "SELECT k, CASE WHEN ts IS NULL THEN date_trunc('day', d) WHEN k = 1 THEN ts_ms"
    " ELSE CAST(TIMESTAMP '1991-09-07 09:07:46.337000' AS TIMESTAMP_MS) END FROM trunc_t;");
}
