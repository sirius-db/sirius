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
 * @file test_gpu_execution_grouping_sets_fallback.cpp
 * @brief Verifies that GPU planning rejects multiple grouping sets and GROUPING().
 *
 * ROLLUP, CUBE and GROUPING SETS must fall back to the CPU at plan time and return
 * DuckDB's rows, including the subtotal and grand-total rows. With the fallback
 * disabled they must fail at plan time instead of returning only the leaf groups.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using GroupingSetsFixture = sirius::test::GpuExecutionFixture;

namespace {

constexpr char kRejection[] = "ROLLUP, CUBE, GROUPING SETS and GROUPING() are not supported";

void create_grouping_table(GroupingSetsFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE gs_t AS SELECT (i % 3)::INTEGER a, (i % 2)::INTEGER b, i::BIGINT v "
    "FROM range(12) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - multiple grouping sets fall back and match the CPU",
                 "[integration][gpu_execution][grouping_sets_fallback]")
{
  create_grouping_table(*this);
  SECTION("ROLLUP")
  {
    expect_plan_fallback_matches_cpu("SELECT a, b, sum(v) s FROM gs_t GROUP BY ROLLUP(a, b);");
  }
  SECTION("CUBE")
  {
    expect_plan_fallback_matches_cpu("SELECT a, b, sum(v) s FROM gs_t GROUP BY CUBE(a, b);");
  }
  SECTION("GROUPING SETS without the set over all keys")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT a, b, sum(v) s FROM gs_t GROUP BY GROUPING SETS ((a), (b));");
  }
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - GROUPING() over a single grouping set falls back",
                 "[integration][gpu_execution][grouping_sets_fallback]")
{
  create_grouping_table(*this);
  expect_plan_fallback_matches_cpu("SELECT a, GROUPING(a) g, sum(v) s FROM gs_t GROUP BY a;");
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - ROLLUP is rejected at plan time when the fallback is disabled",
                 "[integration][gpu_execution][grouping_sets_fallback]")
{
  create_grouping_table(*this);
  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query("SELECT a, b, sum(v) s FROM gs_t GROUP BY ROLLUP(a, b);");
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  run_ok("SET enable_duckdb_fallback = true;");
  REQUIRE(result);
  if (!result->HasError()) {
    UNSCOPED_INFO("expected a plan rejection, got rows: " << result->ToString());
  }
  REQUIRE(result->HasError());
  REQUIRE(result->GetError().find("GPU plan generation failed") != std::string::npos);
  REQUIRE(result->GetError().find(kRejection) != std::string::npos);
  REQUIRE(after.executions == before.executions);
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - a plain GROUP BY still runs on the GPU",
                 "[integration][gpu_execution][grouping_sets_fallback]")
{
  create_grouping_table(*this);
  compare_gpu_vs_cpu("SELECT a, b, sum(v) s FROM gs_t GROUP BY a, b;");
}
