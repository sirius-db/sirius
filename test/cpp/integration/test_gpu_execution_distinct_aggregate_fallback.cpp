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
 * @file test_gpu_execution_distinct_aggregate_fallback.cpp
 * @brief Verifies that GPU planning rejects the DISTINCT aggregates the GPU does not support.
 *
 * The GPU aggregates support DISTINCT only in COUNT, and ungrouped COUNT(DISTINCT) only on a
 * single column, without a FILTER clause. The other DISTINCT aggregates must fall back to the CPU
 * at plan time and return DuckDB's rows. With the fallback disabled they must fail at plan time.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using DistinctAggregateFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Every value of `v` and `s` repeats, so DISTINCT changes SUM, AVG and COUNT.
void create_distinct_table(DistinctAggregateFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE da_t AS SELECT (i % 3)::INTEGER g, (i % 5)::INTEGER v, (i * 7)::BIGINT w, "
    "'s' || (i % 4) s FROM range(60) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(
  DistinctAggregateFixture,
  "distinct aggregates - unsupported grouped DISTINCT aggregates fall back and match the CPU",
  "[integration][gpu_execution][aggregate][distinct_aggregate_fallback]")
{
  create_distinct_table(*this);
  SECTION("SUM(DISTINCT)")
  {
    expect_plan_fallback_matches_cpu("SELECT g, sum(DISTINCT v) s FROM da_t GROUP BY g;");
  }
  SECTION("COUNT(DISTINCT) with a FILTER clause")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT g, count(DISTINCT v) FILTER (WHERE v < 2) c FROM da_t GROUP BY g;");
  }
  SECTION("SUM(DISTINCT) and AVG(DISTINCT) next to COUNT(DISTINCT)")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT g, count(DISTINCT v) c, sum(DISTINCT v) s, avg(DISTINCT v) a FROM da_t GROUP BY g;");
  }
}

TEST_CASE_METHOD(
  DistinctAggregateFixture,
  "distinct aggregates - unsupported ungrouped DISTINCT aggregates fall back and match the CPU",
  "[integration][gpu_execution][aggregate][distinct_aggregate_fallback]")
{
  create_distinct_table(*this);
  SECTION("SUM and AVG with DISTINCT")
  {
    expect_plan_fallback_matches_cpu("SELECT sum(DISTINCT v) s, avg(DISTINCT v) a FROM da_t;");
  }
  SECTION("multi-column COUNT(DISTINCT)")
  {
    expect_plan_fallback_matches_cpu("SELECT count(DISTINCT (g, v)) c FROM da_t;");
  }
  SECTION("COUNT(DISTINCT) with a FILTER clause")
  {
    expect_plan_fallback_matches_cpu("SELECT count(DISTINCT v) FILTER (WHERE v < 2) c FROM da_t;");
  }
}

TEST_CASE_METHOD(DistinctAggregateFixture,
                 "distinct aggregates - refused at plan time when the fallback is disabled",
                 "[integration][gpu_execution][aggregate][distinct_aggregate_fallback]")
{
  create_distinct_table(*this);
  auto const expect_plan_rejection = [this](std::string const& query, std::string const& detail) {
    run_ok("SET gpu_execution = true;");
    run_ok("SET enable_duckdb_fallback = false;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    run_ok("SET enable_duckdb_fallback = true;");
    REQUIRE(result);
    if (!result->HasError()) {
      UNSCOPED_INFO("expected a plan rejection, got rows: " << result->ToString());
    }
    REQUIRE(result->HasError());
    REQUIRE(result->GetError().find("GPU plan generation failed") != std::string::npos);
    REQUIRE(result->GetError().find(detail) != std::string::npos);
    REQUIRE(after.executions == before.executions);
  };
  SECTION("grouped SUM(DISTINCT)")
  {
    expect_plan_rejection("SELECT g, sum(DISTINCT v) s FROM da_t GROUP BY g;",
                          "DISTINCT in grouped aggregates other than COUNT");
  }
  SECTION("ungrouped SUM(DISTINCT)")
  {
    expect_plan_rejection("SELECT sum(DISTINCT v) s FROM da_t;",
                          "DISTINCT in ungrouped aggregates other than COUNT");
  }
}

TEST_CASE_METHOD(DistinctAggregateFixture,
                 "distinct aggregates - grouped COUNT(DISTINCT) still runs on the GPU",
                 "[integration][gpu_execution][aggregate][distinct_aggregate_fallback]")
{
  create_distinct_table(*this);
  compare_gpu_vs_cpu("SELECT g, count(DISTINCT v) c, sum(w) s FROM da_t GROUP BY g;");
}

TEST_CASE_METHOD(DistinctAggregateFixture,
                 "distinct aggregates - ungrouped COUNT(DISTINCT) runs on the GPU",
                 "[integration][gpu_execution][aggregate][distinct_aggregate_fallback]")
{
  create_distinct_table(*this);
  SECTION("next to plain aggregates")
  {
    // The shape of TPC-DS Q16, Q94 and Q95.
    compare_gpu_vs_cpu("SELECT count(DISTINCT v) c, sum(w) s, count(*) n FROM da_t;");
  }
  SECTION("on several columns and types")
  {
    compare_gpu_vs_cpu(
      "SELECT count(DISTINCT v), count(DISTINCT w % 4), count(DISTINCT (v * 1.5)::DOUBLE), "
      "count(DISTINCT (v::DECIMAL(7, 2))), count(DISTINCT s) FROM da_t;");
  }
  SECTION("on an empty input")
  {
    compare_gpu_vs_cpu("SELECT count(DISTINCT v) c, count(*) n FROM da_t WHERE w < 0;");
  }
}

TEST_CASE_METHOD(DistinctAggregateFixture,
                 "distinct aggregates - ungrouped COUNT(DISTINCT) merges sets across batches",
                 "[integration][gpu_execution][aggregate][distinct_aggregate_fallback]")
{
  // Values repeat across batches, so the merge must union the sets rather than add the counts.
  run_ok(
    "CREATE TABLE da_big AS SELECT (i % 1000)::INTEGER v, CASE WHEN i % 7 = 0 THEN NULL ELSE "
    "(i % 333)::BIGINT END n FROM range(300000) r(i);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 65536};
  compare_gpu_vs_cpu("SELECT count(DISTINCT v) c, count(DISTINCT n) d, count(*) r FROM da_big;");
}
