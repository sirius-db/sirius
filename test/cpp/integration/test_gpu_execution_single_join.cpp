/*
 * Copyright 2025, Sirius Contributors.
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
 * @file test_gpu_execution_single_join.cpp
 * @brief Correlated scalar subqueries run on the GPU and match the CPU.
 *
 * DuckDB flattens a correlated non-aggregate scalar subquery into a SINGLE join: a LEFT join that
 * keeps at most one build row per probe row. (A correlated aggregate such as max() lowers to a
 * plain LEFT join instead, so it does not exercise SINGLE.)
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

using SingleJoinFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_single_join_tables(SingleJoinFixture& fx)
{
  // k in {0, 1, 2}; the build only has keys 0 and 1, so a third of the probe rows have no match.
  fx.run_ok("CREATE TABLE sj_a AS SELECT i::INTEGER id, (i % 3)::INTEGER k FROM range(12) r(i);");
  fx.run_ok("CREATE TABLE sj_b AS SELECT i::INTEGER k, (i * 10 + 5)::INTEGER v FROM range(2) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(SingleJoinFixture,
                 "single join - correlated scalar subquery runs on the GPU",
                 "[integration][gpu_execution][single_join]")
{
  create_single_join_tables(*this);
  SECTION("as a column, some rows unmatched")
  {
    compare_gpu_vs_cpu("SELECT a.id, (SELECT b.v FROM sj_b b WHERE b.k = a.k) v FROM sj_a a;");
  }
  SECTION("with LIMIT 1 falls back: the correlated LIMIT decorrelates through a window")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT a.id, (SELECT b.v FROM sj_b b WHERE b.k = a.k LIMIT 1) v FROM sj_a a;");
  }
  SECTION("in a predicate")
  {
    compare_gpu_vs_cpu("SELECT a.id FROM sj_a a WHERE (SELECT b.v FROM sj_b b WHERE b.k = a.k) > 10;");
  }
  SECTION("in an expression with the NULL default")
  {
    compare_gpu_vs_cpu(
      "SELECT a.id, coalesce((SELECT b.v FROM sj_b b WHERE b.k = a.k), -1) v FROM sj_a a;");
  }
}

TEST_CASE_METHOD(SingleJoinFixture,
                 "single join - NULL keys",
                 "[integration][gpu_execution][single_join]")
{
  run_ok(
    "CREATE TABLE sj_an AS SELECT i::INTEGER id, "
    "CASE WHEN i % 4 = 3 THEN NULL ELSE (i % 3)::INTEGER END k FROM range(12) r(i);");
  run_ok("CREATE TABLE sj_bn (k INTEGER, v INTEGER);");
  run_ok("INSERT INTO sj_bn VALUES (0, 10), (1, 20), (NULL, 99);");
  run_ok("CHECKPOINT;");
  SECTION("= never matches a NULL key")
  {
    compare_gpu_vs_cpu("SELECT a.id, (SELECT b.v FROM sj_bn b WHERE b.k = a.k) v FROM sj_an a;");
  }
  SECTION("IS NOT DISTINCT FROM matches NULL to NULL")
  {
    compare_gpu_vs_cpu(
      "SELECT a.id, (SELECT b.v FROM sj_bn b WHERE b.k IS NOT DISTINCT FROM a.k) v FROM sj_an a;");
  }
}
