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
 * @file test_gpu_execution_grouping_sets.cpp
 * @brief GPU execution of ROLLUP, CUBE, GROUPING SETS and GROUPING().
 *
 * Every query runs on the GPU and returns DuckDB's rows, including the subtotal and grand-total
 * rows, duplicate grouping sets, and the grand-total row over an empty input.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

using GroupingSetsFixture = sirius::test::GpuExecutionFixture;

namespace {

// `a` has NULLs, so a NULL key of a leaf group and a NULL of a rolled-up key both occur.
void create_grouping_table(GroupingSetsFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE gs_t AS SELECT "
    "CASE WHEN i % 7 = 0 THEN NULL ELSE (i % 3)::INTEGER END a, "
    "(i % 2)::INTEGER b, "
    "'s' || (i % 4)::VARCHAR c, "
    "i::BIGINT v, "
    "(i * 1.25)::DECIMAL(12, 2) d "
    "FROM range(1000) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - ROLLUP, CUBE and GROUPING SETS match the CPU",
                 "[integration][gpu_execution][grouping_sets]")
{
  create_grouping_table(*this);
  SECTION("ROLLUP")
  {
    compare_gpu_vs_cpu(
      "SELECT a, b, count(*) n, sum(v) s, min(c) lo, max(d) hi FROM gs_t GROUP BY ROLLUP(a, b);");
  }
  SECTION("CUBE")
  {
    compare_gpu_vs_cpu("SELECT a, b, c, count(*) n, sum(d) s FROM gs_t GROUP BY CUBE(a, b, c);");
  }
  SECTION("GROUPING SETS without the set over all keys")
  {
    compare_gpu_vs_cpu(
      "SELECT a, b, c, sum(v) s FROM gs_t GROUP BY GROUPING SETS ((a), (b), (c));");
  }
  SECTION("duplicate grouping sets")
  {
    compare_gpu_vs_cpu("SELECT a, count(*) n FROM gs_t GROUP BY GROUPING SETS ((a), (a), ());");
  }
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - GROUPING() matches the CPU",
                 "[integration][gpu_execution][grouping_sets]")
{
  create_grouping_table(*this);
  SECTION("ROLLUP")
  {
    compare_gpu_vs_cpu(
      "SELECT a, b, GROUPING(a) ga, GROUPING(b) gb, GROUPING(a, b) gab, GROUPING(b, a) gba, "
      "sum(v) s FROM gs_t GROUP BY ROLLUP(a, b);");
  }
  SECTION("single grouping set")
  {
    compare_gpu_vs_cpu("SELECT a, GROUPING(a) g, sum(v) s FROM gs_t GROUP BY a;");
  }
  SECTION("in HAVING")
  {
    compare_gpu_vs_cpu(
      "SELECT a, b, sum(v) s FROM gs_t GROUP BY CUBE(a, b) HAVING GROUPING(a, b) < 3;");
  }
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - AVG and COUNT(DISTINCT) match the CPU",
                 "[integration][gpu_execution][grouping_sets]")
{
  create_grouping_table(*this);
  SECTION("AVG")
  {
    compare_gpu_vs_cpu_approx("SELECT a, b, avg(v) av, avg(d) ad FROM gs_t GROUP BY ROLLUP(a, b);",
                              {2, 3});
  }
  SECTION("COUNT(DISTINCT)")
  {
    compare_gpu_vs_cpu("SELECT a, count(DISTINCT c) n FROM gs_t GROUP BY ROLLUP(a);");
  }
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - empty input matches the CPU",
                 "[integration][gpu_execution][grouping_sets]")
{
  create_grouping_table(*this);
  SECTION("the empty grouping set has one row")
  {
    compare_gpu_vs_cpu(
      "SELECT a, count(*) n, count(c) nc, sum(v) s, min(c) lo, avg(d) ad "
      "FROM gs_t WHERE v % 1000 = 1000 GROUP BY ROLLUP(a);");
  }
  SECTION("the empty grouping set with COUNT(DISTINCT)")
  {
    compare_gpu_vs_cpu(
      "SELECT a, count(DISTINCT c) n FROM gs_t WHERE v % 1000 = 1000 GROUP BY ROLLUP(a);");
  }
  SECTION("no empty grouping set")
  {
    compare_gpu_vs_cpu(
      "SELECT a, b, count(*) n FROM gs_t WHERE v % 1000 = 1000 "
      "GROUP BY GROUPING SETS ((a), (b));");
  }
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - grouping sets without keys fall back",
                 "[integration][gpu_execution][grouping_sets]")
{
  create_grouping_table(*this);
  expect_plan_fallback_matches_cpu("SELECT count(*) n FROM gs_t GROUP BY GROUPING SETS ((), ());");
}
