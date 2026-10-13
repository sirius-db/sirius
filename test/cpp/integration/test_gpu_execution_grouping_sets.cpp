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

#include <cmath>

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
                 "grouping sets - STDDEV_SAMP and mixed partial states match the CPU",
                 "[integration][gpu_execution][grouping_sets][stddev_samp]")
{
  create_grouping_table(*this);
  auto const aggregates = "stddev_samp(v), count(DISTINCT c), avg(d), count(v), stddev_samp(d)";
  SECTION("ROLLUP with GROUPING()")
  {
    compare_gpu_vs_cpu_approx(std::string("SELECT a, b, GROUPING(a, b), ") + aggregates +
                                " FROM gs_t GROUP BY ROLLUP(a, b);",
                              {3, 5, 7});
  }
  SECTION("CUBE")
  {
    compare_gpu_vs_cpu_approx(
      std::string("SELECT a, b, ") + aggregates + " FROM gs_t GROUP BY CUBE(a, b);", {2, 4, 6});
  }
  SECTION("duplicate grouping sets without the set over all keys")
  {
    compare_gpu_vs_cpu_approx(std::string("SELECT a, b, GROUPING(a, b), ") + aggregates +
                                " FROM gs_t GROUP BY GROUPING SETS ((a), (a), (b), ());",
                              {3, 5, 7});
  }
  SECTION("single grouping set with GROUPING()")
  {
    compare_gpu_vs_cpu_approx(
      "SELECT a, GROUPING(a), stddev_samp(v), count(DISTINCT v), avg(v), count(v) "
      "FROM gs_t GROUP BY a;",
      {2, 4});
  }
  SECTION("ordinary GROUP BY with aggregates sharing the input")
  {
    compare_gpu_vs_cpu_approx(
      "SELECT a, count(DISTINCT v), avg(v), stddev_samp(v), count(v) FROM gs_t GROUP BY a;",
      {2, 3});
  }
  SECTION("empty input with a grand total")
  {
    compare_gpu_vs_cpu(
      "SELECT a, GROUPING(a), stddev_samp(v), count(DISTINCT v), avg(v), count(v) "
      "FROM gs_t WHERE v % 1000 = 1000 GROUP BY ROLLUP(a);");
  }
  SECTION("empty input without a grand total")
  {
    compare_gpu_vs_cpu(
      "SELECT a, b, stddev_samp(v) FROM gs_t WHERE v % 1000 = 1000 "
      "GROUP BY GROUPING SETS ((a), (b));");
  }
}

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - STDDEV_SAMP NULL and large offset semantics",
                 "[integration][gpu_execution][grouping_sets][stddev_samp]")
{
  run_ok(
    "CREATE TABLE gs_sd AS SELECT * FROM (VALUES "
    "(0, 1000000000001::DOUBLE), (0, 1000000000001::DOUBLE), "
    "(0, 1000000000003::DOUBLE), (0, 1000000000003::DOUBLE), "
    "(1, NULL::DOUBLE), (2, 42::DOUBLE), (NULL, 7::DOUBLE)) t(g, v);");
  run_ok("CHECKPOINT;");
  // DuckDB's sequential Welford updates lose precision for this large-offset, low-spread
  // group. Its exact sample variance is 4/3, so check GPU accuracy independently of CPU.
  run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query(
    "SELECT stddev_samp(v), count(DISTINCT v), avg(v), count(v) FROM gs_sd GROUP BY ROLLUP(g) "
          "HAVING GROUPING(g) = 0 AND g = 0;");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  auto const after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(result->RowCount() == 1);
  REQUIRE_FALSE(result->GetValue(0, 0).IsNull());
  REQUIRE(result->GetValue(0, 0).GetValue<double>() == Approx(std::sqrt(4.0 / 3.0)).epsilon(1e-12));
  REQUIRE(result->GetValue(1, 0).GetValue<int64_t>() == 2);
  REQUIRE(result->GetValue(2, 0).GetValue<double>() == 1000000000002.0);
  REQUIRE(result->GetValue(3, 0).GetValue<int64_t>() == 4);

  // Compare the remaining leaf groups and grand total against DuckDB, including the
  // distinction between a genuine NULL key and the rolled-up NULL key.
  compare_gpu_vs_cpu_approx(
    "SELECT g, GROUPING(g), stddev_samp(v), count(DISTINCT v), avg(v), count(v) "
    "FROM gs_sd GROUP BY ROLLUP(g) HAVING GROUPING(g) = 1 OR g IS NULL OR g <> 0;",
    {2, 4});
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

TEST_CASE_METHOD(GroupingSetsFixture,
                 "grouping sets - aggregates with a FILTER clause fall back",
                 "[integration][gpu_execution][grouping_sets][aggregate]")
{
  create_grouping_table(*this);
  SECTION("ROLLUP")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT a, b, sum(v) FILTER (WHERE v > 10) s FROM gs_t GROUP BY ROLLUP(a, b);");
  }
  SECTION("GROUP BY")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT a, sum(v) FILTER (WHERE v > 10) s, count(*) n FROM gs_t GROUP BY a;");
  }
  SECTION("ungrouped")
  {
    expect_plan_fallback_matches_cpu("SELECT count(*) FILTER (WHERE v > 10) n FROM gs_t;");
  }
}
