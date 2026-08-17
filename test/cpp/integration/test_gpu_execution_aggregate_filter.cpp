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

// GPU-vs-CPU correctness for the aggregate FILTER (WHERE ...) clause. The planner rewrites each
// filtered aggregate input to CASE WHEN filter THEN input ELSE NULL END below the aggregate and
// lowers count(*) FILTER to count(mask), so these cases prove the mask-to-null lowering on the
// real GPU path: every compare_gpu_vs_cpu query runs once on the GPU with a no-fallback
// assertion and once on DuckDB CPU, then compares results. Rejected shapes (DISTINCT / first
// with FILTER) are asserted to take the plan-time transparent CPU fallback instead.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

namespace {

// `x` is dense (each residue 0-99 appears 10 times), `xn` adds NULLs among passing rows,
// `g` is the group key and contains a NULL group, `d`/`s` cover the DECIMAL and VARCHAR
// aggregate paths.
class AggFilterFixture : public sirius::test::GpuExecutionFixture {
 public:
  AggFilterFixture()
  {
    run_ok(
      "CREATE TABLE agg_f AS SELECT"
      "  CAST(range % 100 AS INTEGER)             AS x,"
      "  CAST(NULLIF(range % 100, 50) AS INTEGER) AS xn,"
      "  CAST(NULLIF(range % 4, 3) AS INTEGER)    AS g,"
      "  CAST(range % 100 AS DECIMAL(10,2))       AS d,"
      "  concat('s', range % 100)                 AS s "
      "FROM range(1000);");
    run_ok("CREATE TABLE marks(y INTEGER);");
    run_ok("INSERT INTO marks VALUES (1),(2),(NULL);");
    run_ok("CHECKPOINT;");
  }

  /// Rejected FILTER shapes decline at plan time: assert exactly one transparent fallback
  /// (no rebind, no GPU execution) and that the fallback result matches a plain CPU run.
  void expect_plan_fallback_and_compare(const std::string& query)
  {
    con->Query("SET gpu_execution = true;");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query(query);
    auto after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);

    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    auto rows     = collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
    auto cpu_rows = collect_rows(cpu_result->Cast<duckdb::MaterializedQueryResult>());
    REQUIRE(rows == cpu_rows);
  }
};

}  // namespace

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution ungrouped FILTER on all supported aggregates",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // Every supported function with the same filter, plus unfiltered aggregates in the same
  // query to prove the mask stays local to the filtered inputs.
  compare_gpu_vs_cpu(
    "SELECT count(*) FILTER (WHERE x < 50), count(xn) FILTER (WHERE x < 50),"
    "       sum(x) FILTER (WHERE x < 50), min(x) FILTER (WHERE x < 50),"
    "       max(x) FILTER (WHERE x < 50), avg(x) FILTER (WHERE x < 50),"
    "       count(*), sum(x) "
    "FROM agg_f");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution grouped FILTER on all supported aggregates with a NULL group key",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  compare_gpu_vs_cpu(
    "SELECT g, count(*) FILTER (WHERE x < 50), count(xn) FILTER (WHERE x < 50),"
    "       sum(x) FILTER (WHERE x < 50), min(x) FILTER (WHERE x < 50),"
    "       max(x) FILTER (WHERE x < 50), avg(x) FILTER (WHERE x < 50),"
    "       count(*) "
    "FROM agg_f GROUP BY g");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution FILTER with a NULL-containing predicate masks NULL out",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // xn is NULL where x = 50, so `xn % 2 = 0` is three-valued: TRUE passes, FALSE and NULL
  // must both mask out.
  compare_gpu_vs_cpu(
    "SELECT count(*) FILTER (WHERE xn % 2 = 0), sum(x) FILTER (WHERE xn % 2 = 0) FROM agg_f");
  compare_gpu_vs_cpu(
    "SELECT g, count(*) FILTER (WHERE xn % 2 = 0), sum(x) FILTER (WHERE xn % 2 = 0) "
    "FROM agg_f GROUP BY g");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution FILTER with a mark-join subquery predicate",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // The NULL in marks makes non-matching IN results NULL (mark-join three-valued output).
  // Only the single-subquery shape is covered: two IN subqueries in one query plan two MARK
  // joins, which aborts on a pre-existing hash-join scheduling bug ("MARK join must run in
  // BUILD_PROBE mode" in refresh_cross_schedule) unrelated to the FILTER lowering.
  compare_gpu_vs_cpu("SELECT count(*) FILTER (WHERE x IN (SELECT y FROM marks)) FROM agg_f");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution FILTER with an empty qualifying set",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // The predicates below are false on every row but not provably false from column statistics,
  // so the mask column really evaluates on the GPU (a stats-foldable predicate like x < 0 takes
  // the scalar-mask runtime fallback instead -- covered by the next case).
  // Ungrouped: xn is NULL exactly where x = 50, so no row passes -> count 0, sum NULL.
  compare_gpu_vs_cpu(
    "SELECT count(*) FILTER (WHERE xn IS NULL AND x < 50),"
    "       sum(x) FILTER (WHERE xn IS NULL AND x < 50) FROM agg_f");
  // Grouped: x < 1 passes only on multiples of 100, which all land in group 0, so groups
  // 1, 2, and NULL have zero passing rows and must still emit 0 / NULL.
  compare_gpu_vs_cpu(
    "SELECT g, count(*) FILTER (WHERE x < 1), sum(x) FILTER (WHERE x < 1) "
    "FROM agg_f GROUP BY g");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution constant-folded FILTER takes the runtime CPU fallback",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // Statistics propagation folds x < 0 to a constant FALSE, so the mask CASE evaluates to a
  // scalar the GPU CASE lowering cannot consume; execution throws and the retained CPU plan
  // runs (fail-safe, never silent). Pin that behavior so a future change to either side (mask
  // lowering or scalar CASE support) surfaces here.
  expect_gpu_fallback(
    "SELECT count(*) FILTER (WHERE x < 0), sum(x) FILTER (WHERE x < 0) FROM agg_f");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution FILTER on VARCHAR and DECIMAL aggregate inputs",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  compare_gpu_vs_cpu(
    "SELECT min(s) FILTER (WHERE x < 50), max(s) FILTER (WHERE x < 50) FROM agg_f");
  compare_gpu_vs_cpu(
    "SELECT sum(d) FILTER (WHERE x < 50), avg(d) FILTER (WHERE x < 50) FROM agg_f");
  compare_gpu_vs_cpu(
    "SELECT g, sum(d) FILTER (WHERE x < 50), avg(d) FILTER (WHERE x < 50) "
    "FROM agg_f GROUP BY g");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution rejected FILTER shapes take the plan-time CPU fallback",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  expect_plan_fallback_and_compare("SELECT count(DISTINCT x) FILTER (WHERE x < 50) FROM agg_f");
  expect_plan_fallback_and_compare("SELECT first(x) FILTER (WHERE x = 42) FROM agg_f");
}
