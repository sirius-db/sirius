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
// assertion and once on DuckDB CPU, then compares results. Rejected shapes (DISTINCT other than
// COUNT(DISTINCT), and first, with FILTER) are asserted to take the plan-time transparent CPU
// fallback instead.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

namespace {

// `x` is dense (each residue 0-99 appears 10 times), `xn` is NULL exactly where x = 50,
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
    run_ok("CREATE TABLE agg_nulls(g INTEGER, v INTEGER, keep BOOLEAN);");
    run_ok(
      "INSERT INTO agg_nulls VALUES (1,1,true),(1,NULL,true),(1,2,true),"
      "(1,100,false),(1,200,NULL),(2,NULL,true),(2,300,false),"
      "(3,400,false),(3,500,NULL);");
    run_ok("CREATE TABLE agg_empty(g INTEGER, v INTEGER, keep BOOLEAN);");
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

    auto rows     = collect_rows(*result);
    auto cpu_rows = collect_rows(*cpu_result);
    REQUIRE(rows == cpu_rows);
  }
};

}  // namespace

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution ungrouped FILTER on all supported aggregates",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // The x=50 rows pass with NULL xn, so count(*) includes them and count(xn) skips them. Unfiltered
  // aggregates prove the mask stays local to the filtered inputs.
  compare_gpu_vs_cpu(
    "SELECT count(*) FILTER (WHERE x <= 50), count(xn) FILTER (WHERE x <= 50),"
    "       sum(x) FILTER (WHERE x <= 50), min(x) FILTER (WHERE x <= 50),"
    "       max(x) FILTER (WHERE x <= 50), avg(x) FILTER (WHERE x <= 50),"
    "       count(*), sum(x) "
    "FROM agg_f");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution grouped FILTER on all supported aggregates with a NULL group key",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  compare_gpu_vs_cpu(
    "SELECT g, count(*) FILTER (WHERE x <= 50), count(xn) FILTER (WHERE x <= 50),"
    "       sum(x) FILTER (WHERE x <= 50), min(x) FILTER (WHERE x <= 50),"
    "       max(x) FILTER (WHERE x <= 50), avg(x) FILTER (WHERE x <= 50),"
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
  // scalar-mask path instead -- covered by the next case).
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
                 "gpu_execution constant-folded FILTER runs on GPU",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // Statistics propagation folds x < 0 to FALSE; CASE broadcasts scalar masks on GPU.
  compare_gpu_vs_cpu(
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
  expect_plan_fallback_and_compare("SELECT sum(DISTINCT x) FILTER (WHERE x < 50) FROM agg_f");
  expect_plan_fallback_and_compare("SELECT first(x) FILTER (WHERE x = 42) FROM agg_f");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution COUNT(DISTINCT) FILTER on several columns and types",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // Every value repeats, so DISTINCT changes each count. xn % 2 = 0 is NULL where x = 50, which
  // must mask out. The unfiltered COUNT(DISTINCT x) proves the mask stays local.
  for (bool grouped : {false, true}) {
    compare_gpu_vs_cpu(std::string("SELECT ") + (grouped ? "g, " : "") +
                       "count(DISTINCT x) FILTER (WHERE x < 50), "
                       "count(DISTINCT xn) FILTER (WHERE x <= 50), "
                       "count(DISTINCT s) FILTER (WHERE x % 3 = 0), "
                       "count(DISTINCT d) FILTER (WHERE xn % 2 = 0), count(DISTINCT x) "
                       "FROM agg_f" +
                       (grouped ? " GROUP BY g" : ""));
  }
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution COUNT(DISTINCT) FILTER is 0 when no non-NULL value passes",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // In agg_nulls group 2 passes only a NULL value and group 3 passes no row. The distinct count
  // shares its masked input with the other aggregates filtered on keep.
  for (bool grouped : {false, true}) {
    for (auto const* table : {"agg_nulls", "agg_empty"}) {
      compare_gpu_vs_cpu(std::string("SELECT ") + (grouped ? "g, " : "") +
                         "count(DISTINCT v) FILTER (WHERE keep), count(v) FILTER (WHERE keep), "
                         "sum(v) FILTER (WHERE keep), count(DISTINCT v) FILTER (WHERE NOT keep) "
                         "FROM " +
                         table + (grouped ? " GROUP BY g" : ""));
    }
  }
  for (auto group : {2, 3}) {
    compare_gpu_vs_cpu("SELECT count(DISTINCT v) FILTER (WHERE keep) FROM agg_nulls WHERE g = " +
                       std::to_string(group));
  }
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution COUNT(DISTINCT) FILTER with a NOT IN subquery (TPC-Havoc Q16_V07)",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // HAVING repeats the filtered aggregate, so the plan has two MARK joins.
  compare_gpu_vs_cpu(
    "SELECT g, count(DISTINCT x) FILTER (WHERE x NOT IN (SELECT y FROM marks WHERE y IS NOT NULL)) "
    "FROM agg_f GROUP BY g "
    "HAVING count(DISTINCT x) FILTER (WHERE x NOT IN (SELECT y FROM marks WHERE y IS NOT NULL)) "
    "> 24");
  // The NULL in marks makes NOT IN FALSE or NULL on every row, so every count is 0.
  compare_gpu_vs_cpu(
    "SELECT g, count(DISTINCT x) FILTER (WHERE x NOT IN (SELECT y FROM marks)) FROM agg_f "
    "GROUP BY g");
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution COUNT(DISTINCT) FILTER merges sets across batches",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  // Each (g, v) pair repeats 150000 rows later, in another batch, so the merge must union the
  // sets. Only the first rows pass i < 100000, so later batches contribute empty or NULL sets.
  run_ok(
    "CREATE TABLE agg_big AS SELECT i, (i % 150000)::INTEGER v, (i % 150000 % 7)::INTEGER g "
    "FROM range(300000) r(i);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 65536};
  for (bool grouped : {false, true}) {
    compare_gpu_vs_cpu(std::string("SELECT ") + (grouped ? "g, " : "") +
                       "count(DISTINCT v) FILTER (WHERE i < 100000), "
                       "count(DISTINCT v) FILTER (WHERE v % 3 = 0), count(DISTINCT v) "
                       "FROM agg_big" +
                       (grouped ? " GROUP BY g" : ""));
  }
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution shared FILTER inputs preserve passing NULLs and empty groups",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  for (bool grouped : {false, true}) {
    compare_gpu_vs_cpu(std::string("SELECT ") + (grouped ? "g, " : "") +
                       "sum(v) FILTER (WHERE keep), min(v) FILTER (WHERE keep), "
                       "max(v) FILTER (WHERE keep), avg(v) FILTER (WHERE keep), "
                       "count(v) FILTER (WHERE keep), count(*) FILTER (WHERE keep), "
                       "count(v) FILTER (WHERE NOT keep), sum(v) FILTER (WHERE NOT keep) "
                       "FROM agg_nulls" +
                       (grouped ? " GROUP BY g" : ""));
  }
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution shared FILTER predicates preserve NULL inputs and empty groups",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  auto current = con->Query("SELECT current_setting('disabled_optimizers')");
  REQUIRE(current);
  REQUIRE_FALSE(current->HasError());
  auto disabled = current->Collection().GetValue(0, 0).ToString();
  if (!disabled.empty()) { disabled += ","; }
  disabled += "common_subexpressions";
  // Exercise Sirius predicate sharing even when DuckDB has not already projected the predicate.
  sirius::test::scoped_sirius_setting guard(
    *con, "disabled_optimizers", std::string_view(disabled));

  for (bool grouped : {false, true}) {
    for (auto const* table : {"agg_nulls", "agg_empty"}) {
      compare_gpu_vs_cpu(
        std::string("SELECT ") + (grouped ? "g, " : "") +
        "sum(v) FILTER (WHERE keep OR v < 0), min(g) FILTER (WHERE keep OR v < 0), "
        "count(v) FILTER (WHERE keep OR v < 0), count(*) FILTER (WHERE keep OR v < 0), "
        "sum(v) FILTER (WHERE NOT keep), count(*) FILTER (WHERE NOT keep), count(*), sum(v) "
        "FROM " +
        table + (grouped ? " GROUP BY g" : ""));
    }
    compare_gpu_vs_cpu(std::string("SELECT ") + (grouped ? "g, " : "") +
                       "sum(d) FILTER (WHERE x <= 50), min(s) FILTER (WHERE x <= 50), "
                       "max(xn) FILTER (WHERE x <= 50), count(xn) FILTER (WHERE x <= 50), "
                       "count(*) FILTER (WHERE x <= 50), sum(x) FROM agg_f" +
                       (grouped ? " GROUP BY g" : ""));
  }
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution FILTER over a truly empty on-disk table",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  for (bool grouped : {false, true}) {
    compare_gpu_vs_cpu(std::string("SELECT ") + (grouped ? "g, " : "") +
                       "sum(v) FILTER (WHERE keep), min(v) FILTER (WHERE keep), "
                       "max(v) FILTER (WHERE keep), avg(v) FILTER (WHERE keep), "
                       "count(v) FILTER (WHERE keep), count(*) FILTER (WHERE keep) "
                       "FROM agg_empty" +
                       (grouped ? " GROUP BY g" : ""));
  }
}

TEST_CASE_METHOD(AggFilterFixture,
                 "gpu_execution shared FILTER reductions with all-NULL or all-failing inputs",
                 "[integration][gpu_execution][aggregate][aggregate_filter]")
{
  for (auto group : {2, 3}) {
    compare_gpu_vs_cpu(
      "SELECT sum(v) FILTER (WHERE keep), min(v) FILTER (WHERE keep), "
      "max(v) FILTER (WHERE keep), avg(v) FILTER (WHERE keep), "
      "count(v) FILTER (WHERE keep), count(*) FILTER (WHERE keep) "
      "FROM agg_nulls WHERE g = " +
      std::to_string(group));
  }
}
