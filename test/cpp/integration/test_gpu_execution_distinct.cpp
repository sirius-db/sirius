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
 * @file test_gpu_execution_distinct.cpp
 * @brief GPU-vs-CPU correctness for `SELECT DISTINCT`, plus the plan-time fallbacks.
 *
 * Every supported case goes through the shared file-backed `GpuExecutionFixture`, whose
 * `compare_gpu_vs_cpu` asserts a real GPU execution with no fallback before comparing against the
 * CPU run. That assertion is what makes these tests meaningful: `DISTINCT` returns a set, so the
 * results match whenever the query is answered at all, and a query that silently fell back to CPU
 * would still compare equal.
 *
 * A carried `DISTINCT ON` column may come from any row of its group on either engine, so a full row
 * comparison is valid only where the key fixes the carried value, as in `dist_fd`, or where an
 * ORDER BY names a single row per group, as in `dist_ord`. Elsewhere
 * `compare_gpu_vs_cpu_on_columns` compares the row count and the columns the query determines.
 *
 * The floating-point cases use `compare_gpu_vs_cpu_canonical` instead: a group holding both 0.0 and
 * -0.0, or NaN and -NaN, prints as whichever member each engine keeps.
 *
 * Every DISTINCT guard throws during `create_plan`, before any GPU work is scheduled, so the
 * guarded shapes use `expect_plan_fallback_matches_cpu` and assert the plan-time fallback counter
 * rather than the runtime one.
 */

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <algorithm>
#include <cstdint>
#include <string>

namespace {

/// Counts the rows of @p groups_query, one per group, and those matching @p violation, on the GPU
/// and the CPU alike, and requires @p groups rows with no violation.
void require_groups_hold(sirius::test::GpuExecutionFixture& fixture,
                         std::string const& groups_query,
                         std::string const& violation,
                         std::int64_t groups)
{
  auto const check =
    "SELECT count(*), count(*) FILTER (WHERE " + violation + ") FROM (" + groups_query + ")";
  fixture.compare_gpu_vs_cpu(check);
  INFO(check);
  auto result = fixture.con->Query(check);
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  CHECK(result->Collection().GetValue(0, 0) == duckdb::Value::BIGINT(groups));
  CHECK(result->Collection().GetValue(1, 0) == duckdb::Value::BIGINT(0));
}

/// Duplicate `(a, b)` pairs, NULLs in either key column, two rows sharing the composite key
/// `(NULL, 1)`, a wholly-NULL column, and a fully-NULL row.
class DistinctFixture : public sirius::test::GpuExecutionFixture {
 public:
  DistinctFixture()
  {
    run_ok(
      "CREATE TABLE dist_t ("
      "  a       INTEGER,"
      "  b       INTEGER,"
      "  s_short VARCHAR,"
      "  d       DECIMAL(15,2),"
      "  dt      DATE,"
      "  allnull INTEGER);");
    run_ok(
      "INSERT INTO dist_t VALUES "
      "(1,    1,    'ab',  1.00,  DATE '2024-01-01', NULL),"
      "(1,    1,    'ab',  1.00,  DATE '2024-01-01', NULL),"  // exact duplicate row
      "(1,    2,    'cd',  2.50,  DATE '2024-01-02', NULL),"
      "(2,    1,    'ab',  1.00,  DATE '2024-01-01', NULL),"  // duplicate (b, s_short, d, dt)
      "(2,    NULL, 'ef',  3.25,  DATE '2024-01-03', NULL),"  // NULL in b
      "(2,    NULL, 'ef',  3.25,  DATE '2024-01-03', NULL),"  // duplicate (2, NULL)
      "(NULL, 1,    'gh',  1.00,  DATE '2024-01-01', NULL),"  // NULL in a
      "(NULL, 1,    'ij',  4.75,  DATE '2024-01-04', NULL),"  // second (NULL, 1): must collapse
      "(NULL, NULL, NULL,  NULL,  NULL,              NULL);"  // fully-NULL row
    );
    run_ok("CREATE TABLE dist_r (a INTEGER, x INTEGER);");
    run_ok("INSERT INTO dist_r VALUES (1, 10), (2, 20), (2, 21), (3, 30), (NULL, 40);");
    run_ok("CREATE TABLE dist_empty (a INTEGER, b INTEGER);");
    // `v` is a function of `k`, so a DISTINCT ON over `k` has one correct answer whichever row
    // represents a group. The carried and guarded shapes need that: DISTINCT ON without ORDER BY
    // may pick a different row on each run and on each engine.
    run_ok("CREATE TABLE dist_fd (k INTEGER, v INTEGER);");
    run_ok("INSERT INTO dist_fd VALUES (1, 10), (1, 10), (2, 20), (2, 20), (3, 30), (NULL, NULL);");
    // Per-`k` sums reach both ends of BIGINT exactly, plus a negative, a zero and a NULL group.
    run_ok("CREATE TABLE dist_sum (k INTEGER, x BIGINT);");
    run_ok(
      "INSERT INTO dist_sum VALUES "
      "(1, -4611686018427387904), (1, -4611686018427387904),"
      "(2, 4611686018427387904), (2, 4611686018427387903),"
      "(3, -5), (3, 2), (4, 0), (NULL, 7);");
    // Several `(a, b)` groups per `a`, so a DISTINCT ON (a) over a GROUP BY a, b has a choice.
    run_ok("CREATE TABLE dist_agg (a INTEGER, b VARCHAR, c INTEGER);");
    run_ok(
      "INSERT INTO dist_agg VALUES (1, 'x', 10), (1, 'x', 5), (1, 'y', 7), "
      "(2, 'z', 3), (2, 'z', 4), (3, 'p', 1), (3, 'q', 1), (3, 'r', 2);");
    // Every ORDER BY an ordered case uses names one row per `k`, so the CPU answer is fixed, and
    // the ascending winner of `t`, `s`, `d` and `(u DESC, t)` is its group's last row, which the
    // unordered route would not pick. `v` names the row. Groups 1 and 2 each hold one NULL `t`
    // in the middle; group 4 is a single all-NULL row. `d` holds NaN, infinities and at most one
    // zero per group, and `s` prefixes and upper case.
    run_ok(
      "CREATE TABLE dist_ord (k INTEGER, v INTEGER, t INTEGER, u INTEGER, s VARCHAR, d DOUBLE);");
    run_ok(
      "INSERT INTO dist_ord VALUES "
      "(1, 1, 50, 1, 'b', 2.0), (1, 2, 40, 1, 'c', 'inf'::DOUBLE),"
      "(1, 3, NULL, 1, 'bb', 'nan'::DOUBLE), (1, 4, 30, 5, 'ab', 0.0),"
      "(1, 5, 20, 5, 'a', '-inf'::DOUBLE),"
      "(2, 6, 45, 2, 'B', 1.0), (2, 7, 35, 2, 'abc', 'nan'::DOUBLE),"
      "(2, 8, 25, 9, 'A', '-0.0'::DOUBLE), (2, 9, NULL, 9, 'Ba', 3.0), (2, 10, 15, 9, '9', -5.0),"
      "(3, 11, 60, 3, 'x', 'inf'::DOUBLE), (3, 12, 55, 3, 'xy', 0.0),"
      "(3, 13, 52, 7, 'w', 'nan'::DOUBLE), (3, 14, 51, 7, '', -1.0),"
      "(NULL, 15, 70, 4, 'm', 7.0), (NULL, 16, 65, 4, 'n', 8.0), (NULL, 17, 61, 4, 'l', 6.0),"
      "(4, 18, NULL, NULL, NULL, NULL);");
    run_ok("CHECKPOINT;");
  }
};

/// Separate from DistinctFixture so the cheap cases above do not load a million rows once per
/// Catch2 test case.
class DistinctBulkFixture : public sirius::test::GpuExecutionFixture {
 public:
  DistinctBulkFixture()
  {
    // ~22 bytes average length and 20 distinct values in 100k rows, so both dictionary-encode
    // gates in local_grouped_aggregate hold: avg_len >= 8.0 and NDV/rows < 0.10.
    run_ok(
      "CREATE TABLE dist_str AS "
      "SELECT 'a_long_string_value_' || (i % 20) AS s FROM range(100000) t(i);");
    run_ok("CREATE TABLE dist_dup AS SELECT (i % 10) AS k, i AS payload FROM range(1000000) t(i);");
    run_ok("CHECKPOINT;");
  }

  /// Compares the per-key rows of @p winners, a query over dist_dup's ten keys, on the GPU and the
  /// CPU, then requires ten rows with none matching @p mismatch.
  void compare_winners(std::string const& winners, std::string const& mismatch)
  {
    compare_gpu_vs_cpu(winners);
    require_groups_hold(*this, winners, mismatch, 10);
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Supported shapes
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a composite key",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT a, b FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT keeps a NULL key as its own value",
                 "[integration][gpu_execution][distinct][nulls]")
{
  // cudf::null_policy::INCLUDE keeps NULL-keyed rows in play, so one NULL row survives as DuckDB
  // returns it.
  compare_gpu_vs_cpu("SELECT DISTINCT a FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a wholly-NULL column",
                 "[integration][gpu_execution][distinct][nulls]")
{
  // Degenerate one-group case: the column checkpoints to CONSTANT all-null validity and the
  // native scan synthesizes its null mask, so the key path must see NULLs rather than sentinels.
  compare_gpu_vs_cpu("SELECT DISTINCT allnull FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT collapses equal composite NULL keys",
                 "[integration][gpu_execution][distinct][nulls]")
{
  // cudf's hash groupby compares keys with null_equality::EQUAL, so the two (NULL, 1) rows land in
  // the same group. INCLUDE alone would not do that.
  compare_gpu_vs_cpu("SELECT DISTINCT a, b FROM dist_t WHERE a IS NULL");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over short string keys",
                 "[integration][gpu_execution][distinct]")
{
  // Average length below 8 bytes, so local_grouped_aggregate leaves the STRING key uncoded.
  compare_gpu_vs_cpu("SELECT DISTINCT s_short FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over DECIMAL and DATE keys",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT d, dt FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over every column",
                 "[integration][gpu_execution][distinct]")
{
  // Every output column is a group key.
  compare_gpu_vs_cpu("SELECT DISTINCT * FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT with ORDER BY runs on the GPU",
                 "[integration][gpu_execution][distinct]")
{
  // LogicalDistinct::order_by is set only for DISTINCT ON, so a plain DISTINCT under an ORDER BY
  // reaches the builder unguarded and its ORDER BY sorts the deduplicated result. This must not
  // fall back.
  compare_gpu_vs_cpu_ordered(
    "SELECT DISTINCT a, b FROM dist_t ORDER BY a NULLS LAST, b NULLS "
    "LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with no carried columns",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT ON (a) a FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with a key that is not an output column",
                 "[integration][gpu_execution][distinct]")
{
  // `b` is appended to the projection under the distinct and pruned again above it, so this is a
  // two-key dedup feeding a single-column projection.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (a, b) a FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with keys in a different order to the outputs",
                 "[integration][gpu_execution][distinct]")
{
  // Group position 0 reads child column 1, so the builder emits its reorder projection.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (b, a) a, b FROM dist_t");
}

//===----------------------------------------------------------------------===//
// Carried columns: an output column no distinct target covers
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with a carried column runs on the GPU",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_fd");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT with an ORDER BY outside the select list runs on the GPU",
                 "[integration][gpu_execution][distinct]")
{
  // `v` is added to the select list after the targets were synthesized from it, so the node is
  // two columns wide with a single target and `v` is carried.
  compare_gpu_vs_cpu_ordered("SELECT DISTINCT k FROM dist_fd ORDER BY v NULLS LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON an expression key runs on the GPU",
                 "[integration][gpu_execution][distinct]")
{
  // The binder appends `k + v` to the select list, so the one key is a bare reference and both
  // `k` and `v` are carried.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k + v) k, v FROM dist_fd");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with permuted keys and a carried column",
                 "[integration][gpu_execution][distinct]")
{
  // group_idx is {1, 0}: the local stage has to emit the keys first for PARTITION, and the
  // builder's projection has to restore the select order. The two `(NULL, 1)` rows differ in
  // `s_short`, so only the keys are compared.
  compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (b, a) a, b, s_short FROM dist_t", {0, 1});
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON ordered only by its keys runs on the GPU",
                 "[integration][gpu_execution][distinct]")
{
  // A key is constant within its group, so an ORDER BY on keys alone cannot choose a carried row.
  SECTION("a carried column")
  {
    compare_gpu_vs_cpu_ordered("SELECT DISTINCT ON (k) k, v FROM dist_fd ORDER BY k NULLS LAST");
  }

  // Every group of `a` holds rows whose carried values differ, so only the key is compared.
  SECTION("a carried column that varies within its group")
  {
    compare_gpu_vs_cpu_on_columns(
      "SELECT DISTINCT ON (a) a, b, s_short FROM dist_t ORDER BY a NULLS LAST", {0});
  }

  // Every column is a key, so nothing is carried and the ORDER BY only sorts the result.
  SECTION("every column a key")
  {
    compare_gpu_vs_cpu_ordered("SELECT DISTINCT ON (k, v) k, v FROM dist_fd ORDER BY v NULLS LAST");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution carried DECIMAL, DATE and all-NULL columns",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Without NULL in `a`, every (a, b) group's rows are identical, so each carried value is fixed.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT ON (a, b) a, b, s_short, d, dt, allnull FROM dist_t WHERE a IS NOT NULL");
  compare_gpu_vs_cpu(
    "SELECT a, b, first(d), first(dt), first(allnull), count(*) FROM dist_t WHERE a IS NOT NULL "
    "GROUP BY a, b");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution carried columns over a filter that keeps no row",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // 15 lies inside v's min/max, so the scan and its filter survive planning and every carried path
  // runs on an empty batch.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_fd WHERE v = 15");
  compare_gpu_vs_cpu("SELECT k, first(v), sum(v), avg(v) FROM dist_fd WHERE v = 15 GROUP BY k");
  compare_gpu_vs_cpu("SELECT k, first(v), stddev_samp(v) FROM dist_fd WHERE v = 15 GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution GROUP BY with only FIRST aggregates runs on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The operator cannot tell this from a lowered DISTINCT ON, so it takes the same route.
  compare_gpu_vs_cpu("SELECT k, first(v) FROM dist_fd GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST preserves carried columns beside low-cardinality SUM",
                 "[integration][gpu_execution][distinct][aggregate][lowcard_groupby]")
{
  // The default-on specialized SUM kernel cannot emit FIRSTs.
  run_ok("CREATE TABLE first_lowcard (k VARCHAR NOT NULL, v BIGINT NOT NULL);");
  run_ok("INSERT INTO first_lowcard VALUES ('A', 10), ('A', 20), ('B', 30), ('B', 40);");
  run_ok("CHECKPOINT;");
  require_groups_hold(
    *this,
    "SELECT k, first(v) AS a, first(100 - v) AS b, sum(v) AS s FROM first_lowcard "
    "GROUP BY k",
    "a + b IS DISTINCT FROM 100 OR s IS DISTINCT FROM CASE k WHEN 'A' THEN 30 "
    "ELSE 70 END",
    2);
  compare_gpu_vs_cpu("SELECT k, first(k), sum(v) FROM first_lowcard GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "low-cardinality aggregation agrees with cuDF across SQL policy changes",
                 "[integration][gpu_execution][lowcard_groupby][optimization_settings]")
{
  run_ok("CREATE TABLE policy_lowcard (k VARCHAR NOT NULL, v BIGINT NOT NULL)");
  run_ok(
    "INSERT INTO policy_lowcard SELECT CASE WHEN range % 2 = 0 THEN 'A' ELSE 'B' END, "
    "range::BIGINT FROM range(4096)");
  run_ok("CHECKPOINT");
  for (bool const enabled : {true, false, true}) {
    sirius::test::scoped_sirius_setting policy(*con, "enable_low_cardinality_groupby", enabled);
    sirius::test::scoped_recording_log_sink logs("debug");
    compare_gpu_vs_cpu("SELECT k, sum(v), count(*) FROM policy_lowcard GROUP BY k");
    auto const records     = logs.records();
    bool const specialized = std::any_of(records.begin(), records.end(), [](auto const& record) {
      return record.message.find("local_grouped_agg: low-cardinality path") != std::string::npos;
    });
    REQUIRE(specialized == enabled);
    compare_gpu_vs_cpu("SELECT k, sum(v), count(*) FROM policy_lowcard GROUP BY ROLLUP(k)");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST beside another aggregate runs on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  compare_gpu_vs_cpu("SELECT k, first(v), sum(v) FROM dist_fd GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST beside other aggregates keeps the declared column order",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The partial layout is [k, partials..., carried...], so each result must be moved into the
  // declared order. With both value columns BIGINT, a swap passes every type check and only the
  // row comparison catches it.
  SECTION("count")
  {
    compare_gpu_vs_cpu("SELECT k, first(v::BIGINT), count(*) FROM dist_fd GROUP BY k");
  }

  SECTION("avg before the FIRST")
  {
    compare_gpu_vs_cpu("SELECT k, avg(v), first(v) FROM dist_fd GROUP BY k");
  }

  // COLLECT_SET puts the groupby, and its row selector, on cuDF's sort path.
  SECTION("COUNT(DISTINCT)")
  {
    compare_gpu_vs_cpu("SELECT k, first(v), count(DISTINCT v) FROM dist_fd GROUP BY k");
  }

  // `0 - v`, not `-v`: the expression evaluator reads a second argument for unary minus.
  SECTION("two FIRSTs and a min")
  {
    compare_gpu_vs_cpu("SELECT k, first(v), first(0 - v), min(v) FROM dist_fd GROUP BY k");
  }

  SECTION("a carried key beside a sum, with a NULL-key group")
  {
    compare_gpu_vs_cpu("SELECT k, first(k), sum(v) FROM dist_fd GROUP BY k");
  }

  // M2 sorts the local input by key before the groupby.
  SECTION("stddev_samp")
  {
    compare_gpu_vs_cpu("SELECT k, first(v), first(0 - v), stddev_samp(v) FROM dist_fd GROUP BY k");
  }
}

TEST_CASE_METHOD(
  DistinctFixture,
  "gpu_execution an all-FIRST list with an expression or key argument runs on the GPU",
  "[integration][gpu_execution][distinct][aggregate]")
{
  SECTION("two arguments, one an expression")
  {
    compare_gpu_vs_cpu("SELECT k, first(v), first(0 - v) FROM dist_fd GROUP BY k");
  }

  SECTION("key argument") { compare_gpu_vs_cpu("SELECT k, first(k) FROM dist_fd GROUP BY k"); }
}

//===----------------------------------------------------------------------===//
// Composition with the operators either side of the DISTINCT
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT on a join probe side",
                 "[integration][gpu_execution][distinct][join]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT l.a FROM dist_t l JOIN dist_r r ON l.a = r.a");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT feeding a join build side",
                 "[integration][gpu_execution][distinct][join]")
{
  compare_gpu_vs_cpu(
    "SELECT r.x FROM dist_r r JOIN (SELECT DISTINCT a FROM dist_t) d ON r.a = d.a");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT under an outer aggregate",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT DISTINCT a, b FROM dist_t)");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT directly over a grouped sum",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT k, sum(x) FROM dist_sum GROUP BY k");
  // Reorder projection above the narrowed aggregate, over keys of different types.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT ON (s, k) k, s FROM (SELECT k, sum(x) AS s FROM dist_sum GROUP BY k)");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON carrying a grouped sum",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Each `k` is one row, so the carried HUGEINT value is determined. Sums at both ends of BIGINT
  // exercise the native 128-bit carrier.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT ON (k) k, s FROM (SELECT k, sum(x) AS s FROM dist_sum GROUP BY k)");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON over a GROUP BY carries the aggregate results",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The GROUP BY's results reach the DISTINCT ON as carried columns, and sum() stays HUGEINT
  // under a HUGEINT declaration.
  SECTION("one group per key")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v, sum(v) FROM dist_fd GROUP BY k, v");
  }

  SECTION("a mixed aggregate list below")
  {
    compare_gpu_vs_cpu(
      "SELECT DISTINCT ON (k) k, v, first(v), sum(v), count(DISTINCT v) "
      "FROM dist_fd GROUP BY k, v");
  }

  // Any group of a key may survive, so each output row is joined back to the groups: a `b` and a
  // sum taken from different groups match none.
  SECTION("several groups per key")
  {
    compare_gpu_vs_cpu(
      "SELECT count(*) FROM "
      "(SELECT DISTINCT ON (a) a, b, sum(c) AS s FROM dist_agg GROUP BY a, b) d "
      "JOIN (SELECT a, b, sum(c) AS s FROM dist_agg GROUP BY a, b) g "
      "ON d.a = g.a AND d.b = g.b AND d.s = g.s");
    compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (a) a, b, sum(c) FROM dist_agg GROUP BY a, b",
                                  {0});
  }

  SECTION("an aggregate result as the key")
  {
    compare_gpu_vs_cpu(
      "SELECT DISTINCT ON (s) s, a FROM (SELECT a, sum(c) AS s FROM dist_agg GROUP BY a)");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT inside a correlated subquery",
                 "[integration][gpu_execution][distinct][join]")
{
  // Decorrelation appends the correlated column to the distinct targets, so the builder sees a
  // target list no uncorrelated query produces. `b = 1` occurs under both `a = 1` and `a = 2`, so a
  // dedup that dropped the appended key would lose a row.
  SECTION("lateral join")
  {
    compare_gpu_vs_cpu(
      "SELECT r.a, s.b FROM dist_r r, LATERAL (SELECT DISTINCT t.b FROM dist_t t WHERE t.a = r.a) "
      "s");
  }

  // A second correlated predicate that is not an equality: the subquery cannot be computed for
  // every key, so the DISTINCT reads a join against the outer rows' keys.
  SECTION("two correlated predicates")
  {
    compare_gpu_vs_cpu(
      "SELECT r.a, r.x, s.b FROM dist_r r, "
      "LATERAL (SELECT DISTINCT t.b FROM dist_t t WHERE t.a = r.a AND t.b < r.x) s");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a materialized CTE",
                 "[integration][gpu_execution][distinct]")
{
  // With equal widths Sirius omits the identity projection, so the CTE node is the DISTINCT's
  // direct child. The narrower select keeps its projection.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT * FROM (WITH c AS MATERIALIZED (SELECT a, b FROM dist_t) SELECT * FROM c) s");
  compare_gpu_vs_cpu(
    "SELECT DISTINCT a FROM (WITH c AS MATERIALIZED (SELECT a, b FROM dist_t) SELECT * FROM c) s");
  // Nested: `d` is read twice and joined on `b`, so it stays wider than the body under `c`.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT * FROM (WITH c AS MATERIALIZED (SELECT a FROM dist_t), "
    "d AS MATERIALIZED (SELECT a, b FROM dist_t) "
    "SELECT d1.a FROM d d1 JOIN d d2 ON d1.b = d2.b JOIN c ON c.a = d1.a) s");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over zero input rows",
                 "[integration][gpu_execution][distinct]")
{
  SECTION("empty table") { compare_gpu_vs_cpu("SELECT DISTINCT a, b FROM dist_empty"); }

  // 15 lies inside x's min/max, so the scan and its filter survive planning.
  SECTION("filter that keeps no row")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT a, x FROM dist_r WHERE x = 15");
  }

  // Outside a's min/max, statistics fold the scan (not the DISTINCT) to EMPTY_RESULT.
  SECTION("scan folded to an empty result")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT a, b FROM dist_t WHERE a > 1000");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution nested carried columns run on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The scan refuses stored LIST and STRUCT columns, so a stored fixed-size array and a computed
  // struct are the nested payloads that reach the DISTINCT. Each is a function of `k`.
  SECTION("fixed-size array")
  {
    run_ok(
      "CREATE TABLE dist_nest AS SELECT k, [k, k + 1, k + 2]::INTEGER[3] AS arr FROM dist_fd;");
    run_ok("CHECKPOINT;");
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, arr FROM dist_nest");
    compare_gpu_vs_cpu("SELECT k, first(arr), count(*) FROM dist_nest GROUP BY k");
  }

  SECTION("struct")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, {'x': k, 'y': k * 10} AS st FROM dist_fd");
    compare_gpu_vs_cpu("SELECT k, first({'x': k, 'y': k * 10}), sum(k) FROM dist_fd GROUP BY k");
  }
}

//===----------------------------------------------------------------------===//
// Volume: the paths a nine-row table cannot reach
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT over long low-cardinality string keys",
                 "[integration][gpu_execution][distinct]")
{
  // Both dictionary-encode gates hold for this column, so the STRING key goes through
  // cudf::dictionary::encode before grouping.
  compare_gpu_vs_cpu("SELECT DISTINCT s FROM dist_str");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT over a heavily duplicated key",
                 "[integration][gpu_execution][distinct]")
{
  // A million rows over ten groups, so the local dedup shrinks each batch to at most ten rows
  // before the hash shuffle.
  SECTION("default batch size") { compare_gpu_vs_cpu("SELECT DISTINCT k FROM dist_dup"); }

  // The default batch size scans dist_dup in one batch, leaving the merge nothing to combine.
  SECTION("many scan batches")
  {
    sirius::test::scoped_sirius_setting batch_size{
      *con, "scan_task_batch_size", std::uint64_t{1048576}};
    compare_gpu_vs_cpu("SELECT DISTINCT k FROM dist_dup");
  }
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT ON a heavily duplicated key with a carried column",
                 "[integration][gpu_execution][distinct]")
{
  // Each of the ten groups holds 100k different payloads, so only the keys are compared.
  SECTION("default batch size")
  {
    compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (k) k, payload FROM dist_dup", {0});
  }

  // Small scan batches give a partition more than one local result, which the merge's single-batch
  // shortcut would otherwise skip. With the key second, group_idx is {1}: a merge that keyed on
  // group_idx rather than on its leading column would dedup on `payload` and return every row.
  SECTION("many scan batches")
  {
    sirius::test::scoped_sirius_setting batch_size{
      *con, "scan_task_batch_size", std::uint64_t{1048576}};
    compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (k) k, payload FROM dist_dup", {0});
    compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (k) payload, k FROM dist_dup", {1});
  }
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT ON merges NULL and NaN keys across batches",
                 "[integration][gpu_execution][distinct]")
{
  // Small scan batches split each key's rows over several partials, so the merge's cudf::distinct
  // must treat NULL keys as equal, every NaN as equal, and -0.0 as 0.0.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  run_ok(
    "CREATE TABLE dist_dup_null AS SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i % 10 END AS k, "
    "i AS payload FROM range(1000000) t(i);");
  // 0.0 and -0.0 are never adjacent, so RLE cannot fold the sign away at CHECKPOINT.
  run_ok(
    "CREATE TABLE dist_dup_fp AS SELECT CASE i % 6 WHEN 0 THEN 'NaN'::DOUBLE "
    "WHEN 1 THEN 0.0::DOUBLE WHEN 2 THEN -('NaN'::DOUBLE) WHEN 3 THEN -(0.0::DOUBLE) "
    "WHEN 4 THEN NULL ELSE 1.5 END AS d, i AS payload FROM range(1000000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (k) k, payload FROM dist_dup_null", {0});
  compare_gpu_vs_cpu_on_columns("SELECT DISTINCT ON (d) d, payload FROM dist_dup_fp", {0});
  // The groupby route with the row selector, over the same NULL group.
  require_groups_hold(*this,
                      "SELECT k, first(payload) AS a, count(*) AS c FROM dist_dup_null GROUP BY k",
                      "coalesce(k, 0) IS DISTINCT FROM a % 10 OR c IS DISTINCT FROM 100000",
                      10);
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT ON over a GROUP BY over many batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Small scan batches make both merges combine partial rows: the GROUP BY's sums and the
  // DISTINCT ON's carried rows. Each output row must still be one real (k, m, s) group.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT DISTINCT ON (k) k, m, s FROM (SELECT k, payload % 3 AS m, "
    "sum(payload) AS s FROM dist_dup GROUP BY k, m)) d "
    "JOIN (SELECT k, payload % 3 AS m, sum(payload) AS s FROM dist_dup GROUP BY k, m) g "
    "ON d.k = g.k AND d.m = g.m AND d.s = g.s");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution FIRST beside other aggregates over many batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Small scan batches make the merge combine partial rows. `payload` varies within each group, so
  // the check is structural rather than a row comparison: `a % 10 = k` catches a gather from the
  // wrong group, `a + b` two FIRSTs taken from different rows, and `c` a column swap.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ("
    "  SELECT k, first(payload) AS a, first(1000000 - payload) AS b, count(*) AS c,"
    "         sum(payload) AS s"
    "  FROM dist_dup GROUP BY k) "
    "WHERE a % 10 <> k OR a + b <> 1000000 OR c <> 100000");
  compare_gpu_vs_cpu_on_columns(
    "SELECT k, first(payload) AS a, first(1000000 - payload) AS b, "
    "count(*) AS c, sum(payload) AS s FROM dist_dup GROUP BY k",
    {0, 3, 4});

  // MERGE_SETS puts the merge groupby, and its row selector, on cuDF's sort path.
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ("
    "  SELECT k, first(payload) AS a, count(DISTINCT payload % 7) AS c"
    "  FROM dist_dup GROUP BY k) "
    "WHERE a % 10 <> k OR c <> 7");
  compare_gpu_vs_cpu_on_columns(
    "SELECT k, first(payload) AS a, count(DISTINCT payload % 7) AS c FROM dist_dup GROUP BY k",
    {0, 2});
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution FIRST beside stddev_samp keeps each group's row",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // M2 sorts the local input by key, and MERGE_M2 puts the merge on cuDF's sort path. The checks
  // match the many-batch FIRST case above; rounding keeps the stddev comparison exact.
  auto const check = [&] {
    compare_gpu_vs_cpu(
      "SELECT count(*) FROM ("
      "  SELECT k, first(payload) AS a, first(1000000 - payload) AS b,"
      "         stddev_samp(payload) AS sd"
      "  FROM dist_dup GROUP BY k) "
      "WHERE a % 10 <> k OR a + b <> 1000000 OR sd IS NULL");
    compare_gpu_vs_cpu_on_columns(
      "SELECT k, first(payload) AS a, first(1000000 - payload) AS b, "
      "round(stddev_samp(payload), 1) AS sd FROM dist_dup GROUP BY k",
      {0, 3});
  };

  SECTION("default batch size") { check(); }

  SECTION("many scan batches")
  {
    sirius::test::scoped_sirius_setting batch_size{
      *con, "scan_task_batch_size", std::uint64_t{1048576}};
    check();
  }
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution nested carried columns over many batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  run_ok("CREATE TABLE dist_dup_arr AS SELECT k, [k, k]::BIGINT[2] AS arr FROM dist_dup;");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, arr FROM dist_dup_arr");
  compare_gpu_vs_cpu("SELECT k, first(arr), count(*) FROM dist_dup_arr GROUP BY k");
  compare_gpu_vs_cpu("SELECT k, first({'x': k, 'y': k * 10}), count(*) FROM dist_dup GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST beside a COUNT(DISTINCT) over a label-encoded key",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // COLLECT_SET, a VARCHAR key and at least 2^20 rows in one batch make the local groupby
  // label-encode its keys, so the row selector runs over the encoded groupby.
  run_ok(
    "CREATE TABLE dist_big AS "
    "SELECT (i % 10)::VARCHAR AS ks, i AS payload FROM range(2000000) t(i);");
  run_ok("CHECKPOINT;");
  // `min(payload)` is the group's digit, which avoids a VARCHAR cast in the outer filter.
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ("
    "  SELECT ks, first(payload) AS a, min(payload) AS m, count(DISTINCT payload % 7) AS c"
    "  FROM dist_big GROUP BY ks) "
    "WHERE a % 10 <> m OR c <> 7");
  compare_gpu_vs_cpu_on_columns(
    "SELECT ks, first(payload) AS a, min(payload) AS m, "
    "count(DISTINCT payload % 7) AS c FROM dist_big GROUP BY ks",
    {0, 2, 3});
}

//===----------------------------------------------------------------------===//
// Ordered carried rows: DISTINCT ON under an ORDER BY, and first(v ORDER BY ...)
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON keeps each group's first row under the ORDER BY",
                 "[integration][gpu_execution][distinct]")
{
  // Each group's first row under `t` is its last row, so a dropped order returns other rows.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON honors direction and NULL placement",
                 "[integration][gpu_execution][distinct]")
{
  // The descending winners of `t` are first rows, which the unordered route also keeps, so this
  // pins the direction only; `v` rises within each group, so its descending winners are last rows.
  SECTION("DESC")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY t DESC");
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, t FROM dist_ord ORDER BY v DESC");
  }

  SECTION("NULLS FIRST")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY t NULLS FIRST");
  }

  SECTION("DESC NULLS FIRST")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY t DESC NULLS FIRST");
  }

  // Mixed directions over two keys: `u` ties at its maximum and `t` breaks the tie.
  SECTION("two keys")
  {
    compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY u DESC, t");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON drops ORDER BY terms that are its keys",
                 "[integration][gpu_execution][distinct]")
{
  // `ORDER BY k, t` orders each group by `t` alone.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY k, t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON carries its ORDER BY column",
                 "[integration][gpu_execution][distinct]")
{
  // `t` is carried and is also the order key.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v, t FROM dist_ord ORDER BY t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON over a GROUP BY ordered by an aggregate",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The top and bottom group per key, ordered by an integer SUM, ties broken by `b`. Keys 1 and 3
  // keep different groups in the two directions, so one of the two fails whichever group an
  // unordered route keeps.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT ON (a) a, b, sum(c) FROM dist_agg GROUP BY a, b ORDER BY a, sum(c) DESC, b");
  compare_gpu_vs_cpu(
    "SELECT DISTINCT ON (a) a, b, sum(c) FROM dist_agg GROUP BY a, b ORDER BY a, sum(c), b");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON orders by a computed key",
                 "[integration][gpu_execution][distinct]")
{
  // `100 - t DESC` orders as `t` ascending, so the winners are last rows; sorting `t` itself
  // descending would keep the first rows.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord ORDER BY 100 - t DESC");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution grouped first(v ORDER BY ...) runs on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  SECTION("beside a SUM and a COUNT(DISTINCT)")
  {
    // COLLECT_SET puts the groupby, and its ranked row selector, on cuDF's sort path.
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY t), sum(v), count(DISTINCT u) FROM dist_ord GROUP BY k");
  }

  SECTION("VARCHAR key")
  {
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY s) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY s DESC) FROM dist_ord GROUP BY k");
  }

  SECTION("DOUBLE key with NaN and infinities")
  {
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY d) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY d DESC) FROM dist_ord GROUP BY k");
  }

  SECTION("beside an unordered FIRST")
  {
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY t), first(k) FROM dist_ord GROUP BY k");
  }

  SECTION("two keys")
  {
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY u DESC, t) FROM dist_ord GROUP BY k");
  }

  // The rewrites share one create_sort_key, which DuckDB projects below the aggregate.
  SECTION("two FIRSTs sharing an ORDER BY")
  {
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY t), first(s ORDER BY t) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY u DESC, t), first(d ORDER BY u DESC, t) FROM dist_ord GROUP BY "
      "k");
  }

  // `100 - t DESC` orders as `t` ascending; a key column holding `t` would keep the first rows.
  SECTION("a computed key, alone and shared by two FIRSTs")
  {
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY 100 - t DESC) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY 100 - t DESC), first(s ORDER BY 100 - t DESC) FROM dist_ord "
      "GROUP BY k");
  }

  // The FILTER masks groups 1 and 2's winners (`t` 20, 15) for the sum only; a key column shared
  // with the masked input would sort them last and keep another row.
  SECTION("beside a FILTERed aggregate over the key column")
  {
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY t), sum(t) FILTER (WHERE t > 25) FROM dist_ord GROUP BY k");
  }

  // `t` as DECIMAL and as DATE, so the sort runs on fixed-point and timestamp-days storage. Under
  // `DESC NULLS FIRST` groups 1 and 2 keep their middle NULL row, which no unordered route keeps.
  SECTION("DECIMAL and DATE keys")
  {
    run_ok(
      "CREATE TABLE dist_ord_types AS SELECT k, v, t::DECIMAL(10, 2) AS dec, "
      "DATE '2024-01-01' + t AS dt FROM dist_ord ORDER BY v;");
    run_ok("CHECKPOINT;");
    compare_gpu_vs_cpu("SELECT k, first(v ORDER BY dec) FROM dist_ord_types GROUP BY k");
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY dt DESC NULLS FIRST) FROM dist_ord_types GROUP BY k");
  }

  // stddev_samp is an M2 request, so the groupby sorts every request's values, the rank included.
  SECTION("beside an M2 aggregate")
  {
    compare_gpu_vs_cpu_on_columns(
      "SELECT k, first(v ORDER BY t), stddev_samp(t) FROM dist_ord GROUP BY k", {0, 1});
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution grouped last(v ORDER BY ...) runs on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Groups 1 and 2 hold a NULL `t` in a middle row. Each query's winners differ from both the
  // un-reversed order's and the order with only the direction reversed.
  SECTION("both directions and NULL placements")
  {
    compare_gpu_vs_cpu("SELECT k, last(v ORDER BY t) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu("SELECT k, last(v ORDER BY t DESC) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu("SELECT k, last(v ORDER BY t NULLS FIRST) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu("SELECT k, last(v ORDER BY t DESC NULLS FIRST) FROM dist_ord GROUP BY k");
  }

  // Group 4's winner has a NULL `v`; group 1's NULL `v` sits in a row that does not win.
  SECTION("a NULL value at the winner")
  {
    run_ok("CREATE TABLE dist_last (k INTEGER, v VARCHAR, t INTEGER);");
    run_ok(
      "INSERT INTO dist_last VALUES (1, 'a', 30), (1, 'b', 10), (1, NULL, 20), (2, 'c', 5), "
      "(2, 'd', NULL), (3, 'e', 7), (4, 'f', 8), (4, NULL, 9);");
    run_ok("CHECKPOINT;");
    compare_gpu_vs_cpu("SELECT k, last(v ORDER BY t) FROM dist_last GROUP BY k");
  }

  SECTION("beside a SUM and a COUNT(DISTINCT)")
  {
    compare_gpu_vs_cpu(
      "SELECT k, last(v ORDER BY t), sum(v), count(DISTINCT u) FROM dist_ord GROUP BY k");
  }

  // The rewrites share one create_sort_key, which DuckDB projects below the aggregate.
  SECTION("two LASTs sharing an ORDER BY")
  {
    compare_gpu_vs_cpu("SELECT k, last(v ORDER BY t), last(s ORDER BY t) FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu(
      "SELECT k, last(v ORDER BY u DESC, t), last(d ORDER BY u DESC, t) FROM dist_ord GROUP BY k");
  }

  // `t DESC NULLS FIRST` is the reverse of `t`, so both read the same row.
  SECTION("beside a FIRST under the reversed order")
  {
    compare_gpu_vs_cpu(
      "SELECT k, last(v ORDER BY t), first(s ORDER BY t DESC NULLS FIRST) FROM dist_ord GROUP BY "
      "k");
  }

  // The shared key's `t` and the lone aggregate's `t` read one column, so one order remains.
  SECTION("two LASTs sharing an ORDER BY beside a FIRST under the reversed order")
  {
    compare_gpu_vs_cpu(
      "SELECT k, last(v ORDER BY t), last(s ORDER BY t), first(u ORDER BY t DESC NULLS FIRST) "
      "FROM dist_ord GROUP BY k");
    compare_gpu_vs_cpu(
      "SELECT k, first(v ORDER BY t), first(s ORDER BY t), last(u ORDER BY t DESC NULLS FIRST) "
      "FROM dist_ord GROUP BY k");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution ordered FIRST and LAST with DISTINCT run on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // `u` repeats within groups 1 to 3 and rises, so the winners are later rows than the first.
  compare_gpu_vs_cpu("SELECT k, first(DISTINCT u ORDER BY u DESC) FROM dist_ord GROUP BY k");
  compare_gpu_vs_cpu("SELECT k, last(DISTINCT u ORDER BY u) FROM dist_ord GROUP BY k");
  compare_gpu_vs_cpu("SELECT k, first(DISTINCT v ORDER BY v DESC) FROM dist_ord GROUP BY k");
  // Groups 1 and 2 keep their NULL `t`, which MIN and MAX would skip.
  compare_gpu_vs_cpu("SELECT k, last(DISTINCT t ORDER BY t) FROM dist_ord GROUP BY k");
  compare_gpu_vs_cpu("SELECT k, first(DISTINCT t ORDER BY t NULLS FIRST) FROM dist_ord GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution ordered FIRST breaks signed-zero and NaN ties by its next key",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Each group's first key ties (-0.0 = 0.0, -NaN = NaN), so `t` alone picks the row: a
  // sign-aware float order keeps v 1 (ASC) and v 5 or v 3 (DESC) instead.
  run_ok("CREATE TABLE dist_zero (k INTEGER, v INTEGER, d DOUBLE, t INTEGER);");
  run_ok(
    "INSERT INTO dist_zero VALUES (1, 1, '-0.0'::DOUBLE, 20), (1, 2, 0.0, 10), "
    "(2, 3, 'nan'::DOUBLE, 20), (2, 4, -('nan'::DOUBLE), 10), "
    "(3, 5, 0.0, 20), (3, 6, '-0.0'::DOUBLE, 10);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT k, first(v ORDER BY d, t) FROM dist_zero GROUP BY k");
  compare_gpu_vs_cpu("SELECT k, first(v ORDER BY d DESC, t) FROM dist_zero GROUP BY k");
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_zero ORDER BY d DESC, t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution ordered FIRST over a filter that keeps no row",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // 26 lies inside t's min/max, so statistics keep the scan and its filter and the ranked selector
  // sees 0 rows.
  compare_gpu_vs_cpu("SELECT k, first(v ORDER BY t) FROM dist_ord WHERE t = 26 GROUP BY k");
  compare_gpu_vs_cpu("SELECT DISTINCT ON (k) k, v FROM dist_ord WHERE t = 26 ORDER BY t");
}

// The GPU lower() maps U+0130 to "i" + U+0307 where DuckDB maps it to "i", and the GPU upper() maps
// U+00DF to "SS" where DuckDB maps it to U+1E9E (#2598), so a NOCASE, lower() or upper() key could
// keep another row; every such key falls back.
TEST_CASE_METHOD(
  DistinctFixture,
  "gpu_execution ordered FIRST and DISTINCT ON over a case-mapped key fall back at plan time",
  "[integration][gpu_execution][distinct][aggregate]")
{
  // DuckDB orders 'ia' (from 'İa') before 'ib' and keeps v 1; the GPU's 'i̇a' sorts after 'ib'.
  // DuckDB orders 'STRASZ' before 'STRAẞE' and keeps v 4; the GPU's 'STRASSE' sorts first.
  run_ok("CREATE TABLE dist_nocase (k INTEGER, s VARCHAR, v INTEGER);");
  run_ok(
    "INSERT INTO dist_nocase VALUES (1, 'İa', 1), (1, 'ib', 2), (2, 'straße', 3), (2, 'strasz', "
    "4);");
  run_ok("CHECKPOINT;");

  SECTION("an explicit collation")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v ORDER BY s COLLATE NOCASE) FROM dist_nocase GROUP BY k");
    expect_plan_fallback_matches_cpu(
      "SELECT k, last(v ORDER BY s COLLATE NOCASE DESC) FROM dist_nocase GROUP BY k");
  }

  SECTION("a lower() or upper() key")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v ORDER BY lower(s)) FROM dist_nocase GROUP BY k");
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v ORDER BY upper(s)) FROM dist_nocase GROUP BY k");
  }

  // DuckDB projects an explicit lower() or upper() key below the Distinct.
  SECTION("a DISTINCT ON key")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT DISTINCT ON (k) k, v FROM dist_nocase ORDER BY k, s COLLATE NOCASE");
    expect_plan_fallback_matches_cpu(
      "SELECT DISTINCT ON (k) k, v FROM dist_nocase ORDER BY k, lower(s)");
    expect_plan_fallback_matches_cpu(
      "SELECT DISTINCT ON (k) k, v FROM dist_nocase ORDER BY k, upper(s)");
  }

  // Two FIRSTs share the sort key, and a max(lower(s)) beside one moves lower(s) into the
  // projection below, so in both the aggregate's key reads a projected column.
  SECTION("a collated key the aggregate reads from a projection")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v ORDER BY s COLLATE NOCASE), first(s ORDER BY s COLLATE NOCASE) FROM "
      "dist_nocase GROUP BY k");
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v ORDER BY s COLLATE NOCASE), max(lower(s)) FROM dist_nocase GROUP BY k");
  }

  SECTION("a default collation")
  {
    auto const prior_result = con->Query("SELECT current_setting('default_collation');");
    REQUIRE(prior_result);
    REQUIRE_FALSE(prior_result->HasError());
    auto const prior = prior_result->Collection().GetValue(0, 0);
    {
      sirius::test::scoped_sirius_setting collation{*con, "default_collation", "nocase"};
      expect_plan_fallback_matches_cpu("SELECT k, first(v ORDER BY s) FROM dist_nocase GROUP BY k");
      expect_plan_fallback_matches_cpu(
        "SELECT DISTINCT ON (k) k, v FROM dist_nocase ORDER BY k, s");
    }
    // A failed restore would leak nocase into every later integration case.
    REQUIRE(
      con->Query("SELECT current_setting('default_collation');")->Collection().GetValue(0, 0) ==
      prior);
  }
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution ordered carried rows over many batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Each key's largest payload sits in the last scan batch, so the merge must rank the partials
  // again; the unordered route keeps the smallest.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  compare_winners("SELECT DISTINCT ON (k) k, payload AS w FROM dist_dup ORDER BY payload DESC",
                  "w IS DISTINCT FROM 999990 + k");
  compare_winners("SELECT DISTINCT ON (k) k, payload AS w FROM dist_dup ORDER BY payload",
                  "w IS DISTINCT FROM k");
  // MERGE_SETS puts the merge groupby, and its ranked row selector, on cuDF's sort path.
  compare_gpu_vs_cpu(
    "SELECT k, first(payload ORDER BY payload DESC), count(*), count(DISTINCT payload % 7) "
    "FROM dist_dup GROUP BY k");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution ordered carried rows rank NULL keys across batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The first and third queries null the key from 500000 on and the second below it, so NULL-keyed
  // partials reach the merge from both ends; the winner, 499990 + k, needs NULLs placed per key at
  // the merge as well.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  compare_winners(
    "SELECT k, first(payload ORDER BY CASE WHEN payload >= 500000 THEN NULL ELSE payload END DESC) "
    "AS w FROM dist_dup GROUP BY k",
    "w IS DISTINCT FROM 499990 + k");
  compare_winners(
    "SELECT k, first(payload ORDER BY CASE WHEN payload < 500000 THEN NULL ELSE payload END NULLS "
    "FIRST, payload DESC) AS w FROM dist_dup GROUP BY k",
    "w IS DISTINCT FROM 499990 + k");
  compare_winners(
    "SELECT DISTINCT ON (k) k, payload AS w FROM dist_dup ORDER BY CASE WHEN payload >= 500000 "
    "THEN NULL ELSE payload END DESC",
    "w IS DISTINCT FROM 499990 + k");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution ordered LAST over many batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Each key's largest payload sits in the last scan batch, so the merge must rank the partials
  // again under the reversed order. The second query's key is NULL from 500000 on.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  compare_winners(
    "SELECT k, last(payload ORDER BY payload) AS a, last(payload + 1 ORDER BY payload) AS b FROM "
    "dist_dup GROUP BY k",
    "a IS DISTINCT FROM 999990 + k OR b IS DISTINCT FROM a + 1");
  compare_winners(
    "SELECT k, last(payload ORDER BY CASE WHEN payload >= 500000 THEN NULL ELSE payload END NULLS "
    "FIRST) AS w FROM dist_dup GROUP BY k",
    "w IS DISTINCT FROM 499990 + k");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution ordered carried rows over several merge partitions",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // A tiny partition size splits the ten groups' partials over several merge tasks.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  sirius::test::scoped_sirius_setting partitions{*con, "hash_partition_bytes", std::uint64_t{1000}};
  compare_winners(
    "SELECT k, last(payload ORDER BY payload) AS a, last(payload + 1 ORDER BY payload) AS b FROM "
    "dist_dup GROUP BY k",
    "a IS DISTINCT FROM 999990 + k OR b IS DISTINCT FROM a + 1");
  compare_winners(
    "SELECT k, first(payload ORDER BY CASE WHEN payload < 500000 THEN NULL ELSE payload END NULLS "
    "FIRST, payload DESC) AS w FROM dist_dup GROUP BY k",
    "w IS DISTINCT FROM 499990 + k");
  compare_winners(
    "SELECT DISTINCT ON (k) k, payload AS w FROM dist_dup ORDER BY CASE WHEN payload >= 500000 "
    "THEN NULL ELSE payload END DESC",
    "w IS DISTINCT FROM 499990 + k");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution ordered FIRST over a VARCHAR key across batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Under string order the smallest `b` of keys 2 to 9 is '10000' || k, in the second scan batch;
  // numeric order and the unordered route keep payload k.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  run_ok("CREATE TABLE dist_vkey AS SELECT k, payload, payload::VARCHAR AS b FROM dist_dup;");
  run_ok("CHECKPOINT;");
  compare_winners("SELECT k, first(payload ORDER BY b) AS w FROM dist_vkey GROUP BY k",
                  "w IS DISTINCT FROM CASE WHEN k < 2 THEN k ELSE 100000 + k END");
  compare_winners("SELECT DISTINCT ON (k) k, payload AS w FROM dist_vkey ORDER BY b",
                  "w IS DISTINCT FROM CASE WHEN k < 2 THEN k ELSE 100000 + k END");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution ordered FIRST over a DOUBLE key with NaN and signed zeros across "
                 "batches",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Each key's smallest `d` is a 0.0 in the first scan batch and a -0.0 in the last, or the
  // reverse; the signs alternate row by row so no run folds them. Tied, `payload DESC` keeps the
  // last batch's row, where a sign-aware order keeps the -0.0 and fails the even keys. Each key's
  // NaN or -NaN sits in the last batch and is the largest `d`.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  run_ok(
    "CREATE TABLE dist_dkey AS SELECT k, payload, CASE "
    "WHEN payload < 10 THEN CASE WHEN k % 2 = 0 THEN -(0.0::DOUBLE) ELSE 0.0::DOUBLE END "
    "WHEN payload >= 999990 THEN CASE WHEN k % 2 = 0 THEN 0.0::DOUBLE ELSE -(0.0::DOUBLE) END "
    "WHEN payload >= 999980 THEN CASE WHEN k % 2 = 0 THEN 'NaN'::DOUBLE ELSE -('NaN'::DOUBLE) END "
    "ELSE payload + 1.0 END AS d FROM dist_dup;");
  run_ok("CHECKPOINT;");
  compare_winners(
    "SELECT k, first(payload ORDER BY d, payload DESC) AS w FROM dist_dkey GROUP BY k",
    "w IS DISTINCT FROM 999990 + k");
  compare_winners("SELECT k, first(payload ORDER BY d DESC) AS w FROM dist_dkey GROUP BY k",
                  "w IS DISTINCT FROM 999980 + k");
  compare_winners("SELECT DISTINCT ON (k) k, payload AS w FROM dist_dkey ORDER BY d DESC",
                  "w IS DISTINCT FROM 999980 + k");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT ON carries several columns across batches and partitions",
                 "[integration][gpu_execution][distinct]")
{
  // Every carried column is a function of `payload`, so a column gathered from another row than
  // the winner's breaks its relation to `w`.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  run_ok(
    "CREATE TABLE dist_wide AS SELECT k, payload, payload * 2 AS a, payload::VARCHAR AS b, "
    "payload % 7 AS c, payload / 2 AS d FROM dist_dup;");
  run_ok("CHECKPOINT;");
  auto const* const winners =
    "SELECT DISTINCT ON (k) k, payload AS w, a, b, c, d FROM dist_wide ORDER BY payload DESC";
  auto const* const mismatch =
    "w IS DISTINCT FROM 999990 + k OR a IS DISTINCT FROM w * 2 OR b IS DISTINCT FROM w::VARCHAR "
    "OR c IS DISTINCT FROM w % 7 OR d IS DISTINCT FROM w / 2";
  compare_winners(winners, mismatch);
  sirius::test::scoped_sirius_setting partitions{*con, "hash_partition_bytes", std::uint64_t{1000}};
  compare_winners(winners, mismatch);
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution integer AVG runs on the GPU without its ORDER BY",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  compare_gpu_vs_cpu("SELECT k, avg(v ORDER BY t DESC) FROM dist_ord GROUP BY k");
  compare_gpu_vs_cpu("SELECT avg(v ORDER BY t) FROM dist_ord");
}

//===----------------------------------------------------------------------===//
// Guarded shapes: plan-time CPU fallback, not a result divergence
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution ordered FIRST with a different ORDER BY falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // One carried row per group cannot answer two row orders.
  expect_plan_fallback_matches_cpu(
    "SELECT k, first(v ORDER BY t), first(v ORDER BY s) FROM dist_ord GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution exported ordered FIRST state falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  expect_plan_fallback_matches_cpu(
    "SELECT k, first(v ORDER BY t) EXPORT_STATE AS s FROM dist_ord GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution floating AVG with an ORDER BY falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // DuckDB adds in the ORDER BY's order, which fixes the rounding the GPU cannot match.
  expect_plan_fallback_matches_cpu("SELECT k, avg(d ORDER BY t) FROM dist_ord GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution ungrouped ordered FIRST and LAST fall back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The smallest and the largest `t` are unique, so both CPU runs agree.
  expect_plan_fallback_matches_cpu("SELECT first(v ORDER BY t) FROM dist_ord");
  expect_plan_fallback_matches_cpu("SELECT last(v ORDER BY t NULLS FIRST) FROM dist_ord");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST and LAST over one ORDER BY fall back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // LAST decodes to a FIRST under the reversed order, and one carried row cannot answer two orders.
  // Every group's smallest `t` is unique, and its last row is a NULL `t` row or a unique largest.
  expect_plan_fallback_matches_cpu(
    "SELECT k, first(v ORDER BY t), last(v ORDER BY t) FROM dist_ord GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST over several grouping sets falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // `k = 1` holds one v, so the grand-total row's first(v) is the same on every run.
  SECTION("ROLLUP")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v), sum(v) FROM dist_fd WHERE k = 1 GROUP BY ROLLUP (k)");
  }

  SECTION("GROUPING()")
  {
    expect_plan_fallback_matches_cpu(
      "SELECT k, first(v), GROUPING(k) FROM dist_fd GROUP BY GROUPING SETS ((k))");
  }

  // The refusal is FIRST's alone.
  SECTION("ROLLUP without FIRST still runs")
  {
    compare_gpu_vs_cpu("SELECT k, sum(v) FROM dist_fd GROUP BY ROLLUP (k)");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST beside a filtered SUM or an unsupported DISTINCT sum",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The filtered SUM masks only its own input; FIRST must still see every row in the group.
  // DISTINCT remains supported only for COUNT.
  SECTION("FILTER")
  {
    compare_gpu_vs_cpu("SELECT k, first(v), sum(v) FILTER (WHERE v > 15) FROM dist_fd GROUP BY k");
  }

  SECTION("DISTINCT sum")
  {
    expect_plan_fallback_matches_cpu("SELECT k, first(v), sum(DISTINCT v) FROM dist_fd GROUP BY k");
  }
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST with a FILTER falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The CPU answer is (1, NULL) (2, 20) (3, 30) (NULL, NULL). A route that ignored the filter would
  // return (1, 10), so a result comparison alone would also catch the defect. FIRST does not skip
  // NULLs, so the mask-to-null lowering used by supported filtered aggregates cannot answer it.
  expect_plan_fallback_matches_cpu(
    "SELECT k, first(v) FILTER (WHERE v > 15) FROM dist_fd GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution ungrouped FIRST with a FILTER falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // Only v = 30 passes the filter, so the CPU answer is deterministic. Ignoring the filter would
  // return whichever row comes first. Pins `main`'s refusal of every aggregate FILTER, kept for
  // when it lifts.
  expect_plan_fallback_matches_cpu("SELECT first(v) FILTER (WHERE v > 25) FROM dist_fd");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution arg_min_null with its own ORDER BY falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // `w` breaks the ties on `t`: DuckDB keeps v 20 and v 50, where decoding the sort key as the
  // whole order would keep v 10 and v 40.
  run_ok(
    "CREATE TABLE dist_argmin AS SELECT * FROM (VALUES (1, 5, 3, 10), (1, 5, 1, 20), "
    "(1, 7, 0, 30), (2, 5, 9, 40), (2, 5, 2, 50)) x(k, t, w, v);");
  run_ok("CHECKPOINT;");
  expect_plan_fallback_matches_cpu(
    "SELECT k, arg_min_null(v, create_sort_key(t, 'ASC NULLS LAST') ORDER BY w) FROM dist_argmin "
    "GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a collated VARCHAR key falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // Binder::BindModifiers pushes a collation over every distinct target, so under a non-binary
  // default_collation the key arrives as a call rather than as a bare reference and the builder's
  // uncovered-output guard refuses it. Every `s_short` value is already lower case, so each nocase
  // group holds one original value and the two CPU runs cannot disagree about which row it is.
  auto const prior_result = con->Query("SELECT current_setting('default_collation');");
  REQUIRE(prior_result);
  REQUIRE_FALSE(prior_result->HasError());
  auto const prior = prior_result->Collection().GetValue(0, 0);
  {
    sirius::test::scoped_sirius_setting collation{*con, "default_collation", "nocase"};
    expect_plan_fallback_matches_cpu("SELECT DISTINCT s_short FROM dist_t");
  }
  // A failed restore would leak nocase into every later integration case.
  REQUIRE(con->Query("SELECT current_setting('default_collation');")->Collection().GetValue(0, 0) ==
          prior);
}

/// Floating-point keys carrying every value whose equality is decided by something other than
/// bit equality: NaN against -NaN (different bit patterns, one DuckDB group) and +0.0 against
/// -0.0 (different bit patterns, one DuckDB group). Separate from DistinctFixture so the
/// integer-keyed cases keep a table whose every column is exactly comparable.
class DistinctFloatFixture : public sirius::test::GpuExecutionFixture {
 public:
  DistinctFloatFixture()
  {
    run_ok("CREATE TABLE dist_fp (d DOUBLE, f REAL);");
    run_ok(
      "INSERT INTO dist_fp VALUES "
      "('NaN'::DOUBLE,       'NaN'::REAL),"
      "('NaN'::DOUBLE,       'NaN'::REAL),"     // duplicate NaN: must collapse
      "(-('NaN'::DOUBLE),    -('NaN'::REAL)),"  // sign bit set: same group as NaN
      "(0.0,                 0.0),"
      "('Infinity'::DOUBLE,  'Infinity'::REAL),"
      "(-(0.0::DOUBLE),      -(0.0::REAL)),"  // same group as +0.0; not beside it, see below
      "('-Infinity'::DOUBLE, '-Infinity'::REAL),"
      "(1.5,                 1.5),"
      "(1.5,                 1.5),"
      "(NULL,                NULL);");
    run_ok("CHECKPOINT;");

    // A `-0.0` literal parses as DECIMAL, and RLE at CHECKPOINT folds a -0.0 into an adjacent
    // equal 0.0, so prove each column stored a negative zero. Read on the CPU so the check does not
    // depend on the code under test, and restore GPU execution before asserting: the connection is
    // shared, so a failed REQUIRE must not leave it off for every later case.
    run_ok("SET gpu_execution = false;");
    auto const d_zeros = con->Query("SELECT count(*) FROM dist_fp WHERE d = 0 AND signbit(d)");
    auto const f_zeros = con->Query("SELECT count(*) FROM dist_fp WHERE f = 0 AND signbit(f)");
    run_ok("SET gpu_execution = true;");
    for (auto* zeros : {d_zeros.get(), f_zeros.get()}) {
      REQUIRE(zeros);
      REQUIRE_FALSE(zeros->HasError());
      REQUIRE(zeros->Collection().GetValue(0, 0).GetValue<int64_t>() == 1);
    }
  }
};

TEST_CASE_METHOD(DistinctFloatFixture,
                 "gpu_execution DISTINCT over a floating-point key groups NaN and signed zero as "
                 "DuckDB does",
                 "[integration][gpu_execution][distinct]")
{
  // cudf::groupby's row comparator decides NaN == NaN and -0.0 == 0.0 on its own terms; DuckDB
  // groups both pairs. A disagreement is a wrong answer, not a fallback: the exact row count
  // catches a pair that fails to collapse, and the keys are compared with each pair's two spellings
  // treated as one, because either engine may keep either member.
  SECTION("DOUBLE") { compare_gpu_vs_cpu_canonical("SELECT DISTINCT d FROM dist_fp"); }

  SECTION("REAL") { compare_gpu_vs_cpu_canonical("SELECT DISTINCT f FROM dist_fp"); }

  // A composite key runs the same comparator over a two-column row, which is the shape the
  // single-column cases cannot reach.
  SECTION("both columns") { compare_gpu_vs_cpu_canonical("SELECT DISTINCT d, f FROM dist_fp"); }

  // Exact, not canonicalized: the sections above prove nothing if the scan drops the sign bit.
  SECTION("the GPU scan keeps a negative zero")
  {
    compare_gpu_vs_cpu("SELECT d, f FROM dist_fp WHERE d = 0");
  }
}

TEST_CASE_METHOD(DistinctFloatFixture,
                 "gpu_execution DISTINCT ON a floating-point key groups NaN and signed zero as "
                 "DuckDB does",
                 "[integration][gpu_execution][distinct]")
{
  // cudf::distinct decides key equality separately from cudf::groupby, so the case above says
  // nothing about this route. Within each collapsed group the carried values differ only as NaN and
  // -NaN or 0.0 and -0.0, which the canonical comparison treats as one value, so the carried column
  // is compared too.
  SECTION("DOUBLE key")
  {
    compare_gpu_vs_cpu_canonical("SELECT DISTINCT ON (d) d, f FROM dist_fp");
  }

  SECTION("REAL key") { compare_gpu_vs_cpu_canonical("SELECT DISTINCT ON (f) f, d FROM dist_fp"); }
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution FIRST beside SUM over a dictionary-encoded merge key",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // 2M rows give the merge enough partial rows for `s` to pass both dictionary-encode gates in
  // merge_grouped_aggregate: avg_len >= 8.0 and NDV/rows < 0.10.
  sirius::test::scoped_sirius_setting batch_size{
    *con, "scan_task_batch_size", std::uint64_t{1048576}};
  run_ok(
    "CREATE TABLE dist_str_first AS "
    "SELECT 'a_long_string_value_' || (i % 20) AS s, i % 20 AS k, i AS v "
    "FROM range(2000000) t(i);");
  run_ok("CHECKPOINT;");
  // first(v) is any row of its group, so only the key and the sum are compared.
  compare_gpu_vs_cpu_on_columns("SELECT s, first(v), sum(v) FROM dist_str_first GROUP BY s",
                                {0, 2});
  // v % 20 equals its row's k, so a FIRST gathered from another group is a violation.
  require_groups_hold(
    *this,
    "SELECT s, first(v) AS f, min(k) AS mk, sum(v) AS t FROM dist_str_first GROUP BY s",
    "f % 20 IS DISTINCT FROM mk",
    20);
}
