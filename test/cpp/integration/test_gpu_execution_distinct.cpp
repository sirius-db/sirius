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
 * Every DISTINCT guard throws during `create_plan`, before any GPU work is scheduled, so the
 * guarded shapes use `expect_plan_fallback_matches_cpu` and assert the plan-time fallback counter
 * rather than the runtime one.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

namespace {

/// Sets a session/global setting for the enclosing scope and resets it on the way out, including
/// when a REQUIRE fails and unwinds. Every `[integration]` fixture borrows its connection from the
/// shared `g_integration_env`, which outlives the case, and `default_collation` is
/// GLOBAL_DEFAULT-scoped -- so a setting left behind by a failed assertion would reach every later
/// integration case in the binary.
class scoped_setting {
 public:
  /// @p literal is spliced into the SET statement verbatim, so string values carry their quotes.
  scoped_setting(sirius::test::GpuExecutionFixture& fixture,
                 std::string name,
                 std::string const& literal)
    : fixture_(fixture), name_(std::move(name))
  {
    fixture_.run_ok("SET " + name_ + " = " + literal + ";");
  }

  /// Resets through the raw connection rather than run_ok(): a Catch2 assertion during unwinding
  /// terminates, and a poisoned connection here is already being reported by the failure above.
  ~scoped_setting() { fixture_.con->Query("RESET " + name_ + ";"); }

  scoped_setting(scoped_setting const&)            = delete;
  scoped_setting& operator=(scoped_setting const&) = delete;
  scoped_setting(scoped_setting&&)                 = delete;
  scoped_setting& operator=(scoped_setting&&)      = delete;

 private:
  sirius::test::GpuExecutionFixture& fixture_;
  std::string name_;
};

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
    // `v` is a function of `k`, so a DISTINCT ON over `k` has one correct answer whichever row
    // represents a group. The guarded shapes need that: they compare a fallback run against a
    // second CPU run, and DISTINCT ON without ORDER BY may pick a different row each time.
    run_ok("CREATE TABLE dist_fd (k INTEGER, v INTEGER);");
    run_ok("INSERT INTO dist_fd VALUES (1, 10), (1, 10), (2, 20), (2, 20), (3, 30), (NULL, NULL);");
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
};

/// Run @p query with gpu_execution on: it must succeed via a plan-time CPU fallback, moving the
/// fallback counter but not the execution counter, and return exactly the CPU results.
void expect_plan_fallback_matches_cpu(sirius::test::GpuExecutionFixture& fx,
                                      std::string const& query)
{
  fx.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto result       = fx.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(after.fallbacks == before.fallbacks + 1);
  REQUIRE(after.executions == before.executions);

  fx.run_ok("SET gpu_execution = false;");
  auto cpu_result = fx.con->Query(query);
  fx.run_ok("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  auto rows = sirius::test::GpuExecutionFixture::collect_rows(
    result->Cast<duckdb::MaterializedQueryResult>());
  auto cpu_rows = sirius::test::GpuExecutionFixture::collect_rows(
    cpu_result->Cast<duckdb::MaterializedQueryResult>());
  REQUIRE(rows == cpu_rows);
}

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
  compare_gpu_vs_cpu("SELECT DISTINCT k FROM dist_dup");
}

//===----------------------------------------------------------------------===//
// Guarded shapes: plan-time CPU fallback, not a result divergence
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with ORDER BY falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // This names a specific row per group, and a hash-partitioned dedup would return an arbitrary
  // one: a plausible wrong answer rather than an error.
  expect_plan_fallback_matches_cpu(
    *this, "SELECT DISTINCT ON (k) k, v FROM dist_fd ORDER BY v NULLS LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with carried columns falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // Carrying `v` out of each `k` group needs a grouped FIRST, which Sirius cannot lower yet.
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT ON (k) k, v FROM dist_fd");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT with an ORDER BY outside the select list falls back",
                 "[integration][gpu_execution][distinct]")
{
  // The binder synthesizes one distinct target per select-list entry and only then hoists `v` into
  // the select list to order by it, so the node is two columns wide with a single target and `v`
  // would need a grouped FIRST.
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT k FROM dist_fd ORDER BY v NULLS LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a collated VARCHAR key falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // Binder::BindModifiers pushes a collation over every distinct target, so under a non-binary
  // default_collation the key arrives as a call rather than as a bare reference and the builder's
  // uncovered-output guard refuses it. Every `s_short` value is already lower case, so each nocase
  // group holds one original value and the two CPU runs cannot disagree about which row it is.
  scoped_setting collation(*this, "default_collation", "'nocase'");
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT s_short FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON an expression key falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT ON (k + v) k, v FROM dist_fd");
}
