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
 * comparison is only valid over `dist_fd`, whose carried column is a function of its key. Elsewhere
 * `compare_gpu_vs_cpu_on_keys` compares the row count and the key columns alone. The floating-point
 * cases use it too, because a group holding both 0.0 and -0.0, or NaN and -NaN, prints as whichever
 * member each engine keeps.
 *
 * Every DISTINCT guard throws during `create_plan`, before any GPU work is scheduled, so the
 * guarded shapes use `expect_plan_fallback_matches_cpu` and assert the plan-time fallback counter
 * rather than the runtime one.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <string>
#include <vector>

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

/// NaN and -NaN, and 0.0 and -0.0, are one group to both engines and print differently, so either
/// engine may report either member. Maps each pair to one spelling.
std::string canonical_key(std::string cell)
{
  if (cell == "-nan") { return "nan"; }
  if (cell.size() > 1 && cell.front() == '-' &&
      cell.find_first_not_of("0.", 1) == std::string::npos) {
    cell.erase(0, 1);
  }
  return cell;
}

/// Runs @p query on the GPU and on the CPU and compares the row count and the @p key_columns of
/// each row, after asserting one GPU execution with no fallback. Reads the query's own result
/// rather than wrapping it: an outer query that drops the carried columns lets the optimizer prune
/// them below the DISTINCT ON, leaving a key-only dedup with nothing carried.
void compare_gpu_vs_cpu_on_keys(sirius::test::GpuExecutionFixture& fixture,
                                std::string const& query,
                                std::vector<std::size_t> const& key_columns)
{
  fixture.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*fixture.con);
  auto gpu_result   = fixture.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fixture.con);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("GPU execution error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

  fixture.run_ok("SET gpu_execution = false;");
  auto cpu_result = fixture.con->Query(query);
  fixture.run_ok("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());
  auto const keys_of = [&key_columns](duckdb::MaterializedQueryResult& result) {
    std::vector<std::vector<std::string>> keys;
    for (auto const& row : sirius::test::collect_rows(result)) {
      std::vector<std::string> key;
      for (auto const column : key_columns) {
        REQUIRE(column < row.size());
        key.push_back(canonical_key(row[column]));
      }
      keys.push_back(std::move(key));
    }
    std::sort(keys.begin(), keys.end());
    return keys;
  };
  REQUIRE(keys_of(gpu_result->Cast<duckdb::MaterializedQueryResult>()) ==
          keys_of(cpu_result->Cast<duckdb::MaterializedQueryResult>()));
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
    // `v` is a function of `k`, so a DISTINCT ON over `k` has one correct answer whichever row
    // represents a group. The carried and guarded shapes need that: DISTINCT ON without ORDER BY
    // may pick a different row on each run and on each engine.
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
  compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT ON (b, a) a, b, s_short FROM dist_t", {0, 1});
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution GROUP BY with only FIRST aggregates runs on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The operator cannot tell this from a lowered DISTINCT ON, so it takes the same route.
  compare_gpu_vs_cpu("SELECT k, first(v) FROM dist_fd GROUP BY k");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST beside another aggregate runs on the GPU",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  compare_gpu_vs_cpu("SELECT k, first(v), sum(v) FROM dist_fd GROUP BY k");
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

  // A second correlated predicate keeps the delim join, so the DISTINCT reads a join against the
  // DELIM_GET.
  SECTION("delim join")
  {
    compare_gpu_vs_cpu(
      "SELECT r.a, r.x, s.b FROM dist_r r, "
      "LATERAL (SELECT DISTINCT t.b FROM dist_t t WHERE t.a = r.a AND t.b < r.x) s");
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
    scoped_setting batch_size(*this, "scan_task_batch_size", "1048576");
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
    compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT ON (k) k, payload FROM dist_dup", {0});
  }

  // Small scan batches give a partition more than one local result, which the merge's single-batch
  // shortcut would otherwise skip. With the key second, group_idx is {1}: a merge that keyed on
  // group_idx rather than on its leading column would dedup on `payload` and return every row.
  SECTION("many scan batches")
  {
    scoped_setting batch_size(*this, "scan_task_batch_size", "1048576");
    compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT ON (k) k, payload FROM dist_dup", {0});
    compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT ON (k) payload, k FROM dist_dup", {1});
  }
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
    "SELECT DISTINCT ON (k) k, v FROM dist_fd ORDER BY v NULLS LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution FIRST with a FILTER falls back at plan time",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  // The CPU answer is (1, NULL) (2, 20) (3, 30) (NULL, NULL). A route that ignored
  // the filter would return (1, 10), so a result comparison alone would also catch the defect.
  expect_plan_fallback_matches_cpu(
    "SELECT k, first(v) FILTER (WHERE v > 15) FROM dist_fd GROUP BY k");
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
  expect_plan_fallback_matches_cpu("SELECT DISTINCT s_short FROM dist_t");
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
      REQUIRE(zeros->GetValue(0, 0).GetValue<int64_t>() == 1);
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
  SECTION("DOUBLE") { compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT d FROM dist_fp", {0}); }

  SECTION("REAL") { compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT f FROM dist_fp", {0}); }

  // A composite key runs the same comparator over a two-column row, which is the shape the
  // single-column cases cannot reach.
  SECTION("both columns")
  {
    compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT d, f FROM dist_fp", {0, 1});
  }

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
  // nothing about this route. Each collapsed group holds rows whose carried values differ, so only
  // the count and the keys are compared.
  SECTION("DOUBLE key")
  {
    compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT ON (d) d, f FROM dist_fp", {0});
  }

  SECTION("REAL key")
  {
    compare_gpu_vs_cpu_on_keys(*this, "SELECT DISTINCT ON (f) f, d FROM dist_fp", {0});
  }
}
