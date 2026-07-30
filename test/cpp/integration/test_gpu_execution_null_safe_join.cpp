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

// GPU-vs-CPU correctness for null-safe joins: a join keyed on IS NOT DISTINCT
// FROM must match NULL to NULL, unlike a plain '=' join. Previously the GPU hash
// join hardcoded cudf::null_equality::UNEQUAL for every condition, so NULL keys
// never matched and IS NOT DISTINCT FROM joins silently undercounted. The fix
// passes null_equality::EQUAL when the equi-key condition is null-safe.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <utils/gpu_execution_fixture.hpp>

namespace {

// RAII: disable a DuckDB optimizer for the current scope and restore it on exit
// (disabled_optimizers is database-level, so restoring avoids affecting other
// tests on the shared connection).
struct disabled_optimizer_guard {
  duckdb::ClientContext& ctx;
  duckdb::set<duckdb::OptimizerType> saved;
  disabled_optimizer_guard(duckdb::ClientContext& c, duckdb::OptimizerType type) : ctx(c)
  {
    auto& opts = duckdb::DBConfig::GetConfig(ctx).options;
    saved      = opts.disabled_optimizers;
    opts.disabled_optimizers.insert(type);
  }
  ~disabled_optimizer_guard()
  {
    duckdb::DBConfig::GetConfig(ctx).options.disabled_optimizers = saved;
  }
};

// Both sides carry NULL keys (so NULL-to-NULL matching is exercised) plus a
// matching non-NULL key and non-matching keys on each side.
class NullSafeJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  NullSafeJoinFixture()
  {
    run_ok("CREATE TABLE l (id INTEGER, k INTEGER);");
    run_ok("CREATE TABLE r (id INTEGER, k INTEGER);");
    run_ok("INSERT INTO l VALUES (1, 10), (2, NULL), (3, NULL), (4, 20);");
    run_ok("INSERT INTO r VALUES (100, 10), (101, NULL), (102, 30);");
    run_ok("CHECKPOINT;");
  }
};

// A join mixing plain '=' with IS NOT DISTINCT FROM (as delim joins for correlated
// subqueries do). `b` has no NULLs, so the result is independent of null semantics;
// this fixture guards that the mixed-key join stays on the GPU. NULL-to-NULL matching
// on the null-safe key is covered by MixedKeyNullSafeJoinFixture below.
class MixedKeyJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  MixedKeyJoinFixture()
  {
    run_ok("CREATE TABLE ml (a INTEGER, b INTEGER);");
    run_ok("CREATE TABLE mr (a INTEGER, b INTEGER);");
    run_ok("INSERT INTO ml VALUES (10, 100), (10, 200), (20, 100);");
    run_ok("INSERT INTO mr VALUES (10, 100), (10, 999), (20, 100);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(NullSafeJoinFixture,
                 "gpu_execution null-safe join matches NULL to NULL (IS NOT DISTINCT FROM)",
                 "[integration][gpu_execution][join][nulls]")
{
  // NULL keys on both sides match each other: (l.k=10↔r.k=10) plus the NULL rows
  // (l ids 2,3) × (r id 101) = 3 matched rows, vs 1 for a plain '=' join.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON l.k IS NOT DISTINCT FROM r.k");
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON l.k IS NOT DISTINCT FROM r.k");
}

TEST_CASE_METHOD(NullSafeJoinFixture,
                 "gpu_execution plain '=' join still drops NULL keys",
                 "[integration][gpu_execution][join][nulls]")
{
  // Regression guard: null-safe handling must not leak into ordinary equality
  // joins, where NULL <> NULL means the NULL-keyed rows do not match.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON l.k = r.k");
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON l.k = r.k");
}

TEST_CASE_METHOD(NullSafeJoinFixture,
                 "gpu_execution null-safe LEFT join matches NULL and pads unmatched",
                 "[integration][gpu_execution][join][nulls]")
{
  // Every left row is emitted; its NULL key matches r's NULL key, while the
  // unmatched left row (k=20) is NULL-padded on the right.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON l.k IS NOT DISTINCT FROM r.k");
}

TEST_CASE_METHOD(NullSafeJoinFixture,
                 "gpu_execution null-safe FULL OUTER join matches NULL to NULL",
                 "[integration][gpu_execution][join][nulls]")
{
  // FULL OUTER is symmetric, so DuckDB can't rewrite it away -- it genuinely
  // exercises the full-outer path with a null-safe key.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l FULL OUTER JOIN r ON l.k IS NOT DISTINCT FROM r.k");
}

TEST_CASE_METHOD(NullSafeJoinFixture,
                 "gpu_execution null-safe RIGHT join hits the right-family path (no swap)",
                 "[integration][gpu_execution][join][nulls]")
{
  // DuckDB normally lowers RIGHT JOIN to a swapped LEFT JOIN, which would hide the
  // hash join's right-family path. Disable the build-side/probe-side optimizer so the
  // RIGHT join type is preserved, forcing that path with a null-safe key.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l RIGHT JOIN r ON l.k IS NOT DISTINCT FROM r.k");
}

TEST_CASE_METHOD(MixedKeyJoinFixture,
                 "gpu_execution mixed '=' and IS NOT DISTINCT FROM join runs on GPU",
                 "[integration][gpu_execution][join][nulls]")
{
  // Regression guard: a join mixing '=' with IS NOT DISTINCT FROM must stay on the
  // GPU and must NOT be rejected -- rejecting it routed delim joins to the
  // unsupported nested-loop path. `b` has no NULLs, so the result is the same under
  // either null semantics; this only asserts the mixed-key join runs on the GPU.
  compare_gpu_vs_cpu(
    "SELECT ml.a, ml.b FROM ml JOIN mr ON ml.a = mr.a AND ml.b IS NOT DISTINCT FROM mr.b");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ml JOIN mr ON ml.a = mr.a AND ml.b IS NOT DISTINCT FROM mr.b");
}

// A join mixing a plain '=' key with a null-safe (IS NOT DISTINCT FROM) key where the
// null-safe column carries NULLs on both sides. A single cuDF null_equality flag can't
// serve both keys at once -- the '=' key needs UNEQUAL (NULL != NULL) while the null-safe
// key needs EQUAL (NULL == NULL) -- so the old single-flag path forced UNEQUAL and
// silently dropped the NULL-to-NULL matches on `b`. The fix keeps '=' as the (UNEQUAL)
// hash key and moves the null-safe key into a NULL_EQUAL predicate on a cuDF mixed join.
class MixedKeyNullSafeJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  MixedKeyNullSafeJoinFixture()
  {
    run_ok("CREATE TABLE mnl (id INTEGER, a INTEGER, b INTEGER);");
    run_ok("CREATE TABLE mnr (id INTEGER, a INTEGER, b INTEGER);");
    // For non-NULL `a`, `b` mixes a matching non-NULL pair, two NULL-to-NULL
    // pairs (which plain '=' would drop), and a non-matching (300 vs 999) pair.
    // Row mnl(5) has no `a` partner. The final pair has matching b=500 but NULL
    // in the plain '=' key `a`; it must not match because NULL = NULL is unknown.
    run_ok(
      "INSERT INTO mnl VALUES (1, 10, 100), (2, 10, NULL), (3, 20, NULL), (4, 30, 300), "
      "(5, 40, NULL), (6, NULL, 500);");
    run_ok(
      "INSERT INTO mnr VALUES (100, 10, 100), (101, 10, NULL), (102, 20, NULL), "
      "(103, 30, 999), (104, NULL, 500);");
    run_ok("CHECKPOINT;");
  }
};

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe INNER join matches NULL to NULL",
                 "[integration][gpu_execution][join][nulls]")
{
  // Matches: (a=10,b=100), (a=10,b=NULL), (a=20,b=NULL) = 3 rows. A plain '=' on `b`
  // would drop the two NULL-to-NULL pairs and return only 1; NULL_EQUAL on the hash key
  // would incorrectly add the (a=NULL,b=500) pair and return 4.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, mnr.id FROM mnl JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM mnl JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe LEFT join matches NULL and pads unmatched",
                 "[integration][gpu_execution][join][nulls]")
{
  // All six left rows are emitted; rows 4, 5, and 6 are NULL-padded on the right.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, mnr.id FROM mnl LEFT JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe SEMI and ANTI joins",
                 "[integration][gpu_execution][join][nulls]")
{
  // Keep the output side on the left so these queries exercise mixed_left_semi_join
  // and mixed_left_anti_join. SEMI returns rows 1-3; ANTI returns rows 4-6,
  // including row 6 because NULL must not match in the plain '=' key.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu(
    "SELECT mnl.id FROM mnl SEMI JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
  compare_gpu_vs_cpu(
    "SELECT mnl.id FROM mnl ANTI JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe FULL OUTER join matches NULL to NULL",
                 "[integration][gpu_execution][join][nulls]")
{
  // FULL OUTER is symmetric, so DuckDB can't rewrite it away -- it exercises the
  // mixed full-outer path with a null-safe key.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, mnr.id FROM mnl FULL OUTER JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

TEST_CASE_METHOD(
  MixedKeyNullSafeJoinFixture,
  "gpu_execution mixed '=' + null-safe RIGHT join hits the right-family path (no swap)",
  "[integration][gpu_execution][join][nulls]")
{
  // Disable the build-side/probe-side optimizer so DuckDB preserves the RIGHT join
  // type instead of lowering it to a swapped LEFT join, forcing the mixed right-family
  // path with a null-safe key.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu(
    "SELECT mnl.id, mnr.id FROM mnl RIGHT JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

// Documents the remaining null-safe limitation. A MARK-family join (here a projected
// correlated EXISTS) can't run in MIXED_JOIN mode, so its null-safe key is NOT routed
// to a NULL_EQUAL predicate -- it stays a UNEQUAL hash key and NULL-to-NULL matches can
// be dropped on the GPU (see the compare_nulls_ note in the sirius_physical_hash_join
// ctor). Tagged [!mayfail] because the observable behavior (silent GPU/CPU divergence
// vs. CPU fallback) depends on the delim-join plan shape; tighten to a hard assertion
// (or expect_gpu_fallback) once null-safe MARK joins are supported.
TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution null-safe MARK join (projected EXISTS) is a known limitation",
                 "[integration][gpu_execution][join][nulls][!mayfail]")
{
  compare_gpu_vs_cpu(
    "SELECT mnl.id, EXISTS (SELECT 1 FROM mnr "
    "WHERE mnr.a = mnl.a AND mnr.b IS NOT DISTINCT FROM mnl.b) AS has_match "
    "FROM mnl");
}
