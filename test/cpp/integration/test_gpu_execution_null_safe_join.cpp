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
  // Keep null-free mixed keys on GPU; rejecting them breaks delim joins.
  compare_gpu_vs_cpu(
    "SELECT ml.a, ml.b FROM ml JOIN mr ON ml.a = mr.a AND ml.b IS NOT DISTINCT FROM mr.b");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ml JOIN mr ON ml.a = mr.a AND ml.b IS NOT DISTINCT FROM mr.b");
}

// Mixed plain and null-safe keys require UNEQUAL hashing plus a NULL_EQUAL predicate.
class MixedKeyNullSafeJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  MixedKeyNullSafeJoinFixture()
  {
    run_ok("CREATE TABLE mnl (id INTEGER, a INTEGER, b INTEGER);");
    run_ok("CREATE TABLE mnr (id INTEGER, a INTEGER, b INTEGER);");
    // Includes null-safe NULL matches and a NULL pair in the plain key that must not match.
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
  // Expected matches: one non-NULL pair and two null-safe NULL pairs.
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
  // Preserve left SEMI/ANTI planning; row 6 must not match on its plain NULL key.
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
  // FULL OUTER cannot be rewritten to a swapped join.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, mnr.id FROM mnl FULL OUTER JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

TEST_CASE_METHOD(
  MixedKeyNullSafeJoinFixture,
  "gpu_execution mixed '=' + null-safe RIGHT join hits the right-family path (no swap)",
  "[integration][gpu_execution][join][nulls]")
{
  // Preserve RIGHT instead of lowering it to a swapped LEFT join.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu(
    "SELECT mnl.id, mnr.id FROM mnl RIGHT JOIN mnr "
    "ON mnl.a = mnr.a AND mnl.b IS NOT DISTINCT FROM mnr.b");
}

// `b` requires materializing an AST-unsupported SMALLINT-to-INTEGER cast; `c` exercises
// the inline SMALLINT-to-BIGINT cast supported by cuDF AST.
class MixedTypeNullSafeJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  MixedTypeNullSafeJoinFixture()
  {
    run_ok("CREATE TABLE mtl (id INTEGER, a INTEGER, b SMALLINT, c SMALLINT);");
    run_ok("CREATE TABLE mtr (id INTEGER, a INTEGER, b INTEGER, c BIGINT);");
    // Same null-matching cases as MixedKeyNullSafeJoinFixture, with mixed key types.
    run_ok(
      "INSERT INTO mtl VALUES (1, 10, 100, 100), (2, 10, NULL, NULL), (3, 20, NULL, NULL), "
      "(4, 30, 300, 300), (5, 40, NULL, NULL), (6, NULL, 500, 500);");
    run_ok(
      "INSERT INTO mtr VALUES (100, 10, 100, 100), (101, 10, NULL, NULL), (102, 20, NULL, NULL), "
      "(103, 30, 999, 999), (104, NULL, 500, 500);");
    run_ok("CHECKPOINT;");
  }
};

TEST_CASE_METHOD(MixedTypeNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe join with an AST-untranslatable cast",
                 "[integration][gpu_execution][join][nulls]")
{
  // SMALLINT-to-INTEGER must be materialized while preserving three expected matches.
  compare_gpu_vs_cpu(
    "SELECT mtl.id, mtr.id FROM mtl JOIN mtr "
    "ON mtl.a = mtr.a AND mtl.b IS NOT DISTINCT FROM mtr.b");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM mtl JOIN mtr "
    "ON mtl.a = mtr.a AND mtl.b IS NOT DISTINCT FROM mtr.b");
}

TEST_CASE_METHOD(MixedTypeNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe join with a cast and no NULLs",
                 "[integration][gpu_execution][join][nulls]")
{
  // The pre-routing, null-free case must continue to run without fallback.
  compare_gpu_vs_cpu(
    "SELECT mtl.id, mtr.id FROM mtl JOIN mtr "
    "ON mtl.a = mtr.a AND mtl.b IS NOT DISTINCT FROM mtr.b "
    "WHERE mtl.b IS NOT NULL AND mtr.b IS NOT NULL");
}

TEST_CASE_METHOD(MixedTypeNullSafeJoinFixture,
                 "gpu_execution mixed '=' + null-safe join with an AST-translatable cast",
                 "[integration][gpu_execution][join][nulls]")
{
  // SMALLINT-to-BIGINT remains inline because cuDF AST supports the target type.
  compare_gpu_vs_cpu(
    "SELECT mtl.id, mtr.id FROM mtl JOIN mtr "
    "ON mtl.a = mtr.a AND mtl.c IS NOT DISTINCT FROM mtr.c");
}

TEST_CASE_METHOD(MixedTypeNullSafeJoinFixture,
                 "gpu_execution mixed-type null-safe join does not leak the materialized key",
                 "[integration][gpu_execution][join][nulls]")
{
  // SELECT * catches synthetic key columns leaking into join output.
  compare_gpu_vs_cpu(
    "SELECT * FROM mtl JOIN mtr ON mtl.a = mtr.a AND mtl.b IS NOT DISTINCT FROM mtr.b");
  compare_gpu_vs_cpu(
    "SELECT * FROM mtl LEFT JOIN mtr ON mtl.a = mtr.a AND mtl.b IS NOT DISTINCT FROM mtr.b");
}

// A correlated IN is the one MARK shape that genuinely mixes key policies: decorrelation makes
// the correlated predicate null-safe while the IN operand stays a plain `=`, giving conditions
// `(a IS NOT DISTINCT FROM a)` AND `(b = #0)`. That mixture is rejected at plan time.
TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution mixed plain + null-safe MARK join falls back at plan time",
                 "[integration][gpu_execution][join][nulls]")
{
  expect_plan_fallback_matches_cpu(
    "SELECT mnl.id, mnl.b IN (SELECT mnr.b FROM mnr WHERE mnr.a = mnl.a) AS in_b FROM mnl");
}

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution all-null-safe MARK join runs on GPU with definite marks",
                 "[integration][gpu_execution][join][nulls]")
{
  // Every key null-safe => EQUAL matching and definite marks. mnl rows 2, 3 and 5 have a NULL
  // `b` that must match mnr's NULL `b`.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, EXISTS (SELECT 1 FROM mnr "
    "WHERE mnr.b IS NOT DISTINCT FROM mnl.b) AS has_match "
    "FROM mnl");
  // Both correlated predicates decorrelate to null-safe delim keys, so this is all-null-safe
  // too -- it is the query that was [!mayfail] before.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, EXISTS (SELECT 1 FROM mnr "
    "WHERE mnr.a = mnl.a AND mnr.b IS NOT DISTINCT FROM mnl.b) AS has_match "
    "FROM mnl");
}

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution correlated projected EXISTS keeps definite marks on GPU",
                 "[integration][gpu_execution][join][nulls]")
{
  // The common shape this path exists for: DuckDB decorrelates a projected correlated EXISTS
  // into an all-null-safe MARK over a build side pre-filtered to `a IS NOT NULL`. The null-safe
  // key is what makes an unmatched row a definite FALSE, so under the old UNEQUAL pin mnl row 6
  // (a IS NULL, unmatched) came back NULL instead of false.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, EXISTS (SELECT 1 FROM mnr WHERE mnr.a = mnl.a) AS has_match FROM mnl");
  // NOT EXISTS negates the same mark; a stray NULL would surface here as a missing row.
  compare_gpu_vs_cpu(
    "SELECT mnl.id, NOT EXISTS (SELECT 1 FROM mnr WHERE mnr.a = mnl.a) AS no_match FROM mnl");
}

TEST_CASE_METHOD(MixedKeyNullSafeJoinFixture,
                 "gpu_execution null-safe MARK rejection does not spill onto plain IN/EXISTS",
                 "[integration][gpu_execution][join][nulls]")
{
  // Regression guard on the plan-time screen: an uncorrelated MARK, whose only key is a plain
  // '=', keeps running on the GPU.
  compare_gpu_vs_cpu("SELECT mnl.id, mnl.b IN (SELECT mnr.b FROM mnr) AS in_b FROM mnl");
  compare_gpu_vs_cpu("SELECT mnl.id, mnl.a IN (SELECT mnr.a FROM mnr) AS in_a FROM mnl");
}

// Below, every key is IS NOT DISTINCT FROM, the join a set operation lowers to. Each case disables
// BUILD_SIDE_PROBE_SIDE so the join keeps its left-side type instead of flipping to a RIGHT_ form.

// Two-column keys where a NULL in either column must still match the same NULL on the other side.
class PartialNullKeyJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  PartialNullKeyJoinFixture()
  {
    run_ok("CREATE TABLE pl (id INTEGER, a INTEGER, b INTEGER);");
    run_ok("CREATE TABLE pr (id INTEGER, a INTEGER, b INTEGER);");
    run_ok("INSERT INTO pl VALUES (1, 1, NULL), (2, NULL, NULL), (3, 1, 2);");
    run_ok("INSERT INTO pr VALUES (100, 1, NULL), (101, NULL, NULL), (102, 1, 3);");
    run_ok("CHECKPOINT;");
  }
};

// DuckDB compares -0.0 equal to 0.0 and NaN equal to NaN; the GPU join must too.
class FloatKeyJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  FloatKeyJoinFixture()
  {
    run_ok("CREATE TABLE fl (id INTEGER, k DOUBLE);");
    run_ok("CREATE TABLE fr (id INTEGER, k DOUBLE);");
    run_ok("INSERT INTO fl VALUES (1, -(0.0::DOUBLE)), (2, 'NaN'::DOUBLE), (3, 1.5);");
    run_ok("INSERT INTO fr VALUES (100, 0.0), (101, 'NaN'::DOUBLE), (102, 2.5);");
    run_ok("CHECKPOINT;");
  }
};

TEST_CASE_METHOD(NullSafeJoinFixture,
                 "gpu_execution all-null-safe SEMI join matches NULL to NULL",
                 "[integration][gpu_execution][join][nulls]")
{
  // Ids 1, 2 and 3 survive: l's NULL keys match r's NULL key.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu("SELECT l.id, l.k FROM l SEMI JOIN r ON l.k IS NOT DISTINCT FROM r.k");
  compare_gpu_vs_cpu("SELECT count(*) FROM l SEMI JOIN r ON l.k IS NOT DISTINCT FROM r.k");
}

TEST_CASE_METHOD(PartialNullKeyJoinFixture,
                 "gpu_execution two-key all-null-safe SEMI join matches partial NULLs",
                 "[integration][gpu_execution][join][nulls]")
{
  // Ids 1 and 2 survive; id 3 differs from (1, 3) in its non-NULL column.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu(
    "SELECT pl.id, pl.a, pl.b FROM pl SEMI JOIN pr "
    "ON pl.a IS NOT DISTINCT FROM pr.a AND pl.b IS NOT DISTINCT FROM pr.b");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM pl SEMI JOIN pr "
    "ON pl.a IS NOT DISTINCT FROM pr.a AND pl.b IS NOT DISTINCT FROM pr.b");
}

TEST_CASE_METHOD(FloatKeyJoinFixture,
                 "gpu_execution all-null-safe SEMI join on DOUBLE keys matches signed zero and NaN",
                 "[integration][gpu_execution][join][nulls]")
{
  // The CPU keeps ids 1 and 2. Ids only, so the comparison does not depend on how -0.0 prints.
  disabled_optimizer_guard guard(*con->context, duckdb::OptimizerType::BUILD_SIDE_PROBE_SIDE);
  compare_gpu_vs_cpu("SELECT fl.id FROM fl SEMI JOIN fr ON fl.k IS NOT DISTINCT FROM fr.k");
  compare_gpu_vs_cpu("SELECT count(*) FROM fl SEMI JOIN fr ON fl.k IS NOT DISTINCT FROM fr.k");
}
