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
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

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

// An all-null-safe MARK over a build larger than both hash_partition_bytes and
// max_build_hash_table_bytes runs hash-partitioned (STANDARD, natural partition count) instead of
// one forced whole-build BUILD_PROBE partition. Both sides carry NULL keys, so NULL-to-NULL
// matches and definite FALSE misses are exercised across partitions. The null-propagating IN
// MARK over the same data keeps its single whole-build partition and its NULL marks.
TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution hash-partitioned null-safe MARK join matches the CPU",
                 "[integration][gpu_execution][join][nulls][mark][partition]")
{
  run_ok(
    "CREATE TABLE pml AS SELECT i AS id, CASE WHEN i % 97 = 0 THEN NULL ELSE (i * 7) % 400000 END "
    "AS k FROM range(300000) AS t(i);");
  run_ok(
    "CREATE TABLE pmr AS SELECT i AS id, CASE WHEN i % 89 = 0 THEN NULL ELSE (i * 3) % 500000 END "
    "AS k FROM range(600000) AS t(i);");
  run_ok("CHECKPOINT;");

  sirius::test::scoped_setting small_partitions(*con, "hash_partition_bytes", 1024 * 1024);
  sirius::test::scoped_setting small_build(*con, "max_build_hash_table_bytes", 512 * 1024);
  sirius::test::scoped_setting small_batches(*con, "scan_task_batch_size", 256 * 1024);

  // Correlated projected EXISTS: decorrelates to an all-null-safe MARK.
  compare_gpu_vs_cpu(
    "SELECT has_match, count(*), sum(id) FROM (SELECT pml.id, EXISTS (SELECT 1 FROM pmr WHERE "
    "pmr.k = pml.k) AS has_match FROM pml) GROUP BY has_match");
  compare_gpu_vs_cpu(
    "SELECT has_match, count(*), sum(id) FROM (SELECT pml.id, EXISTS (SELECT 1 FROM pmr WHERE "
    "pmr.k IS NOT DISTINCT FROM pml.k) AS has_match FROM pml) GROUP BY has_match");
}

// Null-propagating ('=') MARK joins (IN / NOT IN, = ANY / <> ALL) over a build larger than both
// hash_partition_bytes and max_build_hash_table_bytes are hash-partitioned too. SQL's NULL rule is
// global, so a single NULL build key -- which lands in one partition -- must still turn every
// unmatched probe row in every other partition into NULL.
TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution hash-partitioned null-propagating MARK join matches the CPU",
                 "[integration][gpu_execution][join][nulls][mark][partition]")
{
  // Probe keys span every partition; 1 in 97 probe keys is NULL.
  run_ok(
    "CREATE TABLE ppl AS SELECT i AS id, CASE WHEN i % 97 = 0 THEN NULL ELSE (i * 7) % 400000 END "
    "AS k, i % 13 AS g FROM range(300000) AS t(i);");
  // Build sides: no NULL; exactly one NULL; NULLs throughout.
  run_ok(
    "CREATE TABLE ppr_clean AS SELECT i AS id, (i * 3) % 500000 AS k, i % 13 AS g "
    "FROM range(600000) AS t(i);");
  run_ok(
    "CREATE TABLE ppr_one_null AS SELECT i AS id, CASE WHEN i = 12345 THEN NULL ELSE (i * 3) % "
    "500000 END AS k, i % 13 AS g FROM range(600000) AS t(i);");
  run_ok(
    "CREATE TABLE ppr_nulls AS SELECT i AS id, CASE WHEN i % 89 = 0 THEN NULL ELSE (i * 3) % "
    "500000 END AS k, i % 13 AS g FROM range(600000) AS t(i);");
  run_ok("CHECKPOINT;");

  sirius::test::scoped_setting small_partitions(*con, "hash_partition_bytes", 1024 * 1024);
  sirius::test::scoped_setting small_build(*con, "max_build_hash_table_bytes", 512 * 1024);
  sirius::test::scoped_setting small_batches(*con, "scan_task_batch_size", 256 * 1024);

  for (std::string const build : {"ppr_clean", "ppr_one_null", "ppr_nulls"}) {
    CAPTURE(build);
    // Projected IN / NOT IN marks, including NULL probe keys: true / false / NULL per row.
    compare_gpu_vs_cpu("SELECT in_k, count(*), sum(id) FROM (SELECT id, k IN (SELECT k FROM " +
                       build + ") AS in_k FROM ppl) GROUP BY in_k");
    compare_gpu_vs_cpu(
      "SELECT not_in_k, count(*), sum(id) FROM (SELECT id, k NOT IN (SELECT k FROM " + build +
      ") AS not_in_k FROM ppl) GROUP BY not_in_k");
    // NOT IN as a filter: any NULL build key filters every row out.
    compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM ppl WHERE k NOT IN (SELECT k FROM " + build +
                       ")");
    // = ANY / <> ALL spellings.
    compare_gpu_vs_cpu("SELECT m, count(*) FROM (SELECT k = ANY (SELECT k FROM " + build +
                       ") AS m FROM ppl) GROUP BY m");
    compare_gpu_vs_cpu("SELECT m, count(*) FROM (SELECT k <> ALL (SELECT k FROM " + build +
                       ") AS m FROM ppl) GROUP BY m");
  }

  // Correlated IN mixes a null-safe correlation key with the plain IN key; that MARK shape is
  // rejected at plan time and runs on the CPU (unchanged by partitioning).
  expect_plan_fallback_matches_cpu(
    "SELECT in_k, count(*) FROM (SELECT ppl.k IN (SELECT ppr_nulls.k FROM ppr_nulls WHERE "
    "ppr_nulls.g = ppl.g) AS in_k FROM ppl) GROUP BY in_k");
}
