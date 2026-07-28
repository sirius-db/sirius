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
// subqueries do). `b` has no NULLs, so IS NOT DISTINCT FROM behaves like '=' and
// the single-flag GPU join matches CPU.
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
  // unsupported nested-loop path. cuDF applies one null_equality flag to all keys,
  // so this uses UNEQUAL; `b` has no NULLs, so that matches CPU here.
  compare_gpu_vs_cpu(
    "SELECT ml.a, ml.b FROM ml JOIN mr ON ml.a = mr.a AND ml.b IS NOT DISTINCT FROM mr.b");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ml JOIN mr ON ml.a = mr.a AND ml.b IS NOT DISTINCT FROM mr.b");
}
