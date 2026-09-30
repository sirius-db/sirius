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

// GPU-vs-CPU correctness for the SQL-integrated threshold (radius) vector join:
// `l JOIN r ON array_distance(l.v, r.v) <= eps` is recognized from a LOGICAL_ANY_JOIN and
// executed by sirius_physical_vector_threshold_join (tiled-GEMM threshold kernel over the two
// scan children, no pinning). compare_gpu_vs_cpu asserts the query ran on the GPU (one execution,
// no fallback) and returns exactly the CPU result.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>

namespace {

// RAII guard forcing sirius_physical_vector_threshold_join::execute() to split each left batch into
// tiny query tiles (one row per kernel call), so the multi-tile path runs on these small tables.
struct query_tile_env_guard {
  explicit query_tile_env_guard(const char* rows) { setenv("SIRIUS_VSS_QUERY_TILE_ROWS", rows, 1); }
  ~query_tile_env_guard() { unsetenv("SIRIUS_VSS_QUERY_TILE_ROWS"); }
};

class VectorThresholdJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  VectorThresholdJoinFixture()
  {
    // dim-3 tables: distances span the threshold so empty/partial/full match sets are exercised.
    run_ok("CREATE TABLE l (id INTEGER, v FLOAT[3]);");
    run_ok("CREATE TABLE r (id INTEGER, v FLOAT[3]);");
    run_ok("INSERT INTO l VALUES (1, [0,0,0]), (2, [1,1,1]), (3, [2,2,2]);");
    run_ok("INSERT INTO r VALUES (10, [3,3,3]), (11, [4,4,4]);");

    // dim-5 tables: same shape at a different dimensionality.
    run_ok("CREATE TABLE l5 (id INTEGER, v FLOAT[5]);");
    run_ok("CREATE TABLE r5 (id INTEGER, v FLOAT[5]);");
    run_ok("INSERT INTO l5 VALUES (1, [0,0,0,0,0]), (2, [1,1,1,1,1]), (3, [5,5,5,5,5]);");
    run_ok("INSERT INTO r5 VALUES (10, [1,1,1,1,1]), (11, [9,9,9,9,9]);");

    // cosine tables: non-zero vectors with well-separated cosine distances.
    run_ok("CREATE TABLE lc (id INTEGER, v FLOAT[3]);");
    run_ok("CREATE TABLE rc (id INTEGER, v FLOAT[3]);");
    run_ok("INSERT INTO lc VALUES (1, [1,0,0]), (2, [1,1,0]), (3, [0,1,0]);");
    run_ok("INSERT INTO rc VALUES (10, [1,0,0]), (11, [0,0,1]);");

    run_ok("CHECKPOINT;");
  }
};

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution L2 threshold join matches CPU across match-set sizes",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // eps=5 -> partial (3 pairs), eps=0.5 -> empty, eps=100 -> all 6 pairs.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON array_distance(l.v, r.v) <= 5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON array_distance(l.v, r.v) <= 0.5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON array_distance(l.v, r.v) <= 100");
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON array_distance(l.v, r.v) <= 5");
  // constant on the left exercises the operand-normalization path (`const >= func`).
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON 5 >= array_distance(l.v, r.v)");
}

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution L2 threshold join with dim-5 vectors matches CPU",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  compare_gpu_vs_cpu("SELECT l5.id, r5.id FROM l5 JOIN r5 ON array_distance(l5.v, r5.v) <= 5");
}

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution cosine-distance threshold join matches CPU",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  compare_gpu_vs_cpu(
    "SELECT lc.id, rc.id FROM lc JOIN rc ON array_cosine_distance(lc.v, rc.v) <= 0.5");
}

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution cosine-similarity threshold join matches CPU",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // similarity >= eps  <=>  cosine distance <= 1 - eps; same pairs as the <= 0.5 distance case.
  compare_gpu_vs_cpu(
    "SELECT lc.id, rc.id FROM lc JOIN rc ON array_cosine_similarity(lc.v, rc.v) >= 0.5");
  // constant on the left exercises the operand-normalization path (`const <= func`).
  compare_gpu_vs_cpu(
    "SELECT lc.id, rc.id FROM lc JOIN rc ON 0.5 <= array_cosine_similarity(lc.v, rc.v)");
}

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution LEFT threshold join pads unmatched left rows with NULLs",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // eps=5 -> l.id=1 has no match (padded NULL); eps=0.5 -> every left row unmatched;
  // eps=100 -> every left row matched (no padding).
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 0.5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 100");
  compare_gpu_vs_cpu("SELECT count(*) FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5");
  // cosine LEFT join.
  compare_gpu_vs_cpu(
    "SELECT lc.id, rc.id FROM lc LEFT JOIN rc ON array_cosine_distance(lc.v, rc.v) <= 0.5");
}

// The distance the kernel already computes for the predicate is reused as an output column when the
// SELECT list references the same distance function, instead of falling back to CPU for the
// recomputed projection. The distance column is compared with a relative tolerance because the GPU
// expanded-GEMM path and DuckDB's CPU array_distance legitimately differ in the low bits; keys
// (l.id, r.id) and NULLs still compare exactly.
TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution threshold join returns array_distance as an output column",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // eps=100 -> all 6 pairs, so every emitted row carries a distance to check.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v) FROM l JOIN r ON array_distance(l.v, r.v) <= 100",
    {2},
    1e-5);
  // Aliased distance: the whole expression is still the matched call.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v) AS dist "
    "FROM l JOIN r ON array_distance(l.v, r.v) <= 100",
    {2},
    1e-5);
  // Two references to the same distance both rewrite to the one emitted column.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v), array_distance(l.v, r.v) "
    "FROM l JOIN r ON array_distance(l.v, r.v) <= 100",
    {2, 3},
    1e-5);
  // The distance nested inside a larger expression exercises the recursive subexpression rewrite.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v) * 2 "
    "FROM l JOIN r ON array_distance(l.v, r.v) <= 100",
    {2},
    1e-5);
}

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution threshold join returns cosine distance/similarity as an output "
                 "column",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // cosine distance reused as-is.
  compare_gpu_vs_cpu_approx(
    "SELECT lc.id, rc.id, array_cosine_distance(lc.v, rc.v) "
    "FROM lc JOIN rc ON array_cosine_distance(lc.v, rc.v) <= 1",
    {2},
    1e-5);
  // cosine similarity: the operator emits 1 - distance, which must equal CPU
  // array_cosine_similarity.
  compare_gpu_vs_cpu_approx(
    "SELECT lc.id, rc.id, array_cosine_similarity(lc.v, rc.v) "
    "FROM lc JOIN rc ON array_cosine_similarity(lc.v, rc.v) >= 0.0",
    {2},
    1e-5);
}

TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution LEFT threshold join returns NULL distance for unmatched left rows",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // eps=5 -> l.id=1 is unmatched, so its distance column must be NULL (verified exactly, not
  // approximately). l.id=2,3 have real distances checked within tolerance.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v) "
    "FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5",
    {2},
    1e-5);
}

// With the query tile forced to one row per kernel call, execute() splits the left batch into many
// single-row tiles. The result must stay identical to the untiled CPU answer, which exercises the
// per-tile bookkeeping: shifting tile-local query rows back to global left indices, accumulating
// the LEFT matched flag across tiles (including tiles that produce zero edges), and emitting each
// tile separately. The same queries pass untiled in the cases above, so a divergence here is
// tiling-only.
TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution threshold join is invariant to query tiling",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  query_tile_env_guard const tile{"1"};

  // INNER across empty / partial / full match sets; the partial case has a leading zero-edge tile.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON array_distance(l.v, r.v) <= 5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON array_distance(l.v, r.v) <= 0.5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l JOIN r ON array_distance(l.v, r.v) <= 100");
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON array_distance(l.v, r.v) <= 5");

  // LEFT: unmatched left rows must be padded exactly once even though the flag is built across
  // tiles.
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 0.5");
  compare_gpu_vs_cpu("SELECT l.id, r.id FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 100");

  // cosine distance under tiling.
  compare_gpu_vs_cpu(
    "SELECT lc.id, rc.id FROM lc JOIN rc ON array_cosine_distance(lc.v, rc.v) <= 0.5");

  // Reused distance output column: the per-tile distance must line up with the offset query rows.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v) FROM l JOIN r ON array_distance(l.v, r.v) <= 100",
    {2},
    1e-5);

  // LEFT with a reused distance column: unmatched rows keep a NULL distance across tiles.
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, r.id, array_distance(l.v, r.v) "
    "FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5",
    {2},
    1e-5);
}

// An ungrouped count(*) directly over the join takes the row-count-only path: the join skips every
// output-column gather and emits a narrow row-count carrier instead of the FLOAT[dim] vectors. The
// count must still equal DuckDB's, across INNER / LEFT / cosine / dim-5, and combined with tiling
// (where the carrier is emitted per tile and, for LEFT, must also count unmatched rows once).
TEST_CASE_METHOD(VectorThresholdJoinFixture,
                 "gpu_execution threshold join count(*) row-count-only matches CPU",
                 "[integration][gpu_execution][join][vss][vector_threshold]")
{
  // eps=5 -> 3 pairs, eps=0.5 -> 0 pairs, eps=100 -> all 6 pairs.
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON array_distance(l.v, r.v) <= 5");
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON array_distance(l.v, r.v) <= 0.5");
  compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON array_distance(l.v, r.v) <= 100");
  // LEFT count includes unmatched left rows (padded, still counted).
  compare_gpu_vs_cpu("SELECT count(*) FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5");
  compare_gpu_vs_cpu("SELECT count(*) FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 0.5");
  // cosine and dim-5 exercise the path across metrics / dimensionalities.
  compare_gpu_vs_cpu("SELECT count(*) FROM lc JOIN rc ON array_cosine_distance(lc.v, rc.v) <= 0.5");
  compare_gpu_vs_cpu("SELECT count(*) FROM l5 JOIN r5 ON array_distance(l5.v, r5.v) <= 5");

  // Row-count-only combined with many single-row tiles: the carrier is emitted per tile and the
  // LEFT unmatched carrier must still be counted exactly once.
  {
    query_tile_env_guard const tile{"1"};
    compare_gpu_vs_cpu("SELECT count(*) FROM l JOIN r ON array_distance(l.v, r.v) <= 5");
    compare_gpu_vs_cpu("SELECT count(*) FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 5");
    compare_gpu_vs_cpu("SELECT count(*) FROM l LEFT JOIN r ON array_distance(l.v, r.v) <= 0.5");
  }
}

}  // namespace
