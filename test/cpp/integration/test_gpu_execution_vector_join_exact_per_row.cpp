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
 * @file test_gpu_execution_vector_join_exact_per_row.cpp
 * @brief End-to-end tests for the sirius_knn_join() table function, scoped to the
 *        exact per-row top-k vector join (search_mode exact / exact-gemm). Both sides
 *        must be pinned.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using VectorJoinFixture = sirius::test::GpuExecutionFixture;

namespace {

// Sorted rows from a query that must succeed. Sorting both sides lets us compare
// result sets independent of the arrival order of the join's output rows.
std::vector<std::vector<std::string>> ok_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  return sirius::test::GpuExecutionFixture::collect_rows(mat, /*sort=*/true);
}

// Single FLOAT scalar from a one-cell query that must succeed (e.g. min/max checks).
float single_float(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return r->GetValue(0, 0).GetValue<float>();
}

// Assert a query fails, and that its error mentions `needle`.
void expect_error(duckdb::Connection& con, const std::string& sql, const std::string& needle)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  UNSCOPED_INFO("error was: " << r->GetError());
  REQUIRE(r->GetError().find(needle) != std::string::npos);
}

}  // namespace

// -----------------------------------------------------------------------------
// Adversarial large-magnitude L2: `exact` (unexpanded) must be bit-correct; this
// is precisely the input the `exact` mode exists to handle.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact L2 on large-magnitude vectors",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE vj_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vj_corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] FROM range(50000) "
    "t(i);");
  run_ok("CREATE TABLE vj_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vj_probe VALUES "
    "(0, [49800.0, 49801.0, 49802.0]), "
    "(1, [49890.0, 49891.0, 49892.0]), "
    "(2, [49810.0, 49811.0, 49812.0]), "
    "(3, [49780.0, 49781.0, 49782.0]), "
    "(4, [49820.0, 49821.0, 49822.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vj_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'vj_probe', tier => 'gpu', format => 'duckdb');");

  // k=9 lands on complete, tie-free shells around each probe ({P, P+-1..P+-4} at
  // 0, sqrt3, 2*sqrt3, ...), so the top-k SET is unambiguous. Compare ids only:
  // CPU array_distance vs GPU cuVS agree on membership, but their float distance
  // strings can differ in the last digit.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM vj_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM vj_corpus c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 9) n;");
  con->Query("SET gpu_execution = true;");

  // sirius_knn_join in exact mode must match the reference neighbor set exactly.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'vj_probe','vec','vj_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 9);");
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('vj_probe');");
  run_ok("SELECT * FROM unpin_table('vj_corpus');");
}

// -----------------------------------------------------------------------------
// Well-conditioned corpus: exact and exact-gemm must agree neighbor-for-neighbor,
// and both must match the CPU exact reference. This is the GEMM happy-path check.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact and exact-gemm agree on well-conditioned L2",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Magnitudes ~1, directions spread over the sphere -> GEMM is well-conditioned.
  run_ok("CREATE TABLE gemm_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gemm_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(50000) t(i);");
  run_ok("CREATE TABLE gemm_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gemm_probe SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(5) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gemm_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gemm_probe', tier => 'gpu', format => 'duckdb');");

  // CPU exact reference (ids only: distances match to ~1e-6, but the neighbor
  // SET is what must be identical, and these directions are tie-free at k=10).
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM gemm_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM gemm_corpus c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 10) n;");
  con->Query("SET gpu_execution = true;");

  auto join_ids = [&](const std::string& mode) {
    return ok_rows(*con,
                   "SELECT left_id, right_id FROM sirius_knn_join("
                   "'gemm_probe','vec','gemm_corpus','vec', "
                   "search_mode => '" +
                     mode + "', metric => 'l2', k => 10);");
  };

  auto exact_ids      = join_ids("exact");
  auto exact_gemm_ids = join_ids("exact-gemm");

  // exact-gemm agrees with exact, and both match the CPU reference.
  REQUIRE(exact_ids == reference);
  REQUIRE(exact_gemm_ids == exact_ids);

  run_ok("SELECT * FROM unpin_table('gemm_probe');");
  run_ok("SELECT * FROM unpin_table('gemm_corpus');");
}

// -----------------------------------------------------------------------------
// Self-join (left table == right table, the dedup shape) with k larger than the
// table: k is lowered to the row count, so every row pairs with all rows.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - self-join with k larger than the table",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE sj (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO sj VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (2, [0.0, 0.0, 1.0]), "
    "(3, [1.0, 1.0, 0.0]), (4, [2.0, 3.0, 4.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sj', tier => 'gpu', format => 'duckdb');");

  // k = 10 is more than the 5 rows, so it is lowered to 5: every left row pairs
  // with all 5 right rows (including itself at distance 0) -- the full 5x5 set.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'sj','vec','sj','vec', search_mode => 'exact', metric => 'l2', k => 10);");

  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con, "SELECT a.id, b.id FROM sj a, sj b;");
  con->Query("SET gpu_execution = true;");

  REQUIRE(joined.size() == 25);
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('sj');");
}

// -----------------------------------------------------------------------------
// Regular self-join: k smaller than the table, so real per-row top-k selection
// happens. Geometric spacing (x = 1,2,4,...) makes every pairwise distance
// distinct, so the top-k is tie-free and matches the CPU reference exactly.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - self-join per-row top-k with k below the table size",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE dedup (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO dedup VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [2.0, 0.0, 0.0]), (2, [4.0, 0.0, 0.0]), "
    "(3, [8.0, 0.0, 0.0]), (4, [16.0, 0.0, 0.0]), (5, [32.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'dedup', tier => 'gpu', format => 'duckdb');");

  // Each row's 3 nearest (itself + 2 closest) are unambiguous because all gaps
  // differ; compare the id pairs against DuckDB's own per-row top-3.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM dedup p, LATERAL ("
                           "  SELECT c.id FROM dedup c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 3) n;");
  con->Query("SET gpu_execution = true;");

  auto joined =
    ok_rows(*con,
            "SELECT left_id, right_id FROM sirius_knn_join("
            "'dedup','vec','dedup','vec', search_mode => 'exact', metric => 'l2', k => 3);");

  REQUIRE(joined.size() == 18);  // 6 rows x 3 neighbors
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('dedup');");
}

// -----------------------------------------------------------------------------
// Multi-batch right table: under the test's 1 MB block size a ~300k-row corpus
// spans several pinned batches, so reduce_local merges partials across batches
// (n_parts > 1) -- the cross-batch path a single-batch corpus never runs.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact L2 across multiple right batches",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE mb_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO mb_corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] "
    "FROM range(300000) t(i);");
  run_ok("CREATE TABLE mb_probe (id INTEGER, vec FLOAT[3]);");
  // Probes spread across the corpus so the winning neighbors come from different
  // batches -- the merge has to pick each probe's near shell out of far partials
  // contributed by every other batch.
  run_ok(
    "INSERT INTO mb_probe VALUES "
    "(0, [150000.0, 150001.0, 150002.0]), "
    "(1, [37000.0, 37001.0, 37002.0]), "
    "(2, [260000.0, 260001.0, 260002.0]), "
    "(3, [150090.0, 150091.0, 150092.0]), "
    "(4, [90000.0, 90001.0, 90002.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'mb_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'mb_probe', tier => 'gpu', format => 'duckdb');");

  // k=9 lands on complete, tie-free shells ({P, P+-1..P+-4}); compare neighbor ids
  // to DuckDB's own per-row top-9.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM mb_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM mb_corpus c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 9) n;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'mb_probe','vec','mb_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 9);");
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('mb_probe');");
  run_ok("SELECT * FROM unpin_table('mb_corpus');");
}

// -----------------------------------------------------------------------------
// Cosine: correctness on direction-varied vectors, both output types return the
// same neighbor set, and the clamp keeps distance >= 0 and similarity <= 1.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact cosine matches CPU and both outputs stay in range",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Varied directions so cosine is discriminative (the [i,i+1,i+2] corpus is all
  // parallel -> every pair ~1, degenerate for cosine).
  run_ok("CREATE TABLE cos_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO cos_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(50000) t(i);");
  run_ok("CREATE TABLE cos_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO cos_probe SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(5) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'cos_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'cos_probe', tier => 'gpu', format => 'duckdb');");

  // True cosine top-5 per probe from DuckDB.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM cos_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM cos_corpus c "
                           "  ORDER BY array_cosine_distance(p.vec, c.vec) LIMIT 5) n;");
  con->Query("SET gpu_execution = true;");

  // The neighbor set is the same whether the score is reported as distance or
  // similarity -- output_type only changes the score column, not the ranking.
  auto dist_ids = ok_rows(*con,
                          "SELECT left_id, right_id FROM sirius_knn_join("
                          "'cos_probe','vec','cos_corpus','vec', "
                          "search_mode => 'exact', metric => 'cosine', k => 5, "
                          "output_type => 'distance');");
  auto sim_ids  = ok_rows(*con,
                         "SELECT left_id, right_id FROM sirius_knn_join("
                          "'cos_probe','vec','cos_corpus','vec', "
                          "search_mode => 'exact', metric => 'cosine', k => 5, "
                          "output_type => 'similarity');");
  REQUIRE(dist_ids == reference);
  REQUIRE(sim_ids == reference);

  // Clamp holds on the self-matches (probe i == corpus i): distance stays >= 0
  // (would read a hair below 0 without the floor) and similarity stays <= 1.
  REQUIRE(single_float(*con,
                       "SELECT min(distance) FROM sirius_knn_join("
                       "'cos_probe','vec','cos_corpus','vec', "
                       "search_mode => 'exact', metric => 'cosine', k => 5, "
                       "output_type => 'distance');") >= 0.0F);
  REQUIRE(single_float(*con,
                       "SELECT max(similarity) FROM sirius_knn_join("
                       "'cos_probe','vec','cos_corpus','vec', "
                       "search_mode => 'exact', metric => 'cosine', k => 5, "
                       "output_type => 'similarity');") <= 1.0F);

  run_ok("SELECT * FROM unpin_table('cos_probe');");
  run_ok("SELECT * FROM unpin_table('cos_corpus');");
}

// -----------------------------------------------------------------------------
// L2 has no natural similarity, so the combination is rejected at bind.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - l2 with similarity output is rejected",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE rej (id INTEGER, vec FLOAT[3]);");
  run_ok("INSERT INTO rej VALUES (0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'rej', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('rej','vec','rej','vec', "
               "metric => 'l2', output_type => 'similarity');",
               "only meaningful for metric => 'cosine'");

  run_ok("SELECT * FROM unpin_table('rej');");
}

// -----------------------------------------------------------------------------
// Out-of-core: a HOST-tier corpus is streamed chunk-by-chunk through the fused
// operator's running top-k instead of being GPU-resident, and must produce exactly
// what the GPU-resident corpus produces. Multiple right batches, so the fold is the
// thing under test. Streaming-path only -- the split operators require GPU residency.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - host-tier corpus matches a GPU-tier corpus",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  setenv("SIRIUS_VECTOR_JOIN_STREAMING", "1", 1);

  run_ok("CREATE TABLE oc_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO oc_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);");
  run_ok("CREATE TABLE oc_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO oc_probe SELECT i, "
    "[sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(64) t(i);");
  run_ok("CHECKPOINT;");

  const std::string join_sql =
    "SELECT left_id, right_id FROM sirius_knn_join("
    "'oc_probe','vec','oc_corpus','vec', search_mode => 'exact', metric => 'l2', k => 8);";

  run_ok("SELECT * FROM pin_table(name => 'oc_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'oc_corpus', tier => 'gpu', format => 'duckdb');");
  auto const gpu_tier_rows = ok_rows(*con, join_sql);
  REQUIRE(gpu_tier_rows.size() == 64 * 8);
  run_ok("SELECT * FROM unpin_table('oc_corpus');");

  run_ok("SELECT * FROM pin_table(name => 'oc_corpus', tier => 'host', format => 'duckdb');");
  auto const host_tier_rows = ok_rows(*con, join_sql);

  REQUIRE(host_tier_rows == gpu_tier_rows);

  run_ok("SELECT * FROM unpin_table('oc_corpus');");
  run_ok("SELECT * FROM unpin_table('oc_probe');");
  unsetenv("SIRIUS_VECTOR_JOIN_STREAMING");
}

// -----------------------------------------------------------------------------
// Global top-k: the k closest pairs over the whole join, not k per left row. The CPU
// reference is an exhaustive cross join ordered by distance, which is the definition.
// The corpus spans several right batches so the fold, the per-batch depth bound and
// the TOP_N above materialize are all exercised together.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - global top-k matches an exhaustive CPU ranking",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE g_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO g_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);");
  run_ok("CREATE TABLE g_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO g_probe SELECT i, "
    "[sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(32) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'g_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'g_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT p.id, c.id FROM g_probe p, g_corpus c "
                                 "ORDER BY array_distance(p.vec, c.vec) LIMIT 12;");
  con->Query("SET gpu_execution = true;");

  auto const joined = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'g_probe','vec','g_corpus','vec', search_mode => 'exact', "
                              "metric => 'l2', k => 12, join_mode => 'global', "
                              "left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(joined.size() == 12);
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('g_corpus');");
  run_ok("SELECT * FROM unpin_table('g_probe');");
}

// -----------------------------------------------------------------------------
// Threshold (range) join: every pair within eps, which is ragged by construction --
// left rows contribute anywhere from zero to many pairs. This is the case the old
// fixed-k output contract could not represent at all, so it is also the regression
// test for materialize gathering by an explicit left row instead of repeating.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold join matches an exhaustive CPU range query",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE t_corpus (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO t_corpus SELECT i, [(i%100)::float, (i/100)::float] FROM range(10000) t(i);");
  run_ok("CREATE TABLE t_probe (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO t_probe VALUES (0,[10.0,10.0]),(1,[50.5,50.5]),(2,[99.0,99.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 't_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 't_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT p.id, c.id FROM t_probe p, t_corpus c "
                                 "WHERE array_distance(p.vec, c.vec) <= 2.0;");
  con->Query("SET gpu_execution = true;");

  auto const joined = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'t_probe','vec','t_corpus','vec', search_mode => 'exact', "
                              "metric => 'l2', k => 64, join_mode => 'threshold', eps => 2.0, "
                              "left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(joined == reference);
  // Ragged by construction: this is not a whole multiple of the probe row count.
  REQUIRE(joined.size() % 3 != 0);

  run_ok("SELECT * FROM unpin_table('t_corpus');");
  run_ok("SELECT * FROM unpin_table('t_probe');");
}

// -----------------------------------------------------------------------------
// A threshold join has no k. This case used to assert the opposite -- the exact path
// searched each left row only to depth k, so an eps admitting more than k neighbours
// was refused as "truncated". `brute_force_threshold` thresholds inside the tiled GEMM
// instead of filtering an already-computed top-k block, so k is not consulted at all
// and the answer is complete by construction.
//
// The assertion is deliberately STRONGER than the refusal it replaces: an eps that
// admits thousands of pairs against a k of 4 must return every one of them, matched
// against an exhaustive CPU range query rather than merely "did not error".
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold join answers past k without truncating",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE tt_corpus (id INTEGER, vec FLOAT[2]);");
  run_ok(
    "INSERT INTO tt_corpus SELECT i, [(i%100)::float, (i/100)::float] FROM range(10000) t(i);");
  run_ok("CREATE TABLE tt_probe (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO tt_probe VALUES (0,[50.0,50.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'tt_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'tt_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT p.id, c.id FROM tt_probe p, tt_corpus c "
                                 "WHERE array_distance(p.vec, c.vec) <= 50.0;");
  con->Query("SET gpu_execution = true;");
  REQUIRE(reference.size() > 4);  // the point of the case: far more pairs than k

  // eps => 50 admits thousands of pairs and k => 4 is ignored, not a bound.
  auto const joined = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'tt_probe','vec','tt_corpus','vec', search_mode => 'exact', "
                              "metric => 'l2', k => 4, join_mode => 'threshold', eps => 50.0, "
                              "left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(joined.size() == reference.size());
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('tt_corpus');");
  run_ok("SELECT * FROM unpin_table('tt_probe');");
}

// -----------------------------------------------------------------------------
// A threshold answer that outgrows the device leaves it in pieces: the stream stage moves what
// it holds to host memory each time it passes a budget, and materialize brings the pieces back
// one at a time. The budget is lowered so a small join takes that path across several corpus
// chunks; every pair and its distance must come out as an exhaustive CPU range query has them.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - a threshold answer moved to host in pieces is unchanged",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Grid points padded with zeros: 130 MB of corpus is two pin chunks under the test config's
  // 100 MB scan batches, so the pairs arrive in more than one part.
  run_ok("CREATE TABLE ro_corpus (id INTEGER, vec FLOAT[64]);");
  run_ok(
    "INSERT INTO ro_corpus SELECT i, list_resize([(i%500)::float, (i//500)::float], 64, 0::float) "
    "FROM range(500000) t(i);");
  run_ok("CREATE TABLE ro_probe (id INTEGER, vec FLOAT[64]);");
  run_ok(
    "INSERT INTO ro_probe SELECT i, list_resize([(i*37%500)::float, (i*101%1000)::float], 64, "
    "0::float) FROM range(64) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'ro_probe',  tier => 'gpu',  format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'ro_corpus', tier => 'host', format => 'duckdb');");

  // Squared distances between grid points are whole numbers, so they compare exactly.
  con->Query("SET gpu_execution = false;");
  auto const reference =
    ok_rows(*con,
            "SELECT p.id, c.id, round(array_distance(p.vec, c.vec) ^ 2)::BIGINT "
            "FROM ro_probe p, ro_corpus c "
            "WHERE array_distance(p.vec, c.vec) <= 6.0;");
  con->Query("SET gpu_execution = true;");
  REQUIRE(reference.size() > 4000);

  ::setenv("SIRIUS_VSS_RADIUS_FLUSH_PAIRS", "500", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_RADIUS_FLUSH_PAIRS"); }
  } const unset;
  run_ok(
    "CREATE TEMP TABLE ro_joined AS SELECT left_id, right_id, distance FROM "
    "sirius_knn_join('ro_probe','vec','ro_corpus','vec', search_mode => 'exact', "
    "metric => 'l2', join_mode => 'threshold', eps => 6.0, left_output_columns => ['id'], "
    "right_output_columns => ['id']);");
  // Rounded on the CPU: the GPU's FLOAT -> BIGINT cast truncates.
  con->Query("SET gpu_execution = false;");
  auto const joined =
    ok_rows(*con, "SELECT left_id, right_id, round(distance ^ 2)::BIGINT FROM ro_joined;");
  con->Query("SET gpu_execution = true;");
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('ro_corpus');");
  run_ok("SELECT * FROM unpin_table('ro_probe');");
}

// -----------------------------------------------------------------------------
// Query-side partitioning: the probe side no longer has to be device-resident either.
// A task owns one probe chunk and searches the whole corpus against it, so both sides
// can exceed VRAM at once -- the shape every rec-sys candidate-generation join has.
// Answers must not depend on where either side was pinned.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - host-tier probe side matches a GPU-tier probe side",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE qp_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO qp_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);");
  run_ok("CREATE TABLE qp_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO qp_probe SELECT i, "
    "[sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(50000) t(i);");
  run_ok("CHECKPOINT;");

  const std::string join_sql =
    "SELECT left_id, right_id FROM sirius_knn_join("
    "'qp_probe','vec','qp_corpus','vec', search_mode => 'exact', metric => 'l2', k => 8, "
    "left_output_columns => ['id'], right_output_columns => ['id']);";

  run_ok("SELECT * FROM pin_table(name => 'qp_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'qp_probe',  tier => 'gpu', format => 'duckdb');");
  auto const gpu_probe_rows = ok_rows(*con, join_sql);
  REQUIRE(gpu_probe_rows.size() == 50000 * 8);
  run_ok("SELECT * FROM unpin_table('qp_probe');");

  // Probe streamed, corpus resident.
  run_ok("SELECT * FROM pin_table(name => 'qp_probe', tier => 'host', format => 'duckdb');");
  REQUIRE(ok_rows(*con, join_sql) == gpu_probe_rows);
  run_ok("SELECT * FROM unpin_table('qp_corpus');");

  // Both sides streamed: neither has to fit device memory.
  run_ok("SELECT * FROM pin_table(name => 'qp_corpus', tier => 'host', format => 'duckdb');");
  REQUIRE(ok_rows(*con, join_sql) == gpu_probe_rows);

  run_ok("SELECT * FROM unpin_table('qp_corpus');");
  run_ok("SELECT * FROM unpin_table('qp_probe');");
}

// -----------------------------------------------------------------------------
// An approximate search must be backed by something that actually prunes. Without
// a clustering the join would run exhaustively under a query that claimed otherwise,
// which is indistinguishable in the results from an approximate run that got
// lucky -- the one failure mode a user cannot detect. The clustered path that lifts
// this refusal is covered in test_gpu_execution_kmeans.cpp.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - approximate knobs are refused without a clustering",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE ap_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO ap_corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] FROM range(64) t(i);");
  run_ok("CREATE TABLE ap_probe (id INTEGER, vec FLOAT[3]);");
  run_ok("INSERT INTO ap_probe VALUES (0, [1.0, 2.0, 3.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'ap_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'ap_probe', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('ap_probe','vec','ap_corpus','vec', "
               "search_mode => 'approx', k => 4);",
               "needs clustering =>");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('ap_probe','vec','ap_corpus','vec', "
               "n_clusters => 8, k => 4);",
               "belongs to sirius_kmeans_fit");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('ap_probe','vec','ap_corpus','vec', "
               "n_probes => 4, k => 4);",
               "only means anything with clustering");

  // The defaults these guard must keep working.
  REQUIRE(ok_rows(*con,
                  "SELECT left_id, right_id FROM sirius_knn_join("
                  "'ap_probe','vec','ap_corpus','vec', search_mode => 'exact', k => 4);")
            .size() == 4);

  run_ok("SELECT * FROM unpin_table('ap_probe');");
  run_ok("SELECT * FROM unpin_table('ap_corpus');");
}

// -----------------------------------------------------------------------------
// Build phase: the corpus taken from a child scan must answer identically to the
// corpus taken from the pin.
//
// The corpus must span several batches or this test proves almost nothing: a
// neighbour id is a position in the corpus row order, and the fold and materialize
// derive that order independently unless they share the build side's snapshot. One
// batch cannot be ordered wrongly, so a small corpus passes even when the orders
// disagree -- which is exactly how the first cut of this path returned a million
// wrong ids against correct distances.
//
// Batching follows DuckDB row groups (122,880 rows), not the byte target, so the
// row count below is what makes this multi-batch; the vectors stay narrow and the
// whole table is a few MB. Keep it above ~3 row groups.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - build phase matches the pinned corpus",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Wide vectors and enough of them to exceed concat_batch_bytes (100 MB in integration.yaml):
  // the build side is coalesced by BYTES, so a narrow-vector corpus of any row count arrives as
  // a single batch and cannot exercise the ordering contract at all. 600k x FLOAT[64] is ~154 MB
  // and lands as several batches.
  run_ok("CREATE TABLE bp_corpus (id INTEGER, vec FLOAT[64]);");
  // Hashed, not periodic: a smooth generator like sin(i*c + j) repeats every 2*pi/c rows, which
  // at this row count gives every point ~95 near-identical twins and makes the k-th neighbour a
  // coin toss between tied rows. Comparing ids then fails for reasons that have nothing to do
  // with the build phase.
  run_ok(
    "INSERT INTO bp_corpus SELECT i, "
    "list_transform(range(0, 64), j -> ((hash(i * 64 + j) % 100000) / 100000.0)::FLOAT)"
    "::FLOAT[64] FROM range(600000) t(i);");
  run_ok("CREATE TABLE bp_probe (id INTEGER, vec FLOAT[64]);");
  run_ok(
    "INSERT INTO bp_probe SELECT i, "
    "list_transform(range(0, 64), j -> ((hash(i * 977 + j * 13 + 7) % 100000) / 100000.0)::FLOAT)"
    "::FLOAT[64] FROM range(200) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'bp_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'bp_probe', tier => 'gpu', format => 'duckdb');");

  auto const pinned = ok_rows(*con,
                              "SELECT left_id, right_id, distance FROM sirius_knn_join("
                              "'bp_probe','vec','bp_corpus','vec', search_mode => 'exact', "
                              "metric => 'l2', k => 5, left_output_columns => ['id'], "
                              "right_output_columns => ['id']);");
  REQUIRE(pinned.size() == 200 * 5);

  auto const built = ok_rows(*con,
                             "SELECT left_id, right_id, distance FROM sirius_knn_join("
                             "'bp_probe','vec','bp_corpus','vec', search_mode => 'exact', "
                             "metric => 'l2', k => 5, left_output_columns => ['id'], "
                             "right_output_columns => ['id'], build_source => 'scan');");
  REQUIRE(built == pinned);

  // Agreeing with the pin path is necessary but not sufficient, and waiting for the two to
  // disagree on their own does not work: whether the corpus batches arrive in the table's
  // order is a race, and on most runs they do, so the bug this guards against stays dormant.
  // Reversing the snapshot forces the disagreement instead. Any order is a correct corpus
  // order as long as every stage uses the same one, so a reversed run must return exactly the
  // same rows -- and does not if some stage derived an order of its own.
  // Compared against the pinned result, not against `built`: the two build-phase queries would
  // otherwise be the same SQL text, and the second reuses the first's plan -- and with it the
  // snapshot already taken -- so the reversal never happens and the check passes vacuously.
  setenv("SIRIUS_VECTOR_JOIN_REVERSE_BUILD_ORDER", "1", 1);
  auto const reversed = ok_rows(*con,
                                "SELECT left_id, right_id, distance FROM sirius_knn_join("
                                "'bp_probe','vec','bp_corpus','vec', search_mode => 'exact', "
                                "metric => 'l2', k => 5, right_output_columns => ['id'], "
                                "left_output_columns => ['id'], build_source => 'scan');");
  unsetenv("SIRIUS_VECTOR_JOIN_REVERSE_BUILD_ORDER");
  REQUIRE(reversed == pinned);

  run_ok("SELECT * FROM unpin_table('bp_probe');");
  run_ok("SELECT * FROM unpin_table('bp_corpus');");
}

// -----------------------------------------------------------------------------
// What the build phase is for: the corpus does not have to be pinned, and the
// columns it emits are not confined to whatever a pin happens to hold.
//
// Pinning a column subset used to make the rest of that table unusable in the
// same query -- `items` pinned on ['id','vec'] meant nothing could read
// `items.category` -- which is why the join benchmark has to split attributes
// into a side table. The corpus scan reads the base table, so a pin is a cache
// in front of it rather than the source.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - build phase does not require a pinned corpus",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE up_corpus (id INTEGER, vec FLOAT[8], category INTEGER);");
  run_ok(
    "INSERT INTO up_corpus SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 8 + j) % 100000) / 100000.0)::FLOAT)::FLOAT[8], "
    "(i % 7) FROM range(20000) t(i);");
  run_ok("CREATE TABLE up_probe (id INTEGER, vec FLOAT[8]);");
  run_ok(
    "INSERT INTO up_probe SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 31 + j * 7 + 3) % 100000) / 100000.0)::FLOAT)"
    "::FLOAT[8] FROM range(100) t(i);");
  run_ok("CHECKPOINT;");

  // Only the probe is pinned. The corpus is never pinned at all.
  run_ok("SELECT * FROM pin_table(name => 'up_probe', tier => 'gpu', format => 'duckdb');");

  const std::string join_sql =
    "SELECT left_id, right_id, right_category FROM sirius_knn_join("
    "'up_probe','vec','up_corpus','vec', search_mode => 'exact', metric => 'l2', k => 4, "
    "left_output_columns => ['id'], right_output_columns => ['id','category'], "
    "build_source => 'scan');";
  auto const unpinned = ok_rows(*con, join_sql);
  REQUIRE(unpinned.size() == 100 * 4);

  // category is a real value from the corpus row the id names, not a placeholder.
  for (auto const& row : unpinned) {
    REQUIRE(std::stoi(row[2]) == std::stoi(row[1]) % 7);
  }

  // Pinning a strict column subset must not change the answer, and must not make
  // `category` -- absent from the pin -- unavailable.
  run_ok(
    "SELECT * FROM pin_table(name => 'up_corpus', tier => 'gpu', format => 'duckdb', "
    "cols => ['id','vec']);");
  REQUIRE(ok_rows(*con, join_sql) == unpinned);

  run_ok("SELECT * FROM unpin_table('up_corpus');");
  run_ok("SELECT * FROM unpin_table('up_probe');");
}

// -----------------------------------------------------------------------------
// Probe side from a scan too, and then both sides at once with nothing pinned.
//
// The probe is read once per task, so unlike the corpus it needs no re-reads --
// but its batch order still names the output partitions materialize gathers left
// columns by, so the two stages have to agree on it exactly as they do for the
// corpus. Answers must be identical to the all-pinned path either way.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - probe side from a scan matches the pinned probe",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE ps_corpus (id INTEGER, vec FLOAT[8]);");
  run_ok(
    "INSERT INTO ps_corpus SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 8 + j) % 100000) / 100000.0)::FLOAT)::FLOAT[8] "
    "FROM range(30000) t(i);");
  run_ok("CREATE TABLE ps_probe (id INTEGER, vec FLOAT[8], region INTEGER);");
  run_ok(
    "INSERT INTO ps_probe SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 41 + j * 3 + 11) % 100000) / 100000.0)::FLOAT)"
    "::FLOAT[8], (i % 5) FROM range(300) t(i);");
  run_ok("CHECKPOINT;");

  const std::string cols =
    "search_mode => 'exact', metric => 'l2', k => 4, left_output_columns => ['id','region'], "
    "right_output_columns => ['id']";

  run_ok("SELECT * FROM pin_table(name => 'ps_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'ps_probe', tier => 'gpu', format => 'duckdb');");
  auto const pinned = ok_rows(*con,
                              "SELECT left_id, left_region, right_id, distance FROM "
                              "sirius_knn_join('ps_probe','vec','ps_corpus','vec', " +
                                cols + ");");
  REQUIRE(pinned.size() == 300 * 4);
  run_ok("SELECT * FROM unpin_table('ps_probe');");

  // Probe from a scan, corpus still pinned.
  REQUIRE(ok_rows(*con,
                  "SELECT left_id, left_region, right_id, distance FROM "
                  "sirius_knn_join('ps_probe','vec','ps_corpus','vec', " +
                    cols + ", probe_source => 'scan');") == pinned);

  // Both sides from scans, nothing pinned at all.
  run_ok("SELECT * FROM unpin_table('ps_corpus');");
  REQUIRE(ok_rows(*con,
                  "SELECT left_id, left_region, right_id, distance FROM "
                  "sirius_knn_join('ps_probe','vec','ps_corpus','vec', " +
                    cols + ", probe_source => 'scan', build_source => 'scan');") == pinned);

  // Reversing both snapshots must not move a single row: any order is correct as long as every
  // stage uses the same one.
  setenv("SIRIUS_VECTOR_JOIN_REVERSE_BUILD_ORDER", "1", 1);
  auto const reversed = ok_rows(*con,
                                "SELECT left_id, left_region, right_id, distance FROM "
                                "sirius_knn_join('ps_probe','vec','ps_corpus','vec', " +
                                  cols + ", build_source => 'scan', probe_source => 'scan');");
  unsetenv("SIRIUS_VECTOR_JOIN_REVERSE_BUILD_ORDER");
  REQUIRE(reversed == pinned);
}

// -----------------------------------------------------------------------------
// Relational surface: the probe is a subquery, so a predicate on it is DuckDB's
// to push into the scan before the join ever binds. That is the one thing the
// name-taking form cannot express -- a filtered probe had to be materialized as
// its own table first -- and it must give the same answer as doing so.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join_rel - filtered probe matches a pre-filtered table",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE rl_corpus (id INTEGER, vec FLOAT[8]);");
  run_ok(
    "INSERT INTO rl_corpus SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 8 + j) % 100000) / 100000.0)::FLOAT)::FLOAT[8] "
    "FROM range(20000) t(i);");
  run_ok("CREATE TABLE rl_probe (id INTEGER, vec FLOAT[8], region INTEGER);");
  run_ok(
    "INSERT INTO rl_probe SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 53 + j * 5 + 2) % 100000) / 100000.0)::FLOAT)"
    "::FLOAT[8], (i % 10) FROM range(1000) t(i);");
  // The workaround the relational form removes: the filtered probe as its own table.
  run_ok("CREATE TABLE rl_probe_r3 AS SELECT id, vec FROM rl_probe WHERE region = 3;");
  run_ok("CHECKPOINT;");

  run_ok("SELECT * FROM pin_table(name => 'rl_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'rl_probe_r3', tier => 'gpu', format => 'duckdb');");
  auto const workaround =
    ok_rows(*con,
            "SELECT left_id, right_id, distance FROM sirius_knn_join("
            "'rl_probe_r3','vec','rl_corpus','vec', search_mode => 'exact', metric => 'l2', "
            "k => 5, left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(workaround.size() == 100 * 5);
  run_ok("SELECT * FROM unpin_table('rl_probe_r3');");

  // Same answer with the filter pushed into the probe scan instead, and rl_probe never pinned.
  auto const pushed =
    ok_rows(*con,
            "SELECT left_id, right_id, distance FROM sirius_knn_join_rel("
            "(SELECT id, vec FROM rl_probe WHERE region = 3), 'vec', 'rl_corpus', 'vec', "
            "search_mode => 'exact', metric => 'l2', k => 5, left_output_columns => ['id'], "
            "right_output_columns => ['id']);");
  REQUIRE(pushed == workaround);

  // The probe can be any relation, not just a filtered scan.
  auto const from_cte =
    ok_rows(*con,
            "WITH picked AS (SELECT id, vec FROM rl_probe WHERE region = 3) "
            "SELECT left_id, right_id, distance FROM sirius_knn_join_rel("
            "(SELECT id, vec FROM picked), 'vec', 'rl_corpus', 'vec', "
            "search_mode => 'exact', metric => 'l2', k => 5, left_output_columns => ['id'], "
            "right_output_columns => ['id']);");
  REQUIRE(from_cte == workaround);

  // Corpus unpinned as well: neither side needs a pin on this surface.
  run_ok("SELECT * FROM unpin_table('rl_corpus');");
  REQUIRE(ok_rows(*con,
                  "SELECT left_id, right_id, distance FROM sirius_knn_join_rel("
                  "(SELECT id, vec FROM rl_probe WHERE region = 3), 'vec', 'rl_corpus', 'vec', "
                  "search_mode => 'exact', metric => 'l2', k => 5, "
                  "left_output_columns => ['id'], right_output_columns => ['id'], "
                  "build_source => 'scan');") == workaround);

  // probe_source has no meaning here -- the probe is the relation.
  expect_error(*con,
               "SELECT * FROM sirius_knn_join_rel("
               "(SELECT id, vec FROM rl_probe), 'vec', 'rl_corpus', 'vec', k => 5, "
               "probe_source => 'pin');",
               "does not apply");
}

// -----------------------------------------------------------------------------
// Projection pushdown narrows the declared output to what the query reads, which
// changes the shape materialize assembles: either side can end up contributing
// no columns at all. Correctness of the surviving columns is the thing to pin
// down -- the saving itself is structural (an unread corpus column is one fewer
// column concatenated across the whole corpus) and not visible from SQL.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - reading a subset of the declared output columns",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE pp_corpus (id INTEGER, vec FLOAT[8], category INTEGER);");
  run_ok(
    "INSERT INTO pp_corpus SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 8 + j) % 100000) / 100000.0)::FLOAT)::FLOAT[8], "
    "(i % 7) FROM range(20000) t(i);");
  run_ok("CREATE TABLE pp_probe (id INTEGER, vec FLOAT[8], region INTEGER);");
  run_ok(
    "INSERT INTO pp_probe SELECT i, "
    "list_transform(range(0, 8), j -> ((hash(i * 61 + j * 9 + 4) % 100000) / 100000.0)::FLOAT)"
    "::FLOAT[8], (i % 5) FROM range(200) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'pp_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'pp_probe', tier => 'gpu', format => 'duckdb');");

  const std::string args =
    "'pp_probe','vec','pp_corpus','vec', search_mode => 'exact', metric => 'l2', k => 4, "
    "left_output_columns => ['id','region'], right_output_columns => ['id','category']";

  auto const everything =
    ok_rows(*con,
            "SELECT left_id, left_region, right_id, right_category, distance FROM "
            "sirius_knn_join(" +
              args + ");");
  REQUIRE(everything.size() == 200 * 4);

  // One column from each side, and the score dropped: the narrowed layout must still name the
  // same rows as the full one.
  auto const narrowed =
    ok_rows(*con, "SELECT left_id, right_category FROM sirius_knn_join(" + args + ");");
  std::vector<std::vector<std::string>> expected;
  for (auto const& row : everything) {
    expected.push_back({row[0], row[3]});
  }
  std::sort(expected.begin(), expected.end());
  REQUIRE(narrowed == expected);

  // Nothing from the corpus at all -- the right-side gather has no columns to work on.
  auto const left_only = ok_rows(*con, "SELECT left_region FROM sirius_knn_join(" + args + ");");
  REQUIRE(left_only.size() == 200 * 4);

  // Nothing from either side: only the score survives.
  auto const score_only = ok_rows(*con, "SELECT distance FROM sirius_knn_join(" + args + ");");
  REQUIRE(score_only.size() == 200 * 4);

  // And no output columns whatsoever.
  auto const counted = ok_rows(*con, "SELECT count(*) FROM sirius_knn_join(" + args + ");");
  REQUIRE(counted.size() == 1);
  REQUIRE(counted[0][0] == std::to_string(200 * 4));

  run_ok("SELECT * FROM unpin_table('pp_probe');");
  run_ok("SELECT * FROM unpin_table('pp_corpus');");
}

// -----------------------------------------------------------------------------
// A VIEW named as a join side must be REFUSED, not reinterpreted.
//
// `Catalog::GetEntry(TABLE_ENTRY, ...)` also resolves views, so an unchecked
// `Cast<DuckTableEntry>()` would reinterpret a ViewCatalogEntry as a table -- in a release build
// undefined behaviour rather than an error. A view is the obvious way to express a filtered
// corpus, so on the scanned corpus side it is now bound and streamed; everywhere else it has to
// fail by name, never reach the cast.
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - a view is streamed on the scanned corpus side and refused by "
                 "name elsewhere",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE vw_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vw_corpus SELECT i, "
    "list_transform(range(0, 3), j -> ((hash(i * 3 + j) % 1000) / 1000.0)::FLOAT)::FLOAT[3] "
    "FROM range(500) t(i);");
  run_ok("CREATE TABLE vw_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vw_probe SELECT i, "
    "list_transform(range(0, 3), j -> ((hash(i * 7 + j) % 1000) / 1000.0)::FLOAT)::FLOAT[3] "
    "FROM range(20) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vw_probe', tier => 'gpu', format => 'duckdb');");
  run_ok("CREATE VIEW vw_corpus_v AS SELECT id, vec FROM vw_corpus WHERE id % 10 < 3;");

  // Corpus side, on the scan path -- the shape a user reaches for to filter a corpus: bound and
  // streamed, answering from the 150 rows the view keeps.
  auto const via_view = ok_rows(*con,
                                "SELECT left_id, right_id FROM sirius_knn_join("
                                "'vw_probe','vec','vw_corpus_v','vec', k => 2, metric => 'l2', "
                                "right_output_columns => ['id'], build_source => 'scan');");
  REQUIRE(via_view.size() == 20 * 2);
  for (auto const& r : via_view) {
    REQUIRE(std::stoi(r[1]) % 10 < 3);
  }

  // The same view on the pinned corpus path is refused, and the message says what to pass.
  expect_error(*con,
               "SELECT left_id FROM sirius_knn_join('vw_probe','vec','vw_corpus_v','vec', "
               "k => 2, metric => 'l2');",
               "build_source => 'scan'");

  // Probe side: a view is refused by name and pointed at the relational form.
  run_ok("CREATE VIEW vw_probe_v AS SELECT id, vec FROM vw_probe;");
  expect_error(*con,
               "SELECT left_id FROM sirius_knn_join('vw_probe_v','vec','vw_corpus','vec', "
               "k => 2, metric => 'l2', build_source => 'scan');",
               "sirius_knn_join_rel");
}

// -----------------------------------------------------------------------------
// A filter or aggregate ABOVE the join, reading the score.
//
// This is the shape that made the join return silently wrong answers. DuckDB narrows the read
// set AND is free to order it however it likes: with a `GROUP BY` it asks for
// `[distance, left_id]`, i.e. column_ids = [4, 0]. The operator emitted a fixed
// `[left..., right..., score]` block regardless, so the predicate above it read `left_id` and
// the grouping key read `distance` -- no error, just a wrong number.
//
// The data is built so no assertion here can turn on a floating-point last digit: every
// probe's neighbours sit at exact distances 0.5, 1.5, 2.5, ... and the threshold is 2.0, so
// the nearest pair to the cut is 0.5 away from it.
//
//   corpus  c_j = [j, 0]          j = 0 .. 2047
//   probe   p_i = [32*i + 0.5, 0] i = 0 .. 63
//
// so p_i's six nearest are at 0.5, 0.5, 1.5, 1.5, 2.5, 2.5 -- and four of the six are within
// 2.0. Probe 0 sits at the edge of the corpus and has only one neighbour below it, giving
// 0.5, 0.5, 1.5, 2.5, 3.5, 4.5 and so three within 2.0: 63*4 + 3 = 255 surviving pairs.
namespace {

void create_halves_dataset(sirius::test::GpuExecutionFixture& fx)
{
  fx.run_ok("CREATE TABLE hv_corpus (id INTEGER, vec FLOAT[2]);");
  fx.run_ok("INSERT INTO hv_corpus SELECT i, [i::FLOAT, 0.0::FLOAT] FROM range(2048) t(i);");
  fx.run_ok("CREATE TABLE hv_probe (id INTEGER, vec FLOAT[2]);");
  fx.run_ok(
    "INSERT INTO hv_probe SELECT i, [(32*i + 0.5)::FLOAT, 0.0::FLOAT] FROM range(64) t(i);");
  fx.run_ok("CHECKPOINT;");
  fx.run_ok("SELECT * FROM pin_table(name => 'hv_probe',  tier => 'gpu', format => 'duckdb');");
  fx.run_ok("SELECT * FROM pin_table(name => 'hv_corpus', tier => 'gpu', format => 'duckdb');");
}

constexpr int kHalvesSurvivors = 63 * 4 + 3;

}  // namespace

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - a score filter under an aggregate reads the score",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  create_halves_dataset(*this);

  const std::string args =
    "'hv_probe','vec','hv_corpus','vec', search_mode => 'exact', metric => 'l2', k => 6";

  // The CPU reference: the same top-6 per probe, filtered and grouped the same way, with
  // gpu_execution off so nothing about this side can depend on the operator under test.
  con->Query("SET gpu_execution = false;");
  auto const cpu_groups = ok_rows(*con,
                                  "SELECT p.id, count(*) FROM hv_probe p, LATERAL ("
                                  "  SELECT array_distance(p.vec, c.vec) AS d FROM hv_corpus c "
                                  "  ORDER BY array_distance(p.vec, c.vec) LIMIT 6) n "
                                  "WHERE n.d <= 2.0 GROUP BY p.id;");
  auto const cpu_pairs =
    ok_rows(*con,
            "SELECT array_distance(p.vec, c.vec), p.id FROM hv_probe p, hv_corpus c "
            "WHERE c.id IN (SELECT n.cid FROM LATERAL ("
            "  SELECT c2.id AS cid FROM hv_corpus c2 "
            "  ORDER BY array_distance(p.vec, c2.vec) LIMIT 6) n) "
            "AND array_distance(p.vec, c.vec) <= 2.0;");
  con->Query("SET gpu_execution = true;");

  // The reference itself must be the dataset we think it is, or agreeing with it proves nothing.
  REQUIRE(cpu_groups.size() == 64);
  REQUIRE(cpu_pairs.size() == kHalvesSurvivors);

  // B1: filter on the score, then group by a non-score column. Returned 755 pairs / 746 groups
  // against a true 6,611 / 2,158 on the SIFT repro; here a wrong layout shows up as grouping on
  // the near-unique distance instead of on left_id.
  auto const gpu_groups = ok_rows(*con,
                                  "SELECT left_id, count(*) FROM sirius_knn_join(" + args +
                                    ") WHERE distance <= 2.0 GROUP BY left_id;");
  REQUIRE(gpu_groups == cpu_groups);

  // Same rows, and with the score requested FIRST so the requested order is explicitly the
  // reverse of the emitted one.
  auto const gpu_pairs = ok_rows(
    *con, "SELECT distance, left_id FROM sirius_knn_join(" + args + ") WHERE distance <= 2.0;");
  REQUIRE(gpu_pairs == cpu_pairs);

  // The total is the thing a user reads, so assert it directly rather than only as a set.
  auto const gpu_total =
    ok_rows(*con, "SELECT count(*) FROM sirius_knn_join(" + args + ") WHERE distance <= 2.0;");
  REQUIRE(gpu_total[0][0] == std::to_string(kHalvesSurvivors));

  // HAVING on top of that filter: 63 probes keep 4 of their 6, probe 0 keeps 3.
  auto const gpu_having =
    ok_rows(*con,
            "SELECT count(*) FROM (SELECT left_id FROM sirius_knn_join(" + args +
              ") WHERE distance <= 2.0 GROUP BY left_id "
              "HAVING count(*) >= 4);");
  REQUIRE(gpu_having[0][0] == "63");

  // B4: min/max of a non-score column under a score filter. This failed as
  // "CUDF failure ... min() operation requires matching output type" -- the aggregate was
  // handed the INTEGER left_id where the plan said FLOAT.
  auto const gpu_minmax = ok_rows(
    *con,
    "SELECT min(left_id), max(left_id) FROM sirius_knn_join(" + args + ") WHERE distance <= 2.0;");
  REQUIRE(gpu_minmax.size() == 1);
  REQUIRE(gpu_minmax[0][0] == "0");
  REQUIRE(gpu_minmax[0][1] == "63");

  run_ok("SELECT * FROM unpin_table('hv_probe');");
  run_ok("SELECT * FROM unpin_table('hv_corpus');");
}

// -----------------------------------------------------------------------------
// Global top-k, in the two shapes that used to die with "TopN order index out of range":
// without an explicit output_columns list, and underneath an aggregate. Both come from the
// TOP_N above materialize ordering by the score at its DECLARED index while materialize emits
// only the narrowed set of columns the query actually reads.
//
// The halves dataset puts 128 pairs at distance 0.5 (two per probe), so any correct global
// top-12 is some 12 of those. Asserting "12 rows, every distance 0.5" is exact under that tie
// where naming the winning ids would not be.
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - global top-k with a narrowed or aggregated output",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  create_halves_dataset(*this);

  const std::string args =
    "'hv_probe','vec','hv_corpus','vec', search_mode => 'exact', metric => 'l2', k => 12, "
    "join_mode => 'global'";

  // B2: no left/right_output_columns, so the declared output is the full
  // [left_id, left_vec, right_id, right_vec, distance] while only three columns are read.
  auto const narrowed =
    ok_rows(*con, "SELECT left_id, right_id, distance FROM sirius_knn_join(" + args + ");");
  REQUIRE(narrowed.size() == 12);
  for (auto const& row : narrowed) {
    REQUIRE(row[2] == "0.5");
  }

  // Narrower still: the score alone, and no columns at all.
  auto const score_only = ok_rows(*con, "SELECT distance FROM sirius_knn_join(" + args + ");");
  REQUIRE(score_only.size() == 12);

  // B3: an aggregate above global mode. Nothing but the row count is read here, which is the
  // narrowest the output ever gets.
  auto const counted = ok_rows(*con, "SELECT count(*) FROM sirius_knn_join(" + args + ");");
  REQUIRE(counted.size() == 1);
  REQUIRE(counted[0][0] == "12");

  // And with an explicit output_columns list, which is the only shape that used to work.
  auto const explicit_cols =
    ok_rows(*con,
            "SELECT count(*) FROM sirius_knn_join(" + args +
              ", left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(explicit_cols[0][0] == "12");

  run_ok("SELECT * FROM unpin_table('hv_probe');");
  run_ok("SELECT * FROM unpin_table('hv_corpus');");
}

// -----------------------------------------------------------------------------
// Expression composition over the join, for everything Sirius's GPU translator actually
// supports (`src/expression/function_id.cpp` -- 28 function ids plus CASE / CAST / BETWEEN /
// comparison / conjunction).
//
// This exists as the gate for optimization work: the join is only worth tuning if ordinary SQL
// composes on top of it, and until now nothing checked that. Every case compares against a
// reference materialized on the CPU with `gpu_execution = false`, so a disagreement is the join.
//
// Geometry is the halves dataset (see above): six neighbours per probe at exactly 0.5, 0.5, 1.5,
// 1.5, 2.5, 2.5, so 255 of the 384 pairs are within 2.0 and nothing turns on a float last digit.
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - supported expressions compose over the join",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE xc (id INTEGER, nm VARCHAR, vec FLOAT[2]);");
  run_ok(
    "INSERT INTO xc SELECT i, 'corpus_' || i::VARCHAR, [i::FLOAT, 0.0::FLOAT] FROM range(2048) "
    "t(i);");
  run_ok("CREATE TABLE xp (id INTEGER, lb VARCHAR, vec FLOAT[2]);");
  run_ok(
    "INSERT INTO xp SELECT i, 'probe_' || i::VARCHAR, [(32*i + 0.5)::FLOAT, 0.0::FLOAT] "
    "FROM range(64) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'xp', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'xc', tier => 'gpu', format => 'duckdb');");

  const std::string args = "'xp','vec','xc','vec', search_mode => 'exact', metric => 'l2', k => 6";

  // The reference pair set, built once on the CPU. Same columns the TVF emits, same names.
  con->Query("SET gpu_execution = false;");
  run_ok(
    "CREATE TABLE xref AS SELECT p.id AS left_id, p.lb AS left_lb, n.cid AS right_id, "
    "n.nm AS right_nm, n.d AS distance FROM xp p, LATERAL ("
    "  SELECT c.id AS cid, c.nm AS nm, array_distance(p.vec, c.vec) AS d FROM xc c "
    "  ORDER BY array_distance(p.vec, c.vec) LIMIT 6) n;");
  con->Query("SET gpu_execution = true;");

  auto const reference_rows = [&](const std::string& select_list, const std::string& tail) {
    con->Query("SET gpu_execution = false;");
    auto rows = ok_rows(*con, "SELECT " + select_list + " FROM xref " + tail + ";");
    con->Query("SET gpu_execution = true;");
    return rows;
  };

  // Each entry is a select list plus a trailing clause, evaluated over the join and over the
  // CPU reference. The pair must be identical row for row.
  struct composition_case {
    const char* what;
    const char* select_list;
    const char* tail;
  };
  const std::vector<composition_case> cases = {
    // arithmetic operators
    {"add", "left_id + 1", ""},
    {"sub", "right_id - left_id", ""},
    {"mul", "right_id * 2", ""},
    {"div", "distance / 2", ""},
    {"mod", "right_id % 4", ""},
    // string functions
    {"length", "length(right_nm)", ""},
    {"substring", "substring(left_lb, 1, 5)", ""},
    {"concat", "left_lb || '-' || right_nm", ""},
    {"contains", "left_id, right_id", "WHERE contains(right_nm, 'corpus_1')"},
    {"like", "left_id, right_id", "WHERE right_nm LIKE 'corpus_1%'"},
    // expression classes
    {"case", "CASE WHEN distance <= 2.0 THEN 1 ELSE 0 END", ""},
    {"cast", "CAST(distance AS DOUBLE)", ""},
    {"between", "left_id, right_id", "WHERE distance BETWEEN 0.0 AND 2.0"},
    {"conjunction", "left_id, right_id", "WHERE distance <= 2.0 AND right_id % 2 = 0"},
    // an expression above an aggregate above the join
    {"agg_over_expr", "left_id, sum(right_id % 4)", "WHERE distance <= 2.0 GROUP BY left_id"},
  };

  for (auto const& c : cases) {
    UNSCOPED_INFO("composition case: " << c.what);
    auto const gpu = ok_rows(*con,
                             "SELECT " + std::string(c.select_list) + " FROM sirius_knn_join(" +
                               args + ") " + c.tail + ";");
    auto const cpu = reference_rows(c.select_list, c.tail);
    REQUIRE(gpu.size() == cpu.size());
    REQUIRE(gpu == cpu);
  }

  // The reference must be the dataset we think it is, or matching it proves nothing.
  auto const survivors =
    ok_rows(*con, "SELECT count(*) FROM sirius_knn_join(" + args + ") WHERE distance <= 2.0;");
  REQUIRE(survivors[0][0] == std::to_string(kHalvesSurvivors));

  run_ok("SELECT * FROM unpin_table('xp');");
  run_ok("SELECT * FROM unpin_table('xc');");
}

// -----------------------------------------------------------------------------
// The boundary: a function Sirius's GPU translator does NOT implement.
//
// `round`, `sqrt`, `abs` and friends used to be absent from the forward table in
// `src/expression/function_id.cpp`; over an ordinary table that meant a CPU fallback, over the
// join a hard error, because `sirius_knn_join`'s CPU callback is a stub that throws. They are now
// evaluated on the GPU, so the same expression works over both, and this test pins that the join
// no longer depends on a fallback it does not have.
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - named math functions over the join run on the GPU",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  create_halves_dataset(*this);

  auto const on_a_table =
    ok_rows(*con,
            "SELECT count(*) FROM (SELECT round(id / 7.0, 2) AS r FROM hv_corpus) WHERE "
            "r > 0;");
  REQUIRE(on_a_table[0][0] == "2047");

  auto const plain   = ok_rows(*con,
                             "SELECT count(*) FROM sirius_knn_join("
                               "'hv_probe','vec','hv_corpus','vec', search_mode => 'exact', "
                               "metric => 'l2', k => 6);");
  auto const rounded = ok_rows(*con,
                               "SELECT count(*) FROM (SELECT round(distance, 1) AS x FROM "
                               "sirius_knn_join('hv_probe','vec','hv_corpus','vec', "
                               "search_mode => 'exact', metric => 'l2', k => 6)) WHERE x >= 0;");
  REQUIRE(rounded[0][0] == plain[0][0]);

  run_ok("SELECT * FROM unpin_table('hv_probe');");
  run_ok("SELECT * FROM unpin_table('hv_corpus');");
}

// -----------------------------------------------------------------------------
// The default output omits the vector column each side joins on.
//
// `SELECT *` is the first thing anyone types. With the embedding in the declared output it
// printed 128 raw coordinates per side per row and buried the ids and the score -- and the
// relational probe side had already decided against that, for reasons written in its own
// comment. This pins the pinned-table side to the same rule.
//
// Note what is NOT claimed here: the vector is still available, it just has to be asked for.
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - the default output omits the joined vector column",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  auto column_names = [&](const std::string& sql) {
    auto r = con->Query(sql);
    REQUIRE(r);
    if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
    REQUIRE_FALSE(r->HasError());
    return std::vector<std::string>(r->names.begin(), r->names.end());
  };

  run_ok("CREATE TABLE dv_corpus (id INTEGER, tag VARCHAR, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO dv_corpus SELECT i, 'c' || i::VARCHAR, [i::FLOAT, 0.0::FLOAT, 0.0::FLOAT] "
    "FROM range(500) t(i);");
  run_ok("CREATE TABLE dv_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO dv_probe SELECT i, [(i + 0.5)::FLOAT, 0.0::FLOAT, 0.0::FLOAT] FROM range(20) "
    "t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'dv_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'dv_corpus', tier => 'gpu', format => 'duckdb');");

  const std::string args = "'dv_probe','vec','dv_corpus','vec', metric => 'l2', k => 2";

  // `SELECT *` is now the three things a reader wants, plus the corpus's own scalar column.
  auto const defaulted = column_names("SELECT * FROM sirius_knn_join(" + args + ") LIMIT 1;");
  REQUIRE(defaulted == std::vector<std::string>{"left_id", "right_id", "right_tag", "distance"});

  // Asking for it by name still works -- the column is omitted from the default, not withdrawn.
  auto const explicit_vec =
    column_names("SELECT * FROM sirius_knn_join(" + args +
                 ", left_output_columns => ['id','vec'], right_output_columns => ['vec']) "
                 "LIMIT 1;");
  REQUIRE(explicit_vec == std::vector<std::string>{"left_id", "left_vec", "right_vec", "distance"});

  // And the rows are unchanged by the narrower default: same pairs either way.
  auto const with_default =
    ok_rows(*con, "SELECT left_id, right_id FROM sirius_knn_join(" + args + ");");
  auto const with_explicit =
    ok_rows(*con,
            "SELECT left_id, right_id FROM sirius_knn_join(" + args +
              ", left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(with_default == with_explicit);
  REQUIRE(with_default.size() == 20 * 2);

  // The case the rule exists for: TWO embedding columns in ONE table. Dropping only the column
  // each side joins on would leave each side echoing the other's vector, so `SELECT *` would
  // still be a wall of coordinates.
  run_ok("CREATE TABLE dv_two (id INTEGER, a_vec FLOAT[3], b_vec FLOAT[3]);");
  run_ok(
    "INSERT INTO dv_two SELECT i, [i::FLOAT, 0.0::FLOAT, 0.0::FLOAT], "
    "[(i + 0.25)::FLOAT, 0.0::FLOAT, 0.0::FLOAT] FROM range(100) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'dv_two', tier => 'gpu', format => 'duckdb');");
  auto const two_col = column_names(
    "SELECT * FROM sirius_knn_join('dv_two','a_vec','dv_two','b_vec', metric => 'l2', k => 2) "
    "LIMIT 1;");
  REQUIRE(two_col == std::vector<std::string>{"left_id", "right_id", "distance"});
  run_ok("SELECT * FROM unpin_table('dv_two');");

  run_ok("SELECT * FROM unpin_table('dv_probe');");
  run_ok("SELECT * FROM unpin_table('dv_corpus');");
}

// -----------------------------------------------------------------------------
// The degenerate consequence of that default: a table whose only column IS the vector now
// contributes nothing to the output, so `SELECT *` is the score alone. That is a legitimate
// query -- "how far apart are these, ignoring which rows they were" -- and it must not become
// a zero-column crash on the way through materialize.
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - a vector-only table contributes no default output columns",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE vo (vec FLOAT[2]);");
  run_ok("INSERT INTO vo SELECT [i::FLOAT, 0.0::FLOAT] FROM range(200) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vo', tier => 'gpu', format => 'duckdb');");

  auto const r =
    con->Query("SELECT * FROM sirius_knn_join('vo','vec','vo','vec', metric => 'l2', k => 3);");
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  REQUIRE(mat.names.size() == 1);
  REQUIRE(mat.names[0] == "distance");
  REQUIRE(sirius::test::GpuExecutionFixture::collect_rows(mat, true).size() == 200 * 3);

  // Each row is its own nearest neighbour, so a third of the pairs sit at distance 0.
  auto const zeros = ok_rows(*con,
                             "SELECT count(*) FROM sirius_knn_join('vo','vec','vo','vec', "
                             "metric => 'l2', k => 3) WHERE distance = 0.0;");
  REQUIRE(zeros[0][0] == "200");

  run_ok("SELECT * FROM unpin_table('vo');");
}

// -----------------------------------------------------------------------------
// A filter on the corpus side of a threshold join is applied to the corpus before the search,
// not to the join's output after it. Both orders give the same pairs -- a pair within eps is
// one whatever the other corpus rows are -- and the test holds the pushed form to the answer
// an exhaustive CPU range query gives with the predicate in its WHERE clause.
//
// A top-k join must NOT be rewritten that way: dropping corpus rows first would promote other
// rows into each probe's k, a different query than filtering the k that were returned.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - corpus predicates are pushed into a threshold join only",
                 "[integration][gpu_execution][array][vss][vector_join][pushdown]")
{
  run_ok("CREATE TABLE f_corpus (id INTEGER, cat INTEGER, tag VARCHAR, vec FLOAT[2]);");
  run_ok(
    "INSERT INTO f_corpus SELECT i, i % 3, CASE WHEN i % 7 = 0 THEN 'x' ELSE 'y' END, "
    "[(i%100)::float, (i/100)::float] FROM range(10000) t(i);");
  run_ok("CREATE TABLE f_probe (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO f_probe VALUES (0,[10.0,10.0]),(1,[50.5,50.5]),(2,[99.0,99.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'f_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'f_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT p.id, c.id FROM f_probe p, f_corpus c "
                                 "WHERE array_distance(p.vec, c.vec) <= 6.0 AND c.cat = 1 "
                                 "AND c.tag <> 'x' AND c.id >= 500;");
  con->Query("SET gpu_execution = true;");
  REQUIRE_FALSE(reference.empty());

  auto const threshold =
    "FROM sirius_knn_join('f_probe','vec','f_corpus','vec', search_mode => 'exact', "
    "metric => 'l2', join_mode => 'threshold', eps => 6.0, left_output_columns => ['id'], "
    "right_output_columns => ['id','cat','tag']) "
    "WHERE right_cat = 1 AND right_tag <> 'x' AND 500 <= right_id";
  auto const joined = ok_rows(*con, std::string("SELECT left_id, right_id ") + threshold + ";");
  REQUIRE(joined == reference);

  // All three conjuncts were taken by the join: nothing is left for a filter above it.
  auto plan = con->Query(std::string("EXPLAIN SELECT left_id, right_id ") + threshold + ";");
  REQUIRE_FALSE(plan->HasError());
  UNSCOPED_INFO("plan: " << plan->ToString());
  CHECK(plan->ToString().find("FILTER") == std::string::npos);

  // Per-row top-k keeps its filter-after-search meaning.
  run_ok(
    "CREATE TABLE f_topk AS SELECT left_id, right_id, right_cat FROM sirius_knn_join("
    "'f_probe','vec','f_corpus','vec', search_mode => 'exact', metric => 'l2', k => 9, "
    "left_output_columns => ['id'], right_output_columns => ['id','cat']);");
  auto const topk_reference =
    ok_rows(*con, "SELECT left_id, right_id FROM f_topk WHERE right_cat = 1;");
  auto const topk = ok_rows(*con,
                            "SELECT left_id, right_id FROM sirius_knn_join("
                            "'f_probe','vec','f_corpus','vec', search_mode => 'exact', "
                            "metric => 'l2', k => 9, left_output_columns => ['id'], "
                            "right_output_columns => ['id','cat']) WHERE right_cat = 1;");
  REQUIRE(topk == topk_reference);
  REQUIRE(topk.size() < 3 * 9);

  run_ok("SELECT * FROM unpin_table('f_corpus');");
  run_ok("SELECT * FROM unpin_table('f_probe');");
}

// -----------------------------------------------------------------------------
// The GEMM-ranked threshold (exact-gemm): the pairs come from comparing GEMM scores against a
// per-row bound rather than from distances, so the boundary is what has to be right. Checked for
// L2 and cosine against DuckDB's own range predicate. Run it with SIRIUS_VSS_GEMM_TILE_MB=1 to
// spread the corpus over several score tiles.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - GEMM threshold matches the CPU range query for l2 and cosine",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE g_corpus (id INTEGER, vec FLOAT[16]);");
  run_ok(
    "INSERT INTO g_corpus SELECT i, list_transform(range(16), lambda d: "
    "((hash(i * 100 + d) % 1000)::FLOAT / 1000.0))::FLOAT[16] FROM range(20000) t(i);");
  run_ok("CREATE TABLE g_probe (id INTEGER, vec FLOAT[16]);");
  run_ok(
    "INSERT INTO g_probe SELECT i, list_transform(range(16), lambda d: "
    "((hash(i * 100 + d + 7777777) % 1000)::FLOAT / 1000.0))::FLOAT[16] FROM range(300) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'g_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'g_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const l2_ref  = ok_rows(*con,
                              "SELECT p.id, c.id FROM g_probe p, g_corpus c "
                               "WHERE array_distance(p.vec, c.vec) <= 0.8;");
  auto const cos_ref = ok_rows(*con,
                               "SELECT p.id, c.id FROM g_probe p, g_corpus c "
                               "WHERE array_cosine_similarity(p.vec, c.vec) >= 0.93;");
  con->Query("SET gpu_execution = true;");
  REQUIRE(l2_ref.size() > 1000);
  REQUIRE(cos_ref.size() > 1000);

  auto const join = [](const std::string& args) {
    return "SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', "
           "search_mode => 'exact-gemm', join_mode => 'threshold', " +
           args + ", left_output_columns => ['id'], right_output_columns => ['id']);";
  };
  CHECK(ok_rows(*con, join("metric => 'l2', eps => 0.8")) == l2_ref);
  CHECK(ok_rows(*con, join("metric => 'cosine', eps => 0.93, output_type => 'similarity'")) ==
        cos_ref);

  run_ok("SELECT * FROM unpin_table('g_corpus');");
  run_ok("SELECT * FROM unpin_table('g_probe');");
}

// -----------------------------------------------------------------------------
// Global top-k past the 1024 that knn_merge_parts capped the fold at: the answer is the k
// nearest pairs overall, held to DuckDB's ORDER BY ... LIMIT over the cross product.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - global top-k above 1024 matches ORDER BY LIMIT",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE gk_corpus (id INTEGER, vec FLOAT[8]);");
  run_ok(
    "INSERT INTO gk_corpus SELECT i, list_transform(range(8), lambda d: "
    "((hash(i * 100 + d) % 1000)::FLOAT / 1000.0))::FLOAT[8] FROM range(5000) t(i);");
  run_ok("CREATE TABLE gk_probe (id INTEGER, vec FLOAT[8]);");
  run_ok(
    "INSERT INTO gk_probe SELECT i, list_transform(range(8), lambda d: "
    "((hash(i * 100 + d + 999999) % 1000)::FLOAT / 1000.0))::FLOAT[8] FROM range(40) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gk_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gk_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT d FROM (SELECT array_distance(p.vec, c.vec) AS d "
                                 "FROM gk_probe p, gk_corpus c ORDER BY d LIMIT 2000);");
  con->Query("SET gpu_execution = true;");

  run_ok(
    "CREATE TABLE gk_out AS SELECT distance FROM sirius_knn_join('gk_probe','vec','gk_corpus',"
    "'vec', search_mode => 'exact-gemm', metric => 'l2', join_mode => 'global', k => 2000);");
  con->Query("SET gpu_execution = false;");
  auto const joined = ok_rows(*con, "SELECT distance FROM gk_out;");
  con->Query("SET gpu_execution = true;");
  REQUIRE(joined.size() == 2000);
  REQUIRE(reference.size() == 2000);
  // Rank by rank the same distances, to FP32 rounding of the expanded metric. Compared as
  // numbers: the string rows sort lexically, which is numeric order here (all 0.x).
  std::vector<double> got, want;
  for (auto const& r : joined) {
    got.push_back(std::stod(r.at(0)));
  }
  for (auto const& r : reference) {
    want.push_back(std::stod(r.at(0)));
  }
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i) {
    INFO("rank " << i << ": sirius " << got[i] << " duckdb " << want[i]);
    REQUIRE(std::abs(got[i] - want[i]) < 1e-4);
  }

  run_ok("SELECT * FROM unpin_table('gk_corpus');");
  run_ok("SELECT * FROM unpin_table('gk_probe');");
}
