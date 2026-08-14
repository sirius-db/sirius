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
 * @file test_gpu_execution_vector_join_exact_global.cpp
 * @brief End-to-end tests for the sirius_knn_join() table function, scoped to the
 *        exact GLOBAL top-k vector join. Global top-k returns the k globally-best
          pairs across the whole join.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>
#include <vector>

using VectorJoinGlobalFixture = sirius::test::GpuExecutionFixture;

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

// Single integer from a one-cell query that must succeed (e.g. count(*) checks).
int64_t single_int(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return r->GetValue(0, 0).GetValue<int64_t>();
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
// Global top-k on the adversarial large-magnitude L2 corpus. The corpus is
// [i, i+1, i+2], so the distance between probe P and corpus C is |P - C| * sqrt3:
// each probe sits at distance 0 on its own row and sqrt3 on its two neighbors. With
// 5 probes at least 10 apart, k=15 lands on a complete, tie-free boundary -- the 5
// self-matches (distance 0) plus all 10 immediate neighbors (distance sqrt3); the
// 16th pair is at 2*sqrt3. Global k counts pairs across all probes, so the winners
// interleave probe rows rather than giving each probe a fixed share.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - exact L2 on large-magnitude vectors",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE vjg_corpus (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vjg_corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] FROM range(50000) "
    "t(i);");
  run_ok("CREATE TABLE vjg_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vjg_probe VALUES "
    "(0, [49800.0, 49801.0, 49802.0]), "
    "(1, [49890.0, 49891.0, 49892.0]), "
    "(2, [49810.0, 49811.0, 49812.0]), "
    "(3, [49780.0, 49781.0, 49782.0]), "
    "(4, [49820.0, 49821.0, 49822.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vjg_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'vjg_probe', tier => 'gpu', format => 'duckdb');");

  // The 15 globally-smallest pairs are unambiguous: 5 self-pairs at 0 and 10 neighbor
  // pairs at sqrt3; the 16th is at 2*sqrt3. Compare id pairs against DuckDB's own
  // global order over the full cross product.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM vjg_probe p, vjg_corpus c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 15;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'vjg_probe','vec','vjg_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 15, join_mode => 'global');");

  REQUIRE(joined.size() == 15);  // global k pairs total, not k per left row
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('vjg_probe');");
  run_ok("SELECT * FROM unpin_table('vjg_corpus');");
}

// -----------------------------------------------------------------------------
// Global self-join with k larger than the number of pairs. The per-row cap lowers
// the candidate k to the right-table row count (6), so every left row keeps all 6
// right rows as candidates; the global output limit stays at the user's k (100),
// which exceeds the 6*6 = 36 total pairs, so the join returns the entire cross
// product -- exactly 36 rows.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - k larger than the total pair count returns all pairs",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gall (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gall VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [2.0, 0.0, 0.0]), (2, [4.0, 0.0, 0.0]), "
    "(3, [8.0, 0.0, 0.0]), (4, [16.0, 0.0, 0.0]), (5, [32.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gall', tier => 'gpu', format => 'duckdb');");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'gall','vec','gall','vec', search_mode => 'exact', metric => 'l2', "
                        "k => 100, join_mode => 'global');");

  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con, "SELECT a.id, b.id FROM gall a, gall b;");
  con->Query("SET gpu_execution = true;");

  REQUIRE(joined.size() == 36);  // full 6x6 cross product
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('gall');");
}

// -----------------------------------------------------------------------------
// Global top-k below the cross-product size. k (8) exceeds the right-table row
// count (6): a per-row query would cap k to 6 and emit 6*6 = 36 rows, but global k
// counts pairs across all left rows, so it must emit exactly 8 -- the 6 self-matches
// (distance 0) plus the single closest cross pair in both directions (ids 0<->1,
// whose vectors 1.0 and 2.0 are distance 1 apart). Geometric spacing (x = 1,2,4,...)
// makes every pairwise distance distinct, so the 8/9 boundary is tie-free (the 9th
// pair is at distance 2).
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - top-k selects the best pairs across all left rows",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gdedup (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gdedup VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [2.0, 0.0, 0.0]), (2, [4.0, 0.0, 0.0]), "
    "(3, [8.0, 0.0, 0.0]), (4, [16.0, 0.0, 0.0]), (5, [32.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gdedup', tier => 'gpu', format => 'duckdb');");

  // The 8 smallest pairwise distances are unambiguous: 6 self-pairs at distance 0,
  // then exactly 2 pairs at distance 1 (0->1 and 1->0); the 9th is distance 2. Compare
  // the id pairs against DuckDB's own global order over the full cross product.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM gdedup p, gdedup c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 8;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'gdedup','vec','gdedup','vec', search_mode => 'exact', metric => 'l2', "
                        "k => 8, join_mode => 'global');");

  REQUIRE(joined.size() == 8);  // global k pairs total, not k per left row
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('gdedup');");
}

// -----------------------------------------------------------------------------
// Global top-k across a multi-batch corpus, with the two left rows contributing
// unevenly so the result cannot be mistaken for a per-row query. Probe 0 sits
// exactly on corpus id 122880 (distance 0), while probe 1 sits at 245760.5, equally
// close (distance 0.5) to corpus ids 245760 and 245761. The three globally-smallest
// distances are therefore {0, 0.5, 0.5}, so probe 1 contributes two of the three
// winning pairs and probe 0 only one -- an uneven split a per-row top-k could never
// produce. Both of probe 1's winners live in the last corpus batch while probe 0's
// straddles a batch boundary, so this also exercises the cross-batch reduce_local
// feeding the global reduction.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - top-k across multiple right batches",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gmb_corpus (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  run_ok(
    "INSERT INTO gmb_corpus SELECT i, list_resize([i::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM range(250000) t(i);");
  run_ok("CREATE TABLE gmb_probe (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  run_ok(
    "INSERT INTO gmb_probe SELECT id, list_resize([v::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM (VALUES (0, 122880.0), (1, 245760.5)) t(id, v);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gmb_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gmb_probe', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM gmb_probe p, gmb_corpus c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 3;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'gmb_probe','vec','gmb_corpus','vec', search_mode => 'exact', "
                        "metric => 'l2', k => 3, join_mode => 'global');");

  REQUIRE(joined.size() == 3);
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('gmb_probe');");
  run_ok("SELECT * FROM unpin_table('gmb_corpus');");
}

// -----------------------------------------------------------------------------
// Global top-k where the winners come from BOTH left batches. The probe has 125000
// rows, so it splits at the row-group boundary (122880) into two pinned batches
// (ids 0..122879 and 122880..124999); the corpus stays a single 2000-row batch (id
// lives in dim 0, rest zero, so L2 distance is |i - j|). Nearly every probe is set
// far away (dim 0 = 100000 + i, so distance >= ~98000), except four planted probes
// -- two in each left batch (ids 5, 60000 in batch 0; 122880, 124999 in batch 1) --
// placed exactly on a distinct corpus row (distance 0). Each planted probe's three
// nearest are itself (0) and its two neighbors (1), so the 12 globally-smallest
// pairs are exactly those four probes' top-3; the 13th is at distance 2. Because two
// of the four winners live in the second left batch, this proves the global
// reduction drains and merges more than one left partition -- a per-row query keeps
// each probe's neighbors, but only global lets a subset of probes own the whole
// output.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - top-k winners span multiple left batches",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gml_corpus (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  run_ok(
    "INSERT INTO gml_corpus SELECT i, list_resize([i::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM range(2000) t(i);");
  run_ok("CREATE TABLE gml_probe (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  // Default probes sit at dim 0 = 100000 + i (far from every corpus row); four planted
  // probes -- two per left batch -- land exactly on distinct corpus rows.
  run_ok(
    "INSERT INTO gml_probe SELECT i, list_resize([("
    "  CASE i WHEN 5 THEN 500.0 WHEN 60000 THEN 1000.0 "
    "         WHEN 122880 THEN 100.0 WHEN 124999 THEN 1500.0 "
    "         ELSE (100000.0 + i) END)::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM range(125000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gml_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gml_probe', tier => 'gpu', format => 'duckdb');");

  const std::string q =
    "sirius_knn_join('gml_probe','vec','gml_corpus','vec', "
    "search_mode => 'exact', metric => 'l2', k => 12, join_mode => 'global')";

  // The 12 winners are exactly each planted probe's top-3 (self at 0, two neighbors
  // at 1); everything else is at distance >= 2. Build the reference from just those
  // four probes' per-row top-3 -- a full cross product over 125000*2000 pairs would
  // be far too large to materialize on the CPU.
  const std::string planted = "(5, 60000, 122880, 124999)";
  con->Query("SET gpu_execution = false;");
  auto reference =
    ok_rows(*con,
            "SELECT p.id, n.id FROM (SELECT * FROM gml_probe WHERE id IN " + planted +
              ") p, LATERAL ("
              "  SELECT c.id FROM gml_corpus c "
              "  ORDER BY array_distance(p.vec, c.vec) LIMIT 3) n;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con, "SELECT left_id, right_id FROM " + q + ";");

  REQUIRE(joined.size() == 12);  // 4 planted probes x 3 nearest, drawn from both left batches
  REQUIRE(joined == reference);
  // Both second-batch probes actually won pairs (they are not dropped by the drain).
  REQUIRE(single_int(*con, "SELECT count(*) FROM " + q + " WHERE left_id = 122880;") == 3);
  REQUIRE(single_int(*con, "SELECT count(*) FROM " + q + " WHERE left_id = 124999;") == 3);

  run_ok("SELECT * FROM unpin_table('gml_probe');");
  run_ok("SELECT * FROM unpin_table('gml_corpus');");
}

// -----------------------------------------------------------------------------
// Global top-k with the cosine metric. Corpus rows are unit vectors spaced 10 apart
// in the xy-plane (0..50 degrees), so cosine is discriminative and the ordering is
// just |angle difference|. The two probes sit exactly on the 0-degree and 50-degree
// corpus rows. The four globally-smallest cosine distances are the two self-matches
// (distance 0) and the two 10-degree neighbors (distance 1 - cos 10, one per probe),
// which tie each other but sit strictly below the 20-degree pairs -- so the 4/5
// boundary is tie-free. Both output types must return the same neighbor set (only
// the score column changes), and the clamp keeps distance >= 0 and similarity <= 1.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - exact cosine matches CPU and both outputs stay in range",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gcos_corpus (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  // Unit vectors at 0,10,20,30,40,50 degrees -> well-separated directions.
  run_ok(
    "INSERT INTO gcos_corpus VALUES "
    "(0, [1.0, 0.0, 0.0]), "
    "(1, [0.98480775, 0.17364818, 0.0]), "
    "(2, [0.93969262, 0.34202014, 0.0]), "
    "(3, [0.86602540, 0.5, 0.0]), "
    "(4, [0.76604444, 0.64278761, 0.0]), "
    "(5, [0.64278761, 0.76604444, 0.0]);");
  run_ok("CREATE TABLE gcos_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  // Probe 0 aligns with corpus 0 (0 deg), probe 1 with corpus 5 (50 deg).
  run_ok(
    "INSERT INTO gcos_probe VALUES "
    "(0, [1.0, 0.0, 0.0]), "
    "(1, [0.64278761, 0.76604444, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gcos_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gcos_probe', tier => 'gpu', format => 'duckdb');");

  // The 4 smallest cosine pairs globally: 2 self-matches at 0, then the two 10-degree
  // neighbors; the 5th pair is at 20 degrees. Compare ids against DuckDB's global order.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM gcos_probe p, gcos_corpus c "
                           "ORDER BY array_cosine_distance(p.vec, c.vec) LIMIT 4;");
  con->Query("SET gpu_execution = true;");

  auto dist_ids = ok_rows(*con,
                          "SELECT left_id, right_id FROM sirius_knn_join("
                          "'gcos_probe','vec','gcos_corpus','vec', "
                          "search_mode => 'exact', metric => 'cosine', k => 4, "
                          "join_mode => 'global', output_type => 'distance');");
  auto sim_ids  = ok_rows(*con,
                         "SELECT left_id, right_id FROM sirius_knn_join("
                          "'gcos_probe','vec','gcos_corpus','vec', "
                          "search_mode => 'exact', metric => 'cosine', k => 4, "
                          "join_mode => 'global', output_type => 'similarity');");

  REQUIRE(dist_ids.size() == 4);  // global k pairs total
  REQUIRE(dist_ids == reference);
  REQUIRE(sim_ids == reference);  // output_type changes only the score, not the ranking

  // Clamp holds on the self-matches: distance stays >= 0 (would read a hair below 0
  // without the floor) and similarity stays <= 1.
  REQUIRE(single_float(*con,
                       "SELECT min(distance) FROM sirius_knn_join("
                       "'gcos_probe','vec','gcos_corpus','vec', "
                       "search_mode => 'exact', metric => 'cosine', k => 4, "
                       "join_mode => 'global', output_type => 'distance');") >= 0.0F);
  REQUIRE(single_float(*con,
                       "SELECT max(similarity) FROM sirius_knn_join("
                       "'gcos_probe','vec','gcos_corpus','vec', "
                       "search_mode => 'exact', metric => 'cosine', k => 4, "
                       "join_mode => 'global', output_type => 'similarity');") <= 1.0F);

  run_ok("SELECT * FROM unpin_table('gcos_probe');");
  run_ok("SELECT * FROM unpin_table('gcos_corpus');");
}

// -----------------------------------------------------------------------------
// exact and exact-gemm must agree under global top-k. exact-gemm expands the L2
// distance as ‖a‖² - 2a·b + ‖b‖², which loses precision to catastrophic cancellation
// as the true distance approaches 0 (near-duplicate or self-match rows). Global
// ranking compares the winning distances ACROSS probes, so that error -- harmless to
// a per-row set comparison -- can reorder the global top-k when the winners sit near
// 0. So this test deliberately avoids the near-zero regime: each corpus row is a unit
// axis vector (well-separated directions) and each probe is one axis scaled out, so
// its single nearest neighbor sits at a distinct, moderate distance (0.1, 0.2, …,
// 0.5) -- gaps far larger than gemm's ~1e-6 error at unit magnitude. The 5 winners
// are those nearest pairs; the 6th globally-smallest pair is ~1.49 away, a tie-free
// boundary both search modes resolve identically.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - exact and exact-gemm agree on well-conditioned L2",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  // Unit axis vectors: pairwise distances are sqrt2 (orthogonal) or 2 (opposite).
  run_ok("CREATE TABLE ggemm_corpus (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO ggemm_corpus VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (2, [0.0, 0.0, 1.0]), "
    "(3, [-1.0, 0.0, 0.0]), (4, [0.0, -1.0, 0.0]), (5, [0.0, 0.0, -1.0]);");
  // Each probe is one axis scaled out by 1 + 0.1*(id+1), so its nearest neighbor is
  // that axis at distance 0.1*(id+1) and every other corpus row is >= ~1.49 away.
  run_ok("CREATE TABLE ggemm_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO ggemm_probe VALUES "
    "(0, [1.1, 0.0, 0.0]), "     // -> corpus 0 at 0.1
    "(1, [0.0, 1.2, 0.0]), "     // -> corpus 1 at 0.2
    "(2, [0.0, 0.0, 1.3]), "     // -> corpus 2 at 0.3
    "(3, [-1.4, 0.0, 0.0]), "    // -> corpus 3 at 0.4
    "(4, [0.0, -1.5, 0.0]);");   // -> corpus 4 at 0.5
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'ggemm_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'ggemm_probe', tier => 'gpu', format => 'duckdb');");

  // The 5 globally-smallest pairs are each probe's nearest axis, at 0.1..0.5; the 6th
  // is ~1.49, so the set is tie-free at k=5.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM ggemm_probe p, ggemm_corpus c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 5;");
  con->Query("SET gpu_execution = true;");

  auto join_ids = [&](const std::string& mode) {
    return ok_rows(*con,
                   "SELECT left_id, right_id FROM sirius_knn_join("
                   "'ggemm_probe','vec','ggemm_corpus','vec', "
                   "search_mode => '" +
                     mode + "', metric => 'l2', k => 5, join_mode => 'global');");
  };

  auto exact_ids      = join_ids("exact");
  auto exact_gemm_ids = join_ids("exact-gemm");

  REQUIRE(exact_ids == reference);
  REQUIRE(exact_gemm_ids == exact_ids);

  run_ok("SELECT * FROM unpin_table('ggemm_probe');");
  run_ok("SELECT * FROM unpin_table('ggemm_corpus');");
}

// -----------------------------------------------------------------------------
// Global top-k pads a short last batch just like per-row. The 122883-row FLOAT[768]
// corpus splits into one full 122880-row batch and a 3-row tail (one row group is
// ~377 MB, over the batch byte cap, so each pins on its own). The single probe sits
// on the last corpus row (122882): its 9 nearest are the 3-row tail plus 6 rows from
// the full first batch. Because the tail is shorter than k=9, select pads it up to k
// with dummies (distance +inf, id -1); reduce_local must drop those dummies before
// they can reach the global reduction. id lives in dim 0 (rest zero), so distance is
// |i - j| -- one-sided at the end of the corpus, hence tie-free.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - pads a short last batch across 768-dim batches",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE glb_corpus (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  run_ok(
    "INSERT INTO glb_corpus SELECT i, list_resize([i::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM range(122883) t(i);");
  run_ok("CREATE TABLE glb_probe (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  run_ok(
    "INSERT INTO glb_probe SELECT id, list_resize([v::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM (VALUES (0, 122882.0)) t(id, v);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'glb_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'glb_probe', tier => 'gpu', format => 'duckdb');");

  const std::string q =
    "sirius_knn_join('glb_probe','vec','glb_corpus','vec', "
    "search_mode => 'exact', metric => 'l2', k => 9, join_mode => 'global')";

  // The single probe's 9 nearest are the 9 globally-smallest pairs (distances 0..8).
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM glb_probe p, glb_corpus c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 9;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con, "SELECT left_id, right_id FROM " + q + ";");

  REQUIRE(joined == reference);   // padded dummies excluded from the global winners
  REQUIRE(joined.size() == 9);
  // No dummy filler (its id is -1) leaked through the merge into the global top-k.
  REQUIRE(single_int(*con, "SELECT count(*) FROM " + q + " WHERE right_id = -1;") == 0);

  run_ok("SELECT * FROM unpin_table('glb_probe');");
  run_ok("SELECT * FROM unpin_table('glb_corpus');");
}

// -----------------------------------------------------------------------------
// Global top-k payload path, single pinned batch. A multi-column right output
// (id + val) forces the non-fast path, where materialize gathers the requested
// columns by global row number. Global mode runs the TOP_N before materialize, so
// the gather happens on just the k winners -- this test proves that reordering still
// resolves the right columns correctly. The corpus id starts at 10000 so id != row
// position; a bug that used the position as the id would return wrong values. The
// FLOAT[3] corpus never hits the batch byte cap, so it stays one batch -- the
// multi-batch route-once gather is covered by the FLOAT[768] test below. With the
// [i,i+1,i+2] corpus the distance is |P - C| * sqrt3, so k=15 lands on a tie-free
// boundary: the 5 self-matches (0) plus all 10 neighbors (sqrt3).
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - payload path gathers right columns across batches",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gpay_corpus (id INTEGER PRIMARY KEY, val INTEGER, vec FLOAT[3]);");
  // id = 10000 + row position, val = a second payload column; both must come back via
  // the gather, not from the row number.
  run_ok(
    "INSERT INTO gpay_corpus SELECT 10000 + i, i * 2, [i::float, (i+1)::float, (i+2)::float] "
    "FROM range(300000) t(i);");
  run_ok("CREATE TABLE gpay_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gpay_probe VALUES "
    "(90000, [150000.0, 150001.0, 150002.0]), "
    "(90001, [37000.0, 37001.0, 37002.0]), "
    "(90002, [260000.0, 260001.0, 260002.0]), "
    "(90003, [150090.0, 150091.0, 150092.0]), "
    "(90004, [90000.0, 90001.0, 90002.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gpay_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gpay_probe', tier => 'gpu', format => 'duckdb');");

  // CPU global reference: the 15 smallest (probe, corpus) pairs over the full cross
  // product, returning the corpus id + val (the payload columns) with the probe id.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id, c.val FROM gpay_probe p, gpay_corpus c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 15;");
  con->Query("SET gpu_execution = true;");

  // right_output_columns has two columns, so the join takes the payload (gather) path
  // rather than the id-only fast path.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id, right_val FROM sirius_knn_join("
                        "'gpay_probe','vec','gpay_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 15, join_mode => 'global', "
                        "right_output_columns => ['id', 'val']);");

  REQUIRE(joined.size() == 15);  // global k pairs total
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('gpay_probe');");
  run_ok("SELECT * FROM unpin_table('gpay_corpus');");
}

// -----------------------------------------------------------------------------
// Global payload path across GENUINELY multiple pinned batches. The FLOAT[3] gpay
// test above stays in one batch (small rows never reach the byte cap), so it never
// runs the per-batch route-once gather -- the same blind spot that once hid a
// multi-batch merge bug. Here the corpus is FLOAT[768] with 250000 rows: each row
// group (~360 MB) exceeds the batch byte cap and pins on its own, so the corpus
// spans three batches. id = 10000 + row position (so id != position) and val is a
// second payload column, forcing the gather (non-fast) path. id lives in dim 0 (rest
// zero), so the L2 distance between a probe at value P and corpus row j is |P - j|.
// Three probes sit exactly on distinct corpus rows -- one per batch (positions 1000,
// 200000, 249000) -- so the k=9 global winners (each probe's self at distance 0 plus
// its two neighbors at distance 1; the 10th pair is at distance 2, a tie-free
// boundary) draw their payload from all three batches. A route-once bug that
// mis-mapped a batch offset would return the wrong id/val for the batch-1 or batch-2
// winners.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - payload path gathers across real multi-batch corpus",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE mbpay_corpus (id INTEGER PRIMARY KEY, val INTEGER, vec FLOAT[768]);");
  // id = 10000 + row position, val = a second payload column; both must come back via
  // the per-batch gather, not from the row number. dim 0 = row position so L2 = |P - j|.
  run_ok(
    "INSERT INTO mbpay_corpus SELECT 10000 + i, i * 2, "
    "list_resize([i::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] FROM range(250000) t(i);");
  run_ok("CREATE TABLE mbpay_probe (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  // One probe per corpus batch: positions 1000 (batch 0), 200000 (batch 1),
  // 249000 (batch 2). Each lands exactly on a corpus row.
  run_ok(
    "INSERT INTO mbpay_probe SELECT id, list_resize([v::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM (VALUES (90000, 1000.0), (90001, 200000.0), (90002, 249000.0)) t(id, v);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'mbpay_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'mbpay_probe', tier => 'gpu', format => 'duckdb');");

  // CPU global reference: the 9 smallest (probe, corpus) pairs over the full cross
  // product, returning the corpus id + val (the payload columns) with the probe id.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id, c.val FROM mbpay_probe p, mbpay_corpus c "
                           "ORDER BY array_distance(p.vec, c.vec) LIMIT 9;");
  con->Query("SET gpu_execution = true;");

  // right_output_columns has two columns, so the join takes the payload (gather) path.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id, right_val FROM sirius_knn_join("
                        "'mbpay_probe','vec','mbpay_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 9, join_mode => 'global', "
                        "right_output_columns => ['id', 'val']);");

  REQUIRE(joined.size() == 9);  // global k pairs total: 3 probes x (self + 2 neighbors)
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('mbpay_probe');");
  run_ok("SELECT * FROM unpin_table('mbpay_corpus');");
}

// -----------------------------------------------------------------------------
// The k <= 1024 cross-batch merge limit is a PER-ROW constraint, not a global one.
// Global top-k skips knn_merge_parts (its 1024 cap) and reduces brute_force's
// candidates with a plain TOP_N, so it accepts k > 1024; only per-row still rejects
// it at bind. The 2000-row single-batch corpus lets a single probe produce up to 2000
// global pairs, so k = 1025 is a valid, tie-free ask.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - accepts k above the per-row merge limit",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE gkcap_corpus (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gkcap_corpus SELECT i, [i::float, 0.0::float, 0.0::float] FROM range(2000) "
    "t(i);");
  run_ok("CREATE TABLE gkcap_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok("INSERT INTO gkcap_probe VALUES (0, [1.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gkcap_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gkcap_probe', tier => 'gpu', format => 'duckdb');");

  // Global mode accepts k > 1024 (no knn_merge_parts): the one probe yields 1025 global pairs.
  REQUIRE(single_int(*con,
                     "SELECT count(*) FROM sirius_knn_join("
                     "'gkcap_probe','vec','gkcap_corpus','vec', "
                     "search_mode => 'exact', metric => 'l2', k => 1025, join_mode => 'global');") ==
          1025);

  // Per-row still rejects k > 1024 at bind, since it does go through knn_merge_parts.
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('gkcap_probe','vec','gkcap_corpus','vec', "
               "search_mode => 'exact', metric => 'l2', k => 1025, join_mode => 'per-row');",
               "k must be <= 1024");

  run_ok("SELECT * FROM unpin_table('gkcap_probe');");
  run_ok("SELECT * FROM unpin_table('gkcap_corpus');");
}

// -----------------------------------------------------------------------------
// L2 has no natural similarity, so the combination is rejected at bind regardless of
// join_mode.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinGlobalFixture,
                 "sirius_knn_join global - l2 with similarity output is rejected",
                 "[integration][gpu_execution][array][vss][vector_join][global]")
{
  run_ok("CREATE TABLE grej (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok("INSERT INTO grej VALUES (0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'grej', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('grej','vec','grej','vec', "
               "metric => 'l2', output_type => 'similarity', join_mode => 'global');",
               "only meaningful for metric => 'cosine'");

  run_ok("SELECT * FROM unpin_table('grej');");
}
