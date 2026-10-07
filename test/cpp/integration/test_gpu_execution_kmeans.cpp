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
 * @file test_gpu_execution_kmeans.cpp
 * @brief End-to-end tests for sirius_kmeans_fit() / sirius_kmeans_assign(), the pair that
 *        clusters a pinned vector column and emits the assignment edge list that a
 *        cluster-ordered copy of the table is built from.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using KMeansFixture = sirius::test::GpuExecutionFixture;

namespace {

std::unique_ptr<duckdb::MaterializedQueryResult> query_ok(duckdb::Connection& con,
                                                          const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return std::unique_ptr<duckdb::MaterializedQueryResult>(
    static_cast<duckdb::MaterializedQueryResult*>(r.release()));
}

std::int64_t scalar_i64(duckdb::Connection& con, const std::string& sql)
{
  auto r = query_ok(con, sql);
  return r->GetValue(0, 0).GetValue<std::int64_t>();
}

// Sorted rows from a query that must succeed, so two result sets compare independent of the
// order rows happen to arrive in.
std::vector<std::vector<std::string>> ok_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = query_ok(con, sql);
  return sirius::test::GpuExecutionFixture::collect_rows(*r, /*sort=*/true);
}

// (probe id, distance rounded) pairs, sorted. Rounding happens here rather than in SQL because a
// projection over the join falls back to the CPU, which cannot execute the rewrite target.
std::vector<std::pair<std::string, long long>> distance_multiset(duckdb::Connection& con,
                                                                 const std::string& sql)
{
  auto const rows = ok_rows(con, sql);
  std::vector<std::pair<std::string, long long>> out;
  out.reserve(rows.size());
  for (auto const& r : rows) {
    out.emplace_back(r.at(0), std::llround(std::stod(r.at(1)) * 1000.0));
  }
  std::sort(out.begin(), out.end());
  return out;
}

void expect_error(duckdb::Connection& con, const std::string& sql, const std::string& needle)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  UNSCOPED_INFO("error was: " << r->GetError());
  REQUIRE(r->GetError().find(needle) != std::string::npos);
}

// Two tight, well-separated groups in 3-D: ids 0-199 near the origin, ids 200-399 near
// (100,100,100). Any sane 2-means splits them, so cluster membership is checkable without
// depending on which label each group happens to get.
void create_two_group_table(KMeansFixture& fixture, const std::string& table)
{
  fixture.run_ok("CREATE TABLE " + table + " (id INTEGER, vec FLOAT[3]);");
  fixture.run_ok("INSERT INTO " + table +
                 " SELECT i, [(i % 5)::float, (i % 3)::float, (i % 7)::float] "
                 "FROM range(200) t(i);");
  fixture.run_ok("INSERT INTO " + table +
                 " SELECT 200 + i, "
                 "[100.0 + (i % 5), 100.0 + (i % 3), 100.0 + (i % 7)]::FLOAT[3] "
                 "FROM range(200) t(i);");
  fixture.run_ok("CHECKPOINT;");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + table +
                 "', tier => 'gpu', format => 'duckdb');");
}

}  // namespace

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_fit reports the knobs it resolved",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmr_corpus");

  auto r = query_ok(*con,
                    "SELECT n_clusters, dim, train_rows, n_rows FROM sirius_kmeans_fit("
                    "'kmr_corpus','vec', name => 'kmr_c', n_clusters => 2);");
  REQUIRE(r->RowCount() == 1);
  CHECK(r->GetValue(0, 0).GetValue<std::int64_t>() == 2);
  CHECK(r->GetValue(1, 0).GetValue<std::int64_t>() == 3);
  CHECK(r->GetValue(3, 0).GetValue<std::int64_t>() == 400);
  // The default sample is 256 rows per cluster, which the 400-row table cannot supply.
  CHECK(r->GetValue(2, 0).GetValue<std::int64_t>() == 400);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_fit defaults n_clusters to sqrt(n_rows)",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kma_corpus");

  auto const n = scalar_i64(*con,
                            "SELECT n_clusters FROM sirius_kmeans_fit("
                            "'kma_corpus','vec', name => 'kma_auto');");
  CHECK(n == 20);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign emits one edge per row and separates the two groups",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmb_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmb_corpus','vec', name => 'kmb_c', n_clusters => 2);");
  run_ok(
    "CREATE TABLE kmb_asg AS SELECT * FROM sirius_kmeans_assign('kmb_corpus','vec','kmb_c', "
    "n_probes => 1);");

  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmb_asg;") == 400);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT row_id) FROM kmb_asg;") == 400);
  CHECK(scalar_i64(*con, "SELECT min(row_id) FROM kmb_asg;") == 0);
  CHECK(scalar_i64(*con, "SELECT max(row_id) FROM kmb_asg;") == 399);
  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmb_asg WHERE cluster_id NOT IN (0,1);") == 0);
  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmb_asg WHERE distance < 0;") == 0);

  // Each separated group lands wholly in one cluster, and the two land in different ones.
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT cluster_id) FROM kmb_asg WHERE row_id < 200;") ==
        1);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT cluster_id) FROM kmb_asg WHERE row_id >= 200;") ==
        1);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT cluster_id) FROM kmb_asg;") == 2);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign repeats each row once per probe",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmc_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmc_corpus','vec', name => 'kmc_c', n_clusters => 4);");
  run_ok(
    "CREATE TABLE kmc_asg AS SELECT * FROM sirius_kmeans_assign('kmc_corpus','vec','kmc_c', "
    "n_probes => 3);");

  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmc_asg;") == 1200);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT row_id) FROM kmc_asg;") == 400);
  // A row never takes the same cluster twice.
  CHECK(scalar_i64(*con,
                   "SELECT count(*) FROM (SELECT row_id, cluster_id FROM kmc_asg "
                   "GROUP BY row_id, cluster_id HAVING count(*) > 1);") == 0);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign caps probes at the cluster count",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmd_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmd_corpus','vec', name => 'kmd_c', n_clusters => 2);");
  run_ok(
    "CREATE TABLE kmd_asg AS SELECT * FROM sirius_kmeans_assign('kmd_corpus','vec','kmd_c', "
    "n_probes => 10);");

  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmd_asg;") == 800);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign radius mode keeps every row and varies the edge count",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kme_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kme_corpus','vec', name => 'kme_c', n_clusters => 4);");
  run_ok(
    "CREATE TABLE kme_asg AS SELECT * FROM sirius_kmeans_assign('kme_corpus','vec','kme_c', "
    "radius_factor => 0.05, max_probes => 4);");

  // Every row keeps its nearest centroid, so no row can drop out...
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT row_id) FROM kme_asg;") == 400);
  // ...and the two groups are far enough apart that a 5% radius admits nothing across them,
  // which is what makes this fewer edges than a fixed n_probes => 4 would produce.
  CHECK(scalar_i64(*con, "SELECT count(*) FROM kme_asg;") < 1600);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign row ids line up with the table's rowid",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmf_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmf_corpus','vec', name => 'kmf_c', n_clusters => 2);");
  run_ok(
    "CREATE TABLE kmf_asg AS SELECT * FROM sirius_kmeans_assign('kmf_corpus','vec','kmf_c', "
    "n_probes => 1);");

  // The whole cluster-ordering flow rests on the assignment's row_id addressing the same row
  // the table's rowid does; the ids were built to make that checkable.
  con->Query("SET gpu_execution = false;");
  auto const mismatches = scalar_i64(*con,
                                     "SELECT count(*) FROM kmf_corpus c JOIN kmf_asg a ON c.rowid "
                                     "= a.row_id WHERE c.id <> a.row_id;");
  con->Query("SET gpu_execution = true;");
  CHECK(mismatches == 0);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans functions reject bad arguments",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmg_corpus");

  expect_error(*con,
               "SELECT * FROM sirius_kmeans_fit('kmg_corpus','id', name => 'x');",
               "must be a FLOAT[N] array column");
  expect_error(*con, "SELECT * FROM sirius_kmeans_fit('kmg_corpus','vec');", "'name'");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_fit('kmg_corpus','vec', name => 'x', n_iters => 0);",
               "n_iters must be >= 1");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_assign('kmg_corpus','vec','no_such_clustering');",
               "no clustering named");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_assign('kmg_corpus','vec','x', n_probes => 0);",
               "n_probes must be >= 1");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign rejects a clustering trained on a different width",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmh_corpus");
  run_ok("CREATE TABLE kmh_wide (id INTEGER, vec FLOAT[4]);");
  run_ok("INSERT INTO kmh_wide SELECT i, [i::float, 1, 2, 3] FROM range(100) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'kmh_wide', tier => 'gpu', format => 'duckdb');");

  run_ok("SELECT * FROM sirius_kmeans_fit('kmh_corpus','vec', name => 'kmh_c', n_clusters => 2);");
  expect_error(
    *con, "SELECT * FROM sirius_kmeans_assign('kmh_wide','vec','kmh_c');", "is over FLOAT[3]");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign matches a CPU-computed nearest centroid",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmo_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmo_corpus','vec', name => 'kmo_c', n_clusters => 4);");
  run_ok("CREATE TABLE kmo_cent AS SELECT * FROM sirius_kmeans_centroids('kmo_c');");
  run_ok(
    "CREATE TABLE kmo_asg AS SELECT * FROM sirius_kmeans_assign('kmo_corpus','vec','kmo_c', "
    "n_probes => 1);");

  // The oracle is DuckDB on the CPU: pivot the centroids back to one row each, compute every
  // (row, centroid) distance in SQL, and keep each row's argmin. Nothing here shares code with
  // the GPU path, so agreement is evidence the assignment is right rather than self-consistent.
  con->Query("SET gpu_execution = false;");
  auto const disagreements =
    scalar_i64(*con,
               "WITH cent AS ("
               "  SELECT cluster_id,"
               "         max(CASE WHEN dim_index = 0 THEN value END) AS c0,"
               "         max(CASE WHEN dim_index = 1 THEN value END) AS c1,"
               "         max(CASE WHEN dim_index = 2 THEN value END) AS c2"
               "  FROM kmo_cent GROUP BY cluster_id),"
               "ranked AS ("
               "  SELECT t.rowid AS row_id, k.cluster_id,"
               "         row_number() OVER (PARTITION BY t.rowid ORDER BY"
               "           (t.vec[1]-k.c0)*(t.vec[1]-k.c0) + (t.vec[2]-k.c1)*(t.vec[2]-k.c1)"
               "         + (t.vec[3]-k.c2)*(t.vec[3]-k.c2), k.cluster_id) AS rn"
               "  FROM kmo_corpus t CROSS JOIN cent k),"
               "oracle AS (SELECT row_id, cluster_id FROM ranked WHERE rn = 1)"
               "SELECT count(*) FROM oracle o JOIN kmo_asg a USING (row_id)"
               "  WHERE o.cluster_id <> a.cluster_id;");
  con->Query("SET gpu_execution = true;");

  CHECK(disagreements == 0);
}

// -----------------------------------------------------------------------------
// The approximate join. Its first gate is not recall but exactness: probing every
// cluster skips nothing, so the answer must match the exhaustive join it is built on.
// -----------------------------------------------------------------------------

namespace {

// A corpus stored in cluster order, plus a probe table, plus a clustering over both.
//
// @p scan_batch_bytes, when non-zero, lowers `scan_task_batch_size` so the corpus pins as more
// than one chunk. That knob -- not the row count -- is what decides chunking: pinned chunks are
// coalesced to a byte budget that defaults to 512 MB, so a corpus of any plausible test size is
// a single chunk. Measured: 300,000 FLOAT[3] rows is 3.6 MB and one chunk at the default, and
// three chunks at 256 KB.
void create_clustered_join(KMeansFixture& fixture,
                           duckdb::Connection& con,
                           const std::string& prefix,
                           int n_clusters,
                           int corpus_rows                = 4000,
                           std::uint64_t scan_batch_bytes = 0)
{
  auto const corpus = prefix + "_corpus";
  auto const probe  = prefix + "_probe";
  auto const clust  = prefix + "_c";

  if (scan_batch_bytes != 0) {
    fixture.run_ok("SET scan_task_batch_size = " + std::to_string(scan_batch_bytes) + ";");
  }

  fixture.run_ok("CREATE TABLE " + prefix + "_raw (id INTEGER, vec FLOAT[3]);");
  fixture.run_ok("INSERT INTO " + prefix +
                 "_raw SELECT i, "
                 "[(i % 97)::float, ((i * 7) % 89)::float, ((i * 13) % 83)::float] "
                 "FROM range(" +
                 std::to_string(corpus_rows) + ") t(i);");
  fixture.run_ok("CREATE TABLE " + probe + " (id INTEGER, vec FLOAT[3]);");
  fixture.run_ok("INSERT INTO " + probe +
                 " SELECT i, "
                 "[((i * 3) % 97)::float, ((i * 11) % 89)::float, ((i * 5) % 83)::float] "
                 "FROM range(200) t(i);");
  fixture.run_ok("CHECKPOINT;");

  // Cluster the raw table, then materialize a copy ordered by cluster. The order is not an
  // optimization: the join reads each chunk's cluster runs, and a chunk whose labels are not
  // non-decreasing is refused rather than silently answered from a partial range.
  fixture.run_ok("SELECT * FROM pin_table(name => '" + prefix +
                 "_raw', tier => 'gpu', format => 'duckdb');");
  fixture.run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_raw','vec', name => '" + clust +
                 "', n_clusters => " + std::to_string(n_clusters) + ");");
  fixture.run_ok("CREATE TABLE " + prefix + "_asg AS SELECT * FROM sirius_kmeans_assign('" +
                 prefix + "_raw','vec','" + clust + "', n_probes => 1);");
  fixture.run_ok("CREATE TABLE " + corpus + " AS SELECT r.id, r.vec, a.cluster_id FROM " + prefix +
                 "_raw r JOIN " + prefix + "_asg a ON r.rowid = a.row_id ORDER BY a.cluster_id;");
  fixture.run_ok("CHECKPOINT;");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + corpus +
                 "', tier => 'gpu', format => 'duckdb');");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + probe +
                 "', tier => 'gpu', format => 'duckdb');");
}

}  // namespace

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx probing every cluster equals the exhaustive join",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  constexpr int n_clusters = 8;
  create_clustered_join(*this, *con, "apx", n_clusters);

  // Compared on DISTANCES, not neighbour ids. Equidistant corpus rows are interchangeable
  // answers, and the two paths visit clusters in different orders, so they break those ties
  // differently -- an id comparison reports that as a failure and undercuts by exactly the tie
  // multiplicity. The distance multiset is what "same answer" actually means here.
  auto const exhaustive =
    distance_multiset(*con,
                      "SELECT left_id, distance FROM sirius_knn_join("
                      "'apx_probe','vec','apx_corpus','vec', "
                      "search_mode => 'exact-gemm', metric => 'l2', k => 5);");

  // n_probes == n_clusters wants every cluster, so nothing is skipped and the approximate path
  // must reproduce the exhaustive answer. This is what catches a mistake in the neighbour-id
  // base, which pruning would otherwise hide as a plausible-looking wrong answer.
  auto const before    = sirius::test::get_vector_join_prune_stats(*con);
  auto const probe_all = distance_multiset(*con,
                                           "SELECT left_id, distance FROM sirius_knn_join("
                                           "'apx_probe','vec','apx_corpus','vec', "
                                           "search_mode => 'approx', metric => 'l2', k => 5, "
                                           "clustering => 'apx_c', cluster_column => 'cluster_id', "
                                           "n_probes => " +
                                             std::to_string(n_clusters) + ");");

  REQUIRE(probe_all == exhaustive);

  // The other half of the gate. Reproducing the exhaustive answer is only evidence of a correct
  // neighbour-id base if this run really did visit every cluster -- and it is the same counter
  // the pruning test asserts is small, so asserting it is maximal here is what stops that test
  // passing on a counter that is simply always zero.
  auto const after            = sirius::test::get_vector_join_prune_stats(*con);
  auto const scored           = after.pairs_scored - before.pairs_scored;
  auto const exhaustive_pairs = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(exhaustive_pairs > 0);
  CHECK(scored == exhaustive_pairs);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx returns fewer, still-valid neighbours when pruning",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_clustered_join(*this, *con, "apy", 8);

  // One cluster per probe row prunes hard. Recall is expected to drop -- that is the trade --
  // but every pair returned must still be a real corpus row, and every probe row must be
  // answered, which is what separates "approximate" from "broken".
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  auto const pruned = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'apy_probe','vec','apy_corpus','vec', "
                              "search_mode => 'approx', metric => 'l2', k => 5, "
                              "clustering => 'apy_c', cluster_column => 'cluster_id', "
                              "n_probes => 1);");
  CHECK(pruned.size() == 200 * 5);

  // What "approximate" has to mean. Every assertion below this one holds just as well for a run
  // that scored the whole corpus and called itself pruned -- which is exactly what the
  // chunk-granularity version did while passing this test. One cluster of eight is ~12.5% of the
  // corpus; the bound is loose because k-means does not make clusters equal.
  auto const after      = sirius::test::get_vector_join_prune_stats(*con);
  auto const scored     = after.pairs_scored - before.pairs_scored;
  auto const exhaustive = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(exhaustive > 0);
  CHECK(scored * 4 < exhaustive);

  // Counted here rather than in SQL: an aggregate directly over the join falls back to the CPU,
  // which cannot execute the rewrite target at all.
  std::set<std::string> answered;
  std::set<std::string> neighbours;
  for (auto const& row : pruned) {
    answered.insert(row.at(0));
    neighbours.insert(row.at(1));
  }
  // Every probe row is answered...
  CHECK(answered.size() == 200);

  // ...and every neighbour is a real corpus row, which is what a mis-based neighbour id would
  // break: pruning changes which rows come back, never whether they exist.
  auto const corpus_ids = ok_rows(*con, "SELECT id FROM apy_corpus;");
  std::set<std::string> valid;
  for (auto const& row : corpus_ids) {
    valid.insert(row.at(0));
  }
  for (auto const& n : neighbours) {
    CHECK(valid.count(n) == 1);
  }
}

// -----------------------------------------------------------------------------
// The same two gates against a corpus of more than one pinned chunk.
//
// Every clustered test above uses a corpus small enough to arrive as one chunk, and
// that is the one shape in which the clustered path cannot be wrong about chunks: it
// staged chunk 0 and treated a cluster's row range as an offset into it, which is
// correct for one chunk and a read past the end of it for two.
//
// Getting a second chunk takes the batch-size knob, not more rows. Pinned chunks are
// coalesced to `scan_task_batch_size`, which defaults to 512 MB -- so SIFT1M, at
// exactly 512 MB, is itself a single chunk, and no corpus a test can afford to build
// would ever have split on row count alone. That is why this was never covered.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx spans more than one corpus chunk",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  constexpr int n_clusters           = 8;
  constexpr int corpus_rows          = 300000;
  constexpr std::uint64_t scan_batch = 256 * 1024;  // measured: 3 chunks, 10 cluster runs
  create_clustered_join(*this, *con, "apm", n_clusters, corpus_rows, scan_batch);

  auto const exhaustive =
    distance_multiset(*con,
                      "SELECT left_id, distance FROM sirius_knn_join("
                      "'apm_probe','vec','apm_corpus','vec', "
                      "search_mode => 'exact-gemm', metric => 'l2', k => 5);");

  auto const before    = sirius::test::get_vector_join_prune_stats(*con);
  auto const probe_all = distance_multiset(*con,
                                           "SELECT left_id, distance FROM sirius_knn_join("
                                           "'apm_probe','vec','apm_corpus','vec', "
                                           "search_mode => 'approx', metric => 'l2', k => 5, "
                                           "clustering => 'apm_c', cluster_column => 'cluster_id', "
                                           "n_probes => " +
                                             std::to_string(n_clusters) + ");");

  REQUIRE(probe_all == exhaustive);

  // The premise of this test, asserted rather than assumed -- and it has already earned its
  // keep: the first version of this test set the row count and no batch size, so the corpus
  // arrived as one chunk and the test covered nothing. Only this assertion said so. The counter
  // accumulates one corpus-chunk count per probe batch, and 200 probe rows are a single batch,
  // so the delta is the corpus's chunk count.
  auto const after = sirius::test::get_vector_join_prune_stats(*con);
  CHECK(after.chunks_available - before.chunks_available > 1);

  // Pruning across chunks is where a neighbour id is most easily mis-based: an id is local to
  // the slice it came from, and turning it back into a corpus row now takes the chunk's base
  // as well as the slice's own start. A wrong base still returns k plausible ids per probe
  // row, so what catches it is that every id must name a real corpus row.
  auto const pruned = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'apm_probe','vec','apm_corpus','vec', "
                              "search_mode => 'approx', metric => 'l2', k => 5, "
                              "clustering => 'apm_c', cluster_column => 'cluster_id', "
                              "n_probes => 1);");
  CHECK(pruned.size() == 200 * 5);

  std::set<std::string> answered;
  std::set<std::string> neighbours;
  for (auto const& row : pruned) {
    answered.insert(row.at(0));
    neighbours.insert(row.at(1));
  }
  CHECK(answered.size() == 200);

  auto const corpus_ids = ok_rows(*con, "SELECT id FROM apm_corpus;");
  std::set<std::string> valid;
  for (auto const& row : corpus_ids) {
    valid.insert(row.at(0));
  }
  for (auto const& n : neighbours) {
    CHECK(valid.count(n) == 1);
  }
}

// -----------------------------------------------------------------------------
// A corpus that is not stored in cluster order is refused, not answered.
//
// The join reads each chunk's labels as a run per cluster. An unordered corpus makes
// that reading wrong rather than merely slow -- a cluster would be a partial range,
// and the rows outside it would never be scored while the answer still looked whole.
//
// The refusal is thrown at execution, not at bind, because it needs the corpus labels
// on the device. If DuckDB turns that into a CPU fallback the query still fails, but
// with the GPU-only-TVF message instead of this one -- in which case it is the needle
// that is wrong here, not the refusal.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx refuses a corpus not in cluster order",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_clustered_join(*this, *con, "apu", 8);

  // Same rows, same clustering, only the storage order differs.
  run_ok("CREATE TABLE apu_shuffled AS SELECT * FROM apu_corpus ORDER BY id;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'apu_shuffled', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT left_id, distance FROM sirius_knn_join("
               "'apu_probe','vec','apu_shuffled','vec', "
               "search_mode => 'approx', metric => 'l2', k => 5, "
               "clustering => 'apu_c', cluster_column => 'cluster_id', n_probes => 2);",
               "not stored in cluster order");
}

// -----------------------------------------------------------------------------
// The approximate join no longer needs the corpus pinned.
//
// Clustering used to refuse build_source => 'scan' outright, so an approximate join
// meant materializing a cluster-ordered copy AND pinning it -- and a pin of a column
// subset makes that table's other columns unusable in the same query. The cluster ids
// now ride along as a column of the corpus scan, in the same batches the fold
// searches, which is the only way they can mean the same row order the neighbour ids
// are resolved against.
//
// Verified against the pin path rather than against a recomputed oracle: the two must
// answer identically, which is what isolates "where the bytes came from" from "what
// the search did". Compared on distances because the build side's batch order is a
// race -- any order is correct as long as every stage uses the same one, but two
// orders break ties between equidistant rows differently.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx takes the corpus from a child scan",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_clustered_join(*this, *con, "apb", 8);

  auto const pinned = distance_multiset(*con,
                                        "SELECT left_id, distance FROM sirius_knn_join("
                                        "'apb_probe','vec','apb_corpus','vec', "
                                        "search_mode => 'approx', metric => 'l2', k => 5, "
                                        "clustering => 'apb_c', cluster_column => 'cluster_id', "
                                        "n_probes => 2, right_output_columns => ['id']);");

  auto const before  = sirius::test::get_vector_join_prune_stats(*con);
  auto const scanned = distance_multiset(*con,
                                         "SELECT left_id, distance FROM sirius_knn_join("
                                         "'apb_probe','vec','apb_corpus','vec', "
                                         "search_mode => 'approx', metric => 'l2', k => 5, "
                                         "clustering => 'apb_c', cluster_column => 'cluster_id', "
                                         "n_probes => 2, right_output_columns => ['id'], "
                                         "build_source => 'scan');");
  auto const after   = sirius::test::get_vector_join_prune_stats(*con);

  REQUIRE(scanned == pinned);

  // Agreeing with the pin path is not enough on its own: an unpruned build-phase run would
  // agree too, and more of the time. Two clusters of eight is ~25% of the corpus.
  auto const scored     = after.pairs_scored - before.pairs_scored;
  auto const exhaustive = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(exhaustive > 0);
  CHECK(scored * 2 < exhaustive);

  // The point of the change. Nothing above proves the pin was unused -- it was still there.
  run_ok("SELECT * FROM unpin_table('apb_corpus');");
  auto const unpinned = distance_multiset(*con,
                                          "SELECT left_id, distance FROM sirius_knn_join("
                                          "'apb_probe','vec','apb_corpus','vec', "
                                          "search_mode => 'approx', metric => 'l2', k => 5, "
                                          "clustering => 'apb_c', cluster_column => 'cluster_id', "
                                          "n_probes => 2, right_output_columns => ['id'], "
                                          "build_source => 'scan');");
  REQUIRE(unpinned == pinned);

  // The cluster ids are a column of that scan, so a cluster_column the catalog does not have
  // has to be caught at bind: the plan generator would throw instead, and a throw there reads
  // as "this query cannot run on the GPU" and is reported as something unrelated.
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apb_probe','vec','apb_corpus','vec', "
               "search_mode => 'approx', metric => 'l2', k => 5, clustering => 'apb_c', "
               "cluster_column => 'not_a_column', build_source => 'scan');",
               "not found in table");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join rejects approx without a usable clustering",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_two_group_table(*this, "apz_corpus");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "search_mode => 'approx', k => 2);",
               "needs clustering =>");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "search_mode => 'approx', k => 2, clustering => 'nope');",
               "requires cluster_column =>");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "k => 2, clustering => 'nope', cluster_column => 'c');",
               "only applies under search_mode");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "search_mode => 'approx', k => 2, clustering => 'nope', "
               "cluster_column => 'cluster_id');",
               "no clustering named");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx routes each probe row by its own nearest centroids",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  // Three groups: A near the origin, C near x=10, B spread over y=26..34. A probe at (0,14,0)
  // is nearer A's centroid (~14) than B's (~16), yet its nearest corpus row is in B (12 against
  // ~13.4). Routing the probe through its home cluster's nearest centroids visits A and C --
  // C's centroid is 10 from A's, B's is 30 -- and misses; routing the row by its own two nearest
  // centroids visits A and B and finds it. The groups carry spread in every axis because a
  // near-degenerate group let balanced k-means split A and C by parity instead of by position.
  run_ok("CREATE TABLE rt_raw (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO rt_raw SELECT i, [((i % 17) * 0.05)::float, ((i % 13) * 0.05)::float, "
    "((i % 7) * 0.05)::float] FROM range(300) t(i);");
  run_ok(
    "INSERT INTO rt_raw SELECT 300 + i, [(10 + (i % 17) * 0.05)::float, "
    "((i % 13) * 0.05)::float, ((i % 7) * 0.05)::float] FROM range(300) t(i);");
  run_ok(
    "INSERT INTO rt_raw SELECT 600 + i, [((i % 17) * 0.05)::float, (26 + (i % 9))::float, "
    "((i % 7) * 0.05)::float] FROM range(300) t(i);");
  run_ok("CREATE TABLE rt_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO rt_probe SELECT i, [(i * 0.01)::float, 14::float, 0::float] "
    "FROM range(10) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'rt_raw', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM sirius_kmeans_fit('rt_raw','vec', name => 'rt_c', n_clusters => 3);");
  run_ok(
    "CREATE TABLE rt_asg AS SELECT * FROM sirius_kmeans_assign('rt_raw','vec','rt_c', "
    "n_probes => 1);");
  run_ok(
    "CREATE TABLE rt_corpus AS SELECT r.id, r.vec, a.cluster_id FROM rt_raw r "
    "JOIN rt_asg a ON r.rowid = a.row_id ORDER BY a.cluster_id;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'rt_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'rt_probe', tier => 'gpu', format => 'duckdb');");

  // The oracle is per-row routing itself, computed on the CPU: each probe's nearest corpus row
  // among the clusters that sirius_kmeans_assign ranks as ITS OWN n nearest. That is what the
  // operator claims to do, and unlike the exact answer it holds for whatever clustering k-means
  // happened to converge to.
  auto const oracle = [&](int n_probes) {
    return distance_multiset(*con,
                             "SELECT q.id, min(array_distance(q.vec, c.vec)) FROM rt_probe q "
                             "JOIN sirius_kmeans_assign('rt_probe','vec','rt_c', n_probes => " +
                               std::to_string(n_probes) +
                               ") a ON a.row_id = q.rowid "
                               "JOIN rt_corpus c ON c.cluster_id = a.cluster_id GROUP BY q.id;");
  };
  auto const approx = [&](int n_probes) {
    return distance_multiset(*con,
                             "SELECT left_id, distance FROM sirius_knn_join("
                             "'rt_probe','vec','rt_corpus','vec', "
                             "search_mode => 'approx', metric => 'l2', k => 1, "
                             "clustering => 'rt_c', cluster_column => 'cluster_id', "
                             "n_probes => " +
                               std::to_string(n_probes) + ");");
  };
  REQUIRE(approx(1) == oracle(1));

  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  auto const routed = approx(2);
  auto const after  = sirius::test::get_vector_join_prune_stats(*con);
  REQUIRE(routed == oracle(2));

  // The geometric claim on top: two probes are enough to reach the exact answer, and the home
  // cluster alone is not -- so the second probe is what found it.
  auto const exact = distance_multiset(*con,
                                       "SELECT left_id, distance FROM sirius_knn_join("
                                       "'rt_probe','vec','rt_corpus','vec', "
                                       "search_mode => 'exact-gemm', metric => 'l2', k => 1);");
  CHECK(routed == exact);
  CHECK(approx(1) != exact);

  // ...and it got there by visiting two of the three clusters, not all of them.
  auto const scored = after.pairs_scored - before.pairs_scored;
  auto const total  = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(total > 0);
  CHECK(scored * 3 < total * 2 + total / 10);
}

// ---------------------------------------------------------------------------------------------
// The three planner gaps that blocked composing the join with the rest of a SQL plan.
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join under CREATE TABLE AS, INSERT and COPY runs on the GPU",
                 "[integration][gpu_execution][array][vss][kmeans][approx][boundary]")
{
  create_clustered_join(*this, *con, "snk", 8);
  auto const q = std::string(
    "SELECT left_id, right_id, distance FROM sirius_knn_join('snk_probe','vec','snk_corpus','vec', "
    "search_mode => 'exact-gemm', metric => 'l2', k => 5)");
  auto const direct = ok_rows(*con, q + ";");
  REQUIRE(direct.size() == 200 * 5);

  // CREATE TABLE AS: DuckDB's sink stays, the GPU join feeds it.
  run_ok("CREATE TABLE snk_ctas AS " + q + ";");
  auto const via_ctas = ok_rows(*con, "SELECT left_id, right_id, distance FROM snk_ctas;");
  REQUIRE(via_ctas == direct);

  // INSERT ... SELECT into an existing table.
  run_ok("CREATE TABLE snk_ins (left_id INTEGER, right_id INTEGER, distance FLOAT);");
  run_ok("INSERT INTO snk_ins " + q + ";");
  auto const via_insert = ok_rows(*con, "SELECT left_id, right_id, distance FROM snk_ins;");
  REQUIRE(via_insert == direct);

  // COPY ... TO a file and read it back.
  run_ok("COPY (" + q + ") TO '/var/tmp/vj_test_snk_copy.csv' (FORMAT csv, HEADER);");
  auto const via_copy = ok_rows(
    *con, "SELECT left_id, right_id, distance FROM read_csv('/var/tmp/vj_test_snk_copy.csv');");
  REQUIRE(via_copy.size() == direct.size());
}

TEST_CASE_METHOD(KMeansFixture,
                 "named math functions over the join's output stay on the GPU",
                 "[integration][gpu_execution][array][vss][kmeans][approx][boundary]")
{
  create_clustered_join(*this, *con, "mth", 8);
  // A named function used to be refused by the plan generator, which re-ran the whole query on
  // the CPU where the join has no implementation. The CPU reference is DuckDB computing the
  // same functions over the join's stored output.
  run_ok(
    "CREATE TABLE mth_out AS SELECT left_id, right_id, distance FROM sirius_knn_join("
    "'mth_probe','vec','mth_corpus','vec', search_mode => 'exact-gemm', metric => 'l2', "
    "k => 3);");
  auto const gpu =
    ok_rows(*con,
            "SELECT left_id, right_id, round(distance, 2), round(sqrt(distance), 3), "
            "abs(distance - 10.0), floor(distance), ceil(distance) "
            "FROM sirius_knn_join('mth_probe','vec','mth_corpus','vec', "
            "search_mode => 'exact-gemm', metric => 'l2', k => 3);");
  auto const cpu =
    ok_rows(*con,
            "SELECT left_id, right_id, round(distance, 2), round(sqrt(distance), 3), "
            "abs(distance - 10.0), floor(distance), ceil(distance) FROM mth_out;");
  REQUIRE(gpu.size() == cpu.size());
  // Values compare after rounding to 3 decimals: cuDF rounds half to even and DuckDB half away
  // from zero, and the two float paths differ in the last ulp.
  auto norm = [](std::vector<std::vector<std::string>> rows) {
    for (auto& r : rows) {
      for (std::size_t i = 2; i < r.size(); ++i) {
        r[i] = std::to_string(std::llround(std::stod(r[i]) * 100.0));
      }
    }
    std::sort(rows.begin(), rows.end());
    return rows;
  };
  REQUIRE(norm(gpu) == norm(cpu));
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join takes the corpus from a VIEW through build_source => 'scan'",
                 "[integration][gpu_execution][array][vss][kmeans][approx][boundary]")
{
  create_clustered_join(*this, *con, "vw", 8);
  // A filtered corpus as a view: no CTAS, no pin. The reference is the same filter applied to
  // the pinned join's output, which is the answer a corpus-side subquery must reproduce.
  run_ok("CREATE VIEW vw_even AS SELECT id, vec, cluster_id FROM vw_corpus WHERE id % 2 = 0;");
  // The reference: the pinned join's full answer at a depth large enough that every probe's
  // three nearest even rows are inside it, written to a table (the GPU-under-sink splice) and
  // then filtered on the CPU with a window, which the GPU plan does not support.
  run_ok(
    "CREATE TABLE vw_full AS SELECT left_id, right_id, distance FROM sirius_knn_join("
    "'vw_probe','vec','vw_corpus','vec', search_mode => 'exact-gemm', metric => 'l2', "
    "k => 40, right_output_columns => ['id']);");
  auto const reference = distance_multiset(*con,
                                           "SELECT left_id, distance FROM vw_full "
                                           "WHERE right_id % 2 = 0 "
                                           "QUALIFY row_number() OVER (PARTITION BY left_id "
                                           "ORDER BY distance) <= 3;");
  auto const via_view  = distance_multiset(*con,
                                          "SELECT left_id, distance FROM sirius_knn_join("
                                           "'vw_probe','vec','vw_even','vec', "
                                           "search_mode => 'exact-gemm', metric => 'l2', k => 3, "
                                           "right_output_columns => ['id'], "
                                           "build_source => 'scan');");
  REQUIRE(via_view.size() == 200 * 3);
  REQUIRE(via_view == reference);

  // A view corpus on the approximate path, cluster column carried through the view.
  auto const via_view_approx =
    distance_multiset(*con,
                      "SELECT left_id, distance FROM sirius_knn_join("
                      "'vw_probe','vec','vw_even','vec', "
                      "search_mode => 'approx', metric => 'l2', k => 3, "
                      "clustering => 'vw_c', cluster_column => 'cluster_id', "
                      "n_probes => 8, build_source => 'scan');");
  REQUIRE(via_view_approx == via_view);

  // Without build_source => 'scan' a view is refused with a message that says what to pass.
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('vw_probe','vec','vw_even','vec', "
               "search_mode => 'exact-gemm', metric => 'l2', k => 3);",
               "build_source => 'scan'");
}

// -----------------------------------------------------------------------------
// Cluster lists: sirius_kmeans_build_lists writes the pinned corpus in cluster order, and the
// join reads that copy when it is given a clustering without a cluster column.
//
// FLOAT[256] so that one host block holds only 1024 rows: 70,000 rows then span 69 blocks and
// two staged chunks on the HOST tier, and most lists cross a block boundary -- the shapes the
// copy-by-run scatter and the block-wise staging can get wrong. Probing every cluster must give
// back exactly what the exhaustive join gives, which is what makes this an oracle rather than a
// recall check: any row misplaced, duplicated or dropped by the lists changes some row's top-k.
// -----------------------------------------------------------------------------
namespace {

void create_lists_tables(KMeansFixture& fixture,
                         const std::string& prefix,
                         const std::string& tier,
                         bool bytes = false)
{
  // Hashed rather than modular so no two rows repeat: duplicate rows make every top-k a set of
  // exact ties, and the comparison below would then only be testing tie-breaking. With `bytes`
  // every component is an integer in [0, 255], which is what lets the lists store UINT8.
  auto const gen = [bytes](const std::string& seed) {
    return bytes ? "list_transform(range(256), lambda d: (hash(i * 1000 + d + " + seed +
                     ") % 256)::FLOAT)::FLOAT[256]"
                 : "list_transform(range(256), lambda d: ((hash(i * 1000 + d + " + seed +
                     ") % 1000)::FLOAT / 1000.0))::FLOAT[256]";
  };
  fixture.run_ok("CREATE TABLE " + prefix + "_corpus AS SELECT i::INTEGER AS id, " + gen("0") +
                 " AS vec FROM range(70000) t(i);");
  fixture.run_ok("CREATE TABLE " + prefix + "_probe AS SELECT i::INTEGER AS id, " +
                 gen("100000000") + " AS vec FROM range(50) t(i);");
  fixture.run_ok("CHECKPOINT;");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + prefix + "_corpus', tier => '" + tier +
                 "', format => 'duckdb');");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + prefix +
                 "_probe', tier => 'gpu', format => 'duckdb');");
}

}  // namespace

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join over cluster lists probing every cluster equals the exact join",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // FP32 lists on each tier, and byte-valued data, which the lists store as UINT8 on the device
  // whatever tier the pin is on and widen back when a chunk is staged.
  auto const tier   = GENERATE(std::string("gpu"), std::string("host"));
  auto const bytes  = GENERATE(false, true);
  auto const prefix = std::string("kml_") + tier + (bytes ? "_u8" : "_f32");
  create_lists_tables(*this, prefix, tier, bytes);

  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  auto built =
    query_ok(*con,
             "SELECT n_rows, n_clusters, tier, encoding FROM sirius_kmeans_build_lists('" + prefix +
               "_corpus','vec','" + prefix + "_c');");
  CHECK(built->GetValue(0, 0).GetValue<std::int64_t>() == 70000);
  CHECK(built->GetValue(1, 0).GetValue<std::int64_t>() == 16);
  CHECK(built->GetValue(2, 0).ToString() == (bytes ? std::string("gpu") : tier));
  CHECK(built->GetValue(3, 0).ToString() == (bytes ? "uint8" : "float32"));

  auto const join = [&](const std::string& extra) {
    return "SELECT left_id, distance FROM sirius_knn_join('" + prefix + "_probe','vec','" + prefix +
           "_corpus','vec', metric => 'l2', k => 5, " + extra + ");";
  };
  auto const exact = distance_multiset(*con, join("search_mode => 'exact-gemm'"));
  auto const lists = distance_multiset(
    *con, join("search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 16"));
  REQUIRE(exact.size() == 50 * 5);
  REQUIRE(lists.size() == exact.size());
  // Per probe row, the same k distances. Compared within 2e-3 rather than exactly: the two
  // searches tile the GEMM differently, so their float sums round differently.
  for (std::size_t i = 0; i < exact.size(); ++i) {
    CHECK(lists[i].first == exact[i].first);
    CHECK(std::llabs(lists[i].second - exact[i].second) <= 2);
  }

  // And the ids are pin rows: every neighbour is a corpus id at its reported distance.
  auto const rows    = ok_rows(*con,
                            "SELECT left_id, right_id FROM sirius_knn_join('" + prefix +
                              "_probe','vec','" + prefix +
                              "_corpus','vec', metric => 'l2', k => 5, search_mode => 'approx', "
                                 "clustering => '" +
                              prefix + "_c', n_probes => 16);");
  auto const ex_rows = ok_rows(*con,
                               "SELECT left_id, right_id FROM sirius_knn_join('" + prefix +
                                 "_probe','vec','" + prefix +
                                 "_corpus','vec', metric => 'l2', k => 5, "
                                 "search_mode => 'exact-gemm');");
  std::size_t same   = 0;
  std::set<std::vector<std::string>> ex_set(ex_rows.begin(), ex_rows.end());
  for (auto const& r : rows) {
    same += ex_set.count(r);
  }
  // Equal distance multisets already rule out a misplaced row; ties can still swap which of two
  // equidistant corpus rows is reported, so the pairs are only required to mostly agree.
  CHECK(same * 10 >= rows.size() * 9);
}

TEST_CASE_METHOD(KMeansFixture,
                 "cluster lists do not outlive a re-fit of their clustering",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  create_lists_tables(*this, "kmx", "gpu");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmx_corpus','vec', name => 'kmx_c', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('kmx_corpus','vec','kmx_c');");
  auto const join =
    "SELECT left_id FROM sirius_knn_join('kmx_probe','vec','kmx_corpus','vec', metric => 'l2', "
    "k => 5, search_mode => 'approx', clustering => 'kmx_c', n_probes => 2);";
  CHECK(ok_rows(*con, join).size() == 50 * 5);

  // Lists built under the old centroids would route rows to clusters the new ones do not have.
  run_ok("SELECT * FROM sirius_kmeans_fit('kmx_corpus','vec', name => 'kmx_c', n_clusters => 8);");
  expect_error(*con, join, "sirius_kmeans_build_lists");
}

TEST_CASE_METHOD(KMeansFixture,
                 "cluster lists store UINT8 only where it is lossless",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  create_lists_tables(*this, "kmu", "gpu");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmu_corpus','vec', name => 'kmu_c', n_clusters => 8);");
  // Values in [0, 1) are not bytes: asking for UINT8 is refused rather than rounded...
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_build_lists('kmu_corpus','vec','kmu_c', "
               "storage => 'uint8');",
               "integer in [0, 255]");
  // ...and left to choose, the build keeps FP32.
  auto built =
    query_ok(*con, "SELECT encoding FROM sirius_kmeans_build_lists('kmu_corpus','vec','kmu_c');");
  CHECK(built->GetValue(0, 0).ToString() == "float32");
}

TEST_CASE_METHOD(KMeansFixture,
                 "lists that could not answer exactly are refused at build",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // FLOAT[3]: no width the bounded search re-checks in FP32, so INT8 lists could never be
  // searched and FLOAT16 lists would answer from rounded rows. Both are refused, and the
  // refusal leaves lists already built for the clustering in place.
  create_two_group_table(*this, "kmrb_corpus");
  run_ok(
    "SELECT * FROM sirius_kmeans_fit('kmrb_corpus','vec', name => 'kmrb_c', n_clusters => 2);");
  auto built =
    query_ok(*con, "SELECT encoding FROM sirius_kmeans_build_lists('kmrb_corpus','vec','kmrb_c');");
  CHECK(built->GetValue(0, 0).ToString() == "uint8");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_build_lists('kmrb_corpus','vec','kmrb_c', "
               "storage => 'int8');",
               "multiple of 16 and at most 256");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_build_lists('kmrb_corpus','vec','kmrb_c', "
               "storage => 'float16');",
               "multiple of 16");
  CHECK(ok_rows(*con,
                "SELECT left_id FROM sirius_knn_join('kmrb_corpus','vec','kmrb_corpus','vec', "
                "metric => 'l2', k => 3, search_mode => 'approx', clustering => 'kmrb_c', "
                "n_probes => 2);")
          .size() == 400 * 3);
}

TEST_CASE_METHOD(KMeansFixture,
                 "float16 cluster lists still give the exact join's answer",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // Values in [0, 1) at 1/1000 steps, most of which half cannot represent: the lists hold rounded
  // rows, and probing every cluster must still give back the FP32 answer, because the search only
  // uses the half rows to filter and re-scores what passes against the FP32 ones.
  create_lists_tables(*this, "kmh", "host");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmh_corpus','vec', name => 'kmh_c', n_clusters => 16);");
  auto built = query_ok(*con,
                        "SELECT tier, encoding FROM sirius_kmeans_build_lists('kmh_corpus','vec',"
                        "'kmh_c', storage => 'float16');");
  CHECK(built->GetValue(0, 0).ToString() == "gpu");
  CHECK(built->GetValue(1, 0).ToString() == "float16");

  for (auto const k : {5, 50}) {
    auto const join = [&](const std::string& extra) {
      return "SELECT left_id, right_id, distance FROM sirius_knn_join('kmh_probe','vec',"
             "'kmh_corpus','vec', metric => 'l2', k => " +
             std::to_string(k) + ", " + extra + ");";
    };
    // Distances to 1e-5 relative, which half rows are nowhere near (theirs are off by ~1e-4),
    // and the pairs themselves: the two FP32 searches round differently, so only a near-tie may
    // swap which of two rows is reported.
    auto const fetch = [&](const std::string& extra) {
      std::vector<std::tuple<std::string, double, std::string>> out;
      for (auto const& r : ok_rows(*con, join(extra))) {
        out.emplace_back(r.at(0), std::stod(r.at(2)), r.at(1));
      }
      std::sort(out.begin(), out.end());
      return out;
    };
    auto const exact = fetch("search_mode => 'exact-gemm'");
    auto const lists = fetch("search_mode => 'approx', clustering => 'kmh_c', n_probes => 16");
    REQUIRE(exact.size() == static_cast<std::size_t>(50 * k));
    REQUIRE(lists.size() == exact.size());
    std::set<std::pair<std::string, std::string>> exact_pairs;
    for (auto const& [left, d, right] : exact) {
      exact_pairs.emplace(left, right);
    }
    std::size_t same = 0;
    for (std::size_t i = 0; i < exact.size(); ++i) {
      CHECK(std::get<0>(lists[i]) == std::get<0>(exact[i]));
      CHECK(std::abs(std::get<1>(lists[i]) - std::get<1>(exact[i])) <=
            1e-5 * std::max(1.0, std::get<1>(exact[i])));
      same += exact_pairs.count({std::get<0>(lists[i]), std::get<2>(lists[i])});
    }
    CHECK(same * 100 >= exact.size() * 99);
  }
}

TEST_CASE_METHOD(
  KMeansFixture,
  "a threshold join over UINT8 and float16 cluster lists gives the exact join's pairs",
  "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // The lists search the radius with the bounded tensor-core GEMM: UINT8 rows are scored exactly,
  // float16 rows only filter and what passes is re-scored in FP32 and held to the radius again.
  // Probing every cluster must then return the exact join's pairs, and fewer probes a subset.
  auto const bytes  = GENERATE(true, false);
  auto const prefix = std::string(bytes ? "kmr_u8" : "kmr_f16");
  create_lists_tables(*this, prefix, bytes ? "gpu" : "host", bytes);
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  auto built =
    query_ok(*con,
             "SELECT encoding FROM sirius_kmeans_build_lists('" + prefix + "_corpus','vec','" +
               prefix + "_c', storage => '" + (bytes ? "uint8" : "float16") + "');");
  CHECK(built->GetValue(0, 0).ToString() == (bytes ? "uint8" : "float16"));

  // A radius around the median 10th-nearest distance: some rows have many pairs, some none.
  std::vector<double> tenth;
  for (auto const& r :
       ok_rows(*con,
               "SELECT max(distance) FROM sirius_knn_join('" + prefix + "_probe','vec','" + prefix +
                 "_corpus','vec', metric => 'l2', k => 10, "
                 "search_mode => 'exact-gemm') GROUP BY left_id;")) {
    tenth.push_back(std::stod(r.at(0)));
  }
  std::sort(tenth.begin(), tenth.end());
  auto const eps = tenth.at(tenth.size() / 2);

  auto const fetch = [&](const std::string& extra) {
    std::map<std::pair<std::string, std::string>, double> out;
    for (auto const& r : ok_rows(*con,
                                 "SELECT left_id, right_id, distance FROM sirius_knn_join('" +
                                   prefix + "_probe','vec','" + prefix +
                                   "_corpus','vec', metric => 'l2', join_mode => 'threshold', "
                                   "eps => " +
                                   std::to_string(eps) + ", " + extra + ");")) {
      out[{r.at(0), r.at(1)}] = std::stod(r.at(2));
    }
    return out;
  };
  auto const exact = fetch("search_mode => 'exact-gemm'");
  REQUIRE(exact.size() > 50);
  // A pair only one side keeps must sit on the radius, where the two searches' roundings differ.
  auto const on_radius = [&](double d) { return std::abs(d - eps) <= 1e-4 * eps; };
  for (auto const probes : {16, 2}) {
    auto const lists = fetch("search_mode => 'approx', clustering => '" + prefix +
                             "_c', n_probes => " + std::to_string(probes));
    for (auto const& [pair, d] : lists) {
      auto const it = exact.find(pair);
      if (it == exact.end()) {
        CHECK(on_radius(d));
        continue;
      }
      CHECK(std::abs(d - it->second) <= 1e-5 * std::max(1.0, it->second));
    }
    if (probes == 16) {
      for (auto const& [pair, d] : exact) {
        if (lists.count(pair) == 0) { CHECK(on_radius(d)); }
      }
    } else {
      CHECK(lists.size() < exact.size());
    }
  }

  // With the distance unread, a pair the float16 filter already proves inside the radius keeps
  // that verdict instead of being re-scored. The proof leaves a margin far wider than FP32
  // rounding, so the pairs are exactly the re-scored search's.
  auto const all_probes =
    "search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 16";
  std::set<std::pair<std::string, std::string>> scored, unread;
  for (auto const& [pair, d] : fetch(all_probes)) {
    scored.insert(pair);
  }
  for (auto const& r : ok_rows(*con,
                               "SELECT left_id, right_id FROM sirius_knn_join('" + prefix +
                                 "_probe','vec','" + prefix +
                                 "_corpus','vec', metric => 'l2', join_mode => 'threshold', "
                                 "eps => " +
                                 std::to_string(eps) + ", " + all_probes + ");")) {
    unread.emplace(r.at(0), r.at(1));
  }
  CHECK(unread == scored);
}

namespace {

/// A radius at the 1/@p at-th quantile of @p pairs' distances, set halfway between two distinct
/// ones: a pair sitting on the radius is in or out depending on how each search rounds it.
std::string radius_between(std::vector<std::pair<std::string, double>> const& pairs, double at)
{
  std::vector<double> d;
  for (auto const& p : pairs) {
    d.push_back(p.second);
  }
  std::sort(d.begin(), d.end());
  auto const i = static_cast<std::size_t>(static_cast<double>(d.size()) / at);
  auto j       = i + 1;
  while (j < d.size() && d[j] <= d[i] * (1 + 1e-4)) {
    ++j;
  }
  REQUIRE(j < d.size());
  std::ostringstream out;
  out << std::setprecision(17) << (d[i] + d[j]) / 2;
  return out.str();
}

}  // namespace

TEST_CASE_METHOD(KMeansFixture,
                 "UINT8 and float16 cluster lists on the host tier give the exact join's answer",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // Lists that do not fit the device are copied in a chunk at a time, still compact, and searched
  // by the same bounded GEMMs as on the device. 70,000 rows span two chunks either way.
  auto const bytes  = GENERATE(true, false);
  auto const prefix = std::string(bytes ? "kmhu" : "kmhh");
  create_lists_tables(*this, prefix, "host", bytes);
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  auto built = query_ok(*con,
                        "SELECT tier, encoding FROM sirius_kmeans_build_lists('" + prefix +
                          "_corpus','vec','" + prefix + "_c', storage => '" +
                          (bytes ? "uint8" : "float16") + "', tier => 'host');");
  CHECK(built->GetValue(0, 0).ToString() == "host");
  CHECK(built->GetValue(1, 0).ToString() == (bytes ? "uint8" : "float16"));

  auto const fetch = [&](const std::string& mode) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const& r :
         ok_rows(*con,
                 "SELECT left_id, distance FROM sirius_knn_join('" + prefix + "_probe','vec','" +
                   prefix + "_corpus','vec', metric => 'l2', " + mode + ");")) {
      out.emplace_back(r.at(0), std::stod(r.at(1)));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  auto const same = [](auto const& a, auto const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].first == b[i].first);
      CHECK(std::abs(a[i].second - b[i].second) <= 1e-5 * std::max(1.0, b[i].second));
    }
  };
  auto const approx = "search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 16";
  for (auto const k : {1, 20}) {
    auto const kk = "k => " + std::to_string(k) + ", ";
    same(fetch(kk + approx), fetch(kk + "search_mode => 'exact-gemm'"));
  }
  auto const eps = "join_mode => 'threshold', eps => " +
                   radius_between(fetch("k => 10, search_mode => 'exact-gemm'"), 10.0 / 9) + ", ";
  auto const exact = fetch(eps + "search_mode => 'exact-gemm'");
  REQUIRE(exact.size() > 50);
  same(fetch(eps + approx), exact);
}

TEST_CASE_METHOD(
  KMeansFixture,
  "a few probe rows over UINT8 and float16 cluster lists give the exact join's answer",
  "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // Every slice then serves at most three probe rows, which the bounded search scores one corpus
  // row per thread instead of in tensor-core tiles; the answers must not change.
  auto const bytes  = GENERATE(true, false);
  auto const prefix = std::string(bytes ? "kmsu" : "kmsh");
  create_lists_tables(*this, prefix, bytes ? "gpu" : "host", bytes);
  run_ok("CREATE TABLE " + prefix + "_few AS SELECT * FROM " + prefix + "_probe WHERE id < 3;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => '" + prefix +
         "_few', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('" + prefix + "_corpus','vec','" + prefix +
         "_c', storage => '" + (bytes ? "uint8" : "float16") + "');");

  auto const fetch = [&](const std::string& mode) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const& r :
         ok_rows(*con,
                 "SELECT left_id, distance FROM sirius_knn_join('" + prefix + "_few','vec','" +
                   prefix + "_corpus','vec', metric => 'l2', " + mode + ");")) {
      out.emplace_back(r.at(0), std::stod(r.at(1)));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  auto const same = [](auto const& a, auto const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].first == b[i].first);
      CHECK(std::abs(a[i].second - b[i].second) <= 1e-5 * std::max(1.0, b[i].second));
    }
  };
  auto const approx = "search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 16";
  for (auto const k : {1, 10, 100}) {
    auto const kk = "k => " + std::to_string(k) + ", ";
    same(fetch(kk + approx), fetch(kk + "search_mode => 'exact-gemm'"));
  }
  auto const eps = "join_mode => 'threshold', eps => " +
                   radius_between(fetch("k => 50, search_mode => 'exact-gemm'"), 2) + ", ";
  auto const exact = fetch(eps + "search_mode => 'exact-gemm'");
  REQUIRE(exact.size() > 10);
  same(fetch(eps + approx), exact);
}

TEST_CASE_METHOD(KMeansFixture,
                 "corpus output columns are gathered from each neighbour's own row",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // The join reads a corpus output column only from the pin chunks its neighbours fall in. At
  // ~1 KB a row this corpus pins as more than one chunk, and the probes are copies of rows near
  // its end, so each one's nearest neighbour is itself and k => 1 touches the last chunk alone.
  // Every column must come from the neighbour's own row (tag is a function of id), and k => 5 must
  // be DuckDB's own top-k -- not another Sirius search, which would share this gather.
  auto const tier   = GENERATE(std::string("gpu"), std::string("host"));
  auto const prefix = "kmo_" + tier;
  run_ok("CREATE TABLE " + prefix + "_corpus AS SELECT i::INTEGER AS id, i * 7 + 3 AS tag, " +
         "list_transform(range(256), lambda d: ((hash(i * 1000 + d) % 1000)::FLOAT / 1000.0))" +
         "::FLOAT[256] AS vec FROM range(240000) t(i);");
  run_ok("CREATE TABLE " + prefix + "_probe AS SELECT id, vec FROM " + prefix +
         "_corpus WHERE id >= 220000 AND id % 2000 = 7;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => '" + prefix + "_corpus', tier => '" + tier +
         "', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => '" + prefix +
         "_probe', tier => 'gpu', format => "
         "'duckdb');");
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('" + prefix + "_corpus','vec','" + prefix +
         "_c');");

  auto const join = [&](int k) {
    return ok_rows(*con,
                   "SELECT left_id, right_id, right_tag FROM sirius_knn_join('" + prefix +
                     "_probe','vec','" + prefix + "_corpus','vec', metric => 'l2', k => " +
                     std::to_string(k) + ", search_mode => 'approx', clustering => '" + prefix +
                     "_c', n_probes => 16, left_output_columns => ['id'], "
                     "right_output_columns => ['id', 'tag']);");
  };
  auto const nearest = join(1);
  REQUIRE(nearest.size() == 10);
  for (auto const& r : nearest) {
    CHECK(r.at(1) == r.at(0));
    CHECK(std::stoll(r.at(2)) == std::stoll(r.at(1)) * 7 + 3);
  }

  auto const got = join(5);
  REQUIRE(got.size() == 10 * 5);
  std::set<std::pair<std::string, std::string>> got_pairs;
  for (auto const& r : got) {
    CHECK(std::stoll(r.at(2)) == std::stoll(r.at(1)) * 7 + 3);
    got_pairs.emplace(r.at(0), r.at(1));
  }
  con->Query("SET gpu_execution = false;");
  auto const want =
    ok_rows(*con,
            "SELECT p.id, n.id FROM " + prefix + "_probe p, LATERAL (SELECT c.id FROM " + prefix +
              "_corpus c ORDER BY array_distance(p.vec, c.vec) LIMIT 5) n;");
  con->Query("SET gpu_execution = true;");
  REQUIRE(want.size() == got.size());
  std::size_t same = 0;
  for (auto const& r : want) {
    same += got_pairs.count({r.at(0), r.at(1)});
  }
  // The GEMM and DuckDB round differently, so a near-tie may swap which row is k-th.
  CHECK(same * 100 >= want.size() * 98);
}

TEST_CASE_METHOD(KMeansFixture,
                 "int8 cluster lists still give the exact join's answer",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // Values in [0, 1) at 1/1000 steps, which 8-bit codes cannot hold: the lists keep codes, and
  // probing every cluster must still give back the FP32 answer, because the search only filters
  // with the codes (under a bound covering both sides' coding error) and re-scores what passes
  // against the FP32 rows. On both tiers, for top-k and for a radius.
  auto const tier   = GENERATE(std::string("gpu"), std::string("host"));
  auto const prefix = "kmq_" + tier;
  create_lists_tables(*this, prefix, tier);
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  auto built = query_ok(*con,
                        "SELECT encoding FROM sirius_kmeans_build_lists('" + prefix +
                          "_corpus','vec','" + prefix + "_c', storage => 'int8');");
  CHECK(built->GetValue(0, 0).ToString() == "int8");

  auto const fetch = [&](const std::string& mode) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const& r :
         ok_rows(*con,
                 "SELECT left_id, distance FROM sirius_knn_join('" + prefix + "_probe','vec','" +
                   prefix + "_corpus','vec', metric => 'l2', " + mode + ");")) {
      out.emplace_back(r.at(0), std::stod(r.at(1)));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  auto const same = [](auto const& a, auto const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].first == b[i].first);
      CHECK(std::abs(a[i].second - b[i].second) <= 1e-5 * std::max(1.0, b[i].second));
    }
  };
  auto const approx = "search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 16";
  for (auto const k : {5, 50}) {
    auto const kk = "k => " + std::to_string(k) + ", ";
    same(fetch(kk + approx), fetch(kk + "search_mode => 'exact-gemm'"));
  }
  auto const eps = "join_mode => 'threshold', eps => " +
                   radius_between(fetch("k => 10, search_mode => 'exact-gemm'"), 2) + ", ";
  auto const exact = fetch(eps + "search_mode => 'exact-gemm'");
  REQUIRE(exact.size() > 50);
  same(fetch(eps + approx), exact);
  // With the distance unread, pairs the codes already prove inside the radius skip the re-score.
  auto const pairs = [&](const std::string& columns) {
    std::set<std::pair<std::string, std::string>> out;
    for (auto const& r :
         ok_rows(*con,
                 "SELECT " + columns + " FROM sirius_knn_join('" + prefix + "_probe','vec','" +
                   prefix + "_corpus','vec', metric => 'l2', " + eps + approx + ");")) {
      out.emplace(r.at(0), r.at(1));
    }
    return out;
  };
  CHECK(pairs("left_id, right_id") == pairs("left_id, right_id, distance"));
  // One probe still re-scores: with a single cluster the answer is the exact one inside it.
  CHECK(fetch("k => 5, search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 1")
          .size() == 50 * 5);
}

TEST_CASE_METHOD(KMeansFixture,
                 "cosine lists of unit rows give the exact cosine join's answer",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // metric => 'cosine' keeps every row divided by its norm, so the bounded float16 search runs a
  // cosine join as L2 over unit vectors and re-scores in FP32. Probing every cluster must give
  // the exact cosine join's answer, top-k and threshold, on both tiers; an l2 join over those
  // lists would rank by the wrong distance and is refused.
  auto const tier   = GENERATE(std::string("gpu"), std::string("host"));
  auto const prefix = "kmcos_" + tier;
  create_lists_tables(*this, prefix, tier);
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  auto built =
    query_ok(*con,
             "SELECT encoding FROM sirius_kmeans_build_lists('" + prefix + "_corpus','vec','" +
               prefix + "_c', storage => 'float16', metric => 'cosine');");
  CHECK(built->GetValue(0, 0).ToString() == "float16");

  auto const fetch = [&](const std::string& mode) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const& r :
         ok_rows(*con,
                 "SELECT left_id, similarity FROM sirius_knn_join('" + prefix + "_probe','vec','" +
                   prefix + "_corpus','vec', metric => 'cosine', " + mode + ");")) {
      out.emplace_back(r.at(0), std::stod(r.at(1)));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  auto const same = [](auto const& a, auto const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].first == b[i].first);
      CHECK(std::abs(a[i].second - b[i].second) <= 1e-5);
    }
  };
  auto const approx = "search_mode => 'approx', clustering => '" + prefix + "_c', n_probes => 16";
  for (auto const k : {5, 50}) {
    auto const kk = "k => " + std::to_string(k) + ", ";
    same(fetch(kk + approx), fetch(kk + "search_mode => 'exact-gemm'"));
  }
  auto const eps = "join_mode => 'threshold', eps => " +
                   radius_between(fetch("k => 10, search_mode => 'exact-gemm'"), 2) + ", ";
  auto const exact = fetch(eps + "search_mode => 'exact-gemm'");
  REQUIRE(exact.size() > 50);
  same(fetch(eps + approx), exact);
  expect_error(*con,
               "SELECT count(*) FROM sirius_knn_join('" + prefix + "_probe','vec','" + prefix +
                 "_corpus','vec', metric => 'l2', k => 5, " + approx + ");",
               "metric => 'cosine'");
}

TEST_CASE_METHOD(KMeansFixture,
                 "a seeded sweep over UINT8 lists gives what the GEMM sweep gives",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // Device-resident UINT8 lists search each row's nearest cluster with the bounded kernel under a
  // bound taken from a sample of that cluster, even at one probe. SIRIUS_VSS_SEED=0 searches it
  // with a GEMM and a selection instead; the two must agree whatever the probe count and k.
  create_lists_tables(*this, "kmseed", "gpu", true);
  run_ok(
    "SELECT * FROM sirius_kmeans_fit('kmseed_corpus','vec', name => 'kmseed_c', n_clusters => "
    "16);");
  run_ok(
    "SELECT * FROM sirius_kmeans_build_lists('kmseed_corpus','vec','kmseed_c', storage => "
    "'uint8');");
  auto const fetch = [&](const std::string& args) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const& r : ok_rows(
           *con,
           "SELECT left_id, distance FROM sirius_knn_join('kmseed_probe','vec','kmseed_corpus',"
           "'vec', metric => 'l2', search_mode => 'approx', clustering => 'kmseed_c', " +
             args + ");")) {
      out.emplace_back(r.at(0), std::stod(r.at(1)));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  for (auto const probes : {1, 4}) {
    for (auto const k : {1, 10}) {
      auto const args   = "n_probes => " + std::to_string(probes) + ", k => " + std::to_string(k);
      auto const seeded = fetch(args);
      ::setenv("SIRIUS_VSS_SEED", "0", 1);
      auto const gemm = fetch(args);
      ::unsetenv("SIRIUS_VSS_SEED");
      REQUIRE(seeded.size() == gemm.size());
      REQUIRE(seeded.size() == static_cast<std::size_t>(50 * k));
      for (std::size_t i = 0; i < seeded.size(); ++i) {
        CHECK(seeded[i].first == gemm[i].first);
        CHECK(seeded[i].second == gemm[i].second);
      }
    }
  }
}

TEST_CASE_METHOD(KMeansFixture,
                 "a seeded sweep over INT8 and float16 lists gives what the GEMM sweep gives",
                 "[integration][gpu_execution][array][vss][kmeans][approx][lists]")
{
  // Values at 1/1000 steps, which neither 8-bit codes nor halves hold exactly: the seed's sample
  // distance is raised by the coding or rounding error before it bounds anything, and every
  // survivor is re-scored in FP32, so the seeded search must return the unseeded one's rows and
  // FP32 distances, at one probe, at a few, and at every cluster.
  auto const storage = GENERATE(std::string("int8"), std::string("float16"));
  auto const prefix  = "kmsd_" + storage;
  create_lists_tables(*this, prefix, "gpu");
  run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus','vec', name => '" + prefix +
         "_c', n_clusters => 16);");
  auto built = query_ok(*con,
                        "SELECT encoding FROM sirius_kmeans_build_lists('" + prefix +
                          "_corpus','vec','" + prefix + "_c', storage => '" + storage + "');");
  CHECK(built->GetValue(0, 0).ToString() == storage);
  auto const fetch = [&](const std::string& args) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const& r :
         ok_rows(*con,
                 "SELECT left_id, distance FROM sirius_knn_join('" + prefix + "_probe','vec','" +
                   prefix + "_corpus','vec', metric => 'l2', search_mode => 'approx', " +
                   "clustering => '" + prefix + "_c', " + args + ");")) {
      out.emplace_back(r.at(0), std::stod(r.at(1)));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  for (auto const probes : {1, 4, 16}) {
    for (auto const k : {1, 10}) {
      auto const args   = "n_probes => " + std::to_string(probes) + ", k => " + std::to_string(k);
      auto const seeded = fetch(args);
      ::setenv("SIRIUS_VSS_SEED", "0", 1);
      auto const gemm = fetch(args);
      ::unsetenv("SIRIUS_VSS_SEED");
      REQUIRE(seeded.size() == gemm.size());
      REQUIRE(seeded.size() == static_cast<std::size_t>(50 * k));
      for (std::size_t i = 0; i < seeded.size(); ++i) {
        CHECK(seeded[i].first == gemm[i].first);
        CHECK(std::abs(seeded[i].second - gemm[i].second) <= 1e-5 * std::max(1.0, gemm[i].second));
      }
    }
  }
}
