/*
 * Copyright 2025, Sirius Contributors.
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
 * @file test_gpu_execution_vector_join_sql_rewrite.cpp
 * @brief Plain-SQL vector joins (no sirius_knn_join call) run as the vector-join operator: each
 *        shape must run on the GPU -- without the rewrite DuckDB's plan for it falls back to the
 *        CPU -- and return what DuckDB returns for the same statement.
 */

#include "duckdb/main/config.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using SqlRewriteFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Rows of @p sql, sorted.
std::vector<std::vector<std::string>> sorted_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto rows = sirius::test::GpuExecutionFixture::collect_rows(
    r->Cast<duckdb::MaterializedQueryResult>(), /*sort=*/true);
  return rows;
}

/// Equal rows, numbers within 1e-4 relative: the GPU scores pairs with a GEMM and DuckDB
/// directly, and the two round differently in the last places.
bool rows_match(const std::vector<std::vector<std::string>>& a,
                const std::vector<std::vector<std::string>>& b)
{
  if (a.size() != b.size()) { return false; }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) { return false; }
    for (std::size_t j = 0; j < a[i].size(); ++j) {
      if (a[i][j] == b[i][j]) { continue; }
      char* ea       = nullptr;
      char* eb       = nullptr;
      double const x = std::strtod(a[i][j].c_str(), &ea);
      double const y = std::strtod(b[i][j].c_str(), &eb);
      if (*ea != '\0' || *eb != '\0') { return false; }
      if (std::fabs(x - y) > 1e-4 * std::max(1.0, std::fabs(y))) { return false; }
    }
  }
  return true;
}

/// Run @p sql on the GPU (asserting it ran there, one rebind and one execution) and on DuckDB,
/// and require the same rounded rows.
void require_gpu_matches_duckdb(duckdb::Connection& con, const std::string& sql)
{
  con.Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(con, sql);
  con.Query("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(con);
  auto const got    = sorted_rows(con, sql);
  auto const after  = sirius::test::get_transparent_execution_stats(con);
  UNSCOPED_INFO("query: " << sql);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(got.size() == expected.size());
  // Every query here leads with the row's ids, so both sides sort into the same order.
  REQUIRE(rows_match(got, expected));
}

}  // namespace

// Vectors from a hash, so no two corpus rows share a distance to a probe (ties would make a
// top-k answer a matter of tie-breaking), and a small category column to join and filter on.
class SqlRewriteTables {
 public:
  explicit SqlRewriteTables(SqlRewriteFixture& f) : _f(f)
  {
    // The database's optimizer mask is shared, and an earlier test in a full run can leave it
    // cleared; compressed materialization then wraps small-range columns in functions the GPU
    // plan does not translate, and the queries here run on DuckDB instead. Pin the mask Sirius
    // publishes at load while these tables exist.
    auto& disabled  = duckdb::DBConfig::GetConfig(*f.con->context).options.disabled_optimizers;
    _saved_disabled = disabled;
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    auto const vec = [](const std::string& seed) {
      return "list_transform(range(8), lambda d: ((hash(i * 100 + d + " + seed +
             ") % 2000)::FLOAT / 1000.0 - 1.0))::FLOAT[8]";
    };
    f.run_ok("CREATE TABLE sr_corpus AS SELECT i::INTEGER AS id, (i % 20)::INTEGER AS cat, " +
             vec("0") + " AS vec FROM range(20000) t(i);");
    f.run_ok("CREATE TABLE sr_probe AS SELECT i::INTEGER AS id, (i % 20)::INTEGER AS cat, " +
             vec("77777777") + " AS vec FROM range(30) t(i);");
    f.run_ok("CREATE TABLE sr_one AS SELECT 0::INTEGER AS id, " + vec("55555555") +
             " AS vec FROM range(1) t(i);");
    f.run_ok(
      "CREATE TABLE sr_cats AS SELECT i::INTEGER AS cat, 'c' || i AS name FROM range(20) t(i);");
    f.run_ok("CHECKPOINT;");
    for (auto const* t : {"sr_corpus", "sr_probe", "sr_one", "sr_cats"}) {
      f.run_ok(std::string("SELECT * FROM pin_table(name => '") + t +
               "', tier => 'gpu', format => 'duckdb');");
    }
  }
  ~SqlRewriteTables()
  {
    for (auto const* t : {"sr_corpus", "sr_probe", "sr_one", "sr_cats"}) {
      _f.con->Query(std::string("SELECT * FROM unpin_table('") + t + "');");
    }
    duckdb::DBConfig::GetConfig(*_f.con->context).options.disabled_optimizers = _saved_disabled;
  }

 private:
  SqlRewriteFixture& _f;
  std::set<duckdb::OptimizerType> _saved_disabled;
};

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain-SQL threshold join runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // l2 distance bound, the distance read back in the select list.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, array_distance(p.vec, c.vec) AS d FROM sr_probe p, "
    "sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;");
  // cosine similarity bound, with the distance form of the same pair in the select list.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, array_cosine_distance(p.vec, c.vec) AS d FROM sr_probe p, sr_corpus c "
    "WHERE array_cosine_similarity(p.vec, c.vec) >= 0.8;");
  // strict comparison and the constant on the left.
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM sr_probe p, sr_corpus c "
                             "WHERE 0.9 > array_distance(p.vec, c.vec);");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain-SQL threshold join keeps the corpus-side join and filter",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, k.name FROM sr_probe p, sr_corpus c, sr_cats k WHERE c.cat = k.cat AND "
    "k.cat < 5 AND array_cosine_similarity(p.vec, c.vec) >= 0.7;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain-SQL threshold join under a <> join condition runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // DuckDB makes `p.cat <> c.cat` the join and the similarity a filter above it; the join keeps
  // nearly every pair, so it becomes a filter over the vector join. `c.cat < 10` is pushed into
  // the corpus scan, and from there into the operator.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON array_cosine_similarity(p.vec, c.vec) "
    ">= 0.7 WHERE p.cat <> c.cat AND c.cat < 10;");
  // A filtered probe streams in as a relation; a bare pinned one is read from its pin.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id FROM (SELECT * FROM sr_probe WHERE id < 20) p JOIN sr_corpus c ON "
    "array_cosine_similarity(p.vec, c.vec) >= 0.7 WHERE p.cat <> c.cat;");
  // One range condition too, as a self-join counts each pair once; it keeps about half the pairs.
  require_gpu_matches_duckdb(
    *con,
    "SELECT a.id, b.id FROM sr_corpus a JOIN sr_corpus b ON array_distance(a.vec, b.vec) <= 0.6 "
    "WHERE a.id < b.id AND a.cat <> b.cat;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "LATERAL top-k with a join DuckDB moved into it runs as the per-row vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // DuckDB's join order puts the join with sr_cats inside the LATERAL's subquery side.
  require_gpu_matches_duckdb(
    *con,
    "SELECT k.name, count(*), round(avg(n.d), 4) FROM sr_probe p, LATERAL (SELECT c.cat, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 5) n JOIN sr_cats k ON "
    "n.cat = k.cat WHERE k.cat <> 3 GROUP BY k.name;");
  // A join on a probe column pushed in makes the LATERAL correlated on that column too.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, k.name, n.id FROM sr_probe p, LATERAL (SELECT c.id, c.cat, "
    "array_cosine_similarity(p.vec, c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 4) n JOIN "
    "sr_cats k ON k.cat = p.cat WHERE n.cat <> p.cat;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "LATERAL LIMIT 1 runs as the per-row vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // DuckDB decorrelates LIMIT 1 without the list and unnest: max(similarity) when only the score
  // is selected, arg_max(value, similarity) otherwise.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, round(t.s, 5) FROM sr_probe p, LATERAL (SELECT array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 1) t;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, t.id, round(t.d, 5) FROM sr_probe p, LATERAL (SELECT c.id, array_distance(p.vec, "
    "c.vec) AS d FROM sr_corpus c WHERE c.cat = 3 ORDER BY d LIMIT 1) t;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "LATERAL top-k runs as the per-row vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;");
  // A filter inside the subquery restricts the corpus before the top-k, as LATERAL's does.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id FROM sr_probe p, LATERAL (SELECT c.id, array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c WHERE c.cat = 3 ORDER BY s DESC LIMIT 4) n;");
  // Counting the matches reads none of the join's columns, the probe's vector included.
  require_gpu_matches_duckdb(
    *con,
    "SELECT count(*) FROM sr_probe p, LATERAL (SELECT c.id, array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 4) n;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "ORDER BY distance to a scalar-subquery vector runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT id, array_cosine_distance(vec, (SELECT vec FROM sr_one)) AS d "
                             "FROM sr_corpus ORDER BY d LIMIT 25;");
  // Every row's distance, joined and ranked afterwards.
  require_gpu_matches_duckdb(
    *con,
    "SELECT r.id, k.name, r.d FROM sr_cats k JOIN (SELECT id, cat, array_distance(vec, (SELECT vec "
    "FROM sr_one)) AS d FROM sr_corpus) r ON k.cat = r.cat WHERE k.cat = 7 ORDER BY r.d, r.id "
    "LIMIT 30;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a scalar subquery with more than one row still fails",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  auto r = con->Query(
    "SELECT id FROM sr_corpus ORDER BY array_distance(vec, (SELECT vec FROM sr_probe)) LIMIT 3;");
  REQUIRE(r);
  REQUIRE(r->HasError());
  REQUIRE(r->GetError().find("More than one row") != std::string::npos);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a vector read above the join leaves the plan to DuckDB",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // c.vec itself is selected, which the operator does not emit: no rewrite, and the CPU answer.
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con,
                                    "SELECT p.id, c.vec FROM sr_probe p, sr_corpus c "
                                    "WHERE array_distance(p.vec, c.vec) <= 0.6;");
  con->Query("SET gpu_execution = true;");
  auto const got = sorted_rows(*con,
                               "SELECT p.id, c.vec FROM sr_probe p, sr_corpus c "
                               "WHERE array_distance(p.vec, c.vec) <= 0.6;");
  REQUIRE(got == expected);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a top-k rewritten under a correlated subquery keeps DuckDB's answer",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // Vec-H q2's shape, scaled down: the top-k joins beside a correlated subquery that repeats the
  // outer join, under a filter that prunes columns by position. The rewritten operator emits the
  // columns the top-k did, in the same positions, so nothing above it is re-laid out.
  require_gpu_matches_duckdb(
    *con,
    "SELECT r.id, k.name, c.cat, r.d FROM sr_cats k, sr_corpus c, (SELECT id, array_distance(vec, "
    "(SELECT vec FROM sr_one)) AS d FROM sr_corpus ORDER BY d LIMIT 40) r WHERE k.cat = c.cat AND "
    "c.id = r.id AND length(k.name) >= 2 AND c.id >= (SELECT min(c2.id) FROM sr_cats k2, sr_corpus "
    "c2 WHERE k2.cat = c2.cat AND length(k2.name) >= 2 AND c2.cat = c.cat) ORDER BY r.d, r.id;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a vector join in a plan Sirius declines elsewhere still runs on DuckDB",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // The correlated scalar subquery plans as a SINGLE join, which the GPU does not run. The
  // rewritten join has no CPU form, so the rewrite must be undone and DuckDB answer as usual.
  auto const sql =
    "SELECT r.id, k.name, r.d FROM sr_cats k, (SELECT id, cat, array_distance(vec, (SELECT vec "
    "FROM "
    "sr_one)) AS d FROM sr_corpus ORDER BY d LIMIT 40) r WHERE k.cat = r.cat AND r.id = (SELECT "
    "min(c2.id) FROM sr_corpus c2 WHERE c2.cat = k.cat AND c2.id >= r.id) ORDER BY r.d, r.id;";
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con, sql);
  con->Query("SET gpu_execution = true;");
  auto const got = sorted_rows(*con, sql);
  REQUIRE(got.size() == expected.size());
  REQUIRE(rows_match(got, expected));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain SQL searches through cluster lists that answer exactly",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // FLOAT32 lists (these values are not bytes) answer exactly once every cluster is probed, but
  // hold the same bytes as the pin, so by cost the rewrite keeps brute force: the prune
  // statistics, which only the clustered search records, stay put. Forced through the lists they
  // move, and the answers stay DuckDB's. The lists hold rows as they are, not unit rows, so a
  // cosine join keeps the exact GEMM either way.
  run_ok("SELECT * FROM sirius_kmeans_fit('sr_corpus','vec', name => 'sr_c', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('sr_corpus','vec','sr_c');");
  auto const unforced = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive ==
        unforced.pairs_exhaustive);
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  } unset_access_path;
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;");
  auto const topk = sirius::test::get_vector_join_prune_stats(*con);
  CHECK(topk.pairs_exhaustive > before.pairs_exhaustive);
  CHECK(topk.pairs_scored - before.pairs_scored == topk.pairs_exhaustive - before.pairs_exhaustive);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p, sr_corpus c "
                             "WHERE array_distance(p.vec, c.vec) <= 0.9;");
  auto const radius = sirius::test::get_vector_join_prune_stats(*con);
  CHECK(radius.pairs_exhaustive > topk.pairs_exhaustive);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id FROM sr_probe p, LATERAL (SELECT c.id, array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 4) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive ==
        radius.pairs_exhaustive);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a band join can run as a vector join with the band as a filter",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // Two range conditions between the sides (a band) next to a distance threshold;
  // SIRIUS_VSS_BAND_VECTOR_FIRST=1 takes the vector join first with the band as a filter whatever
  // the estimate says, on the GPU, with DuckDB's answer.
  SqlRewriteTables tables(*this);
  ::setenv("SIRIUS_VSS_BAND_VECTOR_FIRST", "1", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_BAND_VECTOR_FIRST"); }
  } unset_band;
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON "
                             "c.cat BETWEEN p.cat - 2 AND p.cat + 2 "
                             "WHERE array_distance(p.vec, c.vec) <= 0.9;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a self-join DuckDB shares as a materialized CTE is still rewritten",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // The same relation on both sides (probe joined to its category, twice): DuckDB's common-subplan
  // pass turns it into a CTE scanned twice, which carries no vectors to search until the rewrite
  // puts it back in place.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT a.id, b.id FROM sr_probe a JOIN sr_cats ca ON a.cat = ca.cat "
                             "JOIN sr_probe b ON array_distance(a.vec, b.vec) <= 1.5 "
                             "JOIN sr_cats cb ON b.cat = cb.cat WHERE a.id < b.id;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a band join's own statistics decide whether the vector join goes first",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // cat is 0..19, so a +-2 band keeps about 4/19 of the pairs: vector join first, on the GPU. id
  // is 0..19,999, so a band one wide keeps about 1/20,000: left to DuckDB's band join, same answer.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON "
                             "c.cat BETWEEN p.cat - 2 AND p.cat + 2 "
                             "WHERE array_distance(p.vec, c.vec) <= 0.9;");

  auto const narrow =
    "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON "
    "c.id BETWEEN p.id * 600 AND p.id * 600 + 1 WHERE array_distance(p.vec, c.vec) <= 3.0;";
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con, narrow);
  con->Query("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto const got    = sorted_rows(*con, narrow);
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);
  REQUIRE(!expected.empty());
  REQUIRE(rows_match(got, expected));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a join over a pinned corpus with no lists can build them inside the query",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // With no lists, SIRIUS_VSS_ACCESS_PATH=lists has the rewrite fit a clustering and write lists
  // of the pinned column first (as SIRIUS_VSS_BUILD_IN_QUERY=1 does when the cost model says the
  // build pays), then search them in full: DuckDB's answer, and the lists stay for later queries.
  // Byte values and a width of 16, so the lists it builds are UINT8 and searched exactly.
  SqlRewriteTables tables(*this);
  auto const bytes = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
           ") % 256)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_bytes AS SELECT i::INTEGER AS id, " + bytes("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_bytes_probe AS SELECT i::INTEGER AS id, " + bytes("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_bytes', tier => 'gpu', format => 'duckdb');");
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  } unset_access_path;
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_bytes_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_bytes c ORDER BY d LIMIT 5) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive > before.pairs_exhaustive);
  auto centroids =
    con->Query("SELECT count(*) FROM sirius_kmeans_centroids('__sirius_auto_sr_bytes_vec');");
  REQUIRE_FALSE(centroids->HasError());
  CHECK(centroids->GetValue(0, 0).GetValue<std::int64_t>() > 0);
}
