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
                 "count(*) over a threshold join is counted inside the join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // The aggregate directly over the join: the join counts, the aggregate sums the batches.
  std::string const counted =
    "SELECT count(*) FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;";
  require_gpu_matches_duckdb(*con, counted);
  REQUIRE(con->ExtractPlan(counted)->ToString().find("sum_no_overflow") != std::string::npos);
  // The explicit call, a cosine threshold, against the same count on DuckDB.
  std::string const called =
    "SELECT count(*) FROM sirius_knn_join('sr_probe','vec','sr_corpus','vec', metric => "
    "'cosine', join_mode => 'threshold', eps => 0.8);";
  REQUIRE(con->ExtractPlan(called)->ToString().find("sum_no_overflow") != std::string::npos);
  run_ok("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con,
                                    "SELECT count(*) FROM sr_probe p, sr_corpus c "
                                    "WHERE array_cosine_similarity(p.vec, c.vec) >= 0.8;");
  run_ok("SET gpu_execution = true;");
  auto const got = sorted_rows(*con, called);
  REQUIRE(got == expected);
  // Shapes the pushdown must leave alone still answer: a second aggregate, groups, a filter
  // above the join.
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*), min(array_distance(p.vec, c.vec)) FROM sr_probe "
                             "p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.cat, count(*) FROM sr_probe p, sr_corpus c WHERE "
                             "array_distance(p.vec, c.vec) <= 0.9 GROUP BY p.cat;");
  std::string const filtered =
    "SELECT count(*) FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9 "
    "AND p.id <> c.id;";
  require_gpu_matches_duckdb(*con, filtered);
  REQUIRE(con->ExtractPlan(filtered)->ToString().find("sum_no_overflow") == std::string::npos);
  // Over the lists: the count equals the number of pairs the same approximate join returns.
  run_ok("SET vector_join_probes = 2;");
  auto const approx_rows = sorted_rows(
    *con,
    "SELECT p.id, c.id FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;");
  auto const approx_count = sorted_rows(*con, counted);
  run_ok("RESET vector_join_probes;");
  REQUIRE(approx_count.size() == 1);
  REQUIRE(approx_count[0][0] == std::to_string(approx_rows.size()));
  // Enough probe rows for the kernel to score them in several tiles, so the count crosses tile
  // boundaries (a 512 MB score tile holds 6 400 rows against this corpus).
  run_ok(
    "CREATE TABLE sr_probe_big AS SELECT i::INTEGER AS id, list_transform(range(8), lambda "
    "d: ((hash(i * 100 + d + 31337) % 2000)::FLOAT / 1000.0 - 1.0))::FLOAT[8] AS vec FROM "
    "range(15000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_probe_big', tier => 'gpu', format => 'duckdb');");
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM sr_probe_big p, sr_corpus c WHERE "
                             "array_distance(p.vec, c.vec) <= 1.2;");
  con->Query("SELECT * FROM unpin_table('sr_probe_big');");
  run_ok("DROP TABLE sr_probe_big;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a range on the similarity or the distance runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // DuckDB folds two bounds on one expression into a BETWEEN. The join takes the bound that
  // makes it a threshold join and the other is applied to the score it returns.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p, sr_corpus c "
                             "WHERE array_cosine_similarity(p.vec, c.vec) BETWEEN 0.7 AND 0.9;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, array_distance(p.vec, c.vec) AS d FROM sr_probe p, "
    "sr_corpus c WHERE array_distance(p.vec, c.vec) BETWEEN 0.5 AND 0.9;");
  // The upper bound sits above the join, on the score the select list reads.
  require_gpu_matches_duckdb(
    *con,
    "SELECT id, cid FROM (SELECT p.id, c.id AS cid, array_cosine_similarity(p.vec, c.vec) AS s "
    "FROM sr_probe p JOIN sr_corpus c ON array_cosine_similarity(p.vec, c.vec) >= 0.7) WHERE s < "
    "0.9;");
  // Strict on both sides.
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM sr_probe p, sr_corpus c "
                             "WHERE array_distance(p.vec, c.vec) > 0.5 AND array_distance(p.vec, "
                             "c.vec) < 0.9;");
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
                 "SET vector_join_probes makes a plain SQL join approximate through the lists",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  run_ok("SELECT * FROM sirius_kmeans_fit('sr_corpus','vec', name => 'sr_c', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('sr_corpus','vec','sr_c');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clustering;");
    }
  } reset{*con};
  // Ids are row positions, so the table function's answer reads as the plain SQL one does.
  auto const topk_sql =
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;";
  auto const radius_sql =
    "SELECT p.id, c.id FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;";
  auto const tvf = [](std::string const& args) {
    return "sirius_knn_join('sr_probe','vec','sr_corpus','vec', search_mode => 'approx', "
           "clustering => 'sr_c', n_probes => 2, " +
           args + ")";
  };
  auto const topk_tvf =
    sorted_rows(*con, "SELECT left_id, right_id, distance FROM " + tvf("k => 7") + ";");
  auto const radius_tvf = sorted_rows(
    *con, "SELECT left_id, right_id FROM " + tvf("join_mode => 'threshold', eps => 0.9") + ";");

  run_ok("SET vector_join_probes = 2;");
  for (auto const& [sql, expected] :
       {std::pair{topk_sql, topk_tvf}, std::pair{radius_sql, radius_tvf}}) {
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto const got    = sorted_rows(*con, sql);
    UNSCOPED_INFO("query: " << sql);
    sirius::test::require_transparent_execution_delta(
      before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
    REQUIRE(rows_match(got, expected));
  }
  // Two of eight clusters drop some true neighbours, so the approximate answer is not DuckDB's.
  con->Query("SET gpu_execution = false;");
  auto const exact = sorted_rows(*con, topk_sql);
  con->Query("SET gpu_execution = true;");
  CHECK_FALSE(rows_match(topk_tvf, exact));

  // Naming the clustering selects the same lists; a name that is not this column's is ignored.
  run_ok("SET vector_join_clustering = 'sr_c';");
  CHECK(rows_match(sorted_rows(*con, topk_sql), topk_tvf));
  run_ok("SET vector_join_clustering = 'no_such_clustering';");
  CHECK(rows_match(sorted_rows(*con, topk_sql), topk_tvf));

  // Every cluster probed, or the setting back at 0, is the exact answer again.
  run_ok("SET vector_join_probes = 8;");
  require_gpu_matches_duckdb(*con, topk_sql);
  run_ok("RESET vector_join_probes;");
  require_gpu_matches_duckdb(*con, topk_sql);
  require_gpu_matches_duckdb(*con, radius_sql);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "SET vector_join_probes with no lists builds them inside the query",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // No clustering exists for the corpus: the setting has the rewrite fit one and write its lists
  // inside the statement, then search p of them. The lists stay, so the table function over the
  // auto-built clustering returns the same answer, and probing every cluster is DuckDB's.
  SqlRewriteTables tables(*this);
  auto const bytes = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
           ") % 256)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_fresh AS SELECT i::INTEGER AS id, " + bytes("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_fresh_probe AS SELECT i::INTEGER AS id, " + bytes("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_fresh', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit() { con.Query("RESET vector_join_probes;"); }
  } reset{*con};
  auto const sql =
    "SELECT p.id, n.id, n.d FROM sr_fresh_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_fresh c ORDER BY d LIMIT 5) n;";
  run_ok("SET vector_join_probes = 2;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto const got    = sorted_rows(*con, sql);
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
  auto centroids =
    con->Query("SELECT count(*) FROM sirius_kmeans_centroids('__sirius_auto_sr_fresh_vec');");
  REQUIRE_FALSE(centroids->HasError());
  auto const n_clusters = centroids->GetValue(0, 0).GetValue<std::int64_t>();
  REQUIRE(n_clusters > 2);
  auto const tvf = sorted_rows(
    *con,
    "SELECT left_id, right_id, distance FROM sirius_knn_join('sr_fresh_probe','vec','sr_fresh',"
    "'vec', probe_source => 'scan', search_mode => 'approx', "
    "clustering => '__sirius_auto_sr_fresh_vec', n_probes => 2, k => 5);");
  REQUIRE(rows_match(got, tvf));
  run_ok("SET vector_join_probes = " + std::to_string(n_clusters) + ";");
  require_gpu_matches_duckdb(*con, sql);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "vector_join_clusters sets the count of an in-query build, and probes follow it",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // 4,000 rows: the reference count is 63. An explicit vector_join_clusters builds that many
  // lists and searches exactly vector_join_probes of them. With the setting back at 0 the probes
  // are stated against the reference and scaled to what the kept lists have: 2 of 63 is
  // ceil(2 * sqrt(252 / 63)) = 4 of 252.
  SqlRewriteTables tables(*this);
  auto const bytes = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
           ") % 256)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_count AS SELECT i::INTEGER AS id, " + bytes("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_count_probe AS SELECT i::INTEGER AS id, " + bytes("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_count', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clusters;");
    }
  } reset{*con};
  auto const sql =
    "SELECT p.id, n.id, n.d FROM sr_count_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_count c ORDER BY d LIMIT 5) n;";
  auto const tvf = [&](int probes) {
    return sorted_rows(
      *con,
      "SELECT left_id, right_id, distance FROM sirius_knn_join('sr_count_probe','vec','sr_count',"
      "'vec', probe_source => 'scan', search_mode => 'approx', "
      "clustering => '__sirius_auto_sr_count_vec', n_probes => " +
        std::to_string(probes) + ", k => 5);");
  };
  run_ok("SET vector_join_clusters = 252;");
  run_ok("SET vector_join_probes = 2;");
  auto const fixed = sorted_rows(*con, sql);
  auto centroids   = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
      "sirius_kmeans_centroids('__sirius_auto_sr_count_vec');");
  REQUIRE_FALSE(centroids->HasError());
  REQUIRE(centroids->GetValue(0, 0).GetValue<std::int64_t>() == 252);
  auto const two = tvf(2), four = tvf(4);
  REQUIRE_FALSE(rows_match(two, four));
  CHECK(rows_match(fixed, two));

  run_ok("RESET vector_join_clusters;");
  CHECK(rows_match(sorted_rows(*con, sql), four));
  // Every reference cluster is still every cluster.
  run_ok("SET vector_join_probes = 63;");
  require_gpu_matches_duckdb(*con, sql);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a large batch of pairs reads a host-pinned corpus's columns chunk by chunk",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // 30,000 probe rows x top-10 is 300,000 values per corpus column, past what a task gathers
  // straight from host memory, so each task copies in the corpus chunks its neighbours fall in
  // (three row groups here) on the reservation it was admitted with. Both columns of a pair
  // must come from the same corpus row, and a few probe rows' pairs must be DuckDB's.
  SqlRewriteTables tables(*this);
  run_ok(
    "CREATE TABLE sr_wide AS SELECT i::INTEGER AS id, (i * 7 + 3)::BIGINT AS tag, "
    "[sin(i)::FLOAT, cos(i * 1.3)::FLOAT, sin(i * 0.7)::FLOAT, cos(i * 0.31)::FLOAT]::FLOAT[4] "
    "AS vec FROM range(300000) t(i);");
  run_ok(
    "CREATE TABLE sr_wide_probe AS SELECT i::INTEGER AS id, "
    "[sin(i * 2.1)::FLOAT, cos(i * 0.9)::FLOAT, sin(i * 1.7)::FLOAT, cos(i * 0.53)::FLOAT]"
    "::FLOAT[4] AS vec FROM range(30000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_wide', tier => 'host', format => 'duckdb');");
  std::string const pairs =
    " FROM sr_wide_probe p, LATERAL (SELECT c.id, c.tag, array_distance(p.vec, c.vec) AS d "
    "FROM sr_wide c ORDER BY d LIMIT 10) n";
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto const got    = sorted_rows(
    *con,
    "SELECT count(*), sum((n.tag - n.id * 7 - 3) * (n.tag - n.id * 7 - 3)), min(n.id) >= 0, "
       "max(n.id) < 300000" +
      pairs + ";");
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
  REQUIRE(got.size() == 1);
  CHECK(got[0][0] == "300000");
  CHECK(got[0][1] == "0");
  CHECK(got[0][2] == "true");
  CHECK(got[0][3] == "true");
  require_gpu_matches_duckdb(*con, "SELECT p.id, n.id, n.tag" + pairs + " WHERE p.id < 20;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a join over a pinned corpus with no lists can build them inside the query",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // With no lists, SIRIUS_VSS_ACCESS_PATH=lists has the rewrite write the exact join's compact
  // layout first -- one list of codes, no clustering to fit -- (as SIRIUS_VSS_BUILD_IN_QUERY=1 does
  // when the cost model says it pays), then search it whole: DuckDB's answer, and the list stays
  // for later queries under its own name, apart from the approximate join's partitions.
  // Byte values and a width of 16, so the codes are UINT8 and searched exactly.
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
  auto centroids = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_codes_sr_bytes_vec');");
  REQUIRE_FALSE(centroids->HasError());
  CHECK(centroids->GetValue(0, 0).GetValue<std::int64_t>() == 1);

  // An approximate join does not take the single list for its partitions: it builds its own.
  run_ok("SET vector_join_probes = 2;");
  struct reset_on_exit {
    duckdb::Connection& c;
    ~reset_on_exit() { c.Query("RESET vector_join_probes;"); }
  } reset_probes{*con};
  auto const approx = con->Query(
    "SELECT count(*) FROM (SELECT p.id, n.id FROM sr_bytes_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_bytes c ORDER BY d LIMIT 5) n);");
  REQUIRE_FALSE(approx->HasError());
  CHECK(approx->GetValue(0, 0).GetValue<std::int64_t>() == 30 * 5);
  auto partitions = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_sr_bytes_vec');");
  REQUIRE_FALSE(partitions->HasError());
  CHECK(partitions->GetValue(0, 0).GetValue<std::int64_t>() > 1);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "lists built inside the query over float vectors answer exactly",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // Values that are not bytes: the lists the query builds are FLOAT16 with their FP32 rows, and
  // every candidate is re-scored in FP32, so the answer is DuckDB's.
  SqlRewriteTables tables(*this);
  auto const floats = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: ((hash(i * 100 + d + " + seed +
           ") % 100000) / 977.0)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_floats AS SELECT i::INTEGER AS id, " + floats("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_floats_probe AS SELECT i::INTEGER AS id, " + floats("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_floats', tier => 'gpu', format => 'duckdb');");
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  } unset_access_path;
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_floats_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_floats c ORDER BY d LIMIT 5) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive > before.pairs_exhaustive);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a probe that is a pinned table is read from its pin",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // A self-join over a pinned table reads both sides from the pin; a probe with a filter is still
  // scanned. DuckDB's answers either way.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(
    *con,
    "SELECT a.id, n.id, n.d FROM sr_probe a, LATERAL (SELECT b.id, array_distance(a.vec, b.vec) "
    "AS d FROM sr_probe b ORDER BY d LIMIT 3) n;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT a.id, n.id FROM sr_corpus a, LATERAL (SELECT b.id, array_cosine_similarity(a.vec, "
    "b.vec) AS s FROM sr_corpus b ORDER BY s DESC LIMIT 2) n WHERE a.id < 50;");
  // The 30 self-pairs at distance 0, then the closest pair in both orders.
  require_gpu_matches_duckdb(*con,
                             "SELECT a.id, b.id, array_distance(a.vec, b.vec) AS d FROM sr_probe "
                             "a, sr_probe b ORDER BY d LIMIT 32;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM sr_corpus c WHERE c.cat = 3 ORDER BY d LIMIT 4) n;");
}
