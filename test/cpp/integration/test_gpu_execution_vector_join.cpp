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
 * @file test_gpu_execution_vector_join.cpp
 * @brief End-to-end tests for the sirius_knn_join() table function.
 *
 * sirius_knn_join is a Sirius-owned surface: for every probe row it finds the k
 * nearest corpus rows (brute force) under a distance metric, and emits the probe
 * output columns, the corpus output columns, then a trailing FLOAT `distance`.
 * Both tables must be pinned (GPU or HOST tier), and the four tier combinations
 * are all exercised here.
 *
 * Correctness is checked against a pure-DuckDB (gpu_execution=false) oracle over
 * array_distance / array_cosine_distance.
 *
 * The dataset is chosen so every assertion is tie-free at the k values tested.
 * Corpus row i has vec [i, i+1, i+2]; probe row with base a has vec [a, a+1, a+2],
 * so the difference is the constant vector [i-a, i-a, i-a] and the L2 distance is
 * exactly sqrt(3)*|i-a|. Distances therefore come in symmetric shells around each
 * probe: {a} at 0, {a-1, a+1} at sqrt(3), {a-2, a+2} at 2*sqrt(3), ... An ODD
 * per-row k (1,3,5,7) lands on complete shells, so the top-k SET is unambiguous.
 * The five probe bases are >=10 apart, so no two probes share a nearby corpus row
 * and the global shells (5 pairs at 0, 10 at sqrt(3), 10 at 2*sqrt(3), ...) are
 * likewise unambiguous at shell-completing global k (5, 15, 25).
 *
 * Data is checkpointed before pinning: pin_table(format='duckdb') reads on-disk
 * blocks through the native scan path, so WAL-resident rows would be invisible.
 *
 * The tiled-corpus path (n_corpus_tiles > 1, and therefore
 * merge_corpus_tile_candidates) triggers only when the corpus staging estimate
 * exceeds free GPU memory, which no realistic SQL-level input can force here.
 * The last test caps the staging budget explicitly with max_stage_bytes to reach
 * it; everywhere else the corpus is staged contiguously.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius_context.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using VectorJoinFixture = sirius::test::GpuExecutionFixture;

namespace {

// Sorted rows from a query that must succeed. Both sides of a comparison sort
// identically, so set equality holds regardless of the physical row order.
std::vector<std::vector<std::string>> ok_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  return sirius::test::GpuExecutionFixture::collect_rows(mat, /*sort=*/true);
}

// Assert a query fails, and (when given) that its error mentions `needle`.
void expect_error(duckdb::Connection& con, const std::string& sql, const std::string& needle = "")
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  if (!needle.empty()) {
    UNSCOPED_INFO("error was: " << r->GetError());
    REQUIRE(r->GetError().find(needle) != std::string::npos);
  }
}

// Number of pinned chunks for a table, on whichever tier it was pinned to.
std::size_t pinned_chunk_count(duckdb::Connection& con, const std::string& name, bool host)
{
  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  const auto* entry = sirius_ctx->get_scan_manager().find_pinned_entry(name);
  REQUIRE(entry != nullptr);
  if (host) { return entry->host_chunks.size(); }
  auto it = entry->data_batches_by_column.find("vec");
  REQUIRE(it != entry->data_batches_by_column.end());
  return it->second.size();
}

// Per-probe-row top-k from a pure-DuckDB oracle. Only valid where the ranking is tie-free.
std::vector<std::vector<std::string>> oracle_per_row(duckdb::Connection& con,
                                                     const std::string& probe,
                                                     const std::string& corpus,
                                                     int k)
{
  con.Query("SET gpu_execution = false;");
  auto rows = ok_rows(con,
                      "SELECT p.id, c.id FROM " + probe + " p, " + corpus +
                        " c QUALIFY row_number() OVER (PARTITION BY p.id ORDER BY "
                        "array_distance(p.vec, c.vec)) <= " +
                        std::to_string(k) + ";");
  con.Query("SET gpu_execution = true;");
  return rows;
}

// The five probe rows: id -> base value a (vec is [a, a+1, a+2]).
const std::map<int, int> kProbeBase = {{0, 49800}, {1, 49890}, {2, 49810}, {3, 49780}, {4, 49820}};

// Create + checkpoint the corpus/probe tables and pin them at the given tiers.
void setup(VectorJoinFixture& fx, const std::string& corpus_tier, const std::string& probe_tier)
{
  fx.run_ok("CREATE TABLE corpus (id INTEGER, vec FLOAT[3]);");
  fx.run_ok(
    "INSERT INTO corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] FROM range(50000) t(i);");
  fx.run_ok("CREATE TABLE probe (id INTEGER, vec FLOAT[3]);");
  fx.run_ok(
    "INSERT INTO probe VALUES "
    "(0, [49800.0, 49801.0, 49802.0]),"
    "(1, [49890.0, 49891.0, 49892.0]),"
    "(2, [49810.0, 49811.0, 49812.0]),"
    "(3, [49780.0, 49781.0, 49782.0]),"
    "(4, [49820.0, 49821.0, 49822.0]);");
  fx.run_ok("CHECKPOINT;");
  fx.run_ok("SELECT * FROM pin_table(name => 'corpus', tier => '" + corpus_tier +
            "', format => 'duckdb');");
  fx.run_ok("SELECT * FROM pin_table(name => 'probe', tier => '" + probe_tier +
            "', format => 'duckdb');");
}

}  // namespace

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - l2 per-row/global/threshold across tier combinations",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Every (corpus tier, probe tier) combination: resident-corpus + resident-probe,
  // host-streamed corpus and/or host-streamed probe.
  auto tiers = GENERATE(table<std::string, std::string>({
    {"gpu", "gpu"},
    {"host", "host"},
    {"gpu", "host"},
    {"host", "gpu"},
  }));
  INFO("corpus tier=" << std::get<0>(tiers) << " probe tier=" << std::get<1>(tiers));
  setup(*this, std::get<0>(tiers), std::get<1>(tiers));

  auto gpu_ids = [&](const std::string& tail) {
    return ok_rows(*con,
                   "SELECT \"probe.id\", \"corpus.id\" FROM sirius_knn_join("
                   "'probe','vec','corpus','vec', probe_output_columns => ['id'], "
                   "corpus_output_columns => ['id']" +
                     tail + ");");
  };
  auto cpu_ids = [&](const std::string& sql) {
    con->Query("SET gpu_execution = false;");
    auto rows = ok_rows(*con, sql);
    con->Query("SET gpu_execution = true;");
    return rows;
  };

  // All groups run sequentially against one pinned setup per tier combo (kept flat
  // rather than SECTIONs so the 50k-row pin isn't rebuilt per assertion group).

  // Per-row top-k vs exact oracle. Odd k lands on complete tie shells, so the
  // top-k SET is unambiguous under ties.
  for (int k : {1, 3, 5, 7}) {
    INFO("per-row k = " << k);
    auto oracle = cpu_ids(
      "SELECT p.id, c.id FROM probe p, corpus c QUALIFY row_number() OVER "
      "(PARTITION BY p.id ORDER BY array_distance(p.vec, c.vec)) <= " +
      std::to_string(k) + ";");
    REQUIRE(gpu_ids(", k => " + std::to_string(k)) == oracle);
  }

  // k => 1: each probe matches its own base at distance 0, and the trailing
  // distance column is exactly 0.
  {
    auto r = con->Query(
      "SELECT \"probe.id\", \"corpus.id\", distance FROM sirius_knn_join("
      "'probe','vec','corpus','vec', k => 1, probe_output_columns => ['id'], "
      "corpus_output_columns => ['id']) ORDER BY \"probe.id\";");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(mat.RowCount() == kProbeBase.size());
    for (duckdb::idx_t i = 0; i < mat.RowCount(); i++) {
      int const pid  = mat.GetValue(0, i).GetValue<int32_t>();
      int const cid  = mat.GetValue(1, i).GetValue<int32_t>();
      double const d = mat.GetValue(2, i).GetValue<double>();
      INFO("probe id " << pid);
      REQUIRE(cid == kProbeBase.at(pid));
      REQUIRE(d == Approx(0.0).margin(1e-3));
    }
  }

  // Global top-k vs exact oracle at shell-completing k.
  for (int k : {5, 15, 25}) {
    INFO("global k = " << k);
    auto oracle = cpu_ids(
      "SELECT p.id, c.id FROM probe p, corpus c ORDER BY "
      "array_distance(p.vec, c.vec) LIMIT " +
      std::to_string(k) + ";");
    REQUIRE(gpu_ids(", k => " + std::to_string(k) + ", mode => 'global'") == oracle);
  }

  // Global returns exactly k rows, ranked nearest-first. k = 10 straddles a tie
  // shell (5 pairs at 0, 10 at sqrt(3)), so assert only count + ordering.
  {
    auto r = con->Query(
      "SELECT distance FROM sirius_knn_join('probe','vec','corpus','vec', "
      "k => 10, mode => 'global');");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(mat.RowCount() == 10);
    double prev = -1.0;
    for (duckdb::idx_t i = 0; i < mat.RowCount(); i++) {
      double const d = mat.GetValue(0, i).GetValue<double>();
      REQUIRE(d >= prev - 1e-4);
      prev = d;
    }
  }

  // Per-row threshold: sqrt(3) ~= 1.732 <= 2.0, 2*sqrt(3) ~= 3.464 > 2.0, so 3
  // neighbors survive per probe.
  {
    auto oracle = cpu_ids(
      "SELECT p.id, c.id FROM probe p, corpus c WHERE array_distance(p.vec, c.vec) <= 2.0;");
    REQUIRE(gpu_ids(", k => 20, threshold => 2.0") == oracle);
  }

  // Global + threshold: bound first, then globally rank. k is large enough that
  // every surviving pair is returned.
  {
    auto within_2 = cpu_ids(
      "SELECT p.id, c.id FROM probe p, corpus c WHERE array_distance(p.vec, c.vec) <= 2.0;");
    REQUIRE(gpu_ids(", k => 20, mode => 'global', threshold => 2.0") == within_2);

    auto within_5 = cpu_ids(
      "SELECT p.id, c.id FROM probe p, corpus c WHERE array_distance(p.vec, c.vec) <= 5.0;");
    REQUIRE(gpu_ids(", k => 40, mode => 'global', threshold => 5.0") == within_5);
  }

  con->Query("SELECT * FROM unpin_table('corpus');");
  con->Query("SELECT * FROM unpin_table('probe');");
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - cosine metric matches exact top-k",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Corpus directions along a strictly increasing angle in the x-z plane; every
  // probe direction sits at a smaller angle than the whole corpus, so cosine
  // distance to corpus i is 1 - cos(theta_i - phi) -- strictly increasing in i.
  // Nearest for every probe is therefore ids 0,1,2,... in order (tie-free).
  run_ok(
    "CREATE TABLE vj_cos_corpus AS SELECT i AS id, "
    "[sin(0.3 + i * 0.0013)::float, 0.0::float, cos(0.3 + i * 0.0013)::float]::FLOAT[3] AS vec "
    "FROM range(1500) t(i);");
  run_ok(
    "CREATE TABLE vj_cos_probe AS SELECT j AS id, "
    "[sin(0.05 + j * 0.03)::float, 0.0::float, cos(0.05 + j * 0.03)::float]::FLOAT[3] AS vec "
    "FROM range(6) t(j);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vj_cos_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'vj_cos_probe', tier => 'gpu', format => 'duckdb');");

  for (int k : {1, 5, 20}) {
    INFO("cosine k = " << k);
    con->Query("SET gpu_execution = false;");
    auto oracle = ok_rows(*con,
                          "SELECT p.id, c.id FROM vj_cos_probe p, vj_cos_corpus c QUALIFY "
                          "row_number() OVER (PARTITION BY p.id ORDER BY "
                          "array_cosine_distance(p.vec, c.vec)) <= " +
                            std::to_string(k) + ";");
    con->Query("SET gpu_execution = true;");
    auto got = ok_rows(*con,
                       "SELECT \"vj_cos_probe.id\", \"vj_cos_corpus.id\" FROM sirius_knn_join("
                       "'vj_cos_probe','vec','vj_cos_corpus','vec', k => " +
                         std::to_string(k) +
                         ", metric => 'cosine', probe_output_columns => ['id'], "
                         "corpus_output_columns => ['id']);");
    REQUIRE(got == oracle);
  }

  run_ok("SELECT * FROM unpin_table('vj_cos_corpus');");
  run_ok("SELECT * FROM unpin_table('vj_cos_probe');");
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - output schema (default all, subset, order, k>rows)",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Both sides carry a VARCHAR passthrough: strings take a different path through
  // gather/repeat and through host_table_chunk_reader than the fixed-width columns.
  run_ok(
    "CREATE TABLE vj_p AS SELECT i AS id, ('p' || i) AS tag, "
    "[i::float, i::float]::FLOAT[2] AS vec FROM range(3) t(i);");
  run_ok(
    "CREATE TABLE vj_c AS SELECT i AS id, (i * 10) AS label, ('c' || i) AS name, "
    "[i::float, i::float]::FLOAT[2] AS vec FROM range(4) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vj_p', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'vj_c', tier => 'gpu', format => 'duckdb');");

  SECTION("omitted output columns => all base columns of both tables + trailing distance")
  {
    auto r = con->Query("SELECT * FROM sirius_knn_join('vj_p','vec','vj_c','vec', k => 2);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    // probe(id, tag, vec) + corpus(id, label, name, vec) + distance
    REQUIRE(r->names.size() == 8);
    REQUIRE(r->names[0] == "vj_p.id");
    REQUIRE(r->names[1] == "vj_p.tag");
    REQUIRE(r->names[2] == "vj_p.vec");
    REQUIRE(r->names[3] == "vj_c.id");
    REQUIRE(r->names[4] == "vj_c.label");
    REQUIRE(r->names[5] == "vj_c.name");
    REQUIRE(r->names[6] == "vj_c.vec");
    REQUIRE(r->names[7] == "distance");
    auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(mat.RowCount() == 3 * 2);

    // The vector column is both the search column and an output column, so it is
    // staged once and both roles read the same slot. A wrong output_slots mapping
    // still yields these names, so check the values: within a row, vec must agree
    // with that row's own id ([i, i] for id i), and the passthroughs with it.
    // Tie-independent, so it holds whichever neighbors k => 2 returns.
    for (duckdb::idx_t i = 0; i < mat.RowCount(); i++) {
      INFO("row " << i);
      auto const pid = mat.GetValue(0, i).GetValue<int32_t>();
      auto const cid = mat.GetValue(3, i).GetValue<int32_t>();
      REQUIRE(mat.GetValue(1, i).GetValue<std::string>() == "p" + std::to_string(pid));
      REQUIRE(mat.GetValue(4, i).GetValue<int32_t>() == cid * 10);
      REQUIRE(mat.GetValue(5, i).GetValue<std::string>() == "c" + std::to_string(cid));
      auto const& pvec = duckdb::ArrayValue::GetChildren(mat.GetValue(2, i));
      auto const& cvec = duckdb::ArrayValue::GetChildren(mat.GetValue(6, i));
      REQUIRE(pvec.size() == 2);
      REQUIRE(cvec.size() == 2);
      for (std::size_t d = 0; d < 2; d++) {
        REQUIRE(pvec[d].GetValue<float>() == Approx(static_cast<float>(pid)));
        REQUIRE(cvec[d].GetValue<float>() == Approx(static_cast<float>(cid)));
      }
    }
  }

  SECTION("explicit subset + column order is honored, distance appended last")
  {
    // The subset omits the vector column, so nothing maps back to staged slot 0.
    auto r = con->Query(
      "SELECT * FROM sirius_knn_join('vj_p','vec','vj_c','vec', k => 2, "
      "probe_output_columns => ['tag', 'id'], "
      "corpus_output_columns => ['name', 'label', 'id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->names.size() == 6);
    REQUIRE(r->names[0] == "vj_p.tag");
    REQUIRE(r->names[1] == "vj_p.id");
    REQUIRE(r->names[2] == "vj_c.name");
    REQUIRE(r->names[3] == "vj_c.label");
    REQUIRE(r->names[4] == "vj_c.id");
    REQUIRE(r->names[5] == "distance");

    auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(mat.RowCount() == 3 * 2);
    for (duckdb::idx_t i = 0; i < mat.RowCount(); i++) {
      INFO("row " << i);
      auto const pid = mat.GetValue(1, i).GetValue<int32_t>();
      auto const cid = mat.GetValue(4, i).GetValue<int32_t>();
      REQUIRE(mat.GetValue(0, i).GetValue<std::string>() == "p" + std::to_string(pid));
      REQUIRE(mat.GetValue(2, i).GetValue<std::string>() == "c" + std::to_string(cid));
      REQUIRE(mat.GetValue(3, i).GetValue<int32_t>() == cid * 10);
    }
  }

  SECTION("k larger than the corpus clamps to the corpus row count")
  {
    auto r = con->Query(
      "SELECT * FROM sirius_knn_join('vj_p','vec','vj_c','vec', k => 100, "
      "probe_output_columns => ['id'], corpus_output_columns => ['id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    // 3 probe rows * min(k=100, 4 corpus rows) = 12.
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 3 * 4);
  }

  run_ok("SELECT * FROM unpin_table('vj_p');");
  run_ok("SELECT * FROM unpin_table('vj_c');");
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - error handling",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok(
    "CREATE TABLE ep AS SELECT i AS id, [i::float, i::float, i::float] AS vec FROM range(20) "
    "t(i);");
  run_ok(
    "CREATE TABLE ec AS SELECT i AS id, [i::float, i::float, i::float] AS vec FROM range(20) "
    "t(i);");
  // A table whose vector column has a different dimensionality (FLOAT[2]).
  run_ok("CREATE TABLE ec2 AS SELECT i AS id, [i::float, i::float] AS vec FROM range(20) t(i);");
  run_ok("CHECKPOINT;");

  SECTION("bind errors (raised before execution)")
  {
    // Missing / NULL required positional argument.
    expect_error(*con, "SELECT * FROM sirius_knn_join(NULL, 'vec', 'ec', 'vec');", "four non-NULL");
    // Vector dimensionality mismatch between probe and corpus.
    expect_error(*con, "SELECT * FROM sirius_knn_join('ep','vec','ec2','vec');", "FLOAT[2]");
    // Unknown output column.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', probe_output_columns => "
                 "['nope']);",
                 "not found");
    // Vector column is not a FLOAT[N] array.
    expect_error(*con, "SELECT * FROM sirius_knn_join('ep','id','ec','vec');", "FLOAT[N]");
    // k must be >= 1.
    expect_error(
      *con, "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', k => 0);", "k must be");
    // Invalid metric.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', metric => 'bogus');",
                 "metric must be");
    // Invalid mode.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', mode => 'bogus');",
                 "mode must be");
  }

  SECTION("execution errors: tables must be pinned")
  {
    // Neither table pinned.
    expect_error(
      *con, "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', k => 5);", "must be pinned");

    // Corpus pinned but probe not.
    run_ok("SELECT * FROM pin_table(name => 'ec', tier => 'gpu', format => 'duckdb');");
    expect_error(
      *con, "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', k => 5);", "probe table 'ep'");
    run_ok("SELECT * FROM unpin_table('ec');");

    // Probe pinned but corpus not.
    run_ok("SELECT * FROM pin_table(name => 'ep', tier => 'gpu', format => 'duckdb');");
    expect_error(
      *con, "SELECT * FROM sirius_knn_join('ep','vec','ec','vec', k => 5);", "corpus table 'ec'");
    run_ok("SELECT * FROM unpin_table('ep');");
  }
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - null vectors are excluded from both sides",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Corpus id 5 is the probe's near-exact match but is NULL, so it must never be
  // returned; the next-nearest non-null rows are. The probe's .3 offset keeps the
  // ranking tie-free. Probe id 1 is a NULL vector and contributes no pairs.
  run_ok(
    "CREATE TABLE nc AS SELECT i AS id, "
    "CASE WHEN i = 5 THEN NULL ELSE [i::float, i::float, i::float] END::FLOAT[3] AS vec "
    "FROM range(50) t(i);");
  run_ok("CREATE TABLE np (id INTEGER, vec FLOAT[3]);");
  run_ok("INSERT INTO np VALUES (0, [5.3, 5.3, 5.3]), (1, NULL);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'nc', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'np', tier => 'gpu', format => 'duckdb');");

  auto gpu = [&](int k) {
    return ok_rows(*con,
                   "SELECT \"np.id\", \"nc.id\" FROM sirius_knn_join('np','vec','nc','vec', k => " +
                     std::to_string(k) +
                     ", probe_output_columns => ['id'], corpus_output_columns => ['id']);");
  };
  auto oracle = [&](int k) {
    con->Query("SET gpu_execution = false;");
    auto rows = ok_rows(*con,
                        "SELECT p.id, c.id FROM np p, nc c "
                        "WHERE p.vec IS NOT NULL AND c.vec IS NOT NULL "
                        "QUALIFY row_number() OVER (PARTITION BY p.id "
                        "ORDER BY array_distance(p.vec, c.vec)) <= " +
                          std::to_string(k) + ";");
    con->Query("SET gpu_execution = true;");
    return rows;
  };

  for (int k : {1, 3, 5}) {
    INFO("k = " << k);
    auto got = gpu(k);
    REQUIRE(got == oracle(k));
    for (auto const& row : got) {
      REQUIRE(row[0] != "1");  // null-vector probe row never produces pairs
      REQUIRE(row[1] != "5");  // null-vector corpus row is never a neighbor
    }
  }

  run_ok("SELECT * FROM unpin_table('nc');");
  run_ok("SELECT * FROM unpin_table('np');");
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - empty results preserve the output schema",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE ep2 (id INTEGER, vec FLOAT[3]);");
  run_ok("INSERT INTO ep2 VALUES (0, [1000000.0, 1000000.0, 1000000.0]);");
  run_ok(
    "CREATE TABLE ec3 AS SELECT i AS id, [i::float, i::float, i::float]::FLOAT[3] AS vec "
    "FROM range(100) t(i);");
  run_ok("CREATE TABLE en (id INTEGER, vec FLOAT[3]);");
  run_ok("INSERT INTO en SELECT i, NULL FROM range(20) t(i);");  // all-null corpus
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'ep2', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'ec3', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'en', tier => 'gpu', format => 'duckdb');");

  SECTION("threshold excludes every pair -> empty, schema intact")
  {
    // Probe is ~1e6 away from the whole corpus; a threshold of 1.0 keeps nothing.
    auto r = con->Query(
      "SELECT * FROM sirius_knn_join('ep2','vec','ec3','vec', k => 5, threshold => 1.0, "
      "probe_output_columns => ['id'], corpus_output_columns => ['id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 0);
    REQUIRE(r->names.size() == 3);
    REQUIRE(r->names[0] == "ep2.id");
    REQUIRE(r->names[1] == "ec3.id");
    REQUIRE(r->names[2] == "distance");
  }

  SECTION("corpus of only null vectors -> empty, schema intact")
  {
    auto r = con->Query(
      "SELECT * FROM sirius_knn_join('ep2','vec','en','vec', k => 5, "
      "probe_output_columns => ['id'], corpus_output_columns => ['id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 0);
    REQUIRE(r->names.size() == 3);
    REQUIRE(r->names[2] == "distance");
  }

  run_ok("SELECT * FROM unpin_table('ep2');");
  run_ok("SELECT * FROM unpin_table('ec3');");
  run_ok("SELECT * FROM unpin_table('en');");
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - self join (same table on both sides)",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Rows are 3 apart in each coordinate, so every row's unique nearest neighbor is
  // itself at distance 0. Both sides resolve to the same name, so the two 'id'
  // outputs collide by name -> read positionally (probe cols first, then corpus).
  run_ok(
    "CREATE TABLE selfj AS SELECT i AS id, "
    "[(i * 3)::float, (i * 3)::float, (i * 3)::float]::FLOAT[3] AS vec FROM range(500) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'selfj', tier => 'gpu', format => 'duckdb');");

  auto r = con->Query(
    "SELECT * FROM sirius_knn_join('selfj','vec','selfj','vec', k => 1, "
    "probe_output_columns => ['id'], corpus_output_columns => ['id']);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  REQUIRE(mat.RowCount() == 500);
  for (duckdb::idx_t i = 0; i < mat.RowCount(); i++) {
    int const probe_id  = mat.GetValue(0, i).GetValue<int32_t>();
    int const corpus_id = mat.GetValue(1, i).GetValue<int32_t>();
    double const d      = mat.GetValue(2, i).GetValue<double>();
    INFO("row " << i);
    REQUIRE(probe_id == corpus_id);
    REQUIRE(d == Approx(0.0).margin(1e-3));
  }

  run_ok("SELECT * FROM unpin_table('selfj');");
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - multi-chunk tables (corpus concat + probe iteration)",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // The chunk layout is baked into the pinned entry at pin time, so shrink the
  // scan batch to force >1 chunk, pin, then restore the default (512 MiB).
  auto chunk_count = [&](const std::string& name, bool host) {
    return pinned_chunk_count(*con, name, host);
  };
  auto oracle_perrow = [&](const std::string& p, const std::string& c, int k) {
    return oracle_per_row(*con, p, c, k);
  };

  SECTION("corpus spans multiple chunks (GPU concat, HOST assemble+concat)")
  {
    auto corpus_tier = GENERATE(std::string("gpu"), std::string("host"));
    INFO("corpus tier = " << corpus_tier);
    run_ok(
      "CREATE TABLE mcc AS SELECT i AS id, [i::float, (i+1)::float, (i+2)::float]::FLOAT[3] AS vec "
      "FROM range(20000) t(i);");
    run_ok("CREATE TABLE mcp (id INTEGER, vec FLOAT[3]);");
    run_ok(
      "INSERT INTO mcp VALUES (0, [1000.3, 1001.3, 1002.3]), (1, [6000.3, 6001.3, 6002.3]), "
      "(2, [15000.3, 15001.3, 15002.3]);");
    run_ok("CHECKPOINT;");
    run_ok("SET scan_task_batch_size = 4096;");
    run_ok("SELECT * FROM pin_table(name => 'mcc', tier => '" + corpus_tier +
           "', format => 'duckdb');");
    run_ok("SET scan_task_batch_size = 536870912;");
    run_ok("SELECT * FROM pin_table(name => 'mcp', tier => 'gpu', format => 'duckdb');");

    INFO("corpus chunk count = " << chunk_count("mcc", corpus_tier == "host"));
    REQUIRE(chunk_count("mcc", corpus_tier == "host") > 1);

    for (int k : {1, 3, 5}) {
      INFO("per-row k = " << k);
      auto got =
        ok_rows(*con,
                "SELECT \"mcp.id\", \"mcc.id\" FROM sirius_knn_join('mcp','vec','mcc','vec',"
                " k => " +
                  std::to_string(k) +
                  ", probe_output_columns => ['id'], corpus_output_columns => ['id']);");
      REQUIRE(got == oracle_perrow("mcp", "mcc", k));
    }
    for (int k : {3, 9}) {
      INFO("global k = " << k);
      con->Query("SET gpu_execution = false;");
      auto oracle = ok_rows(*con,
                            "SELECT p.id, c.id FROM mcp p, mcc c ORDER BY "
                            "array_distance(p.vec, c.vec) LIMIT " +
                              std::to_string(k) + ";");
      con->Query("SET gpu_execution = true;");
      auto got =
        ok_rows(*con,
                "SELECT \"mcp.id\", \"mcc.id\" FROM sirius_knn_join('mcp','vec','mcc','vec',"
                " k => " +
                  std::to_string(k) +
                  ", mode => 'global', probe_output_columns => ['id'], "
                  "corpus_output_columns => ['id']);");
      REQUIRE(got == oracle);
    }

    run_ok("SELECT * FROM unpin_table('mcc');");
    run_ok("SELECT * FROM unpin_table('mcp');");
  }

  SECTION("probe spans multiple chunks (GPU iteration, HOST streaming)")
  {
    auto probe_tier = GENERATE(std::string("gpu"), std::string("host"));
    INFO("probe tier = " << probe_tier);
    // Small corpus, large probe. The .3 offset keeps every probe row's ranking
    // tie-free against the integer corpus.
    run_ok(
      "CREATE TABLE mcc2 AS SELECT i AS id, [i::float, (i+1)::float, (i+2)::float]::FLOAT[3] AS "
      "vec "
      "FROM range(60) t(i);");
    run_ok(
      "CREATE TABLE mcp2 AS SELECT i AS id, "
      "[(i + 0.3)::float, (i + 1.3)::float, (i + 2.3)::float]::FLOAT[3] AS vec "
      "FROM range(8000) t(i);");
    run_ok("CHECKPOINT;");
    run_ok("SELECT * FROM pin_table(name => 'mcc2', tier => 'gpu', format => 'duckdb');");
    run_ok("SET scan_task_batch_size = 4096;");
    run_ok("SELECT * FROM pin_table(name => 'mcp2', tier => '" + probe_tier +
           "', format => 'duckdb');");
    run_ok("SET scan_task_batch_size = 536870912;");

    INFO("probe chunk count = " << chunk_count("mcp2", probe_tier == "host"));
    REQUIRE(chunk_count("mcp2", probe_tier == "host") > 1);

    for (int k : {1, 3, 5}) {
      INFO("per-row k = " << k);
      auto got =
        ok_rows(*con,
                "SELECT \"mcp2.id\", \"mcc2.id\" FROM sirius_knn_join('mcp2','vec','mcc2','vec', "
                "k => " +
                  std::to_string(k) +
                  ", probe_output_columns => ['id'], corpus_output_columns => ['id']);");
      REQUIRE(got == oracle_perrow("mcp2", "mcc2", k));
    }

    // Global mode folds each probe chunk into a running top-k, so it only means
    // anything with more than one probe chunk -- which is what this section sets up.
    //
    // Probe row i is [i+.3, i+1.3, i+2.3] and corpus row j is [j, j+1, j+2], so the
    // distance is sqrt(3)*|0.3 - (j-i)|: one shell per integer offset j-i, all at
    // distinct distances. Offset 0 holds 60 pairs (j = i, i < 60) and offset +1 holds
    // 59 (j = i+1 <= 59), so k = 60 and k = 119 land on complete shells and the global
    // top-k SET is unambiguous.
    for (int k : {60, 119}) {
      INFO("global k = " << k);
      con->Query("SET gpu_execution = false;");
      auto oracle = ok_rows(*con,
                            "SELECT p.id, c.id FROM mcp2 p, mcc2 c ORDER BY "
                            "array_distance(p.vec, c.vec) LIMIT " +
                              std::to_string(k) + ";");
      con->Query("SET gpu_execution = true;");
      auto got =
        ok_rows(*con,
                "SELECT \"mcp2.id\", \"mcc2.id\" FROM sirius_knn_join('mcp2','vec','mcc2','vec', "
                "k => " +
                  std::to_string(k) +
                  ", mode => 'global', probe_output_columns => ['id'], "
                  "corpus_output_columns => ['id']);");
      REQUIRE(got == oracle);
    }

    // Threshold with a multi-chunk probe: it is applied per probe chunk, before the
    // chunk's pairs are handed to the host. Shells sit at sqrt(3)*0.3 = 0.52,
    // sqrt(3)*0.7 = 1.21 and sqrt(3)*1.3 = 2.25, so a bound of 2.0 keeps the first
    // two and drops the rest. k = 10 is well past that, so the bound is what binds.
    {
      con->Query("SET gpu_execution = false;");
      auto oracle = ok_rows(*con,
                            "SELECT p.id, c.id FROM mcp2 p, mcc2 c "
                            "WHERE array_distance(p.vec, c.vec) <= 2.0;");
      con->Query("SET gpu_execution = true;");
      auto got =
        ok_rows(*con,
                "SELECT \"mcp2.id\", \"mcc2.id\" FROM sirius_knn_join('mcp2','vec','mcc2','vec', "
                "k => 10, threshold => 2.0, probe_output_columns => ['id'], "
                "corpus_output_columns => ['id']);");
      REQUIRE(got == oracle);
    }

    run_ok("SELECT * FROM unpin_table('mcc2');");
    run_ok("SELECT * FROM unpin_table('mcp2');");
  }
}

TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - tiled corpus (forced via max_stage_bytes)",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // max_stage_bytes caps the corpus staging budget, so a corpus that would otherwise
  // be staged contiguously is searched one tile per pinned chunk and each probe row's
  // top-k is merged across tiles. A small scan batch at pin time is what produces the
  // >1 chunk to tile over; the budget is what makes the join actually tile.
  auto corpus_tier = GENERATE(std::string("gpu"), std::string("host"));
  INFO("corpus tier = " << corpus_tier);

  // Probe row at base i is [i+.3, i+1.3, i+2.3] and corpus row j is [j, j+1, j+2], so
  // the distance is sqrt(3)*|0.3 - (j-i)|: one shell per integer offset j-i, all at
  // distinct distances. Per probe row the ranking is therefore tie-free at every k.
  run_ok(
    "CREATE TABLE tj_c AS SELECT i AS id, [i::float, (i+1)::float, (i+2)::float]::FLOAT[3] AS vec "
    "FROM range(2000) t(i);");
  run_ok("CREATE TABLE tj_p (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO tj_p VALUES (0, [100.3, 101.3, 102.3]), (1, [500.3, 501.3, 502.3]), "
    "(2, [900.3, 901.3, 902.3]), (3, [1500.3, 1501.3, 1502.3]);");
  run_ok("CHECKPOINT;");
  run_ok("SET scan_task_batch_size = 4096;");
  run_ok("SELECT * FROM pin_table(name => 'tj_c', tier => '" + corpus_tier +
         "', format => 'duckdb');");
  run_ok("SET scan_task_batch_size = 536870912;");
  run_ok("SELECT * FROM pin_table(name => 'tj_p', tier => 'gpu', format => 'duckdb');");

  INFO("corpus chunk count = " << pinned_chunk_count(*con, "tj_c", corpus_tier == "host"));
  REQUIRE(pinned_chunk_count(*con, "tj_c", corpus_tier == "host") > 1);

  // tail is appended inside the argument list; budget => "" runs the same query untiled.
  auto run_join = [&](const std::string& tail, const std::string& budget) {
    return ok_rows(*con,
                   "SELECT \"tj_p.id\", \"tj_c.id\" FROM sirius_knn_join('tj_p','vec','tj_c','vec',"
                   " probe_output_columns => ['id'], corpus_output_columns => ['id']" +
                     tail + budget + ");");
  };
  const std::string kTiled = ", max_stage_bytes => 1";

  // Per-row top-k across many tiles. k < kept + k_t at every merge, so this is the
  // modulo-rank survivor path in merge_corpus_tile_candidates.
  for (int k : {1, 3, 7}) {
    INFO("per-row k = " << k);
    auto const tail = ", k => " + std::to_string(k);
    auto const got  = run_join(tail, kTiled);
    REQUIRE(got == oracle_per_row(*con, "tj_p", "tj_c", k));
    // Tiling must not change the answer.
    REQUIRE(got == run_join(tail, ""));
  }

  // k >= the whole corpus: every merge takes the `k >= m_per_row` early return, and
  // every probe row keeps every corpus row. Also spans several output DataChunks
  // (4 * 2000 rows), so it exercises the multi-chunk reader drain.
  {
    INFO("k >= corpus rows");
    con->Query("SET gpu_execution = false;");
    auto const oracle = ok_rows(*con, "SELECT p.id, c.id FROM tj_p p, tj_c c;");
    con->Query("SET gpu_execution = true;");
    REQUIRE(oracle.size() == static_cast<std::size_t>(4 * 2000));
    REQUIRE(run_join(", k => 2000", kTiled) == oracle);
  }

  // Global top-k. Each offset shell holds exactly one pair per probe row, so
  // multiples of 4 are shell-complete and the global top-k SET is unambiguous.
  for (int k : {4, 8, 12}) {
    INFO("global k = " << k);
    con->Query("SET gpu_execution = false;");
    auto const oracle = ok_rows(*con,
                                "SELECT p.id, c.id FROM tj_p p, tj_c c ORDER BY "
                                "array_distance(p.vec, c.vec) LIMIT " +
                                  std::to_string(k) + ";");
    con->Query("SET gpu_execution = true;");
    auto const tail = ", k => " + std::to_string(k) + ", mode => 'global'";
    auto const got  = run_join(tail, kTiled);
    REQUIRE(got == oracle);
    REQUIRE(got == run_join(tail, ""));
  }

  // Threshold under tiling: shells sit at sqrt(3)*0.3 = 0.52, sqrt(3)*0.7 = 1.21 and
  // sqrt(3)*1.3 = 2.25, so a bound of 2.0 keeps the first two. k = 10 is well past
  // that, so the bound is what binds.
  {
    con->Query("SET gpu_execution = false;");
    auto const oracle = ok_rows(
      *con, "SELECT p.id, c.id FROM tj_p p, tj_c c WHERE array_distance(p.vec, c.vec) <= 2.0;");
    con->Query("SET gpu_execution = true;");
    REQUIRE(run_join(", k => 10, threshold => 2.0", kTiled) == oracle);
  }

  run_ok("SELECT * FROM unpin_table('tj_c');");
  run_ok("SELECT * FROM unpin_table('tj_p');");
}
