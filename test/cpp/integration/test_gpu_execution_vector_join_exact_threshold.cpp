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
 * @file test_gpu_execution_vector_join_exact_threshold.cpp
 * @brief End-to-end tests for sirius_knn_join() in threshold (radius) mode: every
 *        pair within eps is returned (no k, no truncation). Both sides must be pinned.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>
#include <vector>

using VectorJoinFixture = sirius::test::GpuExecutionFixture;

namespace {

// Sorted rows from a query that must succeed, so result sets compare independent
// of the join's output arrival order.
std::vector<std::vector<std::string>> ok_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  return sirius::test::GpuExecutionFixture::collect_rows(mat, /*sort=*/true);
}

int64_t single_int(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return r->GetValue(0, 0).GetValue<int64_t>();
}

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
// L2 threshold, single batch: every pair within the radius comes back, and the
// per-probe neighbor counts vary (ragged output), unlike a fixed-k join. The id
// lives in dim 0, so the L2 distance between rows i and j is |i - j|; integer
// spacing keeps every distance off the eps boundary (no float ties at 2.5).
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - L2 threshold returns every pair within eps",
                 "[integration][gpu_execution][array][vss][vector_join][threshold]")
{
  run_ok("CREATE TABLE thr_corpus (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok("INSERT INTO thr_corpus SELECT i, [i::float, 0.0::float, 0.0::float] FROM range(20) t(i);");
  run_ok("CREATE TABLE thr_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  // Probe 0 sits near the low edge (4 neighbors), 1 is interior (5), 2 near the
  // high edge (4) -> deliberately different counts per probe.
  run_ok(
    "INSERT INTO thr_probe VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [10.0, 0.0, 0.0]), (2, [18.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'thr_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'thr_probe', tier => 'gpu', format => 'duckdb');");

  // CPU reference: the exact radius join is a cross product filtered by distance.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM thr_probe p, thr_corpus c "
                           "WHERE array_distance(p.vec, c.vec) <= 2.5;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'thr_probe','vec','thr_corpus','vec', "
                        "metric => 'l2', join_mode => 'threshold', eps => 2.5);");

  REQUIRE(joined == reference);
  REQUIRE(joined.size() == 13);  // 4 + 5 + 4, a ragged (not fixed-k) result

  run_ok("SELECT * FROM unpin_table('thr_probe');");
  run_ok("SELECT * FROM unpin_table('thr_corpus');");
}

// -----------------------------------------------------------------------------
// Cosine threshold with the default similarity output: eps is a similarity floor,
// which the join thresholds as cosine distance <= 1 - eps. Small hand-placed
// directions keep every similarity clear of the 0.85 cutoff (no boundary ties).
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - cosine threshold keeps pairs above a similarity floor",
                 "[integration][gpu_execution][array][vss][vector_join][threshold]")
{
  run_ok("CREATE TABLE thc_corpus (id INTEGER PRIMARY KEY, vec FLOAT[2]);");
  run_ok(
    "INSERT INTO thc_corpus VALUES "
    "(0, [1.0, 0.0]), (1, [0.99, 0.14]), (2, [0.87, 0.5]), "
    "(3, [0.5, 0.87]), (4, [0.0, 1.0]), (5, [-1.0, 0.0]);");
  run_ok("CREATE TABLE thc_probe (id INTEGER PRIMARY KEY, vec FLOAT[2]);");
  // Probe 0 points along +x -> near ids {0,1,2}; probe 1 along +y -> near {3,4}.
  run_ok("INSERT INTO thc_probe VALUES (0, [1.0, 0.0]), (1, [0.0, 1.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'thc_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'thc_probe', tier => 'gpu', format => 'duckdb');");

  // CPU reference: cosine similarity >= eps is cosine distance <= 1 - eps, matching
  // how the join thresholds internally.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM thc_probe p, thc_corpus c "
                           "WHERE array_cosine_distance(p.vec, c.vec) <= 1.0 - 0.85;");
  con->Query("SET gpu_execution = true;");

  // Default output_type for cosine is similarity, so eps is the similarity floor.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'thc_probe','vec','thc_corpus','vec', "
                        "metric => 'cosine', join_mode => 'threshold', eps => 0.85);");

  REQUIRE(joined == reference);
  REQUIRE(joined.size() == 5);  // probe 0 -> {0,1,2}, probe 1 -> {3,4}

  run_ok("SELECT * FROM unpin_table('thc_probe');");
  run_ok("SELECT * FROM unpin_table('thc_corpus');");
}

// -----------------------------------------------------------------------------
// Multi-batch corpus: the threshold reduce is a UNION across right batches. The
// corpus splits at DuckDB's 122880-row group boundary (one FLOAT[768] row group is
// ~377 MB, over the batch byte cap), so 250000 rows span three batches. A probe on
// the first boundary has neighbors on both sides of it, so the union must stitch
// edges from two different right batches. id lives in dim 0, so L2 distance is |i-j|.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold unions neighbors across right batches",
                 "[integration][gpu_execution][array][vss][vector_join][threshold]")
{
  run_ok("CREATE TABLE thmb_corpus (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  run_ok(
    "INSERT INTO thmb_corpus SELECT i, list_resize([i::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM range(250000) t(i);");
  run_ok("CREATE TABLE thmb_probe (id INTEGER PRIMARY KEY, vec FLOAT[768]);");
  // Probe 0 sits exactly on the first batch boundary (122880): its within-eps
  // neighbors span batch 0 (122878, 122879) and batch 1 (122880..122882). Probe 1
  // sits deep inside batch 0, where all its neighbors come from one batch.
  run_ok(
    "INSERT INTO thmb_probe SELECT id, list_resize([v::FLOAT], 768, 0.0::FLOAT)::FLOAT[768] "
    "FROM (VALUES (0, 122880.0), (1, 60000.0)) t(id, v);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'thmb_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'thmb_probe', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id FROM thmb_probe p, thmb_corpus c "
                           "WHERE array_distance(p.vec, c.vec) <= 2.5;");
  con->Query("SET gpu_execution = true;");

  // search_mode 'exact' (unexpanded L2): magnitudes reach ~122880, where expanded
  // GEMM L2's cancellation error would dwarf the eps=2.5 radius. This is the same
  // reason the per-row multi-batch test uses exact.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'thmb_probe','vec','thmb_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', join_mode => 'threshold', "
                        "eps => 2.5);");

  REQUIRE(joined == reference);
  REQUIRE(joined.size() == 10);  // 5 neighbors each, probe 0's split across two batches

  run_ok("SELECT * FROM unpin_table('thmb_probe');");
  run_ok("SELECT * FROM unpin_table('thmb_corpus');");
}

// -----------------------------------------------------------------------------
// Payload path: a multi-column right output forces the gather path (not the id-only
// fast path). The corpus id starts at 10000 so id != row position; a route-once bug
// that used the position as the id would return wrong ids/vals.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold payload path gathers right columns",
                 "[integration][gpu_execution][array][vss][vector_join][threshold]")
{
  run_ok("CREATE TABLE thpay_corpus (id INTEGER PRIMARY KEY, val INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO thpay_corpus SELECT 10000 + i, i * 2, [i::float, 0.0::float, 0.0::float] "
    "FROM range(40) t(i);");
  run_ok("CREATE TABLE thpay_probe (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok("INSERT INTO thpay_probe VALUES (0, [5.0, 0.0, 0.0]), (1, [30.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'thpay_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'thpay_probe', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, c.id, c.val FROM thpay_probe p, thpay_corpus c "
                           "WHERE array_distance(p.vec, c.vec) <= 2.5;");
  con->Query("SET gpu_execution = true;");

  // Two right output columns -> payload (gather) path rather than the id-only fast path.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id, right_val FROM sirius_knn_join("
                        "'thpay_probe','vec','thpay_corpus','vec', "
                        "metric => 'l2', join_mode => 'threshold', eps => 2.5, "
                        "right_output_columns => ['id', 'val']);");

  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('thpay_probe');");
  run_ok("SELECT * FROM unpin_table('thpay_corpus');");
}

// -----------------------------------------------------------------------------
// Threshold mode selects by radius, so eps must be set (> 0); rejected at bind.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold without eps is rejected",
                 "[integration][gpu_execution][array][vss][vector_join][threshold]")
{
  run_ok("CREATE TABLE threj (id INTEGER PRIMARY KEY, vec FLOAT[3]);");
  run_ok("INSERT INTO threj VALUES (0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'threj', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('threj','vec','threj','vec', "
               "metric => 'l2', join_mode => 'threshold');",
               "requires eps > 0");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('threj','vec','threj','vec', "
               "metric => 'l2', join_mode => 'threshold', eps => 0);",
               "requires eps > 0");

  run_ok("SELECT * FROM unpin_table('threj');");
}
