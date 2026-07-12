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
 * @file test_gpu_execution_vector_search.cpp
 * @brief End-to-end tests for the sirius_vector_search() table function.
 *
 * Unlike the ORDER BY array_distance ... LIMIT auto-routing, sirius_vector_search
 * is an explicit Sirius-owned surface that runs the GPU k-NN search directly in
 * its function body and exposes ANN knobs (k, n_probes, use_index, metric,
 * output_columns). These tests drive the whole stack: bind (arg parse + catalog
 * resolution) -> init (search + gather + host materialize) -> scan (stream chunks).
 *
 * Correctness oracle: DuckDB's own exact `ORDER BY array_distance(...) LIMIT k`,
 * computed with gpu_execution OFF. The ANN path (use_index=true) is made exact by
 * building the index with n_lists <= 32, so it probes every list and considers all
 * vectors; the ENN path (use_index=false) is exact by construction. The linear
 * vec=[i,i,i] data is tie-free, so the top-k id SET is unambiguous between the two.
 *
 * Data is checkpointed before pinning: pin_table(format='duckdb') reads on-disk
 * blocks through the native scan path, so WAL-resident rows would be invisible.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

using VectorSearchFixture = sirius::test::GpuExecutionFixture;

namespace {

// Sorted rows (each a single-column vector) from a query that must succeed. Both
// sides of a comparison sort identically, so the set equality holds regardless of
// the lexical string ordering of numeric ids.
std::vector<std::vector<std::string>> ok_col(duckdb::Connection& con, const std::string& sql)
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

}  // namespace

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - ANN (IVF-Flat) l2sq matches exact top-k",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok("CREATE TABLE vs_l2 AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_l2', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM sirius_create_ann_index('vs_l2', 'vec', metric => 'l2sq', n_lists => 16);");

  // Exact reference (gpu off) vs. sirius_vector_search (gpu on), id set, several k.
  auto exact_ids = [&](const std::string& q, int k) {
    con->Query("SET gpu_execution = false;");
    auto ids = ok_col(*con,
                      "SELECT id FROM vs_l2 ORDER BY array_distance(vec, " + q + ") LIMIT " +
                        std::to_string(k) + ";");
    con->Query("SET gpu_execution = true;");
    return ids;
  };
  auto search_ids = [&](const std::string& q, int k) {
    return ok_col(*con,
                  "SELECT id FROM sirius_vector_search('vs_l2', 'vec', " + q + ", k => " +
                    std::to_string(k) + ", output_columns => ['id']);");
  };

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";
  for (int k : {1, 5, 20, 100}) {
    INFO("k = " << k);
    REQUIRE(search_ids(origin, k) == exact_ids(origin, k));
  }

  // Query vector INSIDE the dataset: distance is symmetric around row 1000, so
  // an ODD k lands on complete tie-shells -> the top-k SET is unambiguous.
  const std::string interior = "[1000.0, 1000.0, 1000.0]::FLOAT[3]";
  for (int k : {1, 7, 21}) {
    INFO("interior k = " << k);
    REQUIRE(search_ids(interior, k) == exact_ids(interior, k));
  }

  run_ok("SELECT * FROM unpin_table('vs_l2');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - explicit n_probes and distance column",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_probe AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(2000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_probe', tier => 'gpu', format => 'duckdb');");
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_probe', 'vec', metric => 'l2sq', n_lists => 16);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  // n_probes == n_lists probes every list -> exact, matches the reference ids.
  con->Query("SET gpu_execution = false;");
  auto exact =
    ok_col(*con, "SELECT id FROM vs_probe ORDER BY array_distance(vec, " + origin + ") LIMIT 10;");
  con->Query("SET gpu_execution = true;");
  auto probed = ok_col(*con,
                       "SELECT id FROM sirius_vector_search('vs_probe', 'vec', " + origin +
                         ", k => 10, output_columns => ['id'], n_probes => 16);");
  REQUIRE(probed == exact);

  // The trailing distance column equals array_distance (Euclidean), within fp tol.
  // Row i=0..9 has vec=[i,i,i]; distance to origin is sqrt(3)*i.
  auto r = con->Query("SELECT distance FROM sirius_vector_search('vs_probe', 'vec', " + origin +
                      ", k => 10, output_columns => ['id']) ORDER BY distance;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  REQUIRE(mat.RowCount() == 10);
  for (duckdb::idx_t i = 0; i < mat.RowCount(); i++) {
    double const d        = mat.GetValue(0, i).GetValue<double>();
    double const expected = std::numbers::sqrt3 * static_cast<double>(i);
    REQUIRE(d == Approx(expected).epsilon(1e-4).margin(1e-4));
  }

  run_ok("SELECT * FROM unpin_table('vs_probe');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - ENN (use_index=false) brute force over pinned table",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_enn AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(3000) t(i);");
  run_ok("CHECKPOINT;");
  // Pinned but NO index built: use_index=false brute-forces the pinned column.
  run_ok("SELECT * FROM pin_table(name => 'vs_enn', tier => 'gpu', format => 'duckdb');");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";
  con->Query("SET gpu_execution = false;");
  auto exact =
    ok_col(*con, "SELECT id FROM vs_enn ORDER BY array_distance(vec, " + origin + ") LIMIT 25;");
  con->Query("SET gpu_execution = true;");
  auto enn = ok_col(*con,
                    "SELECT id FROM sirius_vector_search('vs_enn', 'vec', " + origin +
                      ", k => 25, output_columns => ['id'], use_index => false);");
  REQUIRE(enn == exact);

  // Regression for float32 catastrophic cancellation in the L2 distance
  const std::string big_q = "[2990.0, 2990.0, 2990.0]::FLOAT[3]";
  con->Query("SET gpu_execution = false;");
  auto exact_d =
    con->Query("SELECT array_distance(vec, " + big_q + ") AS d FROM vs_enn ORDER BY d LIMIT 15;");
  REQUIRE(exact_d);
  REQUIRE_FALSE(exact_d->HasError());
  con->Query("SET gpu_execution = true;");
  auto enn_d = con->Query("SELECT distance FROM sirius_vector_search('vs_enn', 'vec', " + big_q +
                          ", k => 15, output_columns => ['id'], use_index => false) "
                          "ORDER BY distance;");
  REQUIRE(enn_d);
  REQUIRE_FALSE(enn_d->HasError());
  auto& enn_mat   = enn_d->Cast<duckdb::MaterializedQueryResult>();
  auto& exact_mat = exact_d->Cast<duckdb::MaterializedQueryResult>();
  REQUIRE(enn_mat.RowCount() == exact_mat.RowCount());
  for (duckdb::idx_t i = 0; i < enn_mat.RowCount(); i++) {
    double const got      = enn_mat.GetValue(0, i).GetValue<double>();
    double const expected = exact_mat.GetValue(0, i).GetValue<double>();
    INFO("rank " << i << " got=" << got << " expected=" << expected);
    REQUIRE(got == Approx(expected).epsilon(1e-4).margin(1e-3));
  }

  run_ok("SELECT * FROM unpin_table('vs_enn');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - cosine metric matches exact top-k",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  // vec=[1,i,0]: cosine distance to [1,0,0] is strictly increasing in i and
  // well-separated for small i -> tie-free top-k (the [i,i,i] trick can't be
  // reused for cosine since those rows all share one direction).
  run_ok(
    "CREATE TABLE vs_cos AS SELECT i AS id, [1.0, i, 0.0]::FLOAT[3] AS vec FROM range(2000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_cos', tier => 'gpu', format => 'duckdb');");
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_cos', 'vec', metric => 'cosine', n_lists => 16);");

  const std::string q = "[1.0, 0.0, 0.0]::FLOAT[3]";
  con->Query("SET gpu_execution = false;");
  auto exact =
    ok_col(*con, "SELECT id FROM vs_cos ORDER BY array_cosine_distance(vec, " + q + ") LIMIT 5;");
  con->Query("SET gpu_execution = true;");
  auto ann = ok_col(*con,
                    "SELECT id FROM sirius_vector_search('vs_cos', 'vec', " + q +
                      ", k => 5, output_columns => ['id'], metric => 'cosine');");
  REQUIRE(ann == exact);

  run_ok("SELECT * FROM unpin_table('vs_cos');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - ENN (use_index=false) cosine matches exact top-k",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_enn_cos AS SELECT i AS id, [1.0, i, 0.0]::FLOAT[3] AS vec FROM range(2000) "
    "t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_enn_cos', tier => 'gpu', format => 'duckdb');");

  const std::string q = "[1.0, 0.0, 0.0]::FLOAT[3]";
  con->Query("SET gpu_execution = false;");
  auto exact = ok_col(
    *con, "SELECT id FROM vs_enn_cos ORDER BY array_cosine_distance(vec, " + q + ") LIMIT 5;");
  con->Query("SET gpu_execution = true;");
  auto enn =
    ok_col(*con,
           "SELECT id FROM sirius_vector_search('vs_enn_cos', 'vec', " + q +
             ", k => 5, output_columns => ['id'], metric => 'cosine', use_index => false);");
  REQUIRE(enn == exact);

  run_ok("SELECT * FROM unpin_table('vs_enn_cos');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - output schema (default all, subset, order, k>rows)",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_schema AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_schema', tier => 'gpu', format => 'duckdb');");
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_schema', 'vec', metric => 'l2sq', n_lists => 4);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  SECTION("omitted output_columns => all base columns + trailing distance")
  {
    auto r =
      con->Query("SELECT * FROM sirius_vector_search('vs_schema', 'vec', " + origin + ", k => 3);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->names.size() == 3);
    REQUIRE(r->names[0] == "id");
    REQUIRE(r->names[1] == "vec");
    REQUIRE(r->names[2] == "distance");
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 3);
  }

  SECTION("subset + explicit order is honored, distance appended last")
  {
    auto r = con->Query("SELECT * FROM sirius_vector_search('vs_schema', 'vec', " + origin +
                        ", k => 3, output_columns => ['vec', 'id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->names.size() == 3);
    REQUIRE(r->names[0] == "vec");
    REQUIRE(r->names[1] == "id");
    REQUIRE(r->names[2] == "distance");
  }

  SECTION("k larger than the table clamps to the row count")
  {
    auto r = con->Query("SELECT id FROM sirius_vector_search('vs_schema', 'vec', " + origin +
                        ", k => 100, output_columns => ['id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 5);
  }

  run_ok("SELECT * FROM unpin_table('vs_schema');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_vector_search - error handling",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok("CREATE TABLE vs_err AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(100) t(i);");
  run_ok("CHECKPOINT;");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  SECTION("bind errors (raised before execution)")
  {
    // Dimensionality mismatch: query is FLOAT[2] but the column is FLOAT[3].
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'vec', [0.0, 0.0]::FLOAT[2], "
                 "output_columns => ['id']);",
                 "FLOAT[3]");
    // Unknown output column.
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['nope']);",
                 "not found");
    // Vector column is not a FLOAT[N] array.
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'id', " + origin +
                   ", output_columns => ['id']);",
                 "FLOAT[N]");
    // k must be >= 1.
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'vec', " + origin +
                   ", k => 0, output_columns => ['id']);",
                 "k must be");
    // Invalid metric.
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['id'], metric => 'bogus');",
                 "metric must be");
  }

  SECTION("execution errors")
  {
    // Table not pinned -> both ANN and ENN require a GPU-pinned table today.
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['id'], use_index => false);",
                 "must be pinned");

    // Pinned but no ANN index built, use_index defaults true -> clear error.
    run_ok("SELECT * FROM pin_table(name => 'vs_err', tier => 'gpu', format => 'duckdb');");
    expect_error(*con,
                 "SELECT * FROM sirius_vector_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['id']);",
                 "no ANN index");
    run_ok("SELECT * FROM unpin_table('vs_err');");
  }
}
