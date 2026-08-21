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
 * @brief End-to-end tests for the sirius_knn_search() table function.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius_context.hpp>
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

}  // namespace

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - ENN brute force over pinned table",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_enn AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(3000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_enn', tier => 'gpu', format => 'duckdb');");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";
  con->Query("SET gpu_execution = false;");
  auto exact =
    ok_col(*con, "SELECT id FROM vs_enn ORDER BY array_distance(vec, " + origin + ") LIMIT 25;");
  con->Query("SET gpu_execution = true;");
  auto enn = ok_col(*con,
                    "SELECT id FROM sirius_knn_search('vs_enn', 'vec', " + origin +
                      ", k => 25, output_columns => ['id']);");
  REQUIRE(enn == exact);

  // Regression for float32 catastrophic cancellation in the L2 distance
  const std::string big_q = "[2990.0, 2990.0, 2990.0]::FLOAT[3]";
  con->Query("SET gpu_execution = false;");
  auto exact_d =
    con->Query("SELECT array_distance(vec, " + big_q + ") AS d FROM vs_enn ORDER BY d LIMIT 15;");
  REQUIRE(exact_d);
  REQUIRE_FALSE(exact_d->HasError());
  con->Query("SET gpu_execution = true;");
  auto enn_d = con->Query("SELECT distance FROM sirius_knn_search('vs_enn', 'vec', " + big_q +
                          ", k => 15, output_columns => ['id']) ORDER BY distance;");
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

// sirius_knn_search defaults output_columns to the pinned columns, not every
// catalog column. The search gathers straight from GPU-resident chunks, so a
// catalog-wide default is unsatisfiable on any subset pin.
TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - default output_columns on a subset-pinned table",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_subset AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec, "
    "'row' || i AS payload FROM range(100) t(i);");
  run_ok("CHECKPOINT;");

  // Pin only [id, vec]: `payload` exists in the catalog but never reaches the GPU.
  run_ok(
    "SELECT * FROM pin_table(name => 'vs_subset', tier => 'gpu', format => 'duckdb', "
    "cols => ['id', 'vec']);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  // Baseline: naming only pinned columns works, so the pin itself is searchable.
  {
    auto r = con->Query("SELECT id FROM sirius_knn_search('vs_subset', 'vec', " + origin +
                        ", k => 5, output_columns => ['id']);");
    REQUIRE(r);
    if (r->HasError()) { UNSCOPED_INFO("explicit output_columns error: " << r->GetError()); }
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 5);
  }

  // Same query with output_columns omitted: the default expands to the pinned
  // columns [id, vec] (not the catalog-wide [id, vec, payload]), so it succeeds.
  {
    auto r =
      con->Query("SELECT * FROM sirius_knn_search('vs_subset', 'vec', " + origin + ", k => 5);");
    REQUIRE(r);
    if (r->HasError()) { UNSCOPED_INFO("default output_columns error: " << r->GetError()); }
    REQUIRE_FALSE(r->HasError());
  }

  run_ok("SELECT * FROM unpin_table('vs_subset');");
}

// output_columns => [] is stored as an empty vector; it must be rejected as a user
// error, NOT treated like omitting the parameter (which defaults to the pinned
// columns). Probed against a fully-pinned table, where "rejected" and "expanded to
// all" actually differ.
TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - explicit empty output_columns is rejected",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_empty_cols AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec, "
    "'row' || i AS payload FROM range(20) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_empty_cols', tier => 'gpu', format => 'duckdb');");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  auto empty_list = con->Query("SELECT * FROM sirius_knn_search('vs_empty_cols', 'vec', " + origin +
                               ", k => 2, output_columns => []);");
  REQUIRE(empty_list);
  UNSCOPED_INFO("empty output_columns error: " << (empty_list->HasError() ? empty_list->GetError()
                                                                          : std::string("<none>")));
  // An explicitly empty list is a user error, not a request for everything: rejected
  // at bind with a typed BinderException rather than silently expanding to all columns.
  REQUIRE(empty_list->HasError());

  run_ok("SELECT * FROM unpin_table('vs_empty_cols');");
}

// An explicitly-requested column that exists in the catalog but was not pinned must
// fail at bind (typed BinderException), not deep in execution with an untyped
// internal_exception("VSS: pinned table missing column ...") after the pin is set up.
TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - explicit unpinned output column is rejected at bind",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_unpinned_col AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec, "
    "'row' || i AS payload FROM range(50) t(i);");
  run_ok("CHECKPOINT;");
  // Pin only [id, vec]; 'payload' is catalog-visible but never pinned.
  run_ok(
    "SELECT * FROM pin_table(name => 'vs_unpinned_col', tier => 'gpu', format => 'duckdb', "
    "cols => ['id', 'vec']);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";
  auto r = con->Query("SELECT * FROM sirius_knn_search('vs_unpinned_col', 'vec', " + origin +
                      ", k => 5, output_columns => ['id', 'payload']);");
  REQUIRE(r);
  UNSCOPED_INFO(
    "unpinned output column error: " << (r->HasError() ? r->GetError() : std::string("<none>")));
  REQUIRE(r->HasError());

  run_ok("SELECT * FROM unpin_table('vs_unpinned_col');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - ENN cosine matches exact top-k",
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
  auto enn = ok_col(*con,
                    "SELECT id FROM sirius_knn_search('vs_enn_cos', 'vec', " + q +
                      ", k => 5, output_columns => ['id'], metric => 'cosine');");
  REQUIRE(enn == exact);

  run_ok("SELECT * FROM unpin_table('vs_enn_cos');");
}
