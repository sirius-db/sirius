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
 *
 * sirius_knn_search is an explicit Sirius-owned surface that runs the GPU k-NN
 * search directly in its function body and exposes ANN knobs.
 *
 * Data is checkpointed before pinning: pin_table(format='duckdb') reads on-disk
 * blocks through the native scan path, so WAL-resident rows would be invisible.
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
                 "sirius_knn_search - ANN (IVF-Flat) l2 matches exact top-k",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok("CREATE TABLE vs_l2 AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_l2', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM sirius_create_ann_index('vs_l2', 'vec', metric => 'l2', n_lists => 16);");

  // Exact reference (gpu off) vs. sirius_knn_search (gpu on), id set, several k.
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
                  "SELECT id FROM sirius_knn_search('vs_l2', 'vec', " + q + ", k => " +
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
                 "sirius_knn_search - explicit n_probes and distance column",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_probe AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(2000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_probe', tier => 'gpu', format => 'duckdb');");
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_probe', 'vec', metric => 'l2', n_lists => 16);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  // n_probes == n_lists probes every list -> exact, matches the reference ids.
  con->Query("SET gpu_execution = false;");
  auto exact =
    ok_col(*con, "SELECT id FROM vs_probe ORDER BY array_distance(vec, " + origin + ") LIMIT 10;");
  con->Query("SET gpu_execution = true;");
  auto probed = ok_col(*con,
                       "SELECT id FROM sirius_knn_search('vs_probe', 'vec', " + origin +
                         ", k => 10, output_columns => ['id'], n_probes => 16);");
  REQUIRE(probed == exact);

  // The trailing distance column equals array_distance (Euclidean), within fp tol.
  // Row i=0..9 has vec=[i,i,i]; distance to origin is sqrt(3)*i.
  auto r = con->Query("SELECT distance FROM sirius_knn_search('vs_probe', 'vec', " + origin +
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
  auto enn_d = con->Query("SELECT distance FROM sirius_knn_search('vs_enn', 'vec', " + big_q +
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
                        ", k => 5, output_columns => ['id'], use_index => false);");
    REQUIRE(r);
    if (r->HasError()) { UNSCOPED_INFO("explicit output_columns error: " << r->GetError()); }
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 5);
  }

  // Same query with output_columns omitted: the default expands to the pinned
  // columns [id, vec] (not the catalog-wide [id, vec, payload]), so it succeeds.
  {
    auto r = con->Query("SELECT * FROM sirius_knn_search('vs_subset', 'vec', " + origin +
                        ", k => 5, use_index => false);");
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
                 "sirius_knn_search - cosine metric matches exact top-k",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  // theta in [0.3, ~2.9] rad (strictly < pi, so cos is 1:1 and distances are
  // distinct); phi advances by the golden angle (~137.5 deg) to spread directions.
  run_ok(
    "CREATE TABLE vs_cos AS SELECT i AS id, "
    "[sin(0.3 + i * 0.0013) * cos(i * 2.39996323), "
    " sin(0.3 + i * 0.0013) * sin(i * 2.39996323), "
    " cos(0.3 + i * 0.0013)]::FLOAT[3] AS vec "
    "FROM range(2000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_cos', tier => 'gpu', format => 'duckdb');");
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_cos', 'vec', metric => 'cosine', n_lists => 16);");

  // Query is the +z axis: nearest by cosine is the smallest theta, i.e. ids 0,1,2,...
  // in order, tie-free -> the top-k SET is unambiguous at every k.
  const std::string q = "[0.0, 0.0, 1.0]::FLOAT[3]";
  auto exact_ids      = [&](int k) {
    con->Query("SET gpu_execution = false;");
    auto ids = ok_col(*con,
                      "SELECT id FROM vs_cos ORDER BY array_cosine_distance(vec, " + q +
                        ") LIMIT " + std::to_string(k) + ";");
    con->Query("SET gpu_execution = true;");
    return ids;
  };

  for (int k : {1, 5, 20, 100}) {
    INFO("cosine k = " << k);
    auto ann = ok_col(*con,
                      "SELECT id FROM sirius_knn_search('vs_cos', 'vec', " + q + ", k => " +
                        std::to_string(k) + ", output_columns => ['id'], metric => 'cosine');");
    REQUIRE(ann == exact_ids(k));
  }

  run_ok("SELECT * FROM unpin_table('vs_cos');");
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
  auto enn =
    ok_col(*con,
           "SELECT id FROM sirius_knn_search('vs_enn_cos', 'vec', " + q +
             ", k => 5, output_columns => ['id'], metric => 'cosine', use_index => false);");
  REQUIRE(enn == exact);

  run_ok("SELECT * FROM unpin_table('vs_enn_cos');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - output schema (default all, subset, order, k>rows)",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_schema AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_schema', tier => 'gpu', format => 'duckdb');");
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_schema', 'vec', metric => 'l2', n_lists => 4);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  SECTION("omitted output_columns => all base columns + trailing distance")
  {
    auto r =
      con->Query("SELECT * FROM sirius_knn_search('vs_schema', 'vec', " + origin + ", k => 3);");
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
    auto r = con->Query("SELECT * FROM sirius_knn_search('vs_schema', 'vec', " + origin +
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
    auto r = con->Query("SELECT id FROM sirius_knn_search('vs_schema', 'vec', " + origin +
                        ", k => 100, output_columns => ['id']);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->Cast<duckdb::MaterializedQueryResult>().RowCount() == 5);
  }

  run_ok("SELECT * FROM unpin_table('vs_schema');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - error handling",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok("CREATE TABLE vs_err AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(100) t(i);");
  run_ok("CHECKPOINT;");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";

  SECTION("bind errors (raised before execution)")
  {
    run_ok("SELECT * FROM pin_table(name => 'vs_err', tier => 'gpu', format => 'duckdb');");

    // Dimensionality mismatch: query is FLOAT[2] but the column is FLOAT[3].
    expect_error(*con,
                 "SELECT * FROM sirius_knn_search('vs_err', 'vec', [0.0, 0.0]::FLOAT[2], "
                 "output_columns => ['id']);",
                 "FLOAT[3]");
    // Unknown output column.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['nope']);",
                 "not found");
    // Vector column is not a FLOAT[N] array.
    expect_error(
      *con,
      "SELECT * FROM sirius_knn_search('vs_err', 'id', " + origin + ", output_columns => ['id']);",
      "FLOAT[N]");
    // k must be >= 1.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_search('vs_err', 'vec', " + origin +
                   ", k => 0, output_columns => ['id']);",
                 "k must be");
    // n_probes must be >= 0.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_search('vs_err', 'vec', " + origin +
                   ", n_probes => -1, output_columns => ['id']);",
                 "n_probes must be");
    // Invalid metric.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['id'], metric => 'bogus');",
                 "metric must be");

    run_ok("SELECT * FROM unpin_table('vs_err');");
  }

  SECTION("execution errors")
  {
    // Table not pinned -> both ANN and ENN require a GPU-pinned table today.
    expect_error(*con,
                 "SELECT * FROM sirius_knn_search('vs_err', 'vec', " + origin +
                   ", output_columns => ['id'], use_index => false);",
                 "must be pinned");

    // Pinned but no ANN index built, use_index defaults true -> clear error.
    run_ok("SELECT * FROM pin_table(name => 'vs_err', tier => 'gpu', format => 'duckdb');");
    expect_error(
      *con,
      "SELECT * FROM sirius_knn_search('vs_err', 'vec', " + origin + ", output_columns => ['id']);",
      "no ANN index");
    run_ok("SELECT * FROM unpin_table('vs_err');");
  }
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_knn_search - multi-chunk pinned table (ANN + ENN)",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  // A pinned chunk == one coalesced DuckDB row group, and row groups are
  // hard-capped at 122880 rows, so >122880 rows guarantees >=2 row groups.
  //
  // The vector values are a permutation of the row order (v = 7*i mod N, a bijection
  // since gcd(7,N)==1), not [i,i,i]. That scatters the small-magnitude vectors
  // (v near 0) across every chunk, so an origin query's nearest neighbors land in
  // multiple chunks -> the ANN cross-chunk gather (gather_pinned_by_global_index's
  // base-offset null-coalesce) actually fires. With plain [i,i,i] the nearest rows
  // all sit in chunk 0, and an interior query at a high row id would hit float32
  // catastrophic cancellation in the ANN path's Expanded L2.
  constexpr int kRows = 200000;  // > 122880 => 2 row groups => 2 pinned chunks
  constexpr int kMod  = 200000;  // gcd(7, 200000) == 1, so v = 7*i mod kMod is a bijection
  run_ok(
    "CREATE TABLE vs_mc AS SELECT i AS id, [v, v, v]::FLOAT[3] AS vec FROM "
    "(SELECT i, (i * 7) % " +
    std::to_string(kMod) + " AS v FROM range(" + std::to_string(kRows) + ") t(i));");
  run_ok("CHECKPOINT;");

  // 4 KiB cap << one row group's decoded bytes, so every row group is its own chunk.
  // Restore the default (512 MiB) before searching, the chunk layout is baked into
  // the pinned entry at pin time; later reads don't re-batch.
  run_ok("SET scan_task_batch_size = 4096;");
  run_ok("SELECT * FROM pin_table(name => 'vs_mc', tier => 'gpu', format => 'duckdb');");
  run_ok("SET scan_task_batch_size = 536870912;");

  // Assert the pin really produced >1 chunk, so this test can never silently
  // degrade into the single-chunk case.
  {
    auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    auto const& mgr                                 = sirius_ctx->get_scan_manager();
    const sirius::scan_manager::pinned_entry* entry = nullptr;
    mgr.visit_pinned_entries(
      [&](std::string_view name, const sirius::scan_manager::pinned_entry& e) {
        if (name == "vs_mc") {
          entry = &e;
          return false;  // found it, stop walking
        }
        return true;  // keep walking
      });
    REQUIRE(entry != nullptr);
    auto it = entry->data_batches_by_column.find("vec");
    REQUIRE(it != entry->data_batches_by_column.end());
    INFO("vec chunk count = " << it->second.size());
    REQUIRE(it->second.size() > 1);
  }

  run_ok("SELECT * FROM sirius_create_ann_index('vs_mc', 'vec', metric => 'l2', n_lists => 16);");

  const std::string origin = "[0.0, 0.0, 0.0]::FLOAT[3]";
  auto exact_ids           = [&](int k) {
    con->Query("SET gpu_execution = false;");
    auto ids = ok_col(*con,
                      "SELECT id FROM vs_mc ORDER BY array_distance(vec, " + origin + ") LIMIT " +
                        std::to_string(k) + ";");
    con->Query("SET gpu_execution = true;");
    return ids;
  };

  // k >= 4 already pulls neighbors out of a later chunk (the permutation places the
  // 4th-nearest vector past the row-group boundary), so the cross-chunk gather runs.
  const std::vector<int> ks{1, 5, 50, 500};

  SECTION("ENN matches exact across chunks")
  {
    for (int k : ks) {
      INFO("ENN k=" << k);
      auto enn = ok_col(*con,
                        "SELECT id FROM sirius_knn_search('vs_mc', 'vec', " + origin + ", k => " +
                          std::to_string(k) + ", output_columns => ['id'], use_index => false);");
      REQUIRE(enn == exact_ids(k));
    }
  }

  SECTION("ANN (IVF-Flat, all lists probed) matches exact across chunks")
  {
    // n_probes == n_lists probes every list -> exact, so it must match the oracle;
    // this is the path that tags global ids at build and gathers them across chunks.
    for (int k : ks) {
      INFO("ANN k=" << k);
      auto ann = ok_col(*con,
                        "SELECT id FROM sirius_knn_search('vs_mc', 'vec', " + origin + ", k => " +
                          std::to_string(k) + ", output_columns => ['id'], n_probes => 16);");
      REQUIRE(ann == exact_ids(k));
    }
  }

  run_ok("SELECT * FROM unpin_table('vs_mc');");
}

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_create_ann_index - error handling and default n_lists",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE idx_err AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(400) t(i);");
  run_ok("CHECKPOINT;");

  SECTION("bind errors (raised before execution)")
  {
    // A NULL positional (arity is fixed at 2, so a short call is a bind error in
    // DuckDB itself; a NULL arg is what reaches our own check).
    expect_error(*con,
                 "SELECT * FROM sirius_create_ann_index(NULL, 'vec');",
                 "two non-NULL positional arguments");
    // Unsupported index_type.
    expect_error(*con,
                 "SELECT * FROM sirius_create_ann_index('idx_err', 'vec', index_type => 'hnsw');",
                 "only 'ivf_flat'");
    // Invalid metric.
    expect_error(*con,
                 "SELECT * FROM sirius_create_ann_index('idx_err', 'vec', metric => 'bogus');",
                 "metric must be");
    // Negative n_lists.
    expect_error(*con,
                 "SELECT * FROM sirius_create_ann_index('idx_err', 'vec', n_lists => -1);",
                 "n_lists must be");
  }

  SECTION("execution errors")
  {
    // Column resolution happens (from the catalog) before the pin check, so these
    // fire without pinning first.
    expect_error(*con, "SELECT * FROM sirius_create_ann_index('idx_err', 'nope');", "not found");
    // Column is not a FLOAT[N] array.
    expect_error(*con, "SELECT * FROM sirius_create_ann_index('idx_err', 'id');", "FLOAT[N]");
    // Table not pinned on the GPU tier.
    expect_error(
      *con, "SELECT * FROM sirius_create_ann_index('idx_err', 'vec');", "must be pinned");
  }

  SECTION("omitted n_lists picks a default and the index still serves a search")
  {
    run_ok("SELECT * FROM pin_table(name => 'idx_err', tier => 'gpu', format => 'duckdb');");
    // default_ivf_n_lists chooses the list count; the build must succeed.
    run_ok("SELECT * FROM sirius_create_ann_index('idx_err', 'vec', metric => 'l2');");

    // n_probes large enough to probe every default list -> exact top-k.
    con->Query("SET gpu_execution = false;");
    auto exact = ok_col(
      *con,
      "SELECT id FROM idx_err ORDER BY array_distance(vec, [0.0,0.0,0.0]::FLOAT[3]) LIMIT 5;");
    con->Query("SET gpu_execution = true;");
    auto got = ok_col(*con,
                      "SELECT id FROM sirius_knn_search('idx_err', 'vec', [0.0,0.0,0.0]::FLOAT[3],"
                      " k => 5, output_columns => ['id'], n_probes => 1000000);");
    REQUIRE(got == exact);

    run_ok("SELECT * FROM unpin_table('idx_err');");
  }
}
