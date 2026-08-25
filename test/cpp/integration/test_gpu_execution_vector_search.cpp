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
#include <cuvs/distance/distance.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius_context.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <vss/cuvs_index_cache.hpp>

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

TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_create_ann_index - prepared statement rebuilds on every execution",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_reexec AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(1000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_reexec', tier => 'gpu', format => 'duckdb');");

  // Prepare once, execute twice. finished now lives in per-execution state, so each
  // execution rebuilds and returns its one success row. When the flag lived in bind
  // data (reused across executions of a prepared plan) the second execution returned
  // zero rows and skipped the rebuild.
  auto prep = con->Prepare(
    "SELECT * FROM sirius_create_ann_index('vs_reexec', 'vec', metric => 'l2', n_lists => 16);");
  REQUIRE(prep);
  if (prep->HasError()) { UNSCOPED_INFO("prepare error: " << prep->GetError()); }
  REQUIRE_FALSE(prep->HasError());

  for (int exec = 1; exec <= 2; ++exec) {
    INFO("execution #" << exec);
    // allow_stream_result = false so we get a materialized result to count rows.
    duckdb::vector<duckdb::Value> params;
    auto res = prep->Execute(params, /*allow_stream_result=*/false);
    REQUIRE(res);
    if (res->HasError()) { UNSCOPED_INFO("execute error: " << res->GetError()); }
    REQUIRE_FALSE(res->HasError());
    auto& mat = res->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(mat.RowCount() == 1);  // rebuilt and returned its success row
  }

  run_ok("SELECT * FROM unpin_table('vs_reexec');");
}

// A deterministic (non-OOM) failed rebuild must not destroy the existing index.
// The builder trains centroids on a single batch, so n_lists that exceeds the
// largest batch can never build even though it is within the total row count.
// We force a two-batch pin (one batch per storage row group) so the largest
// batch (122880 rows) sits strictly below a chosen n_lists that is still under
// the total, then assert the rejected rebuild left the prior index in place.
TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_create_ann_index - failed rebuild leaves the existing index in place",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  // One storage row group is 122880 rows; 200000 rows spans two. A tiny scan
  // batch target puts each row group in its own batch, so the largest batch is
  // 122880 < 150000 <= 200000 total. (scan_task_batch_size is a test-only option,
  // enabled for the C++ test binary.)
  run_ok("SET scan_task_batch_size = 1;");
  run_ok(
    "CREATE TABLE vs_badrebuild AS "
    "SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(200000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_badrebuild', tier => 'gpu', format => 'duckdb');");

  // Build a valid index. Routing looks it up by identity, so we assert on that.
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_badrebuild', 'vec', "
    "metric => 'l2', n_lists => 64);");

  // The identity's catalog is the attached database this fixture routed DDL into.
  auto catq = con->Query("SELECT current_database();");
  REQUIRE(catq);
  REQUIRE_FALSE(catq->HasError());
  auto const catalog = catq->GetValue(0, 0).ToString();
  using Metric       = cuvs::distance::DistanceType;

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& index_cache = sirius_ctx->get_cuvs_index_cache();
  {
    auto entry =
      index_cache.find_by_column(catalog, "main", "vs_badrebuild", "vec", Metric::L2SqrtExpanded);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->meta.n_lists == 64);
  }

  // Rebuild with n_lists above the largest batch but within the total row count.
  // This is deterministic: it cannot build regardless of free memory.
  auto bad = con->Query(
    "SELECT * FROM sirius_create_ann_index('vs_badrebuild', 'vec', "
    "metric => 'l2', n_lists => 150000);");
  REQUIRE(bad);
  REQUIRE(bad->HasError());
  INFO("rebuild error: " << bad->GetError());
  REQUIRE(bad->GetError().find("largest batch size") != std::string::npos);
  REQUIRE(bad->GetError().find("left in place") != std::string::npos);

  // The failure contract: the rebuild was rejected before any erase, so the
  // original index is unchanged, not removed.
  auto entry =
    index_cache.find_by_column(catalog, "main", "vs_badrebuild", "vec", Metric::L2SqrtExpanded);
  REQUIRE(entry != nullptr);
  REQUIRE(entry->meta.n_lists == 64);

  run_ok("SELECT * FROM unpin_table('vs_badrebuild');");
  run_ok("SET scan_task_batch_size = 1048576;");
}

// A created index must resolve two ways: by its management name (the key always
// embeds the routing identity, then the user's name or "default"), and by its
// routing identity via find_by_column. Rebuilding under a different name moves
// the management key but keeps exactly one entry for the identity, so identity
// lookup still resolves.
TEST_CASE_METHOD(VectorSearchFixture,
                 "sirius_create_ann_index - findable by identity and management name",
                 "[integration][gpu_execution][array][vss][vector_search]")
{
  run_ok(
    "CREATE TABLE vs_lookup AS SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(1000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vs_lookup', tier => 'gpu', format => 'duckdb');");

  auto catq = con->Query("SELECT current_database();");
  REQUIRE(catq);
  REQUIRE_FALSE(catq->HasError());
  auto const catalog = catq->GetValue(0, 0).ToString();
  using Metric       = cuvs::distance::DistanceType;

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& index_cache = sirius_ctx->get_cuvs_index_cache();

  auto const identity_prefix = catalog + "_main_vs_lookup_vec_l2_ann_";

  // The cache is shared across test cases in this process, so assert on the net
  // change this test makes rather than an absolute count.
  auto const baseline = index_cache.size();

  // Explicit name => the management key ends with that name; identity lookup
  // resolves to the same entry.
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_lookup', 'vec', name => 'my_idx', "
    "metric => 'l2', n_lists => 16);");
  {
    auto by_name = index_cache.find(identity_prefix + "my_idx");
    auto by_identity =
      index_cache.find_by_column(catalog, "main", "vs_lookup", "vec", Metric::L2SqrtExpanded);
    REQUIRE(by_name != nullptr);
    REQUIRE(by_identity != nullptr);
    REQUIRE(by_name == by_identity);              // same entry, reached two ways
    REQUIRE(index_cache.size() == baseline + 1);  // one entry added
  }

  // No name => the suffix defaults to "default". Rebuilding the same identity
  // replaces the old entry, so the old "my_idx" key is gone but identity lookup
  // still resolves (to the new "default" entry).
  run_ok(
    "SELECT * FROM sirius_create_ann_index('vs_lookup', 'vec', "
    "metric => 'l2', n_lists => 16);");
  {
    // The old name is gone (replaced, not appended) and identity still resolves.
    REQUIRE(index_cache.find(identity_prefix + "my_idx") == nullptr);
    auto by_name = index_cache.find(identity_prefix + "default");
    auto by_identity =
      index_cache.find_by_column(catalog, "main", "vs_lookup", "vec", Metric::L2SqrtExpanded);
    REQUIRE(by_name != nullptr);
    REQUIRE(by_identity != nullptr);
    REQUIRE(by_name == by_identity);
    REQUIRE(index_cache.size() == baseline + 1);  // replaced, not appended
  }

  run_ok("SELECT * FROM unpin_table('vs_lookup');");
}
