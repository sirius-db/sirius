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

// DuckDB UPDATE chains version values in place, which the pinned cache and
// row-visibility masks cannot apply. These tests prove query preparation
// rejects updated scanned columns before serving and transparent execution
// replays DuckDB's CPU plan in the same transaction (partial #1160).

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/transaction/duck_transaction_manager.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <chrono>
#include <future>
#include <string>

using PinMvccUpdateFixture = sirius::test::GpuExecutionFixture;

namespace {

void expect_runtime_fallback_matches_cpu(sirius::test::GpuExecutionFixture& fx,
                                         const std::string& query)
{
  fx.con->Query("SET gpu_execution = true;");
  auto before     = sirius::test::get_transparent_execution_stats(*fx.con);
  auto gpu_result = fx.con->Query(query);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) {
    UNSCOPED_INFO("runtime-fallback query error: " << gpu_result->GetError());
  }
  REQUIRE_FALSE(gpu_result->HasError());
  auto after = sirius::test::get_transparent_execution_stats(*fx.con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 1);

  fx.con->Query("SET gpu_execution = false;");
  auto cpu_result = fx.con->Query(query);
  fx.con->Query("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  auto gpu_rows = sirius::test::GpuExecutionFixture::collect_rows(
    gpu_result->Cast<duckdb::MaterializedQueryResult>());
  auto cpu_rows = sirius::test::GpuExecutionFixture::collect_rows(
    cpu_result->Cast<duckdb::MaterializedQueryResult>());
  REQUIRE(gpu_rows == cpu_rows);
}

}  // namespace

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: updated cached values fall back before zone-map pruning",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // Move values beyond the pin-time maximum. Without the update guard, stale
  // zone maps can prune the only matching rows and the cache serves an empty
  // result instead of the updated values.
  run_ok("UPDATE t SET v = v + 1000000 WHERE k < 10;");
  expect_runtime_fallback_matches_cpu(*this, "SELECT count(*), sum(v) FROM t WHERE v > 900000;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: host-tier pins fall back for updated scanned columns",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range * 2)::INTEGER AS v "
    "FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='host');");
  run_ok("UPDATE t SET v = -1 WHERE k < 25;");

  expect_runtime_fallback_matches_cpu(*this, "SELECT count(*), sum(v), min(v) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: the writer transaction sees its own updated values",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(1000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("BEGIN TRANSACTION;");
  run_ok("UPDATE t SET v = 77 WHERE k = 1;");
  expect_runtime_fallback_matches_cpu(*this, "SELECT v FROM t WHERE k = 1;");
  run_ok("COMMIT;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: checkpoint and repin restore GPU cache serving",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(1000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("UPDATE t SET v = 99 WHERE k = 1;");

  expect_runtime_fallback_matches_cpu(*this, "SELECT v FROM t WHERE k = 1;");
  run_ok("CALL unpin_table('t');");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // The checkpoint folds away the update chain and the refreshed pin contains
  // the updated value, so the query must execute on GPU without a fallback.
  compare_gpu_vs_cpu("SELECT v FROM t WHERE k = 1;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: a checkpoint under a live pin never serves stale data",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(1000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("UPDATE t SET v = 99 WHERE k = 1;");

  // CHECKPOINT folds away the update chain but does not refresh the pin. The
  // checkpoint-generation guard must still reject the old cached image.
  run_ok("CHECKPOINT;");
  expect_runtime_fallback_matches_cpu(*this, "SELECT v FROM t WHERE k = 1;");

  run_ok("SET enable_duckdb_fallback = false;");
  auto result = con->Query("SELECT v FROM t WHERE k = 1;");
  REQUIRE(result);
  REQUIRE(result->HasError());
  REQUIRE_THAT(result->GetError(), Catch::Contains("checkpointed after pin_table"));
  REQUIRE_THAT(result->GetError(), Catch::Contains("CALL unpin_table('t')"));

  run_ok("SET enable_duckdb_fallback = true;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: query validation waits for an active checkpoint fence",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(1000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("UPDATE t SET v = 99 WHERE k = 1;");

  auto attached = duckdb::DatabaseManager::Get(*con->context).GetDatabase(attach_alias);
  REQUIRE(attached);
  auto checkpoint_lock = duckdb::DuckTransactionManager::Get(*attached).TryGetCheckpointLock();
  REQUIRE(checkpoint_lock);

  auto query =
    std::async(std::launch::async, [&] { return con->Query("SELECT v FROM t WHERE k = 1;"); });
  auto const wait_status = query.wait_for(std::chrono::milliseconds(100));
  checkpoint_lock.reset();
  REQUIRE(wait_status == std::future_status::timeout);

  auto result = query.get();
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->GetValue(0, 0) == duckdb::Value::INTEGER(99));
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: updates to insert-delta rows fall back",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("INSERT INTO t VALUES (50000, 5), (50001, 6);");
  run_ok("UPDATE t SET v = 600000 WHERE k = 50001;");

  expect_runtime_fallback_matches_cpu(*this, "SELECT k, v FROM t WHERE k >= 50000 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: an update on an unscanned column does not poison the pin",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("UPDATE t SET v = v + 1 WHERE k < 10;");

  // k is unchanged, so a scan materializing only k remains exact and stays on GPU.
  compare_gpu_vs_cpu("SELECT sum(k) FROM t;");
  // A filter-only reference still materializes v and must activate the guard.
  expect_runtime_fallback_matches_cpu(*this, "SELECT sum(k) FROM t WHERE v < 10;");
  // Materializing the updated column activates the guard.
  expect_runtime_fallback_matches_cpu(*this, "SELECT sum(v) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: fallback-disabled queries return a user-facing error",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(1000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("UPDATE t SET v = 99 WHERE k = 1;");
  run_ok("SET enable_duckdb_fallback = false;");

  auto result = con->Query("SELECT v FROM t WHERE k = 1;");
  REQUIRE(result);
  REQUIRE(result->HasError());
  REQUIRE_THAT(result->GetError(), Catch::Contains("updated after pin_table"));
  REQUIRE_THAT(result->GetError(), Catch::Contains("SET enable_duckdb_fallback = true"));

  run_ok("SET enable_duckdb_fallback = true;");
  run_ok("CALL unpin_table('t');");
}
