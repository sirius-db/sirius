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

// DuckDB UPDATE chains version values in place, which the pinned cache cannot
// apply. These tests prove update-producing statements fail before modifying a
// pinned table and checkpoint replacement cannot race a pinned scan.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/transaction/duck_transaction_manager.hpp>
#include <sirius_context.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <chrono>
#include <future>
#include <string>

using PinMvccUpdateFixture = sirius::test::GpuExecutionFixture;

namespace {

void require_pinned_update_error(duckdb::QueryResult& result,
                                 const std::string& table_name,
                                 const std::string& pinned_name)
{
  REQUIRE(result.HasError());
  REQUIRE_THAT(result.GetError(), Catch::Contains("does not support UPDATE"));
  REQUIRE_THAT(result.GetError(), Catch::Contains("pinned DuckDB table '" + table_name + "'"));
  REQUIRE_THAT(result.GetError(), Catch::Contains("CALL unpin_table('" + pinned_name + "')"));
}

}  // namespace

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: UPDATE fails before modifying a pinned table",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT 1::INTEGER AS k, 10::INTEGER AS v;");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  auto result = con->Query("UPDATE t SET v = 99 WHERE k = 1;");
  REQUIRE(result);
  require_pinned_update_error(*result, "t", "t");

  run_ok("SET gpu_execution = false;");
  auto unchanged = con->Query("SELECT v FROM t;");
  REQUIRE(unchanged);
  REQUIRE_FALSE(unchanged->HasError());
  REQUIRE(unchanged->GetValue(0, 0) == duckdb::Value::INTEGER(10));

  run_ok("CALL unpin_table('t');");
  run_ok("UPDATE t SET v = 99 WHERE k = 1;");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: prepared UPDATE observes pins created after prepare",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT 1::INTEGER AS k, 10::INTEGER AS v;");
  run_ok("CHECKPOINT;");
  auto prepared = con->Prepare("UPDATE t SET v = ? WHERE k = 1;");
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());

  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  auto result = prepared->Execute(99);
  REQUIRE(result);
  require_pinned_update_error(*result, "t", "t");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: another connection cannot update a pinned table",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT 1::INTEGER AS k, 10::INTEGER AS v;");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='host');");

  duckdb::Connection writer(*con->context->db);
  auto use_result = writer.Query("USE " + attach_alias + ";");
  REQUIRE(use_result);
  REQUIRE_FALSE(use_result->HasError());
  auto result = writer.Query("UPDATE t SET v = 99 WHERE k = 1;");
  REQUIRE(result);
  require_pinned_update_error(*result, "t", "t");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: pinning waits for an executing update",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT 1::INTEGER AS k, 10::INTEGER AS v;");
  run_ok("CHECKPOINT;");

  auto prepared = con->Prepare("UPDATE t SET v = v WHERE false;");
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  auto pending_update = prepared->PendingQuery();
  REQUIRE(pending_update);
  REQUIRE_FALSE(pending_update->HasError());
  auto connection_state = duckdb::get_sirius_connection_state(*con->context);
  REQUIRE(connection_state);
  REQUIRE(connection_state->has_pinned_update_guard());

  duckdb::Connection pinner(*con->context->db);
  auto use_result = pinner.Query("USE " + attach_alias + ";");
  REQUIRE(use_result);
  REQUIRE_FALSE(use_result->HasError());
  auto pin               = std::async(std::launch::async, [&] {
    return pinner.Query("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  });
  auto const wait_status = pin.wait_for(std::chrono::milliseconds(100));

  auto update_result = pending_update->Execute();
  REQUIRE(update_result);
  REQUIRE_FALSE(update_result->HasError());
  REQUIRE(wait_status == std::future_status::timeout);

  auto pin_result = pin.get();
  REQUIRE(pin_result);
  REQUIRE_FALSE(pin_result->HasError());
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: update-producing INSERT and MERGE forms are rejected",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t(k INTEGER PRIMARY KEY, v INTEGER);");
  run_ok("INSERT INTO t VALUES (1, 10);");
  run_ok("CREATE TABLE source(k INTEGER, v INTEGER);");
  run_ok("INSERT INTO source VALUES (1, 99);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  auto conflict =
    con->Query("INSERT INTO t VALUES (1, 99) ON CONFLICT(k) DO UPDATE SET v = excluded.v;");
  REQUIRE(conflict);
  require_pinned_update_error(*conflict, "t", "t");

  auto merge = con->Query(
    "MERGE INTO t USING source ON t.k = source.k "
    "WHEN MATCHED THEN UPDATE SET v = source.v;");
  REQUIRE(merge);
  require_pinned_update_error(*merge, "t", "t");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: pinning existing update chains requires a checkpoint",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT 1::INTEGER AS k, 10::INTEGER AS v;");
  run_ok("CHECKPOINT;");
  run_ok("UPDATE t SET v = 99 WHERE k = 1;");

  auto pin = con->Query("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  REQUIRE(pin);
  REQUIRE(pin->HasError());
  REQUIRE_THAT(pin->GetError(), Catch::Contains("has in-memory update chains"));
  REQUIRE_THAT(pin->GetError(), Catch::Contains("run CHECKPOINT before pinning"));

  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  compare_gpu_vs_cpu("SELECT v FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccUpdateFixture,
                 "mvcc update guard: a checkpoint under a live pin never serves stale data",
                 "[integration][gpu_execution][pin_table_mvcc_update]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("INSERT INTO t VALUES (50000, 50000);");
  run_ok("CHECKPOINT;");

  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");
  auto result = con->Query("SELECT sum(v) FROM t;");
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
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("SET gpu_execution = true;");

  auto attached = duckdb::DatabaseManager::Get(*con->context).GetDatabase(attach_alias);
  REQUIRE(attached);
  auto checkpoint_lock = duckdb::DuckTransactionManager::Get(*attached).TryGetCheckpointLock();
  REQUIRE(checkpoint_lock);

  auto query = std::async(std::launch::async, [&] { return con->Query("SELECT sum(v) FROM t;"); });
  auto const wait_status = query.wait_for(std::chrono::milliseconds(100));
  checkpoint_lock.reset();
  REQUIRE(wait_status == std::future_status::timeout);

  auto result = query.get();
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  run_ok("CALL unpin_table('t');");
}
