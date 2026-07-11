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

// MVCC delete end-to-end: a duckdb-pinned table keeps serving GPU queries
// while DELETEs land — every query's results must equal DuckDB CPU at its own
// snapshot (per-query keep-masks over the cached base). States the masks
// cannot represent — post-pin INSERTs, in-place UPDATEs, snapshots older than
// the pin, columns the pin does not cover — must decline at plan time into a
// clean transparent CPU fallback (counters {0 rebinds, 1 fallback, 0
// executions}) with correct results, never the MVCC-blind disk-native read.
// Cache-served queries keep the {1, 0, 1} signature compare_gpu_vs_cpu
// asserts.
//
// Unpinned tables have no masks: deletes, in-flight or txn-local inserts, and
// update chains on scanned columns must decline the same way, while tables
// that are merely probe-dirty from this session's committed writes keep the
// GPU path.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <memory>
#include <string>
#include <vector>

using PinMvccDeleteFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Run @p query under gpu_execution=true expecting the plan-time guard to
/// decline (transparent CPU fallback: {0 rebinds, 1 fallback, 0 executions})
/// and the results to match a plain CPU run.
void expect_fallback_matches_cpu(sirius::test::GpuExecutionFixture& fx, const std::string& query)
{
  fx.con->Query("SET gpu_execution = true;");
  auto before     = sirius::test::get_transparent_execution_stats(*fx.con);
  auto gpu_result = fx.con->Query(query);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("guarded query error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());
  auto after = sirius::test::get_transparent_execution_stats(*fx.con);
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);

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

/// A second connection to the same database instance (concurrent-writer /
/// older-snapshot roles).
std::unique_ptr<duckdb::Connection> second_connection(sirius::test::GpuExecutionFixture& fx)
{
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    return std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
  }
  REQUIRE(fx.db);
  return std::make_unique<duckdb::Connection>(*fx.db);
}

void run_ok_on(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

}  // namespace

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: committed post-pin deletes are masked out of the cached serve",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range * 2)::INTEGER AS v "
    "FROM range(300000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // Word/vector/row-group boundary rows plus a spread — the same shapes the
  // unit tests cover, now through the full transparent pipeline.
  run_ok(
    "DELETE FROM t WHERE k IN (0, 31, 32, 2047, 2048, 122879, 122880, 122881, 299999) "
    "OR k % 50000 = 17;");

  compare_gpu_vs_cpu("SELECT count(*), sum(k), sum(v) FROM t;");
  compare_gpu_vs_cpu("SELECT k, v FROM t WHERE k < 5000;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: pre-pin checkpointed tombstones are filtered",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000);");
  run_ok("CHECKPOINT;");
  // Sparse deletes survive the checkpoint as persisted tombstones (the vacuum
  // only rewrites row groups whose counts drop enough to merge) — the cache
  // then holds the tombstoned rows byte-for-byte and the mask must drop them.
  run_ok("DELETE FROM t WHERE k % 1000 = 7;");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  compare_gpu_vs_cpu("SELECT k FROM t WHERE k < 3000;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(
  PinMvccDeleteFixture,
  "mvcc delete: the query's own uncommitted deletes are honored and rollback restores",
  "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(100000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("BEGIN TRANSACTION;");
  run_ok("DELETE FROM t WHERE k < 1000;");
  compare_gpu_vs_cpu("SELECT count(*), min(k) FROM t;");
  run_ok("ROLLBACK;");

  compare_gpu_vs_cpu("SELECT count(*), min(k) FROM t;");  // every row back
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: an untouched pin serves the all-visible fast path",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(200000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: host-tier pins mask deletes too",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='host');");
  run_ok("DELETE FROM t WHERE k IN (5, 2048, 122880) OR k % 70000 = 3;");
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: multi-chunk pin masks the dirty chunk and fast-paths the clean one",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  // integration.yaml caps scan_task_batch_size at 100 MB; 13M BIGINTs decode
  // past it, so the pin materializes as more than one chunk (the same sizing
  // the mvcc foundation test uses).
  run_ok("CREATE TABLE t AS SELECT range AS k FROM range(13000000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  // All deletes land in the first row groups: later chunks stay mask-less.
  run_ok("DELETE FROM t WHERE k < 100000 AND k % 7 = 0;");
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: a fully-deleted table serves an empty result",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("DELETE FROM t;");
  // A bare count(*) compiles to a rowid-only scan the cache cannot serve
  // (rowid is not cached; unguarded, such scans silently read the STALE disk
  // image and would count 50000 here) — once diverged, cache-or-CPU sends
  // them to the CPU fallback with the correct (empty) answer.
  expect_fallback_matches_cpu(*this, "SELECT count(*) FROM t;");
  // Column-anchored scans serve from the cache, masked down to emptiness.
  compare_gpu_vs_cpu("SELECT count(k) FROM t;");
  compare_gpu_vs_cpu("SELECT k FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: masks are recomputed per query as deletes accumulate",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(100000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("DELETE FROM t WHERE k < 10;");
  compare_gpu_vs_cpu("SELECT count(*), min(k) FROM t;");
  run_ok("DELETE FROM t WHERE k >= 99990;");
  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");
  run_ok("DELETE FROM t WHERE k % 2 = 0;");
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc guards: a transaction older than the pin falls back until it commits",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(100000);");
  run_ok("CHECKPOINT;");

  // Connection A opens its snapshot on the attached db BEFORE the pin exists.
  auto con_a = second_connection(*this);
  run_ok_on(*con_a, "USE " + attach_alias + ";");
  run_ok_on(*con_a, "SET gpu_execution = true;");
  run_ok_on(*con_a, "BEGIN TRANSACTION;");
  run_ok_on(*con_a, "SELECT count(*) FROM t;");  // fixes A's per-db start_time

  // Connection B (the fixture) deletes and pins: v_base is newer than A's
  // snapshot, and A must still see the pre-delete rows.
  run_ok("DELETE FROM t WHERE k < 500;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  {
    // count(k) anchors a cached column (a bare count(*) is a rowid-only scan
    // the cache never serves).
    auto before = sirius::test::get_transparent_execution_stats(*con_a);
    auto result = con_a->Query("SELECT count(k) FROM t;");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto after = sirius::test::get_transparent_execution_stats(*con_a);
    sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);
    // A's snapshot predates the delete: the fallback must serve all rows.
    REQUIRE(result->GetValue(0, 0).ToString() == "100000");
  }

  run_ok_on(*con_a, "COMMIT;");

  {
    // Fresh snapshot: newer than v_base, no inserts — served from the cache.
    auto before = sirius::test::get_transparent_execution_stats(*con_a);
    auto result = con_a->Query("SELECT count(k) FROM t;");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto after = sirius::test::get_transparent_execution_stats(*con_a);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE(result->GetValue(0, 0).ToString() == "99500");
  }

  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc guards: committed post-pin inserts decline to CPU with correct rows",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 50000 FROM range(100);");
  expect_fallback_matches_cpu(*this, "SELECT count(*), max(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc guards: own-transaction inserts decline, rollback resumes GPU serving",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("BEGIN TRANSACTION;");
  run_ok("INSERT INTO t VALUES (900001), (900002);");
  expect_fallback_matches_cpu(*this, "SELECT count(*), max(k) FROM t;");  // sees its own rows
  run_ok("ROLLBACK;");

  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");  // cache serving resumes
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc guards: in-place updates decline to CPU with correct values",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range * 2)::INTEGER AS v "
    "FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  SECTION("value update")
  {
    run_ok("UPDATE t SET v = v + 1 WHERE k < 10;");
    expect_fallback_matches_cpu(*this, "SELECT sum(v) FROM t;");
  }

  SECTION("validity-only update (SET NULL)")
  {
    run_ok("UPDATE t SET v = NULL WHERE k < 10;");
    expect_fallback_matches_cpu(*this, "SELECT count(v), count(*) FROM t;");
  }
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc guards: column-mismatch serves the disk path only while exact",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS a, (range * 3)::INTEGER AS b "
    "FROM range(100000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu', cols=['a']);");

  // The pin cannot serve column b, but every row is visible — the scan takes
  // the native GPU path ({1,0,1}) even though session writes left the table
  // probe-dirty.
  compare_gpu_vs_cpu("SELECT sum(b) FROM t;");

  // After a committed DELETE the disk image is stale: the same scan must now
  // decline to CPU — never the disk-native read.
  run_ok("DELETE FROM t WHERE a < 100;");
  expect_fallback_matches_cpu(*this, "SELECT sum(b) FROM t;");

  // The covered column keeps serving from the cache, masked.
  compare_gpu_vs_cpu("SELECT count(*), sum(a) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc guards: pinning a table with update chains is refused until CHECKPOINT",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range * 2)::INTEGER AS v "
    "FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("UPDATE t SET v = v + 1 WHERE k < 10;");

  auto refused = con->Query("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  REQUIRE(refused);
  REQUIRE(refused->HasError());
  REQUIRE_THAT(refused->GetError(), Catch::Contains("update chains"));

  run_ok("CHECKPOINT;");  // folds the chains into the base data
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  compare_gpu_vs_cpu("SELECT sum(v) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "mvcc delete: a masked pinned table joins an unpinned table correctly",
                 "[integration][gpu_execution][pin_table_mvcc_delete]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(100000);");
  run_ok(
    "CREATE TABLE u AS SELECT range::INTEGER AS k, (range % 10)::INTEGER AS g "
    "FROM range(100000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("DELETE FROM t WHERE k % 3 = 0;");

  compare_gpu_vs_cpu("SELECT u.g, count(*) FROM t JOIN u ON t.k = u.k GROUP BY u.g ORDER BY u.g;");
  run_ok("CALL unpin_table('t');");
}

//===----------------------------------------------------------------------===//
// Unpinned tables: the disk-native read must decline any divergence (#1143)
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "native mvcc guard: uncheckpointed deletes on an unpinned table fall back",
                 "[integration][gpu_execution][duckdb_native_mvcc_guard]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000);");
  // Session-written table: probe-dirty but every row visible — stays on GPU.
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");

  run_ok("DELETE FROM t WHERE k IN (0, 2048, 122880) OR k % 50000 = 17;");
  expect_fallback_matches_cpu(*this, "SELECT count(*), sum(k) FROM t;");
  // rowid queries must not see deleted rows.
  expect_fallback_matches_cpu(*this, "SELECT min(rowid), count(*) FROM t;");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "native mvcc guard: update chains gate only the scanned columns",
                 "[integration][gpu_execution][duckdb_native_mvcc_guard]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range * 2)::INTEGER AS v "
    "FROM range(200000);");
  compare_gpu_vs_cpu("SELECT sum(v) FROM t;");

  run_ok("UPDATE t SET v = v + 1 WHERE k % 1000 = 3;");
  expect_fallback_matches_cpu(*this, "SELECT sum(v) FROM t;");
  // The untouched column carries no chains — still GPU-served.
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");

  run_ok("CHECKPOINT;");  // folds the chains into the base data
  compare_gpu_vs_cpu("SELECT sum(v) FROM t;");
}

TEST_CASE_METHOD(PinMvccDeleteFixture,
                 "native mvcc guard: transaction-local inserts fall back until resolved",
                 "[integration][gpu_execution][duckdb_native_mvcc_guard]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(200000);");
  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");

  run_ok("BEGIN TRANSACTION;");
  run_ok("INSERT INTO t VALUES (1000000);");
  // The txn must see its own row; the disk-native read cannot.
  expect_fallback_matches_cpu(*this, "SELECT count(*), max(k) FROM t;");
  run_ok("ROLLBACK;");

  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");
}
