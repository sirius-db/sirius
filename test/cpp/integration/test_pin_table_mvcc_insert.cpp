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

// End-to-end MVCC INSERT tests for duckdb-pinned tables: post-pin rows serve
// through the insert delta with snapshot-correct results, and shapes the delta
// cannot represent decline into the transparent CPU fallback. The basic
// served/declined flips live in test_pin_table_mvcc_delete.cpp.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <memory>
#include <string>

using PinMvccInsertFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Runs @p query expecting a plan-time decline into the transparent CPU
/// fallback, then checks the results against a plain CPU run.
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

/// Opens a second connection to the same database (concurrent writer / older snapshot).
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

/// Compares GPU vs CPU results on the given connection (no counter assertions).
void compare_gpu_vs_cpu_on(duckdb::Connection& con, const std::string& query)
{
  run_ok_on(con, "SET gpu_execution = true;");
  auto gpu_result = con.Query(query);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("gpu query error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());
  run_ok_on(con, "SET gpu_execution = false;");
  auto cpu_result = con.Query(query);
  run_ok_on(con, "SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());
  auto gpu_rows = sirius::test::GpuExecutionFixture::collect_rows(
    gpu_result->Cast<duckdb::MaterializedQueryResult>());
  auto cpu_rows = sirius::test::GpuExecutionFixture::collect_rows(
    cpu_result->Cast<duckdb::MaterializedQueryResult>());
  REQUIRE(gpu_rows == cpu_rows);
}

}  // namespace

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: rows inserted then deleted before the query never appear",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 50000 FROM range(200);");
  run_ok("DELETE FROM t WHERE k >= 50100;");  // half the delta, tombstoned pre-query

  compare_gpu_vs_cpu("SELECT count(*), max(k), sum(k) FROM t;");
  compare_gpu_vs_cpu("SELECT k FROM t WHERE k >= 49990 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: an older snapshot does not see newer committed inserts",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  auto reader = second_connection(*this);
  run_ok_on(*reader, "USE " + attach_alias + ";");
  run_ok_on(*reader, "BEGIN TRANSACTION;");
  run_ok_on(*reader, "SELECT count(*) FROM t;");  // pin the snapshot (lazy txn spawn)

  run_ok("INSERT INTO t SELECT range::INTEGER + 50000 FROM range(500);");

  // The old snapshot: delta rows exist physically but are invisible to it.
  compare_gpu_vs_cpu_on(*reader, "SELECT count(*), max(k) FROM t;");
  run_ok_on(*reader, "ROLLBACK;");

  // A fresh snapshot sees them.
  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: base deletes and post-pin inserts combine in one query",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range * 2)::INTEGER AS v "
    "FROM range(200000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("DELETE FROM t WHERE k % 1000 = 7;");  // masks over the cached base
  run_ok("INSERT INTO t SELECT range::INTEGER + 200000, 0 FROM range(3000);");
  run_ok("DELETE FROM t WHERE k >= 202000;");  // and over the delta

  compare_gpu_vs_cpu("SELECT count(*), sum(k), sum(v) FROM t;");
  compare_gpu_vs_cpu("SELECT k, v FROM t WHERE k > 199990 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: bulk inserts ride the persistent MergeStorage lane",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // A single commit of at least one full row group: MergeStorage flushes it as
  // persistent row groups, the delta's file lane.
  run_ok("INSERT INTO t SELECT range::INTEGER + 50000 FROM range(130000);");
  // Plus a small transient append: both lanes in one delta.
  run_ok("INSERT INTO t VALUES (180000);");

  compare_gpu_vs_cpu("SELECT count(*), max(k), sum(k) FROM t;");
  compare_gpu_vs_cpu("SELECT count(*) FROM t WHERE k >= 100000;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: bulk constant inserts decode from blockless CONSTANT segments",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(20000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // A full-row-group commit of a projected literal: MergeStorage flushes the
  // constant column as blockless persistent CONSTANT segments, decoded from
  // stats rather than the file.
  run_ok("INSERT INTO t SELECT range::INTEGER + 20000, 7 FROM range(130000);");

  compare_gpu_vs_cpu("SELECT count(*), sum(k), sum(v), min(v), max(v) FROM t;");
  compare_gpu_vs_cpu("SELECT k, v FROM t WHERE k >= 149995 ORDER BY k;");
  // Constant-column-only projection: the persistent delta splits become
  // blockless-only (no file reads, no datasource) and must still serve.
  compare_gpu_vs_cpu("SELECT sum(v), min(v), max(v) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: appends into an indexed tail row group keep per-segment constants",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  // With a PRIMARY KEY, DuckDB appends into the existing tail row group
  // instead of opening a fresh one, so the bulk-flushed CONSTANT segment's
  // row group later gains rows with a different value: the row-group-level
  // stats drift (min drops to 7) while the segment's own constant stays 9.
  run_ok("CREATE TABLE t (k INTEGER PRIMARY KEY, v INTEGER);");
  run_ok("INSERT INTO t SELECT range::INTEGER, range::INTEGER FROM range(20000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 20000, 9 FROM range(130000);");
  run_ok("INSERT INTO t VALUES (150001, 7);");

  compare_gpu_vs_cpu("SELECT count(*), sum(v), min(v), max(v) FROM t;");
  compare_gpu_vs_cpu("SELECT k, v FROM t WHERE k >= 149998 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: bulk all-NULL inserts keep their NULLs",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  // The base carries an all-NULL column too, so the cached read exercises the
  // same CONSTANT-validity decode as the delta.
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, NULL::INTEGER AS v FROM range(20000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // Bulk all-NULL column: the flushed data segments are CONSTANT sentinels and
  // the NULL-ness lives only in the CONSTANT validity segments' stats.
  run_ok("INSERT INTO t SELECT range::INTEGER + 20000, NULL::INTEGER FROM range(130000);");

  compare_gpu_vs_cpu("SELECT count(*), count(v), sum(k) FROM t;");
  compare_gpu_vs_cpu("SELECT k, v FROM t WHERE k >= 149995 ORDER BY k;");
  // All-NULL-column-only projection: blockless-only splits, no datasource.
  compare_gpu_vs_cpu("SELECT count(v) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: bulk all-NULL varchar keeps its NULLs",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  // All-NULL VARCHAR flushes dictionary data with CONSTANT all-null validity
  // (varchar data itself never compresses CONSTANT), exercising the varchar
  // decoder's synthesized-validity path through both the base and the delta.
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, NULL::VARCHAR AS s FROM range(20000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 20000, NULL::VARCHAR FROM range(130000);");

  compare_gpu_vs_cpu("SELECT count(*), count(s) FROM t;");
  compare_gpu_vs_cpu("SELECT k, s FROM t WHERE k >= 149995 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: an unaligned all-NULL row group does not bleed nulls forward",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  // 10001 all-NULL rows end mid-byte. When the next row group's valid rows
  // decode in the same coalesced split, the synthesized validity's trailing
  // padding bits must not null the following rows.
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, NULL::INTEGER AS v FROM range(10001);");
  run_ok("CHECKPOINT;");
  run_ok("INSERT INTO t SELECT range::INTEGER + 10001, range::INTEGER FROM range(5000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  compare_gpu_vs_cpu("SELECT count(*), count(v), sum(v) FROM t;");
  compare_gpu_vs_cpu("SELECT k, v FROM t WHERE k BETWEEN 9998 AND 10008 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: varchar and NULLs decode through the delta",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, 'base_'||range AS s, "
    "CASE WHEN range % 3 = 0 THEN NULL ELSE range::INTEGER END AS opt "
    "FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok(
    "INSERT INTO t SELECT range::INTEGER + 50000, "
    "CASE WHEN range % 97 = 0 THEN '' ELSE 'delta_'||range END, "
    "CASE WHEN range % 5 = 0 THEN NULL ELSE range::INTEGER END "
    "FROM range(2000);");

  compare_gpu_vs_cpu("SELECT count(*), count(opt), min(s), max(s) FROM t;");
  compare_gpu_vs_cpu("SELECT k, s, opt FROM t WHERE k >= 49995 ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: host-tier pins serve the delta too",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(80000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='host');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 80000 FROM range(1000);");
  run_ok("DELETE FROM t WHERE k IN (5, 80003);");

  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: a self-join shares one delta capture across both sides",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, (range % 100)::INTEGER AS g "
    "FROM range(60000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 60000, 7 FROM range(300);");

  compare_gpu_vs_cpu(
    "SELECT a.g, count(*) FROM t a JOIN t b ON a.k = b.k GROUP BY a.g ORDER BY a.g;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: joins with unpinned tables see the delta rows",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(60000);");
  run_ok(
    "CREATE TABLE u AS SELECT range::INTEGER AS k, (range % 10)::INTEGER AS g "
    "FROM range(70000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 60000 FROM range(5000);");

  compare_gpu_vs_cpu("SELECT u.g, count(*) FROM t JOIN u ON t.k = u.k GROUP BY u.g ORDER BY u.g;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: a delta growing across queries stays snapshot-exact",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(40000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t VALUES (40000);");
  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");

  run_ok("INSERT INTO t SELECT range::INTEGER + 40001 FROM range(999);");
  compare_gpu_vs_cpu("SELECT count(*), max(k) FROM t;");

  run_ok("DELETE FROM t WHERE k = 40500;");
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: a tiny pin serves a delta larger than its base",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t SELECT range::INTEGER + 10 FROM range(5000);");
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc insert: zone-map-pruned base still serves matching delta rows",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(100000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t VALUES (1000000), (1000001);");

  // The pushed filter zone-map-prunes every cached chunk; only the delta rows
  // match and must still arrive.
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t WHERE k >= 500000;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc guards: big-string deltas decline at plan time",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, 'short_'||range AS s FROM range(20000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // The overflow string lives only in the delta, but the table-level stat
  // (merged on append) makes the plan-time varchar probe decline. The query
  // must project the string column; without it the scan legitimately serves.
  run_ok("INSERT INTO t VALUES (20000, repeat('x', 5000));");
  expect_fallback_matches_cpu(*this, "SELECT count(*), max(k), max(strlen(s)) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc guards: rowid-only scans keep declining with a delta present",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(30000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t VALUES (30000);");
  // Bare count(*) binds only rowid, which is never cached, so the
  // column-mismatch guard declines; the fallback must still count the delta row.
  expect_fallback_matches_cpu(*this, "SELECT count(*) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccInsertFixture,
                 "mvcc guards: ARRAY projections decline while a delta exists",
                 "[integration][gpu_execution][pin_table_mvcc_insert]")
{
  run_ok(
    "CREATE TABLE t AS SELECT range::INTEGER AS k, "
    "CAST([range::INTEGER, 1, 2] AS INTEGER[3]) AS a FROM range(20000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  run_ok("INSERT INTO t VALUES (20000, CAST([9, 9, 9] AS INTEGER[3]));");

  // Projecting the ARRAY column with delta rows present declines; the fallback
  // must still return correct rows.
  expect_fallback_matches_cpu(*this, "SELECT k, a FROM t WHERE k >= 19998 ORDER BY k;");
  // Projecting only the scalar column serves from cache + delta as usual.
  compare_gpu_vs_cpu("SELECT count(*), sum(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}
