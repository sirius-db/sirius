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

// End-to-end delta promotion: whole closed row groups of a duckdb pin's insert
// delta are cached into the entry at query end, so later queries serve them
// from the base and the per-query delta shrinks. Correctness (GPU == CPU) with
// promotion live is also covered across test_pin_table_mvcc_insert.cpp's
// matrix; here we assert the promotion-specific observable behavior via
// sirius_pinned_tables().
//
// These tests use a row-only GPU-vs-CPU check (not the fixture's
// compare_gpu_vs_cpu, whose exact transparent-execution counter assertions
// are perturbed by promotion mutating the cache between queries) and start by
// unpinning 't' defensively, since the SiriusContext / scan manager is shared
// across the suite. GpuExecutionFixture, real GPU.

#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdint>
#include <string>

using PinMvccPromotionFixture = sirius::test::GpuExecutionFixture;

namespace {

/// A single scalar column of sirius_pinned_tables() for pin @p name, as a
/// string ("NULL" for SQL NULL, "<none>" when the pin is absent).
std::string tvf_scalar(duckdb::Connection& con, const std::string& col, const std::string& name)
{
  auto r = con.Query("SELECT " + col + " FROM sirius_pinned_tables() WHERE name = '" + name + "';");
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("tvf error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  if (r->RowCount() == 0) { return "<none>"; }
  auto v = r->GetValue(0, 0);
  return v.IsNull() ? "NULL" : v.ToString();
}

std::uint64_t tvf_u64(duckdb::Connection& con, const std::string& col, const std::string& name)
{
  return std::stoull(tvf_scalar(con, col, name));
}

/// GPU result equals the CPU result (multiset). Unlike the fixture's
/// compare_gpu_vs_cpu, asserts no transparent-execution counts — so it tolerates
/// the rebind/fallback shifts promotion causes and never aborts a test partway
/// (which would leak the shared pin).
void rows_match(duckdb::Connection& con, const std::string& query)
{
  con.Query("SET gpu_execution = true;");
  auto gpu = con.Query(query);
  REQUIRE(gpu);
  if (gpu->HasError()) { UNSCOPED_INFO("gpu error: " << gpu->GetError()); }
  REQUIRE_FALSE(gpu->HasError());
  con.Query("SET gpu_execution = false;");
  auto cpu = con.Query(query);
  con.Query("SET gpu_execution = true;");
  REQUIRE(cpu);
  REQUIRE_FALSE(cpu->HasError());
  auto g =
    sirius::test::GpuExecutionFixture::collect_rows(gpu->Cast<duckdb::MaterializedQueryResult>());
  auto c =
    sirius::test::GpuExecutionFixture::collect_rows(cpu->Cast<duckdb::MaterializedQueryResult>());
  REQUIRE(g == c);
}

}  // namespace

TEST_CASE_METHOD(PinMvccPromotionFixture,
                 "promotion: a bulk delta promotes into the cache and shrinks the delta",
                 "[integration][gpu_execution][pin_table_mvcc_promotion]")
{
  run_ok("CALL unpin_table('t');");  // defensive: shared scan manager
  run_ok("SET enable_delta_promotion = true;");
  run_ok("SET threads TO 1;");  // deterministic row-group layout
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, (range*2)::BIGINT AS v FROM range(10000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  // Bulk insert: two full persistent row groups plus a sealed partial tail.
  run_ok(
    "INSERT INTO t SELECT (10000+range)::INTEGER, ((10000+range)*2)::BIGINT FROM range(260000);");

  // Nothing promoted yet; the whole delta is on the per-query path.
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") == 0);
  REQUIRE(tvf_u64(*con, "delta_insert_rows", "t") == 260000);

  // A read decodes the delta and promotes its closed row groups at QueryEnd.
  rows_match(*con, "SELECT count(*), sum(k), max(v) FROM t;");

  auto const promoted = tvf_u64(*con, "promoted_rows", "t");
  REQUIRE(promoted >= 245760);  // at least the two full row groups
  // The base grew by exactly the promoted rows, so the per-query delta shrank.
  REQUIRE(tvf_u64(*con, "delta_insert_rows", "t") == 260000 - promoted);
  REQUIRE(tvf_u64(*con, "base_rows", "t") == 10000 + promoted);

  // Results stay exact after promotion, and a second read is a no-op for the promoted rows.
  rows_match(*con, "SELECT k, v FROM t WHERE k IN (5, 10000, 150000, 269999) ORDER BY k;");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") == promoted);
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccPromotionFixture,
                 "promotion: DELETEs against promoted rows are masked correctly",
                 "[integration][gpu_execution][pin_table_mvcc_promotion]")
{
  run_ok("CALL unpin_table('t');");
  run_ok("SET enable_delta_promotion = true;");
  run_ok("SET threads TO 1;");
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("INSERT INTO t SELECT (10000+range)::INTEGER FROM range(200000);");

  // A projecting read (not count(*)-only) becomes the carrier and promotes.
  rows_match(*con, "SELECT count(*), sum(k) FROM t;");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") > 0);

  // Delete rows that now live inside a promoted (base) chunk.
  run_ok("DELETE FROM t WHERE k IN (20000, 50000, 100000);");
  rows_match(*con, "SELECT count(*), sum(k) FROM t;");
  rows_match(*con, "SELECT k FROM t WHERE k BETWEEN 19998 AND 20002 ORDER BY k;");
  REQUIRE(tvf_u64(*con, "delta_delete_rows", "t") >= 3);
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccPromotionFixture,
                 "promotion: an indexed table (k_offset > 0) does not promote",
                 "[integration][gpu_execution][pin_table_mvcc_promotion]")
{
  run_ok("CALL unpin_table('t');");
  run_ok("SET enable_delta_promotion = true;");
  run_ok("CREATE TABLE t (k INTEGER PRIMARY KEY, v INTEGER);");
  run_ok("INSERT INTO t SELECT range, range FROM range(10000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  // A PRIMARY KEY appends into the boundary row group, so the delta starts
  // mid-row-group and promotion is disabled for this pin.
  run_ok("INSERT INTO t VALUES (10000, 1), (10001, 2);");

  rows_match(*con, "SELECT count(*), sum(k) FROM t;");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") == 0);
  // Repeated reads stay correct whether the delta serves or declines to CPU.
  rows_match(*con, "SELECT k FROM t WHERE k >= 9998 ORDER BY k;");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") == 0);
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccPromotionFixture,
                 "promotion: SET enable_delta_promotion = false keeps the delta per-query",
                 "[integration][gpu_execution][pin_table_mvcc_promotion]")
{
  run_ok("CALL unpin_table('t');");
  run_ok("SET threads TO 1;");
  run_ok("SET enable_delta_promotion = false;");
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("INSERT INTO t SELECT (10000+range)::INTEGER FROM range(200000);");

  rows_match(*con, "SELECT count(*), sum(k) FROM t;");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") == 0);           // off: nothing absorbed
  REQUIRE(tvf_u64(*con, "delta_insert_rows", "t") == 200000);  // whole delta stays per-query

  // Turning it back on resumes absorption on the next (projecting) read.
  run_ok("SET enable_delta_promotion = true;");
  rows_match(*con, "SELECT count(*), sum(k) FROM t;");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") > 0);
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccPromotionFixture,
                 "promotion: re-pin after promotion replaces the entry cleanly",
                 "[integration][gpu_execution][pin_table_mvcc_promotion]")
{
  run_ok("CALL unpin_table('t');");
  run_ok("SET enable_delta_promotion = true;");
  run_ok("SET threads TO 1;");
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  run_ok("INSERT INTO t SELECT (10000+range)::INTEGER FROM range(200000);");
  rows_match(*con, "SELECT count(*), sum(k) FROM t;");  // projecting read promotes
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") > 0);

  // Fold the delta into the disk image and re-pin: the grown entry's chunk
  // shape no longer matches the fresh materialization, so the re-pin replaces
  // (rather than failing) — counters reset to a fresh pin.
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") == 0);
  REQUIRE(tvf_u64(*con, "base_rows", "t") == 210000);  // fresh whole-table image
  REQUIRE(tvf_u64(*con, "delta_insert_rows", "t") == 0);
  rows_match(*con, "SELECT count(*), sum(k), max(k) FROM t;");
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(PinMvccPromotionFixture,
                 "promotion: host-tier delta promotes into host chunks",
                 "[integration][gpu_execution][pin_table_mvcc_promotion]")
{
  run_ok("CALL unpin_table('t');");
  run_ok("SET enable_delta_promotion = true;");
  run_ok("SET threads TO 1;");
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, (range*2)::BIGINT AS v FROM range(10000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='host');");
  run_ok(
    "INSERT INTO t SELECT (10000+range)::INTEGER, ((10000+range)*2)::BIGINT FROM range(200000);");

  rows_match(*con, "SELECT count(*), sum(k), max(v) FROM t;");
  REQUIRE(tvf_scalar(*con, "tier", "t") == "host");
  REQUIRE(tvf_u64(*con, "promoted_rows", "t") > 0);
  rows_match(*con, "SELECT k, v FROM t WHERE k IN (42, 10000, 150000) ORDER BY k;");
  run_ok("CALL unpin_table('t');");
}
