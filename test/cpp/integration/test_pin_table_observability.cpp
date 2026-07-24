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

// End-to-end sirius_pinned_tables() introspection surface and the schema-drift
// guard it exposes. Uses a row-only GPU-vs-CPU check (the drift cases
// intentionally fall back to CPU, which the fixture's compare_gpu_vs_cpu
// forbids) and unpins 't' defensively (shared scan manager across the suite).
// GpuExecutionFixture, real GPU.

#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdint>
#include <string>

using PinObservabilityFixture = sirius::test::GpuExecutionFixture;

namespace {

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

std::uint64_t tvf_count(duckdb::Connection& con)
{
  auto r = con.Query("SELECT count(*) FROM sirius_pinned_tables();");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  return r->GetValue(0, 0).GetValue<std::uint64_t>();
}

/// GPU result equals the CPU result, tolerating a transparent CPU fallback (no
/// counter assertions), so drifted-pin queries that decline to DuckDB still
/// verify correctness.
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

TEST_CASE_METHOD(PinObservabilityFixture,
                 "observability: sirius_pinned_tables tracks the pin lifecycle",
                 "[integration][gpu_execution][pin_table_observability]")
{
  run_ok("CALL unpin_table('t');");
  REQUIRE(tvf_count(*con) == 0);  // nothing pinned

  run_ok("SET threads TO 1;");
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, (range*2)::BIGINT AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");

  REQUIRE(tvf_count(*con) == 1);
  REQUIRE(tvf_scalar(*con, "format", "t") == "duckdb");
  REQUIRE(tvf_scalar(*con, "tier", "t") == "gpu");
  REQUIRE(tvf_scalar(*con, "table_name", "t") == "t");
  REQUIRE(tvf_scalar(*con, "column_count", "t") == "2");
  REQUIRE(tvf_scalar(*con, "base_rows", "t") == "50000");
  REQUIRE(tvf_scalar(*con, "is_valid", "t") == "true");
  REQUIRE(tvf_scalar(*con, "stale", "t") == "false");
  REQUIRE(tvf_scalar(*con, "delta_insert_rows", "t") == "0");
  REQUIRE(tvf_scalar(*con, "delta_delete_rows", "t") == "0");

  run_ok("INSERT INTO t SELECT (50000+range)::INTEGER, 0::BIGINT FROM range(300);");
  run_ok("DELETE FROM t WHERE k IN (1, 2, 3, 4);");
  REQUIRE(tvf_scalar(*con, "delta_insert_rows", "t") == "300");
  REQUIRE(tvf_scalar(*con, "delta_delete_rows", "t") == "4");

  run_ok("CALL unpin_table('t');");
  REQUIRE(tvf_count(*con) == 0);
}

TEST_CASE_METHOD(PinObservabilityFixture,
                 "observability: a schema-altered pin goes stale and falls back to CPU",
                 "[integration][gpu_execution][pin_table_observability]")
{
  run_ok("CALL unpin_table('t');");
  run_ok("SET threads TO 1;");
  run_ok("CREATE TABLE t AS SELECT range::INTEGER AS k, (range*2)::BIGINT AS v FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  REQUIRE(tvf_scalar(*con, "stale", "t") == "false");
  rows_match(*con, "SELECT count(*), sum(k) FROM t;");  // served from the cache

  SECTION("ADD COLUMN replaces the DataTable")
  {
    run_ok("ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 7;");
    REQUIRE(tvf_scalar(*con, "stale", "t") == "true");
    // The drift guard declines the pin; the query falls back to DuckDB and stays correct.
    rows_match(*con, "SELECT count(*), sum(k), sum(w) FROM t;");
  }

  SECTION("DROP COLUMN replaces the DataTable")
  {
    run_ok("ALTER TABLE t DROP COLUMN v;");
    REQUIRE(tvf_scalar(*con, "stale", "t") == "true");
    rows_match(*con, "SELECT count(*), sum(k) FROM t;");
  }

  SECTION("RENAME COLUMN keeps the DataTable and keeps serving")
  {
    run_ok("ALTER TABLE t RENAME COLUMN k TO id;");
    // A rename reuses the DataTable, so the positional cache still serves.
    REQUIRE(tvf_scalar(*con, "stale", "t") == "false");
    rows_match(*con, "SELECT count(*), sum(id) FROM t;");
  }

  run_ok("CALL unpin_table('t');");
}
