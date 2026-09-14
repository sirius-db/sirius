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

// A recreated table may reuse a qualified name but has a new catalog object id.
// These tests ensure an old pin is neither served nor merged into the new table.
// Fresh GPU reads remain blocked until a checkpoint replaces the on-disk image.

#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <set>
#include <string>

using PinRecreateFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Expect a plan-time GPU decline and results matching a CPU run.
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

/// Copy the cached column names for @p name.
std::set<std::string> cached_column_names(duckdb::Connection& con, const std::string& name)
{
  auto sirius_ctx = sirius::test::get_registered_sirius_context(con);
  REQUIRE(sirius_ctx != nullptr);
  std::set<std::string> names;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&](std::string_view entry_name, const sirius::scan_manager::pinned_entry& entry) {
      if (entry_name != name) { return true; }  // keep scanning
      auto const& cached = entry.cache_info.column_names();
      names.insert(cached.begin(), cached.end());
      return false;  // stop
    });
  return names;
}

bool entry_exists(duckdb::Connection& con, const std::string& name)
{
  auto sirius_ctx = sirius::test::get_registered_sirius_context(con);
  REQUIRE(sirius_ctx != nullptr);
  bool found = false;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&](std::string_view entry_name, const sirius::scan_manager::pinned_entry&) {
      if (entry_name != name) { return true; }  // keep scanning
      found = true;
      return false;  // stop
    });
  return found;
}

}  // namespace

TEST_CASE_METHOD(PinRecreateFixture,
                 "pin_table - a table recreated under a pinned name is never served the pin",
                 "[integration][gpu_execution][pin_table][pin_table_mvcc]")
{
  run_ok("CREATE TABLE recreate_t AS SELECT range AS a, range * 2 AS b FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='recreate_t', tier='gpu');");

  // Confirm the original table is served from the pin.
  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM recreate_t;");

  // Recreate the same shape with different values, without checkpointing.
  run_ok("DROP TABLE recreate_t;");
  run_ok(
    "CREATE TABLE recreate_t AS SELECT range + 1000000 AS a, range * 3 AS b FROM "
    "range(50000);");

  // The old pin remains registered but no longer matches.
  REQUIRE(entry_exists(*con, "recreate_t"));

  // Fall back instead of serving the old pin or stale disk image.
  expect_fallback_matches_cpu(*this, "SELECT sum(a), sum(b) FROM recreate_t;");

  // Re-pin the new incarnation.
  run_ok("CHECKPOINT;");
  run_ok("CALL unpin_table('recreate_t');");
  run_ok("CALL pin_table(format='duckdb', name='recreate_t', tier='gpu');");
  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM recreate_t;");

  run_ok("CALL unpin_table('recreate_t');");
}

TEST_CASE_METHOD(PinRecreateFixture,
                 "pin_table - a checkpointed recreate under a pinned name serves a fresh read",
                 "[integration][gpu_execution][pin_table][pin_table_mvcc]")
{
  run_ok("CREATE TABLE ckpt_recreate_t AS SELECT range AS a, range * 2 AS b FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='ckpt_recreate_t', tier='gpu');");
  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM ckpt_recreate_t;");

  // After checkpointing, the old pin misses and a fresh GPU read is safe.
  run_ok("DROP TABLE ckpt_recreate_t;");
  run_ok(
    "CREATE TABLE ckpt_recreate_t AS SELECT range + 1000000 AS a, range * 3 AS b FROM "
    "range(50000);");
  run_ok("CHECKPOINT;");

  // Checkpointing makes the disk image safe; it does not change the pin identity.
  REQUIRE(entry_exists(*con, "ckpt_recreate_t"));

  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM ckpt_recreate_t;");

  run_ok("CALL unpin_table('ckpt_recreate_t');");
}

TEST_CASE_METHOD(PinRecreateFixture,
                 "pin_table - re-pinning a recreated table replaces the entry instead of merging",
                 "[integration][gpu_execution][pin_table][pin_table_mvcc]")
{
  // Pin only column 'a' of the first incarnation.
  run_ok("CREATE TABLE repin_t AS SELECT range AS a, range * 2 AS b FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='repin_t', tier='gpu', cols=['a']);");
  REQUIRE(cached_column_names(*con, "repin_t") == std::set<std::string>{"a"});

  // Recreate with the same row count and pin a different column under the same name.
  run_ok("DROP TABLE repin_t;");
  run_ok("CREATE TABLE repin_t AS SELECT range + 1000000 AS a, range * 3 AS b FROM range(50000);");
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='repin_t', tier='gpu', cols=['b']);");

  // The second pin replaces the first instead of merging columns.
  REQUIRE(cached_column_names(*con, "repin_t") == std::set<std::string>{"b"});

  compare_gpu_vs_cpu("SELECT sum(b) FROM repin_t;");

  run_ok("CALL unpin_table('repin_t');");
}
