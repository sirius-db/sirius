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

// A pinned duckdb table that is DROPped and recreated under the same qualified
// name is a DIFFERENT table, and the pin still holds the dropped one's rows.
// Nothing about the name, the column layout or the chunk shape distinguishes the
// two incarnations — only DuckDB's per-catalog-entry object id does — so the pin
// cache keys on that id as well.
//
// Two ways the old identity leaked stale rows, both covered here:
//   * serving — a scan of the recreated table hit the pin and returned the
//     dropped table's values;
//   * re-pinning — a second pin_table under the same pin name took the
//     same-row-count MERGE path and built one entry out of two unrelated tables.
//
// A scan of the recreated table must not fall through to the disk-native read
// either: that path is MVCC-blind and the pin's checkpoint suppression is still
// in force, so it declines at plan time into a clean transparent CPU fallback
// ({0 rebinds, 1 fallback, 0 executions}) with correct results.

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

/// Run @p query under gpu_execution=true expecting the plan-time guard to decline
/// (transparent CPU fallback: {0 rebinds, 1 fallback, 0 executions}) and the
/// results to match a plain CPU run.
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

/// Cached column names of the pinned entry named @p name, copied out inside the
/// visitor so no reference escapes. Empty when no such entry exists — which the
/// callers distinguish from "exists but caches nothing" via @ref entry_exists.
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

  // Baseline: the pin serves this scan on the GPU ({1 rebind, 0 fallbacks, 1
  // execution}), so the assertion below is about identity and not about the pin
  // having failed to take effect.
  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM recreate_t;");

  // Same qualified name, same schema, same ROW COUNT — so every chunk-shape and
  // column-layout check the cache applies still passes — but different values.
  //
  // Deliberately NO checkpoint here. A checkpoint would bump the database's
  // checkpoint generation and trip the pre-existing staleness guard in
  // prepare_for_query, which masks this bug behind a runtime error. Without one,
  // nothing but the table identity stands between the scan and the dropped table's
  // cached rows — pre-fix, the GPU answered sum(a) over range(50000) while DuckDB
  // answered over range(50000) + 1000000, silently and with no error.
  run_ok("DROP TABLE recreate_t;");
  run_ok(
    "CREATE TABLE recreate_t AS SELECT range + 1000000 AS a, range * 3 AS b FROM "
    "range(50000);");

  // The pin is still registered: what changed is that it no longer matches this
  // table, not that it vanished.
  REQUIRE(entry_exists(*con, "recreate_t"));

  // The regression assertion — both halves matter. The counters prove the plan
  // declined instead of serving (or falling through to the MVCC-blind disk read),
  // and the row comparison proves the values are the recreated table's.
  expect_fallback_matches_cpu(*this, "SELECT sum(a), sum(b) FROM recreate_t;");

  // Unpinning clears the superseded entry, and a fresh pin of the new incarnation
  // serves it on the GPU again.
  run_ok("CHECKPOINT;");
  run_ok("CALL unpin_table('recreate_t');");
  run_ok("CALL pin_table(format='duckdb', name='recreate_t', tier='gpu');");
  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM recreate_t;");

  run_ok("CALL unpin_table('recreate_t');");
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

  // Recreate with the SAME row count, then pin only column 'b' under the same pin
  // name. insert_pinned_entry's merge path keys on the row count and the chunk
  // boundaries, both of which match here — only the identity check keeps it from
  // fusing the old table's 'a' with the new table's 'b' into one entry.
  run_ok("DROP TABLE repin_t;");
  run_ok("CREATE TABLE repin_t AS SELECT range + 1000000 AS a, range * 3 AS b FROM range(50000);");
  // pin_table refuses uncheckpointed rows, so this one is required.
  run_ok("CHECKPOINT;");
  run_ok("CALL pin_table(format='duckdb', name='repin_t', tier='gpu', cols=['b']);");

  // Replaced, not merged: the entry caches the second pin's columns only.
  REQUIRE(cached_column_names(*con, "repin_t") == std::set<std::string>{"b"});

  // And it serves the new incarnation on the GPU with the new values.
  compare_gpu_vs_cpu("SELECT sum(b) FROM repin_t;");

  run_ok("CALL unpin_table('repin_t');");
}
