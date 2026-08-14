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

/**
 * @file test_gpu_execution_reader_pruning_gate.cpp
 * @brief End-to-end WI-0b: the reader pruning gate over one clustered parquet file (main doc,
 *        "The siting rule is necessary but not sufficient: the reader path needs a runtime
 *        gate").
 *
 * The same file carries both motivating shapes. Queried ascending, the Top-N boundary excludes
 * every later row group, so the gate observes pruning and stays on -- the -57% winner must not
 * be taxed. Queried descending, the boundary always trails the scan frontier and prunes nothing,
 * so the gate disables and later splits skip the merge -- the measured +8.4% adversary stops
 * paying. A pinned rerun pins the no-evidence rule (no reader runs, so zero SAMPLES, never "zero
 * pruning"), and a flag-off rerun pins that a channel-less scan never consults the gate.
 * Publication races split arrival by design, so awaited counters use the documented bounded
 * retry loop and direction asserts; every section also checks GPU/CPU result equivalence.
 */

#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

using rows_t = std::vector<std::vector<std::string>>;
using sirius::op::dynamic_filter_stats_snapshot;

// 1M rows x 2 int64 columns at 65536 rows per row group -> 16 row groups of ~1 MiB decoded
// each; 1 MiB scan batches then give one split per row group, so a query spans enough splits
// for the gate to both learn (4 barren samples) and act (later splits skip).
constexpr std::int64_t kRows         = 1'000'000;
constexpr std::int64_t kRowGroupRows = 65'536;
constexpr std::uint64_t kBatchBytes  = 1ull << 20;
constexpr std::int64_t kMinRowGroups = 16;

// Enable the Top-N flag for one scope and restore the previous setting.
struct top_n_flag_guard {
  explicit top_n_flag_guard(duckdb::Connection& c, bool enabled) : con(c)
  {
    original = sirius::test::get_registered_sirius_context(c)
                 ->get_config()
                 .get_operator_params()
                 .enable_top_n_dynamic_filter;
    con.Query(std::string{"SET enable_top_n_dynamic_filter = "} + (enabled ? "true" : "false") +
              ";");
  }
  ~top_n_flag_guard()
  {
    con.Query(std::string{"SET enable_top_n_dynamic_filter = "} + (original ? "true" : "false") +
              ";");
  }

  top_n_flag_guard(const top_n_flag_guard&)            = delete;
  top_n_flag_guard& operator=(const top_n_flag_guard&) = delete;

  duckdb::Connection& con;
  bool original = false;
};

//! Shrink scan batches for one scope so the query spans many splits.
struct scan_batch_size_guard {
  explicit scan_batch_size_guard(duckdb::Connection& c, std::uint64_t bytes) : con(c)
  {
    con.Query("SET scan_task_batch_size = " + std::to_string(bytes) + ";");
  }
  ~scan_batch_size_guard() { con.Query("RESET scan_task_batch_size"); }

  scan_batch_size_guard(const scan_batch_size_guard&)            = delete;
  scan_batch_size_guard& operator=(const scan_batch_size_guard&) = delete;

  duckdb::Connection& con;
};

//! Best-effort unpin on scope exit, so a failing assertion cannot leak a pinned entry into the
//! shared environment (destructor results are not asserted -- Catch2 macros throw).
struct scoped_pin {
  scoped_pin(duckdb::Connection& c, std::string const& path, std::string name, char const* tier)
    : con(c), entry_name(std::move(name))
  {
    auto pin =
      con.Query("CALL pin_table('" + path + "', tier='" + tier + "', name='" + entry_name + "');");
    REQUIRE(pin);
    if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
    REQUIRE_FALSE(pin->HasError());
  }
  ~scoped_pin() { con.Query("CALL unpin_table('" + entry_name + "');"); }

  scoped_pin(const scoped_pin&)            = delete;
  scoped_pin& operator=(const scoped_pin&) = delete;

  duckdb::Connection& con;
  std::string entry_name;
};

rows_t query_rows(duckdb::Connection& con, const std::string& query)
{
  auto result = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
}

//! Result equivalence across the three legs the gate must not disturb: GPU with the Top-N flag
//! on, GPU with it off, and the CPU baseline.
void require_flag_equivalence(duckdb::Connection& con, const std::string& query)
{
  con.Query("SET gpu_execution = true;");
  rows_t flag_on;
  {
    top_n_flag_guard flag(con, true);
    flag_on = query_rows(con, query);
  }
  rows_t flag_off;
  {
    top_n_flag_guard flag(con, false);
    flag_off = query_rows(con, query);
  }
  con.Query("SET gpu_execution = false;");
  auto const cpu = query_rows(con, query);
  con.Query("SET gpu_execution = true;");

  REQUIRE(flag_on == flag_off);
  REQUIRE(flag_on == cpu);
}

//! All six reader-gate counters flat between two snapshots.
void require_reader_gate_flat(dynamic_filter_stats_snapshot const& before,
                              dynamic_filter_stats_snapshot const& after)
{
  REQUIRE(after.reader_gate_row_groups_considered == before.reader_gate_row_groups_considered);
  REQUIRE(after.reader_gate_row_groups_pruned == before.reader_gate_row_groups_pruned);
  REQUIRE(after.reader_gate_measurements == before.reader_gate_measurements);
  REQUIRE(after.reader_gate_disabled == before.reader_gate_disabled);
  REQUIRE(after.reader_gate_rearmed == before.reader_gate_rearmed);
  REQUIRE(after.reader_gate_merges_skipped == before.reader_gate_merges_skipped);
}

}  // namespace

TEST_CASE("gpu_execution - reader pruning gate on a fresh-read parquet scan",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  con.Query("SET gpu_execution = true;");

  auto tmp = fs::temp_directory_path() / ("sirius-reader-gate-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  auto const parquet_path = (tmp / "gate_facts.parquet").string();
  std::string const entry = "gate_facts_" + std::to_string(::getpid());

  // Clustered ascending on `v`, so an ascending Top-N boundary excludes every later row group
  // while a descending one always trails the scan frontier and can exclude nothing.
  {
    auto r = con.Query("COPY (SELECT range AS id, range AS v FROM range(" + std::to_string(kRows) +
                       ") ORDER BY v) TO '" + parquet_path + "' (FORMAT PARQUET, ROW_GROUP_SIZE " +
                       std::to_string(kRowGroupRows) + ");");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }

  // Premise: the file must span enough row groups (= splits under the small batch size) for the
  // gate to learn within one execution, or every counter assertion below would be vacuous.
  {
    auto r = con.Query("SELECT count(DISTINCT row_group_id) FROM parquet_metadata('" +
                       parquet_path + "');");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    auto const n_row_groups = r->GetValue(0, 0).GetValue<std::int64_t>();
    UNSCOPED_INFO("fixture parquet has " << n_row_groups << " row groups");
    REQUIRE(n_row_groups >= kMinRowGroups);
  }

  scan_batch_size_guard batches(con, kBatchBytes);

  std::string const ascending_query =
    "SELECT id, v FROM read_parquet('" + parquet_path + "') ORDER BY v LIMIT 10";
  std::string const descending_query =
    "SELECT id, v FROM read_parquet('" + parquet_path + "') ORDER BY v DESC LIMIT 10";

  SECTION("clustered ascending keeps the gate on")
  {
    require_flag_equivalence(con, ascending_query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    // Publication races split arrival, so pruning may need more than one execution to land
    // inside a split's merged read; retry within a small bound rather than asserting one run's
    // schedule. The gate is per-execution state, so the disable counter staying flat across the
    // whole loop is the real assertion: the winning shape never disables.
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    for (int runs = 0; runs < 5; ++runs) {
      auto const rows = query_rows(con, ascending_query);
      REQUIRE(rows.size() == 10);
      auto const now = sirius::test::get_dynamic_filter_stats_snapshot(con);
      if (now.reader_gate_row_groups_pruned > before.reader_gate_row_groups_pruned) { break; }
    }
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.reader_gate_row_groups_considered > before.reader_gate_row_groups_considered);
    REQUIRE(after.reader_gate_row_groups_pruned > before.reader_gate_row_groups_pruned);
    REQUIRE(after.reader_gate_measurements > before.reader_gate_measurements);
    REQUIRE(after.reader_gate_disabled == before.reader_gate_disabled);
  }

  SECTION("the adversary disables the merge")
  {
    require_flag_equivalence(con, descending_query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    // Await both halves of the behavior: the disable decision, and a later split of some
    // execution actually skipping its merge. A disable that lands on an execution's last merged
    // split leaves no split behind it to skip, so the skip may need a further run.
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    for (int runs = 0; runs < 5; ++runs) {
      auto const rows = query_rows(con, descending_query);
      REQUIRE(rows.size() == 10);
      auto const now = sirius::test::get_dynamic_filter_stats_snapshot(con);
      if (now.reader_gate_disabled > before.reader_gate_disabled &&
          now.reader_gate_merges_skipped > before.reader_gate_merges_skipped) {
        break;
      }
    }
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.reader_gate_disabled - before.reader_gate_disabled >= 1);
    REQUIRE(after.reader_gate_merges_skipped > before.reader_gate_merges_skipped);
    // The boundary always trails the frontier on this shape: nothing was ever prunable.
    REQUIRE(after.reader_gate_row_groups_pruned == before.reader_gate_row_groups_pruned);
  }

  SECTION("a pinned serve is zero samples, not zero pruning")
  {
    scoped_pin pin(con, parquet_path, entry, "gpu");
    require_flag_equivalence(con, ascending_query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    // No reader runs on a pinned serve, so the reader-gate counters are deterministically flat;
    // the boundary is consumed post-decode instead (the Phase-1 flip), whose delivery-time
    // counter needs the bounded retry loop.
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    for (int runs = 0; runs < 5; ++runs) {
      auto const rows = query_rows(con, ascending_query);
      REQUIRE(rows.size() == 10);
      auto const now = sirius::test::get_dynamic_filter_stats_snapshot(con);
      if (now.post_decode_apply_rows_in > before.post_decode_apply_rows_in) { break; }
    }
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.post_decode_apply_rows_in > before.post_decode_apply_rows_in);
    require_reader_gate_flat(before, after);
  }

  SECTION("flag off moves nothing")
  {
    // No producer -> the wrapper elides the channel, the scan has no filters, and the gate is
    // never consulted: the feature-off plan shape stays free, deterministically.
    top_n_flag_guard flag(con, false);
    con.Query("SET gpu_execution = true;");
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const asc    = query_rows(con, ascending_query);
    auto const desc   = query_rows(con, descending_query);
    auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(asc.size() == 10);
    REQUIRE(desc.size() == 10);
    require_reader_gate_flat(before, after);

    con.Query("SET gpu_execution = false;");
    auto const asc_cpu  = query_rows(con, ascending_query);
    auto const desc_cpu = query_rows(con, descending_query);
    con.Query("SET gpu_execution = true;");
    REQUIRE(asc == asc_cpu);
    REQUIRE(desc == desc_cpu);
  }

  fs::remove_all(tmp, ec);
}
