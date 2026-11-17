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

// A deferral over a FILTERED scan rebuilds its pin-order rowid from the surviving
// row positions, which only an ingestible that reports them can supply. The
// duckdb-native one filters with a plain select and does not, so installation must
// refuse the shape rather than let the scan reach substitution with a compacted
// batch and no record of which rows lived.
//
// Uses the default-on enable_late_materialization query policy.

#include <catch.hpp>
#include <late_mat/defer_directive.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/pinned_entry_census.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace {

using NativeFilterFixture = sirius::test::GpuExecutionFixture;

constexpr std::int64_t kCustomers = 20'000;
constexpr std::int64_t kOrders    = 60'000;
constexpr std::int64_t kLines     = 120'000;

/// Same shape as the parquet end-to-end case: payload columns wide enough to clear
/// the value floor, read only by the aggregate at the far end, and two joins so the
/// ride clears the four-crossing floor.
std::string query(bool filtered)
{
  return std::string{
           "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, count(*) AS n "
           "FROM c JOIN o ON c.c_custkey = o.o_custkey "
           "JOIN l ON o.o_orderkey = l.l_orderkey "} +
         (filtered ? "WHERE c.c_custkey % 3 = 0 " : "") +
         "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment "
         "ORDER BY c.c_custkey";
}

std::vector<std::string> rows_of(duckdb::QueryResult& result)
{
  std::vector<std::string> rows;
  auto result_rows = result.Collection().GetRows();
  for (duckdb::idx_t i = 0; i < result_rows.size(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row += result_rows.GetValue(c, i).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

// Row equality alone cannot distinguish successful GPU execution from CPU replay.
auto query_on_gpu(duckdb::Connection& con, std::string const& sql)
{
  auto const before = sirius::test::get_transparent_execution_stats(con);
  auto result       = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  auto const after = sirius::test::get_transparent_execution_stats(con);
  REQUIRE(after.executions == before.executions + 1);
  REQUIRE(after.fallbacks == before.fallbacks);
  REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks);
  return result;
}

}  // namespace

TEST_CASE_METHOD(NativeFilterFixture,
                 "late-mat declines a filtered duckdb-native pin and still answers correctly",
                 "[integration][gpu_execution][late_mat][native_filter]")
{
  run_ok(
    "CREATE TABLE c AS SELECT range AS c_custkey, "
    "'Customer#' || lpad(CAST(range AS VARCHAR), 12, '0') AS c_name, "
    "'address-' || repeat(CAST(range % 97 AS VARCHAR), 6) AS c_address, "
    "'comment for customer ' || CAST(range AS VARCHAR) AS c_comment "
    "FROM range(" +
    std::to_string(kCustomers) + ");");
  run_ok("CREATE TABLE o AS SELECT range AS o_orderkey, range % " + std::to_string(kCustomers) +
         " AS o_custkey FROM range(" + std::to_string(kOrders) + ");");
  run_ok("CREATE TABLE l AS SELECT range % " + std::to_string(kOrders) +
         " AS l_orderkey FROM range(" + std::to_string(kLines) + ");");
  run_ok("CHECKPOINT;");

  // The CPU answers, taken before anything is pinned.
  run_ok("SET gpu_execution = false;");
  auto cpu_unfiltered = con->Query(query(false));
  REQUIRE(cpu_unfiltered);
  REQUIRE_FALSE(cpu_unfiltered->HasError());
  auto const expected_unfiltered = rows_of(*cpu_unfiltered);
  auto cpu_filtered              = con->Query(query(true));
  REQUIRE(cpu_filtered);
  REQUIRE_FALSE(cpu_filtered->HasError());
  auto const expected_filtered = rows_of(*cpu_filtered);
  REQUIRE_FALSE(expected_filtered.empty());
  REQUIRE(expected_filtered.size() < expected_unfiltered.size());
  run_ok("SET gpu_execution = true;");

  // c_custkey rides real as the join and group key, so proving it distinct over the
  // pin is what moves the port past the aggregate. Without it both arms decline for
  // an unrelated reason — the join's fan-out is still uncollapsed at the port — and
  // the control below could not tell a working plan from a rejected one. The pin
  // proves it, because late materialization is on.
  // This control tests the filtered-scan refusal, independently of compressed-origin refusal.
  run_ok("CALL pin_table(format='duckdb', name='c', tier='gpu', compression => false);");

  // Control. Without it a decline proves nothing: a query that never had a
  // deferral to install declines for free, and the case below would pass on a
  // plan the policy had already rejected on its own merits.
  auto const before_unfiltered = sirius::late_mat::deferrals_installed();
  auto gpu_unfiltered          = query_on_gpu(*con, query(false));
  REQUIRE(gpu_unfiltered);
  if (gpu_unfiltered->HasError()) { UNSCOPED_INFO(gpu_unfiltered->GetError()); }
  REQUIRE_FALSE(gpu_unfiltered->HasError());
  REQUIRE(rows_of(*gpu_unfiltered) == expected_unfiltered);
  REQUIRE(sirius::late_mat::deferrals_installed() > before_unfiltered);

  // The connection-local threshold changes the next plan without repinning the table.
  run_ok("SET late_materialization_min_boundaries = 1000000;");
  auto const before_threshold = sirius::late_mat::deferrals_installed();
  auto threshold_result       = query_on_gpu(*con, query(false));
  REQUIRE(threshold_result);
  REQUIRE_FALSE(threshold_result->HasError());
  REQUIRE(rows_of(*threshold_result) == expected_unfiltered);
  REQUIRE(sirius::late_mat::deferrals_installed() == before_threshold);
  run_ok("RESET late_materialization_min_boundaries;");
  auto restored_result = query_on_gpu(*con, query(false));
  REQUIRE(restored_result);
  REQUIRE_FALSE(restored_result->HasError());
  REQUIRE(rows_of(*restored_result) == expected_unfiltered);
  REQUIRE(sirius::late_mat::deferrals_installed() > before_threshold);

  // The case under test: same plan, plus a filter the scan applies itself.
  auto const before_filtered = sirius::late_mat::deferrals_installed();
  auto gpu_filtered          = query_on_gpu(*con, query(true));
  REQUIRE(gpu_filtered);
  if (gpu_filtered->HasError()) { UNSCOPED_INFO(gpu_filtered->GetError()); }
  REQUIRE_FALSE(gpu_filtered->HasError());
  REQUIRE(rows_of(*gpu_filtered) == expected_filtered);
  REQUIRE(sirius::late_mat::deferrals_installed() == before_filtered);

  run_ok("CALL unpin_table('c');");
}

TEST_CASE_METHOD(NativeFilterFixture,
                 "SQL count deferral changes an existing pin and refuses nullable values",
                 "[integration][gpu_execution][late_mat][optimization_settings]")
{
  bool const nullable = GENERATE(false, true);
  CAPTURE(nullable);
  run_ok("CREATE TABLE counts AS SELECT i AS k, " +
         std::string(nullable ? "CASE WHEN i % 3 = 0 THEN NULL ELSE repeat('x', 80) END"
                              : "repeat('x', 80)") +
         " AS payload FROM range(12000) t(i);");
  run_ok("CREATE TABLE keys AS SELECT i AS k FROM range(12000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SET enable_duckdb_fallback = false;");
  // Preserve COUNT(payload): null-free statistics would otherwise rewrite it to COUNT(*).
  run_ok("SET disabled_optimizers = 'statistics_propagation';");
  run_ok("SET late_materialization_min_boundaries = 0;");
  run_ok("SET late_materialization_min_value_x_boundaries = 0;");
  run_ok("CALL pin_table(format='duckdb', name='counts', tier='gpu');");
  for (bool const enabled : {false, true, false}) {
    CAPTURE(enabled);
    run_ok(std::string("SET late_materialization_count_defer = ") + (enabled ? "true;" : "false;"));
    sirius::test::scoped_recording_log_sink logs{"info"};
    auto const before = sirius::late_mat::deferrals_installed();
    auto result =
      query_on_gpu(*con,
                   "SELECT k.k % 4 AS g, count(c.payload) FROM counts c JOIN keys k ON c.k = k.k "
                   "GROUP BY g ORDER BY g;");
    REQUIRE(result->RowCount() == 4);
    for (duckdb::idx_t g = 0; g < 4; ++g) {
      REQUIRE(result->Collection().GetValue(0, g).GetValue<int64_t>() == g);
      REQUIRE(result->Collection().GetValue(1, g).GetValue<int64_t>() == (nullable ? 2000 : 3000));
    }
    auto const records = logs.records();
    for (auto const& r : records) {
      UNSCOPED_INFO(r.message);
    }
    bool const count_deferred = std::any_of(records.begin(), records.end(), [](auto const& r) {
      return r.message.find("column(s) with NO far end") != std::string::npos;
    });
    REQUIRE(count_deferred == (enabled && !nullable));
    REQUIRE(sirius::late_mat::deferrals_installed() == before + (enabled && !nullable ? 1 : 0));
  }
  run_ok("CALL unpin_table('counts');");
}

TEST_CASE_METHOD(NativeFilterFixture,
                 "SQL fused scan settings reach decoding on an existing compressed pin",
                 "[integration][gpu_execution][fused_scan_filter][optimization_settings]")
{
  sirius::test::scratch_dir scratch{"fused_scan_settings"};
  {
    std::ofstream plan(scratch.path() / "fused_settings.txt");
    REQUIRE(plan.good());
    for (int i = 0; i < 2; ++i) {
      plan << (i == 0 ? "" : "---\n")
           << "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
    }
  }
  run_ok(
    "CREATE TABLE fused_settings AS SELECT (i % 100)::INTEGER AS k, "
    "i::INTEGER AS payload FROM range(10000) t(i);");
  run_ok("CHECKPOINT;");
  auto const parquet = sirius::test::sql_literal((scratch.path() / "input.parquet").string());
  run_ok("SET gpu_execution = false;");
  run_ok("COPY fused_settings TO " + parquet + " (FORMAT PARQUET);");
  // Parquet exposes its static range analysis to the compressed decoder; native scans do not.
  std::string const sql =
    "SELECT payload FROM read_parquet(" + parquet + ") WHERE k >= 0 AND k < 6 ORDER BY payload;";
  auto cpu = con->Query(sql);
  REQUIRE(cpu);
  REQUIRE_FALSE(cpu->HasError());
  auto const expected = rows_of(*cpu);
  REQUIRE(expected.size() == 600);
  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");
  run_ok("SET enable_late_materialization = false;");
  run_ok("SET pin_table_compression = true;");
  run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
  run_ok("SET pin_table_compression_max_compressed_fraction = 1000;");
  run_ok("SET pin_table_input_compression_plan_dir = " +
         sirius::test::sql_literal(scratch.path().string()) + ";");
  run_ok("CALL pin_table(" + parquet + ", name='fused_settings', tier='gpu');");
  auto const census = sirius::test::census_entry(*con, "fused_settings");
  REQUIRE(census.chunks > 0);
  REQUIRE(census.compressed_chunks == census.chunks);
  run_ok("SET fused_scan_diagnostics = true;");

  struct policy_case {
    bool enabled;
    double cap;
  };
  for (auto const [enabled, cap] : {policy_case{false, 0.10},
                                    {true, 0.10},
                                    {false, 0.10},
                                    {true, 0.01},
                                    {true, 0.10},
                                    {true, 0.01}}) {
    CAPTURE(enabled, cap);
    run_ok(std::string("SET enable_fused_scan_filter = ") + (enabled ? "true;" : "false;"));
    run_ok("SET fused_scan_max_selectivity = " + std::to_string(cap) + ";");
    sirius::test::scoped_recording_log_sink logs{"info"};
    auto result = query_on_gpu(*con, sql);
    REQUIRE(rows_of(*result) == expected);
    auto const records = logs.records();
    for (auto const& r : records) {
      UNSCOPED_INFO(r.message);
    }
    auto contains = [&](char const* text) {
      return std::any_of(records.begin(), records.end(), [&](auto const& r) {
        return r.message.find(text) != std::string::npos;
      });
    };
    REQUIRE(contains("[decompression-pushdown] decode APPLIED") == (enabled && cap > 0.06));
    if (enabled && cap < 0.06) { REQUIRE(contains("selection_unprofitable=true")); }
  }
  run_ok("CALL unpin_table('fused_settings');");
}
