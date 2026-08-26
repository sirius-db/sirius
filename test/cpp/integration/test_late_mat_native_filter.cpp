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
// Requires SIRIUS_EXP_LATE_MAT=1 in the ENVIRONMENT: the gate is read per process.

#include <catch.hpp>
#include <duckdb.hpp>
#include <late_mat/defer_directive.hpp>
#include <late_mat/defer_policy.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>
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

std::vector<std::string> rows_of(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::string> rows;
  for (duckdb::idx_t i = 0; i < result.RowCount(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row += result.GetValue(c, i).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace

TEST_CASE_METHOD(NativeFilterFixture,
                 "late-mat declines a filtered duckdb-native pin and still answers correctly",
                 "[integration][gpu_execution][late_mat][native_filter]")
{
  if (!sirius::late_mat::late_mat_enabled()) {
    WARN("SIRIUS_EXP_LATE_MAT unset; skipping the duckdb-native filtered-pin case");
    return;
  }

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
  // the control below could not tell a working plan from a rejected one. Read at pin
  // time, so it must be set before the pin.
  ::setenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", "c_custkey", 1);
  run_ok("CALL pin_table(format='duckdb', name='c', tier='gpu');");

  // Control. Without it a decline proves nothing: a query that never had a
  // deferral to install declines for free, and the case below would pass on a
  // plan the policy had already rejected on its own merits.
  auto const before_unfiltered = sirius::late_mat::deferrals_installed();
  auto gpu_unfiltered          = con->Query(query(false));
  REQUIRE(gpu_unfiltered);
  if (gpu_unfiltered->HasError()) { UNSCOPED_INFO(gpu_unfiltered->GetError()); }
  REQUIRE_FALSE(gpu_unfiltered->HasError());
  REQUIRE(rows_of(*gpu_unfiltered) == expected_unfiltered);
  REQUIRE(sirius::late_mat::deferrals_installed() > before_unfiltered);

  // The case under test: same plan, plus a filter the scan applies itself.
  auto const before_filtered = sirius::late_mat::deferrals_installed();
  auto gpu_filtered          = con->Query(query(true));
  REQUIRE(gpu_filtered);
  if (gpu_filtered->HasError()) { UNSCOPED_INFO(gpu_filtered->GetError()); }
  REQUIRE_FALSE(gpu_filtered->HasError());
  REQUIRE(rows_of(*gpu_filtered) == expected_filtered);
  REQUIRE(sirius::late_mat::deferrals_installed() == before_filtered);

  ::unsetenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS");
  run_ok("CALL unpin_table('c');");
}
