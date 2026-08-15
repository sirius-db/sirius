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

// The sirius_dynamic_filter_stats() table function is the SQL surface external runners use to
// assert dynamic-filter arming (test/tpch_performance/performance_test.py --mode ab). These tests
// pin its contract: the row set is exactly the dynamic_filter_stats_snapshot field set, the values
// match an in-process snapshot, and movement caused by a query is visible through SQL.

#include "op/dynamic_filter/dynamic_filter_stats.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <thread>

namespace {

using stats_map = std::map<std::string, std::uint64_t>;

stats_map to_map(const sirius::op::dynamic_filter_stats_snapshot& snapshot)
{
  stats_map fields;
  sirius::op::for_each_field(
    snapshot, [&](const char* name, std::uint64_t value) { fields.emplace(name, value); });
  return fields;
}

stats_map query_stats_rows(duckdb::Connection& con)
{
  auto result = con.Query("SELECT name, value FROM sirius_dynamic_filter_stats();");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  stats_map fields;
  auto& materialized = result->Cast<duckdb::MaterializedQueryResult>();
  for (duckdb::idx_t r = 0; r < materialized.RowCount(); ++r) {
    auto const inserted = fields.emplace(materialized.GetValue(0, r).ToString(),
                                         materialized.GetValue(1, r).GetValue<std::uint64_t>());
    REQUIRE(inserted.second);  // duplicate field names would fold silently in the map
  }
  return fields;
}

// Two in-process snapshots bracketing the SQL read, retried until they agree so a straggling
// delivery-time increment between the reads cannot fail the equality this test is about.
struct bracketed_read {
  stats_map snapshot;
  stats_map sql_rows;
};

bracketed_read read_stats_both_ways(duckdb::Connection& con)
{
  for (int attempt = 0; attempt < 5; ++attempt) {
    auto const before = to_map(sirius::test::get_dynamic_filter_stats_snapshot(con));
    auto sql_rows     = query_stats_rows(con);
    auto const after  = to_map(sirius::test::get_dynamic_filter_stats_snapshot(con));
    if (before == after) { return {after, std::move(sql_rows)}; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  FAIL("dynamic-filter counters kept moving with no query in flight");
  return {};
}

// Parquet copy of a deterministic Top-N-eligible table, removed on destruction. The parquet scan
// is the consumer path whose producer eligibility the movement test needs; a duckdb-native table
// would work too, but parquet avoids the file-backed-database scaffolding.
struct scoped_stats_parquet {
  explicit scoped_stats_parquet(duckdb::Connection& con)
    : path(std::filesystem::temp_directory_path() /
           ("sirius_dynamic_filter_stats_fn." + std::to_string(::getpid()) + ".parquet"))
  {
    auto rows = con.Query(
      "COPY (SELECT CAST(i AS INTEGER) AS id, "
      "             CAST((i * 37) % 10007 AS INTEGER) AS v, "
      "             CAST((i * 11) % 97 AS INTEGER) AS w "
      "      FROM range(10000) t(i)) TO '" +
      path.string() + "' (FORMAT PARQUET);");
    REQUIRE(rows);
    REQUIRE_FALSE(rows->HasError());
  }
  ~scoped_stats_parquet()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  scoped_stats_parquet(const scoped_stats_parquet&)            = delete;
  scoped_stats_parquet& operator=(const scoped_stats_parquet&) = delete;

  [[nodiscard]] std::string scan() const { return "read_parquet('" + path.string() + "')"; }

  std::filesystem::path path;
};

}  // namespace

TEST_CASE("sirius_dynamic_filter_stats() mirrors the in-process snapshot",
          "[integration][gpu_execution][dynamic_filter_stats]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  SECTION("row set is exactly the snapshot field set with matching values")
  {
    auto const read = read_stats_both_ways(con);
    REQUIRE(read.sql_rows.size() == sirius::op::dynamic_filter_stats_field_count);
    REQUIRE(read.sql_rows == read.snapshot);
  }

  SECTION("counter movement from a query is visible through the SQL surface")
  {
    scoped_stats_parquet parquet(con);
    sirius::test::scoped_setting flag(con, "enable_top_n_dynamic_filter", 1);
    auto gpu_on = con.Query("SET gpu_execution = true;");
    REQUIRE(gpu_on);
    REQUIRE_FALSE(gpu_on->HasError());

    auto const before = read_stats_both_ways(con);
    REQUIRE(before.sql_rows == before.snapshot);

    auto rows = con.Query("SELECT id, v, w FROM " + parquet.scan() + " ORDER BY v, w LIMIT 10;");
    REQUIRE(rows);
    REQUIRE_FALSE(rows->HasError());
    REQUIRE(rows->Cast<duckdb::MaterializedQueryResult>().RowCount() == 10);

    auto const after = read_stats_both_ways(con);
    REQUIRE(after.sql_rows == after.snapshot);
    // The eligible Top-N producer is a plan-time fact, so the query must have moved it, and the
    // movement must read back identically through SQL (checked via the map equalities above).
    REQUIRE(after.sql_rows.at("top_n_producers_eligible") >
            before.sql_rows.at("top_n_producers_eligible"));
  }
}
