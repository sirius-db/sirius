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

// Aggregates with a FILTER clause on the GPU path. Before the planner rewrite these ran with the
// predicate hoisted into a projection and then ignored, so count(*) FILTER (WHERE c) returned the
// unfiltered count with no error.

#include "duckdb/main/config.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <system_error>

namespace {

void require_ok(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

std::string sql_string_literal(const std::string& value)
{
  std::string escaped = "'";
  for (char c : value) {
    if (c == '\'') { escaped += '\''; }
    escaped += c;
  }
  return escaped + "'";
}

class AggregateFilterFixture {
 public:
  AggregateFilterFixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    if (!sirius::test::g_integration_env->is_active()) {
      sirius::test::g_integration_env->resume();
    }
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    require_ok(*con, "SET enable_duckdb_fallback = false");
    // The shared database's optimizer mask is DB-global and an earlier test can leave it
    // cleared; DuckDB's compressed materialization then rewrites these small-range columns into
    // shapes the GPU plan does not translate. Pin the mask Sirius publishes at load for this
    // fixture, the way the pipeline-conversion fixtures do, and restore it afterwards.
    auto& disabled = duckdb::DBConfig::GetConfig(*con->context).options.disabled_optimizers;
    original_disabled_optimizers = disabled;
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    // The shared environment's database is in-memory and the GPU scan reads a database
    // file's block manager, so the table is written to a file by a plain DuckDB instance
    // (Sirius is disabled for secondary instances) and attached read-only, as the TPC-H
    // fixtures attach their prebuilt file.
    db_path = std::filesystem::temp_directory_path() / "sirius_aggregate_filter_test.db";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + ".wal", ec);
    {
      duckdb::DuckDB file_db(db_path.string());
      duckdb::Connection writer(file_db);
      require_ok(writer,
                 "CREATE TABLE af AS "
                 "SELECT CAST(i % 13 AS INTEGER) AS g, "
                 "       CASE WHEN i % 11 = 0 THEN NULL ELSE CAST(i AS BIGINT) END AS v, "
                 "       CAST(i % 5 AS DOUBLE) AS d "
                 "FROM range(50000) AS t(i)");
      require_ok(writer, "CHECKPOINT");
    }
    require_ok(*con, "ATTACH '" + db_path.string() + "' AS afdb (READ_ONLY)");
    require_ok(*con, "USE afdb");
  }

  ~AggregateFilterFixture()
  {
    if (con) {
      duckdb::DBConfig::GetConfig(*con->context).options.disabled_optimizers =
        original_disabled_optimizers;
      con->Query("USE memory");
      con->Query("DETACH afdb");
    }
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + ".wal", ec);
  }

  // The same statement through transparent GPU execution (asserted to have run on the GPU via
  // the execution counters) and through DuckDB, compared value for value.
  void compare(const std::string& query)
  {
    require_ok(*con, "SET gpu_execution = true");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu    = con->Query(query);
    REQUIRE(gpu);
    if (gpu->HasError()) { UNSCOPED_INFO(gpu->GetError()); }
    REQUIRE_FALSE(gpu->HasError());
    auto after = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

    require_ok(*con, "SET gpu_execution = false");
    auto cpu = con->Query(query);
    require_ok(*con, "SET gpu_execution = true");
    REQUIRE(cpu);
    REQUIRE_FALSE(cpu->HasError());
    REQUIRE(gpu->RowCount() == cpu->RowCount());
    REQUIRE(gpu->ColumnCount() == cpu->ColumnCount());
    for (duckdb::idx_t r = 0; r < cpu->RowCount(); ++r) {
      for (duckdb::idx_t c = 0; c < cpu->ColumnCount(); ++c) {
        UNSCOPED_INFO("row " << r << " col " << c);
        REQUIRE(gpu->GetValue(c, r).ToString() == cpu->GetValue(c, r).ToString());
      }
    }
  }

  std::unique_ptr<duckdb::Connection> con;
  std::filesystem::path db_path;
  std::set<duckdb::OptimizerType> original_disabled_optimizers;
};

}  // namespace

TEST_CASE_METHOD(AggregateFilterFixture,
                 "aggregate FILTER - ungrouped count(*)/count/sum/min/max/avg",
                 "[integration][gpu_execution][aggregate][filter]")
{
  compare(
    "SELECT count(*) FILTER (WHERE v > 25000) AS c_star, "
    "       count(v) FILTER (WHERE g = 3) AS c_v, "
    "       sum(v) FILTER (WHERE g IN (1, 2)) AS s, "
    "       min(v) FILTER (WHERE g = 7) AS mn, "
    "       max(v) FILTER (WHERE g = 7) AS mx, "
    "       avg(d) FILTER (WHERE v IS NOT NULL) AS a "
    "FROM af");
}

TEST_CASE_METHOD(AggregateFilterFixture,
                 "aggregate FILTER - grouped, mixed with unfiltered aggregates",
                 "[integration][gpu_execution][aggregate][filter]")
{
  compare(
    "SELECT g, count(*) AS n, count(*) FILTER (WHERE v > 25000) AS n_hi, "
    "       sum(v) FILTER (WHERE d > 2) AS s_d, sum(v) AS s "
    "FROM af GROUP BY g ORDER BY g");
}

TEST_CASE_METHOD(AggregateFilterFixture,
                 "aggregate FILTER - predicate that matches no row and one that matches all",
                 "[integration][gpu_execution][aggregate][filter]")
{
  // Predicates statistics cannot fold: a filter the optimizer proves constant becomes a
  // constant CASE, and two of those in one projection hit a pre-existing translation gap
  // that has nothing to do with FILTER (the same query with explicit CASE fails the same way).
  compare(
    "SELECT count(*) FILTER (WHERE v % 7 = 99) AS none, sum(v) FILTER (WHERE v % 7 = 99) AS "
    "s_none, "
    "       count(*) FILTER (WHERE g % 1 = 0) AS all_rows FROM af");
}

TEST_CASE_METHOD(AggregateFilterFixture,
                 "avg over a nullable column divides by the non-NULL count",
                 "[integration][gpu_execution][aggregate][filter]")
{
  // v is NULL on every 11th row. Before the fix the ungrouped avg divided by the row count.
  compare("SELECT avg(v) AS a, avg(v) FILTER (WHERE g < 6) AS af, avg(d) AS ad FROM af");
  compare("SELECT g, avg(v) AS a, avg(v) FILTER (WHERE d > 1) AS af FROM af GROUP BY g ORDER BY g");
}

TEST_CASE_METHOD(AggregateFilterFixture,
                 "aggregate FILTER - unsupported aggregate is declined, not silently unfiltered",
                 "[integration][gpu_execution][aggregate][filter]")
{
  // string_agg keeps every input; there is no NULL-skipping rewrite for it, so with fallback off
  // the statement must fail rather than return the unfiltered aggregate.
  require_ok(*con, "SET gpu_execution = true");
  auto gpu = con->Query("SELECT string_agg(CAST(g AS VARCHAR), ',') FILTER (WHERE g < 2) FROM af");
  REQUIRE(gpu);
  REQUIRE(gpu->HasError());
}
