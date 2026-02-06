/*
 * Copyright 2025, Sirius Contributors.
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

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static fs::path get_project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT);
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

static fs::path get_tpch_db_path()
{
  auto db_path = get_project_root() / "tpch.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

/**
 * @brief RAII guard to set SIRIUS_CONFIG_FILE env var for the duration of a test.
 *
 * Points to the integration.cfg config which sets up GPU/host memory spaces
 * and the pipeline executor configuration needed by gpu_execution.
 */
struct config_env_guard {
  config_env_guard()
  {
    auto cfg_path = fs::path(__FILE__).parent_path() / "integration.cfg";
    REQUIRE(fs::exists(cfg_path));
    setenv("SIRIUS_CONFIG_FILE", cfg_path.string().c_str(), 1);
  }
  ~config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

/**
 * @brief Run a query through gpu_execution and return the result.
 */
static duckdb::unique_ptr<duckdb::MaterializedQueryResult> run_gpu_execution(
  duckdb::Connection& con, const std::string& query)
{
  auto sql    = "CALL gpu_execution('" + query + "')";
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("gpu_execution error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return result;
}

//===----------------------------------------------------------------------===//
// Scan tests - basic table reads
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - scan single column from nation", "[integration][gpu_execution][scan]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey from nation;");

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 1);
}

TEST_CASE("gpu_execution - scan multiple integer columns from nation",
          "[integration][gpu_execution][scan]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey, n_regionkey from nation;");

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 2);
}

TEST_CASE("gpu_execution - scan from region table", "[integration][gpu_execution][scan]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select r_regionkey from region;");

  REQUIRE(result->RowCount() == 5);
  REQUIRE(result->ColumnCount() == 1);
}

//===----------------------------------------------------------------------===//
// Projection tests - computed columns / expressions
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - projection with arithmetic expression",
          "[integration][gpu_execution][projection]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey + n_regionkey as total from nation;");

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 1);
}

TEST_CASE("gpu_execution - projection with multiply", "[integration][gpu_execution][projection]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result =
    run_gpu_execution(con, "select n_nationkey * 2 as doubled, n_regionkey from nation;");

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 2);
}

//===----------------------------------------------------------------------===//
// Filter tests - WHERE clause predicates
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - filter equality", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey from nation where n_regionkey = 1;");

  REQUIRE(result->ColumnCount() == 1);
  // TPCH has 5 nations in region 1: ARGENTINA, BRAZIL, CANADA, PERU, UNITED STATES
  REQUIRE(result->RowCount() == 5);
}

TEST_CASE("gpu_execution - filter greater than", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey from nation where n_regionkey > 2;");

  REQUIRE(result->ColumnCount() == 1);
  REQUIRE(result->RowCount() > 0);
}

TEST_CASE("gpu_execution - filter not equal", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select r_regionkey from region where r_regionkey != 3;");

  REQUIRE(result->ColumnCount() == 1);
  REQUIRE(result->RowCount() == 4);
}

TEST_CASE("gpu_execution - filter with projection", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result =
    run_gpu_execution(con, "select n_nationkey, n_regionkey from nation where n_regionkey = 0;");

  REQUIRE(result->ColumnCount() == 2);
  // TPCH has 5 nations in region 0: ALGERIA, ETHIOPIA, KENYA, MOROCCO, MOZAMBIQUE
  REQUIRE(result->RowCount() == 5);
}

// Pre-existing issue: empty result set causes "Port default not found in operator RESULT_COLLECTOR"
TEST_CASE("gpu_execution - filter returns empty result", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey from nation where n_regionkey = 99;");

  REQUIRE(result->ColumnCount() == 1);
  REQUIRE(result->RowCount() == 0);
}

//===----------------------------------------------------------------------===//
// Multi-pipeline queries (pre-existing hang issue - hidden by default)
// These are hidden with [.] tag and only run when explicitly requested.
//===----------------------------------------------------------------------===//

// Pre-existing issue: multi-pipeline queries hang because completion is not
// signaled across pipeline boundaries. Affects GROUP BY, ORDER BY, JOINs, etc.

TEST_CASE("gpu_execution - aggregation count by regionkey",
          "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result =
    run_gpu_execution(con, "select n_regionkey, count(*) from nation group by n_regionkey;");

  REQUIRE(result->RowCount() == 5);
  REQUIRE(result->ColumnCount() == 2);
}

TEST_CASE("gpu_execution - order by", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(con, "select n_nationkey from nation order by n_regionkey;");

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 1);
}

TEST_CASE("gpu_execution - top n (order by + limit)", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result =
    run_gpu_execution(con, "select n_nationkey from nation order by n_regionkey desc limit 5;");

  REQUIRE(result->RowCount() == 5);
  REQUIRE(result->ColumnCount() == 1);
}

TEST_CASE("gpu_execution - join nation and region", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  auto result = run_gpu_execution(
    con,
    "select n.n_nationkey, r.r_regionkey from nation n join region r on n.n_regionkey = "
    "r.r_regionkey;");

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 2);
}
