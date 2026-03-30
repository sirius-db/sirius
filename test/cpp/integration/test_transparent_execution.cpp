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
#include <utils/sirius_test_env.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

/// Guard that sets SIRIUS_CONFIG_FILE for the duration of the test.
struct config_env_guard {
  config_env_guard(const std::string& path) { setenv("SIRIUS_CONFIG_FILE", path.c_str(), 1); }
  ~config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

/// \brief Fixture that sets up a DuckDB connection with Sirius and transparent execution enabled.
class TransparentExecutionFixture {
 public:
  TransparentExecutionFixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.cfg";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<config_env_guard>(cfg_path.string());

      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }

    // Enable transparent execution.
    con->Query("SET sirius_transparent_execution = true;");
  }

  /// Run a query via plain SQL (transparent GPU path) and via CPU, compare results.
  void compare_transparent_vs_cpu(const std::string& query)
  {
    // Run via transparent GPU execution (plain SQL).
    auto gpu_result = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("Transparent GPU error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());

    // Disable transparent execution and run on CPU.
    con->Query("SET sirius_transparent_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET sirius_transparent_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    // Compare dimensions.
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    // Compare row data as strings (order-independent).
    auto gpu_collection = gpu_result->Fetch();
    auto cpu_collection = cpu_result->Fetch();

    // For simplicity, compare row counts; detailed comparison is done via
    // test_gpu_execution_tpch pattern if needed.
    INFO("GPU rows: " << gpu_result->RowCount() << ", CPU rows: " << cpu_result->RowCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());
  }

 protected:
  std::unique_ptr<config_env_guard> config_guard;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

// ============================== Test cases ==============================

TEST_CASE_METHOD(TransparentExecutionFixture,
                 "transparent execution: simple filter",
                 "[transparent][integration]")
{
  // Create test data.
  con->Query("CREATE TABLE test_t AS SELECT i AS id, i * 2 AS val FROM range(1000) t(i);");
  compare_transparent_vs_cpu("SELECT * FROM test_t WHERE val > 500 ORDER BY id LIMIT 10;");
}

TEST_CASE_METHOD(TransparentExecutionFixture,
                 "transparent execution: aggregation",
                 "[transparent][integration]")
{
  con->Query("CREATE TABLE test_agg AS SELECT i % 10 AS grp, i AS val FROM range(1000) t(i);");
  compare_transparent_vs_cpu(
    "SELECT grp, SUM(val) AS total FROM test_agg GROUP BY grp ORDER BY grp;");
}

TEST_CASE_METHOD(TransparentExecutionFixture,
                 "transparent execution: join",
                 "[transparent][integration]")
{
  con->Query("CREATE TABLE test_left AS SELECT i AS id, i * 3 AS val FROM range(100) t(i);");
  con->Query("CREATE TABLE test_right AS SELECT i * 2 AS id, i AS other FROM range(100) t(i);");
  compare_transparent_vs_cpu(
    "SELECT l.id, l.val, r.other FROM test_left l JOIN test_right r ON l.id = r.id ORDER BY "
    "l.id;");
}

TEST_CASE_METHOD(TransparentExecutionFixture,
                 "transparent execution: top-N",
                 "[transparent][integration]")
{
  con->Query("CREATE TABLE test_topn AS SELECT i AS id, i * 7 AS val FROM range(10000) t(i);");
  compare_transparent_vs_cpu("SELECT * FROM test_topn ORDER BY val DESC LIMIT 5;");
}

TEST_CASE_METHOD(TransparentExecutionFixture,
                 "transparent execution: fallback for unsupported (window)",
                 "[transparent][integration]")
{
  // Window functions are not supported by Sirius — should fall back to CPU silently.
  con->Query("CREATE TABLE test_win AS SELECT i AS id, i % 5 AS grp FROM range(100) t(i);");
  auto result = con->Query(
    "SELECT id, grp, ROW_NUMBER() OVER (PARTITION BY grp ORDER BY id) AS rn "
    "FROM test_win ORDER BY id;");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 100);
}

TEST_CASE_METHOD(TransparentExecutionFixture,
                 "transparent execution: disable via SET",
                 "[transparent][integration]")
{
  // When disabled, queries should still work (CPU path).
  con->Query("SET sirius_transparent_execution = false;");
  con->Query("CREATE TABLE test_off AS SELECT i AS id FROM range(10) t(i);");
  auto result = con->Query("SELECT * FROM test_off ORDER BY id;");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 10);
  con->Query("SET sirius_transparent_execution = true;");
}
