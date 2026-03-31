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

#include "util/sirius_plan_renderer.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// ─── Fixture: loads the Sirius extension into an in-memory DuckDB ───────────

class GPUExplainFixture {
 public:
  GPUExplainFixture()
  {
    // Save the current SIRIUS_CONFIG_FILE so we can restore it on destruction.
    const char* prev = std::getenv("SIRIUS_CONFIG_FILE");
    if (prev) {
      had_prev_config_ = true;
      prev_config_     = prev;
    }

    // Set config file so the Sirius extension initializes the GPU context.
    // Without this, join plan generation fails because it accesses SiriusContext.
    auto cfg_path = fs::path(__FILE__).parent_path() / "integration.cfg";
    if (fs::exists(cfg_path)) { setenv("SIRIUS_CONFIG_FILE", cfg_path.string().c_str(), 1); }

    db  = std::make_unique<duckdb::DuckDB>(nullptr);
    con = std::make_unique<duckdb::Connection>(*db);

    // Create test tables
    con->Query("CREATE TABLE t1 (a INTEGER, b VARCHAR, c DOUBLE)");
    con->Query("INSERT INTO t1 VALUES (1, 'hello', 1.5), (2, 'world', 2.5)");
    con->Query("CREATE TABLE t2 (x INTEGER, y VARCHAR)");
    con->Query("INSERT INTO t2 VALUES (1, 'alpha'), (3, 'beta')");
  }

  ~GPUExplainFixture()
  {
    con.reset();
    db.reset();
    if (had_prev_config_) {
      setenv("SIRIUS_CONFIG_FILE", prev_config_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

 private:
  bool had_prev_config_ = false;
  std::string prev_config_;
};

// ─── Tests ──────────────────────────────────────────────────────────────────

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain returns two columns with correct schema",
                 "[gpu_explain]")
{
  auto result = con->Query("CALL gpu_explain('SELECT a, b FROM t1')");
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("Error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());

  // Check column names
  REQUIRE(result->ColumnCount() == 2);
  REQUIRE(result->names[0] == "explain_key");
  REQUIRE(result->names[1] == "explain_value");

  // Should have 2 rows: logical plan + physical plan
  REQUIRE(result->RowCount() == 2);

  auto key0 = result->GetValue(0, 0).ToString();
  auto key1 = result->GetValue(0, 1).ToString();
  REQUIRE(key0 == "duckdb_logical_plan");
  REQUIRE(key1 == "sirius_physical_plan");
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain shows physical plan with TABLE_SCAN",
                 "[gpu_explain]")
{
  // DuckDB may push the filter into the scan, so check for TABLE_SCAN
  auto result = con->Query("CALL gpu_explain('SELECT a FROM t1 WHERE a > 1')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto plan_text = result->GetValue(1, 1).ToString();
  REQUIRE(plan_text.find("TABLE_SCAN") != std::string::npos);
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain shows physical plan for join query",
                 "[gpu_explain][gpu]")
{
  // This test requires the SiriusContext to be initialized (needs GPU), because the
  // hash join plan generator accesses operator_params from the Sirius config.
  auto result = con->Query("CALL gpu_explain('SELECT t1.a, t2.y FROM t1 JOIN t2 ON t1.a = t2.x')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto key1      = result->GetValue(0, 1).ToString();
  auto plan_text = result->GetValue(1, 1).ToString();
  UNSCOPED_INFO("key1: " << key1);
  UNSCOPED_INFO("plan_text: " << plan_text);
  REQUIRE(plan_text.find("HASH_JOIN") != std::string::npos);
  // Should show join type
  REQUIRE(plan_text.find("INNER") != std::string::npos);
  // Should show both table scans
  auto first_scan  = plan_text.find("TABLE_SCAN");
  auto second_scan = plan_text.find("TABLE_SCAN", first_scan + 1);
  REQUIRE(first_scan != std::string::npos);
  REQUIRE(second_scan != std::string::npos);
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain shows physical plan for aggregate query",
                 "[gpu_explain]")
{
  auto result = con->Query("CALL gpu_explain('SELECT a, SUM(c) FROM t1 GROUP BY a')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto plan_text = result->GetValue(1, 1).ToString();
  REQUIRE(plan_text.find("HASH_GROUP_BY") != std::string::npos);
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain reports error for unsupported query (WINDOW)",
                 "[gpu_explain]")
{
  auto result = con->Query("CALL gpu_explain('SELECT a, ROW_NUMBER() OVER (ORDER BY a) FROM t1')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  // Should have an error row instead of physical plan
  auto key1 = result->GetValue(0, 1).ToString();
  REQUIRE(key1 == "error");

  auto error_text = result->GetValue(1, 1).ToString();
  REQUIRE(error_text.find("not supported") != std::string::npos);
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain includes estimated cardinality in output",
                 "[gpu_explain]")
{
  auto result = con->Query("CALL gpu_explain('SELECT * FROM t1')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto plan_text = result->GetValue(1, 1).ToString();
  REQUIRE(plan_text.find("est.") != std::string::npos);
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain tree has connectors for multi-level plan",
                 "[gpu_explain]")
{
  // GROUP BY produces HASH_GROUP_BY -> TABLE_SCAN, which has a tree connector
  auto result = con->Query("CALL gpu_explain('SELECT a, SUM(c) FROM t1 GROUP BY a')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto plan_text = result->GetValue(1, 1).ToString();
  REQUIRE(plan_text.find("\u2514\u2500\u2500") != std::string::npos);
}

TEST_CASE_METHOD(GPUExplainFixture, "gpu_explain rejects NULL parameter", "[gpu_explain]")
{
  auto result = con->Query("CALL gpu_explain(NULL)");
  REQUIRE(result);
  REQUIRE(result->HasError());
}

TEST_CASE_METHOD(GPUExplainFixture,
                 "gpu_explain tree shows proper indentation for nested operators",
                 "[gpu_explain][gpu]")
{
  // Requires GPU context for join plan generation
  auto result = con->Query("CALL gpu_explain('SELECT t1.a, t2.y FROM t1 JOIN t2 ON t1.a = t2.x')");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto plan_text = result->GetValue(1, 1).ToString();
  // Tree connectors should be present
  REQUIRE((plan_text.find("\u251C\u2500\u2500") != std::string::npos ||
           plan_text.find("\u2514\u2500\u2500") != std::string::npos));
}
