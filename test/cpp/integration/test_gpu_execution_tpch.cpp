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

#include "op/sirius_physical_partition.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/main/config.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/tpch_queries.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

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
  const char* env = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH");
  auto db_path =
    env ? fs::path(env) : fs::path(__FILE__).parent_path() / "data/duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

struct sirius_config_env_guard {
  sirius_config_env_guard(const std::string& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }

  ~sirius_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

class GPUExecutionFixtureBase {
 public:
  GPUExecutionFixtureBase()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      // Use the shared DuckDB instance managed by the test listener
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      // Fallback: create an isolated DuckDB (e.g. when running a single test directly)
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard>(cfg_path.string());

      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }
  }

  virtual ~GPUExecutionFixtureBase() { release_env(); }

  /**
   * @brief Subclass hook — called after each env swap to re-establish views /
   * attach databases on the new connection. Default no-op; DuckDB-fixture
   * attaches the tpch database, Parquet-fixture creates views over read_parquet.
   */
  virtual void setup_schema() {}

  /**
   * @brief Bind the fixture's connection to the shared env for the given
   * num_gpus configuration. Pauses the previously-active env (if any) first
   * so at most one Sirius context is live. Returns false if the requested
   * env is unavailable on this host (e.g., num_gpus=2 on a single-GPU host);
   * caller should WARN+return per Catch2 v2 convention.
   */
  bool bind_env(int num_gpus)
  {
    release_env();
    auto* env = sirius::test::acquire_integration_env_for(num_gpus);
    if (env == nullptr) { return false; }
    if (!env->is_active()) { env->resume(); }
    active_env_ = env;
    con         = std::make_unique<duckdb::Connection>(env->make_connection());
    setup_schema();
    return true;
  }

  /**
   * @brief Pauses the currently-bound env (if any) and drops the connection.
   * Safe to call multiple times. Called automatically from the destructor.
   */
  void release_env()
  {
    con.reset();
    if (active_env_ != nullptr) {
      active_env_->pause();
      active_env_ = nullptr;
    }
  }

  /**
   * @brief Runs compare_gpu_vs_cpu on the chosen num_gpus config. Returns false
   * if the 2-GPU path is unavailable (single-GPU host) — caller should WARN+return.
   */
  bool compare_gpu_vs_cpu_for(int num_gpus,
                              const std::string& query,
                              std::optional<float> float_tolerance = std::nullopt)
  {
    if (!bind_env(num_gpus)) { return false; }
    compare_gpu_vs_cpu(query, float_tolerance);
    return true;
  }

  /**
   * @brief Returns SIRIUS_TEST_SF10_PATH env var value, or empty if unset.
   * TEST-04 SF10 smoke TEST_CASEs gate on this — caller WARN+returns when empty.
   */
  static std::string sf10_path()
  {
    const char* p = std::getenv("SIRIUS_TEST_SF10_PATH");
    return p ? std::string{p} : std::string{};
  }

  /**
   * @brief Create views over the 8 TPC-H parquet tables at SIRIUS_TEST_SF10_PATH
   * on the current connection. Must be called AFTER bind_env() so the views
   * are attached to the newly-bound connection. Uses CREATE OR REPLACE VIEW
   * so it can re-run after a schema-owning subclass setup_schema() also ran.
   */
  void attach_sf10_tables()
  {
    auto base = sf10_path();
    REQUIRE_FALSE(base.empty());
    static const char* kTables[] = {
      "lineitem", "orders", "customer", "nation", "region", "part", "partsupp", "supplier"};
    for (auto* t : kTables) {
      auto r =
        con->Query("CREATE OR REPLACE VIEW " + std::string{t} + " AS SELECT * FROM read_parquet('" +
                   base + "/" + std::string{t} + ".parquet');");
      REQUIRE(r);
      REQUIRE_FALSE(r->HasError());
    }
  }

  /**
   * @brief bind_env + attach_sf10_tables + compare_gpu_vs_cpu. Returns false
   * if the requested env is unavailable. Caller should WARN+return on false.
   */
  bool compare_gpu_vs_cpu_sf10_for(int num_gpus,
                                   const std::string& query,
                                   std::optional<float> float_tolerance = std::nullopt)
  {
    if (!bind_env(num_gpus)) { return false; }
    attach_sf10_tables();
    compare_gpu_vs_cpu(query, float_tolerance);
    return true;
  }

  /**
   * @brief Run a query via transparent GPU execution and via DuckDB CPU, then compare results.
   *
   * Transparent execution is enabled by default when SiriusContext is initialized.
   * The CPU baseline is obtained by temporarily disabling transparent execution.
   *
   * Values are compared as strings via Value::ToString() which normalizes type differences
   * (e.g., HUGEINT vs BIGINT both render "50"). Row order is ignored by collecting rows
   * as sorted sets of string tuples.
   */
  static bool is_floating_point(duckdb::LogicalTypeId id)
  {
    return id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
  }

  /// Collect all rows from a MaterializedQueryResult as sorted vectors of stringified values.
  static std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
  {
    std::vector<std::vector<std::string>> rows;
    for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
      std::vector<std::string> row;
      row.reserve(result.ColumnCount());
      for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
        row.push_back(result.GetValue(c, r).ToString());
      }
      rows.push_back(std::move(row));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
  {
    // Enable transparent GPU execution
    con->Query("SET gpu_execution = true;");
    auto before_gpu_stats    = sirius::test::get_transparent_execution_stats(*con);
    auto before_filter_stats = sirius::test::get_dynamic_filter_stats_snapshot(*con);

    // Run on GPU (transparent — plain SQL goes through Sirius optimizer hook)
    auto gpu_result = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("transparent GPU execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());
    auto after_gpu_stats = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before_gpu_stats, after_gpu_stats, 1, 0, 1);

    // Domain evidence must remain an upper bound for every build shape exercised by this suite.
    auto after_filter_stats = sirius::test::get_dynamic_filter_stats_snapshot(*con);
    REQUIRE(after_filter_stats.keys_build_exceeded_domain ==
            before_filter_stats.keys_build_exceeded_domain);

    // Run on CPU (disable transparent execution)
    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());
    auto after_cpu_stats = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(after_gpu_stats, after_cpu_stats, 0, 0, 0);

    // Compare dimensions
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    if (gpu_result->RowCount() > 50000) {
      std::cout << "WARNING: Integration result num rows is: " << gpu_result->RowCount()
                << ". Please consider modifying test to make it smaller and run faster."
                << std::endl;
    }

    // Build a per-column flag for which columns are floating-point.
    std::vector<bool> col_is_float(gpu_result->ColumnCount());
    for (duckdb::idx_t c = 0; c < gpu_result->ColumnCount(); c++) {
      col_is_float[c] = is_floating_point(gpu_result->types[c].id());
    }

    // Collect and sort rows from already-materialized results for deterministic comparison.
    // This avoids re-running the query (which could fail for wrapped subqueries).
    auto& gpu_mat = gpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto& cpu_mat = cpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto gpu_rows = collect_rows(gpu_mat);
    auto cpu_rows = collect_rows(cpu_mat);

    for (duckdb::idx_t r = 0; r < gpu_rows.size(); r++) {
      for (duckdb::idx_t c = 0; c < gpu_rows[r].size(); c++) {
        if (float_tolerance.has_value() && col_is_float[c]) {
          double gpu_d = std::stod(gpu_rows[r][c]);
          double cpu_d = std::stod(cpu_rows[r][c]);
          double diff  = std::fabs(gpu_d - cpu_d);
          if (diff > static_cast<double>(float_tolerance.value())) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " float mismatch: GPU=[" << gpu_d
                                 << "] CPU=[" << cpu_d << "] diff=" << diff
                                 << " tolerance=" << float_tolerance.value());
            REQUIRE(diff <= static_cast<double>(float_tolerance.value()));
          }
        } else {
          if (gpu_rows[r][c] != cpu_rows[r][c]) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_rows[r][c]
                                 << "] CPU=[" << cpu_rows[r][c] << "]");
          }
          REQUIRE(gpu_rows[r][c] == cpu_rows[r][c]);
        }
      }
    }
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_config_env_guard> config_guard;
  sirius::test::shared_test_env* active_env_ = nullptr;
};

/**
 * @brief Catch2 test fixture for GPU execution tests.
 *
 * Initializes a DuckDB instance with the integration.yaml config and provides
 * a compare_gpu_vs_cpu method for validating GPU execution against CPU results.
 */
class GPUExecutionDuckDBFixture : public GPUExecutionFixtureBase {
 public:
  GPUExecutionDuckDBFixture() { setup_schema(); }

  void setup_schema() override
  {
    auto db_path = get_tpch_db_path().string();
    auto result  = con->Query("ATTACH IF NOT EXISTS '" + db_path + "' AS tpch (READ_ONLY);");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("USE tpch;");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }

  // // Disabled: these tests scan native (DuckDB-storage) tables via seq_scan. The
  // // legacy duckdb_scan path was removed and GPU native scan has no IO backend
  // // wired in this harness ("missing io_ctx, io_obj"), so every query throws
  // // "Unsupported scan function: seq_scan" and poisons the shared integration
  // // DB. Shadow the comparison helpers to skip until native scan is supported.
  // void compare_gpu_vs_cpu(const std::string& /*query*/,
  //                         std::optional<float> /*float_tolerance*/ = std::nullopt)
  // {
  //   WARN("duckdb-native tpch scan skipped — legacy duckdb_scan path removed");
  // }

  // bool compare_gpu_vs_cpu_for(int /*num_gpus*/,
  //                             const std::string& /*query*/,
  //                             std::optional<float> /*float_tolerance*/ = std::nullopt)
  // {
  //   WARN("duckdb-native tpch scan skipped — legacy duckdb_scan path removed");
  //   return false;  // RUN_TPCH_MGPU returns out of the test on false
  // }
};

/**
 * @brief Catch2 test fixture for GPU execution tests.
 *
 * Initializes a DuckDB instance with the integration.yaml config and provides
 * a compare_gpu_vs_cpu method for validating GPU execution against CPU results.
 */
class GPUExecutionParquetFixture : public GPUExecutionFixtureBase {
 public:
  GPUExecutionParquetFixture() { setup_schema(); }

  void setup_schema() override
  {
    auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
    auto result = con->Query("CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM read_parquet('" +
                             parquet_dir.string() + "/nation.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS region AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/region.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/customer.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/orders.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS part AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/part.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/partsupp.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/supplier.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());

    result = con->Query("CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM read_parquet('" +
                        parquet_dir.string() + "/lineitem.parquet');");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
};

//===----------------------------------------------------------------------===//
// Scan tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - scan single column",
                 "[integration][gpu_execution][scan]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - scan single column parquet",
                 "[integration][gpu_execution][parquet][scan]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - scan multiple columns",
                 "[integration][gpu_execution][scan]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - scan multiple columns parquet",
                 "[integration][gpu_execution][parquet][scan]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - scan region table",
                 "[integration][gpu_execution][scan]")
{
  compare_gpu_vs_cpu("select r_regionkey from region;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - scan region table parquet",
                 "[integration][gpu_execution][parquet][scan]")
{
  compare_gpu_vs_cpu("select r_regionkey from region;");
}

//===----------------------------------------------------------------------===//
// Projection tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - projection add",
                 "[integration][gpu_execution][projection]")
{
  compare_gpu_vs_cpu("select n_nationkey + n_regionkey as total from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - projection add parquet",
                 "[integration][gpu_execution][parquet][projection]")
{
  compare_gpu_vs_cpu("select n_nationkey + n_regionkey as total from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - projection multiply",
                 "[integration][gpu_execution][projection]")
{
  compare_gpu_vs_cpu("select n_nationkey * 2 as doubled, n_regionkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - projection multiply parquet",
                 "[integration][gpu_execution][parquet][projection]")
{
  compare_gpu_vs_cpu("select n_nationkey * 2 as doubled, n_regionkey from nation;");
}

//===----------------------------------------------------------------------===//
// Filter tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - filter equality",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - filter equality parquet",
                 "[integration][gpu_execution][parquet][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - filter greater than",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey > 2;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - filter greater than parquet",
                 "[integration][gpu_execution][parquet][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey > 2;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - filter not equal",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select r_regionkey from region where r_regionkey != 3;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - filter not equal parquet",
                 "[integration][gpu_execution][parquet][filter]")
{
  compare_gpu_vs_cpu("select r_regionkey from region where r_regionkey != 3;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - filter with projection",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation where n_regionkey = 0;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - filter with projection parquet",
                 "[integration][gpu_execution][parquet][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation where n_regionkey = 0;");
}

//===----------------------------------------------------------------------===//
// Ungrouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped min max",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu("select min(n_regionkey), max(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped min max parquet",
                 "[integration][gpu_execution][parquet][aggregate]")
{
  compare_gpu_vs_cpu("select min(n_regionkey), max(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped min with filter",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu("select min(n_nationkey) from nation where n_regionkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped min with filter parquet",
                 "[integration][gpu_execution][parquet][aggregate]")
{
  compare_gpu_vs_cpu("select min(n_nationkey) from nation where n_regionkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped sum count",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu("select sum(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped sum count parquet",
                 "[integration][gpu_execution][parquet][aggregate]")
{
  compare_gpu_vs_cpu("select sum(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped all agg functions",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu(
    "select sum(n_regionkey), min(n_nationkey), max(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped all agg functions parquet",
                 "[integration][gpu_execution][parquet][aggregate]")
{
  compare_gpu_vs_cpu(
    "select sum(n_regionkey), min(n_nationkey), max(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped avg integer",
                 "[integration][gpu_execution][aggregate][avg]")
{
  compare_gpu_vs_cpu("select avg(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped avg integer parquet",
                 "[integration][gpu_execution][parquet][aggregate][avg]")
{
  compare_gpu_vs_cpu("select avg(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped avg decimal",
                 "[integration][gpu_execution][aggregate][avg]")
{
  compare_gpu_vs_cpu("select avg(l_quantity), avg(l_discount) from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped avg decimal parquet",
                 "[integration][gpu_execution][parquet][aggregate][avg]")
{
  compare_gpu_vs_cpu("select avg(l_quantity), avg(l_discount) from lineitem;");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - single group by key: min max, sum, count(*)",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, min(c_custkey), max(c_custkey), sum(c_custkey), count(*) "
    "from customer group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - single group by key: min max, sum, count(*) parquet",
                 "[integration][gpu_execution][parquet][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, min(c_custkey), max(c_custkey), sum(c_custkey), count(*) "
    "from customer group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - single group by key: min max, count string ",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, min(C_NAME), max(C_NAME), count(C_NAME) from customer "
    "group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - single group by key: min max, count string  parquet",
                 "[integration][gpu_execution][parquet][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, min(C_NAME), max(C_NAME), count(C_NAME) from customer "
    "group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - two group by key: min max, but not showing the group by keys",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select min(c_custkey), max(c_custkey) from customer group by c_nationkey, c_mktsegment;");
}

TEST_CASE_METHOD(
  GPUExecutionParquetFixture,
  "gpu_execution - two group by key: min max, but not showing the group by keys parquet",
  "[integration][gpu_execution][parquet][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select min(c_custkey), max(c_custkey) from customer group by c_nationkey, c_mktsegment;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - two group keys and noaggregations",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, c_mktsegment from customer group by c_mktsegment, c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - two group keys and noaggregations parquet",
                 "[integration][gpu_execution][parquet][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, c_mktsegment from customer group by c_mktsegment, c_nationkey;");
}

//===----------------------------------------------------------------------===//
// Limit tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - limit",
                 "[integration][gpu_execution][limit]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation limit 10;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - limit parquet",
                 "[integration][gpu_execution][parquet][limit]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation limit 10;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - limit with filter",
                 "[integration][gpu_execution][limit]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation where n_regionkey = 1 limit 3;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - limit with filter parquet",
                 "[integration][gpu_execution][parquet][limit]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation where n_regionkey = 1 limit 3;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - limit on large table",
                 "[integration][gpu_execution][limit][limit_multi_batch]")
{
  // lineitem has ~6K rows at SF-0.01, ensuring multiple batches.
  // A limit of 100 should produce exactly 100 rows regardless of batch count.
  compare_gpu_vs_cpu("select l_orderkey from lineitem limit 100");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - limit on large table parquet",
                 "[.][integration_disabled][gpu_execution][parquet][limit][limit_multi_batch]")
{
  // lineitem has ~6K rows at SF-0.01, ensuring multiple batches.
  // A limit of 100 should produce exactly 100 rows regardless of batch count.
  compare_gpu_vs_cpu("select l_orderkey from lineitem limit 100");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - limit with offset on large table",
                 "[integration][gpu_execution][limit][limit_multi_batch]")
{
  compare_gpu_vs_cpu("select l_orderkey, l_partkey from lineitem limit 50 offset 200;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - limit with offset on large table parquet",
                 "[.][integration_disabled][gpu_execution][parquet][limit][limit_multi_batch]")
{
  compare_gpu_vs_cpu("select l_orderkey, l_partkey from lineitem limit 50 offset 200;");
}

//===----------------------------------------------------------------------===//
// Join tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic inner join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic inner join 0 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic inner join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic inner join 1 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic inner join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic inner join 2 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic inner join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n join customer c on "
    "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic inner join 3 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n join customer c on "
    "n.n_nationkey = c.c_nationkey;");
}

// issue #329: expressions in hash-join equality conditions. The join key is materialized into a
// column below the join and partitioned on that column; the compare_gpu_vs_cpu delta assertion
// (1 GPU exec, 0 fallbacks) also proves these run on the GPU rather than falling back to CPU.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - join with expression key on build side",
                 "[integration][gpu_execution][join]")
{
  // The canonical issue #329 example: expression on the (small) nation side.
  compare_gpu_vs_cpu(
    "select n.n_nationkey, c.c_custkey from customer c "
    "join nation n on c.c_custkey = n.n_nationkey * 10;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - join with expression key on probe side",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, c.c_custkey from customer c "
    "join nation n on c.c_nationkey * 2 = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - join with expressions on both sides",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, c.c_custkey from customer c "
    "join nation n on c.c_nationkey * 10 = n.n_nationkey * 10;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed join with expression equality key and inequality",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, c.c_custkey from customer c "
    "join nation n on c.c_custkey = n.n_nationkey * 10 and c.c_custkey > n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 0 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 1 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 2 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 3 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 0 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 1 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 2 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left join 3 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 0 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 1 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 2 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 3 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 0 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 1 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 2 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right join 3 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped inner join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped inner join 0 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped inner join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped inner join 1 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped inner join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped inner join 2 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped inner join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c join nation n on "
    "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped inner join 3 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c join nation n on "
    "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 0 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 1 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 2 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 3 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 0 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 1 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 2 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped left join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped left join 3 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 0 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 1 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 2 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 3 parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 0 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 1 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 2 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped right join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped right join 3 making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic full outer join",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
    "on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic full outer join parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
    "on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic full outer join making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
    "on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic full outer join making nulls parquet",
                 "[integration][gpu_execution][parquet][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
    "on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left semi join",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left semi join parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic left semi join 2",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic left semi join 2 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right semi join",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right semi join parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic right semi join 2",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic right semi join 2 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic semi join 3",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey "
    "from nation n semi join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic semi join 3 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey "
    "from nation n semi join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic semi join 4",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey  from nation n "
    "semi join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic semi join 4 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey  from nation n "
    "semi join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic semi join 5",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_name from nation n semi join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic semi join 5 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_name from nation n semi join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

//===----------------------------------------------------------------------===//
// SEMI join in BUILD_PROBE mode: a large probe (orders, 150k) over a small build/filter subset
// (customer where c_nationkey < 3) drives the planner into BUILD_PROBE, where one
// cudf::filtered_join is built once on the right (filter) side and reused across the streamed left
// probe batches via semi_join.
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - semi join build_probe large probe parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select o.o_orderkey from orders o "
    "semi join (select c_custkey from customer where c_nationkey < 3) c "
    "on o.o_custkey = c.c_custkey;");
}

//===----------------------------------------------------------------------===//
// MARK join tests (issue #921: BUILD_PROBE mode for MARK join)
//
// `OR` combined with `IN (subquery)`, and `IN (subquery)` projected as a value,
// both lower to a HASH_JOIN with "Join Type: MARK" in DuckDB. A large probe
// (orders, 150k) over a small build/filter subquery (customer subset) drives the
// planner into BUILD_PROBE, where one cudf::filtered_join is built on the right
// (filter) side and reused across the streamed left probe batches.
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - mark join via OR + IN subquery parquet",
                 "[integration][gpu_execution][parquet][markjoin]")
{
  // OR forces the IN membership to be materialized as a MARK join rather than a
  // semi join; the customer subset is the small build side.
  compare_gpu_vs_cpu(
    "select count(*) as n from orders "
    "where o_orderkey < 0 "
    "   or o_custkey in (select c_custkey from customer where c_nationkey < 3);");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - mark join via projected IN subquery parquet",
                 "[integration][gpu_execution][parquet][markjoin]")
{
  // Projecting the IN result as a boolean value produces a MARK join; grouping on
  // the mark exercises both the matched (true) and unmatched (false) partitions.
  compare_gpu_vs_cpu(
    "select (o_custkey in (select c_custkey from customer where c_nationkey < 3)) as is_member, "
    "       count(*) as n "
    "from orders group by 1 order by 1;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic semi join misfit 0",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey  "
    "from nation n semi join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic semi join misfit 0 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey  "
    "from nation n semi join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - basic semi join mistit 1",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey  from nation n "
    "semi join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - basic semi join mistit 1 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey  from nation n "
    "semi join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped semi join 0",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c semi join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped semi join 0 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c semi join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped semi join 1",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "semi join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped semi join 1 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "semi join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped semi join misfit 0",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c semi join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped semi join misfit 0 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c semi join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - swapped semi join misfit 1",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "semi join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - swapped semi join misfit 1 parquet",
                 "[integration][gpu_execution][parquet][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "semi join nation n on n.n_nationkey = c.c_custkey;");
}

/*
Anti Join Tests
===============
Each test mirrors its semi join counterpart, replacing `semi join` with `anti join`.
All tests use `compare_gpu_vs_cpu` to validate GPU results against CPU execution.

left anti join
-  nation ANTI JOIN region on matching keys (n_regionkey = r_regionkey)

left anti join 2
- nation ANTI JOIN region on mismatched keys (n_nationkey = r_regionkey)

left anti join 3-4
- nation ANTI JOIN customer on n_nationkey = c_nationkey, varying selected columns (both keys,
non-key only, string column)

left anti join misfit 0-1
- customer ANTI JOIN nation, reversed table order with mismatched keys

 */

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - left anti join",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n anti join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - left anti join parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n anti join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - left anti join 2",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n anti join region r on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - left anti join 2 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n anti join region r on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - left anti join 3",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_name "
    "from customer c anti join nation n on c.c_nationkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - left anti join 3 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_name "
    "from customer c anti join nation n on c.c_nationkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - left anti join 4",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c anti join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - left anti join 4 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c anti join nation n on n.n_nationkey = c.c_nationkey;");
}

// ANTI join in BUILD_PROBE mode: a large probe (orders, 150k) over a small build/filter subset
// (customer where c_nationkey < 3) drives the planner into BUILD_PROBE, where one
// cudf::filtered_join is built once on the right (filter) side and reused across the streamed left
// probe batches via anti_join.
TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - anti join build_probe large probe parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  // count(*) keeps the materialized result small while the full orders probe still streams through
  // anti_join + gather across many batches, exercising the reused filtered_join.
  compare_gpu_vs_cpu(
    "select count(*) as n from orders o "
    "anti join (select c_custkey from customer where c_nationkey < 3) c "
    "on o.o_custkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - left anti join misfit 0",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c anti join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - left anti join misfit 0 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c anti join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - left anti join misfit 1",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "anti join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - left anti join misfit 1 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "anti join nation n on n.n_nationkey = c.c_custkey;");
}

/*
Right Anti Join Tests
=====================
DuckDB's optimizer promotes an anti join to RIGHT_ANTI when the smaller table
is on the left. These tests place the smaller table (region/nation) on the left
so the planner chooses RIGHT_ANTI, exercising the RIGHT_ANTI code path.
All tests use `compare_gpu_vs_cpu` to validate GPU results against CPU execution.

right anti join
- region ANTI JOIN nation on matching keys (r_regionkey = n_regionkey)

right anti join 2
- region ANTI JOIN nation on mismatched keys (r_regionkey = n_nationkey)

right anti join 3
- nation ANTI JOIN customer on n_nationkey = c_nationkey, varying selected columns (both keys,
non-key only, string column)

right anti join misfit
- nation ANTI JOIN customer on n_nationkey = c_custkey, keys that don't naturally align, producing
different filtering

 */

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - right anti join",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r anti join nation n on r.r_regionkey = n.n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - right anti join parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r anti join nation n on r.r_regionkey = n.n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - right anti join 2",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r anti join nation n on r.r_regionkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - right anti join 2 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r anti join nation n on r.r_regionkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - right anti join 3",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey "
    "from nation n anti join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - right anti join 3 parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey "
    "from nation n anti join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - right anti join misfit",
                 "[integration][gpu_execution][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey  "
    "from nation n anti join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - right anti join misfit parquet",
                 "[integration][gpu_execution][parquet][antijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey  "
    "from nation n anti join customer c on n.n_nationkey = c.c_custkey;");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Partitioned join tests
// ======================
// Force _num_partitions >= 2 by setting the partition size to 1 row, so that hash partitioning
// actually runs for all joins even with small TPC-H tables. Without this, all existing anti/semi
// join tests use tables small enough that _num_partitions = ceil(n / 10M) = 1, which skips
// partitioning entirely.
///////////////////////////////////////////////////////////////////////////////////////////////////

// RAII guard to reset partition size after each test, even on failure.
struct partition_size_guard {
  duckdb::Connection& con;
  explicit partition_size_guard(duckdb::Connection& con, duckdb::idx_t size) : con(con)
  {
    con.Query("SET hash_partition_bytes = " + std::to_string(size));
  }
  ~partition_size_guard() { con.Query("RESET hash_partition_bytes"); }
};

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - partitioned anti join (probe key not at col 0)",
                 "[integration][gpu_execution][antijoin][partitioned_join]")
{
  // n.n_regionkey is not column 0 in nation — this is the index mismatch that triggered the bug.
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n anti join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - partitioned anti join (probe key not at col 0) parquet",
                 "[integration][gpu_execution][parquet][antijoin][partitioned_join]")
{
  // n.n_regionkey is not column 0 in nation — this is the index mismatch that triggered the bug.
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n anti join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - partitioned semi join (probe key not at col 0)",
                 "[integration][gpu_execution][semijoin][partitioned_join]")
{
  // Same shape as the anti join above — verifies the fix didn't break semi join partitioning.
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - partitioned semi join (probe key not at col 0) parquet",
                 "[integration][gpu_execution][parquet][semijoin][partitioned_join]")
{
  // Same shape as the anti join above — verifies the fix didn't break semi join partitioning.
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - partitioned inner join (key not at col 0)",
                 "[integration][gpu_execution][partitioned_join]")
{
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, r.r_name "
    "from nation n join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - partitioned inner join (key not at col 0) parquet",
                 "[integration][gpu_execution][parquet][partitioned_join]")
{
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, r.r_name "
    "from nation n join region r on n.n_regionkey = r.r_regionkey;");
}

// Regression test: n_nationkey (INT32) and c_custkey (INT64) were hashed with different physical
// types, so cuDF murmur3 sent the same integer to different partitions and matching keys were
// missed. The bug is deterministic per key value, so restricting the build side to c_custkey <= 25
// still triggers it (INT32/INT64 misfit + partitioning both preserved) while keeping the anti-join
// result identical — only n_nationkey=0 is unmatched, since keys 1..24 match customers 1..25.
// The subset also avoids partitioning the full 150k-row customer table (hash_partition_bytes=1
// makes num_partitions grow with row count), which is what made this test slow.
constexpr const char* kMisfitAntiJoin =
  "select n.n_nationkey, n.n_regionkey from nation n "
  "anti join (select c_custkey from customer where c_custkey <= 25) c "
  "on n.n_nationkey = c.c_custkey;";

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - partitioned anti join (misfit key)",
                 "[integration][gpu_execution][antijoin][partitioned_join]")
{
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(kMisfitAntiJoin);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - partitioned anti join (misfit key) parquet",
                 "[integration][gpu_execution][antijoin][partitioned_join]")
{
  partition_size_guard guard(*con, 1);
  compare_gpu_vs_cpu(kMisfitAntiJoin);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - bigger inner join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l join orders o on l.l_orderkey = o.o_orderkey order by "
    "l.l_orderkey, l.l_linenumber limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - bigger inner join parquet",
                 "[integration][gpu_execution][parquet][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l join orders o on l.l_orderkey = o.o_orderkey order by "
    "l.l_orderkey, l.l_linenumber limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - bigger left join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l left join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber  limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - bigger left join parquet",
                 "[integration][gpu_execution][parquet][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l left join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber  limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - bigger right join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l right join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber  limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - bigger right join parquet",
                 "[integration][gpu_execution][parquet][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l right join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber  limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - bigger full outer join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l full outer join orders o on l.l_orderkey = "
    "o.o_orderkey order by l.l_orderkey, l.l_linenumber  limit 5000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - bigger full outer join parquet",
                 "[integration][gpu_execution][parquet][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l full outer join orders o on l.l_orderkey = "
    "o.o_orderkey order by l.l_orderkey, l.l_linenumber  limit 5000;");
}

//===----------------------------------------------------------------------===//
// Nested loop join tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop inner join single inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n join "
    "customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop inner join single inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n join "
    "customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop inner join double inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop inner join double inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop inner join double inequality condition, one "
                 "condition needing casting",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop inner join double inequality condition, one  parquet"
                 "condition needing casting",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop left join single inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n left "
    "join customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop left join single inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n left "
    "join customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop left join double inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l left join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop left join double inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l left join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop left join double inequality condition, one condition "
                 "needing casting",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(
  GPUExecutionParquetFixture,
  "gpu_execution - nested loop left join double inequality condition, one condition  parquet"
  "needing casting",
  "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop right join single inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n right "
    "join customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop right join single inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n right "
    "join customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop right join double inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l right join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop right join double inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l right join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop right join double inequality condition, one "
                 "condition needing casting",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop right join double inequality condition, one  parquet"
                 "condition needing casting",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop full outer join single inequality condition",
                 "[.][integration_disabled][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n full "
    "outer join customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop full outer join single inequality condition parquet",
                 "[.][integration_disabled][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n full "
    "outer join customer c "
    "on n.n_nationkey < c.c_nationkey where c.c_custkey < 100 "
    "order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop full outer join double inequality condition",
                 "[.][integration_disabled][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l full outer join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop full outer join double inequality condition parquet",
                 "[.][integration_disabled][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l full outer join partsupp ps "
    "on l.l_partkey < ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000"
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop full outer join double inequality condition, one "
                 "condition needing casting",
                 "[.][integration_disabled][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "full outer join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(
  GPUExecutionParquetFixture,
  "gpu_execution - nested loop full outer join double inequality condition, one  parquet"
  "condition needing casting",
  "[.][integration_disabled][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "full outer join customer c on n.n_nationkey > c.c_custkey and n.n_nationkey <= c.c_nationkey "
    "where c.c_custkey < 1000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop inner join one equality and one inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey and n.n_regionkey * 1000 < c.c_custkey order "
    "by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(
  GPUExecutionParquetFixture,
  "gpu_execution - nested loop inner join one equality and one inequality condition parquet",
  "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey and n.n_regionkey * 1000 < c.c_custkey order "
    "by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - nested loop inner join two inequality condition",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey < c.c_nationkey * 2 and n.n_regionkey * 1000 > c.c_custkey "
    "order "
    "by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - nested loop inner join two inequality condition parquet",
                 "[integration][gpu_execution][parquet][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey < c.c_nationkey * 2 and n.n_regionkey * 1000 > c.c_custkey "
    "order "
    "by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(
  GPUExecutionDuckDBFixture,
  "gpu_execution - nested loop inner join two inequality condition and expression eval",
  "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey < c.c_nationkey * 2 and n.n_regionkey * 1000 > c.c_custkey "
    "order "
    "by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed inner join one equality and one inequality condition but "
                 "equality column is shared (triggers nested join)",
                 "[integration][gpu_execution][nested_loop_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l right join partsupp ps "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_partkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000 "
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

//===----------------------------------------------------------------------===//
// Mixed join tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(
  GPUExecutionDuckDBFixture,
  "gpu_execution - mixed inner join one equality and one inequality condition with cast needed",
  "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_name,  c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey and n.n_regionkey < c.c_custkey  "
    "where c.c_custkey < 10000 order by c.c_custkey, n.n_nationkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed right join one equality and one inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l right join partsupp ps "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000 "
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed left join one equality and two inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l left join partsupp ps "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey and l.l_orderkey < "
    "ps.ps_suppkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000 "
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed inner join two equality and one inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey, l.l_orderkey from lineitem l join partsupp ps "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey and l.l_orderkey = "
    "ps.ps_partkey "
    "where l.l_orderkey < 1000 and ps.ps_partkey < 1000 "
    "order by ps.ps_partkey, ps.ps_suppkey, l.l_orderkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed semi join one equality and one inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber from lineitem l semi join partsupp ps "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 "
    "order by l.l_orderkey, l.l_linenumber limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed right semi join one equality and one inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey  from partsupp ps semi join lineitem l "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where ps.ps_partkey < 1000 "
    "order by ps.ps_partkey, ps.ps_suppkey limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed anti join one equality and one inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber from lineitem l anti join partsupp ps "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where l.l_orderkey < 1000 "
    "order by l.l_orderkey, l.l_linenumber limit 1000;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - mixed anti semi join one equality and one inequality condition",
                 "[integration][gpu_execution][mixed_join]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, ps.ps_suppkey  from partsupp ps anti join lineitem l "
    "on l.l_partkey = ps.ps_partkey and l.l_suppkey > ps.ps_suppkey "
    "where ps.ps_partkey < 1000 "
    "order by ps.ps_partkey, ps.ps_suppkey limit 1000;");
}

//===----------------------------------------------------------------------===//
// Disabled tests - known issues
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - two group by key: min max, sum, count of doubles",
                 "[.][integration_disabled][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, c_mktsegment, min(C_ACCTBAL), max(C_ACCTBAL), sum(C_ACCTBAL), "
    "count(C_ACCTBAL) from customer group by c_nationkey, c_mktsegment;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - two group by key: min max, sum, count of doubles parquet",
                 "[.][integration_disabled][gpu_execution][parquet][aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, c_mktsegment, min(C_ACCTBAL), max(C_ACCTBAL), sum(C_ACCTBAL), "
    "count(C_ACCTBAL) from customer group by c_nationkey, c_mktsegment;");
}

// Empty result set: "Port default not found in operator RESULT_COLLECTOR"
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - filter returns empty result",
                 "[.][integration_disabled][gpu_execution]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 99;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - filter returns empty result parquet",
                 "[.][integration_disabled][gpu_execution][parquet]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 99;");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by count",
                 "[integration][gpu_execution][group_by]")
{
  compare_gpu_vs_cpu("select n_regionkey, count(*) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by count parquet",
                 "[integration][gpu_execution][parquet][group_by]")
{
  compare_gpu_vs_cpu("select n_regionkey, count(*) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by min max count",
                 "[integration][gpu_execution][group_by]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, min(n_nationkey), max(n_nationkey), count(n_nationkey) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by min max count parquet",
                 "[integration][gpu_execution][parquet][group_by]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, min(n_nationkey), max(n_nationkey), count(n_nationkey) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by avg integer",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu("select n_regionkey, avg(n_nationkey) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by avg integer parquet",
                 "[integration][gpu_execution][parquet][group_by][avg]")
{
  compare_gpu_vs_cpu("select n_regionkey, avg(n_nationkey) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by avg with other aggregates",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, avg(n_nationkey), sum(n_nationkey), count(*) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by avg with other aggregates parquet",
                 "[integration][gpu_execution][parquet][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, avg(n_nationkey), sum(n_nationkey), count(*) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by avg decimal",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_returnflag, avg(l_quantity), avg(l_discount) "
    "from lineitem group by l_returnflag;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by avg decimal parquet",
                 "[integration][gpu_execution][parquet][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_returnflag, avg(l_quantity), avg(l_discount) "
    "from lineitem group by l_returnflag;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by sum avg on lineitem",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*) "
    "from lineitem group by l_returnflag, l_linestatus;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by sum avg on lineitem parquet",
                 "[integration][gpu_execution][parquet][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*) "
    "from lineitem group by l_returnflag, l_linestatus;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by min, max, avg on decimal on lineitem",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_tax, min(l_extendedprice), max(l_extendedprice), avg(l_extendedprice)"
    "from lineitem group by l_tax;",
    0.0001);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by min, max, avg on decimal on lineitem parquet",
                 "[integration][gpu_execution][parquet][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_tax, min(l_extendedprice), max(l_extendedprice), avg(l_extendedprice)"
    "from lineitem group by l_tax;",
    0.0001);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - group by min, max, avg, sum on decimal on lineitem",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_discount, min(l_extendedprice), sum(l_extendedprice), max(l_extendedprice), "
    "avg(l_extendedprice), sum(l_tax)"
    "from lineitem group by l_discount;",
    0.0001);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - group by min, max, avg, sum on decimal on lineitem parquet",
                 "[integration][gpu_execution][parquet][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_discount, min(l_extendedprice), sum(l_extendedprice), max(l_extendedprice), "
    "avg(l_extendedprice), sum(l_tax)"
    "from lineitem group by l_discount;",
    0.0001);
}

//===----------------------------------------------------------------------===//
// Order by tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by",
                 "[integration][gpu_execution][order_by]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by parquet",
                 "[integration][gpu_execution][parquet][order_by]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by column not in select",
                 "[integration][gpu_execution][order_by][order_by_proj]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by column not in select parquet",
                 "[integration][gpu_execution][parquet][order_by][order_by_proj]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by column not in select lineitem",
                 "[integration][gpu_execution][order_by][order_by_proj]")
{
  compare_gpu_vs_cpu("select l_orderkey from lineitem order by l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by column not in select lineitem parquet",
                 "[integration][gpu_execution][parquet][order_by][order_by_proj]")
{
  compare_gpu_vs_cpu("select l_orderkey from lineitem order by l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by multipartition",
                 "[integration][gpu_execution][order_by]")
{
  // Force small partition size (1 KB) so lineitem data is split into multiple partitions
  con->Query("SET max_sort_partition_bytes = 1024;");

  compare_gpu_vs_cpu("select l_orderkey, l_partkey from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by multipartition parquet",
                 "[integration][gpu_execution][parquet][order_by]")
{
  // Force small partition size (1 KB) so lineitem data is split into multiple partitions
  con->Query("SET max_sort_partition_bytes = 1024;");

  compare_gpu_vs_cpu("select l_orderkey, l_partkey from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by multiple columns",
                 "[integration][gpu_execution][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_linenumber, l_quantity from lineitem order by l_orderkey, l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by multiple columns parquet",
                 "[integration][gpu_execution][parquet][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_linenumber, l_quantity from lineitem order by l_orderkey, l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by desc",
                 "[integration][gpu_execution][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_partkey, l_suppkey from lineitem order by l_partkey desc;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by desc parquet",
                 "[integration][gpu_execution][parquet][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_partkey, l_suppkey from lineitem order by l_partkey desc;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by many selected columns",
                 "[integration][gpu_execution][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity "
    "from lineitem order by l_suppkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by many selected columns parquet",
                 "[integration][gpu_execution][parquet][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity "
    "from lineitem order by l_suppkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by with decimal column",
                 "[integration][gpu_execution][order_by][order_by_types]")
{
  compare_gpu_vs_cpu("select o_orderkey, o_totalprice from orders order by o_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by with decimal column parquet",
                 "[integration][gpu_execution][parquet][order_by][order_by_types]")
{
  compare_gpu_vs_cpu("select o_orderkey, o_totalprice from orders order by o_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - scan lineitem with varchar column",
                 "[integration][gpu_execution][varchar_scan_lineitem]")
{
  compare_gpu_vs_cpu("select l_orderkey, l_shipinstruct from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - scan lineitem with varchar column parquet",
                 "[integration][gpu_execution][parquet][varchar_scan_lineitem]")
{
  compare_gpu_vs_cpu("select l_orderkey, l_shipinstruct from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by lineitem with short varchar column",
                 "[integration][gpu_execution][order_by][varchar_order]")
{
  compare_gpu_vs_cpu(
    "select l_orderkey, l_shipinstruct, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by lineitem with short varchar column parquet",
                 "[integration][gpu_execution][parquet][order_by][varchar_order]")
{
  compare_gpu_vs_cpu(
    "select l_orderkey, l_shipinstruct, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by lineitem with long varchar column",
                 "[integration][gpu_execution][order_by][varchar_order]")
{
  compare_gpu_vs_cpu(
    "select l_orderkey, l_comment, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by lineitem with long varchar column parquet",
                 "[integration][gpu_execution][parquet][order_by][varchar_order]")
{
  compare_gpu_vs_cpu(
    "select l_orderkey, l_comment, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - scan with varchar column",
                 "[integration][gpu_execution][order_by_types][varchar]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_name from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - scan with varchar column parquet",
                 "[integration][gpu_execution][parquet][order_by_types][varchar]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_name from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - order by with varchar column",
                 "[integration][gpu_execution][order_by][order_by_types]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_name from nation order by n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - order by with varchar column parquet",
                 "[integration][gpu_execution][parquet][order_by][order_by_types]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_name from nation order by n_nationkey;");
}

//===----------------------------------------------------------------------===//
// String concat tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - string concat || operator parquet",
                 "[integration][gpu_execution][parquet][string_concat]")
{
  compare_gpu_vs_cpu(
    "SELECT l_orderkey, l_returnflag || '-' || l_linestatus FROM lineitem ORDER BY l_orderkey "
    "LIMIT 100;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - string concat() function parquet",
                 "[integration][gpu_execution][parquet][string_concat]")
{
  compare_gpu_vs_cpu(
    "SELECT l_orderkey, concat(l_returnflag, l_linestatus) FROM lineitem ORDER BY l_orderkey LIMIT "
    "100;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - string concat column with longer varchar parquet",
                 "[integration][gpu_execution][parquet][string_concat]")
{
  compare_gpu_vs_cpu(
    "SELECT p_partkey, p_brand || ': ' || p_type FROM part ORDER BY p_partkey LIMIT 100;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - string concat in WHERE clause parquet",
                 "[integration][gpu_execution][parquet][string_concat]")
{
  compare_gpu_vs_cpu(
    "SELECT l_orderkey FROM lineitem WHERE l_returnflag || l_linestatus = 'NF' ORDER BY l_orderkey "
    "LIMIT 100;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - string concat with nested expression parquet",
                 "[integration][gpu_execution][parquet][string_concat]")
{
  // ORDER BY l_orderkey, l_linenumber for a deterministic primary-key sort —
  // l_orderkey alone is non-unique in lineitem so LIMIT would pick different rows on GPU vs CPU.
  compare_gpu_vs_cpu(
    "SELECT l_orderkey, substring(l_comment, 1, 5) || '...' FROM lineitem "
    "ORDER BY l_orderkey, l_linenumber LIMIT 100;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - string concat NULL propagation parquet",
                 "[integration][gpu_execution][parquet][string_concat][nulls]")
{
  // TPC-H has no nullable VARCHAR columns; introduce nulls via CASE so that
  // concat's null-propagation semantics (any NULL input → NULL output) are exercised.
  compare_gpu_vs_cpu(
    "SELECT l_orderkey, "
    "  CASE WHEN l_orderkey % 7 = 0 THEN NULL ELSE l_returnflag END || '-' || l_linestatus "
    "FROM lineitem ORDER BY l_orderkey, l_linenumber LIMIT 100;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - cast integer to decimal preserves scale",
                 "[integration][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu("select n_nationkey, cast(n_nationkey as Decimal(18,2)) as d from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - cast integer to decimal preserves scale parquet",
                 "[integration][gpu_execution][parquet][cast][decimal]")
{
  compare_gpu_vs_cpu("select n_nationkey, cast(n_nationkey as Decimal(18,2)) as d from nation;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - cast integer to decimal with aggregation",
                 "[integration][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, max(cast(n_nationkey as Decimal(18,2))) as max_d "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - cast integer to decimal with aggregation parquet",
                 "[integration][gpu_execution][parquet][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, max(cast(n_nationkey as Decimal(18,2))) as max_d "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - cast to decimal different scales",
                 "[integration][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select cast(n_nationkey as Decimal(9,0)) as d0, "
    "cast(n_nationkey as Decimal(9,4)) as d4 from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - cast to decimal different scales parquet",
                 "[integration][gpu_execution][parquet][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select cast(n_nationkey as Decimal(9,0)) as d0, "
    "cast(n_nationkey as Decimal(9,4)) as d4 from nation;");
}

// Disabled: avg() in grouped aggregates not yet supported (separate PR)
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - issue 227 cast decimal with avg and group by",
                 "[.][integration_disabled][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select avg(n_regionkey), avg(n_nationkey), n_name, "
    "max(cast(n_nationkey as Decimal(18,2))) "
    "from nation group by n_regionkey, n_name;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - issue 227 cast decimal with avg and group by parquet",
                 "[.][integration_disabled][gpu_execution][parquet][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select avg(n_regionkey), avg(n_nationkey), n_name, "
    "max(cast(n_nationkey as Decimal(18,2))) "
    "from nation group by n_regionkey, n_name;");
}

//===----------------------------------------------------------------------===//
// Count distinct tests
//===----------------------------------------------------------------------===//

// nation: 25 rows, n_regionkey in {0..4} with exactly 5 nations per region.
// count(distinct n_nationkey) per region must equal 5.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: single group key",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: single group key parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;");
}

// count(distinct n_name): n_name is unique per nation, so same result as above.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: string column",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_name) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: string column parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_name) from nation group by n_regionkey;");
}

// count(distinct) mixed with other aggregations in the same grouped aggregate.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: mixed with min and count",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey), min(n_nationkey), count(*) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: mixed with min and count parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey), min(n_nationkey), count(*) "
    "from nation group by n_regionkey;");
}

// Larger table: customer (15000 rows), two group-by keys.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: larger table two group keys",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment) from customer group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: larger table two group keys parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment) from customer group by c_nationkey;");
}

// ---------------------------------------------------------------------------
// Multi-partition count distinct tests
//
// PARTITION_SIZE is temporarily lowered so the engine creates multiple
// partitions even with the small TPC-H test tables.  A RAII guard restores
// the original value after each test regardless of pass/fail.
// ---------------------------------------------------------------------------

// nation (25 rows) with partition_size=5 → ceil(25/5) = 5 partitions.
// count(distinct n_nationkey) per region must still equal 5.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: multi-partition forced, single group key",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 5);
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: multi-partition forced, single group key parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 5);
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;");
}

// customer (15000 rows) with partition_size=1000 → 15 partitions.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: multi-partition forced, customer table",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 1000);
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment) from customer group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: multi-partition forced, customer table parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 1000);
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment) from customer group by c_nationkey;");
}

// Mixed aggregations across multiple forced partitions.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: multi-partition forced, mixed aggregations",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 1000);
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment), min(c_custkey), count(*) "
    "from customer group by c_nationkey;");
}

TEST_CASE_METHOD(
  GPUExecutionParquetFixture,
  "gpu_execution - count distinct: multi-partition forced, mixed aggregations parquet",
  "[integration][gpu_execution][parquet][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 1000);
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment), min(c_custkey), count(*) "
    "from customer group by c_nationkey;");
}

// ---------------------------------------------------------------------------
// Multi-column COUNT(DISTINCT) integration tests
// count(distinct (col1, col2)) counts distinct combinations, not individual values.
// ---------------------------------------------------------------------------

// nation: 25 rows, 5 unique (n_nationkey, n_name) combos per region.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: multi-column struct",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct (n_nationkey, n_name)) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - count distinct: multi-column struct parquet",
                 "[integration][gpu_execution][parquet][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct (n_nationkey, n_name)) from nation group by n_regionkey;");
}

// Multi-column count distinct with a forced multi-partition execution.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count distinct: multi-column struct, multi-partition forced",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 5);
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct (n_nationkey, n_name)) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(
  GPUExecutionParquetFixture,
  "gpu_execution - count distinct: multi-column struct, multi-partition forced parquet",
  "[integration][gpu_execution][parquet][group_by][count_distinct][multi_partition]")
{
  partition_size_guard guard(*con, 5);
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct (n_nationkey, n_name)) from nation group by n_regionkey;");
}

//===----------------------------------------------------------------------===//
// Top N / Join tests (disabled)
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - top n",
                 "[.][integration_disabled][gpu_execution][top_n]")
{
  compare_gpu_vs_cpu(
    "select n_nationkey, n_regionkey from nation order by n_regionkey desc limit 5;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - top n parquet",
                 "[.][integration_disabled][gpu_execution][parquet][top_n]")
{
  compare_gpu_vs_cpu(
    "select n_nationkey, n_regionkey from nation order by n_regionkey desc limit 5;");
}

//===----------------------------------------------------------------------===//
// Empty result queries
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - empty simple query",
                 "[integration][gpu_execution][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l_linestatus, l_orderkey, l_comment, l_receiptdate from lineitem where l_linestatus = "
    "'J' and l_orderkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - empty simple query parquet",
                 "[integration][gpu_execution][parquet][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l_linestatus, l_orderkey, l_comment, l_receiptdate from lineitem where l_linestatus = "
    "'J' and l_orderkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - empty aggregation with group by query",
                 "[integration][gpu_execution][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l_linestatus, count(*), min(l_orderkey) as mino, sum(l_orderkey), count(l_orderkey), "
    "count(l_receiptdate), min(l_receiptdate), count(l_comment), min(l_comment) from lineitem "
    "where l_linestatus = 'J' group by l_linestatus;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - empty aggregation with group by query parquet",
                 "[integration][gpu_execution][parquet][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l_linestatus, count(*), min(l_orderkey) as mino, sum(l_orderkey), count(l_orderkey), "
    "count(l_receiptdate), min(l_receiptdate), count(l_comment), min(l_comment) from lineitem "
    "where l_linestatus = 'J' group by l_linestatus;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - empty aggregation without group by query",
                 "[integration][gpu_execution][empty_result]")
{
  compare_gpu_vs_cpu(
    "select count(*), min(l_orderkey), sum(l_orderkey) as sumo, count(l_orderkey), "
    "count(l_receiptdate), min(l_receiptdate), count(l_comment), min(l_comment) from lineitem "
    "where l_linestatus = 'J';");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - empty aggregation without group by query parquet",
                 "[integration][gpu_execution][parquet][empty_result]")
{
  compare_gpu_vs_cpu(
    "select count(*), min(l_orderkey), sum(l_orderkey) as sumo, count(l_orderkey), "
    "count(l_receiptdate), min(l_receiptdate), count(l_comment), min(l_comment) from lineitem "
    "where l_linestatus = 'J';");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - join with empty one side",
                 "[integration][gpu_execution][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey as lokey, l.l_linestatus, o.o_custkey from lineitem l inner join orders o "
    "on l.l_orderkey = o.o_orderkey where l_linestatus = 'J';");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - join with empty one side parquet",
                 "[integration][gpu_execution][parquet][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey as lokey, l.l_linestatus, o.o_custkey from lineitem l inner join orders o "
    "on l.l_orderkey = o.o_orderkey where l_linestatus = 'J';");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - join with empty two sides",
                 "[integration][gpu_execution][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linestatus, o.o_custkey as ockey from lineitem l inner join orders o "
    "on l.l_orderkey = o.o_orderkey where l_linestatus = 'J' and o.o_comment = 'Special';");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - join with empty two sides parquet",
                 "[integration][gpu_execution][parquet][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linestatus, o.o_custkey as ockey from lineitem l inner join orders o "
    "on l.l_orderkey = o.o_orderkey where l_linestatus = 'J' and o.o_comment = 'Special';");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - join with empty output and order by",
                 "[integration][gpu_execution][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linestatus, o.o_custkey from lineitem l inner join orders o on "
    "l.l_orderkey = o.o_orderkey where l.l_orderkey > 10000 and o.o_orderkey < 10000 order by "
    "l.l_orderkey, o.o_custkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - join with empty output and order by parquet",
                 "[integration][gpu_execution][parquet][empty_result]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linestatus, o.o_custkey from lineitem l inner join orders o on "
    "l.l_orderkey = o.o_orderkey where l.l_orderkey > 10000 and o.o_orderkey < 10000 order by "
    "l.l_orderkey, o.o_custkey;");
}

//===----------------------------------------------------------------------===//
// All-pruned scan and empty-side join queries
//
// Local (parquet + duckdb-native) port of the empty-result coverage from the S3
// pushdown suite (test_s3_sql_surface.cpp, "S3 pushdown ..." cases). A predicate
// like `n_regionkey = 99` matches no rows; with reader-side filter pushdown (on
// by default for local read_parquet) it prunes every row group, producing the
// all-pruned scan this exercises. Each query must still complete on the GPU and
// emit join-type-correct / aggregate-identity output against the surviving side.
//===----------------------------------------------------------------------===//

namespace {

struct labeled_query {
  std::string label;
  std::string sql;
};

std::string sql_string_literal(std::string const& value)
{
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('\'');
  for (auto const c : value) {
    if (c == '\'') { escaped.push_back('\''); }
    escaped.push_back(c);
  }
  escaped.push_back('\'');
  return escaped;
}

void require_query_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("setup query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

struct watchdog_query_result {
  bool timed_out{false};
  duckdb::idx_t row_count{0};
  duckdb::idx_t column_count{0};
  std::vector<std::string> column_names;
  std::vector<std::string> column_types;
  std::vector<std::vector<std::string>> rows;
  std::string error;
};

watchdog_query_result run_query_with_watchdog(duckdb::Connection& con,
                                              std::string sql,
                                              std::chrono::seconds timeout)
{
  struct shared_state {
    std::mutex mtx;
    std::condition_variable cv;
    bool done{false};
    watchdog_query_result result;
  };

  auto state = std::make_shared<shared_state>();
  std::thread worker([&con, sql = std::move(sql), state]() {
    watchdog_query_result out;
    try {
      auto result = con.Query(sql);
      if (!result) {
        out.error = "query returned nullptr";
      } else if (result->HasError()) {
        out.error = result->GetError();
      } else {
        out.row_count    = result->RowCount();
        out.column_count = result->ColumnCount();
        out.column_names.reserve(result->ColumnCount());
        out.column_types.reserve(result->ColumnCount());
        for (duckdb::idx_t c = 0; c < result->ColumnCount(); ++c) {
          out.column_names.push_back(result->ColumnName(c));
          out.column_types.push_back(result->types[c].ToString());
        }
        auto& materialized = result->Cast<duckdb::MaterializedQueryResult>();
        out.rows           = GPUExecutionFixtureBase::collect_rows(materialized);
      }
    } catch (std::exception const& e) {
      out.error = e.what();
    } catch (...) {
      out.error = "query threw an unknown exception";
    }

    {
      std::lock_guard<std::mutex> lock(state->mtx);
      state->result = std::move(out);
      state->done   = true;
    }
    state->cv.notify_one();
  });

  {
    std::unique_lock<std::mutex> lock(state->mtx);
    if (!state->cv.wait_for(lock, timeout, [&] { return state->done; })) {
      con.Interrupt();
      if (!state->cv.wait_for(lock, std::chrono::seconds{5}, [&] { return state->done; })) {
        worker.detach();
        watchdog_query_result out;
        out.timed_out = true;
        out.error     = "query timed out after " + std::to_string(timeout.count()) + " seconds";
        return out;
      }
    }
  }

  worker.join();
  return std::move(state->result);
}

watchdog_query_result compare_gpu_vs_cpu_with_watchdog(duckdb::Connection& con,
                                                       std::string const& query,
                                                       std::chrono::seconds timeout,
                                                       std::function<void()> on_timeout = {})
{
  require_query_ok(con, "SET gpu_execution = true;");
  auto before_gpu_stats = sirius::test::get_transparent_execution_stats(con);

  auto gpu_result = run_query_with_watchdog(con, query, timeout);
  INFO("query: " << query);
  INFO(gpu_result.error);
  if (gpu_result.timed_out && on_timeout) { on_timeout(); }
  REQUIRE_FALSE(gpu_result.timed_out);
  REQUIRE(gpu_result.error.empty());

  auto after_gpu_stats = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before_gpu_stats, after_gpu_stats, 1, 0, 1);

  require_query_ok(con, "SET gpu_execution = false;");
  auto cpu_result = con.Query(query);
  require_query_ok(con, "SET gpu_execution = true;");
  REQUIRE(cpu_result);
  if (cpu_result->HasError()) { UNSCOPED_INFO("CPU oracle error: " << cpu_result->GetError()); }
  REQUIRE_FALSE(cpu_result->HasError());
  auto after_cpu_stats = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(after_gpu_stats, after_cpu_stats, 0, 0, 0);

  std::vector<std::string> cpu_column_names;
  std::vector<std::string> cpu_column_types;
  cpu_column_names.reserve(cpu_result->ColumnCount());
  cpu_column_types.reserve(cpu_result->ColumnCount());
  for (duckdb::idx_t c = 0; c < cpu_result->ColumnCount(); ++c) {
    cpu_column_names.push_back(cpu_result->ColumnName(c));
    cpu_column_types.push_back(cpu_result->types[c].ToString());
  }
  auto& cpu_materialized = cpu_result->Cast<duckdb::MaterializedQueryResult>();
  auto cpu_rows          = GPUExecutionFixtureBase::collect_rows(cpu_materialized);

  REQUIRE(gpu_result.column_count == cpu_result->ColumnCount());
  REQUIRE(gpu_result.row_count == cpu_result->RowCount());
  CHECK(gpu_result.column_names == cpu_column_names);
  CHECK(gpu_result.column_types == cpu_column_types);
  CHECK(gpu_result.rows == cpu_rows);
  return gpu_result;
}

class local_sirius_config_guard {
 public:
  explicit local_sirius_config_guard(fs::path config_path)
  {
    if (auto* current = std::getenv("SIRIUS_CONFIG_FILE"); current != nullptr) {
      had_original_config_env_ = true;
      original_config_env_     = current;
    }
    if (auto* current = std::getenv("SIRIUS_DISABLE"); current != nullptr) {
      had_original_disable_env_ = true;
      original_disable_env_     = current;
    }
    setenv("SIRIUS_CONFIG_FILE", config_path.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
  }

  ~local_sirius_config_guard()
  {
    if (had_original_config_env_) {
      setenv("SIRIUS_CONFIG_FILE", original_config_env_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
    if (had_original_disable_env_) {
      setenv("SIRIUS_DISABLE", original_disable_env_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_DISABLE");
    }
  }

 private:
  std::string original_config_env_;
  std::string original_disable_env_;
  bool had_original_config_env_{false};
  bool had_original_disable_env_{false};
};

void pause_shared_envs_for_local_duckdb()
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }
}

fs::path scan_memory_config_path()
{
  auto path = get_project_root() / "test" / "cpp" / "scan" / "memory.yaml";
  REQUIRE(fs::exists(path));
  return path;
}

class empty_native_table_fixture {
 public:
  empty_native_table_fixture()
  {
    pause_shared_envs_for_local_duckdb();
    config_guard_ = std::make_unique<local_sirius_config_guard>(scan_memory_config_path());

    static std::atomic<std::uint64_t> counter{0};
    auto const id = counter.fetch_add(1, std::memory_order_relaxed);
    dir_          = fs::temp_directory_path() / ("sirius-empty-native-" + std::to_string(id));
    std::error_code ec;
    fs::remove_all(dir_, ec);
    fs::create_directories(dir_);
    db_path_ = dir_ / "empty.duckdb";

    db  = std::make_unique<duckdb::DuckDB>(db_path_.string());
    con = std::make_unique<duckdb::Connection>(*db);

    require_query_ok(
      *con, "ATTACH " + sql_string_literal(get_tpch_db_path().string()) + " AS tpch (READ_ONLY);");
    require_query_ok(*con, "CREATE TABLE e(i INTEGER);");
    require_query_ok(*con, "CHECKPOINT;");
  }

  ~empty_native_table_fixture()
  {
    if (!leaked_) {
      con.reset();
      db.reset();
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }

  void leak_after_timeout()
  {
    leaked_ = true;
    (void)con.release();
    (void)db.release();
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

 private:
  std::unique_ptr<local_sirius_config_guard> config_guard_;
  fs::path dir_;
  fs::path db_path_;
  bool leaked_{false};
};

// Programmatic port of "S3 pushdown shape-C zero-side joins match the local
// parquet oracle": every supported join type crossed with which side is pruned
// to empty. `nation`/`region` are the full tables; the parenthesized subqueries
// prune one side to zero rows. Includes a both-sides-alive MARK control and a
// both-sides-pruned case. Sirius must match the CPU oracle for each.
std::vector<labeled_query> build_empty_side_join_matrix()
{
  const std::string nation        = "nation";
  const std::string region        = "region";
  const std::string pruned_nation = "(select * from nation where n_regionkey = 99)";
  const std::string pruned_region = "(select * from region where r_regionkey = 99)";
  return {
    {"hash inner dead left",
     "select n.n_nationkey, r.r_name from " + pruned_nation + " n inner join " + region +
       " r on n.n_regionkey = r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"hash inner dead right",
     "select n.n_nationkey, r.r_name from " + nation + " n inner join " + pruned_region +
       " r on n.n_regionkey = r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"hash left dead left",
     "select n.n_nationkey, r.r_name from " + pruned_nation + " n left join " + region +
       " r on n.n_regionkey = r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"hash left dead right",
     "select n.n_nationkey, r.r_name from " + nation + " n left join " + pruned_region +
       " r on n.n_regionkey = r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"hash right dead left",
     "select n.n_nationkey, r.r_name from " + pruned_nation + " n right join " + region +
       " r on n.n_regionkey = r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"hash right dead right",
     "select n.n_nationkey, r.r_name from " + nation + " n right join " + pruned_region +
       " r on n.n_regionkey = r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"hash full outer dead left",
     "select n.n_nationkey, r.r_name from " + pruned_nation + " n full outer join " + region +
       " r on n.n_regionkey = r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"hash full outer dead right",
     "select n.n_nationkey, r.r_name from " + nation + " n full outer join " + pruned_region +
       " r on n.n_regionkey = r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"hash not exists dead inner",
     "select n.n_nationkey from " + nation + " n where not exists (select 1 from " + pruned_region +
       " r where r.r_regionkey = n.n_regionkey) order by n.n_nationkey;"},
    {"hash in mark dead inner",
     "select n.n_nationkey, n.n_regionkey in (select r_regionkey from " + pruned_region +
       ") as in_pruned from " + nation + " n order by n.n_nationkey;"},
    {"hash exists dead inner",
     "select n.n_nationkey from " + nation + " n where exists (select 1 from " + pruned_region +
       " r where r.r_regionkey = n.n_regionkey) order by n.n_nationkey;"},
    {"hash count over zero-side join",
     "select count(*) from " + pruned_nation + " n inner join " + region +
       " r on n.n_regionkey = r.r_regionkey;"},
    {"hash both sides pruned",
     "select n.n_nationkey, r.r_name from " + pruned_nation + " n inner join " + pruned_region +
       " r on n.n_regionkey = r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"nlj left dead right",
     "select n.n_nationkey, r.r_regionkey from " + nation + " n left join " + pruned_region +
       " r on n.n_regionkey < r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"nlj right dead left",
     "select n.n_nationkey, r.r_regionkey from " + pruned_nation + " n right join " + region +
       " r on n.n_regionkey < r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"nlj full outer dead left",
     "select n.n_nationkey, r.r_regionkey from " + pruned_nation + " n full outer join " + region +
       " r on n.n_regionkey < r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"nlj full outer dead right",
     "select n.n_nationkey, r.r_regionkey from " + nation + " n full outer join " + pruned_region +
       " r on n.n_regionkey < r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"nlj anti dead right",
     "select n.n_nationkey from " + nation + " n anti join " + pruned_region +
       " r on n.n_regionkey < r.r_regionkey order by n.n_nationkey;"},
    {"nlj inner dead left",
     "select n.n_nationkey, r.r_regionkey from " + pruned_nation + " n inner join " + region +
       " r on n.n_regionkey < r.r_regionkey order by r.r_regionkey, n.n_nationkey;"},
    {"nlj inner dead right",
     "select n.n_nationkey, r.r_regionkey from " + nation + " n inner join " + pruned_region +
       " r on n.n_regionkey < r.r_regionkey order by n.n_nationkey, r.r_regionkey;"},
    {"nlj mark both alive",
     "select n.n_nationkey, n.n_regionkey < any (select r_regionkey from " + region +
       ") as lt_any_region from " + nation + " n order by n.n_nationkey;"},
    {"nlj mark dead right",
     "select n.n_nationkey, n.n_regionkey < any (select r_regionkey from " + pruned_region +
       ") as lt_any_region from " + nation + " n order by n.n_nationkey;"},
    {"nlj mark dead left",
     "select n.n_nationkey, n.n_regionkey < any (select r_regionkey from " + region +
       ") as lt_any_region from " + pruned_nation + " n order by n.n_nationkey;"},
  };
}

}  // namespace

TEST_CASE("gpu_execution - empty native table count identity",
          "[integration][gpu_execution][empty_result][empty-table]")
{
  empty_native_table_fixture fixture;
  auto result = compare_gpu_vs_cpu_with_watchdog(
    *fixture.con, "select count(*) as c from e;", std::chrono::seconds{30}, [&fixture] {
      fixture.leak_after_timeout();
    });
  REQUIRE(result.row_count == 1);
  REQUIRE(result.column_count == 1);
  CHECK(result.rows == std::vector<std::vector<std::string>>{{"0"}});
}

TEST_CASE("gpu_execution - empty native table scan preserves schema",
          "[integration][gpu_execution][empty_result][empty-table]")
{
  empty_native_table_fixture fixture;
  auto result = compare_gpu_vs_cpu_with_watchdog(
    *fixture.con, "select i from e;", std::chrono::seconds{30}, [&fixture] {
      fixture.leak_after_timeout();
    });
  REQUIRE(result.row_count == 0);
  REQUIRE(result.column_count == 1);
  CHECK(result.column_names == std::vector<std::string>{"i"});
  CHECK(result.column_types == std::vector<std::string>{"INTEGER"});
}

TEST_CASE("gpu_execution - empty native table left join pads survivor rows",
          "[integration][gpu_execution][empty_result][empty-table]")
{
  empty_native_table_fixture fixture;
  auto result = compare_gpu_vs_cpu_with_watchdog(
    *fixture.con,
    "select n.n_nationkey, e.i from tpch.nation n left join e on n.n_nationkey = e.i "
    "order by n.n_nationkey;",
    std::chrono::seconds{30},
    [&fixture] { fixture.leak_after_timeout(); });
  REQUIRE(result.row_count == 25);
  REQUIRE(result.column_count == 2);
}

// FULL OUTER and RIGHT joins are excluded from BUILD_PROBE and always run the STANDARD partial-
// barrier path. When one side is a genuinely empty native table it delivers ZERO batches (its
// concat emits nothing), so the STANDARD scheduler must still (a) drain the surviving side and (b)
// emit its rows NULL-padded. These reproduce the empty-opposite-side terminal case for that path in
// both directions.
TEST_CASE("gpu_execution - empty native table full outer join pads survivor rows (empty build)",
          "[integration][gpu_execution][empty_result][empty-table]")
{
  empty_native_table_fixture fixture;
  auto result = compare_gpu_vs_cpu_with_watchdog(
    *fixture.con,
    "select n.n_nationkey, e.i from tpch.nation n full outer join e on n.n_nationkey = e.i "
    "order by n.n_nationkey;",
    std::chrono::seconds{30},
    [&fixture] { fixture.leak_after_timeout(); });
  REQUIRE(result.row_count == 25);
  REQUIRE(result.column_count == 2);
}

TEST_CASE("gpu_execution - empty native table full outer join pads survivor rows (empty probe)",
          "[integration][gpu_execution][empty_result][empty-table]")
{
  empty_native_table_fixture fixture;
  auto result = compare_gpu_vs_cpu_with_watchdog(
    *fixture.con,
    "select n.n_nationkey, e.i from e full outer join tpch.nation n on e.i = n.n_nationkey "
    "order by n.n_nationkey;",
    std::chrono::seconds{30},
    [&fixture] { fixture.leak_after_timeout(); });
  REQUIRE(result.row_count == 25);
  REQUIRE(result.column_count == 2);
}

TEST_CASE("gpu_execution - empty parquet count identity",
          "[integration][gpu_execution][parquet][empty_result][empty-table]")
{
  empty_native_table_fixture fixture;
  static std::atomic<std::uint64_t> counter{0};
  auto const id  = counter.fetch_add(1, std::memory_order_relaxed);
  auto const dir = fs::temp_directory_path() / ("sirius-empty-parquet-" + std::to_string(id));
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir);
  auto const parquet_path = dir / "empty.parquet";

  require_query_ok(*fixture.con, "SET gpu_execution = false;");
  require_query_ok(*fixture.con,
                   "COPY (SELECT 1 AS i WHERE false) TO " +
                     sql_string_literal(parquet_path.string()) + " (FORMAT PARQUET);");
  require_query_ok(*fixture.con, "SET gpu_execution = true;");
  require_query_ok(*fixture.con, "SET enable_compressed_materialization = false;");
  auto const before_stats = sirius::test::get_compressed_materialization_stats(*fixture.con);
  auto result             = compare_gpu_vs_cpu_with_watchdog(
    *fixture.con,
    "select count(*) as c from read_parquet(" + sql_string_literal(parquet_path.string()) + ");",
    std::chrono::seconds{30},
    [&fixture] { fixture.leak_after_timeout(); });
  auto const after_stats = sirius::test::get_compressed_materialization_stats(*fixture.con);
  REQUIRE(after_stats.scan_columns_narrowed == before_stats.scan_columns_narrowed);
  REQUIRE(after_stats.scan_columns_restored == before_stats.scan_columns_restored);
  REQUIRE(result.row_count == 1);
  REQUIRE(result.column_count == 1);
  CHECK(result.rows == std::vector<std::vector<std::string>>{{"0"}});

  fs::remove_all(dir, ec);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - all-pruned empty filter",
                 "[integration][gpu_execution][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 99 order by n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - all-pruned empty filter parquet",
                 "[integration][gpu_execution][parquet][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 99 order by n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - all-pruned ungrouped count identity",
                 "[integration][gpu_execution][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu("select count(*) as c from nation where n_regionkey = 99;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - all-pruned ungrouped count identity parquet",
                 "[integration][gpu_execution][parquet][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu("select count(*) as c from nation where n_regionkey = 99;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - all-pruned ungrouped aggregates identity and nulls",
                 "[integration][gpu_execution][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu(
    "select count(*) as c_all, count(n_name) as c_name, sum(n_nationkey) as sum_key, "
    "min(n_name) as min_name, max(n_name) as max_name, avg(n_nationkey) as avg_key, "
    "first(n_name) as first_name from nation where n_regionkey = 99;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - all-pruned ungrouped aggregates identity and nulls parquet",
                 "[integration][gpu_execution][parquet][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu(
    "select count(*) as c_all, count(n_name) as c_name, sum(n_nationkey) as sum_key, "
    "min(n_name) as min_name, max(n_name) as max_name, avg(n_nationkey) as avg_key, "
    "first(n_name) as first_name from nation where n_regionkey = 99;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - all-pruned grouped aggregate emits no groups",
                 "[integration][gpu_execution][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(*) as c from nation where n_regionkey = 99 group "
    "by n_regionkey order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - all-pruned grouped aggregate emits no groups parquet",
                 "[integration][gpu_execution][parquet][empty_result][all_pruned]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(*) as c from nation where n_regionkey = 99 group "
    "by n_regionkey order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - empty-side join matrix",
                 "[integration][gpu_execution][empty_result][all_pruned]")
{
  for (auto const& q : build_empty_side_join_matrix()) {
    INFO("empty-side join case: " << q.label);
    compare_gpu_vs_cpu(q.sql);
  }
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - empty-side join matrix parquet",
                 "[integration][gpu_execution][parquet][empty_result][all_pruned]")
{
  for (auto const& q : build_empty_side_join_matrix()) {
    INFO("empty-side join case: " << q.label);
    compare_gpu_vs_cpu(q.sql);
  }
}

//===----------------------------------------------------------------------===//
// TPC-H queries
//
// TEST-01/02 (v1.2): each TPC-H TEST_CASE is parameterized on num_gpus ∈ {1, 2}
// via Catch2's GENERATE. The RUN_TPCH_MGPU macro:
//   - picks num_gpus = 1 then 2 (two Catch2 sections per TEST_CASE)
//   - CAPTUREs num_gpus so failures report which variant failed
//   - acquires the matching shared_test_env (integration.yaml for 1,
//     integration-2gpu.yaml for 2) via compare_gpu_vs_cpu_for()
//   - WARN+returns when num_gpus == 2 on a single-GPU host
// This expands each TEST_CASE to run twice; per AUDIT-03, the 2-GPU variant
// MUST execute in the default unit-tests run, so no [.] hide-tag is applied.
//===----------------------------------------------------------------------===//
#define RUN_TPCH_MGPU(...)                                          \
  do {                                                              \
    auto const num_gpus = GENERATE(1, 2);                           \
    CAPTURE(num_gpus);                                              \
    if (!compare_gpu_vs_cpu_for(num_gpus, __VA_ARGS__)) { return; } \
  } while (0)

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 1",
                 "[integration][gpu_execution][TPC-H][Q1]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ1, sirius::test::kTpchQueries[0].float_tolerance);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 1 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q1]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ1, sirius::test::kTpchQueries[0].float_tolerance);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 2",
                 "[integration][gpu_execution][TPC-H][Q2]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ2);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 2 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q2]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ2);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 3",
                 "[integration][gpu_execution][TPC-H][Q3]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ3);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 3 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q3]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ3);
}

// TPC-H Q4 parquet has a pre-existing intermittent flake (see ROADMAP Phase 8
// Success Criterion 2: "Q4 parquet flake policy: retry once per v1.1 precedent,
// not treated as regression"). The retry is scoped to Q4 ONLY — real regressions
// on other queries must fail loudly. We wrap the SAME body shape as RUN_TPCH_MGPU
// but handle any std::exception from compare_gpu_vs_cpu by retrying once with
// a fresh bind_env.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 4",
                 "[integration][gpu_execution][TPC-H][Q4]")
{
  auto const num_gpus = GENERATE(1, 2);
  CAPTURE(num_gpus);
  try {
    if (!compare_gpu_vs_cpu_for(num_gpus, sirius::test::kTpchQ4)) { return; }
  } catch (std::exception const& first_err) {
    WARN(
      "tpch_q4 first attempt failed (pre-existing flake per ROADMAP Phase 8 "
      "Success Criterion 2); retrying once: "
      << first_err.what());
    if (!compare_gpu_vs_cpu_for(num_gpus, sirius::test::kTpchQ4)) { return; }
  }
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 4 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q4]")
{
  auto const num_gpus = GENERATE(1, 2);
  CAPTURE(num_gpus);
  try {
    if (!compare_gpu_vs_cpu_for(num_gpus, sirius::test::kTpchQ4)) { return; }
  } catch (std::exception const& first_err) {
    WARN(
      "tpch_q4 parquet first attempt failed (pre-existing flake per ROADMAP "
      "Phase 8 Success Criterion 2); retrying once: "
      << first_err.what());
    if (!compare_gpu_vs_cpu_for(num_gpus, sirius::test::kTpchQ4)) { return; }
  }
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 5",
                 "[integration][gpu_execution][TPC-H][Q5]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ5);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 5 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q5]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ5);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 6",
                 "[integration][gpu_execution][TPC-H][Q6]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ6);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 6 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q6]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ6);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 7",
                 "[integration][gpu_execution][TPC-H][Q7]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ7);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 7 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q7]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ7);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 8",
                 "[integration][gpu_execution][TPC-H][Q8]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ8);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 8 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q8]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ8);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 9",
                 "[integration][gpu_execution][TPC-H][Q9]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ9);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 9 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q9]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ9);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 10",
                 "[integration][gpu_execution][TPC-H][Q10]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ10);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 10 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q10]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ10);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 11",
                 "[integration][gpu_execution][TPC-H][Q11]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ11);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 11 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q11]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ11);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 12",
                 "[integration][gpu_execution][TPC-H][Q12]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ12);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 12 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q12]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ12);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 13",
                 "[integration][gpu_execution][TPC-H][Q13]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ13);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 13 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q13]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ13);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 14",
                 "[integration][gpu_execution][TPC-H][Q14]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ14);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 14 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q14]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ14);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 15",
                 "[integration][gpu_execution][TPC-H][Q15]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ15);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 15 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q15]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ15);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 16",
                 "[integration][gpu_execution][TPC-H][Q16]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ16);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 16 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q16]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ16);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 17",
                 "[integration][gpu_execution][TPC-H][Q17]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ17);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 17 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q17]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ17);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 18",
                 "[integration][gpu_execution][TPC-H][Q18]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ18);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 18 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q18]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ18);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 19",
                 "[integration][gpu_execution][TPC-H][Q19]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ19);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 19 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q19]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ19);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 20",
                 "[integration][gpu_execution][TPC-H][Q20]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ20);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 20 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q20]")
{
  sirius::test::scoped_recording_log_sink logs{"warn"};
  RUN_TPCH_MGPU(sirius::test::kTpchQ20);
  for (auto const& record : logs.records()) {
    CHECK(record.message.find("RIGHT_DELIM_JOIN") == std::string::npos);
    CHECK(record.message.find("output batch") == std::string::npos);
  }
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 21",
                 "[integration][gpu_execution][TPC-H][Q21]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ21);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 21 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q21]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ21);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - TPC-H Query 22",
                 "[integration][gpu_execution][TPC-H][Q22]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ22);
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 22 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q22]")
{
  RUN_TPCH_MGPU(sirius::test::kTpchQ22);
}

//===----------------------------------------------------------------------===//
// TPC-H SF10 smoke variants (TEST-04)
//
// These TEST_CASEs run TPC-H Q1, Q6, Q12 at SF10 on num_gpus=2. They are
// gated on the SIRIUS_TEST_SF10_PATH env var (skip with WARN if unset) AND
// on >=2 GPUs (WARN+return per Catch2 v2 convention). The views are built on
// top of the SF10 parquet via compare_gpu_vs_cpu_sf10_for which CREATE OR
// REPLACE VIEWs the 8 TPC-H tables after bind_env.
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - tpch_q1_sf10_2gpu",
                 "[integration][tpch_sf10][mgpu-audit][gpu_execution][TPC-H][Q1]")
{
  if (sf10_path().empty()) {
    WARN("SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q1 variant (TEST-04 gate)");
    return;
  }
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("tpch_q1_sf10_2gpu requires >=2 GPUs; skipping");
    return;
  }
  if (!compare_gpu_vs_cpu_sf10_for(
        /*num_gpus=*/2,
        "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, "
        "sum(l_extendedprice) as sum_base_price, "
        "sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
        "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
        "avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, "
        "avg(l_discount) as avg_disc, count(*) as count_order "
        "from lineitem "
        "where l_shipdate <= date '1995-08-19' "
        "group by l_returnflag, l_linestatus "
        "order by l_returnflag, l_linestatus;",
        0.0001f)) {
    return;
  }
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - tpch_q6_sf10_2gpu",
                 "[integration][tpch_sf10][mgpu-audit][gpu_execution][TPC-H][Q6]")
{
  if (sf10_path().empty()) {
    WARN("SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q6 variant (TEST-04 gate)");
    return;
  }
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("tpch_q6_sf10_2gpu requires >=2 GPUs; skipping");
    return;
  }
  if (!compare_gpu_vs_cpu_sf10_for(
        /*num_gpus=*/2,
        "select sum(l_extendedprice * l_discount) as revenue "
        "from lineitem "
        "where l_shipdate >= date '1995-01-01' "
        "and l_shipdate < date '1996-01-01' "
        "and l_discount between 0.07 - 0.01 and 0.07 + 0.01 "
        "and l_quantity < 24;",
        0.0001f)) {
    return;
  }
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - tpch_q12_sf10_2gpu",
                 "[integration][tpch_sf10][mgpu-audit][gpu_execution][TPC-H][Q12]")
{
  if (sf10_path().empty()) {
    WARN("SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q12 variant (TEST-04 gate)");
    return;
  }
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("tpch_q12_sf10_2gpu requires >=2 GPUs; skipping");
    return;
  }
  if (!compare_gpu_vs_cpu_sf10_for(
        /*num_gpus=*/2,
        "select l_shipmode, "
        "sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' "
        "         then 1 else 0 end) as high_line_count, "
        "sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' "
        "         then 1 else 0 end) as low_line_count "
        "from orders, lineitem "
        "where o_orderkey = l_orderkey "
        "and l_shipmode in ('SHIP', 'AIR') "
        "and l_commitdate < l_receiptdate "
        "and l_shipdate < l_commitdate "
        "and l_receiptdate >= date '1995-01-01' "
        "and l_receiptdate < date '1996-01-01' "
        "group by l_shipmode "
        "order by l_shipmode;")) {
    return;
  }
}

// Q11 SF10 exercises materialized CTE with a non-trivial result-set (~8.6K rows).
// Phase 22.3 rationale: the original SF1 Q11 test exercises the CTE planner but
// the CTE materialization is small enough that the validator-detected type
// mismatch (right->types vs producer-shape passthrough) was silently absorbed.
// SF10 makes the bug user-visible — at SF10 sirius_plan_cte declared 2-col
// _types while CTE.execute() forwarded 5-col producer batches. The fraction
// uses 0.0001/SF (= 0.00001 at SF10) per TPC-H spec convention so the result
// has a non-zero rowset that meaningfully validates GROUP BY + HAVING +
// NESTED_LOOP_JOIN downstream of the CTE.
TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - tpch_q11_sf10_2gpu",
                 "[integration][tpch_sf10][mgpu-audit][gpu_execution][TPC-H][Q11]")
{
  if (sf10_path().empty()) {
    WARN("SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q11 variant (TEST-04 gate)");
    return;
  }
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("tpch_q11_sf10_2gpu requires >=2 GPUs; skipping");
    return;
  }
  if (!compare_gpu_vs_cpu_sf10_for(
        /*num_gpus=*/2,
        "select ps_partkey, "
        "sum(ps_supplycost * ps_availqty) as value "
        "from partsupp, supplier, nation "
        "where ps_suppkey = s_suppkey "
        "and s_nationkey = n_nationkey "
        "and n_name = 'GERMANY' "
        "group by ps_partkey "
        "having sum(ps_supplycost * ps_availqty) > ("
        "  select sum(ps_supplycost * ps_availqty) * 0.00001 "
        "  from partsupp, supplier, nation "
        "  where ps_suppkey = s_suppkey "
        "  and s_nationkey = n_nationkey "
        "  and n_name = 'GERMANY'"
        ") "
        "order by value desc;",
        0.01f)) {
    return;
  }
}

//===----------------------------------------------------------------------===//
// GPU_VALUES tests — plan-materialized sources
//
// The GPU_VALUES source operator serves COLUMN_DATA_SCAN (VALUES clauses,
// materialized subqueries, STATISTICS_PROPAGATION constant folds), DUMMY_SCAN
// (constant-only queries), and EMPTY_RESULT (WHERE false). When
// STATISTICS_PROPAGATION is enabled, DuckDB folds ungrouped count(*), MIN,
// and MAX into constant expressions (EXPRESSION_GET -> DUMMY_SCAN), which
// the Sirius planner converts to a GPU_VALUES source. These tests ensure
// those paths work and don't regress.
//===----------------------------------------------------------------------===//

struct scan_task_batch_size_guard {
  duckdb::Connection& con;
  explicit scan_task_batch_size_guard(duckdb::Connection& con, std::size_t size) : con(con)
  {
    con.Query("SET scan_task_batch_size = " + std::to_string(size));
  }
  ~scan_task_batch_size_guard() { con.Query("RESET scan_task_batch_size"); }
};

struct optimizer_disable_guard {
  duckdb::ClientContext& context;
  std::set<duckdb::OptimizerType> original_disabled;

  optimizer_disable_guard(duckdb::ClientContext& context, duckdb::OptimizerType optimizer)
    : context(context),
      original_disabled(duckdb::DBConfig::GetConfig(context).options.disabled_optimizers)
  {
    auto disabled = original_disabled;
    disabled.insert(optimizer);
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = std::move(disabled);
  }

  ~optimizer_disable_guard()
  {
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = std::move(original_disabled);
  }
};

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - empty result (WHERE false)",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where 1=0;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - aggregate over empty result",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select count(*) from nation where 1=0;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - dummy scan (SELECT literal)",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select 42 as x;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - values source",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select b from (values (1), (2), (3)) t(b);");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - count over projection-pruned values preserves cardinality",
                 "[integration][gpu_execution][gpu_values]")
{
  // Prevent DuckDB from folding the aggregate to a constant: UNUSED_COLUMNS
  // can then prune the VALUES payload while preserving its three logical rows.
  // GPU_VALUES must retain those rows even though cuDF cannot represent a
  // positive-row table with zero columns.
  optimizer_disable_guard guard(*con->context, duckdb::OptimizerType::STATISTICS_PROPAGATION);
  compare_gpu_vs_cpu("select count(*) from (values (1), (2), (3)) t(i);");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - boolean values",
                 "[integration][gpu_execution][gpu_values][types]")
{
  compare_gpu_vs_cpu("select b from (values (true), (false), (NULL::BOOLEAN)) t(b);");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - temporal values",
                 "[integration][gpu_execution][gpu_values][types]")
{
  compare_gpu_vs_cpu(
    "select d, ts from (values "
    "(DATE '2024-01-02', TIMESTAMP '2024-01-02 03:04:05.123456'), "
    "(DATE '1999-12-31', TIMESTAMP '1999-12-31 23:59:59.999999'), "
    "(NULL::DATE, NULL::TIMESTAMP)) t(d, ts);");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - DECIMAL32, DECIMAL64, and DECIMAL128 values",
                 "[integration][gpu_execution][gpu_values][types][decimal]")
{
  compare_gpu_vs_cpu(
    "select d32, d64, d128 from (values "
    "(CAST('1234567.89' AS DECIMAL(9,2)), "
    " CAST('12345678901234.5678' AS DECIMAL(18,4)), "
    " CAST('12345678901234567890123456789012.345678' AS DECIMAL(38,6))), "
    "(NULL::DECIMAL(9,2), NULL::DECIMAL(18,4), NULL::DECIMAL(38,6))) "
    "t(d32, d64, d128);");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - values with varchar and nulls",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu(
    "select a, b from (values (1, 'alpha'), (NULL, 'beta'), (3, NULL)) t(a, b) order by a nulls "
    "first;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - values spanning multiple chunks",
                 "[integration][gpu_execution][gpu_values][large_input]")
{
  // > STANDARD_VECTOR_SIZE (2048) rows so the ColumnDataCollection scans
  // multiple DataChunks through the GPU_VALUES staging path. Disable
  // statistics propagation so the aggregate cannot be folded before staging.
  optimizer_disable_guard guard(*con->context, duckdb::OptimizerType::STATISTICS_PROPAGATION);
  std::string query = "select count(*), min(i), max(i) from (values (0)";
  for (int i = 1; i < 5000; i++) {
    query += ", (" + std::to_string(i) + ")";
  }
  query += ") t(i);";
  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - values source fanned out to multiple consumers",
                 "[integration][gpu_execution][gpu_values]")
{
  // A VALUES-backed CTE referenced twice fans the GPU_VALUES output out to
  // multiple downstream data repositories.
  compare_gpu_vs_cpu(
    "with t(b) as (values (1), (2), (3)) select a.b, c.b from t a join t c using (b);");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - values joined with base table",
                 "[integration][gpu_execution][gpu_values]")
{
  // GPU_VALUES and GPU_SCAN sources in one plan: exercises kickoff when the
  // task scheduler only schedules the first scan-like source directly.
  compare_gpu_vs_cpu(
    "select n.n_name from nation n join (values (0), (1), (2)) t(k) on n.n_nationkey = t.k order "
    "by n.n_name;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - oversized values source falls back before GPU materialization",
                 "[integration][gpu_execution][gpu_values][fallback][large_input]")
{
  // GPU_VALUES is intentionally a single-table source. Force a tiny source
  // cap and verify an oversized collection is refused during planning, where
  // transparent execution can safely replay it on DuckDB's streaming path.
  scan_task_batch_size_guard guard(*con, 64);
  auto before = sirius::test::get_transparent_execution_stats(*con);

  std::string const payload(128, 'x');
  auto result = con->Query("select sum(length(s)) from (values ('" + payload + "'), ('" + payload +
                           "')) t(s);");
  REQUIRE(result);
  if (result->HasError()) {
    UNSCOPED_INFO("oversized VALUES fallback error: " << result->GetError());
  }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->GetValue(0, 0).GetValue<int64_t>() == 256);

  auto after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before,
                                                    after,
                                                    /*expected_rebind_delta=*/0,
                                                    /*expected_fallback_delta=*/1,
                                                    /*expected_execution_delta=*/0);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - unsupported HUGEINT values fall back",
                 "[integration][gpu_execution][gpu_values][fallback][types]")
{
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query(
    "select x from (values "
    "(CAST('9223372036854775808' AS HUGEINT)), "
    "(CAST('-9223372036854775809' AS HUGEINT))) t(x) order by x;");

  REQUIRE(result);
  if (result->HasError()) {
    UNSCOPED_INFO("unsupported HUGEINT VALUES fallback error: " << result->GetError());
  }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 2);
  REQUIRE(result->GetValue(0, 0).ToString() == "-9223372036854775809");
  REQUIRE(result->GetValue(0, 1).ToString() == "9223372036854775808");

  auto after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before,
                                                    after,
                                                    /*expected_rebind_delta=*/0,
                                                    /*expected_fallback_delta=*/1,
                                                    /*expected_execution_delta=*/0);
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped count(*)",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select count(*) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped count(*) parquet",
                 "[integration][gpu_execution][parquet][gpu_values]")
{
  compare_gpu_vs_cpu("select count(*) from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped min",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select min(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped min parquet",
                 "[integration][gpu_execution][parquet][gpu_values]")
{
  compare_gpu_vs_cpu("select min(l_orderkey) from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped max",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select max(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped max parquet",
                 "[integration][gpu_execution][parquet][gpu_values]")
{
  compare_gpu_vs_cpu("select max(l_orderkey) from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped min and max",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select min(n_nationkey), max(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped min and max parquet",
                 "[integration][gpu_execution][parquet][gpu_values]")
{
  compare_gpu_vs_cpu("select min(l_orderkey), max(l_orderkey) from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - ungrouped count(*) with min and max",
                 "[integration][gpu_execution][gpu_values]")
{
  compare_gpu_vs_cpu("select count(*), min(n_nationkey), max(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - ungrouped count(*) with min and max parquet",
                 "[integration][gpu_execution][parquet][gpu_values]")
{
  compare_gpu_vs_cpu("select count(*), min(l_orderkey), max(l_orderkey) from lineitem;");
}

//===----------------------------------------------------------------------===//
// pin_table tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - pin_table gpu tier scan and aggregate",
                 "[integration][gpu_execution][parquet][pin_table]")
{
  auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
  auto pin_query =
    "CALL pin_table('" + parquet_dir.string() + "/lineitem.parquet', tier='gpu', name='lineitem');";
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, count(*), sum(l_quantity) "
    "from lineitem group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - pin_table gpu tier filter drops pure-filter column",
                 "[integration][gpu_execution][parquet][pin_table][filter]")
{
  auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
  auto pin_query =
    "CALL pin_table('" + parquet_dir.string() + "/lineitem.parquet', tier='gpu', name='lineitem');";
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  // l_linenumber is referenced only by the predicate, so the cached-scan post-filter fold must
  // gather just l_orderkey and never materialize l_linenumber (#987).
  compare_gpu_vs_cpu(
    "select l_orderkey from lineitem where l_linenumber = 1 and l_orderkey < 1000;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - pin_table host tier scan and aggregate",
                 "[integration][gpu_execution][parquet][pin_table_host]")
{
  auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
  auto pin_query   = "CALL pin_table('" + parquet_dir.string() +
                   "/lineitem.parquet', tier='host', name='lineitem');";
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, count(*), sum(l_quantity) "
    "from lineitem group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - pin_table host tier filter drops pure-filter column",
                 "[integration][gpu_execution][parquet][pin_table_host][filter]")
{
  auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
  auto pin_query   = "CALL pin_table('" + parquet_dir.string() +
                   "/lineitem.parquet', tier='host', name='lineitem');";
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  // l_linenumber is referenced only by the predicate, so the cached-scan post-filter fold must
  // gather just l_orderkey and never materialize l_linenumber (#987).
  compare_gpu_vs_cpu(
    "select l_orderkey from lineitem where l_linenumber = 1 and l_orderkey < 1000;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

// duckdb-native pin: pin a table in the attached tpch .db (format='duckdb',
// table='lineitem'), then a SELECT over the same table must be served from the
// pinned cache (matched by DataTable* identity), bypassing the native scan.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - pin_table duckdb-native gpu tier scan and aggregate",
                 "[integration][gpu_execution][duckdb_native][pin_table_duckdb]")
{
  auto pin_query  = std::string("CALL pin_table(format='duckdb', name='lineitem', tier='gpu');");
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, count(*), sum(l_quantity) "
    "from lineitem group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

// duckdb-native host-tier pin: same as above but the pinned columns live in
// pinned host memory; the cached host batches are sliced + served on hit.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - pin_table duckdb-native host tier scan and aggregate",
                 "[integration][gpu_execution][duckdb_native][pin_table_duckdb_host]")
{
  auto pin_query  = std::string("CALL pin_table(format='duckdb', name='lineitem', tier='host');");
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, count(*), sum(l_quantity) "
    "from lineitem group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

// Overflow (big-string) refusal: strings at/over GetStringBlockLimit (4 KB at the
// default block size) live in overflow blocks the GPU scan cannot decode. The query
// must route to DuckDB CPU and return correct results; pin_table must fail cleanly.
TEST_CASE_METHOD(GPUExecutionDuckDBFixture,
                 "gpu_execution - duckdb-native overflow strings fall back to CPU",
                 "[integration][gpu_execution][duckdb_native][overflow_string]")
{
  auto db_file = fs::temp_directory_path() / "sirius_overflow_strings_test.db";
  fs::remove(db_file);
  fs::remove(fs::path(db_file.string() + ".wal"));

  auto exec = [&](const std::string& q) {
    auto result = con->Query(q);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  };

  exec("ATTACH '" + db_file.string() + "' AS ovf;");
  exec(
    "CREATE TABLE ovf.main.bigstr AS SELECT range AS id, "
    "CASE WHEN range = 7 THEN repeat('x', 5000) ELSE 'short_' || range END AS s "
    "FROM range(0, 1000);");
  exec("CHECKPOINT ovf;");

  // The 5000-char string must come back intact. These queries are DESIGNED to fall
  // back, so assert the fallback counters positively instead of compare_gpu_vs_cpu
  // (whose contract demands a successful GPU rebind).
  auto stats_before = sirius::test::get_transparent_execution_stats(*con);

  auto check = con->Query("SELECT max(length(s)), count(*) FROM ovf.main.bigstr;");
  REQUIRE(check);
  if (check->HasError()) { UNSCOPED_INFO("intercepted query error: " << check->GetError()); }
  REQUIRE_FALSE(check->HasError());
  REQUIRE(check->GetValue(0, 0).GetValue<int64_t>() == 5000);
  REQUIRE(check->GetValue(1, 0).GetValue<int64_t>() == 1000);

  auto intact = con->Query("SELECT s = repeat('x', 5000) FROM ovf.main.bigstr WHERE id = 7;");
  REQUIRE(intact);
  REQUIRE_FALSE(intact->HasError());
  REQUIRE(intact->GetValue(0, 0).GetValue<bool>());

  auto stats_after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(stats_before,
                                                    stats_after,
                                                    /*expected_rebind_delta=*/0,
                                                    /*expected_fallback_delta=*/2,
                                                    /*expected_execution_delta=*/0);

  // Pinning the table must fail with the refusal reason, not cache garbage.
  auto pin_result =
    con->Query("CALL pin_table(format='duckdb', name='ovf.main.bigstr', tier='gpu');");
  REQUIRE(pin_result);
  REQUIRE(pin_result->HasError());
  REQUIRE(pin_result->GetError().find("overflow") != std::string::npos);

  exec("DETACH ovf;");
  fs::remove(db_file);
}

// Pin a column subset (cols=[...]) and then run a query that requests a strict
// subset of those pinned columns — it must be served from the cache. A miss would
// fall through to the separate (non-cached) scan path, so a passing run also
// confirms the cache hit; compare_gpu_vs_cpu validates the served data.
TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - pin_table column subset serves a subset query",
                 "[integration][gpu_execution][parquet][pin_table_cols_subset]")
{
  auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
  auto pin_query   = "CALL pin_table('" + parquet_dir.string() +
                   "/lineitem.parquet', tier='gpu', name='lineitem_subset', "
                   "cols=['l_orderkey', 'l_returnflag', 'l_linestatus', 'l_quantity']);";
  auto pin_result = con->Query(pin_query);
  REQUIRE(pin_result);
  if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
  REQUIRE_FALSE(pin_result->HasError());

  // Requests only l_returnflag, l_linestatus, l_quantity — a strict subset of the
  // pinned columns (l_orderkey is pinned but unused here).
  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, count(*), sum(l_quantity) "
    "from lineitem group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;");

  auto unpin_result = con->Query("CALL unpin_table('lineitem_subset');");
  REQUIRE(unpin_result);
  REQUIRE_FALSE(unpin_result->HasError());
}

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - standalone filter drops pure-filter column via projection map",
                 "[integration][gpu_execution][parquet][filter][projection_map]")
{
  // A HAVING predicate over an aggregate cannot be pushed below the aggregate, so it materializes
  // as a standalone LogicalFilter above it. The aggregate it filters on is dropped from the output,
  // so DuckDB attaches a projection_map to that filter. The filter must gather only the projected
  // columns and never materialize the aggregate(s) referenced only by the predicate (#987) — the
  // trailing projection is folded into the filter's select() rather than emitted as its own op.

  // Drops one pure-filter aggregate (sum(l_quantity)); keeps only the group key l_orderkey.
  compare_gpu_vs_cpu(
    "select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 100;");

  // Drops two pure-filter aggregates; keeps a two-column group-key prefix.
  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus from lineitem group by l_returnflag, l_linestatus "
    "having sum(l_quantity) > 100 and count(*) > 5;");
}
