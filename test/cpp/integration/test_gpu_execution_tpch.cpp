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

#include <catch.hpp>
#include <duckdb.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
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
  const char* env = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH");
  auto db_path    = env ? fs::path(env) : fs::path(__FILE__).parent_path() / "integration.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

/**
 * @brief Catch2 test fixture for GPU execution tests.
 *
 * Initializes a DuckDB instance with the integration.cfg config and provides
 * a compare_gpu_vs_cpu method for validating GPU execution against CPU results.
 */
class GPUExecutionFixture {
 public:
  GPUExecutionFixture()
  {
    // Set up environment variable for config file
    auto cfg_path = fs::path(__FILE__).parent_path() / "integration.cfg";
    REQUIRE(fs::exists(cfg_path));
    setenv("SIRIUS_CONFIG_FILE", cfg_path.string().c_str(), 1);

    // Initialize DuckDB with integration database
    db  = std::make_unique<duckdb::DuckDB>(get_tpch_db_path().string());
    con = std::make_unique<duckdb::Connection>(*db);
  }

  ~GPUExecutionFixture() = default;

  /**
   * @brief Run a query through gpu_execution and through DuckDB CPU, then compare results.
   *
   * Values are compared as strings via Value::ToString() which normalizes type differences
   * (e.g., HUGEINT vs BIGINT both render "50"). Row order is ignored by collecting rows
   * as sorted sets of string tuples.
   */
  static bool is_floating_point(duckdb::LogicalTypeId id)
  {
    return id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
  {
    // Disable fallback so GPU errors are not silently hidden
    con->Query("SET enable_fallback_check = true;");

    // Run on GPU
    auto gpu_sql    = "CALL gpu_execution(\"" + query + "\")";
    auto gpu_result = con->Query(gpu_sql);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("gpu_execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());

    // Run on CPU (plain DuckDB)
    auto cpu_result = con->Query(query);
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    // Compare dimensions
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    // Use DuckDB to sort both result sets by all columns for deterministic comparison.
    // This avoids lexicographic vs numeric sort issues.
    auto ncols               = gpu_result->ColumnCount();
    std::string order_clause = " ORDER BY ";
    for (duckdb::idx_t c = 0; c < ncols; c++) {
      if (c > 0) order_clause += ", ";
      order_clause += std::to_string(c + 1);
    }

    // Strip trailing semicolons from query for subquery wrapping
    auto clean_query = query;
    while (!clean_query.empty() && (clean_query.back() == ';' || clean_query.back() == ' '))
      clean_query.pop_back();

    auto gpu_sorted =
      con->Query("SELECT * FROM gpu_execution(\"" + clean_query + "\")" + order_clause);
    auto cpu_sorted = con->Query("SELECT * FROM (" + clean_query + ") t" + order_clause);
    REQUIRE(gpu_sorted);
    if (gpu_sorted->HasError()) { UNSCOPED_INFO("gpu sorted error: " << gpu_sorted->GetError()); }
    REQUIRE_FALSE(gpu_sorted->HasError());
    REQUIRE(cpu_sorted);
    if (cpu_sorted->HasError()) { UNSCOPED_INFO("cpu sorted error: " << cpu_sorted->GetError()); }
    REQUIRE_FALSE(cpu_sorted->HasError());

    for (duckdb::idx_t r = 0; r < gpu_sorted->RowCount(); r++) {
      for (duckdb::idx_t c = 0; c < gpu_sorted->ColumnCount(); c++) {
        auto gpu_value = gpu_sorted->GetValue(c, r);
        auto cpu_value = cpu_sorted->GetValue(c, r);

        if (float_tolerance.has_value() && is_floating_point(gpu_value.type().id())) {
          double gpu_d = gpu_value.GetValue<double>();
          double cpu_d = cpu_value.GetValue<double>();
          double diff  = std::fabs(gpu_d - cpu_d);
          if (diff > static_cast<double>(float_tolerance.value())) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " float mismatch: GPU=[" << gpu_d
                                 << "] CPU=[" << cpu_d << "] diff=" << diff
                                 << " tolerance=" << float_tolerance.value());
          }
          REQUIRE(diff <= static_cast<double>(float_tolerance.value()));
        } else {
          auto gpu_str = gpu_value.ToString();
          auto cpu_str = cpu_value.ToString();
          if (gpu_str != cpu_str) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_str << "] CPU=["
                                 << cpu_str << "]");
          }
          REQUIRE(gpu_str == cpu_str);
        }
      }
    }
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

//===----------------------------------------------------------------------===//
// Scan tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - scan single column",
                 "[integration][gpu_execution][scan]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - scan multiple columns",
                 "[integration][gpu_execution][scan]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - scan region table",
                 "[integration][gpu_execution][scan]")
{
  compare_gpu_vs_cpu("select r_regionkey from region;");
}

//===----------------------------------------------------------------------===//
// Projection tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - projection add",
                 "[integration][gpu_execution][projection]")
{
  compare_gpu_vs_cpu("select n_nationkey + n_regionkey as total from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - projection multiply",
                 "[integration][gpu_execution][projection]")
{
  compare_gpu_vs_cpu("select n_nationkey * 2 as doubled, n_regionkey from nation;");
}

//===----------------------------------------------------------------------===//
// Filter tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - filter equality",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - filter greater than",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey > 2;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - filter not equal",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select r_regionkey from region where r_regionkey != 3;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - filter with projection",
                 "[integration][gpu_execution][filter]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation where n_regionkey = 0;");
}

//===----------------------------------------------------------------------===//
// Ungrouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - ungrouped min max",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu("select min(n_regionkey), max(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - ungrouped min with filter",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu("select min(n_nationkey) from nation where n_regionkey = 1;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - ungrouped sum count",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu("select sum(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - ungrouped all agg functions",
                 "[integration][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu(
    "select sum(n_regionkey), min(n_nationkey), max(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - ungrouped avg integer",
                 "[integration][gpu_execution][aggregate][avg]")
{
  compare_gpu_vs_cpu("select avg(n_nationkey) from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - ungrouped avg decimal",
                 "[integration][gpu_execution][aggregate][avg]")
{
  compare_gpu_vs_cpu("select avg(l_quantity), avg(l_discount) from lineitem;");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - single group by key: min max, sum, count(*)",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, min(c_custkey), max(c_custkey), sum(c_custkey), count(*) "
    "from customer group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - single group by key: min max, count string ",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, min(C_NAME), max(C_NAME), count(C_NAME) from customer "
    "group by c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - two group by key: min max, but not showing the group by keys",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select min(c_custkey), max(c_custkey) from customer group by c_nationkey, c_mktsegment;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - two group keys and noaggregations",
                 "[integration][gpu_execution][grouped_aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, c_mktsegment from customer group by c_mktsegment, c_nationkey;");
}

//===----------------------------------------------------------------------===//
// Limit tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - limit",
                 "[integration][gpu_execution][limit]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation limit 10;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - limit with filter",
                 "[integration][gpu_execution][limit]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation where n_regionkey = 1 limit 3;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - limit on large table",
                 "[integration][gpu_execution][limit][limit_multi_batch]")
{
  // lineitem has ~6K rows at SF-0.01, ensuring multiple batches.
  // A limit of 100 should produce exactly 100 rows regardless of batch count.
  compare_gpu_vs_cpu("select l_orderkey from lineitem limit 100");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - limit with offset on large table",
                 "[integration][gpu_execution][limit][limit_multi_batch]")
{
  compare_gpu_vs_cpu("select l_orderkey, l_partkey from lineitem limit 50 offset 200;");
}

//===----------------------------------------------------------------------===//
// Join tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic inner join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic inner join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic inner join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic inner join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n join customer c on "
    "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from nation n right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
    "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped inner join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped inner join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped inner join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped inner join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c join nation n on "
    "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped left join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 0",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 1",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 2",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 3",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 0 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 1 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 2 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
    "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped right join 3 making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
    "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic full outer join",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
    "on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic full outer join making nulls",
                 "[integration][gpu_execution][join]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
    "on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left semi join",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic left semi join 2",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey from nation n semi join region r on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right semi join",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic right semi join 2",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic semi join 3",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey "
    "from nation n semi join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic semi join 4",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey  from nation n "
    "semi join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic semi join 5",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_name from nation n semi join customer c "
    "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic semi join misfit 0",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_nationkey, n.n_regionkey  "
    "from nation n semi join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - basic semi join mistit 1",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select n.n_regionkey  from nation n "
    "semi join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped semi join 0",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c semi join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped semi join 1",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "semi join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped semi join misfit 0",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_nationkey, c.c_custkey, c.c_name  "
    "from customer c semi join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - swapped semi join misfit 1",
                 "[integration][gpu_execution][semijoin]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name  from customer c "
    "semi join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - bigger inner join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l join orders o on l.l_orderkey = o.o_orderkey order by "
    "l.l_orderkey, l.l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - bigger left join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l left join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - bigger right join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l right join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - bigger full outer join",
                 "[integration][gpu_execution][bigger_join]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l full outer join orders o on l.l_orderkey = "
    "o.o_orderkey order by l.l_orderkey, l.l_linenumber;");
}

//===----------------------------------------------------------------------===//
// Disabled tests - known issues
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - two group by key: min max, sum, count of doubles",
                 "[.][integration_disabled][gpu_execution][aggregate]")
{
  compare_gpu_vs_cpu(
    "select c_nationkey, c_mktsegment, min(C_ACCTBAL), max(C_ACCTBAL), sum(C_ACCTBAL), "
    "count(C_ACCTBAL) from customer group by c_nationkey, c_mktsegment;");
}

// Empty result set: "Port default not found in operator RESULT_COLLECTOR"
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - filter returns empty result",
                 "[.][integration_disabled][gpu_execution]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation where n_regionkey = 99;");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by count",
                 "[integration][gpu_execution][group_by]")
{
  compare_gpu_vs_cpu("select n_regionkey, count(*) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by min max count",
                 "[integration][gpu_execution][group_by]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, min(n_nationkey), max(n_nationkey), count(n_nationkey) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by avg integer",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu("select n_regionkey, avg(n_nationkey) from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by avg with other aggregates",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, avg(n_nationkey), sum(n_nationkey), count(*) "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by avg decimal",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_returnflag, avg(l_quantity), avg(l_discount) "
    "from lineitem group by l_returnflag;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by sum avg on lineitem",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*) "
    "from lineitem group by l_returnflag, l_linestatus;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by min, max, avg on decimal on lineitem",
                 "[integration][gpu_execution][group_by][avg]")
{
  compare_gpu_vs_cpu(
    "select l_tax, min(l_extendedprice), max(l_extendedprice), avg(l_extendedprice)"
    "from lineitem group by l_tax;",
    0.0001);
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - group by min, max, avg, sum on decimal on lineitem",
                 "[integration][gpu_execution][group_by][avg]")
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

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by",
                 "[integration][gpu_execution][order_by]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_regionkey from nation order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by column not in select",
                 "[integration][gpu_execution][order_by][order_by_proj]")
{
  compare_gpu_vs_cpu("select n_nationkey from nation order by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by column not in select lineitem",
                 "[integration][gpu_execution][order_by][order_by_proj]")
{
  compare_gpu_vs_cpu("select l_orderkey from lineitem order by l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by multipartition",
                 "[integration][gpu_execution][order_by]")
{
  // Force small partition size (1 KB) so lineitem data is split into multiple partitions
  con->Query("SET max_sort_partition_bytes = 1024;");

  std::string query = "select l_orderkey, l_partkey from lineitem order by l_orderkey";

  // Run on GPU
  auto gpu_result = con->Query("CALL gpu_execution('" + query + "')");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("gpu error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());

  // Run on CPU
  auto cpu_result = con->Query(query + ";");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  // Verify dimensions match
  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  // With 600K rows and 1KB max partition, we must have many partitions.
  // Verify the data is non-trivially large (ensures partitioning actually happened).
  REQUIRE(gpu_result->RowCount() > 1000);
  // Sort both result sets for deterministic comparison
  auto gpu_sorted = con->Query("SELECT * FROM gpu_execution('" + query + "') ORDER BY 1, 2");
  auto cpu_sorted = con->Query("SELECT * FROM (" + query + ") t ORDER BY 1, 2");
  REQUIRE(gpu_sorted);
  REQUIRE_FALSE(gpu_sorted->HasError());
  REQUIRE(cpu_sorted);
  REQUIRE_FALSE(cpu_sorted->HasError());

  // Compare every cell
  duckdb::idx_t mismatches = 0;
  for (duckdb::idx_t r = 0; r < gpu_sorted->RowCount(); r++) {
    for (duckdb::idx_t c = 0; c < gpu_sorted->ColumnCount(); c++) {
      auto gpu_val = gpu_sorted->GetValue(c, r).ToString();
      auto cpu_val = cpu_sorted->GetValue(c, r).ToString();
      if (gpu_val != cpu_val) {
        if (mismatches < 5) {
          UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_val << "] CPU=["
                               << cpu_val << "]");
        }
        mismatches++;
      }
      REQUIRE(gpu_val == cpu_val);
    }
  }
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by multiple columns",
                 "[integration][gpu_execution][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_linenumber, l_quantity from lineitem order by l_orderkey, l_linenumber;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by desc",
                 "[integration][gpu_execution][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_partkey, l_suppkey from lineitem order by l_partkey desc;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by many selected columns",
                 "[integration][gpu_execution][order_by]")
{
  con->Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity "
    "from lineitem order by l_suppkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by with decimal column",
                 "[integration][gpu_execution][order_by][order_by_types]")
{
  auto gpu_result = con->Query(
    "CALL gpu_execution('select o_orderkey, o_totalprice from orders order by o_orderkey')");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) {
    std::cerr << "DECIMAL error: " << gpu_result->GetError() << std::endl;
  }
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->RowCount() > 0);
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - scan lineitem with varchar column",
                 "[integration][gpu_execution][varchar_scan_lineitem]")
{
  compare_gpu_vs_cpu("select l_orderkey, l_shipinstruct from lineitem;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by lineitem with short varchar column",
                 "[integration][gpu_execution][order_by][varchar_order]")
{
  compare_gpu_vs_cpu(
    "select l_orderkey, l_shipinstruct, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by lineitem with long varchar column",
                 "[integration][gpu_execution][order_by][varchar_order]")
{
  compare_gpu_vs_cpu(
    "select l_orderkey, l_comment, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - scan with varchar column",
                 "[integration][gpu_execution][order_by_types][varchar]")
{
  compare_gpu_vs_cpu("select n_nationkey, n_name from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - order by with varchar column",
                 "[integration][gpu_execution][order_by][order_by_types]")
{
  auto gpu_result =
    con->Query("CALL gpu_execution('select n_nationkey, n_name from nation order by n_nationkey')");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) {
    std::cerr << "VARCHAR order by error: " << gpu_result->GetError() << std::endl;
  }
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->RowCount() > 0);
  std::cerr << "VARCHAR order by: " << gpu_result->RowCount() << " rows OK" << std::endl;
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - cast integer to decimal preserves scale",
                 "[integration][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu("select n_nationkey, cast(n_nationkey as Decimal(18,2)) as d from nation;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - cast integer to decimal with aggregation",
                 "[integration][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, max(cast(n_nationkey as Decimal(18,2))) as max_d "
    "from nation group by n_regionkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - cast to decimal different scales",
                 "[integration][gpu_execution][cast][decimal]")
{
  compare_gpu_vs_cpu(
    "select cast(n_nationkey as Decimal(9,0)) as d0, "
    "cast(n_nationkey as Decimal(9,4)) as d4 from nation;");
}

// Disabled: avg() in grouped aggregates not yet supported (separate PR)
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - issue 227 cast decimal with avg and group by",
                 "[.][integration_disabled][gpu_execution][cast][decimal]")
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
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: single group key",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;");
}

// count(distinct n_name): n_name is unique per nation, so same result as above.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: string column",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_name) from nation group by n_regionkey;");
}

// count(distinct) mixed with other aggregations in the same grouped aggregate.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: mixed with min and count",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey), min(n_nationkey), count(*) "
    "from nation group by n_regionkey;");
}

// Larger table: customer (15000 rows), two group-by keys.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: larger table two group keys",
                 "[integration][gpu_execution][group_by][count_distinct]")
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

struct PartitionSizeGuard {
  explicit PartitionSizeGuard(duckdb::idx_t override_size)
  {
    sirius::op::sirius_physical_partition::set_partition_size(override_size);
  }
  ~PartitionSizeGuard() { sirius::op::sirius_physical_partition::reset_partition_size(); }
};

// nation (25 rows) with partition_size=5 → ceil(25/5) = 5 partitions.
// count(distinct n_nationkey) per region must still equal 5.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: multi-partition forced, single group key",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  PartitionSizeGuard guard(5);
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;");
}

// customer (15000 rows) with partition_size=1000 → 15 partitions.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: multi-partition forced, customer table",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  PartitionSizeGuard guard(1000);
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment) from customer group by c_nationkey;");
}

// Mixed aggregations across multiple forced partitions.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: multi-partition forced, mixed aggregations",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  PartitionSizeGuard guard(1000);
  compare_gpu_vs_cpu(
    "select c_nationkey, count(distinct c_mktsegment), min(c_custkey), count(*) "
    "from customer group by c_nationkey;");
}

// ---------------------------------------------------------------------------
// Multi-column COUNT(DISTINCT) integration tests
// count(distinct (col1, col2)) counts distinct combinations, not individual values.
// ---------------------------------------------------------------------------

// nation: 25 rows, 5 unique (n_nationkey, n_name) combos per region.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: multi-column struct",
                 "[integration][gpu_execution][group_by][count_distinct]")
{
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct (n_nationkey, n_name)) from nation group by n_regionkey;");
}

// Multi-column count distinct with a forced multi-partition execution.
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - count distinct: multi-column struct, multi-partition forced",
                 "[integration][gpu_execution][group_by][count_distinct][multi_partition]")
{
  PartitionSizeGuard guard(5);
  compare_gpu_vs_cpu(
    "select n_regionkey, count(distinct (n_nationkey, n_name)) from nation group by n_regionkey;");
}

//===----------------------------------------------------------------------===//
// Top N / Join tests (disabled)
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - top n",
                 "[.][integration_disabled][gpu_execution][top_n]")
{
  compare_gpu_vs_cpu(
    "select n_nationkey, n_regionkey from nation order by n_regionkey desc limit 5;");
}

//===----------------------------------------------------------------------===//
// TPC-H queries
//===----------------------------------------------------------------------===//
TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 1",
                 "[integration][gpu_execution][TPC-H][Q1]")
{
  compare_gpu_vs_cpu(
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
    0.00001f);
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 2",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q2]")
{
  compare_gpu_vs_cpu(
    "select s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, "
    "s.s_address, s.s_phone, s.s_comment "
    "from part p, supplier s, partsupp ps, nation n, region r "
    "where p.p_partkey = ps.ps_partkey and s.s_suppkey = ps.ps_suppkey "
    "and p.p_size = 41 and p.p_type like '%NICKEL' "
    "and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey "
    "and r.r_name = 'EUROPE' "
    "and ps.ps_supplycost = ("
    "  select min(ps.ps_supplycost) "
    "  from partsupp ps, supplier s, nation n, region r "
    "  where p.p_partkey = ps.ps_partkey and s.s_suppkey = ps.ps_suppkey "
    "  and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey "
    "  and r.r_name = 'EUROPE'"
    ") "
    "order by s.s_acctbal desc, n.n_name, s.s_name, p.p_partkey "
    "limit 100;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 3",
                 "[integration][gpu_execution][TPC-H][Q3]")
{
  compare_gpu_vs_cpu(
    "select l.l_orderkey, "
    "sum(l.l_extendedprice * (1 - l.l_discount)) as revenue, "
    "o.o_orderdate, o.o_shippriority "
    "from customer c, orders o, lineitem l "
    "where c.c_mktsegment = 'HOUSEHOLD' and c.c_custkey = o.o_custkey "
    "and l.l_orderkey = o.o_orderkey "
    "and o.o_orderdate < date '1995-03-25' "
    "and l.l_shipdate > date '1995-03-25' "
    "group by l.l_orderkey, o.o_orderdate, o.o_shippriority "
    "order by revenue desc, o.o_orderdate "
    "limit 10;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 4",
                 "[integration][gpu_execution][TPC-H][Q4]")
{
  compare_gpu_vs_cpu(
    "select o.o_orderpriority, count(*) as order_count "
    "from orders o "
    "where o.o_orderdate >= date '1996-10-01' "
    "and o.o_orderdate < date '1997-01-01' "
    "and exists ("
    "  select * from lineitem l "
    "  where l.l_orderkey = o.o_orderkey "
    "  and l.l_commitdate < l.l_receiptdate"
    ") "
    "group by o.o_orderpriority "
    "order by o.o_orderpriority;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 5",
                 "[integration][gpu_execution][TPC-H][Q5]")
{
  compare_gpu_vs_cpu(
    "select n.n_name, "
    "sum(l.l_extendedprice * (1 - l.l_discount)) as revenue "
    "from orders o, lineitem l, supplier s, nation n, region r, customer c "
    "where c.c_custkey = o.o_custkey and l.l_orderkey = o.o_orderkey "
    "and l.l_suppkey = s.s_suppkey and c.c_nationkey = s.s_nationkey "
    "and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey "
    "and r.r_name = 'EUROPE' "
    "and o.o_orderdate >= date '1997-01-01' "
    "and o.o_orderdate < date '1998-01-01' "
    "group by n.n_name "
    "order by revenue desc;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 6",
                 "[integration][gpu_execution][TPC-H][Q6]")
{
  compare_gpu_vs_cpu(
    "select sum(l_extendedprice * l_discount) as revenue "
    "from lineitem "
    "where l_shipdate >= date '1997-01-01' "
    "and l_shipdate < date '1998-01-01' "
    "and l_discount between 0.03 - 0.01 and 0.03 + 0.01 "
    "and l_quantity < 24;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 7",
                 "[integration][gpu_execution][TPC-H][Q7]")
{
  compare_gpu_vs_cpu(
    "select supp_nation, cust_nation, l_year, sum(volume) as revenue "
    "from ("
    "  select n1.n_name as supp_nation, n2.n_name as cust_nation, "
    "  extract(year from l.l_shipdate) as l_year, "
    "  l.l_extendedprice * (1 - l.l_discount) as volume "
    "  from supplier s, lineitem l, orders o, customer c, nation n1, nation n2 "
    "  where s.s_suppkey = l.l_suppkey and o.o_orderkey = l.l_orderkey "
    "  and c.c_custkey = o.o_custkey and s.s_nationkey = n1.n_nationkey "
    "  and c.c_nationkey = n2.n_nationkey "
    "  and ((n1.n_name = 'EGYPT' and n2.n_name = 'UNITED STATES') "
    "    or (n1.n_name = 'UNITED STATES' and n2.n_name = 'EGYPT')) "
    "  and l.l_shipdate between date '1995-01-01' and date '1996-12-31'"
    ") as shipping "
    "group by supp_nation, cust_nation, l_year "
    "order by supp_nation, cust_nation, l_year;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 8",
                 "[integration][gpu_execution][TPC-H][Q8]")
{
  compare_gpu_vs_cpu(
    "select o_year, "
    "sum(case when nation = 'EGYPT' then volume else 0 end) / sum(volume) as mkt_share "
    "from ("
    "  select extract(year from o.o_orderdate) as o_year, "
    "  l.l_extendedprice * (1 - l.l_discount) as volume, "
    "  n2.n_name as nation "
    "  from lineitem l, part p, supplier s, orders o, customer c, "
    "  nation n1, nation n2, region r "
    "  where p.p_partkey = l.l_partkey and s.s_suppkey = l.l_suppkey "
    "  and l.l_orderkey = o.o_orderkey and o.o_custkey = c.c_custkey "
    "  and c.c_nationkey = n1.n_nationkey and n1.n_regionkey = r.r_regionkey "
    "  and r.r_name = 'MIDDLE EAST' and s.s_nationkey = n2.n_nationkey "
    "  and o.o_orderdate between date '1995-01-01' and date '1996-12-31' "
    "  and p.p_type = 'PROMO BRUSHED COPPER'"
    ") as all_nations "
    "group by o_year "
    "order by o_year;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 9",
                 "[integration][gpu_execution][TPC-H][Q9]")
{
  compare_gpu_vs_cpu(
    "select nation, o_year, sum(amount) as sum_profit "
    "from ("
    "  select n.n_name as nation, "
    "  extract(year from o.o_orderdate) as o_year, "
    "  l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity as amount "
    "  from part p, supplier s, lineitem l, partsupp ps, orders o, nation n "
    "  where s.s_suppkey = l.l_suppkey and ps.ps_suppkey = l.l_suppkey "
    "  and ps.ps_partkey = l.l_partkey and p.p_partkey = l.l_partkey "
    "  and o.o_orderkey = l.l_orderkey and s.s_nationkey = n.n_nationkey "
    "  and p.p_name like '%yellow%'"
    ") as profit "
    "group by nation, o_year "
    "order by nation, o_year desc;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 10",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q10]")
{
  compare_gpu_vs_cpu(
    "select c.c_custkey, c.c_name, "
    "sum(l.l_extendedprice * (1 - l.l_discount)) as revenue, "
    "c.c_acctbal, n.n_name, c.c_address, c.c_phone, c.c_comment "
    "from customer c, orders o, lineitem l, nation n "
    "where c.c_custkey = o.o_custkey and l.l_orderkey = o.o_orderkey "
    "and o.o_orderdate >= date '1994-03-01' "
    "and o.o_orderdate < date '1994-06-01' "
    "and l.l_returnflag = 'R' "
    "and c.c_nationkey = n.n_nationkey "
    "group by c.c_custkey, c.c_name, c.c_acctbal, c.c_phone, "
    "n.n_name, c.c_address, c.c_comment "
    "order by revenue desc "
    "limit 20;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 11",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q11]")
{
  compare_gpu_vs_cpu(
    "select ps.ps_partkey, "
    "sum(ps.ps_supplycost * ps.ps_availqty) as value "
    "from partsupp ps, supplier s, nation n "
    "where ps.ps_suppkey = s.s_suppkey "
    "and s.s_nationkey = n.n_nationkey "
    "and n.n_name = 'JAPAN' "
    "group by ps.ps_partkey "
    "having sum(ps.ps_supplycost * ps.ps_availqty) > ("
    "  select sum(ps.ps_supplycost * ps.ps_availqty) * 0.0001000000 "
    "  from partsupp ps, supplier s, nation n "
    "  where ps.ps_suppkey = s.s_suppkey "
    "  and s.s_nationkey = n.n_nationkey "
    "  and n.n_name = 'JAPAN'"
    ") "
    "order by value desc;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 12",
                 "[integration][gpu_execution][TPC-H][Q12]")
{
  compare_gpu_vs_cpu(
    "select l.l_shipmode, "
    "sum(case when o.o_orderpriority = '1-URGENT' "
    "  or o.o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count, "
    "sum(case when o.o_orderpriority <> '1-URGENT' "
    "  and o.o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count "
    "from orders o, lineitem l "
    "where o.o_orderkey = l.l_orderkey "
    "and l.l_shipmode in ('TRUCK', 'REG AIR') "
    "and l.l_commitdate < l.l_receiptdate "
    "and l.l_shipdate < l.l_commitdate "
    "and l.l_receiptdate >= date '1994-01-01' "
    "and l.l_receiptdate < date '1995-01-01' "
    "group by l.l_shipmode "
    "order by l.l_shipmode;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 13",
                 "[integration][gpu_execution][TPC-H][Q13]")
{
  compare_gpu_vs_cpu(
    "select c_count, count(*) as custdist "
    "from ("
    "  select c.c_custkey, count(o.o_orderkey) "
    "  from customer c "
    "  left outer join orders o "
    "    on c.c_custkey = o.o_custkey "
    "    and o.o_comment not like '%special%requests%' "
    "  group by c.c_custkey"
    ") as orders (c_custkey, c_count) "
    "group by c_count "
    "order by custdist desc, c_count desc;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 14",
                 "[integration][gpu_execution][TPC-H][Q14]")
{
  compare_gpu_vs_cpu(
    "select 100.00 * sum(case when p.p_type like 'PROMO%' "
    "  then l.l_extendedprice * (1 - l.l_discount) else 0 end) "
    "  / sum(l.l_extendedprice * (1 - l.l_discount)) as promo_revenue "
    "from lineitem l, part p "
    "where l.l_partkey = p.p_partkey "
    "and l.l_shipdate >= date '1994-08-01' "
    "and l.l_shipdate < date '1994-09-01';");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 15",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q15]")
{
  compare_gpu_vs_cpu(
    "with revenue_view as ("
    "  select l_suppkey as supplier_no, "
    "  sum(l_extendedprice * (1 - l_discount)) as total_revenue "
    "  from lineitem "
    "  where l_shipdate >= date '1993-05-01' "
    "  and l_shipdate < date '1993-08-01' "
    "  group by l_suppkey"
    ") "
    "select s.s_suppkey, s.s_name, s.s_address, s.s_phone, r.total_revenue "
    "from supplier s, revenue_view r "
    "where s.s_suppkey = r.supplier_no "
    "and r.total_revenue = ("
    "  select max(total_revenue) from revenue_view"
    ") "
    "order by s.s_suppkey;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 16",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q16]")
{
  compare_gpu_vs_cpu(
    "select p.p_brand, p.p_type, p.p_size, "
    "count(distinct ps.ps_suppkey) as supplier_cnt "
    "from partsupp ps, part p "
    "where p.p_partkey = ps.ps_partkey "
    "and p.p_brand <> 'Brand#21' "
    "and p.p_type not like 'MEDIUM PLATED%' "
    "and p.p_size in (38, 2, 8, 31, 44, 5, 14, 24) "
    "and ps.ps_suppkey not in ("
    "  select s.s_suppkey from supplier s "
    "  where s.s_comment like '%Customer%Complaints%'"
    ") "
    "group by p.p_brand, p.p_type, p.p_size "
    "order by supplier_cnt desc, p.p_brand, p.p_type, p.p_size;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 17",
                 "[integration][gpu_execution][TPC-H][Q17]")
{
  compare_gpu_vs_cpu(
    "select sum(l.l_extendedprice) / 7.0 as avg_yearly "
    "from lineitem l, part p "
    "where p.p_partkey = l.l_partkey "
    "and p.p_brand = 'Brand#13' "
    "and p.p_container = 'JUMBO CAN' "
    "and l.l_quantity < ("
    "  select 0.2 * avg(l2.l_quantity) "
    "  from lineitem l2 "
    "  where l2.l_partkey = p.p_partkey"
    ");");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 18",
                 "[integration][gpu_execution][TPC-H][Q18]")
{
  compare_gpu_vs_cpu(
    "select c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, "
    "o.o_totalprice, sum(l.l_quantity) "
    "from customer c, orders o, lineitem l "
    "where o.o_orderkey in ("
    "  select l_orderkey from lineitem "
    "  group by l_orderkey having sum(l_quantity) > 300"
    ") "
    "and c.c_custkey = o.o_custkey "
    "and o.o_orderkey = l.l_orderkey "
    "group by c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice "
    "order by o.o_totalprice desc, o.o_orderdate "
    "limit 100;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 19",
                 "[integration][gpu_execution][TPC-H][Q19]")
{
  compare_gpu_vs_cpu(
    "select sum(l.l_extendedprice* (1 - l.l_discount)) as revenue "
    "from lineitem l, part p "
    "where ("
    "  p.p_partkey = l.l_partkey "
    "  and p.p_brand = 'Brand#41' "
    "  and p.p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
    "  and l.l_quantity >= 2 and l.l_quantity <= 2 + 10 "
    "  and p.p_size between 1 and 5 "
    "  and l.l_shipmode in ('AIR', 'AIR REG') "
    "  and l.l_shipinstruct = 'DELIVER IN PERSON'"
    ") or ("
    "  p.p_partkey = l.l_partkey "
    "  and p.p_brand = 'Brand#13' "
    "  and p.p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
    "  and l.l_quantity >= 14 and l.l_quantity <= 14 + 10 "
    "  and p.p_size between 1 and 10 "
    "  and l.l_shipmode in ('AIR', 'AIR REG') "
    "  and l.l_shipinstruct = 'DELIVER IN PERSON'"
    ") or ("
    "  p.p_partkey = l.l_partkey "
    "  and p.p_brand = 'Brand#55' "
    "  and p.p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
    "  and l.l_quantity >= 23 and l.l_quantity <= 23 + 10 "
    "  and p.p_size between 1 and 15 "
    "  and l.l_shipmode in ('AIR', 'AIR REG') "
    "  and l.l_shipinstruct = 'DELIVER IN PERSON'"
    ");");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 20",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q20]")
{
  compare_gpu_vs_cpu(
    "select s.s_name, s.s_address "
    "from supplier s, nation n "
    "where s.s_suppkey in ("
    "  select ps.ps_suppkey from partsupp ps "
    "  where ps.ps_partkey in ("
    "    select p.p_partkey from part p where p.p_name like 'antique%'"
    "  ) "
    "  and ps.ps_availqty > ("
    "    select 0.5 * sum(l.l_quantity) "
    "    from lineitem l "
    "    where l.l_partkey = ps.ps_partkey "
    "    and l.l_suppkey = ps.ps_suppkey "
    "    and l.l_shipdate >= date '1993-01-01' "
    "    and l.l_shipdate < date '1994-01-01'"
    "  )"
    ") "
    "and s.s_nationkey = n.n_nationkey "
    "and n.n_name = 'KENYA' "
    "order by s.s_name;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 21",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q21]")
{
  compare_gpu_vs_cpu(
    "select s.s_name, count(*) as numwait "
    "from supplier s, lineitem l1, orders o, nation n "
    "where s.s_suppkey = l1.l_suppkey "
    "and o.o_orderkey = l1.l_orderkey "
    "and o.o_orderstatus = 'F' "
    "and l1.l_receiptdate > l1.l_commitdate "
    "and exists ("
    "  select * from lineitem l2 "
    "  where l2.l_orderkey = l1.l_orderkey "
    "  and l2.l_suppkey <> l1.l_suppkey"
    ") "
    "and not exists ("
    "  select * from lineitem l3 "
    "  where l3.l_orderkey = l1.l_orderkey "
    "  and l3.l_suppkey <> l1.l_suppkey "
    "  and l3.l_receiptdate > l3.l_commitdate"
    ") "
    "and s.s_nationkey = n.n_nationkey "
    "and n.n_name = 'BRAZIL' "
    "group by s.s_name "
    "order by numwait desc, s.s_name "
    "limit 100;");
}

TEST_CASE_METHOD(GPUExecutionFixture,
                 "gpu_execution - TPC-H Query 22",
                 "[.][integration_disabled][gpu_execution][TPC-H][Q22]")
{
  compare_gpu_vs_cpu(
    "select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal "
    "from ("
    "  select substring(c_phone from 1 for 2) as cntrycode, c_acctbal "
    "  from customer c "
    "  where substring(c_phone from 1 for 2) in "
    "    ('24', '31', '11', '16', '21', '20', '34') "
    "  and c_acctbal > ("
    "    select avg(c_acctbal) from customer "
    "    where c_acctbal > 0.00 "
    "    and substring(c_phone from 1 for 2) in "
    "      ('24', '31', '11', '16', '21', '20', '34')"
    "  ) "
    "  and not exists ("
    "    select * from orders o where o.o_custkey = c.c_custkey"
    "  )"
    ") as custsale "
    "group by cntrycode "
    "order by cntrycode;");
}
