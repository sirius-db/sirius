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

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
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
  auto db_path = fs::path(__FILE__).parent_path() / "integration.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

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
 * @brief Run a query through gpu_execution and through DuckDB CPU, then compare results.
 *
 * Values are compared as strings via Value::ToString() which normalizes type differences
 * (e.g., HUGEINT vs BIGINT both render "50"). Row order is ignored by collecting rows
 * as sorted sets of string tuples.
 */
static void compare_gpu_vs_cpu(duckdb::Connection& con, const std::string& query)
{
  // Disable fallback so GPU errors are not silently hidden
  con.Query("SET enable_fallback_check = true;");

  // Run on GPU
  auto gpu_sql    = "CALL gpu_execution('" + query + "')";
  auto gpu_result = con.Query(gpu_sql);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("gpu_execution error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());

  // Run on CPU (plain DuckDB)
  auto cpu_result = con.Query(query);
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

  auto gpu_sorted = con.Query("SELECT * FROM gpu_execution('" + clean_query + "')" + order_clause);
  auto cpu_sorted = con.Query("SELECT * FROM (" + clean_query + ") t" + order_clause);
  REQUIRE(gpu_sorted);
  if (gpu_sorted->HasError()) { UNSCOPED_INFO("gpu sorted error: " << gpu_sorted->GetError()); }
  REQUIRE_FALSE(gpu_sorted->HasError());
  REQUIRE(cpu_sorted);
  if (cpu_sorted->HasError()) { UNSCOPED_INFO("cpu sorted error: " << cpu_sorted->GetError()); }
  REQUIRE_FALSE(cpu_sorted->HasError());

  // Compare row by row using string values (handles type differences like HUGEINT vs BIGINT)
  for (duckdb::idx_t r = 0; r < gpu_sorted->RowCount(); r++) {
    for (duckdb::idx_t c = 0; c < gpu_sorted->ColumnCount(); c++) {
      auto gpu_val = gpu_sorted->GetValue(c, r).ToString();
      auto cpu_val = cpu_sorted->GetValue(c, r).ToString();
      if (gpu_val != cpu_val) {
        UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_val << "] CPU=["
                             << cpu_val << "]");
      }
      REQUIRE(gpu_val == cpu_val);
    }
  }
}

//===----------------------------------------------------------------------===//
// Scan tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - scan single column", "[integration][gpu_execution][scan]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey from nation;");
}

TEST_CASE("gpu_execution - scan multiple columns", "[integration][gpu_execution][scan]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey, n_regionkey from nation;");
}

TEST_CASE("gpu_execution - scan region table", "[integration][gpu_execution][scan]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select r_regionkey from region;");
}

//===----------------------------------------------------------------------===//
// Projection tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - projection add", "[integration][gpu_execution][projection]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey + n_regionkey as total from nation;");
}

TEST_CASE("gpu_execution - projection multiply", "[integration][gpu_execution][projection]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey * 2 as doubled, n_regionkey from nation;");
}

//===----------------------------------------------------------------------===//
// Filter tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - filter equality", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey from nation where n_regionkey = 1;");
}

TEST_CASE("gpu_execution - filter greater than", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey from nation where n_regionkey > 2;");
}

TEST_CASE("gpu_execution - filter not equal", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select r_regionkey from region where r_regionkey != 3;");
}

TEST_CASE("gpu_execution - filter with projection", "[integration][gpu_execution][filter]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey, n_regionkey from nation where n_regionkey = 0;");
}

//===----------------------------------------------------------------------===//
// Ungrouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - ungrouped min max", "[integration][gpu_execution][aggregate]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select min(n_regionkey), max(n_nationkey) from nation;");
}

TEST_CASE("gpu_execution - ungrouped min with filter", "[integration][gpu_execution][aggregate]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select min(n_nationkey) from nation where n_regionkey = 1;");
}

TEST_CASE("gpu_execution - ungrouped sum count", "[integration][gpu_execution][aggregate]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select sum(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE("gpu_execution - ungrouped all agg functions", "[integration][gpu_execution][aggregate]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(
    con,
    "select sum(n_regionkey), min(n_nationkey), max(n_regionkey), count(n_nationkey) from nation;");
}

//===----------------------------------------------------------------------===//
// Limit tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - limit", "[integration][gpu_execution][limit]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey from nation limit 10;");
}

TEST_CASE("gpu_execution - limit with filter", "[integration][gpu_execution][limit]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con,
                     "select n_nationkey, n_regionkey from nation where n_regionkey = 1 limit 3;");
}

//===----------------------------------------------------------------------===//
// Disabled tests - known issues
//===----------------------------------------------------------------------===//

// Empty result set: "Port default not found in operator RESULT_COLLECTOR"
TEST_CASE("gpu_execution - filter returns empty result", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey from nation where n_regionkey = 99;");
}

// Multi-pipeline queries hang (GROUP BY, ORDER BY, JOINs)
TEST_CASE("gpu_execution - group by", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_regionkey, count(*) from nation group by n_regionkey;");
}

TEST_CASE("gpu_execution - order by", "[integration][gpu_execution][order_by]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey, n_regionkey from nation order by n_regionkey;");
}

TEST_CASE("gpu_execution - order by column not in select",
          "[integration][gpu_execution][order_by][order_by_proj]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey from nation order by n_regionkey;");
}

TEST_CASE("gpu_execution - order by column not in select lineitem",
          "[integration][gpu_execution][order_by][order_by_proj]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select l_orderkey from lineitem order by l_linenumber;");
}

TEST_CASE("gpu_execution - order by multipartition", "[integration][gpu_execution][order_by]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);

  // Force small partition size (1 KB) so lineitem data is split into multiple partitions
  con.Query("SET max_sort_partition_bytes = 1024;");

  std::string query = "select l_orderkey, l_partkey from lineitem order by l_orderkey";

  // Run on GPU
  auto gpu_result = con.Query("CALL gpu_execution('" + query + "')");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("gpu error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());

  // Run on CPU
  auto cpu_result = con.Query(query + ";");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  // Verify dimensions match
  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  // With 600K rows and 1KB max partition, we must have many partitions.
  // Verify the data is non-trivially large (ensures partitioning actually happened).
  REQUIRE(gpu_result->RowCount() > 1000);
  // Sort both result sets for deterministic comparison
  auto gpu_sorted = con.Query("SELECT * FROM gpu_execution('" + query + "') ORDER BY 1, 2");
  auto cpu_sorted = con.Query("SELECT * FROM (" + query + ") t ORDER BY 1, 2");
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

  // Reset to auto
  con.Query("SET max_sort_partition_bytes = 0;");
}

TEST_CASE("gpu_execution - order by multiple columns", "[integration][gpu_execution][order_by]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  con.Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    con,
    "select l_orderkey, l_linenumber, l_quantity from lineitem order by l_orderkey, l_linenumber;");
  con.Query("SET max_sort_partition_bytes = 0;");
}

TEST_CASE("gpu_execution - order by desc", "[integration][gpu_execution][order_by]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  con.Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    con, "select l_orderkey, l_partkey, l_suppkey from lineitem order by l_partkey desc;");
  con.Query("SET max_sort_partition_bytes = 0;");
}

TEST_CASE("gpu_execution - order by many selected columns",
          "[integration][gpu_execution][order_by]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  con.Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(con,
                     "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity "
                     "from lineitem order by l_suppkey;");
  con.Query("SET max_sort_partition_bytes = 0;");
}

TEST_CASE("gpu_execution - order by with decimal column",
          "[integration][gpu_execution][order_by][order_by_types]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  auto gpu_result = con.Query(
    "CALL gpu_execution('select o_orderkey, o_totalprice from orders order by o_orderkey')");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) {
    std::cerr << "DECIMAL error: " << gpu_result->GetError() << std::endl;
  }
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->RowCount() > 0);
}

TEST_CASE("gpu_execution - scan lineitem with varchar column",
          "[integration][gpu_execution][varchar_scan_lineitem]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select l_orderkey, l_shipinstruct from lineitem;");
}

TEST_CASE("gpu_execution - order by lineitem with short varchar column",
          "[integration][gpu_execution][order_by][varchar_order]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(
    con, "select l_orderkey, l_shipinstruct, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE("gpu_execution - order by lineitem with long varchar column",
          "[integration][gpu_execution][order_by][varchar_order]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(
    con, "select l_orderkey, l_comment, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE("gpu_execution - scan with varchar column",
          "[integration][gpu_execution][order_by_types][varchar]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(con, "select n_nationkey, n_name from nation;");
}

TEST_CASE("gpu_execution - order by with varchar column",
          "[integration][gpu_execution][order_by][order_by_types]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  auto gpu_result =
    con.Query("CALL gpu_execution('select n_nationkey, n_name from nation order by n_nationkey')");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) {
    std::cerr << "VARCHAR order by error: " << gpu_result->GetError() << std::endl;
  }
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->RowCount() > 0);
  std::cerr << "VARCHAR order by: " << gpu_result->RowCount() << " rows OK" << std::endl;
}

TEST_CASE("gpu_execution - top n", "[.][integration_disabled][gpu_execution][top_n]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(
    con, "select n_nationkey, n_regionkey from nation order by n_regionkey desc limit 5;");
}

TEST_CASE("gpu_execution - join", "[.][integration_disabled][gpu_execution]")
{
  config_env_guard env;
  duckdb::DuckDB db(get_tpch_db_path().string());
  duckdb::Connection con(db);
  compare_gpu_vs_cpu(
    con,
    "select n.n_nationkey, r.r_regionkey from nation n join region r on n.n_regionkey = "
    "r.r_regionkey;");
}
