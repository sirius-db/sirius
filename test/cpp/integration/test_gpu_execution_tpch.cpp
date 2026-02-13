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

// Returns a single DuckDB instance shared across all integration tests.
// The env guard and DB are created once (on first call) and destroyed at program exit.
static duckdb::DuckDB& get_shared_db()
{
  static struct env_init {
    env_init()
    {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.cfg";
      REQUIRE(fs::exists(cfg_path));
      setenv("SIRIUS_CONFIG_FILE", cfg_path.string().c_str(), 1);
    }
  } env;

  static duckdb::DuckDB db(get_tpch_db_path().string());
  return db;
}

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
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey from nation;");
}

TEST_CASE("gpu_execution - scan multiple columns", "[integration][gpu_execution][scan]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey, n_regionkey from nation;");
}

TEST_CASE("gpu_execution - scan region table", "[integration][gpu_execution][scan]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select r_regionkey from region;");
}

//===----------------------------------------------------------------------===//
// Projection tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - projection add", "[integration][gpu_execution][projection]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey + n_regionkey as total from nation;");
}

TEST_CASE("gpu_execution - projection multiply", "[integration][gpu_execution][projection]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey * 2 as doubled, n_regionkey from nation;");
}

//===----------------------------------------------------------------------===//
// Filter tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - filter equality", "[integration][gpu_execution][filter]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey from nation where n_regionkey = 1;");
}

TEST_CASE("gpu_execution - filter greater than", "[integration][gpu_execution][filter]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey from nation where n_regionkey > 2;");
}

TEST_CASE("gpu_execution - filter not equal", "[integration][gpu_execution][filter]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select r_regionkey from region where r_regionkey != 3;");
}

TEST_CASE("gpu_execution - filter with projection", "[integration][gpu_execution][filter]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey, n_regionkey from nation where n_regionkey = 0;");
}

//===----------------------------------------------------------------------===//
// Ungrouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - ungrouped min max", "[integration][gpu_execution][aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select min(n_regionkey), max(n_nationkey) from nation;");
}

TEST_CASE("gpu_execution - ungrouped min with filter", "[integration][gpu_execution][aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select min(n_nationkey) from nation where n_regionkey = 1;");
}

TEST_CASE("gpu_execution - ungrouped sum count", "[integration][gpu_execution][aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select sum(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE("gpu_execution - ungrouped all agg functions", "[integration][gpu_execution][aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select sum(n_regionkey), min(n_nationkey), max(n_regionkey), count(n_nationkey) from nation;");
}

TEST_CASE("gpu_execution - ungrouped avg integer", "[integration][gpu_execution][aggregate][avg]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select avg(n_nationkey) from nation;");
}

TEST_CASE("gpu_execution - ungrouped avg decimal", "[integration][gpu_execution][aggregate][avg]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select avg(l_quantity), avg(l_discount) from lineitem;");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - single group by key: min max, sum, count(*)",
          "[integration][gpu_execution][grouped_aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select c_nationkey, min(c_custkey), max(c_custkey), sum(c_custkey), count(*) "
                     "from customer group by c_nationkey;");
}

TEST_CASE("gpu_execution - single group by key: min max, count string ",
          "[integration][gpu_execution][grouped_aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select c_nationkey, min(C_NAME), max(C_NAME), count(C_NAME) from customer "
                     "group by c_nationkey;");
}

TEST_CASE("gpu_execution - two group by key: min max, but not showing the group by keys",
          "[integration][gpu_execution][grouped_aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select min(c_custkey), max(c_custkey) from customer group by c_nationkey, c_mktsegment;");
}

TEST_CASE("gpu_execution - two group keys and noaggregations",
          "[integration][gpu_execution][grouped_aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select c_nationkey, c_mktsegment from customer group by c_mktsegment, c_nationkey;");
}

//===----------------------------------------------------------------------===//
// Limit tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - limit", "[integration][gpu_execution][limit]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey from nation limit 10;");
}

TEST_CASE("gpu_execution - limit with filter", "[integration][gpu_execution][limit]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n_nationkey, n_regionkey from nation where n_regionkey = 1 limit 3;");
}

//===----------------------------------------------------------------------===//
// Join tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - basic inner join 0", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from nation n join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic inner join 1", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
                     "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic inner join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
                     "join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic inner join 3", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from nation n join customer c on "
                     "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic left join 0", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from nation n left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic left join 1", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
                     "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic left join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
                     "left join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic left join 3", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
                     "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic left join 0 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from nation n left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic left join 1 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
                     "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic left join 2 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
                     "left join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic left join 3 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from nation n left join customer c "
                     "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic right join 0", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from nation n right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic right join 1", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
                     "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic right join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
                     "right join customer c on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic right join 3", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
                     "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - basic right join 0 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from nation n right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic right join 1 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from nation n "
                     "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic right join 2 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from nation n "
                     "right join customer c on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic right join 3 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from nation n right join customer c "
                     "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped inner join 0", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from customer c join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped inner join 1", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
                     "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped inner join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
                     "join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped inner join 3", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from customer c join nation n on "
                     "n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped left join 0", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from customer c left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped left join 1", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
                     "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped left join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
                     "left join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped left join 3", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
                     "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped left join 0 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from customer c left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped left join 1 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
                     "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped left join 2 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
                     "left join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped left join 3 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from customer c left join nation n "
                     "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped right join 0", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from customer c right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped right join 1", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
                     "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped right join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
                     "right join nation n on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped right join 3", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
                     "on n.n_nationkey = c.c_nationkey;");
}

TEST_CASE("gpu_execution - swapped right join 0 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  "
                     "from customer c right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped right join 1 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, n.n_regionkey, c.c_custkey, c.c_name  from customer c "
                     "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped right join 2 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name  from customer c "
                     "right join nation n on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - swapped right join 3 making nulls", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_name, c.c_custkey, c.c_name  from customer c right join nation n "
                     "on n.n_nationkey = c.c_custkey;");
}

TEST_CASE("gpu_execution - basic full outer join", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
                     "on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE("gpu_execution - basic full outer join making nulls",
          "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n.n_nationkey, r.r_regionkey from nation n full outer join region r "
                     "on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE("gpu_execution - basic left semi join", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;");
}

TEST_CASE("gpu_execution - basic left semi join 2", "[integration][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select n.n_nationkey from nation n semi join region r on n.n_nationkey = r.r_regionkey;");
}

TEST_CASE("gpu_execution - basic right semi join", "[.][integration_disabled][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_regionkey;");
}

TEST_CASE("gpu_execution - basic right semi join 2",
          "[.][integration_disabled][gpu_execution][join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_nationkey;");
}

TEST_CASE("gpu_execution - bigger inner join", "[integration][gpu_execution][bigger_join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l join orders o on l.l_orderkey = o.o_orderkey order by "
    "l.l_orderkey, l.l_linenumber;");
}

TEST_CASE("gpu_execution - bigger left join", "[integration][gpu_execution][bigger_join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l left join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber;");
}

TEST_CASE("gpu_execution - bigger right join", "[integration][gpu_execution][bigger_join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l right join orders o on l.l_orderkey = o.o_orderkey "
    "order by l.l_orderkey, l.l_linenumber;");
}

TEST_CASE("gpu_execution - bigger full outer join", "[integration][gpu_execution][bigger_join]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey, o.o_orderkey, o.o_totalprice, "
    "o.o_custkey, o_comment from lineitem l full outer join orders o on l.l_orderkey = "
    "o.o_orderkey order by l.l_orderkey, l.l_linenumber;");
}

//===----------------------------------------------------------------------===//
// Disabled tests - known issues
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - two group by key: min max, sum, count of doubles",
          "[.][integration_disabled][gpu_execution][aggregate]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select c_nationkey, c_mktsegment, min(C_ACCTBAL), max(C_ACCTBAL), sum(C_ACCTBAL), "
    "count(C_ACCTBAL) from customer group by c_nationkey, c_mktsegment;");
}

// Empty result set: "Port default not found in operator RESULT_COLLECTOR"
TEST_CASE("gpu_execution - filter returns empty result", "[.][integration_disabled][gpu_execution]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey from nation where n_regionkey = 99;");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - group by count", "[integration][gpu_execution][group_by]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_regionkey, count(*) from nation group by n_regionkey;");
}

TEST_CASE("gpu_execution - group by min max count", "[integration][gpu_execution][group_by]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n_regionkey, min(n_nationkey), max(n_nationkey), count(n_nationkey) "
                     "from nation group by n_regionkey;");
}

TEST_CASE("gpu_execution - group by avg integer", "[integration][gpu_execution][group_by][avg]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_regionkey, avg(n_nationkey) from nation group by n_regionkey;");
}

TEST_CASE("gpu_execution - group by avg with other aggregates",
          "[integration][gpu_execution][group_by][avg]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n_regionkey, avg(n_nationkey), sum(n_nationkey), count(*) "
                     "from nation group by n_regionkey;");
}

TEST_CASE("gpu_execution - group by avg decimal", "[integration][gpu_execution][group_by][avg]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select l_returnflag, avg(l_quantity), avg(l_discount) "
                     "from lineitem group by l_returnflag;");
}

TEST_CASE("gpu_execution - group by sum avg on lineitem",
          "[integration][gpu_execution][group_by][avg]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*) "
    "from lineitem group by l_returnflag, l_linestatus;");
}

//===----------------------------------------------------------------------===//
// Order by tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - order by", "[integration][gpu_execution][order_by]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey, n_regionkey from nation order by n_regionkey;");
}

TEST_CASE("gpu_execution - order by column not in select",
          "[integration][gpu_execution][order_by][order_by_proj]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey from nation order by n_regionkey;");
}

TEST_CASE("gpu_execution - order by column not in select lineitem",
          "[integration][gpu_execution][order_by][order_by_proj]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select l_orderkey from lineitem order by l_linenumber;");
}

TEST_CASE("gpu_execution - order by multipartition", "[integration][gpu_execution][order_by]")
{
  duckdb::Connection con(get_shared_db());

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
}

TEST_CASE("gpu_execution - order by multiple columns", "[integration][gpu_execution][order_by]")
{
  duckdb::Connection con(get_shared_db());
  con.Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    con,
    "select l_orderkey, l_linenumber, l_quantity from lineitem order by l_orderkey, l_linenumber;");
}

TEST_CASE("gpu_execution - order by desc", "[integration][gpu_execution][order_by]")
{
  duckdb::Connection con(get_shared_db());
  con.Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(
    con, "select l_orderkey, l_partkey, l_suppkey from lineitem order by l_partkey desc;");
}

TEST_CASE("gpu_execution - order by many selected columns",
          "[integration][gpu_execution][order_by]")
{
  duckdb::Connection con(get_shared_db());
  con.Query("SET max_sort_partition_bytes = 1024;");
  compare_gpu_vs_cpu(con,
                     "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity "
                     "from lineitem order by l_suppkey;");
}

TEST_CASE("gpu_execution - order by with decimal column",
          "[integration][gpu_execution][order_by][order_by_types]")
{
  duckdb::Connection con(get_shared_db());
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
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select l_orderkey, l_shipinstruct from lineitem;");
}

TEST_CASE("gpu_execution - order by lineitem with short varchar column",
          "[integration][gpu_execution][order_by][varchar_order]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select l_orderkey, l_shipinstruct, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE("gpu_execution - order by lineitem with long varchar column",
          "[integration][gpu_execution][order_by][varchar_order]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select l_orderkey, l_comment, l_linenumber from lineitem order by l_orderkey;");
}

TEST_CASE("gpu_execution - scan with varchar column",
          "[integration][gpu_execution][order_by_types][varchar]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con, "select n_nationkey, n_name from nation;");
}

TEST_CASE("gpu_execution - order by with varchar column",
          "[integration][gpu_execution][order_by][order_by_types]")
{
  duckdb::Connection con(get_shared_db());
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

//===----------------------------------------------------------------------===//
// Cast to decimal tests
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - cast integer to decimal preserves scale",
          "[integration][gpu_execution][cast][decimal]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n_nationkey, cast(n_nationkey as Decimal(18,2)) as d from nation;");
}

TEST_CASE("gpu_execution - cast integer to decimal with aggregation",
          "[integration][gpu_execution][cast][decimal]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select n_regionkey, max(cast(n_nationkey as Decimal(18,2))) as max_d "
                     "from nation group by n_regionkey;");
}

TEST_CASE("gpu_execution - cast to decimal different scales",
          "[integration][gpu_execution][cast][decimal]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select cast(n_nationkey as Decimal(9,0)) as d0, "
                     "cast(n_nationkey as Decimal(9,4)) as d4 from nation;");
}

// Disabled: avg() in grouped aggregates not yet supported (separate PR)
TEST_CASE("gpu_execution - issue 227 cast decimal with avg and group by",
          "[.][integration_disabled][gpu_execution][cast][decimal]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(con,
                     "select avg(n_regionkey), avg(n_nationkey), n_name, "
                     "max(cast(n_nationkey as Decimal(18,2))) "
                     "from nation group by n_regionkey, n_name;");
}

//===----------------------------------------------------------------------===//
// Top N / Join tests (disabled)
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_execution - top n", "[.][integration_disabled][gpu_execution][top_n]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con, "select n_nationkey, n_regionkey from nation order by n_regionkey desc limit 5;");
}

TEST_CASE("gpu_execution - join", "[.][integration_disabled][gpu_execution]")
{
  duckdb::Connection con(get_shared_db());
  compare_gpu_vs_cpu(
    con,
    "select n.n_nationkey, r.r_regionkey from nation n join region r on n.n_regionkey = "
    "r.r_regionkey;");
}
