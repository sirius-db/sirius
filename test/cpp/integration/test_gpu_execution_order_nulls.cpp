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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace {

void require_ok(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

std::string strip_trailing_sql(const std::string& sql)
{
  auto clean = sql;
  while (!clean.empty() && (clean.back() == ';' || clean.back() == ' ' || clean.back() == '\n' ||
                            clean.back() == '\t')) {
    clean.pop_back();
  }
  return clean;
}

std::string sql_string_literal(const std::string& value)
{
  std::string escaped = "'";
  for (char c : value) {
    if (c == '\'') { escaped += '\''; }
    escaped += c;
  }
  escaped += "'";
  return escaped;
}

std::string query_single_value(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return result->GetValue(0, 0).ToString();
}

class runtime_setting_guard {
 public:
  runtime_setting_guard(duckdb::Connection& con, std::string setting, uint64_t value)
    : con(con),
      setting(std::move(setting)),
      old_value(query_single_value(con, "SELECT current_setting('" + this->setting + "')"))
  {
    require_ok(con, "SET " + this->setting + " = " + std::to_string(value));
  }

  ~runtime_setting_guard() { restore(); }

 private:
  void restore()
  {
    auto result = con.Query("SET " + setting + " = " + old_value);
    if (!result || result->HasError()) {
      WARN("Failed to restore " << setting << " to " << old_value
                                << (result ? ": " + result->GetError() : ""));
    }
  }

  duckdb::Connection& con;
  std::string setting;
  std::string old_value;
};

struct null_order_case {
  std::string sort_direction;
  std::string null_position;
};

const null_order_case kNullOrderCases[] = {
  {"ASC", "FIRST"},
  {"ASC", "LAST"},
  {"DESC", "FIRST"},
  {"DESC", "LAST"},
};

std::string case_name(const null_order_case& order_case)
{
  return order_case.sort_direction + " NULLS " + order_case.null_position;
}

class OrderNullsGPUExecutionFixture {
 public:
  OrderNullsGPUExecutionFixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    REQUIRE(sirius::test::g_integration_env->is_active());
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());

    require_ok(*con, "SET enable_duckdb_fallback = false");
    require_ok(*con,
               "CREATE TEMP TABLE ord_n AS "
               "SELECT CASE WHEN i % 7 = 0 THEN NULL ELSE CAST(i % 100 AS INTEGER) END AS k, "
               "       CAST(i AS INTEGER) AS id "
               "FROM range(30000) AS t(i)");
  }

  void compare_gpu_vs_cpu_ordered(const std::string& query)
  {
    const auto clean_query = strip_trailing_sql(query);
    require_ok(*con, "DROP TABLE IF EXISTS order_nulls_gpu_result");
    require_ok(*con, "DROP TABLE IF EXISTS order_nulls_cpu_result");
    require_ok(*con,
               "CREATE TEMP TABLE order_nulls_gpu_result AS "
               "SELECT row_number() OVER () AS observed_position, * "
               "FROM gpu_execution(" +
                 sql_string_literal(clean_query) + ")");
    require_ok(*con,
               "CREATE TEMP TABLE order_nulls_cpu_result AS "
               "SELECT row_number() OVER () AS observed_position, * "
               "FROM (" +
                 clean_query + ") cpu_result");
    require_empty(
      "SELECT * FROM order_nulls_gpu_result "
      "EXCEPT ALL "
      "SELECT * FROM order_nulls_cpu_result",
      "GPU minus CPU");
    require_empty(
      "SELECT * FROM order_nulls_cpu_result "
      "EXCEPT ALL "
      "SELECT * FROM order_nulls_gpu_result",
      "CPU minus GPU");
  }

 private:
  void require_empty(const std::string& sql, const std::string& direction)
  {
    auto result = con->Query("SELECT count(*) FROM (" + sql + ") diff");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(direction << ": " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());

    const auto count = result->GetValue(0, 0).GetValue<int64_t>();
    if (count != 0) { UNSCOPED_INFO(direction << " returned " << count << " unexpected rows"); }
    REQUIRE(count == 0);
  }

 public:
  std::unique_ptr<duckdb::Connection> con;
};

}  // namespace

TEST_CASE_METHOD(OrderNullsGPUExecutionFixture,
                 "gpu_execution ORDER BY places NULLs correctly for ASC and DESC",
                 "[integration][gpu_execution][order_by][nulls]")
{
  runtime_setting_guard max_sort_partition_bytes(*con, "max_sort_partition_bytes", 65536);

  for (const auto& order_case : kNullOrderCases) {
    DYNAMIC_SECTION(case_name(order_case))
    {
      compare_gpu_vs_cpu_ordered(
        "SELECT k, id "
        "FROM ord_n "
        "ORDER BY k " +
        order_case.sort_direction + " NULLS " + order_case.null_position + ", id ASC");
    }
  }
}

TEST_CASE_METHOD(OrderNullsGPUExecutionFixture,
                 "gpu_execution TOP-N single-key places NULLs correctly for ASC and DESC",
                 "[integration][gpu_execution][top_n][nulls]")
{
  for (const auto& order_case : kNullOrderCases) {
    DYNAMIC_SECTION(case_name(order_case))
    {
      compare_gpu_vs_cpu_ordered(
        "SELECT k "
        "FROM ord_n "
        "ORDER BY k " +
        order_case.sort_direction + " NULLS " + order_case.null_position +
        " "
        "LIMIT 50");
    }
  }
}

TEST_CASE_METHOD(OrderNullsGPUExecutionFixture,
                 "gpu_execution TOP-N multi-key places NULLs correctly for ASC and DESC",
                 "[integration][gpu_execution][top_n][nulls]")
{
  for (const auto& order_case : kNullOrderCases) {
    DYNAMIC_SECTION(case_name(order_case))
    {
      compare_gpu_vs_cpu_ordered(
        "SELECT k, id "
        "FROM ord_n "
        "ORDER BY k " +
        order_case.sort_direction + " NULLS " + order_case.null_position +
        ", id ASC "
        "LIMIT 50");
    }
  }
}
