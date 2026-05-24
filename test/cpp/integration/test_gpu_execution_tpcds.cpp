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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tpcds_db_path()
{
  const char* env = std::getenv("SIRIUS_TPCDS_TEST_DB_PATH");
  if (!env) { env = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH"); }
  auto db_path =
    env ? fs::path(env) : fs::path(__FILE__).parent_path() / "data/duckdb/tpcds.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

struct tpcds_config_env_guard {
  explicit tpcds_config_env_guard(const std::string& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }

  ~tpcds_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

class TpcDsGPUExecutionFixture {
 public:
  TpcDsGPUExecutionFixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<tpcds_config_env_guard>(cfg_path.string());

      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }

    auto result =
      con->Query("ATTACH IF NOT EXISTS '" + tpcds_db_path().string() + "' AS tpcds (READ_ONLY);");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());

    result = con->Query("USE tpcds;");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  }

  static bool is_floating_point(duckdb::LogicalTypeId id)
  {
    return id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
  {
    con->Query("SET enable_duckdb_fallback = false;");

    auto gpu_sql    = "CALL gpu_execution(\"" + query + "\")";
    auto gpu_result = con->Query(gpu_sql);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("gpu_execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());

    auto cpu_result = con->Query(query);
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("cpu error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());

    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    auto ncols               = gpu_result->ColumnCount();
    std::string order_clause = " ORDER BY ";
    for (duckdb::idx_t c = 0; c < ncols; c++) {
      if (c > 0) { order_clause += ", "; }
      order_clause += std::to_string(c + 1);
    }

    auto clean_query = query;
    while (!clean_query.empty() && (clean_query.back() == ';' || clean_query.back() == ' ')) {
      clean_query.pop_back();
    }

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
          auto diff    = std::fabs(gpu_d - cpu_d);
          if (diff > static_cast<double>(float_tolerance.value())) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " float mismatch: GPU=[" << gpu_d
                                 << "] CPU=[" << cpu_d << "] diff=" << diff);
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
  std::unique_ptr<tpcds_config_env_guard> config_guard;
};

}  // namespace

TEST_CASE_METHOD(TpcDsGPUExecutionFixture,
                 "gpu_execution TPC-DS row_number over store_sales",
                 "[tpcds][gpu][window]")
{
  compare_gpu_vs_cpu(
    "SELECT ss_store_sk, ss_item_sk, ss_ticket_number, rn "
    "FROM ("
    "  SELECT "
    "    ss_store_sk,"
    "    ss_item_sk,"
    "    ss_ticket_number,"
    "    row_number() OVER ("
    "      PARTITION BY ss_store_sk "
    "      ORDER BY ss_sold_date_sk ASC NULLS LAST,"
    "               ss_ticket_number ASC NULLS LAST,"
    "               ss_item_sk ASC NULLS LAST,"
    "               ss_customer_sk ASC NULLS LAST"
    "    ) AS rn "
    "  FROM store_sales "
    "  WHERE ss_store_sk IS NOT NULL "
    "    AND ss_sold_date_sk IS NOT NULL "
    "    AND ss_ticket_number IS NOT NULL "
    "    AND ss_item_sk IS NOT NULL "
    "    AND ss_customer_sk IS NOT NULL"
    ") ranked "
    "ORDER BY ss_store_sk, rn, ss_item_sk, ss_ticket_number "
    "LIMIT 200");
}

TEST_CASE_METHOD(TpcDsGPUExecutionFixture,
                 "gpu_execution TPC-DS rank and dense_rank over customer",
                 "[tpcds][gpu][window]")
{
  compare_gpu_vs_cpu(
    "SELECT c_birth_country, c_customer_sk, c_birth_year, rnk, dr "
    "FROM ("
    "  SELECT "
    "    c_birth_country,"
    "    c_customer_sk,"
    "    c_birth_year,"
    "    rank() OVER ("
    "      PARTITION BY c_birth_country "
    "      ORDER BY c_birth_year ASC NULLS LAST"
    "    ) AS rnk,"
    "    dense_rank() OVER ("
    "      PARTITION BY c_birth_country "
    "      ORDER BY c_birth_year ASC NULLS LAST"
    "    ) AS dr "
    "  FROM customer "
    "  WHERE c_birth_country IS NOT NULL"
    ") ranked "
    "ORDER BY c_birth_country, rnk, c_customer_sk "
    "LIMIT 200");
}
