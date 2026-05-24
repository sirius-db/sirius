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

    auto clean_query = query;
    while (!clean_query.empty() && (clean_query.back() == ';' || clean_query.back() == ' ')) {
      clean_query.pop_back();
    }

    // CPU reference; also yields the column count used to build the deterministic ORDER BY.
    auto cpu_result = con->Query("SELECT * FROM (" + clean_query + ") t");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("cpu error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());

    std::string order_clause = " ORDER BY ";
    for (duckdb::idx_t c = 0; c < cpu_result->ColumnCount(); c++) {
      if (c > 0) { order_clause += ", "; }
      order_clause += std::to_string(c + 1);
    }

    // Compare GPU vs CPU under a deterministic ordering; gpu_execution runs only once.
    auto gpu_sorted =
      con->Query("SELECT * FROM gpu_execution(\"" + clean_query + "\")" + order_clause);
    auto cpu_sorted = con->Query("SELECT * FROM (" + clean_query + ") t" + order_clause);
    REQUIRE(gpu_sorted);
    if (gpu_sorted->HasError()) { UNSCOPED_INFO("gpu sorted error: " << gpu_sorted->GetError()); }
    REQUIRE_FALSE(gpu_sorted->HasError());
    REQUIRE(cpu_sorted);
    if (cpu_sorted->HasError()) { UNSCOPED_INFO("cpu sorted error: " << cpu_sorted->GetError()); }
    REQUIRE_FALSE(cpu_sorted->HasError());

    REQUIRE(gpu_sorted->ColumnCount() == cpu_sorted->ColumnCount());
    REQUIRE(gpu_sorted->RowCount() == cpu_sorted->RowCount());

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
                 "gpu_execution TPC-DS Q44 ranking query",
                 "[.][integration][tpcds][gpu][window]")
{
  compare_gpu_vs_cpu(
    "SELECT low_side.rnk,"
    "       i1.i_product_name best_performing,"
    "       i2.i_product_name worst_performing "
    "FROM "
    "  (SELECT * "
    "   FROM "
    "     (SELECT item_sk,"
    "             rank() OVER (ORDER BY rank_col ASC) rnk "
    "      FROM "
    "        (SELECT ss_item_sk item_sk,"
    "                avg(ss_net_profit) rank_col "
    "         FROM store_sales ss1 "
    "         WHERE ss_store_sk = 1 "
    "         GROUP BY ss_item_sk "
    "         HAVING avg(ss_net_profit) > 0.9 * "
    "           (SELECT avg(ss_net_profit) rank_col "
    "            FROM store_sales "
    "            WHERE ss_store_sk = 1 "
    "              AND ss_addr_sk IS NULL "
    "            GROUP BY ss_store_sk)) v1) v11 "
    "   WHERE rnk < 11) low_side,"
    "  (SELECT * "
    "   FROM "
    "     (SELECT item_sk,"
    "             rank() OVER (ORDER BY rank_col DESC) rnk "
    "      FROM "
    "        (SELECT ss_item_sk item_sk,"
    "                avg(ss_net_profit) rank_col "
    "         FROM store_sales ss1 "
    "         WHERE ss_store_sk = 1 "
    "         GROUP BY ss_item_sk "
    "         HAVING avg(ss_net_profit) > 0.9 * "
    "           (SELECT avg(ss_net_profit) rank_col "
    "            FROM store_sales "
    "            WHERE ss_store_sk = 1 "
    "              AND ss_addr_sk IS NULL "
    "            GROUP BY ss_store_sk)) v2) v21 "
    "   WHERE rnk < 11) high_side,"
    "  item i1,"
    "  item i2 "
    "WHERE low_side.rnk = high_side.rnk "
    "  AND i1.i_item_sk = low_side.item_sk "
    "  AND i2.i_item_sk = high_side.item_sk "
    "ORDER BY low_side.rnk "
    "LIMIT 100");
}
