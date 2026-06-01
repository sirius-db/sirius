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

/**
 * @file test_lance_vector_search_transparent.cpp
 * @brief Self-skipping e2e hooks for the real DuckDB Lance extension.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct config_env_guard {
  explicit config_env_guard(const std::string& path)
  {
    if (const char* current = std::getenv("SIRIUS_CONFIG_FILE")) {
      had_original_value = true;
      original_value     = current;
    }
    setenv("SIRIUS_CONFIG_FILE", path.c_str(), 1);
  }

  ~config_env_guard()
  {
    if (had_original_value) {
      setenv("SIRIUS_CONFIG_FILE", original_value.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
  }

  std::string original_value;
  bool had_original_value = false;
};

class LanceVectorSearchTransparentFixture {
 public:
  LanceVectorSearchTransparentFixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<config_env_guard>(cfg_path.string());

      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }

    con->Query("SET gpu_execution = true;");
  }

  bool load_lance_extension()
  {
    const char* extension_env        = std::getenv("SIRIUS_TEST_LANCE_EXTENSION");
    const std::string extension_name = extension_env ? extension_env : "lance";

    auto load_result = con->Query("LOAD " + extension_name + ";");
    if (load_result && !load_result->HasError()) { return true; }

    auto install_result = con->Query("INSTALL " + extension_name + ";");
    if (!install_result || install_result->HasError()) {
      WARN("Lance DuckDB extension unavailable; skipping: "
           << (install_result ? install_result->GetError() : "null result"));
      return false;
    }

    load_result = con->Query("LOAD " + extension_name + ";");
    if (!load_result || load_result->HasError()) {
      WARN("Lance DuckDB extension failed to load; skipping: "
           << (load_result ? load_result->GetError() : "null result"));
      return false;
    }
    return true;
  }

  static std::string require_env_query(const char* env_var)
  {
    const char* value = std::getenv(env_var);
    if (!value || std::string(value).empty()) {
      WARN(env_var << " is unset; skipping Lance vector-search e2e test");
      return {};
    }
    std::string query(value);
    if (query.find("lance_vector_search") == std::string::npos) {
      WARN(env_var << " does not reference lance_vector_search; skipping");
      return {};
    }
    return query;
  }

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

  void compare_gpu_vs_cpu(const std::string& query)
  {
    auto before_gpu_stats = sirius::test::get_transparent_execution_stats(*con);
    auto gpu_result       = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) { UNSCOPED_INFO("GPU error: " << gpu_result->GetError()); }
    REQUIRE_FALSE(gpu_result->HasError());
    auto after_gpu_stats = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before_gpu_stats, after_gpu_stats, 1, 0, 1);

    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("CPU error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());

    auto& gpu_mat = gpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto& cpu_mat = cpu_result->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(collect_rows(gpu_mat) == collect_rows(cpu_mat));
  }

  void require_clean_fallback_and_cpu_parity(const std::string& query)
  {
    auto before_stats = sirius::test::get_transparent_execution_stats(*con);
    auto gpu_result   = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("fallback query error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());
    auto after_stats = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before_stats, after_stats, 0, 1, 0);

    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("CPU error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());

    auto& gpu_mat = gpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto& cpu_mat = cpu_result->Cast<duckdb::MaterializedQueryResult>();
    REQUIRE(collect_rows(gpu_mat) == collect_rows(cpu_mat));
  }

 protected:
  std::unique_ptr<config_env_guard> config_guard;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

}  // namespace

TEST_CASE_METHOD(LanceVectorSearchTransparentFixture,
                 "lance_vector_search transparent e2e - scalar tail stays on GPU",
                 "[lance_vector_search][transparent][integration]")
{
  auto query = require_env_query("SIRIUS_TEST_LANCE_SCALAR_QUERY");
  if (query.empty()) { return; }
  if (!load_lance_extension()) { return; }

  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(LanceVectorSearchTransparentFixture,
                 "lance_vector_search transparent e2e - projected vector cleanly falls back",
                 "[lance_vector_search][transparent][integration]")
{
  auto query = require_env_query("SIRIUS_TEST_LANCE_VECTOR_QUERY");
  if (query.empty()) { return; }
  if (!load_lance_extension()) { return; }

  require_clean_fallback_and_cpu_parity(query);
}
