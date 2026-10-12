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

#pragma once

#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>

namespace sirius::test {

/// Returns the dynamic-filter counter snapshot for @p con.
inline sirius::op::dynamic_filter_stats_snapshot get_dynamic_filter_stats_snapshot(
  duckdb::Connection& con)
{
  return get_registered_sirius_context(con)->get_dynamic_filter_stats_snapshot();
}

/// Sets one numeric Sirius setting for a scope and restores its previous value on destruction.
///
/// The setting lives on the shared `SiriusContext` and outlives the connection, so the restore
/// matters; its result is unchecked because destructors must not throw.
struct scoped_setting {
  scoped_setting(duckdb::Connection& c, std::string setting, std::uint64_t value)
    : con(c), name(std::move(setting))
  {
    auto current = con.Query("SELECT current_setting('" + name + "');");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    original = current->GetValue(0, 0).ToString();

    auto result = con.Query("SET " + name + " = " + std::to_string(value) + ";");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  ~scoped_setting() { con.Query("SET " + name + " = " + original + ";"); }

  scoped_setting(const scoped_setting&)            = delete;
  scoped_setting& operator=(const scoped_setting&) = delete;

  duckdb::Connection& con;
  std::string name;
  std::string original;
};

/// Disables the domain-coverage gate and restores the previous threshold on destruction.
struct coverage_gate_disable_guard {
  explicit coverage_gate_disable_guard(duckdb::Connection& c)
    : con(c),
      original(get_registered_sirius_context(c)
                 ->get_config()
                 .get_operator_params()
                 .dynamic_filter_domain_coverage_threshold)
  {
    con.Query("SET dynamic_filter_domain_coverage_threshold = 2.0;");
  }
  ~coverage_gate_disable_guard()
  {
    con.Query("SET dynamic_filter_domain_coverage_threshold = " + std::to_string(original) + ";");
  }

  coverage_gate_disable_guard(const coverage_gate_disable_guard&)            = delete;
  coverage_gate_disable_guard& operator=(const coverage_gate_disable_guard&) = delete;

  duckdb::Connection& con;
  double original;
};

/// Appends the given optimizers to `disabled_optimizers` and restores the setting on destruction.
struct disabled_optimizers_guard {
  disabled_optimizers_guard(duckdb::Connection& c, const std::string& optimizers) : con(c)
  {
    auto current = con.Query("SELECT current_setting('disabled_optimizers');");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    original    = current->GetValue(0, 0).ToString();
    auto merged = original.empty() ? optimizers : original + "," + optimizers;
    auto result = con.Query("SET disabled_optimizers = '" + merged + "';");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  ~disabled_optimizers_guard() { con.Query("SET disabled_optimizers = '" + original + "';"); }

  disabled_optimizers_guard(const disabled_optimizers_guard&)            = delete;
  disabled_optimizers_guard& operator=(const disabled_optimizers_guard&) = delete;

  duckdb::Connection& con;
  std::string original;
};

/// Path to the vendored TPC-H integration DuckDB, honoring the
/// `SIRIUS_INTEGRATION_TEST_DB_PATH` override. Fails the test if the database is absent.
inline std::filesystem::path integration_tpch_db_path()
{
  namespace fs         = std::filesystem;
  char const* env_path = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH");
  auto db_path         = env_path ? fs::path{env_path}
                                  : fs::path{SIRIUS_PROJECT_ROOT} /
                              "test/cpp/integration/data/duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

}  // namespace sirius::test
