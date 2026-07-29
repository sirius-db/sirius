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

#include <cstdlib>
#include <filesystem>
#include <string>

namespace sirius::test {

/// Snapshot of the connection's `SiriusContext`-owned dynamic-filter counters. Tests assert
/// deltas around a query, and only the deterministic-policy family as equalities -- the
/// opportunistic-delivery family races with probe-side draining and supports directional
/// assertions only (see `op/dynamic_filter/dynamic_filter_stats.hpp`).
inline sirius::op::dynamic_filter_stats_snapshot get_dynamic_filter_stats_snapshot(
  duckdb::Connection& con)
{
  return get_registered_sirius_context(con)->get_dynamic_filter_stats_snapshot();
}

/// RAII disable of the domain-coverage gate: a threshold above 1.0 is the gate's explicit disabled
/// state (the documented rollback lever). The SET mutates the shared SiriusContext, which outlives
/// any one test, so the destructor restores whatever value the constructor found rather than the
/// default -- a literal restore would clobber an enclosing guard.
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

/// Path to the integration DuckDB carrying the SF1 TPC-H schema, honoring the
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
