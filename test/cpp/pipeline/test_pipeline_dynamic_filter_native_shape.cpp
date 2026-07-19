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
#include <sirius_config.hpp>
#include <sirius_context.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

//! Path to the integration DuckDB with the SF1 TPC-H schema pre-loaded.
fs::path integration_db_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/duckdb/integration.duckdb";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/duckdb/integration.duckdb";
#endif
}

//! RAII flip of the dynamic-filter master switch on the connection's shared SiriusContext.
//! The plan-gen router reads it live, and the context outlives this test — restore is mandatory.
class pushdown_switch_guard {
 public:
  pushdown_switch_guard(duckdb::Connection& con, bool enabled)
    : _state(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state")),
      _original(_state->get_config().get_operator_params().enable_dynamic_filter_pushdown)
  {
    REQUIRE(_state != nullptr);
    _state->get_config().get_operator_params().enable_dynamic_filter_pushdown = enabled;
  }
  ~pushdown_switch_guard()
  {
    _state->get_config().get_operator_params().enable_dynamic_filter_pushdown = _original;
  }

  pushdown_switch_guard(const pushdown_switch_guard&)            = delete;
  pushdown_switch_guard& operator=(const pushdown_switch_guard&) = delete;

 private:
  duckdb::shared_ptr<duckdb::SiriusContext> _state;
  bool _original;
};

bool contains(const std::string& haystack, const std::string& needle)
{
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

//! A selective build over `part` feeding a `lineitem` probe makes DuckDB's join-filter-pushdown
//! optimizer wire a DynamicTableFilterSet to the lineitem seq_scan. The duckdb-native GPU scan
//! must then carry a DYNAMIC_FILTER operator above it (conversion only, no GPU execution).
TEST_CASE("duckdb-native scans consume dynamic filters", "[integration][pipeline][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  auto db_path = integration_db_path();
  REQUIRE(fs::exists(db_path));
  auto r = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  const std::string join_query =
    "SELECT count(*) FROM lineitem, part WHERE l_partkey = p_partkey AND p_size = 15";

  SECTION("join over native tables wraps the probe scan in a DYNAMIC_FILTER")
  {
    REQUIRE(contains(sirius::test::convert_query_to_dump(con, join_query), "DYNAMIC_FILTER"));
  }

  SECTION("a single-table native scan carries no DYNAMIC_FILTER")
  {
    const std::string scan_query = "SELECT count(*) FROM lineitem WHERE l_quantity < 10";
    REQUIRE_FALSE(contains(sirius::test::convert_query_to_dump(con, scan_query), "DYNAMIC_FILTER"));
  }

  SECTION("the master switch elides the operator")
  {
    pushdown_switch_guard off(con, /*enabled=*/false);
    REQUIRE_FALSE(contains(sirius::test::convert_query_to_dump(con, join_query), "DYNAMIC_FILTER"));
  }
}
