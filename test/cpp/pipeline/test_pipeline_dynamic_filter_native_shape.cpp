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
// Reaching a join operator through the pipeline headers instantiates
// `vector<join_condition>`'s destructor, which needs the AST node definition.
#include <expression/ast/node.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <pipeline/sirius_pipeline_converter.hpp>
#include <sirius_config.hpp>
#include <sirius_context.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>

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

// Set the dynamic-filter flag for one scope and restore it on exit.
class dynamic_filter_switch_guard {
 public:
  dynamic_filter_switch_guard(duckdb::Connection& con, bool enabled)
    : _con(con),
      _original(sirius::test::get_registered_sirius_context(con)
                  ->get_config()
                  .get_operator_params()
                  .enable_dynamic_filter)
  {
    auto result =
      _con.Query(std::string{"SET enable_dynamic_filter = "} + (enabled ? "true" : "false") + ";");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  ~dynamic_filter_switch_guard()
  {
    _con.Query(std::string{"SET enable_dynamic_filter = "} + (_original ? "true" : "false") + ";");
  }

  dynamic_filter_switch_guard(const dynamic_filter_switch_guard&)            = delete;
  dynamic_filter_switch_guard& operator=(const dynamic_filter_switch_guard&) = delete;

 private:
  duckdb::Connection& _con;
  bool _original;
};

bool contains(const std::string& haystack, const std::string& needle)
{
  return haystack.find(needle) != std::string::npos;
}

// Visit each distinct dynamic-filter endpoint in the converted pipelines.
template <typename Fn>
std::size_t for_each_endpoint(sirius::pipeline::pipeline_conversion_result& result, const Fn& check)
{
  std::unordered_set<const sirius::op::sirius_physical_operator*> seen;
  std::size_t count = 0;

  auto visit = [&](sirius::op::sirius_physical_operator* op) {
    if (op == nullptr || op->type != sirius::op::SiriusPhysicalOperatorType::DYNAMIC_FILTER) {
      return;
    }
    if (!seen.insert(op).second) { return; }
    ++count;
    check(op);
  };

  for (auto const& pipeline : result.scheduled_pipelines) {
    visit(pipeline->get_source().get());
    visit(pipeline->get_sink().get());
    for (auto& op_ref : pipeline->get_operators()) {
      visit(&op_ref.get());
    }
  }
  return count;
}

// Endpoints consume pipelineable batches, never PARTITION output.
void require_not_fed_partitioned_data(sirius::op::sirius_physical_operator* endpoint)
{
  REQUIRE(endpoint->children.size() == 1);
  INFO("endpoint child is " << sirius::op::SiriusPhysicalOperatorToString(
         endpoint->children[0]->type));
  CHECK(endpoint->children[0]->type != sirius::op::SiriusPhysicalOperatorType::PARTITION);
}

}  // namespace

// A selective part build binds a dynamic-filter endpoint to the lineitem GPU scan.
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

  SECTION("the switch elides the operator")
  {
    dynamic_filter_switch_guard off(con, /*enabled=*/false);
    REQUIRE_FALSE(contains(sirius::test::convert_query_to_dump(con, join_query), "DYNAMIC_FILTER"));
  }
}

TEST_CASE("the dynamic-filter endpoint is never fed partitioned data",
          "[integration][pipeline][dynamic_filter]")
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

  std::size_t endpoints_checked = 0;
  sirius::test::with_conversion_result(
    con, join_query, [&](sirius::pipeline::pipeline_conversion_result& result) {
      endpoints_checked = for_each_endpoint(result, require_not_fed_partitioned_data);
    });

  // Guard against a vacuous pass: this query must actually wire an endpoint.
  REQUIRE(endpoints_checked > 0);
}

TEST_CASE("dynamic-filter endpoints obey the data contract on SIP-shaped plans",
          "[integration][pipeline][dynamic_filter]")
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

  // Q5: five nested joins; the top join's `c_nationkey` comes from a build side.
  const std::string build_side_query =
    "SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue "
    "FROM customer, orders, lineitem, supplier, nation, region "
    "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey "
    "AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey "
    "AND n_regionkey = r_regionkey AND r_name = 'ASIA' "
    "AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01' "
    "GROUP BY n_name ORDER BY revenue DESC";

  // Q3-shaped: a GROUP BY over a join, so any endpoint below it sits under that wrapper.
  const std::string group_by_query =
    "SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate "
    "FROM customer, orders, lineitem "
    "WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey "
    "GROUP BY l_orderkey, o_orderdate ORDER BY revenue DESC";

  SECTION("Q5: the feature wires endpoints, and every endpoint obeys the data contract")
  {
    // The pinned join order puts n_regionkey on the nation join's build side, so only join-edge
    // descent reaches it.
    sirius::test::disabled_optimizers_guard shape(con, "join_order,build_side_probe_side");
    std::size_t endpoints_off = 0;
    {
      dynamic_filter_switch_guard switch_off(con, /*enabled=*/false);
      sirius::test::with_conversion_result(
        con, build_side_query, [&](sirius::pipeline::pipeline_conversion_result& result) {
          endpoints_off = for_each_endpoint(result, require_not_fed_partitioned_data);
        });
    }

    std::size_t endpoints_on = 0;
    {
      dynamic_filter_switch_guard switch_on(con, /*enabled=*/true);
      sirius::test::with_conversion_result(
        con, build_side_query, [&](sirius::pipeline::pipeline_conversion_result& result) {
          endpoints_on = for_each_endpoint(result, require_not_fed_partitioned_data);
        });
    }

    REQUIRE(endpoints_on > endpoints_off);
  }

  SECTION("Q3-shaped: every endpoint below the GROUP BY wrapper obeys the contract")
  {
    // No lower bound on the count here: what this pins is that any endpoint the converter produces
    // on this shape is fed pipelineable data.
    dynamic_filter_switch_guard switch_on(con, /*enabled=*/true);
    sirius::test::with_conversion_result(
      con, group_by_query, [&](sirius::pipeline::pipeline_conversion_result& result) {
        for_each_endpoint(result, require_not_fed_partitioned_data);
      });
  }
}
