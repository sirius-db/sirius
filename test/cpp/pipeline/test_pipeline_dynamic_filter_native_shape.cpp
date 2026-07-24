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
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

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

//! Walk every operator reachable from a converted pipeline set and hand each DYNAMIC_FILTER
//! endpoint to `check` exactly once. Returns how many distinct endpoints were seen, so a caller
//! can reject a vacuous pass.
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

//! The invariant every endpoint must satisfy: it masks a plain batch, so its input must be
//! pipelineable -- never the partitioned data a PARTITION emits.
void require_not_fed_partitioned_data(sirius::op::sirius_physical_operator* endpoint)
{
  REQUIRE(endpoint->children.size() == 1);
  INFO("endpoint child is " << sirius::op::SiriusPhysicalOperatorToString(
         endpoint->children[0]->type));
  CHECK(endpoint->children[0]->type != sirius::op::SiriusPhysicalOperatorType::PARTITION);
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

//! The endpoint applies its mask to a plain batch, so it must be fed pipelineable data -- never the
//! partitioned data a PARTITION produces. `wrap_join_child` wraps whatever occupies a join's child
//! slot as `CONCAT -> PARTITION -> <that child>`, so an endpoint already sitting in that slot lands
//! on the source side of the PARTITION. Issue #1010's SIP placement depends on that holding for an
//! endpoint inserted anywhere in a join's subtree, including on a build input, so pin it here: if
//! the wrapper passes are ever reordered, this fails loudly instead of silently mis-shaping the
//! plan. Conversion only -- no GPU execution.
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

//! Same invariant, on the shapes issue #1010's SIP placement introduces and that nothing
//! exercises today: an endpoint on a join's **build** input, and one beneath a GROUP BY's wrapper.
//!
//! `wrap_join_child` wraps `join.children[idx]` as `CONCAT -> PARTITION -> <child>` for **both**
//! children (`wrap_join`, `sirius_physical_plan_generator.cpp:506-515`), and `wrap_hash_group_by`
//! inserts its own PARTITION, so an endpoint already occupying either slot should land on the
//! source side. That is reasoning, not evidence, until a plan actually contains such an endpoint.
//!
//! TPC-H Q5 is the natural carrier: its top join needs `c_nationkey`, produced three levels down
//! on the **build** side of nested joins, so DuckDB's probe-spine walk never reaches it -- exactly
//! the Case 1b shape this feature targets. Q3 supplies the GROUP BY. Conversion only, no GPU
//! execution.
//!
//! Until placement lands, this asserts the reachable half -- every endpoint the converter does
//! produce on these shapes obeys the contract -- and pins the shapes so the build-side and
//! GROUP-BY cases are covered the moment SIP starts creating endpoints there, rather than having
//! to be remembered at that point.
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

  for (auto const& query : {build_side_query, group_by_query}) {
    DYNAMIC_SECTION("query: " << query.substr(0, 48) << "...")
    {
      sirius::test::with_conversion_result(
        con, query, [&](sirius::pipeline::pipeline_conversion_result& result) {
          // No lower bound on the count: until SIP placement lands, these plans may wire zero
          // endpoints. The contract asserted is that any endpoint present obeys the invariant.
          for_each_endpoint(result, require_not_fed_partitioned_data);
        });
    }
  }
}
