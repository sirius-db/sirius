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
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

// Restore the previous SIP setting so nested guards remain valid.
struct sip_switch_guard {
  sip_switch_guard(duckdb::Connection& c, bool enabled)
    : con(c),
      original(sirius::test::get_registered_sirius_context(c)
                 ->get_config()
                 .get_operator_params()
                 .enable_dynamic_filter_sip)
  {
    auto result = con.Query(std::string{"SET enable_dynamic_filter_sip = "} +
                            (enabled ? "true" : "false") + ";");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  ~sip_switch_guard()
  {
    con.Query(std::string{"SET enable_dynamic_filter_sip = "} + (original ? "true" : "false") +
              ";");
  }

  sip_switch_guard(const sip_switch_guard&)            = delete;
  sip_switch_guard& operator=(const sip_switch_guard&) = delete;

  duckdb::Connection& con;
  bool original;
};

// Disable inferred join-key filters so predicate-free queries keep an unfiltered build.
struct statistics_propagation_disable_guard {
  explicit statistics_propagation_disable_guard(duckdb::Connection& c) : con(c)
  {
    auto current = con.Query("SELECT current_setting('disabled_optimizers');");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    original    = current->GetValue(0, 0).ToString();
    auto merged = original.empty() ? std::string{"statistics_propagation"}
                                   : original + ",statistics_propagation";
    auto result = con.Query("SET disabled_optimizers = '" + merged + "';");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  ~statistics_propagation_disable_guard()
  {
    con.Query("SET disabled_optimizers = '" + original + "';");
  }

  statistics_propagation_disable_guard(const statistics_propagation_disable_guard&) = delete;
  statistics_propagation_disable_guard& operator=(const statistics_propagation_disable_guard&) =
    delete;

  duckdb::Connection& con;
  std::string original;
};

// Verify GPU execution and return the result as a sorted bag.
std::vector<std::vector<std::string>> run_on_gpu(duckdb::Connection& con, const std::string& query)
{
  auto const before = sirius::test::get_transparent_execution_stats(con);

  auto result = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) {
    UNSCOPED_INFO("transparent GPU execution error: " << result->GetError());
  }
  REQUIRE_FALSE(result->HasError());

  auto const after = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  return sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
}

// Dynamic-filter counter deltas for one query execution.
struct publication_deltas {
  std::uint64_t producers_enabled        = 0;
  std::uint64_t membership_filters_built = 0;
  std::uint64_t filters_pushed           = 0;
};

// Counter deltas for the same query with SIP disabled and enabled.
struct sip_comparison {
  publication_deltas off;
  publication_deltas on;
};

publication_deltas run_and_measure(duckdb::Connection& con,
                                   const std::string& query,
                                   std::vector<std::vector<std::string>>& rows)
{
  auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
  rows              = run_on_gpu(con, query);
  auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
  return publication_deltas{
    .producers_enabled        = after.producers_enabled - before.producers_enabled,
    .membership_filters_built = after.membership_filters_built - before.membership_filters_built,
    .filters_pushed           = after.filters_pushed - before.filters_pushed};
}

// Compare SIP-disabled and SIP-enabled results and return their counter deltas. With the active
// task creator, publication completes before the probe subtree starts, so delivery counters are
// deterministic for these queries.
sip_comparison require_sip_result_equivalence(duckdb::Connection& con, const std::string& query)
{
  con.Query("SET gpu_execution = true;");

  sip_comparison deltas;
  std::vector<std::vector<std::string>> off_rows;
  {
    sip_switch_guard sip_off(con, /*enabled=*/false);
    deltas.off = run_and_measure(con, query, off_rows);
  }

  std::vector<std::vector<std::string>> on_rows;
  {
    sip_switch_guard sip_on(con, /*enabled=*/true);
    deltas.on = run_and_measure(con, query, on_rows);
  }

  REQUIRE(on_rows == off_rows);
  return deltas;
}

}  // namespace

// Verify that SIP preserves results and is the only target route for predicate-free joins.
// Shape-specific placement is covered by test_plan_tree_shape.cpp.
TEST_CASE("gpu_execution - SIP endpoint placement preserves results",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  auto db_path = sirius::test::integration_tpch_db_path();
  auto r       = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  SECTION("SIP is the only route a filter-free join can take")
  {
    // Suppress inferred predicates and the independent coverage gate. With no filtering build,
    // scan-route discovery cannot create a target, so any enabled producer comes from SIP.
    statistics_propagation_disable_guard stats_off(con);
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_sip_result_equivalence(con,
                                     "SELECT count(*), sum(o.o_custkey) "
                                     "FROM lineitem l "
                                     "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                     "JOIN customer c ON o.o_custkey = c.c_custkey");

    // Confirm that the switch-off plan has no other dynamic-filter route.
    REQUIRE(deltas.off.producers_enabled == 0);

    // Check each publication stage independently.
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }

  SECTION("a LEFT join in the query keeps results identical while SIP is active")
  {
    // Use the same structural attribution as the predicate-free inner-join case. Dedicated
    // plan-shape tests cover LEFT-join descent.
    statistics_propagation_disable_guard stats_off(con);
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_sip_result_equivalence(con,
                                     "SELECT count(*), sum(o.o_custkey) "
                                     "FROM lineitem l "
                                     "LEFT JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                     "JOIN customer c ON o.o_custkey = c.c_custkey");

    REQUIRE(deltas.off.producers_enabled == 0);
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }

  SECTION("a null-equal producing condition still returns identical results")
  {
    // These keys are non-null, so result parity cannot verify null-equal rejection. Admission and
    // plan-shape tests cover that rule directly.
    require_sip_result_equivalence(
      con,
      "SELECT count(*), sum(o.o_custkey) "
      "FROM lineitem l "
      "JOIN orders o ON l.l_orderkey = o.o_orderkey "
      "JOIN customer c ON o.o_custkey IS NOT DISTINCT FROM c.c_custkey "
      "WHERE c.c_nationkey = 3 "
      "AND l.l_shipdate < DATE '1992-03-01'");
  }

  SECTION("an endpoint whose channel receives no filter passes rows through")
  {
    // The customer predicate produces no rows but remains inside the column statistics, preserving
    // the join in the plan. The empty-build endpoint receives no filter and must pass rows through.
    require_sip_result_equivalence(con,
                                   "SELECT count(*), sum(o.o_custkey) "
                                   "FROM lineitem l "
                                   "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                   "JOIN customer c ON o.o_custkey = c.c_custkey "
                                   "WHERE c.c_phone = '25-000-000-0000'");
  }
}
