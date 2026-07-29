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

//! RAII toggle for the temporary SIP switch (default off). The SET mutates the shared
//! SiriusContext, which outlives this test, so the destructor restores whatever value the
//! constructor found rather than the default -- a literal restore would clobber an enclosing guard.
//! The SET result is checked: an unregistered or misspelled option would otherwise leave the switch
//! untouched and every assertion below would still pass.
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

//! RAII disable of DuckDB's statistics propagation. On live table statistics that pass derives
//! table filters from join-key value ranges -- on the integration data `o_custkey = c_custkey`
//! yields `c_custkey <= 14999`, because TPC-H customers whose key is divisible by three place no
//! orders -- so a query with no written predicate can still plan a build that genuinely filters,
//! for DuckDB's own build_side_has_filter hint exactly as for Sirius's mirror. The filter-free
//! sections need the premise "no predicate means no filtering build", so they remove the
//! synthesizing pass rather than weakening their counter anchors.
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

//! Run `query` on the GPU -- asserted through the transparent-execution counters, not assumed --
//! and return its rows as a sorted bag.
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

//! What one run of a query moved in the connection's dynamic-filter counters.
struct publication_deltas {
  std::uint64_t producers_enabled        = 0;
  std::uint64_t membership_filters_built = 0;
  std::uint64_t filters_pushed           = 0;
};

//! Both runs' counter deltas: `off` is the same query with the switch off, `on` with it on.
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

//! Run `query` twice, with the SIP switch off and then on, require the two row bags to be
//! identical, and return both runs' publication-counter deltas so each case can assert what it can
//! attribute.
//!
//! Row equality is the correctness statement: a join-edge endpoint only pre-filters rows the
//! producing join would drop anyway. All queries here aggregate to integers, so the comparison is
//! exact. The counters are the liveness statement, and they separate three failure stages:
//! `producers_enabled` (a plan-time constructor increment) says a target was wired at all --
//! discovery creates a target only when a key actually binds, so an enabled producer always has
//! at least one bound key -- `membership_filters_built` says the publisher constructed a filter
//! for a bound key, and `filters_pushed` says one reached a channel.
//!
//! The two delivery counters are deterministic here because the default task-creator strategy is
//! `active` (`src/include/creator/config.hpp`), under which publication completes on build-batch
//! arrival, before any probe-subtree operator is activated. Under the opt-in `lookahead` strategy
//! that ordering does not hold (design-v2 "Coordination") and only `producers_enabled`, being a
//! plan-time fact, stays exact.
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

//! End-to-end coverage for SIP (sideways information passing) endpoint placement. What this file
//! verifies is that turning the switch on changes no result row, and that on a filter-free join
//! shape it is the only route that can wire a target -- which the publication counters establish.
//!
//! It pins no plan shape. Which join produces, which side is its build, and where a key ends up are
//! all optimizer choices on the transparent path, so a shape written into the SQL is not a shape
//! any assertion here can hold to. The shape-dependent claims -- build-side placement, LEFT
//! build-block descent, null-equal rejection -- live in `test_plan_tree_shape.cpp`, which pins the
//! optimizer passes it depends on and can therefore fail for the reason it names.
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
    // The query carries no predicate anywhere, and the statistics guard keeps DuckDB from
    // synthesizing one out of join-key ranges. `build_subtree_is_filtering` is therefore false
    // for every subtree under every join order, so scan-route discovery is disarmed for every
    // join and no scan target is wired at all. Every target the SIP-on run produces is SIP's,
    // whatever plan the optimizer chose -- the attribution is structural rather than a property
    // of the written join order.
    //
    // Both guards remove a confound rather than weakening the case: an unfiltered build is the
    // shape most likely to arm the domain-coverage gate, that gate has its own suite, and
    // suppressing it can only make filters more available.
    statistics_propagation_disable_guard stats_off(con);
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_sip_result_equivalence(con,
                                     "SELECT count(*), sum(o.o_custkey) "
                                     "FROM lineitem l "
                                     "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                     "JOIN customer c ON o.o_custkey = c.c_custkey");

    // Self-verifying: if a future edit reintroduces a predicate, this fails and names the reason
    // instead of degrading the anchors below into a vacuous comparison of two equal counts.
    REQUIRE(deltas.off.producers_enabled == 0);

    // Asserted in publication order, so a failure names the stage that broke: the switch wired a
    // target, the publisher built a filter for a bound direct key, the filter reached a channel.
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }

  SECTION("a LEFT join in the query keeps results identical while SIP is active")
  {
    // Filter-free for the same reason as the section above, guards included, so the same
    // structural attribution holds. The LEFT build-block descent rule itself is not observable
    // from execution -- it is pinned by `join_block_descent`'s unit coverage and by the LEFT
    // plan-shape section in `test_plan_tree_shape.cpp`, which can assert the join type survived
    // optimization.
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
    // What this verifies: turning SIP on over a producing join written `IS NOT DISTINCT FROM`
    // changes no result row. It does not kill the B1 mutation -- TPC-H `o_custkey` and `c_custkey`
    // are both non-null, so null-equal and `equal` agree on this data. Admission's equality clause
    // is pinned by the `COMPARE_NOT_DISTINCT_FROM` case in `test_dynamic_filter_key_admission.cpp`
    // and end to end by the null-equal section in `test_plan_tree_shape.cpp`.
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
    // No customer row matches, so the outer join's build is empty and its endpoint never receives a
    // filter. What this verifies is the pass-through path, not the publisher's empty-build skip:
    // the inner join in the same query publishes normally, so the counters cannot attribute a skip
    // to the outer one (that skip is unit-tested in `test_dynamic_filter_publisher.cpp`).
    //
    // The predicate is empty on the data but not statically refutable: TPC-H phone numbers never
    // carry an all-zero group, yet this value lies inside the column's [min, max], so statistics
    // propagation cannot erase the scan and collapse the join before a producer is ever planned.
    //
    // This case carries no counter anchor. Making a build empty requires a predicate, and a
    // predicate lets DuckDB wire scan routes under some join orders and not others, so no
    // order-independent attribution exists here. The claim the case does make -- a channel that
    // receives nothing drops no rows -- is exactly what result equivalence establishes. File-level
    // liveness is carried by the two filter-free cases above.
    require_sip_result_equivalence(con,
                                   "SELECT count(*), sum(o.o_custkey) "
                                   "FROM lineitem l "
                                   "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                   "JOIN customer c ON o.o_custkey = c.c_custkey "
                                   "WHERE c.c_phone = '25-000-000-0000'");
  }
}
