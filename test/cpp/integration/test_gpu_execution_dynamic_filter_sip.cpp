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
#include <utility>
#include <vector>

namespace {

// Set the dynamic-filter flag for one scope and restore it on exit.
struct dynamic_filter_switch_guard {
  dynamic_filter_switch_guard(duckdb::Connection& c, bool enabled)
    : con(c),
      original(sirius::test::get_registered_sirius_context(c)
                 ->get_config()
                 .get_operator_params()
                 .enable_dynamic_filter)
  {
    auto result =
      con.Query(std::string{"SET enable_dynamic_filter = "} + (enabled ? "true" : "false") + ";");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  ~dynamic_filter_switch_guard()
  {
    con.Query(std::string{"SET enable_dynamic_filter = "} + (original ? "true" : "false") + ";");
  }

  dynamic_filter_switch_guard(const dynamic_filter_switch_guard&)            = delete;
  dynamic_filter_switch_guard& operator=(const dynamic_filter_switch_guard&) = delete;

  duckdb::Connection& con;
  bool original;
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
  std::uint64_t producers_enabled                    = 0;
  std::uint64_t membership_filters_built             = 0;
  std::uint64_t publications_finished                = 0;
  std::uint64_t publications_skipped_build_not_whole = 0;
  std::uint64_t filters_pushed                       = 0;
};

struct switch_comparison {
  publication_deltas off;
  publication_deltas on;
  std::vector<std::vector<std::string>> rows;
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
    .publications_finished    = after.publications_finished - before.publications_finished,
    .publications_skipped_build_not_whole =
      after.publications_skipped_build_not_whole - before.publications_skipped_build_not_whole,
    .filters_pushed = after.filters_pushed - before.filters_pushed};
}

// Publication completes before these probes, making the enabled/disabled counter deltas
// deterministic.
switch_comparison require_switch_result_equivalence(duckdb::Connection& con,
                                                    const std::string& query)
{
  con.Query("SET gpu_execution = true;");

  switch_comparison deltas;
  std::vector<std::vector<std::string>> off_rows;
  {
    dynamic_filter_switch_guard switch_off(con, /*enabled=*/false);
    deltas.off = run_and_measure(con, query, off_rows);
  }

  std::vector<std::vector<std::string>> on_rows;
  {
    dynamic_filter_switch_guard switch_on(con, /*enabled=*/true);
    deltas.on = run_and_measure(con, query, on_rows);
  }

  REQUIRE(on_rows == off_rows);
  deltas.rows = std::move(on_rows);
  return deltas;
}

}  // namespace

TEST_CASE("gpu_execution - opaque-build and build-block routes preserve results",
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

  SECTION("a key on the inner join's build side, reached by build-block descent")
  {
    // The pinned join order puts o_custkey on an inner build side, so only build-block descent can
    // target it. Inferred predicates and the coverage gate are disabled to keep attribution stable.
    sirius::test::disabled_optimizers_guard shape(
      con, "statistics_propagation,join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_switch_result_equivalence(con,
                                        "SELECT count(*), sum(o.o_custkey) "
                                        "FROM lineitem l "
                                        "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                        "JOIN customer c ON o.o_custkey = c.c_custkey "
                                        "WHERE c.c_nationkey = 3");

    REQUIRE(deltas.off.producers_enabled == 0);
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }

  SECTION("a LEFT join in the query keeps results identical while SIP is active")
  {
    // Dedicated plan-shape tests cover LEFT-join descent.
    sirius::test::disabled_optimizers_guard shape(
      con, "statistics_propagation,join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_switch_result_equivalence(con,
                                        "SELECT count(*), sum(o.o_custkey) "
                                        "FROM lineitem l "
                                        "LEFT JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                        "JOIN customer c ON o.o_custkey = c.c_custkey "
                                        "WHERE c.c_nationkey = 3");

    REQUIRE(deltas.off.producers_enabled == 0);
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }

  SECTION("a RIGHT join admits build-block descent")
  {
    sirius::test::disabled_optimizers_guard shape(
      con, "statistics_propagation,join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_switch_result_equivalence(con,
                                        "SELECT count(*), count(o.o_orderkey) "
                                        "FROM orders o "
                                        "RIGHT JOIN customer c ON o.o_custkey = c.c_custkey "
                                        "JOIN nation n ON c.c_nationkey = n.n_nationkey "
                                        "WHERE n.n_regionkey = 3");

    REQUIRE(deltas.off.producers_enabled == 0);
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
    REQUIRE(deltas.rows.size() == 1);
    REQUIRE(deltas.rows[0].size() == 2);
    CHECK(std::stoull(deltas.rows[0][0]) > std::stoull(deltas.rows[0][1]));
  }

  SECTION("a FULL OUTER join admits build-block descent under equality semantics")
  {
    sirius::test::disabled_optimizers_guard shape(
      con, "statistics_propagation,join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas =
      require_switch_result_equivalence(con,
                                        "SELECT count(*), count(o.o_orderkey) "
                                        "FROM orders o "
                                        "FULL OUTER JOIN customer c ON o.o_custkey = c.c_custkey "
                                        "JOIN nation n ON c.c_nationkey = n.n_nationkey "
                                        "WHERE n.n_regionkey = 3");

    REQUIRE(deltas.off.producers_enabled == 0);
    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
    REQUIRE(deltas.rows.size() == 1);
    REQUIRE(deltas.rows[0].size() == 2);
    CHECK(std::stoull(deltas.rows[0][0]) > std::stoull(deltas.rows[0][1]));
  }

  SECTION("RIGHT, FULL OUTER, and ANTI joins admit probe-block descent")
  {
    sirius::test::disabled_optimizers_guard shape(
      con, "statistics_propagation,join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    std::vector<std::string> const queries{
      "SELECT count(*), sum(o.o_orderkey) "
      "FROM orders o "
      "RIGHT JOIN customer c ON o.o_custkey = c.c_custkey "
      "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
      "WHERE l.l_shipdate < DATE '1992-02-01'",
      "SELECT count(*), sum(o.o_orderkey) "
      "FROM orders o "
      "FULL OUTER JOIN customer c ON o.o_custkey = c.c_custkey "
      "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
      "WHERE l.l_shipdate < DATE '1992-02-01'",
      "SELECT count(*), sum(o.o_orderkey) "
      "FROM orders o "
      "ANTI JOIN nation n ON o.o_custkey = n.n_nationkey "
      "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
      "WHERE l.l_shipdate < DATE '1992-02-01'"};

    for (auto const& query : queries) {
      CAPTURE(query);
      auto const deltas = require_switch_result_equivalence(con, query);

      REQUIRE(deltas.off.producers_enabled == 0);
      REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
      REQUIRE(deltas.on.membership_filters_built > deltas.off.membership_filters_built);
      REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
    }
  }

  SECTION("a null-equal producing condition still returns identical results")
  {
    // These keys are non-null, so result parity cannot verify null-equal rejection. Admission and
    // plan-shape tests cover that rule directly.
    require_switch_result_equivalence(
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
    // the join in the plan.
    require_switch_result_equivalence(con,
                                      "SELECT count(*), sum(o.o_custkey) "
                                      "FROM lineitem l "
                                      "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                                      "JOIN customer c ON o.o_custkey = c.c_custkey "
                                      "WHERE c.c_phone = '25-000-000-0000'");
  }

  SECTION("a single-partition MIXED_JOIN publishes through the partition fold")
  {
    // An equality plus an inequality condition puts the join in MIXED_JOIN mode, which
    // compute_hash_join_partition_strategy excludes from BUILD_PROBE.
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas = require_switch_result_equivalence(
      con,
      "select count(*) from orders o "
      "join (select l_orderkey, l_shipdate from lineitem "
      "      where l_shipdate < date '1992-02-01') l "
      "on o.o_orderkey = l.l_orderkey and o.o_orderdate < l.l_shipdate");

    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.publications_finished > deltas.off.publications_finished);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }

  SECTION("a multi-partition build publishes nothing")
  {
    // Correctness stake: a filter built from one partition's slice of the build keys would drop
    // probe rows that do join, so this must be pinned on a build that really spans partitions.
    // The summed columns exist only to keep the projection from pruning them, widening the build
    // past broadcast candidacy. Reaching a genuinely partition-sliced build needs all three pins:
    // broadcast off, a small hash-partition target so the natural count exceeds one, and a build
    // budget below the build size so BUILD_PROBE cannot fold it back to one table per GPU.
    sirius::test::disabled_optimizers_guard shape(
      con, "statistics_propagation,join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    sirius::test::scoped_setting no_broadcast(con, "max_broadcast_join_size", 1);
    sirius::test::scoped_setting small_partitions(con, "hash_partition_bytes", 8ULL * 1024 * 1024);
    sirius::test::scoped_setting small_build_budget(
      con, "max_build_hash_table_bytes", 8ULL * 1024 * 1024);
    auto const deltas = require_switch_result_equivalence(
      con,
      "select count(*), sum(l.l_partkey), sum(l.l_suppkey), sum(l.l_linenumber), "
      "       sum(l.l_quantity), sum(l.l_extendedprice), sum(l.l_discount), sum(l.l_tax) "
      "from orders o join lineitem l on o.o_orderkey = l.l_orderkey "
      "where l.l_shipdate >= date '1992-01-01'");

    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.publications_finished == 0);
    REQUIRE(deltas.on.publications_skipped_build_not_whole > 0);
  }

  SECTION("an unfiltered aggregate build supplies no publication evidence")
  {
    // The join order is pinned so the aggregate stays the build side. With no filter in that
    // visible subtree and no opaque build root, enabling dynamic filters must arm no producer.
    sirius::test::disabled_optimizers_guard shape(con, "join_order,build_side_probe_side");
    sirius::test::coverage_gate_disable_guard gate_off(con);
    auto const deltas = require_switch_result_equivalence(
      con,
      "select count(*) from lineitem l "
      "join (select l_orderkey from lineitem group by l_orderkey) g "
      "on l.l_orderkey = g.l_orderkey");

    REQUIRE(deltas.off.producers_enabled == 0);
    REQUIRE(deltas.on.producers_enabled == deltas.off.producers_enabled);
  }

  SECTION("TPC-H q17: a delim-scan build wires only through opaque-build evidence")
  {
    auto const deltas =
      require_switch_result_equivalence(con,
                                        "select sum(l.l_extendedprice) / 7.0 as avg_yearly "
                                        "from lineitem l, part p "
                                        "where p.p_partkey = l.l_partkey "
                                        "and p.p_brand = 'Brand#13' "
                                        "and p.p_container = 'JUMBO CAN' "
                                        "and l.l_quantity < ("
                                        "select 0.2 * avg(l2.l_quantity) "
                                        "from lineitem l2 "
                                        "where l2.l_partkey = p.p_partkey)");

    REQUIRE(deltas.on.producers_enabled > deltas.off.producers_enabled);
    REQUIRE(deltas.on.filters_pushed > deltas.off.filters_pushed);
  }
}
