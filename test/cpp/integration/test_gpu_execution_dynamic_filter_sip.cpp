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
#include <utils/scoped_sirius_setting.hpp>
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
  return sirius::test::collect_rows(*result);
}

// Dynamic-filter counter deltas for one query execution.
struct publication_deltas {
  std::uint64_t producers_enabled                    = 0;
  std::uint64_t membership_filters_built             = 0;
  std::uint64_t publications_finished                = 0;
  std::uint64_t publications_skipped_build_not_whole = 0;
  std::uint64_t filters_pushed                       = 0;
  std::uint64_t accumulation_publications_finished   = 0;
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
    .filters_pushed = after.filters_pushed - before.filters_pushed,
    .accumulation_publications_finished =
      after.accumulation_publications_finished - before.accumulation_publications_finished};
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

  SECTION("a multi-partition build publishes nothing without accumulation")
  {
    // Correctness stake: a filter built from one partition's slice of the build keys would drop
    // probe rows that do join, so this must be pinned on a build that really spans partitions.
    // Accumulation (on by default) legitimately publishes for such a build, so it is turned off
    // here to pin the whole-build path alone.
    sirius::test::scoped_sirius_setting accumulation_off(
      con, "enable_dynamic_filter_multi_partition", false);
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

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - deduplicated CTE builds admit dynamic filters",
                 "[integration][gpu_execution][dynamic_filter][cte_dedup]")
{
  run_ok("SET gpu_execution = false;");
  run_ok(
    "CREATE TABLE parts AS SELECT (range % 50)::INTEGER AS pk, "
    "concat('p', range % 3) AS pname FROM range(100);");
  run_ok("INSERT INTO parts VALUES (NULL, 'p1'), (1, 'p1');");
  run_ok(
    "CREATE TABLE items AS SELECT range::INTEGER AS id, (range % 50)::INTEGER AS fk, "
    "(range % 3)::INTEGER AS qty FROM range(3000);");
  run_ok("INSERT INTO items VALUES (3000, NULL, 1), (3001, 1, NULL);");
  run_ok("CHECKPOINT;");

  sirius::test::scoped_sirius_setting native_cte(*con, "delim_join_as_cte", true);
  sirius::test::coverage_gate_disable_guard gate_off(*con);
  std::string query;

  SECTION("the generated correlated domain keeps duplicate and nullable rows correct")
  {
    // The physical-plan regression isolates the dedup publisher. This query also has a
    // filtered base-table build, so its counters only establish that filtering ran.
    query =
      "SELECT p.pk, i.id, i.qty FROM items i JOIN parts p ON p.pk = i.fk "
      "WHERE p.pname = 'p1' AND i.qty < "
      "(SELECT AVG(i2.qty) + 1 FROM items i2 WHERE i2.fk = p.pk)";
  }

  SECTION("an unfiltered deduplicated CTE supplies the publication evidence")
  {
    // No filter is visible in the join's build subtree: only dedup-over-CTE evidence can arm
    // its publisher. The sibling consumer keeps the shared materialization in the plan.
    query =
      "WITH shared AS MATERIALIZED (SELECT pk, pname FROM parts) "
      "SELECT i.fk, i.id FROM items i "
      "JOIN (SELECT pk FROM shared GROUP BY pk) d ON i.fk = d.pk "
      "WHERE i.id < 10 "
      "UNION ALL SELECT pk, -1 FROM shared WHERE pname = 'p1'";
  }

  auto const deltas = require_switch_result_equivalence(*con, query);
  REQUIRE(deltas.off.producers_enabled == 0);
  REQUIRE(deltas.on.producers_enabled > 0);
  // A completed publication may skip an already drained scan; admission is deterministic,
  // while delivery to an independently scheduled probe is opportunistic.
  REQUIRE(deltas.on.publications_finished > 0);

  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - a CTE probe does not prune its shared producer",
                 "[integration][gpu_execution][dynamic_filter][cte_dedup]")
{
  run_ok("SET gpu_execution = false;");
  run_ok(
    "CREATE TABLE shared_rows AS SELECT range::INTEGER AS id, "
    "(range % 10)::INTEGER AS k FROM range(1000);");
  run_ok("INSERT INTO shared_rows VALUES (1000, NULL), (1001, NULL), (1002, 9);");
  run_ok("CREATE TABLE selected_keys(k INTEGER, keep INTEGER);");
  run_ok("INSERT INTO selected_keys VALUES (1, 1), (1, 1), (2, 1), (8, 0), (NULL, 1);");
  run_ok("CHECKPOINT;");

  // A filter for the first consumer's {1,2} domain must never reach the shared producer:
  // the second consumer needs other keys and NULLs from that same materialization.
  std::string const query =
    "WITH shared AS MATERIALIZED (SELECT id, k FROM shared_rows), "
    "domain AS MATERIALIZED (SELECT k FROM selected_keys WHERE keep = 1) "
    "SELECT s.id, s.k FROM shared s JOIN (SELECT k FROM domain GROUP BY k) d ON s.k = d.k "
    "UNION ALL SELECT id, k FROM shared WHERE id < 20 OR k IS NULL";
  sirius::test::coverage_gate_disable_guard gate_off(*con);
  auto const deltas = require_switch_result_equivalence(*con, query);
  REQUIRE(deltas.off.producers_enabled == 0);
  REQUIRE(deltas.on.producers_enabled > 0);
  REQUIRE(deltas.rows.size() == 222);
  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - right semi and anti filter admission preserves results",
                 "[integration][gpu_execution][dynamic_filter][right_family]")
{
  std::string const join_type = GENERATE("SEMI", "ANTI");
  std::string const input_case =
    GENERATE("matching", "no matching keys", "runtime empty probe", "runtime empty build");
  bool const enable_accumulation = GENERATE(false, true);
  CAPTURE(join_type, input_case, enable_accumulation);

  run_ok("SET gpu_execution = false;");
  run_ok("CREATE TABLE retained (id INTEGER, k INTEGER, keep INTEGER);");
  run_ok(
    "INSERT INTO retained VALUES "
    "(1, 1, 1), (2, 1, 1), (3, NULL, 1), (4, 3, 1), (5, 10001, 0);");
  auto const probe_key = input_case == "no matching keys" ? "2" : "range % 4";
  run_ok("CREATE TABLE probes AS SELECT (" + std::string(probe_key) +
         ")::INTEGER AS k, "
         "((range % 2) * 100)::INTEGER AS v FROM range(10000);");
  run_ok("INSERT INTO probes VALUES (NULL, 0);");
  run_ok("CHECKPOINT;");

  // Retain the join when key domains are disjoint. The larger probe also keeps the retained
  // relation on the physical build side when its runtime filter produces no rows.
  sirius::test::disabled_optimizers_guard shape(*con, "statistics_propagation");
  sirius::test::coverage_gate_disable_guard gate_off(*con);
  sirius::test::scoped_sirius_setting accumulation(
    *con, "enable_dynamic_filter_multi_partition", enable_accumulation);
  auto const probe_filter = input_case == "runtime empty probe" ? " AND p.v = 50 " : " ";
  auto const build_filter =
    input_case == "runtime empty build" ? "WHERE s.keep = 2" : "WHERE s.keep = 1";
  auto const query = "SELECT s.id, s.k FROM retained s " + join_type +
                     " JOIN probes p ON s.k = p.k" + probe_filter + build_filter;

  auto explanation = con->Query("EXPLAIN " + query);
  REQUIRE(explanation);
  REQUIRE_FALSE(explanation->HasError());
  INFO(explanation->ToString());
  REQUIRE(explanation->ToString().find("Join Type: RIGHT_" + join_type) != std::string::npos);

  auto const deltas = require_switch_result_equivalence(*con, query);
  REQUIRE(deltas.off.producers_enabled == 0);
  if (enable_accumulation) {
    REQUIRE(deltas.on.producers_enabled > 0);
  } else {
    REQUIRE(deltas.on.producers_enabled == 0);
    REQUIRE(deltas.on.accumulation_publications_finished == 0);
  }
  auto const expected_rows = input_case == "runtime empty build" ? 0
                             : input_case == "matching"          ? (join_type == "SEMI" ? 3 : 1)
                                                                 : (join_type == "SEMI" ? 0 : 4);
  REQUIRE(deltas.rows.size() == expected_rows);

  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  compare_gpu_vs_cpu(query);

  // Publication may arrive after the probe drains. Exercise the pruning semantics
  // deterministically as well: remove every probe key that cannot equal a retained build key.
  // This checks the SQL identity independently of join orientation and runtime filtering.
  auto const prefiltered_query =
    "SELECT s.id, s.k FROM retained s " + join_type +
    " JOIN (SELECT * FROM probes WHERE k = 1 OR k = 3) p ON s.k = p.k" + probe_filter +
    build_filter;
  dynamic_filter_switch_guard switch_off(*con, /*enabled=*/false);
  REQUIRE(run_on_gpu(*con, prefiltered_query) == deltas.rows);
  compare_gpu_vs_cpu(prefiltered_query);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - catalog nonnull probes admit null-safe right-family filters",
                 "[integration][gpu_execution][dynamic_filter][right_family][join_key_nullability]")
{
  std::string const join_type = GENERATE("SEMI", "ANTI");
  std::string const key_type  = GENERATE("INTEGER", "BIGINT");
  bool const empty_probe      = GENERATE(false, true);
  CAPTURE(join_type, key_type, empty_probe);
  run_ok("SET gpu_execution = false;");
  run_ok("CREATE TABLE retained (id INTEGER, k " + key_type + ", keep INTEGER);");
  run_ok(
    "INSERT INTO retained VALUES "
    "(1, 1, 1), (2, 1, 1), (3, NULL, 1), (4, 3, 1), (5, 10001, 0);");
  run_ok("CREATE TABLE probes (k " + key_type + " NOT NULL, v INTEGER);");
  run_ok("INSERT INTO probes SELECT range % 4, (range % 2) * 100 FROM range(10000);");
  run_ok("CHECKPOINT;");

  sirius::test::disabled_optimizers_guard shape(*con, "statistics_propagation");
  sirius::test::coverage_gate_disable_guard gate_off(*con);
  sirius::test::scoped_sirius_setting accumulation(
    *con, "enable_dynamic_filter_multi_partition", true);
  auto const probe_filter = empty_probe ? " AND p.v = 50 " : " ";
  auto const query        = "SELECT s.id, s.k FROM retained s " + join_type +
                     " JOIN probes p ON s.k IS NOT DISTINCT FROM p.k" + probe_filter +
                     "WHERE s.keep = 1";
  auto explanation = con->Query("EXPLAIN " + query);
  REQUIRE(explanation);
  REQUIRE_FALSE(explanation->HasError());
  INFO(explanation->ToString());
  REQUIRE(explanation->ToString().find("Join Type: RIGHT_" + join_type) != std::string::npos);

  // The build remains nullable. Its NULL row survives ANTI and never matches this probe,
  // so normalizing null-safe equality must preserve both the match set and duplicate rows.
  auto const deltas = require_switch_result_equivalence(*con, query);
  REQUIRE(deltas.off.producers_enabled == 0);
  REQUIRE(deltas.on.producers_enabled > 0);
  auto const expected_rows =
    empty_probe ? (join_type == "SEMI" ? 0 : 4) : (join_type == "SEMI" ? 3 : 1);
  REQUIRE(deltas.rows.size() == expected_rows);
  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - catalog nullability preserves join output semantics",
                 "[integration][gpu_execution][join_key_nullability]")
{
  run_ok("SET gpu_execution = false;");
  run_ok("CREATE TABLE nullable_rows(id INTEGER, k INTEGER, k2 INTEGER);");
  run_ok(
    "INSERT INTO nullable_rows VALUES "
    "(1, NULL, NULL), (2, 1, 10), (3, 1, 10), (4, 2, 20), (5, 4, NULL), (6, NULL, 10);");
  run_ok("CREATE TABLE nonnull_rows(id INTEGER, k INTEGER NOT NULL, k2 INTEGER NOT NULL);");
  run_ok("INSERT INTO nonnull_rows VALUES (11, 1, 10), (12, 2, 99), (13, 3, 30), (14, 1, 10);");
  run_ok("CHECKPOINT;");
  sirius::test::disabled_optimizers_guard shape(*con, "statistics_propagation");

  SECTION("LEFT preserves nullable rows and pads unmatched output")
  {
    compare_gpu_vs_cpu(
      "SELECT n.id, c.id, n.k, c.k FROM nullable_rows n "
      "LEFT JOIN nonnull_rows c ON n.k IS NOT DISTINCT FROM c.k");
  }
  SECTION("FULL pads unmatched rows on both sides")
  {
    compare_gpu_vs_cpu(
      "SELECT n.id, c.id, n.k, c.k FROM nullable_rows n "
      "FULL JOIN nonnull_rows c ON n.k IS NOT DISTINCT FROM c.k");
  }
  SECTION("ANTI keeps NULLs from the nullable probe")
  {
    compare_gpu_vs_cpu(
      "SELECT n.* FROM nullable_rows n ANTI JOIN nonnull_rows c "
      "ON n.k IS NOT DISTINCT FROM c.k");
  }
  SECTION("equality and null-safe composite keys normalize together")
  {
    compare_gpu_vs_cpu(
      "SELECT n.id, c.id FROM nullable_rows n JOIN nonnull_rows c "
      "ON n.k = c.k AND n.k2 IS NOT DISTINCT FROM c.k2");
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - prepared null-safe join rebinds after dropping NOT NULL",
                 "[integration][gpu_execution][join_key_nullability][prepared_statement]")
{
  run_ok("SET gpu_execution = false;");
  run_ok("CREATE TABLE nullable_rows(id INTEGER, k INTEGER);");
  run_ok("INSERT INTO nullable_rows VALUES (1, NULL), (2, 1);");
  run_ok("CREATE TABLE constrained_rows(id INTEGER, k INTEGER NOT NULL);");
  run_ok("INSERT INTO constrained_rows VALUES (11, 1);");
  run_ok("CHECKPOINT;");
  sirius::test::disabled_optimizers_guard shape(*con, "statistics_propagation");
  std::string const query =
    "SELECT n.id, c.id FROM nullable_rows n LEFT JOIN constrained_rows c "
    "ON n.k IS NOT DISTINCT FROM c.k";
  run_ok("SET gpu_execution = true;");
  auto prepared = con->Prepare(query);
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  auto execute_and_compare = [&] {
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu          = prepared->Execute();
    REQUIRE(gpu);
    INFO((gpu->HasError() ? gpu->GetError() : "prepared GPU execution"));
    REQUIRE_FALSE(gpu->HasError());
    auto const after = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    auto rows = collect_rows(*gpu);
    run_ok("SET gpu_execution = false;");
    auto cpu = con->Query(query);
    REQUIRE(cpu);
    REQUIRE_FALSE(cpu->HasError());
    REQUIRE(rows == collect_rows(*cpu));
    run_ok("SET gpu_execution = true;");
    return rows;
  };
  REQUIRE(execute_and_compare() ==
          std::vector<std::vector<std::string>>{{"1", "NULL"}, {"2", "11"}});
  run_ok("SET gpu_execution = false;");
  run_ok("ALTER TABLE constrained_rows ALTER COLUMN k DROP NOT NULL;");
  run_ok("INSERT INTO constrained_rows VALUES (12, NULL);");
  run_ok("CHECKPOINT;");
  run_ok("SET gpu_execution = true;");
  REQUIRE(execute_and_compare() == std::vector<std::vector<std::string>>{{"1", "12"}, {"2", "11"}});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - early CTE build filtering preserves shared consumers",
                 "[integration][gpu_execution][dynamic_filter][right_family][cte_dedup]")
{
  std::string const join_type = GENERATE("SEMI", "ANTI");
  bool const deduplicate      = GENERATE(false, true);
  CAPTURE(join_type, deduplicate);
  run_ok("SET gpu_execution = false;");
  run_ok("CREATE TABLE retained(id INTEGER, k INTEGER);");
  run_ok("INSERT INTO retained VALUES (1, 1), (1, 1), (2, 3), (3, NULL), (4, 9);");
  run_ok("CREATE TABLE probes AS SELECT (range % 2)::INTEGER AS k FROM range(1000);");
  run_ok("CHECKPOINT;");
  sirius::test::disabled_optimizers_guard shape(*con, "statistics_propagation");
  sirius::test::coverage_gate_disable_guard gate_off(*con);
  sirius::test::scoped_sirius_setting accumulation(
    *con, "enable_dynamic_filter_multi_partition", true);
  auto const grouping = deduplicate ? " GROUP BY id, k" : "";
  auto const query =
    "WITH shared AS MATERIALIZED (SELECT id, k FROM retained) "
    "SELECT s.id, s.k FROM (SELECT id, k FROM shared" +
    std::string(grouping) + ") s " + join_type +
    " JOIN probes p ON s.k = p.k UNION ALL SELECT id, k FROM shared";
  auto explanation = con->Query("EXPLAIN " + query);
  REQUIRE(explanation);
  REQUIRE_FALSE(explanation->HasError());
  INFO(explanation->ToString());
  REQUIRE(explanation->ToString().find("Join Type: RIGHT_" + join_type) != std::string::npos);
  auto const deltas = require_switch_result_equivalence(*con, query);
  REQUIRE(deltas.off.producers_enabled == 0);
  REQUIRE(deltas.on.producers_enabled > 0);
  auto const expected_rows = join_type == "SEMI" ? (deduplicate ? 6 : 7) : 8;
  REQUIRE(deltas.rows.size() == expected_rows);
  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  compare_gpu_vs_cpu(query);
}
