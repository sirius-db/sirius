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
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace {

// Aggregates over orders joined to customers whose strings the aggregate groups by, which Sirius
// computes per customer key before the join. Customers 5 and 7 appear twice with different names,
// so the join repeats their orders; some orders have a NULL or an unknown customer, some customers
// have no orders, one customer's orders all have a NULL price, and one customer has a NULL key.
class PartialAggregatePushdownFixture : public sirius::test::GpuExecutionFixture {
 public:
  PartialAggregatePushdownFixture()
  {
    enable_guard.emplace(*con, "enable_partial_aggregate_below_join", true);
    run_ok(
      "CREATE TABLE customer (c_custkey BIGINT, c_name VARCHAR, c_address VARCHAR, "
      "c_nationkey INTEGER, c_acctbal DECIMAL(15, 2));");
    run_ok(
      "INSERT INTO customer SELECT range, 'Customer#' || range, 'Street ' || (range * 7), "
      "range % 5, range * 1.25 FROM range(1, 61);");
    run_ok(
      "INSERT INTO customer VALUES (5, 'Customer#5 again', 'Elsewhere', 2, 10.00), "
      "(7, 'Customer#7 again', 'Elsewhere', 3, 20.00), (NULL, 'Nobody', 'Nowhere', 1, 0.00);");
    run_ok("CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR);");
    run_ok("INSERT INTO nation SELECT range, 'Nation ' || range FROM range(5);");
    run_ok(
      "CREATE TABLE orders (o_orderkey BIGINT, o_custkey BIGINT, o_orderdate DATE, "
      "o_totalprice DECIMAL(15, 2), o_shippriority INTEGER);");
    // Customer keys 1..80: 61..80 match no customer. Customer 9's prices are all NULL.
    run_ok(
      "INSERT INTO orders SELECT range, CASE WHEN range % 53 = 0 THEN NULL ELSE range % 80 + 1 "
      "END, DATE '1994-01-01' + ((range * 11) % 300)::INTEGER, "
      "CASE WHEN range % 80 + 1 = 9 THEN NULL ELSE ((range * 37) % 20000 - 5000) / 100 END, "
      "range % 3 FROM range(4000);");
    run_ok(
      "CREATE TABLE lineitem (l_orderkey BIGINT, l_extendedprice DECIMAL(15, 2), "
      "l_discount DECIMAL(15, 2), l_returnflag VARCHAR);");
    run_ok(
      "INSERT INTO lineitem SELECT (range * 3) % 4200, ((range * 13) % 90000) / 100, "
      "(range % 11) / 100, CASE WHEN range % 2 = 0 THEN 'R' ELSE 'N' END FROM range(16000);");
    run_ok("CHECKPOINT;");
  }

  std::uint64_t pushdowns()
  {
    return sirius::test::get_transparent_execution_stats(*con).partial_aggregate_pushdowns;
  }

  /// The pushed and the unpushed GPU plans against the CPU. With @p pushed, the rule must fire
  /// when enabled; it must never fire when disabled.
  void compare_pushed_and_not(std::string const& query, bool pushed = false)
  {
    INFO(query);
    auto const before = pushdowns();
    compare_gpu_vs_cpu(query);
    auto const after = pushdowns();
    if (pushed) { CHECK(after > before); }
    sirius::test::scoped_sirius_setting off{*con, "enable_partial_aggregate_below_join", false};
    compare_gpu_vs_cpu(query);
    CHECK(pushdowns() == after);
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;
};

}  // namespace

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregate below the join that adds its group-by strings",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // One join: the orders are summed per customer key, then joined to the customers' strings.
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, c_address, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey GROUP BY c_custkey, c_name, c_address",
    true);
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_totalprice) AS total, min(o_totalprice) AS low, "
    "max(o_orderdate) AS latest FROM customer, orders WHERE c_custkey = o_custkey "
    "GROUP BY c_custkey, c_name",
    true);
  // The filter is on the aggregated side, which the join keeps.
  compare_pushed_and_not(
    "SELECT c_name, c_address, min(o_totalprice) AS low, max(o_totalprice) AS high "
    "FROM customer, orders WHERE c_custkey = o_custkey AND o_shippriority = 1 "
    "GROUP BY c_name, c_address",
    true);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregate below joins in the shape of TPC-H q10",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, "
    "n_name, c_address FROM customer, orders, lineitem, nation "
    "WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey "
    "AND o_orderdate >= DATE '1994-03-01' AND o_orderdate < DATE '1994-06-01' "
    "AND l_returnflag = 'R' AND c_nationkey = n_nationkey "
    "GROUP BY c_custkey, c_name, c_acctbal, n_name, c_address");
  // A group column from the aggregated side travels with the join key.
  compare_pushed_and_not(
    "SELECT c_name, o_shippriority, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey GROUP BY c_custkey, c_name, o_shippriority");
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregate below a join with no rows to aggregate",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey AND o_orderdate > DATE '2100-01-01' GROUP BY c_custkey, c_name");
  // Only customer 9, whose prices are all NULL: SUM and MIN stay NULL through both steps.
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_totalprice) AS total, min(o_totalprice) AS low "
    "FROM customer, orders WHERE c_custkey = o_custkey AND o_custkey = 9 "
    "GROUP BY c_custkey, c_name");
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregates the partial aggregate pushdown leaves alone",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  auto const before = pushdowns();
  // COUNT and AVG do not combine into a result of their own type.
  compare_gpu_vs_cpu(
    "SELECT c_custkey, c_name, count(*) AS n FROM customer, orders "
    "WHERE c_custkey = o_custkey GROUP BY c_custkey, c_name");
  // The customers are filtered, so the join discards orders an aggregate would have summed.
  compare_gpu_vs_cpu(
    "SELECT c_custkey, c_name, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey AND c_acctbal < 20 GROUP BY c_custkey, c_name");
  // No strings to carry.
  compare_gpu_vs_cpu(
    "SELECT c_custkey, c_nationkey, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey GROUP BY c_custkey, c_nationkey");
  CHECK(pushdowns() == before);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution keeps zeros of either sign apart under a computed join key",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // 0.0 and -0.0 group together, but 1 / k sends them to different rows of dims. Summing per k
  // first would give one of the two names the total of both.
  run_ok("CREATE TABLE facts (k DOUBLE, v DECIMAL(15, 2));");
  run_ok("INSERT INTO facts VALUES (0.0, 1.00), (-0.0, 2.00), (0.0, 4.00), (0.5, 8.00);");
  run_ok("CREATE TABLE dims (k DOUBLE, name VARCHAR, extra VARCHAR);");
  run_ok(
    "INSERT INTO dims VALUES ('Infinity'::DOUBLE, 'pos', 'p'), ('-Infinity'::DOUBLE, 'neg', 'n'), "
    "(2.0, 'two', 't');");
  run_ok("CHECKPOINT;");
  auto const before = pushdowns();
  compare_gpu_vs_cpu(
    "SELECT d.name, d.extra, sum(f.v) AS total FROM facts f JOIN dims d ON 1.0 / f.k = d.k "
    "GROUP BY d.name, d.extra");
  CHECK(pushdowns() == before);
}
