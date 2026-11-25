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
#include <utils/parquet_fixture_utils.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
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
  /// when enabled; it must never fire when disabled. The columns in @p inexact (0-based, never the
  /// first) hold floating-point sums whose last digits depend on the order of addition, and match
  /// within a relative tolerance.
  void compare_pushed_and_not(std::string const& query,
                              bool pushed                          = false,
                              std::set<std::size_t> const& inexact = {})
  {
    INFO(query);
    auto const compare = [&] {
      if (inexact.empty()) {
        compare_gpu_vs_cpu(query);
      } else {
        compare_gpu_vs_cpu_approx(query, inexact, 1e-9);
      }
    };
    auto const before = pushdowns();
    compare();
    auto const after = pushdowns();
    if (pushed) { CHECK(after > before); }
    sirius::test::scoped_sirius_setting off{*con, "enable_partial_aggregate_below_join", false};
    compare();
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
    "SELECT c_custkey, c_address, min(o_totalprice) AS low, max(o_totalprice) AS high "
    "FROM customer, orders WHERE c_custkey = o_custkey AND o_shippriority = 1 "
    "GROUP BY c_custkey, c_address",
    true);
  // The join key is grouped by on the orders' side.
  compare_pushed_and_not(
    "SELECT o_custkey, c_name, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey GROUP BY o_custkey, c_name",
    true);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution sums of integers below a join",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // An INTEGER and a BIGINT sum to a HUGEINT, and so does the sum of the per-key sums. Customer
  // 9's prices are all NULL, so the third sum has nothing to add for that customer.
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_shippriority) AS units, sum(o_orderkey) AS keys, "
    "sum(CASE WHEN o_totalprice IS NULL THEN NULL ELSE o_shippriority END) AS some_units, "
    "min(o_shippriority) AS low FROM customer, orders WHERE c_custkey = o_custkey "
    "GROUP BY c_custkey, c_name",
    true);
  // No rows to add up.
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_shippriority) AS units FROM customer, orders "
    "WHERE c_custkey = o_custkey AND o_orderdate > DATE '2100-01-01' GROUP BY c_custkey, c_name",
    true);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution integer sums past the BIGINT range below a join",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // Forty rows of 9e18 per key leave the BIGINT range. Keys 61 to 80 match no customer: their
  // sums are computed below the join and then dropped by it, and must not fail on the way.
  // Customer 3's sum is past the range in the result itself.
  run_ok("CREATE TABLE payments (p_custkey BIGINT, p_amount BIGINT);");
  run_ok(
    "INSERT INTO payments SELECT range % 80 + 1, CASE WHEN range % 80 + 1 > 60 OR "
    "range % 80 + 1 = 3 THEN 9000000000000000000 ELSE range % 1000 END FROM range(3200);");
  run_ok("CHECKPOINT;");
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(p_amount) AS paid, max(p_amount) AS most "
    "FROM customer, payments WHERE c_custkey = p_custkey GROUP BY c_custkey, c_name",
    true);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution sums of 128-bit values stay above the join",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // The GPU refuses a 128-bit sum when a batch's row count times its largest value could
  // overflow. Key 99 matches no customer, and the square of its 9e18 nearly fills a HUGEINT:
  // summed below the join, that row would send the query to the CPU. The rule leaves the sum
  // above the join, where the GPU never sees the row.
  run_ok("CREATE TABLE ledger (g_custkey BIGINT, g_amount BIGINT);");
  run_ok(
    "INSERT INTO ledger VALUES (1, 3), (1, 4), (2, 5), (99, 9000000000000000000), "
    "(99, 9000000000000000000);");
  run_ok("CHECKPOINT;");
  auto const before = pushdowns();
  compare_gpu_vs_cpu(
    "SELECT c_custkey, c_name, sum(g_amount::HUGEINT * g_amount::HUGEINT) AS squares, "
    "max(g_amount) AS most FROM customer, ledger WHERE c_custkey = g_custkey "
    "GROUP BY c_custkey, c_name");
  CHECK(pushdowns() == before);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregates of arguments that branch on arithmetic below a join",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // TPC-DS q50's shape: rows counted per range of a difference, as sums of a CASE. The join key
  // is left out of the arguments: DuckDB would read the customers' copy of it instead.
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, "
    "sum(CASE WHEN o_orderkey - o_shippriority <= 30 THEN 1 ELSE 0 END) AS near, "
    "sum(CASE WHEN o_orderkey - o_shippriority > 30 AND o_orderkey - o_shippriority <= 600 "
    "THEN 1 ELSE 0 END) AS further, "
    "sum(CASE WHEN o_shippriority * 2 > 2 OR o_totalprice < 0 THEN o_totalprice ELSE 0 END) "
    "AS priced, max(CASE WHEN o_shippriority + 1 > 2 THEN o_orderkey * 2 ELSE o_orderkey END) "
    "AS highest FROM customer, orders WHERE c_custkey = o_custkey GROUP BY c_custkey, c_name",
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
                 "gpu_execution floating-point sums below a join",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // Halves and quarters add up exactly in any order, so the totals match the CPU's digit for
  // digit. Customer 9's prices are all NULL.
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_shippriority * 0.25::DOUBLE) AS quarters, "
    "sum(o_shippriority * 0.5::DOUBLE) AS halves, "
    "sum(CASE WHEN o_totalprice IS NULL THEN NULL ELSE 0.5::DOUBLE END) AS some "
    "FROM customer, orders WHERE c_custkey = o_custkey GROUP BY c_custkey, c_name",
    true);
  // Tenths do not: the last digits of these totals depend on the order of addition, which the
  // CPU, the GPU and the pushed plan each choose differently. Every value has the same sign, so
  // nothing cancels and the totals agree to many more digits than the tolerance asks for. The name
  // leads because it is unique, which a customer key is not.
  compare_pushed_and_not(
    "SELECT c_name, c_custkey, sum(o_orderkey * 0.1::DOUBLE) AS tenths, "
    "sum(o_shippriority * 0.3::DOUBLE + 0.7::DOUBLE) AS sevenths "
    "FROM customer, orders WHERE c_custkey = o_custkey GROUP BY c_custkey, c_name",
    true,
    {2, 3});
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution floating-point sums that overflow below a join",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // Two values add up to the same total in either order, so each customer's total is one the rule
  // cannot change: customer 1's overflows, customer 2's cancels to zero and customer 3's
  // overflows the other way. Key 99 matches no customer, so its sum overflows below the join and
  // nowhere else.
  run_ok("CREATE TABLE readings (r_custkey BIGINT, r_value DOUBLE);");
  run_ok(
    "INSERT INTO readings VALUES (1, 1e308), (1, 1e308), (2, 1e308), (2, -1e308), (3, -1e308), "
    "(3, -1e308), (99, 1e308), (99, 1e308);");
  run_ok("CHECKPOINT;");
  std::string const totals =
    "SELECT c_custkey, c_name, sum(r_value) AS total FROM customer, readings "
    "WHERE c_custkey = r_custkey GROUP BY c_custkey, c_name";
  compare_pushed_and_not(totals, true);
  // A filter on the total keeps the same customer with and without the rule.
  compare_pushed_and_not(totals + " HAVING sum(r_value) = 0", true);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregate below an outer join that keeps its side",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // Orders without a customer, and with a NULL key, stay in the result with a NULL name: one group
  // per such key when the orders' key is grouped by.
  compare_pushed_and_not(
    "SELECT o_custkey, c_name, sum(o_totalprice) AS total, max(o_orderdate) AS latest "
    "FROM orders LEFT JOIN customer ON c_custkey = o_custkey GROUP BY o_custkey, c_name",
    true);
  // The same join written from the customers' side.
  compare_pushed_and_not(
    "SELECT o_custkey, c_name, sum(o_totalprice) AS total, min(o_totalprice) AS low "
    "FROM customer RIGHT JOIN orders ON c_custkey = o_custkey GROUP BY o_custkey, c_name",
    true);
  // No orders at all.
  compare_pushed_and_not(
    "SELECT o_custkey, c_name, sum(o_totalprice) AS total FROM orders LEFT JOIN customer "
    "ON c_custkey = o_custkey WHERE o_orderdate > DATE '2100-01-01' GROUP BY o_custkey, c_name",
    true);
  // Grouped by the customers' key, the orders without a customer all fall into the one group
  // whose key is NULL. The groups are then not the orders' keys, and going by the grouping the
  // rule leaves it alone.
  std::string const padded =
    "SELECT c_custkey, c_name, sum(o_totalprice) AS total, min(o_totalprice) AS low "
    "FROM orders LEFT JOIN customer ON c_custkey = o_custkey GROUP BY c_custkey, c_name";
  {
    sirius::test::scoped_sirius_setting by_grouping{
      *con, "partial_aggregate_min_rows_per_key", 0.0};
    auto const before = pushdowns();
    compare_gpu_vs_cpu(padded);
    CHECK(pushdowns() == before);
  }
  // The orders' statistics show each customer key holds about 50 of them, which is reason enough:
  // the sums of the keys without a customer are added up in the NULL group.
  compare_pushed_and_not(padded, true);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregate below a join whose keys the statistics show repeat",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // The strings are grouped by without the join key. Customers 5 and 7 each have two names, and
  // several keys share an address, so the groups are not the keys. The 4000 orders hold 80 customer
  // keys, 50 orders each, so a sum per key is worth computing first all the same.
  std::string const by_strings =
    "SELECT c_name, c_address, sum(o_totalprice) AS total, max(o_orderdate) AS latest "
    "FROM customer, orders WHERE c_custkey = o_custkey GROUP BY c_name, c_address";
  compare_pushed_and_not(by_strings, true);
  // The same orders and customers as Parquet files. A footer holds no distinct count, but the
  // orders' customer keys lie between 1 and 80: no more than 80 keys for the 4000 orders.
  sirius::test::scratch_dir files{"partial_aggregate_pushdown"};
  for (char const* table : {"orders", "customer"}) {
    auto const file = files.file_literal(std::string{table} + ".parquet");
    run_ok(std::string{"COPY "} + table + " TO " + file + ";");
    run_ok(std::string{"CREATE VIEW file_"} + table + " AS SELECT * FROM read_parquet(" + file +
           ");");
  }
  compare_pushed_and_not(
    "SELECT c_name, c_address, sum(o_totalprice) AS total, max(o_orderdate) AS latest "
    "FROM file_customer, file_orders WHERE c_custkey = o_custkey GROUP BY c_name, c_address",
    true);

  // A threshold no key reaches leaves it to the grouping.
  sirius::test::scoped_sirius_setting unreachable{*con, "partial_aggregate_min_rows_per_key", 1e9};
  auto const before = pushdowns();
  compare_gpu_vs_cpu(by_strings);
  CHECK(pushdowns() == before);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregates the partial aggregate pushdown leaves alone",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // The statistics are left out, so that the grouping alone decides.
  sirius::test::scoped_sirius_setting by_grouping{*con, "partial_aggregate_min_rows_per_key", 0.0};
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
  // The strings are grouped by without the join key. Customers 5 and 7 each have two names, and
  // several keys share an address, so the groups are not the keys.
  compare_gpu_vs_cpu(
    "SELECT c_name, c_address, sum(o_totalprice) AS total FROM customer, orders "
    "WHERE c_custkey = o_custkey GROUP BY c_name, c_address");
  // The join to lineitem reads the order key, which nothing above groups by. Summed below the
  // customers' join, the orders would be grouped one by one.
  compare_gpu_vs_cpu(
    "SELECT c_custkey, c_name, l_returnflag, sum(o_totalprice) AS total "
    "FROM customer, orders, lineitem WHERE c_custkey = o_custkey AND o_orderkey = l_orderkey "
    "GROUP BY c_custkey, c_name, l_returnflag");
  CHECK(pushdowns() == before);
}

TEST_CASE_METHOD(PartialAggregatePushdownFixture,
                 "gpu_execution aggregate below a join that discards most of its groups",
                 "[integration][gpu_execution][partial_agg_pushdown]")
{
  // With the kept-share threshold at 0 the rule no longer minds that the customers are filtered:
  // every customer key is summed, and the join then drops the sums of the customers filtered out.
  sirius::test::scoped_sirius_setting any_other_side{*con, "partial_aggregate_min_kept_share", 0.0};
  compare_pushed_and_not(
    "SELECT c_custkey, c_name, sum(o_totalprice) AS total, max(o_orderdate) AS latest "
    "FROM customer, orders WHERE c_custkey = o_custkey AND c_acctbal < 20 "
    "GROUP BY c_custkey, c_name",
    true);
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
