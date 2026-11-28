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

#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_partial_aggregate_pushdown.hpp"
#include "transparent/sirius_optimizer_extension.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;

namespace {

/// How the joins of a test query are ordered.
enum class join_order {
  optimized,   ///< By DuckDB's join order optimizer
  as_written,  ///< As the query writes them, to pin the shape the rule sees
};

/// The plan as nested joins and aggregates over table names. An aggregate shows its number of
/// groups; operators that only pass rows through are left out.
std::string describe(LogicalOperator& op)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto const table = op.Cast<LogicalGet>().GetTable();
      return table ? table->name.GetIdentifierName() : "?";
    }
    case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      return "AGG[" + std::to_string(op.Cast<LogicalAggregate>().groups.size()) + "](" +
             describe(*op.children[0]) + ")";
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      return "JOIN(" + describe(*op.children[0]) + "," + describe(*op.children[1]) + ")";
    default: return op.children.size() == 1 ? describe(*op.children[0]) : "?";
  }
}

/// Give every scan without a row estimate one: its table's row count, or a fifth of it when the
/// scan is filtered, which is DuckDB's default for a filter. DuckDB's join order optimizer is what
/// estimates scans, so plans kept in written order have none.
void estimate_scans(ClientContext& context, LogicalOperator& op)
{
  if (op.type == LogicalOperatorType::LOGICAL_GET && !op.has_estimated_cardinality) {
    auto const rows = op.EstimateCardinality(context);
    op.SetEstimatedCardinality(op.Cast<LogicalGet>().table_filters.HasFilters() ? rows / 5 : rows);
  }
  for (auto& child : op.children) {
    estimate_scans(context, *child);
  }
}

using pushdown_options = sirius::planner::partial_aggregate_pushdown_options;
using pushdown_result  = sirius::planner::partial_aggregate_pushdown_result;
using reasons          = std::vector<std::string>;

/// Parse and optimize @p query, copy the plan as transparent execution does, apply the rewrite
/// with @p options when @p rewrite is set, copy the result again as plan validation does, then
/// resolve it and hand it to @p consume inside a transaction. Returns what the rewrite did.
template <class Consume>
pushdown_result with_resolved_plan(Connection& con,
                                   std::string const& query,
                                   join_order order,
                                   bool rewrite,
                                   pushdown_options const& options,
                                   Consume&& consume)
{
  pushdown_result pushed;
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  if (order == join_order::as_written) {
    disabled.insert(OptimizerType::JOIN_ORDER);
    disabled.insert(OptimizerType::BUILD_SIDE_PROBE_SIDE);
  }

  con.Query("BEGIN TRANSACTION");
  try {
    Parser parser(context);
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    Optimizer optimizer(*planner.binder, context);
    auto optimized = optimizer.Optimize(std::move(planner.plan));
    estimate_scans(context, *optimized);
    auto plan = sirius::transparent::copy_logical_plan(*optimized, context);
    if (rewrite) {
      pushed = sirius::planner::push_partial_aggregates_below_joins(
        context, *planner.binder, plan, options);
      plan = sirius::transparent::copy_logical_plan(*plan, context);
    }
    plan->ResolveOperatorTypes();
    ColumnBindingResolver verifier(true);
    verifier.VisitOperator(*plan);
    consume(context, plan);
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }
  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return pushed;
}

struct pushdown_fixture {
  pushdown_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    // The GPU seq_scan ingestible requires a single-file block manager: an on-disk database.
    db  = sirius::test::open_sirius_db(db_dir.file("partial_aggregate_pushdown.db").c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    for (char const* sql :
         {"CREATE TABLE customer (c_custkey BIGINT, c_name VARCHAR, c_address VARCHAR, "
          "c_nationkey INTEGER, c_acctbal DECIMAL(15, 2), c_region INTEGER, c_clerk VARCHAR, "
          "c_weight DOUBLE)",
          "INSERT INTO customer SELECT range, 'Customer#' || range, 'Street ' || range, "
          "range % 25, range * 1.5, range % 3, 'Clerk#' || (range % 10), range * 0.5 "
          "FROM range(1, 301)",
          "CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR)",
          "INSERT INTO nation SELECT range, 'N' || range FROM range(25)",
          "CREATE TABLE orders (o_orderkey BIGINT, o_custkey BIGINT, o_region INTEGER, "
          "o_orderdate DATE, o_totalprice DECIMAL(15, 2), o_clerk VARCHAR, "
          "o_shippriority INTEGER, o_weight DOUBLE, o_units INTEGER, o_wide DECIMAL(38, 2))",
          "INSERT INTO orders SELECT range, range % 300 + 1, (range % 300 + 1) % 3, "
          "DATE '1994-01-01' + (range % 365)::INTEGER, (range % 1000) * 2.5, "
          "'Clerk#' || (range % 10), range % 3, range * 0.5, range % 7, range FROM range(3000)",
          "CREATE TABLE lineitem (l_orderkey BIGINT, l_extendedprice DECIMAL(15, 2), "
          "l_discount DECIMAL(15, 2), l_returnflag VARCHAR)",
          "INSERT INTO lineitem SELECT range % 3000, (range % 1000) * 1.5, (range % 10) / 100.0, "
          "CASE WHEN range % 2 = 0 THEN 'R' ELSE 'N' END FROM range(12000)",
          // 100 visits per customer key, in five regions.
          "CREATE TABLE visits (v_custkey BIGINT, v_region INTEGER, v_amount DECIMAL(15, 2))",
          "INSERT INTO visits SELECT range % 300 + 1, range % 5, range % 100 FROM range(30000)",
          // One account per customer key.
          "CREATE TABLE accounts (a_custkey BIGINT PRIMARY KEY, a_balance DECIMAL(15, 2))",
          "INSERT INTO accounts SELECT range, range * 2.5 FROM range(1, 301)",
          // One member per customer key, known by an id of its own: the PRIMARY KEY, where the
          // key the orders join on is declared nothing.
          "CREATE TABLE members (m_custkey BIGINT, m_id VARCHAR PRIMARY KEY, m_name VARCHAR, "
          "m_tier INTEGER)",
          "INSERT INTO members SELECT range, 'M' || range, 'Member#' || range, range % 3 "
          "FROM range(1, 301)"}) {
      auto result = con->Query(sql);
      INFO(sql);
      REQUIRE_FALSE(result->HasError());
    }
  }

  /// The thresholds with the statistics left out, so that a test of the other conditions does not
  /// turn on what DuckDB estimates of these small tables.
  static constexpr pushdown_options by_grouping{.min_rows_per_key = 0};

  std::size_t pushed(std::string const& query, pushdown_options const& options = by_grouping)
  {
    return with_resolved_plan(
             *con, query, join_order::optimized, true, options, [](ClientContext&, auto&) {})
      .pushed;
  }

  /// Why the rule left the aggregates of @p query alone: one entry per aggregate over a join.
  /// With @p order as written the answer does not turn on how DuckDB orders the joins.
  reasons why(std::string const& query,
              pushdown_options const& options = by_grouping,
              join_order order                = join_order::optimized)
  {
    return with_resolved_plan(*con, query, order, true, options, [](ClientContext&, auto&) {})
      .declined;
  }

  /// The shape of @p query's plan with its joins as written, with or without the rewrite. The
  /// plan must also be one Sirius can run.
  std::string shape(std::string const& query,
                    bool rewrite,
                    pushdown_options const& options = by_grouping)
  {
    std::string result;
    with_resolved_plan(*con,
                       query,
                       join_order::as_written,
                       rewrite,
                       options,
                       [&](ClientContext& context, auto& plan) {
                         result = describe(*plan);
                         ColumnBindingResolver resolver;
                         resolver.VisitOperator(*plan);
                         sirius::planner::sirius_physical_plan_generator gen(context);
                         REQUIRE(gen.create_plan(std::move(plan)));
                       });
    return result;
  }

  sirius::test::scratch_dir db_dir{"partial_aggregate_pushdown"};
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

/// TPC-H q10 over @p from, which joins the four tables.
std::string q10_shape(std::string const& from)
{
  return "SELECT c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) AS revenue, "
         "c_acctbal, n_name, c_address FROM " +
         from +
         " WHERE o_orderdate >= DATE '1994-03-01' AND o_orderdate < DATE '1994-06-01' "
         "AND l_returnflag = 'R' GROUP BY c_custkey, c_name, c_acctbal, n_name, c_address";
}

/// An aggregate of orders grouped by a customer's key and name.
std::string per_customer(std::string const& aggregates,
                         std::string const& filter = "",
                         std::string const& groups = "c_custkey, c_name")
{
  return "SELECT " + groups + ", " + aggregates +
         " FROM orders, customer WHERE o_custkey = c_custkey " + filter + " GROUP BY " + groups;
}

/// The orders' total per group, with the customers joined first and, above them, lineitem, whose
/// join reads the order key.
std::string through_lineitem(std::string const& groups)
{
  return "SELECT " + groups +
         ", sum(o_totalprice) FROM (orders JOIN customer ON o_custkey = c_custkey) "
         "JOIN lineitem ON o_orderkey = l_orderkey GROUP BY " +
         groups;
}

}  // namespace

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown sums q10's revenue before the customer join",
                 "[partial_agg_pushdown][plan]")
{
  // lineitem and orders first, then customer, then nation: the order Sirius runs q10 in.
  auto const left_deep = q10_shape(
    "lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON c_custkey = o_custkey "
    "JOIN nation ON c_nationkey = n_nationkey");
  CHECK(shape(left_deep, false) == "AGG[5](JOIN(JOIN(JOIN(lineitem,orders),customer),nation))");
  CHECK(shape(left_deep, true) ==
        "AGG[5](JOIN(JOIN(AGG[1](JOIN(lineitem,orders)),customer),nation))");

  // The same joins with every side swapped.
  auto const mirrored = q10_shape(
    "nation JOIN (customer JOIN (orders JOIN lineitem ON l_orderkey = o_orderkey) "
    "ON c_custkey = o_custkey) ON c_nationkey = n_nationkey");
  CHECK(shape(mirrored, true) ==
        "AGG[5](JOIN(nation,JOIN(customer,AGG[1](JOIN(orders,lineitem)))))");

  // customer joined to nation first, so one join brings every string.
  auto const bushy = q10_shape(
    "(lineitem JOIN orders ON l_orderkey = o_orderkey) JOIN "
    "(customer JOIN nation ON c_nationkey = n_nationkey) ON c_custkey = o_custkey");
  CHECK(shape(bushy, true) == "AGG[5](JOIN(AGG[1](JOIN(lineitem,orders)),JOIN(customer,nation)))");
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown leaves q10 alone when lineitem is joined last",
                 "[partial_agg_pushdown][plan]")
{
  // The only place to aggregate is all of lineitem, before the filtered orders discard most of
  // it.
  auto const lineitem_last = q10_shape(
    "lineitem JOIN (orders JOIN (customer JOIN nation ON c_nationkey = n_nationkey) "
    "ON c_custkey = o_custkey) ON l_orderkey = o_orderkey");
  CHECK(shape(lineitem_last, true) == "AGG[5](JOIN(lineitem,JOIN(orders,JOIN(customer,nation))))");
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown covers SUM, MIN and MAX over one side of the join",
                 "[partial_agg_pushdown][plan]")
{
  CHECK(pushed(per_customer("sum(o_totalprice)")) == 1);
  CHECK(pushed(per_customer("min(o_totalprice), max(o_orderdate)")) == 1);
  CHECK(pushed(per_customer("sum(o_totalprice), min(o_totalprice), max(o_clerk)")) == 1);
  // The join key is also aggregated, and a filter on the aggregated side does not matter.
  CHECK(pushed(per_customer("min(o_custkey), sum(o_totalprice)", "AND o_shippriority = 1")) == 1);
  // The join key is grouped by on the aggregated side instead of the customers'.
  CHECK(pushed(per_customer("sum(o_totalprice)", "", "o_custkey, c_name")) == 1);
  // A group column of the aggregated side travels with the join key.
  CHECK(pushed(per_customer("sum(o_totalprice)", "", "c_custkey, c_name, o_shippriority")) == 1);
  // A two-column join key.
  CHECK(pushed("SELECT c_custkey, c_region, c_name, sum(o_totalprice) FROM orders, customer "
               "WHERE o_custkey = c_custkey AND o_region = c_region "
               "GROUP BY c_custkey, c_region, c_name") == 1);
  // Two aggregates share an expression, which DuckDB computes in a projection under them.
  CHECK(pushed(per_customer("sum(o_totalprice * (1 - o_totalprice)), "
                            "min(o_totalprice * (1 - o_totalprice))")) == 1);
  // Under the default settings a floating-point division by zero yields inf or NaN, so it
  // cannot fail.
  CHECK(pushed(per_customer("min(o_units / o_shippriority)")) == 1);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown covers sums of integers of up to 64 bits",
                 "[partial_agg_pushdown][plan]")
{
  // An INTEGER and a BIGINT sum to a HUGEINT, and the sum of those sums is the same HUGEINT.
  CHECK(pushed(per_customer("sum(o_units)")) == 1);
  CHECK(pushed(per_customer("sum(o_orderkey)")) == 1);
  CHECK(pushed(per_customer("sum(CAST(o_units AS BIGINT)), sum(o_totalprice), min(o_units)")) == 1);
  // The plan is one Sirius can run.
  auto const joined =
    "SELECT c_custkey, c_name, sum(o_units), sum(o_orderkey) "
    "FROM orders JOIN customer ON o_custkey = c_custkey GROUP BY c_custkey, c_name";
  CHECK(shape(joined, false) == "AGG[2](JOIN(orders,customer))");
  CHECK(shape(joined, true) == "AGG[2](JOIN(AGG[1](orders),customer))");
  // A sum of 128-bit values stays above the join, and keeps the sums beside it there. The GPU
  // refuses one when a batch's row count times its largest value could overflow, and below the
  // join the batch holds rows the join would have removed.
  CHECK(pushed(per_customer("sum(o_units), sum(o_wide)")) == 0);
  // MIN and MAX of such values add nothing up.
  CHECK(pushed(per_customer("min(o_wide), max(CAST(o_units AS HUGEINT))")) == 1);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown covers floating-point sums",
                 "[partial_agg_pushdown][plan]")
{
  // A DOUBLE sums to a DOUBLE, and so does a sum of such sums.
  CHECK(pushed(per_customer("sum(o_weight)")) == 1);
  CHECK(pushed(per_customer("sum(o_weight), sum(o_totalprice), min(o_weight)")) == 1);
  CHECK(pushed(per_customer("sum(o_weight * 2 + 1)")) == 1);
  CHECK(shape(per_customer("sum(o_weight)"), true) == "AGG[2](JOIN(AGG[1](orders),customer))");
  // Every order without a customer falls into one group here. Summed per customer key first, two
  // keys could each overflow where the group's sum would not, so the sum stays above the join.
  CHECK(pushed("SELECT c_custkey, c_name, sum(o_weight) FROM orders LEFT JOIN customer "
               "ON o_custkey = c_custkey GROUP BY c_custkey, c_name") == 0);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown covers arguments that branch on arithmetic",
                 "[partial_agg_pushdown][plan]")
{
  // What can throw is the arithmetic alone, under CASE, a comparison, AND and OR.
  CHECK(pushed(per_customer(
          "sum(CASE WHEN o_units * o_shippriority > 4 THEN o_totalprice ELSE 0 END)")) == 1);
  CHECK(pushed(per_customer("sum(CASE WHEN o_units - o_shippriority <= 3 THEN 1 ELSE 0 END)")) ==
        1);
  CHECK(pushed(per_customer("max(CASE WHEN o_units + 1 > 2 AND o_shippriority * 2 < 4 "
                            "OR o_units - 1 = 0 THEN o_units * 2 ELSE o_units END)")) == 1);
  auto const branching =
    "SELECT c_custkey, c_name, sum(CASE WHEN o_units - o_shippriority <= 3 THEN 1 ELSE 0 END) "
    "FROM orders JOIN customer ON o_custkey = c_custkey GROUP BY c_custkey, c_name";
  CHECK(shape(branching, true) == "AGG[2](JOIN(AGG[1](orders),customer))");
  // A cast that can fail still stands in the way, wherever it sits.
  CHECK(pushed(per_customer(
          "sum(CASE WHEN CAST(o_clerk AS INTEGER) > 1 THEN o_totalprice ELSE 0 END)")) == 0);
}

// What the rule leaves alone, with its reason, is in "reports why it leaves an aggregate alone".
TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown declines joins where it would not pay off",
                 "[partial_agg_pushdown][plan]")
{
  // A column a join further up reads is a key too. Below the customers' join the orders would be
  // grouped by their own key, one group per order, which pays only when the aggregate above keeps
  // the orders apart as well.
  CHECK(pushed(through_lineitem("c_custkey, c_name, l_orderkey, l_returnflag")) == 1);
  // The aggregated side is already grouped.
  CHECK(pushed("SELECT c_custkey, c_name, min(spent) FROM customer, "
               "(SELECT o_custkey, min(o_totalprice) AS spent FROM orders GROUP BY o_custkey) "
               "WHERE o_custkey = c_custkey GROUP BY c_custkey, c_name") == 0);
  // A join key computed from a DOUBLE: 0.0 and -0.0 group together, but the key can tell them
  // apart. The bare column and a key computed from an integer are fine.
  auto const joined_on = [](std::string const& condition, std::string const& key) {
    return "SELECT " + key + ", c_name, sum(o_totalprice) FROM orders, customer WHERE " +
           condition + " GROUP BY " + key + ", c_name";
  };
  CHECK(pushed(joined_on("o_weight = c_weight", "c_weight")) == 1);
  CHECK(pushed(joined_on("o_custkey + 1 = c_custkey", "o_custkey")) == 1);
  CHECK(pushed(joined_on("CAST(o_weight AS VARCHAR) = c_clerk", "o_weight")) == 0);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown reports why it leaves an aggregate alone",
                 "[partial_agg_pushdown][plan]")
{
  // An aggregate it pushes, and one that is not over a join, report nothing.
  CHECK(why(per_customer("sum(o_totalprice)")).empty());
  CHECK(why("SELECT o_clerk, sum(o_totalprice) FROM orders GROUP BY o_clerk").empty());

  // Aggregates it cannot combine.
  CHECK(why(per_customer("count(*)")) == reasons{"aggregate:count_star"});
  // DuckDB counts rows instead when the counted column holds no NULL.
  CHECK(why(per_customer("count(o_totalprice), sum(o_totalprice)")) ==
        reasons{"aggregate:count_star"});
  CHECK(why(per_customer("count(NULLIF(o_units, 0))")) == reasons{"aggregate:count"});
  CHECK(why(per_customer("avg(o_totalprice)")) == reasons{"aggregate:avg"});
  CHECK(why(per_customer("avg(o_units)")) == reasons{"aggregate:avg"});
  CHECK(why(per_customer("avg(o_weight)")) == reasons{"aggregate:avg"});
  CHECK(why(per_customer("sum(DISTINCT o_totalprice)")) == reasons{"aggregate:distinct"});
  CHECK(why(per_customer("sum(o_totalprice) FILTER (WHERE o_shippriority = 1)")) ==
        reasons{"aggregate:filter"});
  CHECK(why(per_customer("sum(o_shippriority > 1)")) == reasons{"sum:other"});
  CHECK(why(per_customer("sum(CAST(o_units AS HUGEINT))")) == reasons{"sum:hugeint"});
  CHECK(why(per_customer("sum(o_wide)")) == reasons{"sum:decimal128"});
  // DuckDB keeps the product of two 64-bit DECIMALs in 64 bits, so its sum is one the rule pushes.
  CHECK(why(per_customer("sum(o_totalprice * o_totalprice)")).empty());

  // Arguments: one that reads both sides, and ones that can fail other than through arithmetic.
  CHECK(why(per_customer("sum(o_totalprice * c_acctbal)")) == reasons{"arguments:both-sides"});
  CHECK(why(per_customer(
          "sum(CASE WHEN o_units // o_shippriority > 1 THEN o_totalprice ELSE 0 END)")) ==
        reasons{"argument:fallible"});
  CHECK(why(per_customer("max(CAST(o_clerk AS INTEGER))")) == reasons{"argument:fallible"});
  CHECK(why(per_customer("min(o_units // o_shippriority)")) == reasons{"argument:fallible"});

  // Joins and keys.
  CHECK(why(per_customer("sum(o_totalprice)", "", "c_custkey, c_nationkey")) ==
        reasons{"path:no-payload"});
  CHECK(why(per_customer("sum(o_totalprice)", "AND c_acctbal > 400")) ==
        reasons{"other-side:filtered"});
  // A filter DuckDB keeps above the scan is not trusted to keep rows, whatever it is estimated
  // to keep.
  CHECK(
    why(per_customer("sum(o_totalprice)", "AND c_clerk IN ('Clerk#1', 'Clerk#2', 'Clerk#3')")) ==
    reasons{"other-side:opaque; other-side:filtered"});
  CHECK(why(per_customer("sum(o_totalprice)", "", "c_name")) ==
        reasons{"keys:not-fewer-than-groups; keys:not-grouped"});
  CHECK(why(per_customer("sum(o_totalprice)", "", "c_name, c_address")) ==
        reasons{"keys:not-grouped"});
  // The join to lineitem reads the order key, which no group stands for.
  CHECK(why(through_lineitem("c_custkey, c_name, l_returnflag")) == reasons{"keys:not-grouped"});
  CHECK(why("SELECT c_clerk, c_address, sum(o_totalprice) FROM orders, customer "
            "WHERE o_clerk = c_clerk GROUP BY c_clerk, c_address") ==
        reasons{"keys:variable-width"});
  CHECK(
    why(per_customer("sum(o_totalprice)", "", "c_custkey, c_name, o_shippriority, o_orderdate")) ==
    reasons{"keys:too-many-extra"});
  CHECK(why("SELECT o_weight, c_name, sum(o_totalprice) FROM orders, customer "
            "WHERE 1.0 / o_weight = c_weight GROUP BY o_weight, c_name") ==
        reasons{"key:computed-inexact"});
  CHECK(why("SELECT c_custkey, c_name, sum(o_totalprice) FROM customer LEFT JOIN orders "
            "ON o_custkey = c_custkey GROUP BY c_custkey, c_name") == reasons{"join:outer-padded"});
  CHECK(why("SELECT sum(o_totalprice) FROM orders, customer WHERE o_custkey = c_custkey") ==
        reasons{"shape:ungrouped"});

  // Every independent reason is reported, so that it shows which limits would all have to go.
  CHECK(why(per_customer("count(*)", "AND c_acctbal > 400")) ==
        reasons{"aggregate:count_star; other-side:filtered"});
  CHECK(why(per_customer("count(*), avg(o_weight)", "", "c_name")) ==
        reasons{"aggregate:count_star; aggregate:avg; keys:not-fewer-than-groups; "
                "keys:not-grouped"});
  // A full outer join pads the orders as well as the customers.
  CHECK(why("SELECT c_custkey, c_name, sum(o_totalprice) FROM orders FULL JOIN customer "
            "ON o_custkey = c_custkey GROUP BY c_custkey, c_name") == reasons{"join:full"});
  // An outer join that keeps the orders leaves the customers' key NULL for every order without a
  // customer. Grouped by it, those orders all fall into one group, so the groups are not the
  // orders' keys.
  CHECK(why("SELECT c_custkey, c_name, sum(o_totalprice) FROM orders LEFT JOIN customer "
            "ON o_custkey = c_custkey GROUP BY c_custkey, c_name") == reasons{"keys:not-grouped"});
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown passes outer joins that keep the aggregated side",
                 "[partial_agg_pushdown][plan]")
{
  // The orders are the side every row of which the join keeps, on either hand of it.
  auto const kept = [](std::string const& from) {
    return "SELECT o_custkey, c_name, sum(o_totalprice) FROM " + from +
           " ON o_custkey = c_custkey GROUP BY o_custkey, c_name";
  };
  CHECK(pushed(kept("orders LEFT JOIN customer")) == 1);
  CHECK(pushed(kept("customer RIGHT JOIN orders")) == 1);
  CHECK(shape(kept("orders LEFT JOIN customer"), true) == "AGG[2](JOIN(AGG[1](orders),customer))");
  // The join pads the orders instead, or both sides.
  CHECK(pushed(kept("customer LEFT JOIN orders")) == 0);
  CHECK(pushed(kept("orders FULL JOIN customer")) == 0);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown follows its thresholds",
                 "[partial_agg_pushdown][plan]")
{
  auto const filtered = per_customer("sum(o_totalprice)", "AND c_acctbal > 400");
  CHECK(pushed(filtered) == 0);
  CHECK(pushed(filtered, {.min_kept_share = 0.0}) == 1);
  CHECK(pushed(per_customer("sum(o_totalprice)"), {.min_kept_share = 1.0}) == 1);

  auto const two_extra =
    per_customer("sum(o_totalprice)", "", "c_custkey, c_name, o_shippriority, o_orderdate");
  CHECK(pushed(two_extra) == 0);
  CHECK(pushed(two_extra, {.max_extra_keys = 2}) == 1);
  // The largest value the setting can hold is no limit, not a sum that wraps around to none.
  CHECK(pushed(two_extra, {.max_extra_keys = std::numeric_limits<std::size_t>::max()}) == 1);
  CHECK(why(per_customer("sum(o_totalprice)", "", "c_custkey, c_name, o_shippriority"),
            {.max_extra_keys = 0}) == reasons{"keys:too-many-extra"});
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown leaves alone a key proven unique",
                 "[partial_agg_pushdown][plan]")
{
  // Every account has its own customer key, a PRIMARY KEY. The aggregate above groups by that
  // key, but one below the join would have a group for every account and combine nothing.
  auto const per_account =
    "SELECT c_custkey, c_name, sum(a_balance) FROM accounts, customer "
    "WHERE a_custkey = c_custkey GROUP BY c_custkey, c_name";
  CHECK(why(per_account) == reasons{"keys:unique"});
  // A filter on the accounts keeps the proof.
  CHECK(why("SELECT c_custkey, c_name, sum(a_balance) FROM accounts, customer "
            "WHERE a_custkey = c_custkey AND a_balance > 100 GROUP BY c_custkey, c_name") ==
        reasons{"keys:unique"});
  // The statistics cannot outvote it: they show one account per key as well.
  CHECK(pushed(per_account, {.min_rows_per_key = 0.5}) == 0);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown follows a grouped column proven unique to the key",
                 "[partial_agg_pushdown][plan]")
{
  // The shape of TPC-DS q11 and q74. The aggregate above groups by the members' id and not by
  // the key the orders join on. The id is the members' PRIMARY KEY, so one id is one member and
  // one join key: each group above still comes from one group below.
  std::string const joined = "orders, members WHERE o_custkey = m_custkey";
  auto const per_member    = [&](std::string const& groups,
                              std::string const& from      = "",
                              std::string const& aggregate = "sum(o_totalprice)") {
    return "SELECT " + groups + ", " + aggregate + " FROM " + (from.empty() ? joined : from) +
           " GROUP BY " + groups;
  };
  CHECK(pushed(per_member("m_id, m_name")) == 1);
  CHECK(shape(per_member("m_id, m_name"), true) == "AGG[2](JOIN(AGG[1](orders),members))");
  // A floating-point sum is pushed on the same grounds.
  CHECK(pushed(per_member("m_id, m_name", "", "sum(o_weight)")) == 1);
  // A filter on the members keeps the proof.
  CHECK(pushed(per_member("m_id, m_name", joined + " AND m_tier = 1"),
               {.min_kept_share = 0.0, .min_rows_per_key = 0}) == 1);
  // A second key the aggregate above groups by itself, as q11 does by the year.
  CHECK(pushed(per_member("m_id, m_name, o_region")) == 1);

  // A grouped column that is not proven unique fixes nothing, whatever its values.
  CHECK(why(per_member("m_name, m_tier")) == reasons{"keys:not-grouped"});
  // Nor does the id fix a key the aggregate above does not group by: the order key, which the
  // join with lineitem reads. Joined in another order that key would not be one, so the joins
  // stay as written.
  CHECK(why(per_member("m_id, m_name, m_tier",
                       "(orders JOIN members ON o_custkey = m_custkey) JOIN lineitem "
                       "ON o_orderkey = l_orderkey"),
            by_grouping,
            join_order::as_written) == reasons{"keys:not-grouped"});
  // An outer join pads the id with NULL where an order finds no member, so the orders without
  // one share a group above.
  CHECK(why(per_member("m_id, m_name", "orders LEFT JOIN members ON o_custkey = m_custkey")) ==
        reasons{"keys:not-grouped"});

  // The id fixes the members' key, which a join further up equates with the customers', which
  // the join below equates with the orders'.
  CHECK(shape(per_member("m_id, c_name",
                         "(orders JOIN customer ON o_custkey = c_custkey) JOIN members "
                         "ON c_custkey = m_custkey"),
              true) == "AGG[2](JOIN(JOIN(AGG[1](orders),customer),members))");
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown trusts statistics that show its keys repeat",
                 "[partial_agg_pushdown][plan]")
{
  // Grouped by the customers' strings alone, the aggregate above does not group by the join key.
  // The orders' statistics show that key repeats: 3000 orders over 300 customer keys, 10 each.
  // The count behind that is DuckDB's estimate, so the thresholds here stand well clear of 10.
  auto const by_strings = per_customer("sum(o_totalprice)", "", "c_name, c_address");
  CHECK(pushed(by_strings, {.min_rows_per_key = 2}) == 1);
  CHECK(why(by_strings, {.min_rows_per_key = 40}) == reasons{"keys:not-grouped"});
  CHECK(shape(by_strings, true, {.min_rows_per_key = 2}) ==
        "AGG[2](JOIN(AGG[1](orders),customer))");

  // The same under an outer join that pads the customers' key: the orders without a customer
  // share a group above, and their keys still repeat below.
  auto const padded =
    "SELECT c_custkey, c_name, sum(o_totalprice) FROM orders LEFT JOIN customer "
    "ON o_custkey = c_custkey GROUP BY c_custkey, c_name";
  CHECK(pushed(padded, {.min_rows_per_key = 2}) == 1);
  CHECK(pushed(padded, {.min_rows_per_key = 40}) == 0);

  // A key that an equality filter leaves a single value adds no groups. The visits of one region
  // are 6000 rows over at most 300 customer keys; counted with its five values, the region would
  // bring that down to a few rows per group.
  auto const one_region =
    "SELECT c_name, c_address, v_region, sum(v_amount) FROM visits, customer "
    "WHERE v_custkey = c_custkey AND v_region = 1 GROUP BY c_name, c_address, v_region";
  CHECK(shape(one_region, true, {.min_rows_per_key = 6}) ==
        "AGG[3](JOIN(AGG[2](visits),customer))");
  CHECK(shape("SELECT c_name, c_address, v_region, sum(v_amount) FROM visits, customer "
              "WHERE v_custkey = c_custkey AND v_region > 0 GROUP BY c_name, c_address, v_region",
              true,
              {.min_rows_per_key = 6}) == "AGG[3](JOIN(visits,customer))");

  // A key with one row each shows nothing: every order has its own key.
  CHECK(why("SELECT l_returnflag, sum(o_totalprice) FROM orders, lineitem "
            "WHERE o_orderkey = l_orderkey GROUP BY l_returnflag",
            {.min_rows_per_key = 2})
          .size() == 1);
  // Two keys together can form more groups than there are orders, which shows nothing either.
  CHECK(why(through_lineitem("c_custkey, c_name, l_returnflag"), {.min_rows_per_key = 2}) ==
        reasons{"keys:not-grouped"});

  // A floating-point sum is left to the grouping, however often its keys repeat: where keys share
  // a group above, each key's sum could overflow where the group's would not.
  auto const floats = per_customer("sum(o_weight)", "", "c_name, c_address");
  CHECK(why(floats, {.min_rows_per_key = 2}) == reasons{"keys:not-grouped"});
  CHECK(pushed(per_customer("min(o_weight), max(o_weight)", "", "c_name, c_address"),
               {.min_rows_per_key = 2}) == 1);
}
