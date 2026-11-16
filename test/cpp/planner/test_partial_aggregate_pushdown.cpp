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

#include <filesystem>
#include <memory>
#include <string>

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

/// Parse and optimize @p query, copy the plan as transparent execution does, apply the rewrite
/// when @p rewrite is set, copy the result again as plan validation does, then resolve it and hand
/// it to @p consume inside a transaction. Returns the number of aggregates pushed.
template <class Consume>
std::size_t with_resolved_plan(
  Connection& con, std::string const& query, join_order order, bool rewrite, Consume&& consume)
{
  std::size_t pushed = 0;
  auto& context      = *con.context;

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
      pushed = sirius::planner::push_partial_aggregates_below_joins(context, *planner.binder, plan);
      plan   = sirius::transparent::copy_logical_plan(*plan, context);
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
          "c_nationkey INTEGER, c_acctbal DECIMAL(15, 2), c_region INTEGER, c_clerk VARCHAR)",
          "INSERT INTO customer SELECT range, 'Customer#' || range, 'Street ' || range, "
          "range % 25, range * 1.5, range % 3, 'Clerk#' || (range % 10) FROM range(1, 301)",
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
          "CASE WHEN range % 2 = 0 THEN 'R' ELSE 'N' END FROM range(12000)"}) {
      auto result = con->Query(sql);
      INFO(sql);
      REQUIRE_FALSE(result->HasError());
    }
  }

  std::size_t pushed(std::string const& query, join_order order = join_order::optimized)
  {
    return with_resolved_plan(*con, query, order, true, [](ClientContext&, auto&) {});
  }

  /// The shape of @p query's plan with its joins as written, with or without the rewrite. The
  /// plan must also be one Sirius can run.
  std::string shape(std::string const& query, bool rewrite)
  {
    std::string result;
    with_resolved_plan(
      *con, query, join_order::as_written, rewrite, [&](ClientContext& context, auto& plan) {
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
  // The strings alone are grouped by: the join key still leaves fewer columns to group by.
  CHECK(pushed(per_customer("sum(o_totalprice)", "", "c_name, c_address")) == 1);
  // A group column of the aggregated side travels with the join key.
  CHECK(pushed(per_customer("sum(o_totalprice)", "", "c_custkey, c_name, o_shippriority")) == 1);
  // A two-column join key.
  CHECK(pushed("SELECT c_custkey, c_region, c_name, sum(o_totalprice) FROM orders, customer "
               "WHERE o_custkey = c_custkey AND o_region = c_region "
               "GROUP BY c_custkey, c_region, c_name") == 1);
  // Two aggregates share an expression, which DuckDB computes in a projection under them.
  CHECK(pushed(per_customer("sum(o_totalprice * (1 - o_totalprice)), "
                            "min(o_totalprice * (1 - o_totalprice))")) == 1);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown declines aggregates it cannot combine exactly",
                 "[partial_agg_pushdown][plan]")
{
  CHECK(pushed(per_customer("count(*)")) == 0);
  CHECK(pushed(per_customer("count(o_totalprice), sum(o_totalprice)")) == 0);
  CHECK(pushed(per_customer("avg(o_totalprice)")) == 0);
  CHECK(pushed(per_customer("sum(DISTINCT o_totalprice)")) == 0);
  CHECK(pushed(per_customer("sum(o_totalprice) FILTER (WHERE o_shippriority = 1)")) == 0);
  // Sums that are not a narrow DECIMAL's: floating point, integer and 128-bit decimal.
  CHECK(pushed(per_customer("sum(o_weight)")) == 0);
  CHECK(pushed(per_customer("sum(o_units)")) == 0);
  CHECK(pushed(per_customer("sum(o_wide)")) == 0);
  // An argument that reads both sides, and ones that can fail on a row the join would drop.
  CHECK(pushed(per_customer("sum(o_totalprice * c_acctbal)")) == 0);
  CHECK(pushed(per_customer("max(CAST(o_clerk AS INTEGER))")) == 0);
  CHECK(pushed(per_customer("min(o_units / o_shippriority)")) == 0);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "partial aggregate pushdown declines joins where it would not pay off",
                 "[partial_agg_pushdown][plan]")
{
  // No strings to carry.
  CHECK(pushed(per_customer("sum(o_totalprice)", "", "c_custkey, c_nationkey")) == 0);
  // The customers are filtered, so the join discards orders that would have been aggregated.
  CHECK(pushed(per_customer("sum(o_totalprice)", "AND c_acctbal > 400")) == 0);
  // It would group by as many columns as the aggregate above it, or by a string.
  CHECK(pushed(per_customer("sum(o_totalprice)", "", "c_name")) == 0);
  CHECK(pushed("SELECT c_custkey, c_address, sum(o_totalprice) FROM orders, customer "
               "WHERE o_clerk = c_clerk GROUP BY c_custkey, c_address") == 0);
  // Too many columns of the aggregated side are grouped by.
  CHECK(pushed(per_customer(
          "sum(o_totalprice)", "", "c_custkey, c_name, o_shippriority, o_orderdate")) == 0);
  // The aggregated side is already grouped.
  CHECK(pushed("SELECT c_custkey, c_name, min(spent) FROM customer, "
               "(SELECT o_custkey, min(o_totalprice) AS spent FROM orders GROUP BY o_custkey) "
               "WHERE o_custkey = c_custkey GROUP BY c_custkey, c_name") == 0);
  // An outer join pads the aggregated side, and an ungrouped aggregate has nothing to carry.
  CHECK(pushed("SELECT c_custkey, c_name, sum(o_totalprice) FROM customer LEFT JOIN orders "
               "ON o_custkey = c_custkey GROUP BY c_custkey, c_name") == 0);
  CHECK(pushed("SELECT sum(o_totalprice) FROM orders, customer WHERE o_custkey = c_custkey") == 0);
}
