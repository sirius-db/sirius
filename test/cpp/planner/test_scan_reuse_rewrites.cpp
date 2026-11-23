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

// Which plans `rewrite_repeated_scans` rewrites, on DuckDB's optimized plans as Sirius receives
// them. Answers are checked on the GPU against DuckDB in
// test/cpp/integration/test_gpu_execution_scan_reuse.cpp.

#include "planner/mark_join_conversion.hpp"
#include "planner/scan_reuse_rewrites.hpp"
#include "planner/unsupported_optimizers.hpp"
#include "transparent/sirius_optimizer_extension.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/tpch_queries.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstddef>
#include <filesystem>
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

/// What a test does to the plan before `rewrite_repeated_scans`.
struct plan_options {
  join_order order   = join_order::optimized;
  bool convert_marks = true;  ///< Run `convert_mark_joins` first, as `create_plan` does
};

/// A plan before and after the rewrite.
struct rewrite_outcome {
  std::string before;  ///< The plan just before the call, as text
  std::string after;   ///< The plan after the call, as text
  std::vector<unique_ptr<LogicalOperator>> removed;
  unique_ptr<LogicalOperator> plan;                    ///< The rewritten plan, copied and resolved
  std::vector<std::string> subquery_functions_before;  ///< Of the second aggregate in pre-order
  std::vector<LogicalType> outer_types_before;

  [[nodiscard]] bool unchanged() const { return removed.empty() && before == after; }
};

/// The name of the table @p get reads: its catalog table, or the Parquet file's stem.
std::string table_of(LogicalGet const& get)
{
  if (auto const table = get.GetTable()) { return table->name.GetIdentifierName(); }
  if (get.parameters.empty()) { return "?"; }
  return std::filesystem::path(get.parameters[0].ToString()).stem().string();
}

std::size_t scans_of(LogicalOperator& op, std::string const& table)
{
  std::size_t n =
    op.type == LogicalOperatorType::LOGICAL_GET && table_of(op.Cast<LogicalGet>()) == table ? 1 : 0;
  for (auto& child : op.children) {
    n += scans_of(*child, table);
  }
  return n;
}

std::size_t joins_of(LogicalOperator& op, JoinType type)
{
  std::size_t n = op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
                      op.Cast<LogicalComparisonJoin>().join_type == type
                    ? 1
                    : 0;
  for (auto& child : op.children) {
    n += joins_of(*child, type);
  }
  return n;
}

/// Every aggregate, in pre-order: the outer query's first.
void collect_aggregates(LogicalOperator& op, std::vector<LogicalAggregate*>& out)
{
  if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    out.push_back(&op.Cast<LogicalAggregate>());
  }
  for (auto& child : op.children) {
    collect_aggregates(*child, out);
  }
}

/// The first aggregate in pre-order: the outer query's.
LogicalAggregate* outer_aggregate(LogicalOperator& op)
{
  if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    return &op.Cast<LogicalAggregate>();
  }
  for (auto& child : op.children) {
    if (auto* found = outer_aggregate(*child)) { return found; }
  }
  return nullptr;
}

std::vector<std::string> function_names(LogicalAggregate const& aggregate)
{
  std::vector<std::string> names;
  for (auto const& expression : aggregate.expressions) {
    names.push_back(
      expression->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE
        ? std::string(expression->Cast<BoundAggregateExpression>().Function().GetName())
        : "?");
  }
  return names;
}

/// TPC-H Q18 with its tables, aggregates and predicates as parameters. The defaults give Q18 with
/// its HAVING threshold scaled to the fixture.
struct q18_query {
  std::string outer_aggregates  = "sum(l_quantity)";
  std::string orders            = "orders";
  std::string outer_lineitem    = "lineitem";
  std::string subquery_lineitem = "lineitem";
  std::string subquery_key      = "l_orderkey";
  std::string subquery_where    = "";
  std::string subquery_groups   = "l_orderkey";
  std::string having            = "having sum(l_quantity) > 150";
  std::string membership        = "o_orderkey in";
  std::string outer_where       = "";
  std::string join              = "o_orderkey = l_orderkey";
  std::string groups            = "c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice";

  [[nodiscard]] std::string sql() const
  {
    return "select " + groups + ", " + outer_aggregates + " from customer, " + orders + ", " +
           outer_lineitem + " where " + membership + " (select " + subquery_key + " from " +
           subquery_lineitem + " " + subquery_where + " group by " + subquery_groups + " " +
           having + ") and c_custkey = o_custkey and " + join + outer_where + " group by " +
           groups + " order by o_totalprice desc, o_orderdate limit 100";
  }
};

struct scan_reuse_fixture {
  scan_reuse_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    db  = sirius::test::open_sirius_db(dir.file("scan_reuse.db").c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    auto const lineitem =
      "SELECT (i % 500)::BIGINT AS l_orderkey, (i % 37)::BIGINT AS l_partkey, (i % 11)::BIGINT AS "
      "l_suppkey, (i % 7)::INTEGER AS l_linenumber, ((i % 50) + 1)::DECIMAL(15, 2) AS l_quantity, "
      "(i * 1.5)::DECIMAL(15, 2) AS l_extendedprice, ((i % 9) / 100)::DECIMAL(15, 2) AS l_tax, "
      "DATE '1994-01-01' + (i % 700)::INTEGER AS l_shipdate FROM range(4000) t(i)";
    parquet_view("lineitem", lineitem);
    parquet_view("lineitem2", lineitem);
    // The same file with the key and another BIGINT column, or the two DECIMAL columns, swapped by
    // name: the same source, but not the same columns.
    run(
      "CREATE VIEW lineitem_keyswap AS SELECT l_partkey AS l_orderkey, l_orderkey AS l_partkey, "
      "l_quantity, l_extendedprice FROM read_parquet(" +
      dir.file_literal("lineitem.parquet") + ")");
    run(
      "CREATE VIEW lineitem_sumswap AS SELECT l_orderkey, l_extendedprice AS l_quantity, "
      "l_quantity AS l_extendedprice FROM read_parquet(" +
      dir.file_literal("lineitem.parquet") + ")");
    parquet_view("lineitem_int",
                 "SELECT (i % 500)::BIGINT AS l_orderkey, ((i % 50) + 1)::INTEGER AS l_quantity "
                 "FROM range(4000) t(i)");
    parquet_view("lineitem_null",
                 "SELECT CASE WHEN i % 13 = 5 THEN NULL ELSE (i % 500)::BIGINT END AS l_orderkey, "
                 "((i % 50) + 1)::DECIMAL(15, 2) AS l_quantity FROM range(4000) t(i)");
    parquet_view("lineitem_empty",
                 "SELECT l_orderkey, l_quantity FROM (" + std::string(lineitem) + ") WHERE false");
    parquet_view("orders",
                 "SELECT i::BIGINT AS o_orderkey, (i % 50)::BIGINT AS o_custkey, DATE '1995-01-01' "
                 "+ (i % 300)::INTEGER AS o_orderdate, (i * 3.25)::DECIMAL(15, 2) AS o_totalprice "
                 "FROM range(500) t(i)");
    parquet_view("orders_null",
                 "SELECT CASE WHEN i % 7 = 3 THEN NULL ELSE i::BIGINT END AS o_orderkey, (i % "
                 "50)::BIGINT AS o_custkey, DATE '1995-01-01' + (i % 300)::INTEGER AS o_orderdate, "
                 "(i * 3.25)::DECIMAL(15, 2) AS o_totalprice FROM range(500) t(i)");
    parquet_view("customer",
                 "SELECT i::BIGINT AS c_custkey, 'Customer#' || i AS c_name FROM range(50) t(i)");
    parquet_view("nation",
                 "SELECT i::BIGINT AS n_nationkey, 'N' || i AS n_name FROM range(5) t(i)");
    // A native table has statistics, which make DuckDB pick `sum_no_overflow` for its SUMs.
    run("CREATE TABLE lineitem_native AS " + std::string(lineitem));
  }

  void run(std::string const& sql)
  {
    auto result = con->Query(sql);
    INFO(sql);
    REQUIRE_FALSE(result->HasError());
  }

  void parquet_view(std::string const& name, std::string const& select)
  {
    run("COPY (" + select + ") TO " + dir.file_literal(name + ".parquet") + " (FORMAT parquet)");
    run("CREATE VIEW " + name + " AS FROM read_parquet(" + dir.file_literal(name + ".parquet") +
        ")");
  }

  /// Parse and optimize @p query, copy the plan as transparent execution does, convert its MARK
  /// joins, rewrite it, copy the result again as plan validation does, and resolve it, all inside a
  /// transaction.
  rewrite_outcome rewrite(std::string const& query, plan_options options = {})
  {
    INFO(query);
    rewrite_outcome outcome;
    auto& context = *con->context;

    auto const original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
    auto& disabled               = DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled.insert(OptimizerType::IN_CLAUSE);
    disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(sirius::planner::unsupported_optimizers.begin(),
                    sirius::planner::unsupported_optimizers.end());
    if (options.order == join_order::as_written) {
      disabled.insert(OptimizerType::JOIN_ORDER);
      disabled.insert(OptimizerType::BUILD_SIDE_PROBE_SIDE);
    }

    con->Query("BEGIN TRANSACTION");
    try {
      Parser parser(context);
      parser.ParseQuery(query);
      REQUIRE(!parser.statements.empty());
      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      REQUIRE(planner.plan);
      Optimizer optimizer(*planner.binder, context);
      auto optimized = optimizer.Optimize(std::move(planner.plan));
      auto plan      = sirius::transparent::copy_logical_plan(*optimized, context);
      if (options.convert_marks) { sirius::planner::convert_mark_joins(context, plan); }

      outcome.before = plan->ToString();
      std::vector<LogicalAggregate*> aggregates;
      collect_aggregates(*plan, aggregates);
      if (!aggregates.empty()) {
        for (auto const& expression : aggregates[0]->expressions) {
          outcome.outer_types_before.push_back(expression->GetReturnType());
        }
      }
      if (aggregates.size() > 1) {
        outcome.subquery_functions_before = function_names(*aggregates[1]);
      }
      outcome.removed = sirius::planner::rewrite_repeated_scans(context, plan);
      outcome.after   = plan->ToString();

      // The rewritten plan serializes, and every binding it reads exists below it with its type.
      plan = sirius::transparent::copy_logical_plan(*plan, context);
      plan->ResolveOperatorTypes();
      ColumnBindingResolver verifier(true);
      verifier.VisitOperator(*plan);
      outcome.plan = std::move(plan);
    } catch (...) {
      con->Query("ROLLBACK");
      DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
      throw;
    }
    con->Query("COMMIT");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    return outcome;
  }

  /// Requires that the rewrite removed one scan of @p table and left one, with no semi join.
  void require_reused(rewrite_outcome const& outcome, std::string const& table = "lineitem")
  {
    INFO(outcome.before);
    REQUIRE(outcome.removed.size() == 1);
    REQUIRE(outcome.removed[0]->type == LogicalOperatorType::LOGICAL_GET);
    CHECK(table_of(outcome.removed[0]->Cast<LogicalGet>()) == table);
    CHECK(scans_of(*outcome.plan, table) == 1);
    CHECK(joins_of(*outcome.plan, JoinType::SEMI) == 0);
    CHECK(joins_of(*outcome.plan, JoinType::RIGHT_SEMI) == 0);
    auto* outer = outer_aggregate(*outcome.plan);
    REQUIRE(outer);
    for (auto const& name : function_names(*outer)) {
      CHECK(name == "sum");
    }
    for (std::size_t i = 0; i < outer->expressions.size(); ++i) {
      CHECK(outer->expressions[i]->GetReturnType() == outcome.outer_types_before[i]);
    }
  }

  void require_unchanged(rewrite_outcome const& outcome)
  {
    INFO(outcome.before);
    INFO(outcome.after);
    CHECK(outcome.removed.empty());
    CHECK(outcome.before == outcome.after);
  }

  sirius::test::scratch_dir dir{"scan_reuse_rewrites"};
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(scan_reuse_fixture,
                 "scan reuse: TPC-H Q18 sums the IN-subquery's sums and scans lineitem once",
                 "[planner][scan_reuse]")
{
  auto const outcome = rewrite(sirius::test::kTpchQ18);
  require_reused(outcome);
  // The outer SUM reads the subquery's sum, carried up through the converted join.
  auto* outer = outer_aggregate(*outcome.plan);
  REQUIRE(outer);
  CHECK(outer->children[0]->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
}

TEST_CASE_METHOD(
  scan_reuse_fixture,
  "scan reuse: a native lineitem's sum_no_overflow is reused by a SUM of the same type",
  "[planner][scan_reuse]")
{
  // Statistics make DuckDB sum the subquery with `sum_no_overflow`; the outer SUM, over a join with
  // no maximum cardinality, keeps `sum`.
  auto const outcome = rewrite(
    q18_query{.outer_lineitem = "lineitem_native", .subquery_lineitem = "lineitem_native"}.sql());
  REQUIRE(outcome.subquery_functions_before == std::vector<std::string>{"sum_no_overflow"});
  require_reused(outcome, "lineitem_native");
  REQUIRE(outcome.outer_types_before.size() == 1);
  CHECK(outcome.outer_types_before[0] == LogicalType::DECIMAL(38, 2));
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "scan reuse: Q18 shapes the rule rewrites",
                 "[planner][scan_reuse]")
{
  SECTION("two reused sums")
  {
    require_reused(rewrite(q18_query{.outer_aggregates = "sum(l_quantity), sum(l_extendedprice)",
                                     .having           = "having sum(l_quantity) > 150 and "
                                                         "sum(l_extendedprice) > 0"}
                             .sql()));
  }
  SECTION("one subquery sum reused by two outer sums")
  {
    require_reused(
      rewrite(q18_query{.outer_aggregates = "sum(l_quantity), sum(l_quantity) as again"}.sql()));
  }
  SECTION("the same filter on both scans")
  {
    require_reused(rewrite(q18_query{.subquery_where = "where l_shipdate > DATE '1995-01-01'",
                                     .outer_where    = " and l_shipdate > DATE '1995-01-01'"}
                             .sql()));
  }
  SECTION("NULL keys on both sides")
  {
    require_reused(rewrite(q18_query{.orders            = "orders_null",
                                     .outer_lineitem    = "lineitem_null",
                                     .subquery_lineitem = "lineitem_null"}
                             .sql()),
                   "lineitem_null");
  }
  SECTION("a HAVING that keeps nothing")
  {
    require_reused(rewrite(q18_query{.having = "having sum(l_quantity) > 1000000"}.sql()));
  }
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "scan reuse: lineitem on either side of the outer join",
                 "[planner][scan_reuse]")
{
  // Written order, with the IN subquery kept with orders: lineitem on the right of the outer join,
  // then on the left.
  auto const y =
    std::string("(select * from customer, orders where c_custkey = o_custkey and o_orderkey in ") +
    "(select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 150)) y";
  auto const query = [](std::string const& from) {
    // Grouped by the customer only: DuckDB may group by either side's copy of the join key.
    return "select c_name, sum(l_quantity) from " + from +
           " where o_orderkey = l_orderkey group by c_name";
  };
  require_reused(rewrite(query(y + ", lineitem"), {.order = join_order::as_written}));
  require_reused(rewrite(query("lineitem, " + y), {.order = join_order::as_written}));
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "scan reuse: a RIGHT_SEMI join becomes an INNER join with its children in place",
                 "[planner][scan_reuse]")
{
  // So few orders that DuckDB builds the semi join on them.
  auto const outcome = rewrite(q18_query{
    .outer_where = " and o_totalprice > 1600 and o_custkey < 3 and o_orderdate > DATE '1995-06-01'"}
                                 .sql());
  INFO(outcome.before);
  REQUIRE(outcome.before.find("RIGHT_SEMI") != std::string::npos);
  require_reused(outcome);
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "scan reuse: Q18 shapes the rule leaves alone",
                 "[planner][scan_reuse]")
{
  SECTION("a subquery without a SUM (no HAVING)")
  {
    require_unchanged(rewrite(q18_query{.having = ""}.sql()));
  }
  SECTION("subquery groups that are not unique per key")
  {
    require_unchanged(rewrite(q18_query{.subquery_groups = "l_orderkey, l_linenumber"}.sql()));
  }
  SECTION("outer aggregates other than a plain SUM")
  {
    for (auto const* aggregates : {"avg(l_quantity)",
                                   "count(*)",
                                   "min(l_quantity)",
                                   "sum(DISTINCT l_quantity)",
                                   "sum(l_quantity) FILTER (WHERE l_tax > 0)",
                                   "sum(l_quantity), sum(l_quantity * 2)"}) {
      INFO(aggregates);
      require_unchanged(rewrite(q18_query{.outer_aggregates = aggregates}.sql()));
    }
  }
  SECTION("an outer group on a lineitem column")
  {
    require_unchanged(rewrite(
      q18_query{.groups = "c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, l_linenumber"}
        .sql()));
  }
  SECTION("a filter on one scan only")
  {
    // An empty lineitem: statistics turn the join key into a filter on the outer scan only.
    require_unchanged(rewrite(
      q18_query{.outer_lineitem = "lineitem_empty", .subquery_lineitem = "lineitem_empty"}.sql()));
    require_unchanged(
      rewrite(q18_query{.outer_where = " and l_shipdate > DATE '1995-01-01'"}.sql()));
    require_unchanged(
      rewrite(q18_query{.subquery_where = "where l_shipdate > DATE '1995-01-01'"}.sql()));
  }
  SECTION("another file")
  {
    require_unchanged(rewrite(q18_query{.subquery_lineitem = "lineitem2"}.sql()));
  }
  SECTION("an outer SUM of a column the subquery does not sum")
  {
    require_unchanged(rewrite(q18_query{.having = "having sum(l_extendedprice) > 1000"}.sql()));
  }
  SECTION("an outer join on two conditions, or not on equality")
  {
    require_unchanged(
      rewrite(q18_query{.join = "o_orderkey = l_orderkey and o_custkey = l_suppkey"}.sql()));
    require_unchanged(rewrite(q18_query{.join = "o_orderkey < l_orderkey"}.sql()));
  }
  SECTION("NOT IN, and IN under OR")
  {
    require_unchanged(rewrite(q18_query{.membership = "o_orderkey not in"}.sql()));
    require_unchanged(
      rewrite(q18_query{.membership = "o_orderdate = DATE '1990-01-01' or o_orderkey in"}.sql()));
  }
  SECTION("an integer argument")
  {
    require_unchanged(rewrite(
      q18_query{.outer_lineitem = "lineitem_int", .subquery_lineitem = "lineitem_int"}.sql()));
  }
  SECTION("a subquery grouped by a column other than the outer join's")
  {
    require_unchanged(
      rewrite(q18_query{.subquery_key = "l_partkey", .subquery_groups = "l_partkey"}.sql()));
  }
  SECTION("a LEFT join between the outer join and the semi join")
  {
    require_unchanged(rewrite(
      "select c_name, o_orderkey, sum(l_quantity) from (orders left join customer on c_custkey = "
      "o_custkey), lineitem where o_orderkey in (select l_orderkey from lineitem group by "
      "l_orderkey having sum(l_quantity) > 150) and o_orderkey = l_orderkey group by c_name, "
      "o_orderkey",
      {.order = join_order::as_written}));
  }
  SECTION("a subquery SUM with DISTINCT or FILTER")
  {
    require_unchanged(rewrite(q18_query{.having = "having sum(DISTINCT l_quantity) > 150"}.sql()));
    require_unchanged(rewrite(
      q18_query{.having = "having sum(l_quantity) FILTER (WHERE l_linenumber = 1) > 10"}.sql()));
  }
  SECTION("IN on an orders column other than the join's")
  {
    require_unchanged(rewrite(q18_query{.membership = "o_custkey in"}.sql()));
  }
  SECTION("a view that renames the key or the summed column over the same file")
  {
    require_unchanged(rewrite(q18_query{.outer_lineitem = "lineitem_keyswap"}.sql()));
    require_unchanged(rewrite(q18_query{.outer_lineitem = "lineitem_sumswap"}.sql()));
  }
  SECTION("a sampled scan")
  {
    require_unchanged(rewrite(q18_query{.outer_lineitem = "lineitem tablesample 50%"}.sql()));
    require_unchanged(rewrite(q18_query{.subquery_lineitem = "lineitem tablesample 50%"}.sql()));
  }
}
