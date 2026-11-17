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

#include "planner/delim_cte_rewrites.hpp"
#include "sirius_context.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/scoped_sirius_setting.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>

using namespace duckdb;

namespace {

struct plan_stats {
  std::size_t exists_answered = 0;  ///< What `answer_exists_from_aggregate` returned
  std::size_t keys_carried    = 0;  ///< What `carry_subquery_keys_across_equalities` returned
  std::size_t mark_joins      = 0;
  std::size_t outer_joins     = 0;  ///< LEFT and RIGHT
  std::size_t semi_joins      = 0;  ///< SEMI and RIGHT_SEMI
  std::size_t sales_scans     = 0;
  std::string text;
};

void count_operators(LogicalOperator const& op, plan_stats& stats)
{
  if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    switch (op.Cast<LogicalComparisonJoin>().join_type) {
      case JoinType::MARK: ++stats.mark_joins; break;
      case JoinType::LEFT:
      case JoinType::RIGHT: ++stats.outer_joins; break;
      case JoinType::SEMI:
      case JoinType::RIGHT_SEMI: ++stats.semi_joins; break;
      default: break;
    }
  }
  if (op.type == LogicalOperatorType::LOGICAL_GET) {
    auto const table = op.Cast<LogicalGet>().GetTable();
    if (table && table->name == "sales") { ++stats.sales_scans; }
  }
  for (auto const& child : op.children) {
    count_operators(*child, stats);
  }
}

struct rules {
  bool exists = false;
  bool carry  = false;
  int repeat  = 1;  ///< Times each chosen rule is applied
};

/// The plan as the binder leaves it, with the chosen rules applied to it directly: what the
/// pre-optimize hook does, without DuckDB's optimizer in the way of the plan's shape.
plan_stats bound(Connection& con, std::string const& sql, rules const& apply)
{
  Parser parser(*con.context);
  parser.ParseQuery(sql);
  REQUIRE(parser.statements.size() == 1);
  plan_stats stats;
  con.context->RunFunctionInTransaction([&] {
    Planner planner(*con.context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    for (int i = 0; i < apply.repeat; ++i) {
      if (apply.exists) {
        stats.exists_answered +=
          sirius::planner::answer_exists_from_aggregate(*con.context, *planner.plan);
      }
      if (apply.carry) {
        stats.keys_carried += sirius::planner::carry_subquery_keys_across_equalities(
          *con.context, *planner.binder, *planner.plan);
      }
    }
    count_operators(*planner.plan, stats);
    stats.text = planner.plan->ToString();
    // DuckDB checks this after every optimizer pass in debug builds only: every column an
    // operator reads is one its children output.
    planner.plan->ResolveOperatorTypes();
    ColumnBindingResolver verifier(/*verify_only=*/true);
    REQUIRE_NOTHROW(verifier.VisitOperator(*planner.plan));
  });
  return stats;
}

/// The plan after DuckDB's optimizer, Sirius's hooks included.
plan_stats optimized(Connection& con, std::string const& sql)
{
  Parser parser(*con.context);
  parser.ParseQuery(sql);
  REQUIRE(parser.statements.size() == 1);
  plan_stats stats;
  con.context->RunFunctionInTransaction([&] {
    Planner planner(*con.context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    Optimizer optimizer(*planner.binder, *con.context);
    auto plan = optimizer.Optimize(std::move(planner.plan));
    if (auto state = duckdb::get_sirius_connection_state(*con.context)) {
      state->clear_captured_plan();
    }
    count_operators(*plan, stats);
    stats.text = plan->ToString();
  });
  return stats;
}

/// A scalar aggregate subquery and an EXISTS over the same three joined tables, correlated on the
/// supplier's nation, which an equality ties to the customer's (TPC-Havoc Q05_V01's shape).
struct delim_rewrites_fixture {
  delim_rewrites_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    db  = sirius::test::open_sirius_db(db_dir.file("delim_rewrites.db").c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    for (char const* sql :
         {"SET delim_join_as_cte = true",
          "CREATE TABLE nations (nk INTEGER, region VARCHAR)",
          "INSERT INTO nations SELECT range, 'R' || (range % 5) FROM range(25)",
          // More rows than a quarter of `sales`.
          "CREATE TABLE many_nations (nk INTEGER, region VARCHAR)",
          "INSERT INTO many_nations SELECT range % 25, 'R' || (range % 5) FROM range(4000)",
          "CREATE TABLE suppliers (sk INTEGER, s_nk INTEGER)",
          "INSERT INTO suppliers SELECT range, range % 25 FROM range(200)",
          "CREATE TABLE customers (ck INTEGER, c_nk INTEGER, c_wide_nk BIGINT)",
          "INSERT INTO customers SELECT range, range % 25, range % 25 FROM range(1000)",
          "CREATE TABLE sales (s_sk INTEGER, s_ck INTEGER, amount INTEGER)",
          "INSERT INTO sales SELECT range % 200, range % 1000, range % 17 FROM range(5000)"}) {
      auto result = con->Query(sql);
      INFO(sql);
      REQUIRE_FALSE(result->HasError());
    }
  }

  sirius::test::scratch_dir db_dir{"delim_rewrites"};
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

std::string rows(std::string const& tables      = "sales sa, customers c, suppliers s",
                 std::string const& tie         = "c.c_nk = s.s_nk",
                 std::string const& correlation = "s.s_nk = n.nk",
                 std::string const& more        = "")
{
  return " FROM " + tables + " WHERE sa.s_ck = c.ck AND sa.s_sk = s.sk AND " + tie + " AND " +
         correlation + more;
}

std::string const selective_outer = "nations n WHERE n.region = 'R1'";

std::string scalar_query(std::string const& subquery = "(SELECT sum(sa.amount)" + rows() + ")",
                         std::string const& outer    = selective_outer)
{
  return "SELECT n.nk, " + subquery + " AS revenue FROM " + outer;
}

std::string twin_query(std::string const& subquery  = "(SELECT sum(sa.amount)" + rows() + ")",
                       std::string const& predicate = "EXISTS (SELECT 1" + rows() + ")",
                       std::string const& outer     = selective_outer)
{
  return scalar_query(subquery, outer) + " AND " + predicate;
}

}  // namespace

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - an EXISTS over an aggregate subquery's rows is answered from it",
                 "[delim_cte_rewrites][plan]")
{
  auto const before = bound(*con, twin_query(), {});
  INFO(before.text);
  REQUIRE(before.mark_joins == 1);
  REQUIRE(before.outer_joins == 1);
  REQUIRE(before.sales_scans == 2);

  auto const after = bound(*con, twin_query(), {.exists = true});
  INFO(after.text);
  CHECK(after.exists_answered == 1);
  // The EXISTS join and its side are gone; the join-back keeps only the outer rows with a group.
  CHECK(after.mark_joins == 0);
  CHECK(after.outer_joins == 0);
  CHECK(after.sales_scans == 1);
  CHECK(bound(*con, twin_query(), {.exists = true, .repeat = 2}).exists_answered == 1);

  // An outer side of two joined tables qualifies while the rows their join can produce, 625 here,
  // stay small next to the subquery (TPC-Havoc Q05_V01's nation and region).
  auto const joined_outer =
    twin_query("(SELECT sum(sa.amount)" + rows() + ")",
               "EXISTS (SELECT 1" + rows() + ")",
               "nations n, nations m WHERE n.nk = m.nk AND m.region = 'R1'");
  auto const joined = bound(*con, joined_outer, {.exists = true});
  INFO(joined.text);
  CHECK(joined.exists_answered == 1);
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - an EXISTS that may differ from the subquery's groups is kept",
                 "[delim_cte_rewrites][plan]")
{
  std::string const subquery = "(SELECT sum(sa.amount)" + rows() + ")";
  std::string const exists   = "EXISTS (SELECT 1" + rows() + ")";
  auto const [label, sql]    = GENERATE_COPY(table<std::string, std::string>({
    // A NULL key matches a NULL key in the EXISTS, and no key in the subquery's `=`.
    {"null-safe EXISTS",
        twin_query(subquery,
                "EXISTS (SELECT 1" +
                  rows("sales sa, customers c, suppliers s",
                       "c.c_nk = s.s_nk",
                       "s.s_nk IS NOT DISTINCT FROM n.nk") +
                  ")")},
    // A group the HAVING rejects still has rows for the EXISTS.
    {"HAVING",
        twin_query("(SELECT sum(sa.amount)" + rows() + " HAVING sum(sa.amount) > 10)", exists)},
    // COUNT has a value, 0, for a key without rows.
    {"COUNT", twin_query("(SELECT count(*)" + rows() + ")", exists)},
    {"a filter only the EXISTS has",
        twin_query(subquery,
                "EXISTS (SELECT 1" +
                  rows("sales sa, customers c, suppliers s",
                       "c.c_nk = s.s_nk",
                       "s.s_nk = n.nk",
                       " AND sa.amount > 5") +
                  ")")},
    {"a filter only the subquery has",
        twin_query("(SELECT sum(sa.amount)" +
                  rows("sales sa, customers c, suppliers s",
                       "c.c_nk = s.s_nk",
                       "s.s_nk = n.nk",
                       " AND sa.amount > 5") +
                  ")",
                exists)},
    {"NOT EXISTS", twin_query(subquery, "NOT " + exists)},
    {"EXISTS under OR", twin_query(subquery, "(" + exists + " OR n.nk = 3)")},
    {"another key column",
        twin_query(subquery,
                "EXISTS (SELECT 1" +
                  rows("sales sa, customers c, suppliers s", "c.c_nk = s.s_nk", "c.ck = n.nk") +
                  ")")},
    // Not provably the same rows: the two sides are compared table by table, in order.
    {"another table order",
        twin_query(subquery, "EXISTS (SELECT 1" + rows("customers c, sales sa, suppliers s") + ")")},
    // Without the EXISTS the delim CTE would hold every outer row.
    {"an outer side with no other selection", twin_query(subquery, exists, "nations n WHERE true")},
    {"a large outer side", twin_query(subquery, exists, "many_nations n WHERE n.region = 'R1'")},
    // Two small scans, 225 rows in total, whose join can produce up to 5000.
    {"an outer side whose join can multiply its rows",
        twin_query(subquery, exists, "nations n, suppliers x WHERE n.nk < x.sk")},
  }));
  CAPTURE(label, sql);
  auto const before = bound(*con, sql, {});
  auto const after  = bound(*con, sql, {.exists = true});
  INFO(after.text);
  CHECK(after.exists_answered == 0);
  CHECK(after.mark_joins == before.mark_joins);
  CHECK(after.outer_joins == before.outer_joins);
  CHECK(after.sales_scans == before.sales_scans);
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - an outer side without a row estimate keeps its EXISTS",
                 "[delim_cte_rewrites][plan]")
{
  std::string const subquery = "(SELECT sum(sa.amount)" + rows() + ")";
  std::string const exists   = "EXISTS (SELECT 1" + rows() + ")";

  // Control: the outer rows behind a projection, read from a table, whose row count is known.
  auto const counted =
    bound(*con,
          twin_query(subquery, exists, "(SELECT nk, region FROM nations) n WHERE n.region = 'R1'"),
          {.exists = true});
  INFO(counted.text);
  CHECK(counted.exists_answered == 1);

  // The same shape over a table function that estimates nothing. DuckDB reports such a scan as
  // one row, which would pass for the smallest of tables.
  auto const uncounted = bound(*con,
                               twin_query(subquery,
                                          exists,
                                          "(SELECT \"int\" AS nk, \"varchar\" AS region "
                                          "FROM test_all_types()) n WHERE n.region = 'R1'"),
                               {.exists = true});
  INFO(uncounted.text);
  CHECK(uncounted.exists_answered == 0);
  CHECK(uncounted.mark_joins == 1);
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - a subquery's keys reach the table an equality ties to them",
                 "[delim_cte_rewrites][plan]")
{
  auto const before = bound(*con, scalar_query(), {});
  INFO(before.text);
  REQUIRE(before.semi_joins == 1);

  auto const after = bound(*con, scalar_query(), {.carry = true});
  INFO(after.text);
  CHECK(after.keys_carried == 1);
  CHECK(after.semi_joins == 2);
  // A second pass finds both tables restricted.
  auto const twice = bound(*con, scalar_query(), {.carry = true, .repeat = 2});
  CHECK(twice.keys_carried == 1);
  CHECK(twice.semi_joins == 2);
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - keys are not carried without a plain equality on the key column",
                 "[delim_cte_rewrites][plan]")
{
  auto const [label, tie] = GENERATE(table<std::string, std::string>({
    {"an equality on other columns", "c.ck = s.sk"},
    {"an equality through a cast", "c.c_wide_nk = s.s_nk"},
    {"an equality under OR", "(c.c_nk = s.s_nk OR c.ck = s.sk)"},
    {"an inequality", "c.c_nk <= s.s_nk"},
  }));
  CAPTURE(label);
  auto const sql =
    scalar_query("(SELECT sum(sa.amount)" + rows("sales sa, customers c, suppliers s", tie) + ")");
  auto const before = bound(*con, sql, {});
  auto const after  = bound(*con, sql, {.carry = true});
  INFO(after.text);
  CHECK(after.keys_carried == 0);
  CHECK(after.semi_joins == before.semi_joins);
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - the optimizer hook applies both rules, each under its switch",
                 "[delim_cte_rewrites][plan]")
{
  bool const answer_exists = GENERATE(true, false);
  bool const carry_keys    = GENERATE(true, false);
  CAPTURE(answer_exists, carry_keys);
  sirius::test::scoped_sirius_setting exists_switch(
    *con, "enable_exists_from_aggregate", answer_exists);
  sirius::test::scoped_sirius_setting carry_switch(*con, "enable_subquery_key_carry", carry_keys);

  auto const plan = optimized(*con, twin_query());
  INFO(plan.text);
  // The subquery's tables are joined once when the EXISTS is answered from the aggregate.
  CHECK(plan.sales_scans == (answer_exists ? 1 : 2));
  CHECK(plan.outer_joins == (answer_exists ? 0 : 1));
  // The keys restrict `suppliers`, and with the carry `customers` as well; the EXISTS that is
  // kept is one more join of the semi family or a MARK join.
  auto const key_joins = carry_keys ? std::size_t{2} : std::size_t{1};
  CHECK(plan.semi_joins + plan.mark_joins == key_joins + (answer_exists ? 0 : 1));
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - both rules are on by default",
                 "[delim_cte_rewrites][plan]")
{
  auto const plan = optimized(*con, twin_query());
  INFO(plan.text);
  // The subquery's tables are joined once, the keys restrict `suppliers` and `customers`, and the
  // EXISTS is gone.
  CHECK(plan.sales_scans == 1);
  CHECK(plan.outer_joins == 0);
  CHECK(plan.semi_joins + plan.mark_joins == 2);
}

TEST_CASE_METHOD(delim_rewrites_fixture,
                 "delim rewrites - a query that stays on the CPU keeps DuckDB's plan",
                 "[delim_cte_rewrites][plan]")
{
  // Both rules on, so that only the query's destination keeps them from the plan.
  sirius::test::scoped_sirius_setting exists_switch(*con, "enable_exists_from_aggregate", true);
  sirius::test::scoped_sirius_setting carry_switch(*con, "enable_subquery_key_carry", true);

  SECTION("GPU execution is off")
  {
    sirius::test::scoped_sirius_setting cpu(*con, "gpu_execution", false);
    auto const plan = optimized(*con, twin_query());
    INFO(plan.text);
    CHECK(plan.sales_scans == 2);
  }
  SECTION("the statement returns no rows, so Sirius never runs it")
  {
    auto const plan = optimized(*con, "CREATE TABLE revenue_copy AS " + twin_query());
    INFO(plan.text);
    CHECK(plan.sales_scans == 2);
  }
}
