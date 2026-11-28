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

// Which projection expressions transparent/push_narrowing_expressions.hpp moves below an inner
// join, on DuckDB's optimized plans with Sirius's optimizer settings.

#include "transparent/push_narrowing_expressions.hpp"

#include <catch.hpp>
#include <duckdb.hpp>  // IWYU pragma: keep
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

/// lineitem and orders as Parquet files read through views, as Sirius plans TPC-H.
struct pushdown_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  std::filesystem::path dir;

  void run(std::string const& sql)
  {
    auto result = con.Query(sql);
    INFO(sql);
    REQUIRE_FALSE(result->HasError());
  }

  void parquet_table(std::string const& name, std::string const& select)
  {
    auto const file = (dir / (name + ".parquet")).string();
    run("COPY (" + select + ") TO '" + file + "' (FORMAT parquet)");
    run("CREATE VIEW " + name + " AS FROM read_parquet('" + file + "')");
  }

  pushdown_fixture()
  {
    static std::atomic<unsigned> counter{0};
    dir = std::filesystem::temp_directory_path() /
          ("sirius_pushdown_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir);
    // Sirius plans with these optimizers off
    // (sirius_extension.cpp: publish_transparent_optimizer_mask).
    run(
      "SET disabled_optimizers = 'in_clause,compressed_materialization,"
      "partial_aggregate_pushdown,grouping_sets,aggregate_reuse,distinct_aggregate_rewrite,"
      "scalar_fn_pushdown,late_materialization'");
    parquet_table(
      "lineitem",
      "SELECT (i % 500)::BIGINT AS l_orderkey, (i % 37)::BIGINT AS l_partkey, (i % "
      "7)::INTEGER AS l_linenumber, ((i % 50) + 1)::DECIMAL(15, 2) AS l_quantity, (i * "
      "1.5)::DECIMAL(15, 2) AS l_extendedprice, ((i % 10) / 100.0)::DECIMAL(15, 2) AS "
      "l_discount, ((i % 8) / 100.0)::DECIMAL(15, 2) AS l_tax, ((i % 50) + 1)::DOUBLE AS "
      "f_quantity, (i * 1.5)::DOUBLE AS f_extendedprice, ((i % 10) / 100.0)::DOUBLE AS "
      "f_discount, ((i % 8) / 100.0)::DOUBLE AS f_tax FROM range(4000) t(i)");
    parquet_table("orders",
                  "SELECT i::BIGINT AS o_orderkey, DATE '1995-01-01' + (i % 300)::INTEGER AS "
                  "o_orderdate, (i * 3.25)::DECIMAL(15, 2) AS o_totalprice FROM range(500) t(i)");
  }

  ~pushdown_fixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }

  /// DuckDB's optimized plan for @p sql, as Sirius gets it (bindings, not yet resolved), rewritten;
  /// @p moved gets the number of moved expressions.
  duckdb::unique_ptr<duckdb::LogicalOperator> rewritten(std::string const& sql, std::size_t& moved)
  {
    duckdb::unique_ptr<duckdb::LogicalOperator> plan;
    con.context->RunFunctionInTransaction([&] {
      duckdb::Parser parser(*con.context);
      parser.ParseQuery(sql);
      duckdb::Planner planner(*con.context);
      planner.CreatePlan(std::move(parser.statements[0]));
      duckdb::Optimizer optimizer(*planner.binder, *con.context);
      plan  = optimizer.Optimize(std::move(planner.plan));
      moved = sirius::transparent::push_narrowing_expressions(*plan, *planner.binder);
      // The rewritten plan still resolves: every binding it reads exists below it, with its type.
      plan->ResolveOperatorTypes();
      duckdb::ColumnBindingResolver resolver;
      resolver.VisitOperator(*plan);
    });
    return plan;
  }
};

/// TPC-H Q9's shape: a yearly SUM of @p amount over lineitem joined to orders with @p join.
std::string yearly_sum(std::string const& amount,
                       std::string const& also        = "",
                       std::string const& join        = "join",
                       std::string const& outer_extra = "")
{
  return "select extract(year from o_orderdate) as o_year, sum(amount)" + outer_extra +
         " from (select o_orderdate, " + amount + " as amount" + also + " from lineitem " + join +
         " orders on o_orderkey = l_orderkey) group by o_year";
}

constexpr char const* q9_amount = "l_extendedprice * (1 - l_discount) - l_tax * l_quantity";
/// The same amount over the DOUBLE columns of a dataset without DECIMAL.
constexpr char const* q9_amount_double = "f_extendedprice * (1 - f_discount) - f_tax * f_quantity";

}  // namespace

TEST_CASE_METHOD(pushdown_fixture,
                 "a DECIMAL expression narrower than its inputs is computed below the inner join",
                 "[planner][push_narrowing_expressions]")
{
  std::size_t moved = 0;
  (void)rewritten(yearly_sum(q9_amount), moved);
  CHECK(moved == 1);
}

TEST_CASE_METHOD(pushdown_fixture,
                 "a DOUBLE expression narrower than its inputs is computed below the inner join",
                 "[planner][push_narrowing_expressions]")
{
  std::size_t moved = 0;
  (void)rewritten(yearly_sum(q9_amount_double), moved);
  CHECK(moved == 1);
}

TEST_CASE_METHOD(
  pushdown_fixture,
  "expressions that would not narrow the rows, or are not DECIMAL or DOUBLE arithmetic, stay above "
  "the join",
  "[planner][push_narrowing_expressions]")
{
  struct variant {
    char const* why;
    std::string sql;
  };
  for (auto const& [why, sql] : {
         variant{"inputs from both sides of the join",
                 yearly_sum("l_extendedprice * (1 - l_discount) - o_totalprice * l_quantity")},
         variant{"a result as wide as its one input", yearly_sum("l_extendedprice * 2")},
         variant{"an input the projection also returns",
                 yearly_sum(q9_amount, ", l_quantity", "join", ", max(l_quantity)")},
         variant{"an outer join", yearly_sum(q9_amount, "", "left join")},
         variant{"integer arithmetic",
                 yearly_sum("l_orderkey * l_partkey - l_linenumber * l_partkey")},
       }) {
    INFO(why);
    std::size_t moved = 0;
    (void)rewritten(sql, moved);
    CHECK(moved == 0);
  }
}
