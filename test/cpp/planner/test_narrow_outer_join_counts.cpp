/*
 * Copyright 2025, Sirius Contributors.
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

// Which COUNT arguments transparent/narrow_outer_join_counts.hpp replaces by a one-byte flag below
// an outer join, on DuckDB's optimized plans with Sirius's optimizer settings.

#include "transparent/narrow_outer_join_counts.hpp"

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

/// customer and orders as Parquet files read through views, as Sirius plans TPC-H.
struct count_flag_fixture {
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

  count_flag_fixture()
  {
    static std::atomic<unsigned> counter{0};
    dir = std::filesystem::temp_directory_path() /
          ("sirius_count_flag_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir);
    // Sirius plans with these optimizers off
    // (sirius_extension.cpp: publish_transparent_optimizer_mask).
    run(
      "SET disabled_optimizers = 'in_clause,compressed_materialization,"
      "partial_aggregate_pushdown,grouping_sets,aggregate_reuse,distinct_aggregate_rewrite,"
      "scalar_fn_pushdown,late_materialization'");
    parquet_table("customer", "SELECT i::BIGINT AS c_custkey FROM range(200) t(i)");
    parquet_table("orders",
                  "SELECT i::BIGINT AS o_orderkey, (i % 150)::BIGINT AS o_custkey, (i % "
                  "2 = 0) AS o_flag, 'comment ' || i AS o_comment FROM range(1000) t(i)");
  }

  ~count_flag_fixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }

  /// The number of COUNT arguments the pass narrows in DuckDB's optimized plan for @p sql, as
  /// Sirius gets it (bindings, not yet resolved).
  std::size_t narrowed(std::string const& sql)
  {
    std::size_t count = 0;
    con.context->RunFunctionInTransaction([&] {
      duckdb::Parser parser(*con.context);
      parser.ParseQuery(sql);
      duckdb::Planner planner(*con.context);
      planner.CreatePlan(std::move(parser.statements[0]));
      duckdb::Optimizer optimizer(*planner.binder, *con.context);
      auto plan = optimizer.Optimize(std::move(planner.plan));
      count     = sirius::transparent::narrow_outer_join_counts(*plan, *planner.binder);
      // The rewritten plan still resolves: every binding it reads exists below it, with its type.
      plan->ResolveOperatorTypes();
      duckdb::ColumnBindingResolver resolver;
      resolver.VisitOperator(*plan);
    });
    return count;
  }
};

/// TPC-H Q13's shape: orders per customer through @p join, aggregated by @p aggregates.
std::string orders_per_customer(std::string const& aggregates,
                                std::string const& join = "left outer join")
{
  return "select c_custkey, " + aggregates + " from customer " + join +
         " orders on c_custkey = o_custkey and o_comment not like '%7%' group by c_custkey";
}

}  // namespace

TEST_CASE_METHOD(count_flag_fixture,
                 "a column that an aggregate only counts crosses the outer join as a flag",
                 "[planner][narrow_outer_join_counts]")
{
  CHECK(narrowed(orders_per_customer("count(o_orderkey) as c_count")) == 1);
}

TEST_CASE_METHOD(count_flag_fixture,
                 "the flag is built on the null-extended side of a RIGHT join too",
                 "[planner][narrow_outer_join_counts]")
{
  CHECK(narrowed("select c_custkey, count(o_orderkey) as c_count from orders right outer join "
                 "customer on c_custkey = o_custkey group by c_custkey") == 1);
}

TEST_CASE_METHOD(
  count_flag_fixture,
  "counts that are not a plain count of one wide null-extended column keep their column",
  "[planner][narrow_outer_join_counts]")
{
  struct variant {
    char const* why;
    std::string sql;
  };
  for (auto const& [why, sql] : {
         variant{"count(*) reads no column", orders_per_customer("count(*) as c_count")},
         variant{"the column is also read by another aggregate",
                 orders_per_customer("count(o_orderkey) as c_count, max(o_orderkey) as last")},
         variant{"a DISTINCT count needs the values",
                 orders_per_customer("count(distinct o_orderkey) as c_count")},
         variant{"an inner join extends no side with NULLs",
                 orders_per_customer("count(o_orderkey) as c_count", "join")},
         variant{"the column comes from the preserved side",
                 orders_per_customer("count(c_custkey) as c_count")},
         variant{"a one-byte column is already as narrow as the flag",
                 orders_per_customer("count(o_flag) as c_count")},
       }) {
    INFO(why);
    CHECK(narrowed(sql) == 0);
  }
}
