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

// Which plans planner/scan_reuse_rewrites.hpp rewrites, on DuckDB's optimized plans with
// Sirius's optimizer settings. Answers are checked on the GPU against DuckDB in
// test/cpp/integration/test_gpu_execution_scan_reuse.cpp.

#include "planner/scan_reuse_rewrites.hpp"

#include <catch.hpp>
#include <duckdb.hpp>  // IWYU pragma: keep
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

/// The tables, as Parquet files read through views: Sirius plans TPC-H from views over
/// read_parquet, where DuckDB's COMMON_SUBPLAN optimizer does not share Q2's or Q11's joins (it
/// does over native tables, which would hide the shapes these rules rewrite).
struct scan_reuse_fixture {
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

  scan_reuse_fixture()
  {
    static std::atomic<unsigned> counter{0};
    dir = std::filesystem::temp_directory_path() /
          ("sirius_scan_reuse_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir);
    // Sirius plans with these optimizers off and explicit delim joins
    // (sirius_extension.cpp: publish_transparent_optimizer_mask, default_to_explicit_delim_joins).
    run(
      "SET disabled_optimizers = 'in_clause,compressed_materialization,"
      "partial_aggregate_pushdown,grouping_sets,aggregate_reuse,distinct_aggregate_rewrite,"
      "scalar_fn_pushdown,late_materialization'");
    run("SET delim_join_as_cte = false");
    parquet_table("lineitem",
                  "SELECT (i % 500)::BIGINT AS l_orderkey, (i % 37)::BIGINT AS l_partkey, (i % "
                  "11)::BIGINT AS l_suppkey, (i % 7)::INTEGER AS l_linenumber, ((i % 50) + "
                  "1)::DECIMAL(15, 2) AS l_quantity, (i * 1.5)::DECIMAL(15, 2) AS l_extendedprice "
                  "FROM range(4000) t(i)");
    parquet_table("orders",
                  "SELECT i::BIGINT AS o_orderkey, (i % 50)::BIGINT AS o_custkey, DATE "
                  "'1995-01-01' + (i % 300)::INTEGER AS o_orderdate, (i * 3.25)::DECIMAL(15, 2) "
                  "AS o_totalprice FROM range(500) t(i)");
    parquet_table("customer",
                  "SELECT i::BIGINT AS c_custkey, 'Customer#' || i AS c_name FROM range(50) t(i)");
    parquet_table("part",
                  "SELECT i::BIGINT AS p_partkey, 'Brand#' || (i % 5) AS p_brand, CASE WHEN i % 3 "
                  "= 0 THEN 'JUMBO CAN' ELSE 'SM BOX' END AS p_container, (i % 4)::INTEGER AS "
                  "p_size, CASE WHEN i % 2 = 0 THEN 'STANDARD NICKEL' ELSE 'SMALL BRASS' END AS "
                  "p_type, 'Manufacturer#' || (i % 3) AS p_mfgr FROM range(37) t(i)");
    parquet_table("supplier",
                  "SELECT i::BIGINT AS s_suppkey, 'Supplier#' || i AS s_name, (i % 5)::BIGINT AS "
                  "s_nationkey, (i * 10.5)::DECIMAL(15, 2) AS s_acctbal, 'addr' || i AS "
                  "s_address, 'phone' || i AS s_phone, 'comment' || i AS s_comment FROM "
                  "range(11) t(i)");
    parquet_table("partsupp",
                  "SELECT (i % 37)::BIGINT AS ps_partkey, (i % 11)::BIGINT AS ps_suppkey, ((i * "
                  "13) % 100 + 1)::DECIMAL(15, 2) AS ps_supplycost, (i % 9)::INTEGER AS "
                  "ps_availqty FROM range(400) t(i)");
    parquet_table("nation",
                  "SELECT i::BIGINT AS n_nationkey, CASE WHEN i = 1 THEN 'GERMANY' ELSE 'N' || i "
                  "END AS n_name, (i % 2)::BIGINT AS n_regionkey FROM range(5) t(i)");
    parquet_table("region",
                  "SELECT i::BIGINT AS r_regionkey, CASE WHEN i = 0 THEN 'EUROPE' ELSE 'ASIA' END "
                  "AS r_name FROM range(2) t(i)");
  }

  ~scan_reuse_fixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }

  /// DuckDB's optimized plan for @p sql, as Sirius gets it (bindings, not yet resolved), rewritten;
  /// @p counts gets the rules that fired.
  duckdb::unique_ptr<duckdb::LogicalOperator> rewritten(std::string const& sql,
                                                        sirius::planner::scan_reuse_counts& counts)
  {
    duckdb::unique_ptr<duckdb::LogicalOperator> plan;
    con.context->RunFunctionInTransaction([&] {
      duckdb::Parser parser(*con.context);
      parser.ParseQuery(sql);
      duckdb::Planner planner(*con.context);
      planner.CreatePlan(std::move(parser.statements[0]));
      duckdb::Optimizer optimizer(*planner.binder, *con.context);
      plan   = optimizer.Optimize(std::move(planner.plan));
      counts = sirius::planner::rewrite_repeated_scans(*con.context, plan);
      // The rewritten plan still resolves: every binding it reads exists below it, with its type.
      plan->ResolveOperatorTypes();
      duckdb::ColumnBindingResolver resolver;
      resolver.VisitOperator(*plan);
    });
    return plan;
  }
};

/// Scans of the fixture's Parquet file for @p table.
std::size_t scans_of(duckdb::LogicalOperator& op, std::string const& table)
{
  std::size_t n = 0;
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& get = op.Cast<duckdb::LogicalGet>();
    if (!get.parameters.empty() &&
        get.parameters[0].ToString().find("/" + table + ".parquet") != std::string::npos) {
      ++n;
    }
  }
  for (auto& child : op.children) {
    n += scans_of(*child, table);
  }
  return n;
}

std::size_t joins_of(duckdb::LogicalOperator& op, duckdb::JoinType type)
{
  std::size_t n = 0;
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
      op.Cast<duckdb::LogicalComparisonJoin>().join_type == type) {
    ++n;
  }
  for (auto& child : op.children) {
    n += joins_of(*child, type);
  }
  return n;
}

/// TPC-H Q18 with its parameter scaled to the fixture, and variants of it.
std::string q18(std::string const& outer_sum   = "sum(l_quantity)",
                std::string const& outer_where = "",
                std::string const& groups      = "l_orderkey",
                std::string const& extra_group = "")
{
  return "select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice" + extra_group + ", " +
         outer_sum +
         " from customer, orders, lineitem where o_orderkey in (select l_orderkey from lineitem "
         "group by " +
         groups +
         " having sum(l_quantity) > 150) and c_custkey = o_custkey and o_orderkey = l_orderkey" +
         outer_where + " group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice" +
         extra_group + " order by o_totalprice desc, o_orderdate limit 100";
}

}  // namespace

TEST_CASE_METHOD(scan_reuse_fixture,
                 "TPC-H Q18 reuses its IN-subquery aggregate and scans lineitem once",
                 "[planner][scan_reuse]")
{
  sirius::planner::scan_reuse_counts counts;
  auto plan = rewritten(q18(), counts);
  CHECK(counts.semi_join_aggregates == 1);
  CHECK(scans_of(*plan, "lineitem") == 1);
  CHECK(joins_of(*plan, duckdb::JoinType::SEMI) == 0);
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "Q18 shapes whose outer SUM is not the subquery's are left alone",
                 "[planner][scan_reuse]")
{
  struct variant {
    char const* why;
    std::string sql;
  };
  for (auto const& [why, sql] : {
         variant{"a filter on the outer lineitem only",
                 q18("sum(l_quantity)", " and l_linenumber > 2")},
         variant{"a COUNT in the outer aggregate", q18("count(*)")},
         variant{"an outer SUM of a column the subquery does not sum", q18("sum(l_extendedprice)")},
         variant{"subquery groups that are not unique per key",
                 q18("sum(l_quantity)", "", "l_orderkey, l_linenumber")},
         variant{"an outer group on a lineitem column",
                 q18("sum(l_quantity)", "", "l_orderkey", ", l_linenumber")},
       }) {
    INFO(why);
    sirius::planner::scan_reuse_counts counts;
    auto plan = rewritten(sql, counts);
    CHECK(counts.semi_join_aggregates == 0);
    CHECK(scans_of(*plan, "lineitem") == 2);
  }
}

namespace {

/// TPC-H Q17 with its parameters matched to the fixture, and variants of it.
std::string q17(std::string const& aggregate      = "0.2 * avg(l2.l_quantity)",
                std::string const& outer_where    = "",
                std::string const& subquery_where = "")
{
  return "select sum(l.l_extendedprice) / 7.0 as avg_yearly from lineitem l, part p where "
         "p.p_partkey = l.l_partkey and p.p_brand = 'Brand#1' and p.p_container = 'JUMBO CAN' "
         "and l.l_quantity < (select " +
         aggregate + " from lineitem l2 where l2.l_partkey = p.p_partkey" + subquery_where + ")" +
         outer_where;
}

/// TPC-H Q2, parameters matched to the fixture.
std::string q2(std::string const& subquery_region = "EUROPE")
{
  return "select s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, s.s_address, s.s_phone, "
         "s.s_comment from part p, supplier s, partsupp ps, nation n, region r where p.p_partkey "
         "= ps.ps_partkey and s.s_suppkey = ps.ps_suppkey and p.p_size = 1 and p.p_type like "
         "'%NICKEL' and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey and "
         "r.r_name = 'EUROPE' and ps.ps_supplycost = (select min(ps.ps_supplycost) from partsupp "
         "ps, supplier s, nation n, region r where p.p_partkey = ps.ps_partkey and s.s_suppkey = "
         "ps.ps_suppkey and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey and "
         "r.r_name = '" +
         subquery_region +
         "') order by s.s_acctbal desc, n.n_name, s.s_name, p.p_partkey limit 100";
}

std::size_t materialized_ctes(duckdb::LogicalOperator& op)
{
  std::size_t n = op.type == duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE ? 1 : 0;
  for (auto& child : op.children) {
    n += materialized_ctes(*child);
  }
  return n;
}

}  // namespace

TEST_CASE_METHOD(scan_reuse_fixture,
                 "TPC-H Q17 reads lineitem once, from a CTE of the outer join",
                 "[planner][scan_reuse]")
{
  sirius::planner::scan_reuse_counts counts;
  auto plan = rewritten(q17(), counts);
  CHECK(counts.correlated_aggregates == 1);
  CHECK(scans_of(*plan, "lineitem") == 1);
  CHECK(materialized_ctes(*plan) == 1);
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "TPC-H Q2 reads partsupp, supplier, nation and region once",
                 "[planner][scan_reuse]")
{
  sirius::planner::scan_reuse_counts counts;
  auto plan = rewritten(q2(), counts);
  CHECK(counts.correlated_aggregates == 1);
  for (auto const* table : {"partsupp", "supplier", "nation", "region", "part"}) {
    INFO(table);
    CHECK(scans_of(*plan, table) == 1);
  }
}

TEST_CASE_METHOD(scan_reuse_fixture,
                 "correlated subqueries that the outer rows cannot answer are left alone",
                 "[planner][scan_reuse]")
{
  struct variant {
    char const* why;
    std::string sql;
    char const* table;
    std::size_t scans;
  };
  for (auto const& [why, sql, table, scans] : {
         variant{"a SUM, which repeated outer rows change",
                 q17("0.2 * sum(l2.l_quantity)"),
                 "lineitem",
                 2},
         variant{"a COUNT", q17("count(*)"), "lineitem", 2},
         variant{"a filter on the outer lineitem only",
                 q17("0.2 * avg(l2.l_quantity)", " and l.l_linenumber > 2"),
                 "lineitem",
                 2},
         variant{"a filter in the subquery only",
                 q17("0.2 * avg(l2.l_quantity)", "", " and l2.l_linenumber > 2"),
                 "lineitem",
                 2},
         variant{"a different region in the subquery", q2("ASIA"), "partsupp", 2},
       }) {
    INFO(why);
    sirius::planner::scan_reuse_counts counts;
    auto plan = rewritten(sql, counts);
    CHECK(counts.correlated_aggregates == 0);
    CHECK(scans_of(*plan, table) == scans);
  }
}
