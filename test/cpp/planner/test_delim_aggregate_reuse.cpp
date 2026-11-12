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

#include "op/sirius_physical_delim_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_delim_aggregate_reuse.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/scoped_sirius_setting.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;
using T = sirius::op::SiriusPhysicalOperatorType;

namespace {

/// Parse and optimize @p query, apply the rewrite when @p rewrite is set, then resolve it and hand
/// the logical plan to @p consume inside a transaction. Returns the number of aggregates rewritten.
template <class Consume>
std::size_t with_resolved_plan(Connection& con,
                               std::string const& query,
                               bool rewrite,
                               Consume&& consume)
{
  std::size_t rewrites = 0;
  auto& context        = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");
  try {
    Parser parser(context);
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    auto plan = std::move(planner.plan);
    Optimizer optimizer(*planner.binder, context);
    plan = optimizer.Optimize(std::move(plan));
    if (rewrite) {
      rewrites =
        sirius::planner::reuse_delim_rows_for_subquery_aggregates(context, *planner.binder, plan);
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver verifier(true);
    verifier.VisitOperator(*plan);
    ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    consume(context, plan);
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }
  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return rewrites;
}

std::size_t count(sirius::op::sirius_physical_operator* root, T type)
{
  if (!root) { return 0; }
  std::size_t found = root->type == type ? 1 : 0;
  for (auto& child : root->children) {
    found += count(child.get(), type);
  }
  // A delim join keeps its inner join and distinct outside `children`.
  if (root->type == T::LEFT_DELIM_JOIN || root->type == T::RIGHT_DELIM_JOIN) {
    found += count(root->Cast<sirius::op::sirius_physical_delim_join>().join.get(), type);
  }
  return found;
}

struct delim_reuse_fixture {
  delim_reuse_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    // The GPU seq_scan ingestible requires a single-file block manager: an on-disk database.
    db  = sirius::test::open_sirius_db(db_dir.file("delim_reuse.db").c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    for (char const* sql :
         {"CREATE TABLE part (p_partkey BIGINT, p_brand VARCHAR, p_container VARCHAR, "
          "p_size INTEGER, p_mfgr VARCHAR)",
          "INSERT INTO part SELECT range, 'Brand#' || (range % 5), 'C' || (range % 4), "
          "range % 7, 'M' || (range % 3) FROM range(1, 201)",
          "CREATE TABLE lineitem (l_partkey BIGINT, l_suppkey BIGINT, l_quantity DECIMAL(15, 2), "
          "l_extendedprice DECIMAL(15, 2), l_shipmode VARCHAR, l_discount DOUBLE)",
          "INSERT INTO lineitem SELECT (range * 7) % 200 + 1, range % 40 + 1, range % 50 + 1, "
          "(range % 1000) * 1.5, CASE WHEN range % 3 = 0 THEN 'AIR' ELSE 'RAIL' END, "
          "(range % 10) / 100.0 FROM range(5000)",
          "CREATE TABLE partsupp (ps_partkey BIGINT, ps_suppkey BIGINT, "
          "ps_supplycost DECIMAL(15, 2))",
          "INSERT INTO partsupp SELECT p, (p + s * 10) % 40 + 1, ((p * 31 + s * 17) % 997) / 10 "
          "FROM range(1, 201) t(p), range(4) u(s)",
          "CREATE TABLE supplier (s_suppkey BIGINT, s_name VARCHAR, s_nationkey INTEGER, "
          "s_acctbal DECIMAL(15, 2))",
          "INSERT INTO supplier SELECT range, 'S' || range, range % 25, range * 3.25 "
          "FROM range(1, 41)",
          "CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER)",
          "INSERT INTO nation SELECT range, 'N' || range, range % 5 FROM range(25)",
          "CREATE TABLE region (r_regionkey INTEGER, r_name VARCHAR)",
          "INSERT INTO region VALUES (0, 'AFRICA'), (1, 'AMERICA'), (2, 'ASIA'), (3, 'EUROPE'), "
          "(4, 'MIDDLE EAST')",
          "CREATE TABLE widths (k BIGINT, i INTEGER, b BIGINT, d9 DECIMAL(9, 2), "
          "d18 DECIMAL(18, 2), s SMALLINT, d4 DECIMAL(4, 1), h HUGEINT, d38 DECIMAL(38, 0))",
          "INSERT INTO widths SELECT range % 7, range, range, range, range, range, range / 10, "
          "range, range FROM range(100)"}) {
      auto result = con->Query(sql);
      INFO(sql);
      REQUIRE_FALSE(result->HasError());
    }
  }

  std::size_t rewrites(std::string const& query)
  {
    return with_resolved_plan(*con, query, true, [](ClientContext&, auto&) {});
  }

  /// Scans in @p query's physical plan, with and without the rewrite.
  std::pair<std::size_t, std::size_t> scans(std::string const& query)
  {
    std::size_t rewritten = 0, original = 0;
    for (bool const rewrite : {true, false}) {
      with_resolved_plan(*con, query, rewrite, [&](ClientContext& context, auto& plan) {
        sirius::planner::sirius_physical_plan_generator gen(context);
        auto physical = gen.create_plan(std::move(plan));
        REQUIRE(physical);
        (rewrite ? rewritten : original) = count(physical.get(), T::GPU_SCAN);
      });
    }
    return {rewritten, original};
  }

  sirius::test::scratch_dir db_dir{"delim_reuse"};
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

/// TPC-H q17's shape: an AVG over every lineitem of the outer row's part.
std::string q17_shape(std::string const& outer_filter    = "",
                      std::string const& subquery        = "0.2 * avg(l_quantity)",
                      std::string const& subquery_filter = "l_partkey = p_partkey")
{
  return "SELECT sum(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part "
         "WHERE p_partkey = l_partkey AND p_brand = 'Brand#3' AND p_container = 'C1' " +
         outer_filter + " AND l_quantity < (SELECT " + subquery + " FROM lineitem WHERE " +
         subquery_filter + ")";
}

/// TPC-H q2's shape: a MIN over the part's suppliers in one region, joined the same way.
std::string q2_shape(std::string const& subquery_tables = "partsupp, supplier, nation, region",
                     std::string const& subquery_filter =
                       "s_suppkey = ps_suppkey AND s_nationkey = n_nationkey AND "
                       "n_regionkey = r_regionkey AND r_name = 'EUROPE'")
{
  return "SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr "
         "FROM part, supplier, partsupp, nation, region "
         "WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 3 "
         "AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE' "
         "AND ps_supplycost = (SELECT min(ps_supplycost) FROM " +
         subquery_tables + " WHERE p_partkey = ps_partkey AND " + subquery_filter + ")";
}

}  // namespace

TEST_CASE_METHOD(delim_reuse_fixture,
                 "delim reuse computes q17- and q2-shaped aggregates from the outer rows",
                 "[delim_reuse][plan]")
{
  for (bool const as_cte : {false, true}) {
    CAPTURE(as_cte);
    sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
    CHECK(rewrites(q17_shape()) == 1);
    CHECK(rewrites(q2_shape()) == 1);
    // The subquery's second scan of lineitem, and of partsupp, go. DuckDB already shares
    // supplier, nation and region between the two sides as a CTE.
    CHECK(scans(q17_shape()) == std::pair<std::size_t, std::size_t>{2, 3});
    CHECK(scans(q2_shape()) == std::pair<std::size_t, std::size_t>{5, 6});
  }
}

TEST_CASE_METHOD(delim_reuse_fixture,
                 "delim reuse covers MIN and MAX, and outer rows keyed by the subquery's column",
                 "[delim_reuse][plan]")
{
  for (bool const as_cte : {false, true}) {
    CAPTURE(as_cte);
    sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
    CHECK(rewrites(q17_shape("", "max(l_quantity) - min(l_quantity)")) == 1);
    // Keyed by the lineitems' own, nullable, column, each repeated once per region.
    CHECK(rewrites("SELECT l_partkey, r_name, l_quantity, (SELECT avg(l2.l_quantity) FROM "
                   "lineitem l2 WHERE l2.l_partkey = l1.l_partkey) FROM lineitem l1, region "
                   "WHERE r_regionkey < 2") == 1);
    // The outer rows do not keep the aggregated column.
    CHECK(rewrites("SELECT l_partkey, r_name, (SELECT avg(l2.l_quantity) FROM lineitem l2 "
                   "WHERE l2.l_partkey = l1.l_partkey) FROM lineitem l1, region "
                   "WHERE r_regionkey < 2") == 0);
    // DuckDB already joins this one with an aggregate over all of lineitem, without a delim join.
    CHECK(rewrites("SELECT l_partkey, l_quantity FROM lineitem l1 WHERE l_quantity > "
                   "(SELECT avg(l_quantity) FROM lineitem l2 WHERE l2.l_partkey = l1.l_partkey)") ==
          0);
  }
}

TEST_CASE_METHOD(delim_reuse_fixture,
                 "delim reuse leaves subqueries whose rows differ from the outer rows alone",
                 "[delim_reuse][plan]")
{
  for (bool const as_cte : {false, true}) {
    CAPTURE(as_cte);
    sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
    // The outer rows drop lineitems the subquery counts.
    CHECK(rewrites(q17_shape("AND l_shipmode = 'AIR'")) == 0);
    // The subquery keeps fewer lineitems than the outer rows.
    CHECK(rewrites(q17_shape(
            "", "0.2 * avg(l_quantity)", "l_partkey = p_partkey AND l_shipmode = 'AIR'")) == 0);
    // Correlated on another column than the join key.
    CHECK(rewrites(q17_shape("", "0.2 * avg(l_quantity)", "l_suppkey = p_partkey")) == 0);
    // Aggregates that count repeated rows, and an AVG whose sum depends on the order.
    CHECK(rewrites(q17_shape("", "0.01 * sum(l_quantity)")) == 0);
    CHECK(rewrites(q17_shape("", "0.01 * count(*)")) == 0);
    CHECK(rewrites(q17_shape("", "avg(l_discount)")) == 0);
    // A subquery over another table.
    CHECK(rewrites("SELECT sum(l_extendedprice) FROM lineitem, part WHERE p_partkey = l_partkey "
                   "AND p_brand = 'Brand#3' AND l_quantity < "
                   "(SELECT min(ps_supplycost) FROM partsupp WHERE ps_partkey = p_partkey)") == 0);
    // Another region, or every supplier rather than the outer rows' region.
    CHECK(rewrites(q2_shape("partsupp, supplier, nation, region",
                            "s_suppkey = ps_suppkey AND s_nationkey = n_nationkey AND "
                            "n_regionkey = r_regionkey AND r_name = 'ASIA'")) == 0);
    CHECK(rewrites(q2_shape("partsupp, supplier", "s_suppkey = ps_suppkey")) == 0);
  }
}

TEST_CASE_METHOD(delim_reuse_fixture,
                 "delim reuse limits AVG to inputs whose repeated sum cannot overflow",
                 "[delim_reuse][plan]")
{
  // Each row of widths is repeated once per region, which scales the AVG's sum.
  auto const repeated_avg = [](std::string const& column) {
    return "SELECT a.k, a." + column + ", r_name, (SELECT avg(b." + column +
           ") FROM widths b WHERE b.k = a.k) FROM widths a, region WHERE r_regionkey < 2";
  };
  for (bool const as_cte : {false, true}) {
    CAPTURE(as_cte);
    sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
    for (char const* column : {"i", "b", "d9", "d18"}) {
      CAPTURE(column);
      CHECK(rewrites(repeated_avg(column)) == 1);
    }
    for (char const* column : {"s", "d4", "h", "d38"}) {
      CAPTURE(column);
      CHECK(rewrites(repeated_avg(column)) == 0);
    }
    // MIN and MAX do not sum, so any width is reused.
    CHECK(rewrites("SELECT a.k, a.d38, r_name, (SELECT max(b.d38) FROM widths b WHERE b.k = a.k) "
                   "FROM widths a, region WHERE r_regionkey < 2") == 1);
  }
}
