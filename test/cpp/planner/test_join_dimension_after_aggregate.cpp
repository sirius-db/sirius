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

// Which lookup joins transparent/join_dimension_after_aggregate.hpp moves above their aggregate,
// on DuckDB's optimized plans with Sirius's optimizer settings, and that DuckDB's CPU engine
// returns the original rows from the rewritten plan.

#include "transparent/join_dimension_after_aggregate.hpp"

#include <catch.hpp>
#include <duckdb.hpp>  // IWYU pragma: keep
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

namespace {

/// What the fixture's optimizer extension does to the next plans DuckDB optimizes.
struct rewrite_request : duckdb::OptimizerExtensionInfo {
  bool enabled = false;
  /// The estimate every aggregate of the plan gets before the rewrite, instead of DuckDB's.
  std::optional<duckdb::idx_t> aggregate_estimate;
  std::uint64_t min_carried_bytes = 1;
  std::size_t moved               = 0;
  /// DuckDB's estimate of the rows that reach the plan's first aggregate.
  duckdb::idx_t rows = 0;
  std::string plan;
};

void estimate_aggregates(duckdb::LogicalOperator& op, duckdb::idx_t estimate)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    op.SetEstimatedCardinality(estimate);
  }
  for (auto& child : op.children) {
    estimate_aggregates(*child, estimate);
  }
}

duckdb::LogicalOperator const* first_aggregate(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return &op; }
  for (auto const& child : op.children) {
    if (auto const* found = first_aggregate(*child)) { return found; }
  }
  return nullptr;
}

void rewrite(duckdb::OptimizerExtensionInput& input,
             duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  auto& request = static_cast<rewrite_request&>(*input.info);
  if (!request.enabled) { return; }
  if (request.aggregate_estimate) { estimate_aggregates(*plan, *request.aggregate_estimate); }
  auto const* aggregate = first_aggregate(*plan);
  request.rows          = aggregate ? aggregate->children[0]->estimated_cardinality : 0;
  request.moved         = sirius::transparent::join_dimensions_after_aggregate(
    input.context, plan, input.optimizer.binder, request.min_carried_bytes);
  request.plan = plan->ToString();
}

/// Parquet files read through views, as Sirius plans TPC-H.
///
/// TPC-H Q9's tables: nations 3 and 4 share one name; supplier 9 has no nation and supplier 10 one
/// that is not in the table. Every supplier supplies every part, and every second part is green.
///
/// Readings of meters at sites, in batches: the lookup table `sites` has a VARCHAR key, and sites
/// S0 and S1 share one label and zone; meter 38 has no site and meter 39 one that is not in the
/// table.
struct late_join_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  std::filesystem::path dir;
  duckdb::shared_ptr<rewrite_request> request = duckdb::make_shared_ptr<rewrite_request>();

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

  late_join_fixture()
  {
    static std::atomic<unsigned> counter{0};
    dir = std::filesystem::temp_directory_path() /
          ("sirius_late_join_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir);
    // Sirius plans with these optimizers off
    // (sirius_extension.cpp: publish_transparent_optimizer_mask).
    run(
      "SET disabled_optimizers = 'in_clause,compressed_materialization,"
      "partial_aggregate_pushdown,grouping_sets,aggregate_reuse,distinct_aggregate_rewrite,"
      "scalar_fn_pushdown,late_materialization'");
    parquet_table("lineitem",
                  "SELECT (i % 500)::BIGINT AS l_orderkey, (i % 37)::BIGINT AS l_partkey, (i % "
                  "11)::BIGINT AS l_suppkey, ((i % 50) + 1)::DECIMAL(15, 2) AS l_quantity, (i * "
                  "1.5)::DECIMAL(15, 2) AS l_extendedprice, ((i % 10) / 100.0)::DECIMAL(15, 2) AS "
                  "l_discount FROM range(4000) t(i)");
    parquet_table("orders",
                  "SELECT i::BIGINT AS o_orderkey, DATE '1995-01-01' + i::INTEGER AS o_orderdate "
                  "FROM range(500) t(i)");
    parquet_table("supplier",
                  "SELECT i::BIGINT AS s_suppkey, CASE WHEN i = 9 THEN NULL WHEN i = 10 THEN 7 "
                  "ELSE i % 5 END::BIGINT AS s_nationkey FROM range(11) t(i)");
    parquet_table("nation",
                  "SELECT i::BIGINT AS n_nationkey, CASE WHEN i >= 3 THEN 'SHARED' ELSE 'N' || i "
                  "END AS n_name, (i % 2)::BIGINT AS n_regionkey FROM range(5) t(i)");
    parquet_table("part",
                  "SELECT i::BIGINT AS p_partkey, CASE WHEN i % 2 = 0 THEN 'dark green' ELSE "
                  "'pale red' END AS p_name FROM range(37) t(i)");
    parquet_table("partsupp",
                  "SELECT (i % 37)::BIGINT AS ps_partkey, (i % 11)::BIGINT AS ps_suppkey, ((i * "
                  "13) % 100 + 1)::DECIMAL(15, 2) AS ps_supplycost FROM range(407) t(i)");
    parquet_table("readings",
                  "SELECT (i % 40)::BIGINT AS meter, (i % 21)::BIGINT AS batch, i::BIGINT AS "
                  "reading FROM range(3000) t(i)");
    parquet_table("batches",
                  "SELECT i::BIGINT AS batch_id, (i % 3)::INTEGER AS shift FROM range(21) t(i)");
    parquet_table("meters",
                  "SELECT i::BIGINT AS meter_id, CASE WHEN i = 38 THEN NULL WHEN i = 39 THEN "
                  "'S9' ELSE 'S' || (i % 6) END AS site FROM range(40) t(i)");
    parquet_table("sites",
                  "SELECT 'S' || i AS code, CASE WHEN i <= 1 THEN 'north' ELSE 'L' || i END AS "
                  "label, CASE WHEN i <= 1 THEN 1 ELSE i END::INTEGER AS zone FROM range(6) t(i)");

    duckdb::OptimizerExtension extension;
    extension.optimize_function = rewrite;
    extension.optimizer_info    = request;
    duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*db.instance),
                                         std::move(extension));
  }

  ~late_join_fixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }

  /// Checks that the rewrite moves @p moved lookup joins in DuckDB's optimized plan for @p sql,
  /// and that DuckDB then returns the rows of the original plan, @p rows of them.
  void expect_moved(std::size_t moved, std::string const& sql, duckdb::idx_t rows)
  {
    INFO(sql);
    auto original = con.Query(sql);
    REQUIRE_FALSE(original->HasError());
    request->moved   = 0;
    request->enabled = true;
    auto rewritten   = con.Query(sql);
    request->enabled = false;
    INFO(request->plan);
    INFO(rewritten->ToString());
    REQUIRE_FALSE(rewritten->HasError());
    CHECK(request->moved == moved);
    CHECK(original->RowCount() == rows);
    CHECK(rewritten->ToString() == original->ToString());
  }
};

/// TPC-H Q9's shape: @p aggregates of lineitem joined to orders, and to supplier joined to
/// nation with @p nation_join, by nation name and order year.
std::string by_nation_and_year(
  std::string const& aggregates  = "sum(amount)",
  std::string const& nation_join = "join nation on s_nationkey = n_nationkey")
{
  return "select nation, o_year, " + aggregates +
         " from (select n_name as nation, extract(year from o_orderdate) as o_year, "
         "l_extendedprice * (1 - l_discount) as amount, l_quantity, l_suppkey from lineitem join "
         "orders on o_orderkey = l_orderkey join supplier on s_suppkey = l_suppkey " +
         nation_join + ") group by nation, o_year order by nation, o_year";
}

// Four names over two order years: suppliers 0 to 8 have nations 0 to 4, which carry the names
// N0, N1, N2 and SHARED (twice).
constexpr duckdb::idx_t names_and_years = 4 * 2;

}  // namespace

TEST_CASE_METHOD(late_join_fixture,
                 "a lookup table that only names GROUP BY keys is joined after the aggregate",
                 "[planner][join_dimension_after_aggregate]")
{
  SECTION("TPC-H Q9")
  {
    expect_moved(
      1,
      "select nation, o_year, sum(amount) as sum_profit from (select n_name as nation, "
      "extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) - "
      "ps_supplycost * l_quantity as amount from part, supplier, lineitem, partsupp, orders, "
      "nation where s_suppkey = l_suppkey and ps_suppkey = l_suppkey and ps_partkey = l_partkey "
      "and p_partkey = l_partkey and o_orderkey = l_orderkey and s_nationkey = n_nationkey and "
      "p_name like '%green%') as profits group by nation, o_year order by nation, o_year desc",
      names_and_years);
  }
  SECTION("TPC-H Q9's shape") { expect_moved(1, by_nation_and_year(), names_and_years); }
  SECTION("min, max and a narrow sum")
  {
    expect_moved(
      1, by_nation_and_year("min(l_quantity), max(amount), sum(l_quantity)"), names_and_years);
  }
  SECTION("two lookup columns after another key")
  {
    // N0 and N2 are in region 0, N1 in region 1, and SHARED in both.
    expect_moved(
      1,
      "select extract(year from o_orderdate) as o_year, n_name, n_regionkey, sum(l_quantity), "
      "min(l_extendedprice) from lineitem join orders on o_orderkey = l_orderkey join supplier "
      "on s_suppkey = l_suppkey join nation on s_nationkey = n_nationkey group by o_year, "
      "n_name, n_regionkey order by o_year, n_name, n_regionkey",
      2 * 5);
  }
  SECTION("a lookup table with a VARCHAR key, on another branch than the largest table")
  {
    // Every meter is read in every batch, so each of the three shifts has the five (label, zone)
    // pairs of the six sites.
    expect_moved(1,
                 "select shift, label, zone, sum(reading), min(reading) from readings join "
                 "batches on batch = batch_id join meters on meter = meter_id join sites on site "
                 "= code group by shift, label, zone order by shift, label, zone",
                 3 * 5);
  }
  SECTION("an aggregate estimated at the bound")
  {
    request->aggregate_estimate = sirius::transparent::max_late_join_rows;
    expect_moved(1, by_nation_and_year(), names_and_years);
  }
}

TEST_CASE_METHOD(late_join_fixture,
                 "a lookup join that filters, feeds an expression or cannot be merged stays below "
                 "the aggregate",
                 "[planner][join_dimension_after_aggregate]")
{
  struct variant {
    char const* why;
    std::string sql;
    duckdb::idx_t rows;
  };
  for (auto const& [why, sql, rows] : {
         variant{"a filter on the lookup table",
                 by_nation_and_year("sum(amount)",
                                    "join nation on s_nationkey = n_nationkey and n_regionkey = 0"),
                 3 * 2},
         variant{"a lookup column inside an aggregate's argument",
                 by_nation_and_year("sum(case when nation = 'N1' then amount else 0 end)"),
                 names_and_years},
         variant{"avg", by_nation_and_year("avg(amount)"), names_and_years},
         variant{
           "count(DISTINCT)", by_nation_and_year("count(distinct l_suppkey)"), names_and_years},
         variant{"sum(DISTINCT)", by_nation_and_year("sum(distinct l_quantity)"), names_and_years},
         variant{"a LEFT join, which keeps the suppliers without a nation",
                 by_nation_and_year("sum(amount)", "left join nation on s_nationkey = n_nationkey"),
                 (4 + 1) * 2},
       }) {
    INFO(why);
    expect_moved(0, sql, rows);
  }
}

TEST_CASE_METHOD(late_join_fixture,
                 "an aggregate estimated above the bound keeps its lookup join",
                 "[planner][join_dimension_after_aggregate]")
{
  request->aggregate_estimate = sirius::transparent::max_late_join_rows + 1;
  expect_moved(0, by_nation_and_year(), names_and_years);
}

TEST_CASE_METHOD(late_join_fixture,
                 "a lookup join moves only when the rows above it carry enough of its columns",
                 "[planner][join_dimension_after_aggregate]")
{
  // The join emits n_name, 16 bytes as a variable-size column, and not the 8-byte s_nationkey
  // that replaces it. DuckDB estimates 2,500 rows into this aggregate, which makes 20,000 bytes.
  constexpr std::uint64_t width = 16 - 8;
  expect_moved(1, by_nation_and_year(), names_and_years);
  auto const carried = request->rows * width;
  CAPTURE(request->rows);
  REQUIRE(carried > 0);

  request->min_carried_bytes = carried;
  expect_moved(1, by_nation_and_year(), names_and_years);
  request->min_carried_bytes = carried + 1;
  expect_moved(0, by_nation_and_year(), names_and_years);
}
