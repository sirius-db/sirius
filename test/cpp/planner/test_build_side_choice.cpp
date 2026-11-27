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

// Which inner joins planner/build_side_choice.hpp builds on their filtered side, on DuckDB's
// optimized plans with Sirius's optimizer settings, and that DuckDB's CPU engine returns the
// original rows from the swapped plan.

#include "planner/build_side_choice.hpp"

#include <catch.hpp>
#include <duckdb.hpp>  // IWYU pragma: keep
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

/// What the fixture's optimizer extension does to the next plans DuckDB optimizes.
struct swap_request : duckdb::OptimizerExtensionInfo {
  bool enabled                  = false;
  std::uint64_t min_build_bytes = 1;
  std::size_t swapped           = 0;
  std::string plan;
};

void rewrite(duckdb::OptimizerExtensionInput& input,
             duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  auto& request = static_cast<swap_request&>(*input.info);
  if (!request.enabled) { return; }
  request.swapped =
    sirius::planner::swap_unfiltered_builds(input.context, *plan, request.min_build_bytes);
  request.plan = plan->ToString();
}

/// Parquet files read through views, as Sirius plans TPC-H.
///
/// 1000 orders of 12 bytes each (a BIGINT key and a DATE), and 4000 items of 8 wide columns, four
/// to an order, every second one green. DuckDB estimates 800 items for a filter it cannot rate,
/// fewer rows than the orders but more bytes, so it builds the join on the orders.
struct build_side_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  std::filesystem::path dir;
  duckdb::shared_ptr<swap_request> request = duckdb::make_shared_ptr<swap_request>();

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

  build_side_fixture()
  {
    static std::atomic<unsigned> counter{0};
    dir = std::filesystem::temp_directory_path() /
          ("sirius_build_side_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir);
    // Sirius plans with these optimizers off
    // (sirius_extension.cpp: publish_transparent_optimizer_mask).
    run(
      "SET disabled_optimizers = 'in_clause,compressed_materialization,"
      "partial_aggregate_pushdown,grouping_sets,aggregate_reuse,distinct_aggregate_rewrite,"
      "scalar_fn_pushdown,late_materialization'");
    parquet_table("orders",
                  "SELECT i::BIGINT AS o_orderkey, DATE '1995-01-01' + (i % 900)::INTEGER AS "
                  "o_orderdate FROM range(1000) t(i)");
    parquet_table("items",
                  "SELECT (i % 1000)::BIGINT AS i_orderkey, CASE WHEN i % 2 = 0 THEN 'dark green' "
                  "ELSE 'pale red' END AS i_name, 'a' || i AS c1, 'b' || i AS c2, 'c' || i AS c3, "
                  "'d' || i AS c4, 'e' || i AS c5, 'f' || i AS c6 FROM range(4000) t(i)");

    duckdb::OptimizerExtension extension;
    extension.optimize_function = rewrite;
    extension.optimizer_info    = request;
    duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*db.instance),
                                         std::move(extension));
  }

  ~build_side_fixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }

  /// Checks that @p swapped joins of DuckDB's optimized plan for @p sql are swapped when a build
  /// of @p min_build_bytes is one too large to hold, and that DuckDB then returns the rows of the
  /// original plan.
  void expect_swapped(std::size_t swapped, std::string const& sql, std::uint64_t min_build_bytes)
  {
    INFO(sql);
    auto original = con.Query(sql);
    REQUIRE_FALSE(original->HasError());
    request->swapped         = 0;
    request->min_build_bytes = min_build_bytes;
    request->enabled         = true;
    auto rewritten           = con.Query(sql);
    request->enabled         = false;
    INFO(request->plan);
    REQUIRE_FALSE(rewritten->HasError());
    CHECK(request->swapped == swapped);
    CHECK(original->RowCount() > 0);
    CHECK(rewritten->ToString() == original->ToString());
  }
};

/// The items of each order year, @p filter applied to the items.
std::string items_by_year(std::string const& filter)
{
  return "select extract(year from o_orderdate) as o_year, count(*), max(c1), max(c2), max(c3), "
         "max(c4), max(c5), max(c6) from items join orders on i_orderkey = o_orderkey " +
         filter + " group by o_year order by o_year";
}

}  // namespace

TEST_CASE_METHOD(build_side_fixture,
                 "an unfiltered build too large to hold gives way to the filtered probe",
                 "[build_side_choice]")
{
  auto const green = items_by_year("where i_name like '%green%'");
  // The orders are estimated at 1000 rows of 12 bytes.
  expect_swapped(1, green, 1);
  expect_swapped(1, green, 6'000);
}

TEST_CASE_METHOD(build_side_fixture,
                 "an unfiltered build that can be held keeps its side",
                 "[build_side_choice]")
{
  expect_swapped(0, items_by_year("where i_name like '%green%'"), 24'000);
}

TEST_CASE_METHOD(build_side_fixture,
                 "a join with no filter on its probe keeps its sides",
                 "[build_side_choice]")
{
  expect_swapped(0, items_by_year(""), 1);
}
