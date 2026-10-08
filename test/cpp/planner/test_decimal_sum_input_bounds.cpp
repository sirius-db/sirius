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

// Plan-time magnitude bounds for decimal SUM inputs: resolve_decimal_sum_input_bounds reads
// base-table statistics through value-preserving operators and declines everything else. The
// operator shapes the tracer accepts are covered one by one in test_scan_column_origin.cpp.

#include "planner/decimal_sum_input_bounds.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using bounds_t = std::vector<std::optional<std::uint64_t>>;

duckdb::LogicalAggregate* find_aggregate(duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    return &op.Cast<duckdb::LogicalAggregate>();
  }
  for (auto& child : op.children) {
    if (auto* found = find_aggregate(*child)) { return found; }
  }
  return nullptr;
}

/// Bounds of the first aggregate in the optimized, binding-resolved plan of @p sql, resolved in
/// the connection's current transaction (the statistics callbacks need one).
bounds_t bounds_of(duckdb::Connection& con, std::string const& sql)
{
  auto& context   = *con.context;
  auto plan       = sirius::test::extract_logical_plan_sirius_order(context, sql).logical_plan;
  auto* aggregate = find_aggregate(*plan);
  REQUIRE(aggregate != nullptr);
  return sirius::planner::resolve_decimal_sum_input_bounds(context, *aggregate);
}

/// Resolves @p sql's bounds inside a transaction of its own.
bounds_t bounds_in_transaction(duckdb::Connection& con, std::string const& sql)
{
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION")->HasError());
  bounds_t result;
  try {
    result = bounds_of(con, sql);
  } catch (...) {
    con.Query("ROLLBACK");
    throw;
  }
  REQUIRE_FALSE(con.Query("COMMIT")->HasError());
  return result;
}

/// Plain DuckDB with Sirius disabled (as in test_context.cpp): the bounds are a logical-plan
/// property.
struct decimal_bounds_fixture {
  decimal_bounds_fixture()
  {
    if (char const* previous = getenv("SIRIUS_DISABLE")) { _previous_disable = previous; }
    setenv("SIRIUS_DISABLE", "1", 1);
    db  = std::make_unique<duckdb::DuckDB>(nullptr);
    con = std::make_unique<duckdb::Connection>(*db);
    // v in [1.00, 50.00] (unscaled 5000), w in [-6.00, 0.00] (600), x in [10.50, 31.50] (3150);
    // wide is DECIMAL128 on the GPU and never a candidate.
    run(
      "CREATE TABLE fact AS SELECT (i % 3)::INTEGER g, ((i % 50) + 1)::DECIMAL(15,2) v, "
      "(-(i % 7))::DECIMAL(7,2) w, (i % 1000)::DECIMAL(20,2) wide FROM range(3000) r(i);");
    run(
      "CREATE TABLE dim AS SELECT i::INTEGER g, ((i + 1) * 10.5)::DECIMAL(9,2) x FROM range(3) "
      "r(i);");
  }
  ~decimal_bounds_fixture()
  {
    con.reset();
    db.reset();
    if (_previous_disable) {
      setenv("SIRIUS_DISABLE", _previous_disable->c_str(), 1);
    } else {
      unsetenv("SIRIUS_DISABLE");
    }
  }
  void run(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE_FALSE(result->HasError());
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

 private:
  std::optional<std::string> _previous_disable;
};

}  // namespace

TEST_CASE_METHOD(decimal_bounds_fixture,
                 "decimal sum bounds - scan columns reaching the aggregate unchanged",
                 "[planner][aggregate][decimal_sum_overflow]")
{
  auto const bounds = bounds_in_transaction(
    *con, "SELECT g, sum(v), avg(v), sum(w), sum(wide), count(v), min(v) FROM fact GROUP BY g");
  REQUIRE(bounds.size() == 6);
  CHECK(bounds[0] == std::uint64_t{5000});  // sum(v)
  CHECK(bounds[1] == std::uint64_t{5000});  // avg(v) shares v's bound
  CHECK(bounds[2] == std::uint64_t{600});   // sum(w): the magnitude of -6.00
  CHECK_FALSE(bounds[3].has_value());       // sum(wide) already runs as DECIMAL128
  CHECK_FALSE(bounds[4].has_value());       // count cannot overflow
  CHECK_FALSE(bounds[5].has_value());       // min cannot overflow
}

TEST_CASE_METHOD(decimal_bounds_fixture,
                 "decimal sum bounds - ungrouped SUM and AVG over a filtered scan",
                 "[planner][aggregate][decimal_sum_overflow]")
{
  auto const bounds = bounds_in_transaction(*con, "SELECT sum(v), avg(w) FROM fact WHERE g = 1");
  REQUIRE(bounds.size() == 2);
  CHECK(bounds[0] == std::uint64_t{5000});
  CHECK(bounds[1] == std::uint64_t{600});
}

TEST_CASE_METHOD(decimal_bounds_fixture,
                 "decimal sum bounds - filters and both join sides forward the bound",
                 "[planner][aggregate][decimal_sum_overflow]")
{
  auto const bounds =
    bounds_in_transaction(*con,
                          "SELECT f.g, sum(f.v), sum(d.x) FROM fact f JOIN dim d ON f.g = d.g "
                          "WHERE f.w <= 0 AND d.x > 1 GROUP BY f.g");
  REQUIRE(bounds.size() == 2);
  CHECK(bounds[0] == std::uint64_t{5000});
  CHECK(bounds[1] == std::uint64_t{3150});
}

TEST_CASE_METHOD(decimal_bounds_fixture,
                 "decimal sum bounds - semi and anti joins and cross products forward the bound",
                 "[planner][aggregate][decimal_sum_overflow]")
{
  auto const semi = bounds_in_transaction(
    *con, "SELECT sum(f.v) FROM fact f WHERE EXISTS (SELECT 1 FROM dim d WHERE d.g = f.g)");
  REQUIRE(semi.size() == 1);
  CHECK(semi[0] == std::uint64_t{5000});
  auto const anti = bounds_in_transaction(
    *con, "SELECT sum(f.w) FROM fact f WHERE NOT EXISTS (SELECT 1 FROM dim d WHERE d.g = f.g + 7)");
  REQUIRE(anti.size() == 1);
  CHECK(anti[0] == std::uint64_t{600});
  auto const cross = bounds_in_transaction(*con, "SELECT sum(f.v), sum(d.x) FROM fact f, dim d");
  REQUIRE(cross.size() == 2);
  CHECK(cross[0] == std::uint64_t{5000});
  CHECK(cross[1] == std::uint64_t{3150});
}

TEST_CASE_METHOD(decimal_bounds_fixture,
                 "decimal sum bounds - computed inputs and unmodelled operators decline",
                 "[planner][aggregate][decimal_sum_overflow]")
{
  // v + v stays DECIMAL(16,2), a widening candidate, but is no scan column.
  auto const computed = bounds_in_transaction(*con, "SELECT sum(v + v) FROM fact");
  REQUIRE(computed.size() == 1);
  CHECK_FALSE(computed[0].has_value());
  // UNION ALL is not traced.
  auto const unioned = bounds_in_transaction(
    *con, "SELECT sum(v) FROM (SELECT v FROM fact UNION ALL SELECT v FROM fact)");
  REQUIRE(unioned.size() == 1);
  CHECK_FALSE(unioned[0].has_value());
}

TEST_CASE_METHOD(decimal_bounds_fixture,
                 "decimal sum bounds - transaction-local rows withdraw the statistics",
                 "[planner][aggregate][decimal_sum_overflow]")
{
  REQUIRE_FALSE(con->Query("BEGIN TRANSACTION")->HasError());
  run("INSERT INTO fact VALUES (0, 9999999999999.99, 0, 0)");
  auto const during = bounds_of(*con, "SELECT sum(v) FROM fact");
  REQUIRE_FALSE(con->Query("ROLLBACK")->HasError());
  REQUIRE(during.size() == 1);
  CHECK_FALSE(during[0].has_value());
  // Committed data is covered by the table's statistics again.
  auto const after = bounds_in_transaction(*con, "SELECT sum(v) FROM fact");
  REQUIRE(after.size() == 1);
  CHECK(after[0] == std::uint64_t{5000});
}
