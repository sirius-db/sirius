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

#include "catch.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/planner.hpp"
#include "helper/type_conversions.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_window.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_window.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <algorithm>
#include <map>
#include <set>

using sirius::aggregate_id;
using sirius::planner::plan_window;
using sirius::planner::window_kind;

namespace {

struct window_fixture {
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  window_fixture()
  {
    auto result = con.Query(
      "CREATE TABLE w(p INTEGER, q INTEGER, v INTEGER, f DOUBLE, "
      "d DECIMAL(18,2), wide DECIMAL(38,2), s VARCHAR)");
    REQUIRE_FALSE(result->HasError());
    con.BeginTransaction();
  }
  ~window_fixture() { con.Rollback(); }

  duckdb::unique_ptr<duckdb::LogicalOperator> bind(std::string const& sql, bool optimize = false)
  {
    duckdb::Parser parser(*con.context);
    parser.ParseQuery(sql);
    duckdb::Planner planner(*con.context);
    planner.CreatePlan(std::move(parser.statements[0]));
    auto plan = std::move(planner.plan);
    if (optimize) {
      duckdb::Optimizer optimizer(*planner.binder, *con.context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    return plan;
  }
};

duckdb::LogicalWindow* find_window_if_present(duckdb::LogicalOperator& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_WINDOW) {
    return &plan.Cast<duckdb::LogicalWindow>();
  }
  for (auto& child : plan.children) {
    if (auto* window = find_window_if_present(*child)) { return window; }
  }
  return nullptr;
}

duckdb::LogicalWindow& find_window(duckdb::LogicalOperator& plan)
{
  auto* window = find_window_if_present(plan);
  REQUIRE(window);
  return *window;
}

}  // namespace

TEST_CASE("whole-partition window planning preserves slots and bound types",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT v, sum(v) OVER (PARTITION BY p), "
    "avg(v) OVER (PARTITION BY p) FROM w");
  auto& window = find_window(*logical);
  auto plan    = plan_window(window);
  REQUIRE(plan.input_width == window.children[0]->types.size());
  REQUIRE(plan.output_types == window.types);
  REQUIRE(plan.prepared_types == window.children[0]->types);
  REQUIRE(plan.hidden_expressions.empty());
  REQUIRE(plan.partition_slots.size() == 1);
  REQUIRE(plan.expressions.size() == 2);
  CHECK(plan.expressions[0].function == aggregate_id::sum);
  CHECK(plan.expressions[1].function == aggregate_id::avg);
  CHECK(plan.expressions[0].argument_slot == plan.expressions[1].argument_slot);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::INTEGER);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::HUGEINT);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DOUBLE);
  CHECK(plan.expressions[0].output_slot == plan.input_width);
  CHECK(plan.expressions[1].output_slot == plan.input_width + 1);
  CHECK(plan.expressions[0].start == duckdb::WindowBoundary::UNBOUNDED_PRECEDING);
  CHECK(plan.expressions[0].end == duckdb::WindowBoundary::CURRENT_ROW_RANGE);
}

TEST_CASE("whole-partition window planning shares hidden expressions", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT sum(v+1) OVER (PARTITION BY p+1, q), "
    "avg(v+1) OVER (PARTITION BY q, p+1) FROM w");
  auto plan = plan_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 2);
  REQUIRE(plan.partition_slots.size() == 2);
  CHECK(plan.partition_slots[0] == plan.input_width);
  CHECK(plan.expressions[0].argument_slot == plan.input_width + 1);
  CHECK(plan.expressions[1].argument_slot == plan.input_width + 1);
  CHECK(plan.prepared_types.size() == plan.input_width + 2);
  CHECK(plan.output_types.size() == plan.input_width + 2);

  // An expression used as both a partition key and an argument gets one column.
  logical = fixture.bind("SELECT sum(p+1) OVER (PARTITION BY p+1) FROM w");
  plan    = plan_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 1);
  CHECK(plan.expressions[0].argument_slot == plan.partition_slots[0]);
}

TEST_CASE("whole-partition window planning admits frames that cover the whole partition",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const clause =
    GENERATE("",
             "PARTITION BY p",
             "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
             "PARTITION BY p RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
             // The order cannot change an aggregate over the whole partition, so none is sorted.
             "ORDER BY p ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
             "PARTITION BY q ORDER BY p RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING");
  auto logical = fixture.bind(std::string("SELECT sum(v) OVER (") + clause + ") FROM w");
  auto plan    = plan_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
}

TEST_CASE("whole-partition window planning rejects unsupported bound semantics",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression = GENERATE(
    "sum(v) OVER (ORDER BY p)",
    "sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
    "sum(v) OVER (ROWS BETWEEN 1 PRECEDING AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW)",
    "sum(v) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE GROUP)",
    "sum(v) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE TIES)",
    "sum(DISTINCT v) OVER ()",
    "sum(v) FILTER (WHERE v > 0) OVER ()",
    "sum(v ORDER BY p) OVER ()",
    "row_number() OVER ()",
    "lag(v) OVER ()",
    "sum(v::HUGEINT) OVER ()",
    "sum(v::DECIMAL(4,0)) OVER ()",
    "avg(v::DECIMAL(4,0)) OVER ()",
    "sum(v::UBIGINT) OVER ()",
    "sum(v) OVER (PARTITION BY [p])",
    "sum(v) OVER (PARTITION BY f)",
    "sum(v) OVER (PARTITION BY v::DECIMAL(4,0))",
    "sum(v) OVER (PARTITION BY s COLLATE nocase)",
    "sum(random()) OVER ()",
    "sum(atan2(f, 1)) OVER ()",
    "sum(v) OVER (), avg(v) OVER (PARTITION BY p)",
    "sum(v) OVER (PARTITION BY p, p), avg(v) OVER (PARTITION BY p, q)");
  INFO(expression);
  // The table is empty. Capability rejection must still occur before execution.
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w");
  REQUIRE_THROWS_AS(plan_window(find_window(*logical)), duckdb::NotImplementedException);
}

TEST_CASE("window planning accepts wide DECIMAL and HUGEINT partition keys",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression = GENERATE("sum(v) OVER (PARTITION BY d)",
                                   "sum(v) OVER (PARTITION BY wide)",
                                   "sum(v) OVER (PARTITION BY p::HUGEINT)",
                                   "rank() OVER (PARTITION BY d ORDER BY v)",
                                   "rank() OVER (PARTITION BY p ORDER BY v::HUGEINT)");
  INFO(expression);
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w");
  auto plan    = plan_window(find_window(*logical));
  CHECK(plan.partition_slots.size() == 1);
}

TEST_CASE("whole-partition count and extrema preserve bound result types", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT count(*) OVER (PARTITION BY p),count(v) OVER (PARTITION BY p),"
    "min(v) OVER (PARTITION BY p),max(d) OVER (PARTITION BY p),"
    "min(s) OVER (PARTITION BY p) FROM w");
  auto plan = plan_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 5);
  CHECK(plan.expressions[0].function == aggregate_id::count_star);
  CHECK(plan.expressions[1].function == aggregate_id::count);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::BIGINT);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::BIGINT);
  CHECK(plan.expressions[2].result_type == duckdb::LogicalType::INTEGER);
  CHECK(plan.expressions[3].result_type == duckdb::LogicalType::DECIMAL(18, 2));
  CHECK(plan.expressions[4].result_type == duckdb::LogicalType::VARCHAR);
  CHECK(plan.prepared_types[plan.expressions[0].argument_slot] == duckdb::LogicalType::INTEGER);
}

TEST_CASE("whole-partition window planning retains decimal widening", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind("SELECT sum(d) OVER (), avg(d) OVER (), sum(f) OVER () FROM w");
  auto plan    = plan_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 3);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(18, 2));
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::DECIMAL(38, 2));
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DOUBLE);
  CHECK(plan.expressions[2].result_type == duckdb::LogicalType::DOUBLE);
}

TEST_CASE("whole-partition window planner activates only admitted windows",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind("SELECT sum(v) OVER (ORDER BY p) FROM w");
  sirius::planner::sirius_physical_plan_generator generator(*fixture.con.context);
  REQUIRE_THROWS_WITH(generator.create_plan(std::move(logical)),
                      Catch::Matchers::ContainsSubstring("framed window without PARTITION BY"));
  logical       = fixture.bind("SELECT sum(v) OVER () FROM (VALUES (1),(2)) t(v)");
  auto physical = generator.create_plan(std::move(logical));
  auto find     = [](auto&& self, sirius::op::sirius_physical_operator& op) -> bool {
    if (op.type == sirius::op::SiriusPhysicalOperatorType::WINDOW) {
      CHECK(op.types.back() == sirius::logical_type::make(sirius::type_id::HUGEINT));
      REQUIRE(op.children.size() == 1);
      CHECK(op.children[0]->type == sirius::op::SiriusPhysicalOperatorType::CONCAT);
      CHECK(op.children[0]->children[0]->type == sirius::op::SiriusPhysicalOperatorType::PARTITION);
      return true;
    }
    for (auto& child : op.children) {
      if (self(self, *child)) { return true; }
    }
    return false;
  };
  REQUIRE(find(find, *physical));
}

TEST_CASE("window stages plan as stacked WINDOW operators", "[planner][window_plan]")
{
  window_fixture fixture;
  sirius::planner::sirius_physical_plan_generator generator(*fixture.con.context);
  auto count = [](auto&& self, sirius::op::sirius_physical_operator& op) -> std::size_t {
    std::size_t windows = op.type == sirius::op::SiriusPhysicalOperatorType::WINDOW ? 1 : 0;
    for (auto& child : op.children) {
      windows += self(self, *child);
    }
    return windows;
  };
  std::vector<std::pair<std::string, std::size_t>> const cases = {
    {"SELECT rank() OVER (PARTITION BY p ORDER BY v),sum(v) OVER (PARTITION BY p) FROM w", 2},
    {"SELECT rank() OVER (PARTITION BY p ORDER BY v),dense_rank() OVER (PARTITION BY p ORDER BY v) "
     "FROM w",
     1},
    {"SELECT rank() OVER (PARTITION BY p ORDER BY v),rank() OVER (PARTITION BY p ORDER BY q) "
     "FROM w",
     2},
    {"SELECT avg(d) OVER (PARTITION BY p,q),rank() OVER (PARTITION BY p ORDER BY v) FROM w", 2},
    {"SELECT sum(v) OVER (PARTITION BY p),sum(v) OVER (PARTITION BY q) FROM w", 2},
    {"SELECT sum(v) OVER (PARTITION BY p+1),rank() OVER (PARTITION BY q%2 ORDER BY v*2) FROM w", 2},
    {"SELECT sum(s) OVER (PARTITION BY p) FROM (SELECT p,sum(d) OVER (PARTITION BY q) s FROM w) t",
     2},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT "
     "ROW),max(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT "
     "ROW) FROM w",
     1},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT "
     "ROW),max(v) OVER (PARTITION BY p ORDER BY q DESC ROWS BETWEEN UNBOUNDED PRECEDING AND "
     "CURRENT ROW) FROM w",
     2},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT "
     "ROW),sum(v) OVER (PARTITION BY p) FROM w",
     2},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT "
     "ROW),rank() OVER (PARTITION BY p ORDER BY q) FROM w",
     2}};
  // The physical plan cannot scan the fixture's in-memory table, so read from a VALUES list.
  std::string const source = "FROM (VALUES (1,1,1,1.00::DECIMAL(18,2))) w(p,q,v,d)";
  for (auto const& [query, windows] : cases) {
    auto sql = query;
    for (auto at = sql.find("FROM w"); at != std::string::npos; at = sql.find("FROM w", at)) {
      sql.replace(at, 6, source);
    }
    INFO(sql);
    auto physical = generator.create_plan(fixture.bind(sql));
    CHECK(count(count, *physical) == windows);
  }
}

TEST_CASE("ranking window planning records the order keys and bound types",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT rank() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST),"
    "dense_rank() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST),"
    "row_number() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST) FROM w");
  auto& window = find_window(*logical);
  auto plan    = plan_window(window);
  REQUIRE(plan.expressions.empty());
  REQUIRE(plan.hidden_expressions.empty());
  REQUIRE(plan.partition_slots.size() == 1);
  REQUIRE(plan.rankings.size() == 3);
  CHECK(plan.rankings[0].function == sirius::op::ranking_function::rank);
  CHECK(plan.rankings[1].function == sirius::op::ranking_function::dense_rank);
  CHECK(plan.rankings[2].function == sirius::op::ranking_function::row_number);
  for (std::size_t i = 0; i < plan.rankings.size(); ++i) {
    CHECK(plan.rankings[i].result_type == duckdb::LogicalType::BIGINT);
    CHECK(plan.rankings[i].output_slot == plan.input_width + i);
  }
  REQUIRE(plan.order.columns.size() == 1);
  REQUIRE(plan.order.directions.size() == 1);
  REQUIRE(plan.order.nulls.size() == 1);
  CHECK(plan.order.directions[0] == cudf::order::DESCENDING);
  CHECK(plan.order.nulls[0] == sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                                              duckdb::OrderByNullType::NULLS_LAST));
}

TEST_CASE("ranking window planning evaluates order expressions once and keeps several keys",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT rank() OVER (PARTITION BY p ORDER BY v+1, q),"
    "dense_rank() OVER (PARTITION BY p ORDER BY v+1, q) FROM w");
  auto plan = plan_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 1);
  REQUIRE(plan.order.columns.size() == 2);
  CHECK(static_cast<std::size_t>(plan.order.columns[0]) == plan.input_width);
  CHECK(static_cast<std::size_t>(plan.order.columns[1]) < plan.input_width);
  CHECK(plan.order.directions ==
        std::vector<cudf::order>{cudf::order::ASCENDING, cudf::order::ASCENDING});
}

TEST_CASE("ranking window planning accepts a window without PARTITION BY", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT rank() OVER (ORDER BY v DESC),dense_rank() OVER (ORDER BY v DESC),"
    "row_number() OVER (ORDER BY v DESC) FROM w");
  auto plan = plan_window(find_window(*logical));
  CHECK(plan.partition_slots.empty());
  CHECK(plan.expressions.empty());
  CHECK(plan.rankings.size() == 3);
  CHECK(plan.order.columns.size() == 1);
}

TEST_CASE("ranking window planning keeps the direction and NULL placement of each key",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT rank() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST, q ASC NULLS FIRST) FROM w");
  auto plan = plan_window(find_window(*logical));
  REQUIRE(plan.order.columns.size() == 2);
  CHECK(plan.order.directions ==
        std::vector<cudf::order>{cudf::order::DESCENDING, cudf::order::ASCENDING});
  CHECK(plan.order.nulls ==
        std::vector<cudf::null_order>{
          sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                         duckdb::OrderByNullType::NULLS_LAST),
          sirius::op::to_cudf_null_order(duckdb::OrderType::ASCENDING,
                                         duckdb::OrderByNullType::NULLS_FIRST)});
}

TEST_CASE("ranking window planning rejects unsupported shapes", "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression =
    GENERATE("row_number() OVER (PARTITION BY p)",
             "rank() OVER (PARTITION BY p ORDER BY v::DECIMAL(4,0))",
             "rank() OVER (PARTITION BY p ORDER BY s COLLATE nocase)",
             "rank() OVER (PARTITION BY p ORDER BY [v])",
             "rank() OVER (PARTITION BY f ORDER BY v)",
             "percent_rank() OVER (PARTITION BY p ORDER BY v)",
             "ntile(2) OVER (PARTITION BY p ORDER BY v)",
             "sum(v) OVER (PARTITION BY p ORDER BY v), rank() OVER (PARTITION BY p ORDER BY v)",
             "sum(v) OVER (PARTITION BY p), rank() OVER (PARTITION BY p ORDER BY v)",
             "rank() OVER (PARTITION BY p ORDER BY v), sum(v) OVER (PARTITION BY p)");
  INFO(expression);
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w");
  REQUIRE_THROWS_AS(plan_window(find_window(*logical)), duckdb::NotImplementedException);
}

TEST_CASE("running aggregate planning records the order keys and bound types",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT sum(v) OVER (PARTITION BY p ORDER BY q DESC NULLS LAST ROWS BETWEEN UNBOUNDED "
    "PRECEDING AND CURRENT ROW),"
    "max(d) OVER (PARTITION BY p ORDER BY q DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND "
    "CURRENT ROW),"
    "count(v) OVER (PARTITION BY p ORDER BY q DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND "
    "CURRENT ROW),"
    "count(*) OVER (PARTITION BY p ORDER BY q DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND "
    "CURRENT ROW) FROM w");
  auto plan = plan_window(find_window(*logical));
  CHECK(plan.kind == window_kind::cumulative);
  CHECK(plan.rankings.empty());
  REQUIRE(plan.expressions.size() == 4);
  CHECK(plan.expressions[0].function == aggregate_id::sum);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::HUGEINT);
  CHECK(plan.expressions[1].function == aggregate_id::max);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DECIMAL(18, 2));
  CHECK(plan.expressions[2].function == aggregate_id::count);
  CHECK(plan.expressions[3].function == aggregate_id::count_star);
  CHECK(plan.partition_slots.size() == 1);
  REQUIRE(plan.order.columns.size() == 1);
  CHECK(plan.order.directions[0] == cudf::order::DESCENDING);
  CHECK(plan.order.nulls[0] == sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                                              duckdb::OrderByNullType::NULLS_LAST));
}

TEST_CASE("running aggregate planning accepts a window without PARTITION BY and an expression key",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT min(v) OVER (ORDER BY q+1, p ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),"
    "sum(d) OVER (ORDER BY q+1, p ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w");
  auto plan = plan_window(find_window(*logical));
  CHECK(plan.kind == window_kind::cumulative);
  CHECK(plan.partition_slots.empty());
  CHECK(plan.order.columns.size() == 2);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DECIMAL(38, 2));
}

TEST_CASE("running aggregate planning rejects unsupported shapes", "[planner][window_plan]")
{
  window_fixture fixture;
  std::string const frame = " ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW";
  auto const expression   = GENERATE_COPY(
    // Without PARTITION BY, a frame or function the running evaluator does not take has no other
    // GPU evaluator.
    std::string("sum(v) OVER (ORDER BY q)"),
    std::string("sum(v) OVER (ORDER BY q ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
    "avg(v) OVER (ORDER BY q" + frame + ")",
    "max(s) OVER (ORDER BY q" + frame + ")",
    "sum(v::UBIGINT) OVER (PARTITION BY p ORDER BY q" + frame + ")",
    "sum(DISTINCT v) OVER (PARTITION BY p ORDER BY q" + frame + ")",
    "sum(v) FILTER (WHERE v > 0) OVER (PARTITION BY p ORDER BY q" + frame + ")",
    "sum(v) OVER (PARTITION BY p ORDER BY q" + frame + " EXCLUDE CURRENT ROW)",
    "sum(v) OVER (PARTITION BY p ORDER BY s COLLATE nocase" + frame + ")",
    "sum(v) OVER (PARTITION BY p ORDER BY [v]" + frame + ")",
    "first_value(v) OVER (PARTITION BY p ORDER BY q" + frame + ")",
    "sum(v) OVER (PARTITION BY p ORDER BY q" + frame + "), rank() OVER (PARTITION BY p ORDER BY q)",
    "sum(v) OVER (PARTITION BY p ORDER BY q" + frame + "), sum(v) OVER (PARTITION BY p)",
    "sum(v) OVER (PARTITION BY p ORDER BY q" + frame + "), sum(v) OVER (PARTITION BY p ORDER BY v" +
      frame + ")",
    "sum(v) OVER (PARTITION BY p ORDER BY q" + frame + "), sum(v) OVER (PARTITION BY q ORDER BY q" +
      frame + ")");
  INFO(expression);
  auto logical = fixture.bind("SELECT " + expression + " FROM w");
  REQUIRE_THROWS_AS(plan_window(find_window(*logical)), duckdb::NotImplementedException);
}

TEST_CASE("ordered aggregates the running evaluator does not take are framed",
          "[planner][window_plan]")
{
  window_fixture fixture;
  std::string const frame = " ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW";
  auto const expression   = GENERATE_COPY(
    // The default frame is a RANGE frame, where peers share a result.
    std::string("sum(v) OVER (PARTITION BY p ORDER BY q)"),
    std::string("sum(v) OVER (PARTITION BY p ORDER BY q RANGE BETWEEN UNBOUNDED PRECEDING AND "
                  "CURRENT ROW)"),
    std::string("sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
    std::string("sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN UNBOUNDED PRECEDING AND 1 "
                  "FOLLOWING)"),
    std::string("sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN CURRENT ROW AND UNBOUNDED "
                  "FOLLOWING)"),
    // The running evaluator takes neither AVG nor a string MIN or MAX.
    "avg(v) OVER (PARTITION BY p ORDER BY q" + frame + ")",
    "max(s) OVER (PARTITION BY p ORDER BY q" + frame + ")");
  INFO(expression);
  auto logical = fixture.bind("SELECT " + expression + " FROM w", true);
  auto plan    = plan_window(find_window(*logical));
  CHECK(plan.kind == window_kind::framed);
  CHECK(plan.expressions.empty());
  CHECK(plan.framed.size() == 1);
  CHECK(plan.framed_plan);
}

TEST_CASE("whole-partition window planning rejects invalid mappings and null treatment",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical     = fixture.bind("SELECT sum(v) OVER (PARTITION BY p) FROM w");
  auto& window     = find_window(*logical);
  auto& expression = window.expressions[0]->Cast<duckdb::BoundWindowExpression>();
  SECTION("unsupported null treatment") { expression.IgnoreNullsMutable() = true; }
  SECTION("incorrect visible result type") { window.types.back() = duckdb::LogicalType::DOUBLE; }
  SECTION("incorrect input mapping") { window.types[0] = duckdb::LogicalType::VARCHAR; }
  SECTION("argument outside input schema")
  {
    expression.GetChildrenMutable()[0]->Cast<duckdb::BoundReferenceExpression>().IndexMutable() =
      window.children[0]->types.size();
  }
  REQUIRE_THROWS_AS(plan_window(window), duckdb::NotImplementedException);
}

TEST_CASE("whole-partition planning accepts statistics-optimized SUM", "[planner][window_plan]")
{
  window_fixture fixture;
  REQUIRE_FALSE(fixture.con.Query("INSERT INTO w(p,v) VALUES (1,10),(1,20)")->HasError());
  auto settings = fixture.con.Query("SELECT current_setting('disabled_optimizers')");
  REQUIRE_FALSE(settings->HasError());
  auto const current = settings->Collection().GetValue(0, 0).ToString();
  REQUIRE_FALSE(fixture.con
                  .Query("SET disabled_optimizers='" + current + (current.empty() ? "" : ",") +
                         "window_self_join'")
                  ->HasError());
  auto logical = fixture.bind("SELECT p,sum(v) OVER (PARTITION BY p) FROM w", true);
  auto plan    = plan_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::HUGEINT);
}

TEST_CASE("whole-partition window planning checks optimized TPC-DS shapes",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const aggregate = GENERATE("sum", "avg");
  // The grouped SUM widens decimal sales before the window consumes them.
  auto const query =
    std::string("SELECT ") + aggregate + "(sum(d)) OVER (PARTITION BY p) FROM w GROUP BY p, q";
  auto logical = fixture.bind(query);
  auto plan    = plan_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(38, 2));

  auto optimized = fixture.bind(query, true);
  CHECK(optimized->types == logical->types);
  if (auto* window = find_window_if_present(*optimized)) {
    auto optimized_plan = plan_window(*window);
    REQUIRE(optimized_plan.expressions.size() == 1);
    CHECK(optimized_plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(38, 2));
  } else {
    // WINDOW_SELF_JOIN can replace this entire window with aggregation + join.
    // Such a plan supplies no evidence of GPU window execution.
    SUCCEED("DuckDB eliminated the window from this optimized TPC-DS shape");
  }
}

TEST_CASE("window input builder prepares expressions and materializes complete CONCAT wiring",
          "[planner][window_plan][window_input_plan]")
{
  window_fixture fixture;
  auto const partitioned = GENERATE(false, true);
  auto logical = fixture.bind(partitioned ? "SELECT sum(v+1) OVER (PARTITION BY p+1) FROM w"
                                          : "SELECT sum(v+1) OVER () FROM w");
  auto bound   = plan_window(find_window(*logical));
  auto const input_types = sirius::from_duckdb_vec(find_window(*logical).children[0]->types);
  auto child    = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(input_types, 10);
  auto assembly = sirius::planner::build_window_input(std::move(child), bound, 10);
  auto* input   = &assembly->Cast<sirius::op::sirius_physical_concat>();
  REQUIRE(assembly->types == sirius::from_duckdb_vec(bound.prepared_types));
  REQUIRE(assembly->children.size() == 1);
  auto& exchange = assembly->children[0]->Cast<sirius::op::sirius_physical_partition>();
  std::vector<int> keys(bound.partition_slots.begin(), bound.partition_slots.end());
  CHECK(exchange.partition_keys() == keys);
  CHECK(exchange.input_barrier_for(*exchange.children[0]) == sirius::op::MemoryBarrierType::FULL);
  REQUIRE(exchange.children.size() == 1);
  auto& projection = exchange.children[0]->Cast<sirius::op::sirius_physical_projection>();
  CHECK(projection.select_list.size() == bound.prepared_types.size());

  auto root =
    duckdb::make_uniq<sirius::op::sirius_physical_passthrough_sink>(assembly->types, 10, "default");
  root->children.push_back(std::move(assembly));
  sirius::planner::sirius_physical_plan_generator::set_parent_ops(*root, nullptr);
  sirius::pipeline::pipeline_build_context context{nullptr};
  sirius::pipeline::sirius_pipeline_build_state state;
  auto meta = std::make_shared<sirius::pipeline::sirius_meta_pipeline>(context, state, nullptr);
  meta->build(*root);
  meta->ready();
  sirius::pipeline::sirius_pipeline_converter converter(context);
  auto converted = converter.convert(*meta);
  sirius::pipeline::assign_operator_ids(converted.scheduled_pipelines);
  cucascade::shared_data_repository_manager repositories;
  sirius::pipeline::materialize_repository_wiring(converted.repository_wirings, repositories);
  auto* port = input->get_port("default");
  REQUIRE(port);
  // CONCAT keeps its ordinary streaming port, but concat_all gates claims on completion.
  CHECK(port->type == sirius::op::MemoryBarrierType::PARTIAL);
  auto hint = input->get_next_task_hint();
  REQUIRE(hint);
  CHECK(hint->hint == sirius::op::TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(port->src_pipeline);
  CHECK(port->src_pipeline->get_sink().get() == &exchange);
  CHECK(port->dest_pipeline->get_source().get() == input);
  CHECK(repositories.get_repository(input->get_operator_id(), "default").get() == port->repo);
}

TEST_CASE("window input builder assembles fragmented keyed and unkeyed input",
          "[planner][window_input_plan][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  window_fixture fixture;
  auto const keyed = GENERATE(false, true);
  auto logical = fixture.bind(keyed ? "SELECT sum(p) OVER (PARTITION BY p) FROM (SELECT p FROM w)"
                                    : "SELECT sum(p) OVER () FROM (SELECT p FROM w)");
  auto& window = find_window(*logical);
  auto bound   = plan_window(window);
  REQUIRE(bound.input_width == 1);
  auto types    = sirius::from_duckdb_vec(window.children[0]->types);
  auto first    = make_numeric_batch<int32_t>(space, {1, 2, 1}, cudf::type_id::INT32);
  auto second   = make_numeric_batch<int32_t>(space, {2, 3, 1}, cudf::type_id::INT32);
  auto target   = first->to_read_only().get_data()->get_size_in_bytes();
  auto assembly = sirius::planner::build_window_input(
    duckdb::make_uniq<sirius_physical_dummy_scan>(types, 6), bound, 6, target);
  auto& concat = assembly->Cast<sirius_physical_concat>();
  auto& router = concat.children[0]->Cast<sirius_physical_partition>();
  cucascade::shared_data_repository input, buckets;
  auto attach = [](sirius_physical_operator& op, cucascade::shared_data_repository& repo) {
    auto port  = std::make_unique<sirius_physical_operator::port>();
    port->repo = &repo;
    port->type = MemoryBarrierType::FULL;
    op.add_port("default", std::move(port));
  };
  attach(router, input);
  attach(concat, buckets);
  router.add_next_port_after_sink({&concat, "default"});
  input.add_data_batch(first);
  input.add_data_batch(second);
  while (auto data = router.get_next_task_input_data()) {
    data->prepare_for_processing(&space, default_stream());
    auto result = router.execute(*data, default_stream());
    router.sink(*result, default_stream());
  }
  CHECK(buckets.num_partitions() == (keyed ? 2 : 1));
  std::vector<int32_t> values;
  std::set<std::size_t> claimed;
  std::map<int32_t, std::size_t> key_buckets;
  while (auto data = concat.get_next_task_input_data()) {
    auto const bucket = *dynamic_cast<partitioned_operator_data&>(*data).get_partition_idx();
    CHECK(claimed.insert(bucket).second);
    data->prepare_for_processing(&space, default_stream());
    auto result  = concat.execute(*data, default_stream());
    auto batches = dynamic_cast<pipelineable_operator_data&>(*result).get_read_only_batches();
    REQUIRE(batches.size() == 1);
    for (auto value :
         copy_column_to_host<int32_t>(sirius::get_cudf_table_view(batches[0]).column(0))) {
      auto position = key_buckets.emplace(value, bucket).first;
      CHECK(position->second == bucket);
      values.push_back(value);
    }
  }
  std::sort(values.begin(), values.end());
  CHECK(values == std::vector<int32_t>{1, 1, 1, 2, 2, 3});
}

TEST_CASE("window adapter restores narrow inputs and removes hidden slots", "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  using sirius::logical_type;
  using sirius::type_id;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  // Both columns are compressed INTEGERs. Slot 1 is a hidden argument, not visible payload.
  auto batch = make_two_column_batch<int8_t, int8_t>(
    space, {1, 2, 1, 2}, {10, 20, 30, 40}, cudf::type_id::INT8, std::nullopt, cudf::type_id::INT8);
  auto stream        = space.acquire_stream();
  auto const spilled = GENERATE(false, true);
  if (spilled) {
    auto const* host = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
    auto writable    = batch->to_mutable();
    writable.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host, stream);
    stream.sync();
  }
  sirius_physical_window window(
    {logical_type::make(type_id::INTEGER), logical_type::make(type_id::HUGEINT)},
    {logical_type::make(type_id::INTEGER), logical_type::make(type_id::INTEGER)},
    1,
    plan_whole_partition_window(
      {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
      {0},
      {{aggregate_id::sum, 1}}),
    4);
  pipelineable_operator_data input({batch});
  input.prepare_for_processing(&space, stream);
  auto result = window.execute(input, stream);
  stream.sync();
  auto batches = dynamic_cast<pipelineable_operator_data&>(*result).get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto table = sirius::get_cudf_table_view(batches[0]);
  REQUIRE(table.num_columns() == 2);
  CHECK(table.column(0).type().id() == cudf::type_id::INT8);
  CHECK(table.column(1).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  CHECK(copy_column_to_host<__int128_t>(table.column(1)) ==
        std::vector<__int128_t>{40, 60, 40, 60});
}

TEST_CASE("ranking window adapter ranks narrow inputs without restoring them",
          "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  using sirius::logical_type;
  using sirius::type_id;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  // The partition key and the order key are compressed INTEGERs.
  auto batch = make_two_column_batch<int8_t, int8_t>(
    space, {1, 2, 1, 2}, {30, 10, 20, 40}, cudf::type_id::INT8, std::nullopt, cudf::type_id::INT8);
  auto stream = space.acquire_stream();
  sirius_physical_window window(
    {logical_type::make(type_id::INTEGER),
     logical_type::make(type_id::INTEGER),
     logical_type::make(type_id::BIGINT)},
    {logical_type::make(type_id::INTEGER), logical_type::make(type_id::INTEGER)},
    2,
    {0},
    window_order{{1}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE}},
    {ranking_function::rank},
    4);
  pipelineable_operator_data input({batch});
  input.prepare_for_processing(&space, stream);
  auto result = window.execute(input, stream);
  stream.sync();
  auto batches = dynamic_cast<pipelineable_operator_data&>(*result).get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto table = sirius::get_cudf_table_view(batches[0]);
  REQUIRE(table.num_columns() == 3);
  // The payload keeps its carrier, and the ranks follow the order of the narrow keys.
  CHECK(table.column(0).type().id() == cudf::type_id::INT8);
  CHECK(table.column(1).type().id() == cudf::type_id::INT8);
  CHECK(table.column(2).type().id() == cudf::type_id::INT64);
  CHECK(copy_column_to_host<int64_t>(table.column(2)) == std::vector<int64_t>{2, 1, 1, 2});
}

TEST_CASE("window source drains assembled batches across repository partition gaps",
          "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  auto outputs = types;
  outputs.push_back(sirius::from_duckdb(duckdb::LogicalType::HUGEINT));
  sirius_physical_window window(
    outputs,
    types,
    1,
    plan_whole_partition_window(
      {cudf::data_type{cudf::type_id::INT32}}, {0}, {{aggregate_id::sum, 0}}),
    3);
  cucascade::shared_data_repository repo;
  repo.set_num_partitions(5);
  auto first  = make_numeric_batch<int32_t>(space, {1, 2}, cudf::type_id::INT32);
  auto second = make_numeric_batch<int32_t>(space, {3}, cudf::type_id::INT32);
  repo.add_data_batch(first, 1);
  repo.add_data_batch(second, 4);
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->repo = &repo;
  port->type = MemoryBarrierType::FULL;
  window.add_port("default", std::move(port));
  for (auto const& expected : {first, second}) {
    auto input = window.get_next_task_input_data();
    REQUIRE(input);
    auto const& batches = dynamic_cast<pipelineable_operator_data&>(*input).get_data_batches();
    REQUIRE(batches.size() == 1);
    CHECK(batches[0] == expected);
  }
  CHECK_FALSE(window.get_next_task_input_data());
  CHECK(repo.total_size() == 0);
}

TEST_CASE("ranking window adapter rejects invalid construction", "[window_adapter]")
{
  using namespace sirius::op;
  using sirius::logical_type;
  using sirius::type_id;
  auto const integer = logical_type::make(type_id::INTEGER);
  auto const bigint  = logical_type::make(type_id::BIGINT);
  // Three prepared columns, and a ranking window that appends one BIGINT result to them.
  duckdb::vector<logical_type> const prepared{integer, integer, integer};
  duckdb::vector<logical_type> const output{integer, integer, integer, bigint};
  window_order const order{{1}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE}};
  std::vector<ranking_function> const rank{ranking_function::rank};
  auto build = [&](duckdb::vector<logical_type> const& outputs,
                   std::size_t input_width,
                   std::vector<cudf::size_type> partitions,
                   window_order const& keys,
                   std::vector<ranking_function> const& functions) {
    return sirius_physical_window(
      outputs, prepared, input_width, std::move(partitions), keys, functions, 1);
  };
  CHECK_NOTHROW(build(output, 3, {0}, order, rank));
  SECTION("no functions")
  {
    CHECK_THROWS_AS(build(output, 3, {0}, order, {}), std::invalid_argument);
  }
  SECTION("no partition columns, for a window without PARTITION BY")
  {
    CHECK_NOTHROW(build(output, 3, {}, order, rank));
  }
  SECTION("no order keys")
  {
    CHECK_THROWS_AS(build(output, 3, {0}, window_order{}, rank), std::invalid_argument);
  }
  SECTION("a partition column outside the prepared schema")
  {
    CHECK_THROWS_AS(build(output, 3, {3}, order, rank), std::invalid_argument);
    CHECK_THROWS_AS(build(output, 3, {-1}, order, rank), std::invalid_argument);
  }
  SECTION("an order column outside the prepared schema")
  {
    CHECK_THROWS_AS(build(output, 3, {0}, window_order{{3}, order.directions, order.nulls}, rank),
                    std::invalid_argument);
  }
  SECTION("a direction or NULL placement missing for an order key")
  {
    CHECK_THROWS_AS(build(output, 3, {0}, window_order{{1}, {}, order.nulls}, rank),
                    std::invalid_argument);
    CHECK_THROWS_AS(build(output, 3, {0}, window_order{{1}, order.directions, {}}, rank),
                    std::invalid_argument);
  }
  SECTION("an output schema that does not match the prepared input")
  {
    CHECK_THROWS_AS(build(output, 2, {0}, order, rank), std::invalid_argument);
    CHECK_THROWS_AS(build(output, 4, {0}, order, rank), std::invalid_argument);
  }
}

TEST_CASE("framed window planning records the frame of each aggregate", "[planner][window_plan]")
{
  using namespace sirius::op;
  window_fixture fixture;
  struct expected_frame {
    std::string clause;
    window_frame frame;
  };
  auto const expected = GENERATE(
    expected_frame{"", default_window_frame()},
    expected_frame{"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING",
                   {frame_unit::rows, frame_unbounded{}, frame_row_offset{1, false}}},
    expected_frame{"ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING",
                   {frame_unit::rows, frame_row_offset{2, true}, frame_row_offset{1, false}}},
    expected_frame{"ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
                   {frame_unit::rows, frame_current_row{}, frame_unbounded{}}},
    expected_frame{"ROWS BETWEEN 1 + 1 PRECEDING AND CURRENT ROW",
                   {frame_unit::rows, frame_row_offset{2, true}, frame_current_row{}}},
    expected_frame{"RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
                   {frame_unit::range, frame_current_row{}, frame_unbounded{}}});
  INFO(expected.clause);
  // Offsets are BIGINT constants only after the optimizer folds the binder's cast.
  auto logical = fixture.bind("SELECT sum(v) OVER (PARTITION BY p ORDER BY q DESC NULLS FIRST, v " +
                                expected.clause + ") FROM w",
                              true);
  auto plan    = plan_window(find_window(*logical));
  CHECK(plan.expressions.empty());
  CHECK(plan.rankings.empty());
  REQUIRE(plan.framed.size() == 1);
  CHECK(plan.framed[0].function == aggregate_id::sum);
  CHECK(plan.framed[0].frame == expected.frame);
  CHECK(plan.framed[0].result_type == duckdb::LogicalType::HUGEINT);
  CHECK(plan.framed[0].output_slot == plan.input_width);
  REQUIRE(plan.partition_slots.size() == 1);
  REQUIRE(plan.order.columns.size() == 2);
  CHECK(plan.order.directions ==
        std::vector<cudf::order>{cudf::order::DESCENDING, cudf::order::ASCENDING});
  CHECK(plan.order.nulls[0] ==
        to_cudf_null_order(duckdb::OrderType::DESCENDING, duckdb::OrderByNullType::NULLS_FIRST));
  REQUIRE(plan.framed_plan);
  CHECK(plan.framed_plan->group_columns ==
        std::vector<cudf::size_type>{static_cast<cudf::size_type>(plan.partition_slots[0])});
}

TEST_CASE("framed window planning shares a stage across frames", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT count(*) OVER (PARTITION BY p ORDER BY v ROWS 1 PRECEDING),"
    "max(d) OVER (PARTITION BY p ORDER BY v),"
    "avg(v+1) OVER (PARTITION BY p ORDER BY v ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING) FROM w",
    true);
  auto plan = plan_window(find_window(*logical));
  REQUIRE(plan.framed.size() == 3);
  CHECK(plan.framed[0].function == aggregate_id::count_star);
  CHECK(plan.framed[1].function == aggregate_id::max);
  CHECK(plan.framed[1].result_type == plan.framed[1].argument_type);
  CHECK(plan.framed[2].function == aggregate_id::avg);
  CHECK(plan.framed[2].argument_slot >= plan.input_width);
  CHECK(plan.framed[0].frame.unit == sirius::op::frame_unit::rows);
  CHECK(plan.framed[1].frame == sirius::op::default_window_frame());
  REQUIRE(plan.framed_plan);
  CHECK(plan.framed_plan->outputs.size() == 3);
}

TEST_CASE("framed window planning rejects unsupported frames", "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression =
    GENERATE("sum(v) OVER (ORDER BY q ROWS 1 PRECEDING)",
             "sum(v) OVER (ORDER BY q)",
             "sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN v PRECEDING AND CURRENT ROW)",
             "sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN -1 PRECEDING AND CURRENT ROW)",
             "sum(v) OVER (PARTITION BY p ORDER BY q GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW)",
             "sum(v) OVER (PARTITION BY p ORDER BY q ROWS BETWEEN 1 PRECEDING AND CURRENT ROW "
             "EXCLUDE CURRENT ROW)",
             "sum(v) OVER (PARTITION BY p ORDER BY q RANGE BETWEEN 1 PRECEDING AND CURRENT ROW)",
             "sum(v) OVER (PARTITION BY p ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)",
             "sum(DISTINCT v) OVER (PARTITION BY p ORDER BY q)",
             "sum(v) OVER (PARTITION BY p ORDER BY q ROWS 1 PRECEDING), "
             "rank() OVER (PARTITION BY p ORDER BY q)");
  INFO(expression);
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w", true);
  REQUIRE_THROWS_AS(plan_window(find_window(*logical)), duckdb::NotImplementedException);
}

TEST_CASE("framed windows split into stages by ORDER BY and kind", "[planner][window_plan]")
{
  window_fixture fixture;
  sirius::planner::sirius_physical_plan_generator generator(*fixture.con.context);
  auto count = [](auto&& self, sirius::op::sirius_physical_operator& op) -> std::size_t {
    std::size_t windows = op.type == sirius::op::SiriusPhysicalOperatorType::WINDOW ? 1 : 0;
    for (auto& child : op.children) {
      windows += self(self, *child);
    }
    return windows;
  };
  std::vector<std::pair<std::string, std::size_t>> const cases = {
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS 1 PRECEDING),"
     "max(v) OVER (PARTITION BY p ORDER BY q) FROM w",
     1},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q),sum(v) OVER (PARTITION BY p ORDER BY v) "
     "FROM w",
     2},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q),rank() OVER (PARTITION BY p ORDER BY q) "
     "FROM w",
     2},
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q),sum(v) OVER (PARTITION BY p) FROM w", 2},
    // A running aggregate is a stage of its own kind, even with the same ORDER BY.
    {"SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS UNBOUNDED PRECEDING),"
     "sum(v) OVER (PARTITION BY p ORDER BY q ROWS 1 PRECEDING) FROM w",
     2}};
  std::string const source = "FROM (VALUES (1,1,1)) w(p,q,v)";
  for (auto const& [query, windows] : cases) {
    auto sql = query;
    for (auto at = sql.find("FROM w"); at != std::string::npos; at = sql.find("FROM w", at)) {
      sql.replace(at, 6, source);
    }
    INFO(sql);
    // Offsets are BIGINT constants only after the optimizer folds them.
    auto physical = generator.create_plan(fixture.bind(sql, true));
    CHECK(count(count, *physical) == windows);
  }
}

TEST_CASE("framed window adapter restores arguments and leaves keys narrow",
          "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  using sirius::logical_type;
  using sirius::type_id;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  // Slot 0 is the partition key, slot 1 the order key and slot 2 a hidden argument, all
  // compressed INTEGERs, sorted on (slot 0, slot 1).
  auto mr     = get_resource_ref(space);
  auto stream = space.acquire_stream();
  auto make   = [&](std::vector<int8_t> const& values) {
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT8},
                                            static_cast<cudf::size_type>(values.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);
    REQUIRE(cudaMemcpy(column->mutable_view().data<int8_t>(),
                       values.data(),
                       values.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return column;
  };
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make({1, 1, 1, 2, 2}));
  columns.push_back(make({1, 2, 3, 1, 2}));
  columns.push_back(make({10, 20, 30, 40, 50}));
  auto batch         = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                       space,
                                       stream,
                                       sirius::telemetry::batch_telemetry_info{});
  auto const integer = cudf::data_type{cudf::type_id::INT32};
  window_order const order{{1}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE}};
  auto plan = plan_framed_window(
    {integer, integer, integer},
    {0},
    order,
    {{aggregate_id::sum, 2, {frame_unit::rows, frame_row_offset{2, true}, frame_current_row{}}},
     {aggregate_id::count_star, 2, default_window_frame()}});
  sirius_physical_window window({logical_type::make(type_id::INTEGER),
                                 logical_type::make(type_id::INTEGER),
                                 logical_type::make(type_id::HUGEINT),
                                 logical_type::make(type_id::BIGINT)},
                                {logical_type::make(type_id::INTEGER),
                                 logical_type::make(type_id::INTEGER),
                                 logical_type::make(type_id::INTEGER)},
                                2,
                                plan,
                                5);
  CHECK(window.input_exchange() == window_exchange::sorted);
  CHECK(window.order().columns == order.columns);
  pipelineable_operator_data input({batch});
  input.prepare_for_processing(&space, stream);
  auto result = window.execute(input, stream);
  stream.sync();
  auto batches = dynamic_cast<pipelineable_operator_data&>(*result).get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto table = sirius::get_cudf_table_view(batches[0]);
  REQUIRE(table.num_columns() == 4);
  CHECK(table.column(0).type().id() == cudf::type_id::INT8);
  CHECK(table.column(1).type().id() == cudf::type_id::INT8);
  CHECK(table.column(2).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  // ROWS 2 PRECEDING: the frame preceding counts the current row, so it covers 2 rows back.
  CHECK(copy_column_to_host<__int128_t>(table.column(2)) ==
        std::vector<__int128_t>{10, 30, 60, 40, 90});
  CHECK(copy_column_to_host<int64_t>(table.column(3)) == std::vector<int64_t>{1, 2, 3, 1, 2});
}

TEST_CASE("framed window source drains merged ranges across gaps as they arrive",
          "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  auto manager       = sirius::test::operator_utils::initialize_memory_manager();
  auto& space        = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto const integer = cudf::data_type{cudf::type_id::INT32};
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  auto outputs = types;
  outputs.push_back(sirius::from_duckdb(duckdb::LogicalType::BIGINT));
  sirius_physical_window window(
    outputs,
    types,
    1,
    plan_framed_window(
      {integer}, {0}, window_order{}, {{aggregate_id::count_star, 0, default_window_frame()}}),
    3);
  // A merged range from MERGE_SORT is complete, so the window streams it.
  sirius_physical_merge_sort merge(types, {}, {0}, 3);
  CHECK(window.input_barrier_for(merge) == MemoryBarrierType::PIPELINE);
  cucascade::shared_data_repository repo;
  repo.set_num_partitions(5);
  auto first  = make_numeric_batch<int32_t>(space, {1, 1}, cudf::type_id::INT32);
  auto second = make_numeric_batch<int32_t>(space, {3}, cudf::type_id::INT32);
  repo.add_data_batch(first, 1);
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->repo = &repo;
  port->type = MemoryBarrierType::PIPELINE;
  window.add_port("default", std::move(port));
  auto take = [&] {
    auto input = window.get_next_task_input_data();
    REQUIRE(input);
    auto const& batches = dynamic_cast<pipelineable_operator_data&>(*input).get_data_batches();
    REQUIRE(batches.size() == 1);
    return batches[0];
  };
  CHECK(take() == first);
  CHECK_FALSE(window.get_next_task_input_data());
  repo.add_data_batch(second, 4);
  CHECK(take() == second);
  CHECK_FALSE(window.get_next_task_input_data());
}

TEST_CASE("framed window adapter rejects invalid construction", "[window_adapter]")
{
  using namespace sirius::op;
  using sirius::logical_type;
  using sirius::type_id;
  auto const integer = logical_type::make(type_id::INTEGER);
  auto const bigint  = logical_type::make(type_id::BIGINT);
  auto const native  = cudf::data_type{cudf::type_id::INT32};
  duckdb::vector<logical_type> const prepared{integer, integer};
  duckdb::vector<logical_type> const output{integer, integer, bigint};
  auto plan = [&](std::vector<cudf::size_type> groups) {
    return plan_framed_window(
      {native, native},
      groups,
      window_order{{1}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE}},
      {{aggregate_id::count, 1, default_window_frame()}});
  };
  CHECK_NOTHROW(sirius_physical_window(output, prepared, 2, plan({0}), 1));
  SECTION("no partition columns")
  {
    CHECK_THROWS_AS(sirius_physical_window(output, prepared, 2, plan({}), 1),
                    std::invalid_argument);
  }
  SECTION("an output schema that does not match the prepared input")
  {
    CHECK_THROWS_AS(sirius_physical_window(output, prepared, 1, plan({0}), 1),
                    std::invalid_argument);
  }
  SECTION("a plan built for a wider schema")
  {
    auto wide = plan_framed_window({native, native, native},
                                   {2},
                                   window_order{},
                                   {{aggregate_id::count, 1, default_window_frame()}});
    CHECK_THROWS_AS(sirius_physical_window(output, prepared, 2, wide, 1), std::invalid_argument);
  }
}

TEST_CASE("framed window offsets must be folded to constants", "[planner][window_plan]")
{
  window_fixture fixture;
  auto const sql = "SELECT sum(v) OVER (PARTITION BY p ORDER BY q ROWS 2 PRECEDING) FROM w";
  // The binder casts the literal to BIGINT. Only DuckDB's optimizer folds the cast away.
  auto bound = fixture.bind(sql);
  CHECK_THROWS_WITH(plan_window(find_window(*bound)),
                    Catch::Matchers::ContainsSubstring("non-constant frame offset"));
  auto optimized = fixture.bind(sql, true);
  auto plan      = plan_window(find_window(*optimized));
  REQUIRE(plan.framed.size() == 1);
  CHECK(plan.framed[0].frame.start ==
        sirius::op::frame_bound{sirius::op::frame_row_offset{2, true}});
}
