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

/**
 * @file test_plan_aggregate_filter.cpp
 * @brief Plan-time handling of the aggregate FILTER clause: supported filtered aggregates
 *        convert with their inputs mask-wrapped as CASE projections below the aggregate (and
 *        `count(*) FILTER` lowered to `count(mask)`), while DISTINCT aggregates other than
 *        COUNT(DISTINCT), nested inputs, and `first` with FILTER throw
 *        duckdb::NotImplementedException so the query falls back to CPU.
 */

#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/case_expr.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/aggregation.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_plan_aggregate_filter_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);
    _path = tmpl;
  }

  ~scoped_temp_db_path()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
      std::remove((_path + ".wal").c_str());
    }
  }

  scoped_temp_db_path(const scoped_temp_db_path&)            = delete;
  scoped_temp_db_path& operator=(const scoped_temp_db_path&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

/// Generate a Sirius physical plan from a SQL query string. Throws on any failure (after
/// rolling back and restoring the optimizer settings) so a planner regression fails the
/// test instead of silently skipping it -- and so REQUIRE_THROWS_AS can assert the
/// NotImplementedException the FILTER gate raises for rejected shapes.
duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(
  Connection& con,
  const std::string& query,
  std::function<void(LogicalOperator&)> customize = {},
  bool enable_common_subexpressions               = false)
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  if (enable_common_subexpressions) {
    disabled.erase(OptimizerType::COMMON_SUBEXPRESSIONS);
  } else {
    disabled.insert(OptimizerType::COMMON_SUBEXPRESSIONS);
  }
  disabled.insert(OptimizerType::COMMON_AGGREGATE);
  disabled.insert(OptimizerType::AGGREGATE_FUNCTION_REWRITER);

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius_physical_operator> result;
  try {
    Parser parser(context);
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);

    if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(context)) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver verifier(true);
    verifier.VisitOperator(*plan);
    resolver.VisitOperator(*plan);
    if (customize) { customize(*plan); }

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

sirius_physical_operator* find_first(sirius_physical_operator* root,
                                     SiriusPhysicalOperatorType type)
{
  if (!root) { return nullptr; }
  if (root->type == type) { return root; }
  for (auto& child : root->children) {
    if (auto* found = find_first(child.get(), type)) { return found; }
  }
  return nullptr;
}

std::size_t count_case_entries(sirius::op::sirius_physical_projection const& projection)
{
  std::size_t count = 0;
  for (auto const& entry : projection.select_list) {
    if (entry && entry->holds<sirius::ast::case_expr>()) { count++; }
  }
  return count;
}

void visit_aggregates(LogicalOperator& op,
                      std::function<void(BoundAggregateExpression&)> const& visit)
{
  if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    for (auto& expression : op.expressions) {
      visit(expression->Cast<BoundAggregateExpression>());
    }
  }
  for (auto& child : op.children) {
    visit_aggregates(*child, visit);
  }
}

duckdb::unique_ptr<Expression> synthetic_abs(duckdb::unique_ptr<Expression> child,
                                             FunctionStability stability,
                                             bool ordered)
{
  auto const type = child->GetReturnType();
  ScalarFunction function(Identifier("abs"), duckdb::vector<LogicalType>{type}, type, nullptr);
  function.SetStability(stability);
  function.SetRequiresOrderedExecution(ordered);
  duckdb::vector<duckdb::unique_ptr<Expression>> arguments;
  arguments.push_back(std::move(child));
  return duckdb::make_uniq<BoundFunctionExpression>(
    BoundScalarFunction(function), std::move(arguments), nullptr);
}

struct plan_aggregate_filter_fixture {
  plan_aggregate_filter_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    db  = sirius::test::open_sirius_db(_db_path.path().c_str(), cfg);
    con = std::make_unique<Connection>(*db);

    con->Query("CREATE TABLE events (id INTEGER, val INTEGER)");
    con->Query("INSERT INTO events SELECT range, NULLIF(range * 3 % 7, 0) FROM range(100)");
  }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - grouped filtered aggregates convert to masked CASE inputs",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT val % 2 AS g, sum(val) FILTER (WHERE val < 5), count(*) FILTER (WHERE val < 5) "
    "FROM events GROUP BY g");

  auto* group_by = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
  REQUIRE(group_by != nullptr);

  // count(*) FILTER lowers to count(mask) -> COUNT_VALID; no COUNT_ALL may survive, since
  // COUNT_ALL would count masked-out rows.
  auto& aggregate = group_by->Cast<sirius::op::sirius_physical_grouped_aggregate>();
  std::vector<cudf::aggregation::Kind> const expected{cudf::aggregation::Kind::SUM,
                                                      cudf::aggregation::Kind::COUNT_VALID};
  CHECK(aggregate.cudf_aggregates == expected);

  // The projection below the aggregate carries the mask rewrites: the sum input wrapped as
  // CASE WHEN filter THEN val ELSE NULL END and the count(*) boolean mask column.
  auto* projection = find_first(group_by, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection != nullptr);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 2);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - ungrouped count(*) FILTER converts to count(mask)",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(*con, "SELECT count(*) FILTER (WHERE val < 5) FROM events");

  auto* ungrouped = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(ungrouped != nullptr);

  auto& aggregate = ungrouped->Cast<sirius::op::sirius_physical_ungrouped_aggregate>();
  REQUIRE(aggregate.aggregates.size() == 1);
  REQUIRE(aggregate.aggregates[0]->holds<sirius::ast::aggregate>());
  auto const& agg = aggregate.aggregates[0]->get<sirius::ast::aggregate>();
  CHECK(agg.function() == sirius::aggregate_id::count);
  CHECK_FALSE(agg.distinct());
  REQUIRE(agg.arguments().size() == 1);
  CHECK(agg.arguments()[0]->holds<sirius::ast::reference>());
  CHECK(agg.arguments()[0]->return_type().id() == sirius::type_id::BOOLEAN);
  CHECK(agg.return_type().id() == sirius::type_id::BIGINT);

  // The boolean mask column (CASE WHEN filter THEN true ELSE NULL END) lives in the
  // projection below the aggregate.
  auto* projection = find_first(ungrouped, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection != nullptr);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 1);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - grouped COUNT(DISTINCT) FILTER collects a masked CASE input",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(*con,
                                   "SELECT val % 2 AS g, count(DISTINCT id) FILTER (WHERE id < 5), "
                                   "sum(id) FILTER (WHERE id < 5) FROM events GROUP BY g");

  auto* group_by = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
  REQUIRE(group_by != nullptr);

  // COLLECT_SET skips the NULLs the mask writes, so the distinct count and the sum read one
  // shared masked column.
  auto& aggregate = group_by->Cast<sirius::op::sirius_physical_grouped_aggregate>();
  std::vector<cudf::aggregation::Kind> const expected{cudf::aggregation::Kind::COLLECT_SET,
                                                      cudf::aggregation::Kind::SUM};
  CHECK(aggregate.cudf_aggregates == expected);
  CHECK(aggregate.cudf_aggregate_idx == std::vector<int>{1, 1});

  auto* projection = find_first(group_by, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection != nullptr);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 1);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - ungrouped COUNT(DISTINCT) FILTER keeps DISTINCT over the mask",
                 "[aggregate_filter][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT count(DISTINCT val) FILTER (WHERE id < 50) FROM events");

  auto* ungrouped = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(ungrouped != nullptr);

  auto& aggregate = ungrouped->Cast<sirius::op::sirius_physical_ungrouped_aggregate>();
  REQUIRE(aggregate.aggregates.size() == 1);
  REQUIRE(aggregate.aggregates[0]->holds<sirius::ast::aggregate>());
  auto const& agg = aggregate.aggregates[0]->get<sirius::ast::aggregate>();
  CHECK(agg.function() == sirius::aggregate_id::count);
  CHECK(agg.distinct());
  REQUIRE(agg.arguments().size() == 1);
  CHECK(agg.arguments()[0]->holds<sirius::ast::reference>());

  auto* projection = find_first(ungrouped, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection != nullptr);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 1);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - other DISTINCT aggregates and nested inputs are rejected",
                 "[aggregate_filter][isolated_context]")
{
  // The GPU evaluator cannot build the NULL STRUCT the mask of the multi-column count needs.
  for (auto const* aggregate : {"sum(DISTINCT id)", "count(DISTINCT (id, val))"}) {
    for (bool grouped : {false, true}) {
      CAPTURE(aggregate, grouped);
      auto const query = std::string("SELECT ") + (grouped ? "val % 2 AS g, " : "") + aggregate +
                         " FILTER (WHERE id < 5) FROM events" + (grouped ? " GROUP BY g" : "");
      REQUIRE_THROWS_MATCHES(generate_sirius_plan(*con, query),
                             duckdb::NotImplementedException,
                             Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                               "Aggregate FILTER not supported")));
    }
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - first with FILTER is rejected for CPU fallback",
                 "[aggregate_filter][isolated_context]")
{
  REQUIRE_THROWS_AS(
    generate_sirius_plan(*con, "SELECT first(val) FILTER (WHERE val = 3) FROM events"),
    duckdb::NotImplementedException);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - ordered FILTER is rejected before child planning",
                 "[aggregate_filter][isolated_context]")
{
  for (bool grouped : {false, true}) {
    for (bool filtered : {false, true}) {
      CAPTURE(grouped, filtered);
      auto const query = std::string("SELECT ") + (grouped ? "id % 2 AS g, " : "") + "sum(val)" +
                         (filtered ? " FILTER (WHERE val < 5)" : "") + " FROM events" +
                         (grouped ? " GROUP BY g" : "");
      auto customize = [](LogicalOperator& root) {
        auto* op = &root;
        while (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
          REQUIRE(op->children.size() == 1);
          op = op->children[0].get();
        }
        REQUIRE(op->expressions.size() == 1);
        auto& aggregate = op->expressions[0]->Cast<BoundAggregateExpression>();
        REQUIRE(aggregate.GetChildren().size() == 1);

        // Add ordering after optimization so DuckDB cannot remove it before the support check.
        auto orders = duckdb::make_uniq<BoundOrderModifier>();
        orders->orders.emplace_back(
          OrderType::ASCENDING, OrderByNullType::NULLS_LAST, aggregate.GetChildren()[0]->Copy());
        aggregate.GetOrderBysMutable() = std::move(orders);

        // The child must only be reached by the unfiltered control, which bypasses the FILTER gate.
        REQUIRE(op->children.size() == 1);
        op->children[0]->type = LogicalOperatorType::LOGICAL_INVALID;
      };
      REQUIRE_THROWS_MATCHES(
        generate_sirius_plan(*con, query, customize),
        duckdb::NotImplementedException,
        Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
          filtered ? "Aggregate FILTER not supported" : "Unimplemented logical operator type")));
    }
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - identical masked values share one column across functions",
                 "[aggregate_filter][isolated_context]")
{
  for (bool grouped : {false, true}) {
    auto plan =
      generate_sirius_plan(*con,
                           std::string("SELECT ") + (grouped ? "id % 2 AS g, " : "") +
                             "sum(val) FILTER (WHERE id < 50), min(val) FILTER (WHERE id < 50), "
                             "max(val) FILTER (WHERE id < 50), avg(val) FILTER (WHERE id < 50), "
                             "count(val) FILTER (WHERE id < 50), count(*) FILTER (WHERE id < 50), "
                             "count(*) FILTER (WHERE id < 50) FROM events" +
                             (grouped ? " GROUP BY g" : ""));
    auto* aggregate = find_first(plan.get(),
                                 grouped ? SiriusPhysicalOperatorType::HASH_GROUP_BY
                                         : SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    REQUIRE(aggregate);
    auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(projection);
    auto const& projected = projection->Cast<sirius::op::sirius_physical_projection>();
    CHECK(count_case_entries(projected) == 2);
    CHECK(projected.select_list.size() == (grouped ? 3 : 2));
    if (grouped) {
      auto const& grouped_agg = aggregate->Cast<sirius::op::sirius_physical_grouped_aggregate>();
      // AVG's expanded SUM and COUNT_VALID slots read the same value mask.
      CHECK(grouped_agg.cudf_aggregate_idx == std::vector<int>{1, 1, 1, 1, 1, 1, 2});
      REQUIRE(grouped_agg.aggregate_slots.size() == 6);
      CHECK(grouped_agg.aggregate_slots[3].is_avg);
    } else {
      auto const& ungrouped = aggregate->Cast<sirius::op::sirius_physical_ungrouped_aggregate>();
      REQUIRE(ungrouped.aggregates.size() == 6);
      for (std::size_t i = 0; i < ungrouped.aggregates.size(); ++i) {
        auto const& args = ungrouped.aggregates[i]->get<sirius::ast::aggregate>().arguments();
        REQUIRE(args.size() == 1);
        CHECK(args[0]->get<sirius::ast::reference>().column_index == (i < 5 ? 0 : 1));
      }
    }
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - distinct values predicates and types keep separate masks",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT min(val) FILTER (WHERE id < 50), max(val) FILTER (WHERE id < 51), "
    "min(id) FILTER (WHERE id < 50), min(CAST(val AS DOUBLE)) FILTER (WHERE id < 50), "
    "max(CAST(val AS DOUBLE)) FILTER (WHERE id < 50), sum(val) FROM events");
  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(aggregate);
  auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 4);
  auto const& aggs = aggregate->Cast<sirius::op::sirius_physical_ungrouped_aggregate>().aggregates;
  std::vector<uint32_t> indices;
  for (auto const& agg : aggs) {
    indices.push_back(agg->get<sirius::ast::aggregate>()
                        .arguments()[0]
                        ->get<sirius::ast::reference>()
                        .column_index);
  }
  CHECK(indices == std::vector<uint32_t>{0, 1, 2, 3, 3, 4});
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - complete predicates share one column across different inputs",
                 "[aggregate_filter][isolated_context]")
{
  for (bool grouped : {false, true}) {
    CAPTURE(grouped);
    auto plan =
      generate_sirius_plan(*con,
                           std::string("SELECT ") + (grouped ? "id AS g, " : "") +
                             "min(val) FILTER (WHERE id < 50), max(id) FILTER (WHERE id < 50), "
                             "count(*) FILTER (WHERE id < 50), "
                             "min(CAST(val AS DOUBLE)) FILTER (WHERE id >= 50), "
                             "max(id) FILTER (WHERE id >= 50), count(val) FILTER (WHERE id >= 50), "
                             "sum(id), min(CASE WHEN id < 50 THEN val ELSE NULL END) FROM events" +
                             (grouped ? " GROUP BY g" : ""));
    auto* aggregate = find_first(plan.get(),
                                 grouped ? SiriusPhysicalOperatorType::HASH_GROUP_BY
                                         : SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    REQUIRE(aggregate);
    auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(projection);
    auto const& masks = projection->Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(masks.children.size() == 1);
    REQUIRE(masks.children[0]->type == SiriusPhysicalOperatorType::PROJECTION);
    auto const& predicates = masks.children[0]->Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(predicates.children.size() == 1);
    auto const input_width = predicates.children[0]->types.size();
    REQUIRE(predicates.select_list.size() == input_width + 2);
    REQUIRE(predicates.types.size() == input_width + 2);
    for (std::size_t i = 0; i < input_width; ++i) {
      REQUIRE(predicates.select_list[i]->holds<sirius::ast::reference>());
      CHECK(predicates.select_list[i]->get<sirius::ast::reference>().column_index == i);
      CHECK(predicates.types[i] == predicates.children[0]->types[i]);
      CHECK(predicates.select_list[i]->return_type() == predicates.types[i]);
    }
    for (std::size_t i = input_width; i < input_width + 2; ++i) {
      CHECK(predicates.select_list[i]->holds<sirius::ast::comparison>());
      CHECK(predicates.types[i].id() == sirius::type_id::BOOLEAN);
    }

    auto const offset = grouped ? 1 : 0;
    REQUIRE(masks.select_list.size() == offset + 8);
    for (std::size_t i = 0; i < 6; ++i) {
      auto const& expression = *masks.select_list[offset + i];
      REQUIRE(expression.holds<sirius::ast::case_expr>());
      auto const& mask = expression.get<sirius::ast::case_expr>();
      REQUIRE(mask.cases.size() == 1);
      REQUIRE(mask.cases[0].when_->holds<sirius::ast::reference>());
      CHECK(mask.cases[0].when_->get<sirius::ast::reference>().column_index ==
            input_width + (i < 3 ? 0 : 1));
      CHECK(mask.cases[0].when_->return_type().id() == sirius::type_id::BOOLEAN);
    }
    CHECK(masks.select_list[offset + 3]->return_type().id() == sirius::type_id::DOUBLE);
    CHECK(masks.select_list[offset + 6]->holds<sirius::ast::reference>());
    auto const& user_case = masks.select_list[offset + 7]->get<sirius::ast::case_expr>();
    CHECK(user_case.cases[0].when_->holds<sirius::ast::comparison>());
    if (grouped) {
      REQUIRE(masks.select_list[0]->holds<sirius::ast::reference>());
      CHECK(masks.select_list[0]->get<sirius::ast::reference>().column_index < input_width);
    } else {
      auto const& aggs =
        aggregate->Cast<sirius::op::sirius_physical_ungrouped_aggregate>().aggregates;
      REQUIRE(aggs.size() == 8);
      for (std::size_t i = 0; i < aggs.size(); ++i) {
        CHECK(aggs[i]
                ->get<sirius::ast::aggregate>()
                .arguments()[0]
                ->get<sirius::ast::reference>()
                .column_index == i);
      }
      CHECK(aggs[2]->get<sirius::ast::aggregate>().function() == sirius::aggregate_id::count);
    }
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - DuckDB shared predicates need no additional stage",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT min(val) FILTER (WHERE id < 50), max(id) FILTER (WHERE id < 50), "
    "count(*) FILTER (WHERE id < 50) FROM events",
    [](LogicalOperator& root) {
      visit_aggregates(root, [](BoundAggregateExpression& aggregate) {
        REQUIRE(aggregate.GetFilter());
        CHECK(aggregate.GetFilter()->GetExpressionClass() == ExpressionClass::BOUND_REF);
      });
    },
    true);
  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(aggregate);
  auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection);
  auto const& masks = projection->Cast<sirius::op::sirius_physical_projection>();
  REQUIRE(masks.select_list.size() == 3);
  REQUIRE(masks.children.size() == 1);
  REQUIRE(masks.children[0]->type == SiriusPhysicalOperatorType::PROJECTION);
  auto const& predicates = masks.children[0]->Cast<sirius::op::sirius_physical_projection>();
  std::size_t computed_predicates = 0;
  for (auto const& expression : predicates.select_list) {
    if (expression->holds<sirius::ast::comparison>()) { ++computed_predicates; }
  }
  CHECK(computed_predicates == 1);
  REQUIRE(predicates.children.size() == 1);
  CHECK(predicates.children[0]->type != SiriusPhysicalOperatorType::PROJECTION);
  for (auto const& expression : masks.select_list) {
    auto const& mask = expression->get<sirius::ast::case_expr>();
    CHECK(mask.cases[0].when_->holds<sirius::ast::reference>());
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - a reused whole mask needs no predicate projection",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con, "SELECT min(val) FILTER (WHERE id < 50), max(val) FILTER (WHERE id < 50) FROM events");
  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(aggregate);
  auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection);
  auto const& masks = projection->Cast<sirius::op::sirius_physical_projection>();
  REQUIRE(masks.select_list.size() == 1);
  auto const& mask = masks.select_list[0]->get<sirius::ast::case_expr>();
  CHECK(mask.cases[0].when_->holds<sirius::ast::comparison>());
  REQUIRE(masks.children.size() == 1);
  CHECK(masks.children[0]->type != SiriusPhysicalOperatorType::PROJECTION);
}

TEST_CASE_METHOD(
  plan_aggregate_filter_fixture,
  "aggregate filter - different complete predicates retain their evaluation boundaries",
  "[aggregate_filter][isolated_context]")
{
  auto plan       = generate_sirius_plan(*con,
                                   "SELECT min(val) FILTER (WHERE id < 50 AND val > 1), "
                                         "max(id) FILTER (WHERE id < 50 OR val > 2) FROM events");
  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(aggregate);
  auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection);
  auto const& masks = projection->Cast<sirius::op::sirius_physical_projection>();
  REQUIRE(masks.select_list.size() == 2);
  for (auto const& expression : masks.select_list) {
    auto const& mask = expression->get<sirius::ast::case_expr>();
    CHECK(mask.cases[0].when_->holds<sirius::ast::conjunction>());
  }
  REQUIRE(masks.children.size() == 1);
  CHECK(masks.children[0]->type != SiriusPhysicalOperatorType::PROJECTION);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - trivial predicates need no extra projection",
                 "[aggregate_filter][isolated_context]")
{
  auto created = con->Query("CREATE TABLE flags AS SELECT id, val, id < 50 AS keep FROM events");
  REQUIRE_FALSE(created->HasError());
  for (bool constant : {false, true}) {
    CAPTURE(constant);
    auto plan =
      generate_sirius_plan(*con,
                           "SELECT min(val) FILTER (WHERE keep), max(id) FILTER (WHERE keep), "
                           "count(*) FILTER (WHERE keep) FROM flags",
                           [&](LogicalOperator& root) {
                             if (constant) {
                               visit_aggregates(root, [](BoundAggregateExpression& aggregate) {
                                 aggregate.GetFilterMutable() =
                                   duckdb::make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
                               });
                             }
                           });
    auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    REQUIRE(aggregate);
    auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(projection);
    auto const& masks = projection->Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(masks.select_list.size() == 3);
    for (auto const& expression : masks.select_list) {
      auto const& mask = expression->get<sirius::ast::case_expr>();
      CHECK(mask.cases[0].when_->holds<sirius::ast::constant>() == constant);
      CHECK(mask.cases[0].when_->holds<sirius::ast::reference>() == !constant);
    }
    REQUIRE(masks.children.size() == 1);
    CHECK(masks.children[0]->type != SiriusPhysicalOperatorType::PROJECTION);
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - unsafe value and predicate trees are never shared",
                 "[aggregate_filter][isolated_context]")
{
  for (bool predicate : {false, true}) {
    for (auto stability : {FunctionStability::CONSISTENT,
                           FunctionStability::VOLATILE,
                           FunctionStability::CONSISTENT_WITHIN_QUERY}) {
      bool const ordered = stability == FunctionStability::CONSISTENT;
      CAPTURE(predicate, ordered, static_cast<int>(stability));
      auto plan = generate_sirius_plan(
        *con,
        "SELECT min(val) FILTER (WHERE id < 50), max(val) FILTER (WHERE id < 50) FROM events",
        [&](LogicalOperator& root) {
          visit_aggregates(root, [&](BoundAggregateExpression& agg) {
            REQUIRE(agg.GetChildren().size() == 1);
            // Planner-only functions use supported names but never execute synthetic callbacks.
            if (predicate) {
              agg.GetFilterMutable() = BoundComparisonExpression::Create(
                ExpressionType::COMPARE_GREATERTHAN,
                synthetic_abs(agg.GetChildren()[0]->Copy(), stability, ordered),
                duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0)));
            } else {
              agg.GetChildrenMutable()[0] =
                synthetic_abs(agg.GetChildren()[0]->Copy(), stability, ordered);
            }
          });
        });
      auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
      REQUIRE(aggregate);
      auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
      REQUIRE(projection);
      auto const& masks = projection->Cast<sirius::op::sirius_physical_projection>();
      CHECK(count_case_entries(masks) == 2);
      for (auto const& expression : masks.select_list) {
        auto const& mask = expression->get<sirius::ast::case_expr>();
        CHECK(mask.cases[0].when_->holds<sirius::ast::reference>() == !predicate);
      }
      REQUIRE(masks.children.size() == 1);
      CHECK((masks.children[0]->type == SiriusPhysicalOperatorType::PROJECTION) == !predicate);
      auto const& aggs =
        aggregate->Cast<sirius::op::sirius_physical_ungrouped_aggregate>().aggregates;
      REQUIRE(aggs.size() == 2);
      CHECK(aggs[0]
              ->get<sirius::ast::aggregate>()
              .arguments()[0]
              ->get<sirius::ast::reference>()
              .column_index != aggs[1]
                                 ->get<sirius::ast::aggregate>()
                                 .arguments()[0]
                                 ->get<sirius::ast::reference>()
                                 .column_index);
    }
  }
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - an unsupported shared predicate still declines translation",
                 "[aggregate_filter][isolated_context]")
{
  REQUIRE_THROWS_AS(
    generate_sirius_plan(
      *con,
      "SELECT min(val) FILTER (WHERE id < 50), max(id) FILTER (WHERE id < 50) FROM events",
      [](LogicalOperator& root) {
        duckdb::unique_ptr<Expression> predicate;
        visit_aggregates(root, [&](BoundAggregateExpression& aggregate) {
          if (!predicate) {
            ScalarFunction function(Identifier("unsupported_filter_predicate"),
                                    duckdb::vector<LogicalType>{LogicalType::INTEGER},
                                    LogicalType::BOOLEAN,
                                    nullptr);
            function.SetStability(FunctionStability::CONSISTENT);
            duckdb::vector<duckdb::unique_ptr<Expression>> arguments;
            arguments.push_back(aggregate.GetChildren()[0]->Copy());
            predicate = duckdb::make_uniq<BoundFunctionExpression>(
              BoundScalarFunction(function), std::move(arguments), nullptr);
          }
          aggregate.GetFilterMutable() = predicate->Copy();
        });
      }),
    duckdb::NotImplementedException);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - identical count-star masks reuse the projected column",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT count(*) FILTER (WHERE id < 50), count(*) FILTER (WHERE id < 51) FROM events",
    [](LogicalOperator& root) {
      duckdb::unique_ptr<Expression> filter;
      visit_aggregates(root, [&](BoundAggregateExpression& agg) {
        REQUIRE(agg.GetChildren().empty());
        REQUIRE(agg.GetFilter());
        // Keep two binder slots, then give them identical predicates at the extraction boundary.
        if (!filter) { filter = agg.GetFilter()->Copy(); }
        agg.GetFilterMutable() = filter->Copy();
      });
    });
  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(aggregate);
  auto* projection = find_first(aggregate, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 1);
  auto const& aggs = aggregate->Cast<sirius::op::sirius_physical_ungrouped_aggregate>().aggregates;
  REQUIRE(aggs.size() == 2);
  for (auto const& agg : aggs) {
    CHECK(agg->get<sirius::ast::aggregate>().function() == sirius::aggregate_id::count);
    CHECK(agg->get<sirius::ast::aggregate>()
            .arguments()[0]
            ->get<sirius::ast::reference>()
            .column_index == 0);
  }
}
