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

#include "planner/join_key_nullability.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_cteref.hpp>
#include <duckdb/planner/operator/logical_distinct.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_limit.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_top_n.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace duckdb;
using sirius::planner::normalize_non_null_join_keys;

unique_ptr<Expression> ref(idx_t ordinal)
{
  return make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, ordinal);
}

unique_ptr<LogicalComparisonJoin> make_join(unique_ptr<LogicalOperator> left,
                                            unique_ptr<LogicalOperator> right,
                                            idx_t left_key  = 0,
                                            idx_t right_key = 0,
                                            JoinType type   = JoinType::RIGHT_ANTI)
{
  auto join = make_uniq<LogicalComparisonJoin>(type);
  join->children.push_back(std::move(left));
  join->children.push_back(std::move(right));
  join->conditions.emplace_back(
    ref(left_key), ref(right_key), ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  join->ResolveOperatorTypes();
  return join;
}

unique_ptr<LogicalProjection> project(unique_ptr<LogicalOperator> child, vector<idx_t> ordinals)
{
  vector<unique_ptr<Expression>> expressions;
  for (auto ordinal : ordinals)
    expressions.push_back(ref(ordinal));
  auto result = make_uniq<LogicalProjection>(TableIndex(20), std::move(expressions));
  result->children.push_back(std::move(child));
  return result;
}

struct nullability_fixture {
  nullability_fixture()
  {
    if (auto const* old = std::getenv("SIRIUS_DISABLE")) previous = old;
    setenv("SIRIUS_DISABLE", "1", 1);
    db  = std::make_unique<DuckDB>(nullptr);
    con = std::make_unique<Connection>(*db);
    run("CREATE TABLE constrained(k INTEGER NOT NULL, nullable INTEGER, other INTEGER NOT NULL)");
    run("CREATE TABLE unconstrained(k INTEGER, nullable INTEGER, other INTEGER)");
    run("INSERT INTO constrained VALUES (1, NULL, 10), (2, 20, 20)");
    run("INSERT INTO unconstrained VALUES (1, 10, 10), (NULL, NULL, NULL), (3, 30, 30)");
    run("BEGIN TRANSACTION");
  }
  ~nullability_fixture()
  {
    con.reset();
    db.reset();
    if (previous)
      setenv("SIRIUS_DISABLE", previous->c_str(), 1);
    else
      unsetenv("SIRIUS_DISABLE");
  }
  void run(std::string const& sql)
  {
    auto result = con->Query(sql);
    INFO((result->HasError() ? result->GetError() : sql));
    REQUIRE_FALSE(result->HasError());
  }
  unique_ptr<LogicalOperator> scan(std::string const& table)
  {
    auto& context = *con->context;
    Parser parser(context);
    parser.ParseQuery("SELECT * FROM " + table);
    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    auto plan = std::move(planner.plan);
    plan->ResolveOperatorTypes();
    ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    while (plan->type != LogicalOperatorType::LOGICAL_GET) {
      REQUIRE(plan->children.size() == 1);
      auto child = std::move(plan->children[0]);
      plan       = std::move(child);
    }
    // A constraint remains evidence when no data statistics callback is available.
    auto& get                        = plan->Cast<LogicalGet>();
    get.function.statistics          = nullptr;
    get.function.statistics_extended = nullptr;
    return plan;
  }
  std::optional<std::string> previous;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization uses constraints on either side",
                 "[planner][join_key_nullability]")
{
  auto const constrained_left = GENERATE(false, true);
  auto const type             = GENERATE(JoinType::INNER,
                             JoinType::LEFT,
                             JoinType::RIGHT,
                             JoinType::OUTER,
                             JoinType::SEMI,
                             JoinType::ANTI,
                             JoinType::RIGHT_SEMI,
                             JoinType::RIGHT_ANTI);
  auto join                   = make_join(scan(constrained_left ? "constrained" : "unconstrained"),
                        scan(constrained_left ? "unconstrained" : "constrained"),
                        0,
                        0,
                        type);
  REQUIRE_FALSE(join->conditions[0].GetLeftStats());
  REQUIRE_FALSE(join->conditions[0].GetRightStats());
  REQUIRE(normalize_non_null_join_keys(*join) == 1);
  CHECK(join->conditions[0].GetComparisonType() == ExpressionType::COMPARE_EQUAL);
  CHECK(normalize_non_null_join_keys(*join) == 0);
}

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization refuses nullable keys and MARK joins",
                 "[planner][join_key_nullability]")
{
  SECTION("nullable column beside NOT NULL columns")
  {
    auto join = make_join(scan("constrained"), scan("unconstrained"), 1);
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("current data without NULLs is not a constraint")
  {
    run("DELETE FROM unconstrained WHERE k IS NULL");
    auto join = make_join(scan("unconstrained"), scan("unconstrained"));
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("MARK observes UNKNOWN separately from false")
  {
    auto join = make_join(scan("constrained"), scan("unconstrained"), 0, 0, JoinType::MARK);
    CHECK(normalize_non_null_join_keys(*join) == 0);
    CHECK(join->conditions[0].GetComparisonType() == ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  }
  SECTION("other comparison semantics are unchanged")
  {
    auto join           = make_join(scan("constrained"), scan("unconstrained"));
    join->conditions[0] = JoinCondition(ref(0), ref(0), ExpressionType::COMPARE_DISTINCT_FROM);
    CHECK(normalize_non_null_join_keys(*join) == 0);
    CHECK(join->conditions[0].GetComparisonType() == ExpressionType::COMPARE_DISTINCT_FROM);
  }
}

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization maps scan and operator projections",
                 "[planner][join_key_nullability]")
{
  auto input = scan("constrained");
  auto& get  = input->Cast<LogicalGet>();
  get.SetColumnIds({ColumnIndex(2), ColumnIndex(1), ColumnIndex(0)});
  get.projection_ids     = {ProjectionIndex(1), ProjectionIndex(2)};
  auto projection        = project(std::move(input), {1, 0});
  auto filter            = make_uniq<LogicalFilter>();
  filter->projection_map = {ProjectionIndex(1), ProjectionIndex(0)};
  filter->children.push_back(std::move(projection));
  auto order            = make_uniq<LogicalOrder>(vector<BoundOrderByNode>{});
  order->projection_map = {ProjectionIndex(1), ProjectionIndex(0)};
  order->children.push_back(std::move(filter));
  auto const key = GENERATE(0, 1);
  auto join      = make_join(std::move(order), scan("unconstrained"), key, key);
  CHECK(normalize_non_null_join_keys(*join) == (key == 0 ? 1 : 0));
  CHECK(join->conditions[0].GetComparisonType() ==
        (key == 0 ? ExpressionType::COMPARE_EQUAL : ExpressionType::COMPARE_NOT_DISTINCT_FROM));
}

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization requires every null-safe key to be proven",
                 "[planner][join_key_nullability]")
{
  auto const second_key = GENERATE(1, 2);
  auto join             = make_join(scan("constrained"), scan("unconstrained"));
  join->conditions.emplace_back(
    ref(second_key), ref(second_key), ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  auto const expected =
    second_key == 2 ? ExpressionType::COMPARE_EQUAL : ExpressionType::COMPARE_NOT_DISTINCT_FROM;
  CHECK(normalize_non_null_join_keys(*join) == (second_key == 2 ? 2 : 0));
  CHECK(join->conditions[0].GetComparisonType() == expected);
  CHECK(join->conditions[1].GetComparisonType() == expected);
}

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization follows row selection operators",
                 "[planner][join_key_nullability]")
{
  auto distinct = make_uniq<LogicalDistinct>(DistinctType::DISTINCT);
  distinct->children.push_back(scan("constrained"));
  auto limit = make_uniq<LogicalLimit>(BoundLimitNode::ConstantValue(10), BoundLimitNode());
  limit->children.push_back(std::move(distinct));
  auto top = make_uniq<LogicalTopN>(vector<BoundOrderByNode>{}, 5, 0);
  top->children.push_back(std::move(limit));
  auto join = make_join(std::move(top), scan("unconstrained"));
  CHECK(normalize_non_null_join_keys(*join) == 1);
}

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization stops at computed and null-padding operators",
                 "[planner][join_key_nullability]")
{
  SECTION("constant projection")
  {
    auto projection            = project(scan("constrained"), {0});
    projection->expressions[0] = make_uniq<BoundConstantExpression>(Value::INTEGER(5));
    auto join                  = make_join(std::move(projection), scan("unconstrained"));
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("cast projection")
  {
    auto projection = project(scan("constrained"), {0});
    projection->expressions[0] =
      BoundCastExpression::AddDefaultCastToType(ref(0), LogicalType::BIGINT);
    auto join           = make_join(std::move(projection), scan("unconstrained"));
    join->conditions[0] = JoinCondition(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0),
                                        ref(0),
                                        ExpressionType::COMPARE_NOT_DISTINCT_FROM);
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("CTE reference has no catalog constraint proof")
  {
    auto cte  = make_uniq<LogicalCTERef>(TableIndex(25),
                                        TableIndex(26),
                                        vector<LogicalType>{LogicalType::INTEGER},
                                        vector<Identifier>{"k"});
    auto join = make_join(std::move(cte), scan("unconstrained"));
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("outer join pads a constrained column")
  {
    auto outer = make_join(scan("unconstrained"), scan("constrained"), 0, 0, JoinType::LEFT);
    auto join  = make_join(std::move(outer), scan("unconstrained"), 3);
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("single grouping set can omit a grouping key")
  {
    auto aggregate =
      make_uniq<LogicalAggregate>(TableIndex(21), TableIndex(22), vector<unique_ptr<Expression>>{});
    aggregate->groups.push_back(ref(0));
    aggregate->grouping_sets.emplace_back();
    aggregate->children.push_back(scan("constrained"));
    auto join = make_join(std::move(aggregate), scan("unconstrained"));
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("a pushed scan projection has lost its source expression")
  {
    auto input                                      = scan("constrained");
    input->Cast<LogicalGet>().has_pushed_projection = true;
    auto join = make_join(std::move(input), scan("unconstrained"));
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
  SECTION("virtual column cannot inherit a regular column constraint")
  {
    auto input                   = scan("constrained");
    auto join                    = make_join(std::move(input), scan("unconstrained"));
    auto& get                    = join->children[0]->Cast<LogicalGet>();
    get.GetMutableColumnIds()[0] = ColumnIndex(COLUMN_IDENTIFIER_ROW_ID);
    CHECK(normalize_non_null_join_keys(*join) == 0);
  }
}

TEST_CASE_METHOD(nullability_fixture,
                 "join key normalization reflects a dropped constraint",
                 "[planner][join_key_nullability]")
{
  run("COMMIT");
  run("ALTER TABLE constrained ALTER COLUMN k DROP NOT NULL");
  run("INSERT INTO constrained VALUES (NULL, NULL, 30)");
  run("BEGIN TRANSACTION");
  auto join = make_join(scan("constrained"), scan("unconstrained"));
  CHECK(normalize_non_null_join_keys(*join) == 0);
}
