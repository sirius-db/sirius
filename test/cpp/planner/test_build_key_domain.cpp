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
 * @file test_build_key_domain.cpp
 * @brief Tests positional lineage resolution for build-key domains.
 *
 * The CPU-only fixtures use disjoint ordinal and cardinality ranges so resolving the wrong scan or
 * coordinate produces an unmistakable result.
 */

#include "planner/dynamic_filter/build_key_domain.hpp"

#include <catch.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
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
#include <duckdb/planner/operator/logical_set_operation.hpp>
#include <duckdb/planner/operator/logical_top_n.hpp>
#include <duckdb/planner/operator/logical_unnest.hpp>
#include <duckdb/planner/operator/logical_window.hpp>

#include <cstddef>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

using sirius::planner::build_key_domain_cardinalities;
using sirius::planner::detail::resolve_build_key_scans;
using sirius::planner::detail::resolve_pass_through_scan;

namespace {

// Assign each table a distinct cardinality so resolving the wrong scan is visible.
constexpr std::size_t stub_cardinality_of(duckdb::idx_t table_index)
{
  return std::size_t{1000} * (static_cast<std::size_t>(table_index) + 1);
}

struct stub_cardinality_source {
  std::optional<std::size_t> operator()(duckdb::LogicalGet const& get) const
  {
    return stub_cardinality_of(get.table_index);
  }
};

struct refusing_source {
  std::optional<std::size_t> operator()(duckdb::LogicalGet const&) const { return std::nullopt; }
};

// Count cardinality lookups to verify per-scan memoization.
struct counting_source {
  std::size_t* calls = nullptr;
  std::optional<std::size_t> operator()(duckdb::LogicalGet const& get) const
  {
    ++*calls;
    return stub_cardinality_of(get.table_index);
  }
};

// Build a scan whose default-constructed table function is never invoked.
duckdb::unique_ptr<duckdb::LogicalGet> make_get(duckdb::idx_t table_index, duckdb::idx_t width)
{
  duckdb::vector<duckdb::LogicalType> types(width, duckdb::LogicalType::INTEGER);
  duckdb::vector<duckdb::string> names;
  for (duckdb::idx_t i = 0; i < width; ++i) {
    names.push_back("c" + std::to_string(i));
  }
  auto get = duckdb::make_uniq<duckdb::LogicalGet>(
    table_index, duckdb::TableFunction(), nullptr, std::move(types), std::move(names));
  for (duckdb::idx_t i = 0; i < width; ++i) {
    get->AddColumnId(i);
  }
  return get;
}

duckdb::unique_ptr<duckdb::Expression> make_ref(duckdb::idx_t index)
{
  return duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, index);
}

// A computed expression has no pass-through base-column lineage.
duckdb::unique_ptr<duckdb::Expression> make_computed()
{
  return duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(7));
}

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection(
  duckdb::unique_ptr<duckdb::LogicalOperator> child,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions)
{
  auto projection =
    duckdb::make_uniq<duckdb::LogicalProjection>(/*table_index=*/0, std::move(expressions));
  projection->children.push_back(std::move(child));
  return projection;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join(
  duckdb::JoinType join_type,
  duckdb::unique_ptr<duckdb::LogicalOperator> left,
  duckdb::unique_ptr<duckdb::LogicalOperator> right,
  duckdb::vector<duckdb::idx_t> left_projection_map  = {},
  duckdb::vector<duckdb::idx_t> right_projection_map = {})
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  join->children.push_back(std::move(left));
  join->children.push_back(std::move(right));
  join->left_projection_map  = std::move(left_projection_map);
  join->right_projection_map = std::move(right_projection_map);
  return join;
}

duckdb::JoinCondition make_condition(duckdb::unique_ptr<duckdb::Expression> left,
                                     duckdb::unique_ptr<duckdb::Expression> right)
{
  duckdb::JoinCondition condition;
  condition.left       = std::move(left);
  condition.right      = std::move(right);
  condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  return condition;
}

// Require each resolvable output ordinal to reach the expected scan.
void require_all_ordinals_resolve_to(duckdb::LogicalOperator const& subtree,
                                     std::size_t output_width,
                                     duckdb::LogicalGet const* expected)
{
  for (auto const ordinal : std::views::iota(std::size_t{0}, output_width)) {
    auto const* resolved = resolve_pass_through_scan(subtree, ordinal);
    if (resolved != nullptr) { REQUIRE(resolved == expected); }
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Leaf resolution and admissibility
//===----------------------------------------------------------------------===//

TEST_CASE("walk resolves a plain base scan and refuses ordinals past its width",
          "[dynamic_filter][build_key_domain]")
{
  auto get = make_get(/*table_index=*/0, /*width=*/3);
  get->ResolveOperatorTypes();

  for (auto const ordinal : std::views::iota(std::size_t{0}, std::size_t{3})) {
    REQUIRE(resolve_pass_through_scan(*get, ordinal) == get.get());
  }
  REQUIRE(resolve_pass_through_scan(*get, 3) == nullptr);
}

TEST_CASE("walk uses the projected scan width when projection_ids narrow the scan",
          "[dynamic_filter][build_key_domain]")
{
  auto get            = make_get(/*table_index=*/0, /*width=*/4);
  get->projection_ids = {2};  // scan output is one column
  get->ResolveOperatorTypes();

  REQUIRE(resolve_pass_through_scan(*get, 0) == get.get());
  REQUIRE(resolve_pass_through_scan(*get, 1) == nullptr);
  REQUIRE(resolve_pass_through_scan(*get, 3) == nullptr);
}

TEST_CASE("walk refuses a table-in-out scan wholesale", "[dynamic_filter][build_key_domain]")
{
  // A GET with projected input appends its child's columns after the scan columns, so "the base
  // table's row count" is not meaningful for any ordinal of this node.
  auto get             = make_get(/*table_index=*/0, /*width=*/2);
  auto input           = make_get(/*table_index=*/1, /*width=*/1);
  get->projected_input = {0};
  get->children.push_back(std::move(input));
  get->ResolveOperatorTypes();

  REQUIRE(resolve_pass_through_scan(*get, 0) == nullptr);  // a scan-width ordinal, still refused
  REQUIRE(resolve_pass_through_scan(*get, 2) == nullptr);  // the appended input column
}

//===----------------------------------------------------------------------===//
// Single-child pass-through operators
//===----------------------------------------------------------------------===//

TEST_CASE("walk resolves a projection pass-through at a non-identity position",
          "[dynamic_filter][build_key_domain]")
{
  auto get        = make_get(/*table_index=*/0, /*width=*/3);
  auto const* raw = get.get();
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  expressions.push_back(make_computed());  // ordinal 0: computed, must refuse
  expressions.push_back(make_ref(2));      // ordinal 1: pass-through of child ordinal 2
  auto projection = make_projection(std::move(get), std::move(expressions));
  projection->ResolveOperatorTypes();

  REQUIRE(resolve_pass_through_scan(*projection, 0) == nullptr);
  REQUIRE(resolve_pass_through_scan(*projection, 1) == raw);
  REQUIRE(resolve_pass_through_scan(*projection, 2) == nullptr);  // past the projection width
}

TEST_CASE("walk applies the FILTER projection map", "[dynamic_filter][build_key_domain]")
{
  // The child projection makes the map observable: mapped resolution lands on a pass-through
  // reference, identity resolution lands on a computed expression and refuses.
  auto get = make_get(/*table_index=*/0, /*width=*/2);
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  expressions.push_back(make_computed());
  expressions.push_back(make_ref(1));
  auto projection = make_projection(std::move(get), std::move(expressions));

  auto filter = duckdb::make_uniq<duckdb::LogicalFilter>();
  filter->children.push_back(std::move(projection));

  SECTION("without a projection map: identity")
  {
    filter->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*filter, 0) == nullptr);  // the computed column
    REQUIRE(resolve_pass_through_scan(*filter, 1) != nullptr);
  }
  SECTION("with a projection map: mapped")
  {
    filter->projection_map = {1};
    filter->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*filter, 0) != nullptr);  // map[0] = 1, the reference
    REQUIRE(resolve_pass_through_scan(*filter, 1) == nullptr);  // past the map
  }
}

TEST_CASE("walk applies the ORDER_BY projection map", "[dynamic_filter][build_key_domain]")
{
  auto get = make_get(/*table_index=*/0, /*width=*/2);
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  expressions.push_back(make_computed());
  expressions.push_back(make_ref(0));
  auto projection = make_projection(std::move(get), std::move(expressions));

  auto order = duckdb::make_uniq<duckdb::LogicalOrder>(duckdb::vector<duckdb::BoundOrderByNode>{});
  order->children.push_back(std::move(projection));

  SECTION("without a projection map: identity")
  {
    order->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*order, 0) == nullptr);
    REQUIRE(resolve_pass_through_scan(*order, 1) != nullptr);
  }
  SECTION("with a projection map: mapped")
  {
    order->projection_map = {1};
    order->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*order, 0) != nullptr);
    REQUIRE(resolve_pass_through_scan(*order, 1) == nullptr);
  }
}

TEST_CASE("walk passes through LIMIT, TOP_N, and DISTINCT by identity",
          "[dynamic_filter][build_key_domain]")
{
  auto make_traced_child = [] {
    auto get        = make_get(/*table_index=*/0, /*width=*/2);
    auto const* raw = get.get();
    return std::pair{std::move(get), raw};
  };

  SECTION("LIMIT")
  {
    auto [get, raw] = make_traced_child();
    auto limit = duckdb::make_uniq<duckdb::LogicalLimit>(duckdb::BoundLimitNode::ConstantValue(10),
                                                         duckdb::BoundLimitNode());
    limit->children.push_back(std::move(get));
    limit->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*limit, 1) == raw);
  }
  SECTION("TOP_N")
  {
    auto [get, raw] = make_traced_child();
    auto top_n      = duckdb::make_uniq<duckdb::LogicalTopN>(
      duckdb::vector<duckdb::BoundOrderByNode>{}, /*limit=*/10, /*offset=*/0);
    top_n->children.push_back(std::move(get));
    top_n->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*top_n, 1) == raw);
  }
  SECTION("DISTINCT")
  {
    auto [get, raw] = make_traced_child();
    auto distinct   = duckdb::make_uniq<duckdb::LogicalDistinct>(duckdb::DistinctType::DISTINCT);
    distinct->children.push_back(std::move(get));
    distinct->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*distinct, 1) == raw);
  }
}

TEST_CASE("walk resolves aggregate group ordinals only", "[dynamic_filter][build_key_domain]")
{
  auto make_aggregate = [](duckdb::unique_ptr<duckdb::LogicalOperator> child) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
    aggregates.push_back(make_computed());  // structural stand-in for an aggregate expression
    auto aggregate = duckdb::make_uniq<duckdb::LogicalAggregate>(
      /*group_index=*/50, /*aggregate_index=*/51, std::move(aggregates));
    aggregate->groups.push_back(make_ref(1));
    aggregate->children.push_back(std::move(child));
    return aggregate;
  };

  SECTION("a group that is a plain reference resolves")
  {
    auto get        = make_get(/*table_index=*/0, /*width=*/2);
    auto const* raw = get.get();
    auto aggregate  = make_aggregate(std::move(get));
    aggregate->ResolveOperatorTypes();

    REQUIRE(resolve_pass_through_scan(*aggregate, 0) == raw);
  }
  SECTION("an aggregate output ordinal refuses")
  {
    auto aggregate = make_aggregate(make_get(/*table_index=*/0, /*width=*/2));
    aggregate->ResolveOperatorTypes();

    REQUIRE(resolve_pass_through_scan(*aggregate, 1) == nullptr);
  }
  SECTION("multiple grouping sets refuse even at a group ordinal")
  {
    auto aggregate = make_aggregate(make_get(/*table_index=*/0, /*width=*/2));
    aggregate->grouping_sets.push_back(duckdb::GroupingSet{});
    aggregate->grouping_sets.push_back(duckdb::GroupingSet{});
    aggregate->ResolveOperatorTypes();

    REQUIRE(resolve_pass_through_scan(*aggregate, 0) == nullptr);
  }
}

TEST_CASE("walk composes a three-level stack", "[dynamic_filter][build_key_domain]")
{
  // GET(width 3) -> FILTER(map {2, 0}) -> PROJECTION([ref 1]): projection output 0 reads filter
  // output 1, the map sends 1 to child ordinal 0, which is within the scan. Every level's remap
  // participates; a dropped remap at either level changes the outcome.
  auto get        = make_get(/*table_index=*/0, /*width=*/3);
  auto const* raw = get.get();

  auto filter            = duckdb::make_uniq<duckdb::LogicalFilter>();
  filter->projection_map = {2, 0};
  filter->children.push_back(std::move(get));

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  expressions.push_back(make_ref(1));
  auto projection = make_projection(std::move(filter), std::move(expressions));
  projection->ResolveOperatorTypes();

  REQUIRE(resolve_pass_through_scan(*projection, 0) == raw);
}

//===----------------------------------------------------------------------===//
// Joins
//===----------------------------------------------------------------------===//

TEST_CASE("walk continues through row-subset joins to the emitted side",
          "[dynamic_filter][build_key_domain]")
{
  auto const semi_like_left = {duckdb::JoinType::SEMI, duckdb::JoinType::ANTI};
  for (auto const join_type : semi_like_left) {
    auto left            = make_get(/*table_index=*/0, /*width=*/2);
    auto const* left_raw = left.get();
    auto right           = make_get(/*table_index=*/1, /*width=*/2);
    auto join            = make_join(join_type, std::move(left), std::move(right));
    join->ResolveOperatorTypes();

    // Output is only the left block.
    REQUIRE(join->types.size() == 2);
    REQUIRE(resolve_pass_through_scan(*join, 0) == left_raw);
    REQUIRE(resolve_pass_through_scan(*join, 1) == left_raw);
    REQUIRE(resolve_pass_through_scan(*join, 2) == nullptr);
  }

  auto const semi_like_right = {duckdb::JoinType::RIGHT_SEMI, duckdb::JoinType::RIGHT_ANTI};
  for (auto const join_type : semi_like_right) {
    auto left             = make_get(/*table_index=*/0, /*width=*/2);
    auto right            = make_get(/*table_index=*/1, /*width=*/3);
    auto const* right_raw = right.get();
    auto join             = make_join(join_type, std::move(left), std::move(right));
    join->ResolveOperatorTypes();

    // Output is only the right block.
    REQUIRE(join->types.size() == 3);
    require_all_ordinals_resolve_to(*join, 3, right_raw);
    REQUIRE(resolve_pass_through_scan(*join, 0) == right_raw);
  }
}

TEST_CASE("walk continues through MARK's left block and refuses the mark column",
          "[dynamic_filter][build_key_domain]")
{
  auto left            = make_get(/*table_index=*/0, /*width=*/2);
  auto const* left_raw = left.get();
  auto right           = make_get(/*table_index=*/1, /*width=*/2);
  auto join            = make_join(duckdb::JoinType::MARK, std::move(left), std::move(right));
  join->ResolveOperatorTypes();

  REQUIRE(join->types.size() == 3);  // left block + appended BOOLEAN mark
  REQUIRE(resolve_pass_through_scan(*join, 0) == left_raw);
  REQUIRE(resolve_pass_through_scan(*join, 1) == left_raw);
  REQUIRE(resolve_pass_through_scan(*join, 2) == nullptr);  // the mark column
}

TEST_CASE("walk continues through SINGLE's left block and refuses its right block",
          "[dynamic_filter][build_key_domain]")
{
  auto left            = make_get(/*table_index=*/0, /*width=*/2);
  auto const* left_raw = left.get();
  auto right           = make_get(/*table_index=*/1, /*width=*/3);
  auto join            = make_join(duckdb::JoinType::SINGLE, std::move(left), std::move(right));
  join->ResolveOperatorTypes();

  REQUIRE(join->types.size() == 5);
  REQUIRE(resolve_pass_through_scan(*join, 0) == left_raw);
  REQUIRE(resolve_pass_through_scan(*join, 1) == left_raw);
  // Right-block values may repeat across left rows, so every right ordinal refuses.
  for (auto const ordinal : std::views::iota(std::size_t{2}, std::size_t{5})) {
    REQUIRE(resolve_pass_through_scan(*join, ordinal) == nullptr);
  }
}

TEST_CASE("walk refuses row-multiplying and DELIM joins on every ordinal",
          "[dynamic_filter][build_key_domain]")
{
  SECTION("INNER refuses both sides")
  {
    auto join = make_join(duckdb::JoinType::INNER,
                          make_get(/*table_index=*/0, /*width=*/2),
                          make_get(/*table_index=*/1, /*width=*/2));
    join->ResolveOperatorTypes();

    for (auto const ordinal : std::views::iota(std::size_t{0}, join->types.size())) {
      REQUIRE(resolve_pass_through_scan(*join, ordinal) == nullptr);
    }
  }
  SECTION("LEFT refuses")
  {
    auto join = make_join(duckdb::JoinType::LEFT,
                          make_get(/*table_index=*/0, /*width=*/2),
                          make_get(/*table_index=*/1, /*width=*/2));
    join->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*join, 0) == nullptr);
  }
  SECTION("DELIM join refuses even with row-subset semantics")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(
      duckdb::JoinType::SEMI, duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN);
    join->children.push_back(make_get(/*table_index=*/0, /*width=*/2));
    join->children.push_back(make_get(/*table_index=*/1, /*width=*/2));
    join->ResolveOperatorTypes();
    REQUIRE(resolve_pass_through_scan(*join, 0) == nullptr);
  }
}

TEST_CASE("walk locates join ordinals through non-empty projection maps",
          "[dynamic_filter][build_key_domain]")
{
  SECTION("SEMI with a left projection map")
  {
    // The left child's ordinal 0 is computed and ordinal 1 is a pass-through, so a dropped map
    // (identity) resolves the computed column and refuses.
    auto get = make_get(/*table_index=*/0, /*width=*/2);
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    expressions.push_back(make_computed());
    expressions.push_back(make_ref(0));
    auto projection = make_projection(std::move(get), std::move(expressions));

    auto join = make_join(duckdb::JoinType::SEMI,
                          std::move(projection),
                          make_get(/*table_index=*/1, /*width=*/2),
                          /*left_projection_map=*/{1});
    join->ResolveOperatorTypes();

    REQUIRE(join->types.size() == 1);
    REQUIRE(resolve_pass_through_scan(*join, 0) != nullptr);
    REQUIRE(resolve_pass_through_scan(*join, 1) == nullptr);  // past the mapped width
  }
  SECTION("ANTI with a left projection map")
  {
    auto get = make_get(/*table_index=*/0, /*width=*/2);
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    expressions.push_back(make_computed());
    expressions.push_back(make_ref(0));
    auto projection = make_projection(std::move(get), std::move(expressions));

    auto join = make_join(duckdb::JoinType::ANTI,
                          std::move(projection),
                          make_get(/*table_index=*/1, /*width=*/2),
                          /*left_projection_map=*/{1});
    join->ResolveOperatorTypes();

    REQUIRE(join->types.size() == 1);
    REQUIRE(resolve_pass_through_scan(*join, 0) != nullptr);
    REQUIRE(resolve_pass_through_scan(*join, 1) == nullptr);
  }
  SECTION("SINGLE with a left projection map still refuses the right block")
  {
    auto left             = make_get(/*table_index=*/0, /*width=*/3);
    auto const* left_raw  = left.get();
    auto right            = make_get(/*table_index=*/1, /*width=*/2);
    auto const* right_raw = right.get();
    auto join             = make_join(duckdb::JoinType::SINGLE,
                          std::move(left),
                          std::move(right),
                          /*left_projection_map=*/{2});
    join->ResolveOperatorTypes();

    REQUIRE(join->types.size() == 3);  // one mapped left column + two right columns
    REQUIRE(resolve_pass_through_scan(*join, 0) == left_raw);
    for (auto const ordinal : std::views::iota(std::size_t{1}, std::size_t{3})) {
      auto const* resolved = resolve_pass_through_scan(*join, ordinal);
      REQUIRE(resolved == nullptr);
      REQUIRE(resolved != right_raw);
    }
  }
  SECTION("RIGHT_SEMI with a right projection map")
  {
    auto get = make_get(/*table_index=*/1, /*width=*/2);
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    expressions.push_back(make_computed());
    expressions.push_back(make_ref(1));
    auto projection = make_projection(std::move(get), std::move(expressions));

    auto join = make_join(duckdb::JoinType::RIGHT_SEMI,
                          make_get(/*table_index=*/0, /*width=*/2),
                          std::move(projection),
                          /*left_projection_map=*/{},
                          /*right_projection_map=*/{1});
    join->ResolveOperatorTypes();

    REQUIRE(join->types.size() == 1);
    REQUIRE(resolve_pass_through_scan(*join, 0) != nullptr);
  }
  SECTION("MARK with a left projection map places the mark after the mapped block")
  {
    auto left            = make_get(/*table_index=*/0, /*width=*/3);
    auto const* left_raw = left.get();
    auto join            = make_join(duckdb::JoinType::MARK,
                          std::move(left),
                          make_get(/*table_index=*/1, /*width=*/2),
                          /*left_projection_map=*/{2});
    join->ResolveOperatorTypes();

    REQUIRE(join->types.size() == 2);  // one mapped left column + the mark
    REQUIRE(resolve_pass_through_scan(*join, 0) == left_raw);
    REQUIRE(resolve_pass_through_scan(*join, 1) == nullptr);  // the mark column
  }
}

TEST_CASE("walk never resolves an ordinal to the opposite join side",
          "[dynamic_filter][build_key_domain]")
{
  // Two GETs with different table indexes under one modelled join: a recurse-into-every-child
  // default would resolve the wrong GET for refused ordinals.
  auto left             = make_get(/*table_index=*/0, /*width=*/2);
  auto const* left_raw  = left.get();
  auto right            = make_get(/*table_index=*/1, /*width=*/3);
  auto const* right_raw = right.get();
  auto join             = make_join(duckdb::JoinType::SINGLE, std::move(left), std::move(right));
  join->ResolveOperatorTypes();

  REQUIRE(join->types.size() == 5);
  for (auto const ordinal : std::views::iota(std::size_t{0}, join->types.size())) {
    auto const* resolved = resolve_pass_through_scan(*join, ordinal);
    REQUIRE(resolved != right_raw);
    if (resolved != nullptr) { REQUIRE(resolved == left_raw); }
  }
}

//===----------------------------------------------------------------------===//
// Unmodelled operators
//===----------------------------------------------------------------------===//

TEST_CASE("walk refuses unmodelled operators", "[dynamic_filter][build_key_domain]")
{
  SECTION("WINDOW")
  {
    auto window = duckdb::make_uniq<duckdb::LogicalWindow>(/*window_index=*/60);
    window->children.push_back(make_get(/*table_index=*/0, /*width=*/2));
    REQUIRE(resolve_pass_through_scan(*window, 0) == nullptr);
  }
  SECTION("UNION")
  {
    auto set_op = duckdb::make_uniq<duckdb::LogicalSetOperation>(
      /*table_index=*/61,
      /*column_count=*/2,
      make_get(/*table_index=*/0, /*width=*/2),
      make_get(/*table_index=*/1, /*width=*/2),
      duckdb::LogicalOperatorType::LOGICAL_UNION,
      /*setop_all=*/true);
    REQUIRE(resolve_pass_through_scan(*set_op, 0) == nullptr);
  }
  SECTION("UNNEST")
  {
    auto unnest = duckdb::make_uniq<duckdb::LogicalUnnest>(/*unnest_index=*/62);
    unnest->children.push_back(make_get(/*table_index=*/0, /*width=*/2));
    REQUIRE(resolve_pass_through_scan(*unnest, 0) == nullptr);
  }
  SECTION("CTE_REF")
  {
    auto cte_ref = duckdb::make_uniq<duckdb::LogicalCTERef>(
      /*table_index=*/63,
      /*cte_index=*/0,
      duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
      duckdb::vector<duckdb::string>{"a"});
    REQUIRE(resolve_pass_through_scan(*cte_ref, 0) == nullptr);
  }
}

//===----------------------------------------------------------------------===//
// Per-condition resolution and the evidence template
//===----------------------------------------------------------------------===//

namespace {

// Use out-of-range probe ordinals so tracing the probe side cannot accidentally resolve a build
// scan.
duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_traced_join(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> build_sides)
{
  auto join            = make_join(duckdb::JoinType::INNER,
                        make_get(/*table_index=*/0, /*width=*/1),
                        make_get(/*table_index=*/1, /*width=*/3));
  std::size_t probe_at = 90;
  for (auto& build_side : build_sides) {
    join->conditions.push_back(make_condition(make_ref(probe_at++), std::move(build_side)));
  }
  join->ResolveOperatorTypes();
  return join;
}

}  // namespace

TEST_CASE("build keys resolve through the build child only", "[dynamic_filter][build_key_domain]")
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> build_sides;
  build_sides.push_back(make_ref(2));      // plain reference: resolves
  build_sides.push_back(make_computed());  // computed: refused
  build_sides.push_back(duckdb::BoundCastExpression::AddDefaultCastToType(
    make_ref(1), duckdb::LogicalType::BIGINT));  // cast: refused
  auto join = make_traced_join(std::move(build_sides));

  auto const scans = resolve_build_key_scans(*join);

  REQUIRE(scans.size() == 3);
  REQUIRE(scans[0] == join->children[1].get());
  REQUIRE(scans[1] == nullptr);
  REQUIRE(scans[2] == nullptr);

  // The domain is the BUILD table's cardinality; the probe table's distinct stub value proves the
  // walk read condition.right, not condition.left.
  auto const domains = build_key_domain_cardinalities(*join, stub_cardinality_source{});
  REQUIRE(domains == std::vector<std::size_t>{stub_cardinality_of(1), 0, 0});
}

TEST_CASE("evidence source is consulted once per distinct resolved scan",
          "[dynamic_filter][build_key_domain]")
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> build_sides;
  build_sides.push_back(make_ref(0));
  build_sides.push_back(make_ref(2));  // same scan, different ordinal
  auto join = make_traced_join(std::move(build_sides));

  std::size_t calls  = 0;
  auto const domains = build_key_domain_cardinalities(*join, counting_source{.calls = &calls});

  REQUIRE(domains == std::vector<std::size_t>{stub_cardinality_of(1), stub_cardinality_of(1)});
  REQUIRE(calls == 1);
}

TEST_CASE("a refusing evidence source yields zero for every key",
          "[dynamic_filter][build_key_domain]")
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> build_sides;
  build_sides.push_back(make_ref(0));
  auto join = make_traced_join(std::move(build_sides));

  auto const domains = build_key_domain_cardinalities(*join, refusing_source{});

  REQUIRE(domains == std::vector<std::size_t>{0});
}
