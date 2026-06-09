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

#include "catch.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_projection.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

using sirius::op::sirius_physical_dummy_scan;
using sirius::op::sirius_physical_projection;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

std::unique_ptr<sirius::ast::node> make_reference(uint32_t column_index)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::reference{column_index, sirius::logical_type::make(sirius::type_id::INTEGER)});
}

duckdb::vector<sirius::logical_type> make_integer_types(std::size_t count)
{
  duckdb::vector<sirius::logical_type> types;
  types.reserve(count);
  for (std::size_t i = 0; i < count; i++) {
    types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  }
  return types;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_dummy_scan(std::size_t column_count)
{
  return duckdb::make_uniq<sirius_physical_dummy_scan>(make_integer_types(column_count), 1);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_projection(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child)
{
  auto projection = duckdb::make_uniq<sirius_physical_projection>(
    make_integer_types(select_list.size()), std::move(select_list), child->estimated_cardinality);
  projection->children.push_back(std::move(child));
  return projection;
}

std::size_t count_projection_operators(sirius::op::sirius_physical_operator const& op)
{
  std::size_t count = op.type == SiriusPhysicalOperatorType::PROJECTION ? 1 : 0;
  for (auto const& child : op.children) {
    count += count_projection_operators(*child);
  }
  return count;
}

}  // namespace

TEST_CASE("projection_fold - push_projection composes reorder with expression projection",
          "[projection_fold]")
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(make_reference(2));
  inner_select.push_back(make_reference(0));
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(3));

  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  std::vector<std::unique_ptr<sirius::ast::node>> add_args;
  add_args.push_back(make_reference(0));
  add_args.push_back(make_reference(1));
  outer_select.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add,
                               std::move(add_args),
                               sirius::logical_type::make(sirius::type_id::INTEGER)}));

  plan = sirius::planner::push_projection(
    std::move(plan), make_integer_types(1), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 1);
  auto& folded = plan->Cast<sirius_physical_projection>();
  REQUIRE(folded.select_list.size() == 1);
  REQUIRE(folded.select_list[0]->holds<sirius::ast::function_call>());
  auto const& fc = folded.select_list[0]->get<sirius::ast::function_call>();
  REQUIRE(fc.arguments()[0]->get<sirius::ast::reference>().column_index == 2);
  REQUIRE(fc.arguments()[1]->get<sirius::ast::reference>().column_index == 0);
}

TEST_CASE("projection_fold - fold_adjacent_projections collapses a three-deep chain",
          "[projection_fold]")
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> first;
  first.push_back(make_reference(1));
  first.push_back(make_reference(0));
  auto plan = make_projection(std::move(first), make_dummy_scan(2));

  duckdb::vector<std::unique_ptr<sirius::ast::node>> second;
  second.push_back(make_reference(1));
  second.push_back(make_reference(0));
  plan = make_projection(std::move(second), std::move(plan));

  duckdb::vector<std::unique_ptr<sirius::ast::node>> third;
  third.push_back(make_reference(0));
  plan = make_projection(std::move(third), std::move(plan));

  plan = sirius::planner::fold_adjacent_projections(std::move(plan));

  REQUIRE(count_projection_operators(*plan) == 1);
  auto& folded = plan->Cast<sirius_physical_projection>();
  REQUIRE(folded.select_list.size() == 1);
  // third [#0] → second col 0 → first col 1 → scan col 0
  REQUIRE(folded.select_list[0]->get<sirius::ast::reference>().column_index == 0);
}

TEST_CASE("projection_fold - null select-list slot prevents folding", "[projection_fold]")
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(make_reference(0));
  inner_select.push_back(nullptr);
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(2));

  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(make_reference(0));
  plan = sirius::planner::push_projection(
    std::move(plan), make_integer_types(1), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 2);
}

TEST_CASE("projection_fold - non-trivial inner expr referenced twice is not folded",
          "[projection_fold]")
{
  // inner: [ add(#0, #1) ]  (non-trivial, single output column)
  std::vector<std::unique_ptr<sirius::ast::node>> inner_add_args;
  inner_add_args.push_back(make_reference(0));
  inner_add_args.push_back(make_reference(1));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add,
                               std::move(inner_add_args),
                               sirius::logical_type::make(sirius::type_id::INTEGER)}));
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(2));

  // outer: [ #0, #0 ]  (references the non-trivial inner column twice)
  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(make_reference(0));
  outer_select.push_back(make_reference(0));
  plan = sirius::planner::push_projection(
    std::move(plan), make_integer_types(2), std::move(outer_select), 1);

  // Folding would duplicate add(#0, #1); the guard must keep both projections.
  REQUIRE(count_projection_operators(*plan) == 2);
}

TEST_CASE("projection_fold - non-trivial inner expr referenced once is folded", "[projection_fold]")
{
  // inner: [ add(#0, #1), #0 ]
  std::vector<std::unique_ptr<sirius::ast::node>> inner_add_args;
  inner_add_args.push_back(make_reference(0));
  inner_add_args.push_back(make_reference(1));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add,
                               std::move(inner_add_args),
                               sirius::logical_type::make(sirius::type_id::INTEGER)}));
  inner_select.push_back(make_reference(0));
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(2));

  // outer: [ #0 ]  (references the non-trivial inner column exactly once)
  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(make_reference(0));
  plan = sirius::planner::push_projection(
    std::move(plan), make_integer_types(1), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 1);
  auto& folded = plan->Cast<sirius_physical_projection>();
  REQUIRE(folded.select_list[0]->holds<sirius::ast::function_call>());
}

TEST_CASE("projection_fold - identity outer projection is elided", "[projection_fold]")
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(make_reference(1));
  inner_select.push_back(make_reference(0));
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(2));

  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(make_reference(0));
  outer_select.push_back(make_reference(1));
  plan = sirius::planner::push_projection(
    std::move(plan), make_integer_types(2), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 1);
  auto& folded = plan->Cast<sirius_physical_projection>();
  REQUIRE(folded.select_list[0]->get<sirius::ast::reference>().column_index == 1);
  REQUIRE(folded.select_list[1]->get<sirius::ast::reference>().column_index == 0);
}
