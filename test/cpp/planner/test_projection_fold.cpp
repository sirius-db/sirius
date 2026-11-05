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
#include "expression/ast/case_expr.hpp"
#include "expression/ast/coalesce.hpp"
#include "expression/ast/comparison.hpp"
#include "expression/ast/conjunction.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/function_id.hpp"
#include "expression/value.hpp"
#include "helper/logical_type.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_projection.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using sirius::op::sirius_physical_dummy_scan;
using sirius::op::sirius_physical_filter;
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

duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_filter(
  duckdb::vector<sirius::logical_type> types,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child)
{
  auto filter_expr = std::make_unique<sirius::ast::node>(sirius::ast::constant{
    sirius::value{true}, sirius::logical_type::make(sirius::type_id::BOOLEAN)});
  auto filter      = duckdb::make_uniq<sirius_physical_filter>(
    std::move(types), std::move(filter_expr), child->estimated_cardinality);
  filter->children.push_back(std::move(child));
  return filter;
}

std::size_t count_projection_operators(sirius::op::sirius_physical_operator const& op)
{
  std::size_t count = op.type == SiriusPhysicalOperatorType::PROJECTION ? 1 : 0;
  for (auto const& child : op.children) {
    count += count_projection_operators(*child);
  }
  return count;
}

std::unique_ptr<sirius::ast::node> make_add(std::unique_ptr<sirius::ast::node> left,
                                            std::unique_ptr<sirius::ast::node> right)
{
  std::vector<std::unique_ptr<sirius::ast::node>> args;
  args.push_back(std::move(left));
  args.push_back(std::move(right));
  return std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add,
                               std::move(args),
                               sirius::logical_type::make(sirius::type_id::INTEGER)});
}

std::unique_ptr<sirius::ast::node> make_equal(std::unique_ptr<sirius::ast::node> left,
                                              std::unique_ptr<sirius::ast::node> right)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::equal, std::move(left), std::move(right)});
}

std::unique_ptr<sirius::ast::node> make_and(std::unique_ptr<sirius::ast::node> left,
                                            std::unique_ptr<sirius::ast::node> right)
{
  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  return std::make_unique<sirius::ast::node>(
    sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});
}

std::unique_ptr<sirius::ast::node> make_coalesce(std::unique_ptr<sirius::ast::node> first,
                                                 std::unique_ptr<sirius::ast::node> second)
{
  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.push_back(std::move(first));
  children.push_back(std::move(second));
  return std::make_unique<sirius::ast::node>(sirius::ast::coalesce{
    std::move(children), sirius::logical_type::make(sirius::type_id::INTEGER)});
}

std::unique_ptr<sirius::ast::node> make_case(std::vector<sirius::ast::case_expr::when_then> cases,
                                             std::unique_ptr<sirius::ast::node> otherwise,
                                             bool eager = false)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::case_expr{std::move(cases),
                           std::move(otherwise),
                           sirius::logical_type::make(sirius::type_id::INTEGER),
                           eager});
}

std::unique_ptr<sirius::ast::node> make_case(std::unique_ptr<sirius::ast::node> when,
                                             std::unique_ptr<sirius::ast::node> then,
                                             std::unique_ptr<sirius::ast::node> otherwise,
                                             bool eager = false)
{
  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back({std::move(when), std::move(then)});
  return make_case(std::move(cases), std::move(otherwise), eager);
}

// inner: [ add(#0, #1), #0, #1 ], so an outer #0 reads GPU work and outer #1 and #2 read columns.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_sum_projection()
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select;
  select.push_back(make_add(make_reference(0), make_reference(1)));
  select.push_back(make_reference(0));
  select.push_back(make_reference(1));
  return make_projection(std::move(select), make_dummy_scan(2));
}

// An outer expression over make_sum_projection that reads the sum (#0) once, in @p slot.
std::unique_ptr<sirius::ast::node> read_sum_in(std::string const& slot)
{
  auto const guard = [] { return make_equal(make_reference(1), make_reference(2)); };
  auto const test  = [] { return make_equal(make_reference(0), make_reference(2)); };
  if (slot == "then") { return make_case(guard(), make_reference(0), make_reference(1)); }
  if (slot == "else") { return make_case(guard(), make_reference(1), make_reference(0)); }
  if (slot == "later when") {
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back({guard(), make_reference(1)});
    cases.push_back({test(), make_reference(2)});
    return make_case(std::move(cases), make_reference(1));
  }
  if (slot == "when and operand") {
    return make_case(make_and(guard(), test()), make_reference(1), make_reference(2));
  }
  if (slot == "coalesce argument") { return make_coalesce(make_reference(1), make_reference(0)); }
  if (slot == "then of a nested case") {
    return make_add(make_case(guard(), make_reference(0), make_reference(1)), make_reference(2));
  }
  if (slot == "first when") { return make_case(test(), make_reference(1), make_reference(2)); }
  if (slot == "first coalesce argument") {
    return make_coalesce(make_reference(0), make_reference(1));
  }
  if (slot == "value and operand") { return make_and(guard(), test()); }
  if (slot == "eager case then") {
    return make_case(guard(), make_reference(0), make_reference(1), /*eager=*/true);
  }
  FAIL("unknown slot " << slot);
  return nullptr;
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

TEST_CASE(
  "projection_fold - fold_adjacent_projections collapses nested stack under non-projection parent",
  "[projection_fold]")
{
  // FILTER → PROJ(outer) → PROJ(inner) → SCAN
  // Mimics stacks built without push_projection (e.g. under joins) that only the
  // post-pass in sirius_physical_plan_generator folds.
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(make_reference(1));
  inner_select.push_back(make_reference(0));
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(2));

  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(make_reference(0));
  plan = make_projection(std::move(outer_select), std::move(plan));

  plan = make_filter(make_integer_types(1), std::move(plan));

  REQUIRE(plan->type == SiriusPhysicalOperatorType::FILTER);
  REQUIRE(count_projection_operators(*plan) == 2);

  plan = sirius::planner::fold_adjacent_projections(std::move(plan));

  REQUIRE(plan->type == SiriusPhysicalOperatorType::FILTER);
  REQUIRE(plan->children.size() == 1);
  REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(count_projection_operators(*plan) == 1);
  auto& folded = plan->children[0]->Cast<sirius_physical_projection>();
  REQUIRE(folded.select_list.size() == 1);
  // outer [#0] → inner col 0 → scan col 1
  REQUIRE(folded.select_list[0]->get<sirius::ast::reference>().column_index == 1);
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

TEST_CASE("projection_fold - non-trivial inner expr read where a branch skips rows is not folded",
          "[projection_fold]")
{
  // DuckDB computes the inner projection on every row, so its errors raise even for rows that a
  // CASE or COALESCE branch above it never sees. Inside the branch they would not.
  auto const slot = GENERATE(as<std::string>{},
                             "then",
                             "else",
                             "later when",
                             "when and operand",
                             "coalesce argument",
                             "then of a nested case");
  INFO(slot);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(read_sum_in(slot));
  auto plan = sirius::planner::push_projection(
    make_sum_projection(), make_integer_types(1), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 2);
}

TEST_CASE("projection_fold - non-trivial inner expr read on every row is folded",
          "[projection_fold]")
{
  auto const slot = GENERATE(as<std::string>{},
                             "first when",
                             "first coalesce argument",
                             "value and operand",
                             "eager case then");
  INFO(slot);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(read_sum_in(slot));
  auto plan = sirius::planner::push_projection(
    make_sum_projection(), make_integer_types(1), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 1);
}

TEST_CASE("projection_fold - column references fold into a branch that skips rows",
          "[projection_fold]")
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> inner_select;
  inner_select.push_back(make_reference(1));
  inner_select.push_back(make_reference(0));
  auto plan = make_projection(std::move(inner_select), make_dummy_scan(2));

  // outer: [ CASE WHEN #0 = #1 THEN #0 ELSE #1 END ]
  duckdb::vector<std::unique_ptr<sirius::ast::node>> outer_select;
  outer_select.push_back(make_case(
    make_equal(make_reference(0), make_reference(1)), make_reference(0), make_reference(1)));
  plan = sirius::planner::push_projection(
    std::move(plan), make_integer_types(1), std::move(outer_select), 1);

  REQUIRE(count_projection_operators(*plan) == 1);
  auto const& folded = plan->Cast<sirius_physical_projection>();
  auto const& alt    = folded.select_list[0]->get<sirius::ast::case_expr>();
  REQUIRE(alt.cases[0].then_->get<sirius::ast::reference>().column_index == 1);
  REQUIRE(alt.else_->get<sirius::ast::reference>().column_index == 0);
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
