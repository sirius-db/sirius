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

#include "planner/sirius_plan_projection_utils.hpp"

#include "expression/ast/substitute.hpp"
#include "expression/ast/visit.hpp"
#include "op/sirius_physical_projection.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

using projection = sirius::op::sirius_physical_projection;

// A select-list entry is "fully supported" only if every slot is non-null.
// A null slot is the from_duckdb fallback signal for an unsupported expression;
// folding through it would silently hide the CPU-fallback trigger.
bool select_list_is_fully_supported(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& select_list)
{
  for (auto const& expr : select_list) {
    if (!expr) { return false; }
  }
  return true;
}

// Trivial expressions are free to duplicate: a bare column reference or a
// constant literal carries no GPU work. Folding may clone these into multiple
// outer reference sites without increasing execution cost.
bool is_trivial_node(sirius::ast::node const& n)
{
  return n.holds<sirius::ast::reference>() || n.holds<sirius::ast::constant>();
}

// Compose an outer select list over an inner one: each outer reference #i is
// replaced by a deep clone of inner.select_list[i].
duckdb::vector<std::unique_ptr<sirius::ast::node>> compose_select_lists(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& outer_select_list,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& inner_select_list)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> combined;
  combined.reserve(outer_select_list.size());
  for (auto const& outer_expr : outer_select_list) {
    combined.push_back(sirius::ast::substitute_references(*outer_expr, inner_select_list));
  }
  return combined;
}

// Replace @p outer (whose single child is @p inner) with one projection whose
// child is inner's grandchild. inner's child slot is moved out.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> fold_once(projection const& outer,
                                                                   projection& inner)
{
  auto folded =
    duckdb::make_uniq<projection>(outer.types,
                                  compose_select_lists(outer.select_list, inner.select_list),
                                  outer.estimated_cardinality);
  folded->children.push_back(std::move(inner.children[0]));
  return std::move(folded);
}

// Drop a passthrough [#0, #1, ...] projection when its child already exposes the
// same column count — avoids a no-op GPU pipeline stage.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> try_elide_identity_projection(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> projection_op)
{
  auto& proj = projection_op->Cast<projection>();
  if (!is_identity_ast_projection(proj.select_list)) { return projection_op; }
  if (proj.children.size() != 1) { return projection_op; }
  if (proj.children[0]->types.size() != proj.types.size()) { return projection_op; }
  return std::move(proj.children[0]);
}

// Repeatedly merge this projection with its immediate child projection (outer
// over inner) until the chain breaks or can_combine_projections refuses.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> fold_projection_with_child(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> projection_op)
{
  while (projection_op->type == sirius::op::SiriusPhysicalOperatorType::PROJECTION &&
         projection_op->children.size() == 1 &&
         projection_op->children[0]->type == sirius::op::SiriusPhysicalOperatorType::PROJECTION) {
    auto& outer = projection_op->Cast<projection>();
    auto& inner = projection_op->children[0]->Cast<projection>();
    if (!can_combine_projections(outer, inner)) { break; }
    projection_op = fold_once(outer, inner);
  }
  return try_elide_identity_projection(std::move(projection_op));
}

}  // namespace

// True when the select list is exactly [#0, #1, ..., #n-1] with no reordering.
bool is_identity_ast_projection(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& select_list)
{
  for (std::size_t i = 0; i < select_list.size(); i++) {
    auto const& expr = select_list[i];
    if (!expr || !expr->holds<sirius::ast::reference>()) { return false; }
    if (expr->get<sirius::ast::reference>().column_index != i) { return false; }
  }
  return true;
}

bool can_combine_projections(sirius::op::sirius_physical_projection const& outer,
                             sirius::op::sirius_physical_projection const& inner)
{
  // fold_once splices inner out of the chain; inner must be a single-child link.
  if (inner.children.size() != 1) { return false; }
  if (!select_list_is_fully_supported(outer.select_list) ||
      !select_list_is_fully_supported(inner.select_list)) {
    return false;
  }

  // Safety guard against expression-duplication blowup: substitution clones each
  // inner expression into every outer reference site. Count how often the outer
  // list references each inner column; folding is only safe when every inner
  // column referenced more than once holds a trivial (free-to-duplicate)
  // expression. Otherwise non-trivial GPU work would be re-executed per use.
  std::vector<std::size_t> use_count(inner.select_list.size(), 0);
  for (auto const& outer_expr : outer.select_list) {
    sirius::ast::visit_references(*outer_expr, [&](sirius::ast::reference const& ref) {
      if (ref.column_index < use_count.size()) { use_count[ref.column_index]++; }
    });
  }
  for (std::size_t i = 0; i < inner.select_list.size(); i++) {
    if (use_count[i] > 1 && !is_trivial_node(*inner.select_list[i])) { return false; }
  }
  return true;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> combine_projections(
  sirius::op::sirius_physical_projection const& outer,
  duckdb::unique_ptr<sirius::op::sirius_physical_projection> inner)
{
  return fold_once(outer, *inner);
}

// Plan-builder entry: wrap @p child in a projection, skipping creation when the
// select list is identity, then fold with an adjacent child projection if safe.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> push_projection(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  std::size_t estimated_cardinality)
{
  if (is_identity_ast_projection(select_list) && child->types.size() == types.size()) {
    return child;
  }

  auto projection_op =
    duckdb::make_uniq<projection>(std::move(types), std::move(select_list), estimated_cardinality);
  projection_op->children.push_back(std::move(child));
  return fold_projection_with_child(std::move(projection_op));
}

// Post-pass over the finished plan: recurse into children first, then fold any
// PROJECTION chain at this node. Catches stacks that push_projection missed
// (e.g. projections under joins or built in separate plan-builder steps).
duckdb::unique_ptr<sirius::op::sirius_physical_operator> fold_adjacent_projections(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan)
{
  for (auto& child : plan->children) {
    child = fold_adjacent_projections(std::move(child));
  }
  if (plan->type == sirius::op::SiriusPhysicalOperatorType::PROJECTION) {
    plan = fold_projection_with_child(std::move(plan));
  }
  return plan;
}

}  // namespace sirius::planner
