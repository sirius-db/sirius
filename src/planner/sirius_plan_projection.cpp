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

#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_vector_threshold_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <memory>
#include <optional>
#include <string>

namespace sirius::planner {

namespace {

// Translate at the planner boundary and reject unsupported expressions before
// they enter a physical projection. The source vector is consumed in order.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    auto translated = e ? sirius::ast::from_duckdb(*e) : nullptr;
    if (e && translated == nullptr) {
      throw duckdb::NotImplementedException(
        "Unsupported expression in projection (falling back to CPU): " + e->ToString());
    }
    out.push_back(std::move(translated));
  }
  return out;
}

// Where the vector threshold join exposes its reusable distance, and which columns it joins on.
struct threshold_join_distance_info {
  std::string metric;              // "l2" or "cosine", from the join predicate
  std::size_t left_vec_index;      // output index of the left vector column
  std::size_t right_vec_index;     // output index of the right vector column
  std::size_t distance_col_index;  // where enable_distance_output() appends the distance column
};

// If expr is `fn(a, b)` where fn is a supported vector-distance function whose metric matches the
// join's, and a and b reference the join's two vector columns, return whether the SELECT wants
// similarity (`1 - distance`). Otherwise nullopt. The two arguments may appear in either order
// because the distance functions are symmetric.
std::optional<bool> match_distance_call(const duckdb::Expression& expr,
                                        const threshold_join_distance_info& info)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_FUNCTION) { return std::nullopt; }
  auto const& func = expr.Cast<duckdb::BoundFunctionExpression>();
  if (func.children.size() != 2) { return std::nullopt; }

  std::string fn_metric;
  bool as_similarity = false;
  if (func.function.name == "array_distance") {
    fn_metric = "l2";
  } else if (func.function.name == "array_cosine_distance") {
    fn_metric = "cosine";
  } else if (func.function.name == "array_cosine_similarity") {
    fn_metric     = "cosine";
    as_similarity = true;
  } else {
    return std::nullopt;
  }
  // A cosine SELECT cannot reuse a Euclidean distance and vice versa.
  if (fn_metric != info.metric) { return std::nullopt; }

  if (func.children[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
      func.children[1]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto const i0           = func.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
  auto const i1           = func.children[1]->Cast<duckdb::BoundReferenceExpression>().index;
  bool const matches_cols = (i0 == info.left_vec_index && i1 == info.right_vec_index) ||
                            (i0 == info.right_vec_index && i1 == info.left_vec_index);
  if (!matches_cols) { return std::nullopt; }
  return as_similarity;
}

// Recursively replace every matching distance call in expr with a reference to the join's emitted
// distance column. Records whether anything was rewritten and the single similarity flag the
// rewrites agree on. A call whose similarity flag conflicts with the first match is left untouched,
// so the query falls back cleanly rather than emitting one column with the wrong units.
void rewrite_distance_calls(duckdb::unique_ptr<duckdb::Expression>& expr,
                            const threshold_join_distance_info& info,
                            bool& rewrote_any,
                            std::optional<bool>& target_as_similarity)
{
  if (auto as_sim = match_distance_call(*expr, info)) {
    if (!target_as_similarity) { target_as_similarity = *as_sim; }
    if (*target_as_similarity == *as_sim) {
      expr        = duckdb::make_uniq<duckdb::BoundReferenceExpression>(expr->return_type,
                                                                 info.distance_col_index);
      rewrote_any = true;
    }
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    *expr, [&](duckdb::unique_ptr<duckdb::Expression>& child) {
      rewrite_distance_calls(child, info, rewrote_any, target_as_similarity);
    });
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalProjection& op)
{
  D_ASSERT(op.children.size() == 1);
  auto plan = create_plan(*op.children[0]);

  // When this projection sits directly on the vector threshold join and references the same
  // distance function as the join predicate, reuse the distance the join already computed instead
  // of letting the (GPU-unsupported) array_distance expression force a whole-query CPU fallback. We
  // rewrite the matching SELECT expression to a reference to an extra distance column the join then
  // emits.
  if (plan->type == sirius::op::SiriusPhysicalOperatorType::VECTOR_THRESHOLD_JOIN) {
    auto& join = static_cast<sirius::op::sirius_physical_vector_threshold_join&>(*plan);
    threshold_join_distance_info info{
      join.metric,
      join.left_vector_col_idx,
      join.left_output_col_idxs.size() + join.right_vector_col_idx,
      join.left_output_col_idxs.size() + join.right_output_col_idxs.size()};
    bool rewrote_any = false;
    std::optional<bool> target_as_similarity;
    for (auto& e : op.expressions) {
      if (e) { rewrite_distance_calls(e, info, rewrote_any, target_as_similarity); }
    }
    if (rewrote_any) { join.enable_distance_output(*target_as_similarity); }
  }

#ifdef DEBUG
  for (auto& expr : op.expressions) {
    D_ASSERT(!expr->IsWindow());
    D_ASSERT(!expr->IsAggregate());
  }
#endif
  if (plan->types.size() == op.types.size()) {
    // check if this projection can be omitted entirely
    // this happens if a projection simply emits the columns in the same order
    // e.g. PROJECTION(#0, #1, #2, #3, ...)
    bool omit_projection = true;
    for (std::size_t i = 0; i < op.types.size(); i++) {
      if (op.expressions[i]->type == duckdb::ExpressionType::BOUND_REF) {
        auto& bound_ref = op.expressions[i]->Cast<duckdb::BoundReferenceExpression>();
        if (bound_ref.index == i) { continue; }
      }
      omit_projection = false;
      break;
    }
    if (omit_projection) {
      // the projection only directly projects the child' columns: omit it entirely
      return plan;
    }
  }

  return push_projection(std::move(plan),
                         sirius::from_duckdb_vec(op.types),
                         translate_expressions(std::move(op.expressions)),
                         op.estimated_cardinality);
}

}  // namespace sirius::planner
