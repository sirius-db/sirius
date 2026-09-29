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

#include "duckdb/common/types.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "op/sirius_physical_vector_threshold_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

#include <optional>

namespace sirius::planner {

namespace {

//! A recognized side of the distance predicate: the vector column's output index within its child,
//! plus the fixed dimensionality.
struct resolved_side {
  std::size_t vector_col_idx;
  std::int64_t dim;
};

//! Resolve one FLOAT[dim] vector reference into the join's combined [left cols..., right cols...]
//! row: an index < @p n_left_cols is a left column, otherwise a right column (offset removed).
//! Returns false if the expression is not a FLOAT[dim] positional reference of exactly one side.
bool resolve_vector_column(const duckdb::Expression& expr,
                           std::size_t n_left_cols,
                           std::optional<resolved_side>& left_side,
                           std::optional<resolved_side>& right_side)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return false; }
  auto const& ref = expr.Cast<duckdb::BoundReferenceExpression>();

  auto const& type = ref.return_type;
  if (type.id() != duckdb::LogicalTypeId::ARRAY ||
      duckdb::ArrayType::GetChildType(type).id() != duckdb::LogicalTypeId::FLOAT) {
    return false;
  }
  auto const dim = static_cast<std::int64_t>(duckdb::ArrayType::GetSize(type));

  if (ref.index < n_left_cols) {
    if (left_side) { return false; }  // both operands from the left side: not a join predicate
    left_side = resolved_side{ref.index, dim};
  } else {
    if (right_side) { return false; }
    right_side = resolved_side{ref.index - n_left_cols, dim};
  }
  return true;
}

struct threshold_match {
  resolved_side left;
  resolved_side right;
  float cutoff;
  std::string metric;
};

//! Recognize `array_distance(l.v, r.v) <cmp> const`. Milestone 1: INNER + array_distance (L2) +
//! `<=`/`<` only (const on either side). Returns nullopt when the pattern does not match.
std::optional<threshold_match> match_threshold_join(duckdb::LogicalAnyJoin& op)
{
  if (op.join_type != duckdb::JoinType::INNER && op.join_type != duckdb::JoinType::LEFT) {
    return std::nullopt;
  }
  if (!op.condition || op.children.size() != 2) { return std::nullopt; }
  if (op.condition->GetExpressionClass() != duckdb::ExpressionClass::BOUND_COMPARISON) {
    return std::nullopt;
  }
  auto const& cmp = op.condition->Cast<duckdb::BoundComparisonExpression>();

  // Identify which operand is the distance/similarity function and which is the constant.
  duckdb::Expression* func_expr  = nullptr;
  duckdb::Expression* const_expr = nullptr;
  bool func_on_left              = false;
  if (cmp.left->GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION &&
      cmp.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT) {
    func_expr = cmp.left.get(), const_expr = cmp.right.get(), func_on_left = true;
  } else if (cmp.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION &&
             cmp.left->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT) {
    func_expr = cmp.right.get(), const_expr = cmp.left.get(), func_on_left = false;
  } else {
    return std::nullopt;
  }

  auto const& func = func_expr->Cast<duckdb::BoundFunctionExpression>();
  if (func.children.size() != 2) { return std::nullopt; }
  // Map function -> metric and whether it measures similarity (higher = closer) vs distance.
  std::string metric;
  bool is_similarity = false;
  if (func.function.name == "array_distance") {
    metric = "l2";
  } else if (func.function.name == "array_cosine_distance") {
    metric = "cosine";
  } else if (func.function.name == "array_cosine_similarity") {
    metric        = "cosine";
    is_similarity = true;
  } else {
    return std::nullopt;
  }

  // Normalize the comparison to the form written with the function on the left ("func <op> const").
  auto op_type = op.condition->GetExpressionType();
  if (!func_on_left) {
    if (op_type == duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO) {
      op_type = duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
    } else if (op_type == duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
      op_type = duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO;
    }
  }
  // A threshold ("within") join keeps close pairs: distance functions need `func <= const`, the
  // similarity function needs `func >= const`. Only the inclusive comparison in the correct
  // direction matches -- the kernel is inclusive and we replace the whole join (no residual
  // filter), so a strict comparison would over-include boundary pairs.
  if (op_type != (is_similarity ? duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO
                                : duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO)) {
    return std::nullopt;
  }

  auto const n_left_cols = op.children[0]->GetColumnBindings().size();
  std::optional<resolved_side> left_side, right_side;
  if (!resolve_vector_column(*func.children[0], n_left_cols, left_side, right_side)) {
    return std::nullopt;
  }
  if (!resolve_vector_column(*func.children[1], n_left_cols, left_side, right_side)) {
    return std::nullopt;
  }
  if (!left_side || !right_side || left_side->dim != right_side->dim) { return std::nullopt; }

  auto const& constant = const_expr->Cast<duckdb::BoundConstantExpression>();
  auto const const_val = static_cast<float>(
    constant.value.DefaultCastAs(duckdb::LogicalType::DOUBLE).GetValue<double>());
  // brute_force_threshold keeps cosine *distance* <= cutoff. For the similarity form (higher =
  // closer), `similarity >= eps` is equivalent to `distance <= 1 - eps`.
  auto const cutoff = is_similarity ? (1.0F - const_val) : const_val;

  return threshold_match{*left_side, *right_side, cutoff, metric};
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalAnyJoin& op)
{
  auto match = match_threshold_join(op);
  if (!match) {
    // Only the vector-distance ANY_JOIN is supported on GPU; anything else falls back to CPU.
    throw duckdb::NotImplementedException("Any join not supported");
  }

  auto left_child  = create_plan(*op.children[0]);
  auto right_child = create_plan(*op.children[1]);

  // Output batch byte budget, same source the other operators use (see insert_gpu_pipeline_
  // operators). Falls back to the constructor default when no SiriusContext is registered.
  sirius::operator_params op_params;
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  if (sirius_ctx) { op_params = sirius_ctx->get_config().get_operator_params(); }

  return duckdb::make_uniq<sirius::op::sirius_physical_vector_threshold_join>(
    op,
    std::move(left_child),
    std::move(right_child),
    match->left.vector_col_idx,
    match->right.vector_col_idx,
    match->cutoff,
    match->metric,
    match->left.dim,
    op.join_type,
    op.estimated_cardinality,
    op_params.concat_batch_bytes);
}

}  // namespace sirius::planner
