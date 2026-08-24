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

// Direct lowering of pure-equality EXISTS / NOT EXISTS DELIM joins.
//
// DuckDB plans an equality-correlated EXISTS / NOT EXISTS as a DELIM join whose delim-scan side
// is a "dedup sandwich": DISTINCT(correlation keys) -> INNER join against the inner relation,
// joined back to the outer rows with a null-safe key comparison. Executed literally, the outer
// keys are deduplicated, joined, and membership-tested a second time. This pass proves the
// sandwich's only role is that membership test and collapses the whole construct into one direct
// semi/anti hash join between the outer relation and the inner relation.
//
// Correctness argument (per outer row o, dedup keys K = DISTINCT of the outer key columns,
// correlated conditions C between dedup keys and inner rows i):
//   o survives the delim SEMI  iff  exists k in K: k joins-back-to o.keys and exists i: C(k, i).
// The join-back pins k to o's own key vector (proven per column below), so this reduces to
//   exists i: C(o.keys, i)
// which is exactly the direct semi join on C; the anti case is its complement. A join-back
// condition may reference either the dedup key column itself or the inner-relation column a
// correlated equality equates with it (they coincide on the correlated join's output, and
// DuckDB's projection forwards whichever it likes); both forms pin the key. The reduction is
// exact for any equality-family C, provided (a) every dedup key column is pinned by the
// join-back and (b) every plain `=` join-back is backed by a plain `=` correlated condition on
// the same key column — the NULL outer keys the join-back drops then fail the direct join's own
// `=` comparison identically. Clause (b) refuses two shapes as null_safety: a plain `=`
// join-back over a null-safe correlated condition (the direct null-safe join would match the
// NULL keys the join-back dropped), and a plain `=` join-back on a key column no correlated
// condition constrains (the rewrite deletes that join-back and carries no condition on the
// column at all, so nothing reproduces its NULL-drop).
//
// Staging follows the house planner-pass pattern: collect the candidate roles, match the shape,
// prove the semantics obligations, then rewrite. Every early-out carries a typed refusal reason
// that unit tests pin per plan shape.

#include "planner/sirius_plan_delim_direct.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include <algorithm>
#include <functional>
#include <utility>

namespace sirius::planner {

const char* to_string(delim_direct_refusal refusal)
{
  switch (refusal) {
    case delim_direct_refusal::none: return "none";
    case delim_direct_refusal::unsupported_join_type: return "unsupported_join_type";
    case delim_direct_refusal::orientation_mismatch: return "orientation_mismatch";
    case delim_direct_refusal::sandwich_shape: return "sandwich_shape";
    case delim_direct_refusal::residual_delim_consumer: return "residual_delim_consumer";
    case delim_direct_refusal::non_equality_correlation: return "non_equality_correlation";
    case delim_direct_refusal::inner_condition_shape: return "inner_condition_shape";
    case delim_direct_refusal::join_back_shape: return "join_back_shape";
    case delim_direct_refusal::delim_column_mismatch: return "delim_column_mismatch";
    case delim_direct_refusal::delim_column_type_mismatch: return "delim_column_type_mismatch";
    case delim_direct_refusal::nested_delim_context: return "nested_delim_context";
    case delim_direct_refusal::null_safety: return "null_safety";
    case delim_direct_refusal::residual_predicate: return "residual_predicate";
  }
  return "unknown";
}

namespace {

/// True when the subtree still contains a DELIM_GET, i.e. a consumer of some delim join's
/// duplicate-eliminated data.
bool contains_delim_get(const duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_DELIM_GET) { return true; }
  return std::ranges::any_of(op.children,
                             [](const auto& child) { return contains_delim_get(*child); });
}

bool is_equality_family(duckdb::ExpressionType comparison)
{
  return comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
         comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM;
}

/// Width of one side of @p join's output, honoring its projection map.
std::size_t join_side_width(const duckdb::LogicalComparisonJoin& join, std::size_t side)
{
  const auto& map = side == 0 ? join.left_projection_map : join.right_projection_map;
  if (!map.empty()) { return map.size(); }
  return join.children[side]->types.size();
}

/// Trace a flat index into the sandwich's output back through the (possibly stacked) pure
/// column-reference projections and the inner join's projection maps to (side, column) of the
/// inner join's children. Returns false when the index cannot be followed (out of range
/// anywhere on the path).
bool trace_sandwich_output(const std::vector<const duckdb::LogicalProjection*>& projections,
                           const duckdb::LogicalComparisonJoin& inner_join,
                           std::size_t index,
                           std::size_t& side_out,
                           std::size_t& column_out)
{
  for (const auto* projection : projections) {
    if (index >= projection->expressions.size()) { return false; }
    const auto& expr = *projection->expressions[index];
    if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return false; }
    index = expr.Cast<duckdb::BoundReferenceExpression>().index;
  }
  const std::size_t left_width = join_side_width(inner_join, 0);
  std::size_t side             = 0;
  std::size_t position         = index;
  if (index >= left_width) {
    side     = 1;
    position = index - left_width;
    if (position >= join_side_width(inner_join, 1)) { return false; }
  }
  const auto& map = side == 0 ? inner_join.left_projection_map : inner_join.right_projection_map;
  side_out        = side;
  column_out      = map.empty() ? position : map[position];
  return true;
}

}  // namespace

delim_direct_analysis classify_delim_direct_lowering(duckdb::LogicalComparisonJoin& op)
{
  delim_direct_analysis analysis;
  auto refuse = [&analysis](delim_direct_refusal reason) {
    analysis.refusal = reason;
    return analysis;
  };

  // --- Collect: identify the membership orientation and the candidate sandwich. ---
  // The dedup keys must come from the side the join emits (the outer relation), which puts the
  // sandwich on the opposite side. Any other join type is a different delim construct (the
  // scalar-aggregate correlations plan as LEFT / RIGHT / SINGLE and must keep the delim
  // lowering: they extend outer rows instead of membership-testing them).
  std::size_t sandwich_index = 0;
  switch (op.join_type) {
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
      // Emits children[1]; dedup source must be the RHS (flipped delim).
      if (!op.delim_flipped) { return refuse(delim_direct_refusal::orientation_mismatch); }
      sandwich_index = 0;
      break;
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
      // Emits children[0]; dedup source must be the LHS (unflipped delim).
      if (op.delim_flipped) { return refuse(delim_direct_refusal::orientation_mismatch); }
      sandwich_index = 1;
      break;
    default: return refuse(delim_direct_refusal::unsupported_join_type);
  }
  if (op.children.size() != 2) { return refuse(delim_direct_refusal::sandwich_shape); }
  if (op.predicate) { return refuse(delim_direct_refusal::residual_predicate); }

  // --- Match: the sandwich must be [pure-reference PROJECTIONs ->] INNER join over a bare
  // DELIM_GET and the inner relation. (DuckDB commonly stacks two reference-only projections
  // over the correlated join.) ---
  duckdb::LogicalOperator* sandwich = op.children[sandwich_index].get();
  std::vector<const duckdb::LogicalProjection*> projections;
  while (sandwich->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    const auto& projection = sandwich->Cast<duckdb::LogicalProjection>();
    if (!std::ranges::all_of(projection.expressions, [](const auto& expr) {
          return expr && expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF;
        })) {
      return refuse(delim_direct_refusal::sandwich_shape);
    }
    if (sandwich->children.size() != 1) { return refuse(delim_direct_refusal::sandwich_shape); }
    projections.push_back(&projection);
    sandwich = sandwich->children[0].get();
  }
  if (sandwich->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return refuse(delim_direct_refusal::sandwich_shape);
  }
  auto& inner_join = sandwich->Cast<duckdb::LogicalComparisonJoin>();
  if (inner_join.join_type != duckdb::JoinType::INNER || inner_join.children.size() != 2) {
    return refuse(delim_direct_refusal::sandwich_shape);
  }
  if (inner_join.predicate) { return refuse(delim_direct_refusal::residual_predicate); }

  const bool left_is_delim_get =
    inner_join.children[0]->type == duckdb::LogicalOperatorType::LOGICAL_DELIM_GET;
  const bool right_is_delim_get =
    inner_join.children[1]->type == duckdb::LogicalOperatorType::LOGICAL_DELIM_GET;
  if (left_is_delim_get == right_is_delim_get) {
    return refuse(delim_direct_refusal::sandwich_shape);
  }
  const std::size_t delim_get_index = right_is_delim_get ? 1 : 0;
  const auto& delim_get = inner_join.children[delim_get_index]->Cast<duckdb::LogicalDelimGet>();
  const auto& inner_relation = *inner_join.children[1 - delim_get_index];

  // The delim data must have no consumer beyond this sandwich: the inner relation the rewrite
  // keeps may not replay it (nested correlation), and the sandwich shape above admits no other
  // reader by construction.
  if (contains_delim_get(inner_relation)) {
    return refuse(delim_direct_refusal::residual_delim_consumer);
  }

  // Dedup key vector: DELIM_GET columns are the duplicate-eliminated outer columns, in order.
  const std::size_t key_width = op.duplicate_eliminated_columns.size();
  if (key_width == 0 || delim_get.chunk_types.size() != key_width) {
    return refuse(delim_direct_refusal::join_back_shape);
  }
  if (!std::ranges::all_of(op.duplicate_eliminated_columns, [](const auto& column) {
        return column && column->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF;
      })) {
    return refuse(delim_direct_refusal::join_back_shape);
  }
  // The rewrite substitutes each duplicate-eliminated source column for the DELIM_GET key it
  // produced, so the two must agree on type: a mismatch means the correlated condition was typed
  // against a different value than the one the direct join would compare. Proven rather than
  // assumed, like every other obligation in this pass.
  for (std::size_t k = 0; k < key_width; k++) {
    if (delim_get.chunk_types[k] !=
        op.duplicate_eliminated_columns[k]->Cast<duckdb::BoundReferenceExpression>().return_type) {
      return refuse(delim_direct_refusal::delim_column_type_mismatch);
    }
  }

  // --- Match: the correlated conditions must be equality-family with a plain dedup-key side. ---
  if (inner_join.conditions.empty()) { return refuse(delim_direct_refusal::sandwich_shape); }
  analysis.dedup_column_of_condition.reserve(inner_join.conditions.size());
  // Per correlated condition: the inner-relation side's column when it is a plain reference
  // (used to resolve join-backs expressed through the inner column — see below).
  constexpr std::size_t no_column = static_cast<std::size_t>(-1);
  std::vector<std::size_t> inner_column_of_condition;
  inner_column_of_condition.reserve(inner_join.conditions.size());
  for (const auto& condition : inner_join.conditions) {
    if (!is_equality_family(condition.comparison)) {
      return refuse(delim_direct_refusal::non_equality_correlation);
    }
    const auto& key_side   = delim_get_index == 0 ? *condition.left : *condition.right;
    const auto& inner_side = delim_get_index == 0 ? *condition.right : *condition.left;
    if (key_side.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      return refuse(delim_direct_refusal::inner_condition_shape);
    }
    const auto key_column = key_side.Cast<duckdb::BoundReferenceExpression>().index;
    if (key_column >= key_width) { return refuse(delim_direct_refusal::inner_condition_shape); }
    analysis.dedup_column_of_condition.push_back(key_column);
    inner_column_of_condition.push_back(
      inner_side.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF
        ? inner_side.Cast<duckdb::BoundReferenceExpression>().index
        : no_column);
  }

  // --- Prove: the join-back pins every dedup key column to its own outer source column. ---
  // A join-back condition's sandwich side traces either to the dedup key column itself, or to
  // the inner-relation column a correlated condition equates with that key (on the correlated
  // join's output the two are interchangeable; DuckDB's projection may forward either). Both
  // pin the dedup key to the outer source column on the other side.
  std::vector<bool> pinned(key_width, false);
  std::vector<bool> pinned_with_plain_equal(key_width, false);
  for (const auto& condition : op.conditions) {
    if (!is_equality_family(condition.comparison)) {
      return refuse(delim_direct_refusal::join_back_shape);
    }
    const auto& sandwich_side = sandwich_index == 0 ? *condition.left : *condition.right;
    const auto& outer_side    = sandwich_index == 0 ? *condition.right : *condition.left;
    if (sandwich_side.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
        outer_side.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      return refuse(delim_direct_refusal::join_back_shape);
    }
    std::size_t side   = 0;
    std::size_t column = 0;
    if (!trace_sandwich_output(projections,
                               inner_join,
                               sandwich_side.Cast<duckdb::BoundReferenceExpression>().index,
                               side,
                               column)) {
      return refuse(delim_direct_refusal::join_back_shape);
    }
    // Dedup key columns this join-back condition can pin.
    std::vector<std::size_t> candidates;
    if (side == delim_get_index) {
      if (column >= key_width) { return refuse(delim_direct_refusal::join_back_shape); }
      candidates.push_back(column);
    } else {
      for (std::size_t i = 0; i < inner_join.conditions.size(); i++) {
        if (inner_column_of_condition[i] == column) {
          candidates.push_back(analysis.dedup_column_of_condition[i]);
        }
      }
    }
    const auto outer_column = outer_side.Cast<duckdb::BoundReferenceExpression>().index;
    bool pinned_any         = false;
    for (const auto key_column : candidates) {
      const auto& source =
        op.duplicate_eliminated_columns[key_column]->Cast<duckdb::BoundReferenceExpression>();
      if (outer_column != source.index) { continue; }
      pinned[key_column] = true;
      if (condition.comparison == duckdb::ExpressionType::COMPARE_EQUAL) {
        pinned_with_plain_equal[key_column] = true;
      }
      pinned_any = true;
    }
    if (!pinned_any) { return refuse(delim_direct_refusal::join_back_shape); }
  }
  if (!std::ranges::all_of(pinned, std::identity{})) {
    return refuse(delim_direct_refusal::delim_column_mismatch);
  }

  // --- Prove: NULL-key semantics. A plain `=` join-back drops NULL-keyed outer rows before the
  // membership test; the direct join reproduces that only when the correlated condition on that
  // key is also plain `=` (NULL never matches). A null-safe correlated condition would let a
  // NULL outer key match after the rewrite — refuse. (The converse — null-safe join-back over a
  // plain `=` correlation — is exact: the NULL key group joins back but can never satisfy `=`.)
  for (std::size_t i = 0; i < inner_join.conditions.size(); i++) {
    const auto column = analysis.dedup_column_of_condition[i];
    if (pinned_with_plain_equal[column] &&
        inner_join.conditions[i].comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
      return refuse(delim_direct_refusal::null_safety);
    }
  }
  // A dedup key column constrained by NO correlated condition contributes only its join-back
  // to the original semantics. A null-safe join-back there is vacuous (the row's own key group
  // always matches itself), but a plain `=` join-back drops NULL-keyed outer rows — an effect
  // the direct join cannot reproduce once that join-back is deleted. Refuse those columns.
  std::vector<bool> constrained(key_width, false);
  for (const auto column : analysis.dedup_column_of_condition) {
    constrained[column] = true;
  }
  for (std::size_t column = 0; column < key_width; column++) {
    if (pinned_with_plain_equal[column] && !constrained[column]) {
      return refuse(delim_direct_refusal::null_safety);
    }
  }

  analysis.refusal         = delim_direct_refusal::none;
  analysis.sandwich_index  = sandwich_index;
  analysis.delim_get_index = delim_get_index;
  analysis.inner_join      = &inner_join;
  return analysis;
}

void apply_delim_direct_lowering(duckdb::LogicalComparisonJoin& op,
                                 delim_direct_analysis&& analysis)
{
  D_ASSERT(analysis.eligible() && analysis.inner_join != nullptr);
  auto& inner_join            = *analysis.inner_join;
  const std::size_t sandwich  = analysis.sandwich_index;
  const std::size_t delim_get = analysis.delim_get_index;

  // The direct join is always emitted in right-family form: build = the outer relation
  // (children[1]), probe = the inner relation (children[0]). The outer is the delim's dedup
  // source — the filtered, membership-tested side — so building on it keeps the build small and
  // is the only direction where a published membership filter prunes the (typically far larger)
  // inner probe. A SEMI/ANTI-oriented delim is therefore flipped to RIGHT_SEMI / RIGHT_ANTI; the
  // emitted column set is unchanged (still the outer side).
  //
  // Direct conditions: the correlated join's conditions with each dedup key replaced by the
  // outer source column it was proven to pin, order preserved 1:1.
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.reserve(inner_join.conditions.size());
  for (std::size_t i = 0; i < inner_join.conditions.size(); i++) {
    auto& correlated   = inner_join.conditions[i];
    const auto& source = op.duplicate_eliminated_columns[analysis.dedup_column_of_condition[i]]
                           ->Cast<duckdb::BoundReferenceExpression>();
    auto outer_key =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(source.return_type, source.index);
    duckdb::JoinCondition direct;
    direct.comparison = correlated.comparison;
    direct.left       = std::move(delim_get == 0 ? correlated.right : correlated.left);
    direct.right      = std::move(outer_key);
    conditions.push_back(std::move(direct));
  }

  // Splice: probe (inner relation) at children[0], build (outer relation) at children[1].
  // Detach the inner relation first: the correlated join (and analysis.inner_join with it)
  // lives inside the sandwich subtree the assignments below destroy. Any DuckDB filter-pushdown
  // metadata on the correlated join dies with that subtree — nothing in the live Sirius path
  // reads it; plan_comparison_join re-derives dynamic-filter targets natively, and
  // scan_route_join_type_admissible admits the direct join's RIGHT_SEMI / RIGHT_ANTI types.
  const std::size_t outer_index = 1 - sandwich;
  auto inner_relation           = std::move(inner_join.children[1 - delim_get]);
  auto outer_relation           = std::move(op.children[outer_index]);
  op.children[0]                = std::move(inner_relation);
  op.children[1]                = std::move(outer_relation);

  // The outer side's projection map moves with it to the right slot. Clearing the probe's map is
  // exact rather than lossy: a right-family join gathers no columns from its probe side
  // (gather_join_output sets collect_left = false for RIGHT_SEMI / RIGHT_ANTI, leaving
  // lhs_output_columns unread), so that map contributes nothing. Sirius applies these maps only
  // at the join's output gather, never below the join, so neither map has ever narrowed what the
  // probe's PARTITION / CONCAT carry.
  auto outer_map = std::move(outer_index == 0 ? op.left_projection_map : op.right_projection_map);
  op.left_projection_map.clear();
  op.right_projection_map = std::move(outer_map);

  op.join_type =
    (op.join_type == duckdb::JoinType::SEMI || op.join_type == duckdb::JoinType::RIGHT_SEMI)
      ? duckdb::JoinType::RIGHT_SEMI
      : duckdb::JoinType::RIGHT_ANTI;
  op.conditions = std::move(conditions);
  op.duplicate_eliminated_columns.clear();
  op.delim_flipped = false;
  op.type          = duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN;

  // The conditions these describe no longer exist: the dedup-key side was replaced by its outer
  // source column and the sides were swapped into right-family orientation. Anything indexed
  // against the old conditions is now wrong rather than merely stale, so drop it instead of
  // leaving a future reader -- plan_comparison_join moves join_stats into the physical join -- to
  // be misled by it.
  op.join_stats.clear();
  op.filter_pushdown.reset();
}

}  // namespace sirius::planner
