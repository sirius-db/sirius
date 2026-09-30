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

#include "cuda/vss/knn_merge.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_unnest_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_unnest.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_vector_topk_join.hpp"
#include "op/sirius_physical_vector_topk_merge.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_context.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace sirius::planner {

namespace {

//! A per-row top-k vector join as DuckDB plans it: a LATERAL `ORDER BY dist LIMIT k` subquery
//! decorrelated into a DELIM_JOIN keyed on the left row's vector.
struct topk_delim_match {
  std::size_t left_child_idx;      // child holding the outer (left) table
  std::size_t subquery_child_idx;  // child holding the decorrelated subquery (has the DELIM_GET)
  std::size_t left_vector_col_idx;
  std::size_t subquery_vector_col_idx;  // delim copy of the vector, re-emitted by the subquery
  std::int64_t dim;
  duckdb::JoinType join_type;  // normalized to the left table's view: INNER or LEFT
};

//! FLOAT[dim] positional reference, or nullopt.
std::optional<std::pair<std::size_t, std::int64_t>> as_float_array_ref(
  const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
  auto const& ref  = expr.Cast<duckdb::BoundReferenceExpression>();
  auto const& type = ref.return_type;
  if (type.id() != duckdb::LogicalTypeId::ARRAY ||
      duckdb::ArrayType::GetChildType(type).id() != duckdb::LogicalTypeId::FLOAT) {
    return std::nullopt;
  }
  return std::make_pair(static_cast<std::size_t>(ref.index),
                        static_cast<std::int64_t>(duckdb::ArrayType::GetSize(type)));
}

//! Step 1: recognize the DELIM_JOIN itself -- a single `l.v IS NOT DISTINCT FROM delim.v`
//! condition on a FLOAT[dim] vector, with the left table on the preserved side.
std::optional<topk_delim_match> match_topk_delim_join(duckdb::LogicalComparisonJoin& op)
{
  if (op.type != duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN) { return std::nullopt; }
  if (op.children.size() != 2 || op.conditions.size() != 1 ||
      op.duplicate_eliminated_columns.size() != 1) {
    return std::nullopt;
  }

  // The join-order optimizer may swap the children; delim_flipped says where the subquery went.
  std::size_t const subquery_idx = op.delim_flipped ? 0 : 1;
  std::size_t const left_idx     = 1 - subquery_idx;

  // A swap turns LEFT into RIGHT, so the preserved side must always be the left table.
  duckdb::JoinType join_type;
  if (op.join_type == duckdb::JoinType::INNER) {
    join_type = duckdb::JoinType::INNER;
  } else if ((op.join_type == duckdb::JoinType::LEFT && left_idx == 0) ||
             (op.join_type == duckdb::JoinType::RIGHT && left_idx == 1)) {
    join_type = duckdb::JoinType::LEFT;
  } else {
    return std::nullopt;
  }

  auto const& cond = op.conditions[0];
  if (cond.comparison != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) { return std::nullopt; }
  auto const lhs = as_float_array_ref(*cond.left);
  auto const rhs = as_float_array_ref(*cond.right);
  if (!lhs || !rhs || lhs->second != rhs->second) { return std::nullopt; }

  // Condition sides follow child order: cond.left is bound against children[0].
  auto const& left_ref     = left_idx == 0 ? *lhs : *rhs;
  auto const& subquery_ref = left_idx == 0 ? *rhs : *lhs;

  // The deduplicated column must be the left table's vector.
  auto const delim_col = as_float_array_ref(*op.duplicate_eliminated_columns[0]);
  if (!delim_col || delim_col->first != left_ref.first) { return std::nullopt; }

  return topk_delim_match{
    left_idx, subquery_idx, left_ref.first, subquery_ref.first, left_ref.second, join_type};
}

//! The decorrelated subquery's operators, top to bottom. DuckDB's TopN-window elimination turns
//! `ORDER BY dist LIMIT k` into a grouped arg_min(payload, dist, k), plus an UNNEST when k > 1.
struct topk_subquery_match {
  std::vector<duckdb::LogicalProjection*> top_projections;  // outermost first
  duckdb::LogicalUnnest* unnest;                            // null when k == 1
  duckdb::LogicalAggregate* aggregate;
  duckdb::LogicalProjection* distance_projection;  // computes dist over (r cols, delim vector)
  duckdb::LogicalOperator* inner_join;             // CROSS_PRODUCT, or ANY_JOIN(dist IS NOT NULL)
  std::size_t delim_get_child_idx;                 // which inner_join child is the DELIM_GET
  std::size_t right_child_idx;                     // which inner_join child is the right table
};

//! arg_min / arg_max, with or without the nulls-last variant DuckDB picks for nullable distances.
bool is_topk_aggregate(const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) { return false; }
  auto const& aggr = expr.Cast<duckdb::BoundAggregateExpression>();
  if (aggr.filter || aggr.order_bys || aggr.IsDistinct()) { return false; }
  auto const& name = aggr.function.name;
  return name == "arg_min" || name == "arg_max" || name == "arg_min_nulls_last" ||
         name == "arg_max_nulls_last";
}

//! Step 2: walk the subquery side down to the (DELIM_GET, right table) join, checking each node.
std::optional<topk_subquery_match> match_topk_subquery(duckdb::LogicalOperator& root)
{
  topk_subquery_match m{};
  duckdb::LogicalOperator* node = &root;

  while (node->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    m.top_projections.push_back(&node->Cast<duckdb::LogicalProjection>());
    node = node->children[0].get();
  }

  if (node->type == duckdb::LogicalOperatorType::LOGICAL_UNNEST) {
    m.unnest = &node->Cast<duckdb::LogicalUnnest>();
    if (m.unnest->expressions.size() != 1) { return std::nullopt; }
    node = node->children[0].get();
  }

  if (node->type != duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    return std::nullopt;
  }
  m.aggregate = &node->Cast<duckdb::LogicalAggregate>();
  if (m.aggregate->groups.size() != 1 || m.aggregate->grouping_sets.size() > 1 ||
      m.aggregate->expressions.size() != 1 || !is_topk_aggregate(*m.aggregate->expressions[0])) {
    return std::nullopt;
  }
  node = node->children[0].get();

  if (node->type != duckdb::LogicalOperatorType::LOGICAL_PROJECTION) { return std::nullopt; }
  m.distance_projection = &node->Cast<duckdb::LogicalProjection>();
  node                  = node->children[0].get();

  // NEAREST BY adds `dist IS NOT NULL`, which filter pushdown folds into an ANY_JOIN.
  if (node->type == duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN) {
    auto const& any_join = node->Cast<duckdb::LogicalAnyJoin>();
    if (any_join.join_type != duckdb::JoinType::INNER || !any_join.condition ||
        any_join.condition->GetExpressionType() != duckdb::ExpressionType::OPERATOR_IS_NOT_NULL ||
        !any_join.left_projection_map.empty() || !any_join.right_projection_map.empty()) {
      return std::nullopt;
    }
  } else if (node->type != duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
    return std::nullopt;
  }
  m.inner_join = node;
  if (node->children.size() != 2) { return std::nullopt; }

  // Exactly one side is the DELIM_GET; the other is the right table's subtree.
  bool const delim_first =
    node->children[0]->type == duckdb::LogicalOperatorType::LOGICAL_DELIM_GET;
  bool const delim_second =
    node->children[1]->type == duckdb::LogicalOperatorType::LOGICAL_DELIM_GET;
  if (delim_first == delim_second) { return std::nullopt; }
  m.delim_get_child_idx = delim_first ? 0 : 1;
  m.right_child_idx     = 1 - m.delim_get_child_idx;

  return m;
}

//! What the top-k join computes, read off the matched subquery.
struct topk_params {
  std::int64_t k;
  std::string metric;  // "l2" or "cosine"
  bool is_similarity;  // ranking keeps the largest values (arg_max) instead of smallest
  std::size_t right_vector_col_idx;  // within the right table subtree's output
  std::size_t distance_col_idx;      // the distance column within distance_projection's output
};

//! Metric for a supported distance/similarity function, and whether it measures similarity.
std::optional<std::pair<std::string, bool>> metric_of(const std::string& name)
{
  if (name == "array_distance") { return std::make_pair(std::string("l2"), false); }
  if (name == "array_cosine_distance") { return std::make_pair(std::string("cosine"), false); }
  if (name == "array_cosine_similarity") { return std::make_pair(std::string("cosine"), true); }
  return std::nullopt;
}

//! Step 3: read k, direction, metric and the right table's vector column from the matched nodes.
std::optional<topk_params> extract_topk_params(const topk_subquery_match& m, std::int64_t dim)
{
  auto const& aggr = m.aggregate->expressions[0]->Cast<duckdb::BoundAggregateExpression>();
  if (aggr.children.size() != 2 && aggr.children.size() != 3) { return std::nullopt; }

  // k is the third argument; the 2-argument form is k = 1 and has no UNNEST above it.
  std::int64_t k = 1;
  if (aggr.children.size() == 3) {
    if (!m.unnest ||
        aggr.children[2]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) {
      return std::nullopt;
    }
    auto const& k_val = aggr.children[2]->Cast<duckdb::BoundConstantExpression>().value;
    if (k_val.IsNull()) { return std::nullopt; }
    k = k_val.GetValue<std::int64_t>();
  } else if (m.unnest) {
    return std::nullopt;
  }
  if (k < 1) { return std::nullopt; }
  bool const keeps_largest = aggr.function.name.starts_with("arg_max");

  // The ranking argument must be a distance column computed by the projection below.
  auto const& projected = m.distance_projection->expressions;
  if (aggr.children[1]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto const dist_idx = aggr.children[1]->Cast<duckdb::BoundReferenceExpression>().index;
  if (dist_idx >= projected.size() ||
      projected[dist_idx]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_FUNCTION) {
    return std::nullopt;
  }
  auto const& func  = projected[dist_idx]->Cast<duckdb::BoundFunctionExpression>();
  auto const metric = metric_of(func.function.name);
  if (!metric || func.children.size() != 2) { return std::nullopt; }
  // Nearest means smallest distance or largest similarity; the opposite is a farthest-k query.
  if (keeps_largest != metric->second) { return std::nullopt; }

  // Inner join output is [children[0] cols..., children[1] cols...]; one distance argument must be
  // the DELIM_GET's vector, the other a FLOAT[dim] column of the right table.
  auto const n_first    = m.inner_join->children[0]->GetColumnBindings().size();
  auto const delim_lo   = m.delim_get_child_idx == 0 ? 0 : n_first;
  auto const right_lo   = m.right_child_idx == 0 ? 0 : n_first;
  auto const right_cols = m.inner_join->children[m.right_child_idx]->GetColumnBindings().size();
  std::optional<std::size_t> right_vec, delim_vec;
  for (auto const& arg : func.children) {
    auto const ref = as_float_array_ref(*arg);
    if (!ref || ref->second != dim) { return std::nullopt; }
    if (ref->first >= right_lo && ref->first < right_lo + right_cols) {
      right_vec = ref->first - right_lo;
    } else if (ref->first == delim_lo) {
      delim_vec = ref->first;
    }
  }
  if (!right_vec || !delim_vec) { return std::nullopt; }

  // Rows must be ranked per left vector: the group key is the projected DELIM_GET vector.
  auto const& group = *m.aggregate->groups[0];
  if (group.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
  auto const group_idx = group.Cast<duckdb::BoundReferenceExpression>().index;
  if (group_idx >= projected.size()) { return std::nullopt; }
  auto const group_src = as_float_array_ref(*projected[group_idx]);
  if (!group_src || group_src->first != *delim_vec) { return std::nullopt; }

  return topk_params{k, metric->first, metric->second, *right_vec, dist_idx};
}

//! Where one output column of the delim join comes from, in terms of the fused operator's inputs.
struct topk_output_col {
  enum class source { left, right, distance };
  source from;
  std::size_t idx;  // column within the left or right table's output; unused for distance
};

//! Index of a struct field read by struct_extract / struct_extract_at, or nullopt.
std::optional<std::size_t> struct_field_of(const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_FUNCTION) { return std::nullopt; }
  auto const& func = expr.Cast<duckdb::BoundFunctionExpression>();
  if ((func.function.name != "struct_extract" && func.function.name != "struct_extract_at") ||
      !func.bind_info || func.children.empty()) {
    return std::nullopt;
  }
  return func.bind_info->Cast<duckdb::StructExtractBindData>().index;
}

//! Step 4: follow one output column of the subquery down to the right table, the left vector, or
//! the distance. The payload may be a struct_pack that projections above unpack field by field,
//! so pending field reads are kept on a stack and applied when the struct_pack is reached.
std::optional<topk_output_col> trace_subquery_col(const topk_subquery_match& m,
                                                  const topk_params& p,
                                                  std::size_t left_vector_col_idx,
                                                  std::size_t col)
{
  std::vector<std::size_t> fields;  // innermost read on top

  for (auto* proj : m.top_projections) {
    if (col >= proj->expressions.size()) { return std::nullopt; }
    duckdb::Expression const* expr = proj->expressions[col].get();
    while (auto field = struct_field_of(*expr)) {
      fields.push_back(*field);
      expr = expr->Cast<duckdb::BoundFunctionExpression>().children[0].get();
    }
    if (expr->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
    col = expr->Cast<duckdb::BoundReferenceExpression>().index;
  }

  // UNNEST output is [aggregate cols..., unnested element]; the element is one payload row.
  bool through_unnest = false;
  if (m.unnest) {
    auto const n_child = m.unnest->children[0]->GetColumnBindings().size();
    if (col == n_child) {
      auto const& unnest_expr = *m.unnest->expressions[0];
      if (unnest_expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_UNNEST) {
        return std::nullopt;
      }
      auto const& child = *unnest_expr.Cast<duckdb::BoundUnnestExpression>().child;
      if (child.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
      col            = child.Cast<duckdb::BoundReferenceExpression>().index;
      through_unnest = true;
    } else if (col > n_child) {
      return std::nullopt;
    }
  }

  // Aggregate output is [group (the left vector), top-k payload].
  if (col == 0) {
    if (!fields.empty() || through_unnest) { return std::nullopt; }
    return topk_output_col{topk_output_col::source::left, left_vector_col_idx};
  }
  // With k > 1 the payload is a list and only its unnested elements are row values.
  if (col != 1 || through_unnest != (m.unnest != nullptr)) { return std::nullopt; }

  duckdb::Expression const* payload =
    m.aggregate->expressions[0]->Cast<duckdb::BoundAggregateExpression>().children[0].get();
  while (!fields.empty()) {
    if (payload->GetExpressionClass() != duckdb::ExpressionClass::BOUND_FUNCTION) {
      return std::nullopt;
    }
    auto const& pack = payload->Cast<duckdb::BoundFunctionExpression>();
    if (pack.function.name != "struct_pack" || fields.back() >= pack.children.size()) {
      return std::nullopt;
    }
    payload = pack.children[fields.back()].get();
    fields.pop_back();
  }
  if (payload->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
  col = payload->Cast<duckdb::BoundReferenceExpression>().index;

  // Distance projection: the distance itself, or a pass-through of an inner-join column.
  auto const& projected = m.distance_projection->expressions;
  if (col >= projected.size()) { return std::nullopt; }
  if (col == p.distance_col_idx || projected[col]->Equals(*projected[p.distance_col_idx])) {
    return topk_output_col{topk_output_col::source::distance, 0};
  }
  if (projected[col]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  col = projected[col]->Cast<duckdb::BoundReferenceExpression>().index;

  auto const n_first    = m.inner_join->children[0]->GetColumnBindings().size();
  auto const delim_lo   = m.delim_get_child_idx == 0 ? 0 : n_first;
  auto const right_lo   = m.right_child_idx == 0 ? 0 : n_first;
  auto const right_cols = m.inner_join->children[m.right_child_idx]->GetColumnBindings().size();
  if (col >= right_lo && col < right_lo + right_cols) {
    return topk_output_col{topk_output_col::source::right, col - right_lo};
  }
  if (col == delim_lo) {
    return topk_output_col{topk_output_col::source::left, left_vector_col_idx};
  }
  return std::nullopt;
}

//! Step 4: map every output column of the delim join, honoring its per-side projection maps.
std::optional<std::vector<topk_output_col>> trace_output_cols(duckdb::LogicalComparisonJoin& op,
                                                              const topk_delim_match& d,
                                                              const topk_subquery_match& m,
                                                              const topk_params& p)
{
  std::vector<topk_output_col> out;
  for (std::size_t child = 0; child < 2; ++child) {
    auto const& map = child == 0 ? op.left_projection_map : op.right_projection_map;
    auto const n    = op.children[child]->GetColumnBindings().size();
    std::vector<std::size_t> cols;
    if (map.empty()) {
      for (std::size_t i = 0; i < n; ++i) {
        cols.push_back(i);
      }
    } else {
      cols.assign(map.begin(), map.end());
    }
    for (auto const c : cols) {
      if (child == d.left_child_idx) {
        out.push_back(topk_output_col{topk_output_col::source::left, c});
        continue;
      }
      auto traced = trace_subquery_col(m, p, d.left_vector_col_idx, c);
      if (!traced) { return std::nullopt; }
      out.push_back(*traced);
    }
  }
  return out;
}

//! A global top-k vector join as DuckDB plans `SELECT ... FROM l, r ORDER BY dist LIMIT k`:
//! TOP_N over a PROJECTION that computes the distance over a CROSS_PRODUCT of the two tables.
//! When the distance is reused, common-subexpression elimination computes it once in that bottom
//! projection and adds projections above it that pass it through.
struct global_topk_match {
  std::vector<duckdb::LogicalProjection*> projections;  // top first; the last computes the distance
  duckdb::LogicalOperator* cross_product;
  std::size_t distance_col;          // the ranking column within the bottom projection's output
  std::size_t left_vector_col_idx;   // within cross_product->children[0]'s output
  std::size_t right_vector_col_idx;  // within cross_product->children[1]'s output
  std::int64_t dim;
  std::string metric;
  bool is_similarity;
  std::int64_t k;
};

std::optional<global_topk_match> match_global_topk(duckdb::LogicalTopN& op)
{
  if (op.orders.size() != 1 || op.offset != 0 || op.limit < 1 || op.children.size() != 1) {
    return std::nullopt;
  }
  std::vector<duckdb::LogicalProjection*> projections;
  duckdb::LogicalOperator* node = op.children[0].get();
  while (node->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    projections.push_back(&node->Cast<duckdb::LogicalProjection>());
    node = node->children[0].get();
  }
  if (projections.empty() || node->type != duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
      node->children.size() != 2) {
    return std::nullopt;
  }
  auto& cross = *node;
  auto& proj  = *projections.back();

  // The ORDER BY key must reach the bottom projection's distance/similarity column, passed
  // through unchanged by any projections above it.
  auto const& order = op.orders[0];
  if (order.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto dist_col = order.expression->Cast<duckdb::BoundReferenceExpression>().index;
  for (std::size_t i = 0; i + 1 < projections.size(); ++i) {
    auto const& exprs = projections[i]->expressions;
    if (dist_col >= exprs.size() ||
        exprs[dist_col]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      return std::nullopt;
    }
    dist_col = exprs[dist_col]->Cast<duckdb::BoundReferenceExpression>().index;
  }
  if (dist_col >= proj.expressions.size() ||
      proj.expressions[dist_col]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_FUNCTION) {
    return std::nullopt;
  }
  auto const& func  = proj.expressions[dist_col]->Cast<duckdb::BoundFunctionExpression>();
  auto const metric = metric_of(func.function.name);
  if (!metric || func.children.size() != 2) { return std::nullopt; }
  // Nearest means ascending distance or descending similarity.
  auto const nearest_order =
    metric->second ? duckdb::OrderType::DESCENDING : duckdb::OrderType::ASCENDING;
  if (order.type != nearest_order) { return std::nullopt; }

  // One argument from each side of the cross product (output is [left cols..., right cols...]).
  auto const n_left = cross.children[0]->GetColumnBindings().size();
  std::optional<std::pair<std::size_t, std::int64_t>> left_vec, right_vec;
  for (auto const& arg : func.children) {
    auto const ref = as_float_array_ref(*arg);
    if (!ref) { return std::nullopt; }
    if (ref->first < n_left) {
      left_vec = ref;
    } else {
      right_vec = std::make_pair(ref->first - n_left, ref->second);
    }
  }
  if (!left_vec || !right_vec || left_vec->second != right_vec->second) { return std::nullopt; }

  return global_topk_match{std::move(projections),
                           &cross,
                           dist_col,
                           left_vec->first,
                           right_vec->first,
                           left_vec->second,
                           metric->first,
                           metric->second,
                           static_cast<std::int64_t>(op.limit)};
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::try_plan_vector_perrow_topk_join(duckdb::LogicalComparisonJoin& op)
{
  auto match = match_topk_delim_join(op);
  if (!match) { return nullptr; }

  SIRIUS_LOG_DEBUG(
    "[vector_perrow_topk_join] matched delim join: left_child={} subquery_child={} left_vec_col={} "
    "subquery_vec_col={} dim={} join_type={}",
    match->left_child_idx,
    match->subquery_child_idx,
    match->left_vector_col_idx,
    match->subquery_vector_col_idx,
    match->dim,
    duckdb::EnumUtil::ToString(match->join_type));

  auto sub = match_topk_subquery(*op.children[match->subquery_child_idx]);
  if (!sub) {
    SIRIUS_LOG_DEBUG(
      "[vector_perrow_topk_join] subquery is not the top-k shape; using generic delim join");
    return nullptr;
  }
  SIRIUS_LOG_DEBUG(
    "[vector_perrow_topk_join] matched subquery: top_projections={} unnest={} aggregate={} "
    "inner_join={} delim_get_child={} right_child={}",
    sub->top_projections.size(),
    sub->unnest != nullptr,
    sub->aggregate->expressions[0]->Cast<duckdb::BoundAggregateExpression>().function.name,
    duckdb::EnumUtil::ToString(sub->inner_join->type),
    sub->delim_get_child_idx,
    sub->right_child_idx);

  auto params = extract_topk_params(*sub, match->dim);
  if (!params) {
    SIRIUS_LOG_DEBUG(
      "[vector_perrow_topk_join] unsupported ranking or k; using generic delim join");
    return nullptr;
  }
  SIRIUS_LOG_DEBUG(
    "[vector_perrow_topk_join] params: k={} metric={} similarity={} right_vec_col={} "
    "distance_col={}",
    params->k,
    params->metric,
    params->is_similarity,
    params->right_vector_col_idx,
    params->distance_col_idx);

  auto output_cols = trace_output_cols(op, *match, *sub, *params);
  if (!output_cols) {
    SIRIUS_LOG_DEBUG(
      "[vector_perrow_topk_join] could not trace output columns; using generic delim join");
    return nullptr;
  }
  std::string layout;
  for (auto const& c : *output_cols) {
    if (!layout.empty()) { layout += ' '; }
    switch (c.from) {
      case topk_output_col::source::left: layout += "L" + std::to_string(c.idx); break;
      case topk_output_col::source::right: layout += "R" + std::to_string(c.idx); break;
      case topk_output_col::source::distance: layout += "D"; break;
    }
  }
  SIRIUS_LOG_DEBUG("[vector_perrow_topk_join] output columns: [{}]", layout);

  // The merge stage (cuVS knn_merge_parts) caps k.
  if (params->k > vss::KNN_MERGE_MAX_K) {
    SIRIUS_LOG_DEBUG(
      "[vector_perrow_topk_join] k={} exceeds the merge limit; using generic delim join",
      params->k);
    return nullptr;
  }

  auto left_plan     = create_plan(*op.children[match->left_child_idx]);
  auto right_plan    = create_plan(*sub->inner_join->children[sub->right_child_idx]);
  auto const n_left  = left_plan->get_types().size();
  auto const n_right = right_plan->get_types().size();

  // Output batch byte budget, same source the other operators use.
  sirius::operator_params op_params;
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  if (sirius_ctx) { op_params = sirius_ctx->get_config().get_operator_params(); }

  auto join =
    duckdb::make_uniq<sirius::op::sirius_physical_vector_topk_join>(std::move(left_plan),
                                                                    std::move(right_plan),
                                                                    match->left_vector_col_idx,
                                                                    params->right_vector_col_idx,
                                                                    params->k,
                                                                    params->metric,
                                                                    params->is_similarity,
                                                                    match->dim,
                                                                    match->join_type,
                                                                    op.estimated_cardinality);
  // The join finds each left row's top-k within every right batch; the merge combines them.
  auto merge = duckdb::make_uniq<sirius::op::sirius_physical_vector_topk_merge>(
    *join, op_params.concat_batch_bytes);
  merge->children.push_back(std::move(join));

  // The merge emits [left cols, right cols, ranking]; reorder into the delim join's output layout.
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  for (std::size_t i = 0; i < output_cols->size(); ++i) {
    auto const& c        = (*output_cols)[i];
    std::size_t const at = c.from == topk_output_col::source::left    ? c.idx
                           : c.from == topk_output_col::source::right ? n_left + c.idx
                                                                      : n_left + n_right;
    duckdb::BoundReferenceExpression ref(op.types[i], at);
    select_list.push_back(sirius::ast::from_duckdb(ref));
  }
  return push_projection(std::move(merge),
                         sirius::from_duckdb_vec(op.types),
                         std::move(select_list),
                         op.estimated_cardinality);
}

}  // namespace sirius::planner

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::try_plan_vector_global_topk_join(duckdb::LogicalTopN& op)
{
  auto match = match_global_topk(op);
  if (!match) { return nullptr; }
  SIRIUS_LOG_DEBUG(
    "[vector_global_topk_join] matched: k={} metric={} similarity={} distance_col={} "
    "left_vec_col={} right_vec_col={} dim={} dynamic_filter={}",
    match->k,
    match->metric,
    match->is_similarity,
    match->distance_col,
    match->left_vector_col_idx,
    match->right_vector_col_idx,
    match->dim,
    op.dynamic_filter != nullptr);

  auto& proj         = *match->projections.back();
  auto& cross        = *match->cross_product;
  auto left_plan     = create_plan(*cross.children[0]);
  auto right_plan    = create_plan(*cross.children[1]);
  auto const n_left  = left_plan->get_types().size();
  auto const n_right = right_plan->get_types().size();

  auto join =
    duckdb::make_uniq<sirius::op::sirius_physical_vector_topk_join>(std::move(left_plan),
                                                                    std::move(right_plan),
                                                                    match->left_vector_col_idx,
                                                                    match->right_vector_col_idx,
                                                                    match->k,
                                                                    match->metric,
                                                                    match->is_similarity,
                                                                    match->dim,
                                                                    duckdb::JoinType::INNER,
                                                                    cross.estimated_cardinality,
                                                                    sirius::op::topk_scope::global);

  // The join emits [left cols, right cols, ranking], the cross product's layout plus one column,
  // so the projection only needs its distance calls pointed at that column.
  auto const& distance = *proj.expressions[match->distance_col];
  std::function<void(duckdb::unique_ptr<duckdb::Expression>&)> use_ranking_column =
    [&](duckdb::unique_ptr<duckdb::Expression>& expr) {
      if (expr->Equals(distance)) {
        expr =
          duckdb::make_uniq<duckdb::BoundReferenceExpression>(expr->return_type, n_left + n_right);
        return;
      }
      duckdb::ExpressionIterator::EnumerateChildren(*expr, use_ranking_column);
    };
  // Rebuild the projections bottom-up; only the bottom one computes the distance.
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = std::move(join);
  for (auto it = match->projections.rbegin(); it != match->projections.rend(); ++it) {
    auto& p     = **it;
    bool bottom = &p == &proj;
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    for (auto const& e : p.expressions) {
      auto rewritten = e->Copy();
      if (bottom) { use_ranking_column(rewritten); }
      auto node = sirius::ast::from_duckdb(*rewritten);
      if (!node) {
        throw duckdb::NotImplementedException(
          "Unsupported expression in vector global top-k projection: " + e->ToString());
      }
      select_list.push_back(std::move(node));
    }
    plan = push_projection(std::move(plan),
                           sirius::from_duckdb_vec(p.types),
                           std::move(select_list),
                           p.estimated_cardinality);
  }
  return plan;
}

}  // namespace sirius::planner
