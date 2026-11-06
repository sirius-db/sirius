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

#include "planner/mark_join_conversion.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/expression_nullability.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_materialized_cte.hpp>

#include <map>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

using binding_key = std::pair<duckdb::idx_t, duckdb::idx_t>;

binding_key key_of(duckdb::ColumnBinding const& binding)
{
  return {binding.table_index.index, binding.column_index.GetIndex()};
}

bool has_delim_cte(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE &&
      op.Cast<duckdb::LogicalMaterializedCTE>().ctename.StartsWith("__duckdb_delim")) {
    return true;
  }
  for (auto const& child : op.children) {
    if (has_delim_cte(*child)) return true;
  }
  return false;
}

void count_references(duckdb::LogicalOperator& op, std::map<binding_key, std::size_t>& refs)
{
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    op, [&](duckdb::unique_ptr<duckdb::Expression>* expression) {
      duckdb::ExpressionIterator::VisitExpression<duckdb::BoundColumnRefExpression>(
        **expression,
        [&](duckdb::BoundColumnRefExpression const& ref) { ++refs[key_of(ref.Binding())]; });
    });
  for (auto& child : op.children) {
    count_references(*child, refs);
  }
}

bool is_bare_ref_to(duckdb::Expression const& expr, duckdb::ColumnBinding const& binding)
{
  if (expr.GetExpressionType() != duckdb::ExpressionType::BOUND_COLUMN_REF) return false;
  return expr.Cast<duckdb::BoundColumnRefExpression>().Binding() == binding;
}

bool is_not_of_ref_to(duckdb::Expression const& expr, duckdb::ColumnBinding const& binding)
{
  if (expr.GetExpressionType() != duckdb::ExpressionType::OPERATOR_NOT) return false;
  auto const& children = expr.Cast<duckdb::BoundOperatorExpression>().GetChildren();
  return children.size() == 1 && is_bare_ref_to(*children[0], binding);
}

bool all_comparisons_null_safe(duckdb::LogicalComparisonJoin const& join)
{
  for (auto const& cond : join.conditions) {
    if (!cond.IsComparison()) continue;
    auto const type = cond.GetComparisonType();
    if (type != duckdb::ExpressionType::COMPARE_DISTINCT_FROM &&
        type != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      return false;
    }
  }
  return true;
}

/// A null-safe equality is a plain equality once either key is provably not NULL (DuckDB's
/// SimplifyNullSafeSemiJoinConditions); plain equality is what dynamic filter publication needs.
void simplify_null_safe_semi_conditions(duckdb::ClientContext& context,
                                        duckdb::LogicalComparisonJoin& join)
{
  duckdb::NotNullExpressionAnalyzer analyzer(context);
  for (auto& cond : join.conditions) {
    if (!cond.IsComparison() ||
        cond.GetComparisonType() != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      continue;
    }
    if (!analyzer.IsNotNull(*join.children[0], cond.GetLHS()) &&
        !analyzer.IsNotNull(*join.children[1], cond.GetRHS())) {
      continue;
    }
    cond = duckdb::JoinCondition(
      cond.LeftReference()->Copy(), cond.RightReference()->Copy(), duckdb::ExpressionType::COMPARE_EQUAL);
  }
}

struct ancestor_info {
  duckdb::LogicalOperatorType type;
  bool has_projection_map;
};

/// True when dropping the last column of the filter's output cannot shift any positional consumer.
bool ancestors_address_by_binding(std::vector<ancestor_info> const& ancestors)
{
  for (auto it = ancestors.rbegin(); it != ancestors.rend(); ++it) {
    switch (it->type) {
      case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: return true;
      case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
        if (it->has_projection_map) return false;
        break;
      default: return false;
    }
  }
  return false;  // reached the plan root: the root addresses its input by position
}

struct converter {
  duckdb::ClientContext& context;
  std::map<binding_key, std::size_t> const& refs;
  std::vector<ancestor_info> ancestors;
  std::size_t converted = 0;

  void visit(duckdb::unique_ptr<duckdb::LogicalOperator>& slot)
  {
    if (slot->type == duckdb::LogicalOperatorType::LOGICAL_FILTER) { try_convert(slot); }

    bool has_map = false;
    if (slot->type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
      has_map = slot->Cast<duckdb::LogicalFilter>().HasProjectionMap();
    } else if (slot->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
      has_map = slot->Cast<duckdb::LogicalJoin>().HasProjectionMap();
    }
    ancestors.push_back({slot->type, has_map});
    for (auto& child : slot->children) {
      visit(child);
    }
    ancestors.pop_back();
  }

  void try_convert(duckdb::unique_ptr<duckdb::LogicalOperator>& slot)
  {
    auto& filter = slot->Cast<duckdb::LogicalFilter>();
    if (filter.children.size() != 1) return;
    auto& child = *filter.children[0];
    if (child.type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) return;
    auto& join = child.Cast<duckdb::LogicalComparisonJoin>();
    if (join.join_type != duckdb::JoinType::MARK) return;

    auto const mark = join.GetColumnBindings().back();
    auto const it   = refs.find(key_of(mark));
    if (it == refs.end() || it->second != 1) return;  // the filter must be the only reader

    std::size_t found = filter.expressions.size();
    bool to_anti      = false;
    for (std::size_t i = 0; i < filter.expressions.size(); ++i) {
      if (is_bare_ref_to(*filter.expressions[i], mark)) {
        found = i;
        break;
      }
      if (is_not_of_ref_to(*filter.expressions[i], mark)) {
        found   = i;
        to_anti = true;
        break;
      }
    }
    if (found == filter.expressions.size()) return;
    if (to_anti && !all_comparisons_null_safe(join)) return;

    // The mark is the join's last output column. A filter whose projection map drops it keeps its
    // own output unchanged; otherwise the output shrinks and every consumer must address by binding.
    auto const mark_position = join.GetColumnBindings().size() - 1;
    auto const has_map       = filter.HasProjectionMap();
    if (has_map) {
      for (auto const& kept : filter.projection_map) {
        if (kept.GetIndex() == mark_position) return;
      }
      bool identity = filter.projection_map.size() == mark_position;
      for (std::size_t i = 0; identity && i < filter.projection_map.size(); ++i) {
        identity = filter.projection_map[i].GetIndex() == i;
      }
      if (!identity && filter.expressions.size() == 1) return;  // would leave a bare projection
    } else if (!ancestors_address_by_binding(ancestors)) {
      return;
    }

    join.join_type = to_anti ? duckdb::JoinType::ANTI : duckdb::JoinType::SEMI;
    filter.expressions.erase(filter.expressions.begin() + static_cast<std::ptrdiff_t>(found));
    if (!to_anti) simplify_null_safe_semi_conditions(context, join);
    ++converted;

    if (filter.expressions.empty()) {
      // Only an identity (or absent) map reaches here, so the join's own output is the filter's.
      auto join_op = std::move(filter.children[0]);
      slot         = std::move(join_op);
    }
  }
};

}  // namespace

std::size_t convert_mark_joins(duckdb::ClientContext& context,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!plan || !has_delim_cte(*plan)) return 0;
  std::map<binding_key, std::size_t> refs;
  count_references(*plan, refs);
  converter pass{context, refs, {}, 0};
  pass.visit(plan);
  return pass.converted;
}

}  // namespace sirius::planner
