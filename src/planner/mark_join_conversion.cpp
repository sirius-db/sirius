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

#include "log/logging.hpp"
#include "planner/dynamic_filter/build_filter_evidence.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/enums/join_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/expression_nullability.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>

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

// The identity projections that block DuckDB's own conversion come from the delim-to-CTE rewrite,
// which runs exactly when `delim_join_as_cte` is true. Keying on the setting, not on a CTE being
// left in the plan, also covers plans where DuckDB later inlined the CTE but kept the projections.
bool delim_join_as_cte_enabled(duckdb::ClientContext& context)
{
  duckdb::Value setting;
  return context.TryGetCurrentSetting("delim_join_as_cte", setting) && !setting.IsNull() &&
         setting.GetValue<bool>();
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
    cond = duckdb::JoinCondition(cond.LeftReference()->Copy(),
                                 cond.RightReference()->Copy(),
                                 duckdb::ExpressionType::COMPARE_EQUAL);
  }
}

// DuckDB flips a SEMI join when its filter target is at least this many times larger than the
// outer side (JoinFilterBuildSideHeuristics::SEMI_JOIN_FILTER_TARGET_RATIO).
constexpr duckdb::idx_t outer_build_ratio = 3;

bool reads_cte(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_CTE_REF) return true;
  for (auto const& child : op.children) {
    if (child && reads_cte(*child)) return true;
  }
  return false;
}

// A right-family join publishes before its probe is read, by accumulating a Bloom filter, and
// accumulation takes INT32 and INT64 keys only (accumulated_bloom_builder::supports).
bool is_early_publication_key(duckdb::Expression const& expr)
{
  if (expr.GetExpressionType() != duckdb::ExpressionType::BOUND_COLUMN_REF) return false;
  auto const type = expr.GetReturnType().id();
  return type == duckdb::LogicalTypeId::INTEGER || type == duckdb::LogicalTypeId::BIGINT;
}

/// Every condition must be the same kind of equality, so that the flipped join is a plain hash
/// join: a right-family join has no nested-loop fallback, and a residual would have to be
/// translated again with its sides swapped. One key must also be publishable.
bool conditions_allow_outer_build(duckdb::LogicalComparisonJoin const& join)
{
  bool publishable = false;
  for (auto const& cond : join.conditions) {
    if (!cond.IsComparison()) return false;
    auto const type = cond.GetComparisonType();
    if (type != duckdb::ExpressionType::COMPARE_EQUAL &&
        type != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      return false;
    }
    if (type != join.conditions.front().GetComparisonType()) return false;
    publishable = publishable || (is_early_publication_key(cond.GetLHS()) &&
                                  is_early_publication_key(cond.GetRHS()));
  }
  return publishable;
}

/// DuckDB chooses build sides while the join is still MARK, which has no flipped form, so a
/// converted join builds on its subquery and its dynamic filter reaches only the outer rows. This
/// makes the choice DuckDB could not: when the outer side is the small, filtered one, build on it
/// (RIGHT_SEMI or RIGHT_ANTI) so that its keys filter the subquery's scans. Returns whether the
/// join was flipped.
bool build_on_small_outer_side(duckdb::ClientContext& context, duckdb::LogicalComparisonJoin& join)
{
  // Publication is the flip's only gain, and a right-family join needs these settings for it.
  if (!right_family_publication_enabled(*duckdb::query_operator_options(context))) return false;
  if (!conditions_allow_outer_build(join)) return false;

  auto& outer    = *join.children[0];
  auto& subquery = *join.children[1];
  // A CTE read on the outer side marks the delim rewrite's join-back. Its subquery is already
  // restricted by the CTE's keys, and target discovery does not descend through a right-family
  // join, so a flip would hide the outer-row read from the joins above.
  if (reads_cte(outer)) return false;
  if (!build_subtree_is_filtering(outer) && !build_subtree_has_filtering_join(outer)) return false;

  auto const outer_rows    = outer.EstimateCardinality(context);
  auto const subquery_rows = subquery.EstimateCardinality(context);
  if (outer_rows > subquery_rows / outer_build_ratio) return false;

  // The steps of DuckDB's FlipChildren (build_probe_side_optimizer.cpp), which is not exported.
  // A SEMI or ANTI join emits its left child through the left projection map and the RIGHT form
  // emits its right child through the right one, so the join's output bindings stay the same.
  std::swap(join.children[0], join.children[1]);
  join.join_type = duckdb::InverseJoinType(join.join_type);
  for (auto& cond : join.conditions) {
    cond = duckdb::JoinCondition(cond.RightReference()->Copy(),
                                 cond.LeftReference()->Copy(),
                                 duckdb::FlipComparisonExpression(cond.GetComparisonType()));
  }
  std::swap(join.left_projection_map, join.right_projection_map);

  SIRIUS_LOG_INFO(
    "[mark_join_conversion] Building a converted {} join on its outer side (outer est {} rows, "
    "subquery est {} rows).",
    duckdb::JoinTypeToString(join.join_type),
    outer_rows,
    subquery_rows);
  return true;
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
    // own output unchanged; otherwise the output shrinks and every consumer must address by
    // binding.
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
    // A flipped join keeps its keys as they are: simplifying key by key could leave a mix of
    // plain and null-safe equalities, and a right-family join publishes null-safe keys too.
    if (!build_on_small_outer_side(context, join) && !to_anti) {
      simplify_null_safe_semi_conditions(context, join);
    }
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
  if (!plan || !delim_join_as_cte_enabled(context)) return 0;
  std::map<binding_key, std::size_t> refs;
  count_references(*plan, refs);
  converter pass{context, refs, {}, 0};
  pass.visit(plan);
  return pass.converted;
}

}  // namespace sirius::planner
