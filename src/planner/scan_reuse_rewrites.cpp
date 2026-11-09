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

#include "planner/scan_reuse_rewrites.hpp"

#include <duckdb/function/builtin_function_lookup.hpp>
#include <duckdb/function/function_binder.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::planner {
namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::ProjectionIndex;
using duckdb::unique_ptr;

/// The binding @p expression reads when it is a plain column reference of this query level.
std::optional<ColumnBinding> column_ref(Expression const& expression)
{
  if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
    return std::nullopt;
  }
  auto const& ref = expression.Cast<duckdb::BoundColumnRefExpression>();
  if (ref.Depth() != 0) { return std::nullopt; }
  return ref.Binding();
}

/// Every column binding @p expression reads.
void collect_references(Expression const& expression, std::vector<ColumnBinding>& out)
{
  if (auto const ref = column_ref(expression)) {
    out.push_back(*ref);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expression, [&](Expression const& child) { collect_references(child, out); });
}

std::optional<std::size_t> position_of(std::vector<ColumnBinding> const& bindings,
                                       ColumnBinding binding)
{
  auto const it = std::find(bindings.begin(), bindings.end(), binding);
  if (it == bindings.end()) { return std::nullopt; }
  return static_cast<std::size_t>(it - bindings.begin());
}

bool produces(LogicalOperator& op, ColumnBinding binding)
{
  return position_of(op.GetColumnBindings(), binding).has_value();
}

/// A table column a binding carries unchanged: the scan and the column's primary index.
struct column_origin {
  duckdb::LogicalGet* get;
  duckdb::idx_t column;
};

/**
 * @brief The scan column @p binding carries, through projections of plain references, filters,
 * joins and aggregate group columns; nullopt for computed columns, row ids and virtual columns.
 */
std::optional<column_origin> trace(LogicalOperator& op, ColumnBinding binding)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto& get = op.Cast<duckdb::LogicalGet>();
      if (get.table_index != binding.table_index) { return std::nullopt; }
      auto const& ids = get.GetColumnIds();
      auto const at   = binding.column_index.GetIndex();
      if (at >= ids.size()) { return std::nullopt; }
      auto const& id = ids[at];
      if (id.IsRowIdColumn() || id.IsVirtualColumn() || id.HasChildren()) { return std::nullopt; }
      return column_origin{&get, id.GetPrimaryIndex()};
    }
    case LogicalOperatorType::LOGICAL_PROJECTION: {
      auto& projection = op.Cast<duckdb::LogicalProjection>();
      if (projection.table_index != binding.table_index) { return std::nullopt; }
      auto const at = binding.column_index.GetIndex();
      if (at >= projection.expressions.size()) { return std::nullopt; }
      auto const ref = column_ref(*projection.expressions[at]);
      if (!ref) { return std::nullopt; }
      return trace(*projection.children[0], *ref);
    }
    case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto& aggregate = op.Cast<duckdb::LogicalAggregate>();
      if (aggregate.group_index != binding.table_index) { return std::nullopt; }
      auto const at = binding.column_index.GetIndex();
      if (at >= aggregate.groups.size()) { return std::nullopt; }
      auto const ref = column_ref(*aggregate.groups[at]);
      if (!ref) { return std::nullopt; }
      return trace(*aggregate.children[0], *ref);
    }
    default:
      for (auto& child : op.children) {
        if (produces(*child, binding)) { return trace(*child, binding); }
      }
      return std::nullopt;
  }
}

/// The filters of @p get that change its rows, as text keyed by table column, in a fixed order.
/// Optional filters (dynamic join and top-N filters) only skip rows the plan drops later anyway.
std::vector<std::string> required_filters(duckdb::LogicalGet const& get)
{
  std::vector<std::string> out;
  auto const& ids = get.GetColumnIds();
  for (auto const& entry : get.table_filters) {
    auto const& filter = entry.Filter();
    if (duckdb::ExpressionFilter::IsOptionalFilter(filter)) { continue; }
    auto const at = entry.GetIndex().GetIndex();
    if (at >= ids.size()) {
      out.emplace_back("?");
      continue;
    }
    auto const column = ids[at].GetPrimaryIndex();
    auto const type   = column < get.returned_types.size() ? get.returned_types[column]
                                                           : duckdb::LogicalType::SQLNULL;
    duckdb::BoundColumnRefExpression placeholder(
      duckdb::Identifier("c" + std::to_string(column)),
      type,
      ColumnBinding(duckdb::TableIndex(0), ProjectionIndex(column)));
    out.push_back(std::to_string(column) + ":" + filter.ToExpression(placeholder)->ToString());
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Whether two scans read the same rows: the same table, or the same table function with the same
/// arguments, and the same row filters.
bool same_rows(duckdb::LogicalGet const& a, duckdb::LogicalGet const& b)
{
  if (!(a.function.GetName() == b.function.GetName())) { return false; }
  auto const table = a.GetTable();
  if (table) {
    if (table.get() != b.GetTable().get()) { return false; }
  } else if (a.parameters.empty() || a.parameters != b.parameters ||
             a.named_parameters != b.named_parameters) {
    // Without arguments, a function's identity is only in its bind data (see DuckDB's
    // CommonSubplanOptimizer): equal arguments cannot prove the same rows.
    return false;
  }
  return required_filters(a) == required_filters(b);
}

/// A plain, non-DISTINCT, unfiltered, unordered aggregate @p name over one column reference.
/// "sum" also matches sum_no_overflow, which DuckDB's statistics put where a SUM cannot overflow.
std::optional<ColumnBinding> plain_aggregate_of(Expression const& expression, char const* name)
{
  if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
    return std::nullopt;
  }
  auto const& aggregate = expression.Cast<duckdb::BoundAggregateExpression>();
  auto const& function  = aggregate.Function().GetName();
  bool const named =
    function == name || (std::string_view(name) == "sum" && function == "sum_no_overflow");
  if (!named || aggregate.IsDistinct() || aggregate.GetFilter() || aggregate.GetOrderBys() ||
      aggregate.GetChildren().size() != 1) {
    return std::nullopt;
  }
  return column_ref(*aggregate.GetChildren()[0]);
}

/// A step from an operator down to one of its children.
struct path_step {
  LogicalOperator* op;
  std::size_t child;
};

/// Appends @p binding to @p op's projection map for child @p child when that map is not empty
/// (an empty map passes every child column through).
void keep_through(LogicalOperator& op, std::size_t child, ColumnBinding binding)
{
  auto map = duckdb::LogicalOperatorVisitor::GetProjectionMap(op, child);
  if (!map || map->empty()) { return; }
  auto const at = position_of(op.children[child]->GetColumnBindings(), binding);
  if (!at) { throw std::logic_error("[scan reuse] a carried column vanished below its parent"); }
  if (std::find(map->begin(), map->end(), ProjectionIndex(*at)) == map->end()) {
    map->push_back(ProjectionIndex(*at));
  }
}

//===----------------------------------------------------------------------===//
// Rule: reuse a SEMI join's grouped aggregate (TPC-H Q18)
//===----------------------------------------------------------------------===//

struct semi_join_match {
  /// Which side of the outer INNER join is the repeated scan.
  std::size_t scan_side;
  /// From the other side's root down to the SEMI join.
  std::vector<path_step> to_semi;
  duckdb::LogicalComparisonJoin* semi;
  /// From the SEMI join's build side down to the subquery's aggregate (projections and filters).
  std::vector<LogicalOperator*> to_aggregate;
  duckdb::LogicalAggregate* subquery;
  /// For each outer aggregate, the subquery aggregate (index into its expressions) it reuses.
  std::vector<std::size_t> reused;
};

std::optional<semi_join_match> match_semi_join_aggregate(duckdb::LogicalAggregate& outer)
{
  if (outer.grouping_sets.size() > 1 || !outer.grouping_functions.empty() ||
      outer.expressions.empty()) {
    return std::nullopt;
  }
  auto& below = *outer.children[0];
  if (below.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return std::nullopt; }
  auto& join = below.Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::INNER || join.conditions.size() != 1 ||
      !join.conditions[0].IsComparison() ||
      join.conditions[0].GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
    return std::nullopt;
  }
  for (std::size_t side = 0; side < 2; ++side) {
    if (join.children[side]->type != LogicalOperatorType::LOGICAL_GET) { continue; }
    auto& scan            = join.children[side]->Cast<duckdb::LogicalGet>();
    auto const& condition = join.conditions[0];
    auto const scan_key   = column_ref(side == 0 ? condition.GetLHS() : condition.GetRHS());
    auto const other_key  = column_ref(side == 0 ? condition.GetRHS() : condition.GetLHS());
    if (!scan_key || !other_key) { continue; }
    auto const scan_key_origin = trace(scan, *scan_key);
    if (!scan_key_origin) { continue; }

    // Every outer aggregate is SUM(scan column), and nothing else reads the scan.
    std::vector<duckdb::idx_t> summed;
    bool sums_only = true;
    for (auto const& expression : outer.expressions) {
      auto const argument = plain_aggregate_of(*expression, "sum");
      auto const origin   = argument ? trace(scan, *argument) : std::nullopt;
      if (!origin) {
        sums_only = false;
        break;
      }
      summed.push_back(origin->column);
    }
    if (!sums_only) { continue; }
    std::vector<ColumnBinding> group_refs;
    for (auto const& group : outer.groups) {
      collect_references(*group, group_refs);
    }
    if (std::ranges::any_of(group_refs,
                            [&](auto const& ref) { return ref.table_index == scan.table_index; })) {
      continue;
    }

    // Down the other side, through INNER joins and filters, to the SEMI join on other_key.
    semi_join_match match{.scan_side = side};
    LogicalOperator* at                 = join.children[1 - side].get();
    duckdb::LogicalComparisonJoin* semi = nullptr;
    while (at != nullptr && semi == nullptr) {
      if (at->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        auto& candidate = at->Cast<duckdb::LogicalComparisonJoin>();
        if (candidate.join_type == duckdb::JoinType::SEMI && candidate.conditions.size() == 1 &&
            candidate.conditions[0].IsComparison() &&
            candidate.conditions[0].GetComparisonType() == duckdb::ExpressionType::COMPARE_EQUAL &&
            column_ref(candidate.conditions[0].GetLHS()) == other_key) {
          semi = &candidate;
          break;
        }
        if (candidate.join_type != duckdb::JoinType::INNER) { break; }
      } else if (at->type != LogicalOperatorType::LOGICAL_FILTER) {
        break;
      }
      LogicalOperator* next = nullptr;
      for (std::size_t c = 0; c < at->children.size(); ++c) {
        if (produces(*at->children[c], *other_key)) {
          match.to_semi.push_back({at, c});
          next = at->children[c].get();
          break;
        }
      }
      at = next;
    }
    if (semi == nullptr) { continue; }

    // The build side: projections and filters over the subquery's aggregate, keyed on its group.
    auto key = column_ref(semi->conditions[0].GetRHS());
    if (!key) { continue; }
    LogicalOperator* build = semi->children[1].get();
    while (build->type == LogicalOperatorType::LOGICAL_PROJECTION ||
           build->type == LogicalOperatorType::LOGICAL_FILTER) {
      if (build->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        auto& projection = build->Cast<duckdb::LogicalProjection>();
        if (projection.table_index != key->table_index) { break; }
        key = column_ref(*projection.expressions[key->column_index.GetIndex()]);
        if (!key) { break; }
      }
      match.to_aggregate.push_back(build);
      build = build->children[0].get();
    }
    if (!key || build->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { continue; }
    auto& subquery = build->Cast<duckdb::LogicalAggregate>();
    if (subquery.groups.size() != 1 || subquery.grouping_sets.size() > 1 ||
        !subquery.grouping_functions.empty() ||
        *key != ColumnBinding(subquery.group_index, ProjectionIndex(0))) {
      continue;
    }
    // Its input: projections of plain references over a scan of the same rows as the outer scan.
    LogicalOperator* input = subquery.children[0].get();
    while (input->type == LogicalOperatorType::LOGICAL_PROJECTION &&
           std::ranges::all_of(input->expressions,
                               [](auto const& e) { return column_ref(*e).has_value(); })) {
      input = input->children[0].get();
    }
    if (input->type != LogicalOperatorType::LOGICAL_GET ||
        !same_rows(scan, input->Cast<duckdb::LogicalGet>())) {
      continue;
    }
    auto const group_ref    = column_ref(*subquery.groups[0]);
    auto const group_origin = group_ref ? trace(*subquery.children[0], *group_ref) : std::nullopt;
    if (!group_origin || group_origin->column != scan_key_origin->column) { continue; }
    // Each outer SUM(c) has a SUM(c) in the subquery.
    for (auto const column : summed) {
      for (std::size_t j = 0; j < subquery.expressions.size(); ++j) {
        auto const argument = plain_aggregate_of(*subquery.expressions[j], "sum");
        auto const origin   = argument ? trace(*subquery.children[0], *argument) : std::nullopt;
        if (origin && origin->column == column) {
          match.reused.push_back(j);
          break;
        }
      }
    }
    if (match.reused.size() != summed.size()) { continue; }
    match.semi     = semi;
    match.subquery = &subquery;
    return match;
  }
  return std::nullopt;
}

bool reuse_semi_join_aggregate(duckdb::ClientContext& context,
                               duckdb::LogicalAggregate& outer,
                               removed_subtree_callback const& removed)
{
  auto match = match_semi_join_aggregate(outer);
  if (!match) { return false; }
  auto& subquery = *match->subquery;

  // Bind the new aggregates first: nothing changes unless they keep their result types.
  duckdb::FunctionBinder binder(context);
  std::vector<unique_ptr<Expression>> sums;
  std::vector<ColumnBinding> carried;  // the reused SUMs, as the SEMI join's build side emits them
  for (std::size_t i = 0; i < match->reused.size(); ++i) {
    auto const j    = match->reused[i];
    auto const type = subquery.expressions[j]->GetReturnType();
    // Placeholder binding: replaced below once the build side carries the column.
    std::vector<unique_ptr<Expression>> arguments;
    arguments.push_back(duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
      type, ColumnBinding(subquery.aggregate_index, ProjectionIndex(j))));
    auto sum = binder.BindAggregateFunction(
      duckdb::GetBuiltinAggregateFunction(context, duckdb::Identifier("sum"), {type}),
      std::move(arguments));
    if (sum->GetReturnType() != outer.expressions[i]->GetReturnType()) { return false; }
    sum->SetAlias(outer.expressions[i]->GetAlias());
    sums.push_back(std::move(sum));
  }

  // Carry each reused SUM up the build side: filters pass it through (keeping it in a non-empty
  // projection map), projections get a reference to it.
  for (auto const j : match->reused) {
    ColumnBinding binding(subquery.aggregate_index, ProjectionIndex(j));
    auto const type = subquery.expressions[j]->GetReturnType();
    for (auto it = match->to_aggregate.rbegin(); it != match->to_aggregate.rend(); ++it) {
      auto& op = **it;
      if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
        keep_through(op, 0, binding);
      } else {
        auto& projection = op.Cast<duckdb::LogicalProjection>();
        projection.expressions.push_back(
          duckdb::make_uniq<duckdb::BoundColumnRefExpression>(type, binding));
        binding =
          ColumnBinding(projection.table_index, ProjectionIndex(projection.expressions.size() - 1));
      }
    }
    carried.push_back(binding);
  }

  // The SEMI join becomes INNER (its build keys are unique), and the reused SUMs go up to the
  // outer aggregate.
  auto& semi     = *match->semi;
  semi.join_type = duckdb::JoinType::INNER;
  for (auto const& binding : carried) {
    keep_through(semi, 1, binding);
    for (auto it = match->to_semi.rbegin(); it != match->to_semi.rend(); ++it) {
      keep_through(*it->op, it->child, binding);
    }
  }
  for (std::size_t i = 0; i < sums.size(); ++i) {
    auto& sum = sums[i]->Cast<duckdb::BoundAggregateExpression>();
    sum.GetChildrenMutable()[0]->Cast<duckdb::BoundColumnRefExpression>().BindingMutable() =
      carried[i];
    outer.expressions[i] = std::move(sums[i]);
  }
  // The join with the repeated scan goes.
  if (removed) { removed(*outer.children[0]->children[match->scan_side]); }
  auto other        = std::move(outer.children[0]->children[1 - match->scan_side]);
  outer.children[0] = std::move(other);
  return true;
}

void rewrite(duckdb::ClientContext& context,
             unique_ptr<LogicalOperator>& root,
             unique_ptr<LogicalOperator>& op,
             scan_reuse_counts& counts,
             removed_subtree_callback const& removed)
{
  for (auto& child : op->children) {
    rewrite(context, root, child, counts, removed);
  }
  if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
      reuse_semi_join_aggregate(context, op->Cast<duckdb::LogicalAggregate>(), removed)) {
    ++counts.semi_join_aggregates;
  }
}

}  // namespace

scan_reuse_counts rewrite_repeated_scans(duckdb::ClientContext& context,
                                         unique_ptr<LogicalOperator>& plan,
                                         removed_subtree_callback const& removed)
{
  scan_reuse_counts counts;
  if (!plan) { return counts; }
  rewrite(context, plan, plan, counts, removed);
  if (counts.total() != 0) {
    SIRIUS_LOG_INFO("[scan reuse] rewrote the plan: {} IN-subquery aggregate(s) reused",
                    counts.semi_join_aggregates);
  }
  return counts;
}

}  // namespace sirius::planner
