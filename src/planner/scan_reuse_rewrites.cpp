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

#include "log/logging.hpp"
#include "planner/aggregate_binding.hpp"
#include "planner/sirius_plan_flat_tree.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/table_filter_set.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ranges>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::planner {
namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::idx_t;
using duckdb::JoinType;
using duckdb::LogicalAggregate;
using duckdb::LogicalComparisonJoin;
using duckdb::LogicalGet;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::ProjectionIndex;
using duckdb::unique_ptr;

template <std::ranges::input_range Range, class Value>
bool contains(Range const& range, Value const& value)
{
  return std::ranges::find(range, value) != std::ranges::end(range);
}

/// A plain column reference of this query level (depth 0), else nullopt.
std::optional<ColumnBinding> column_ref(Expression const& expression)
{
  if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
    return std::nullopt;
  }
  auto const& ref = expression.Cast<duckdb::BoundColumnRefExpression>();
  if (ref.Depth() != 0) { return std::nullopt; }
  return ref.Binding();
}

/// A base-table column carried unchanged: the scan, and the column's primary index.
struct scan_column {
  LogicalGet const* get;
  idx_t column;
};

/// The scan column @p binding names, following only PROJECTIONs whose expression for it is a plain
/// column reference, down to a GET. Nullopt for any other operator, row-id, virtual column, or a
/// column id with children.
std::optional<scan_column> trace_scan_column(LogicalOperator const& op, ColumnBinding binding)
{
  auto const at = binding.column_index.GetIndex();
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto const& get = op.Cast<LogicalGet>();
      auto const& ids = get.GetColumnIds();
      if (get.table_index != binding.table_index || at >= ids.size()) { return std::nullopt; }
      auto const& id = ids[at];
      if (id.IsRowIdColumn() || id.IsVirtualColumn() || id.HasChildren()) { return std::nullopt; }
      return scan_column{&get, id.GetPrimaryIndex()};
    }
    case LogicalOperatorType::LOGICAL_PROJECTION: {
      auto const& projection = op.Cast<duckdb::LogicalProjection>();
      if (projection.table_index != binding.table_index || at >= projection.expressions.size()) {
        return std::nullopt;
      }
      auto const ref = column_ref(*projection.expressions[at]);
      if (!ref) { return std::nullopt; }
      return trace_scan_column(*projection.children[0], *ref);
    }
    default: return std::nullopt;
  }
}

/// The argument of a plain SUM: `sum` or `sum_no_overflow`, not DISTINCT, no FILTER, no ORDER BY,
/// exactly one child that is a plain column reference whose type satisfies `sum_combines_exactly`.
/// Nullopt otherwise.
std::optional<ColumnBinding> plain_sum_argument(Expression const& expression)
{
  if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
    return std::nullopt;
  }
  auto const& aggregate = expression.Cast<duckdb::BoundAggregateExpression>();
  auto const& name      = aggregate.Function().GetName();
  if ((name != "sum" && name != "sum_no_overflow") || aggregate.IsDistinct() ||
      aggregate.GetFilter() || aggregate.GetOrderBys() || aggregate.GetChildren().size() != 1) {
    return std::nullopt;
  }
  auto const& argument = *aggregate.GetChildren()[0];
  if (!sum_combines_exactly(argument.GetReturnType())) { return std::nullopt; }
  return column_ref(argument);
}

/// One table filter of a scan, keyed by base-table column, with its column written as a reference
/// to a canonical binding so that the same filter on two scans of one table compares equal.
struct table_filter_on {
  idx_t column;
  unique_ptr<Expression> expression;
  bool optional;
};

/// The single-column filters of @p get, or nullopt when one filters something other than a plain
/// base-table column.
std::optional<std::vector<table_filter_on>> table_filters_of(LogicalGet const& get)
{
  auto const& ids = get.GetColumnIds();
  std::vector<table_filter_on> filters;
  for (auto const& entry : get.table_filters) {
    auto const at = entry.GetIndex().GetIndex();
    if (at >= ids.size()) { return std::nullopt; }
    auto const& id = ids[at];
    if (id.IsRowIdColumn() || id.IsVirtualColumn() || id.HasChildren()) { return std::nullopt; }
    auto const column = id.GetPrimaryIndex();
    if (column >= get.returned_types.size()) { return std::nullopt; }
    duckdb::BoundColumnRefExpression const canonical(
      get.returned_types[column], ColumnBinding(duckdb::TableIndex(0), ProjectionIndex(column)));
    filters.push_back({.column     = column,
                       .expression = entry.Filter().ToExpression(canonical),
                       .optional   = duckdb::ExpressionFilter::IsOptionalFilter(entry.Filter())});
  }
  return filters;
}

/// Whether the required (non-optional) filters of @p a and @p b are equal as multisets.
bool same_required_filters(std::vector<table_filter_on> const& a,
                           std::vector<table_filter_on> const& b)
{
  auto const is_required = [](table_filter_on const& filter) { return !filter.optional; };
  if (std::ranges::count_if(a, is_required) != std::ranges::count_if(b, is_required)) {
    return false;
  }
  std::vector<bool> matched(b.size(), false);
  return std::ranges::all_of(a | std::views::filter(is_required), [&](table_filter_on const& x) {
    for (std::size_t i = 0; i < b.size(); ++i) {
      if (!matched[i] && !b[i].optional && b[i].column == x.column &&
          Expression::Equals(*b[i].expression, *x.expression)) {
        matched[i] = true;
        return true;
      }
    }
    return false;
  });
}

/// Checks that each key retained by the subquery scan @p t has the same input rows as that key in
/// the outer scan @p l. The scans must share a source and required filters. Optional filters on
/// @p t must use @p key_column so they can discard only whole groups.
bool same_rows_per_key(duckdb::ClientContext& context,
                       LogicalGet const& l,
                       LogicalGet const& t,
                       idx_t key_column)
{
  auto const plain_scan = [](LogicalGet const& get) {
    return get.children.empty() && !get.table_filters.HasMultiColumnFilters() &&
           !get.extra_info.sample_options;
  };
  if (!plain_scan(l) || !plain_scan(t)) { return false; }
  // A runtime join filter pushed through the subquery aggregate also acts on the group key,
  // so it can prune keys while keeping every row of each surviving group.
  auto const l_filters = table_filters_of(l);
  auto const t_filters = table_filters_of(t);
  if (!l_filters || !t_filters || !same_required_filters(*l_filters, *t_filters)) { return false; }
  if (!std::ranges::all_of(*t_filters, [&](table_filter_on const& filter) {
        return !filter.optional || filter.column == key_column;
      })) {
    return false;
  }
  return flat::same_source(context, l, t);
}

/// An operator and the index of its child that a path continues into.
struct path_step {
  LogicalOperator* op;
  std::size_t child;
};

/// Matched operators and the mapping from outer SUMs to subquery subtotals.
struct in_subquery_sum_reuse {
  LogicalAggregate* outer;
  LogicalComparisonJoin* outer_join;
  /// The child of `outer_join` that holds the scan to remove.
  std::size_t scan_side;
  /// From `outer_join`'s other child down to `semi`, top-down; each step names the child that
  /// contains the semi join.
  std::vector<path_step> to_semi;
  LogicalComparisonJoin* semi;
  /// The child of `semi` that holds the subquery: 1 for SEMI, 0 for RIGHT_SEMI.
  std::size_t subquery_side;
  /// From `semi`'s subquery child down to `subquery`, top-down: PROJECTIONs and FILTERs only.
  std::vector<LogicalOperator*> to_subquery;
  LogicalAggregate* subquery;
  /// Per outer expression: the index of the subquery expression it reuses.
  std::vector<idx_t> reused;
};

struct outer_scan {
  LogicalComparisonJoin* join;
  std::size_t scan_side;
  scan_column key;
  /// The join key from the child that survives the rewrite.
  ColumnBinding probe_key;
  /// Per outer expression: the base-table column it sums.
  std::vector<idx_t> summed;
};

/// The join below @p outer with the scan on side @p side, when that join is an INNER join on a
/// single `=`, @p outer sums only columns of the scan, and its groups do not read the scan.
std::optional<outer_scan> match_outer(LogicalAggregate& outer, std::size_t side)
{
  if (outer.grouping_sets.size() > 1 || !outer.grouping_functions.empty() ||
      outer.expressions.empty() ||
      outer.children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return std::nullopt;
  }
  auto& join = outer.children[0]->Cast<LogicalComparisonJoin>();
  if (join.join_type != JoinType::INNER || join.conditions.size() != 1 ||
      !join.conditions[0].IsComparison() ||
      join.conditions[0].GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
    return std::nullopt;
  }
  auto const& scan_child = *join.children[side];
  if (scan_child.type != LogicalOperatorType::LOGICAL_GET || !scan_child.children.empty()) {
    return std::nullopt;
  }
  auto const& scan      = scan_child.Cast<LogicalGet>();
  auto const& condition = join.conditions[0];
  auto const scan_ref   = column_ref(side == 0 ? condition.GetLHS() : condition.GetRHS());
  auto const probe_ref  = column_ref(side == 0 ? condition.GetRHS() : condition.GetLHS());
  if (!scan_ref || !probe_ref) { return std::nullopt; }
  auto const key = trace_scan_column(scan, *scan_ref);
  if (!key) { return std::nullopt; }

  std::vector<idx_t> summed;
  summed.reserve(outer.expressions.size());
  for (auto const& expression : outer.expressions) {
    auto const argument = plain_sum_argument(*expression);
    auto const column   = argument ? trace_scan_column(scan, *argument) : std::nullopt;
    if (!column) { return std::nullopt; }
    summed.push_back(column->column);
  }
  auto const reads_scan = [&](unique_ptr<Expression> const& group) {
    return std::ranges::any_of(flat::columns_read(*group), [&](ColumnBinding const& binding) {
      return binding.table_index == scan.table_index;
    });
  };
  if (std::ranges::any_of(outer.groups, reads_scan)) { return std::nullopt; }
  return outer_scan{.join      = &join,
                    .scan_side = side,
                    .key       = *key,
                    .probe_key = *probe_ref,
                    .summed    = std::move(summed)};
}

struct semi_join_on_key {
  std::vector<path_step> to_semi;
  LogicalComparisonJoin* semi;
  std::size_t subquery_side;
  ColumnBinding subquery_key;
};

/// From @p top, through FILTERs and INNER comparison joins, the SEMI or RIGHT_SEMI join whose
/// single `=` condition compares @p probe_key on its returned side.
std::optional<semi_join_on_key> find_semi(LogicalOperator& top, ColumnBinding probe_key)
{
  std::vector<path_step> to_semi;
  LogicalOperator* at = &top;
  while (true) {
    if (at->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
      auto& join = at->Cast<LogicalComparisonJoin>();
      if (join.join_type == JoinType::SEMI || join.join_type == JoinType::RIGHT_SEMI) {
        // A SEMI join returns its left child, a RIGHT_SEMI join its right child; the condition's
        // left operand always refers to the left child (also after `convert_mark_joins` flips one).
        bool const right_semi = join.join_type == JoinType::RIGHT_SEMI;
        if (join.conditions.size() != 1 || !join.conditions[0].IsComparison() ||
            join.conditions[0].GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
          return std::nullopt;
        }
        auto const& condition = join.conditions[0];
        auto const returned   = column_ref(right_semi ? condition.GetRHS() : condition.GetLHS());
        auto const subquery   = column_ref(right_semi ? condition.GetLHS() : condition.GetRHS());
        if (returned != probe_key || !subquery) { return std::nullopt; }
        return semi_join_on_key{.to_semi       = std::move(to_semi),
                                .semi          = &join,
                                .subquery_side = right_semi ? std::size_t{0} : std::size_t{1},
                                .subquery_key  = *subquery};
      }
      if (join.join_type != JoinType::INNER) { return std::nullopt; }
    } else if (at->type != LogicalOperatorType::LOGICAL_FILTER) {
      return std::nullopt;
    }
    auto const child = std::ranges::find_if(at->children, [&](unique_ptr<LogicalOperator>& c) {
      return contains(c->GetColumnBindings(), probe_key);
    });
    if (child == at->children.end()) { return std::nullopt; }
    to_semi.push_back({at, static_cast<std::size_t>(child - at->children.begin())});
    at = child->get();
  }
}

struct grouped_subquery {
  std::vector<LogicalOperator*> to_subquery;
  LogicalAggregate* aggregate;
};

/// From @p top, through PROJECTIONs that pass @p key on as a plain column and FILTERs, an aggregate
/// grouped by exactly @p key.
std::optional<grouped_subquery> match_subquery(LogicalOperator& top, ColumnBinding key)
{
  std::vector<LogicalOperator*> to_subquery;
  LogicalOperator* at = &top;
  while (at->type == LogicalOperatorType::LOGICAL_PROJECTION ||
         at->type == LogicalOperatorType::LOGICAL_FILTER) {
    if (at->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      auto const& projection = at->Cast<duckdb::LogicalProjection>();
      auto const index       = key.column_index.GetIndex();
      if (projection.table_index != key.table_index || index >= projection.expressions.size()) {
        return std::nullopt;
      }
      auto const ref = column_ref(*projection.expressions[index]);
      if (!ref) { return std::nullopt; }
      key = *ref;
    }
    to_subquery.push_back(at);
    at = at->children[0].get();
  }
  if (at->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return std::nullopt; }
  auto& aggregate = at->Cast<LogicalAggregate>();
  if (aggregate.groups.size() != 1 || aggregate.grouping_sets.size() > 1 ||
      !aggregate.grouping_functions.empty() ||
      key != ColumnBinding(aggregate.group_index, ProjectionIndex(0))) {
    return std::nullopt;
  }
  return grouped_subquery{.to_subquery = std::move(to_subquery), .aggregate = &aggregate};
}

/// Per outer expression, the subquery expression it reuses, when the subquery groups a scan of the
/// same rows as the outer scan by the outer join's key column and sums every column the outer
/// aggregate sums.
std::optional<std::vector<idx_t>> match_scan(duckdb::ClientContext& context,
                                             outer_scan const& outer,
                                             LogicalAggregate const& subquery)
{
  auto const& input    = *subquery.children[0];
  auto const group_ref = column_ref(*subquery.groups[0]);
  auto const group     = group_ref ? trace_scan_column(input, *group_ref) : std::nullopt;
  if (!group || group->column != outer.key.column ||
      !same_rows_per_key(context, *outer.key.get, *group->get, group->column)) {
    return std::nullopt;
  }
  auto const sums_column = [&](idx_t column) {
    return [&, column](unique_ptr<Expression> const& expression) {
      auto const argument = plain_sum_argument(*expression);
      auto const origin   = argument ? trace_scan_column(input, *argument) : std::nullopt;
      return origin && origin->get == group->get && origin->column == column;
    };
  };
  std::vector<idx_t> reused;
  reused.reserve(outer.summed.size());
  for (auto const column : outer.summed) {
    auto const found = std::ranges::find_if(subquery.expressions, sums_column(column));
    if (found == subquery.expressions.end()) { return std::nullopt; }
    reused.push_back(static_cast<idx_t>(found - subquery.expressions.begin()));
  }
  return reused;
}

//===----------MATCH----------===//
/// Simplified plan shapes. Intermediate filters, projections, and joins are omitted.
/// The IN join can be SEMI or RIGHT_SEMI.
///
/// BEFORE
/// Aggregate with SUM(value)
///   Inner join to the repeated scan on key
///     Repeated scan: key, value
///     IN semi join on key
///       Input rows tested by IN
///       Subquery grouped by key with SUM(value) AS subtotal
///
/// AFTER
/// Aggregate with SUM(subtotal)
///   IN join converted to an inner join on key, exposing subtotal
///     Input rows tested by IN
///     Subquery grouped by key with SUM(value) AS subtotal
///
/// The subquery emits at most one row per matching key, so the converted join preserves the rows
/// kept by IN. For each surviving key, the subtotal uses the same input rows and summed column as
/// the repeated scan. Summing the subtotals must preserve the original SUM's result type and
/// arithmetic behavior.
std::optional<in_subquery_sum_reuse> match_in_subquery_sum(duckdb::ClientContext& context,
                                                           LogicalAggregate& outer)
{
  for (std::size_t side = 0; side < 2; ++side) {
    auto scan = match_outer(outer, side);
    if (!scan) { continue; }
    auto semi = find_semi(*scan->join->children[1 - side], scan->probe_key);
    if (!semi) { continue; }
    auto subquery = match_subquery(*semi->semi->children[semi->subquery_side], semi->subquery_key);
    if (!subquery) { continue; }
    auto reused = match_scan(context, *scan, *subquery->aggregate);
    if (!reused) { continue; }
    return in_subquery_sum_reuse{.outer         = &outer,
                                 .outer_join    = scan->join,
                                 .scan_side     = side,
                                 .to_semi       = std::move(semi->to_semi),
                                 .semi          = semi->semi,
                                 .subquery_side = semi->subquery_side,
                                 .to_subquery   = std::move(subquery->to_subquery),
                                 .subquery      = subquery->aggregate,
                                 .reused        = std::move(*reused)};
  }
  return std::nullopt;
}

/// Append references to the carried sums to a projection on the subquery side.
struct projection_append {
  duckdb::LogicalProjection* projection;
  std::vector<unique_ptr<Expression>> refs;
};

/// Extends a filter or join's child projection map to expose the carried subtotals. Map entries
/// are positions in the child's output. Adding columns below can shift those positions, so the
/// rewrite remaps existing entries by column binding.
struct projection_map_extension {
  LogicalOperator* op;
  std::size_t child;
  /// The child's output before the rewrite; empty when the map is empty (it then emits every child
  /// column, the carried ones included).
  duckdb::vector<ColumnBinding> child_bindings_before;
  /// The carried sums, as the child emits them.
  std::vector<ColumnBinding> carried;
};

using carry_step = std::variant<projection_append, projection_map_extension>;

/// Replacement aggregates and edits for exposing the subtotals to them.
struct prepared_edits {
  /// From the subquery aggregate up to the outer join's other child, bottom-up.
  std::vector<carry_step> carry;
  /// Per outer expression: its replacement, a SUM over the carried sum.
  std::vector<unique_ptr<Expression>> sums;
};

duckdb::vector<ProjectionIndex>& projection_map(projection_map_extension const& step)
{
  auto map = duckdb::LogicalOperatorVisitor::GetProjectionMap(*step.op, step.child);
  if (!map) {
    throw duckdb::InternalException(
      "scan reuse: an operator on the carry path has no projection map");
  }
  return *map;
}

projection_map_extension extend_map(LogicalOperator& op,
                                    std::size_t child,
                                    std::vector<ColumnBinding> const& carried)
{
  projection_map_extension step{
    .op = &op, .child = child, .child_bindings_before = {}, .carried = carried};
  if (!projection_map(step).empty()) {
    step.child_bindings_before = op.children[child]->GetColumnBindings();
  }
  return step;
}

//===----------PREPARE----------===//
/// Binds the replacement SUMs and builds every reference and map change; nullopt (with the plan
/// unchanged) when a replacement SUM would not have the type of the SUM it replaces.
std::optional<prepared_edits> prepare(duckdb::ClientContext& context,
                                      in_subquery_sum_reuse const& match)
{
  // Each reused subquery expression is carried once, even when several outer SUMs reuse it.
  std::vector<idx_t> carried_expressions;
  std::vector<std::size_t> carried_of;  // per outer expression: index into carried_expressions
  for (auto const j : match.reused) {
    auto const found = std::ranges::find(carried_expressions, j);
    carried_of.push_back(static_cast<std::size_t>(found - carried_expressions.begin()));
    if (found == carried_expressions.end()) { carried_expressions.push_back(j); }
  }
  std::vector<duckdb::LogicalType> types;
  std::vector<ColumnBinding> carried;
  for (auto const j : carried_expressions) {
    types.push_back(match.subquery->expressions[j]->GetReturnType());
    carried.emplace_back(match.subquery->aggregate_index, ProjectionIndex(j));
  }

  prepared_edits edits;
  edits.carry.reserve(match.to_subquery.size() + 1 + match.to_semi.size());
  for (auto* const op : match.to_subquery | std::views::reverse) {
    if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
      edits.carry.emplace_back(extend_map(*op, 0, carried));
      continue;
    }
    auto& projection = op->Cast<duckdb::LogicalProjection>();
    projection_append append{.projection = &projection, .refs = {}};
    for (std::size_t k = 0; k < carried.size(); ++k) {
      append.refs.emplace_back(
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(types[k], carried[k]));
      carried[k] =
        ColumnBinding(projection.table_index, ProjectionIndex(projection.expressions.size() + k));
    }
    edits.carry.emplace_back(std::move(append));
  }
  edits.carry.emplace_back(extend_map(*match.semi, match.subquery_side, carried));
  for (auto const& step : match.to_semi | std::views::reverse) {
    edits.carry.emplace_back(extend_map(*step.op, step.child, carried));
  }

  // Bind SUM over the subtotal's DECIMAL(38) type, then check that it preserves the old SUM's type.
  edits.sums.reserve(match.reused.size());
  for (std::size_t i = 0; i < match.reused.size(); ++i) {
    auto const k = carried_of[i];
    auto sum     = bind_builtin_aggregate(
      context,
      "sum",
      types[k],
      duckdb::make_uniq<duckdb::BoundColumnRefExpression>(types[k], carried[k]));
    auto const& old = *match.outer->expressions[i];
    if (sum->GetReturnType() != old.GetReturnType()) { return std::nullopt; }
    sum->SetAlias(old.GetAlias());
    edits.sums.emplace_back(std::move(sum));
  }

  // So that `apply` only appends within capacity.
  for (auto& step : edits.carry) {
    if (auto const* append = std::get_if<projection_append>(&step)) {
      auto& expressions = append->projection->expressions;
      expressions.reserve(expressions.size() + append->refs.size());
    } else {
      auto const& extension = std::get<projection_map_extension>(step);
      auto& map             = projection_map(extension);
      if (!map.empty()) { map.reserve(map.size() + extension.carried.size()); }
    }
  }
  return edits;
}

/// Finds the current position of a binding. A missing binding aborts Sirius planning.
ProjectionIndex position_of(duckdb::vector<ColumnBinding> const& bindings, ColumnBinding binding)
{
  auto const found = std::ranges::find(bindings, binding);
  if (found == bindings.end()) {
    throw duckdb::InternalException(
      "scan reuse: a column read above is missing from its child's output");
  }
  return ProjectionIndex(static_cast<idx_t>(found - bindings.begin()));
}

void carry(projection_append& step)
{
  for (auto& ref : step.refs) {
    step.projection->expressions.push_back(std::move(ref));
  }
}

void carry(projection_map_extension& step)
{
  auto& map = projection_map(step);
  if (map.empty()) { return; }
  auto const after = step.op->children[step.child]->GetColumnBindings();
  for (auto& entry : map) {
    if (entry.GetIndex() >= step.child_bindings_before.size()) {
      throw duckdb::InternalException("scan reuse: a projection map entry is out of range");
    }
    entry = position_of(after, step.child_bindings_before[entry.GetIndex()]);
  }
  for (auto const& binding : step.carried) {
    auto const at = position_of(after, binding);
    if (!contains(map, at)) { map.push_back(at); }
  }
}

//===----------APPLY----------===//
/// Applies the prepared edits and returns the removed scan. An exception can leave the plan
/// partially rewritten. `create_plan` discards it on fallback.
unique_ptr<LogicalOperator> apply(in_subquery_sum_reuse const& match, prepared_edits&& edits)
{
  // The subquery's keys are unique, so the INNER join emits each returned row once, as SEMI did.
  match.semi->join_type = JoinType::INNER;
  // Apply edits bottom-up so each parent can resolve the subtotals in its child's updated output.
  for (auto& step : edits.carry) {
    std::visit([](auto& s) { carry(s); }, step);
  }
  for (std::size_t i = 0; i < edits.sums.size(); ++i) {
    match.outer->expressions[i] = std::move(edits.sums[i]);
  }
  auto removed             = std::move(match.outer_join->children[match.scan_side]);
  auto kept                = std::move(match.outer_join->children[1 - match.scan_side]);
  match.outer->children[0] = std::move(kept);
  return removed;
}

void rewrite(duckdb::ClientContext& context,
             LogicalOperator& op,
             std::vector<unique_ptr<LogicalOperator>>& removed)
{
  for (auto& child : op.children) {
    rewrite(context, *child, removed);
  }
  // match -> prepare -> apply, or no change.
  if (op.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return; }
  auto const match = match_in_subquery_sum(context, op.Cast<LogicalAggregate>());
  if (!match) { return; }
  auto edits = prepare(context, *match);
  if (!edits) { return; }
  removed.push_back(apply(*match, std::move(*edits)));
}

}  // namespace

std::vector<unique_ptr<LogicalOperator>> rewrite_repeated_scans(duckdb::ClientContext& context,
                                                                unique_ptr<LogicalOperator>& plan)
{
  std::vector<unique_ptr<LogicalOperator>> removed;
  if (!plan) { return removed; }
  rewrite(context, *plan, removed);
  if (!removed.empty()) {
    SIRIUS_LOG_INFO(
      "[scan reuse] reused {} IN-subquery SUM aggregate(s) instead of scanning their table again",
      removed.size());
  }
  return removed;
}

}  // namespace sirius::planner
