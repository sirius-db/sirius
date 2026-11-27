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

#include "transparent/join_dimension_after_aggregate.hpp"

#include <duckdb/common/types.hpp>
#include <duckdb/function/builtin_function_lookup.hpp>
#include <duckdb/function/function_binder.hpp>
#include <duckdb/optimizer/column_binding_replacer.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::ProjectionIndex;
using duckdb::unique_ptr;

/// A column of the lookup scan.
struct lookup_column {
  ColumnBinding binding;
  duckdb::LogicalType type;
};

/// A binding above the lookup join that carries a lookup column unchanged: the scan's own, or the
/// output of a projection that passes one through.
struct carrier {
  ColumnBinding binding;
  lookup_column column;
};

/// The aggregate whose lookup join is searched for.
struct aggregate_site {
  duckdb::ClientContext& context;
  unique_ptr<LogicalOperator>& slot;
  /// DuckDB's estimate of the rows that reach the aggregate.
  std::uint64_t rows;
  std::uint64_t min_carried_bytes;
};

/// A lookup join to move above its aggregate.
struct late_join {
  unique_ptr<LogicalOperator>* aggregate;
  /// The operators between the aggregate and the lookup join.
  std::vector<LogicalOperator*> path;
  unique_ptr<LogicalOperator>* lookup;
  /// The lookup join's equality: the column of its left side, the scan's column, and their type.
  ColumnBinding foreign_key;
  ColumnBinding lookup_key;
  duckdb::LogicalType key_type;
  /// The projection outputs on the path that carry a lookup column.
  std::vector<ColumnBinding> passed_through;
  /// For each group of the aggregate, the lookup column it is, when it is one.
  std::vector<std::optional<lookup_column>> groups;
  /// Each aggregate over its own result, which merges the groups above the late join.
  duckdb::vector<unique_ptr<Expression>> merged;
  /// The bytes of lookup columns each row above the lookup join stops carrying.
  std::uint64_t carried_width = 0;
};

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

bool contains(duckdb::vector<ColumnBinding> const& bindings, ColumnBinding const& binding)
{
  return std::find(bindings.begin(), bindings.end(), binding) != bindings.end();
}

/// The carrier @p expression is a plain reference to, if it is one.
carrier const* carrier_of(Expression const& expression, std::vector<carrier> const& carriers)
{
  auto const binding = column_ref(expression);
  if (!binding) { return nullptr; }
  auto const it = std::find_if(carriers.begin(), carriers.end(), [&](carrier const& candidate) {
    return candidate.binding == *binding;
  });
  return it == carriers.end() ? nullptr : &*it;
}

bool reads_any(Expression const& expression, std::vector<carrier> const& carriers)
{
  if (expression.GetExpressionClass() == duckdb::ExpressionClass::BOUND_COLUMN_REF) {
    auto const& binding = expression.Cast<duckdb::BoundColumnRefExpression>().Binding();
    return std::any_of(carriers.begin(), carriers.end(), [&](carrier const& candidate) {
      return candidate.binding == binding;
    });
  }
  bool found = false;
  duckdb::ExpressionIterator::EnumerateChildren(
    expression, [&](Expression const& child) { found = found || reads_any(child, carriers); });
  return found;
}

bool reads_any(LogicalOperator const& op, std::vector<carrier> const& carriers)
{
  bool found = false;
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    op, [&](unique_ptr<Expression> const* expression) {
      found = found || reads_any(**expression, carriers);
    });
  return found;
}

std::uint64_t width_of(duckdb::LogicalType const& type)
{
  auto const physical = type.InternalType();
  // The rule of push_narrowing_expressions: a variable-size column counts as 16 bytes.
  return duckdb::TypeIsConstantSize(physical) ? duckdb::GetTypeIdSize(physical) : 16;
}

/// Whether @p rows rows of @p width bytes each are at least @p bytes, without overflow.
bool at_least(std::uint64_t rows, std::uint64_t width, std::uint64_t bytes)
{
  if (width == 0) { return bytes == 0; }
  return rows >= bytes / width + (bytes % width == 0 ? 0 : 1);
}

/// The match for the INNER comparison join in @p slot as the lookup join of @p site's aggregate,
/// reached through @p path. Its `merged` is left empty.
std::optional<late_join> match_lookup(aggregate_site const& site,
                                      std::vector<LogicalOperator*> const& path,
                                      unique_ptr<LogicalOperator>& slot)
{
  auto& join = slot->Cast<duckdb::LogicalComparisonJoin>();
  if (join.conditions.size() != 1 || !join.duplicate_eliminated_columns.empty() ||
      join.children[1]->type != LogicalOperatorType::LOGICAL_GET) {
    return std::nullopt;
  }
  auto const& condition = join.conditions[0];
  if (!condition.IsComparison() ||
      condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
    return std::nullopt;
  }
  auto const foreign_key = column_ref(condition.GetLHS());
  auto const lookup_key  = column_ref(condition.GetRHS());
  auto const& key_type   = condition.GetLHS().GetReturnType();
  if (!foreign_key || !lookup_key || condition.GetRHS().GetReturnType() != key_type) {
    return std::nullopt;
  }

  auto& scan = join.children[1]->Cast<duckdb::LogicalGet>();
  // A filtered lookup scan makes its join a filter, which belongs below the large joins.
  if (!scan.children.empty() || scan.table_filters.HasFilters() ||
      scan.table_filters.HasMultiColumnFilters() || !scan.extra_info.file_filters.empty() ||
      scan.EstimateCardinality(site.context) > max_late_join_rows) {
    return std::nullopt;
  }
  auto const scanned = scan.GetColumnBindings();
  if (!contains(join.children[0]->GetColumnBindings(), *foreign_key) ||
      !contains(scanned, *lookup_key)) {
    return std::nullopt;
  }

  late_join match{.aggregate   = &site.slot,
                  .path        = path,
                  .lookup      = &slot,
                  .foreign_key = *foreign_key,
                  .lookup_key  = *lookup_key,
                  .key_type    = key_type};

  scan.ResolveOperatorTypes();
  std::vector<carrier> carriers;
  for (std::size_t i = 0; i < scanned.size(); ++i) {
    carriers.push_back({scanned[i], {scanned[i], scan.types[i]}});
  }
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    auto const& op = **it;
    if (op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
      if (reads_any(op, carriers)) { return std::nullopt; }
      continue;
    }
    auto const table_index = op.Cast<duckdb::LogicalProjection>().table_index;
    for (std::size_t k = 0; k < op.expressions.size(); ++k) {
      if (auto const* carried = carrier_of(*op.expressions[k], carriers)) {
        ColumnBinding const output(table_index, ProjectionIndex(k));
        auto const column = carried->column;
        match.passed_through.push_back(output);
        carriers.push_back({output, column});
      } else if (reads_any(*op.expressions[k], carriers)) {
        return std::nullopt;
      }
    }
  }

  auto const& aggregate = site.slot->Cast<duckdb::LogicalAggregate>();
  bool grouped          = false;
  for (auto const& group : aggregate.groups) {
    if (auto const* carried = carrier_of(*group, carriers)) {
      match.groups.emplace_back(carried->column);
      grouped = true;
    } else if (reads_any(*group, carriers)) {
      return std::nullopt;
    } else {
      match.groups.emplace_back(std::nullopt);
    }
  }
  if (!grouped) { return std::nullopt; }
  for (auto const& expression : aggregate.expressions) {
    if (reads_any(*expression, carriers)) { return std::nullopt; }
  }

  // What the rows stop carrying: each lookup column the join emits, less the copy of the foreign
  // key that takes its place.
  auto const outputs    = join.GetColumnBindings();
  std::uint64_t emitted = 0;
  std::uint64_t copies  = 0;
  for (std::size_t i = 0; i < scanned.size(); ++i) {
    if (!contains(outputs, scanned[i])) { continue; }
    emitted += width_of(scan.types[i]);
    copies += width_of(key_type);
  }
  match.carried_width = emitted - std::min(emitted, copies);
  if (!at_least(site.rows, match.carried_width, site.min_carried_bytes)) {
    SIRIUS_LOG_INFO(
      "Kept a lookup join below its aggregate: {} rows carry {} bytes of its columns each, below "
      "the {} bytes that move it",
      site.rows,
      match.carried_width,
      site.min_carried_bytes);
    return std::nullopt;
  }
  return match;
}

/// The first lookup join of @p site's aggregate at or below @p slot, searching through
/// projections, filters and INNER comparison joins.
std::optional<late_join> find_lookup(aggregate_site const& site,
                                     std::vector<LogicalOperator*>& path,
                                     unique_ptr<LogicalOperator>& slot)
{
  auto& op = *slot;
  if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    if (op.Cast<duckdb::LogicalComparisonJoin>().join_type != duckdb::JoinType::INNER) {
      return std::nullopt;
    }
    if (auto match = match_lookup(site, path, slot)) { return match; }
  } else if (op.type != LogicalOperatorType::LOGICAL_PROJECTION &&
             op.type != LogicalOperatorType::LOGICAL_FILTER) {
    return std::nullopt;
  }
  path.push_back(&op);
  std::optional<late_join> found;
  for (auto& child : op.children) {
    found = find_lookup(site, path, child);
    if (found) { break; }
  }
  path.pop_back();
  return found;
}

/// Each aggregate of @p aggregate over its own result, or nothing when one of them is not a plain
/// `sum`, `min` or `max` that merges into its own result type.
std::optional<duckdb::vector<unique_ptr<Expression>>> merging_aggregates(
  duckdb::ClientContext& context, duckdb::LogicalAggregate const& aggregate)
{
  duckdb::FunctionBinder binder(context);
  duckdb::vector<unique_ptr<Expression>> merged;
  for (std::size_t j = 0; j < aggregate.expressions.size(); ++j) {
    auto const& expression = *aggregate.expressions[j];
    if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
      return std::nullopt;
    }
    auto const& original = expression.Cast<duckdb::BoundAggregateExpression>();
    std::string name     = original.Function().GetName().GetIdentifierName();
    // DuckDB's statistics turn a SUM that cannot overflow into sum_no_overflow.
    if (name == "sum_no_overflow") { name = "sum"; }
    if ((name != "sum" && name != "min" && name != "max") || original.IsDistinct() ||
        original.GetFilter() || original.GetOrderBys()) {
      return std::nullopt;
    }
    auto const& type = original.GetReturnType();
    auto function =
      duckdb::TryGetBuiltinAggregateFunction(context, duckdb::Identifier(name), {type});
    if (!function) { return std::nullopt; }
    duckdb::vector<unique_ptr<Expression>> arguments;
    arguments.push_back(duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
      type, ColumnBinding(aggregate.aggregate_index, ProjectionIndex(j))));
    auto merge = binder.BindAggregateFunction(std::move(function), std::move(arguments));
    if (merge->GetReturnType() != type) { return std::nullopt; }
    merge->SetAlias(original.GetAlias());
    merged.push_back(std::move(merge));
  }
  return merged;
}

std::optional<late_join> find_late_join(duckdb::ClientContext& context,
                                        unique_ptr<LogicalOperator>& slot,
                                        std::uint64_t min_carried_bytes)
{
  auto const& aggregate = slot->Cast<duckdb::LogicalAggregate>();
  auto& child           = slot->children[0];
  if (aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty() ||
      (!aggregate.grouping_sets.empty() &&
       aggregate.grouping_sets[0].size() != aggregate.groups.size()) ||
      !aggregate.has_estimated_cardinality ||
      aggregate.estimated_cardinality > max_late_join_rows || !child->has_estimated_cardinality) {
    return std::nullopt;
  }
  auto merged = merging_aggregates(context, aggregate);
  if (!merged) { return std::nullopt; }
  std::vector<LogicalOperator*> path;
  auto match = find_lookup(aggregate_site{.context           = context,
                                          .slot              = slot,
                                          .rows              = child->estimated_cardinality,
                                          .min_carried_bytes = min_carried_bytes},
                           path,
                           child);
  if (!match) { return std::nullopt; }
  match->merged = std::move(*merged);
  return match;
}

void join_after_aggregate(late_join match, duckdb::Binder& binder)
{
  // A projection takes the lookup join's place and keeps its column positions, so the projection
  // maps above stay valid. It emits the foreign key where the join emitted a lookup column.
  auto lookup         = std::move(*match.lookup);
  auto& probe         = lookup->children[0];
  auto const outputs  = lookup->GetColumnBindings();
  auto const probed   = probe->GetColumnBindings();
  auto const stand_in = binder.GenerateTableIndex();
  probe->ResolveOperatorTypes();
  duckdb::ColumnBindingReplacer replacer;
  duckdb::vector<unique_ptr<Expression>> select_list;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    ColumnBinding const output(stand_in, ProjectionIndex(i));
    auto const it = std::find(probed.begin(), probed.end(), outputs[i]);
    if (it != probed.end()) {
      select_list.push_back(duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        probe->types[static_cast<std::size_t>(it - probed.begin())], outputs[i]));
      replacer.replacement_bindings.emplace_back(outputs[i], output);
    } else {
      select_list.push_back(
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(match.key_type, match.foreign_key));
      replacer.replacement_bindings.emplace_back(outputs[i], output, match.key_type);
    }
  }
  for (auto const& binding : match.passed_through) {
    replacer.replacement_bindings.emplace_back(binding, binding, match.key_type);
  }
  auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(stand_in, std::move(select_list));
  if (lookup->has_estimated_cardinality) {
    projection->SetEstimatedCardinality(lookup->estimated_cardinality);
  }
  auto scan = std::move(lookup->children[1]);
  projection->children.push_back(std::move(probe));
  *match.lookup = std::move(projection);

  auto& lower = (*match.aggregate)->Cast<duckdb::LogicalAggregate>();
  for (auto* op : match.path) {
    replacer.VisitOperatorBindings(*op);
  }
  replacer.VisitOperatorBindings(lower);

  // The lower aggregate gets new indices and the upper one takes over its old ones, so nothing
  // above the aggregate changes.
  auto const group_index     = lower.group_index;
  auto const aggregate_index = lower.aggregate_index;
  auto const groupings_index = lower.groupings_index;
  lower.group_index          = binder.GenerateTableIndex();
  lower.aggregate_index      = binder.GenerateTableIndex();
  if (groupings_index.IsValid()) { lower.groupings_index = binder.GenerateTableIndex(); }

  duckdb::vector<unique_ptr<Expression>> groups;
  ColumnBinding grouped_key;
  for (std::size_t i = 0; i < lower.groups.size(); ++i) {
    ColumnBinding const group(lower.group_index, ProjectionIndex(i));
    if (auto const& column = match.groups[i]) {
      groups.push_back(
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(column->type, column->binding));
      grouped_key = group;
      // The statistics describe the lookup column this group no longer is.
      if (i < lower.group_stats.size()) { lower.group_stats[i].reset(); }
    } else {
      groups.push_back(duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        lower.groups[i]->GetReturnType(), group));
    }
    groups.back()->SetAlias(lower.groups[i]->GetAlias());
  }
  duckdb::ColumnBindingReplacer to_lower;
  for (std::size_t j = 0; j < match.merged.size(); ++j) {
    to_lower.replacement_bindings.emplace_back(
      ColumnBinding(aggregate_index, ProjectionIndex(j)),
      ColumnBinding(lower.aggregate_index, ProjectionIndex(j)));
  }
  for (auto& merge : match.merged) {
    to_lower.VisitExpression(&merge);
  }

  auto const estimate = lower.estimated_cardinality;
  auto join           = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  join->conditions.emplace_back(
    duckdb::make_uniq<duckdb::BoundColumnRefExpression>(match.key_type, grouped_key),
    duckdb::make_uniq<duckdb::BoundColumnRefExpression>(match.key_type, match.lookup_key),
    duckdb::ExpressionType::COMPARE_EQUAL);
  join->SetEstimatedCardinality(estimate);
  auto upper = duckdb::make_uniq<duckdb::LogicalAggregate>(
    group_index, aggregate_index, std::move(match.merged));
  upper->groupings_index = groupings_index;
  upper->groups          = std::move(groups);
  upper->grouping_sets   = lower.grouping_sets;
  upper->SetEstimatedCardinality(estimate);
  join->children.push_back(std::move(*match.aggregate));
  join->children.push_back(std::move(scan));
  upper->children.push_back(std::move(join));
  upper->ResolveOperatorTypes();
  *match.aggregate = std::move(upper);
}

}  // namespace

std::size_t join_dimensions_after_aggregate(duckdb::ClientContext& context,
                                            duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                            duckdb::Binder& binder,
                                            std::uint64_t min_carried_bytes)
{
  // A rewritten aggregate stays in the plan as the lower one, so its children are still the ones
  // to visit; the late join and the upper aggregate above it are not, or they would match again.
  auto& op          = *plan;
  std::size_t moved = 0;
  if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    if (auto match = find_late_join(context, plan, min_carried_bytes)) {
      SIRIUS_LOG_INFO(
        "Moved a lookup join above its aggregate: {} rows carried {} bytes of its columns each, "
        "at least the {} bytes that move it",
        op.children[0]->estimated_cardinality,
        match->carried_width,
        min_carried_bytes);
      join_after_aggregate(std::move(*match), binder);
      ++moved;
    }
  }
  for (auto& child : op.children) {
    moved += join_dimensions_after_aggregate(context, child, binder, min_carried_bytes);
  }
  return moved;
}

}  // namespace sirius::transparent
