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

#include "planner/sirius_plan_flat_tree.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"

#include <exception>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::planner::flat {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::ExpressionType;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::LogicalType;
using duckdb::ProjectionIndex;
using duckdb::TableIndex;
using duckdb::unique_ptr;

unique_ptr<Expression> column(LogicalType const& type, ColumnBinding binding)
{
  return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(type, binding);
}

bool is_column(Expression const& expression)
{
  return expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF;
}

ColumnBinding binding_of(Expression const& expression)
{
  return expression.Cast<duckdb::BoundColumnRefExpression>().Binding();
}

ColumnBinding canonical(std::size_t leaf, duckdb::idx_t column)
{
  return ColumnBinding(TableIndex(leaf), ProjectionIndex(column));
}

/// @p expression over the tree's leaf columns, or null when it reads a column the tree does not
/// output or is volatile (it would be compared, not evaluated, but two calls are not equal).
unique_ptr<Expression> resolve(Expression const& expression, flat_tree const& tree)
{
  if (expression.IsVolatile()) { return nullptr; }
  auto copy     = expression.Copy();
  bool resolved = true;
  duckdb::ExpressionIterator::VisitExpressionClassMutable(
    copy, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& ref) {
      auto const found = tree.columns.find(binding_of(*ref));
      if (found == tree.columns.end()) {
        resolved = false;
        return;
      }
      ref = found->second->Copy();
    });
  return resolved ? std::move(copy) : nullptr;
}

namespace {

/// The plain base-table column behind slot @p slot of @p gets column ids.
std::optional<duckdb::idx_t> base_column(duckdb::LogicalGet const& get, duckdb::idx_t slot)
{
  auto const& ids = get.GetColumnIds();
  if (slot >= ids.size() || !ids[slot].HasPrimaryIndex() || ids[slot].HasChildren() ||
      ids[slot].IsVirtualColumn() || ids[slot].GetPrimaryIndex() >= get.returned_types.size()) {
    return std::nullopt;
  }
  return ids[slot].GetPrimaryIndex();
}

bool add_opaque(LogicalOperator& op, flat_tree& tree)
{
  auto const bindings = op.GetColumnBindings();
  if (bindings.size() != op.types.size()) { return false; }
  auto const id = tree.leaves.size();
  tree.leaves.push_back({leaf_kind::opaque, &op});
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    tree.columns[bindings[i]] = column(op.types[i], canonical(id, i));
  }
  return true;
}

/// A scan as a leaf, its pushed-down filters as predicates. False when a column or filter is not
/// a plain base column, leaving the scan to be an opaque leaf.
bool add_scan(duckdb::LogicalGet& get, flat_tree& tree)
{
  if (!get.children.empty() || get.table_filters.HasMultiColumnFilters()) { return false; }
  auto const id       = tree.leaves.size();
  auto const bindings = get.GetColumnBindings();
  std::vector<std::pair<ColumnBinding, unique_ptr<Expression>>> columns;
  for (auto const& binding : bindings) {
    auto const base = base_column(get, binding.column_index);
    if (!base) { return false; }
    columns.emplace_back(binding, column(get.returned_types[*base], canonical(id, *base)));
  }
  std::vector<unique_ptr<Expression>> filters;
  for (auto const& entry : get.table_filters) {
    auto const base = base_column(get, entry.GetIndex().GetIndex());
    if (!base) { return false; }
    auto const tested = column(get.returned_types[*base], canonical(id, *base));
    filters.push_back(entry.Filter().ToExpression(*tested));
  }
  tree.leaves.push_back({leaf_kind::scan, &get});
  for (auto& [binding, value] : columns) {
    tree.columns[binding] = std::move(value);
  }
  for (auto& filter : filters) {
    tree.predicates.push_back(std::move(filter));
  }
  return true;
}

/// A scan of a materialized CTE as a leaf: every scan of one CTE reads the same rows.
bool add_cte_ref(duckdb::LogicalCTERef& ref, flat_tree& tree)
{
  if (ref.is_recurring || ref.correlated_columns != 0) { return false; }
  auto const id = tree.leaves.size();
  tree.leaves.push_back({leaf_kind::cte_ref, &ref});
  for (std::size_t i = 0; i < ref.chunk_types.size(); ++i) {
    tree.columns[ColumnBinding(ref.table_index, ProjectionIndex(i))] =
      column(ref.chunk_types[i], canonical(id, i));
  }
  return true;
}

bool add_conditions(duckdb::LogicalComparisonJoin const& join, flat_tree& tree)
{
  for (auto const& condition : join.conditions) {
    if (!condition.IsComparison()) { return false; }
    auto lhs = resolve(condition.GetLHS(), tree);
    auto rhs = resolve(condition.GetRHS(), tree);
    if (!lhs || !rhs) { return false; }
    tree.predicates.push_back(duckdb::BoundComparisonExpression::Create(
      condition.GetComparisonType(), std::move(lhs), std::move(rhs)));
  }
  return true;
}

}  // namespace

/// A plain GROUP BY of its one group: no grouping sets beyond `{0}`, which DuckDB may leave
/// implicit.
bool grouped_by_its_group(duckdb::LogicalAggregate const& aggregate)
{
  if (aggregate.groups.size() != 1 || !aggregate.grouping_functions.empty()) { return false; }
  auto const& sets = aggregate.grouping_sets;
  return sets.empty() ||
         (sets.size() == 1 && sets[0].size() == 1 && sets[0].count(ProjectionIndex(0)) == 1);
}

namespace {

/// The distinct values of one CTE column, `[PROJECTION] <- AGGREGATE(GROUP BY ref.c) <- CTE_REF`,
/// as DuckDB builds a delim join's duplicate-eliminated keys under `delim_join_as_cte`.
struct distinct_cte_column {
  TableIndex cte;
  duckdb::idx_t column;
  ColumnBinding output;
  LogicalType type;
};

std::optional<distinct_cte_column> match_distinct_cte_column(LogicalOperator& op)
{
  auto* node                            = &op;
  duckdb::LogicalProjection* projection = nullptr;
  if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
    projection = &node->Cast<duckdb::LogicalProjection>();
    if (projection->expressions.size() != 1 || !is_column(*projection->expressions[0])) {
      return std::nullopt;
    }
    node = node->children[0].get();
  }
  if (node->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY ||
      node->children.size() != 1) {
    return std::nullopt;
  }
  auto& distinct = node->Cast<duckdb::LogicalAggregate>();
  if (!distinct.expressions.empty() || distinct.groups.size() != 1 ||
      !grouped_by_its_group(distinct) || !is_column(*distinct.groups[0]) ||
      distinct.children[0]->type != LogicalOperatorType::LOGICAL_CTE_REF) {
    return std::nullopt;
  }
  ColumnBinding const group(distinct.group_index, ProjectionIndex(0));
  if (projection && !(binding_of(*projection->expressions[0]) == group)) { return std::nullopt; }
  auto const& ref   = distinct.children[0]->Cast<duckdb::LogicalCTERef>();
  auto const source = binding_of(*distinct.groups[0]);
  if (!(source.table_index == ref.table_index) || source.column_index >= ref.chunk_types.size()) {
    return std::nullopt;
  }
  return distinct_cte_column{
    ref.cte_index,
    source.column_index,
    projection ? ColumnBinding(projection->table_index, ProjectionIndex(0)) : group,
    distinct.groups[0]->GetReturnType()};
}

std::size_t add_keys(flat_tree& tree, key_source& keys, LogicalOperator& op)
{
  auto const id = tree.leaves.size();
  tree.leaves.push_back({leaf_kind::keys, &op});
  keys.leaf = id;
  return id;
}

}  // namespace

/// Flatten @p op into @p tree. Operators other than inner joins, cross products, filters,
/// projections, scans and (with @p keys) the key source become opaque leaves. False when a
/// predicate or column cannot be expressed over the leaves.
bool flatten(LogicalOperator& op, flat_tree& tree, key_source* keys)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET:
      return add_scan(op.Cast<duckdb::LogicalGet>(), tree) || add_opaque(op, tree);
    case LogicalOperatorType::LOGICAL_CTE_REF:
      return add_cte_ref(op.Cast<duckdb::LogicalCTERef>(), tree) || add_opaque(op, tree);
    case LogicalOperatorType::LOGICAL_PROJECTION: {
      if (!flatten(*op.children[0], tree, keys)) { return false; }
      auto const& projection = op.Cast<duckdb::LogicalProjection>();
      for (std::size_t i = 0; i < projection.expressions.size(); ++i) {
        auto value = resolve(*projection.expressions[i], tree);
        if (!value) { return false; }
        tree.columns[ColumnBinding(projection.table_index, ProjectionIndex(i))] = std::move(value);
      }
      return true;
    }
    case LogicalOperatorType::LOGICAL_FILTER: {
      if (!flatten(*op.children[0], tree, keys)) { return false; }
      for (auto const& expression : op.expressions) {
        auto predicate = resolve(*expression, tree);
        if (!predicate) { return false; }
        tree.predicates.push_back(std::move(predicate));
      }
      return true;
    }
    case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
      return flatten(*op.children[0], tree, keys) && flatten(*op.children[1], tree, keys);
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
      auto& join = op.Cast<duckdb::LogicalComparisonJoin>();
      if (keys && keys->cte &&
          (join.join_type == duckdb::JoinType::SEMI ||
           join.join_type == duckdb::JoinType::RIGHT_SEMI)) {
        auto const key_child = join.join_type == duckdb::JoinType::SEMI ? 1 : 0;
        if (auto distinct = match_distinct_cte_column(*join.children[key_child])) {
          if (keys->leaf || !flatten(*join.children[1 - key_child], tree, keys) || keys->leaf) {
            return false;
          }
          auto const id                  = add_keys(tree, *keys, *join.children[key_child]);
          keys->cte_index                = distinct->cte;
          keys->cte_column               = distinct->column;
          tree.columns[distinct->output] = column(distinct->type, canonical(id, 0));
          return add_conditions(join, tree);
        }
      }
      if (keys && keys->dedup_ref && join.join_type == duckdb::JoinType::SEMI &&
          join.children[1]->type == LogicalOperatorType::LOGICAL_CTE_REF) {
        auto const& ref = join.children[1]->Cast<duckdb::LogicalCTERef>();
        if (keys->cte_index && ref.cte_index == *keys->cte_index && !ref.is_recurring &&
            ref.correlated_columns == 0) {
          if (keys->leaf || !flatten(*join.children[0], tree, keys) || keys->leaf) { return false; }
          auto const id = add_keys(tree, *keys, *join.children[1]);
          for (std::size_t i = 0; i < ref.chunk_types.size(); ++i) {
            tree.columns[ColumnBinding(ref.table_index, ProjectionIndex(i))] =
              column(ref.chunk_types[i], canonical(id, i));
          }
          return add_conditions(join, tree);
        }
      }
      if (join.join_type != duckdb::JoinType::INNER) { return add_opaque(op, tree); }
      return flatten(*op.children[0], tree, keys) && flatten(*op.children[1], tree, keys) &&
             add_conditions(join, tree);
    }
    case LogicalOperatorType::LOGICAL_DELIM_GET: {
      auto const& delim_get = op.Cast<duckdb::LogicalDelimGet>();
      if (!keys || !keys->delim_get) { return add_opaque(op, tree); }
      if (keys->leaf || delim_get.chunk_types.size() != 1) { return false; }
      auto const id = add_keys(tree, *keys, op);
      tree.columns[ColumnBinding(delim_get.table_index, ProjectionIndex(0))] =
        column(delim_get.chunk_types[0], canonical(id, 0));
      return true;
    }
    default: return add_opaque(op, tree);
  }
}

/// Whether two scans read the same view of the same source. This is the identity the transparent
/// execution path itself compares between plans (`capture_bound_read_view`), which a table
/// function's own bind-data equality does not reliably provide (parquet's does not, even for a
/// scan against itself). Reads of the same files can still select different rows, such as two
/// Iceberg snapshots with different deletes, so their selector evidence must match too.
bool same_source(duckdb::ClientContext& context,
                 duckdb::LogicalGet const& a,
                 duckdb::LogicalGet const& b)
{
  if (a.function.GetName() != b.function.GetName() || a.returned_types != b.returned_types ||
      a.names != b.names) {
    return false;
  }
  try {
    auto const view_a = sirius::op::scan::capture_bound_read_view(a, context);
    auto const view_b = sirius::op::scan::capture_bound_read_view(b, context);
    return view_a.identity && view_b.identity &&
           view_a.identity->fingerprint == view_b.identity->fingerprint &&
           sirius::op::scan::same_selector(view_a, view_b);
  } catch (std::exception const&) {
    return false;
  }
}

/// Whether two leaves produce the same rows: scans of one view, or scans of one CTE.
bool same_rows(duckdb::ClientContext& context, leaf const& a, leaf const& b)
{
  if (a.kind != b.kind) { return false; }
  if (a.kind == leaf_kind::scan) {
    return same_source(context, a.op->Cast<duckdb::LogicalGet>(), b.op->Cast<duckdb::LogicalGet>());
  }
  if (a.kind != leaf_kind::cte_ref) { return false; }
  auto const& ref_a = a.op->Cast<duckdb::LogicalCTERef>();
  auto const& ref_b = b.op->Cast<duckdb::LogicalCTERef>();
  return ref_a.cte_index == ref_b.cte_index && ref_a.chunk_types == ref_b.chunk_types;
}

std::vector<ColumnBinding> columns_read(Expression const& expression)
{
  std::vector<ColumnBinding> read;
  duckdb::ExpressionIterator::VisitExpressionClass(
    expression, ExpressionClass::BOUND_COLUMN_REF, [&](Expression const& ref) {
      read.push_back(binding_of(ref));
    });
  return read;
}

/// Equal, or the same comparison with its operands swapped.
bool same_predicate(Expression const& a, Expression const& b)
{
  using duckdb::BoundComparisonExpression;
  if (a.Equals(b)) { return true; }
  if (!BoundComparisonExpression::IsComparison(a) || !BoundComparisonExpression::IsComparison(b) ||
      b.GetExpressionType() != duckdb::FlipComparisonExpression(a.GetExpressionType())) {
    return false;
  }
  auto const& fa = a.Cast<duckdb::BoundFunctionExpression>();
  auto const& fb = b.Cast<duckdb::BoundFunctionExpression>();
  return BoundComparisonExpression::Left(fa).Equals(BoundComparisonExpression::Right(fb)) &&
         BoundComparisonExpression::Right(fa).Equals(BoundComparisonExpression::Left(fb));
}

/// `x = y` between two plain columns.
std::optional<std::pair<ColumnBinding, ColumnBinding>> column_equality(Expression const& predicate)
{
  using duckdb::BoundComparisonExpression;
  if (predicate.GetExpressionType() != ExpressionType::COMPARE_EQUAL ||
      !BoundComparisonExpression::IsComparison(predicate)) {
    return std::nullopt;
  }
  auto const& comparison = predicate.Cast<duckdb::BoundFunctionExpression>();
  auto const& lhs        = BoundComparisonExpression::Left(comparison);
  auto const& rhs        = BoundComparisonExpression::Right(comparison);
  if (!is_column(lhs) || !is_column(rhs)) { return std::nullopt; }
  return std::pair{binding_of(lhs), binding_of(rhs)};
}

}  // namespace sirius::planner::flat
