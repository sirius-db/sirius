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

#include "planner/sirius_plan_delim_aggregate_reuse.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "log/logging.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace sirius::planner {

namespace {

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

std::size_t count_delim_gets(LogicalOperator const& op)
{
  std::size_t count = op.type == LogicalOperatorType::LOGICAL_DELIM_GET ? 1 : 0;
  for (auto const& child : op.children) {
    count += count_delim_gets(*child);
  }
  return count;
}

// ---------------------------------------------------------------------------------------------
// Flattening
//
// A tree of inner joins, cross products, filters and projections over leaves is the product of
// its leaves filtered by the conjunction of every predicate in it, whatever the join order. Both
// the outer rows and the subquery's input are flattened into that form: their leaves, their
// predicates, and each output column as an expression over leaf columns. Expressions name a leaf
// column canonically as ColumnBinding(leaf ordinal, column). A scan's column is its base-table
// column, so the same column of two scans of one source has the same number.
// ---------------------------------------------------------------------------------------------

enum class leaf_kind { scan, cte_ref, keys, opaque };

struct leaf {
  leaf_kind kind;
  LogicalOperator* op;
};

struct flat_tree {
  std::vector<leaf> leaves;
  duckdb::column_binding_map_t<unique_ptr<Expression>> columns;  ///< Output binding to its value
  std::vector<unique_ptr<Expression>> predicates;
};

/// Where the subquery side gets the outer rows' distinct keys from: a one-column DELIM_GET, or
/// (with `delim_join_as_cte`) a SEMI join against the distinct values of one column of a CTE.
struct key_source {
  bool delim_get = false;
  bool cte       = false;
  std::optional<std::size_t> leaf;
  std::optional<TableIndex> cte_index;
  duckdb::idx_t cte_column = 0;
};

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

/// A plain GROUP BY of its one group: no grouping sets beyond `{0}`, which DuckDB may leave
/// implicit.
bool grouped_by_its_group(duckdb::LogicalAggregate const& aggregate)
{
  if (aggregate.groups.size() != 1 || !aggregate.grouping_functions.empty()) { return false; }
  auto const& sets = aggregate.grouping_sets;
  return sets.empty() ||
         (sets.size() == 1 && sets[0].size() == 1 && sets[0].count(ProjectionIndex(0)) == 1);
}

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

// ---------------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------------

/// Whether two scans read the same view of the same source, by the identity the transparent
/// execution path compares between plans.
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
           view_a.identity->fingerprint == view_b.identity->fingerprint;
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

/// Aggregates whose value over a group does not change when every row of the group is repeated
/// the same number of times. AVG divides a sum by a count that both scale by that number, and is
/// limited to exact (integer and fixed-point) inputs, whose sums do not depend on the order rows
/// are added in.
bool reusable_aggregate(Expression const& expression)
{
  if (expression.GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) { return false; }
  auto const& aggregate = expression.Cast<duckdb::BoundAggregateExpression>();
  if (aggregate.IsDistinct() || aggregate.GetFilter() || aggregate.GetOrderBys() ||
      aggregate.GetChildren().size() != 1) {
    return false;
  }
  auto const name = aggregate.Function().GetName().GetIdentifierName();
  if (name == "min" || name == "max") { return true; }
  if (name != "avg") { return false; }
  auto const& type = aggregate.GetChildren()[0]->GetReturnType();
  return type.IsIntegral() || type.id() == duckdb::LogicalTypeId::DECIMAL;
}

/// What the aggregate reads once rewritten, over the columns of a scan of the outer rows.
struct reuse {
  duckdb::idx_t key_column = 0;  ///< The outer rows' column the delim join is keyed on
  bool key_nullable        = true;
  std::vector<std::vector<unique_ptr<Expression>>> arguments;  ///< Per aggregate
};

/// Whether @p aggregate's input, which reaches the outer rows' distinct values of their column
/// @p key_column (null: whichever column @p keys records) through @p keys, can be replaced by the
/// outer rows @p outer themselves, read through a CTE scan with table index @p scan_index.
///
/// The input is flattened into S, the subquery's own rows, restricted by `S.a = keys.k`. The
/// outer rows must flatten into S' x P with
///  - every leaf of S matching exactly one scan of the same source (a view, or a CTE) in S', and
///    no others in S',
///  - the predicates over S' alone equal to S's (so S' and S hold the same rows),
///  - every predicate across S' and P reading S' only through a' (the image of a),
///  - the key column a', or a column of P that a predicate equates with a'.
/// Then an S' row with a' = k pairs with the same rows of P as any other, so the outer rows with
/// key k are S's rows with a = k, each repeated the same number of times, and grouping them by the
/// key gives the subquery's groups, each repeated uniformly; `reusable_aggregate` is unchanged by
/// that. Keys without outer rows have no subquery rows to group either.
std::optional<reuse> match_aggregate(duckdb::ClientContext& context,
                                     duckdb::LogicalAggregate& aggregate,
                                     LogicalOperator& outer,
                                     key_source keys,
                                     std::optional<duckdb::idx_t> key_column,
                                     TableIndex scan_index)
{
  if (aggregate.children.size() != 1 || aggregate.groups.size() != 1 ||
      !grouped_by_its_group(aggregate) || aggregate.expressions.empty() ||
      !std::ranges::all_of(aggregate.expressions, [](auto const& expression) {
        return reusable_aggregate(*expression);
      })) {
    return std::nullopt;
  }
  flat_tree sub;
  if (!flatten(*aggregate.children[0], sub, &keys) || !keys.leaf) { return std::nullopt; }
  if (!key_column) { key_column = keys.cte_column; }
  flat_tree rows;
  if (!flatten(outer, rows, nullptr)) { return std::nullopt; }

  // Each subquery scan to the one scan of its source among the outer rows.
  std::vector<std::optional<std::size_t>> image(sub.leaves.size());
  std::vector<bool> in_image(rows.leaves.size(), false);
  for (std::size_t i = 0; i < sub.leaves.size(); ++i) {
    if (i == *keys.leaf) { continue; }
    if (sub.leaves[i].kind != leaf_kind::scan && sub.leaves[i].kind != leaf_kind::cte_ref) {
      return std::nullopt;
    }
    for (std::size_t j = 0; j < rows.leaves.size(); ++j) {
      if (!same_rows(context, sub.leaves[i], rows.leaves[j])) { continue; }
      if (image[i] || in_image[j]) { return std::nullopt; }
      image[i]    = j;
      in_image[j] = true;
    }
    if (!image[i]) { return std::nullopt; }
  }
  auto const onto_rows = [&](unique_ptr<Expression> expression) -> unique_ptr<Expression> {
    bool mapped = true;
    duckdb::ExpressionIterator::VisitExpressionClassMutable(
      expression, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& ref) {
        auto const binding = binding_of(*ref);
        auto const target  = image[binding.table_index.index];
        mapped             = mapped && target;
        if (target) {
          ref = column(ref->GetReturnType(), canonical(*target, binding.column_index));
        }
      });
    return mapped ? std::move(expression) : nullptr;
  };
  auto const reads_keys = [&](Expression const& expression) {
    return std::ranges::any_of(columns_read(expression), [&](ColumnBinding binding) {
      return binding.table_index.index == *keys.leaf;
    });
  };

  // S's predicates, and the single `S.a = keys.k` restricting it to the keys.
  ColumnBinding const key_value = canonical(*keys.leaf, 0);
  std::optional<ColumnBinding> sub_key;
  std::vector<unique_ptr<Expression>> sub_predicates;
  for (auto const& predicate : sub.predicates) {
    if (!reads_keys(*predicate)) {
      auto mapped = onto_rows(predicate->Copy());
      if (!mapped) { return std::nullopt; }
      sub_predicates.push_back(std::move(mapped));
      continue;
    }
    auto const equality = column_equality(*predicate);
    if (sub_key || !equality || !(equality->first == key_value || equality->second == key_value)) {
      return std::nullopt;
    }
    sub_key = equality->first == key_value ? equality->second : equality->first;
    if (sub_key->table_index.index == *keys.leaf) { return std::nullopt; }
  }
  if (!sub_key) { return std::nullopt; }
  auto const mapped_key = onto_rows(column(LogicalType::SQLNULL, *sub_key));
  if (!mapped_key) { return std::nullopt; }
  ColumnBinding const row_key = binding_of(*mapped_key);

  // The outer rows' predicates over S' alone must be S's; any across S' and P may read only a'.
  std::vector<bool> matched(sub_predicates.size(), false);
  std::vector<ColumnBinding> equated_with_key;
  for (auto const& predicate : rows.predicates) {
    auto const read = columns_read(*predicate);
    bool const reads_s =
      std::ranges::any_of(read, [&](auto b) { return in_image[b.table_index.index]; });
    bool const reads_p =
      std::ranges::any_of(read, [&](auto b) { return !in_image[b.table_index.index]; });
    if (reads_p && !reads_s) { continue; }
    if (reads_p) {
      bool const only_key = std::ranges::all_of(
        read, [&](auto b) { return !in_image[b.table_index.index] || b == row_key; });
      if (!only_key) { return std::nullopt; }
      if (auto const equality = column_equality(*predicate)) {
        if (equality->first == row_key) { equated_with_key.push_back(equality->second); }
        if (equality->second == row_key) { equated_with_key.push_back(equality->first); }
      }
      continue;
    }
    bool found = false;
    for (std::size_t i = 0; i < sub_predicates.size() && !found; ++i) {
      if (!matched[i] && same_predicate(*predicate, *sub_predicates[i])) {
        matched[i] = found = true;
      }
    }
    if (!found) { return std::nullopt; }
  }
  if (!std::ranges::all_of(matched, [](bool m) { return m; })) { return std::nullopt; }

  // The outer rows' key column is a', or a column of P equated with it.
  auto const outputs = outer.GetColumnBindings();
  if (*key_column >= outputs.size()) { return std::nullopt; }
  auto const& key_output = *rows.columns.at(outputs[*key_column]);
  if (!is_column(key_output)) { return std::nullopt; }
  auto const key_binding = binding_of(key_output);
  if (!(key_binding == row_key) &&
      std::ranges::find(equated_with_key, key_binding) == equated_with_key.end()) {
    return std::nullopt;
  }
  if (outer.types[*key_column] != aggregate.groups[0]->GetReturnType()) { return std::nullopt; }

  // The subquery groups by its key or by the key it is equal to.
  auto const group = resolve(*aggregate.groups[0], sub);
  if (!group || !is_column(*group) ||
      !(binding_of(*group) == *sub_key || binding_of(*group) == key_value)) {
    return std::nullopt;
  }

  // Each argument over the outer rows' output columns.
  duckdb::column_binding_map_t<duckdb::idx_t> output_of;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto const& value = *rows.columns.at(outputs[i]);
    if (is_column(value)) { output_of.emplace(binding_of(value), i); }
  }
  reuse result;
  result.key_column   = *key_column;
  result.key_nullable = equated_with_key.empty();
  for (auto const& expression : aggregate.expressions) {
    auto& arguments = result.arguments.emplace_back();
    for (auto const& child : expression->Cast<duckdb::BoundAggregateExpression>().GetChildren()) {
      auto argument = resolve(*child, sub);
      if (!argument || reads_keys(*argument)) { return std::nullopt; }
      argument      = onto_rows(std::move(argument));
      bool readable = argument != nullptr;
      if (readable) {
        duckdb::ExpressionIterator::VisitExpressionClassMutable(
          argument, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& ref) {
            auto const found = output_of.find(binding_of(*ref));
            readable         = readable && found != output_of.end();
            if (found != output_of.end()) {
              ref = column(ref->GetReturnType(),
                           ColumnBinding(scan_index, ProjectionIndex(found->second)));
            }
          });
      }
      if (!readable) { return std::nullopt; }
      arguments.push_back(std::move(argument));
    }
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Rewriting
// ---------------------------------------------------------------------------------------------

unique_ptr<LogicalOperator> scan_rows(TableIndex scan_index,
                                      TableIndex cte_index,
                                      LogicalOperator const& outer)
{
  duckdb::vector<duckdb::Identifier> names;
  for (std::size_t i = 0; i < outer.types.size(); ++i) {
    names.emplace_back("__sirius_delim_rows_" + std::to_string(i));
  }
  auto scan = duckdb::make_uniq<duckdb::LogicalCTERef>(scan_index, cte_index, outer.types, names);
  scan->SetEstimatedCardinality(outer.estimated_cardinality);
  return scan;
}

/// Point @p aggregate at the outer rows through @p scan, per @p plan.
void reuse_rows(duckdb::LogicalAggregate& aggregate, unique_ptr<LogicalOperator> scan, reuse& plan)
{
  auto const key_type   = aggregate.groups[0]->GetReturnType();
  auto const scan_index = scan->Cast<duckdb::LogicalCTERef>().table_index;
  ColumnBinding const key(scan_index, ProjectionIndex(plan.key_column));
  unique_ptr<LogicalOperator> input = std::move(scan);
  if (plan.key_nullable) {
    // The subquery's `=` never matched a NULL key; a group of NULL keys must not appear either.
    auto not_null = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
      ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
    not_null->GetChildrenMutable().push_back(column(key_type, key));
    auto filter = duckdb::make_uniq<duckdb::LogicalFilter>(std::move(not_null));
    filter->SetEstimatedCardinality(input->estimated_cardinality);
    filter->children.push_back(std::move(input));
    input = std::move(filter);
  }
  aggregate.groups[0] = column(key_type, key);
  for (std::size_t i = 0; i < aggregate.expressions.size(); ++i) {
    auto& children =
      aggregate.expressions[i]->Cast<duckdb::BoundAggregateExpression>().GetChildrenMutable();
    for (std::size_t c = 0; c < children.size(); ++c) {
      children[c] = std::move(plan.arguments[i][c]);
    }
  }
  aggregate.children[0] = std::move(input);
  aggregate.ResolveOperatorTypes();
}

struct rewriter {
  duckdb::ClientContext& context;
  duckdb::Binder& binder;
  unique_ptr<LogicalOperator>& root;
  std::vector<duckdb::LogicalMaterializedCTE*> ctes;  ///< CTEs whose consumer is being visited
  std::size_t rewrites = 0;

  /// `DELIM_JOIN(outer, ... AGGREGATE(input))` whose aggregate can read the outer rows becomes
  /// `CTE(outer) { JOIN(CTE_SCAN, ... AGGREGATE(CTE_SCAN)) }` with the same conditions.
  void delim_join(unique_ptr<LogicalOperator>& slot)
  {
    using duckdb::JoinType;
    auto& delim = slot->Cast<duckdb::LogicalComparisonJoin>();
    if (delim.children.size() != 2 || delim.duplicate_eliminated_columns.size() != 1 ||
        !is_column(*delim.duplicate_eliminated_columns[0]) ||
        std::ranges::any_of(delim.conditions, [](auto const& c) { return !c.IsComparison(); })) {
      return;
    }
    auto const outer_idx = delim.delim_flipped ? std::size_t{1} : std::size_t{0};
    if (!(delim.join_type == JoinType::INNER ||
          (delim.join_type == JoinType::LEFT && outer_idx == 0) ||
          (delim.join_type == JoinType::RIGHT && outer_idx == 1))) {
      return;
    }
    auto& outer = delim.children[outer_idx];
    auto& sub   = delim.children[1 - outer_idx];
    if (count_delim_gets(*outer) != 0 || count_delim_gets(*sub) != 1) { return; }
    // Above the aggregate, only projections and filters of its result.
    auto* node = sub.get();
    while ((node->type == LogicalOperatorType::LOGICAL_PROJECTION ||
            node->type == LogicalOperatorType::LOGICAL_FILTER) &&
           node->children.size() == 1) {
      node = node->children[0].get();
    }
    if (node->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return; }
    auto& aggregate = node->Cast<duckdb::LogicalAggregate>();

    auto const outputs = outer->GetColumnBindings();
    auto const dup     = binding_of(*delim.duplicate_eliminated_columns[0]);
    auto const found   = std::ranges::find(outputs, dup);
    if (found == outputs.end()) { return; }
    key_source keys;
    keys.delim_get        = true;
    auto const scan_index = binder.GenerateTableIndex();
    auto plan             = match_aggregate(context,
                                aggregate,
                                *outer,
                                keys,
                                static_cast<duckdb::idx_t>(found - outputs.begin()),
                                scan_index);
    if (!plan) { return; }

    // ---- Nothing below declines. ----
    auto const cte_index = binder.GenerateTableIndex();
    reuse_rows(aggregate, scan_rows(scan_index, cte_index, *outer), *plan);
    auto const joined_index    = binder.GenerateTableIndex();
    auto joined_rows           = scan_rows(joined_index, cte_index, *outer);
    auto join                  = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(delim.join_type);
    join->conditions           = std::move(delim.conditions);
    join->left_projection_map  = std::move(delim.left_projection_map);
    join->right_projection_map = std::move(delim.right_projection_map);
    join->SetEstimatedCardinality(delim.estimated_cardinality);
    join->children.resize(2);
    join->children[outer_idx]     = std::move(joined_rows);
    join->children[1 - outer_idx] = std::move(sub);
    auto* definition              = outer.get();
    auto const column_count       = outer->types.size();
    auto cte                      = duckdb::make_uniq<duckdb::LogicalMaterializedCTE>(
      duckdb::Identifier("__sirius_delim_rows_" + std::to_string(cte_index.index)),
      cte_index,
      column_count,
      std::move(outer),
      std::move(join),
      duckdb::CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
    cte->SetEstimatedCardinality(delim.estimated_cardinality);
    slot = std::move(cte);
    slot->ResolveOperatorTypes();

    // Everything that read the outer rows' columns, the join conditions included, reads the scan.
    duckdb::vector<ColumnBinding> scanned;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      scanned.emplace_back(joined_index, ProjectionIndex(i));
    }
    duckdb::ColumnBindingReplacer replacer;
    replacer.AddReplacements(outputs, scanned);
    replacer.stop_operator = definition;
    replacer.VisitOperator(*root);
    ++rewrites;
  }

  /// An aggregate restricted to a delim CTE's distinct keys reads that CTE instead.
  void cte_aggregate(LogicalOperator& node)
  {
    auto& aggregate = node.Cast<duckdb::LogicalAggregate>();
    if (ctes.empty() || aggregate.children.size() != 1) { return; }
    key_source probe;
    probe.cte = true;
    flat_tree scratch;
    if (!flatten(*aggregate.children[0], scratch, &probe) || !probe.cte_index) { return; }
    auto const cte = std::ranges::find_if(
      ctes, [&](auto const* candidate) { return candidate->table_index == *probe.cte_index; });
    if (cte == ctes.end()) { return; }
    auto& outer = *(*cte)->children[0];
    key_source keys;
    keys.cte              = true;
    auto const scan_index = binder.GenerateTableIndex();
    auto plan = match_aggregate(context, aggregate, outer, keys, std::nullopt, scan_index);
    if (!plan) { return; }
    reuse_rows(aggregate, scan_rows(scan_index, *probe.cte_index, outer), *plan);
    ++rewrites;
  }

  void visit(unique_ptr<LogicalOperator>& slot)
  {
    if (slot->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) { delim_join(slot); }
    if (slot->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { cte_aggregate(*slot); }
    bool const is_cte = slot->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE;
    for (std::size_t c = 0; c < slot->children.size(); ++c) {
      bool const consumer = is_cte && c == 1;
      if (consumer) { ctes.push_back(&slot->Cast<duckdb::LogicalMaterializedCTE>()); }
      visit(slot->children[c]);
      if (consumer) { ctes.pop_back(); }
    }
  }
};

}  // namespace

std::size_t reuse_delim_rows_for_subquery_aggregates(
  duckdb::ClientContext& context,
  duckdb::Binder& binder,
  duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  plan->ResolveOperatorTypes();
  rewriter pass{context, binder, plan, {}, 0};
  pass.visit(plan);
  if (pass.rewrites > 0) {
    SIRIUS_LOG_INFO(
      "[sirius_plan_delim_aggregate_reuse] Computed {} subquery aggregate(s) from the delim join's "
      "outer rows instead of rescanning the subquery's tables",
      pass.rewrites);
  }
  return pass.rewrites;
}

}  // namespace sirius::planner
