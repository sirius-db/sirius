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

#include "planner/delim_cte_rewrites.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "log/logging.hpp"
#include "planner/delim_cte_proof.hpp"
#include "planner/sirius_plan_flat_tree.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::ExpressionType;
using duckdb::idx_t;
using duckdb::JoinType;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::LogicalType;
using duckdb::ProjectionIndex;
using duckdb::TableIndex;
using duckdb::unique_ptr;
using flat::binding_of;
using flat::canonical;
using flat::column;
using flat::column_equality;
using flat::is_column;

/// Without the `EXISTS`, the delim CTE holds every row of the outer side. The rewrite is limited
/// to outer sides that can produce at most this share of the subquery's largest scan, the bound
/// DuckDB puts on a CTE whose keys it pushes into another scan
/// (`CTEJoinFilterPusher::MAX_SOURCE_RATIO`).
constexpr idx_t outer_rows_divisor = 4;

/// Tables one correlation column's keys are carried to.
constexpr std::size_t max_carries_per_key = 3;

std::size_t count_refs(LogicalOperator const& op, TableIndex cte)
{
  std::size_t count = op.type == LogicalOperatorType::LOGICAL_CTE_REF &&
                          op.Cast<duckdb::LogicalCTERef>().cte_index == cte
                        ? 1
                        : 0;
  for (auto const& child : op.children) {
    count += count_refs(*child, cte);
  }
  return count;
}

LogicalOperator* skip_projections(LogicalOperator* op)
{
  while (op->type == LogicalOperatorType::LOGICAL_PROJECTION && op->children.size() == 1) {
    op = op->children[0].get();
  }
  return op;
}

bool is_key_comparison(ExpressionType type)
{
  return type == ExpressionType::COMPARE_EQUAL || type == ExpressionType::COMPARE_NOT_DISTINCT_FROM;
}

/// The column of `IS NOT NULL(column)`.
std::optional<ColumnBinding> not_null_column(Expression const& predicate)
{
  if (predicate.GetExpressionType() != ExpressionType::OPERATOR_IS_NOT_NULL) {
    return std::nullopt;
  }
  auto const& children = predicate.Cast<duckdb::BoundOperatorExpression>().GetChildren();
  if (children.size() != 1 || !is_column(*children[0])) { return std::nullopt; }
  return binding_of(*children[0]);
}

// ---------------------------------------------------------------------------------------------
// EXISTS from the aggregate
// ---------------------------------------------------------------------------------------------

/// The group of @p aggregate that @p binding, an output of @p top, passes through projections to.
std::optional<idx_t> group_behind(ColumnBinding binding,
                                  LogicalOperator const& top,
                                  duckdb::LogicalAggregate const& aggregate)
{
  auto const* op = &top;
  while (op != &aggregate) {
    if (op->type != LogicalOperatorType::LOGICAL_PROJECTION || op->children.size() != 1) {
      return std::nullopt;
    }
    auto const& projection = op->Cast<duckdb::LogicalProjection>();
    idx_t const slot       = binding.column_index;
    if (!(binding.table_index == projection.table_index) || slot >= projection.expressions.size() ||
        !is_column(*projection.expressions[slot])) {
      return std::nullopt;
    }
    binding = binding_of(*projection.expressions[slot]);
    op      = op->children[0].get();
  }
  idx_t const group = binding.column_index;
  if (!(binding.table_index == aggregate.group_index) || group >= aggregate.groups.size()) {
    return std::nullopt;
  }
  return group;
}

bool grouped_once_by_all_groups(duckdb::LogicalAggregate const& aggregate)
{
  if (!aggregate.grouping_functions.empty()) { return false; }
  auto const& sets = aggregate.grouping_sets;
  return sets.empty() || (sets.size() == 1 && sets[0].size() == aggregate.groups.size());
}

/// A conjunction of equalities between plain columns: a join predicate, which selects nothing on
/// its own (DuckDB's `IsNonSelectiveJoinPredicate`).
bool is_join_equality(Expression const& predicate)
{
  if (predicate.GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
    bool all = true;
    duckdb::ExpressionIterator::EnumerateChildren(
      predicate, [&](Expression const& child) { all = all && is_join_equality(child); });
    return all;
  }
  return column_equality(predicate).has_value();
}

/// Whether @p tree filters its rows by more than join equalities and NULL checks (DuckDB's
/// `HasSelection`, by which its rewrite decides that a delim CTE's keys restrict anything).
bool has_selection(flat::flat_tree const& tree)
{
  return std::ranges::any_of(tree.predicates, [](auto const& predicate) {
    return !is_join_equality(*predicate) && !not_null_column(*predicate);
  });
}

/// The rows DuckDB estimates for table scan @p op, or nothing when its function gives no estimate
/// (CSV, Arrow and most extension scans). `EstimateCardinality` reports such a scan as one row,
/// which would pass for a tiny table.
std::optional<idx_t> scan_rows(LogicalOperator& op, duckdb::ClientContext& context)
{
  auto& get = op.Cast<duckdb::LogicalGet>();
  if (get.has_estimated_cardinality) { return get.estimated_cardinality; }
  if (!get.function.cardinality || !get.bind_data) { return std::nullopt; }
  auto const stats = get.function.cardinality(context, get.bind_data.get());
  if (!stats || !stats->has_estimated_cardinality) { return std::nullopt; }
  return stats->estimated_cardinality;
}

/// The most rows @p tree can produce, or nothing when a leaf is not a table scan with a row
/// estimate. A flattened tree joins its leaves, and without a proof that a join key is unique a
/// join can pair every row of one leaf with every row of another, so the bound is the product of
/// the leaves' rows, not their sum.
std::optional<idx_t> most_rows(flat::flat_tree const& tree, duckdb::ClientContext& context)
{
  constexpr auto unbounded = std::numeric_limits<idx_t>::max();
  idx_t product            = 1;
  for (auto const& leaf : tree.leaves) {
    if (leaf.op->type != LogicalOperatorType::LOGICAL_GET || !leaf.op->children.empty()) {
      return std::nullopt;
    }
    auto const rows = scan_rows(*leaf.op, context);
    if (!rows) { return std::nullopt; }
    product = *rows != 0 && product > unbounded / *rows ? unbounded : product * *rows;
  }
  return product;
}

/// The rows of the largest table scan under @p op that has a row estimate, 0 when none has.
idx_t largest_scan(LogicalOperator& op, duckdb::ClientContext& context)
{
  idx_t largest = 0;
  if (op.type == LogicalOperatorType::LOGICAL_GET && op.children.empty()) {
    largest = scan_rows(op, context).value_or(0);
  }
  for (auto& child : op.children) {
    largest = std::max(largest, largest_scan(*child, context));
  }
  return largest;
}

/// What `answer_exists_from_aggregate` changes for one delim CTE, with everything already built.
struct exists_edit {
  duckdb::LogicalProjection* body          = nullptr;  ///< The CTE's projection of its outer rows
  duckdb::LogicalComparisonJoin* mark      = nullptr;  ///< The `EXISTS` join to remove
  duckdb::LogicalComparisonJoin* join_back = nullptr;
  std::vector<std::pair<std::size_t, unique_ptr<Expression>>> constants;  ///< TRUE per mark use
};

void commit(exists_edit& edit) noexcept
{
  for (auto& [slot, constant] : edit.constants) {
    edit.body->expressions[slot] = std::move(constant);
  }
  auto outer                = std::move(edit.mark->children[0]);
  edit.body->children[0]    = std::move(outer);  // Releases the filter, the MARK join and its side.
  edit.join_back->join_type = JoinType::INNER;
}

std::optional<exists_edit> match_exists_twin(duckdb::ClientContext& context,
                                             LogicalOperator& root,
                                             duckdb::LogicalMaterializedCTE& cte)
{
  auto const parts = match_delim_cte(cte);
  if (!parts || !can_restrict(*parts, context)) { return std::nullopt; }
  auto& join_back = *parts->join_back;
  if (join_back.join_type != JoinType::LEFT && join_back.join_type != JoinType::INNER) {
    return std::nullopt;
  }
  // Only the join-back and the key read see the CTE's rows, and only the subquery sees its keys.
  if (count_refs(root, cte.table_index) != 2 || count_refs(root, parts->dedup->table_index) != 1) {
    return std::nullopt;
  }

  // Key k of the dedup CTE is column key_column[k] of the delim CTE.
  auto const& distinct = parts->dedup->children[0]->Cast<duckdb::LogicalAggregate>();
  std::vector<idx_t> key_column;
  for (auto const& group : distinct.groups) {
    if (!is_column(*group) || !(binding_of(*group).table_index == parts->key_read->table_index)) {
      return std::nullopt;
    }
    key_column.push_back(binding_of(*group).column_index);
  }
  auto const keys = key_column.size();
  if (keys == 0 || join_back.conditions.size() != keys) { return std::nullopt; }

  // The join-back's subquery side: projections over an aggregate grouped exactly by the keys. A
  // HAVING would put a filter in between, whose rejected groups an INNER join-back would drop.
  auto* const aggregate_op = skip_projections(join_back.children[1].get());
  if (aggregate_op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY ||
      aggregate_op->children.size() != 1) {
    return std::nullopt;
  }
  auto& aggregate = aggregate_op->Cast<duckdb::LogicalAggregate>();
  if (aggregate.groups.size() != keys || !grouped_once_by_all_groups(aggregate)) {
    return std::nullopt;
  }

  // The aggregate's input, restricted to the keys by one `column = key` per key.
  flat::key_source source;
  source.dedup_ref = true;
  source.cte_index = parts->dedup->table_index;
  flat::flat_tree sub;
  if (!flat::flatten(*aggregate.children[0], sub, &source) || !source.leaf) { return std::nullopt; }
  auto const keys_leaf  = *source.leaf;
  auto const reads_keys = [&](Expression const& expression) {
    return std::ranges::any_of(flat::columns_read(expression), [&](ColumnBinding binding) {
      return binding.table_index.index == keys_leaf;
    });
  };
  std::vector<std::optional<ColumnBinding>> sub_key(keys);
  for (auto const& predicate : sub.predicates) {
    if (!reads_keys(*predicate)) { continue; }
    auto const equality = column_equality(*predicate);
    if (!equality) { return std::nullopt; }
    auto [mine, key] = *equality;
    if (key.table_index.index != keys_leaf) { std::swap(mine, key); }
    idx_t const k = key.column_index;
    if (key.table_index.index != keys_leaf || mine.table_index.index == keys_leaf || k >= keys ||
        sub_key[k]) {
      return std::nullopt;
    }
    sub_key[k] = mine;
  }
  if (std::ranges::any_of(sub_key, [](auto const& key) { return !key; })) { return std::nullopt; }

  // Group g holds the subquery's column of key group_key[g].
  std::vector<idx_t> group_key(keys);
  std::vector<bool> grouped(keys, false);
  for (std::size_t g = 0; g < keys; ++g) {
    auto const value = flat::resolve(*aggregate.groups[g], sub);
    if (!value || !is_column(*value)) { return std::nullopt; }
    auto const found =
      std::ranges::find_if(sub_key, [&](auto const& key) { return *key == binding_of(*value); });
    auto const k = static_cast<std::size_t>(found - sub_key.begin());
    if (found == sub_key.end() || grouped[k]) { return std::nullopt; }
    group_key[g] = k;
    grouped[k]   = true;
  }

  // The join-back joins each key's column of the outer rows to that key's group, nothing else.
  std::vector<bool> joined(keys, false);
  for (auto const& condition : join_back.conditions) {
    if (!condition.IsComparison() || !is_key_comparison(condition.GetComparisonType()) ||
        !is_column(condition.GetLHS()) || !is_column(condition.GetRHS())) {
      return std::nullopt;
    }
    auto const outer = binding_of(condition.GetLHS());
    auto const group =
      group_behind(binding_of(condition.GetRHS()), *join_back.children[1], aggregate);
    if (!(outer.table_index == parts->outer_row_read->table_index) || !group) {
      return std::nullopt;
    }
    auto const k         = group_key[*group];
    idx_t const outer_at = outer.column_index;
    if (joined[k] || outer_at != key_column[k]) { return std::nullopt; }
    joined[k] = true;
  }

  // The CTE's rows: a projection of exactly the rows the EXISTS keeps.
  if (cte.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION ||
      cte.children[0]->children.size() != 1 ||
      cte.children[0]->children[0]->type != LogicalOperatorType::LOGICAL_FILTER) {
    return std::nullopt;
  }
  auto& body   = cte.children[0]->Cast<duckdb::LogicalProjection>();
  auto& filter = body.children[0]->Cast<duckdb::LogicalFilter>();
  if (filter.expressions.size() != 1 || !filter.projection_map.empty() ||
      filter.children.size() != 1 ||
      filter.children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return std::nullopt;
  }
  auto& mark = filter.children[0]->Cast<duckdb::LogicalComparisonJoin>();
  if (mark.join_type != JoinType::MARK || mark.children.size() != 2 ||
      mark.conditions.size() != keys || !mark.left_projection_map.empty() ||
      !mark.right_projection_map.empty()) {
    return std::nullopt;
  }
  ColumnBinding const mark_column(mark.mark_index, ProjectionIndex(0));
  if (!is_column(*filter.expressions[0]) || !(binding_of(*filter.expressions[0]) == mark_column)) {
    return std::nullopt;
  }

  exists_edit edit;
  edit.body      = &body;
  edit.mark      = &mark;
  edit.join_back = &join_back;
  // Every row the filter keeps has a TRUE mark, so the constant stands in for the column.
  for (std::size_t i = 0; i < body.expressions.size(); ++i) {
    auto const& expression = *body.expressions[i];
    if (is_column(expression) && binding_of(expression) == mark_column) {
      edit.constants.emplace_back(
        i, duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BOOLEAN(true)));
    } else if (std::ranges::any_of(flat::columns_read(expression),
                                   [&](ColumnBinding read) { return read == mark_column; })) {
      return std::nullopt;
    }
  }

  // The EXISTS correlates each key's column of the outer rows with one column of its own side.
  flat::flat_tree exists;
  if (!flat::flatten(*mark.children[1], exists, nullptr)) { return std::nullopt; }
  std::vector<std::optional<ColumnBinding>> exists_key(keys);
  std::vector<bool> null_safe(keys, false);
  for (auto const& condition : mark.conditions) {
    if (!condition.IsComparison() || !is_key_comparison(condition.GetComparisonType()) ||
        !is_column(condition.GetLHS()) || !is_column(condition.GetRHS())) {
      return std::nullopt;
    }
    auto const outer = binding_of(condition.GetLHS());
    std::optional<std::size_t> k;
    for (std::size_t i = 0; i < keys && !k; ++i) {
      if (key_column[i] >= body.expressions.size()) { return std::nullopt; }
      auto const& projected = *body.expressions[key_column[i]];
      if (is_column(projected) && binding_of(projected) == outer && !exists_key[i]) { k = i; }
    }
    auto const value = exists.columns.find(binding_of(condition.GetRHS()));
    if (!k || value == exists.columns.end() || !is_column(*value->second)) { return std::nullopt; }
    exists_key[*k] = binding_of(*value->second);
    null_safe[*k]  = condition.GetComparisonType() == ExpressionType::COMPARE_NOT_DISTINCT_FROM;
  }

  // Both sides read the same rows: the same sources in the same order ...
  std::vector<std::size_t> image(sub.leaves.size(), 0);
  std::size_t paired = 0;
  for (std::size_t i = 0; i < sub.leaves.size(); ++i) {
    if (i == keys_leaf) { continue; }
    if (paired >= exists.leaves.size() ||
        !flat::same_rows(context, sub.leaves[i], exists.leaves[paired])) {
      return std::nullopt;
    }
    image[i] = paired++;
  }
  if (paired != exists.leaves.size()) { return std::nullopt; }
  auto const onto_exists = [&](ColumnBinding binding) {
    return canonical(image[binding.table_index.index], binding.column_index);
  };
  // ... keyed by the same column ...
  for (std::size_t k = 0; k < keys; ++k) {
    if (!(onto_exists(*sub_key[k]) == *exists_key[k])) { return std::nullopt; }
  }
  // ... under the same predicates. `IS NOT NULL` on a key column is left out on both sides: a
  // NULL there matches no key through `=`, and the NULL handling is compared separately below.
  auto const key_of = [&](ColumnBinding binding) -> std::optional<std::size_t> {
    for (std::size_t k = 0; k < keys; ++k) {
      if (*exists_key[k] == binding) { return k; }
    }
    return std::nullopt;
  };
  std::vector<bool> not_null(keys, false);
  std::vector<Expression const*> exists_predicates;
  for (auto const& predicate : exists.predicates) {
    auto const tested = not_null_column(*predicate);
    if (auto const k = tested ? key_of(*tested) : std::nullopt) {
      not_null[*k] = true;
      continue;
    }
    exists_predicates.push_back(predicate.get());
  }
  std::vector<unique_ptr<Expression>> sub_predicates;
  for (auto const& predicate : sub.predicates) {
    if (reads_keys(*predicate)) { continue; }
    auto mapped = predicate->Copy();
    duckdb::ExpressionIterator::VisitExpressionClassMutable(
      mapped, duckdb::ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& ref) {
        ref = column(ref->GetReturnType(), onto_exists(binding_of(*ref)));
      });
    auto const tested = not_null_column(*mapped);
    if (tested && key_of(*tested)) { continue; }
    sub_predicates.push_back(std::move(mapped));
  }
  if (sub_predicates.size() != exists_predicates.size()) { return std::nullopt; }
  std::vector<bool> matched(sub_predicates.size(), false);
  for (auto const* predicate : exists_predicates) {
    bool found = false;
    for (std::size_t i = 0; i < sub_predicates.size() && !found; ++i) {
      if (!matched[i] && flat::same_predicate(*predicate, *sub_predicates[i])) {
        matched[i] = found = true;
      }
    }
    if (!found) { return std::nullopt; }
  }
  // The subquery matches keys with `=`, which no NULL satisfies. A null-safe EXISTS agrees only
  // when its own side has no NULL key to match a NULL outer key with.
  for (std::size_t k = 0; k < keys; ++k) {
    if (null_safe[k] && !not_null[k]) { return std::nullopt; }
  }

  // Without the EXISTS the CTE holds all of the outer side, so that side must still select rows
  // and, at the most rows its joins can produce, be small next to the subquery.
  flat::flat_tree outer;
  if (!flat::flatten(*mark.children[0], outer, nullptr) || !has_selection(outer)) {
    return std::nullopt;
  }
  auto const outer_rows = most_rows(outer, context);
  auto const largest    = largest_scan(*mark.children[1], context);
  if (!outer_rows || *outer_rows > largest / outer_rows_divisor) { return std::nullopt; }
  return edit;
}

std::optional<exists_edit> find_exists_twin(duckdb::ClientContext& context,
                                            LogicalOperator& root,
                                            LogicalOperator& op)
{
  if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && op.children.size() == 2) {
    if (auto edit = match_exists_twin(context, root, op.Cast<duckdb::LogicalMaterializedCTE>())) {
      return edit;
    }
  }
  for (auto& child : op.children) {
    if (auto edit = find_exists_twin(context, root, *child)) { return edit; }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------------------------
// Keys across an equality
// ---------------------------------------------------------------------------------------------

void collect_dedup_ctes(LogicalOperator& op, std::unordered_set<idx_t>& dedups)
{
  if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
    if (auto const parts = match_delim_cte(op.Cast<duckdb::LogicalMaterializedCTE>())) {
      dedups.insert(parts->dedup->table_index.index);
    }
  }
  for (auto& child : op.children) {
    collect_dedup_ctes(*child, dedups);
  }
}

/// A SEMI join whose build side is a scan of one of @p dedups: a subquery's restriction to the
/// outer rows' keys.
duckdb::LogicalComparisonJoin* as_key_join(LogicalOperator& op,
                                           std::unordered_set<idx_t> const& dedups)
{
  if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.children.size() != 2) {
    return nullptr;
  }
  auto& join = op.Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != JoinType::SEMI ||
      join.children[1]->type != LogicalOperatorType::LOGICAL_CTE_REF) {
    return nullptr;
  }
  auto const& ref = join.children[1]->Cast<duckdb::LogicalCTERef>();
  return dedups.count(ref.cte_index.index) != 0 && !ref.is_recurring && ref.correlated_columns == 0
           ? &join
           : nullptr;
}

struct typed_equality {
  ColumnBinding a;
  ColumnBinding b;
  LogicalType type;
};

void add_equality(Expression const& predicate, std::vector<typed_equality>& equalities)
{
  auto const equality = column_equality(predicate);
  if (!equality) { return; }
  auto const& comparison = predicate.Cast<duckdb::BoundFunctionExpression>();
  auto const& lhs        = duckdb::BoundComparisonExpression::Left(comparison).GetReturnType();
  auto const& rhs        = duckdb::BoundComparisonExpression::Right(comparison).GetReturnType();
  if (lhs == rhs) { equalities.push_back({equality->first, equality->second, lhs}); }
}

/// Equalities that hold for every row of @p op, from the filters and inner joins that pass its
/// columns through unchanged.
void equalities_within(LogicalOperator const& op, std::vector<typed_equality>& equalities)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_FILTER:
      if (!op.Cast<duckdb::LogicalFilter>().projection_map.empty()) { return; }
      for (auto const& expression : op.expressions) {
        add_equality(*expression, equalities);
      }
      break;
    case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: break;
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
      auto const& join = op.Cast<duckdb::LogicalComparisonJoin>();
      if (join.join_type != JoinType::INNER || !join.left_projection_map.empty() ||
          !join.right_projection_map.empty()) {
        return;
      }
      for (auto const& condition : join.conditions) {
        if (condition.IsComparison() &&
            condition.GetComparisonType() == ExpressionType::COMPARE_EQUAL &&
            is_column(condition.GetLHS()) && is_column(condition.GetRHS()) &&
            condition.GetLHS().GetReturnType() == condition.GetRHS().GetReturnType()) {
          equalities.push_back({binding_of(condition.GetLHS()),
                                binding_of(condition.GetRHS()),
                                condition.GetLHS().GetReturnType()});
        }
      }
      break;
    }
    default: return;
  }
  for (auto const& child : op.children) {
    equalities_within(*child, equalities);
  }
}

/// One added key join, built and waiting for the join it goes above.
struct carry_edit {
  unique_ptr<LogicalOperator>* slot = nullptr;  ///< Where the subquery's existing key join hangs
  unique_ptr<duckdb::LogicalComparisonJoin> join;
};

void commit(carry_edit& edit) noexcept
{
  edit.join->children[0] = std::move(*edit.slot);
  *edit.slot             = std::move(edit.join);
}

void plan_carries(duckdb::Binder& binder,
                  LogicalOperator& op,
                  std::unordered_set<idx_t> const& dedups,
                  std::vector<carry_edit>& edits)
{
  for (auto& child : op.children) {
    plan_carries(binder, *child, dedups, edits);
  }
  if (op.type != LogicalOperatorType::LOGICAL_FILTER || op.children.size() != 1 ||
      !op.Cast<duckdb::LogicalFilter>().projection_map.empty()) {
    return;
  }
  // The key joins directly under the filter, and the subquery's rows under those.
  std::vector<duckdb::LogicalComparisonJoin*> key_joins;
  auto* input = op.children[0].get();
  while (auto* const join = as_key_join(*input, dedups)) {
    if (!join->left_projection_map.empty() || !join->right_projection_map.empty()) { return; }
    key_joins.push_back(join);
    input = join->children[0].get();
  }
  if (key_joins.empty()) { return; }

  std::vector<typed_equality> equalities;
  for (auto const& expression : op.expressions) {
    add_equality(*expression, equalities);
  }
  equalities_within(*input, equalities);
  if (equalities.empty()) { return; }
  duckdb::column_binding_set_t produced;
  for (auto const& binding : input->GetColumnBindings()) {
    produced.insert(binding);
  }

  // (dedup CTE, key column, restricted column) of every key join, existing or planned.
  struct restriction {
    TableIndex cte;
    idx_t key;
    ColumnBinding column;
  };
  std::vector<restriction> restricted;
  struct key_use {
    duckdb::LogicalCTERef const* ref;
    idx_t key;
    ColumnBinding column;
    LogicalType type;
  };
  std::vector<key_use> uses;
  for (auto const* join : key_joins) {
    auto const& ref = join->children[1]->Cast<duckdb::LogicalCTERef>();
    for (auto const& condition : join->conditions) {
      if (!condition.IsComparison() || !is_key_comparison(condition.GetComparisonType()) ||
          !is_column(condition.GetLHS()) || !is_column(condition.GetRHS())) {
        continue;
      }
      auto const key     = binding_of(condition.GetRHS());
      idx_t const key_at = key.column_index;
      if (!(key.table_index == ref.table_index) || key_at >= ref.chunk_types.size()) { continue; }
      auto const restricted_column = binding_of(condition.GetLHS());
      restricted.push_back({ref.cte_index, key_at, restricted_column});
      if (condition.GetLHS().GetReturnType() == ref.chunk_types[key_at]) {
        uses.push_back({&ref, key_at, restricted_column, ref.chunk_types[key_at]});
      }
    }
  }

  for (auto const& use : uses) {
    // Every column the equalities make equal to the restricted one.
    std::vector<ColumnBinding> equal{use.column};
    for (std::size_t next = 0; next < equal.size(); ++next) {
      for (auto const& equality : equalities) {
        if (!(equality.type == use.type)) { continue; }
        for (auto const& [from, to] :
             {std::pair{equality.a, equality.b}, std::pair{equality.b, equality.a}}) {
          if (from == equal[next] && std::ranges::find(equal, to) == equal.end()) {
            equal.push_back(to);
          }
        }
      }
    }
    std::size_t carried = 0;
    for (std::size_t i = 1; i < equal.size() && carried < max_carries_per_key; ++i) {
      auto const target = equal[i];
      // One restriction per relation: another column of an already restricted relation adds
      // nothing the first does not.
      bool const covered = std::ranges::any_of(restricted, [&](restriction const& r) {
        return r.cte == use.ref->cte_index && r.key == use.key &&
               r.column.table_index == target.table_index;
      });
      if (covered || produced.count(target) == 0) { continue; }

      auto scan = duckdb::make_uniq<duckdb::LogicalCTERef>(binder.GenerateTableIndex(),
                                                           use.ref->cte_index,
                                                           use.ref->chunk_types,
                                                           use.ref->bound_columns);
      auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(JoinType::SEMI);
      join->conditions.emplace_back(
        column(use.type, target),
        column(use.type, ColumnBinding(scan->table_index, ProjectionIndex(use.key))),
        ExpressionType::COMPARE_EQUAL);
      join->children.push_back(nullptr);
      join->children.push_back(std::move(scan));
      edits.push_back({&op.children[0], std::move(join)});
      restricted.push_back({use.ref->cte_index, use.key, target});
      ++carried;
    }
  }
}

}  // namespace

std::size_t answer_exists_from_aggregate(duckdb::ClientContext& context, LogicalOperator& plan)
{
  // One at a time: removing an EXISTS side removes any delim CTE nested inside it.
  std::size_t answered = 0;
  while (auto edit = find_exists_twin(context, plan, plan)) {
    commit(*edit);
    ++answered;
  }
  if (answered > 0) {
    SIRIUS_LOG_INFO(
      "[delim_cte_rewrites] Answered {} EXISTS from the aggregate subquery over the same rows",
      answered);
  }
  return answered;
}

std::size_t carry_subquery_keys_across_equalities(duckdb::ClientContext& /*context*/,
                                                  duckdb::Binder& binder,
                                                  LogicalOperator& plan)
{
  std::unordered_set<idx_t> dedups;
  collect_dedup_ctes(plan, dedups);
  if (dedups.empty()) { return 0; }
  std::vector<carry_edit> edits;
  plan_carries(binder, plan, dedups, edits);
  for (auto& edit : edits) {
    commit(edit);
  }
  if (!edits.empty()) {
    SIRIUS_LOG_INFO(
      "[delim_cte_rewrites] Carried a subquery's keys across an equality to {} more "
      "relation(s)",
      edits.size());
  }
  return edits.size();
}

}  // namespace sirius::planner
