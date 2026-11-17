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
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "log/logging.hpp"
#include "planner/sirius_plan_flat_tree.hpp"

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
using namespace sirius::planner::flat;  // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------------

/// Aggregates whose value over a group does not change when every row of the group is repeated
/// the same number of times. AVG divides a sum by a count that both scale by that number, and is
/// limited to exact (integer and fixed-point) inputs, whose sums do not depend on the order rows
/// are added in. Repetition also scales the sum itself, so AVG is further limited to inputs whose
/// sum cannot overflow at any row count: DuckDB sums INTEGER and BIGINT values, and DECIMALs
/// stored as them, in 128 bits, where (2^64 - 1) rows of magnitude at most 2^63 stay below
/// 2^127 - 1. A narrower input is summed in 64 bits and a 128-bit input in 128 bits, so either
/// can overflow on repeated rows where the original aggregate does not.
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
  if (!type.IsIntegral() && type.id() != duckdb::LogicalTypeId::DECIMAL) { return false; }
  auto const physical = type.InternalType();
  return physical == duckdb::PhysicalType::INT32 || physical == duckdb::PhysicalType::INT64;
}

/// What the aggregate reads once rewritten, over the columns of a scan of the outer rows.
struct reuse {
  duckdb::idx_t key_column = 0;  ///< The outer rows' column the subquery is keyed on
  bool key_nullable        = true;
  std::vector<std::vector<unique_ptr<Expression>>> arguments;  ///< Per aggregate
};

/// Whether @p aggregate's input, which reaches the outer rows' distinct values of the column
/// @p keys records through @p keys, can be replaced by the outer rows @p outer themselves, read
/// through a CTE scan with table index @p scan_index.
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
  auto const key_column = keys.cte_column;
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
  if (key_column >= outputs.size()) { return std::nullopt; }
  auto const& key_output = *rows.columns.at(outputs[key_column]);
  if (!is_column(key_output)) { return std::nullopt; }
  auto const key_binding = binding_of(key_output);
  if (!(key_binding == row_key) &&
      std::ranges::find(equated_with_key, key_binding) == equated_with_key.end()) {
    return std::nullopt;
  }
  if (outer.types[key_column] != aggregate.groups[0]->GetReturnType()) { return std::nullopt; }

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
  result.key_column   = key_column;
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
  std::vector<duckdb::LogicalMaterializedCTE*> ctes;  ///< CTEs whose consumer is being visited
  std::size_t rewrites = 0;

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
    auto plan             = match_aggregate(context, aggregate, outer, keys, scan_index);
    if (!plan) { return; }
    reuse_rows(aggregate, scan_rows(scan_index, *probe.cte_index, outer), *plan);
    ++rewrites;
  }

  void visit(unique_ptr<LogicalOperator>& slot)
  {
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
  rewriter pass{context, binder, {}, 0};
  pass.visit(plan);
  if (pass.rewrites > 0) {
    SIRIUS_LOG_INFO(
      "[sirius_plan_delim_aggregate_reuse] Computed {} subquery aggregate(s) from the delim CTE's "
      "outer rows instead of rescanning the subquery's tables",
      pass.rewrites);
  }
  return pass.rewrites;
}

}  // namespace sirius::planner
