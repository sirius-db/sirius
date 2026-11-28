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

#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Lowering for `SELECT DISTINCT`: a LogicalDistinct becomes one grouped aggregate whose groups are
// the distinct targets. An output column no target covers is carried by a FIRST over that column,
// as DuckDB's own builder does, and the grouped kernels run an all-FIRST list by keeping one row
// per key, the first under the node's ORDER BY when one survives.

namespace sirius::planner {

namespace {

//! Whether @p op is a projection that passes its child's columns through unchanged, in order.
bool is_identity_projection(duckdb::LogicalOperator const& op)
{
  if (op.type != duckdb::LogicalOperatorType::LOGICAL_PROJECTION) { return false; }
  auto const& child_types = op.children[0]->types;
  if (op.expressions.size() != child_types.size()) { return false; }
  for (duckdb::idx_t i = 0; i < op.expressions.size(); ++i) {
    auto const& expression = *op.expressions[i];
    if (expression.GetExpressionType() != duckdb::ExpressionType::BOUND_REF ||
        expression.Cast<duckdb::BoundReferenceExpression>().Index() != i ||
        expression.GetReturnType() != child_types[i]) {
      return false;
    }
  }
  return true;
}

//! Whether every output column of @p op is one of its distinct targets, as a plain reference.
bool targets_every_column(duckdb::LogicalDistinct const& op)
{
  std::vector<bool> targeted(op.types.size(), false);
  for (auto const& target : op.distinct_targets) {
    if (target->GetExpressionType() != duckdb::ExpressionType::BOUND_REF) { return false; }
    auto const index = target->Cast<duckdb::BoundReferenceExpression>().Index();
    if (index >= targeted.size()) { return false; }
    targeted[index] = true;
  }
  return std::ranges::all_of(targeted, std::identity{});
}

//! The child column @p expression reads, or nothing unless it is a bare reference below @p width.
std::optional<duckdb::idx_t> child_column(duckdb::Expression const& expression, duckdb::idx_t width)
{
  if (expression.GetExpressionType() != duckdb::ExpressionType::BOUND_REF) { return std::nullopt; }
  auto const index = expression.Cast<duckdb::BoundReferenceExpression>().Index();
  if (index >= width) { return std::nullopt; }
  return index;
}

}  // namespace

bool sirius_physical_plan_generator::input_rows_are_distinct(duckdb::LogicalOperator const& op)
{
  auto const* input = &op;
  while (is_identity_projection(*input)) {
    input = input->children[0].get();
  }
  switch (input->type) {
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      return set_operation_deduplicates(input->Cast<duckdb::LogicalSetOperation>());
    default: return false;
  }
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalDistinct& op)
{
  D_ASSERT(op.children.size() == 1);

  // A zero-key grouped aggregate is not a shape the operator models.
  if (op.distinct_targets.empty()) {
    throw duckdb::NotImplementedException(
      "DISTINCT with no distinct targets is not supported on the GPU (falling back to CPU)");
  }

  // Targets become group key columns, which cannot be nested. Runs before lowering so the
  // message can name the column.
  for (auto const& target : op.distinct_targets) {
    reject_nested_column_operation(*target, "DISTINCT");
  }

  // Until #2598 is fixed: the GPU lower() and upper() map some characters differently from
  // DuckDB's, so they can rank rows differently. Checked before planning the child, which takes
  // the expressions out of its projections.
  if (op.order_by) {
    auto const surviving = op.order_by->Copy();
    if (!surviving->Simplify(op.distinct_targets, nullptr)) {
      for (auto const& key : surviving->orders) {
        if (reads_case_mapping(*key.expression, *op.children[0])) {
          throw duckdb::NotImplementedException(
            "DISTINCT ON with a NOCASE-collated, lower() or upper() ORDER BY key is not supported "
            "on the GPU (falling back to CPU): '" +
            key.expression->ToString() + "'");
        }
      }
    }
  }

  auto plan = create_plan(*op.children[0]);
  // The dedup's output types are the child's, so the child must plan the declared schema.
  auto const& planned_types = plan->get_types();
  require_planned_schema(planned_types, op.types, "DISTINCT");

  // The binder puts a DISTINCT over every distinct EXCEPT / INTERSECT, whose plan already ends in a
  // full-row deduplication; grouping its rows again would remove nothing.
  if (targets_every_column(op) && input_rows_are_distinct(*op.children[0])) { return plan; }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> projections;
  // Child columns no target covers, each carried by a FIRST, in aggregate order.
  duckdb::vector<duckdb::idx_t> carried_columns;
  // Child column index -> the group position that reads it, for bare BOUND_REF targets only.
  std::unordered_map<duckdb::idx_t, duckdb::idx_t> group_by_references;

  auto const group_count = op.distinct_targets.size();
  std::optional<duckdb::idx_t> first_non_reference_target;
  for (duckdb::idx_t i = 0; i < group_count; i++) {
    auto& target = op.distinct_targets[i];
    if (target->GetExpressionType() == duckdb::ExpressionType::BOUND_REF) {
      auto& bound_ref                        = target->Cast<duckdb::BoundReferenceExpression>();
      group_by_references[bound_ref.Index()] = i;
    } else if (!first_non_reference_target) {
      first_non_reference_target = i;
    }
    groups.push_back(std::move(target));
  }

  // `order_by` is populated for DISTINCT ON only. A term that is a distinct target is constant
  // within each group, so Simplify drops it, as the reference builder's OrderedAggregateOptimizer
  // does; only a term that survives can choose a group's row.
  auto order = op.order_by ? op.order_by->Copy() : nullptr;
  if (order && order->Simplify(groups, nullptr)) { order.reset(); }

  // A carried column lands after the keys in the operator's output, so a node wider than its
  // targets needs the projection.
  bool requires_projection = op.types.size() != group_count;

  for (duckdb::idx_t i = 0; i < op.types.size(); i++) {
    auto const entry = group_by_references.find(i);
    if (entry != group_by_references.end()) {
      auto const group_index = entry->second;
      projections.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::reference{static_cast<std::uint32_t>(group_index), planned_types[i]}));
      if (group_index != i) { requires_projection = true; }
      continue;
    }

    // Output column i has no bare-reference target, for one of two unrelated causes.
    //
    // Cause 1: some target is not a bare reference, so it maps to no output column.
    if (first_non_reference_target) {
      auto const target = *first_non_reference_target;
      throw duckdb::NotImplementedException(
        "DISTINCT: no distinct target is a plain reference to output column " + std::to_string(i) +
        " (falling back to CPU); target " + std::to_string(target) + " is '" +
        groups[target]->ToString() + "'");
    }

    // Cause 2: every target is a bare reference but none reads column i, so it has to be carried
    // out of each group by a FIRST, from the group's first row under the surviving ORDER BY or
    // from any row without one. The reference builder also attaches op.order_by and runs
    // OrderedAggregateOptimizer here, which leaves the FIRST unordered unless a term survives.

    // This FIRST's position in the operator's output, so it must be taken before the push below.
    projections.push_back(std::make_unique<sirius::ast::node>(sirius::ast::reference{
      static_cast<std::uint32_t>(group_count + carried_columns.size()), planned_types[i]}));
    carried_columns.push_back(i);
    requires_projection = true;
  }

  // The operator requires every group to be a bare reference to a column the child has. Checked
  // here because the kernels read the index unguarded at execution time.
  auto const child_column_count = planned_types.size();
  duckdb::vector<duckdb::idx_t> key_columns;
  key_columns.reserve(groups.size());
  for (duckdb::idx_t i = 0; i < groups.size(); i++) {
    auto const column = child_column(*groups[i], child_column_count);
    if (!column) {
      throw duckdb::NotImplementedException(
        "DISTINCT: group key " + std::to_string(i) + " '" + groups[i]->ToString() +
        "' is not a plain reference to a child column (falling back to CPU)");
    }
    key_columns.push_back(*column);
  }

  // The kernels sort on child columns, so every surviving ORDER BY key must be one.
  std::vector<dedup_order_key> order_keys;
  if (order) {
    for (auto const& key : order->orders) {
      reject_nested_column_operation(*key.expression, "a DISTINCT ON ORDER BY");
      auto const column = child_column(*key.expression, child_column_count);
      if (!column) {
        throw duckdb::NotImplementedException(
          "DISTINCT ON: ORDER BY key '" + key.expression->ToString() +
          "' is not a plain reference to a child column (falling back to CPU)");
      }
      auto const modifier = sirius::ast::order_modifier_from_duckdb(key);
      if (!modifier) {
        throw duckdb::NotImplementedException("DISTINCT ON: ORDER BY key '" +
                                              key.expression->ToString() +
                                              "' has no resolved direction or NULL placement "
                                              "(falling back to CPU)");
      }
      order_keys.push_back({.column = *column, .modifier = *modifier});
    }
  }
  auto group_by =
    plan_dedup(std::move(plan), key_columns, carried_columns, order_keys, op.estimated_cardinality);

  if (!requires_projection) { return group_by; }

  // Restore the output order op.types declares. push_projection drops the select list when it is
  // the identity, as for `SELECT DISTINCT ON (k) k, v`.
  auto const estimated_cardinality = group_by->estimated_cardinality;
  return push_projection(
    std::move(group_by), planned_types, std::move(projections), estimated_cardinality);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_physical_plan_generator::plan_dedup(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  std::span<duckdb::idx_t const> key_columns,
  std::span<duckdb::idx_t const> carried_columns,
  std::span<dedup_order_key const> order,
  duckdb::idx_t estimated_cardinality)
{
  // The operator's output is its group key columns, then one column per FIRST.
  auto const& child_types = child->get_types();
  duckdb::vector<sirius::logical_type> output_types;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> group_nodes;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
  output_types.reserve(key_columns.size() + carried_columns.size());
  group_nodes.reserve(key_columns.size());
  aggregates.reserve(carried_columns.size());
  for (auto const index : key_columns) {
    output_types.push_back(child_types[index]);
    group_nodes.push_back(std::make_unique<sirius::ast::node>(
      sirius::ast::reference{static_cast<std::uint32_t>(index), child_types[index]}));
  }
  auto const reference_to = [&](duckdb::idx_t index) {
    return std::make_unique<sirius::ast::node>(
      sirius::ast::reference{static_cast<std::uint32_t>(index), child_types[index]});
  };
  std::vector<sirius::ast::order_modifier> modifiers;
  modifiers.reserve(order.size());
  for (auto const& key : order) {
    modifiers.push_back(key.modifier);
  }
  for (auto const index : carried_columns) {
    // A FIRST's ORDER BY keys follow its value as reference arguments.
    std::vector<std::unique_ptr<sirius::ast::node>> first_arguments;
    first_arguments.push_back(reference_to(index));
    for (auto const& key : order) {
      first_arguments.push_back(reference_to(key.column));
    }
    output_types.push_back(child_types[index]);
    aggregates.push_back(
      std::make_unique<sirius::ast::node>(sirius::ast::aggregate{sirius::aggregate_id::first,
                                                                 std::move(first_arguments),
                                                                 child_types[index],
                                                                 /*distinct=*/false,
                                                                 modifiers}));
  }

  auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                         sirius::op::sirius_physical_grouped_aggregate>(
    std::move(output_types), std::move(aggregates), std::move(group_nodes), estimated_cardinality);
  group_by->children.push_back(std::move(child));
  return group_by;
}

}  // namespace sirius::planner
