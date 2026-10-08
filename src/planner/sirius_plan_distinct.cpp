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
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

// Lowering for `SELECT DISTINCT`: a LogicalDistinct becomes one grouped aggregate whose groups are
// the distinct targets. An output column no target covers is carried by a FIRST over that column,
// as DuckDB's own builder does, and the grouped kernels run an all-FIRST list by keeping one row
// per key.

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalDistinct& op)
{
  D_ASSERT(op.children.size() == 1);

  // The binder lowers a distinct UNION to this node over a UNION ALL; distinct UNION is not on
  // the GPU path yet, so keep it on CPU.
  if (op.children[0]->type == duckdb::LogicalOperatorType::LOGICAL_UNION) {
    throw duckdb::NotImplementedException(
      "DISTINCT over a UNION is not supported on the GPU (falling back to CPU): distinct UNION is "
      "not on the GPU path yet");
  }

  // `order_by` is populated for DISTINCT ON only, so this tests the ordered shape and not
  // distinct_type. Relaxing it returns a plausible wrong row, not an error.
  if (op.order_by) {
    throw duckdb::NotImplementedException(
      "DISTINCT ON with ORDER BY is not supported on the GPU (falling back to CPU): choosing the "
      "first row of each group under a global order is not decomposable across the hash shuffle");
  }

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

  auto plan = create_plan(*op.children[0]);
  // op.types normally equals the planned child's schema, but a child built by another arm can
  // narrow a type without its parents re-resolving, so compare element types and not only width.
  auto const& planned_types = plan->get_output_types();
  auto const declared_types = sirius::from_duckdb_vec(op.types);
  if (planned_types.size() != declared_types.size()) {
    throw duckdb::NotImplementedException(
      "DISTINCT: the planned child produces " + std::to_string(planned_types.size()) +
      (planned_types.size() == 1 ? " column" : " columns") + " but the DISTINCT node declares " +
      std::to_string(declared_types.size()) + " (falling back to CPU)");
  }
  for (duckdb::idx_t i = 0; i < declared_types.size(); i++) {
    if (planned_types[i] == declared_types[i]) { continue; }
    // An aggregate below may plan a result narrower than its parents declare.
    if (planned_types[i] == sirius::from_duckdb(planned_aggregate_type(op.types[i]))) { continue; }
    throw duckdb::NotImplementedException("DISTINCT: the planned child produces " +
                                          planned_types[i].to_string() + " for column " +
                                          std::to_string(i) + " but the DISTINCT node declares " +
                                          declared_types[i].to_string() + " (falling back to CPU)");
  }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> projections;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
  // One FIRST return type per carried column, in aggregate order.
  duckdb::vector<sirius::logical_type> aggregate_types;
  // Child column index -> the group position that reads it, for bare BOUND_REF targets only.
  std::unordered_map<duckdb::idx_t, duckdb::idx_t> group_by_references;

  auto const group_count = op.distinct_targets.size();
  std::optional<duckdb::idx_t> first_non_reference_target;
  for (duckdb::idx_t i = 0; i < group_count; i++) {
    auto& target = op.distinct_targets[i];
    if (target->GetExpressionType() == duckdb::ExpressionType::BOUND_REF) {
      auto& bound_ref                      = target->Cast<duckdb::BoundReferenceExpression>();
      group_by_references[bound_ref.index] = i;
    } else if (!first_non_reference_target) {
      first_non_reference_target = i;
    }
    groups.push_back(std::move(target));
  }

  // A carried column lands after the keys in the operator's output, so a node wider than its
  // targets needs the projection unless that layout is already the output order.
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
    // out of each group by a FIRST. The reference builder also attaches op.order_by and runs
    // OrderedAggregateOptimizer here; the order_by guard above makes both no-ops.
    //
    // The FIRST reads the column as the child plans it, which may be narrower than declared.
    auto const carried_type =
      planned_types[i] == declared_types[i] ? op.types[i] : planned_aggregate_type(op.types[i]);
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> first_children;
    first_children.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(carried_type, i));
    auto first_aggregate = duckdb::FunctionBinder(context).BindAggregateFunction(
      duckdb::FirstFunctionGetter::GetFunction(carried_type),
      std::move(first_children),
      nullptr,
      duckdb::AggregateType::NON_DISTINCT);
    auto first_node = sirius::ast::from_duckdb(*first_aggregate);
    if (first_node == nullptr) {
      throw duckdb::NotImplementedException(
        "Unsupported expression in DISTINCT (falling back to CPU): " + first_aggregate->ToString());
    }
    // This FIRST's position in the operator's output, so it must be taken before the push below.
    projections.push_back(std::make_unique<sirius::ast::node>(sirius::ast::reference{
      static_cast<std::uint32_t>(group_count + aggregates.size()), planned_types[i]}));
    aggregate_types.push_back(planned_types[i]);
    aggregates.push_back(std::move(first_node));
    requires_projection = true;
  }

  // The operator requires every group to be a bare reference to a column the child has. Checked
  // here because neither consumer can still fall back cleanly: one throws in the vocabulary of
  // aggregates, and the other reads the index unguarded at execution time.
  auto const child_column_count = planned_types.size();
  for (duckdb::idx_t i = 0; i < groups.size(); i++) {
    auto const& group = *groups[i];
    if (group.GetExpressionType() != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "DISTINCT: group key " + std::to_string(i) +
        " is not a plain column reference (falling back to CPU): '" + group.ToString() + "'");
    }
    auto const& bound_ref = group.Cast<duckdb::BoundReferenceExpression>();
    if (bound_ref.index >= child_column_count) {
      throw duckdb::NotImplementedException(
        "DISTINCT: group key " + std::to_string(i) + " reads column " +
        std::to_string(bound_ref.index) + " of a child that has " +
        std::to_string(child_column_count) + (child_column_count == 1 ? " column" : " columns") +
        " (falling back to CPU)");
    }
  }

  // The operator's output is its group key columns, then one column per FIRST.
  duckdb::vector<sirius::logical_type> output_types;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> group_nodes;
  output_types.reserve(groups.size() + aggregate_types.size());
  group_nodes.reserve(groups.size());
  for (auto const& group : groups) {
    auto const index = group->Cast<duckdb::BoundReferenceExpression>().index;
    output_types.push_back(planned_types[index]);
    group_nodes.push_back(std::make_unique<sirius::ast::node>(
      sirius::ast::reference{static_cast<std::uint32_t>(index), planned_types[index]}));
  }
  output_types.insert(output_types.end(), aggregate_types.begin(), aggregate_types.end());

  auto group_by =
    duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                           sirius::op::sirius_physical_grouped_aggregate>(std::move(output_types),
                                                                          std::move(aggregates),
                                                                          std::move(group_nodes),
                                                                          op.estimated_cardinality);
  group_by->children.push_back(std::move(plan));

  if (!requires_projection) { return group_by; }

  // Restore the output order op.types declares. push_projection drops the select list when it is
  // the identity, as for `SELECT DISTINCT ON (k) k, v`.
  auto const estimated_cardinality = group_by->estimated_cardinality;
  return push_projection(
    std::move(group_by), planned_types, std::move(projections), estimated_cardinality);
}

}  // namespace sirius::planner
