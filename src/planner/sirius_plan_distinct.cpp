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
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

// Lowering for `SELECT DISTINCT`: a LogicalDistinct becomes one grouped aggregate whose groups
// are the distinct targets and whose aggregate list is empty. An output column no target covers
// would need a grouped FIRST, which Sirius cannot execute, so those shapes throw and run on CPU.

namespace sirius::planner {

namespace {

// Drains `exprs`, preserving size and order. A null node crashes at execution time, so a
// declined translation throws here.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    auto translated = e ? sirius::ast::from_duckdb(*e) : nullptr;
    if (e && translated == nullptr) {
      throw duckdb::NotImplementedException(
        "Unsupported expression in DISTINCT (falling back to CPU): " + e->ToString());
    }
    out.push_back(std::move(translated));
  }
  return out;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalDistinct& op)
{
  D_ASSERT(op.children.size() == 1);

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
  //
  // A CTE node declares its materialization side, not the rows it forwards; the schema that
  // reaches this operator is its body's, which is its second child.
  auto const& child_types = plan->type == sirius::op::SiriusPhysicalOperatorType::CTE
                              ? plan->children[1]->types
                              : plan->types;
  auto declared_types     = sirius::from_duckdb_vec(op.types);
  if (child_types.size() != declared_types.size()) {
    throw duckdb::NotImplementedException(
      "DISTINCT: the planned child produces " + std::to_string(child_types.size()) +
      (child_types.size() == 1 ? " column" : " columns") + " but the DISTINCT node declares " +
      std::to_string(declared_types.size()) + " (falling back to CPU)");
  }
  for (duckdb::idx_t i = 0; i < declared_types.size(); i++) {
    if (child_types[i] == declared_types[i]) { continue; }
    throw duckdb::NotImplementedException("DISTINCT: the planned child produces " +
                                          child_types[i].to_string() + " for column " +
                                          std::to_string(i) + " but the DISTINCT node declares " +
                                          declared_types[i].to_string() + " (falling back to CPU)");
  }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> projections;
  // With zero aggregates this holds exactly the group key types.
  duckdb::vector<duckdb::LogicalType> aggregate_types;
  // Child column index -> the group position that reads it, for bare BOUND_REF targets only.
  std::unordered_map<duckdb::idx_t, duckdb::idx_t> group_by_references;

  auto const group_count          = op.distinct_targets.size();
  bool all_targets_are_references = true;
  for (duckdb::idx_t i = 0; i < group_count; i++) {
    auto& target = op.distinct_targets[i];
    if (target->GetExpressionType() == duckdb::ExpressionType::BOUND_REF) {
      auto& bound_ref                      = target->Cast<duckdb::BoundReferenceExpression>();
      group_by_references[bound_ref.index] = i;
    } else {
      all_targets_are_references = false;
    }
    aggregate_types.push_back(target->return_type);
    groups.push_back(std::move(target));
  }

  // Fewer targets than output columns always throws below, so what actually reaches the
  // projection is the reorder the loop finds, as in `SELECT DISTINCT ON (b, a) a, b`.
  bool requires_projection = op.types.size() != group_count;

  for (duckdb::idx_t i = 0; i < op.types.size(); i++) {
    auto const& logical_type = op.types[i];
    auto const entry         = group_by_references.find(i);
    if (entry != group_by_references.end()) {
      auto const group_index = entry->second;
      projections.push_back(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(logical_type, group_index));
      if (group_index != i) { requires_projection = true; }
      continue;
    }

    // Output column i has no bare-reference target. Two unrelated causes reach here under either
    // distinct_type, so branch on the cause and let distinct_type choose only the wording.
    //
    // Cause 1: some target is not a bare reference, so it maps to no output column. `groups[i]` is
    // in range because columns 0..i-1 were all found, and is context rather than the named cause.
    if (!all_targets_are_references) {
      throw duckdb::NotImplementedException(
        "DISTINCT: no distinct target is a plain reference to output column " + std::to_string(i) +
        " (falling back to CPU); target " + std::to_string(i) + " is '" + groups[i]->ToString() +
        "'");
    }

    // Cause 2: every target is a bare reference but none reads column i, so it has to be carried
    // out of each group.
    if (op.distinct_type == duckdb::DistinctType::DISTINCT_ON) {
      throw duckdb::NotImplementedException(
        "DISTINCT ON with carried (non-key) columns is not supported on the GPU (falling back to "
        "CPU): output column " +
        std::to_string(i) + " would need a grouped FIRST aggregate");
    }
    throw duckdb::NotImplementedException(
      "DISTINCT: output column " + std::to_string(i) +
      " has no distinct target (falling back to CPU): the node has " +
      std::to_string(groups.size()) + (groups.size() == 1 ? " target for " : " targets for ") +
      std::to_string(op.types.size()) +
      (op.types.size() == 1 ? " output column" : " output columns") +
      ", and an output column without one would need a grouped FIRST aggregate");
  }

  // The operator requires every group to be a bare reference to a column the child has. Checked
  // here because neither consumer can still fall back cleanly: one throws in the vocabulary of
  // aggregates, and the other reads the index unguarded at execution time.
  auto const child_column_count = child_types.size();
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

  auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                         sirius::op::sirius_physical_grouped_aggregate>(
    sirius::from_duckdb_vec(aggregate_types),
    duckdb::vector<std::unique_ptr<sirius::ast::node>>{},
    translate_expressions(std::move(groups)),
    op.estimated_cardinality);
  group_by->children.push_back(std::move(plan));

  if (!requires_projection) { return group_by; }

  // Restore the output order op.types declares; the select list is a permutation, not an
  // identity, so this is never elided.
  auto const estimated_cardinality = group_by->estimated_cardinality;
  return push_projection(std::move(group_by),
                         std::move(declared_types),
                         translate_expressions(std::move(projections)),
                         estimated_cardinality);
}

}  // namespace sirius::planner
