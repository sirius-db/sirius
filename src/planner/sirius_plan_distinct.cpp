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

// Lowering for `SELECT DISTINCT`. A duckdb::LogicalDistinct becomes one
// sirius::op::sirius_physical_grouped_aggregate whose groups are the distinct targets and whose
// aggregate list is empty: deduplication is grouping with nothing to compute per group.
// insert_gpu_pipeline_operators then applies the standard aggregate wrap, so DISTINCT inherits
// GROUP BY's local pre-aggregation, hash shuffle and multi-GPU fan-out.
//
// The body follows DuckDB's PhysicalPlanGenerator::CreatePlan(LogicalDistinct&), except where the
// reference binds a grouped FIRST aggregate for an output column that no distinct target covers.
// Sirius has no grouped FIRST (to_cudf_aggregation_kind(aggregate_id::first) is std::nullopt,
// src/op/aggregate/aggregate_op_util.cpp), so this builder throws and that shape runs on the CPU.

namespace sirius::planner {

namespace {

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner boundary. The
// source vector is drained; size and order are preserved. A plan holding a null node crashes at
// execution time, so a shape sirius::ast::from_duckdb declines throws here instead and the query
// falls back to DuckDB's CPU execution.
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

  // Binder::VisitQueryNode populates `order_by` only for DISTINCT ON, so this is an exact test for
  // `DISTINCT ON (k) ... ORDER BY t` and deliberately not a test on distinct_type:
  // `SELECT DISTINCT a, b ... ORDER BY a` arrives with order_by == nullptr and is supported, its
  // ORDER BY having become a separate LOGICAL_ORDER above this node. The ordered shape names a
  // specific row per group and a hash-partitioned dedup returns an arbitrary one, so relaxing this
  // returns a plausible wrong answer rather than an error.
  if (op.order_by) {
    throw duckdb::NotImplementedException(
      "DISTINCT ON with ORDER BY is not supported on the GPU (falling back to CPU): choosing the "
      "first row of each group under a global order is not decomposable across the hash shuffle");
  }

  // The binder never builds this node with an empty target list, and a zero-key grouped aggregate
  // is not a shape this operator models.
  if (op.distinct_targets.empty()) {
    throw duckdb::NotImplementedException(
      "DISTINCT with no distinct targets is not supported on the GPU (falling back to CPU)");
  }

  // The targets become cudf::groupby key columns, and Sirius reads and projects nested columns but
  // cannot operate on them. Runs before lowering so the message can still name the column.
  for (auto const& target : op.distinct_targets) {
    reject_nested_column_operation(*target, "DISTINCT");
  }

  auto plan = create_plan(*op.children[0]);
  // LogicalDistinct::ResolveTypes() copies its child's types, so op.types is also the physical
  // child's schema; reading it here rather than the child's keeps the loop in duckdb::LogicalType.
  // That equivalence does not always hold, and nothing downstream re-establishes it:
  // create_plan(LogicalAggregate&) rewrites HUGEINT to BIGINT in its own logical operator without
  // touching the parents that already resolved against HUGEINT, so `SELECT DISTINCT k, sum(x) FROM
  // t GROUP BY k` reaches here declaring HUGEINT over a BIGINT child. Compare the types and not
  // only their count: a child agreeing on width but not on type is the harder failure to trace.
  auto declared_types = sirius::from_duckdb_vec(op.types);
  if (plan->types.size() != declared_types.size()) {
    throw duckdb::NotImplementedException(
      "DISTINCT: the planned child produces " + std::to_string(plan->types.size()) +
      (plan->types.size() == 1 ? " column" : " columns") + " but the DISTINCT node declares " +
      std::to_string(declared_types.size()) + " (falling back to CPU)");
  }
  for (duckdb::idx_t i = 0; i < declared_types.size(); i++) {
    if (plan->types[i] == declared_types[i]) { continue; }
    throw duckdb::NotImplementedException("DISTINCT: the planned child produces " +
                                          plan->types[i].to_string() + " for column " +
                                          std::to_string(i) + " but the DISTINCT node declares " +
                                          declared_types[i].to_string() + " (falling back to CPU)");
  }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> projections;
  // Named after the reference implementation, which appends the FIRST aggregates' return types.
  // With zero aggregates it holds exactly the group key types.
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

  // Carried over from the reference implementation, and never what reaches the projection here:
  // fewer targets than output columns means some output column has no bare-reference target, and
  // the loop below throws on it. What does reach the projection is the reorder that loop finds, as
  // in `SELECT DISTINCT ON (b, a) a, b`.
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

    // No distinct target is a bare reference to output column i, which is where the reference
    // implementation binds FIRST(child.i). Two unrelated causes land here, either of which can
    // occur under either distinct_type, so branch on the cause and let distinct_type choose only
    // the wording.
    //
    // Cause 1: a target is not a bare reference at all, so it maps to no output column. The route
    // ordinary SQL has into this is a VARCHAR key under a non-binary default_collation, which
    // Binder::BindModifiers wraps in a collation call. `groups[i]` is in range because columns
    // 0..i-1 were all found in the map; it is the offender only when the two lists are 1:1, so it
    // is reported as context rather than named as the cause.
    if (!all_targets_are_references) {
      throw duckdb::NotImplementedException(
        "DISTINCT: no distinct target is a plain reference to output column " + std::to_string(i) +
        " (falling back to CPU); target " + std::to_string(i) + " is '" + groups[i]->ToString() +
        "'");
    }

    // Cause 2: every target is a bare reference, but none reads output column i, so that column has
    // to be carried out of each group. Under DISTINCT ON the select list holds a column the ON list
    // does not; under plain DISTINCT the select list grew after the binder synthesized its targets
    // from it, as in `SELECT DISTINCT a FROM t ORDER BY b`.
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

  // An empty aggregate list is a shape the operator already ships: the delim-join builder builds
  // it for duplicate elimination, where cudf's hash groupby returns the unique keys.
  //
  // The operator requires every group to be a bare reference to a column the child has. Check both
  // here because neither consumer can still fall back cleanly: a non-reference group throws from
  // convert_duckdb_aggregates_to_cudf in the vocabulary of aggregates rather than of DISTINCT, and
  // an out-of-range column index is read unguarded by the aggregate kernel at execution time.
  auto const child_column_count = plan->types.size();
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

  // Restore the output order op.types declares. The select list is a permutation rather than an
  // identity, so push_projection creates a real PROJECTION instead of eliding it.
  auto const estimated_cardinality = group_by->estimated_cardinality;
  return push_projection(std::move(group_by),
                         std::move(declared_types),
                         translate_expressions(std::move(projections)),
                         estimated_cardinality);
}

}  // namespace sirius::planner
