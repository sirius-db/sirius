/*
 * Copyright 2025, Sirius Contributors.
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

#include "planner/sirius_physical_plan_generator.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"

#include <unordered_set>

namespace sirius::planner {

sirius_physical_plan_generator::sirius_physical_plan_generator(duckdb::ClientContext& context)
  : context(context)
{
}

sirius_physical_plan_generator::~sirius_physical_plan_generator() {}

duckdb::OrderPreservationType sirius_physical_plan_generator::order_preservation_recursive(
  sirius::op::sirius_physical_operator& op)
{
  if (op.is_source()) { return op.source_order(); }

  duckdb::idx_t child_idx = 0;
  for (auto& child : op.children) {
    // Do not take the materialization phase of physical CTEs into account
    if (op.type == sirius::op::SiriusPhysicalOperatorType::CTE && child_idx == 0) {
      child_idx++;
      continue;
    }
    auto child_preservation = order_preservation_recursive(*child);
    if (child_preservation != duckdb::OrderPreservationType::INSERTION_ORDER) {
      return child_preservation;
    }
    child_idx++;
  }
  return duckdb::OrderPreservationType::INSERTION_ORDER;
}

bool sirius_physical_plan_generator::preserve_insertion_order(
  duckdb::ClientContext& context, sirius::op::sirius_physical_operator& plan)
{
  auto preservation_type = order_preservation_recursive(plan);
  if (preservation_type == duckdb::OrderPreservationType::FIXED_ORDER) {
    // always need to maintain preservation order
    return true;
  }
  if (preservation_type == duckdb::OrderPreservationType::NO_ORDER) {
    // never need to preserve order
    return false;
  }
  // preserve insertion order - check flags
  if (!duckdb::DBConfig::GetSetting<duckdb::PreserveInsertionOrderSetting>(context)) {
    // preserving insertion order is disabled by config
    return false;
  }
  return true;
}

bool sirius_physical_plan_generator::preserve_insertion_order(
  sirius::op::sirius_physical_operator& plan)
{
  return preserve_insertion_order(context, plan);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::unique_ptr<duckdb::LogicalOperator> op)
{
  auto& profiler = duckdb::QueryProfiler::Get(context);

  // Resolve the types of each operator.
  profiler.StartPhase(duckdb::MetricsType::PHYSICAL_PLANNER_RESOLVE_TYPES);
  op->ResolveOperatorTypes();
  profiler.EndPhase();

  // Resolve the column references.
  profiler.StartPhase(duckdb::MetricsType::PHYSICAL_PLANNER_COLUMN_BINDING);
  duckdb::ColumnBindingResolver resolver;
  resolver.VisitOperator(*op);
  profiler.EndPhase();

  // then create the main physical plan
  profiler.StartPhase(duckdb::MetricsType::PHYSICAL_PLANNER_CREATE_PLAN);
  auto plan = create_plan(*op);
  profiler.EndPhase();

  // Push filter operators through joins when the filter predicates reference
  // only one side of the join.  This reduces the data volume flowing into joins
  // and all downstream operators.
  plan = push_filters_through_joins(std::move(plan));

  plan->verify();
  return plan;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalOperator& op)
{
  SIRIUS_LOG_DEBUG("Creating sirius physical plan for logical operator type: {}",
                   duckdb::LogicalOperatorToString(op.type));
  op.estimated_cardinality                                      = op.EstimateCardinality(context);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = nullptr;

  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      plan = create_plan(op.Cast<duckdb::LogicalGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      plan = create_plan(op.Cast<duckdb::LogicalProjection>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT:
      plan = create_plan(op.Cast<duckdb::LogicalEmptyResult>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      plan = create_plan(op.Cast<duckdb::LogicalFilter>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      plan = create_plan(op.Cast<duckdb::LogicalAggregate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_WINDOW:
      throw duckdb::NotImplementedException("Window not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalWindow>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNNEST:
      throw duckdb::NotImplementedException("Unnest not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalUnnest>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
      plan = create_plan(op.Cast<duckdb::LogicalLimit>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_SAMPLE:
      throw duckdb::NotImplementedException("Sample not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSample>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
      plan = create_plan(op.Cast<duckdb::LogicalOrder>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
      plan = create_plan(op.Cast<duckdb::LogicalTopN>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE:
      throw duckdb::NotImplementedException("Copy to file not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCopyToFile>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN:
      plan = create_plan(op.Cast<duckdb::LogicalDummyScan>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
      throw duckdb::NotImplementedException("Any join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalAnyJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ASOF_JOIN:
      throw duckdb::NotImplementedException("Asof join not supported");
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      plan = create_plan(op.Cast<duckdb::LogicalComparisonJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
      throw duckdb::NotImplementedException("Cross product not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCrossProduct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
      throw duckdb::NotImplementedException("Positional join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPositionalJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      throw duckdb::NotImplementedException("Set operation not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSetOperation>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_INSERT:
      throw duckdb::NotImplementedException("Insert not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalInsert>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELETE:
      throw duckdb::NotImplementedException("Delete not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalDelete>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CHUNK_GET:
      plan = create_plan(op.Cast<duckdb::LogicalColumnDataGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
      plan = create_plan(op.Cast<duckdb::LogicalDelimGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPRESSION_GET:
      plan = create_plan(op.Cast<duckdb::LogicalExpressionGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UPDATE:
      throw duckdb::NotImplementedException("Update not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalUpdate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_TABLE:
      throw duckdb::NotImplementedException("Create table not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateTable>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_INDEX:
      throw duckdb::NotImplementedException("Create index not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateIndex>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SECRET:
      throw duckdb::NotImplementedException("Create secret not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateSecret>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPLAIN:
      throw duckdb::NotImplementedException("Explain not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExplain>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
      throw duckdb::NotImplementedException("Distinct not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalDistinct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PREPARE:
      throw duckdb::NotImplementedException("Prepare not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPrepare>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXECUTE:
      throw duckdb::NotImplementedException("Execute not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExecute>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_VIEW:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_MACRO:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_TYPE:
      throw duckdb::NotImplementedException("Create not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PRAGMA:
      throw duckdb::NotImplementedException("Pragma not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPragma>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_VACUUM:
      throw duckdb::NotImplementedException("Vacuum not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalVacuum>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TRANSACTION:
    case duckdb::LogicalOperatorType::LOGICAL_ALTER:
    case duckdb::LogicalOperatorType::LOGICAL_DROP:
    case duckdb::LogicalOperatorType::LOGICAL_LOAD:
    case duckdb::LogicalOperatorType::LOGICAL_ATTACH:
    case duckdb::LogicalOperatorType::LOGICAL_DETACH:
      throw duckdb::NotImplementedException("Simple not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSimple>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
      throw duckdb::NotImplementedException("Recursive CTE not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalRecursiveCTE>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
      plan = create_plan(op.Cast<duckdb::LogicalMaterializedCTE>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF:
      plan = create_plan(op.Cast<duckdb::LogicalCTERef>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPORT:
      throw duckdb::NotImplementedException("Export not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExport>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_SET:
      throw duckdb::NotImplementedException("Set not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_RESET:
      throw duckdb::NotImplementedException("Reset not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalReset>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PIVOT:
      throw duckdb::NotImplementedException("Pivot not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPivot>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COPY_DATABASE:
      throw duckdb::NotImplementedException("Copy database not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCopyDatabase>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
      throw duckdb::NotImplementedException("Update extensions not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSimple>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
      throw duckdb::NotImplementedException("Extension operator not supported");
      // plan = op.Cast<duckdb::LogicalExtensionOperator>().create_plan(context, *this);

      // if (!plan) {
      // 	throw duckdb::InternalException("Missing sirius_physical_operator for Extension
      // Operator");
      // }
      break;
    case duckdb::LogicalOperatorType::LOGICAL_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_INVALID: {
      throw duckdb::NotImplementedException("Unimplemented logical operator type!");
    }
    default: throw duckdb::NotImplementedException("Unimplemented logical operator type");
  }
  if (!plan) { throw duckdb::InternalException("Physical plan generator - no plan generated"); }

  plan->estimated_cardinality = op.estimated_cardinality;
#ifdef DUCKDB_VERIFY_VECTOR_OPERATOR
  auto verify = duckdb::make_uniq<duckdb::PhysicalVerifyVector>(std::move(plan));
  plan        = std::move(verify);
#endif

  return plan;
}

namespace {

/// Collect all BoundReferenceExpression column indices from an expression tree.
void collect_column_refs(duckdb::Expression& expr, std::unordered_set<duckdb::idx_t>& refs)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    refs.insert(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expr, [&](duckdb::Expression& child) { collect_column_refs(child, refs); });
}

enum class FilterSide { LEFT, RIGHT, BOTH };

/// Determine whether all column references in the filter expression map to
/// the left side, right side, or both sides of the hash join.
FilterSide determine_filter_side(duckdb::Expression& expr,
                                 const sirius::op::sirius_physical_hash_join& join)
{
  std::unordered_set<duckdb::idx_t> refs;
  collect_column_refs(expr, refs);
  if (refs.empty()) { return FilterSide::BOTH; }

  auto lhs_count = static_cast<duckdb::idx_t>(join.lhs_output_columns.col_idxs.size());
  bool has_left  = false;
  bool has_right = false;
  for (auto ref : refs) {
    if (ref < lhs_count) {
      has_left = true;
    } else {
      has_right = true;
    }
  }

  if (has_left && !has_right) { return FilterSide::LEFT; }
  if (!has_left && has_right) { return FilterSide::RIGHT; }
  return FilterSide::BOTH;
}

/// Remap BoundReferenceExpression indices so they reference the child table
/// of the given join side instead of the join output.
void remap_refs_to_child(duckdb::Expression& expr,
                         const sirius::op::sirius_physical_hash_join& join,
                         FilterSide side)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    auto& ref      = expr.Cast<duckdb::BoundReferenceExpression>();
    auto lhs_count = static_cast<duckdb::idx_t>(join.lhs_output_columns.col_idxs.size());
    if (side == FilterSide::LEFT) {
      ref.index = static_cast<duckdb::idx_t>(join.lhs_output_columns.col_idxs[ref.index]);
    } else {
      ref.index =
        static_cast<duckdb::idx_t>(join.rhs_output_columns.col_idxs[ref.index - lhs_count]);
    }
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expr, [&](duckdb::Expression& child) { remap_refs_to_child(child, join, side); });
}

}  // anonymous namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::push_filters_through_joins(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> op)
{
  if (!op) { return op; }

  // Recursively process children first (bottom-up)
  for (auto& child : op->children) {
    child = push_filters_through_joins(std::move(child));
  }

  // Only interested in FILTER whose immediate child is a HASH_JOIN
  if (op->type != sirius::op::SiriusPhysicalOperatorType::FILTER || op->children.empty() ||
      op->children[0]->type != sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) {
    return op;
  }

  auto& filter = op->Cast<sirius::op::sirius_physical_filter>();
  auto& join   = op->children[0]->Cast<sirius::op::sirius_physical_hash_join>();

  // MARK joins produce a virtual boolean column that doesn't map to either
  // child.  OUTER joins introduce NULLs that change filter semantics.  Only
  // push through join types where output columns map directly to children.
  if (join.join_type != duckdb::JoinType::INNER && join.join_type != duckdb::JoinType::SEMI &&
      join.join_type != duckdb::JoinType::ANTI) {
    return op;
  }

  auto side = determine_filter_side(*filter.expression, join);
  if (side == FilterSide::BOTH) { return op; }

  int side_idx = (side == FilterSide::LEFT) ? 0 : 1;

  SIRIUS_LOG_DEBUG("Pushing filter through {} join (id {}) to {} child",
                   (join.join_type == duckdb::JoinType::ANTI    ? "ANTI"
                    : join.join_type == duckdb::JoinType::INNER ? "INNER"
                                                                : "OTHER"),
                   join.operator_id,
                   (side == FilterSide::LEFT ? "left" : "right"));

  // Remap the filter expression's column indices to reference the target
  // child's output instead of the join's output.
  remap_refs_to_child(*filter.expression, join, side);

  // Update the filter's output types to match the target child
  op->types = op->children[0]->children[side_idx]->types;

  // Restructure: FILTER(JOIN(L, R)) → JOIN(FILTER(L), R)  or  JOIN(L, FILTER(R))
  auto join_ptr                = std::move(op->children[0]);
  op->children[0]              = std::move(join_ptr->children[side_idx]);
  join_ptr->children[side_idx] = std::move(op);

  // Recursively try to push the filter further through nested joins
  join_ptr->children[side_idx] =
    push_filters_through_joins(std::move(join_ptr->children[side_idx]));

  return join_ptr;
}

}  // namespace sirius::planner
