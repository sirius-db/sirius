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

#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <memory>

namespace sirius::planner {

namespace {

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner
// boundary. The source vector is drained; size and order are preserved, with a
// null slot wherever from_duckdb declines an unsupported shape (a fallback
// signal) — matching the prior bulk-translation null-skip semantics.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(e ? sirius::ast::from_duckdb(*e) : nullptr);
  }
  return out;
}

// File-local helper (formerly sirius_physical_plan_generator::extract_aggregate_expressions).
// Pulls aggregate child / filter sub-expressions out of the aggregate list and groups into a
// projection fed upstream of the aggregate. Operates on raw DuckDB expressions so the hoist
// is straightforward; the caller translates the resulting groups/aggregates into Sirius AST
// nodes when constructing the aggregate operator.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> extract_aggregate_expressions(
  duckdb::ClientContext& context,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
  duckdb::optional_ptr<duckdb::vector<duckdb::GroupingSet>> grouping_sets)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  duckdb::vector<duckdb::LogicalType> types;

  // bind sorted aggregates
  for (auto& aggr : aggregates) {
    auto& bound_aggr = aggr->Cast<duckdb::BoundAggregateExpression>();
    if (bound_aggr.order_bys) {
      duckdb::FunctionBinder::BindSortedAggregate(context, bound_aggr, groups, grouping_sets);
    }
  }
  for (auto& group : groups) {
    auto ref =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(group->return_type, expressions.size());
    types.push_back(group->return_type);
    expressions.push_back(std::move(group));
    group = std::move(ref);
  }
  for (auto& aggr : aggregates) {
    auto& bound_aggr = aggr->Cast<duckdb::BoundAggregateExpression>();
    for (auto& child_expr : bound_aggr.children) {
      auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(child_expr->return_type,
                                                                     expressions.size());
      types.push_back(child_expr->return_type);
      expressions.push_back(std::move(child_expr));
      child_expr = std::move(ref);
    }
    if (bound_aggr.filter) {
      auto& filter = bound_aggr.filter;
      auto ref     = duckdb::make_uniq<duckdb::BoundReferenceExpression>(filter->return_type,
                                                                     expressions.size());
      types.push_back(filter->return_type);
      expressions.push_back(std::move(filter));
      bound_aggr.filter = std::move(ref);
    }
  }
  if (expressions.empty()) { return child; }
  auto const estimated_cardinality = child->estimated_cardinality;
  return push_projection(std::move(child),
                         sirius::from_duckdb_vec(types),
                         translate_expressions(std::move(expressions)),
                         estimated_cardinality);
}

}  // namespace

static uint32_t required_bits_for_value(uint32_t n)
{
  std::size_t required_bits = 0;
  while (n > 0) {
    n >>= 1;
    required_bits++;
  }
  return duckdb::UnsafeNumericCast<uint32_t>(required_bits);
}

template <class T>
duckdb::hugeint_t get_range_hugeint(const duckdb::BaseStatistics& nstats)
{
  return duckdb::Hugeint::Convert(duckdb::NumericStats::GetMax<T>(nstats)) -
         duckdb::Hugeint::Convert(duckdb::NumericStats::GetMin<T>(nstats));
}

static bool can_use_partitioned_aggregate(duckdb::ClientContext& context,
                                          duckdb::LogicalAggregate& op,
                                          sirius::op::sirius_physical_operator& child,
                                          duckdb::vector<duckdb::column_t>& partition_columns)
{
  if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) { return false; }
  // check if the source is partitioned by the aggregate columns
  // figure out the columns we are grouping by
  for (auto& group_expr : op.groups) {
    // only support bound reference here
    if (group_expr->GetExpressionType() != duckdb::ExpressionType::BOUND_REF) { return false; }
    auto& ref = group_expr->Cast<duckdb::BoundReferenceExpression>();
    partition_columns.push_back(ref.index);
  }
  // traverse the children of the aggregate to find the source operator
  duckdb::reference<sirius::op::sirius_physical_operator> child_ref(child);
  while (child_ref.get().type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& child_op = child_ref.get();
    switch (child_op.type) {
      case sirius::op::SiriusPhysicalOperatorType::PROJECTION: {
        // recompute partition columns
        auto& projection = child_op.Cast<sirius::op::sirius_physical_projection>();
        duckdb::vector<duckdb::column_t> new_columns;
        for (auto& partition_col : partition_columns) {
          // we only support bound reference here
          auto const* expr = projection.select_list[partition_col].get();
          if (!expr->holds<sirius::ast::reference>()) { return false; }
          new_columns.push_back(expr->get<sirius::ast::reference>().column_index);
        }
        // continue into child node with new columns
        partition_columns = std::move(new_columns);
        child_ref         = *child_op.children[0];
        break;
      }
      case sirius::op::SiriusPhysicalOperatorType::FILTER:
        // continue into child operators
        child_ref = *child_op.children[0];
        break;
      default:
        // unsupported operator for partition pass-through
        return false;
    }
  }
  auto& table_scan = child_ref.get().Cast<sirius::op::sirius_physical_table_scan>();
  if (!table_scan.function.get_partition_info) {
    // this source does not expose partition information - skip
    return false;
  }
  // get the base columns by projecting over the projection_ids/column_ids
  if (!table_scan.projection_ids.empty()) {
    for (auto& partition_col : partition_columns) {
      if (partition_col >= table_scan.projection_ids.size()) { return false; }
      partition_col = table_scan.projection_ids[partition_col];
    }
  }
  duckdb::vector<duckdb::column_t> base_columns;
  for (const auto& partition_idx : partition_columns) {
    auto col_idx = partition_idx;
    col_idx      = table_scan.column_ids[col_idx].GetPrimaryIndex();
    base_columns.push_back(col_idx);
  }
  // check if the source operator is partitioned by the grouping columns
  duckdb::TableFunctionPartitionInput input(table_scan.bind_data.get(), base_columns);
  auto partition_info = table_scan.function.get_partition_info(context, input);
  if (partition_info != duckdb::TablePartitionInfo::SINGLE_VALUE_PARTITIONS) {
    // we only support single-value partitions currently
    return false;
  }
  // we have single value partitions!
  return true;
}

static bool can_use_perfect_hash_aggregate(duckdb::ClientContext& context,
                                           duckdb::LogicalAggregate& op,
                                           duckdb::vector<std::size_t>& bits_per_group)
{
  if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) { return false; }
  std::size_t perfect_hash_bits = 0;
  for (std::size_t group_idx = 0; group_idx < op.groups.size(); group_idx++) {
    auto& group = op.groups[group_idx];
    auto& stats = op.group_stats[group_idx];

    switch (group->return_type.InternalType()) {
      case duckdb::PhysicalType::INT8:
      case duckdb::PhysicalType::INT16:
      case duckdb::PhysicalType::INT32:
      case duckdb::PhysicalType::INT64:
      case duckdb::PhysicalType::UINT8:
      case duckdb::PhysicalType::UINT16:
      case duckdb::PhysicalType::UINT32:
      case duckdb::PhysicalType::UINT64: break;
      default:
        // we only support simple integer types for perfect hashing
        return false;
    }
    // check if the group has stats available
    auto& group_type = group->return_type;
    if (!stats) {
      // no stats, but we might still be able to use perfect hashing if the type is small enough
      // for small types we can just set the stats to [type_min, type_max]
      switch (group_type.InternalType()) {
        case duckdb::PhysicalType::INT8:
        case duckdb::PhysicalType::INT16:
        case duckdb::PhysicalType::UINT8:
        case duckdb::PhysicalType::UINT16: break;
        default:
          // type is too large and there are no stats: skip perfect hashing
          return false;
      }
      // construct stats with the min and max value of the type
      stats = duckdb::NumericStats::CreateUnknown(group_type).ToUnique();
      duckdb::NumericStats::SetMin(*stats, duckdb::Value::MinimumValue(group_type));
      duckdb::NumericStats::SetMax(*stats, duckdb::Value::MaximumValue(group_type));
    }
    auto& nstats = *stats;

    if (!duckdb::NumericStats::HasMinMax(nstats)) { return false; }

    if (duckdb::NumericStats::Max(*stats) < duckdb::NumericStats::Min(*stats)) {
      // May result in underflow
      return false;
    }

    // we have a min and a max value for the stats: use that to figure out how many bits we have
    // we add two here, one for the NULL value, and one to make the computation one-indexed
    // (e.g. if min and max are the same, we still need one entry in total)
    duckdb::hugeint_t range_h;
    switch (group_type.InternalType()) {
      case duckdb::PhysicalType::INT8: range_h = get_range_hugeint<int8_t>(nstats); break;
      case duckdb::PhysicalType::INT16: range_h = get_range_hugeint<int16_t>(nstats); break;
      case duckdb::PhysicalType::INT32: range_h = get_range_hugeint<int32_t>(nstats); break;
      case duckdb::PhysicalType::INT64: range_h = get_range_hugeint<int64_t>(nstats); break;
      case duckdb::PhysicalType::UINT8: range_h = get_range_hugeint<uint8_t>(nstats); break;
      case duckdb::PhysicalType::UINT16: range_h = get_range_hugeint<uint16_t>(nstats); break;
      case duckdb::PhysicalType::UINT32: range_h = get_range_hugeint<uint32_t>(nstats); break;
      case duckdb::PhysicalType::UINT64: range_h = get_range_hugeint<uint64_t>(nstats); break;
      default:
        throw duckdb::InternalException(
          "Unsupported type for perfect hash (should be caught before)");
    }

    uint64_t range;
    if (!duckdb::Hugeint::TryCast(range_h, range)) { return false; }

    // bail out on any range bigger than 2^32
    if (range >= duckdb::NumericLimits<int32_t>::Maximum()) { return false; }

    range += 2;
    // figure out how many bits we need
    std::size_t required_bits = required_bits_for_value(duckdb::UnsafeNumericCast<uint32_t>(range));
    bits_per_group.push_back(required_bits);
    perfect_hash_bits += required_bits;
    // check if we have exceeded the bits for the hash
    if (perfect_hash_bits > duckdb::Settings::Get<duckdb::PerfectHtThresholdSetting>(context)) {
      // too many bits for perfect hash
      return false;
    }
  }
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<duckdb::BoundAggregateExpression>();
    if (aggregate.IsDistinct() || !aggregate.function.combine) {
      // distinct aggregates are not supported in perfect hash aggregates
      return false;
    }
  }
  return true;
}

/// cuDF does not support HUGEINT (int128). DuckDB widens aggregates like sum(int32) to HUGEINT
/// to avoid overflow. We downcast to BIGINT at the plan level so all downstream operators
/// (including the result collector) use the correct type.
static void downcast_hugeint_types(duckdb::vector<duckdb::LogicalType>& types,
                                   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs)
{
  for (auto& type : types) {
    if (type == duckdb::LogicalType::HUGEINT) { type = duckdb::LogicalType::BIGINT; }
  }
  for (auto& expr : exprs) {
    if (expr->return_type == duckdb::LogicalType::HUGEINT) {
      expr->return_type = duckdb::LogicalType::BIGINT;
    }
  }
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalAggregate& op)
{
  D_ASSERT(op.children.size() == 1);

  // Downcast HUGEINT to BIGINT since cuDF does not support int128
  downcast_hugeint_types(op.types, op.expressions);

  auto plan = create_plan(*op.children[0]);

  plan = extract_aggregate_expressions(
    context, std::move(plan), op.expressions, op.groups, op.grouping_sets);
  bool can_use_simple_aggregation = true;
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<duckdb::BoundAggregateExpression>();
    if (!aggregate.function.simple_update) {
      // unsupported aggregate for simple aggregation: use hash aggregation
      can_use_simple_aggregation = false;
      break;
    }
  }

  // Check if all groups are valid
  if (op.group_stats.empty()) { op.group_stats.resize(op.groups.size()); }
  auto group_validity = duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES;
  for (const auto& stats : op.group_stats) {
    if (stats && !stats->CanHaveNull()) { continue; }
    group_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
    break;
  }

  if (op.groups.empty() && op.grouping_sets.size() <= 1) {
    // no groups, check if we can use a simple aggregation
    // special case: aggregate entire columns together
    if (can_use_simple_aggregation) {
      auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                             sirius::op::sirius_physical_ungrouped_aggregate>(
        sirius::from_duckdb_vec(op.types),
        translate_expressions(std::move(op.expressions)),
        op.estimated_cardinality,
        op.distinct_validity);
      group_by->children.push_back(std::move(plan));
      return group_by;
    }
    throw duckdb::NotImplementedException("Non simple aggregation is not supported");
  }

  // groups! create a GROUP BY aggregator
  // use a partitioned or perfect hash aggregate if possible
  duckdb::vector<duckdb::column_t> partition_columns;
  duckdb::vector<std::size_t> required_bits;
  if (can_use_simple_aggregation &&
      can_use_partitioned_aggregate(context, op, *plan, partition_columns)) {
    auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                           sirius::op::sirius_physical_grouped_aggregate>(
      sirius::from_duckdb_vec(op.types),
      translate_expressions(std::move(op.expressions)),
      translate_expressions(std::move(op.groups)),
      std::move(op.grouping_sets),
      std::move(op.grouping_functions),
      op.estimated_cardinality,
      group_validity,
      op.distinct_validity);
    group_by->children.push_back(std::move(plan));
    return group_by;
  }

  if (can_use_perfect_hash_aggregate(context, op, required_bits)) {
    auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                           sirius::op::sirius_physical_grouped_aggregate>(
      sirius::from_duckdb_vec(op.types),
      translate_expressions(std::move(op.expressions)),
      translate_expressions(std::move(op.groups)),
      std::move(op.grouping_sets),
      std::move(op.grouping_functions),
      op.estimated_cardinality,
      group_validity,
      op.distinct_validity);
    group_by->children.push_back(std::move(plan));
    return group_by;
  }

  auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                         sirius::op::sirius_physical_grouped_aggregate>(
    sirius::from_duckdb_vec(op.types),
    translate_expressions(std::move(op.expressions)),
    translate_expressions(std::move(op.groups)),
    std::move(op.grouping_sets),
    std::move(op.grouping_functions),
    op.estimated_cardinality,
    group_validity,
    op.distinct_validity);
  group_by->children.push_back(std::move(plan));
  return group_by;
}

}  // namespace sirius::planner
