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

#include "operator/gpu_physical_partition.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_order.hpp"
#include "operator/gpu_physical_top_n.hpp"

namespace duckdb {

GPUPhysicalPartition::GPUPhysicalPartition(vector<LogicalType> types,
                                           idx_t estimated_cardinality,
                                           GPUPhysicalOperator* parent_op,
                                           bool is_build)
  : GPUPhysicalOperator(PhysicalOperatorType::INVALID, std::move(types), estimated_cardinality)
{
  _num_partitions = (estimated_cardinality + PARTITION_SIZE - 1) / PARTITION_SIZE;
  _parent_op      = parent_op;
  _is_build       = is_build;
  GetPartitionKeys(parent_op, is_build);
}

string GPUPhysicalPartition::GetName() const { return "PARTITION"; }

bool GPUPhysicalPartition::IsSource() const { return true; }

bool GPUPhysicalPartition::IsSink() const { return true; }

void GPUPhysicalPartition::GetPartitionKeys(GPUPhysicalOperator* op, bool is_build)
{
  _partition_keys.clear();
  if (op->type == PhysicalOperatorType::HASH_JOIN) {
    auto& hash_join_op = op->Cast<GPUPhysicalHashJoin>();
    if (is_build) {
      for (idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
        auto& condition = hash_join_op.conditions[cond_idx];
        if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(condition.right->Cast<BoundReferenceExpression>().index);
        }
      }
    } else {
      for (idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
        auto& condition = hash_join_op.conditions[cond_idx];
        if (condition.left->GetExpressionClass() == ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(condition.left->Cast<BoundReferenceExpression>().index);
        }
      }
    }
  } else if (op->type == PhysicalOperatorType::HASH_GROUP_BY) {
    auto& grouped_aggregate_op = op->Cast<GPUPhysicalGroupedAggregate>();
    for (idx_t i = 0; i < grouped_aggregate_op.groupings.size(); i++) {
      auto& grouping = grouped_aggregate_op.groupings[i];
      for (auto& group_idx : grouped_aggregate_op.grouping_sets[i]) {
        auto& group = grouped_aggregate_op.grouped_aggregate_data.groups[group_idx];
        if (group->GetExpressionClass() == ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(group->Cast<BoundReferenceExpression>().index);
        }
      }
    }
  } else if (op->type == PhysicalOperatorType::ORDER_BY) {
    auto& order_by_op = op->Cast<GPUPhysicalOrder>();
    for (size_t order_idx = 0; order_idx < order_by_op.orders.size(); order_idx++) {
      auto& expr = order_by_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<BoundReferenceExpression>().index);
      }
    }
  } else if (op->type == PhysicalOperatorType::TOP_N) {
    auto& top_n_op = op->Cast<GPUPhysicalTopN>();
    for (size_t order_idx = 0; order_idx < top_n_op.orders.size(); order_idx++) {
      auto& expr = top_n_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<BoundReferenceExpression>().index);
      }
    }
  }
}

bool GPUPhysicalPartition::isBuildPartition() { return _is_build; }

}  // namespace duckdb
