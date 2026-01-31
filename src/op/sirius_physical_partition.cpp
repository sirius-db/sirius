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

#include "op/sirius_physical_partition.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n.hpp"

namespace sirius {
namespace op {

sirius_physical_partition::sirius_physical_partition(duckdb::vector<duckdb::LogicalType> types,
                                                     duckdb::idx_t estimated_cardinality,
                                                     sirius_physical_operator* parent_op,
                                                     bool is_build)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARTITION, std::move(types), estimated_cardinality)
{
  _num_partitions = (estimated_cardinality + PARTITION_SIZE - 1) / PARTITION_SIZE;
  _parent_op      = parent_op;
  _is_build       = is_build;
  get_partition_keys(parent_op, is_build);
}

std::string sirius_physical_partition::get_name() const { return "PARTITION"; }

bool sirius_physical_partition::is_source() const { return true; }

bool sirius_physical_partition::is_sink() const { return true; }

void sirius_physical_partition::get_partition_keys(sirius_physical_operator* op, bool is_build)
{
  _partition_keys.clear();
  if (op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    auto& hash_join_op = op->Cast<sirius_physical_hash_join>();
    if (is_build) {
      for (duckdb::idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
        auto& condition = hash_join_op.conditions[cond_idx];
        if (condition.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(
            condition.right->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }
    } else {
      for (duckdb::idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
        auto& condition = hash_join_op.conditions[cond_idx];
        if (condition.left->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(condition.left->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    auto& grouped_aggregate_op = op->Cast<sirius_physical_grouped_aggregate>();
    for (duckdb::idx_t i = 0; i < grouped_aggregate_op.groupings.size(); i++) {
      auto& grouping = grouped_aggregate_op.groupings[i];
      for (auto& group_idx : grouped_aggregate_op.grouping_sets[i]) {
        auto& group = grouped_aggregate_op.grouped_aggregate_data.groups[group_idx];
        if (group->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          _partition_keys.push_back(group->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::ORDER_BY) {
    auto& order_by_op = op->Cast<sirius_physical_order>();
    for (size_t order_idx = 0; order_idx < order_by_op.orders.size(); order_idx++) {
      auto& expr = order_by_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<duckdb::BoundReferenceExpression>().index);
      }
    }
  }
}

bool sirius_physical_partition::is_build_partition() { return _is_build; }

}  // namespace op
}  // namespace sirius
