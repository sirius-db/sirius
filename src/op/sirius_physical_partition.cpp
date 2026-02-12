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

#include "creator/task_creator.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "pipeline/sirius_pipeline.hpp"

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
  get_partition_keys_and_type(parent_op, is_build);
}

std::string sirius_physical_partition::get_name() const { return "PARTITION"; }

bool sirius_physical_partition::is_source() const { return true; }

bool sirius_physical_partition::is_sink() const { return true; }

void sirius_physical_partition::get_partition_keys_and_type(sirius_physical_operator* op,
                                                            bool is_build)
{
  _partition_keys.clear();
  _partition_type = PartitionType::NONE;
  if (op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    _partition_type    = PartitionType::HASH;
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
    _partition_type            = PartitionType::HASH;
    auto& grouped_aggregate_op = op->Cast<sirius_physical_grouped_aggregate>();
    _partition_keys            = grouped_aggregate_op.group_idx;

    // WSM TODO: this is the original code for getting the partition keys from the grouped aggregate
    // operator which may be what we want to use when we care about grouping sets for (duckdb::idx_t
    // i = 0; i < grouped_aggregate_op.groupings.size(); i++) {
    //   auto& grouping = grouped_aggregate_op.groupings[i];
    //   for (auto& group_idx : grouped_aggregate_op.grouping_sets[i]) {
    //     auto& group = grouped_aggregate_op.grouped_aggregate_data.groups[group_idx];
    //     if (group->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    //       _partition_keys.push_back(group->Cast<duckdb::BoundReferenceExpression>().index);
    //     }
    //   }
    // }
  } else if (op->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY) {
    _partition_type                  = PartitionType::HASH;
    auto& grouped_aggregate_merge_op = op->Cast<sirius_physical_grouped_aggregate_merge>();
    _partition_keys                  = grouped_aggregate_merge_op.group_idx;

  } else if (op->type == SiriusPhysicalOperatorType::ORDER_BY) {
    _partition_type   = PartitionType::RANGE;
    auto& order_by_op = op->Cast<sirius_physical_order>();
    for (size_t order_idx = 0; order_idx < order_by_op.orders.size(); order_idx++) {
      auto& expr = order_by_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<duckdb::BoundReferenceExpression>().index);
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::TOP_N) {
    _partition_type = PartitionType::CUSTOM;
    auto& top_n_op  = op->Cast<sirius_physical_top_n>();
    for (size_t order_idx = 0; order_idx < top_n_op.orders.size(); order_idx++) {
      auto& expr = top_n_op.orders[order_idx].expression;
      if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        _partition_keys.push_back(expr->Cast<duckdb::BoundReferenceExpression>().index);
      }
    }
  }
}

bool sirius_physical_partition::is_build_partition() { return _is_build; }

operator_data sirius_physical_partition::execute(const operator_data& input_data,
                                                 rmm::cuda_stream_view stream)
{
  const auto& input_batches = input_data.get_data_batches();
  if (input_batches.size() != 1) {
    throw std::runtime_error("We expect only one input batch for partition operator");
  }

  if (_num_partitions < 2) { return input_data; }

  auto input_batch = input_batches[0];
  std::vector<std::shared_ptr<cucascade::data_batch>> partitioned_results;
  switch (_partition_type) {
    case PartitionType::HASH:
      partitioned_results = gpu_partition_impl::hash_partition(
        input_batch, _partition_keys, _num_partitions, stream, *input_batch->get_memory_space());
      break;
    case PartitionType::RANGE:
      throw std::runtime_error("Range partitioning is not implemented yet");
    case PartitionType::EVENLY:
      partitioned_results = gpu_partition_impl::evenly_partition(
        input_batch, _num_partitions, stream, *input_batch->get_memory_space());
      break;
    case PartitionType::CUSTOM:
      throw std::runtime_error("Custom partitioning is not implemented yet");
    default:
      throw std::runtime_error("Unsupported partition type: " +
                               partition_type_to_string(_partition_type));
  }
  return operator_data(partitioned_results);
}

void sirius_physical_partition::sink(const operator_data& input_data, rmm::cuda_stream_view stream)
{
  const auto& input_batches = input_data.get_data_batches();
  (void)stream;  // sink does not use stream for push_data_batch_partitioned
  int partition_id = 0;
  for (auto& batch : input_batches) {
    for (auto& [next_op, port_id] : next_port_after_sink) {
      // the next operator is a partition consumer operator, so we need to push the batch into the
      // specific partition
      auto partition_consumer_op =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_op);
      if (partition_consumer_op) {
        partition_consumer_op->push_data_batch_partitioned(port_id, batch, partition_id);
      } else {
        throw std::runtime_error("Next operator is not a partition consumer operator");
      }
    }
    partition_id++;
  }
}

}  // namespace op
}  // namespace sirius
