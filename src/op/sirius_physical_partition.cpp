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

#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

namespace {

std::optional<duckdb::idx_t> extract_bound_ref_index(const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    return expr.Cast<duckdb::BoundReferenceExpression>().index;
  }
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast_expr = expr.Cast<duckdb::BoundCastExpression>();
    if (cast_expr.child->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
      return cast_expr.child->Cast<duckdb::BoundReferenceExpression>().index;
    }
  }
  return std::nullopt;
}

}  // namespace

sirius_physical_partition::sirius_physical_partition(duckdb::vector<duckdb::LogicalType> types,
                                                     duckdb::idx_t estimated_cardinality,
                                                     sirius_physical_operator* parent_op,
                                                     bool is_build,
                                                     uint64_t hash_partition_bytes)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARTITION, std::move(types), estimated_cardinality)
{
  s_partition_size = hash_partition_bytes;
  _parent_op       = parent_op;
  _is_build        = is_build;
  get_partition_keys_and_type(parent_op, is_build);
}

std::string sirius_physical_partition::get_name() const { return "PARTITION"; }

bool sirius_physical_partition::is_source() const { return true; }

bool sirius_physical_partition::is_sink() const { return true; }

void sirius_physical_partition::get_partition_keys_and_type(sirius_physical_operator* op,
                                                            bool is_build)
{
  if (op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    _partition_type    = PartitionType::HASH;
    auto& hash_join_op = op->Cast<sirius_physical_hash_join>();
    for (duckdb::idx_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
      auto& condition = hash_join_op.conditions[cond_idx];
      if (condition.comparison != duckdb::ExpressionType::COMPARE_EQUAL &&
          condition.comparison != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        continue;
      }
      std::optional<duckdb::idx_t> left_index =
        extract_bound_ref_index(*hash_join_op.conditions[cond_idx].left);
      std::optional<duckdb::idx_t> right_index =
        extract_bound_ref_index(*hash_join_op.conditions[cond_idx].right);
      if (left_index.has_value() && right_index.has_value()) {
        // Determine if a type cast is needed for hash alignment.
        // When the join condition has a BOUND_CAST on one side, the two sides have different
        // physical column types (e.g. INT32 vs INT64). cuDF's murmur3 produces different hash
        // values for the same integer in different representations, so without a cast, matching
        // keys would land in different partitions. We apply the same cast used by the join
        // condition so both sides hash identically.
        const auto& key_expr = is_build ? *hash_join_op.conditions[cond_idx].right
                                        : *hash_join_op.conditions[cond_idx].left;
        if (is_build) {
          _partition_keys.push_back(right_index.value());
        } else {
          _partition_keys.push_back(left_index.value());
        }
        if (key_expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
          _partition_key_cast_types.push_back(duckdb::GetCudfType(key_expr.return_type));
        } else {
          _partition_key_cast_types.push_back(cudf::data_type{cudf::type_id::EMPTY});
        }
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
    _partition_type = PartitionType::NONE;
    _num_partitions = 1;
  } else if (op->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    _partition_type            = PartitionType::HASH;
    auto& grouped_aggregate_op = op->Cast<sirius_physical_grouped_aggregate>();
    _partition_keys            = grouped_aggregate_op.get_output_grouping_indices();

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
    _partition_keys                  = grouped_aggregate_merge_op.get_output_grouping_indices();

  } else if (op->type == SiriusPhysicalOperatorType::CONCAT) {
    auto& parent_concat_op = op->Cast<sirius_physical_concat>();
    bool is_build          = parent_concat_op.is_build_concat();
    _is_build              = is_build;
    if (parent_concat_op.get_parent_op()->type == SiriusPhysicalOperatorType::HASH_JOIN ||
        parent_concat_op.get_parent_op()->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
      get_partition_keys_and_type(parent_concat_op.get_parent_op(), is_build);
    } else {
      throw std::runtime_error("Unsupported operator following partition->concat: " +
                               parent_concat_op.get_parent_op()->get_name());
    }
  } else {
    throw std::runtime_error("Unsupported operator type for partition: " + op->get_name());
  }
}

bool sirius_physical_partition::is_build_partition() { return _is_build; }

std::unique_ptr<operator_data> sirius_physical_partition::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_partition::execute"};
  const auto& input_batches = input_data.get_data_batches();
  if (input_batches.size() != 1) {
    throw std::runtime_error("We expect only one input batch for partition operator " +
                             std::to_string(this->get_operator_id()));
  }
  if (!_num_partitions.has_value()) {
    throw std::runtime_error("Num partitions was not set in sirius_physical_partition operator " +
                             std::to_string(this->get_operator_id()));
  }
  if (_num_partitions.value() < 2 || _partition_keys.empty()) {
    return std::make_unique<operator_data>(input_data);
  }

  auto input_batch = input_batches[0];
  std::vector<std::shared_ptr<cucascade::data_batch>> partitioned_results;
  switch (_partition_type) {
    case PartitionType::HASH:
      partitioned_results = gpu_partition_impl::hash_partition(input_batch,
                                                               _partition_keys,
                                                               _partition_key_cast_types,
                                                               _num_partitions.value(),
                                                               stream,
                                                               *input_batch->get_memory_space());
      break;
    case PartitionType::RANGE:
      throw std::runtime_error("Range partitioning is not implemented yet");
    case PartitionType::EVENLY:
      partitioned_results = gpu_partition_impl::evenly_partition(
        input_batch, _num_partitions.value(), stream, *input_batch->get_memory_space());
      break;
    case PartitionType::NONE: partitioned_results = {input_batch}; break;
    case PartitionType::CUSTOM:
      throw std::runtime_error("Custom partitioning is not implemented yet");
    default:
      throw std::runtime_error("Unsupported partition type: " +
                               partition_type_to_string(_partition_type));
  }
  return std::make_unique<operator_data>(partitioned_results);
}

void sirius_physical_partition::sink(const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_partition::sink"};
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

int sirius_physical_partition::determine_num_partitions()
{
  SIRIUS_LOG_DEBUG("sirius_physical_partition::determine_num_partitions() start for id {}",
                   this->get_operator_id());
  if (ports.find("default") == ports.end()) {
    SIRIUS_LOG_WARN(
      "sirius_physical_partition::determine_num_partitions() did not find default repo for id {}",
      this->get_operator_id());
    throw std::runtime_error(
      "sirius_physical_partition::determine_num_partitions() did not find default repo for id " +
      std::to_string(this->get_operator_id()));
  }
  auto& repo           = ports.at("default")->repo;
  auto batch_ids       = repo->get_batch_ids(0);
  uint64_t total_bytes = 0;
  for (auto batch_id : batch_ids) {
    auto batch = repo->get_data_batch_by_id(batch_id, std::nullopt, 0);
    if (batch && batch->get_data()) { total_bytes += batch->get_data()->get_size_in_bytes(); }
  }
  return static_cast<int>(std::max(uint64_t{1}, total_bytes / s_partition_size));
}

void sirius_physical_partition::set_num_partitions(int num_partitions)
{
  std::unique_lock<std::mutex> guard(lock, std::try_to_lock);
  if (!guard.owns_lock()) {
    throw std::runtime_error(
      "set_num_partitions failed to acquire lock for partition operator " +
      std::to_string(get_operator_id()) +
      " — likely a cross-partition deadlock: both sibling partitions are simultaneously "
      "in get_next_task_input_data");
  }
  _num_partitions = num_partitions;
}

std::unique_ptr<operator_data> sirius_physical_partition::get_next_task_input_data()
{
  {
    std::lock_guard<std::mutex> guard(lock);
    if (!_num_partitions.has_value()) {
      _num_partitions = determine_num_partitions();
      if (_sibling_partition_op) {
        SIRIUS_LOG_DEBUG(
          "sirius_physical_partition id {} determined {} partitions on sibling id {} and {} build "
          "side",
          this->get_operator_id(),
          _num_partitions.value(),
          _sibling_partition_op->get_operator_id(),
          (_is_build ? "is" : "is not"));
        _sibling_partition_op->Cast<sirius_physical_partition>().set_num_partitions(
          _num_partitions.value());
      } else {
        SIRIUS_LOG_DEBUG("sirius_physical_partition id {} determined {} partitions",
                         this->get_operator_id(),
                         _num_partitions.value());
      }
    }
  }
  return sirius_physical_operator::get_next_task_input_data();
}

}  // namespace op
}  // namespace sirius
