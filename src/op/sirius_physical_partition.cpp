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

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "log/logging.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

#include <mutex>

namespace sirius {
namespace op {

namespace {

std::optional<std::size_t> extract_bound_ref_index(const duckdb::Expression& expr)
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

sirius_physical_partition::sirius_physical_partition(duckdb::vector<sirius::logical_type> types,
                                                     std::size_t estimated_cardinality,
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
    _hash_join_op      = op;  // set the hash join operator pointer for later use
    _partition_type    = PartitionType::HASH;
    auto& hash_join_op = op->Cast<sirius_physical_hash_join>();
    for (std::size_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
      auto& condition = hash_join_op.conditions[cond_idx];
      if (condition.comparison != sirius::comparison_type::equal &&
          condition.comparison != sirius::comparison_type::not_distinct_from) {
        continue;
      }
      auto left_owned  = sirius::ast::to_duckdb(*hash_join_op.conditions[cond_idx].left);
      auto right_owned = sirius::ast::to_duckdb(*hash_join_op.conditions[cond_idx].right);
      std::optional<std::size_t> left_index  = extract_bound_ref_index(*left_owned);
      std::optional<std::size_t> right_index = extract_bound_ref_index(*right_owned);
      if (left_index.has_value() && right_index.has_value()) {
        // Determine if a type cast is needed for hash alignment.
        // When the join condition has a BOUND_CAST on one side, the two sides have different
        // physical column types (e.g. INT32 vs INT64). cuDF's murmur3 produces different hash
        // values for the same integer in different representations, so without a cast, matching
        // keys would land in different partitions. We apply the same cast used by the join
        // condition so both sides hash identically.
        const auto& key_expr = is_build ? *right_owned : *left_owned;
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
    // operator which may be what we want to use when we care about grouping sets for (std::size_t
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
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() != 1) {
    throw std::runtime_error("We expect only one input batch for partition operator " +
                             std::to_string(this->get_operator_id()));
  }
  if (!_num_partitions.has_value()) {
    throw std::runtime_error("Num partitions was not set in sirius_physical_partition operator " +
                             std::to_string(this->get_operator_id()));
  }

  auto const& input_batch_ro = input_batches[0];
  auto* space                = input_batch_ro.get_memory_space();

  if (_num_partitions.value() < 2 || _partition_keys.empty()) {
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> partitioned_results;
  switch (_partition_type) {
    case PartitionType::HASH:
      partitioned_results = gpu_partition_impl::hash_partition(input_batch_ro,
                                                               _partition_keys,
                                                               _partition_key_cast_types,
                                                               _num_partitions.value(),
                                                               stream,
                                                               *space);
      break;
    case PartitionType::RANGE:
      throw std::runtime_error("Range partitioning is not implemented yet");
    case PartitionType::EVENLY:
      partitioned_results = gpu_partition_impl::evenly_partition(
        input_batch_ro, _num_partitions.value(), stream, *space);
      break;
    case PartitionType::NONE: {
      partitioned_results = {input_batch_ro.clone(sirius::get_next_batch_id(), stream)};
      break;
    }
    case PartitionType::CUSTOM:
      throw std::runtime_error("Custom partitioning is not implemented yet");
    default:
      throw std::runtime_error("Unsupported partition type: " +
                               partition_type_to_string(_partition_type));
  }
  return std::make_unique<pipelineable_operator_data>(partitioned_results);
}

void sirius_physical_partition::sink(const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_partition::sink"};
  auto& pipelineable_input  = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = pipelineable_input.get_data_batches();
  (void)stream;  // sink does not use stream for push_data_batch_partitioned
  int partition_id = 0;
  for (auto& batch : input_batches) {
    for (auto& next_port_info : next_port_after_sink) {
      // the next operator is a partition consumer operator, so we need to push the batch into the
      // specific partition
      auto partition_consumer_op =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (partition_consumer_op) {
        partition_consumer_op->push_data_batch_partitioned(
          next_port_info.next_operator_port_name, batch, partition_id);
      } else {
        throw std::runtime_error("Next operator is not a partition consumer operator");
      }
    }
    partition_id++;
  }
}

std::pair<int, uint64_t> sirius_physical_partition::determine_num_partitions()
{
  if (ports.find("default") == ports.end()) {
    throw std::runtime_error(
      "sirius_physical_partition::determine_num_partitions() did not find default repo for id " +
      std::to_string(this->get_operator_id()));
  }
  auto& repo           = ports.at("default")->repo;
  auto batch_ids       = repo->get_batch_ids(0);
  uint64_t total_bytes = 0;
  for (auto batch_id : batch_ids) {
    auto batch = repo->get_data_batch_by_id(batch_id, 0);
    if (batch) {
      auto ro = batch->to_read_only();
      if (ro.get_data()) { total_bytes += ro.get_data()->get_size_in_bytes(); }
    }
  }
  int num_partitions = static_cast<int>(std::max(uint64_t{1}, total_bytes / s_partition_size));
  // Multi-GPU floor: if the input is big enough to justify using the second
  // GPU, force at least num_gpus partitions so partition-based operators
  // (hash_join, merge_group_by) get work on every GPU. Below the small-table
  // threshold we keep num_partitions at its natural value — tiny tables run
  // on a single GPU to avoid cross-device overhead.
  if (_min_num_partitions > 1 && total_bytes >= _small_table_bytes) {
    num_partitions = std::max(num_partitions, _min_num_partitions);
  }
  return std::make_pair(num_partitions, total_bytes);
}

void sirius_physical_partition::set_num_partitions(int num_partitions)
{
  std::lock_guard<std::mutex> guard(lock);
  _num_partitions = num_partitions;
}

std::optional<task_creation_hint> sirius_physical_partition::get_next_task_hint(
  std::optional<std::size_t> downstream_request)
{
  std::lock_guard<std::mutex> guard(lock);
  if (!_num_partitions.has_value() && !_is_build && _sibling_partition_op != nullptr) {
    // If this is a probe partition and we haven't determined the number of partitions yet, we
    // should wait for the build sibling to determine it. This is because the build side will drive
    // the partitioning and the probe side needs to know the number of partitions to create the
    // correct number of tasks. Forward the downstream request — the sibling will route it
    // through its own relation rules.
    return _sibling_partition_op->get_next_task_hint(downstream_request);
  } else if (_num_partitions.has_value() && !_is_build && _sibling_partition_op != nullptr) {
    // If this is part of a join and its on the probe side, and we have determined the number of
    // partitions, we have this behave as a pipeline operator and just schedule tasks. Partition
    // is FAN_OUT (1 -> _num_partitions), so a downstream request does not promote upstream
    // demand; we keep the port-barrier default.

    if (ports.size() != 1) {  // Ensure that it only has one port
      throw std::runtime_error("sirius_physical_partition: there should be only one port");
    }
    auto port_ptr = ports.begin()->second;
    if (port_ptr->repo->total_size() > 0) {
      return task_creation_hint{TaskCreationHint::READY, this, task_creation_hint::ALL_TASKS};
    } else if (port_ptr->src_pipeline && !port_ptr->src_pipeline->is_pipeline_finished()) {
      // In pipelineable-probe mode partition can make progress one batch at a time; honor the
      // port's barrier type so PIPELINE ports ask for 1, PARTIAL for 4, etc.
      auto* producer = &(port_ptr->src_pipeline->get_operators()[0].get());
      return task_creation_hint{
        TaskCreationHint::WAITING_FOR_INPUT_DATA, producer, tasks_for_barrier(port_ptr->type)};
    } else {
      return std::nullopt;
    }
  } else {
    return sirius_physical_operator::get_next_task_hint(downstream_request);
  }
}

std::unique_ptr<operator_data> sirius_physical_partition::get_next_task_input_data()
{
  // Lock both this and the sibling partition atomically to prevent ABBA deadlock:
  // without this, two threads entering get_next_task_input_data on sibling partitions
  // simultaneously would each hold their own lock while trying to acquire the other's.
  if (_sibling_partition_op) {
    auto& sibling = _sibling_partition_op->Cast<sirius_physical_partition>();
    std::scoped_lock guard(lock, sibling.lock);
    if (!_num_partitions.has_value()) {
      auto [num_parts, total_bytes] = determine_num_partitions();
      auto& hash_join               = _hash_join_op->Cast<sirius_physical_hash_join>();
      // BUILD_PROBE mode requires the build side to deliver exactly one
      // batch at runtime. The only mechanism that guarantees that is the
      // build-side CONCAT with concat_all enabled. Detect whether either
      // partition has a build-side CONCAT downstream BEFORE asking the
      // join to consider BUILD_PROBE — without this gate, a small build
      // side with no downstream CONCAT would enter BUILD_PROBE and then
      // throw at runtime in get_next_task_input_data_for_build_probe when
      // size(0) != 1.
      auto has_build_concat = [](sirius_physical_operator& part_op) {
        for (auto& next_port : part_op.get_next_ports_after_sink()) {
          if (next_port.next_operator->type != SiriusPhysicalOperatorType::CONCAT) { continue; }
          auto& concat = next_port.next_operator->Cast<sirius_physical_concat>();
          if (concat.is_build_concat()) { return true; }
        }
        return false;
      };
      bool const build_foldable = has_build_concat(*this) || has_build_concat(sibling);
      hash_join.update_join_exec_mode(num_parts, total_bytes, build_foldable);
      if (_hash_join_op->type == SiriusPhysicalOperatorType::HASH_JOIN &&
          hash_join.is_build_probe_mode()) {
        // Either sibling may run this block first; configure the build-side CONCAT only.
        auto enable_build_concat_all = [](sirius_physical_operator& part_op) {
          for (auto& next_port : part_op.get_next_ports_after_sink()) {
            if (next_port.next_operator->type != SiriusPhysicalOperatorType::CONCAT) { continue; }
            auto& concat = next_port.next_operator->Cast<sirius_physical_concat>();
            if (concat.is_build_concat()) { concat.set_concat_all(true); }
          }
        };
        enable_build_concat_all(*this);
        enable_build_concat_all(sibling);
      }
      _num_partitions         = num_parts;
      sibling._num_partitions = num_parts;
      SIRIUS_LOG_DEBUG(
        "sirius_physical_partition id {} determined {} partitions from {} bytes on sibling id {} "
        "and {} build "
        "side",
        this->get_operator_id(),
        _num_partitions.value(),
        total_bytes,
        _sibling_partition_op->get_operator_id(),
        (_is_build ? "is" : "is not"));
    }
  } else {
    std::lock_guard<std::mutex> guard(lock);
    if (!_num_partitions.has_value()) {
      auto [num_parts, total_bytes] = determine_num_partitions();
      _num_partitions               = num_parts;
      SIRIUS_LOG_DEBUG("sirius_physical_partition id {} determined {} partitions from {} bytes",
                       this->get_operator_id(),
                       _num_partitions.value(),
                       total_bytes);
    }
  }
  return sirius_physical_operator::get_next_task_input_data();
}

std::size_t sirius_physical_partition::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  if (_num_partitions.has_value() && *_num_partitions == 1) { return 0; }
  return stats.bytes * 2;
}

}  // namespace op
}  // namespace sirius
