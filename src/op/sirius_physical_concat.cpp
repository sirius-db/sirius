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

#include "op/sirius_physical_concat.hpp"

#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_concat::sirius_physical_concat(duckdb::vector<sirius::logical_type> types,
                                               std::size_t estimated_cardinality,
                                               sirius_physical_operator* parent_op,
                                               bool is_build,
                                               uint64_t concat_batch_bytes)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::CONCAT, std::move(types), estimated_cardinality)
{
  _parent_op          = parent_op;
  _is_build           = is_build;
  _concat_batch_bytes = concat_batch_bytes;
  // check if parent_op is a hash join
  if (parent_op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    auto hash_join = dynamic_cast<sirius_physical_hash_join*>(parent_op);
    if (hash_join->join_type == duckdb::JoinType::LEFT ||
        hash_join->join_type == duckdb::JoinType::ANTI ||
        hash_join->join_type == duckdb::JoinType::SEMI) {
      // if the join type is left or anti, then we need to concat all the batches into one batch for
      // the build side
      _concat_all = is_build;
    } else if (hash_join->join_type == duckdb::JoinType::RIGHT ||
               hash_join->join_type == duckdb::JoinType::RIGHT_ANTI ||
               hash_join->join_type == duckdb::JoinType::RIGHT_SEMI) {
      // if the join type is right or right anti, then we need to concat all the batches into one
      // batch for the probe side
      _concat_all = !is_build;
    } else if (hash_join->join_type == duckdb::JoinType::INNER ||
               hash_join->join_type == duckdb::JoinType::MARK) {
      _concat_all = false;
    } else if (hash_join->join_type == duckdb::JoinType::OUTER) {
      _concat_all = true;
    } else {
      throw std::runtime_error("sirius_physical_concat: unsupported join type: " +
                               duckdb::JoinTypeToString(hash_join->join_type));
    }
  } else if (parent_op->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
    _concat_all = false;
  } else {
    throw std::runtime_error("sirius_physical_concat: parent_op is not a hash join: " +
                             SiriusPhysicalOperatorToString(parent_op->type));
  }
}

std::optional<task_creation_hint> sirius_physical_concat::get_next_task_hint(
  std::optional<std::size_t> /* downstream_request */)
{
  // Concat is FAN_IN: combine_upstream_request ignores downstream requests for this relation, so
  // we intentionally don't forward it. The local barrier semantics (PARTIAL vs concat-all)
  // already encode the right upstream demand.
  std::lock_guard<std::mutex> lg(lock);

  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto port_ptr          = ports.begin()->second;
  bool pipeline_finished = port_ptr->src_pipeline && port_ptr->src_pipeline->is_pipeline_finished();

  // If the source pipeline is done, we're ready to process whatever data remains
  if (pipeline_finished) {
    if (port_ptr->repo->total_size() > 0) {
      return task_creation_hint{TaskCreationHint::READY, this, task_creation_hint::ALL_TASKS};
    }
    return std::nullopt;
  } else if (_concat_all) {
    // if we need to concat all then we need to wait for the pipeline to be finished — concat-all
    // requires every upstream batch before we can do anything, so request ALL_TASKS.
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                              &(port_ptr->src_pipeline->get_operators()[0].get()),
                              task_creation_hint::ALL_TASKS};
  }

  // Source pipeline still running — check if there is enough data to fire a task early.
  // "Enough" means: for some partition, simulating get_next_task_input_data would pull a group
  // of batches AND there would still be at least one batch left in that partition afterward.
  for (size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    auto batch_ids          = port_ptr->repo->get_batch_ids(i);
    size_t total_batch_size = 0;
    size_t pulled_count     = 0;
    for (auto& batch_id : batch_ids) {
      auto batch_idle = port_ptr->repo->get_data_batch_by_id(batch_id, i);
      auto batch_ro   = batch_idle->to_read_only();
      auto batch_size = batch_ro.get_data()->get_size_in_bytes();
      total_batch_size += batch_size;
      if (!_concat_all && total_batch_size > _concat_batch_bytes) {
        // This batch pushes us over the threshold — the loop would stop here.
        // If we already accumulated batches (pulled_count > 0), the overflowing batch stays,
        // so there is at least one batch left after the pull.
        if (pulled_count > 0) {
          return task_creation_hint{TaskCreationHint::READY, this, task_creation_hint::ALL_TASKS};
        }
        // If nothing was accumulated yet, the single oversized batch itself would be pulled,
        // and remaining data is everything after it.
        if (batch_ids.size() > 1) {
          return task_creation_hint{TaskCreationHint::READY, this, task_creation_hint::ALL_TASKS};
        }
        break;
      } else {
        pulled_count++;
      }
    }
  }

  // Not enough data yet — wait for more from the source pipeline. Concat operates incrementally
  // (PARTIAL-barrier semantics): a small batch of upstream tasks is enough to make progress.
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                            &(port_ptr->src_pipeline->get_operators()[0].get()),
                            task_creation_hint::PARTIAL_BARRIER_DEFAULT_TASKS};
}

std::unique_ptr<operator_data> sirius_physical_concat::get_next_task_input_data()
{
  // iterate through all the partition and pull
  std::lock_guard<std::mutex> lg(lock);

  // assert that there is only one port
  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto port_ptr = ports.begin()->second;
  for (size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
    // get all the batch ids from the partition
    auto batch_ids          = port_ptr->repo->get_batch_ids(i);
    size_t total_batch_size = 0;
    for (auto& batch_id : batch_ids) {
      auto batch_idle = port_ptr->repo->get_data_batch_by_id(batch_id, i);
      auto batch_ro   = batch_idle->to_read_only();
      auto batch_size = batch_ro.get_data()->get_size_in_bytes();
      total_batch_size += batch_size;
      // Check if the batch size is already exceed the threshold
      if (!_concat_all && total_batch_size > _concat_batch_bytes) {
        // if the batch size is already exceed the threshold, then we need to return the batch right
        // away
        if (input_batch.size() == 0) {
          // this mean that there is a batch that is bigger than the threshold, then we just output
          // that batch right away
          auto popped_batch = port_ptr->repo->pop_data_batch_by_id(batch_id, i);
          input_batch.push_back(std::move(popped_batch));
        }
        break;
      } else {
        // if the batch size does not exceed the threshold, then we need to add the batch to the
        // input batch
        auto popped_batch = port_ptr->repo->pop_data_batch_by_id(batch_id, i);
        input_batch.push_back(std::move(popped_batch));
      }
    }
    if (input_batch.size() != 0) {
      return std::make_unique<partitioned_operator_data>(std::move(input_batch), i);
    }
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_concat::execute(const operator_data& input_data,
                                                               rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_concat::execute"};
  auto partitioned_input_data = dynamic_cast<const partitioned_operator_data*>(&input_data);
  if (partitioned_input_data == nullptr) {
    throw std::runtime_error(
      "sirius_physical_concat: input_data is not a partitioned_operator_data");
  }
  const auto& input_batches = partitioned_input_data->get_read_only_batches();
  auto partition_idx        = partitioned_input_data->get_partition_idx();
  if (input_batches.empty()) {
    return std::make_unique<partitioned_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}, partition_idx);
  }

  cucascade::memory::memory_space* space = input_batches[0].get_memory_space();
  if (space == nullptr) { throw std::runtime_error("sirius_physical_concat: space is nullptr"); }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(1);
  if (input_batches.size() == 1) {
    auto copy   = input_batches[0];
    auto output = cucascade::data_batch::to_idle(std::move(copy));
    output_batches.push_back(std::move(output));
  } else {
    auto merged_batch = gpu_merge_impl::concat(input_batches, stream, *space);
    output_batches.push_back(std::move(merged_batch));
  }
  return std::make_unique<partitioned_operator_data>(output_batches, partition_idx);
}

void sirius_physical_concat::sink(const operator_data& output_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_concat::sink"};
  auto partitioned_output_data = dynamic_cast<const partitioned_operator_data*>(&output_data);
  auto partition_idx           = partitioned_output_data->get_partition_idx();
  for (auto& batch : partitioned_output_data->get_data_batches()) {
    for (auto& next_port_info : next_port_after_sink) {
      auto partition_consumer_op =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (partition_consumer_op) {
        partition_consumer_op->push_data_batch_partitioned(
          next_port_info.next_operator_port_name, batch, partition_idx);
      } else {
        throw std::runtime_error(
          "sirius_physical_concat::sink(): Next operator is not a partition consumer operator: " +
          SiriusPhysicalOperatorToString(next_port_info.next_operator->type));
      }
    }
  }
}

std::string sirius_physical_concat::get_name() const { return "CONCAT"; }

bool sirius_physical_concat::is_source() const { return true; }

bool sirius_physical_concat::is_sink() const { return true; }

bool sirius_physical_concat::is_build_concat() { return _is_build; }

void sirius_physical_concat::set_concat_all(bool concat_all)
{
  std::lock_guard<std::mutex> lg(lock);
  _concat_all = concat_all;
}

std::size_t sirius_physical_concat::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  if (stats.num_batches <= 1) { return 0; }
  return stats.bytes;
}

}  // namespace op
}  // namespace sirius
