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

#include "data/data_batch_utils.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_concat::sirius_physical_concat(duckdb::vector<sirius::logical_type> types,
                                               std::size_t estimated_cardinality,
                                               sirius_physical_operator* downstream_join,
                                               bool is_build,
                                               uint64_t concat_batch_bytes)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::CONCAT, std::move(types), estimated_cardinality)
{
  _is_build           = is_build;
  _concat_batch_bytes = concat_batch_bytes;
  // `downstream_join` (the HJ/NLJ this CONCAT feeds — not the tree parent) picks
  // `_concat_all` and is stashed for the legacy converter's destination lookup.
  _downstream_join = downstream_join;
  if (downstream_join->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    auto hash_join = dynamic_cast<sirius_physical_hash_join*>(downstream_join);
    if (hash_join->join_type == duckdb::JoinType::LEFT ||
        hash_join->join_type == duckdb::JoinType::ANTI ||
        hash_join->join_type == duckdb::JoinType::SEMI) {
      // if the join type is left or anti, then we need to concat all the batches into one batch for
      // the build side
      _concat_all = is_build;
    } else if (hash_join->is_right_family()) {
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
  } else if (downstream_join->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
    _concat_all = false;
  } else {
    throw std::runtime_error("sirius_physical_concat: downstream_join is not a hash/nlj: " +
                             SiriusPhysicalOperatorToString(downstream_join->type));
  }
}

std::optional<std::vector<uint64_t>> sirius_physical_concat::plan_pull_for_partition(
  ::cucascade::shared_data_repository& repo,
  std::size_t partition_idx,
  bool pipeline_finished) const
{
  auto batch_ids = repo.get_batch_ids(partition_idx);
  if (batch_ids.empty()) { return std::nullopt; }

  if (_concat_all) {
    // A concat-all group is the whole partition, so it can only form once the source is done.
    if (!pipeline_finished) { return std::nullopt; }
    return batch_ids;
  }

  std::vector<uint64_t> group;
  std::size_t total_batch_size = 0;
  for (auto const batch_id : batch_ids) {
    auto batch_idle = repo.get_data_batch_by_id(batch_id, partition_idx);
    auto batch_ro   = batch_idle->to_read_only();
    total_batch_size += batch_ro.get_data()->get_size_in_bytes();
    if (total_batch_size > _concat_batch_bytes) {
      // The accumulated group is complete; the overflowing batch seeds the next group.
      if (!group.empty()) { return group; }
      // The first batch alone exceeds the threshold. Release it as a single-batch group unless
      // it is the only batch and more data may still arrive.
      if (pipeline_finished || batch_ids.size() > 1) { return std::vector<uint64_t>{batch_id}; }
      return std::nullopt;
    }
    group.push_back(batch_id);
  }

  // The whole partition fits under the threshold: keep accumulating until the source is done.
  if (!pipeline_finished) { return std::nullopt; }
  return group;
}

std::optional<task_creation_hint> sirius_physical_concat::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(lock);

  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto const port_ptr          = ports.begin()->second;
  bool const pipeline_finished = is_source_pipeline_finished();

  // If the source pipeline is done, we're ready to process whatever data remains
  if (pipeline_finished) {
    if (port_ptr->repo->total_size() > 0) {
      return task_creation_hint{TaskCreationHint::READY, this};
    }
    return std::nullopt;
  }

  if (_concat_all) {
    // if we need to concat all then we need to wait for the pipeline to be finished
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                              &(port_ptr->src_pipeline->get_operators()[0].get())};
  }

  // Source pipeline still running — fire early only if some partition already holds a group
  // that get_next_task_input_data would release.
  for (std::size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    if (plan_pull_for_partition(*port_ptr->repo, i, /*pipeline_finished=*/false)) {
      return task_creation_hint{TaskCreationHint::READY, this};
    }
  }

  // Not enough data yet — wait for more from the source pipeline
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                            &(port_ptr->src_pipeline->get_operators()[0].get())};
}

std::unique_ptr<operator_data> sirius_physical_concat::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(lock);

  // assert that there is only one port
  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto const port_ptr          = ports.begin()->second;
  bool const pipeline_finished = is_source_pipeline_finished();

  // Pull from the first partition where the group-forming policy releases a group.
  for (std::size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    auto plan = plan_pull_for_partition(*port_ptr->repo, i, pipeline_finished);
    if (!plan) { continue; }
    std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
    input_batch.reserve(plan->size());
    for (auto const batch_id : *plan) {
      input_batch.push_back(port_ptr->repo->pop_data_batch_by_id(batch_id, i));
    }
    return std::make_unique<partitioned_operator_data>(std::move(input_batch), i);
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
    auto merged_batch = gpu_merge_impl::concat(input_batches, stream, *space, batch_telemetry());
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

bool sirius_physical_concat::is_build_concat() const { return _is_build; }

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
