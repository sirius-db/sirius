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

#include "creator/task_creator.hpp"
#include "log/logging.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>

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

std::optional<task_creation_hint> sirius_physical_concat::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(lock);

  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto port_ptr = ports.begin()->second;

  // If the source pipeline is done, we're ready to process whatever data remains.
  if (is_source_pipeline_finished()) {
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

  // Source pipeline still running — fire early once a partition has buffered more than the
  // batching threshold across at least two batches. A lone oversized batch keeps WAITING: it
  // either gains a groupmate or is flushed once the source finishes.
  for (const auto& totals : _input_totals) {
    if (totals.bytes > _concat_batch_bytes && totals.count >= 2) {
      return task_creation_hint{TaskCreationHint::READY, this};
    }
  }

  // Not enough data yet — wait for more from the source pipeline
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                            &(port_ptr->src_pipeline->get_operators()[0].get())};
}

void sirius_physical_concat::push_data_batch_partitioned(
  std::string_view port_id,
  std::shared_ptr<::cucascade::data_batch> batch,
  std::size_t partition_idx)
{
  if (batch) {
    // Measured before taking `lock`: to_read_only() can wait out a downgrade's exclusive batch
    // lock, and the manager-thread hint blocks on `lock`.
    auto const batch_size = batch->to_read_only().get_data()->get_size_in_bytes();
    std::lock_guard<std::mutex> lg(lock);
    if (_input_totals.size() <= partition_idx) { _input_totals.resize(partition_idx + 1); }
    _pushed_batch_bytes[batch->get_batch_id()] = batch_size;
    _input_totals[partition_idx].bytes += batch_size;
    _input_totals[partition_idx].count += 1;
  }
  // Delegated after releasing `lock` for the same reason: the base's telemetry publish takes the
  // batch's shared lock. Ledger-then-insert ordering keeps every poppable batch ledgered; the
  // reverse window (counted but not yet poppable) only makes the hint transiently optimistic.
  sirius_physical_partition_consumer_operator::push_data_batch_partitioned(
    port_id, std::move(batch), partition_idx);
}

std::shared_ptr<::cucascade::data_batch> sirius_physical_concat::pop_and_account(
  ::cucascade::shared_data_repository& repo, uint64_t batch_id, std::size_t partition_idx)
{
  auto popped = repo.pop_data_batch_by_id(batch_id, partition_idx);
  auto ledger = _pushed_batch_bytes.find(batch_id);
  if (ledger != _pushed_batch_bytes.end()) {
    auto& totals = _input_totals[partition_idx];
    totals.bytes -= ledger->second;
    totals.count -= 1;
    _pushed_batch_bytes.erase(ledger);
  } else {
    // Unledgered: tests that insert straight into the repository, or broadcast build slots (the
    // sink deposits one id into every slot; the ledger keys by id). Broadcast implies concat_all,
    // which consults neither the totals nor buffered_batch_size.
    SIRIUS_LOG_DEBUG("sirius_physical_concat: popped batch {} has no push-time ledger entry",
                     batch_id);
  }
  return popped;
}

uint64_t sirius_physical_concat::buffered_batch_size(::cucascade::shared_data_repository& repo,
                                                     uint64_t batch_id,
                                                     std::size_t partition_idx) const
{
  if (auto ledger = _pushed_batch_bytes.find(batch_id); ledger != _pushed_batch_bytes.end()) {
    return ledger->second;
  }
  return repo.get_data_batch_by_id(batch_id, partition_idx)
    ->to_read_only()
    .get_data()
    ->get_size_in_bytes();
}

std::unique_ptr<operator_data> sirius_physical_concat::get_next_task_input_data()
{
  std::unique_ptr<operator_data> task_input;
  bool forwarded = false;

  // Each pass forms one group; a single-batch group with sink wiring is forwarded downstream
  // without a task (execute is an identity for it) and the walk repeats. Every forward removes a
  // batch from the repository, so the loop terminates.
  // Contract: pipeline-wired callers hold get_task_creation_lock() — the forward's in-flight batch
  // is in no repository, and only that lock hides the gap from finish evaluation.
  while (true) {
    std::shared_ptr<::cucascade::data_batch> single_batch;
    std::size_t single_partition_idx = 0;
    {
      std::lock_guard<std::mutex> lg(lock);

      if (ports.size() != 1) {
        throw std::runtime_error("sirius_physical_concat: there should be only one port");
      }

      auto port_ptr = ports.begin()->second;
      for (std::size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
        std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
        auto batch_ids               = port_ptr->repo->get_batch_ids(i);
        std::size_t total_batch_size = 0;
        for (auto& batch_id : batch_ids) {
          // Sizes are push-time snapshots; _concat_all needs none.
          if (!_concat_all) {
            total_batch_size += buffered_batch_size(*port_ptr->repo, batch_id, i);
            if (total_batch_size > _concat_batch_bytes) {
              // An oversized head goes alone; otherwise the group closes and `batch_id` stays.
              if (input_batch.empty()) {
                input_batch.push_back(pop_and_account(*port_ptr->repo, batch_id, i));
              }
              break;
            }
          }
          input_batch.push_back(pop_and_account(*port_ptr->repo, batch_id, i));
        }
        if (input_batch.size() != 0) {
          if (input_batch.size() == 1 && !next_port_after_sink.empty()) {
            single_batch         = std::move(input_batch[0]);
            single_partition_idx = i;
          } else {
            task_input = std::make_unique<partitioned_operator_data>(std::move(input_batch), i);
          }
          break;
        }
      }
    }  // end lock
    if (!single_batch) { break; }

    for (auto const& next_port_info : next_port_after_sink) {
      auto partition_consumer_op =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (partition_consumer_op) {
        partition_consumer_op->push_data_batch_partitioned(
          next_port_info.next_operator_port_name, single_batch, single_partition_idx);
      } else {
        throw std::runtime_error(
          "sirius_physical_concat::get_next_task_input_data(): Next operator is not a partition "
          "consumer operator: " +
          SiriusPhysicalOperatorToString(next_port_info.next_operator->type));
      }
    }
    forwarded = true;
  }

  if (forwarded) {
    // A forward creates no task, so the executor's per-task consumer ping never fires; re-arm the
    // downstream consumers here. schedule() only touches the task creator's thread-safe creation
    // queue, so calling it from this (creator worker) thread is safe; a redundant ping is
    // harmless.
    auto pipeline      = get_pipeline();
    auto* task_creator = pipeline ? pipeline->get_task_creator() : nullptr;
    if (task_creator) {
      std::vector<sirius_physical_operator*> pinged;
      for (auto& next_port_info : next_port_after_sink) {
        auto* consumer = next_port_info.next_operator;
        if (std::find(pinged.begin(), pinged.end(), consumer) == pinged.end()) {
          pinged.push_back(consumer);
          task_creator->schedule(consumer);
        }
      }
    }
  }

  return task_input;
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
