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
#include "log/logging.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/partition/gpu_partition_impl.hpp"
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

uint64_t sirius_physical_concat::effective_batch_bytes()
{
  // Caller holds `lock`.
  int const parts = _probe_split_parts.load(std::memory_order_acquire);
  if (parts <= 1 || _concat_all) { return _concat_batch_bytes; }
  if (uint64_t const cached = _split_budget_bytes.load(std::memory_order_acquire); cached != 0) {
    return cached;
  }

  auto port_ptr = ports.begin()->second;
  // While the source pipeline is still producing we do not know the total probe size, and the
  // default budget already emits one batch per `_concat_batch_bytes` — a probe that large is
  // already multi-task. The pathological case is a probe that fits in one budget, which is only
  // knowable once the source is finished. (No source pipeline at all means no further input can
  // arrive, so what is resident is the whole side.)
  if (port_ptr->src_pipeline != nullptr && !port_ptr->src_pipeline->is_pipeline_finished()) {
    return _concat_batch_bytes;
  }

  uint64_t total = 0;
  for (std::size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    for (auto& batch_id : port_ptr->repo->get_batch_ids(i)) {
      auto batch = port_ptr->repo->get_data_batch_by_id(batch_id, i);
      if (!batch) { continue; }
      auto batch_ro = batch->to_read_only();
      if (batch_ro.get_data()) { total += batch_ro.get_data()->get_size_in_bytes(); }
    }
  }
  if (total == 0) { return _concat_batch_bytes; }  // nothing resident yet; do not fix a budget

  auto const divisor = static_cast<uint64_t>(parts);
  uint64_t budget    = (total + divisor - 1) / divisor;
  // Never split into batches too small to be worth a task, and never make the batch LARGER than
  // the configured concat budget (so this can only ever add parallelism, never remove it). The
  // floor is itself capped by the configured budget, which a user may have set below it.
  uint64_t const floor_bytes = std::min(_min_probe_split_bytes, _concat_batch_bytes);
  budget                     = std::clamp(budget, floor_bytes, _concat_batch_bytes);
  if (budget == 0) { return _concat_batch_bytes; }
  _split_budget_bytes.store(budget, std::memory_order_release);
  SIRIUS_LOG_DEBUG(
    "sirius_physical_concat id {}: BUILD_PROBE probe split into ~{} batches ({} total bytes, {} "
    "bytes per batch)",
    this->get_operator_id(),
    (total + budget - 1) / budget,
    total,
    budget);
  return budget;
}

std::optional<task_creation_hint> sirius_physical_concat::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(lock);

  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto port_ptr          = ports.begin()->second;
  bool pipeline_finished = port_ptr->src_pipeline && port_ptr->src_pipeline->is_pipeline_finished();

  // If the source pipeline is done, we're ready to process whatever data remains
  if (pipeline_finished) {
    if (port_ptr->repo->total_size() > 0) {
      return task_creation_hint{TaskCreationHint::READY, this};
    }
    return std::nullopt;
  } else if (_concat_all) {
    // if we need to concat all then we need to wait for the pipeline to be finished
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                              &(port_ptr->src_pipeline->get_operators()[0].get())};
  }

  // Source pipeline still running — check if there is enough data to fire a task early.
  // "Enough" means: for some partition, simulating get_next_task_input_data would pull a group
  // of batches AND there would still be at least one batch left in that partition afterward.
  // (effective_batch_bytes() is `_concat_batch_bytes` on this path — the probe split only fixes a
  // smaller budget once the source pipeline is finished — but it keeps the two paths in sync.)
  uint64_t const batch_bytes_budget = effective_batch_bytes();
  for (size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    auto batch_ids          = port_ptr->repo->get_batch_ids(i);
    size_t total_batch_size = 0;
    size_t pulled_count     = 0;
    for (auto& batch_id : batch_ids) {
      auto batch_idle = port_ptr->repo->get_data_batch_by_id(batch_id, i);
      auto batch_ro   = batch_idle->to_read_only();
      auto batch_size = batch_ro.get_data()->get_size_in_bytes();
      total_batch_size += batch_size;
      if (!_concat_all && total_batch_size > batch_bytes_budget) {
        // This batch pushes us over the threshold — the loop would stop here.
        // If we already accumulated batches (pulled_count > 0), the overflowing batch stays,
        // so there is at least one batch left after the pull.
        if (pulled_count > 0) { return task_creation_hint{TaskCreationHint::READY, this}; }
        // If nothing was accumulated yet, the single oversized batch itself would be pulled,
        // and remaining data is everything after it.
        if (batch_ids.size() > 1) { return task_creation_hint{TaskCreationHint::READY, this}; }
        break;
      } else {
        pulled_count++;
      }
    }
  }

  // Not enough data yet — wait for more from the source pipeline
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                            &(port_ptr->src_pipeline->get_operators()[0].get())};
}

std::unique_ptr<operator_data> sirius_physical_concat::get_next_task_input_data()
{
  // iterate through all the partition and pull
  std::lock_guard<std::mutex> lg(lock);

  // assert that there is only one port
  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto port_ptr                     = ports.begin()->second;
  uint64_t const batch_bytes_budget = effective_batch_bytes();
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
      if (!_concat_all && total_batch_size > batch_bytes_budget) {
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

  // BUILD_PROBE probe split, case 2: grouping in get_next_task_input_data caps a pulled group at
  // the split budget, but it cannot split a SINGLE upstream batch that is already larger than the
  // budget — and "the whole probe arrived as one scan batch" is exactly the one-task case we are
  // fixing. Slice it into row-disjoint pieces here; every probe row still lands in exactly one
  // output batch, so per-probe-row semantics (LEFT NULL-padding, SEMI/ANTI/MARK one-in-one-out)
  // are untouched. Only ever reached on a probe-side CONCAT of a single-partition BUILD_PROBE join
  // (see compute_hash_join_partition_strategy for the join-type allowlist).
  if (int const split_parts = _probe_split_parts.load(std::memory_order_acquire); split_parts > 1) {
    uint64_t const budget = _split_budget_bytes.load(std::memory_order_acquire);
    auto const* data      = input_batches[0].get_data();
    uint64_t const bytes  = data != nullptr ? data->get_size_in_bytes() : 0;
    if (input_batches.size() == 1 && budget > 0 && bytes > budget) {
      auto const num_rows = get_cudf_table_view(input_batches[0]).num_rows();
      auto parts          = static_cast<int64_t>((bytes + budget - 1) / budget);
      parts               = std::min<int64_t>(parts, split_parts);
      parts               = std::min<int64_t>(parts, num_rows);  // never emit empty slices
      if (parts > 1) {
        auto slices = gpu_partition_impl::evenly_partition(
          input_batches[0], static_cast<int>(parts), stream, *space, batch_telemetry());
        return std::make_unique<partitioned_operator_data>(std::move(slices), partition_idx);
      }
    }
  }

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

void sirius_physical_concat::set_probe_split_parts(int parts, uint64_t min_batch_bytes)
{
  std::lock_guard<std::mutex> lg(lock);
  if (_is_build || _concat_all) {
    // Defensive: the build side must stay foldable to a single batch for BUILD_PROBE, and a
    // concat_all side is whole-side by definition. Callers only target probe-side CONCATs.
    return;
  }
  _min_probe_split_bytes = min_batch_bytes;
  _probe_split_parts.store(std::max(parts, 1), std::memory_order_release);
}

std::size_t sirius_physical_concat::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  // A single input batch is normally forwarded without a copy — except when the probe split is
  // enabled, where an oversized single batch is sliced into a fresh copy of the same total size.
  if (stats.num_batches <= 1) {
    return _probe_split_parts.load(std::memory_order_acquire) > 1 ? stats.bytes : 0;
  }
  return stats.bytes;
}

}  // namespace op
}  // namespace sirius
