/*
 * Copyright 2026, Sirius Contributors.
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

#include "scan_manager/load_balancing_scan_batch_coalecer.hpp"

#include "op/scan/sirius_gpu_scan_operator_data.hpp"

#include <stop_token>
#include <utility>

namespace sirius::scan_manager {

load_balancing_scan_batch_coalecer::metadata_processing_state*
load_balancing_scan_batch_coalecer::register_pipeline(op::scan::sirius_gpu_scan_operator* scan_op,
                                                      std::shared_ptr<balancing_strategy> balancer)
{
  if (!scan_op) return nullptr;

  auto connector   = scan_op->get_split_connector().shared_from_this();
  auto ingestible  = scan_op->get_ingestible().shared_from_this();
  auto coalecer    = ingestible->create_batch_coalecer();
  auto uid         = scan_op->get_operator_id();
  auto pipeline_id = scan_op->get_pipeline()->get_pipeline_id();
  auto state       = std::make_unique<metadata_processing_state>(
    uid, pipeline_id, std::move(coalecer), std::move(connector), std::move(balancer));
  _pipeline_order.push_back(uid);
  auto state_ptr = state.get();
  _slots[uid]    = std::move(state);
  return state_ptr;
}

void load_balancing_scan_batch_coalecer::use_cached_entries_for_pipeline(
  op::scan::sirius_gpu_scan_operator* scan_op, std::unique_ptr<databatch_provider> provider)
{
  if (!scan_op) return;
  auto uid = scan_op->get_operator_id();
  auto it  = _slots.find(uid);
  if (it == _slots.end()) { return; }
  auto& state = *it->second;
  state.attach_batch_provider(std::move(provider));
}

std::function<void(exec::try_t<std::unique_ptr<op::scan::scan_info>>&&)>
load_balancing_scan_batch_coalecer::get_split_provider_bridge(
  op::scan::sirius_gpu_scan_operator* scan_op)
{
  if (!scan_op) return nullptr;
  auto uid = scan_op->get_operator_id();
  auto it  = _slots.find(uid);
  if (it == _slots.end()) { return {}; }
  return [state_ptr = it->second](exec::try_t<std::unique_ptr<op::scan::scan_info>>&& entry) {
    state_ptr->queue.enqueue(std::move(entry));
  };
}

void load_balancing_scan_batch_coalecer::worker_loop([[maybe_unused]] std::stop_token const& stop)
{
  for (auto pipeline_id : _pipeline_order) {
    if (stop.stop_requested()) { break; }
    auto& state = *_slots[pipeline_id];
    if (state.batch_provider) {
      process_cached_entries(state, stop);
    } else {
      process_provider_inputs(state, stop);
    }
  }
}

void load_balancing_scan_batch_coalecer::process_provider_inputs(metadata_processing_state& state,
                                                                 std::stop_token const& stop)
{
  std::stop_callback stop_cb(stop, [&state] { state.queue.enqueue(nullptr); });
  auto& batch_queue = state.queue;
  bool is_closed    = false;
  while (!is_closed && !stop.stop_requested()) {
    metadata_processing_state::provider_value_t entry;
    batch_queue.wait_dequeue(entry);
    if (entry.has_exception()) {
      state.connector->close(entry.exception());
      break;
    }
    is_closed    = entry.is_empty();
    auto batches = [&]() {
      if (is_closed) {
        return state.coalecer->flush();
      } else {
        return state.coalecer->push(std::move(entry).value());
      }
    }();
    for (auto& batch : batches) {
      auto op_data = std::make_unique<op::scan::scan_operator_input>(std::move(batch));
      auto dev_id  = state.balancer->get_next_gpu(state.pipeline_id, op_data.get());
      if (dev_id >= 0) { op_data->set_preferred_device_id(dev_id); }

      auto fadvise_hints = op_data->get_fadvise_hints();
      if (!fadvise_hints.empty()) {
        for (auto& hint : fadvise_hints) {
          if (hint.datasource && !hint.ranges.empty()) {
            hint.datasource->fadvise(hint.ranges, dev_id);
          };
        }
      }
      op_data->prefetch(io::cache::prefetching_stage::opportunistic);
      state.connector->push_split(std::move(op_data));
    }
    if (is_closed) {
      state.connector->close();
      break;
    }
  }
}

void load_balancing_scan_batch_coalecer::process_cached_entries(
  metadata_processing_state& state, [[maybe_unused]] std::stop_token const& stop)
{
  auto& batch_queue = state.queue;
  bool is_closed    = false;
  while (!is_closed) {
    auto databatch = state.batch_provider->get_next_batch();
    balancing_strategy::device_id_hint hint;
    {
      auto rdonly = databatch->try_to_read_only();
      if (rdonly) {
        auto* space = rdonly->get_memory_space();
        if (space) { hint = balancing_strategy::make_target_hint(space->get_id()); }
      }
    }
    is_closed = databatch == nullptr;
    if (!is_closed) {
      auto op_data = std::make_unique<op::scan::scan_operator_input>(std::move(databatch));
      auto dev_id  = state.balancer->get_next_gpu(state.pipeline_id, op_data.get(), hint);
      if (dev_id >= 0) { op_data->set_preferred_device_id(dev_id); }
      state.connector->push_split(std::move(op_data));
    }
  }
  state.connector->close();
}

}  // namespace sirius::scan_manager
