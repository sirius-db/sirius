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

#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"

#include "exec/try.hpp"
#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"

#include <stop_token>
#include <utility>

namespace sirius::scan_manager {

load_balancing_scan_batch_coalescer::metadata_processing_state*
load_balancing_scan_batch_coalescer::register_pipeline(op::scan::sirius_gpu_scan_operator* scan_op,
                                                       std::shared_ptr<balancing_strategy> balancer)
{
  if (!scan_op) return nullptr;

  auto connector   = scan_op->get_split_connector().shared_from_this();
  auto ingestible  = scan_op->get_ingestible().shared_from_this();
  auto coalescer   = ingestible->create_batch_coalescer();
  auto uid         = scan_op->get_operator_id();
  auto pipeline_id = scan_op->get_pipeline()->get_pipeline_id();
  auto state       = std::make_unique<metadata_processing_state>(
    uid, pipeline_id, std::move(coalescer), std::move(connector), std::move(balancer));
  _pipeline_order.push_back(uid);
  auto state_ptr = state.get();
  _slots[uid]    = std::move(state);
  return state_ptr;
}

void load_balancing_scan_batch_coalescer::use_cached_entries_for_pipeline(
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
load_balancing_scan_batch_coalescer::get_split_provider_bridge(
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

void load_balancing_scan_batch_coalescer::worker_loop([[maybe_unused]] std::stop_token const& stop)
{
  for (auto pipeline_id : _pipeline_order) {
    if (stop.stop_requested()) { break; }
    auto& state = *_slots.at(pipeline_id);
    if (state.batch_provider) {
      process_cached_entries(state, stop);
    } else {
      process_provider_inputs(state, stop);
    }
  }
}

void load_balancing_scan_batch_coalescer::process_provider_inputs(metadata_processing_state& state,
                                                                  std::stop_token const& stop)
{
  std::stop_callback stop_cb(stop, [&state] {
    state.queue.enqueue(exec::make_empty_try<std::unique_ptr<op::scan::scan_info>>());
  });
  auto& batch_queue = state.queue;

  // Balance one coalesced batch onto a GPU and hand it to the connector.
  auto emit = [&state](std::unique_ptr<op::scan::scan_info> batch) {
    auto op_data = std::make_unique<op::scan::scan_operator_input>(std::move(batch));
    auto dev_id  = state.balancer->get_next_gpu(state.pipeline_id, op_data.get());
    SIRIUS_LOG_DEBUG("[coalesce-debug] load_balancer emit pipeline={} -> preferred GPU {}",
                     state.pipeline_id,
                     dev_id.has_value() ? dev_id.value() : -1);
    if (dev_id.has_value() && *dev_id >= 0) { op_data->set_preferred_device_id(dev_id.value()); }

    auto fadvise_hints = op_data->get_fadvise_hints();
    if (!fadvise_hints.empty()) {
      for (auto& hint : fadvise_hints) {
        if (hint.datasource && !hint.ranges.empty()) {
          hint.datasource->fadvise(hint.ranges, dev_id);
        }
      }
    }
    op_data->prefetch(io::cache::prefetching_stage::opportunistic);
    state.connector->push_split(std::move(op_data));
  };

  while (!stop.stop_requested()) {
    try {
      metadata_processing_state::provider_value_t entry;
      batch_queue.wait_dequeue(entry);
      if (entry.has_exception()) {
        state.connector->close(entry.exception());
        return;
      }

      if (!entry.is_empty()) {
        auto batches = state.coalescer->push(std::move(entry).value());
        for (auto& batch : batches) {
          emit(std::move(batch));
        }
        continue;
      }

      // End-of-input sentinel. The producer-side completion_controller emits it
      // only after every metadata task has finished enqueuing, so no further
      // entries will be produced. But state.queue is a multi-producer queue with
      // only per-producer (per-thread) FIFO: real entries enqueued before this
      // sentinel on other threads may still be sitting in the queue and get
      // delivered out of order. Drain them before closing so their splits are not
      // silently dropped (which would scan fewer files than requested).
      metadata_processing_state::provider_value_t leftover;
      while (batch_queue.try_dequeue(leftover)) {
        if (leftover.has_exception()) {
          state.connector->close(leftover.exception());
          return;
        }
        if (leftover.is_empty()) { continue; }  // extra (e.g. stop) sentinel — ignore
        auto batches = state.coalescer->push(std::move(leftover).value());
        for (auto& batch : batches) {
          emit(std::move(batch));
        }
      }
      auto final_batches = state.coalescer->flush();
      for (auto& batch : final_batches) {
        emit(std::move(batch));
      }
      state.connector->close();
    } catch (...) {
      state.connector->close(std::current_exception());
    }
    return;
  }

  // Stop requested between iterations — close so downstream consumers unblock.
  state.connector->close();
}

void load_balancing_scan_batch_coalescer::process_cached_entries(metadata_processing_state& state,
                                                                 std::stop_token const& stop)
{
  drain_cached_provider(*state.batch_provider, *state.connector, stop);
}

void load_balancing_scan_batch_coalescer::drain_cached_provider(databatch_provider& provider,
                                                                split_connector& connector,
                                                                std::stop_token const& stop)
{
  try {
    while (!stop.stop_requested()) {
      auto databatch = provider.get_next_batch();
      if (!databatch) { break; }
      connector.push_split(std::make_unique<op::scan::scan_operator_input>(std::move(databatch)));
    }
    connector.close();
  } catch (...) {
    // Surface the provider failure to the consumer: get_next_split() rethrows
    // once the queue drains. Without this close the connector never closes —
    // the dispatcher swallows task exceptions — and the query hangs silently.
    connector.close(std::current_exception());
  }
}

}  // namespace sirius::scan_manager
