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
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/readahead_scan_manager.hpp"

#include <stop_token>
#include <utility>

namespace sirius::scan_manager {

load_balancing_scan_batch_coalescer::metadata_processing_state*
load_balancing_scan_batch_coalescer::register_pipeline(
  op::scan::sirius_gpu_scan_operator* scan_op,
  std::shared_ptr<balancing_strategy> balancer,
  std::shared_ptr<readahead_scan_manager> readahead)
{
  if (!scan_op) return nullptr;

  auto connector   = scan_op->get_split_connector().shared_from_this();
  auto ingestible  = scan_op->get_ingestible().shared_from_this();
  auto coalescer   = ingestible->create_batch_coalescer();
  auto uid         = scan_op->get_operator_id();
  auto pipeline_id = scan_op->get_pipeline()->get_pipeline_id();
  auto state       = std::make_unique<metadata_processing_state>(uid,
                                                           pipeline_id,
                                                           std::move(coalescer),
                                                           std::move(connector),
                                                           std::move(balancer),
                                                           std::move(readahead));
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
  // Cached batches always reach post_filter_and_project unfiltered, so the
  // op's row filter (when present) runs against every drained split — record
  // that so each split's working-set estimate covers the filter-by-copy peak.
  state.row_filter_pending = scan_op->get_ingestible().has_row_filter();
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

void load_balancing_scan_batch_coalescer::slot_loop(std::size_t pipeline_id,
                                                    std::stop_token const& stop)
{
  if (stop.stop_requested()) { return; }
  auto it = _slots.find(pipeline_id);
  if (it == _slots.end()) { return; }
  auto& state = *it->second;
  if (state.batch_provider) {
    process_cached_entries(state, stop);
  } else {
    process_provider_inputs(state, stop);
  }
}

void load_balancing_scan_batch_coalescer::process_provider_inputs(metadata_processing_state& state,
                                                                  std::stop_token const& stop)
{
  std::stop_callback stop_cb(stop, [&state] {
    state.queue.enqueue(exec::make_empty_try<std::unique_ptr<op::scan::scan_info>>());
  });
  auto& batch_queue = state.queue;

  // Every exit from this function closes the slot through here.  The readahead
  // cannot retire an operator until it knows no further split will be emitted,
  // and that includes the failure paths: a slot that closed with an exception
  // produces nothing more either.  Registration happens inside emit() on this
  // same thread, so the close always follows the last register in program order.
  auto close_slot = [&state](std::exception_ptr const& ex = nullptr) {
    if (state.readahead) { state.readahead->mark_operator_closed(state.op_id); }
    state.connector->close(ex);
  };

  // Balance one coalesced batch onto a GPU and hand it to the connector.  The
  // device is chosen before the split exists because the constructor fadvises
  // and publishes in one step, leaving no window afterwards to stamp one in.
  auto emit = [&state](std::unique_ptr<op::scan::scan_info> batch) {
    auto dev_id = state.balancer->get_next_gpu(state.pipeline_id);
    state.connector->push_split(std::make_unique<op::scan::scan_operator_input>(
      std::move(batch), state.readahead, state.op_id, dev_id));
  };

  while (!stop.stop_requested()) {
    try {
      metadata_processing_state::provider_value_t entry;
      batch_queue.wait_dequeue(entry);
      if (entry.has_exception()) {
        close_slot(entry.exception());
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
          close_slot(leftover.exception());
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
      close_slot();
    } catch (...) {
      close_slot(std::current_exception());
    }
    return;
  }

  // Stop requested between iterations — close so downstream consumers unblock.
  close_slot();
}

void load_balancing_scan_batch_coalescer::process_cached_entries(metadata_processing_state& state,
                                                                 std::stop_token const& stop)
{
  drain_cached_provider(*state.batch_provider,
                        *state.connector,
                        stop,
                        state.row_filter_pending,
                        state.readahead,
                        state.op_id);
}

void load_balancing_scan_batch_coalescer::drain_cached_provider(
  databatch_provider& provider,
  split_connector& connector,
  std::stop_token const& stop,
  bool row_filter_pending,
  std::shared_ptr<readahead_scan_manager> readahead,
  std::size_t operator_id)
{
  // See process_provider_inputs: the readahead needs "no more splits" on every
  // exit, success or failure, or the operator can never be retired.
  auto close_slot = [&connector, &readahead, operator_id](std::exception_ptr const& ex = nullptr) {
    if (readahead) { readahead->mark_operator_closed(operator_id); }
    connector.close(ex);
  };

  try {
    while (!stop.stop_requested()) {
      auto next = provider.get_next_batch();
      if (next.data) {
        auto split = std::make_unique<op::scan::scan_operator_input>(
          std::move(next.data), readahead, operator_id);
        split->mvcc_keep_mask               = std::move(next.mvcc_keep_mask);
        split->needs_carrier_conversion     = next.needs_carrier_conversion;
        split->conversion_destination_bytes = next.conversion_destination_bytes;
        split->row_filter_pending           = row_filter_pending;
        connector.push_split(std::move(split));
        continue;
      }
      if (next.scan_info) {
        // Insert-delta split. row_filter_pending stays false: scan_info
        // splits fold filter costs into their own estimates. The constructor
        // fadvises and publishes, as on the walk path; host-backed splits have
        // no file ranges, so the hints no-op.
        std::optional<int> device;
        if (next.preferred_device >= 0) { device = next.preferred_device; }
        auto split = std::make_unique<op::scan::scan_operator_input>(
          std::move(next.scan_info), readahead, operator_id, device);
        split->mvcc_keep_mask = std::move(next.mvcc_keep_mask);
        connector.push_split(std::move(split));
        continue;
      }
      break;  // end-of-stream
    }
    close_slot();
  } catch (...) {
    // Surface the provider failure to the consumer: get_next_split() rethrows
    // once the queue drains. Without this close the connector never closes —
    // the dispatcher swallows task exceptions — and the query hangs silently.
    close_slot(std::current_exception());
  }
}

}  // namespace sirius::scan_manager
