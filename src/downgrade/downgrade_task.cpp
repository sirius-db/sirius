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

#include "downgrade/downgrade_task.hpp"

#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>

#include <chrono>

namespace sirius {
namespace parallel {

void downgrade_task::execute(rmm::cuda_stream_view stream)
{
  auto& batch = _local_state->cast<downgrade_task_local_state>()._batch;

  // Check if already on host tier - nothing to do
  auto memory_space = batch->get_memory_space();
  if (memory_space == nullptr || memory_space->get_tier() != cucascade::memory::Tier::GPU) {
    mark_task_completion();
    return;
  }

  // Save the batch state so we can restore it after the in-transit conversion.
  // The batch may be in task_created state if a pipeline task is pending for it;
  // blindly resetting to idle would cause the pipeline task to fail with invalid_state.
  auto prev_state = batch->get_state();

  // Try to acquire an in-transit lock - if batch is being processed, we can't downgrade
  if (!batch->try_to_lock_for_in_transit()) {
    // Batch is currently being processed or moving, skip downgrade for now
    // The scheduler can retry later
    mark_task_completion();
    return;
  }

  auto data_size = batch->get_data()->get_size_in_bytes();
  auto t_start   = std::chrono::steady_clock::now();

  try {
    auto& mr_manager = _global_state->cast<downgrade_task_global_state>()._reservation_manager;
    auto reservation = mr_manager.request_reservation(
      cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST}, data_size);
    if (!reservation) {
      throw rmm::out_of_memory("Failed to allocate host memory for downgrade task.");
    }

    // Reservation identifies a memory_space (tier + device). Fetch its default allocator.
    auto mem_space = mr_manager.get_memory_space(reservation->tier(), reservation->device_id());
    if (!mem_space) { throw std::runtime_error("Invalid reservation memory_space for HOST tier"); }

    // Use the centralized converter registry to convert GPU representation to HOST
    auto& converter_registry = sirius::converter_registry::get();
    batch->convert_to<cucascade::host_data_representation>(converter_registry, mem_space, stream);

    // Release the in-transit lock, restoring the batch to its previous state
    batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});

    auto duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    double throughput_mbs =
      (duration_ms > 0.0) ? (data_size / (1024.0 * 1024.0)) / (duration_ms / 1000.0) : 0.0;
    SIRIUS_LOG_TRACE("[downgrade] batch {} done: {} B in {:.2f} ms ({:.1f} MB/s)",
                     batch->get_batch_id(),
                     data_size,
                     duration_ms,
                     throughput_mbs);

    mark_task_completion();
    return;
  } catch (...) {
    batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
    throw;
  }
}

void downgrade_task::mark_task_completion()
{
  // notify task_creator about task completion
  uint64_t task_id     = _local_state->cast<downgrade_task_local_state>()._task_id;
  uint64_t pipeline_id = _local_state->cast<downgrade_task_local_state>()._pipeline_id;
  auto message         = std::make_unique<sirius::task_completion_message>();
  message->task_id     = task_id;
  message->pipeline_id = pipeline_id;
  message->source      = sirius::Source::DOWNGRADE;
  _global_state->cast<downgrade_task_global_state>()._message_queue.EnqueueMessage(
    std::move(message));
}

uint64_t downgrade_task::get_task_id() const
{
  return _local_state->cast<downgrade_task_local_state>()._task_id;
}

}  // namespace parallel
}  // namespace sirius
