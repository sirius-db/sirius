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
// #include "downgrade/downgrade_executor.hpp"
#include "memory/memory_reservation.hpp"
#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include <rmm/cuda_stream_view.hpp>
#include "cudf/contiguous_split.hpp"

namespace sirius {
namespace parallel {

void downgrade_task::execute()
{
  // TODO: store this in local state i think
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  // get the memory_space and check that its gpu
  auto memory_space = _local_state->cast<downgrade_task_local_state>()._batch->get_memory_space();
  if (memory_space->get_tier() != memory::Tier::GPU) {
    mark_task_completion();
    return;
  } else {
    auto& batch    = _local_state->cast<downgrade_task_local_state>()._batch;
    auto data_size = batch->get_data()->get_size_in_bytes();

    try {
      auto& mr_manager = sirius::memory::memory_reservation_manager::get_instance();
      auto reservation = mr_manager.request_reservation(
        sirius::memory::any_memory_space_in_tier{sirius::memory::Tier::HOST}, data_size);
      if (!reservation) {
        throw rmm::out_of_memory("Failed to allocate host memory for downgrade task.");
      }
      // Reservation identifies a memory_space (tier + device). Fetch its default allocator.
      auto mem_space = mr_manager.get_memory_space(reservation->tier, reservation->device_id);
      if (!mem_space) {
        throw std::runtime_error("Invalid reservation memory_space for HOST tier");
      }
      auto fixed_mr =
        mem_space->get_default_allocator_as<sirius::memory::fixed_size_host_memory_resource>();
      if (fixed_mr == nullptr) {
        throw std::runtime_error("Default HOST allocator is not fixed_size_host_memory_resource");
      }

      batch->convert_to_memory_space(mem_space, stream);

      mark_task_completion();
      return;

    } catch (const rmm::out_of_memory& e) {
      throw std::runtime_error("Failed to allocate gpu_memory");
    }

    // Obtain HOST-tier memory resource from the memory manager
    // auto& mr_manager = sirius::memory::memory_reservation_manager::get_instance();
    // auto host_spaces = mr_manager.get_memory_spaces_for_tier(sirius::memory::Tier::HOST);
    // if (host_spaces.empty()) {
    //     mark_task_completion();
    //     return;
    // }
    // auto host_allocator_ref = host_spaces[0]->get_default_allocator();
    // auto* host_fixed_mr =
    // dynamic_cast<sirius::memory::fixed_size_host_memory_resource*>(&host_allocator_ref.get());

    // // Fallback: if cast fails, complete task without conversion
    // if (host_fixed_mr == nullptr) {
    //     mark_task_completion();
    //     return;
    // }

    // // Use default CUDA stream for the conversion
    // rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // auto host_table = detail::convert_to_host_representation(table, host_fixed_mr, stream);
    // _local_state->cast<downgrade_task_local_state>()._batch->set_data(std::move(host_table));
    // mark_task_completion();
    // return;
  }

  mark_task_completion();
}

void downgrade_task::mark_task_completion()
{
  // notify task_creator about task completion
  uint64_t task_id     = _local_state->cast<downgrade_task_local_state>()._task_id;
  uint64_t pipeline_id = _local_state->cast<downgrade_task_local_state>()._pipeline_id;
  auto message         = sirius::make_unique<sirius::task_completion_message>();
  message->task_id     = task_id;
  message->pipeline_id = pipeline_id;
  message->source      = sirius::Source::PIPELINE;
  _global_state->cast<downgrade_task_global_state>()._message_queue.EnqueueMessage(
    std::move(message));
}

uint64_t downgrade_task::get_task_id() const
{
  return _local_state->cast<downgrade_task_local_state>()._task_id;
}

}  // namespace parallel
}  // namespace sirius