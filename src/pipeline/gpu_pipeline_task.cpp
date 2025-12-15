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

#include "pipeline/gpu_pipeline_task.hpp"

#include "data/data_repository_manager.hpp"

namespace sirius {
namespace parallel {

gpu_pipeline_task::gpu_pipeline_task(sirius::unique_ptr<itask_local_state> local_state,
                                     sirius::shared_ptr<itask_global_state> global_state)
  : itask(std::move(local_state), std::move(global_state))
{
}

uint64_t gpu_pipeline_task::get_task_id() const
{
  return _local_state->cast<gpu_pipeline_task_local_state>()._task_id;
}

const duckdb::GPUPipeline* gpu_pipeline_task::get_pipeline() const
{
  return _global_state->cast<gpu_pipeline_task_global_state>()._pipeline.get();
}

void gpu_pipeline_task::execute()
{
  // Execute the task
  // Transfer data batch to GPU memory if not in GPU memory
  // set reservation_aware_memory_resource_ref as the default cudf allocator
  // Call cudf operators
  // after each cudf operator, get the peak total bytes to collect statistics
  mark_task_completion();
}

void gpu_pipeline_task::mark_task_completion()
{
  // notify TaskCreator about task completion
  uint64_t task_id     = _local_state->cast<gpu_pipeline_task_local_state>()._task_id;
  uint64_t pipeline_id = _global_state->cast<gpu_pipeline_task_global_state>()._pipeline_id;
  auto message         = sirius::make_unique<sirius::task_completion_message>();
  message->task_id     = task_id;
  message->pipeline_id = pipeline_id;
  message->source      = sirius::Source::PIPELINE;
  _global_state->cast<gpu_pipeline_task_global_state>()._message_queue.enqueue_message(
    std::move(message));
}

}  // namespace parallel
}  // namespace sirius
