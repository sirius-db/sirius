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

#include <data/data_batch_utils.hpp>
#include <data/data_repository.hpp>
#include <data/data_repository_manager.hpp>

namespace sirius {
namespace pipeline {

gpu_pipeline_task::gpu_pipeline_task(
  uint64_t task_id,
  std::vector<cucascade::shared_data_repository*> data_repos,
  std::unique_ptr<sirius::parallel::itask_local_state> local_state,
  std::shared_ptr<sirius::parallel::itask_global_state> global_state)
  : itask(std::move(local_state), std::move(global_state)),
    _task_id(task_id),
    _data_repos(std::move(data_repos))
{
}

uint64_t gpu_pipeline_task::get_task_id() const { return _task_id; }

const duckdb::GPUPipeline* gpu_pipeline_task::get_pipeline() const
{
  return _global_state->cast<gpu_pipeline_task_global_state>()._pipeline.get();
}

void gpu_pipeline_task::execute()
{
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();

  // Acquire processing handles for all input batches.
  // This prevents the batches from being downgraded while we're processing them.
  // The handles will automatically release when they go out of scope.
  auto processing_handles = sirius::acquire_processing_handles(local_state._batches);
  if (!processing_handles) {
    // Some batch is being downgraded - cannot process right now
    // TODO: Add retry logic or re-queue the task
    return;
  }

  // At this point, all input batches are locked for processing.
  // They will remain locked until the processing_handles go out of scope.

  // TODO: Implement actual pipeline execution:
  // 1. Transfer data batch to GPU memory if not already there
  // 2. Set reservation_aware_memory_resource_ref as the default cudf allocator
  // 3. Execute cudf operators on the pipeline
  // 4. After each cudf operator, get peak total bytes to collect statistics
  // 5. Push output batches to the data repository

  // Processing handles are automatically released here when they go out of scope
}

}  // namespace pipeline
}  // namespace sirius
