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

#include <data/cpu_data_representation.hpp>
#include <data/data_batch_utils.hpp>
#include <data/data_repository.hpp>
#include <data/data_repository_manager.hpp>
#include <data/gpu_data_representation.hpp>
#include <data/sirius_converter_registry.hpp>
#include <memory/memory_space.hpp>

#include <optional>

namespace sirius {
namespace pipeline {

namespace {

std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space)
{
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : batch->get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  auto lock_result = batch->try_to_lock_for_processing(target_space->get_id());

  auto cancel_task_if_needed = []() {};

  const bool needs_conversion =
    requested_memory_space != nullptr &&
    lock_result.status == cucascade::lock_for_processing_status::memory_space_mismatch;

  if (!lock_result.success && needs_conversion) {
    try {
      auto& registry = sirius::converter_registry::get();
      auto stream    = requested_memory_space->acquire_stream();
      switch (requested_memory_space->get_tier()) {
        case cucascade::memory::Tier::GPU: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            cancel_task_if_needed();
            return std::nullopt;
          }
          batch->convert_to<cucascade::gpu_table_representation>(
            registry, requested_memory_space, stream);
          batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
          break;
        }
        case cucascade::memory::Tier::HOST: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            cancel_task_if_needed();
            return std::nullopt;
          }
          batch->convert_to<cucascade::host_table_representation>(
            registry, requested_memory_space, stream);
          batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
          break;
        }
        default: cancel_task_if_needed(); return std::nullopt;
      }

      lock_result = batch->try_to_lock_for_processing(requested_memory_space->get_id());
    } catch (...) {
      cancel_task_if_needed();
      throw;
    }
  }

  if (!lock_result.success) {
    cancel_task_if_needed();
    return std::nullopt;
  }

  return std::move(lock_result.handle);
}

}  // namespace

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

const sirius_pipeline* gpu_pipeline_task::get_pipeline() const
{
  return _global_state->cast<gpu_pipeline_task_global_state>()._pipeline.get();
}

void gpu_pipeline_task::execute()
{
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();

  const auto* reservation = local_state.get_reservation();
  const auto* requested_memory_space =
    reservation != nullptr ? &reservation->get_memory_space() : nullptr;
  std::vector<cucascade::data_batch_processing_handle> processing_handles;
  processing_handles.reserve(local_state._batches.size());

  for (const auto& batch : local_state._batches) {
    auto handle = lock_or_prepare_batch(batch, requested_memory_space);
    if (!handle) {
      // Failed to lock (or convert) one of the batches. Caller can retry later.
      return;
    }
    processing_handles.emplace_back(std::move(*handle));
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
