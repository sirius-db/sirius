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

#pragma once

#include "log/logging.hpp"

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/sirius_converter_registry.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>

namespace sirius {
namespace pipeline {

/**
 * @brief Lock or prepare a single data batch for processing in the requested memory space.
 *
 * If the batch is already in the requested memory space it is locked in place. If it resides
 * elsewhere the batch is first converted (moved) to the requested space and then locked.
 *
 * @param batch                   The batch to lock/prepare.
 * @param requested_memory_space  Target memory space; may be nullptr to use the batch's current
 *                                space.
 * @param stream                  CUDA stream used for any data-movement kernels.
 * @return A processing handle that keeps the batch locked, or std::nullopt on failure.
 * @throws rmm::out_of_memory  If a GPU memory allocation fails during the conversion.
 */
inline std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  using status = cucascade::lock_for_processing_status;
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : batch->get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  // NOTE: only works in single gpu setup
  // wait for processing in case a shared batch is in transit in another thread.
  auto lock_result = batch->wait_to_lock_for_processing(target_space->get_id());

  auto cancel_task_if_needed = []() {
    SIRIUS_LOG_ERROR(
      "gpu_pipeline_task: failed to lock batch for processing and cannot prepare batch for "
      "processing. This likely means the batch is in transit and there is a bug in "
      "the in-transit locking logic. Cancelling task to avoid deadlock.");
  };

  while (!lock_result.success && lock_result.status == status::memory_space_mismatch) {
    try {
      auto& registry = sirius::converter_registry::get();
      switch (requested_memory_space->get_tier()) {
        case cucascade::memory::Tier::GPU: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            auto current_state = batch->get_state();
            if (current_state == cucascade::batch_state::in_transit) {
              // If another thread has taken the in_transit lock, wait to acquire the processing
              // lock.
              lock_result = batch->wait_to_lock_for_processing(target_space->get_id());
              continue;
            }
            cancel_task_if_needed();
            return std::nullopt;
          }
          try {
            batch->convert_to<cucascade::gpu_table_representation>(
              registry, requested_memory_space, stream);
          } catch (...) {
            batch->try_to_release_in_transit();
            throw;
          }
          batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
          break;
        }
        case cucascade::memory::Tier::HOST: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            cancel_task_if_needed();
            return std::nullopt;
          }
          try {
            batch->convert_to<cucascade::host_data_representation>(
              registry, requested_memory_space, stream);
          } catch (...) {
            batch->try_to_release_in_transit();
            throw;
          }
          batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
          break;
        }
        default: cancel_task_if_needed(); return std::nullopt;
      }

      lock_result = batch->try_to_lock_for_processing(requested_memory_space->get_id());
    } catch (...) {
      throw;
    }
  }

  if (!lock_result.success) {
    cancel_task_if_needed();
    return std::nullopt;
  }

  return std::move(lock_result.handle);
}

}  // namespace pipeline
}  // namespace sirius
