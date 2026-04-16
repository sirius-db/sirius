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

#include "data/convertible_data_batch.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace pipeline {

/**
 * @brief Lock or prepare a single data batch for processing in the requested memory space.
 *
 * If the batch is already in the requested memory space it is locked in place. If it resides
 * elsewhere the batch is first converted (moved) to the requested space via
 * convertible_data_batch::convert() and then locked.
 *
 * @param batch                   The batch to lock/prepare.
 * @param requested_memory_space  Target memory space; may be nullptr to use the batch's current
 *                                space.
 * @param stream                  CUDA stream used for any data-movement kernels.
 * @param res_mgr                 Reservation manager for polite reservation checks during
 *                                conversion.
 * @return A processing handle that keeps the batch locked, or std::nullopt on failure.
 * @throws rmm::out_of_memory  If a GPU memory allocation fails during the conversion.
 */
inline std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream,
  sirius::memory::sirius_memory_reservation_manager& res_mgr)
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
    // Delegate tier-switching conversion to convertible_data_batch::convert().
    // This unifies the forward-path conversion with the downgrade path, ensuring
    // both benefit from the same failure-safety guarantees (state restore on error).
    sirius::convertible_data_batch convertible(batch);
    auto* mutable_space = const_cast<cucascade::memory::memory_space*>(target_space);
    bool converted =
      convertible.convert(std::vector<cucascade::memory::memory_space*>{mutable_space},
                          stream,
                          res_mgr);

    if (!converted) {
      // convert() returns false if another thread holds the in_transit lock or no
      // reservation is available. Re-attempt wait_to_lock_for_processing which
      // blocks until the batch is available again, matching the original contention
      // handling.
      lock_result = batch->wait_to_lock_for_processing(target_space->get_id());
      continue;
    }

    lock_result = batch->try_to_lock_for_processing(target_space->get_id());
  }

  if (!lock_result.success) {
    cancel_task_if_needed();
    return std::nullopt;
  }

  return std::move(lock_result.handle);
}

}  // namespace pipeline
}  // namespace sirius
