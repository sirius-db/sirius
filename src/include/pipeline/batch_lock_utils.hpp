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

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/sirius_converter_registry.hpp>

#include <memory>
#include <optional>

namespace sirius {
namespace pipeline {

/**
 * @brief Lock or prepare a single data batch for processing in the requested memory space.
 *
 * Acquires a read-only (shared) lock on the batch. If the batch resides in a different memory
 * space than requested, it is first upgraded to a mutable (exclusive) lock, converted in-place
 * to the target representation, then downgraded back to a read-only lock.
 *
 * @param batch                   The batch to lock/prepare.
 * @param requested_memory_space  Target memory space; may be nullptr to use the batch's current
 *                                space.
 * @param stream                  CUDA stream used for any data-movement kernels.
 * @return A read_only_data_batch holding the shared lock, or std::nullopt on failure.
 * @throws rmm::out_of_memory  If a GPU memory allocation fails during the conversion.
 */
inline std::optional<cucascade::read_only_data_batch> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  if (!batch) { return std::nullopt; }

  // Opportunistic stream rebind (best-effort, non-blocking): if the batch already resides in
  // the target GPU memory space, rebind its deallocation stream to this task's stream so that
  // when the task later frees the data the free lands on the active stream's free list rather
  // than on the stream the data was produced on. We use try_get_mutable() so we never block --
  // if another reader holds the batch (e.g. a probe task on another GPU sharing a build batch)
  // or it is otherwise busy, we simply skip the rebind; correctness is unaffected. Gating on a
  // space match guarantees the stream and the data live on the same device (important for
  // multi-GPU). The mismatch case below converts via `stream`, which already allocates the new
  // table on it, so no rebind is needed there.
  if (auto mut = batch->try_get_mutable()) {
    const auto* current_space = (*mut)->get_memory_space();
    const auto* rebind_target =
      requested_memory_space != nullptr ? requested_memory_space : current_space;
    if (current_space != nullptr && rebind_target != nullptr &&
        current_space->get_id() == rebind_target->get_id() &&
        rebind_target->get_tier() == cucascade::memory::Tier::GPU) {
      (*mut)->rebind_stream(stream);
    }
  }

  // Acquire a read-only lock
  auto read_accessor = batch->get_read_only();

  // Determine the target memory space
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : read_accessor->get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  // Memory space matches — return the read-only accessor directly
  if (read_accessor->get_memory_space() != nullptr &&
      read_accessor->get_memory_space()->get_id() == target_space->get_id()) {
    return std::move(read_accessor);
  }

  // Memory space mismatch: release the shared lock, take the exclusive lock,
  // convert in-place, then return a fresh read-only accessor.
  cucascade::read_only_data_batch::to_idle(std::move(read_accessor));
  auto& registry    = sirius::converter_registry::get();
  auto mut_accessor = batch->get_mutable();

  if (mut_accessor->get_memory_space() != nullptr &&
      mut_accessor->get_memory_space()->get_id() == target_space->get_id()) {
    cucascade::mutable_data_batch::to_idle(std::move(mut_accessor));
    return batch->get_read_only();
  }

  switch (target_space->get_tier()) {
    case cucascade::memory::Tier::GPU:
      mut_accessor->convert_to<cucascade::gpu_table_representation>(registry, target_space, stream);
      break;
    case cucascade::memory::Tier::HOST:
      mut_accessor->convert_to<cucascade::host_data_representation>(registry, target_space, stream);
      break;
    default:
      SIRIUS_LOG_ERROR("lock_or_prepare_batch: unsupported target tier for batch {}",
                       mut_accessor->get_batch_id());
      return std::nullopt;
  }

  cucascade::mutable_data_batch::to_idle(std::move(mut_accessor));
  return batch->get_read_only();
}

}  // namespace pipeline
}  // namespace sirius
