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
#include "telemetry/batch_telemetry.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>

#include <memory>
#include <optional>

namespace sirius {
namespace pipeline {

struct lock_and_prepare_batch_result {
  // A new batch that may have been created to appropriately prepare the batch.
  // If set, the ro_lock is a lock on this new batch which may be stored for
  // future reference.
  std::optional<std::shared_ptr<cucascade::data_batch>> new_batch;
  cucascade::read_only_data_batch ro_lock;
};

/**
 * @brief Lock or prepare a single data batch for processing in the requested memory space.
 *
 * Acquires a read-only (shared) lock on the batch. On a memory-space mismatch the behavior
 * depends on where the data currently lives:
 *
 * - GPU -> GPU (cross-device): the batch is cloned into the target space via
 *   read_only_data_batch::clone_to while only the shared lock is held. The source is never
 *   exclusively locked or mutated — concurrent readers (e.g. probe tasks on another GPU sharing
 *   a build batch) proceed during the transfer — and it stays resident in its original space for
 *   consumers local to that device. The returned accessor references the NEW (cloned) batch.
 *   Source lifetime is ownership-driven: whoever still needs the original (repositories, other
 *   tasks) holds a reference to it; once the last reference drops it is freed. The source is
 *   back in the idle state as soon as this function returns, so it is downgrade-eligible.
 *
 * - HOST/DISK -> GPU (upgrade/readback): the batch is upgraded to a mutable (exclusive) lock and
 *   converted in-place, freeing the spilled copy — intentional move semantics for the common
 *   single-consumer case. The returned accessor references the same batch. If a concurrent
 *   consumer converted the batch during the (non-atomic) lock upgrade, the state observed under
 *   the exclusive lock is re-dispatched: same space -> no-op; a different GPU -> clone (a
 *   GPU-resident batch is never moved cross-device); still host/disk -> move as planned.
 *
 * @param batch                   The batch to lock/prepare.
 * @param requested_memory_space  Target memory space; may be nullptr to use the batch's current
 *                                space.
 * @param stream                  CUDA stream used for any data-movement kernels.
 * @return A read_only_data_batch holding the shared lock (on the clone for cross-GPU inputs),
 *         or std::nullopt on failure.
 * @throws rmm::out_of_memory  If a GPU memory allocation fails during the conversion or clone.
 */
inline std::optional<lock_and_prepare_batch_result> lock_and_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  if (!batch) { return std::nullopt; }

  // Opportunistic stream rebind (best-effort, non-blocking): if the batch already resides in
  // the target GPU memory space, rebind its deallocation stream to this task's stream so that
  // when the task later frees the data the free lands on the active stream's free list rather
  // than on the stream the data was produced on. We use try_to_mutable() so we never block --
  // if another reader holds the batch (e.g. a probe task on another GPU sharing a build batch)
  // or it is otherwise busy, we simply skip the rebind; correctness is unaffected. Gating on a
  // space match guarantees the stream and the data live on the same device (important for
  // multi-GPU). The mismatch case below converts via `stream`, which already allocates the new
  // table on it, so no rebind is needed there.
  if (std::optional<cucascade::mutable_data_batch> mut = batch->try_to_mutable()) {
    const auto* current_space = mut->get_memory_space();
    const auto* rebind_target =
      requested_memory_space != nullptr ? requested_memory_space : current_space;
    if (current_space != nullptr && rebind_target != nullptr &&
        current_space->get_id() == rebind_target->get_id() &&
        rebind_target->get_tier() == cucascade::memory::Tier::GPU) {
      mut->rebind_stream(stream);
    }
  }

  // Acquire a read-only lock
  cucascade::read_only_data_batch read_accessor = batch->to_read_only();

  // Determine the target memory space
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : read_accessor.get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  // Memory space matches — return the read-only accessor directly
  if (read_accessor.get_memory_space() != nullptr &&
      read_accessor.get_memory_space()->get_id() == target_space->get_id()) {
    return lock_and_prepare_batch_result{
      .new_batch = std::nullopt,
      .ro_lock   = std::move(read_accessor),
    };
  }

  // Memory space mismatch — clone or move depending on where the data lives.
  auto& registry = sirius::converter_registry::get();

  switch (target_space->get_tier()) {
    case cucascade::memory::Tier::GPU: {
      if (read_accessor.get_current_tier() == cucascade::memory::Tier::GPU) {
        // Cross-GPU: clone into the target space under the shared lock. The source is never
        // exclusively locked or mutated. clone_to routes through convert_gpu_to_gpu, which waits
        // on the source's writer event and synchronizes its copy stream before returning, so
        // holding read_accessor across the call is sufficient ordering and the copy is complete
        // when it returns. No normalize_gpu_list_offsets here: that fixup only reverses the
        // INT64 offsets promotion of host->GPU reconstruction, and a GPU->GPU copy preserves the
        // source's column types (a GPU-resident source is already normalized — the same-space
        // fast path above depends on that invariant).
        // TODO(dhruv9vats): thread operator batch telemetry to cloned batch
        auto clone = read_accessor.clone_to<cucascade::gpu_table_representation>(
          registry, get_next_batch_id(), target_space, stream);
        return lock_and_prepare_batch_result{
          .new_batch = clone,
          .ro_lock   = clone->to_read_only(),
        };
        // read_accessor dropped.
      }

      // HOST/DISK -> GPU upgrade/readback: convert in place (move), freeing the spilled copy
      {
        cucascade::data_batch::to_idle(std::move(read_accessor));  // release the read lock
        cucascade::mutable_data_batch mut_accessor = batch->to_mutable();

        // this upgrade to mutable is not atomic (shared released, then exclusive acquired): a
        // concurrent consumer of a shared spilled batch may have converted it in the gap.
        // Re-dispatch on the state observed under the exclusive lock:
        const auto* current_space = mut_accessor.get_memory_space();
        if (current_space != nullptr && current_space->get_id() == target_space->get_id()) {
          // Already in the target space — skip the wasteful same-space deep copy.
          cucascade::data_batch::to_idle(
            std::move(mut_accessor));  // release the exclusive write lock
          return lock_and_prepare_batch_result{
            .new_batch = std::nullopt,
            .ro_lock   = batch->to_read_only(),
          };
        }

        if (mut_accessor.get_current_tier() == cucascade::memory::Tier::GPU) {
          // Upgraded to a DIFFERENT GPU in the gap: never move a GPU-resident batch cross-device
          // (that would steal it from the device that just materialized it) — clone instead. The
          // clone happens under the exclusive lock we already hold; pinning the state here avoids
          // yet another non-atomic lock transition.
          // TODO(dhruv9vats): thread operator batch telemetry to cloned batch
          auto clone = mut_accessor.clone_to<cucascade::gpu_table_representation>(
            registry, sirius::get_next_batch_id(), target_space, stream);
          return lock_and_prepare_batch_result{
            .new_batch = clone,
            .ro_lock   = clone->to_read_only(),
          };
        }
        mut_accessor.convert_to<cucascade::gpu_table_representation>(
          registry, target_space, stream);
        telemetry::batch_telemetry_registry::instance().on_tier_change(
          mut_accessor.get_batch_id(),
          target_space->get_tier(),
          target_space->get_id().device_id,
          mut_accessor.get_data()->get_size_in_bytes());

        // exclusive mutable access dropped.
      }

      return lock_and_prepare_batch_result{
        .new_batch = std::nullopt,
        .ro_lock   = batch->to_read_only(),
      };
    }
    case cucascade::memory::Tier::HOST: {
      {
        // release the read lock to get mutable access
        cucascade::data_batch::to_idle(std::move(read_accessor));
        cucascade::mutable_data_batch mut_accessor = batch->to_mutable();
        // Same non-atomic-upgrade race as the GPU arm: skip if already in the target space.
        if (mut_accessor.get_memory_space() == nullptr ||
            mut_accessor.get_memory_space()->get_id() != target_space->get_id()) {
          auto host_reservation =
            const_cast<cucascade::memory::memory_space*>(target_space)
              ->make_reservation_or_null(mut_accessor.get_data()->get_size_in_bytes());
          if (host_reservation != nullptr) {
            mut_accessor.convert_to<cucascade::host_data_representation>(
              registry, *host_reservation, stream);
          } else {
            SIRIUS_LOG_WARN(
              "lock_or_prepare_batch: host reservation failed for batch {} ({} bytes) — "
              "proceeding without reservation, converter may OOM",
              mut_accessor.get_batch_id(),
              mut_accessor.get_data()->get_size_in_bytes());
            mut_accessor.convert_to<cucascade::host_data_representation>(
              registry, target_space, stream);
          }
          telemetry::batch_telemetry_registry::instance().on_tier_change(
            mut_accessor.get_batch_id(),
            target_space->get_tier(),
            target_space->get_id().device_id,
            mut_accessor.get_data()->get_size_in_bytes());
        }
        // exclusive mutable access dropped.
      }
      return lock_and_prepare_batch_result{
        .new_batch = std::nullopt,
        .ro_lock   = batch->to_read_only(),
      };
    }
    default:
      SIRIUS_LOG_ERROR("lock_or_prepare_batch: unsupported target tier for batch {}",
                       read_accessor.get_batch_id());
      return std::nullopt;
  }
}

}  // namespace pipeline
}  // namespace sirius
