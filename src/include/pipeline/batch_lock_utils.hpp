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

/**
 * @file batch_lock_utils.hpp
 * @brief RAII helpers for acquiring locks on cucascade::data_batch.
 *
 * Phase 18 / DB-01 migration: this header was rewritten for the post-#117
 * cucascade RAII model. The pre-#117 FSM-based locking API (handle types,
 * processing-state locks, in-transit transitions, blocking wait helpers,
 * and the task_created / processing / in_transit batch states) was removed
 * in cucascade pin 1c1e648. Locking is now expressed by holding
 * read_only_data_batch / mutable_data_batch accessor objects whose
 * destructors release the lock.
 *
 * The helpers below wrap the common Sirius patterns:
 *   - prepare_and_acquire_mutable: blocking exclusive lock + in-place
 *     conversion to a target memory space.
 *   - try_acquire_mutable: non-blocking variant.
 *   - acquire_read_only: shared lock, NO conversion (caller asserts the
 *     batch is already in the requested space).
 *
 * P1 lock-scope warning: every accessor returned by these helpers must be
 * scoped to the narrowest possible block. NEVER hold an accessor across a
 * function call that re-acquires on the same batch — the second acquisition
 * blocks on the first, and the first cannot release because it is still on
 * the call stack. For upgrade paths, use
 * cucascade::data_batch::readonly_to_mutable(std::move(ro)) instead of
 * acquiring both directly.
 */

#include "data/sirius_converter_registry.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace sirius {
namespace pipeline {

/**
 * @brief Acquire an exclusive (mutable) accessor on @p batch in the requested
 *        memory space, converting in-place if the batch lives elsewhere.
 *
 * Replaces the post-#117-removed lock_or_prepare_batch. Returns a
 * cucascade::mutable_data_batch by value — destruction releases the
 * exclusive lock; lifetime IS the lock guard.
 *
 * P1 lock-scope warning: caller MUST scope the returned accessor to the
 * narrowest block; NEVER hold the accessor across a call that re-acquires on
 * the same batch (would self-deadlock — second writer blocks on the first
 * exclusive lock; first writer is stuck on the call stack).
 *
 * Blocking semantics: this helper calls cucascade::data_batch::to_mutable(),
 * which blocks until the exclusive lock is acquired. Use try_acquire_mutable
 * for non-blocking variant.
 *
 * @param batch                   Batch to lock/prepare. nullptr -> nullopt.
 * @param requested_memory_space  Target memory space; nullptr -> use the
 *                                batch's current space (no conversion).
 * @param stream                  CUDA stream for any conversion kernels.
 *                                Pass the caller's actual operator stream;
 *                                never the cudaStream_t default (HYG-02).
 * @return cucascade::mutable_data_batch on success; std::nullopt on failure
 *         (batch null, no target tier resolvable, or unsupported tier).
 * @throws rmm::out_of_memory if a GPU memory allocation fails during
 *         conversion (RAII drop releases the exclusive lock on stack
 *         unwind).
 */
[[nodiscard]] inline std::optional<cucascade::mutable_data_batch> prepare_and_acquire_mutable(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  if (!batch) { return std::nullopt; }

  // Acquire exclusive lock first (single-step under #117 — no FSM).
  cucascade::mutable_data_batch acc = batch->to_mutable();

  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : acc.get_memory_space();
  if (target_space == nullptr) { return std::move(acc); }

  // Already in target space?
  if (acc.get_memory_space() == target_space) { return std::move(acc); }

  // Convert in-place. mutable_data_batch::convert_to handles stream sync
  // before destroying the source representation.
  auto& registry = sirius::converter_registry::get();
  switch (target_space->get_tier()) {
    case cucascade::memory::Tier::GPU:
      acc.convert_to<cucascade::gpu_table_representation>(registry, target_space, stream);
      break;
    case cucascade::memory::Tier::HOST:
      acc.convert_to<cucascade::host_data_representation>(registry, target_space, stream);
      break;
    default:
      // Disk tier not supported by this helper.
      return std::nullopt;
  }
  return std::move(acc);
}

/**
 * @brief Non-blocking variant of prepare_and_acquire_mutable.
 *
 * Returns std::nullopt without conversion if the exclusive lock is
 * unavailable. Otherwise behaves identically.
 *
 * @param batch                   Batch to lock/prepare. nullptr -> nullopt.
 * @param requested_memory_space  Target memory space; nullptr -> use the
 *                                batch's current space (no conversion).
 * @param stream                  CUDA stream for any conversion kernels.
 * @return cucascade::mutable_data_batch on success; std::nullopt if the
 *         exclusive lock could not be acquired immediately, or on the same
 *         failure conditions as prepare_and_acquire_mutable.
 */
[[nodiscard]] inline std::optional<cucascade::mutable_data_batch> try_acquire_mutable(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  if (!batch) { return std::nullopt; }

  auto try_acc = batch->try_to_mutable();
  if (!try_acc) { return std::nullopt; }

  cucascade::mutable_data_batch acc = std::move(*try_acc);

  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : acc.get_memory_space();
  if (target_space == nullptr) { return std::move(acc); }

  if (acc.get_memory_space() == target_space) { return std::move(acc); }

  auto& registry = sirius::converter_registry::get();
  switch (target_space->get_tier()) {
    case cucascade::memory::Tier::GPU:
      acc.convert_to<cucascade::gpu_table_representation>(registry, target_space, stream);
      break;
    case cucascade::memory::Tier::HOST:
      acc.convert_to<cucascade::host_data_representation>(registry, target_space, stream);
      break;
    default:
      return std::nullopt;
  }
  return std::move(acc);
}

/**
 * @brief Acquire a read-only (shared) accessor on @p batch.
 *
 * ASSUMES the batch is already in @p requested_memory_space — does NOT
 * convert. Callers that need conversion must use prepare_and_acquire_mutable
 * first (or use cucascade::data_batch::readonly_to_mutable to upgrade).
 *
 * Polling note: this is non-blocking from the caller's perspective in the
 * sense that no FSM transition gates the read — concurrent readers are
 * permitted. Internally cucascade::data_batch::to_read_only() blocks only on
 * an outstanding exclusive lock. Callers that must avoid that block should
 * poll via cucascade::data_batch::try_to_read_only() directly.
 * See §Recipe R6 in 18-RESEARCH.md for the post-#117 polling discipline.
 *
 * @param batch                   Batch to read-lock. nullptr -> nullopt.
 * @param requested_memory_space  Memory space the caller asserts the batch
 *                                lives in. If non-null and the batch lives
 *                                elsewhere, returns std::nullopt (caller
 *                                must use prepare_and_acquire_mutable for
 *                                conversion). nullptr disables the check.
 * @return cucascade::read_only_data_batch on success; std::nullopt on
 *         failure (batch null, or memory_space mismatch).
 */
[[nodiscard]] inline std::optional<cucascade::read_only_data_batch> acquire_read_only(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space)
{
  if (!batch) { return std::nullopt; }

  cucascade::read_only_data_batch ro = batch->to_read_only();

  if (requested_memory_space != nullptr && ro.get_memory_space() != requested_memory_space) {
    // Caller must use prepare_and_acquire_mutable for conversion — this
    // helper does not convert (see P1 lock-scope warning above).
    return std::nullopt;
  }
  return std::move(ro);
}

}  // namespace pipeline
}  // namespace sirius
