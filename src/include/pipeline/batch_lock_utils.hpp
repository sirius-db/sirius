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
 *   - acquire_read_only: shared lock, NO conversion (caller asserts the
 *     batch is already in the requested space).
 *
 * The previous Phase 18-01 version exported a non-blocking try_acquire_mutable
 * helper alongside the blocking one. It had zero production callers and was
 * removed in 18-07 — its only function in the original design was to support
 * the R5 lock-and-hold pattern that 18-VERIFICATION.md confirmed deadlocks
 * under glibc EDEADLK. Restore it only if a future plan finds a non-lock-and-
 * hold use case.
 *
 * P1 lock-scope warning (Path A — Phase 18-07 gap closure):
 * Every accessor returned by these helpers must be scoped to the narrowest
 * possible block. NEVER hold an accessor across a function call that
 * re-acquires on the same batch — glibc std::shared_mutex detects same-thread
 * re-lock attempts and aborts with "Resource deadlock avoided" (POSIX
 * EDEADLK), not just blocks. The Phase 18-02 R5 lock-and-hold design that
 * held a vector<mutable_data_batch> across op->execute() was reverted in
 * 18-07 after this exact failure mode fired on every [mgpu] test (see
 * .planning/phases/18-databatch-raii-migration-cucascade-117-surface/
 * 18-VERIFICATION.md for the gap analysis and 18-07-SUMMARY.md for the
 * closure record). For upgrade paths, use
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
 * Path A (Phase 18-07 gap closure): callers MUST scope the returned accessor
 * to a narrow `{}` block. The previous Phase 18-02 R5 lock-and-hold pattern
 * (vector<mutable_data_batch> held across op->execute()) was confirmed to
 * deadlock under glibc EDEADLK on shared_mutex re-lock and was reverted in
 * 18-07. The single production caller is now
 * pipelineable_operator_data::prepare_for_processing, which iterates batches
 * and acquires/drops this accessor under a `{}` block per iteration so the
 * exclusive lock is released BEFORE the function returns.
 *
 * Blocking semantics: this helper calls cucascade::data_batch::to_mutable(),
 * which blocks until the exclusive lock is acquired. Restore a non-blocking
 * try_-prefixed variant only if a Path-A-compatible use case appears (see
 * file-level doc block).
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

// Phase 18-07 (Path A): try_acquire_mutable removed. The non-blocking variant
// of prepare_and_acquire_mutable had ZERO production callers and existed only
// to support patterns conceptually adjacent to the now-reverted R5 lock-and-
// hold design. Callers that need a non-blocking exclusive accessor today
// should call cucascade::data_batch::try_to_mutable() directly and perform
// any required memory-space conversion inline. Restore this helper if a
// future plan finds a Path-A-compatible use case (i.e. one that scopes the
// returned accessor to a narrow `{}` block — see file-level doc).

/**
 * @brief Acquire a read-only (shared) accessor on @p batch.
 *
 * Path A (Phase 18-07): scope the returned accessor to a narrow `{}` block
 * — never hold across a downstream call that re-acquires on the same batch.
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
