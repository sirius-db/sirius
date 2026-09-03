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

// GPU-free, header-only: the decision rule for failing a query fast when an OOM retry cannot
// make progress. No CUDA, no cucascade; the executor feeds it plain numbers.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>

namespace sirius::pipeline {

/// Retry budget for a rescheduled GPU pipeline task (OOM or CUDA-launch reschedule).
///
/// Bumped from 10 to 100 as part of follow-up #17. SF100 Q11 with cache=table_gpu + num_gpus=2
/// exhausted the old 10-retry budget against cross-GPU BUILD_PROBE batch-lock contention: the
/// batch was held in `processing` on one GPU while the probe task on the other GPU needed it.
/// Each convert-release cycle is O(100ms) at SF100 scale, so 10 retries x 5ms backoff (50 ms
/// total) was far too short. With 100 retries x 50 ms backoff (~5 s) the probe tasks get enough
/// patience to clear the contention window while still bailing out on truly wedged queries.
/// Lives here so the futility reason can name the cap it short-circuits.
inline constexpr std::uint32_t kMaxTaskRetries = 100;

/// Context-wide progress signals consulted by the OOM fail-fast. One instance per task_scheduler,
/// shared by every gpu_pipeline_executor it owns; relaxed/acquire atomics, no locks.
struct execution_progress {
  /// +1 per task whose execute() returned normally, on any executor.
  std::atomic<std::uint64_t> completed_epoch{0};
  /// First-attempt tasks (retry_count == 0) currently inside execute(), on any executor.
  std::atomic<std::int64_t> inflight_first_attempts{0};
};

/// What the GPU executor's reservation gate saw for one attempt. Written by the manager thread
/// right before set_reservation(); read by the reschedule path after the attempt threw.
struct retry_gate_observation {
  std::size_t requested_bytes    = 0;  ///< bytes_needs after the space-max clamp
  std::size_t granted_bytes      = 0;  ///< reservation->size() handed to the task
  std::size_t freed_by_downgrade = 0;  ///< bytes the gate's downgrade request freed (0 if none ran)
  bool downgrade_requested  = false;   ///< the gate asked a downgrade executor (partial + executor)
  bool disk_tier_configured = false;   ///< downgrade_executor::has_disk_tier()
  std::size_t space_max_bytes = 0;     ///< memory_space::get_max_memory() (the reservation limit)
  std::uint64_t completed_epoch_at_gate = 0;  ///< execution_progress::completed_epoch read before
                                              ///< make_reservation()
};

struct retry_futility_input {
  bool is_oom               = false;  ///< the exception is an oom_reschedule_exception
  std::uint32_t retry_count = 0;      ///< of the task that just threw (0 = first attempt)
  std::size_t oom_required_bytes =
    0;  ///< live + requested recorded by the OOM handler; 0 = unknown
  std::optional<retry_gate_observation> gate;  ///< this attempt's gate observation
  std::uint64_t completed_epoch_now        = 0;
  std::int64_t inflight_first_attempts_now = 0;
};

/**
 * @brief Decide whether another OOM retry of this task can possibly be granted what it needs.
 *
 * Returns std::nullopt when another retry may succeed; otherwise the human-readable reason it
 * cannot (used verbatim in the query error). Pure: no I/O, no globals.
 *
 * The proof rests on two facts about cucascade's reservation gate. (A) A *partial* grant is
 * issued only when no other reservation is alive on the space, so `space_max - granted` is
 * exactly the number of bytes held outside any task reservation (parked fragment output, pinned
 * tables, idle batches). (B) A LIMIT_EXCEEDED OOM proves the attempt needed strictly more than
 * its reservation on its own stream, and the handler records that `live + requested` lower
 * bound. When the retry's gate was partial, the downgrade freed nothing, the recorded need
 * exceeds the grant, and nothing completed or was running anywhere that could release memory,
 * the next retry is granted at most the same and hits the same limit.
 */
inline std::optional<std::string> assess_retry_futility(const retry_futility_input& in)
{
  // 1. CUDA-launch reschedules keep the plain retry budget.
  if (!in.is_oom) { return std::nullopt; }
  // 2. A first attempt never fails fast; partial first attempts often succeed.
  if (in.retry_count == 0) { return std::nullopt; }
  // 3. Defensive: the gate did not record.
  if (!in.gate) { return std::nullopt; }
  auto const& gate = *in.gate;
  // 4. Full grant: the gate was not the constraint (the #732 batch-lock contention shape).
  if (gate.granted_bytes >= gate.requested_bytes) { return std::nullopt; }
  // 5. Spilling is making progress.
  if (gate.freed_by_downgrade != 0) { return std::nullopt; }
  // 6. Unknown need, or it would fit in the grant.
  if (in.oom_required_bytes == 0 || in.oom_required_bytes <= gate.granted_bytes) {
    return std::nullopt;
  }
  // 7. Something finished since the gate; its memory may now be convertible or free.
  if (in.completed_epoch_now != gate.completed_epoch_at_gate) { return std::nullopt; }
  // 8. A first attempt is running somewhere and may release memory.
  if (in.inflight_first_attempts_now != 0) { return std::nullopt; }

  // 9. Futile.
  constexpr double kGiB = 1073741824.0;
  auto const held       = gate.space_max_bytes > gate.granted_bytes
                            ? gate.space_max_bytes - gate.granted_bytes
                            : std::size_t{0};
  auto const held_pct   = gate.space_max_bytes > 0 ? 100.0 * static_cast<double>(held) /
                                                     static_cast<double>(gate.space_max_bytes)
                                                   : 0.0;
  std::string const downgrade_clause =
    gate.downgrade_requested
      ? std::format(
          "and were not convertible by the downgrade executor, which freed 0 bytes "
          "(disk tier {}configured)",
          gate.disk_tier_configured ? "" : "not ")
      : std::string("and no downgrade executor is attached to this memory space");
  return std::format(
    "the last attempt needed at least {} bytes ({:.2f} GiB) on its own stream but the "
    "reservation gate could grant only {} bytes ({:.2f} GiB) of the {} requested; {} bytes "
    "({:.1f}% of the {}-byte space) are held outside any task reservation {}; no task completed "
    "and no first attempt was running since this attempt's reservation was granted, so another "
    "retry would hit the same limit (retry cap {})",
    in.oom_required_bytes,
    static_cast<double>(in.oom_required_bytes) / kGiB,
    gate.granted_bytes,
    static_cast<double>(gate.granted_bytes) / kGiB,
    gate.requested_bytes,
    held,
    held_pct,
    gate.space_max_bytes,
    downgrade_clause,
    kMaxTaskRetries);
}

}  // namespace sirius::pipeline
