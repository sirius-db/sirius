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

#include "exec/config.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace sirius::pipeline {

/**
 * @brief Process-wide "memory may have been released" event.
 *
 * Posted when a GPU pipeline task completes (its reservation and inputs are released) and
 * when a downgrade frees bytes. A rescheduled task waits on it instead of sleeping a fixed
 * interval, and its epoch is the engine's progress clock: a retrying task that sees the epoch
 * unchanged since its previous retry knows that nothing completed and nothing was freed.
 */
class memory_progress_signal {
 public:
  /// The signal shared by the GPU pipeline executors and downgrade executors.
  static memory_progress_signal& global() noexcept;

  /// Record progress and wake every waiter.
  void notify() noexcept;

  [[nodiscard]] std::uint64_t epoch() const noexcept
  {
    return _epoch.load(std::memory_order_acquire);
  }

  /**
   * @brief Wait for the epoch to move past @p seen.
   *
   * Never returns before @p min_wait (so a stream of events cannot turn retries into a busy
   * loop), and returns at @p max_wait even without an event.
   *
   * @return true when the epoch moved past @p seen.
   */
  bool wait_for_progress(std::uint64_t seen,
                         std::chrono::milliseconds min_wait,
                         std::chrono::milliseconds max_wait);

  /// Marks work in flight that may still free memory: a GPU task executing or a downgrade
  /// request being processed. While none is, a retry has nothing to wait for.
  class activity_scope {
   public:
    explicit activity_scope(memory_progress_signal& signal) noexcept : _signal(&signal)
    {
      _signal->_active.fetch_add(1, std::memory_order_acq_rel);
    }
    ~activity_scope() { _signal->_active.fetch_sub(1, std::memory_order_acq_rel); }
    activity_scope(const activity_scope&)            = delete;
    activity_scope& operator=(const activity_scope&) = delete;

   private:
    memory_progress_signal* _signal;
  };

  [[nodiscard]] activity_scope track_activity() noexcept { return activity_scope{*this}; }

  /// True when no GPU task is executing and no downgrade request is being processed.
  [[nodiscard]] bool idle() const noexcept { return _active.load(std::memory_order_acquire) == 0; }

 private:
  std::atomic<std::uint64_t> _epoch{0};
  std::atomic<std::int64_t> _active{0};
  std::mutex _mutex;
  std::condition_variable _cv;
};

/**
 * @brief Retry bookkeeping for one task lineage (the original task and its reschedules).
 *
 * Carried from each rescheduled task's local state to the next.
 */
struct gpu_task_retry_state {
  /// Every reschedule, for logging.
  std::uint32_t total_retries = 0;
  /// Reschedules whose failed attempt had its full reservation (counted against max_retries).
  std::uint32_t charged_retries = 0;
  /// Consecutive reschedules without observed progress; drives the backoff.
  std::uint32_t retries_since_progress = 0;
  /// Consecutive reschedules without progress whose failure found the engine idle.
  std::uint32_t idle_retries = 0;
  /// Progress epoch observed at the previous reschedule.
  std::uint64_t last_seen_epoch = 0;
  /// When this lineage last observed progress (or first failed).
  std::chrono::steady_clock::time_point last_progress_time{};
  bool initialized = false;
};

enum class gpu_task_retry_verdict { retry, fail_max_retries, fail_no_progress, fail_idle };

struct gpu_task_retry_decision {
  gpu_task_retry_verdict verdict = gpu_task_retry_verdict::retry;
  /// Earliest the retry may proceed.
  std::chrono::milliseconds min_wait{0};
  /// Latest the retry waits for a memory-release event.
  std::chrono::milliseconds max_wait{0};
};

/**
 * @brief Account one failed attempt and decide whether and how long to wait before retrying.
 *
 * @param state            The lineage's bookkeeping (updated in place).
 * @param config           Retry policy.
 * @param full_reservation Whether the failed attempt ran with its full reservation.
 * @param epoch_now        memory_progress_signal epoch at the failure.
 * @param engine_idle      memory_progress_signal::idle() at the failure (after this attempt
 *                         stopped executing).
 * @param now              Current time.
 */
gpu_task_retry_decision on_gpu_task_retry(gpu_task_retry_state& state,
                                          const exec::gpu_task_retry_config& config,
                                          bool full_reservation,
                                          std::uint64_t epoch_now,
                                          bool engine_idle,
                                          std::chrono::steady_clock::time_point now) noexcept;

}  // namespace sirius::pipeline
