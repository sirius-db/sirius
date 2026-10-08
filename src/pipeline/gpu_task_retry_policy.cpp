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

#include "pipeline/gpu_task_retry_policy.hpp"

#include <algorithm>
#include <thread>

namespace sirius::pipeline {

memory_progress_signal& memory_progress_signal::global() noexcept
{
  static memory_progress_signal signal;
  return signal;
}

void memory_progress_signal::notify() noexcept
{
  {
    // Bumped under the mutex so a waiter that has just checked the epoch cannot miss it.
    std::lock_guard<std::mutex> lock(_mutex);
    _epoch.fetch_add(1, std::memory_order_acq_rel);
  }
  _cv.notify_all();
}

bool memory_progress_signal::wait_for_progress(std::uint64_t seen,
                                               std::chrono::milliseconds min_wait,
                                               std::chrono::milliseconds max_wait)
{
  auto const start = std::chrono::steady_clock::now();
  max_wait         = std::max(max_wait, min_wait);
  if (min_wait.count() > 0) { std::this_thread::sleep_for(min_wait); }
  std::unique_lock<std::mutex> lock(_mutex);
  return _cv.wait_until(lock, start + max_wait, [&] { return epoch() != seen; });
}

gpu_task_retry_decision on_gpu_task_retry(gpu_task_retry_state& state,
                                          const exec::gpu_task_retry_config& config,
                                          bool full_reservation,
                                          std::uint64_t epoch_now,
                                          bool engine_idle,
                                          std::chrono::steady_clock::time_point now) noexcept
{
  bool progressed = false;
  if (!state.initialized) {
    // The no-progress clock starts at the lineage's first failure.
    state.initialized        = true;
    state.last_seen_epoch    = epoch_now;
    state.last_progress_time = now;
  } else if (epoch_now != state.last_seen_epoch) {
    progressed                   = true;
    state.last_seen_epoch        = epoch_now;
    state.last_progress_time     = now;
    state.retries_since_progress = 0;
    state.idle_retries           = 0;
  }
  ++state.total_retries;
  ++state.retries_since_progress;
  if (full_reservation) { ++state.charged_retries; }
  // Idle and nothing progressed: no task is running and no downgrade is in flight, so nothing
  // can free memory for this task. A first failure counts too -- there was no previous retry to
  // compare against, and an idle engine is idle either way.
  if (engine_idle && !progressed) {
    ++state.idle_retries;
  } else {
    state.idle_retries = 0;
  }

  gpu_task_retry_decision decision;
  if (config.max_retries > 0 && state.charged_retries > config.max_retries) {
    decision.verdict = gpu_task_retry_verdict::fail_max_retries;
    return decision;
  }
  if (config.max_idle_retries > 0 && state.idle_retries > config.max_idle_retries) {
    decision.verdict = gpu_task_retry_verdict::fail_idle;
    return decision;
  }
  if (config.no_progress_timeout.count() > 0 &&
      now - state.last_progress_time > config.no_progress_timeout) {
    decision.verdict = gpu_task_retry_verdict::fail_no_progress;
    return decision;
  }

  // initial * 2^(retries_since_progress - 1), capped; the shift is bounded so it cannot overflow.
  auto const initial  = std::max(config.initial_backoff, std::chrono::milliseconds{0});
  auto const cap      = std::max(config.max_backoff, initial);
  auto const exponent = std::min<std::uint32_t>(state.retries_since_progress - 1, 20);
  auto backoff        = initial * (std::int64_t{1} << exponent);
  decision.min_wait   = initial;
  decision.max_wait   = std::min(backoff, cap);
  return decision;
}

}  // namespace sirius::pipeline
