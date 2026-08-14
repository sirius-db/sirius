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

#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_request.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::parallel {
class downgrade_executor;
}  // namespace sirius::parallel

namespace sirius::telemetry {
class telemetry_context;
}  // namespace sirius::telemetry

namespace sirius {

namespace creator {
class task_creator;
}

namespace pipeline {

struct executor_metrics {
  size_t tasks_executed{0};
};

/**
 * @brief Storm-observability counters for the OOM-retry pacing path.
 *
 * All monotonically increasing over the executor's lifetime; readable from any
 * thread. See gpu_pipeline_executor::get_oom_pacing_stats().
 */
struct oom_pacing_stats {
  /// Total OOM/contention reschedules (every task_reschedule_exception retried).
  size_t oom_reschedules{0};
  /// Admissions that went through a no-progress downgrade (0 bytes freed) and
  /// proceeded on a partial reservation — the storm signature.
  size_t starved_admissions{0};
  /// Reschedules whose backoff was escalated beyond the base interval.
  size_t backoff_events{0};
  /// Total milliseconds actually spent in reschedule backoff sleeps.
  size_t backoff_ms{0};
};

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and manages a pool of threads
 * dedicated to executing GPU pipeline tasks with specialized GPU resource
 * management.
 */
class gpu_pipeline_executor : public sirius::parallel::itask_executor {
 public:
  /**
   * @brief Constructs a new gpu_pipeline_executor with task execution configuration
   *
   * @param config Configuration for the task executor (thread count, retry policy, etc.)
   * @param mem_space Pointer to the memory space for GPU allocations
   * @param task_request_publisher Publisher to submit task requests
   * @param downgrade_executor Pointer to the downgrade executor. This is used so that the
   * gpu_pipeline_executor can request memory downgrade if it cannot obtain a reservation from the
   * memory space.
   */
  explicit gpu_pipeline_executor(
    exec::thread_pool_config config,
    cucascade::memory::memory_space* mem_space,
    exec::publisher<std::unique_ptr<task_request>> task_request_publisher,
    sirius::parallel::downgrade_executor* downgrade_executor,
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context);

  /**
   * @brief Destructor for the gpu_pipeline_executor.
   */
  ~gpu_pipeline_executor();

  // Non-copyable but movable
  gpu_pipeline_executor(const gpu_pipeline_executor&)            = delete;
  gpu_pipeline_executor& operator=(const gpu_pipeline_executor&) = delete;
  gpu_pipeline_executor(gpu_pipeline_executor&&)                 = delete;
  gpu_pipeline_executor& operator=(gpu_pipeline_executor&&)      = delete;

  /**
   * @brief Set the task creator for scheduling output consumers
   *
   * @param task_creator Pointer to the task creator
   */
  void set_task_creator(creator::task_creator* task_creator);

  /**
   * @brief Check if the internal task queue is empty.
   *
   * Useful for verifying that drain_and_wait() has fully cleared the queue.
   * Only reliable when the executor is quiescent (no concurrent producers).
   *
   * @return true if the task queue contains no pending tasks.
   */
  [[nodiscard]] bool is_task_queue_empty() const noexcept;

  /**
   * @brief Return a snapshot of this executor's runtime metrics.
   */
  [[nodiscard]] executor_metrics get_metrics() const noexcept;

  /**
   * @brief Snapshot of the OOM-retry pacing counters (storm observability).
   */
  [[nodiscard]] oom_pacing_stats get_oom_pacing_stats() const noexcept;

  /**
   * @brief Backoff before re-admitting an OOM-rescheduled task.
   *
   * @p starved_streak counts consecutive attempts that OOM'd after a "starved"
   * admission (no-progress downgrade + partial reservation; see
   * gpu_pipeline_task_local_state::starved_admission). A streak of 0 — the
   * clean-admission OOM or cross-GPU batch-contention case (follow-up #17) —
   * keeps the historical base interval so transient contention still clears
   * fast. Under starvation only other work freeing memory can help, so the
   * interval doubles per consecutive failure up to a cap: 50, 100, 200, 400,
   * 800, 800... ms. This collapses the retry-storm duty cycle (observed ~11
   * attempts/s/task, each dragging a futile downgrade pass) roughly 10x and
   * stretches the MAX_RETRIES budget from ~9 s of pressure to >70 s, so
   * transient multi-second memory cliffs no longer terminate queries at the
   * retry cap. The sleep itself polls a wake source (memory release / query
   * error / executor stop) — see the reschedule path.
   */
  [[nodiscard]] static std::chrono::milliseconds compute_oom_backoff(
    uint32_t starved_streak) noexcept
  {
    constexpr uint32_t kMaxShift = 4;  // cap at 50 << 4 = 800 ms
    return std::chrono::milliseconds{50u << std::min(starved_streak, kMaxShift)};
  }

  /**
   * @brief Set the completion handler for query completion signaling
   *
   * @param handler Pointer to the completion handler
   */
  void set_completion_handler(completion_handler* handler) noexcept;

 protected:
  void manager_loop() override;

  absl::AnyInvocable<void() noexcept> get_per_thread_init() override;

 private:
  /**
   * @brief Safely casts itask to gpu_pipeline_task with type validation
   *
   * @param task The itask pointer to cast
   * @return gpu_pipeline_task* The casted gpu_pipeline_task pointer
   * @throws std::bad_cast if the task is not of type gpu_pipeline_task
   */
  gpu_pipeline_task* cast_to_gpu_pipeline_task(sirius::parallel::itask* task);

  cucascade::memory::exclusive_stream_pool _stream_pool;
  exec::publisher<std::unique_ptr<task_request>> _task_request_publisher;
  cucascade::memory::memory_space* _memory_space;
  sirius::parallel::downgrade_executor* _downgrade_executor{nullptr};
  sirius::creator::task_creator* _task_creator{nullptr};
  completion_handler* _completion_handler{nullptr};
  std::atomic<size_t> _tasks_executed{0};

  /// OOM-retry pacing counters (see oom_pacing_stats).
  std::atomic<size_t> _oom_reschedules_total{0};
  std::atomic<size_t> _starved_admissions_total{0};
  std::atomic<size_t> _backoff_events_total{0};
  std::atomic<size_t> _backoff_ms_total{0};
  /// Workers currently in an escalated backoff sleep. Escalated sleeps are
  /// capped at (num_threads - 1) concurrent sleepers so the bounded pool always
  /// keeps at least one slot free for dispatching runnable tasks; a retryer
  /// over the cap falls back to the base interval.
  std::atomic<uint32_t> _backoff_sleepers{0};
};

}  // namespace pipeline
}  // namespace sirius
