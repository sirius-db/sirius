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

#include <atomic>
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
struct TaskManagerLoopThreadHandleWrapper;
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

  /**
   * @brief Prepare and dispatch one popped task: reservation, downgrade-on-shortfall, local-state
   *        wiring, then hand-off to the bounded pool.
   *
   * Split out of manager_loop() so a failure here is contained to @p pipeline_task's query. Every
   * failure path reports to that task's own completion handler and returns; none of them may stop
   * the manager thread, which serves every in-flight query on this device. noexcept because
   * manager_loop() is a std::thread entry function — an escaping exception would be
   * std::terminate, not a query failure.
   *
   * @param pipeline_task The task popped from this executor's queue. Consumed.
   * @param slot The reserved pool slot. Consumed by dispatch, or released on any early return.
   * @param manager_thread_telemetry This manager thread's telemetry handle, for resource
   * attribution.
   */
  void process_task(
    std::unique_ptr<sirius::parallel::itask> pipeline_task,
    exec::bounded_thread_pool::slot slot,
    telemetry::TaskManagerLoopThreadHandleWrapper& manager_thread_telemetry) noexcept;

  cucascade::memory::exclusive_stream_pool _stream_pool;
  exec::publisher<std::unique_ptr<task_request>> _task_request_publisher;
  cucascade::memory::memory_space* _memory_space;
  sirius::parallel::downgrade_executor* _downgrade_executor{nullptr};
  sirius::creator::task_creator* _task_creator{nullptr};
  std::atomic<size_t> _tasks_executed{0};
};

}  // namespace pipeline
}  // namespace sirius
