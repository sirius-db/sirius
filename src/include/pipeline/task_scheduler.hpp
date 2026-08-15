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
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/task_request.hpp"
#include "planner/query.hpp"

#include <cucascade/memory/topology_discovery.hpp>

#include <atomic>
#include <future>
#include <memory>
#include <optional>
#include <unordered_map>

namespace sirius::parallel {
class downgrade_executor;
}  // namespace sirius::parallel

namespace sirius::telemetry {
class telemetry_context;
struct TaskQueueHandleWrapper;
}  // namespace sirius::telemetry

namespace sirius {

namespace creator {
class task_creator;
}

namespace pipeline {

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and uses a gpu_pipeline_task_queue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class task_scheduler {
 public:
  /**
   * @brief Constructs a new task_scheduler with task execution configuration
   *
   * @param gpu_executor_config Configuration for the GPU pipeline executor thread pool
   * @param mem_mgr Reference to the memory reservation manager
   * @param telemetry_context Shared pointer to the telemetry context
   * @param sys_topology Optional system topology info for CPU affinity
   * @param downgrade_executors Optional vector of downgrade executors
   */
  explicit task_scheduler(const exec::thread_pool_config& gpu_executor_config,
                          sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                          std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
                          const cucascade::memory::system_topology_info* sys_topology = nullptr,
                          const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>*
                            downgrade_executors = nullptr);

  /**
   * @brief Destructor for the task_scheduler.
   */
  ~task_scheduler();

  // Non-copyable but movable
  task_scheduler(const task_scheduler&)            = delete;
  task_scheduler& operator=(const task_scheduler&) = delete;
  task_scheduler(task_scheduler&&)                 = delete;
  task_scheduler& operator=(task_scheduler&&)      = delete;

  /**
   * @brief Schedules a task for execution with GPU-specific logic
   *
   * Overrides the base class schedule method to provide specialized scheduling
   * behavior for GPU pipeline operations, including resource allocation and
   * GPU context management.
   *
   * @param task The task to schedule (must be a gpu_pipeline_task)
   */
  void schedule(std::unique_ptr<sirius::parallel::itask> task);

  /**
   * @brief Starts the executor and initializes worker threads
   *
   * Initializes the thread pool and begins accepting tasks for execution.
   */
  void start();

  /**
   * @brief Stops the executor and cleanly shuts down worker threads
   *
   * Stops accepting new tasks and waits for all worker threads to complete
   * their current tasks before shutting down.
   */
  void stop();

  /**
   * @brief Set the task creator reference
   *
   * Sets the task creator for this executor and propagates it to all GPU executors.
   *
   * @param task_creator Reference to the task creator
   */
  void set_task_creator(sirius::creator::task_creator& task_creator);

  /**
   * @brief Get a pointer to the pipeline-level task queue.
   */
  [[nodiscard]] exec::multi_index_priority_queue<sirius::parallel::itask>*
  get_pipeline_task_queue() noexcept
  {
    return &_task_queue;
  }

  /**
   * @brief Call @p fn(device_id, executor) for each GPU executor, in ascending device_id order.
   *
   * Safe to call after a query completes (quiescent executors). Useful for
   * collecting per-GPU metrics without exposing the internal executor map.
   */
  template <typename Fn>
  void visit_executors(Fn&& fn) const
  {
    for (auto const& [device_id, exec] : _gpu_executors) {
      fn(device_id, *exec);
    }
  }

  /**
   * @brief Kick off query execution by scheduling its first scan.
   *
   * Completion is signalled through the query's own completion_handler, which its sirius_engine
   * owns and already holds the future for, so nothing is returned here.
   *
   * @param query The query to start; must have at least one schedulable scan source.
   */
  void start_query(const planner::query& query);

  /**
   * @brief Drop every queued task belonging to @p query_id.
   *
   * Clears the scheduler's queue and each GPU executor's queue of that query's pending work,
   * leaving every other query's tasks in place. In-flight tasks are unaffected.
   *
   * Called from the per-query cleanup so a finished or failed query leaves nothing queued that
   * points into the plan about to be destroyed.
   */
  void drain_query_tasks(sirius::query_id_t query_id);

  /**
   * @brief Fail one query by reporting @p error to its own completion handler.
   *
   * Touches no shared subsystem: other in-flight queries keep running. The caller's query is
   * unwound by sirius_engine::execute, whose future.get() catch runs drain_after_error(query_id).
   *
   * @param handler The failing query's completion handler.
   * @param error The error to report.
   */
  void terminate_query(const std::shared_ptr<completion_handler>& handler,
                       std::exception_ptr error);

  /**
   * @brief Bind the per-query lifecycle gate, propagating it to every GPU executor.
   *
   * Without one (the default, and what most unit tests use) every query is treated as accepting
   * work, i.e. the pre-gate behaviour.
   */
  void set_query_lifecycle_registry(sirius::exec::query_lifecycle_registry* registry);

  /**
   * @brief Drain all in-flight tasks after a query error.
   *
   * Drains the top-level task queue and waits for each GPU executor to finish
   * all in-flight thread-pool tasks.  After this call it is safe for QueryEnd
   * to destroy data repositories without causing a use-after-free in executing
   * tasks.  Each GPU executor's manager thread is restarted so the executor is
   * ready for the next query.
   */
  void drain_after_error(sirius::query_id_t query_id);

  /**
   * @brief This function interrupts executors and waits for all in-flight tasks to complete.
   * If any tasks are still in flight, an error is logged and an exception is thrown.
   * This is used to ensure that all tasks have completed before the query returns and tears down
   * the plan.
   * @throws std::runtime_error if any tasks are still in flight.
   */
  void wait_for_completion(sirius::query_id_t query_id);

 private:
  void management_eventloop();

  /// Pipeline-level task queue, ordered by task priority (highest dispatched first).
  exec::multi_index_priority_queue<sirius::parallel::itask> _task_queue;
  exec::channel<std::unique_ptr<task_request>> _task_request_channel;
  /// Publisher used by schedule() to wake the management event loop when a new
  /// task is pushed into _task_queue. The event loop blocks on _task_request_channel
  /// for two event kinds: device_ready (sent by gpu_pipeline_executors when a worker
  /// thread is reserved) and task_available (sent by schedule()).
  std::optional<exec::publisher<std::unique_ptr<task_request>>> _self_publisher;
  std::thread _management_thread;
  std::atomic<bool> _running{false};

  /// Set of GPU device_ids that have a reserved worker thread waiting for a task.
  /// Only mutated by the management thread (matches device_ready signals from
  /// _task_request_channel and erases on dispatch), so no synchronization needed.
  std::vector<int> _ready_devices;

  /// device_id -> GPU executor. Dispatch never iterates this map — tasks are
  /// dispatched by keyed lookup (`at(device_id)`) for a ready device, with the
  /// device order coming from _ready_devices — so the unordered iteration
  /// order (used only for start/stop/wiring propagation and per-query drains)
  /// does not affect scheduling.
  std::unordered_map<int, std::unique_ptr<gpu_pipeline_executor>> _gpu_executors;
  sirius::creator::task_creator* _task_creator{nullptr};
  /// Non-owning; owned by SiriusContext and outlives this scheduler. Null in unit tests.
  sirius::exec::query_lifecycle_registry* _query_lifecycle{nullptr};
  std::shared_ptr<const telemetry::telemetry_context> _telemetry_context;
  std::unique_ptr<telemetry::TaskQueueHandleWrapper> _task_queue_telemetry;
};

}  // namespace pipeline
}  // namespace sirius
