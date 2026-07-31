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
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/downgraded_task_prefetcher.hpp"
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
   * @brief Prepare scheduler state for a query.
   *
   * Drains tasks left by the previous query, installs the new query and completion handler,
   * and resets per-query scheduler state.
   *
   * @param query Query whose tasks will be scheduled
   */
  void prepare_for_query(duckdb::shared_ptr<planner::query> query);

  /**
   * @brief Start query execution and return a future for completion.
   *
   * Sets up the completion handler and returns a future that will be satisfied
   * when the query completes or errors. Note: prepare_for_query must be called
   * before this method.
   *
   * @return A future that will be satisfied when the query completes.
   */
  std::future<void> start_query();

  /**
   * @brief Terminate the query execution and report the error to duckdb.
   *
   * @param error The error to report.
   */
  void terminate_query(std::exception_ptr error);

  /**
   * @brief Drain all in-flight tasks after a query error.
   *
   * Drains the top-level task queue and waits for each GPU executor to finish
   * all in-flight thread-pool tasks.  After this call it is safe for QueryEnd
   * to destroy data repositories without causing a use-after-free in executing
   * tasks.  Each GPU executor's manager thread is restarted so the executor is
   * ready for the next query.
   */
  void drain_after_error();

  /**
   * @brief This function interrupts executors and waits for all in-flight tasks to complete.
   * If any tasks are still in flight, an error is logged and an exception is thrown.
   * This is used to ensure that all tasks have completed before the query returns and tears down
   * the plan.
   * @throws std::runtime_error if any tasks are still in flight.
   */
  void wait_for_completion();

  // for testing/stress only — see Doxygen below.
  /**
   * @brief Inject an initial value into the round-robin counter.
   *
   * For testing/stress only. Intended to verify that the round-robin
   * distribution path (and the per-task-device contract documented in
   * `docs/super-sirius/pipeline-execution.md`) is correct under arbitrary
   * counter starting offsets — catches hash-bucket-order dependent bugs and
   * off-by-one drift that a counter starting at 0 each query might mask.
   *
   * Must be called AFTER `prepare_for_query` (which resets the counter
   * to 0) but BEFORE the first task is dispatched into
   * `management_eventloop`. Concurrent calls are safe (atomic store).
   */
  void set_no_pref_rr_counter_for_testing(size_t value) noexcept
  {
    _no_pref_rr_counter.store(value, std::memory_order_relaxed);
  }

 private:
  void management_eventloop();

  /// Build and start the HOST/DISK->GPU input prefetcher for queued tasks when
  /// enabled via SIRIUS_TASK_PREFETCH=1 (see downgraded_task_prefetcher.hpp).
  /// No-op otherwise.
  void maybe_start_downgraded_task_prefetcher();

  std::mutex _query_mutex;
  duckdb::shared_ptr<planner::query> _query;

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

  /// device_id -> GPU executor. std::map (not unordered_map) so iteration
  /// order is deterministic (ascending by device_id) — keeps preference-less
  /// task dispatch reproducible across runs.
  std::unordered_map<int, std::unique_ptr<gpu_pipeline_executor>> _gpu_executors;
  /// Deprecated as of pull-signal restoration: no-preference distribution now
  /// arises from which executor sends a device_ready signal first, not from a
  /// counter. Field retained for source compat with set_no_pref_rr_counter_for_testing.
  std::atomic<size_t> _no_pref_rr_counter{0};

  /// For resolving the GPU memory space and reservations of the downgraded-task
  /// prefetcher (started in start() when SIRIUS_TASK_PREFETCH=1).
  sirius::memory::sirius_memory_reservation_manager& _mem_mgr;
  /// Background HOST/DISK->GPU upgrader for queued tasks' downgraded inputs;
  /// stopped first in stop() so no conversion is in flight during teardown.
  std::unique_ptr<downgraded_task_prefetcher> _task_prefetcher;

  sirius::creator::task_creator* _task_creator{nullptr};
  std::unique_ptr<completion_handler> _completion_handler;
  std::shared_ptr<const telemetry::telemetry_context> _telemetry_context;
  std::unique_ptr<telemetry::TaskQueueHandleWrapper> _task_queue_telemetry;
};

}  // namespace pipeline
}  // namespace sirius
