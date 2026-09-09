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

#include "exec/bounded_thread_pool.hpp"
#include "exec/config.hpp"
#include "exec/inspectable_mpsc.hpp"
#include "exec/invocable.hpp"
#include "parallel/task.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <thread>

namespace sirius {
namespace telemetry {
class telemetry_context;
struct TaskQueueHandleWrapper;
}  // namespace telemetry

namespace parallel {

/**
 * @brief Abstract base class for all task executors.
 *
 * Holds the common infrastructure shared by gpu_pipeline_executor,
 * duckdb_scan_executor, and downgrade_executor:
 *   - a bounded_thread_pool for concurrency control and task execution
 *   - an inspectable MPSC task queue
 *   - a manager thread that drives the dispatch loop
 *
 * Subclasses must implement manager_loop() and may override the virtual
 * hooks get_per_thread_init(), on_start(), on_stop(), and on_stopped() for
 * any executor-specific startup/shutdown behaviour.
 */
class itask_executor {
 public:
  /**
   * @param device_id GPU this executor is bound to, if any. Used to parent the
   * task-queue telemetry under that GPU's device group instead of the engine.
   */
  explicit itask_executor(exec::thread_pool_config config,
                          std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
                          std::optional<int> device_id = std::nullopt);

  virtual ~itask_executor();

  // Non-copyable and non-movable
  itask_executor(const itask_executor&)            = delete;
  itask_executor& operator=(const itask_executor&) = delete;
  itask_executor(itask_executor&&)                 = delete;
  itask_executor& operator=(itask_executor&&)      = delete;

  /**
   * @brief Schedule a task for execution.
   */
  void schedule(std::unique_ptr<itask> task);

  /**
   * @brief Start the executor: creates the thread pool and manager thread.
   *
   * Calls get_per_thread_init() to obtain any per-worker-thread init function,
   * then calls on_start() after launching the manager thread so subclasses can
   * start additional threads (e.g. a monitor thread).
   */
  void start();

  /**
   * @brief Stop the executor and wait for all in-flight work to finish.
   *
   * Stops the kiosk, interrupts the task queue, calls on_stop() (so subclasses
   * can join extra threads before the manager thread is joined), joins the
   * manager thread, waits for all kiosk tickets to be released, stops the
   * thread pool, then calls on_stopped() for any final cleanup.
   */
  void stop();

  /**
   * @brief Block until all in-flight tasks complete. Convenience wrapper over the pool.
   */
  void wait_all();

  /**
   * @brief Drain any leftover tasks remaining in the queue.
   */
  void drain_leftover_tasks();

  /**
   * @brief Drain in-flight tasks and restart the manager, ready for the next query.
   *
   * Stops the kiosk and interrupts the queue so the manager exits, waits for
   * all in-flight thread-pool tasks, drains the queue, then re-enables both
   * and restarts the manager thread.
   */
  void drain_and_wait();

  /**
   * @brief Like drain_and_wait(), but VALIDATES the queue is empty instead of
   * draining it.
   *
   * Stops the kiosk and interrupts the queue so the manager exits, waits for all
   * in-flight thread-pool tasks, then — instead of draining — checks that the
   * task queue is empty. If it is not, logs an error and throws: a non-empty queue
   * at query completion means tasks were still scheduled when we declared the query
   * done. Re-enables the queue/pool and restarts the manager thread either way
   * (the throw happens after the executor is left in a restartable state).
   */
  void wait_and_validate_empty();

 protected:
  /**
   * @brief Main dispatch loop — must be implemented by each subclass.
   *
   * Called on the dedicated manager thread. Responsible for acquiring kiosk
   * tickets, popping tasks from _task_queue, and submitting them to
   * _thread_pool.
   */
  virtual void manager_loop() = 0;

  /**
   * @brief Return a per-worker-thread init function for the thread pool.
   *
   * Called once during start() before the thread pool is created. The default
   * returns nullptr (no per-thread init). Override to set the CUDA device or
   * perform other per-thread setup.
   */
  virtual sirius::exec::invocable<void() noexcept> get_per_thread_init() { return nullptr; }

  /**
   * @brief Called from start() after the manager thread is launched.
   *
   * Override to start additional threads (e.g. a monitor thread in
   * downgrade_executor).
   */
  virtual void on_start() {}

  /**
   * @brief Called from stop() after the task queue is interrupted, before the
   * manager thread is joined.
   *
   * Override to join any extra threads (e.g. the monitor thread in
   * downgrade_executor) that must finish before the manager exits.
   */
  virtual void on_stop() {}

  /**
   * @brief Called from stop() after the thread pool has been stopped.
   *
   * Override for any final cleanup (e.g. destroying a CUDA stream in
   * downgrade_executor).
   */
  virtual void on_stopped() {}

 protected:
  std::atomic<bool> _running{false};
  exec::thread_pool_config _config;
  std::unique_ptr<exec::bounded_thread_pool> _bounded_pool;
  exec::inspectable_mpsc<itask> _task_queue;
  std::thread _manager_thread;
  std::shared_ptr<const telemetry::telemetry_context> _telemetry_context;
  std::unique_ptr<telemetry::TaskQueueHandleWrapper> _task_queue_telemetry;
};

}  // namespace parallel
}  // namespace sirius
