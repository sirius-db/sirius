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
#include "exec/interruptible_mpmc.hpp"
#include "exec/kiosk.hpp"
#include "exec/thread_pool.hpp"
#include "parallel/task.hpp"

#include <absl/functional/any_invocable.h>

#include <atomic>
#include <memory>
#include <thread>

namespace sirius {
namespace parallel {

/**
 * @brief Abstract base class for all task executors.
 *
 * Holds the common infrastructure shared by gpu_pipeline_executor,
 * duckdb_scan_executor, and downgrade_executor:
 *   - a bounded kiosk for concurrency control
 *   - an interruptible MPMC task queue
 *   - a thread pool for task execution
 *   - a manager thread that drives the dispatch loop
 *
 * Subclasses must implement manager_loop() and may override the virtual
 * hooks get_per_thread_init(), on_start(), on_stop(), and on_stopped() for
 * any executor-specific startup/shutdown behaviour.
 */
class itask_executor {
 public:
  explicit itask_executor(exec::thread_pool_config config)
    : _config(std::move(config)), _kiosk(_config.num_threads)
  {
  }

  virtual ~itask_executor() { stop(); }

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
  virtual absl::AnyInvocable<void() noexcept> get_per_thread_init() { return nullptr; }

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
  exec::kiosk _kiosk;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  exec::interruptible_mpmc<std::unique_ptr<itask>> _task_queue;
  std::thread _manager_thread;
};

}  // namespace parallel
}  // namespace sirius
