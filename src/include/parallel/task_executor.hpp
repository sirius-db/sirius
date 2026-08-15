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
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "parallel/task.hpp"
#include "query_id.hpp"

#include <absl/functional/any_invocable.h>

#include <atomic>
#include <memory>
#include <mutex>
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
   *
   * A no-op when the lifecycle gate reports that the task's query is tearing down: the OOM
   * reschedule path re-enters here from a worker thread long after a drain may have passed.
   */
  void schedule(std::unique_ptr<itask> task);

  /**
   * @brief Bind the per-query lifecycle gate consulted before enqueuing.
   *
   * Without one (the default, and what most unit tests use) every query is treated as accepting
   * work, i.e. the pre-gate behaviour.
   */
  void set_query_lifecycle_registry(exec::query_lifecycle_registry* registry) noexcept
  {
    _query_lifecycle = registry;
  }

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
   * Interrupts the bounded pool and the task queue, calls on_stop() (so
   * subclasses can join extra threads before the manager thread is joined),
   * joins the manager thread, waits for all pool slots to be released, stops
   * the thread pool, then calls on_stopped() for any final cleanup.
   */
  void stop();

  /**
   * @brief Block until all in-flight tasks complete. Convenience wrapper over the pool.
   */
  void wait_all();

  /**
   * @brief Drain any leftover tasks remaining in the queue, for every query.
   */
  void drain_leftover_tasks();

  /**
   * @brief Drop the queued tasks belonging to one query.
   *
   * Tasks of other queries are left in place and the queue stays open, so unlike interrupt()
   * this does not stall any other query's producers or consumers. Only queued work is affected;
   * a task already dispatched to the thread pool runs to completion.
   */
  void drain_query_tasks(sirius::query_id_t query_id);

  /**
   * @brief Drain in-flight tasks and restart the manager, ready for the next query.
   *
   * Interrupts the bounded pool and the queue so the manager exits, waits for
   * all in-flight thread-pool tasks, drains the queue, then re-enables both
   * and restarts the manager thread.
   */
  void drain_and_wait();

  /**
   * @brief Wait for in-flight work, then assert @p query_id has nothing queued here.
   *
   * The success-path counterpart of wait_and_drain_query(). A non-empty queue for @p query_id at
   * completion means tasks were still being scheduled when the query was declared done, so this
   * throws rather than draining, which would hide the bug.
   *
   * Unlike the whole-executor version it replaced, this neither interrupts the shared queue nor
   * stops and restarts the manager thread — both of which dropped co-tenant queries' queued work
   * and stalled their producers on every completion. It also no longer validates the *whole*
   * queue, which made one query fail because another had work legitimately queued. The in-flight
   * wait is per-query too (bounded_thread_pool::wait_for_query).
   */
  void wait_and_validate_empty(sirius::query_id_t query_id);

  /**
   * @brief Wait for @p query_id's in-flight work, then drop its queued tasks.
   *
   * The error-path counterpart of wait_and_validate_empty(). Waiting first guarantees no thread is
   * still executing a task that references the failing query's plan before the caller lets that
   * plan be destroyed. The wait covers this query's attributed slots plus any UNTAGGED slot (a
   * task with no pipeline, whose query cannot be determined) — never a co-tenant's attributed
   * work, so a peer's parked memory wait does not extend this query's cleanup.
   */
  void wait_and_drain_query(sirius::query_id_t query_id);

 protected:
  /**
   * @brief Stop and join the manager thread, then reopen the queue.
   *
   * Releasing the manager's pool slot is a PRECONDITION for the caller's in-flight wait, not an
   * optimization: manager_loop() reserves a slot and then blocks in pop(), so an idle manager
   * holds an active untagged slot forever and any wait that needs the untagged count (or
   * active_) to reach zero would never return. The join also closes the pop-to-attach window: a
   * popped task in the manager's hand belongs to neither the queue nor the per-query slot
   * count, so per-query waits are only trustworthy once the manager is down.
   *
   * Waiting for in-flight pool work is the CALLER's job (per-query in the error bracket,
   * whole-pool in stop()/drain_and_wait()), after this returns.
   */
  void quiesce_manager();

  /// \brief Re-arm the pool and queue and restart the manager thread after quiesce_manager().
  void resume_manager();

  /**
   * @brief Main dispatch loop — must be implemented by each subclass.
   *
   * Called on the dedicated manager thread. Responsible for reserving pool
   * slots, popping tasks from _task_queue, and dispatching them on
   * _bounded_pool.
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
  std::unique_ptr<exec::bounded_thread_pool> _bounded_pool;
  /// Ordered by task priority and indexed by query, so one query's queued work can be
  /// dropped without touching another's. Keys come from pipeline::index_keys_for, the
  /// same extractor the task_scheduler's queue uses.
  exec::multi_index_priority_queue<itask> _task_queue;
  /// Non-owning; owned by SiriusContext and outlives this executor. Null in unit tests.
  exec::query_lifecycle_registry* _query_lifecycle{nullptr};
  /// Serializes every stop-join-restart transition of _manager_thread (stop(), drain_and_wait(),
  /// wait_and_drain_query()'s quiesce/resume bracket). Two queries failing concurrently reach
  /// drain_after_error on the same executor; unserialized, the second resume_manager() assigns
  /// onto the joinable thread the first just created — std::thread::operator= terminates. Same
  /// bug class as downgrade_executor::_lifecycle_mutex. The success path
  /// (wait_and_validate_empty) never touches _manager_thread and deliberately does not take this.
  std::mutex _manager_lifecycle_mutex;
  std::thread _manager_thread;
  std::shared_ptr<const telemetry::telemetry_context> _telemetry_context;
  std::unique_ptr<telemetry::TaskQueueHandleWrapper> _task_queue_telemetry;
};

}  // namespace parallel
}  // namespace sirius
