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

#include "config.hpp"
#include "duckdb/main/client_context.hpp"
#include "exec/bounded_thread_pool.hpp"
#include "exec/config.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "exec/queue_priority.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace sirius::pipeline {
class task_scheduler;
class sirius_pipeline_task_global_state;
}  // namespace sirius::pipeline

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace sirius::creator {

/**
 * @brief Manages the creation and scheduling of GPU pipeline tasks.
 *
 * The task_creator is responsible for creating tasks from GPU pipelines and scheduling
 * them for execution. It maintains a thread pool that processes task creation requests
 * from the task_creation_queue. The creator prioritizes table scan pipelines and uses
 * hints from operators to determine the next tasks to create.
 *
 * Usage:
 *   1. Construct with a task_creation_queue, thread count, and pipeline map.
 *   2. Call start_thread_pool() to begin processing tasks.
 *   3. Call start() to schedule initial scan pipelines.
 *   4. Call stop_thread_pool() when done.
 */

struct task_creation_request {
  op::sirius_physical_operator* node;
  request_type type = request_type::active;
};

class task_creator {
 public:
  /**
   * @brief Construct a new task_creator.
   *
   * @param config Configuration for the thread pool (thread count, name prefix, CPU affinity).
   * @param mem_res_mgr Reference to the memory reservation manager.
   * @param topology_index Optional shared GPU<->NUMA index for NUMA-aware GPU routing.
   */
  task_creator(task_creator_config config,
               sirius::memory::sirius_memory_reservation_manager& mem_res_mgr,
               std::shared_ptr<const sirius::memory::topology_index> topology_index = nullptr);

  /**
   * @brief Destructor that ensures the thread pool is stopped.
   */
  virtual ~task_creator();

  // Non-copyable and movable
  task_creator(const task_creator&)            = delete;
  task_creator& operator=(const task_creator&) = delete;
  task_creator(task_creator&&)                 = delete;
  task_creator& operator=(task_creator&&)      = delete;

  /// \brief sets client context needed for task creation
  void set_client_context(::duckdb::ClientContext& client_context);

  /// \brief sets pipeline executor reference
  void set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler);

  /// \brief prepare global states for all pipelines in the query
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief clean-up query bound resources and prepare the task creator for next query
  void reset();

  /**
   * @brief Stop the task creator and its thread pool.
   */
  void stop();

  /**
   * @brief Start the worker thread pool.
   *
   * Creates and starts the worker threads that process task creation requests.
   * This method is idempotent - calling it multiple times has no additional effect.
   */
  void start_thread_pool();

  /**
   * @brief Stop the worker thread pool.
   *
   * Stops all worker threads and waits for them to finish. This method is
   * idempotent - calling it multiple times has no additional effect.
   */
  void stop_thread_pool();

  /**
   * @brief Drain all pending task creation requests and wait for in-flight tasks to complete.
   *
   * Call this after a query completes (future resolved) but before destroying the engine/operators
   * to ensure no stale operator pointers are accessed by the task creator threads.
   */
  void drain_pending_tasks();

  /**
   * @brief Schedule a task creation info for processing.
   *
   * @param info The task creation info to schedule.
   */
  virtual void schedule(op::sirius_physical_operator* request);

  void schedule_lookahead(std::optional<int> device_id_hint = std::nullopt);

  /**
   * @brief Notification that the scheduler found its task queue empty.
   *
   * Plain @c std::function rather than @c std::function<void() noexcept>: the latter is not
   * usable with libstdc++/libc++, so the no-throw requirement lives in this contract and is
   * backstopped by a @c try/catch(...) at the fire site.
   *
   * @warning Fired on the @c task_scheduler management thread with **no lock held**. The
   *          callback must be non-blocking, must not throw (a throw is caught and logged, but
   *          the hook is then useless), and must not re-enter @c task_creator.
   */
  using task_queue_depleted_hook = std::function<void()>;

  /**
   * @brief Notification that a task-creation request produced no task.
   *
   * Plain @c std::function for the same reason as @ref task_queue_depleted_hook; the no-throw
   * requirement below is a contract, backstopped by a @c try/catch(...) at the fire site.
   *
   * @param requested The operator the request started from (never null). Borrowed, valid only
   *                  for the duration of the call.
   * @param kind      Whether the request was active or speculative look-ahead.
   *
   * @warning Fired on the task_creator's **single** manager thread with **no lock held**, from
   *          a code path that is *outside* the dispatch lambda's @c try block. The callback must
   *          be non-blocking and must not throw: the manager thread is the only task-creation
   *          thread in the engine, so blocking it stalls task creation engine-wide and an escaping
   *          exception would end all task creation silently. The implementation wraps the call in
   *          @c try/catch(...) as a backstop, but the contract is still "do not throw".
   */
  using task_not_created_hook =
    std::function<void(const op::sirius_physical_operator* requested, request_type kind)>;

  /**
   * @brief Install the queue-depleted hook. Single slot — the last setter wins.
   *
   * The callable is moved onto the heap once, here, and the fire path only ever copies the
   * owning @c shared_ptr. That is deliberate: the installed lambdas capture a @c std::weak_ptr,
   * which is not trivially copyable, so libstdc++'s @c std::function small-object optimisation
   * does not apply and copying the @c std::function itself would @c malloc on **every** fire —
   * once per matcher iteration on the single thread that dispatches every GPU task.
   *
   * The slot is snapshotted atomically and invoked on the copy, so replacing a callback does not
   * synchronize with an in-flight invocation (a fire already in progress runs to completion
   * against the old callable). Callbacks must not capture raw pointers to objects the
   * task_creator can outlive — capture a @c std::weak_ptr. In particular a hook must **never**
   * reach the scan manager through @c duckdb::SiriusContext::get_scan_manager(): that accessor
   * throws once the context has been terminated, and the not-created hook fires from outside any
   * @c try block. Cleared by @ref reset.
   */
  void set_on_task_queue_depleted(task_queue_depleted_hook hook);

  /// @copydoc set_on_task_queue_depleted
  void set_on_task_not_created(task_not_created_hook hook);

  /**
   * @brief Get the next task id.
   *
   * @return uint64_t The next task id.
   */
  uint64_t get_next_task_id();

  /**
   * @brief Compute a scheduling priority for every pipeline in the query.
   *
   * Partitions the pipeline DAG into branches (via query_index) and assigns each pipeline a
   * priority so that earlier (closer-to-scan) branches get lower values and run first (priority
   * ascends with execution order), honoring the configured priority_order within each branch.
   * Exposed for unit testing.
   *
   * @param query The query whose pipelines are prioritized.
   * @return Map from pipeline to its scheduling priority (pipelines absent from the map keep the
   *         default priority of 0).
   */
  [[nodiscard]] std::unordered_map<const pipeline::sirius_pipeline*, exec::queue_priority>
  compute_pipeline_priorities(const sirius::planner::query& query) const;

 protected:
  /**
   * @brief Stop the worker thread pool.
   *
   * Stops all worker threads and waits for them to finish. This method is
   * idempotent - calling it multiple times has no additional effect.
   */
  void do_stop_thread_pool();

  /**
   * @brief Find the operator for which to create the next task based on operator hints.
   *
   * This method queries the given node for a hint about what task to create next.
   *
   * @param node The operator node to get the next task hint from.
   * @return The operator node that should be scheduled next, or nullptr if no task should be
   * scheduled.
   */
  op::sirius_physical_operator* get_operator_for_next_task(op::sirius_physical_operator* node);

  /**
   * @brief Manager loop to consume task creation requests and dispatch to the thread pool.
   *
   * Reserves slots from the bounded pool (ensuring controlled concurrency), pulls task
   * creation requests from the queue, and dispatches work to the pool.
   */
  void manager_loop();

  /**
   * @brief Invoke the queue-depleted hook, if one is installed.
   *
   * Snapshot the slot, then invoke the snapshot, so a hook never runs with a task_creator lock
   * held and replacing a hook does not synchronize with an in-flight call. The snapshot is a
   * @c shared_ptr copy — one refcount bump, **no allocation** — which is why this fire path takes
   * no task_creator lock at all. The invocation is wrapped in @c try/catch(...) and the exception
   * logged and dropped, which is what makes this @c noexcept honest.
   *
   * Protected rather than private so the hook plumbing is drivable from a test subclass without
   * having to reach the anchor that calls it.
   */
  void fire_task_queue_depleted() noexcept;

  /**
   * @brief Invoke the not-created hook, if one is installed.
   *
   * Same lock-free snapshot-then-fire discipline and the same @c try/catch(...) backstop as
   * @ref fire_task_queue_depleted. The backstop is load-bearing here: the call site sits outside
   * the dispatch lambda's @c try block on the single manager thread, so an escaping exception
   * would silently end all task creation.
   *
   * @param requested The operator the failed request started from. Borrowed for the call only.
   * @param kind      Whether the request was active or speculative look-ahead.
   */
  void fire_task_not_created(const op::sirius_physical_operator* requested,
                             request_type kind) noexcept;

  std::atomic<bool> _running;
  task_creator_config _config;
  std::unique_ptr<exec::bounded_thread_pool> _bounded_pool;
  std::thread _manager_thread;
  ::duckdb::ClientContext* _client_context;
  sirius::pipeline::task_scheduler* _task_scheduler{nullptr};
  sirius::memory::sirius_memory_reservation_manager& _mem_res_mgr;
  std::atomic<uint64_t> _task_id{0};

  std::mutex _lookahead_mutex;              // Protect concurrent access to the lookahead scheduling
  std::size_t _index_of_next_lookahead{0};  // Index of the next operator to lookahead for
  std::vector<op::sirius_physical_operator*> _lookahead_queue;

  /// The two single-slot hooks. Held by @c shared_ptr so a fire is a refcount bump rather than a
  /// @c std::function copy: the installed lambdas capture a @c std::weak_ptr, which defeats
  /// libstdc++'s small-object optimisation, so copying the @c std::function would allocate on
  /// **every** fire — once per matcher iteration on the single thread that dispatches every GPU
  /// task.
  ///
  /// @c std::atomic so the slot is readable with no task_creator lock at all. (libstdc++ backs
  /// this with an internal spinlock rather than a lock-free CAS, but that spinlock is a strict
  /// leaf — held only across a pointer copy, never while calling out — so it cannot participate in
  /// any cycle with _lookahead_mutex -> sirius_pipeline::_status_mutex ->
  /// split_connector::_mutex, and it never allocates.)
  ///
  /// @c const because a snapshot is shared with an in-flight invocation: the callable itself must
  /// not be mutated behind a running hook's back, only the slot repointed.
  std::atomic<std::shared_ptr<const task_queue_depleted_hook>> _on_task_queue_depleted;
  std::atomic<std::shared_ptr<const task_not_created_hook>> _on_task_not_created;

  // Queue for creating tasks based on operators. The operator is the starting point to start
  // looking which task should be created, not necessarily the operator for whose pipeline the task
  // will be created
  exec::interruptible_mpmc<std::unique_ptr<task_creation_request>> _task_creation_queue;

  // Map of operator ID to global state for scan operators
  std::unordered_map<size_t, std::shared_ptr<pipeline::sirius_pipeline_task_global_state>>
    _gpu_operator_global_state_map;
  std::unique_ptr<duckdb::ThreadContext> _thread_context;
  std::unique_ptr<duckdb::ExecutionContext> _execution_context;
  std::mutex _global_state_mutex;  // Protect concurrent access to the map

  /// Shared GPU<->NUMA topology index for NUMA-aware GPU routing (may be null).
  /// Scoped to the memory manager's reserved GPU/HOST spaces:
  ///  - gpus_of(numa) drives HOST-data locality (a NUMA node can host multiple
  ///    GPUs; the round-robin below spreads work across them). NUMA node -1 is
  ///    the "unknown" key (non-NUMA / single-NUMA hosts) and is queried
  ///    verbatim from the host memory space's device id.
  ///  - gpu_ids() is the active executor set that partition affinity indexes,
  ///    so the pin resolves to a real executor when num_gpus < physical count.
  std::shared_ptr<const sirius::memory::topology_index> _topology_index;
  /// Round-robin counter for NUMA-affinity routing when multiple GPUs share a NUMA node.
  std::atomic<uint64_t> _numa_affinity_rr{0};
  /// Sorted, deduped active GPU device ids, materialized from
  /// `_topology_index->gpu_ids()` at construction. Partition affinity indexes
  /// this (`_active_gpu_ids[partition_idx % size]`); it must stay in the same
  /// sorted order that sirius_physical_partition uses for its device->slot
  /// mapping (see sirius_engine.cpp) so the two remain inverse to each other.
  std::vector<int> _active_gpu_ids;
};

}  // namespace sirius::creator
