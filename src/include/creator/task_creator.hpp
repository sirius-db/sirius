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
#include "exec/multi_index_priority_queue.hpp"
#include "exec/queue_priority.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "query_id.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <condition_variable>
#include <map>
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
  //! The query `node` belongs to. Indexes the request in the creation queue so a finished or
  //! failed query's pending requests can be dropped without touching any other query's.
  sirius::query_id_t query_id = sirius::make_query_id(0);
  //! Scheduling priority of `node`'s pipeline; orders the creation queue the same way the
  //! execution queue is ordered (query first, then within-query pipeline rank).
  exec::queue_priority priority = 0;
  //! Preferred GPU, when the caller had a hint. Only a secondary index; does not bind the task.
  int device_id = exec::no_preferred_device;
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

  /// \brief Bind @p query_id to the connection that is running it.
  /// Called at execution-window begin, before prepare_for_query.
  void set_client_context(sirius::query_id_t query_id, ::duckdb::ClientContext& client_context);

  /// \brief sets pipeline executor reference
  void set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler);

  /// \brief Register the per-query state for @p query's pipelines.
  ///
  /// Adds an entry; it does NOT clear other queries' entries. Call reset(query_id) to drop one.
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief Drop everything held for @p query_id: pending creation requests, in-flight creation
  /// work, and the per-query state entry. Other queries are untouched.
  ///
  /// Runs on both the success and the failure path (SiriusContext::run_mandatory_cleanup, which
  /// StandaloneQueryScope guarantees exactly once), so a failed query cannot leave the shared
  /// creator holding stale operator pointers.
  ///
  /// Must complete before @p query_id's planner::query is destroyed: queued requests hold raw
  /// operator pointers into its plan.
  void reset(sirius::query_id_t query_id);

  /// \brief Drop every query's state. Teardown only (SiriusContext::terminate).
  void reset_all();

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
   * @brief Drop @p query_id's pending creation requests and wait for its in-flight creation
   *        work to finish. Other queries keep running.
   *
   * Call this after a query completes (future resolved) but before destroying the engine/operators
   * to ensure no stale operator pointers are accessed by the task creator threads. Unlike the
   * previous global drain, this neither interrupts the queue nor waits on other queries' work.
   */
  void drain_pending_tasks(sirius::query_id_t query_id);

  /**
   * @brief Schedule a task creation info for processing.
   *
   * @param info The task creation info to schedule.
   */
  virtual void schedule(op::sirius_physical_operator* request);

  /// \brief Overload for callers that already know the query; avoids re-deriving it.
  void schedule(op::sirius_physical_operator* request, sirius::query_id_t query_id);

  void schedule_lookahead(sirius::query_id_t query_id,
                          std::optional<int> device_id_hint = std::nullopt);

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

  std::atomic<bool> _running;
  task_creator_config _config;
  std::unique_ptr<exec::bounded_thread_pool> _bounded_pool;
  std::thread _manager_thread;
  ::duckdb::ClientContext* _client_context;
  sirius::pipeline::task_scheduler* _task_scheduler{nullptr};
  sirius::memory::sirius_memory_reservation_manager& _mem_res_mgr;
  std::atomic<uint64_t> _task_id{0};

  /**
   * @brief Everything the task creator holds on behalf of ONE query.
   *
   * Handed out as a `shared_ptr`: a worker resolves it once and then reads `global_states`,
   * which is never mutated after `prepare_for_query`. That makes the lookup race-free even if
   * the query is erased concurrently — the worker's copy keeps the state alive until it is done.
   * Operator ids restart at 0 for every query, so `global_states` is only unique *within* an
   * entry; keying it globally is what would let two queries fetch each other's state.
   */
  struct query_task_global_state {
    //! Source operator id -> that pipeline's task global state. Written once by
    //! prepare_for_query, read-only afterwards.
    std::unordered_map<size_t, std::shared_ptr<pipeline::sirius_pipeline_task_global_state>>
      global_states;

    //! Client context of the connection running this query. Per query because two concurrent
    //! queries on different connections have different contexts.
    ::duckdb::ClientContext* client_context{nullptr};

    std::mutex lookahead_mutex;
    std::size_t index_of_next_lookahead{0};
    std::vector<op::sirius_physical_operator*> lookahead_queue;

    //! Per-query stand-in for `bounded_thread_pool::wait_all()`, which can only wait on every
    //! query's creation work at once. Incremented before dispatch, decremented when the lambda
    //! leaves (including by exception); `wait_for_in_flight()` blocks until it reaches zero.
    std::mutex in_flight_mutex;
    std::condition_variable in_flight_cv;
    std::size_t in_flight{0};

    void enter_in_flight();
    void leave_in_flight();
    void wait_for_in_flight();
  };

  //! Resolve a query's state, or nullptr when it has already been reset.
  std::shared_ptr<query_task_global_state> get_query_task_global_state(
    sirius::query_id_t query_id) const;

  // Queue for creating tasks based on operators. The operator is the starting point to start
  // looking which task should be created, not necessarily the operator for whose pipeline the task
  // will be created.
  //
  // A multi_index_priority_queue rather than a plain FIFO so that (a) creation is ordered the
  // same way execution is — earlier query first, then pipeline rank — and (b) a single query's
  // pending requests can be dropped via drain(query_index{...}) without disturbing any other
  // query's.
  exec::multi_index_priority_queue<task_creation_request> _task_creation_queue;

  //! One entry per in-flight query. Guarded by _global_state_mutex.
  std::map<sirius::query_id_t, std::shared_ptr<query_task_global_state>> _query_task_global_states;
  mutable std::mutex _global_state_mutex;  // Protect concurrent access to _query_task_global_states

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
