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
#include "exec/query_lifecycle_registry.hpp"
#include "exec/queue_priority.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "query_id.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

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
  //! `node`'s operator type, captured at schedule() time.
  //!
  //! Stored rather than read back off `node` so the queue's key extractor — which runs inside the
  //! queue mutex on every push — dereferences no operator. A schedule() racing its query's
  //! teardown would otherwise read a freed operator while holding that mutex, and unlike the
  //! other keys this one had no reason to be resolved late.
  op::SiriusPhysicalOperatorType operator_type = op::SiriusPhysicalOperatorType::INVALID;
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

  /// \brief Narrow this query to a GPU subset, replacing the constructor's topology-derived
  /// list. Called once per query by sirius_engine::initialize_internal.
  ///
  /// @param full_count how many GPUs existed before narrowing. Passed in rather than inferred
  /// so it and @p ids are cut from the same list.
  void set_active_gpu_ids(std::vector<int> ids, std::size_t full_count);

  /// \brief The GPU subset this query was admitted onto.
  [[nodiscard]] const std::vector<int>& get_active_gpu_ids() const noexcept;

  /// \brief Bind @p query_id to the connection that is running it.
  /// Called at execution-window begin, before prepare_for_query.
  void set_client_context(sirius::query_id_t query_id, ::duckdb::ClientContext& client_context);

  /// \brief sets pipeline executor reference
  void set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler);

  /// \brief Bind the per-query lifecycle gate consulted before every enqueue.
  ///
  /// Without one (the default, and what most unit tests use) every query is treated as accepting
  /// work, i.e. the pre-gate behaviour.
  void set_query_lifecycle_registry(sirius::exec::query_lifecycle_registry* registry) noexcept
  {
    _query_lifecycle = registry;
  }

  /// \brief Register the per-query state for @p query's pipelines.
  ///
  /// Adds an entry; it does NOT clear other queries' entries. Call reset(query_id) to drop one.
  ///
  /// \param handler The query's completion signal
  void prepare_for_query(const sirius::planner::query& query,
                         std::shared_ptr<pipeline::completion_handler> handler);

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

  /// \brief Warm up one not-yet-activated scan of the oldest live query.
  /// No-op when no query is registered.
  void schedule_lookahead(std::optional<int> device_id_hint = std::nullopt);

  /// \brief Fail @p query_id with @p error, touching no shared subsystem.
  ///
  /// schedule() throws on an operator that carries no pipeline. Callers on paths that must not
  /// propagate (sirius_pipeline::notify_downstream_pipelines runs from ~gpu_pipeline_task and
  /// from the streaming-source close callback) route the exception here instead, so the query
  /// surfaces the error rather than the process terminating. The error goes to that query's own
  /// completion handler and nothing else; other in-flight queries keep running, and the failing
  /// query is unwound by sirius_engine::execute's drain_after_error(query_id).
  void report_fatal_error(sirius::query_id_t query_id, std::exception_ptr error);

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
   * @brief Whether the lifecycle gate still accepts work for @p query_id.
   *
   * True when no registry is bound (the unit-test default), preserving pre-gate behaviour.
   */
  [[nodiscard]] bool accepts_work(sirius::query_id_t query_id) const noexcept;

  /// \brief Log loudly when a push was refused for a query that is still accepting work.
  /// A refused push destroys the request, so a live query silently loses a task it is waiting on.
  void report_if_dropped(bool pushed, sirius::query_id_t query_id) const;

  /// \brief Whether the calling thread is one of this creator's task-creation pool workers.
  /// Used only to assert that stop() is never called from inside its own pool, which would
  /// self-deadlock in wait_all(). See task_creator::stop.
  [[nodiscard]] static bool is_pool_worker_thread();

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
  /// Follows WAITING_FOR_INPUT_DATA hints upstream to the operator that can
  /// produce next. get_next_task_hint() is side-effecting at every level (it
  /// can drain ports and make that pipeline finishable), so every pipeline the
  /// walk visits is appended to @p visited_pipelines for the caller to
  /// re-evaluate — a pipeline whose tasks all completed earlier gets no later
  /// mark_task_completed() to do it.
  op::sirius_physical_operator* get_operator_for_next_task(
    op::sirius_physical_operator* node,
    std::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& visited_pipelines);

  /// \brief report_fatal_error for callers that already hold the query's handler (the creation
  /// worker), avoiding a second lookup of a state it has in scope.
  void report_fatal_error(const std::shared_ptr<pipeline::completion_handler>& handler,
                          std::exception_ptr error);

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
  /// Non-owning; owned by SiriusContext and outlives this creator. Null in unit tests.
  sirius::exec::query_lifecycle_registry* _query_lifecycle{nullptr};
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

    //! This query's completion signal, for the creation-failure path (which has this state in
    //! scope but no task). Same handler every pipeline's global state carries.
    std::shared_ptr<pipeline::completion_handler> completion_handler;

    std::mutex lookahead_mutex;
    std::size_t index_of_next_lookahead{0};
    std::vector<op::sirius_physical_operator*> lookahead_queue;

    // (In-flight creation work is tracked by the pool itself, keyed by the query the slot is
    // attached to; see bounded_thread_pool::drain_and_wait. The bespoke counter that used to live
    // here duplicated that accounting and had to be kept in sync by hand across every exit path.)
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
  /// Serializes pool/manager-thread lifecycle (start_thread_pool / stop_thread_pool / stop).
  ///
  /// Deliberately NOT _global_state_mutex. do_stop_thread_pool() joins _manager_thread, and
  /// manager_loop() takes _global_state_mutex on every request via get_query_task_global_state().
  /// Holding that mutex across the join therefore deadlocked: the stopper waited for the manager
  /// thread while holding the very mutex the manager thread needed to finish its iteration. It
  /// fires whenever a query takes the error path (drain_after_error -> stop_thread_pool) while the
  /// creator is still running. The two mutexes guard disjoint state -- this one covers
  /// _bounded_pool / _manager_thread / _running, the other covers _query_task_global_states and
  /// _task_scheduler -- so splitting them removes the cycle rather than papering over it.
  std::mutex _pool_lifecycle_mutex;

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
  /// Separate from _numa_affinity_rr. A task can take a NUMA pick and then be clamped, so
  /// sharing one counter advances it twice per task: against an even subset size the stride
  /// never changes parity and every clamped task lands on the same GPU.
  std::atomic<uint64_t> _admission_rr{0};
  /// Sorted, deduped GPU device ids this query is admitted onto: every executor at
  /// construction, narrowed per query by `set_active_gpu_ids()`. Partition affinity indexes
  /// it (`_active_gpu_ids[partition_idx % size]`) and must stay in the same sorted order
  /// sirius_physical_partition uses for its device->slot map, so the two stay inverse.
  std::vector<int> _active_gpu_ids;
  /// GPU count before this query was narrowed, from the same list the admitted set was cut
  /// from; `_active_gpu_ids.size() < this` means the query is on a strict subset.
  std::size_t _full_gpu_count{0};
};

}  // namespace sirius::creator
