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
#include "exec/query_stage_manager.hpp"
#include "exec/queue_priority.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
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
};

/// Why @ref task_creator::get_operator_for_next_task stopped.
enum class next_task_state {
  /// An operator is ready to have a task created for it.
  ready,
  /// No operator in the chain can produce right now.  Not necessarily terminal:
  /// an operator waiting on input reports this until its input arrives.
  depleted,
};

/// The operator @ref task_creator::get_operator_for_next_task settled on,
/// together with what it means -- see @c next_task_state.
struct next_task_result {
  op::sirius_physical_operator* op{nullptr};
  next_task_state state{next_task_state::depleted};
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

  /// \brief sets client context needed for task creation
  void set_client_context(::duckdb::ClientContext& client_context);

  /// \brief sets pipeline executor reference
  void set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler);

  /// Attach the query-stage observer, sharing ownership of it so the handle is
  /// neither null nor dangling for as long as this creator can report.  Until
  /// this is called the creator reports into its own private manager, which has
  /// no listeners and so is a no-op.
  void set_query_stage_manager(sirius::exec::query_stage_manager& manager)
  {
    _query_stage_manager = manager.shared_from_this();
  }

  /// Report that @p pipeline_id has closed, so its source will produce no more
  /// tasks.  Called by sirius_pipeline; this supplies the query id, which the
  /// pipeline cannot see.
  void notify_pipeline_closure(std::size_t pipeline_id, std::size_t source_operator_id) noexcept;

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
   * This method queries the given node for a hint about what task to create next,
   * walking through operators that are themselves waiting on input.
   *
   * @param node The operator node to get the next task hint from.
   * @return The operator and what it means.  On @c ready, the operator is the
   *         one to schedule.  On @c depleted, it is the operator that was asked
   *         last and could not produce -- which is what makes the failure
   *         attributable: the walk can descend several operators past the one
   *         the request named, and reporting the request's own operator would
   *         name a node that is waiting perfectly happily.  Null only when
   *         @p node is null.
   */
  /// Follows WAITING_FOR_INPUT_DATA hints upstream to the operator that can
  /// produce next. get_next_task_hint() is side-effecting at every level (it
  /// can drain ports and make that pipeline finishable), so every pipeline the
  /// walk visits is appended to @p visited_pipelines for the caller to
  /// re-evaluate — a pipeline whose tasks all completed earlier gets no later
  /// mark_task_completed() to do it.
  [[nodiscard]] next_task_result get_operator_for_next_task(
    op::sirius_physical_operator* node,
    std::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& visited_pipelines);

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
  // Non-owning; SiriusContext stops and joins this creator before destroying the scheduler.
  sirius::pipeline::task_scheduler* _task_scheduler{nullptr};
  /// Observer of query stage transitions.  Never null: seeded with an
  /// unsubscribed manager so the reporting paths need no check, and replaced by
  /// the context's own in set_query_stage_manager.
  std::shared_ptr<sirius::exec::query_stage_manager> _query_stage_manager{
    std::make_shared<sirius::exec::query_stage_manager>()};
  /// Stamped by prepare_for_query; the manager loop has no other handle on it.
  sirius::query_id_t _query_id{};
  sirius::memory::sirius_memory_reservation_manager& _mem_res_mgr;
  std::atomic<uint64_t> _task_id{0};

  std::mutex _lookahead_mutex;              // Protect concurrent access to the lookahead scheduling
  std::size_t _index_of_next_lookahead{0};  // Index of the next operator to lookahead for
  std::vector<op::sirius_physical_operator*> _lookahead_queue;

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
