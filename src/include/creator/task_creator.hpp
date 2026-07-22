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
#include "exec/inspectable_priority_queue.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <atomic>
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
   * @param sys_topology Optional system topology info for NUMA-aware GPU routing.
   */
  task_creator(task_creator_config config,
               sirius::memory::sirius_memory_reservation_manager& mem_res_mgr,
               const cucascade::memory::system_topology_info* sys_topology = nullptr);

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

  /// System topology for NUMA-aware GPU routing (non-owning, may be null)
  const cucascade::memory::system_topology_info* _sys_topology{nullptr};
  /// Maps NUMA node ID -> all GPU device_ids on that NUMA node (for HOST data
  /// locality). A NUMA node can host multiple GPUs; the NUMA-affinity rule
  /// round-robins across the vector so work spreads instead of pinning to the
  /// first GPU.
  /// GPUs that report numa_node=-1 (non-NUMA / single-NUMA hosts, per the
  /// Linux /sys/bus/pci/devices/*/numa_node convention) are normalized to
  /// NUMA 0 so they match the host memory space built for the single node.
  std::unordered_map<int, std::vector<int>> _numa_to_gpu;
  /// Round-robin counter for NUMA-affinity routing when multiple GPUs share a NUMA node.
  std::atomic<uint64_t> _numa_to_gpu_rr{0};
  /// Sorted device ids that actually have a GPU executor (memory-manager GPU
  /// spaces). Partition affinity indexes this, not the physical topology, so the
  /// pin resolves to a real executor when num_gpus < physical GPU count.
  std::vector<int> _active_gpu_ids;
};

}  // namespace sirius::creator
