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
#include "exec/interruptible_mpmc.hpp"
#include "exec/kiosk.hpp"
#include "exec/thread_pool.hpp"
#include "op/scan/config.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "parallel/task.hpp"
#include "pipeline/task_request.hpp"

#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::creator {
class task_creator;
}  // namespace sirius::creator

namespace sirius::pipeline {
class completion_handler;
}  // namespace sirius::pipeline

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// DuckDB Scan Executor
//===----------------------------------------------------------------------===//

/**
 * @brief A task executor for duckdb scan tasks.
 *
 * This class manages a pool of threads dedicated to executing DuckDB scan
 * tasks with kiosk-based concurrency control.
 */
class duckdb_scan_executor {
 public:
  /**
   * @brief Constructs a new duckdb_scan_executor with task execution configuration
   *
   * @param config Configuration for the thread pool (thread count, etc.)
   * @param mem_mgr Pointer to the memory reservation manager for host allocations
   * @param task_request_publisher Publisher to submit task requests
   */
  explicit duckdb_scan_executor(
    exec::thread_pool_config config,
    cucascade::memory::memory_reservation_manager* mem_mgr,
    exec::publisher<std::unique_ptr<sirius::pipeline::task_request>> task_request_publisher);

  /**
   * @brief Destructor for the duckdb_scan_executor.
   */
  ~duckdb_scan_executor();

  // Non-copyable and non-movable
  duckdb_scan_executor(const duckdb_scan_executor&)            = delete;
  duckdb_scan_executor& operator=(const duckdb_scan_executor&) = delete;
  duckdb_scan_executor(duckdb_scan_executor&&)                 = delete;
  duckdb_scan_executor& operator=(duckdb_scan_executor&&)      = delete;

  /**
   * @brief Schedule a new task for execution.
   *
   * @param task The task to be scheduled.
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
   * @brief Wait for all scheduled tasks to complete.
   */
  void wait_all();

  /**
   * @brief Get the number of threads in the thread pool for this executor.
   *
   * @return The number of threads in the thread pool for this executor.
   */
  [[nodiscard]] int32_t get_num_threads() const { return _config.num_threads; }

  /**
   * @brief Set the task creator for scheduling output consumers
   *
   * @param task_creator Pointer to the task creator
   */
  void set_task_creator(sirius::creator::task_creator* task_creator);

  /**
   * @brief Drain any leftover tasks from the queue
   *
   * Clears the task queue of any remaining tasks from a previous query.
   */
  void drain_leftover_tasks();

  /**
   * @brief Drain tasks and wait for all in-flight work to complete.
   *
   * Interrupts the manager loop so it releases its kiosk ticket, waits for
   * all in-flight thread-pool tasks, then restarts the manager so the
   * executor is ready for the next query.  Used during error cleanup.
   */
  void drain_and_wait();

  /**
   * @brief Set the completion handler for query completion signaling
   *
   * @param handler Pointer to the completion handler
   */
  void set_completion_handler(sirius::pipeline::completion_handler* handler) noexcept;

  /**
   * @brief Cache scan results for the given query
   *
   * @param query The query string to cache results for
   * @return True if this is a re-execution of the same query (cache hit),
   *         false if the query changed (cache miss / cleared).
   */
  [[nodiscard]] bool cache_scan_results_for_query(const std::string& query);

  /**
   * @brief Configure scan result caching level
   *
   * @param level The cache level to use
   */
  void set_scan_caching_enabled(cache_level level);

  /**
   * @brief Check if scan result caching is enabled
   *
   * @return True if caching is enabled, false otherwise
   */
  [[nodiscard]] bool is_scan_caching_enabled() const noexcept
  {
    return _cache_level != cache_level::NONE;
  }

  /**
   * @brief Check if the scan executor is in preload mode (cache is hot).
   *
   * @return True if preload mode is active, false otherwise.
   */
  [[nodiscard]] bool is_preload_mode() const noexcept { return _preload_mode; }

  /**
   * @brief Prepare cache for scan operators
   *
   * In CACHE mode: ensures cache is empty and creates entries for each operator's ID
   * In PRELOAD mode: verifies all operator IDs are present in the cache
   *
   * @param scan_operators Vector of scan operators to prepare cache for
   */
  void prepare_cache_for_scan_operators(
    const std::vector<sirius::op::sirius_physical_operator*>& scan_operators);

  /**
   * @brief Inject cached scan data directly into pipeline data repositories.
   *
   * In preload mode, this bypasses the entire scan task infrastructure
   * (global state init, local state, compute_task) and pushes cached
   * data_batch objects straight into the downstream operator ports.
   *
   * For each scan operator (DuckDB or parquet) whose data is cached:
   *   1. Cached batches are cloned or shared based on cache level
   *   2. All batches are pushed to the sink's downstream repos
   *   3. The scan operator is marked exhausted / pipeline finished
   *
   * @param scan_operators All scan operators in the current query
   * @return Downstream consumer operators that should be scheduled,
   *         or empty if not in preload mode.
   */
  std::vector<op::sirius_physical_operator*> preload_into_pipelines(
    const std::vector<op::sirius_physical_operator*>& scan_operators);

 private:
  /**
   * @brief Manager loop to consume tasks from queue and dispatch to the thread pool
   */
  void manager_loop();

  /**
   * @brief Submit a scan task request to pipeline_executor
   */
  void submit_scan_request();

  std::unique_ptr<op::operator_data> get_scan_output(pipeline::sirius_pipeline_itask* task,
                                                     rmm::cuda_stream_view stream);

  /// Identifies the type of scan for cache cloning dispatch.
  enum class scan_type { DUCKDB, PARQUET };

  /**
   * @brief Clone or share cached batches based on cache level and scan type.
   *
   * Single source of truth for the cloning strategy:
   *   - TABLE_GPU: zero-copy (return as-is)
   *   - Other levels + DUCKDB: deep clone via data_batch::clone()
   *   - Other levels + PARQUET: shallow_clone() on host representations
   *
   * @param batches The cached batch vector to clone
   * @param type    The scan type that produced these batches
   * @param stream  CUDA stream for async copy (used by clone)
   * @return Cloned or shared batch vector ready for pipeline injection
   */
  std::vector<std::shared_ptr<cucascade::data_batch>> clone_cached_batches(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
    scan_type type,
    rmm::cuda_stream_view stream);

  /**
   * @brief Mark a scan operator's pipeline as finished for preload.
   *
   * Sets the appropriate exhausted/has_more_partitions flag based on
   * scan type, then calls update_pipeline_status().
   */
  static void mark_scan_pipeline_finished(op::sirius_physical_operator* scan_op);

  struct cache_entry {
    std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> batches;
    std::size_t batch_index{0};
  };

  mutable std::mutex _cache_mutex;
  std::unordered_map<size_t, std::unique_ptr<cache_entry>> _cache;
  std::size_t _query_hash{0};
  cache_level _cache_level{cache_level::NONE};
  bool _preload_mode{false};

  std::atomic<bool> _running{false};
  exec::thread_pool_config _config;
  exec::kiosk _kiosk;
  std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  exec::interruptible_mpmc<std::unique_ptr<sirius::parallel::itask>> _task_queue;
  std::thread _manager_thread;
  exec::publisher<std::unique_ptr<sirius::pipeline::task_request>> _task_request_publisher;
  cucascade::memory::memory_reservation_manager* _mem_mgr{nullptr};
  sirius::creator::task_creator* _task_creator{nullptr};
  sirius::pipeline::completion_handler* _completion_handler{nullptr};
  cucascade::memory::memory_space* _gpu_memory_space{nullptr};
};

}  // namespace sirius::op::scan
