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
#include "op/scan/config.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "parallel/task.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/task_request.hpp"

#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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
 * tasks with bounded concurrency control.
 */
class duckdb_scan_executor : public sirius::parallel::itask_executor {
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
   * @brief Get the number of threads in the thread pool for this executor.
   *
   * @return The number of threads in the thread pool for this executor.
   */
  [[nodiscard]] int32_t get_num_threads() const { return _config.num_threads; }

  /**
   * @brief Wait for all scheduled tasks to complete.
   */
  void wait_all() { itask_executor::wait_all(); }

  /**
   * @brief Set the task creator for scheduling output consumers
   *
   * @param task_creator Pointer to the task creator
   */
  void set_task_creator(sirius::creator::task_creator* task_creator);

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

 protected:
  void manager_loop() override;

 private:
  /**
   * @brief Submit a scan task request to pipeline_executor
   */
  void submit_scan_request();

  /**
   * @brief Select the target GPU for the next scan batch.
   *
   * Distributes scan batches across GPUs proportional to available GPU memory.
   * Falls back to round-robin when all GPUs are at capacity.
   *
   * @return The device_id of the GPU to target for the next scan batch.
   */
  int select_target_gpu();

  std::unique_ptr<op::operator_data> get_scan_output(pipeline::sirius_pipeline_itask* task,
                                                     rmm::cuda_stream_view stream);

  struct cache_entry {
    std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> batches;
    std::size_t batch_index{0};
  };

  mutable std::mutex _cache_mutex;
  std::unordered_map<size_t, std::unique_ptr<cache_entry>> _cache;
  std::size_t _query_hash{0};
  cache_level _cache_level{cache_level::NONE};
  bool _preload_mode{false};

  std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;
  exec::publisher<std::unique_ptr<sirius::pipeline::task_request>> _task_request_publisher;
  cucascade::memory::memory_reservation_manager* _mem_mgr{nullptr};
  cucascade::memory::memory_space* _gpu_memory_space{nullptr};  ///< First GPU space (backward compat)
  std::vector<cucascade::memory::memory_space*> _gpu_memory_spaces;  ///< All GPU memory spaces
  std::atomic<size_t> _scan_round_robin{0};  ///< Round-robin counter for scan distribution
  sirius::creator::task_creator* _task_creator{nullptr};
  sirius::pipeline::completion_handler* _completion_handler{nullptr};
};

}  // namespace sirius::op::scan
