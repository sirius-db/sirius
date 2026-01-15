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

// sirius
#include <op/scan/duckdb_scan_task_queue.hpp>
#include <parallel/task_executor.hpp>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// DuckDB Scan Executor
//===----------------------------------------------------------------------===//

/**
 * @brief A task executor for duckdb scan tasks.
 *
 * This class extends the generic itask_executor simply by instantiating it with a
 * duckdb_scan_task_queue.
 *
 */
class duckdb_scan_executor : public sirius::parallel::itask_executor {
 public:
  //===----------Constructor----------===//
  explicit duckdb_scan_executor(sirius::parallel::task_executor_config config)
    : sirius::parallel::itask_executor(std::make_unique<duckdb_scan_task_queue>(config.num_threads),
                                       config)
  {
  }

  //===----------Methods----------===//
  /**
   * @brief Schedule a new task for execution.
   *
   * @param task The task to be scheduled.
   */
  void schedule(std::unique_ptr<sirius::parallel::itask> task) override;

  /**
   * @brief Wait for all scheduled tasks to complete.
   */
  void wait();

  /**
   * @brief Worker thread loop.
   *
   * @param worker_id The ID of the worker thread.
   */
  void worker_loop(int32_t worker_id) override;

  /**
   * @brief Get the number of threads in the thread pool for this executor.
   *
   * @return The number of threads in the thread pool for this executor.
   */
  [[nodiscard]] int32_t get_num_threads() const { return _config.num_threads; }

  //===----------Fields----------===//
 private:
  std::atomic<uint64_t> _total_tasks    = 0;  ///< The total number of scheduled tasks
  std::atomic<uint64_t> _finished_tasks = 0;  ///< The total number of finished tasks
  std::mutex _finish_mutex;                   ///< Mutex to protect condition variable
  std::condition_variable _finish_cv;         ///< Condition variable to signal task completion
};

}  // namespace sirius::op::scan
