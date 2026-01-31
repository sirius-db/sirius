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
#include "exec/config.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_request.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <unordered_map>

namespace sirius::creator {
class task_creator;
}  // namespace sirius::creator

namespace sirius {
namespace pipeline {

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and uses a gpu_pipeline_task_queue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class pipeline_executor {
 public:
  /**
   * @brief Constructs a new pipeline_executor with task execution configuration
   *
   * @param gpu_executor_config Configuration for the GPU pipeline executor thread pool
   * @param scan_executor_config Configuration for the scan executor thread pool
   * @param mem_mgr Reference to the memory reservation manager
   * @param sys_topology Optional system topology info for CPU affinity
   */
  explicit pipeline_executor(const exec::thread_pool_config& gpu_executor_config,
                             const exec::thread_pool_config& scan_executor_config,
                             sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                             const cucascade::memory::system_topology_info* sys_topology = nullptr);

  /**
   * @brief Destructor for the gpu_pipeline_executor.
   */
  ~pipeline_executor() = default;

  // Non-copyable but movable
  pipeline_executor(const pipeline_executor&)            = delete;
  pipeline_executor& operator=(const pipeline_executor&) = delete;
  pipeline_executor(pipeline_executor&&)                 = delete;
  pipeline_executor& operator=(pipeline_executor&&)      = delete;

  /**
   * @brief Schedules a task for execution with GPU-specific logic
   *
   * Overrides the base class schedule method to provide specialized scheduling
   * behavior for GPU pipeline operations, including resource allocation and
   * GPU context management.
   *
   * @param task The task to schedule (must be a gpu_pipeline_task)
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
   * @brief Submit a task request to task_request_queue
   */
  void submit_task_request(std::unique_ptr<task_request> request);

  /**
   * @brief Set the task creator reference
   *
   * Sets the task creator for this executor and propagates it to all GPU executors.
   *
   * @param task_creator Reference to the task creator
   */
  void set_task_creator(sirius::creator::task_creator& task_creator);

  /**
   * @brief Get the scan executor reference
   *
   * @return Reference to the duckdb scan executor
   */
  [[nodiscard]] sirius::op::scan::duckdb_scan_executor& get_scan_executor() noexcept
  {
    return *_scan_executor;
  }

  [[nodiscard]] const sirius::op::scan::duckdb_scan_executor& get_scan_executor() const noexcept
  {
    return *_scan_executor;
  }

 private:
  void management_eventloop();

  std::mutex _priority_scans_mutex;
  std::queue<op::sirius_physical_operator*> _priority_scans;
  exec::interruptible_mpmc<std::unique_ptr<sirius::parallel::itask>>
    _task_queue;  ///< Queue for GPU pipeline tasks
  exec::interruptible_mpmc<std::unique_ptr<task_request>> _task_request_queue;
  std::thread _management_thread;
  std::atomic<bool> _running{false};

  std::unordered_map<int, std::unique_ptr<gpu_pipeline_executor>>
    _gpu_executors;  ///< Map of device_id to GPU executor

  sirius::creator::task_creator* _task_creator{nullptr};
  std::unique_ptr<sirius::op::scan::duckdb_scan_executor> _scan_executor;
};

}  // namespace pipeline
}  // namespace sirius
