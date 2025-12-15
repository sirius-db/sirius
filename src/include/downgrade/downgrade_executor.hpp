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
#include "data/data_repository.hpp"
#include "downgrade/downgrade_task.hpp"
#include "memory/memory_reservation.hpp"
#include "parallel/task_executor.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Executor specialized for performing memory downgrade operations across tier hierarchies.
 *
 * The downgrade_executor inherits from itask_executor and manages a pool of threads dedicated
 * to executing downgrade tasks that move data between different memory tiers (e.g., GPU to CPU,
 * CPU to disk). It uses a downgrade_task_queue for task scheduling and coordination.
 *
 * While similar to gpu_pipeline_executor in its threading model, it includes specialized
 * memory management logic required for efficient spilling operations and tier management.
 */
class downgrade_executor : public itask_executor {
 public:
  /**
   * @brief Constructs a new downgrade_executor with task execution configuration
   *
   * @param config Configuration for the task executor (thread count, retry policy, etc.)
   * @param data_repo_mgr Reference to the data repository for accessing and storing data batches
   */
  explicit downgrade_executor(task_executor_config config, data_repository_manager& data_repo_mgr)
    : itask_executor(sirius::make_unique<downgrade_task_queue>(), config),
      _data_repo_mgr(data_repo_mgr)
  {
  }

  /**
   * @brief Destructor for the downgrade_executor.
   */
  ~downgrade_executor() override = default;

  // Non-copyable but movable
  downgrade_executor(const downgrade_executor&)            = delete;
  downgrade_executor& operator=(const downgrade_executor&) = delete;
  downgrade_executor(downgrade_executor&&)                 = default;
  downgrade_executor& operator=(downgrade_executor&&)      = default;

  /**
   * @brief Schedules a downgrade task for execution
   *
   * This is a type-safe wrapper that converts the downgrade task to the base itask
   * interface and delegates to the parent's schedule method.
   *
   * @param downgrade_task The downgrade task to schedule for execution
   */
  void schedule_downgrade_task(sirius::unique_ptr<downgrade_task> downgrade_task)
  {
    // Convert to itask and use parent's schedule method
    schedule(std::move(downgrade_task));
  }

  /**
   * @brief Schedules a task for execution with downgrade-specific logic
   *
   * Overrides the base class schedule method to provide specialized scheduling
   * behavior for downgrade operations, including memory tier validation and
   * resource management.
   *
   * @param task The task to schedule (must be a downgrade_task)
   */
  void schedule(sirius::unique_ptr<itask> task) override;

  /**
   * @brief Main worker loop for executing downgrade tasks
   *
   * Each worker thread runs this loop to continuously pull and execute downgrade
   * tasks from the queue. Includes specialized error handling and resource cleanup
   * for memory tier operations.
   *
   * @param worker_id The unique identifier for this worker thread
   */
  void worker_loop(int worker_id) override;

  /**
   * @brief Starts the executor and initializes worker threads
   *
   * Initializes the thread pool and begins accepting tasks for execution.
   */
  void start() override;

  /**
   * @brief Stops the executor and cleanly shuts down worker threads
   *
   * Stops accepting new tasks and waits for all worker threads to complete
   * their current tasks before shutting down.
   */
  void stop() override;

 private:
  /**
   * @brief Safely casts itask to downgrade_task with type validation
   *
   * @param task The base task pointer to cast
   * @return downgrade_task* The casted downgrade_task pointer
   * @throws std::bad_cast if the task is not of type downgrade_task
   */
  downgrade_task* cast_to_downgrade_task(itask* task);

 private:
  data_repository_manager& _data_repo_mgr;  ///< Reference to the data repository manager for
                                            ///< accessing data during downgrade operations
};

}  // namespace parallel
}  // namespace sirius
