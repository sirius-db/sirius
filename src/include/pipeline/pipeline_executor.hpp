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
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_request.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>

namespace sirius {
namespace pipeline {

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and uses a gpu_pipeline_task_queue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class pipeline_executor : public sirius::parallel::itask_executor {
 public:
  /**
   * @brief Constructs a new pipeline_executor with task execution configuration
   *
   * @param pipeline_config Configuration for the pipeline executor (thread count, retry policy,
   * etc.)
   * @param gpu_executor_config Configuration for the GPU pipeline executor
   * @param num_gpus Number of GPU executors to create
   */
  explicit pipeline_executor(const parallel::task_executor_config& gpu_task_executor_config,
                             sirius::memory::sirius_memory_reservation_manager& mem_mgr);

  /**
   * @brief Destructor for the gpu_pipeline_executor.
   */
  ~pipeline_executor() override = default;

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
  void schedule(std::unique_ptr<sirius::parallel::itask> task) override;

  /**
   * @brief Main worker loop for executing GPU pipeline tasks
   *
   * Each worker thread runs this loop to continuously pull and execute GPU
   * pipeline tasks from the queue. Handles GPU-specific operations including
   * kernel launches, memory transfers, and synchronization.
   *
   * @param worker_id The unique identifier for this worker thread
   */
  void worker_loop(int worker_id) override;

  void on_start() override;
  void on_stop() override;

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

  /**
   * @brief Dispatch a task to a specific GPU executor based on GPU ID
   *
   * @param task The task to schedule
   * @param gpu_id The GPU ID to which the task should be scheduled
   */
  void dispatch_to_gpu_executor(std::unique_ptr<sirius::parallel::itask> task, int gpu_id);

  /**
   * @brief Submit a task request to task_request_queue
   */
  void submit_task_request(std::unique_ptr<task_request> request);

 private:
  std::vector<std::unique_ptr<gpu_pipeline_executor>> _gpu_executors;  ///< Vector of GPU executors
  std::unique_ptr<task_request_queue> _task_request_queue;
};

}  // namespace pipeline
}  // namespace sirius
