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

#include "exec/config.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "exec/kiosk.hpp"
#include "exec/thread_pool.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_request.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <queue>
#include <thread>

namespace sirius {
namespace pipeline {

class pipeline_executor;

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and uses a gpu_pipeline_task_queue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class gpu_pipeline_executor {
 public:
  /**
   * @brief Constructs a new gpu_pipeline_executor with task execution configuration
   *
   * @param config Configuration for the task executor (thread count, retry policy, etc.)
   */
  explicit gpu_pipeline_executor(exec::thread_pool_config config,
                                 cucascade::memory::memory_space* mem_space,
                                 pipeline_executor* pipeline_exec);

  /**
   * @brief Destructor for the gpu_pipeline_executor.
   */
  ~gpu_pipeline_executor() = default;

  // Non-copyable but movable
  gpu_pipeline_executor(const gpu_pipeline_executor&)            = delete;
  gpu_pipeline_executor& operator=(const gpu_pipeline_executor&) = delete;
  gpu_pipeline_executor(gpu_pipeline_executor&&)                 = delete;
  gpu_pipeline_executor& operator=(gpu_pipeline_executor&&)      = delete;

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

 private:
  /**
   * @brief Manager loop to consume task from local buffer and dispatch to the thread pool
   */
  void manager_loop();

  /**
   * @brief Submit a task request to task_request_queue
   */
  void submit_task_request();

  /**
   * @brief Safely casts itask to gpu_pipeline_task with type validation
   *
   * @param task The itask pointer to cast
   * @return gpu_pipeline_task* The casted gpu_pipeline_task pointer
   * @throws std::bad_cast if the task is not of type gpu_pipeline_task
   */
  gpu_pipeline_task* cast_to_gpu_pipeline_task(sirius::parallel::itask* task);

  std::atomic<bool> _running{false};
  exec::thread_pool_config _config;
  exec::kiosk _kiosk;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  exec::interruptible_mpmc<std::unique_ptr<sirius::parallel::itask>> _task_queue;
  std::thread _manager_thread;
  pipeline_executor* _pipeline_exec;
  cucascade::memory::memory_space* _memory_space;
};

}  // namespace pipeline
}  // namespace sirius
