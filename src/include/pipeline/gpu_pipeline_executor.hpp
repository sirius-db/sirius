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
#include "memory/memory_reservation.hpp"
#include "memory/memory_space.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_request.hpp"

#include <blockingconcurrentqueue.h>
#include <data/data_repository.hpp>

#include <queue>

namespace sirius {
namespace pipeline {

class pipeline_executor;

class local_task_buffer {
 public:
  local_task_buffer() = default;
  void produce(std::unique_ptr<sirius::parallel::itask> task);
  std::unique_ptr<sirius::parallel::itask> consume();
  void open();
  void close();

 private:
  mutable std::mutex _mtx;
  std::condition_variable _cv;
  std::queue<std::unique_ptr<sirius::parallel::itask>> _queue;
  std::atomic<bool> _is_open{false};  ///< Whether the queue is open for pushing/pulling tasks
};

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and uses a gpu_pipeline_task_queue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class gpu_pipeline_executor : public sirius::parallel::itask_executor {
 public:
  /**
   * @brief Constructs a new gpu_pipeline_executor with task execution configuration
   *
   * @param config Configuration for the task executor (thread count, retry policy, etc.)
   */
  explicit gpu_pipeline_executor(sirius::parallel::task_executor_config config,
                                 const cucascade::memory::memory_space* mem_space,
                                 pipeline_executor* pipeline_exec);

  /**
   * @brief Destructor for the gpu_pipeline_executor.
   */
  ~gpu_pipeline_executor() override = default;

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

  void on_start() override;
  void on_stop() override;

  /**
   * @brief Get the memory space view associated with this executor
   *
   * @return cucascade::memory::memory_space* Pointer to the memory space
   */
  cucascade::memory::memory_space* get_memory_space_view();

  /**
   * @brief Manager loop to consume task from local buffer and dispatch to the thread pool
   */
  void manager_loop();

  /**
   * @brief Submit a task request to task_request_queue
   */
  void submit_task_request(std::unique_ptr<task_request> request);

 private:
  /**
   * @brief Safely casts itask to gpu_pipeline_task with type validation
   *
   * @param task The itask pointer to cast
   * @return gpu_pipeline_task* The casted gpu_pipeline_task pointer
   * @throws std::bad_cast if the task is not of type gpu_pipeline_task
   */
  gpu_pipeline_task* cast_to_gpu_pipeline_task(sirius::parallel::itask* task);

  std::unique_ptr<std::thread> _gpu_pipeline_executor_manager_thread;
  std::unique_ptr<local_task_buffer> _local_task_buffer;
  pipeline_executor* _pipeline_exec;
  const cucascade::memory::memory_space*
    _memory_space_view;  // this is supposed to be the memory space
                         // associated with this pipeline executor
};

}  // namespace pipeline
}  // namespace sirius
