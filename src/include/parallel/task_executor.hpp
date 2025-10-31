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

#include "task_queue.hpp"
#include <condition_variable>
#include "helper/helper.hpp"

namespace sirius {
namespace parallel {

struct task_executor_thread {
  explicit task_executor_thread(sirius::unique_ptr<std::thread> thread)
    : _internal_thread(std::move(thread)) {}

  sirius::unique_ptr<std::thread> _internal_thread;
};

struct task_executor_config {
  int num_threads;
  bool retry_on_error;
};

/**
 * Interface for a thread pool used by different concrete executors like `gpu_pipeline_executor`, can be
 * extended to support various kinds of tasks and scheduling policies.
 */
class itask_executor {
public:
  itask_executor(sirius::unique_ptr<itask_queue> task_queue, task_executor_config config)
    : _task_queue(std::move(task_queue)), _config(config), _running(false) {}
  
  virtual ~itask_executor() {
    stop();
  }

  // Non-copyable and movable
  itask_executor(const itask_executor&) = delete;
  itask_executor& operator=(const itask_executor&) = delete;
  itask_executor(itask_executor&&) = default;
  itask_executor& operator=(itask_executor&&) = default;

  // Start worker threads
  virtual void start();

  // Stop accepting new tasks, and join worker threads.
  virtual void stop();

  // Schedule a task.
  virtual void schedule(sirius::unique_ptr<itask> task);

protected:
  // Helper functions.
  virtual void on_start();
  virtual void on_stop();
  virtual void on_task_error(int worker_id, sirius::unique_ptr<itask> task, const std::exception& e);

  // Main thread loop.
  virtual void worker_loop(int worker_id);

protected:
  sirius::unique_ptr<itask_queue> _task_queue;
  task_executor_config _config;
  sirius::atomic<bool> _running;
  sirius::vector<sirius::unique_ptr<task_executor_thread>> _threads;
};

} // namespace parallel
} // namespace sirius
