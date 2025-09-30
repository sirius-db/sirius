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

#include "task_scheduler.hpp"

#include "duckdb/common/vector.hpp"

#include <atomic>
#include <thread>

namespace sirius {
namespace parallel {

/**
 * Interface for a thread pool used by different concrete executors like `GPUPipelineExecutor`, can be
 * extended to support various kinds of tasks and scheduling policies.
 */
class ITaskExecutor {
public:
  struct WorkerThread {
    explicit WorkerThread(duckdb::unique_ptr<std::thread> thread)
      : internal_thread_(std::move(thread)) {}

	  duckdb::unique_ptr<std::thread> internal_thread_;
  };

  struct Config {
    int num_threads = std::thread::hardware_concurrency();
    bool retry_on_error = true;
  };

  ITaskExecutor(duckdb::unique_ptr<ITaskScheduler> scheduler, Config config)
    : scheduler_(std::move(scheduler)), config_(config), running_(false) {}
  
  virtual ~ITaskExecutor() {
    Stop();
  }

  // Non-copyable and movable
  ITaskExecutor(const ITaskExecutor&) = delete;
  ITaskExecutor& operator=(const ITaskExecutor&) = delete;
  ITaskExecutor(ITaskExecutor&&) = default;
  ITaskExecutor& operator=(ITaskExecutor&&) = default;

  // Start worker threads
  virtual void Start();

  // Stop accepting new tasks, and join worker threads.
  virtual void Stop();

  // Schedule a task.
  virtual void Schedule(duckdb::unique_ptr<ITask> task);

private:
  // Helper functions.
  virtual void OnStart();
  virtual void OnStop();
  virtual void OnTaskError(int worker_id, duckdb::unique_ptr<ITask> task, const std::exception& e);

  // Main thread loop.
  virtual void WorkerLoop(int worker_id);

private:
  duckdb::unique_ptr<ITaskScheduler> scheduler_;
  Config config_;
  std::atomic<bool> running_;
  duckdb::vector<duckdb::unique_ptr<WorkerThread>> threads_;
};

} // namespace parallel
} // namespace sirius
