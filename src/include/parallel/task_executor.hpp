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

#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace sirius {
namespace parallel {

struct TaskExecutorThread {
  explicit TaskExecutorThread(std::unique_ptr<std::thread> thread)
    : internal_thread_(std::move(thread)) {}

  std::unique_ptr<std::thread> internal_thread_;
};

struct TaskExecutorConfig {
  int num_threads;
  bool retry_on_error;
};

/**
 * Interface for a thread pool used by different concrete executors like `GPUPipelineExecutor`, can be
 * extended to support various kinds of tasks and scheduling policies.
 */
class ITaskExecutor {
public:
  ITaskExecutor(std::unique_ptr<ITaskQueue> task_queue, TaskExecutorConfig config)
    : task_queue_(std::move(task_queue)), config_(config), running_(false) {}
  
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
  virtual void Schedule(std::unique_ptr<ITask> task);

  // Wait until all tasks are finished.
  virtual void Wait();

private:
  // Helper functions.
  virtual void OnStart();
  virtual void OnStop();
  virtual void OnTaskError(int worker_id, std::unique_ptr<ITask> task, const std::exception& e);

  // Main thread loop.
  virtual void WorkerLoop(int worker_id);

private:
  std::unique_ptr<ITaskQueue> task_queue_;
  TaskExecutorConfig config_;
  std::atomic<bool> running_;
  std::vector<std::unique_ptr<TaskExecutorThread>> threads_;
  std::atomic<uint64_t> total_tasks_ = 0;
  std::atomic<uint64_t> finished_tasks_ = 0;
  std::mutex finish_mutex_;
  std::condition_variable finish_cv_;
};

} // namespace parallel
} // namespace sirius
