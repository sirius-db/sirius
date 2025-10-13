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

#include "parallel/task_executor.hpp"

namespace sirius {
namespace parallel {

void ITaskExecutor::Start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  OnStart();
  threads_.reserve(config_.num_threads);
  for (int i = 0; i < config_.num_threads; ++i) {
    threads_.push_back(
      std::make_unique<TaskExecutorThread>(std::make_unique<std::thread>(&ITaskExecutor::WorkerLoop, this, i)));
  }
}

void ITaskExecutor::Stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  OnStop();
  for (auto& thread : threads_) {
    if (thread->internal_thread_->joinable()) {
      thread->internal_thread_->join();
    }
  }
  threads_.clear();
}

void ITaskExecutor::Schedule(std::unique_ptr<ITask> task) {
  task_queue_->Push(std::move(task));
  total_tasks_.fetch_add(1);
}

void ITaskExecutor::Wait() {
  std::unique_lock<std::mutex> lock(finish_mutex_);
  finish_cv_.wait(lock, [&]() {
    return total_tasks_.load() == finished_tasks_.load();
  });
}

void ITaskExecutor::OnStart() {
  task_queue_->Open();
}

void ITaskExecutor::OnStop() {
  task_queue_->Close();
}

void ITaskExecutor::OnTaskError(int worker_id, std::unique_ptr<ITask> task, const std::exception& e) {
  if (config_.retry_on_error) {
    Schedule(std::move(task));
  } else {
    Stop();
  }
}

void ITaskExecutor::WorkerLoop(int worker_id) {
  while (true) {
    if (!running_.load()) {
      // Executor is stopped.
      break;
    }
    auto task = task_queue_->Pull();
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      task->Execute();
    } catch (const std::exception& e) {
      OnTaskError(worker_id, std::move(task), e);
    }
    {
      std::unique_lock<std::mutex> lock(finish_mutex_);
      finished_tasks_.fetch_add(1);
      if (total_tasks_.load() == finished_tasks_.load()) {
        finish_cv_.notify_one();
      }
    }
  }
}

} // namespace parallel
} // namespace sirius
