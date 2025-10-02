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

#include "duckdb/common/helper.hpp"

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
      duckdb::make_uniq<TaskExecutorThread>(duckdb::make_uniq<std::thread>(&ITaskExecutor::WorkerLoop, this, i)));
  }
}

void ITaskExecutor::Stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  OnStop();
  for (auto& thread : threads_) {
    thread->internal_thread_->join();
  }
  threads_.clear();
}

void ITaskExecutor::Schedule(duckdb::unique_ptr<ITask> task) {
  scheduler_->Push(std::move(task));
}

void ITaskExecutor::OnStart() {
  scheduler_->Open();
}

void ITaskExecutor::OnStop() {
  scheduler_->Close();
}

void ITaskExecutor::OnTaskError(int worker_id, duckdb::unique_ptr<ITask> task, const std::exception& e) {
  if (config_.retry_on_error) {
    scheduler_->Push(std::move(task));
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
    auto task = scheduler_->Pull();
    if (task == nullptr) {
      // Scheduler is closed.
      break;
    }
    try {
      task->Execute();
    } catch (const std::exception& e) {
      OnTaskError(worker_id, std::move(task), e);
    }
  }
}

} // namespace parallel
} // namespace sirius
