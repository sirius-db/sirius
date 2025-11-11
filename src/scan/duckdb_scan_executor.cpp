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

// sirius
#include <scan/duckdb_scan_executor.hpp>

namespace sirius::parallel {

void duckdb_scan_executor::schedule(std::unique_ptr<itask> task)
{
  itask_executor::schedule(std::move(task));
  _total_tasks.fetch_add(1, std::memory_order_relaxed);
}

void duckdb_scan_executor::wait()
{
  std::unique_lock<sirius::mutex> lock(_finish_mutex);
  _finish_cv.wait(lock, [&]() { return _total_tasks.load() == _finished_tasks.load(); });
}

void duckdb_scan_executor::worker_loop(int32_t worker_id)
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    auto task = _task_queue->pull();
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      task->execute();
    } catch (const std::exception& e) {
      on_task_error(worker_id, std::move(task), e);
    }
    {
      std::unique_lock<sirius::mutex> lock(_finish_mutex);
      _finished_tasks.fetch_add(1);
      if (_total_tasks.load() == _finished_tasks.load()) { _finish_cv.notify_one(); }
    }
  }
}

}  // namespace sirius::parallel