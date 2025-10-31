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

void itask_executor::start() {
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) {
    return;
  }
  on_start();
  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    _threads.push_back(
      sirius::make_unique<task_executor_thread>(sirius::make_unique<sirius::thread>(&itask_executor::worker_loop, this, i)));
  }
}

void itask_executor::stop() {
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) {
    return;
  }
  on_stop();
  for (auto& thread : _threads) {
    if (thread->_internal_thread->joinable()) {
      thread->_internal_thread->join();
    }
  }
  _threads.clear();
}

void itask_executor::schedule(sirius::unique_ptr<itask> task) {
  _task_queue->push(std::move(task));
}

void itask_executor::on_start() {
  _task_queue->open();
}

void itask_executor::on_stop() {
  _task_queue->close();
}

void itask_executor::on_task_error(int worker_id, sirius::unique_ptr<itask> task, const std::exception& e) {
  if (_config.retry_on_error) {
    schedule(std::move(task));
  } else {
    stop();
  }
}

void itask_executor::worker_loop(int worker_id) {
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
  }
}

} // namespace parallel
} // namespace sirius