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

#include "downgrade/downgrade_executor.hpp"

namespace sirius {
namespace parallel {

void downgrade_executor::schedule(std::unique_ptr<itask> task)
{
  // Downgrade-specific scheduling logic
  auto downgrade_task = cast_to_downgrade_task(task.get());
  if (!downgrade_task) {
    // If it's not a downgrade_task, use the parent's implementation
    itask_executor::schedule(std::move(task));
    return;
  }

  // Schedule the downgrade task using the parent's method
  itask_executor::schedule(std::move(task));
}

downgrade_task* downgrade_executor::cast_to_downgrade_task(itask* task)
{
  // Safely cast to downgrade_task
  return dynamic_cast<downgrade_task*>(task);
}

void downgrade_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    _threads.push_back(std::make_unique<task_executor_thread>(
      std::make_unique<std::thread>(&downgrade_executor::worker_loop, this, i)));
  }
}

void downgrade_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  on_stop();
  for (auto& thread : _threads) {
    if (thread->_internal_thread->joinable()) { thread->_internal_thread->join(); }
  }
  _threads.clear();
}

void downgrade_executor::worker_loop(int worker_id)
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
  }
}

}  // namespace parallel
}  // namespace sirius
