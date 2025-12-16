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

#include "creator/task_creator.hpp"

namespace sirius {

// spawn the internal thread that runs the task creator loop
void itask_creator::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _thread = sirius::make_unique<parallel::task_executor_thread>(
    sirius::make_unique<sirius::thread>(&itask_creator::worker_loop, this));
}

void itask_creator::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  // stop the internal thread
  if (_thread->_internal_thread && _thread->_internal_thread->joinable()) {
    _thread->_internal_thread->join();
    _thread->_internal_thread.reset();
  }
}

void itask_creator::worker_loop()
{
  // Default implementation does nothing
}

void itask_creator::signal()
{
  {
    std::lock_guard<std::mutex> lock(_mtx);
    _ready = true;
  }
  _cv.notify_one();
}

void itask_creator::wait()
{
  std::unique_lock<std::mutex> lock(_mtx);
  _cv.wait(lock, [this] { return _ready; });
  _ready = false;
}

uint64_t itask_creator::get_next_task_id() { return _next_task_id++; }

}  // namespace sirius
