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

#include "helper/helper.hpp"
#include "task_executor.hpp"

#include <condition_variable>

namespace sirius {

/**
 * Interface for a Task Creator, can be extended to support various kinds of task creation policies.
 */
class itask_creator {
 public:
  itask_creator() : _running(false) {}

  virtual ~itask_creator() { stop(); }

  // Non-copyable and movable
  itask_creator(const itask_creator&)            = delete;
  itask_creator& operator=(const itask_creator&) = delete;
  itask_creator(itask_creator&&)                 = default;
  itask_creator& operator=(itask_creator&&)      = default;

  // Start worker threads
  virtual void start();

  // Stop accepting new tasks, and join worker threads.
  virtual void stop();

  virtual void signal();

  virtual void wait();

 protected:
  // Main thread loop.
  virtual void worker_loop();

  virtual uint64_t get_next_task_id();

 protected:
  sirius::atomic<bool> _running;
  sirius::unique_ptr<parallel::task_executor_thread> _thread;
  sirius::atomic<uint64_t> _next_task_id = 0;  ///< Atomic counter for generating unique task IDs
  sirius::mutex _mtx;                          ///< Mutex for synchronization
  std::condition_variable _cv;                 ///< Condition variable for thread coordination
  bool _ready = false;                         ///< Flag indicating readiness state
};

}  // namespace sirius
