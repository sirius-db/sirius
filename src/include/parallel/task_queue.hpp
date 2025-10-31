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

#include "task.hpp"

#include <memory>

namespace sirius {
namespace parallel {

/**
 * Interface for concrete task queues for customized scheduling policies.
 */
class itask_queue {
public:
  virtual ~itask_queue() = default;

  // Open the queue and start accepting new tasks.
  virtual void open() = 0;

  // Close the queue and stop processing new tasks.
  virtual void close() = 0;

  // Add a task to the queue.
  virtual void push(std::unique_ptr<itask> task) = 0;

  // Pull a task from the queue. Wait until a task available or the queue is closed.
  virtual std::unique_ptr<itask> pull() = 0;
};

} // namespace parallel
} // namespace sirius