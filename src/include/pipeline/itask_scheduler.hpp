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
 * See the License for the specific language governing permissions and limitations under the
 * License.
 */

#pragma once

#include "parallel/task.hpp"

#include <exception>
#include <memory>

namespace sirius::pipeline {

/// Receives tasks and terminal errors produced asynchronously by a task creator.
class itask_scheduler {
 public:
  virtual ~itask_scheduler() = default;

  virtual void schedule(std::unique_ptr<sirius::parallel::itask> task) = 0;
  virtual void terminate_query(std::exception_ptr error)               = 0;
};

}  // namespace sirius::pipeline
