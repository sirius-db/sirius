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

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/unique_ptr.hpp"

namespace sirius {
namespace parallel {

/**
 * Interface for concrete task local states.
 */
class ITaskLocalState {
public:
  virtual ~ITaskLocalState() = default;
};

/**
 * Interface for concrete task global states.
 */
class ITaskGlobalState {
public:
  virtual ~ITaskGlobalState() = default;
};

/**
 * Interface for concrete executor tasks.
 */
class ITask {
public:
  ITask(duckdb::unique_ptr<ITaskLocalState> local_state, duckdb::shared_ptr<ITaskGlobalState> global_state)
    : local_state_(std::move(local_state)), global_state_(global_state) {}

  virtual ~ITask() = default;

  // Non-copyable and movable.
  ITask(const ITask&) = delete;
  ITask& operator=(const ITask&) = delete;
  ITask(ITask&&) = default;
  ITask& operator=(ITask&&) = default;

  // Execution function.
  virtual void Execute() = 0;

protected:
  duckdb::unique_ptr<ITaskLocalState> local_state_;
  duckdb::shared_ptr<ITaskGlobalState> global_state_;
};

} // namespace parallel
} // namespace sirius
