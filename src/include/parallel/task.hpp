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

#include "exec/work_tracker.hpp"
#include "helper/helper.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/memory/memory_reservation.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace sirius {
namespace parallel {

/**
 * Interface for concrete task local states.
 */
class itask_local_state {
 public:
  virtual ~itask_local_state() = default;

  template <class TargetType>
  TargetType& cast()
  {
    DynamicCastCheck<TargetType>(this);
    return reinterpret_cast<TargetType&>(*this);
  }

  template <class TargetType>
  const TargetType& cast() const
  {
    DynamicCastCheck<TargetType>(this);
    return reinterpret_cast<const TargetType&>(*this);
  }
};

/**
 * Interface for concrete task global states.
 */
class itask_global_state {
 public:
  virtual ~itask_global_state() = default;

  template <class TargetType>
  TargetType& cast()
  {
    DynamicCastCheck<TargetType>(this);
    return reinterpret_cast<TargetType&>(*this);
  }

  template <class TargetType>
  const TargetType& cast() const
  {
    DynamicCastCheck<TargetType>(this);
    return reinterpret_cast<const TargetType&>(*this);
  }
};

/**
 * Interface for concrete executor tasks.
 */
class itask {
  //! Query-work slot. Declared first so it is released after task state and input batches.
  exec::work_tracker::slot _work_slot;

 public:
  itask(uint64_t task_id,
        std::unique_ptr<itask_local_state> local_state,
        std::shared_ptr<itask_global_state> global_state)
    : _task_id(task_id),
      _local_state(std::move(local_state)),
      _global_state(std::move(global_state))
  {
  }

  virtual ~itask() = default;

  // Non-copyable and non-movable. Tasks are moved by unique_ptr, not by object move.
  itask(const itask&)            = delete;
  itask& operator=(const itask&) = delete;
  itask(itask&&)                 = delete;
  itask& operator=(itask&&)      = delete;

  // Execution function.
  virtual void execute(rmm::cuda_stream_view stream) = 0;

  template <typename T>
  T* as() noexcept
  {
    return dynamic_cast<T*>(this);
  }

  template <typename T>
  const T* as() const noexcept
  {
    return dynamic_cast<const T*>(this);
  }

  template <typename T>
  [[nodiscard]] bool is() const noexcept
  {
    return dynamic_cast<const T*>(this) != nullptr;
  }

  itask_local_state* local_state() noexcept { return _local_state.get(); }
  [[nodiscard]] itask_global_state* global_state() noexcept { return _global_state.get(); }
  [[nodiscard]] uint64_t get_task_id() const noexcept { return _task_id; }

  /// Set this task's query-work slot.
  void set_work_slot(exec::work_tracker::slot slot) noexcept { _work_slot = std::move(slot); }

  /// Move the slot to a continuation of this task.
  [[nodiscard]] exec::work_tracker::slot take_work_slot() noexcept { return std::move(_work_slot); }

 protected:
  uint64_t _task_id;
  std::unique_ptr<itask_local_state> _local_state;
  std::shared_ptr<itask_global_state> _global_state;
};

}  // namespace parallel
}  // namespace sirius
