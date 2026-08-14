/*
 * Copyright 2026, Sirius Contributors.
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

#include "exec/queue_priority.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "query_id.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace sirius::exec {

/**
 * @brief Observer of where a query is in its execution, assembled from the
 *        points at which work is created, dispatched, and runs out.
 *
 * The task creator and the task scheduler each see one half of the picture: the
 * creator knows what work exists and why it could not make more, the scheduler
 * knows what got dispatched and when a GPU went hungry.  Neither can say on its
 * own whether a query is scan-bound, waiting on a barrier, or simply done.
 * Reporting both halves here is what lets that be answered in one place.
 *
 * Every hook is an observation, not a decision: they are @c noexcept, must not
 * block, and are called from the creator/scheduler hot paths.  Implementations
 * that need to do real work should hand it off rather than do it inline.
 *
 * Thread safety: hooks are called concurrently from creator, scheduler and
 * executor threads.  Implementations must be safe under that; the base does
 * nothing and so trivially is.
 */
class query_stage_manager {
 public:
  query_stage_manager()          = default;
  virtual ~query_stage_manager() = default;

  query_stage_manager(query_stage_manager const&)            = delete;
  query_stage_manager& operator=(query_stage_manager const&) = delete;

  /// A task has been created for @p operator_id and handed to the scheduler.
  virtual void on_task_created(query_id_t query_id,
                               std::size_t operator_id,
                               op::SiriusPhysicalOperatorType operator_type,
                               queue_priority priority) noexcept
  {
  }

  /// A task has been popped from the queue and pushed to the executor for
  /// @p gpu_id -- the point at which queued work becomes running work.
  virtual void on_task_deployed(query_id_t query_id,
                                std::size_t operator_id,
                                op::SiriusPhysicalOperatorType operator_type,
                                int gpu_id) noexcept
  {
  }

  /// A source operator could not find a next producer, so no task was created.
  /// Distinguishes "nothing to do yet" from "nothing left to do".
  virtual void on_failed_to_create_task(query_id_t query_id,
                                        std::size_t source_operator_id) noexcept
  {
  }

  /// The scheduler found its queue empty.  Says nothing about whether more work
  /// is coming -- pair with @c on_failed_to_create_task to tell those apart.
  virtual void on_task_queue_empty() noexcept {}

  /// A pipeline reached its closed state, so its source operator will produce
  /// no further tasks.
  virtual void on_pipeline_closed(query_id_t query_id,
                                  std::size_t pipeline_id,
                                  std::size_t source_operator_id) noexcept
  {
  }

  /// An executor asked for work for @p gpu_id and the scheduler had tasks but
  /// none it could send there.  Not the same as an empty queue: this is work
  /// existing but being unplaceable, i.e. a GPU idling against a non-empty
  /// queue.
  virtual void on_executor_awaiting_task(int gpu_id) noexcept {}

  // -- fan-out ---------------------------------------------------------------
  //
  // Components that want these events subscribe rather than being reached into,
  // so the creator and scheduler keep one collaborator and know nothing about
  // who is listening.  Listeners are raw pointers with a lifetime shorter than
  // this manager's; each must remove itself before it dies.

  void add_listener(query_stage_manager* listener)
  {
    if (listener == nullptr || listener == this) { return; }
    std::lock_guard g{_listeners_mtx};
    _listeners.push_back(listener);
  }

  void remove_listener(query_stage_manager* listener) noexcept
  {
    std::lock_guard g{_listeners_mtx};
    std::erase(_listeners, listener);
  }

  /// Deliver @p fn to every listener.  Called by the reporting hooks below.
  template <class Fn>
  void fan_out(Fn&& fn) noexcept
  {
    std::lock_guard g{_listeners_mtx};
    for (auto* l : _listeners) {
      fn(*l);
    }
  }

 private:
  mutable std::mutex _listeners_mtx;
  std::vector<query_stage_manager*> _listeners;
};

/// A @c query_stage_manager that forwards every hook to its listeners.  This is
/// what SiriusContext owns: the creator and scheduler report into it, and it
/// relays to whoever subscribed.
class fan_out_query_stage_manager final : public query_stage_manager {
 public:
  void on_task_created(query_id_t query_id,
                       std::size_t operator_id,
                       op::SiriusPhysicalOperatorType operator_type,
                       queue_priority priority) noexcept override
  {
    fan_out([&](query_stage_manager& l) {
      l.on_task_created(query_id, operator_id, operator_type, priority);
    });
  }

  void on_task_deployed(query_id_t query_id,
                        std::size_t operator_id,
                        op::SiriusPhysicalOperatorType operator_type,
                        int gpu_id) noexcept override
  {
    fan_out([&](query_stage_manager& l) {
      l.on_task_deployed(query_id, operator_id, operator_type, gpu_id);
    });
  }

  void on_failed_to_create_task(query_id_t query_id,
                                std::size_t source_operator_id) noexcept override
  {
    fan_out([&](query_stage_manager& l) {
      l.on_failed_to_create_task(query_id, source_operator_id);
    });
  }

  void on_task_queue_empty() noexcept override
  {
    fan_out([](query_stage_manager& l) { l.on_task_queue_empty(); });
  }

  void on_pipeline_closed(query_id_t query_id,
                          std::size_t pipeline_id,
                          std::size_t source_operator_id) noexcept override
  {
    fan_out([&](query_stage_manager& l) {
      l.on_pipeline_closed(query_id, pipeline_id, source_operator_id);
    });
  }

  void on_executor_awaiting_task(int gpu_id) noexcept override
  {
    fan_out([&](query_stage_manager& l) { l.on_executor_awaiting_task(gpu_id); });
  }
};

}  // namespace sirius::exec
