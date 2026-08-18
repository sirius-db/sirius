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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace sirius::exec {

/**
 * @brief Receives the execution-stage events a @ref query_stage_manager
 *        publishes.
 *
 * Every hook is an observation, not a decision: they are @c noexcept, must not
 * block, and run on the creator/scheduler hot paths.  Implementations that need
 * to do real work should hand it off rather than do it inline.
 *
 * Thread safety: hooks are called concurrently from creator, scheduler and
 * executor threads.  Implementations must be safe under that; the defaults do
 * nothing and so trivially are.
 */
class query_stage_listener {
 public:
  query_stage_listener()          = default;
  virtual ~query_stage_listener() = default;

  query_stage_listener(query_stage_listener const&)            = delete;
  query_stage_listener& operator=(query_stage_listener const&) = delete;

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
};

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
 * Reporters call the @c notify_* entry points, which relay to every subscribed
 * @ref query_stage_listener.  Nothing here is virtual: the manager is the fixed
 * relay and the listener is the extension point.
 *
 * Lifetime: constructed by SiriusContext into a @c shared_ptr and handed to its
 * reporters by reference, each of which extends it via @c shared_from_this ---
 * so a reporter's handle is never null and never dangles.
 *
 * Thread safety: the listener set is guarded by a shared mutex.  Publishing
 * takes it shared, so the reporting threads do not serialise against each
 * other; @ref subscribe and @ref unsubscribe take it exclusively.  Subscription
 * is expected to happen before events flow, but the lock means a late one --
 * the per-query readahead registering while the scheduler's event loop is
 * live -- is safe rather than merely unlikely.
 */
class query_stage_manager : public std::enable_shared_from_this<query_stage_manager> {
 public:
  query_stage_manager()  = default;
  ~query_stage_manager() = default;

  query_stage_manager(query_stage_manager const&)            = delete;
  query_stage_manager& operator=(query_stage_manager const&) = delete;

  // -- subscription ----------------------------------------------------------
  //
  // Components that want these events subscribe rather than being reached into,
  // so the creator and scheduler keep one collaborator and know nothing about
  // who is listening.  The manager shares ownership of its listeners, so a
  // listener cannot die while still subscribed.

  void subscribe(std::shared_ptr<query_stage_listener> listener)
  {
    if (listener == nullptr) { return; }
    std::unique_lock g{_listeners_mtx};
    _listeners.push_back(std::move(listener));
  }

  /// Drops @p listener's subscription along with this manager's share of its
  /// ownership.  Blocks until any in-flight publish has finished, so the
  /// listener is not being called once this returns.
  void unsubscribe(query_stage_listener const* listener) noexcept
  {
    std::unique_lock g{_listeners_mtx};
    std::erase_if(_listeners, [listener](auto const& l) { return l.get() == listener; });
  }

  // -- reporting -------------------------------------------------------------

  void notify_task_created(query_id_t query_id,
                           std::size_t operator_id,
                           op::SiriusPhysicalOperatorType operator_type,
                           queue_priority priority) noexcept
  {
    std::shared_lock g{_listeners_mtx};
    for (auto const& l : _listeners) {
      l->on_task_created(query_id, operator_id, operator_type, priority);
    }
  }

  void notify_task_deployed(query_id_t query_id,
                            std::size_t operator_id,
                            op::SiriusPhysicalOperatorType operator_type,
                            int gpu_id) noexcept
  {
    std::shared_lock g{_listeners_mtx};
    for (auto const& l : _listeners) {
      l->on_task_deployed(query_id, operator_id, operator_type, gpu_id);
    }
  }

  void notify_failed_to_create_task(query_id_t query_id, std::size_t source_operator_id) noexcept
  {
    std::shared_lock g{_listeners_mtx};
    for (auto const& l : _listeners) {
      l->on_failed_to_create_task(query_id, source_operator_id);
    }
  }

  void notify_task_queue_empty() noexcept
  {
    std::shared_lock g{_listeners_mtx};
    for (auto const& l : _listeners) {
      l->on_task_queue_empty();
    }
  }

  void notify_pipeline_closed(query_id_t query_id,
                              std::size_t pipeline_id,
                              std::size_t source_operator_id) noexcept
  {
    std::shared_lock g{_listeners_mtx};
    for (auto const& l : _listeners) {
      l->on_pipeline_closed(query_id, pipeline_id, source_operator_id);
    }
  }

  void notify_executor_awaiting_task(int gpu_id) noexcept
  {
    std::shared_lock g{_listeners_mtx};
    for (auto const& l : _listeners) {
      l->on_executor_awaiting_task(gpu_id);
    }
  }

 private:
  mutable std::shared_mutex _listeners_mtx;
  std::vector<std::shared_ptr<query_stage_listener>> _listeners;
};

}  // namespace sirius::exec
