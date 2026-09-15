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

#include "event/event.hpp"
#include "event/query_event_publisher.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string_view>
#include <thread>

namespace sirius::event {

/**
 * @brief Receives the execution-stage events a @ref query_event_publisher
 *        publishes, on a thread of its own.
 *
 * The subscriber registers a mailbox in its constructor and drops it in its
 * destructor.  In between, @ref start puts a worker on that mailbox and @ref
 * stop takes it off; both are just gates on the worker.  Events published
 * while the worker is off ACCUMULATE in the mailbox and are drained the next
 * time it comes on --- there is one subscription per subscriber, and its
 * lifetime is the subscriber's own.
 *
 * The hooks below run on the subscriber's worker, not on the creator,
 * scheduler or executor thread that raised the event, so an implementation is
 * free to do real work in them and cannot stall execution by doing so.  The
 * flip side is that a hook is a report of something that already happened and
 * may no longer be true by the time it is read.  Implementations should treat
 * the arguments as a snapshot rather than as live state.
 *
 * Thread safety: hooks are called from the subscriber's worker and nowhere
 * else, so they are serialised against each other and see events in publication
 * order.  They still race against the implementation's own public API, which
 * is called from elsewhere.
 *
 * Subclassing: the worker dispatches into virtual hooks, so it must be stopped
 * before the derived part of the object is destroyed.  A derived class whose
 * destructor can run while events are in flight must call @ref stop itself
 * --- the base destructor is too late, the derived members are already gone
 * by then.
 */
class query_event_subscriber {
 public:
  /// Register a mailbox subscribed to @p events on @p publisher.  Registration
  /// happens here, so @p events is decided by the derived type's constructor;
  /// listing an event but not overriding its hook lands the payload on the
  /// no-op base implementation, which is the trade for @c override still
  /// catching a misspelt or drifted callback signature.
  ///
  /// Throws @c std::invalid_argument if @p publisher is not owned by a @c
  /// shared_ptr, since the subscriber holds the publisher by weak reference
  /// so it can never keep it alive.
  query_event_subscriber(query_event_publisher& publisher, std::span<event_type const> events);

  /// Delegating overload so callers can write @c {event_type::foo, ...} at the
  /// call site rather than forcing an array declaration.
  query_event_subscriber(query_event_publisher& publisher,
                         std::initializer_list<event_type> events);

  virtual ~query_event_subscriber();

  query_event_subscriber(query_event_subscriber const&)            = delete;
  query_event_subscriber& operator=(query_event_subscriber const&) = delete;

  /// Start the worker draining the mailbox.  Idempotent: a second call while
  /// the worker runs does nothing.  A no-op once @ref stop or the publisher
  /// has stopped --- a subscriber's lifecycle is start-once, stop-once, so
  /// once torn down it stays that way.
  ///
  /// Anything published between construction and this call is still in the
  /// mailbox and gets replayed in order.
  void start();

  /// Stop the worker and join it.  Terminal: subsequent @ref start calls are
  /// no-ops.  Does NOT drop the registration --- that happens once, at
  /// destruction.  Safe to call when not started, and safe to call twice.
  /// Does not stop the publisher.
  void stop() noexcept;

  /// Whether the worker is up.  Named apart from any @c is_running a subclass
  /// has for its own work, which is a different question.
  [[nodiscard]] bool is_subscribed() const noexcept { return _worker.joinable(); }

  /// Name for logs and for the worker's thread name, which is
  /// @c "subscriber-<name>" truncated to what pthread accepts.
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  // -- hooks -----------------------------------------------------------------
  //
  // One per @ref event_type; the default body is a no-op so a subscriber that
  // only cares about a subset can leave the rest alone.  A hook the subscriber
  // did not name in its constructor is never called: the publisher filters at
  // registration time.

  /// A task has been created for @p operator_id and handed to the scheduler.
  virtual void on_task_created(event_id_t,
                               timestamp_t,
                               query_id_t query_id,
                               std::size_t operator_id,
                               op::SiriusPhysicalOperatorType operator_type,
                               exec::queue_priority priority) noexcept;

  /// A task has been popped from the queue and pushed to the executor for
  /// @p gpu_id --- the point at which queued work becomes running work.
  virtual void on_task_deployed(event_id_t,
                                timestamp_t,
                                query_id_t query_id,
                                std::size_t operator_id,
                                op::SiriusPhysicalOperatorType operator_type,
                                int gpu_id) noexcept;

  /// No task was created for @p source_operator_id: the walk from it found
  /// nobody able to produce.  Distinguishes "nothing to do yet" from "nothing
  /// left to do".
  ///
  /// @p failed_operator_id is where the walk actually stopped.  It differs
  /// from @p source_operator_id whenever the walk descended through operators
  /// that were themselves waiting on input, and it is the one that says where
  /// the pipeline is stuck --- the source may simply be waiting on it.  The
  /// two are equal when the source itself could not produce.
  virtual void on_failed_to_create_task(event_id_t,
                                        timestamp_t,
                                        query_id_t query_id,
                                        std::size_t source_operator_id,
                                        std::size_t failed_operator_id) noexcept;

  /// The scheduler found its queue empty.  Says nothing about whether more
  /// work is coming --- pair with @c on_failed_to_create_task to tell those
  /// apart.
  virtual void on_task_queue_empty(event_id_t, timestamp_t) noexcept;

  /// A pipeline reached its closed state, so its source operator will produce
  /// no further tasks.
  virtual void on_pipeline_closed(event_id_t,
                                  timestamp_t,
                                  query_id_t query_id,
                                  std::size_t pipeline_id,
                                  std::size_t source_operator_id) noexcept;

  /// An executor asked for work for @p gpu_id and the scheduler had tasks but
  /// none it could send there.  Not the same as an empty queue: this is work
  /// existing but being unplaceable, i.e. a GPU idling against a non-empty
  /// queue.
  virtual void on_executor_awaiting_task(event_id_t, timestamp_t, int gpu_id) noexcept;

  /// An executor could not reserve the memory the task from @p operator_id
  /// needs on @p gpu_id, and is spilling to free @p shortfall_bytes before it
  /// can run.  That task is parked for as long as that takes, so the GPU is
  /// about to do no work at all --- which makes it the one moment the device's
  /// IO path is unambiguously free.
  virtual void on_memory_downgrade_for_task(event_id_t,
                                            timestamp_t,
                                            query_id_t query_id,
                                            std::size_t operator_id,
                                            int gpu_id,
                                            std::size_t shortfall_bytes) noexcept;

  /// An executor could not reserve @p bytes_needed for the task from
  /// @p operator_id on @p gpu_id and is about to block until the memory frees
  /// up.  Distinct from @ref on_memory_downgrade_for_task, which is the
  /// executor actively spilling to make room: here it is simply waiting on
  /// someone else to release.  Either way the task is parked and the GPU is
  /// about to do no work, which is what makes it worth reporting.
  ///
  /// Raised BEFORE the blocking call, because a subscriber told only once the
  /// wait is over learns nothing it can act on.
  virtual void on_wait_for_memory_for_task(event_id_t,
                                           timestamp_t,
                                           query_id_t query_id,
                                           std::size_t operator_id,
                                           int gpu_id,
                                           std::size_t bytes_needed) noexcept;

 protected:
  /// Invoked once when the publisher requests stop, on the thread that called
  /// @ref query_event_publisher::stop --- not on the worker, which may still
  /// be finishing an event.  The hook for tearing down whatever the subscriber
  /// was driving; the worker's own exit needs no help.
  virtual void on_stop_requested() noexcept;

 private:
  /// Drain until stopped, replaying each event into its hook.
  ///
  /// The timed wait rather than a plain blocking one is what makes the token a
  /// real stop signal: a publisher that goes away without pushing the sentinel
  /// still gets the worker out within the poll interval, instead of leaving a
  /// thread parked forever on a queue nobody will write to again.
  void run() noexcept;

  /// Replay one event into the hook that matches its tag.  Kept out of line:
  /// the @c if @c constexpr chain is exhaustive over @ref event_type, so a new
  /// event that forgets an arm here fails to compile rather than being
  /// dropped.
  void dispatch(query_events const& event) noexcept;

  /// Set the worker's thread name; hides that it uses the shared thread-name
  /// helper and its 15-byte truncation.
  void set_thread_name() noexcept;

  /// Weak so a subscriber never keeps the publisher alive; used only to
  /// deregister from the destructor.
  std::weak_ptr<query_event_publisher> _publisher;
  std::shared_ptr<event_queue> _queue;
  std::stop_token _stop_token;
  std::optional<std::stop_callback<std::function<void()>>> _stop_cb;
  /// Cleared by @ref stop so the worker leaves without draining the backlog.
  std::atomic<bool> _draining{true};
  /// Latched by @ref stop; makes the subscriber's lifecycle one-shot so
  /// @ref start after it is a no-op instead of a restart.
  std::atomic<bool> _stopped{false};
  std::jthread _worker;
};

}  // namespace sirius::event
