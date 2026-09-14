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

#include "blockingconcurrentqueue.h"
#include "event/event.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stop_token>
#include <tuple>
#include <utility>
#include <vector>

namespace sirius::event {

class query_event_subscriber;

using event_queue = duckdb_moodycamel::BlockingConcurrentQueue<std::shared_ptr<query_events>>;

/// One subscriber's mailbox plus the publisher-wide stop token.  The queue is
/// what the publisher pushes into; the token is what the subscriber parks
/// against while waiting.
struct subscriber_registration {
  std::shared_ptr<event_queue> queue;
  std::stop_token stop_token;
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
 * Reporters call the @c publish_* entry points, which build the event once and
 * hand a reference to it to every registered subscriber's queue.  Publishing is
 * therefore a push and never a callback: a reporter is never made to wait on
 * what a subscriber does with the event, which matters because these are raised
 * from the creator, scheduler and executor hot paths.  Nothing here is virtual:
 * the publisher is the fixed relay and the subscriber is the extension point.
 *
 * Registration is not a public entry point of this class.  A @ref
 * query_event_subscriber registers itself in its own constructor and drops
 * itself in its own destructor, and nothing else needs to touch the routing
 * table --- so those two operations are private and only that friend can reach
 * them.  Callers hold a subscriber, not a mailbox.
 *
 * Lifetime: constructed by SiriusContext into a @c shared_ptr and handed to its
 * reporters by reference, each of which extends it via @c shared_from_this ---
 * so a reporter's handle is never null and never dangles.  Queues are owned by
 * their subscribers and only referenced here, so the publisher never keeps a
 * subscriber alive.
 *
 * Thread safety: the queue set is guarded by a shared mutex.  Publishing takes
 * it shared, so the reporting threads do not serialise against each other;
 * registration and deregistration take it exclusively.
 */
class query_event_publisher : public std::enable_shared_from_this<query_event_publisher> {
 public:
  query_event_publisher()  = default;
  ~query_event_publisher() = default;

  query_event_publisher(query_event_publisher const&)            = delete;
  query_event_publisher& operator=(query_event_publisher const&) = delete;

  /// Stop every subscriber and close the publisher to further publishing.
  /// Requests stop on the shared token and pushes the null sentinel to each
  /// queue, which is what wakes a subscriber parked on an empty mailbox;
  /// subsequent @c publish_* calls are no-ops.  Does not join the subscribers:
  /// a subscriber owns its own thread and joins it in its own destructor.
  void stop() noexcept;

  // -- reporting -------------------------------------------------------------
  //
  // One entry point per @ref event_type.  The wrappers exist so a reporter
  // names the event by intent ("task deployed") rather than by wiring
  // ("publish a task_deployed_event") --- and so the private @c publish
  // template stays private.

  void publish_task_created(query_id_t query_id,
                            std::size_t operator_id,
                            op::SiriusPhysicalOperatorType operator_type,
                            exec::queue_priority priority) noexcept;

  void publish_task_deployed(query_id_t query_id,
                             std::size_t operator_id,
                             op::SiriusPhysicalOperatorType operator_type,
                             int gpu_id) noexcept;

  void publish_failed_to_create_task(query_id_t query_id,
                                     std::size_t source_operator_id,
                                     std::size_t failed_operator_id) noexcept;

  void publish_task_queue_empty() noexcept;

  void publish_pipeline_closed(query_id_t query_id,
                               std::size_t pipeline_id,
                               std::size_t source_operator_id) noexcept;

  void publish_executor_awaiting_task(int gpu_id) noexcept;

  void publish_memory_downgrade_for_task(query_id_t query_id,
                                         std::size_t operator_id,
                                         int gpu_id,
                                         std::size_t shortfall_bytes) noexcept;

  void publish_wait_for_memory_for_task(query_id_t query_id,
                                        std::size_t operator_id,
                                        int gpu_id,
                                        std::size_t bytes_needed) noexcept;

 private:
  friend class query_event_subscriber;

  /// Mint a mailbox subscribed to @p events and nothing else, and add it to
  /// the routing table.  Called by the subscriber base in its constructor.
  ///
  /// Publishing walks one subscriber list per event, so a subscriber that
  /// names two events is skipped outright by the other six rather than being
  /// handed a payload it will drop.  With a handful of subscribers that is a
  /// rounding error; it is worth having because the cost grows with
  /// subscribers x events while the useful work does not.
  ///
  /// Registering after events have started flowing is safe; the subscriber
  /// simply misses what was published before it arrived.  Registering after
  /// @ref stop yields an already-stopped token and a queue nothing will ever
  /// be pushed to, so the subscriber never gets past a no-op @c start.
  [[nodiscard]] subscriber_registration register_subscriber(std::span<event_type const> events);

  /// Drop @p queue's registration.  Blocks until any in-flight publish has
  /// finished, so no further event reaches @p queue once this returns.  Called
  /// by the subscriber base in its destructor.
  void unregister_subscriber(std::shared_ptr<event_queue> const& queue) noexcept;

  /// Build the event once and fan a reference to it out to every mailbox.
  ///
  /// The empty check ahead of the allocation is not just an optimisation: with
  /// nobody listening these entry points sit on hot paths and should cost a
  /// lock and a branch, not a heap allocation.  Publishing is @c noexcept, so a
  /// failed allocation drops the event rather than propagating out into a
  /// reporter that has no way to handle it.
  ///
  /// The template body stays in the header even though nothing else does: the
  /// @c publish_* wrappers in the .cpp instantiate it, and moving it out would
  /// mean spelling every instantiation there by hand.
  template <typename Event, typename... Args>
  void publish(Args&&... args) noexcept
  {
    auto const event_id  = _next_event_id.fetch_add(1, std::memory_order_relaxed);
    auto const timestamp = std::chrono::system_clock::now();
    try {
      std::shared_lock g{_queues_mtx};
      auto const& subscribers = _by_event[event_index_v<Event>];
      if (subscribers.empty()) { return; }
      auto payload = std::make_shared<query_events>(
        Event{event_id, timestamp, typename Event::param_type{std::forward<Args>(args)...}});
      for (auto* q : subscribers) {
        q->enqueue(payload);
      }
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Telemetry is not worth failing execution over.
    }
  }

  inline static std::atomic<event_id_t> _next_event_id{0};

  /// Shared by every subscriber: one request_stop takes them all down together.
  std::stop_source _stop_source;

  mutable std::shared_mutex _queues_mtx;
  /// Owns the registered queues; the buckets below only point into it.
  std::vector<std::shared_ptr<event_queue>> _queues;
  /// One subscriber list per event, indexed by @ref event_index_v.  This is
  /// what makes a publish cost only the subscribers that asked for that event.
  std::array<std::vector<event_queue*>, n_query_events> _by_event;
  bool _stopped{false};
};

}  // namespace sirius::event
