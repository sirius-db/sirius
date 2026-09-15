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

#include "event/query_event_subscriber.hpp"

#include "exec/thread_util.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sirius::event {
namespace {

/// Always-false, but dependent on its argument, so the exhaustiveness
/// @c static_assert in the dispatch only fires for a tag with no arm.
template <event_type>
inline constexpr bool unhandled_event_type = false;

}  // namespace

// ---------------------------------------------------------------------------
// construction / destruction
// ---------------------------------------------------------------------------

query_event_subscriber::query_event_subscriber(query_event_publisher& publisher,
                                               std::span<event_type const> events)
  : _publisher(publisher.weak_from_this())
{
  if (_publisher.expired()) {
    throw std::invalid_argument("query_event_publisher must be owned by a shared_ptr");
  }
  // Registering here rather than in @c start decouples subscription from the
  // worker's lifetime: events published while the worker is off still land in
  // the mailbox and are drained on the next start.  There is one subscription
  // per subscriber, and it is the subscriber's own lifetime.
  auto registration = publisher.register_subscriber(events);
  _queue            = std::move(registration.queue);
  _stop_token       = std::move(registration.stop_token);
}

query_event_subscriber::query_event_subscriber(query_event_publisher& publisher,
                                               std::initializer_list<event_type> events)
  : query_event_subscriber(publisher, std::span<event_type const>{events.begin(), events.size()})
{
}

query_event_subscriber::~query_event_subscriber()
{
  // Order matters: the worker may still be running a hook against the derived
  // object, so it has to be down before the base ends.  Only then is the
  // routing pointer safe to drop; leaving it in place would let a concurrent
  // publish write into a queue whose subscriber is halfway through teardown.
  stop();
  if (auto publisher = _publisher.lock()) { publisher->unregister_subscriber(_queue); }
}

// ---------------------------------------------------------------------------
// lifecycle (just a gate on the worker)
// ---------------------------------------------------------------------------

void query_event_subscriber::start()
{
  if (_worker.joinable()) { return; }
  if (_stopped.load(std::memory_order_relaxed)) { return; }  // torn down for good
  if (_stop_token.stop_requested()) { return; }              // publisher already stopped
  _draining.store(true, std::memory_order_relaxed);
  // Armed here rather than in the constructor so a subscriber that is never
  // started never gets the callback -- and so @ref on_stop_requested cannot
  // fire before the derived object is fully built.
  _stop_cb.emplace(_stop_token, [this] { on_stop_requested(); });
  _worker = std::jthread([this] { run(); });
  set_thread_name();
}

void query_event_subscriber::stop() noexcept
{
  // Latch first so a concurrent @ref start (racing with our teardown from a
  // different thread) turns into a no-op rather than resurrecting the worker
  // we are about to join.
  _stopped.store(true, std::memory_order_relaxed);
  if (!_worker.joinable()) {
    _stop_cb.reset();
    return;
  }
  // The sentinel wakes a worker parked on an empty mailbox; the flag stops
  // it from working through whatever is queued behind the sentinel.
  // A stale sentinel left in the queue would kill a restart, but subscribers
  // are one-shot: once stopped, @ref start is a no-op, so there is no next
  // worker to be tripped by it.
  _draining.store(false, std::memory_order_relaxed);
  std::ignore = _queue->try_enqueue(nullptr);  // see query_event_publisher::stop
  _worker.join();
  _stop_cb.reset();
}

// ---------------------------------------------------------------------------
// hook defaults (empty --- a subscriber only overrides what it cares about)
// ---------------------------------------------------------------------------

void query_event_subscriber::on_task_created(event_id_t,
                                             timestamp_t,
                                             query_id_t,
                                             std::size_t,
                                             op::SiriusPhysicalOperatorType,
                                             exec::queue_priority) noexcept
{
}

void query_event_subscriber::on_task_deployed(
  event_id_t, timestamp_t, query_id_t, std::size_t, op::SiriusPhysicalOperatorType, int) noexcept
{
}

void query_event_subscriber::on_failed_to_create_task(
  event_id_t, timestamp_t, query_id_t, std::size_t, std::size_t) noexcept
{
}

void query_event_subscriber::on_task_queue_empty(event_id_t, timestamp_t) noexcept {}

void query_event_subscriber::on_pipeline_closed(
  event_id_t, timestamp_t, query_id_t, std::size_t, std::size_t) noexcept
{
}

void query_event_subscriber::on_executor_awaiting_task(event_id_t, timestamp_t, int) noexcept {}

void query_event_subscriber::on_memory_downgrade_for_task(
  event_id_t, timestamp_t, query_id_t, std::size_t, int, std::size_t) noexcept
{
}

void query_event_subscriber::on_wait_for_memory_for_task(
  event_id_t, timestamp_t, query_id_t, std::size_t, int, std::size_t) noexcept
{
}

void query_event_subscriber::on_stop_requested() noexcept {}

// ---------------------------------------------------------------------------
// worker
// ---------------------------------------------------------------------------

void query_event_subscriber::run() noexcept
{
  using namespace std::chrono_literals;
  constexpr auto poll_interval = 100ms;

  while (_draining.load(std::memory_order_relaxed) && !_stop_token.stop_requested()) {
    std::shared_ptr<query_events> event;
    if (!_queue->wait_dequeue_timed(event, poll_interval)) { continue; }
    if (event == nullptr) { break; }  // close sentinel
    dispatch(*event);
  }
}

void query_event_subscriber::dispatch(query_events const& event) noexcept
{
  std::visit(
    [this](auto const& e) {
      constexpr auto type = std::decay_t<decltype(e)>::type;
      std::apply(
        [this, &e](auto const&... args) {
          if constexpr (type == event_type::task_created) {
            on_task_created(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::task_deployed) {
            on_task_deployed(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::failed_to_create_task) {
            on_failed_to_create_task(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::task_queue_empty) {
            on_task_queue_empty(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::pipeline_closed) {
            on_pipeline_closed(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::executor_awaiting_task) {
            on_executor_awaiting_task(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::memory_downgrade_for_task) {
            on_memory_downgrade_for_task(e.event_id, e.timestamp, args...);
          } else if constexpr (type == event_type::wait_for_memory_for_task) {
            on_wait_for_memory_for_task(e.event_id, e.timestamp, args...);
          } else {
            static_assert(unhandled_event_type<type>, "unhandled event_type");
          }
        },
        e.data);
    },
    event);
}

void query_event_subscriber::set_thread_name() noexcept
{
  // Same "subscriber-<name>" convention as before; @ref exec::thread_util
  // takes care of the 15-byte truncation the kernel imposes.
  std::string thread_name{"subscriber-"};
  thread_name.append(name());
  std::ignore = exec::thread_util::set_thread_name(_worker, thread_name);
}

}  // namespace sirius::event
