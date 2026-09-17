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

#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
  // One lock over the whole transition rather than a flag per step: the checks
  // and the assignment to @c _worker have to be indivisible, or a @ref stop
  // running between them decides there is no worker and leaves one behind.
  std::lock_guard g{_mtx};
  if (_state != worker_state::idle) { return; }  // already running, or torn down for good
  if (_stop_token.stop_requested()) { return; }  // publisher already stopped
  _state = worker_state::running;
  _worker_live.store(true, std::memory_order_release);
  _worker = std::jthread([this] { run(); });
  set_thread_name();
}

void query_event_subscriber::stop() noexcept
{
  std::lock_guard g{_mtx};
  if (_state == worker_state::stopped) { return; }
  auto const was_running = _state == worker_state::running;
  _state                 = worker_state::stopped;  // terminal: @ref start is a no-op from here
  if (!was_running) { return; }

  // Closing the mailbox is the whole teardown: it wakes the worker at once, and
  // it turns every later publish into a drop, so a stopped subscriber cannot
  // accumulate events nobody will drain.  interrupt() closes before it enqueues
  // its wake-up sentinels, so a throw from that allocation still leaves the
  // mailbox closed and the worker comes out on the queue's poll backstop --
  // which is what lets this stay noexcept.
  try {
    _queue->interrupt();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }

  // A hook that stops its own subscriber would be joining itself, which throws
  // -- and throwing out of a noexcept function terminates.  Leave instead: the
  // mailbox is closed, so the worker exits as soon as the hook returns, and
  // @c ~jthread joins it.
  if (_worker.get_id() == std::this_thread::get_id()) { return; }
  _worker.join();
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
  // pop() returns null exactly once the mailbox closes, which every teardown
  // path does -- so the loop needs no second exit condition of its own.
  while (auto event = _queue->pop()) {
    dispatch(*event);
  }
  // Reported here, after the last hook, so a subscriber sees its events and its
  // shutdown in one order on one thread.  Only when the publisher is what
  // stopped us: a caller of @ref stop does not need telling.
  if (_stop_token.stop_requested()) { on_stop_requested(); }
  _worker_live.store(false, std::memory_order_release);
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
