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

#include "catch.hpp"
#include "exec/query_stage_manager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace sirius;
using namespace sirius::exec;
using namespace std::chrono_literals;

namespace {

/// Records what it was told, in the order it was told.  Everything is under one
/// mutex rather than a set of atomics because the assertions are about the
/// sequence, not just the counts.
class recording_listener : public query_stage_listener {
 public:
  using query_stage_listener::query_stage_listener;

  ~recording_listener() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "recorder"; }

  void on_task_created(query_id_t query_id,
                       std::size_t operator_id,
                       op::SiriusPhysicalOperatorType,
                       queue_priority) noexcept override
  {
    record("created:" + std::to_string(value_of(query_id)) + ":" + std::to_string(operator_id));
  }

  void on_task_deployed(query_id_t,
                        std::size_t operator_id,
                        op::SiriusPhysicalOperatorType,
                        int gpu_id) noexcept override
  {
    record("deployed:" + std::to_string(operator_id) + ":" + std::to_string(gpu_id));
  }

  void on_task_queue_empty() noexcept override { record("empty"); }

  void on_executor_awaiting_task(int gpu_id) noexcept override
  {
    record("awaiting:" + std::to_string(gpu_id));
  }

  void on_wait_for_memory_for_task(query_id_t,
                                   std::size_t operator_id,
                                   int gpu_id,
                                   std::size_t bytes_needed) noexcept override
  {
    record("wait:" + std::to_string(operator_id) + ":" + std::to_string(gpu_id) + ":" +
           std::to_string(bytes_needed));
  }

  [[nodiscard]] std::vector<std::string> seen() const
  {
    std::lock_guard g{_mtx};
    return _seen;
  }

  [[nodiscard]] std::size_t count() const
  {
    std::lock_guard g{_mtx};
    return _seen.size();
  }

  [[nodiscard]] bool stop_seen() const noexcept { return _stop_seen.load(); }

 protected:
  void on_stop_requested() noexcept override { _stop_seen.store(true); }

 private:
  void record(std::string what)
  {
    std::lock_guard g{_mtx};
    _seen.push_back(std::move(what));
  }

  mutable std::mutex _mtx;
  std::vector<std::string> _seen;
  std::atomic<bool> _stop_seen{false};
};

/// Delivery is asynchronous, so every assertion about what arrived has to be a
/// poll rather than a read.  Fails by timing out, which is the honest outcome:
/// an event that never shows up is indistinguishable from one that is merely
/// slow, and the test should not pretend otherwise.
template <typename Listener>
bool wait_for(Listener const& l, std::size_t n, std::chrono::milliseconds timeout = 2s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (l.count() >= n) { return true; }
    std::this_thread::sleep_for(1ms);
  }
  return l.count() >= n;
}

/// Records every event it is given, and is subscribed to only one.
///
/// The recording is what makes this discriminating. A listener that merely
/// omitted the other hooks would inherit the base's empty defaults, so a
/// wrongly-delivered event would land on a no-op and the test would pass with
/// the routing table deleted. Overriding them all means an event this listener
/// did not subscribe to cannot arrive unnoticed.
class selective_listener : public query_stage_listener {
 public:
  explicit selective_listener(query_stage_manager& manager,
                              std::initializer_list<query_stage_event_type> events)
    : query_stage_listener(manager, events), _subscribed(events)
  {
  }

  ~selective_listener() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "selective"; }

  void on_task_created(query_id_t,
                       std::size_t,
                       op::SiriusPhysicalOperatorType,
                       queue_priority) noexcept override
  {
    record(query_stage_event_type::task_created, "task_created");
  }

  void on_task_deployed(query_id_t,
                        std::size_t,
                        op::SiriusPhysicalOperatorType,
                        int) noexcept override
  {
    record(query_stage_event_type::task_deployed, "task_deployed");
  }

  void on_failed_to_create_task(query_id_t, std::size_t, std::size_t) noexcept override
  {
    record(query_stage_event_type::failed_to_create_task, "failed_to_create_task");
  }

  void on_task_queue_empty() noexcept override
  {
    record(query_stage_event_type::task_queue_empty, "task_queue_empty");
  }

  void on_pipeline_closed(query_id_t, std::size_t, std::size_t) noexcept override
  {
    record(query_stage_event_type::pipeline_closed, "pipeline_closed");
  }

  void on_executor_awaiting_task(int) noexcept override
  {
    record(query_stage_event_type::executor_awaiting_task, "executor_awaiting_task");
  }

  void on_memory_downgrade_for_task(query_id_t, std::size_t, int, std::size_t) noexcept override
  {
    record(query_stage_event_type::memory_downgrade_for_task, "memory_downgrade_for_task");
  }

  void on_wait_for_memory_for_task(query_id_t, std::size_t, int, std::size_t) noexcept override
  {
    record(query_stage_event_type::wait_for_memory_for_task, "wait_for_memory_for_task");
  }

  [[nodiscard]] std::vector<std::string> seen() const
  {
    std::lock_guard g{_mtx};
    return _seen;
  }

  [[nodiscard]] std::size_t count() const
  {
    std::lock_guard g{_mtx};
    return _seen.size();
  }

  /// Deliveries of events this listener did NOT subscribe to. The routing is
  /// correct only while this stays zero -- and unlike the recorded names, it
  /// says so without the test having to know which events were published.
  [[nodiscard]] std::size_t unsubscribed_count() const noexcept
  {
    return _unsubscribed.load(std::memory_order_relaxed);
  }

 private:
  void record(query_stage_event_type type, std::string what)
  {
    if (std::ranges::find(_subscribed, type) == _subscribed.end()) {
      _unsubscribed.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard g{_mtx};
    _seen.push_back(std::move(what));
  }

  mutable std::mutex _mtx;
  std::vector<std::string> _seen;
  std::vector<query_stage_event_type> _subscribed;
  std::atomic<std::size_t> _unsubscribed{0};
};

/// Publish one of every event, in enumerator order.
void publish_one_of_each(query_stage_manager& manager)
{
  manager.notify_task_created(make_query_id(1), 2, op::SiriusPhysicalOperatorType::GPU_SCAN, 0);
  manager.notify_task_deployed(make_query_id(1), 2, op::SiriusPhysicalOperatorType::GPU_SCAN, 0);
  manager.notify_failed_to_create_task(make_query_id(1), 2, 3);
  manager.notify_task_queue_empty();
  manager.notify_pipeline_closed(make_query_id(1), 2, 3);
  manager.notify_executor_awaiting_task(0);
  manager.notify_memory_downgrade_for_task(make_query_id(1), 2, 0, 4096);
  manager.notify_wait_for_memory_for_task(make_query_id(1), 2, 0, 4096);
}

constexpr auto some_op_type               = op::SiriusPhysicalOperatorType::GPU_SCAN;
constexpr queue_priority default_priority = 0;

}  // namespace

// =============================================================================
// delivery
// =============================================================================

TEST_CASE("a started listener receives every event in publication order",
          "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  listener.start();
  REQUIRE(listener.is_listening());

  manager->notify_task_created(make_query_id(7), 3, some_op_type, default_priority);
  manager->notify_task_deployed(make_query_id(7), 3, some_op_type, 1);
  manager->notify_task_queue_empty();
  manager->notify_executor_awaiting_task(2);
  manager->notify_wait_for_memory_for_task(make_query_id(7), 3, 1, 4096);

  REQUIRE(wait_for(listener, 5));
  CHECK(listener.seen() ==
        std::vector<std::string>{
          "created:7:3", "deployed:3:1", "empty", "awaiting:2", "wait:3:1:4096"});
}

TEST_CASE("every registered listener gets its own copy of an event", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener first{*manager};
  recording_listener second{*manager};
  first.start();
  second.start();

  manager->notify_task_queue_empty();

  REQUIRE(wait_for(first, 1));
  REQUIRE(wait_for(second, 1));
  CHECK(first.seen() == std::vector<std::string>{"empty"});
  CHECK(second.seen() == std::vector<std::string>{"empty"});
}

TEST_CASE("publishing is decoupled from the reporter's thread", "[exec][query_stage_manager]")
{
  // The point of the queue: a reporter hands the event over and moves on, so a
  // listener that has not been started yet -- and so is draining nothing --
  // must not hold up the notify_* call.
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};

  for (int i = 0; i < 1000; ++i) {
    manager->notify_executor_awaiting_task(i);
  }
  CHECK(listener.count() == 0);

  // ...and the backlog is there waiting once it does start.
  listener.start();
  REQUIRE(wait_for(listener, 1000));
  CHECK(listener.seen().front() == "awaiting:0");
  CHECK(listener.seen().back() == "awaiting:999");
}

TEST_CASE("events raised with nobody registered are dropped", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  manager->notify_task_queue_empty();

  recording_listener listener{*manager};
  listener.start();
  manager->notify_executor_awaiting_task(5);

  REQUIRE(wait_for(listener, 1));
  // Only the one published after it arrived: registration is not a replay.
  CHECK(listener.seen() == std::vector<std::string>{"awaiting:5"});
}

// =============================================================================
// lifecycle
// =============================================================================

TEST_CASE("a listener that was never started stops without hanging", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  CHECK_FALSE(listener.is_listening());

  listener.stop();
  CHECK_FALSE(listener.is_listening());
}

TEST_CASE("listener start and stop are idempotent", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};

  listener.start();
  listener.start();  // already running -- must not spawn a second worker
  CHECK(listener.is_listening());

  listener.stop();
  listener.stop();
  CHECK_FALSE(listener.is_listening());
}

TEST_CASE("a stopped listener hears nothing further", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  listener.start();

  manager->notify_task_queue_empty();
  REQUIRE(wait_for(listener, 1));
  listener.stop();

  manager->notify_executor_awaiting_task(1);
  std::this_thread::sleep_for(50ms);
  CHECK(listener.count() == 1);
}

TEST_CASE("a stopped listener can be restarted with a fresh mailbox", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  listener.start();
  listener.stop();

  // Published while stopped, so deregistered: this one is gone for good.
  manager->notify_executor_awaiting_task(1);

  listener.start();
  CHECK(listener.is_listening());
  manager->notify_executor_awaiting_task(2);

  REQUIRE(wait_for(listener, 1));
  std::this_thread::sleep_for(50ms);
  CHECK(listener.seen() == std::vector<std::string>{"awaiting:2"});
}

TEST_CASE("the destructor stops a running listener", "[exec][query_stage_manager]")
{
  // The failure mode here is a hang, not a wrong value: a worker parked on an
  // empty queue with nothing to wake it would never be joined.
  auto manager  = std::make_shared<query_stage_manager>();
  auto listener = std::make_unique<recording_listener>(*manager);
  listener->start();
  REQUIRE(listener->is_listening());

  listener.reset();
  SUCCEED("destructor joined the worker");
}

TEST_CASE("a listener outliving its manager stops without hanging", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  listener.start();

  // No stop(), no sentinel: the worker has to come out on the token alone.
  manager->stop();
  manager.reset();

  listener.stop();
  CHECK_FALSE(listener.is_listening());
}

// =============================================================================
// manager stop
// =============================================================================

TEST_CASE("stopping the manager takes every listener down with it", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener first{*manager};
  recording_listener second{*manager};
  first.start();
  second.start();

  manager->stop();

  // The sentinel is what makes this prompt; the stop token alone would take up
  // to a poll interval.
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline && first.is_listening()) {
    std::this_thread::sleep_for(1ms);
  }
  first.stop();
  second.stop();
  CHECK(first.stop_seen());
  CHECK(second.stop_seen());
}

TEST_CASE("a stopped manager publishes nothing further", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  listener.start();

  manager->stop();
  manager->notify_task_queue_empty();
  manager->notify_executor_awaiting_task(1);

  std::this_thread::sleep_for(50ms);
  CHECK(listener.count() == 0);
}

TEST_CASE("registering with an already-stopped manager yields a listener that finishes",
          "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  manager->stop();

  recording_listener listener{*manager};
  listener.start();
  // The token is already stopped, so there is nothing to run and nothing to
  // join -- the failure this guards against is start() parking a worker
  // forever on a queue the manager will never write to.
  CHECK_FALSE(listener.is_listening());

  manager->notify_task_queue_empty();
  CHECK(listener.count() == 0);
}

TEST_CASE("reporters publishing concurrently all get through", "[exec][query_stage_manager]")
{
  constexpr int n_reporters  = 4;
  constexpr int per_reporter = 250;

  auto manager = std::make_shared<query_stage_manager>();
  recording_listener listener{*manager};
  listener.start();

  std::vector<std::thread> reporters;
  reporters.reserve(n_reporters);
  for (int r = 0; r < n_reporters; ++r) {
    reporters.emplace_back([manager, r] {
      for (int i = 0; i < per_reporter; ++i) {
        manager->notify_task_created(make_query_id(static_cast<std::uint32_t>(r)),
                                     static_cast<std::size_t>(i),
                                     some_op_type,
                                     default_priority);
      }
    });
  }
  for (auto& t : reporters) {
    t.join();
  }

  CHECK(wait_for(listener, n_reporters * per_reporter, 10s));
}

// =============================================================================
// per-event routing
// =============================================================================

// The tag doubles as the routing index, so a reordered enum or variant would
// misroute every event; the header static_asserts the ends of that mapping.
static_assert(n_system_events == all_system_events.size());

TEST_CASE("an unsubscribed event is never delivered", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  // Records all eight, subscribed to one. Anything but "task_queue_empty" in
  // the result means the routing delivered something nobody asked for.
  selective_listener listener{*manager, {query_stage_event_type::task_queue_empty}};
  // A listener subscribed to everything, purely as the sync point: once it has
  // all eight, every delivery this round has been made.
  selective_listener witness{*manager,
                             {query_stage_event_type::task_created,
                              query_stage_event_type::task_deployed,
                              query_stage_event_type::failed_to_create_task,
                              query_stage_event_type::task_queue_empty,
                              query_stage_event_type::pipeline_closed,
                              query_stage_event_type::executor_awaiting_task,
                              query_stage_event_type::memory_downgrade_for_task,
                              query_stage_event_type::wait_for_memory_for_task}};
  listener.start();
  witness.start();

  publish_one_of_each(*manager);

  REQUIRE(wait_for(witness, 8));
  CHECK(listener.unsubscribed_count() == 0);
  CHECK(listener.seen() == std::vector<std::string>{"task_queue_empty"});
  // The witness subscribed to all eight, so nothing it got was unsubscribed
  // either -- otherwise the counter would be measuring the wrong thing.
  CHECK(witness.unsubscribed_count() == 0);
}

TEST_CASE("a listener subscribed to nothing receives nothing", "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  selective_listener listener{*manager, {}};
  selective_listener witness{*manager, {query_stage_event_type::wait_for_memory_for_task}};
  listener.start();
  witness.start();

  publish_one_of_each(*manager);

  REQUIRE(wait_for(witness, 1));
  CHECK(listener.unsubscribed_count() == 0);
  CHECK(listener.count() == 0);
}

TEST_CASE("each event reaches exactly its own subscriber", "[exec][query_stage_manager]")
{
  // One listener per event, each recording all eight: a payload routed to the
  // wrong bucket shows up as a second entry on somebody.
  auto manager = std::make_shared<query_stage_manager>();
  std::vector<std::unique_ptr<selective_listener>> listeners;
  std::vector<std::string> const names{"task_created",
                                       "task_deployed",
                                       "failed_to_create_task",
                                       "task_queue_empty",
                                       "pipeline_closed",
                                       "executor_awaiting_task",
                                       "memory_downgrade_for_task",
                                       "wait_for_memory_for_task"};
  for (std::size_t i = 0; i < n_system_events; ++i) {
    listeners.push_back(std::make_unique<selective_listener>(
      *manager,
      std::initializer_list<query_stage_event_type>{static_cast<query_stage_event_type>(i)}));
    listeners.back()->start();
  }

  publish_one_of_each(*manager);

  for (std::size_t i = 0; i < listeners.size(); ++i) {
    INFO("listener for " << names[i]);
    REQUIRE(wait_for(*listeners[i], 1));
    CHECK(listeners[i]->unsubscribed_count() == 0);
    CHECK(listeners[i]->seen() == std::vector<std::string>{names[i]});
  }
}

TEST_CASE("unregistering drops a listener from every event's routing",
          "[exec][query_stage_manager]")
{
  auto manager = std::make_shared<query_stage_manager>();
  selective_listener narrow{*manager, {query_stage_event_type::task_queue_empty}};
  narrow.start();
  manager->notify_task_queue_empty();
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline && narrow.count() == 0) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(narrow.count() == 1);

  narrow.stop();
  // The routing table holds raw pointers into the queue the listener owns, so a
  // stale entry here would be a dangling write rather than a wasted one.
  manager->notify_task_queue_empty();
  std::this_thread::sleep_for(50ms);
  CHECK(narrow.count() == 1);
}

TEST_CASE("every event is published and no unsubscribed one is delivered",
          "[exec][query_stage_manager]")
{
  // Each listener subscribes to one event; every event is then published. The
  // counter is the assertion: it counts callbacks that fired for an event the
  // listener never asked for, so zero across all eight means the routing
  // delivered nothing it should not have.
  auto manager = std::make_shared<query_stage_manager>();
  std::vector<std::unique_ptr<selective_listener>> listeners;
  for (std::size_t i = 0; i < n_system_events; ++i) {
    listeners.push_back(std::make_unique<selective_listener>(
      *manager,
      std::initializer_list<query_stage_event_type>{static_cast<query_stage_event_type>(i)}));
    listeners.back()->start();
  }

  // Twice, so a stale routing entry has a second chance to show up.
  publish_one_of_each(*manager);
  publish_one_of_each(*manager);

  for (auto const& l : listeners) {
    REQUIRE(wait_for(*l, 2));
  }
  // Settle: a wrongly-routed payload would arrive around now, not before.
  std::this_thread::sleep_for(100ms);
  for (std::size_t i = 0; i < listeners.size(); ++i) {
    INFO("listener " << i);
    CHECK(listeners[i]->unsubscribed_count() == 0);
    CHECK(listeners[i]->count() == 2);
  }
}

TEST_CASE("registration alone decides delivery", "[exec][query_stage_manager]")
{
  // Two listeners of the SAME class, so they implement the same callbacks and
  // differ in exactly one thing: what each registered for. Publish one event and
  // one of them must fire while the other stays at zero. Nothing but the routing
  // can produce that difference.
  auto manager = std::make_shared<query_stage_manager>();
  selective_listener empties{*manager, {query_stage_event_type::task_queue_empty}};
  selective_listener closes{*manager, {query_stage_event_type::pipeline_closed}};
  empties.start();
  closes.start();

  manager->notify_task_queue_empty();

  REQUIRE(wait_for(empties, 1));
  std::this_thread::sleep_for(100ms);  // a misroute would land about now
  CHECK(empties.count() == 1);
  CHECK(closes.count() == 0);

  // And the mirror, so the result cannot be an artefact of which listener was
  // registered first or which event happens to be published.
  manager->notify_pipeline_closed(make_query_id(1), 2, 3);

  REQUIRE(wait_for(closes, 1));
  std::this_thread::sleep_for(100ms);
  CHECK(closes.count() == 1);
  CHECK(empties.count() == 1);
}
