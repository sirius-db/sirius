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
#include "event/query_event_publisher.hpp"

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
#include <utility>
#include <vector>

using namespace sirius;
using namespace sirius::event;
using namespace std::chrono_literals;

namespace {

/// Records what it was told, in the order it was told.  Everything is under one
/// mutex rather than a set of atomics because the assertions are about the
/// sequence, not just the counts.
class recording_subscriber : public query_event_subscriber {
 public:
  using query_event_subscriber::query_event_subscriber;

  ~recording_subscriber() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "recorder"; }

  [[nodiscard]] std::vector<event_type> has_hooks_for_events() const override
  {
    return {event_type::task_created,
            event_type::task_deployed,
            event_type::task_queue_empty,
            event_type::executor_awaiting_task,
            event_type::wait_for_memory_for_task};
  }

  void on_task_created(event_id_t,
                       timestamp_t,
                       query_id_t query_id,
                       std::size_t operator_id,
                       op::SiriusPhysicalOperatorType,
                       exec::queue_priority) noexcept override
  {
    record("created:" + std::to_string(value_of(query_id)) + ":" + std::to_string(operator_id));
  }

  void on_task_deployed(event_id_t,
                        timestamp_t,
                        query_id_t,
                        std::size_t operator_id,
                        op::SiriusPhysicalOperatorType,
                        int gpu_id) noexcept override
  {
    record("deployed:" + std::to_string(operator_id) + ":" + std::to_string(gpu_id));
  }

  void on_task_queue_empty(event_id_t, timestamp_t) noexcept override { record("empty"); }

  void on_executor_awaiting_task(event_id_t, timestamp_t, int gpu_id) noexcept override
  {
    record("awaiting:" + std::to_string(gpu_id));
  }

  void on_wait_for_memory_for_task(event_id_t,
                                   timestamp_t,
                                   query_id_t,
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
template <typename Subscriber>
bool wait_for(Subscriber const& l, std::size_t n, std::chrono::milliseconds timeout = 2s)
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
/// The recording is what makes this discriminating. A subscriber that merely
/// omitted the other hooks would inherit the base's empty defaults, so a
/// wrongly-delivered event would land on a no-op and the test would pass with
/// the routing table deleted. Overriding them all means an event this subscriber
/// did not subscribe to cannot arrive unnoticed.
class selective_subscriber : public query_event_subscriber {
 public:
  explicit selective_subscriber(query_event_publisher& publisher,
                                std::initializer_list<event_type> events)
    : query_event_subscriber(publisher), _subscribed(events)
  {
  }

  ~selective_subscriber() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "selective"; }

  [[nodiscard]] std::vector<event_type> has_hooks_for_events() const override
  {
    return _subscribed;
  }

  void on_task_created(event_id_t,
                       timestamp_t,
                       query_id_t,
                       std::size_t,
                       op::SiriusPhysicalOperatorType,
                       exec::queue_priority) noexcept override
  {
    record(event_type::task_created, "task_created");
  }

  void on_task_deployed(event_id_t,
                        timestamp_t,
                        query_id_t,
                        std::size_t,
                        op::SiriusPhysicalOperatorType,
                        int) noexcept override
  {
    record(event_type::task_deployed, "task_deployed");
  }

  void on_failed_to_create_task(
    event_id_t, timestamp_t, query_id_t, std::size_t, std::size_t) noexcept override
  {
    record(event_type::failed_to_create_task, "failed_to_create_task");
  }

  void on_task_queue_empty(event_id_t, timestamp_t) noexcept override
  {
    record(event_type::task_queue_empty, "task_queue_empty");
  }

  void on_pipeline_closed(
    event_id_t, timestamp_t, query_id_t, std::size_t, std::size_t) noexcept override
  {
    record(event_type::pipeline_closed, "pipeline_closed");
  }

  void on_executor_awaiting_task(event_id_t, timestamp_t, int) noexcept override
  {
    record(event_type::executor_awaiting_task, "executor_awaiting_task");
  }

  void on_memory_downgrade_for_task(
    event_id_t, timestamp_t, query_id_t, std::size_t, int, std::size_t) noexcept override
  {
    record(event_type::memory_downgrade_for_task, "memory_downgrade_for_task");
  }

  void on_wait_for_memory_for_task(
    event_id_t, timestamp_t, query_id_t, std::size_t, int, std::size_t) noexcept override
  {
    record(event_type::wait_for_memory_for_task, "wait_for_memory_for_task");
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

  /// Deliveries of events this subscriber did NOT subscribe to. The routing is
  /// correct only while this stays zero -- and unlike the recorded names, it
  /// says so without the test having to know which events were published.
  [[nodiscard]] std::size_t unsubscribed_count() const noexcept
  {
    return _unsubscribed.load(std::memory_order_relaxed);
  }

 private:
  void record(event_type type, std::string what)
  {
    if (std::ranges::find(_subscribed, type) == _subscribed.end()) {
      _unsubscribed.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard g{_mtx};
    _seen.push_back(std::move(what));
  }

  mutable std::mutex _mtx;
  std::vector<std::string> _seen;
  std::vector<event_type> _subscribed;
  std::atomic<std::size_t> _unsubscribed{0};
};

/// Publish one of every event, in enumerator order.
void publish_one_of_each(query_event_publisher& publisher)
{
  publisher.publish_task_created(make_query_id(1), 2, op::SiriusPhysicalOperatorType::GPU_SCAN, 0);
  publisher.publish_task_deployed(make_query_id(1), 2, op::SiriusPhysicalOperatorType::GPU_SCAN, 0);
  publisher.publish_failed_to_create_task(make_query_id(1), 2, 3);
  publisher.publish_task_queue_empty();
  publisher.publish_pipeline_closed(make_query_id(1), 2, 3);
  publisher.publish_executor_awaiting_task(0);
  publisher.publish_memory_downgrade_for_task(make_query_id(1), 2, 0, 4096);
  publisher.publish_wait_for_memory_for_task(make_query_id(1), 2, 0, 4096);
}

constexpr auto some_op_type                     = op::SiriusPhysicalOperatorType::GPU_SCAN;
constexpr exec::queue_priority default_priority = 0;

}  // namespace

// =============================================================================
// delivery
// =============================================================================

TEST_CASE("published events carry IDs and timestamps", "[event][query_event_publisher]")
{
  auto publisher    = std::make_shared<query_event_publisher>();
  auto registration = publisher->register_subscriber({event_type::task_queue_empty});
  auto const before = std::chrono::system_clock::now();

  publisher->publish_task_queue_empty();
  publisher->publish_task_queue_empty();
  auto const after = std::chrono::system_clock::now();

  std::shared_ptr<query_events> first;
  std::shared_ptr<query_events> second;
  REQUIRE(registration.queue->wait_dequeue_timed(first, 2s));
  REQUIRE(registration.queue->wait_dequeue_timed(second, 2s));

  auto const metadata = [](query_events const& published) {
    return std::visit([](auto const& event) { return std::pair{event.event_id, event.timestamp}; },
                      published);
  };
  auto const [first_id, first_timestamp]   = metadata(*first);
  auto const [second_id, second_timestamp] = metadata(*second);

  CHECK(second_id == first_id + 1);
  CHECK(first_timestamp >= before);
  CHECK(first_timestamp <= after);
  CHECK(second_timestamp >= first_timestamp);
  CHECK(second_timestamp <= after);

  publisher->unregister_subscriber(registration.queue);
}

TEST_CASE("a started subscriber receives every event in publication order",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();
  REQUIRE(subscriber.is_subscribed());

  publisher->publish_task_created(make_query_id(7), 3, some_op_type, default_priority);
  publisher->publish_task_deployed(make_query_id(7), 3, some_op_type, 1);
  publisher->publish_task_queue_empty();
  publisher->publish_executor_awaiting_task(2);
  publisher->publish_wait_for_memory_for_task(make_query_id(7), 3, 1, 4096);

  REQUIRE(wait_for(subscriber, 5));
  CHECK(subscriber.seen() ==
        std::vector<std::string>{
          "created:7:3", "deployed:3:1", "empty", "awaiting:2", "wait:3:1:4096"});
}

TEST_CASE("every registered subscriber gets its own copy of an event",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber first{*publisher};
  recording_subscriber second{*publisher};
  first.start();
  second.start();

  publisher->publish_task_queue_empty();

  REQUIRE(wait_for(first, 1));
  REQUIRE(wait_for(second, 1));
  CHECK(first.seen() == std::vector<std::string>{"empty"});
  CHECK(second.seen() == std::vector<std::string>{"empty"});
}

TEST_CASE("a started subscriber receives a burst in publication order",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();

  for (int i = 0; i < 1000; ++i) {
    publisher->publish_executor_awaiting_task(i);
  }

  REQUIRE(wait_for(subscriber, 1000));
  CHECK(subscriber.seen().front() == "awaiting:0");
  CHECK(subscriber.seen().back() == "awaiting:999");
}

TEST_CASE("events raised with nobody registered are dropped", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  publisher->publish_task_queue_empty();

  recording_subscriber subscriber{*publisher};
  subscriber.start();
  publisher->publish_executor_awaiting_task(5);

  REQUIRE(wait_for(subscriber, 1));
  // Only the one published after it arrived: registration is not a replay.
  CHECK(subscriber.seen() == std::vector<std::string>{"awaiting:5"});
}

// =============================================================================
// lifecycle
// =============================================================================

TEST_CASE("a subscriber that was never started stops without hanging",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  CHECK_FALSE(subscriber.is_subscribed());

  subscriber.stop();
  CHECK_FALSE(subscriber.is_subscribed());
}

TEST_CASE("subscriber start and stop are idempotent", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};

  subscriber.start();
  subscriber.start();  // already running -- must not spawn a second worker
  CHECK(subscriber.is_subscribed());

  subscriber.stop();
  subscriber.stop();
  CHECK_FALSE(subscriber.is_subscribed());
}

TEST_CASE("a stopped subscriber hears nothing further", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();

  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(subscriber, 1));
  subscriber.stop();

  publisher->publish_executor_awaiting_task(1);
  std::this_thread::sleep_for(50ms);
  CHECK(subscriber.count() == 1);
}

TEST_CASE("a stopped subscriber can be restarted with a fresh mailbox",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();
  subscriber.stop();

  // Published while stopped, so deregistered: this one is gone for good.
  publisher->publish_executor_awaiting_task(1);

  subscriber.start();
  CHECK(subscriber.is_subscribed());
  publisher->publish_executor_awaiting_task(2);

  REQUIRE(wait_for(subscriber, 1));
  std::this_thread::sleep_for(50ms);
  CHECK(subscriber.seen() == std::vector<std::string>{"awaiting:2"});
}

TEST_CASE("the destructor stops a running subscriber", "[event][query_event_publisher]")
{
  // The failure mode here is a hang, not a wrong value: a worker parked on an
  // empty queue with nothing to wake it would never be joined.
  auto publisher  = std::make_shared<query_event_publisher>();
  auto subscriber = std::make_unique<recording_subscriber>(*publisher);
  subscriber->start();
  REQUIRE(subscriber->is_subscribed());

  subscriber.reset();
  SUCCEED("destructor joined the worker");
}

TEST_CASE("a subscriber outliving its publisher stops without hanging",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();

  // No stop(), no sentinel: the worker has to come out on the token alone.
  publisher->stop();
  publisher.reset();

  subscriber.stop();
  CHECK_FALSE(subscriber.is_subscribed());
}

// =============================================================================
// publisher stop
// =============================================================================

TEST_CASE("stopping the publisher takes every subscriber down with it",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber first{*publisher};
  recording_subscriber second{*publisher};
  first.start();
  second.start();

  publisher->stop();

  // The sentinel is what makes this prompt; the stop token alone would take up
  // to a poll interval.
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline && first.is_subscribed()) {
    std::this_thread::sleep_for(1ms);
  }
  first.stop();
  second.stop();
  CHECK(first.stop_seen());
  CHECK(second.stop_seen());
}

TEST_CASE("a stopped publisher publishes nothing further", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();

  publisher->stop();
  publisher->publish_task_queue_empty();
  publisher->publish_executor_awaiting_task(1);

  std::this_thread::sleep_for(50ms);
  CHECK(subscriber.count() == 0);
}

TEST_CASE("registering with an already-stopped publisher yields a subscriber that finishes",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  publisher->stop();

  recording_subscriber subscriber{*publisher};
  subscriber.start();
  // The token is already stopped, so there is nothing to run and nothing to
  // join -- the failure this guards against is start() parking a worker
  // forever on a queue the publisher will never write to.
  CHECK_FALSE(subscriber.is_subscribed());

  publisher->publish_task_queue_empty();
  CHECK(subscriber.count() == 0);
}

TEST_CASE("reporters publishing concurrently all get through", "[event][query_event_publisher]")
{
  constexpr int n_reporters  = 4;
  constexpr int per_reporter = 250;

  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();

  std::vector<std::thread> reporters;
  reporters.reserve(n_reporters);
  for (int r = 0; r < n_reporters; ++r) {
    reporters.emplace_back([publisher, r] {
      for (int i = 0; i < per_reporter; ++i) {
        publisher->publish_task_created(make_query_id(static_cast<std::uint32_t>(r)),
                                        static_cast<std::size_t>(i),
                                        some_op_type,
                                        default_priority);
      }
    });
  }
  for (auto& t : reporters) {
    t.join();
  }

  CHECK(wait_for(subscriber, n_reporters * per_reporter, 10s));
}

// =============================================================================
// per-event routing
// =============================================================================

// The tag doubles as the routing index, so a reordered enum or variant would
// misroute every event; the header static_asserts the ends of that mapping.
static_assert(n_query_events == all_query_events.size());

TEST_CASE("an unsubscribed event is never delivered", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  // Records all eight, subscribed to one. Anything but "task_queue_empty" in
  // the result means the routing delivered something nobody asked for.
  selective_subscriber subscriber{*publisher, {event_type::task_queue_empty}};
  // A subscriber subscribed to everything, purely as the sync point: once it has
  // all eight, every delivery this round has been made.
  selective_subscriber witness{*publisher,
                               {event_type::task_created,
                                event_type::task_deployed,
                                event_type::failed_to_create_task,
                                event_type::task_queue_empty,
                                event_type::pipeline_closed,
                                event_type::executor_awaiting_task,
                                event_type::memory_downgrade_for_task,
                                event_type::wait_for_memory_for_task}};
  subscriber.start();
  witness.start();

  publish_one_of_each(*publisher);

  REQUIRE(wait_for(witness, 8));
  CHECK(subscriber.unsubscribed_count() == 0);
  CHECK(subscriber.seen() == std::vector<std::string>{"task_queue_empty"});
  // The witness subscribed to all eight, so nothing it got was unsubscribed
  // either -- otherwise the counter would be measuring the wrong thing.
  CHECK(witness.unsubscribed_count() == 0);
}

TEST_CASE("a subscriber subscribed to nothing receives nothing", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  selective_subscriber subscriber{*publisher, {}};
  selective_subscriber witness{*publisher, {event_type::wait_for_memory_for_task}};
  subscriber.start();
  witness.start();

  publish_one_of_each(*publisher);

  REQUIRE(wait_for(witness, 1));
  CHECK(subscriber.unsubscribed_count() == 0);
  CHECK(subscriber.count() == 0);
}

TEST_CASE("each event reaches exactly its own subscriber", "[event][query_event_publisher]")
{
  // One subscriber per event, each recording all eight: a payload routed to the
  // wrong bucket shows up as a second entry on somebody.
  auto publisher = std::make_shared<query_event_publisher>();
  std::vector<std::unique_ptr<selective_subscriber>> subscribers;
  std::vector<std::string> const names{"task_created",
                                       "task_deployed",
                                       "failed_to_create_task",
                                       "task_queue_empty",
                                       "pipeline_closed",
                                       "executor_awaiting_task",
                                       "memory_downgrade_for_task",
                                       "wait_for_memory_for_task"};
  for (std::size_t i = 0; i < n_query_events; ++i) {
    subscribers.push_back(std::make_unique<selective_subscriber>(
      *publisher, std::initializer_list<event_type>{static_cast<event_type>(i)}));
    subscribers.back()->start();
  }

  publish_one_of_each(*publisher);

  for (std::size_t i = 0; i < subscribers.size(); ++i) {
    INFO("subscriber for " << names[i]);
    REQUIRE(wait_for(*subscribers[i], 1));
    CHECK(subscribers[i]->unsubscribed_count() == 0);
    CHECK(subscribers[i]->seen() == std::vector<std::string>{names[i]});
  }
}

TEST_CASE("unregistering drops a subscriber from every event's routing",
          "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  selective_subscriber narrow{*publisher, {event_type::task_queue_empty}};
  narrow.start();
  publisher->publish_task_queue_empty();
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline && narrow.count() == 0) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(narrow.count() == 1);

  narrow.stop();
  // The routing table holds raw pointers into the queue the subscriber owns, so a
  // stale entry here would be a dangling write rather than a wasted one.
  publisher->publish_task_queue_empty();
  std::this_thread::sleep_for(50ms);
  CHECK(narrow.count() == 1);
}

TEST_CASE("every event is published and no unsubscribed one is delivered",
          "[event][query_event_publisher]")
{
  // Each subscriber subscribes to one event; every event is then published. The
  // counter is the assertion: it counts callbacks that fired for an event the
  // subscriber never asked for, so zero across all eight means the routing
  // delivered nothing it should not have.
  auto publisher = std::make_shared<query_event_publisher>();
  std::vector<std::unique_ptr<selective_subscriber>> subscribers;
  for (std::size_t i = 0; i < n_query_events; ++i) {
    subscribers.push_back(std::make_unique<selective_subscriber>(
      *publisher, std::initializer_list<event_type>{static_cast<event_type>(i)}));
    subscribers.back()->start();
  }

  // Twice, so a stale routing entry has a second chance to show up.
  publish_one_of_each(*publisher);
  publish_one_of_each(*publisher);

  for (auto const& l : subscribers) {
    REQUIRE(wait_for(*l, 2));
  }
  // Settle: a wrongly-routed payload would arrive around now, not before.
  std::this_thread::sleep_for(100ms);
  for (std::size_t i = 0; i < subscribers.size(); ++i) {
    INFO("subscriber " << i);
    CHECK(subscribers[i]->unsubscribed_count() == 0);
    CHECK(subscribers[i]->count() == 2);
  }
}

TEST_CASE("registration alone decides delivery", "[event][query_event_publisher]")
{
  // Two subscribers of the SAME class, so they implement the same callbacks and
  // differ in exactly one thing: what each registered for. Publish one event and
  // one of them must fire while the other stays at zero. Nothing but the routing
  // can produce that difference.
  auto publisher = std::make_shared<query_event_publisher>();
  selective_subscriber empties{*publisher, {event_type::task_queue_empty}};
  selective_subscriber closes{*publisher, {event_type::pipeline_closed}};
  empties.start();
  closes.start();

  publisher->publish_task_queue_empty();

  REQUIRE(wait_for(empties, 1));
  std::this_thread::sleep_for(100ms);  // a misroute would land about now
  CHECK(empties.count() == 1);
  CHECK(closes.count() == 0);

  // And the mirror, so the result cannot be an artefact of which subscriber was
  // registered first or which event happens to be published.
  publisher->publish_pipeline_closed(make_query_id(1), 2, 3);

  REQUIRE(wait_for(closes, 1));
  std::this_thread::sleep_for(100ms);
  CHECK(closes.count() == 1);
  CHECK(empties.count() == 1);
}
