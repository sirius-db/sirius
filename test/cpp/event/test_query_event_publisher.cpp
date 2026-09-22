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
#include "event/query_event_subscriber.hpp"
#include "utils/compressed_materialization_recorder.hpp"

#include <future>

// Query lifecycle producers and consumers live on different threads and most
// subscribers care about only a subset of events. This suite documents why the
// publisher owns event IDs/timestamps and filters delivery at registration,
// while subscriber teardown remains safe during concurrent publication.

#include <algorithm>
#include <atomic>
#include <barrier>
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

namespace sirius::event {
struct query_event_test_access {
  static auto hold_routing(query_event_publisher& publisher)
  {
    return std::unique_lock{publisher._queues_mtx};
  }

  static bool has_subscribers(query_event_publisher& publisher)
  {
    std::shared_lock lock{publisher._queues_mtx};
    return !publisher._queues.empty();
  }
};
}  // namespace sirius::event

namespace {

/// Records what it was told, in the order it was told.  Everything is under one
/// mutex rather than a set of atomics because the assertions are about the
/// sequence, not just the counts.
class recording_subscriber : public query_event_subscriber {
 public:
  explicit recording_subscriber(query_event_publisher& publisher)
    : query_event_subscriber(publisher,
                             {event_type::task_created,
                              event_type::task_deployed,
                              event_type::task_queue_empty,
                              event_type::executor_awaiting_task,
                              event_type::wait_for_memory_for_task})
  {
  }

  ~recording_subscriber() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "recorder"; }

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

 private:
  void record(std::string what)
  {
    std::lock_guard g{_mtx};
    _seen.push_back(std::move(what));
  }

  mutable std::mutex _mtx;
  std::vector<std::string> _seen;
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
    : query_event_subscriber(publisher, events, /*drain_on_stop=*/true), _subscribed(events)
  {
  }

  ~selective_subscriber() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "selective"; }

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

  void on_compressed_materialization(event_id_t,
                                     timestamp_t,
                                     compressed_materialization_activity,
                                     std::uint64_t) noexcept override
  {
    record(event_type::compressed_materialization, "compressed_materialization");
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
  publisher.publish_compressed_materialization(
    compressed_materialization_activity::pin_columns_narrowed, 3);
}

constexpr auto some_op_type                     = op::SiriusPhysicalOperatorType::GPU_SCAN;
constexpr exec::queue_priority default_priority = 0;

}  // namespace

// =============================================================================
// delivery
// =============================================================================

namespace {

/// Captures the event id and timestamp of every task_queue_empty it is told
/// about, so an assertion can compare against the wall clock the reporter saw.
class metadata_subscriber : public query_event_subscriber {
 public:
  explicit metadata_subscriber(query_event_publisher& publisher)
    : query_event_subscriber(publisher, {event_type::task_queue_empty})
  {
  }
  ~metadata_subscriber() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "meta"; }

  void on_task_queue_empty(event_id_t id, timestamp_t ts) noexcept override
  {
    std::lock_guard g{_mtx};
    _events.emplace_back(id, ts);
  }

  [[nodiscard]] std::vector<std::pair<event_id_t, timestamp_t>> events() const
  {
    std::lock_guard g{_mtx};
    return _events;
  }

  [[nodiscard]] std::size_t count() const
  {
    std::lock_guard g{_mtx};
    return _events.size();
  }

 private:
  mutable std::mutex _mtx;
  std::vector<std::pair<event_id_t, timestamp_t>> _events;
};

}  // namespace

TEST_CASE("published events carry IDs and timestamps", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  metadata_subscriber subscriber{*publisher};
  subscriber.start();
  auto const before = std::chrono::system_clock::now();

  publisher->publish_task_queue_empty();
  publisher->publish_task_queue_empty();
  auto const after = std::chrono::system_clock::now();

  REQUIRE(wait_for(subscriber, 2));
  auto const events                        = subscriber.events();
  auto const [first_id, first_timestamp]   = events[0];
  auto const [second_id, second_timestamp] = events[1];

  CHECK(second_id == first_id + 1);
  CHECK(first_timestamp >= before);
  CHECK(first_timestamp <= after);
  CHECK(second_timestamp >= first_timestamp);
  CHECK(second_timestamp <= after);
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

TEST_CASE("stop is terminal and start after it is a no-op", "[event][query_event_publisher]")
{
  // Subscribers are one-shot: once torn down they stay that way, so @ref start
  // after @ref stop is a no-op rather than a restart.  Anything published
  // between is dropped at the publisher --- @ref stop closes the mailbox, so
  // there is nothing queued for a worker that is not coming back either.
  auto publisher = std::make_shared<query_event_publisher>();
  recording_subscriber subscriber{*publisher};
  subscriber.start();
  subscriber.stop();
  REQUIRE_FALSE(subscriber.is_subscribed());

  publisher->publish_executor_awaiting_task(1);
  subscriber.start();
  CHECK_FALSE(subscriber.is_subscribed());
  publisher->publish_executor_awaiting_task(2);

  std::this_thread::sleep_for(50ms);
  CHECK(subscriber.count() == 0);
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

  // Closing each mailbox is what makes this prompt; the queue's own poll
  // backstop alone would take longer.
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline &&
         (first.is_subscribed() || second.is_subscribed())) {
    std::this_thread::sleep_for(1ms);
  }
  // is_subscribed() tracks the worker, not merely whether one was ever spawned,
  // so it reports the publisher's teardown without anyone calling stop().
  CHECK_FALSE(first.is_subscribed());
  CHECK_FALSE(second.is_subscribed());
  first.stop();
  second.stop();
}

namespace {

/// Stops itself from inside a hook, which is the one call of @ref stop that
/// cannot join --- it would be joining the thread it is running on.
class self_stopping_subscriber : public query_event_subscriber {
 public:
  explicit self_stopping_subscriber(query_event_publisher& publisher)
    : query_event_subscriber(publisher, {event_type::task_queue_empty})
  {
  }

  ~self_stopping_subscriber() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "self-stop"; }

  void on_task_queue_empty(event_id_t, timestamp_t) noexcept override
  {
    _count.fetch_add(1, std::memory_order_relaxed);
    stop();
  }

  [[nodiscard]] std::size_t count() const noexcept
  {
    return _count.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<std::size_t> _count{0};
};

}  // namespace

TEST_CASE("a subscriber can stop itself from a hook", "[event][query_event_publisher]")
{
  // The failure mode is a crash, not a wrong value: stop() joins, and a join
  // from the joined thread throws out of a noexcept function, which terminates.
  auto publisher = std::make_shared<query_event_publisher>();
  self_stopping_subscriber subscriber{*publisher};
  subscriber.start();

  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(subscriber, 1));

  // The mailbox is closed, so the second one never reaches the hook -- and the
  // worker is on its way out rather than joined, so give it room to get there.
  publisher->publish_task_queue_empty();
  std::this_thread::sleep_for(50ms);
  CHECK(subscriber.count() == 1);
}

TEST_CASE("concurrent start and stop leave no worker behind", "[event][query_event_publisher]")
{
  // start() reads its own state and then writes the worker; a stop() landing
  // between the two used to conclude there was nothing to join and return,
  // leaving a live worker no one would ever take down.
  constexpr int n_rounds = 200;
  for (int round = 0; round < n_rounds; ++round) {
    auto publisher = std::make_shared<query_event_publisher>();
    recording_subscriber subscriber{*publisher};

    std::thread starter{[&subscriber] { subscriber.start(); }};
    std::thread stopper{[&subscriber] { subscriber.stop(); }};
    starter.join();
    stopper.join();

    // stop() is terminal however the race fell out, so the worker is down.
    CHECK_FALSE(subscriber.is_subscribed());
  }
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

TEST_CASE("subscriber interest lasts until the last registration is removed",
          "[event][query_event_interest]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  for (auto type : all_query_events) {
    INFO("event " << static_cast<std::size_t>(type));
    CHECK_FALSE(publisher->has_subscribers(type));
    {
      // Registration, not worker start, decides interest. Duplicate event types
      // must not keep interest alive after the subscriber goes away.
      selective_subscriber first{*publisher, {type, type}};
      CHECK(publisher->has_subscribers(type));
      for (auto other : all_query_events) {
        CHECK(publisher->has_subscribers(other) == (other == type));
      }
      selective_subscriber second{*publisher, {type}};
      second.stop();
      CHECK(publisher->has_subscribers(type));
    }
    CHECK_FALSE(publisher->has_subscribers(type));
    selective_subscriber replacement{*publisher, {type}};
    CHECK(publisher->has_subscribers(type));
    replacement.stop();
    CHECK_FALSE(publisher->has_subscribers(type));
  }
  CHECK_FALSE(publisher->has_subscribers(static_cast<event_type>(n_query_events)));
}

TEST_CASE("publisher shutdown clears interest and prevents new interest",
          "[event][query_event_interest]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  std::vector<std::unique_ptr<selective_subscriber>> subscribers;
  for (auto type : all_query_events) {
    subscribers.push_back(
      std::make_unique<selective_subscriber>(*publisher, std::initializer_list<event_type>{type}));
    CHECK(publisher->has_subscribers(type));
  }
  publisher->stop();
  subscribers.clear();
  for (auto type : all_query_events) {
    CHECK_FALSE(publisher->has_subscribers(type));
    selective_subscriber late{*publisher, {type}};
    CHECK_FALSE(publisher->has_subscribers(type));
  }
}

TEST_CASE("unsubscribed publications bypass the routing lock", "[event][query_event_interest]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  // A registered mailbox with no interests must not defeat the per-event check.
  selective_subscriber empty{*publisher, {}};
  auto lock         = query_event_test_access::hold_routing(*publisher);
  auto published    = std::async(std::launch::async, [&] {
    bool interested = false;
    for (auto type : all_query_events) {
      interested |= publisher->has_subscribers(type);
    }
    publish_one_of_each(*publisher);
    return interested;
  });
  auto const status = published.wait_for(2s);
  // Release before assertions so a regression fails rather than hanging teardown.
  lock.unlock();
  CHECK(status == std::future_status::ready);
  CHECK_FALSE(published.get());
}

TEST_CASE("changing other subscriptions does not drop a live subscriber's events",
          "[event][query_event_interest]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  selective_subscriber stable{*publisher, {event_type::task_queue_empty}};
  stable.start();
  std::barrier ready{5};
  std::vector<std::jthread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      ready.arrive_and_wait();
      for (int j = 0; j < 500; ++j) {
        publisher->publish_task_queue_empty();
        publisher->publish_pipeline_closed(make_query_id(1), 2, 3);
      }
    });
  }
  threads.emplace_back([&] {
    ready.arrive_and_wait();
    for (int i = 0; i < 100; ++i) {
      // queue-empty retains its stable listener; pipeline-closed alternates
      // between having a listener and having none while reporters are active.
      selective_subscriber transient{*publisher,
                                     {event_type::task_queue_empty, event_type::pipeline_closed}};
    }
  });
  threads.clear();
  stable.stop();
  CHECK(stable.count() == 2000);
  CHECK(stable.unsubscribed_count() == 0);
  CHECK_FALSE(publisher->has_subscribers(event_type::task_queue_empty));
  CHECK_FALSE(publisher->has_subscribers(event_type::pipeline_closed));
}

TEST_CASE("an unsubscribed event is never delivered", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  // Records all events, subscribed to one. Anything but "task_queue_empty" in
  // the result means the routing delivered something nobody asked for.
  selective_subscriber subscriber{*publisher, {event_type::task_queue_empty}};
  // A subscriber subscribed to everything verifies all publications were made.
  // Each subscriber drains its own mailbox before assertions.
  selective_subscriber witness{*publisher,
                               {event_type::task_created,
                                event_type::task_deployed,
                                event_type::failed_to_create_task,
                                event_type::task_queue_empty,
                                event_type::pipeline_closed,
                                event_type::executor_awaiting_task,
                                event_type::memory_downgrade_for_task,
                                event_type::wait_for_memory_for_task,
                                event_type::compressed_materialization}};
  subscriber.start();
  witness.start();

  publish_one_of_each(*publisher);

  witness.stop();
  REQUIRE(witness.count() == n_query_events);
  subscriber.stop();
  CHECK(subscriber.unsubscribed_count() == 0);
  CHECK(subscriber.seen() == std::vector<std::string>{"task_queue_empty"});
  // The witness subscribed to all events, so nothing it got was unsubscribed
  // either -- otherwise the counter would be measuring the wrong thing.
  CHECK(witness.unsubscribed_count() == 0);
}

TEST_CASE("a subscriber subscribed to nothing receives nothing", "[event][query_event_publisher]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  selective_subscriber subscriber{*publisher, {}};
  selective_subscriber witness{
    *publisher, {event_type::wait_for_memory_for_task, event_type::compressed_materialization}};
  subscriber.start();
  witness.start();

  publish_one_of_each(*publisher);

  REQUIRE(wait_for(witness, 1));
  CHECK(subscriber.unsubscribed_count() == 0);
  CHECK(subscriber.count() == 0);
}

TEST_CASE("each event reaches exactly its own subscriber", "[event][query_event_publisher]")
{
  // One subscriber per event, each recording all events: a payload routed to the
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
                                       "wait_for_memory_for_task",
                                       "compressed_materialization"};
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

TEST_CASE("destroying a subscriber drops it from every event's routing",
          "[event][query_event_publisher]")
{
  // Registration is the subscriber's own lifetime: destroying it clears its
  // queue from the publisher's routing table.  Publishing after that must be
  // safe --- the routing pointers held into the destroyed queue would be
  // dangling writes rather than merely wasted ones.
  auto publisher = std::make_shared<query_event_publisher>();
  auto narrow    = std::make_unique<selective_subscriber>(
    *publisher, std::initializer_list<event_type>{event_type::task_queue_empty});
  narrow->start();
  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(*narrow, 1));
  REQUIRE(narrow->count() == 1);

  narrow.reset();

  // Bring a fresh subscriber in behind the destroyed one so the publish below
  // has someone to serve; if the destroyed subscriber's routing entry were
  // still there, the publish would touch its freed queue as it walked past.
  selective_subscriber witness{*publisher, {event_type::task_queue_empty}};
  witness.start();
  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(witness, 1));
  CHECK(witness.count() == 1);
}

TEST_CASE("stopping a subscriber drops it from every event's routing",
          "[event][query_event_publisher]")
{
  // A stopped subscriber is silent either way, so silence proves nothing.  The
  // event ID does: publish() takes an ID only after finding the event's bucket
  // non-empty, so an ID that does not advance is the bucket being genuinely
  // empty -- i.e. the stopped subscriber off the routing list, not merely
  // ignoring what it is still handed.
  auto publisher = std::make_shared<query_event_publisher>();

  metadata_subscriber quitter{*publisher};
  quitter.start();
  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(quitter, 1));
  auto const before = quitter.events().front().first;

  quitter.stop();
  for (int i = 0; i < 100; ++i) {
    publisher->publish_task_queue_empty();
  }

  metadata_subscriber after{*publisher};
  after.start();
  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(after, 1));

  // Consecutive across the gap: the hundred in between cost no ID because
  // nobody was registered.  Still routed to the stopped subscriber, they would
  // have taken one each.
  CHECK(after.events().front().first == before + 1);
}

TEST_CASE("stopping the publisher drops every subscriber from its routing",
          "[event][query_event_publisher]")
{
  // Same instrument, other end: after the publisher stops, its own routing
  // table is empty, so a publish costs no ID either.
  auto publisher = std::make_shared<query_event_publisher>();
  metadata_subscriber subscriber{*publisher};
  subscriber.start();
  publisher->publish_task_queue_empty();
  REQUIRE(wait_for(subscriber, 1));
  auto const before = subscriber.events().front().first;

  publisher->stop();
  for (int i = 0; i < 100; ++i) {
    publisher->publish_task_queue_empty();
  }

  auto fresh = std::make_shared<query_event_publisher>();
  metadata_subscriber after{*fresh};
  after.start();
  fresh->publish_task_queue_empty();
  REQUIRE(wait_for(after, 1));

  // The ID counter is process-wide, so a fresh publisher continues the same
  // sequence -- which is what makes it readable across the stopped one.
  CHECK(after.events().front().first == before + 1);
}

TEST_CASE("every event is published and no unsubscribed one is delivered",
          "[event][query_event_publisher]")
{
  // Each subscriber subscribes to one event; every event is then published. The
  // counter is the assertion: it counts callbacks that fired for an event the
  // subscriber never asked for, so zero across all events means the routing
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
  for (auto const& subscriber : subscribers) {
    subscriber->stop();
  }
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

  SECTION("only queue-empty is published")
  {
    publisher->publish_task_queue_empty();
    empties.stop();
    closes.stop();
    CHECK(empties.count() == 1);
    CHECK(closes.count() == 0);
  }
  SECTION("only pipeline-closed is published")
  {
    publisher->publish_pipeline_closed(make_query_id(1), 2, 3);
    empties.stop();
    closes.stop();
    CHECK(closes.count() == 1);
    CHECK(empties.count() == 0);
  }
}

TEST_CASE("compressed materialization snapshots consume every activity and preserve counts",
          "[event][compressed_materialization]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  sirius::test::compressed_materialization_recorder recorder{*publisher};
  auto const before = recorder.snapshot();
  CHECK(before.scan_columns_narrowed == 0);
  using enum compressed_materialization_activity;
  publisher->publish_compressed_materialization(scan_columns_narrowed, 2);
  publisher->publish_compressed_materialization(scan_columns_restored, 3);
  publisher->publish_compressed_materialization(pin_columns_narrowed, 5);
  publisher->publish_compressed_materialization(scan_sidecar_installed);
  publisher->publish_compressed_materialization(partition_narrow_columns, 7);
  publisher->publish_compressed_materialization(scan_narrow_targets_retracted, 11);
  auto const after = recorder.snapshot();
  CHECK(after.scan_columns_narrowed == 2);
  CHECK(after.scan_columns_restored == 3);
  CHECK(after.pin_columns_narrowed == 5);
  CHECK(after.scan_sidecars_installed == 1);
  CHECK(after.partition_narrow_columns == 7);
  CHECK(after.scan_narrow_targets_retracted == 11);
  // No new publication is still a complete, meaningful observation.
  CHECK(recorder.snapshot().pin_columns_narrowed == 5);
}

TEST_CASE("compressed materialization observations are scoped to their publisher",
          "[event][compressed_materialization]")
{
  auto first  = std::make_shared<query_event_publisher>();
  auto second = std::make_shared<query_event_publisher>();
  first->publish_compressed_materialization(
    compressed_materialization_activity::pin_columns_narrowed, 99);
  sirius::test::compressed_materialization_recorder one{*first};
  sirius::test::compressed_materialization_recorder two{*second};
  first->publish_compressed_materialization(
    compressed_materialization_activity::pin_columns_narrowed, 3);
  CHECK(one.snapshot().pin_columns_narrowed == 3);
  CHECK(two.snapshot().pin_columns_narrowed == 0);
}

TEST_CASE("drained snapshots include every concurrent reporter and can be reused",
          "[event][compressed_materialization]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  sirius::test::compressed_materialization_recorder recorder{*publisher};
  for (std::uint64_t round = 1; round <= 3; ++round) {
    std::vector<std::jthread> reporters;
    for (int thread = 0; thread < 4; ++thread) {
      reporters.emplace_back([&] {
        for (int i = 0; i < 500; ++i) {
          publisher->publish_compressed_materialization(
            compressed_materialization_activity::partition_narrow_columns, 2);
        }
      });
    }
    reporters.clear();  // Join before capturing the observation boundary.
    CHECK(recorder.snapshot().partition_narrow_columns == round * 4000);
  }
}

namespace {
class blocked_subscriber : public query_event_subscriber {
 public:
  explicit blocked_subscriber(query_event_publisher& publisher,
                              bool drain_on_stop  = false,
                              bool stop_from_hook = false)
    : query_event_subscriber(publisher, {event_type::task_queue_empty}, drain_on_stop),
      release_future(release.get_future().share()),
      _stop_from_hook(stop_from_hook)
  {
    start();
  }
  ~blocked_subscriber() override
  {
    unblock();
    stop();
  }
  std::string_view name() const noexcept override { return "blocked"; }
  void on_task_queue_empty(event_id_t, timestamp_t) noexcept override
  {
    if (count == 0) {
      entered.set_value();
      release_future.wait();
    }
    ++count;
    if (_stop_from_hook) { stop(); }
  }
  void unblock()
  {
    std::call_once(released, [&] { release.set_value(); });
  }
  std::promise<void> entered;
  std::size_t count{0};  // Read only after stop() joins the worker.

 private:
  std::once_flag released;
  std::promise<void> release;
  std::shared_future<void> release_future;
  bool const _stop_from_hook;
};
}  // namespace

TEST_CASE("shutdown drains queued callbacks only when opted in", "[event][query_event_drain]")
{
  bool const drain_on_stop  = GENERATE(false, true);
  bool const stop_from_hook = GENERATE(false, true);
  auto publisher            = std::make_shared<query_event_publisher>();
  blocked_subscriber subscriber{*publisher, drain_on_stop, stop_from_hook};
  publisher->publish_task_queue_empty();
  REQUIRE(subscriber.entered.get_future().wait_for(2s) == std::future_status::ready);

  // Leave a backlog in several producer queues while the first hook is blocked.
  // Shutdown sentinels must not hide any of it from the final drain.
  std::vector<std::jthread> reporters;
  for (int thread = 0; thread < 4; ++thread) {
    reporters.emplace_back([&] {
      for (int i = 0; i < 25; ++i) {
        publisher->publish_task_queue_empty();
      }
    });
  }
  reporters.clear();

  SECTION("subscriber stops") {}
  SECTION("publisher stops") { publisher->stop(); }

  auto stopped        = std::async(std::launch::async, [&] { subscriber.stop(); });
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (query_event_test_access::has_subscribers(*publisher) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  bool const detached = !query_event_test_access::has_subscribers(*publisher);
  auto const status   = stopped.wait_for(10ms);
  // This must not reach the detached subscriber, even during its drain.
  publisher->publish_task_queue_empty();
  subscriber.unblock();
  stopped.get();

  CHECK(detached);
  CHECK(status == std::future_status::timeout);
  CHECK(subscriber.count == (drain_on_stop ? 101 : 1));
  CHECK_FALSE(subscriber.is_subscribed());
  subscriber.stop();
  subscriber.start();
  CHECK_FALSE(subscriber.is_subscribed());
}

TEST_CASE("drain handles an empty mailbox and an unstarted subscriber",
          "[event][query_event_drain]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  selective_subscriber subscriber{*publisher, {event_type::task_queue_empty}};
  SECTION("started with no events") { subscriber.start(); }
  SECTION("never started") { publisher->publish_task_queue_empty(); }
  subscriber.stop();
  CHECK_FALSE(subscriber.is_subscribed());
  CHECK(subscriber.count() == 0);
}

TEST_CASE("mailbox delivery failures stay visible after draining", "[event][query_event_drain]")
{
  event_queue queue;
  CHECK_FALSE(queue.failed());
  queue.delivery_failed();
  queue.interrupt();
  CHECK(queue.try_pop() == nullptr);
  CHECK(queue.failed());
}

TEST_CASE("snapshots reject a stopped publisher", "[event][compressed_materialization]")
{
  auto publisher = std::make_shared<query_event_publisher>();
  sirius::test::compressed_materialization_recorder recorder{*publisher};
  publisher->stop();
  CHECK_THROWS_AS(recorder.snapshot(), std::runtime_error);
}
