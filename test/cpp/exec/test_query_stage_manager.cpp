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
bool wait_for(recording_listener const& l, std::size_t n, std::chrono::milliseconds timeout = 2s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (l.count() >= n) { return true; }
    std::this_thread::sleep_for(1ms);
  }
  return l.count() >= n;
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
