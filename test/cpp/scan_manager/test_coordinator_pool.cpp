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
#include "scan_manager/coordinator_pool.hpp"

#include <atomic>
#include <future>
#include <stdexcept>
using namespace std::chrono_literals;
using sirius::scan_manager::coordinator_pool;
using progress = coordinator_pool::progress;
using result   = coordinator_pool::step_result;

TEST_CASE("Coordinator pool serializes steps and retains racing wakes", "[coordinator_pool]")
{
  auto workers = GENERATE(1u, 2u);
  coordinator_pool pool(workers);
  CHECK(pool.started_workers() == 0);
  std::atomic<unsigned> calls{0}, failures{0};
  std::atomic<bool> active{false}, overlap{false}, stopped{false}, neighbor_ran{false},
    unfair{false};
  std::promise<void> completed;
  auto done = completed.get_future();
  std::shared_ptr<coordinator_pool::registration> attempt, neighbor;
  attempt = pool.add(
    [&]() -> result {
      if (active.exchange(true)) overlap = true;
      if (++calls == 1 && !stopped) {
        neighbor->wake();
        for (int i = 0; i < 100; ++i)
          attempt->wake();
        active = false;
        return {progress::parked};
      }
      if (workers == 1 && !neighbor_ran) unfair = true;
      active = false;
      completed.set_value();
      return {progress::finished};
    },
    [&] { stopped = true; },
    [&](auto) {
      ++failures;
      stopped = true;
    });
  neighbor = pool.add(
    [&]() -> result {
      neighbor_ran = true;
      return {progress::finished};
    },
    [] {},
    [&](auto) { ++failures; });
  auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) { pool.stop(); });
  CHECK(pool.started_workers() == workers);
  attempt->wake();
  REQUIRE(done.wait_for(2s) == std::future_status::ready);
  attempt->wait();
  attempt->wake();  // A stale wake cannot reopen the completed attempt.
  CHECK(calls == 2);
  CHECK_FALSE(overlap);
  CHECK_FALSE(unfair);
  CHECK(failures == 0);
}

TEST_CASE("Coordinator pool handles deadlines failures and parked retirement independently",
          "[coordinator_pool]")
{
  coordinator_pool pool(1);
  std::atomic<bool> stopped{false};
  std::atomic<unsigned> failures{0};
  std::promise<void> timed, recovered;
  auto deadline_done = timed.get_future();
  auto failure_done  = recovered.get_future();
  auto due           = coordinator_pool::clock::now() + 30ms;
  auto timer         = pool.add(
    [&]() -> result {
      if (!stopped && coordinator_pool::clock::now() < due) return {progress::parked, due};
      timed.set_value();
      return {progress::finished};
    },
    [&] { stopped = true; },
    [&](auto) { ++failures; });
  auto failure = pool.add(
    [&]() -> result {
      if (!failures) throw std::runtime_error("one query failed");
      recovered.set_value();
      return {progress::finished};
    },
    [] {},
    [&](auto) { ++failures; });
  auto parked =
    pool.add([&]() -> result { return {stopped ? progress::finished : progress::parked}; },
             [&] { stopped = true; },
             [&](auto) { ++failures; });
  auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) { pool.stop(); });
  timer->wake();
  failure->wake();
  parked->wake();
  REQUIRE(failure_done.wait_for(2s) == std::future_status::ready);
  REQUIRE(deadline_done.wait_for(2s) == std::future_status::ready);
  CHECK(coordinator_pool::clock::now() >= due);
  pool.stop();
  timer->wait();
  failure->wait();
  parked->wait();
  CHECK(failures == 1);
  CHECK_THROWS(pool.add([] { return result{progress::finished}; }, [] {}, [](auto) {}));
}

TEST_CASE("Coordinator pool startup failure leaves no accepted attempt", "[coordinator_pool]")
{
  coordinator_pool pool(2);
  pool.before_thread_for_testing([](size_t i) {
    if (i == 1) throw std::runtime_error("startup");
  });
  auto step = [] { return result{progress::finished}; };
  CHECK_THROWS_WITH(pool.add(step, [] {}, [](auto) {}), "startup");
  CHECK(pool.started_workers() == 0);
  pool.before_thread_for_testing({});
  auto attempt = pool.add(step, [] {}, [](auto) {});
  auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) { pool.stop(); });
  attempt->wake();
  pool.stop();
  attempt->wait();
}

TEST_CASE("Coordinator wake handles release captures and ignore reused registrations",
          "[coordinator_pool]")
{
  coordinator_pool pool(1);
  auto backing                = std::make_shared<int>(1);
  std::weak_ptr<int> released = backing;
  auto attempt = pool.add([backing] { return result{progress::finished}; }, [] {}, [](auto) {});
  auto stale   = attempt->notifier();
  backing.reset();
  attempt->wake();
  attempt->wait();
  attempt.reset();
  CHECK(released.expired());

  unsigned calls   = 0;
  auto replacement = pool.add(
    [&] {
      ++calls;
      return result{progress::finished};
    },
    [] {},
    [](auto) {});
  auto fence   = pool.add([] { return result{progress::finished}; }, [] {}, [](auto) {});
  auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) { pool.stop(); });
  stale();
  fence->wake();
  fence->wait();
  // One worker scans the reused slot before the fence, if the stale wake marked it runnable.
  CHECK(calls == 0);
  pool.stop();
  CHECK(calls == 1);
}

TEST_CASE("Coordinator drain includes an in-flight pool stop notification", "[coordinator_pool]")
{
  coordinator_pool pool(1);
  std::promise<void> release_stop, finished;
  auto released      = release_stop.get_future();
  auto step_finished = finished.get_future();
  std::shared_ptr<coordinator_pool::registration> attempt;
  attempt = pool.add(
    [&] {
      finished.set_value();
      return result{progress::finished};
    },
    [&] {
      attempt->wake();
      released.wait();
    },
    [](auto) {});
  auto stopping = std::async(std::launch::async, [&] { pool.stop(); });
  auto cleanup  = std::shared_ptr<void>(nullptr, [&](void*) { release_stop.set_value(); });
  REQUIRE(step_finished.wait_for(2s) == std::future_status::ready);
  auto drained = std::async(std::launch::async, [&] { attempt->wait(); });
  auto status  = drained.wait_for(20ms);
  cleanup.reset();
  drained.get();
  stopping.get();
  CHECK(status == std::future_status::timeout);
}
