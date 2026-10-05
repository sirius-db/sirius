/*
 * Copyright 2025, Sirius Contributors.
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

#include "exec/query_admission.hpp"

#include <catch.hpp>

#include <atomic>
#include <future>
#include <thread>
#include <vector>
using sirius::exec::query_admission;
using access_kind = query_admission::access;
using namespace std::chrono_literals;
namespace {
template <class Predicate>
bool await(Predicate p)
{
  auto end = std::chrono::steady_clock::now() + 2s;
  while (!p() && std::chrono::steady_clock::now() < end)
    std::this_thread::yield();
  return p();
}
}  // namespace
TEST_CASE("query admission enforces limits and FIFO arrival", "[query_admission]")
{
  for (auto limit : {1, 2, 4}) {
    query_admission monitor;
    monitor.configure(limit);
    std::vector<query_admission::permit> holders;
    for (int i = 0; i < limit; ++i)
      holders.push_back(monitor.acquire(access_kind::query, [] {}));
    REQUIRE(monitor.snapshot().active_queries == limit);
    std::promise<void> release_first;
    auto release = release_first.get_future();
    std::atomic<int> order{0};
    std::atomic<int> first_order{0}, second_order{0};
    auto first        = std::async(std::launch::async, [&] {
      auto p      = monitor.acquire(access_kind::query, [] {});
      first_order = ++order;
      release.wait();
      return p.ticket();
    });
    bool first_queued = await([&] { return monitor.snapshot().queued_queries == 1; });
    auto second       = std::async(std::launch::async, [&] {
      auto p       = monitor.acquire(access_kind::query, [] {});
      second_order = ++order;
      return p.ticket();
    });
    bool both_queued  = await([&] { return monitor.snapshot().queued_queries == 2; });
    holders.back().reset();
    bool entered  = await([&] { return first_order.load() != 0; });
    int premature = second_order.load();
    release_first.set_value();
    auto first_id  = first.get();
    auto second_id = second.get();
    CHECK(first_queued);
    CHECK(both_queued);
    CHECK(entered);
    CHECK(premature == 0);
    CHECK(first_order == 1);
    CHECK(second_order == 2);
    CHECK(first_id < second_id);
  }
}
TEST_CASE("admission cancellation removes waiters without consuming capacity", "[query_admission]")
{
  query_admission monitor;
  auto held = monitor.acquire(access_kind::query, [] {});
  std::atomic<bool> cancel{false};
  auto waiter = std::async(std::launch::async, [&] {
    try {
      auto p = monitor.acquire(access_kind::query, [&] {
        if (cancel) throw std::runtime_error("cancelled");
      });
      return false;
    } catch (std::runtime_error const& e) {
      return std::string(e.what()) == "cancelled";
    }
  });
  bool queued = await([&] { return monitor.snapshot().queued_queries == 1; });
  cancel      = true;
  bool ready  = waiter.wait_for(1s) == std::future_status::ready;
  if (!ready) monitor.close();
  CHECK(waiter.get());
  CHECK(queued);
  CHECK(ready);
  CHECK(monitor.snapshot().queued_queries == 0);
  CHECK(monitor.snapshot().active_queries == 1);
}
TEST_CASE("pending maintenance blocks new queries and planning", "[query_admission]")
{
  query_admission monitor;
  monitor.configure(2);
  auto held     = monitor.acquire(access_kind::query, [] {});
  auto planning = monitor.acquire(access_kind::planning, [] {});
  std::promise<void> release_maintenance;
  auto release = release_maintenance.get_future();
  std::atomic<bool> entered{false};
  auto maintenance = std::async(std::launch::async, [&] {
    auto p  = monitor.acquire(access_kind::maintenance, [] {});
    entered = true;
    release.wait();
  });
  bool waiting     = await([&] { return monitor.snapshot().maintenance_waiters == 1; });
  auto next =
    std::async(std::launch::async, [&] { return monitor.acquire(access_kind::query, [] {}); });
  bool queued = await([&] { return monitor.snapshot().queued_queries == 1; });
  held.reset();
  bool too_early = entered.load();
  planning.reset();
  bool exclusive = await([&] { return entered.load(); });
  bool passed    = next.wait_for(0ms) == std::future_status::ready;
  release_maintenance.set_value();
  maintenance.get();
  auto query = next.get();
  CHECK(waiting);
  CHECK(queued);
  CHECK_FALSE(too_early);
  CHECK(exclusive);
  CHECK_FALSE(passed);
}
TEST_CASE("admission permits transfer threads and shutdown wakes waiters", "[query_admission]")
{
  query_admission monitor;
  auto held = monitor.acquire(access_kind::query, [] {});
  std::thread thread([token = std::move(held)]() mutable { token.reset(); });
  thread.join();
  CHECK(monitor.snapshot().active_queries == 0);
  held        = monitor.acquire(access_kind::query, [] {});
  auto waiter = std::async(std::launch::async, [&] {
    try {
      auto p = monitor.acquire(access_kind::query, [] {});
      return false;
    } catch (std::runtime_error const&) {
      return true;
    }
  });
  bool queued = await([&] { return monitor.snapshot().queued_queries == 1; });
  monitor.close();
  CHECK(waiter.get());
  CHECK(queued);
  held.reset();
  monitor.wait_until_idle();
  CHECK(monitor.snapshot().closing);
}
