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

TEST_CASE("writers finish ahead of pending maintenance without exceeding capacity",
          "[query_admission]")
{
  query_admission monitor;
  auto writer  = monitor.acquire(access_kind::writer, [] {});
  auto busy    = monitor.acquire(access_kind::query, [] {});
  auto acquire = [&](access_kind kind, const query_admission::permit* parent = nullptr) {
    return std::async(std::launch::async, [&, kind, parent] {
      try {
        return monitor.acquire(kind, [] {}, parent);
      } catch (std::runtime_error const&) {
        return query_admission::permit{};  // close() bounds failure cleanup.
      }
    });
  };
  auto maintenance    = acquire(access_kind::maintenance);
  bool waiting        = await([&] { return monitor.snapshot().maintenance_waiters == 1; });
  auto unrelated      = acquire(access_kind::query);
  bool queued         = await([&] { return monitor.snapshot().queued_queries == 1; });
  auto new_writer     = acquire(access_kind::writer);
  auto planning       = acquire(access_kind::planning, &writer);
  bool planning_ready = planning.wait_for(1s) == std::future_status::ready;
  auto continuation   = acquire(access_kind::query, &writer);
  bool both_queued    = await([&] { return monitor.snapshot().queued_queries == 2; });
  bool exceeded_limit = continuation.wait_for(0ms) == std::future_status::ready;
  busy.reset();
  bool progressed        = continuation.wait_for(1s) == std::future_status::ready;
  bool maintenance_early = maintenance.wait_for(0ms) == std::future_status::ready;
  bool unrelated_early   = unrelated.wait_for(0ms) == std::future_status::ready;
  bool writer_early      = new_writer.wait_for(0ms) == std::future_status::ready;
  if (!progressed || !planning_ready) monitor.close();
  continuation.get().reset();
  planning.get().reset();
  writer.reset();
  bool exclusive = maintenance.wait_for(1s) == std::future_status::ready;
  if (!exclusive) monitor.close();
  auto maintenance_permit = maintenance.get();
  bool maintenance_active = monitor.snapshot().maintenance_active;
  maintenance_permit.reset();
  monitor.close();
  unrelated.get().reset();
  new_writer.get().reset();
  monitor.wait_until_idle();

  CHECK(waiting);
  CHECK(queued);
  CHECK(planning_ready);
  CHECK(both_queued);
  CHECK_FALSE(exceeded_limit);
  CHECK(progressed);
  CHECK_FALSE(maintenance_early);
  CHECK_FALSE(unrelated_early);
  CHECK_FALSE(writer_early);
  CHECK(exclusive);
  CHECK(maintenance_active);
}

TEST_CASE("maintenance waiting on a writer is cancellable", "[query_admission]")
{
  query_admission monitor;
  auto writer = monitor.acquire(access_kind::writer, [] {});
  std::atomic<bool> cancel{false};
  auto maintenance = std::async(std::launch::async, [&] {
    try {
      auto p = monitor.acquire(access_kind::maintenance, [&] {
        if (cancel) throw std::runtime_error("cancelled");
      });
      return false;
    } catch (std::runtime_error const& e) {
      return std::string(e.what()) == "cancelled";
    }
  });
  bool waiting     = await([&] { return monitor.snapshot().maintenance_waiters == 1; });
  bool active      = monitor.snapshot().maintenance_active;
  cancel           = true;
  bool ready       = maintenance.wait_for(1s) == std::future_status::ready;
  if (!ready) monitor.close();
  CHECK(maintenance.get());
  CHECK(waiting);
  CHECK_FALSE(active);
  REQUIRE(ready);
  CHECK(monitor.snapshot().maintenance_waiters == 0);
  CHECK(monitor.snapshot().writers == 1);
  auto query = monitor.acquire(access_kind::query, [] {});
  CHECK(monitor.snapshot().active_queries == 1);
}

TEST_CASE("writer permits enforce ownership and participate in shutdown", "[query_admission]")
{
  query_admission monitor, other;
  auto writer = monitor.acquire(access_kind::writer, [] {});
  auto query  = monitor.acquire(access_kind::query, [] {});
  query_admission::permit empty;
  CHECK_THROWS_AS(monitor.acquire(access_kind::planning, [] {}, &empty), std::logic_error);
  CHECK_THROWS_AS(monitor.acquire(access_kind::planning, [] {}, &query), std::logic_error);
  CHECK_THROWS_AS(other.acquire(access_kind::planning, [] {}, &writer), std::logic_error);
  CHECK_THROWS_AS(monitor.acquire(access_kind::maintenance, [] {}, &writer), std::logic_error);
  query.reset();
  CHECK_THROWS_AS(monitor.configure(2), std::logic_error);
  monitor.close();
  auto idle   = std::async(std::launch::async, [&] { monitor.wait_until_idle(); });
  bool waited = idle.wait_for(50ms) == std::future_status::timeout;
  std::thread release([token = std::move(writer)]() mutable { token.reset(); });
  release.join();
  idle.get();
  CHECK(waited);
  CHECK(monitor.snapshot().writers == 0);
}
