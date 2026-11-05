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

#include "scan_manager/shared_scan_budget.hpp"

#include <catch.hpp>

#include <future>
#include <thread>
using sirius::make_query_id;
using sirius::scan_manager::shared_scan_budget;
using namespace std::chrono_literals;
namespace {
template <class F>
bool budget_await(F f)
{
  auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!f() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return f();
}
}  // namespace
TEST_CASE("shared scan budget bounds speculation and prioritizes demand", "[shared_scan_budget]")
{
  shared_scan_budget budget;
  auto first = budget.acquire({{1, 1}}, make_query_id(1), {}, false);
  REQUIRE(budget.active(1) == 1);
  auto demand = budget.acquire({{1, 1}}, make_query_id(2), {}, true);
  REQUIRE(budget.active(1) == 2);
  std::stop_source stop;
  auto next    = std::async(std::launch::async, [&] {
    return budget.acquire({{1, 1}}, make_query_id(3), stop.get_token(), false);
  });
  bool pending = budget_await([&] { return budget.pending() == 1; });
  first.reset();
  bool premature = next.wait_for(20ms) == std::future_status::ready;
  demand.reset();
  bool ready = next.wait_for(1s) == std::future_status::ready;
  if (!ready) stop.request_stop();
  auto token = next.get();
  CHECK(pending);
  CHECK_FALSE(premature);
  CHECK(ready);
  REQUIRE(token);
  CHECK(budget.active(1) == 1);
  token.reset();
  CHECK(budget.active(1) == 0);
}
TEST_CASE("shared scan cancellation preserves another subscriber's capacity",
          "[shared_scan_budget]")
{
  shared_scan_budget budget;
  auto first = budget.acquire({{1, 1}, {1, 1}, {2, 2}}, make_query_id(1), {}, false);
  CHECK(budget.active(1) == 1);
  CHECK(budget.active(2) == 1);
  std::stop_source cancel;
  auto waiter = std::async(std::launch::async, [&] {
    return budget.acquire({{1, 1}}, make_query_id(2), cancel.get_token(), false);
  });
  bool queued = budget_await([&] { return budget.pending() == 1; });
  cancel.request_stop();
  CHECK_FALSE(waiter.get());
  CHECK(queued);
  CHECK(budget.active(1) == 1);
  CHECK(budget.pending() == 0);
  first.reset();
  CHECK(budget.active(1) == 0);
  CHECK(budget.active(2) == 0);
}
TEST_CASE("shared scan waiters favor oldest query", "[shared_scan_budget]")
{
  shared_scan_budget budget;
  auto held = budget.acquire({{1, 1}}, make_query_id(1), {}, false);
  std::stop_source cancel;
  auto newer         = std::async(std::launch::async, [&] {
    return budget.acquire({{1, 1}}, make_query_id(3), cancel.get_token(), false);
  });
  bool newer_waiting = budget_await([&] { return budget.pending() == 1; });
  auto older         = std::async(std::launch::async, [&] {
    return budget.acquire({{1, 1}}, make_query_id(2), cancel.get_token(), false);
  });
  bool both_waiting  = budget_await([&] { return budget.pending() == 2; });
  held.reset();
  bool older_ready = older.wait_for(1s) == std::future_status::ready;
  bool newer_ready = newer.wait_for(0ms) == std::future_status::ready;
  if (!older_ready) cancel.request_stop();
  auto first = older.get();
  first.reset();
  auto second = newer.get();
  CHECK(newer_waiting);
  CHECK(both_waiting);
  CHECK(older_ready);
  CHECK_FALSE(newer_ready);
  CHECK(second);
}

TEST_CASE("shared scan limits remain stable until all users retire", "[shared_scan_budget]")
{
  shared_scan_budget budget;
  auto held = budget.acquire({{1, 1}}, make_query_id(1), {}, false);
  REQUIRE_THROWS_AS(budget.acquire({{1, 2}}, make_query_id(2), {}, true), std::invalid_argument);
  REQUIRE_THROWS_AS(budget.acquire({{2, 3}, {1, 2}}, make_query_id(2), {}, false),
                    std::invalid_argument);
  CHECK(budget.active(1) == 1);
  CHECK(budget.active(2) == 0);
  CHECK(budget.pending() == 0);
  // A rejected composite request must not register its other backend.
  auto other = budget.acquire({{2, 4}}, make_query_id(2), {}, false);
  CHECK(budget.active(2) == 1);
  held.reset();
  // Reusing an idle address starts new accounting with the new backend's limit.
  auto replacement = budget.acquire({{1, 2}}, make_query_id(3), {}, false);
  auto second      = budget.acquire({{1, 2}}, make_query_id(4), {}, false);
  CHECK(budget.active(1) == 2);
}

TEST_CASE("shared scan requests validate duplicate and zero limits", "[shared_scan_budget]")
{
  shared_scan_budget budget;
  REQUIRE_THROWS_AS(budget.acquire({{1, 1}, {1, 2}}, make_query_id(1), {}, false),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(budget.acquire({{1, 0}}, make_query_id(1), {}, false), std::invalid_argument);
  CHECK(budget.active(1) == 0);
  auto held = budget.acquire({{1, 2}, {1, 2}}, make_query_id(1), {}, false);
  CHECK(budget.active(1) == 1);
}

TEST_CASE("shared scan cancellation cleans idle backend state but preserves live tickets",
          "[shared_scan_budget]")
{
  shared_scan_budget budget;
  auto held = budget.acquire({{1, 1}}, make_query_id(1), {}, false);
  std::stop_source stop;
  auto waiting = std::async(std::launch::async, [&] {
    return budget.acquire({{1, 1}, {2, 3}}, make_query_id(2), stop.get_token(), false);
  });
  bool queued  = budget_await([&] { return budget.pending() == 1; });
  // Backend 2 is idle but belongs to a live waiter; its limit must remain stable.
  bool rejected = false;
  try {
    auto conflicting = budget.acquire({{2, 4}}, make_query_id(3), {}, true);
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  stop.request_stop();
  auto cancelled = waiting.get();
  CHECK(queued);
  CHECK(rejected);
  CHECK_FALSE(cancelled);
  CHECK(budget.pending() == 0);
  CHECK(budget.active(1) == 1);
  auto replacement = budget.acquire({{2, 4}}, make_query_id(3), {}, false);
  CHECK(budget.active(2) == 1);
}

TEST_CASE("shared scan demand and empty requests bypass speculative waiters",
          "[shared_scan_budget]")
{
  shared_scan_budget budget;
  auto held = budget.acquire({{1, 1}}, make_query_id(1), {}, false);
  std::stop_source stop;
  auto waiting = std::async(std::launch::async, [&] {
    return budget.acquire({{1, 1}}, make_query_id(2), stop.get_token(), false);
  });
  bool queued  = budget_await([&] { return budget.pending() == 1; });
  // Use the same stop token so a regression in the empty-request fast path
  // cannot leave a blocked future when the test fails.
  auto empty       = std::async(std::launch::async, [&] {
    return budget.acquire({}, make_query_id(3), stop.get_token(), false);
  });
  bool empty_ready = empty.wait_for(1s) == std::future_status::ready;
  auto demand      = budget.acquire({{1, 1}}, make_query_id(4), {}, true);
  auto pending     = budget.pending();
  stop.request_stop();
  auto empty_ticket = empty.get();
  auto cancelled    = waiting.get();
  CHECK(queued);
  CHECK(empty_ready);
  CHECK(empty_ticket);
  CHECK_FALSE(cancelled);
  CHECK(pending == 1);
  CHECK(budget.active(1) == 2);
}

TEST_CASE("shared scan tickets outlive their budget facade", "[shared_scan_budget]")
{
  shared_scan_budget::token ticket;
  {
    shared_scan_budget budget;
    ticket = budget.acquire({{1, 1}}, make_query_id(1), {}, false);
    CHECK(budget.active(1) == 1);
  }
  ticket.reset();
}
