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
