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

#include "exec/work_tracker.hpp"

#include <catch.hpp>

#include <chrono>
#include <thread>
#include <utility>
#include <vector>

using sirius::exec::work_tracker;
using namespace std::chrono_literals;

TEST_CASE("work_tracker counts slots from acquire to destruction", "[work_tracker]")
{
  work_tracker tracker;
  REQUIRE(tracker.outstanding() == 0);
  REQUIRE(tracker.wait_quiescent(0ms));

  {
    auto a = tracker.acquire();
    auto b = tracker.acquire();
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(tracker.outstanding() == 2);
    REQUIRE_FALSE(tracker.wait_quiescent(0ms));
  }
  REQUIRE(tracker.outstanding() == 0);
  REQUIRE(tracker.wait_quiescent(0ms));
}

TEST_CASE("work_tracker slots move without double counting", "[work_tracker]")
{
  work_tracker tracker;
  auto a = tracker.acquire();
  REQUIRE(tracker.outstanding() == 1);

  work_tracker::slot b = std::move(a);
  REQUIRE(tracker.outstanding() == 1);
  REQUIRE_FALSE(a);
  REQUIRE(b);

  auto c = tracker.acquire();
  REQUIRE(tracker.outstanding() == 2);
  c = std::move(b);
  REQUIRE(tracker.outstanding() == 1);

  auto& c_ref = c;
  c           = std::move(c_ref);
  REQUIRE(tracker.outstanding() == 1);
}

TEST_CASE("work_tracker wait_quiescent wakes when the last slot releases from another thread",
          "[work_tracker]")
{
  work_tracker tracker;
  auto slot = tracker.acquire();

  std::thread releaser([held = std::move(slot)]() mutable { std::this_thread::sleep_for(50ms); });

  REQUIRE(tracker.wait_quiescent(10000ms));
  releaser.join();
  REQUIRE(tracker.outstanding() == 0);
}

TEST_CASE("work_tracker keeps counting work acquired while a waiter is blocked", "[work_tracker]")
{
  work_tracker tracker;
  auto first = tracker.acquire();

  std::thread handoff([&tracker, held = std::move(first)]() mutable {
    std::this_thread::sleep_for(20ms);
    auto second = tracker.acquire();  // new work while the waiter blocks
    held        = work_tracker::slot{};
    std::this_thread::sleep_for(20ms);
  });

  REQUIRE(tracker.wait_quiescent(10000ms));
  REQUIRE(tracker.outstanding() == 0);
  handoff.join();
}

TEST_CASE("work_tracker slots may outlive the tracker", "[work_tracker]")
{
  work_tracker::slot survivor;
  {
    work_tracker tracker;
    survivor = tracker.acquire();
  }
  survivor = work_tracker::slot{};
  SUCCEED("slot released after tracker destruction without fault");
}

TEST_CASE("work_tracker close makes zero permanent", "[work_tracker]")
{
  work_tracker tracker;
  auto before_close = tracker.acquire();
  REQUIRE(before_close);

  tracker.close();

  auto after_close = tracker.acquire();
  REQUIRE_FALSE(after_close);
  REQUIRE(tracker.outstanding() == 1);
  REQUIRE_FALSE(tracker.wait_quiescent(0ms));

  before_close = {};
  REQUIRE(tracker.wait_quiescent(0ms));
  REQUIRE(tracker.outstanding() == 0);
}

TEST_CASE("work_tracker concurrent acquire/release converges to zero", "[work_tracker]")
{
  work_tracker tracker;
  constexpr int threads_n  = 8;
  constexpr int per_thread = 500;

  std::vector<std::thread> threads;
  threads.reserve(threads_n);
  for (int t = 0; t < threads_n; ++t) {
    threads.emplace_back([&tracker] {
      for (int i = 0; i < per_thread; ++i) {
        auto slot                     = tracker.acquire();
        [[maybe_unused]] auto carrier = std::move(slot);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  REQUIRE(tracker.outstanding() == 0);
  REQUIRE(tracker.wait_quiescent(0ms));
}
