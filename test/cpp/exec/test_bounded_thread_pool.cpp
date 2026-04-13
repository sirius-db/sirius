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

#include "catch.hpp"
#include "exec/bounded_thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// =============================================================================
// Bounded concurrency
// =============================================================================

TEST_CASE("bounded_thread_pool respects capacity — never exceeds N concurrent tasks",
          "[bounded_thread_pool]")
{
  constexpr int capacity = 3;
  bounded_thread_pool pool(capacity, "test");

  std::atomic<int> active{0};
  std::atomic<int> peak{0};
  std::mutex mu;

  // Schedule more tasks than capacity; each holds briefly so overlap is observable.
  for (int i = 0; i < 12; ++i) {
    auto s = pool.reserve();
    pool.dispatch(std::move(s), [&active, &peak, &mu] {
      int cur = active.fetch_add(1) + 1;
      {
        std::lock_guard lock(mu);
        if (cur > peak.load()) { peak.store(cur); }
      }
      std::this_thread::sleep_for(5ms);
      active.fetch_sub(1);
    });
  }

  pool.wait_all();
  REQUIRE(peak.load() <= capacity);
}

// =============================================================================
// reserve() + dispatch()
// =============================================================================

TEST_CASE("bounded_thread_pool reserve and dispatch executes task", "[bounded_thread_pool]")
{
  bounded_thread_pool pool(2, "test");

  std::atomic<bool> ran{false};
  auto slot = pool.reserve();
  REQUIRE(slot.is_valid());

  pool.dispatch(std::move(slot), [&ran] { ran = true; });
  REQUIRE_FALSE(slot.is_valid());  // consumed by move

  pool.wait_all();
  REQUIRE(ran.load());
}

TEST_CASE("bounded_thread_pool reserve blocks when at capacity", "[bounded_thread_pool]")
{
  constexpr int capacity = 1;
  bounded_thread_pool pool(capacity, "test");

  std::atomic<bool> gate{false};
  auto slot = pool.reserve();
  REQUIRE(slot.is_valid());
  pool.dispatch(std::move(slot), [&gate] {
    while (!gate.load()) {
      std::this_thread::yield();
    }
  });

  // Second reserve() should block.
  std::atomic<bool> second_reserved{false};
  std::thread t([&pool, &second_reserved] {
    auto s          = pool.reserve();
    second_reserved = s.is_valid();
  });

  std::this_thread::sleep_for(30ms);
  REQUIRE_FALSE(second_reserved.load());

  gate = true;
  t.join();
  pool.wait_all();
  REQUIRE(second_reserved.load());
}

// =============================================================================
// RAII slot release without dispatch
// =============================================================================

TEST_CASE("bounded_thread_pool slot dropped without dispatch releases slot",
          "[bounded_thread_pool]")
{
  constexpr int capacity = 1;
  bounded_thread_pool pool(capacity, "test");

  {
    auto slot = pool.reserve();
    REQUIRE(slot.is_valid());
    // Destroy slot without calling dispatch() — should release the slot.
  }

  // Slot released; we should be able to reserve again without blocking.
  std::atomic<bool> done{false};
  std::thread t([&pool, &done] {
    auto slot = pool.reserve();
    done      = slot.is_valid();
  });
  t.join();

  REQUIRE(done.load());
}

// =============================================================================
// interrupt() / resume() lifecycle
// =============================================================================

TEST_CASE("bounded_thread_pool interrupt causes reserve to return invalid slot",
          "[bounded_thread_pool]")
{
  constexpr int capacity = 1;
  bounded_thread_pool pool(capacity, "test");

  std::atomic<bool> gate{false};
  auto slot = pool.reserve();
  pool.dispatch(std::move(slot), [&gate] {
    while (!gate.load()) {
      std::this_thread::yield();
    }
  });

  std::atomic<bool> reserve_returned{false};
  std::atomic<bool> slot_valid{true};
  std::thread t([&pool, &reserve_returned, &slot_valid] {
    auto s           = pool.reserve();
    slot_valid       = s.is_valid();
    reserve_returned = true;
  });

  std::this_thread::sleep_for(30ms);
  REQUIRE_FALSE(reserve_returned.load());

  pool.interrupt();
  t.join();

  REQUIRE(reserve_returned.load());
  REQUIRE_FALSE(slot_valid.load());

  gate = true;
  pool.wait_all();
}

TEST_CASE("bounded_thread_pool interrupt wakes multiple blocked callers", "[bounded_thread_pool]")
{
  constexpr int capacity = 1;
  bounded_thread_pool pool(capacity, "test");

  std::atomic<bool> gate{false};
  {
    auto s = pool.reserve();
    pool.dispatch(std::move(s), [&gate] {
      while (!gate.load()) {
        std::this_thread::yield();
      }
    });
  }

  constexpr int num_waiters = 4;
  std::atomic<int> woken{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < num_waiters; ++i) {
    threads.emplace_back([&pool, &woken] {
      pool.reserve();  // blocks until a slot is available or interrupted
      woken.fetch_add(1);
    });
  }

  std::this_thread::sleep_for(30ms);
  REQUIRE(woken.load() == 0);

  pool.interrupt();
  for (auto& t : threads) {
    t.join();
  }
  REQUIRE(woken.load() == num_waiters);

  gate = true;
  pool.wait_all();
}

TEST_CASE("bounded_thread_pool resume re-enables reserve after interrupt", "[bounded_thread_pool]")
{
  bounded_thread_pool pool(2, "test");

  pool.interrupt();

  // reserve() should return an invalid slot while interrupted.
  {
    auto s = pool.reserve();
    REQUIRE_FALSE(s.is_valid());
  }

  pool.resume();

  // reserve() should work again after resume.
  std::atomic<bool> ran{false};
  auto s = pool.reserve();
  REQUIRE(s.is_valid());
  pool.dispatch(std::move(s), [&ran] { ran = true; });
  pool.wait_all();
  REQUIRE(ran.load());
}

// =============================================================================
// wait_all()
// =============================================================================

TEST_CASE("bounded_thread_pool wait_all returns only after all tasks complete",
          "[bounded_thread_pool]")
{
  bounded_thread_pool pool(4, "test");

  std::atomic<int> completed{0};
  constexpr int num_tasks = 8;

  for (int i = 0; i < num_tasks; ++i) {
    auto s = pool.reserve();
    pool.dispatch(std::move(s), [&completed] {
      std::this_thread::sleep_for(10ms);
      completed.fetch_add(1);
    });
  }

  pool.wait_all();
  REQUIRE(completed.load() == num_tasks);
}

TEST_CASE("bounded_thread_pool wait_all returns immediately when pool is idle",
          "[bounded_thread_pool]")
{
  bounded_thread_pool pool(2, "test");
  REQUIRE_NOTHROW(pool.wait_all());
}

// =============================================================================
// Exception safety
// =============================================================================

TEST_CASE("bounded_thread_pool exception in task does not crash the pool", "[bounded_thread_pool]")
{
  bounded_thread_pool pool(2, "test");

  // Task that throws — pool should catch it and remain functional.
  {
    auto s = pool.reserve();
    pool.dispatch(std::move(s), [] { throw std::runtime_error("intentional"); });
  }
  pool.wait_all();

  // Pool still works after the exception.
  std::atomic<bool> ran{false};
  auto s = pool.reserve();
  REQUIRE(s.is_valid());
  pool.dispatch(std::move(s), [&ran] { ran = true; });
  pool.wait_all();
  REQUIRE(ran.load());
}

// =============================================================================
// stop() safety
// =============================================================================

TEST_CASE("bounded_thread_pool stop is idempotent", "[bounded_thread_pool]")
{
  bounded_thread_pool pool(2, "test");
  REQUIRE_NOTHROW(pool.stop());
  REQUIRE_NOTHROW(pool.stop());
}

TEST_CASE("bounded_thread_pool destructor stops cleanly with in-flight tasks",
          "[bounded_thread_pool]")
{
  std::atomic<int> completed{0};
  {
    bounded_thread_pool pool(2, "test");
    auto s = pool.reserve();
    pool.dispatch(std::move(s), [&completed] {
      std::this_thread::sleep_for(5ms);
      completed.fetch_add(1);
    });
    // Destructor calls stop(), which joins workers after they finish.
  }
  REQUIRE(completed.load() == 1);
}

// =============================================================================
// Concurrent producers
// =============================================================================

TEST_CASE("bounded_thread_pool concurrent producers all tasks execute", "[bounded_thread_pool]")
{
  constexpr int capacity    = 4;
  constexpr int num_threads = 8;
  constexpr int tasks_each  = 10;

  bounded_thread_pool pool(capacity, "test");
  std::atomic<int> counter{0};

  std::vector<std::thread> producers;
  for (int i = 0; i < num_threads; ++i) {
    producers.emplace_back([&pool, &counter] {
      for (int j = 0; j < tasks_each; ++j) {
        auto s = pool.reserve();
        pool.dispatch(std::move(s), [&counter] { counter.fetch_add(1); });
      }
    });
  }
  for (auto& t : producers) {
    t.join();
  }

  pool.wait_all();
  REQUIRE(counter.load() == num_threads * tasks_each);
}
