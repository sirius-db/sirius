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
#include "exec/thread_pool.hpp"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

namespace {

cpu_set_t current_cpu_mask()
{
  cpu_set_t mask;
  CPU_ZERO(&mask);
  REQUIRE(sched_getaffinity(0, sizeof(cpu_set_t), &mask) == 0);
  return mask;
}

std::vector<int> cpu_ids_in(const cpu_set_t& mask)
{
  std::vector<int> ids;
  for (int id = 0; id < CPU_SETSIZE; ++id) {
    if (CPU_ISSET(id, &mask)) { ids.push_back(id); }
  }
  return ids;
}

bool masks_equal(const cpu_set_t& lhs, const cpu_set_t& rhs)
{
  for (int id = 0; id < CPU_SETSIZE; ++id) {
    if (CPU_ISSET(id, &lhs) != CPU_ISSET(id, &rhs)) { return false; }
  }
  return true;
}

cpu_set_t mask_for(const std::vector<int>& cpu_ids)
{
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int id : cpu_ids) {
    CPU_SET(id, &mask);
  }
  return mask;
}

}  // namespace

// =============================================================================
// CPU affinity
// =============================================================================

TEST_CASE("thread pools apply one shared CPU affinity mask to every worker",
          "[thread_pool][bounded_thread_pool][cpu_affinity]")
{
  auto const allowed = cpu_ids_in(current_cpu_mask());
  REQUIRE_FALSE(allowed.empty());

  std::vector<int> requested{allowed.front()};
  if (allowed.size() > 1) { requested.push_back(allowed[1]); }
  auto const expected = mask_for(requested);

  SECTION("static_thread_pool")
  {
    std::atomic<int> initialized{0};
    std::atomic<bool> exact{true};
    static_thread_pool pool(2, "affinity", requested, [&]() noexcept {
      cpu_set_t actual;
      CPU_ZERO(&actual);
      if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &actual) != 0 ||
          !masks_equal(actual, expected)) {
        exact = false;
      }
      initialized.fetch_add(1);
    });
    CHECK(initialized.load() == 2);
    CHECK(exact.load());
  }

  SECTION("bounded_thread_pool")
  {
    std::atomic<int> initialized{0};
    std::atomic<bool> exact{true};
    bounded_thread_pool pool(2, "affinity", requested, [&]() noexcept {
      cpu_set_t actual;
      CPU_ZERO(&actual);
      if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &actual) != 0 ||
          !masks_equal(actual, expected)) {
        exact = false;
      }
      initialized.fetch_add(1);
    });
    CHECK(initialized.load() == 2);
    CHECK(exact.load());
  }
}

TEST_CASE("empty CPU affinity inherits the constructing thread mask",
          "[thread_pool][bounded_thread_pool][cpu_affinity]")
{
  auto const expected = current_cpu_mask();

  SECTION("static_thread_pool")
  {
    std::atomic<bool> inherited{true};
    static_thread_pool pool(1, "affinity", {}, [&]() noexcept {
      cpu_set_t actual;
      CPU_ZERO(&actual);
      inherited = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &actual) == 0 &&
                  masks_equal(actual, expected);
    });
    CHECK(inherited.load());
  }

  SECTION("bounded_thread_pool")
  {
    std::atomic<bool> inherited{true};
    bounded_thread_pool pool(1, "affinity", {}, [&]() noexcept {
      cpu_set_t actual;
      CPU_ZERO(&actual);
      inherited = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &actual) == 0 &&
                  masks_equal(actual, expected);
    });
    CHECK(inherited.load());
  }
}

TEST_CASE("thread pools reject invalid CPU affinity before creating workers",
          "[thread_pool][bounded_thread_pool][cpu_affinity]")
{
  int invalid_id = -1;

  SECTION("negative ID") { invalid_id = -1; }
  SECTION("ID at CPU_SETSIZE") { invalid_id = CPU_SETSIZE; }
  SECTION("ID outside the current allowed mask")
  {
    auto const allowed = current_cpu_mask();
    invalid_id         = -1;
    for (int id = 0; id < CPU_SETSIZE; ++id) {
      if (!CPU_ISSET(id, &allowed)) {
        invalid_id = id;
        break;
      }
    }
    if (invalid_id < 0) {
      WARN("current process is allowed on every CPU_SETSIZE slot; disallowed-ID case skipped");
      return;
    }
  }

  CHECK_THROWS_AS(static_thread_pool(1, "invalid", {invalid_id}), std::invalid_argument);
  CHECK_THROWS_AS(bounded_thread_pool(1, "invalid", {invalid_id}), std::invalid_argument);

  // A failed construction must not leave a joinable worker behind or poison later pools.
  CHECK_NOTHROW(static_thread_pool(1, "valid"));
  CHECK_NOTHROW(bounded_thread_pool(1, "valid"));
}

TEST_CASE("thread pool startup propagates worker errors after joining every worker",
          "[thread_pool][cpu_affinity][startup]")
{
  std::vector<std::thread> threads;
  threads.reserve(3);
  std::atomic<int> startup_count{0};
  std::atomic<bool> stop{false};

  CHECK_THROWS_WITH(detail::start_thread_pool_workers(
                      threads,
                      3,
                      "startup",
                      [&] {
                        if (startup_count.fetch_add(1) == 1) {
                          throw std::runtime_error("forced affinity startup failure");
                        }
                      },
                      [&] {
                        while (!stop.load()) {
                          std::this_thread::yield();
                        }
                      },
                      [&] { stop = true; }),
                    "forced affinity startup failure");

  CHECK(startup_count.load() == 3);
  CHECK(std::none_of(
    threads.begin(), threads.end(), [](auto const& thread) { return thread.joinable(); }));
}

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
      (void)pool.reserve();  // blocks until a slot is available or interrupted
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
