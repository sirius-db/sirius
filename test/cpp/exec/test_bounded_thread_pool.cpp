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
#include "query_id.hpp"

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

//===----------------------------------------------------------------------===//
// Per-query accounting: wait_for_query / drain_and_wait
//===----------------------------------------------------------------------===//

namespace {
//! Releases a blocking task on every exit path, so a failed REQUIRE cannot leave a worker spinning
//! and hang the pool destructor's join().
struct release_on_exit {
  std::atomic<bool>& flag;
  ~release_on_exit() { flag.store(true, std::memory_order_release); }
};
}  // namespace

TEST_CASE("wait_for_query runs the query's work and ignores other queries",
          "[bounded_thread_pool][concurrency]")
{
  bounded_thread_pool pool(4, "test");
  const auto qa = sirius::make_query_id(1);
  const auto qb = sirius::make_query_id(2);

  std::atomic<bool> release_b{false};
  release_on_exit guard{release_b};
  std::atomic<int> a_done{0};

  // B blocks until told otherwise. If wait_for_query(A) waited on the whole pool it would hang
  // here -- tracking per query is exactly what makes it return.
  {
    auto sb = pool.reserve();
    sb.attach(qb);
    pool.dispatch(std::move(sb), [&] {
      while (!release_b.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  for (int i = 0; i < 3; ++i) {
    auto sa = pool.reserve();
    sa.attach(qa);
    pool.dispatch(std::move(sa), [&] { a_done.fetch_add(1, std::memory_order_relaxed); });
  }

  pool.wait_for_query(qa);
  // RUN, not dropped: the success path must never discard work the query scheduled.
  REQUIRE(a_done.load() == 3);
  REQUIRE(pool.active_for_query(qa) == 0);
  REQUIRE(pool.active_for_query(qb) == 1);  // B untouched

  release_b.store(true, std::memory_order_release);
  pool.wait_all();
}

TEST_CASE("an untagged reservation does not block a per-query wait",
          "[bounded_thread_pool][concurrency]")
{
  // The load-bearing case. Every manager loop in Sirius reserves a slot and THEN blocks waiting
  // for a task, so a slot with no query is permanently held whenever a manager is idle. wait_all()
  // can never return in that state; the per-query waits must, or query teardown deadlocks -- which
  // is exactly how removing the manager-quiesce bracket broke the suite.
  bounded_thread_pool pool(4, "test");
  const auto q = sirius::make_query_id(7);

  auto parked = pool.reserve();  // untagged, held for the whole test
  REQUIRE(parked.is_valid());

  std::atomic<int> ran{0};
  {
    auto s = pool.reserve();
    s.attach(q);
    pool.dispatch(std::move(s), [&] { ran.fetch_add(1, std::memory_order_relaxed); });
  }

  pool.wait_for_query(q);  // must not hang despite `parked` still being held
  REQUIRE(ran.load() == 1);
  REQUIRE(pool.active_for_query(q) == 0);
}

TEST_CASE("wait_for_query_and_untagged waits out untagged slots but not co-tenants",
          "[bounded_thread_pool][concurrency]")
{
  // The error bracket's wait. A task dispatched WITHOUT an attach (its query is unknowable)
  // could belong to the failing query, so it must be waited for; a co-tenant's attributed task
  // must not be — that is the whole point of replacing wait_all() in the bracket.
  bounded_thread_pool pool(4, "test");
  const auto qa = sirius::make_query_id(1);
  const auto qb = sirius::make_query_id(2);

  std::atomic<bool> release_b{false};
  release_on_exit guard_b{release_b};
  std::atomic<bool> release_untagged{false};
  release_on_exit guard_u{release_untagged};

  // Co-tenant B: parked indefinitely, attributed to B.
  {
    auto sb = pool.reserve();
    sb.attach(qb);
    pool.dispatch(std::move(sb), [&] {
      while (!release_b.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  // An untagged task: dispatched without attach, blocked until released.
  {
    auto su = pool.reserve();
    pool.dispatch(std::move(su), [&] {
      while (!release_untagged.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  // A quick task of the waiting query.
  std::atomic<int> a_done{0};
  {
    auto sa = pool.reserve();
    sa.attach(qa);
    pool.dispatch(std::move(sa), [&] { a_done.fetch_add(1, std::memory_order_relaxed); });
  }

  std::atomic<bool> wait_returned{false};
  std::thread waiter([&] {
    pool.wait_for_query_and_untagged(qa);
    wait_returned.store(true, std::memory_order_release);
  });

  // Blocked by the untagged slot even after A's own work is long done.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const bool blocked_on_untagged = !wait_returned.load(std::memory_order_acquire);

  // Releasing the untagged task is enough — the co-tenant stays parked.
  release_untagged.store(true, std::memory_order_release);
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!wait_returned.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool returned_with_cotenant_parked =
    wait_returned.load(std::memory_order_acquire) && pool.active_for_query(qb) == 1;

  release_b.store(true, std::memory_order_release);
  waiter.join();
  pool.wait_all();

  REQUIRE(a_done.load() == 1);
  REQUIRE(blocked_on_untagged);
  REQUIRE(returned_with_cotenant_parked);
}

TEST_CASE("active_untagged tracks the reserve-to-attach window", "[bounded_thread_pool]")
{
  bounded_thread_pool pool(2, "test");
  REQUIRE(pool.active_untagged() == 0);

  const auto q = sirius::make_query_id(9);
  {
    auto s = pool.reserve();
    REQUIRE(pool.active_untagged() == 1);  // reserved but not yet attributed
    s.attach(q);
    REQUIRE(pool.active_untagged() == 0);  // attribution moved it into the query's count
    REQUIRE(pool.active_for_query(q) == 1);
  }
  REQUIRE(pool.active_untagged() == 0);
  REQUIRE(pool.active_for_query(q) == 0);
}

TEST_CASE("drain_and_wait discards the query's queued work", "[bounded_thread_pool][concurrency]")
{
  // The error-path counterpart: queued work is dropped rather than run, because its query is
  // failing and its plan is about to be destroyed. Single worker, held by a blocker, so everything
  // dispatched behind it is still queued when the drain runs.
  bounded_thread_pool pool(1, "test");
  const auto q = sirius::make_query_id(3);

  std::atomic<bool> release{false};
  release_on_exit guard{release};
  std::atomic<int> ran{0};

  auto blocker = pool.reserve();
  pool.dispatch(std::move(blocker), [&] {
    while (!release.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  // Capacity is 1 and taken, so reserve() would block here; dispatch from another thread.
  std::atomic<int> dispatched{0};
  std::thread producer([&] {
    for (int i = 0; i < 2; ++i) {
      auto s = pool.reserve();
      if (!s) { return; }
      s.attach(q);
      pool.dispatch(std::move(s), [&] { ran.fetch_add(1, std::memory_order_relaxed); });
      dispatched.fetch_add(1, std::memory_order_relaxed);
    }
  });

  release.store(true, std::memory_order_release);
  producer.join();
  pool.drain_and_wait(q);
  REQUIRE(pool.active_for_query(q) == 0);
  REQUIRE(ran.load() <= dispatched.load());  // some may have run before the drain; none may leak
  pool.wait_all();
}
