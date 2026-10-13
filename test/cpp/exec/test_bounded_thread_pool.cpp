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
#include "exec/leased_callable.hpp"
#include "query_id.hpp"

#include <absl/cleanup/cleanup.h>

#include <atomic>
#include <chrono>
#include <future>
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
// Query retirement uses leases; pool tags identify queued work only.
//===----------------------------------------------------------------------===//

namespace {
struct release_on_exit {
  std::atomic<bool>& flag;
  ~release_on_exit() { flag.store(true, std::memory_order_release); }
};
}  // namespace

TEST_CASE("query retirement ignores other queries and idle pool reservations",
          "[bounded_thread_pool][query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry lifecycle;
  const auto qa = sirius::make_query_id(1);
  const auto qb = sirius::make_query_id(2);
  lifecycle.open_query(qa);
  lifecycle.open_query(qb);
  bounded_thread_pool pool(5, "test");
  auto parked = pool.reserve();  // An idle manager is capacity, not query work.
  std::atomic<bool> release_b{false};
  release_on_exit guard{release_b};
  std::atomic<int> a_done{0};
  auto work_b = lifecycle.try_acquire_work(qb);
  dispatch_with_work(
    pool,
    pool.reserve(),
    qb,
    std::move(work_b),
    [](std::exception_ptr) noexcept {},
    [&] {
      while (!release_b.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    });
  for (int i = 0; i < 3; ++i) {
    auto work = lifecycle.try_acquire_work(qa);
    dispatch_with_work(
      pool,
      pool.reserve(),
      qa,
      std::move(work),
      [](std::exception_ptr) noexcept {},
      [&] { ++a_done; });
  }
  lifecycle.quiesce_and_wait_for_submissions(qa);
  lifecycle.wait_for_work(qa);
  CHECK(a_done == 3);  // Successful retirement ran, rather than discarded, the accepted work.
  CHECK(lifecycle.activity(qa).work == 0);
  CHECK(lifecycle.activity(qb).work == 1);
  CHECK(lifecycle.accepts_work(qb));
  CHECK(parked.is_valid());
  release_b.store(true, std::memory_order_release);
  lifecycle.quiesce_and_wait_for_submissions(qb);
  lifecycle.wait_for_work(qb);
  lifecycle.close(qa);
  lifecycle.close(qb);
}

TEST_CASE("a manager-held lease covers dispatch after the pool cancellation sweep",
          "[bounded_thread_pool][query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry lifecycle;
  const auto q = sirius::make_query_id(7);
  lifecycle.open_query(q);
  bounded_thread_pool pool(1, "test");
  auto slot      = pool.reserve();
  auto held_work = lifecycle.try_acquire_work(q);
  lifecycle.quiesce_and_wait_for_submissions(q);
  pool.discard_queued(q);  // The manager has popped the task but has not dispatched it yet.
  auto retired = std::async(std::launch::async, [&] { lifecycle.wait_for_work(q); });
  CHECK(retired.wait_for(20ms) == std::future_status::timeout);
  std::atomic<bool> ran{false};
  dispatch_with_work(
    pool,
    std::move(slot),
    q,
    std::move(held_work),
    [](std::exception_ptr) noexcept {},
    [&] { ran = true; });
  retired.get();
  CHECK(ran);
  CHECK(lifecycle.activity(q).work == 0);
  pool.wait_all();
  lifecycle.close(q);
}

TEST_CASE("pool cancellation discards only matching queued work outside its mutex",
          "[bounded_thread_pool][query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry lifecycle;
  const auto qa = sirius::make_query_id(3);
  const auto qb = sirius::make_query_id(4);
  lifecycle.open_query(qa);
  lifecycle.open_query(qb);
  bounded_thread_pool pool(3, "test");
  auto first  = pool.reserve();
  auto second = pool.reserve();
  auto other  = pool.reserve();
  // Stop idle workers after reserving so queued work cannot race with worker execution.
  pool.stop();
  std::atomic<int> ran{0};
  bool destroyed = false;
  struct reentrant_capture {
    bounded_thread_pool& pool;
    bool& destroyed;
    ~reentrant_capture()
    {
      pool.discard_queued(sirius::make_query_id(99));
      destroyed = true;
    }
  };
  auto owned      = std::unique_ptr<reentrant_capture>(new reentrant_capture{pool, destroyed});
  auto first_work = lifecycle.try_acquire_work(qa);
  dispatch_with_work(
    pool,
    std::move(first),
    qa,
    std::move(first_work),
    [](std::exception_ptr) noexcept {},
    [&, owned = std::move(owned)] { ++ran; });
  auto second_work = lifecycle.try_acquire_work(qa);
  dispatch_with_work(
    pool,
    std::move(second),
    qa,
    std::move(second_work),
    [](std::exception_ptr) noexcept {},
    [&] { ++ran; });
  auto other_work = lifecycle.try_acquire_work(qb);
  dispatch_with_work(
    pool,
    std::move(other),
    qb,
    std::move(other_work),
    [](std::exception_ptr) noexcept {},
    [&] { ++ran; });
  absl::Cleanup cleanup = [&] {
    pool.discard_queued(qa);
    pool.discard_queued(qb);
  };
  lifecycle.quiesce_and_wait_for_submissions(qa);
  lifecycle.quiesce_and_wait_for_submissions(qb);
  CHECK(lifecycle.activity(qa).work == 2);
  CHECK(lifecycle.activity(qb).work == 1);
  pool.discard_queued(qa);
  CHECK(destroyed);
  CHECK(ran == 0);
  CHECK(lifecycle.activity(qa).work == 0);
  CHECK(lifecycle.activity(qb).work == 1);
  pool.discard_queued(qb);
  CHECK(ran == 0);
  CHECK(lifecycle.activity(qb).work == 0);
  lifecycle.wait_for_work(qa);
  lifecycle.wait_for_work(qb);
  lifecycle.close(qa);
  lifecycle.close(qb);
  pool.wait_all();
}

TEST_CASE("a worker lease outlives its callback and captured destructors",
          "[bounded_thread_pool][query_lifecycle_gate][concurrency]")
{
  query_lifecycle_registry lifecycle;
  const auto q = sirius::make_query_id(1);
  lifecycle.open_query(q);
  std::promise<void> destroying_capture;
  std::promise<void> release_capture;
  auto release = release_capture.get_future().share();
  std::atomic<std::size_t> work_at_destruction{0};
  struct capture {
    query_lifecycle_registry& lifecycle;
    sirius::query_id_t query;
    std::promise<void>& destroying;
    std::shared_future<void> release;
    std::atomic<std::size_t>& observed;
    ~capture()
    {
      observed = lifecycle.activity(query).work;
      destroying.set_value();
      release.wait();
    }
  };
  bounded_thread_pool pool(1, "lease-test");
  absl::Cleanup unblock = [&] { release_capture.set_value(); };
  auto owned            = std::unique_ptr<capture>(
    new capture{lifecycle, q, destroying_capture, release, work_at_destruction});
  auto work = lifecycle.try_acquire_work(q);
  pool.dispatch(pool.reserve(), leased_callable{std::move(work), [owned = std::move(owned)] {}});
  destroying_capture.get_future().wait();
  lifecycle.quiesce_and_wait_for_submissions(q);
  // The body has returned, but resource-using capture destruction has not finished.
  CHECK(work_at_destruction == 1);
  CHECK(lifecycle.activity(q).work == 1);
  auto retired = std::async(std::launch::async, [&] { lifecycle.wait_for_work(q); });
  CHECK(retired.wait_for(20ms) == std::future_status::timeout);
  release_capture.set_value();
  std::move(unblock).Cancel();
  retired.get();
  pool.wait_all();
  CHECK(lifecycle.activity(q).work == 0);
  lifecycle.close(q);
}

TEST_CASE("leased callables retain captures on exception and discard",
          "[bounded_thread_pool][query_lifecycle_gate]")
{
  query_lifecycle_registry lifecycle;
  const auto q = sirius::make_query_id(1);
  lifecycle.open_query(q);
  std::size_t observed = 0;
  struct capture {
    query_lifecycle_registry& lifecycle;
    sirius::query_id_t query;
    std::size_t& observed;
    ~capture() { observed = lifecycle.activity(query).work; }
  };
  auto owned                 = std::unique_ptr<capture>(new capture{lifecycle, q, observed});
  auto work                  = lifecycle.try_acquire_work(q);
  invocable<void()> callable = leased_callable{
    std::move(work), [owned = std::move(owned)] { throw std::runtime_error("worker failure"); }};
  SECTION("exception") { CHECK_THROWS_AS(callable(), std::runtime_error); }
  SECTION("discard without invocation") {}
  lifecycle.quiesce_and_wait_for_submissions(q);
  CHECK(lifecycle.activity(q).work == 1);
  callable = nullptr;
  CHECK(observed == 1);
  lifecycle.wait_for_work(q);
  lifecycle.close(q);
}
