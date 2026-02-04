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
#include "exec/kiosk.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// =============================================================================
// Unbounded kiosk tests
// =============================================================================

TEST_CASE("unbounded kiosk basic acquire and release", "[kiosk][unbounded]")
{
  kiosk k;

  REQUIRE_FALSE(k.is_bounded());
  REQUIRE(k.max_capacity() == 0);
  REQUIRE(k.active_count() == 0);
  REQUIRE(k.total_issued() == 0);

  {
    auto t = k.acquire();
    REQUIRE(t.is_valid());
    REQUIRE(k.active_count() == 1);
    REQUIRE(k.total_issued() == 1);
  }

  // Ticket released on scope exit
  REQUIRE(k.active_count() == 0);
  REQUIRE(k.total_issued() == 1);
}

TEST_CASE("unbounded kiosk multiple concurrent acquisitions", "[kiosk][unbounded]")
{
  kiosk k;
  constexpr int num_tickets = 100;

  std::vector<ticket> tickets;
  tickets.reserve(num_tickets);

  for (int i = 0; i < num_tickets; ++i) {
    tickets.push_back(k.acquire());
    REQUIRE(tickets.back().is_valid());
  }

  REQUIRE(k.active_count() == num_tickets);
  REQUIRE(k.total_issued() == num_tickets);

  tickets.clear();
  REQUIRE(k.active_count() == 0);
  REQUIRE(k.total_issued() == num_tickets);
}

TEST_CASE("unbounded kiosk try_acquire always succeeds", "[kiosk][unbounded]")
{
  kiosk k;

  std::vector<ticket> tickets;
  tickets.reserve(50);
  for (int i = 0; i < 50; ++i) {
    auto t = k.try_acquire();
    REQUIRE(t.has_value());
    REQUIRE(t->is_valid());
    tickets.push_back(std::move(*t));
  }

  REQUIRE(k.active_count() == 50);
}

TEST_CASE("unbounded kiosk try_acquire_for always succeeds", "[kiosk][unbounded]")
{
  kiosk k;

  auto t = k.try_acquire_for(1ms);
  REQUIRE(t.has_value());
  REQUIRE(t->is_valid());
  REQUIRE(k.active_count() == 1);
}

// =============================================================================
// Bounded kiosk tests
// =============================================================================

TEST_CASE("bounded kiosk basic properties", "[kiosk][bounded]")
{
  kiosk k(5);

  REQUIRE(k.is_bounded());
  REQUIRE(k.max_capacity() == 5);
  REQUIRE(k.active_count() == 0);
  REQUIRE(k.total_issued() == 0);
}

TEST_CASE("bounded kiosk acquire within capacity", "[kiosk][bounded]")
{
  kiosk k(3);

  auto t1 = k.acquire();
  REQUIRE(t1.is_valid());
  REQUIRE(k.active_count() == 1);

  auto t2 = k.acquire();
  REQUIRE(t2.is_valid());
  REQUIRE(k.active_count() == 2);

  auto t3 = k.acquire();
  REQUIRE(t3.is_valid());
  REQUIRE(k.active_count() == 3);

  REQUIRE(k.total_issued() == 3);
}

TEST_CASE("bounded kiosk try_acquire fails at capacity", "[kiosk][bounded]")
{
  kiosk k(2);

  auto t1 = k.try_acquire();
  REQUIRE(t1.has_value());

  auto t2 = k.try_acquire();
  REQUIRE(t2.has_value());

  // At capacity - should fail
  auto t3 = k.try_acquire();
  REQUIRE_FALSE(t3.has_value());

  REQUIRE(k.active_count() == 2);
}

TEST_CASE("bounded kiosk try_acquire succeeds after release", "[kiosk][bounded]")
{
  kiosk k(1);

  auto t1 = k.try_acquire();
  REQUIRE(t1.has_value());

  auto t2 = k.try_acquire();
  REQUIRE_FALSE(t2.has_value());

  t1->release();

  auto t3 = k.try_acquire();
  REQUIRE(t3.has_value());
}

TEST_CASE("bounded kiosk try_acquire_for times out at capacity", "[kiosk][bounded]")
{
  kiosk k(1);

  auto t1 = k.try_acquire();
  REQUIRE(t1.has_value());

  auto start  = std::chrono::steady_clock::now();
  auto t2     = k.try_acquire_for(100ms);
  auto end    = std::chrono::steady_clock::now();
  auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  REQUIRE_FALSE(t2.has_value());
  REQUIRE(waited >= 90ms);  // Allow some tolerance
}

TEST_CASE("bounded kiosk try_acquire_for succeeds when released", "[kiosk][bounded]")
{
  kiosk k(1);

  auto t1 = k.acquire();
  REQUIRE(t1.is_valid());

  std::thread releaser([&]() {
    std::this_thread::sleep_for(50ms);
    t1.release();
  });

  auto t2 = k.try_acquire_for(500ms);
  REQUIRE(t2.has_value());
  REQUIRE(t2->is_valid());

  releaser.join();
}

TEST_CASE("bounded kiosk blocking acquire waits for availability", "[kiosk][bounded]")
{
  kiosk k(1);

  auto t1 = k.acquire();
  REQUIRE(t1.is_valid());

  std::atomic<bool> acquired{false};

  std::thread waiter([&]() {
    auto t2  = k.acquire();
    acquired = t2.is_valid();
  });

  // Give waiter time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE_FALSE(acquired.load());

  // Release ticket - should unblock waiter
  t1.release();

  waiter.join();
  REQUIRE(acquired.load());
}

TEST_CASE("bounded kiosk multiple waiters", "[kiosk][bounded]")
{
  kiosk k(1);
  std::atomic<int> acquired_count{0};
  constexpr int num_waiters = 5;

  // Acquire the only ticket
  auto holder = k.acquire();

  std::vector<std::thread> waiters;
  for (int i = 0; i < num_waiters; ++i) {
    waiters.emplace_back([&]() {
      auto t = k.acquire();
      if (t.is_valid()) {
        acquired_count.fetch_add(1);
        std::this_thread::sleep_for(10ms);  // Hold briefly
      }
    });
  }

  // Give waiters time to start
  std::this_thread::sleep_for(50ms);
  REQUIRE(acquired_count.load() == 0);

  // Release - waiters should proceed one at a time
  holder.release();

  for (auto& w : waiters) {
    w.join();
  }

  REQUIRE(acquired_count.load() == num_waiters);
  REQUIRE(k.total_issued() == num_waiters + 1);
}

// =============================================================================
// Stop/Interruption tests
// =============================================================================

TEST_CASE("kiosk stop causes acquire to return invalid ticket", "[kiosk][stop]")
{
  kiosk k(1);

  // Hold the only ticket
  auto t1 = k.acquire();
  REQUIRE(t1.is_valid());

  std::atomic<bool> got_invalid{false};

  std::thread waiter([&]() {
    auto t2     = k.acquire();
    got_invalid = !t2.is_valid();
  });

  // Give waiter time to block
  std::this_thread::sleep_for(50ms);

  // Stop the kiosk
  k.stop();

  waiter.join();
  REQUIRE(got_invalid.load());
}

TEST_CASE("kiosk stop wakes up multiple blocked threads", "[kiosk][stop]")
{
  kiosk k(1);
  constexpr int num_waiters = 4;
  std::atomic<int> returned_count{0};

  auto holder = k.acquire();

  std::vector<std::thread> waiters;
  for (int i = 0; i < num_waiters; ++i) {
    waiters.emplace_back([&]() {
      auto t = k.acquire();
      returned_count.fetch_add(1);
      // All should get invalid tickets after stop
      REQUIRE_FALSE(t.is_valid());
    });
  }

  // Give waiters time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE(returned_count.load() == 0);

  // Stop should wake all waiters
  k.stop();

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (returned_count.load() < num_waiters) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      for (auto& w : waiters) {
        if (w.joinable()) w.detach();
      }
      FAIL("Timeout waiting for waiters to return");
    }
  }

  for (auto& w : waiters) {
    w.join();
  }
  REQUIRE(returned_count.load() == num_waiters);
}

TEST_CASE("kiosk acquire after stop returns invalid ticket", "[kiosk][stop]")
{
  kiosk k;

  k.stop();

  auto t = k.acquire();
  REQUIRE_FALSE(t.is_valid());
}

TEST_CASE("bounded kiosk acquire after stop returns invalid ticket", "[kiosk][stop]")
{
  kiosk k(5);

  k.stop();

  auto t = k.acquire();
  REQUIRE_FALSE(t.is_valid());
}

// =============================================================================
// wait_all tests
// =============================================================================

TEST_CASE("kiosk wait_all returns immediately when no active tickets", "[kiosk][wait]")
{
  kiosk k;

  auto start = std::chrono::steady_clock::now();
  k.wait_all();
  auto elapsed = std::chrono::steady_clock::now() - start;

  REQUIRE(elapsed < 50ms);
}

TEST_CASE("kiosk wait_all blocks until all tickets released", "[kiosk][wait]")
{
  kiosk k;

  auto t1 = k.acquire();
  auto t2 = k.acquire();

  std::atomic<bool> wait_completed{false};

  std::thread waiter([&]() {
    k.wait_all();
    wait_completed = true;
  });

  // Give waiter time to start
  std::this_thread::sleep_for(50ms);
  REQUIRE_FALSE(wait_completed.load());

  // Release one ticket
  t1.release();
  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(wait_completed.load());

  // Release second ticket - wait should complete
  t2.release();

  waiter.join();
  REQUIRE(wait_completed.load());
}

TEST_CASE("kiosk wait_all_for returns true when all released", "[kiosk][wait]")
{
  kiosk k;

  auto t = k.acquire();

  std::thread releaser([&]() {
    std::this_thread::sleep_for(50ms);
    t.release();
  });

  bool result = k.wait_all_for(500ms);
  REQUIRE(result);

  releaser.join();
}

TEST_CASE("kiosk wait_all_for returns false on timeout", "[kiosk][wait]")
{
  kiosk k;

  auto t = k.acquire();

  auto start  = std::chrono::steady_clock::now();
  bool result = k.wait_all_for(100ms);
  auto end    = std::chrono::steady_clock::now();

  REQUIRE_FALSE(result);
  REQUIRE((end - start) >= 90ms);

  // Clean up
  t.release();
}

TEST_CASE("kiosk wait_all with multiple threads releasing", "[kiosk][wait]")
{
  kiosk k;
  constexpr int num_tickets = 10;

  std::vector<ticket> tickets;
  for (int i = 0; i < num_tickets; ++i) {
    tickets.push_back(k.acquire());
  }

  REQUIRE(k.active_count() == num_tickets);

  std::atomic<bool> wait_done{false};
  std::thread waiter([&]() {
    k.wait_all();
    wait_done = true;
  });

  // Release tickets in separate threads
  std::vector<std::thread> releasers;
  for (int i = 0; i < num_tickets; ++i) {
    releasers.emplace_back([&tickets, i]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(10 * i));
      tickets[i].release();
    });
  }

  for (auto& r : releasers) {
    r.join();
  }

  waiter.join();
  REQUIRE(wait_done.load());
  REQUIRE(k.active_count() == 0);
}

// =============================================================================
// Ticket tests
// =============================================================================

TEST_CASE("ticket RAII auto-release", "[kiosk][ticket]")
{
  kiosk k;

  {
    auto t = k.acquire();
    REQUIRE(k.active_count() == 1);
  }

  REQUIRE(k.active_count() == 0);
}

TEST_CASE("ticket move semantics", "[kiosk][ticket]")
{
  kiosk k;

  auto t1 = k.acquire();
  REQUIRE(t1.is_valid());
  REQUIRE(k.active_count() == 1);

  // Move construct
  ticket t2 = std::move(t1);
  REQUIRE_FALSE(t1.is_valid());
  REQUIRE(t2.is_valid());
  REQUIRE(k.active_count() == 1);

  // Move assign
  ticket t3;
  t3 = std::move(t2);
  REQUIRE_FALSE(t2.is_valid());
  REQUIRE(t3.is_valid());
  REQUIRE(k.active_count() == 1);
}

TEST_CASE("ticket manual release", "[kiosk][ticket]")
{
  kiosk k;

  auto t = k.acquire();
  REQUIRE(k.active_count() == 1);

  t.release();
  REQUIRE(k.active_count() == 0);
  REQUIRE_FALSE(t.is_valid());
}

TEST_CASE("ticket double release is safe", "[kiosk][ticket]")
{
  kiosk k;

  auto t = k.acquire();
  REQUIRE(k.active_count() == 1);

  t.release();
  REQUIRE(k.active_count() == 0);

  // Double release should be safe
  REQUIRE_NOTHROW(t.release());
  REQUIRE(k.active_count() == 0);
}

TEST_CASE("ticket move assign releases previous", "[kiosk][ticket]")
{
  kiosk k;

  auto t1 = k.acquire();
  auto t2 = k.acquire();
  REQUIRE(k.active_count() == 2);

  // Move assign t2 to t1 - should release t1's original ticket
  t1 = std::move(t2);
  REQUIRE(k.active_count() == 1);
  REQUIRE(t1.is_valid());
  REQUIRE_FALSE(t2.is_valid());
}

TEST_CASE("ticket default constructed is invalid", "[kiosk][ticket]")
{
  ticket t;
  REQUIRE_FALSE(t.is_valid());
  REQUIRE_FALSE(static_cast<bool>(t));
}

TEST_CASE("ticket bool conversion", "[kiosk][ticket]")
{
  kiosk k;

  ticket t1;
  REQUIRE_FALSE(static_cast<bool>(t1));

  auto t2 = k.acquire();
  REQUIRE(static_cast<bool>(t2));

  t2.release();
  REQUIRE_FALSE(static_cast<bool>(t2));
}

// =============================================================================
// Edge cases and stress tests
// =============================================================================

TEST_CASE("kiosk concurrent acquire and release stress test", "[kiosk][stress]")
{
  kiosk k(10);
  constexpr int num_threads    = 20;
  constexpr int ops_per_thread = 100;
  std::atomic<int> successful_acquires{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < ops_per_thread; ++j) {
        auto t = k.acquire();
        if (t.is_valid()) {
          successful_acquires.fetch_add(1);
          std::this_thread::sleep_for(1ms);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(successful_acquires.load() == num_threads * ops_per_thread);
  REQUIRE(k.active_count() == 0);
  REQUIRE(k.total_issued() == num_threads * ops_per_thread);
}

TEST_CASE("bounded kiosk capacity of 1 serializes access", "[kiosk][bounded]")
{
  kiosk k(1);
  std::atomic<int> concurrent{0};
  std::atomic<int> max_concurrent{0};
  constexpr int num_threads = 10;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      auto t      = k.acquire();
      int current = concurrent.fetch_add(1) + 1;

      // Track max concurrent
      int expected = max_concurrent.load();
      while (current > expected && !max_concurrent.compare_exchange_weak(expected, current)) {}

      std::this_thread::sleep_for(10ms);
      concurrent.fetch_sub(1);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // With capacity 1, max concurrent should be 1
  REQUIRE(max_concurrent.load() == 1);
}
