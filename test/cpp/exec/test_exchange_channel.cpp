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

// Pure CPU tests — no GPU required.

#include "catch.hpp"
#include "exec/exchange_channel.hpp"

#include <atomic>
#include <chrono>
#include <set>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// ============================================================================
// Helpers
// ============================================================================

static exchange_batch_handle make_handle(uint64_t id, std::size_t bytes = 0)
{
  return exchange_batch_handle{id, bytes};
}

// ============================================================================
// CH-1: fresh channel state
// ============================================================================

TEST_CASE("exchange_channel: fresh channel", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});
  REQUIRE(ch.empty());
  REQUIRE_FALSE(ch.full());
  REQUIRE(ch.size() == 0);
  REQUIRE(ch.size_bytes() == 0);
  REQUIRE_FALSE(ch.closed());
  REQUIRE_FALSE(ch.drained());
}

// ============================================================================
// CH-2: fill to capacity_items
// ============================================================================

TEST_CASE("exchange_channel: fill to capacity_items", "[exchange_channel]")
{
  const std::size_t N = 4;
  exchange_channel ch(exchange_channel::config{.capacity_items = N});

  for (std::size_t i = 0; i < N; ++i) {
    REQUIRE(ch.try_push(make_handle(i)));
  }
  REQUIRE(ch.size() == N);
  REQUIRE(ch.full());
  REQUIRE_FALSE(ch.empty());

  // One more item must be rejected.
  REQUIRE_FALSE(ch.try_push(make_handle(N)));
}

// ============================================================================
// CH-3: byte bound
// ============================================================================

TEST_CASE("exchange_channel: byte bound blocks push when non-empty", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 10, .capacity_bytes = 100});

  // Push an item that brings total bytes to exactly the bound.
  REQUIRE(ch.try_push(make_handle(0, 100)));
  REQUIRE(ch.size_bytes() == 100);
  REQUIRE(ch.full());

  // Next push must be rejected (bytes >= bound AND non-empty).
  REQUIRE_FALSE(ch.try_push(make_handle(1, 1)));
}

// ============================================================================
// CH-4: oversized-batch rule
// ============================================================================

TEST_CASE("exchange_channel: oversized batch admitted into empty channel", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4, .capacity_bytes = 50});

  // A batch larger than capacity_bytes must be admitted when the channel is empty.
  REQUIRE_FALSE(ch.full());
  REQUIRE(ch.try_push(make_handle(0, 200)));
  REQUIRE(ch.size() == 1);
  REQUIRE(ch.size_bytes() == 200);
  // After admission the channel is "full" by bytes.
  REQUIRE(ch.full());
}

// ============================================================================
// CH-5: FIFO ordering and size_bytes tracking
// ============================================================================

TEST_CASE("exchange_channel: FIFO ordering and size_bytes tracking", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 8});

  REQUIRE(ch.try_push(make_handle(10, 100)));
  REQUIRE(ch.try_push(make_handle(20, 200)));
  REQUIRE(ch.try_push(make_handle(30, 300)));
  REQUIRE(ch.size_bytes() == 600);

  auto h0 = ch.try_pop();
  REQUIRE(h0.has_value());
  REQUIRE(h0->batch_id == 10);
  REQUIRE(ch.size_bytes() == 500);

  auto h1 = ch.try_pop();
  REQUIRE(h1.has_value());
  REQUIRE(h1->batch_id == 20);
  REQUIRE(ch.size_bytes() == 300);

  auto h2 = ch.try_pop();
  REQUIRE(h2.has_value());
  REQUIRE(h2->batch_id == 30);
  REQUIRE(ch.size_bytes() == 0);
  REQUIRE(ch.empty());
}

// ============================================================================
// CH-6: blocking push unblocks when consumer pops
// ============================================================================

TEST_CASE("exchange_channel: blocking push unblocks on pop", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 1});

  REQUIRE(ch.try_push(make_handle(0)));  // fills the channel
  REQUIRE(ch.full());

  std::atomic<bool> unblocked{false};
  std::thread producer([&] {
    bool ok = ch.push(make_handle(1));  // blocks until consumer pops
    REQUIRE(ok);
    unblocked.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(10ms);
  REQUIRE_FALSE(unblocked.load());

  // Consumer pops — producer should unblock.
  auto h = ch.pop();
  REQUIRE(h.has_value());

  producer.join();
  REQUIRE(unblocked.load());
}

// ============================================================================
// CH-7: close-then-drain semantics
// ============================================================================

TEST_CASE("exchange_channel: close-then-drain", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  REQUIRE(ch.try_push(make_handle(1)));
  REQUIRE(ch.try_push(make_handle(2)));

  ch.close();
  REQUIRE(ch.closed());
  REQUIRE_FALSE(ch.drained());  // items still queued

  // Pushes after close are rejected.
  REQUIRE_FALSE(ch.try_push(make_handle(3)));
  REQUIRE_FALSE(ch.push(make_handle(4)));

  // Queued items still pop in FIFO order.
  auto h1 = ch.pop();
  REQUIRE(h1.has_value());
  REQUIRE(h1->batch_id == 1);
  REQUIRE_FALSE(ch.drained());

  auto h2 = ch.pop();
  REQUIRE(h2.has_value());
  REQUIRE(h2->batch_id == 2);

  // Now drained.
  REQUIRE(ch.empty());
  REQUIRE(ch.drained());

  // pop on a drained channel returns nullopt immediately.
  auto h3 = ch.pop();
  REQUIRE_FALSE(h3.has_value());
}

// ============================================================================
// CH-8: blocked pop wakes on close
// ============================================================================

TEST_CASE("exchange_channel: blocked pop wakes on close", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  std::atomic<bool> returned_nullopt{false};
  std::thread consumer([&] {
    auto h = ch.pop();  // blocks on empty open channel
    REQUIRE_FALSE(h.has_value());
    returned_nullopt.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(10ms);
  REQUIRE_FALSE(returned_nullopt.load());

  ch.close();
  consumer.join();
  REQUIRE(returned_nullopt.load());
}

// ============================================================================
// CH-9: close is idempotent
// ============================================================================

TEST_CASE("exchange_channel: close is idempotent", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  ch.close();
  ch.close();  // second call must not throw or deadlock
  REQUIRE(ch.closed());
  REQUIRE_FALSE(ch.try_push(make_handle(0)));
  REQUIRE_FALSE(ch.push(make_handle(0)));
}

// ============================================================================
// CH-10: MPMC stress — each handle delivered exactly once
// ============================================================================

TEST_CASE("exchange_channel: MPMC stress — each handle delivered exactly once",
          "[exchange_channel]")
{
  constexpr int N_PRODUCERS          = 4;
  constexpr int N_CONSUMERS          = 3;
  constexpr int HANDLES_PER_PRODUCER = 200;
  constexpr int TOTAL                = N_PRODUCERS * HANDLES_PER_PRODUCER;

  exchange_channel ch(exchange_channel::config{.capacity_items = 16});

  std::atomic<int> consumed{0};
  std::set<uint64_t> seen;
  std::mutex seen_mutex;

  // Consumers: drain until they have seen TOTAL items.
  std::vector<std::thread> consumers;
  for (int c = 0; c < N_CONSUMERS; ++c) {
    consumers.emplace_back([&] {
      while (true) {
        auto h = ch.pop();
        if (!h.has_value()) break;  // drained
        {
          std::lock_guard<std::mutex> lk(seen_mutex);
          seen.insert(h->batch_id);
        }
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Producers: push handles, then close once last producer is done.
  std::vector<std::thread> producers;
  std::atomic<int> done_producers{0};
  for (int p = 0; p < N_PRODUCERS; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < HANDLES_PER_PRODUCER; ++i) {
        uint64_t id = static_cast<uint64_t>(p * HANDLES_PER_PRODUCER + i);
        while (!ch.push(make_handle(id))) {
          // push returns false only if closed — shouldn't happen here
          break;
        }
      }
      if (done_producers.fetch_add(1, std::memory_order_acq_rel) + 1 == N_PRODUCERS) { ch.close(); }
    });
  }

  for (auto& t : producers)
    t.join();
  for (auto& t : consumers)
    t.join();

  REQUIRE(consumed.load() == TOTAL);
  REQUIRE(static_cast<int>(seen.size()) == TOTAL);
}

// ============================================================================
// CH-11: hooks fire once per op and are called outside the lock
// ============================================================================

TEST_CASE("exchange_channel: hooks fire outside the lock", "[exchange_channel]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  std::atomic<int> push_count{0};
  std::atomic<int> pop_count{0};
  std::atomic<std::size_t> size_in_push_cb{0};
  std::atomic<std::size_t> size_in_pop_cb{0};

  ch.set_on_push([&] {
    push_count.fetch_add(1, std::memory_order_relaxed);
    // size() acquires the mutex; if the callback were fired inside the lock this would deadlock.
    size_in_push_cb.store(ch.size(), std::memory_order_relaxed);
  });
  ch.set_on_pop([&] {
    pop_count.fetch_add(1, std::memory_order_relaxed);
    size_in_pop_cb.store(ch.size(), std::memory_order_relaxed);
  });

  REQUIRE(ch.try_push(make_handle(1)));
  REQUIRE(push_count.load() == 1);
  REQUIRE(size_in_push_cb.load() == 1);

  REQUIRE(ch.try_push(make_handle(2)));
  REQUIRE(push_count.load() == 2);

  auto h = ch.try_pop();
  REQUIRE(h.has_value());
  REQUIRE(pop_count.load() == 1);
  REQUIRE(size_in_pop_cb.load() == 1);
}
