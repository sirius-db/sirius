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
#include <limits>
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
    // CHECK (not REQUIRE): a throwing assertion in a std::thread terminates the process.
    CHECK(ok);
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
    // CHECK (not REQUIRE): a throwing assertion in a std::thread terminates the process.
    CHECK_FALSE(h.has_value());
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

// ============================================================================
// CH-16: on_close fires exactly once, outside the lock, only on the
// first successful close().
// ============================================================================

TEST_CASE("exchange_channel: on_close fires exactly once outside the lock",
          "[exchange_channel][pipeline_completion]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  std::atomic<int> close_count{0};
  std::atomic<bool> was_closed_in_cb{false};
  ch.set_on_close([&] {
    close_count.fetch_add(1, std::memory_order_relaxed);
    // closed() acquires the mutex; if the callback fired inside the lock this would deadlock.
    was_closed_in_cb.store(ch.closed(), std::memory_order_relaxed);
  });

  REQUIRE(close_count.load() == 0);  // not fired before close() is ever called

  ch.close();
  REQUIRE(close_count.load() == 1);
  REQUIRE(was_closed_in_cb.load());

  // Repeated close() calls must not re-fire the callback.
  ch.close();
  ch.close();
  REQUIRE(close_count.load() == 1);
}

// ============================================================================
// CH-17: on_close fires even for a channel that closes with items
// still queued (close-then-drain — the callback signals "no more pushes will
// ever happen", not "the queue is empty").
// ============================================================================

TEST_CASE("exchange_channel: on_close fires even when items remain queued",
          "[exchange_channel][pipeline_completion]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  std::atomic<int> close_count{0};
  ch.set_on_close([&] { close_count.fetch_add(1, std::memory_order_relaxed); });

  REQUIRE(ch.try_push(make_handle(0)));
  ch.close();

  REQUIRE(close_count.load() == 1);
  REQUIRE_FALSE(ch.drained());  // item still queued — on_close still fired
}

// ============================================================================
// CH-12 (reproduction): a push that would push the cumulative total
// past capacity_bytes must be rejected. full_unlocked() only inspects bytes
// already queued, not the incoming candidate, so a 40-byte handle is wrongly
// admitted on top of 40 already-queued bytes in a 50-byte-bound channel.
// ============================================================================

TEST_CASE("exchange_channel: cumulative push crossing byte bound is rejected",
          "[exchange_channel][byte_admission]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 10, .capacity_bytes = 50});

  REQUIRE(ch.try_push(make_handle(0, 40)));
  REQUIRE_FALSE(ch.full());  // 40 < 50 — current full() correctly reports not-full here.

  // BUG: the incoming 40-byte handle would bring the total to 80 (> 50), but
  // full_unlocked() only compares the *already-queued* 40 bytes against the bound,
  // so this push is wrongly admitted today.
  REQUIRE_FALSE(ch.try_push(make_handle(1, 40)));
  REQUIRE(ch.size_bytes() == 40);
}

// ============================================================================
// CH-13 (reproduction): the oversized-batch rule is documented to
// admit an oversized handle only into an *empty* channel. A non-empty channel
// must reject an oversized handle instead of wedging past its byte bound.
// ============================================================================

TEST_CASE("exchange_channel: oversized handle rejected while queue is non-empty",
          "[exchange_channel][byte_admission]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 10, .capacity_bytes = 50});

  REQUIRE(ch.try_push(make_handle(0, 1)));  // 1 byte queued — far under the bound.

  // BUG: full_unlocked() only compares the queued total (1) against the bound (50),
  // so this 200-byte handle is wrongly admitted even though the queue is non-empty.
  REQUIRE_FALSE(ch.try_push(make_handle(1, 200)));
  REQUIRE(ch.size_bytes() == 1);
}

// ============================================================================
// CH-14 (must-not-regress): the same oversized handle IS accepted
// once the queue drains back to empty — the oversized-batch rule must keep
// working for a channel that becomes empty again, not just a freshly-built one.
// ============================================================================

TEST_CASE("exchange_channel: oversized handle accepted once queue becomes empty",
          "[exchange_channel][byte_admission]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 10, .capacity_bytes = 50});

  REQUIRE(ch.try_push(make_handle(0, 1)));
  auto h = ch.try_pop();
  REQUIRE(h.has_value());
  REQUIRE(ch.empty());

  REQUIRE(ch.try_push(make_handle(1, 200)));
  REQUIRE(ch.size_bytes() == 200);
}

// ============================================================================
// CH-15 (blocking push): a blocking push() whose candidate would
// cross the byte bound must wait rather than being admitted immediately, and
// must succeed once popping frees enough byte capacity.
// ============================================================================

TEST_CASE("exchange_channel: blocking push waits for byte headroom (blocking path)",
          "[exchange_channel][byte_admission]")
{
  exchange_channel ch(exchange_channel::config{.capacity_items = 10, .capacity_bytes = 50});

  REQUIRE(ch.try_push(make_handle(0, 40)));  // 40 queued, 10 bytes of headroom left.

  std::atomic<bool> unblocked{false};
  std::atomic<bool> push_ok{false};
  std::thread producer([&] {
    // A 40-byte handle would cross the 50-byte bound on top of the 40 already
    // queued; with a correct admission check this call must block instead of
    // returning immediately. Today's buggy full_unlocked()-only check admits it
    // right away, so this thread will observe `unblocked` flip to true before the
    // pop below ever runs.
    //
    // Assertions here use CHECK (non-throwing) rather than REQUIRE: a throwing
    // assertion on this thread would propagate as an unhandled exception across
    // the thread boundary and crash the whole test binary via std::terminate.
    bool ok = ch.push(make_handle(1, 40));
    push_ok.store(ok, std::memory_order_release);
    unblocked.store(true, std::memory_order_release);
  });

  // CHECK (not REQUIRE): if this fails, `producer` must still be joined below —
  // a REQUIRE failure here would throw and unwind past producer.join(), and a
  // still-joinable std::thread's destructor calls std::terminate().
  std::this_thread::sleep_for(10ms);
  CHECK_FALSE(unblocked.load());

  // Popping the first handle frees all 40 bytes, leaving 50 bytes of headroom —
  // enough for the pending 40-byte push to proceed.
  auto h = ch.pop();
  CHECK(h.has_value());

  producer.join();
  REQUIRE(unblocked.load());
  REQUIRE(push_ok.load());
}

// ============================================================================
// CH-18: byte accounting never overflows
// ============================================================================

TEST_CASE("exchange_channel: rejects a push that would overflow byte accounting",
          "[exchange_channel]")
{
  constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();

  // Byte-unbounded channel: this is the only configuration where the cumulative
  // total is not already capped by capacity_bytes at admission time.
  exchange_channel ch(exchange_channel::config{.capacity_items = 4});

  REQUIRE(ch.try_push(make_handle(1, max_bytes)));
  REQUIRE(ch.size_bytes() == max_bytes);

  // Admitting even one more byte would wrap _total_bytes.
  REQUIRE_FALSE(ch.try_push(make_handle(2, 1)));
  // A zero-sized handle still fits exactly.
  REQUIRE(ch.try_push(make_handle(3, 0)));
  REQUIRE(ch.size_bytes() == max_bytes);

  // Popping the oversized handle restores headroom.
  auto h = ch.try_pop();
  REQUIRE(h.has_value());
  REQUIRE(h->batch_id == 1);
  REQUIRE(ch.size_bytes() == 0);
  REQUIRE(ch.try_push(make_handle(4, 1)));
}
