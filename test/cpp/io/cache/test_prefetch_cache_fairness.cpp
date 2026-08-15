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

// Cross-query fairness of the prefetching cache (concurrency issues F3/F4).
// Everything here is CPU-side: no GPU, no IO context, no buffer pool — these
// are the cache's scoring and queueing components exercised directly.
//
// F3 (epochs): scoring used to be newest-wins on one global ticker, so a
// second query's prepare_for_query demoted every chunk the first query had
// prefetched-but-not-yet-read to eviction tier 0, and a second query touching
// a shared chunk reset the first query's demand counters.  The fix scores
// against the OLDEST live epoch (query_epoch_tracker::min_live_epoch) and
// unions the counters while any live query wants the chunk.
//
// F4 (queues): the cache's preparation/prefetch/evictor queues were strict
// process-wide FIFOs, so query A's flood of requests fully blocked query B's
// first prefetch.  The fix keeps FIFO within a query's band and round-robins
// across bands at pop (fair_band_queue).

#include "catch.hpp"
#include "io/cache/fair_band_queue.hpp"
#include "io/cache/query_epoch_tracker.hpp"
#include "io/cache/types.hpp"
#include "query_id.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using sirius::make_query_id;
using sirius::io::cache::chunk_lifecycle;
using sirius::io::cache::fair_band_queue;
using sirius::io::cache::query_epoch_tracker;

// ---------------------------------------------------------------------------
// query_epoch_tracker
// ---------------------------------------------------------------------------

TEST_CASE("query_epoch_tracker mints monotonic epochs and tracks the oldest live one",
          "[cache][prefetching_cache][fairness]")
{
  query_epoch_tracker tracker;
  REQUIRE(tracker.newest_epoch() == 0);
  REQUIRE(tracker.min_live_epoch() == 0);
  REQUIRE(tracker.live_count() == 0);

  auto const a = tracker.begin_query(make_query_id(11));
  REQUIRE(a == 1);
  REQUIRE(tracker.newest_epoch() == 1);
  REQUIRE(tracker.min_live_epoch() == 1);

  auto const b = tracker.begin_query(make_query_id(22));
  REQUIRE(b == 2);
  REQUIRE(tracker.newest_epoch() == 2);
  // A is still live, so the staleness bar must NOT advance past its epoch.
  REQUIRE(tracker.min_live_epoch() == 1);
  REQUIRE(tracker.live_count() == 2);

  REQUIRE(tracker.epoch_of(make_query_id(11)) == 1);
  REQUIRE(tracker.epoch_of(make_query_id(22)) == 2);
  // Unknown queries fall back to the newest epoch (the old ticker stamp).
  REQUIRE(tracker.epoch_of(make_query_id(99)) == 2);

  tracker.end_query(make_query_id(11));
  REQUIRE(tracker.min_live_epoch() == 2);
  REQUIRE(tracker.live_count() == 1);

  // Retiring the last query parks the bar at the newest epoch — the last
  // query's chunks keep their demand tier between queries, exactly the old
  // single-ticker between-queries behavior.
  tracker.end_query(make_query_id(22));
  REQUIRE(tracker.live_count() == 0);
  REQUIRE(tracker.min_live_epoch() == tracker.newest_epoch());

  // end_query is idempotent; unknown ids are ignored.
  tracker.end_query(make_query_id(22));
  tracker.end_query(make_query_id(12345));
  REQUIRE(tracker.min_live_epoch() == 2);
}

TEST_CASE("query_epoch_tracker re-registration of a stale id re-stamps the epoch",
          "[cache][prefetching_cache][fairness]")
{
  query_epoch_tracker tracker;
  REQUIRE(tracker.begin_query(make_query_id(7)) == 1);
  // A window whose cleanup never ran registers again: fresh epoch, one entry.
  REQUIRE(tracker.begin_query(make_query_id(7)) == 2);
  REQUIRE(tracker.live_count() == 1);
  REQUIRE(tracker.min_live_epoch() == 2);
}

// ---------------------------------------------------------------------------
// F3 (a): a newer query's prepare must not demote a live peer's unread chunks
// ---------------------------------------------------------------------------

TEST_CASE("a second query's prepare does not demote a live peer's unread chunks",
          "[cache][prefetching_cache][fairness]")
{
  query_epoch_tracker tracker;

  // Query A prefetches a chunk but has not read it yet.
  auto const epoch_a = tracker.begin_query(make_query_id(1));
  chunk_lifecycle chunk;
  chunk.on_request(epoch_a, tracker.min_live_epoch());
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 1);

  // Query B prepares.  Pre-fix, the evictor scored against the newest ticker,
  // which demoted A's chunk to tier 0 (evict first) the moment B arrived:
  auto const epoch_b = tracker.begin_query(make_query_id(2));
  REQUIRE(epoch_b > epoch_a);
  REQUIRE(chunk.load().eviction_tier(tracker.newest_epoch()) == 0);  // the old, broken score

  // Post-fix the evictor scores against the oldest LIVE epoch: A's unread
  // chunk keeps its protected tier for as long as A is live.
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 1);

  // A consumes the chunk: demand satisfied, tier drops to 0.
  chunk.on_consume();
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 0);

  tracker.end_query(make_query_id(1));
  tracker.end_query(make_query_id(2));
}

TEST_CASE("a query's chunks become evictable once its epoch is retired",
          "[cache][prefetching_cache][fairness]")
{
  query_epoch_tracker tracker;

  auto const epoch_a = tracker.begin_query(make_query_id(1));
  chunk_lifecycle chunk;
  chunk.on_request(epoch_a, tracker.min_live_epoch());

  tracker.begin_query(make_query_id(2));
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 1);  // protected while A lives

  // A finishes without consuming (e.g. the scan was cancelled): its unread
  // demand must NOT pin the chunk forever — the bar advances to B's epoch and
  // the chunk falls to tier 0.
  tracker.end_query(make_query_id(1));
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 0);

  tracker.end_query(make_query_id(2));
}

// ---------------------------------------------------------------------------
// F3 (b): shared-chunk counters are a union across live queries
// ---------------------------------------------------------------------------

TEST_CASE("a shared chunk keeps the first query's accounting when a second query requests it",
          "[cache][prefetching_cache][fairness]")
{
  query_epoch_tracker tracker;

  // A requests the chunk twice and consumes once: one outstanding read.
  auto const epoch_a = tracker.begin_query(make_query_id(1));
  chunk_lifecycle chunk;
  chunk.on_request(epoch_a, tracker.min_live_epoch());
  chunk.on_request(epoch_a, tracker.min_live_epoch());
  chunk.on_consume();
  {
    auto const s = chunk.load();
    REQUIRE(s.inserts == 2);
    REQUIRE(s.reads == 1);
  }

  // B (a newer live query) requests the SAME chunk.  Pre-fix, on_request
  // reset the counters to (inserts=1, reads=0) because B's tick was newer —
  // erasing A's outstanding demand.  Post-fix the request accumulates.
  auto const epoch_b = tracker.begin_query(make_query_id(2));
  chunk.on_request(epoch_b, tracker.min_live_epoch());
  {
    auto const s = chunk.load();
    REQUIRE(s.inserts == 3);  // union: A's two + B's one
    REQUIRE(s.reads == 1);    // A's consume preserved
    REQUIRE(s.tick == epoch_b);
  }

  // Outstanding demand (2) caps the tier while either query lives.
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 2);

  // A retires; the chunk was re-stamped with B's epoch, so B's demand still
  // protects it.
  tracker.end_query(make_query_id(1));
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 2);

  // Both remaining reads happen; the tier decays to 0.
  chunk.on_consume();
  chunk.on_consume();
  REQUIRE(chunk.load().eviction_tier(tracker.min_live_epoch()) == 0);

  tracker.end_query(make_query_id(2));
}

TEST_CASE("counters from retired epochs are reset by the next request",
          "[cache][prefetching_cache][fairness]")
{
  query_epoch_tracker tracker;

  auto const epoch_a = tracker.begin_query(make_query_id(1));
  chunk_lifecycle chunk;
  chunk.on_request(epoch_a, tracker.min_live_epoch());
  chunk.on_request(epoch_a, tracker.min_live_epoch());
  tracker.end_query(make_query_id(1));

  // A is gone; C is a fresh query touching the leftover chunk.  Its stale
  // counters (tick < min live) must not leak into C's accounting.
  auto const epoch_c = tracker.begin_query(make_query_id(3));
  chunk.on_request(epoch_c, tracker.min_live_epoch());
  {
    auto const s = chunk.load();
    REQUIRE(s.tick == epoch_c);
    REQUIRE(s.inserts == 1);
    REQUIRE(s.reads == 0);
  }
  tracker.end_query(make_query_id(3));
}

// ---------------------------------------------------------------------------
// single-epoch backward compatibility: the one-argument overload is the
// historical newest-wins behavior, byte for byte
// ---------------------------------------------------------------------------

TEST_CASE("single-epoch on_request keeps the historical reset-on-newer-tick semantics",
          "[cache][prefetching_cache][fairness]")
{
  chunk_lifecycle chunk;

  chunk.on_request(1);
  chunk.on_request(1);
  chunk.on_consume();
  {
    auto const s = chunk.load();
    REQUIRE(s.tick == 1);
    REQUIRE(s.inserts == 2);
    REQUIRE(s.reads == 1);
  }
  REQUIRE(chunk.load().eviction_tier(1) == 1);
  REQUIRE(chunk.load().eviction_tier(2) == 0);  // older tick == stale

  // A newer single-epoch request resets the counters (the old behavior —
  // with one query at a time, min live == the requester's epoch).
  chunk.on_request(2);
  {
    auto const s = chunk.load();
    REQUIRE(s.tick == 2);
    REQUIRE(s.inserts == 1);
    REQUIRE(s.reads == 0);
  }
}

// ---------------------------------------------------------------------------
// F4 (c): fair_band_queue — a flood in one band cannot starve another band
// ---------------------------------------------------------------------------

TEST_CASE("fair_band_queue serves a second band within a bounded number of pops",
          "[cache][prefetching_cache][fairness]")
{
  fair_band_queue<int> queue;

  // Query A floods 1000 requests, then query B submits a single one.  Under
  // the pre-fix strict FIFO, B's request was served at pop #1001.  Under the
  // round-robin pops it must surface within 2 pops (one A batch is allowed
  // ahead of it, never the whole flood).
  constexpr int flood = 1000;
  for (int i = 0; i < flood; ++i) {
    queue.enqueue(/*band=*/1, /*item=*/i);
  }
  queue.enqueue(/*band=*/2, /*item=*/-1);

  int pops_until_b = 0;
  int item         = 0;
  do {
    REQUIRE(queue.try_dequeue(item));
    ++pops_until_b;
  } while (item != -1);
  REQUIRE(pops_until_b <= 2);

  // The flood still drains completely, in FIFO order.
  int expected = 1;  // item 0 was band 1's first pop above
  while (queue.try_dequeue(item)) {
    REQUIRE(item == expected);
    ++expected;
  }
  REQUIRE(expected == flood);
}

TEST_CASE("fair_band_queue is strict FIFO within a single band",
          "[cache][prefetching_cache][fairness]")
{
  fair_band_queue<int> queue;
  for (int i = 0; i < 100; ++i) {
    queue.enqueue(/*band=*/7, i);
  }
  for (int i = 0; i < 100; ++i) {
    int item = -1;
    REQUIRE(queue.try_dequeue(item));
    REQUIRE(item == i);  // single band == plain FIFO, single-query order unchanged
  }
  int item = -1;
  REQUIRE_FALSE(queue.try_dequeue(item));
}

TEST_CASE("fair_band_queue round-robins across live bands", "[cache][prefetching_cache][fairness]")
{
  fair_band_queue<int> queue;
  // Three bands, three items each, enqueued as one band's full flood at a
  // time (the adversarial order for a FIFO).
  for (int band = 1; band <= 3; ++band) {
    for (int i = 0; i < 3; ++i) {
      queue.enqueue(static_cast<std::uint32_t>(band), band * 10 + i);
    }
  }

  std::vector<int> order;
  int item = 0;
  while (queue.try_dequeue(item)) {
    order.push_back(item);
  }
  REQUIRE(order == std::vector<int>{10, 20, 30, 11, 21, 31, 12, 22, 32});
}

TEST_CASE("fair_band_queue sentinels on the no-band lane take part in the rotation",
          "[cache][prefetching_cache][fairness]")
{
  fair_band_queue<int> queue;
  for (int i = 0; i < 10; ++i) {
    queue.enqueue(/*band=*/5, i);
  }
  queue.enqueue(fair_band_queue<int>::no_band, -1);  // e.g. a shutdown wake-up

  // The sentinel must surface within one full rotation (2 pops here), not
  // behind the whole flood.
  int pops = 0;
  int item = 0;
  do {
    REQUIRE(queue.try_dequeue(item));
    ++pops;
  } while (item != -1);
  REQUIRE(pops <= 2);
}

TEST_CASE("fair_band_queue blocking pop wakes on a cross-thread enqueue",
          "[cache][prefetching_cache][fairness]")
{
  // Racy by nature: repeat to shake out lost-wakeup interleavings.
  for (int round = 0; round < 5; ++round) {
    fair_band_queue<int> queue;
    std::atomic<int> received{-1};

    std::thread consumer([&] {
      int item = -1;
      queue.wait_dequeue(item);
      received.store(item, std::memory_order_release);
    });

    std::thread producer([&] { queue.enqueue(/*band=*/3, 42); });

    producer.join();
    consumer.join();
    REQUIRE(received.load(std::memory_order_acquire) == 42);
  }
}

TEST_CASE("fair_band_queue keeps every item under concurrent multi-band producers",
          "[cache][prefetching_cache][fairness]")
{
  // Racy by nature: repeat to shake out interleavings.
  for (int round = 0; round < 5; ++round) {
    fair_band_queue<int> queue;
    constexpr int producers      = 4;
    constexpr int items_per_band = 250;

    std::vector<std::thread> threads;
    threads.reserve(producers);
    for (int band = 1; band <= producers; ++band) {
      threads.emplace_back([&queue, band] {
        for (int i = 0; i < items_per_band; ++i) {
          queue.enqueue(static_cast<std::uint32_t>(band), band * 1000 + i);
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }

    // Every item comes out exactly once, and each band's items in FIFO order.
    std::vector<int> next_expected(producers + 1, 0);
    int item    = 0;
    int drained = 0;
    while (queue.try_dequeue(item)) {
      int const band = item / 1000;
      REQUIRE(band >= 1);
      REQUIRE(band <= producers);
      REQUIRE(item % 1000 == next_expected[band]);
      ++next_expected[band];
      ++drained;
    }
    REQUIRE(drained == producers * items_per_band);
  }
}
