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

#include "catch.hpp"
#include "exec/completion_controller.hpp"
#include "io/cache/types.hpp"
#include "io/types.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

using sirius::io::cache::cached_chunk;
using sirius::io::cache::chunk_fill;
using sirius::io::cache::chunk_state;
using sirius::io::cache::covers;
using sirius::io::cache::fill_span;
using sirius::io::cache::merge;
using sirius::io::cache::needed_fill;

namespace {

constexpr std::size_t PAGE  = sirius::io::IO_BLOCK_SIZE;
constexpr std::size_t CHUNK = 16 * PAGE;  // 64 KiB — the layout maths is size-parametric
constexpr std::size_t OFF   = 4 * CHUNK;  // an arbitrary chunk-aligned position

/// Drive a fresh chunk to `allocated`, the state a loader can claim.
void make_allocated(chunk_state& s)
{
  REQUIRE(s.mark_queued());
  REQUIRE(s.mark_allocated());
}

/// Drive a fresh chunk to `cached` holding the extent @p f.
void make_cached(chunk_state& s, chunk_fill f)
{
  make_allocated(s);
  chunk_fill got{};
  REQUIRE(s.take_loading_merging(f, got));
  REQUIRE(got == f);
  REQUIRE(s.mark_cached());
}

}  // namespace

// ===========================================================================
// chunk_fill — extent arithmetic
// ===========================================================================

TEST_CASE("an unset extent covers nothing and is distinct from a full one", "[cache][fill]")
{
  auto const unset = chunk_fill::unset();
  auto const whole = chunk_fill::whole();

  CHECK(unset.is_unset());
  CHECK_FALSE(whole.is_unset());
  CHECK(unset != whole);

  // The distinction is the whole point: a freshly created chunk must not read
  // as "already populated", or a merge into it would drop the desired extent.
  CHECK_FALSE(covers(unset, OFF, CHUNK, OFF, OFF + PAGE));
  CHECK(covers(whole, OFF, CHUNK, OFF, OFF + CHUNK));
}

TEST_CASE("needed_fill anchors an extent to the nearer chunk edge", "[cache][fill]")
{
  SECTION("a request spanning the whole chunk is full")
  {
    CHECK(needed_fill(OFF, CHUNK, OFF, OFF + CHUNK) == chunk_fill::whole());
    CHECK(needed_fill(OFF, CHUNK, 0, OFF + 10 * CHUNK) == chunk_fill::whole());
  }

  SECTION("a request clipping the head is a prefix")
  {
    auto const f = needed_fill(OFF, CHUNK, OFF, OFF + 2 * PAGE);
    CHECK(f == chunk_fill::prefix_of(2));
  }

  SECTION("a request clipping the tail is a suffix")
  {
    auto const f = needed_fill(OFF, CHUNK, OFF + 14 * PAGE, OFF + CHUNK);
    CHECK(f == chunk_fill::suffix_of(2));
  }

  SECTION("an interior request nearer the tail becomes a suffix, never the whole chunk")
  {
    auto const f = needed_fill(OFF, CHUNK, OFF + 8 * PAGE, OFF + 9 * PAGE);
    CHECK(f == chunk_fill::suffix_of(8));  // 8 pages to the tail vs 9 to the head
    CHECK_FALSE(f.full);
  }

  SECTION("an interior request nearer the head becomes a prefix")
  {
    // The case anchoring unconditionally to the tail gets badly wrong: one page
    // sitting one page in costs 2 pages as a prefix and CHUNK-1 as a suffix.
    auto const f = needed_fill(OFF, CHUNK, OFF + PAGE, OFF + 2 * PAGE);
    CHECK(f == chunk_fill::prefix_of(2));
    CHECK_FALSE(f.full);
  }

  SECTION("an interior request is covered by whichever edge it is anchored to")
  {
    // Whatever shape is chosen must contain the bytes that motivated it.
    for (std::size_t start = 0; start < CHUNK / PAGE; ++start) {
      auto const lo = OFF + start * PAGE;
      auto const hi = lo + PAGE;
      auto const f  = needed_fill(OFF, CHUNK, lo, hi);
      CHECK(covers(f, OFF, CHUNK, lo, hi));
    }
  }

  SECTION("a sub-page request is rounded out to whole pages")
  {
    auto const f = needed_fill(OFF, CHUNK, OFF, OFF + 10);
    CHECK(f == chunk_fill::prefix_of(1));
  }

  SECTION("a request that does not overlap the chunk is unset")
  {
    CHECK(needed_fill(OFF, CHUNK, OFF + CHUNK, OFF + 2 * CHUNK).is_unset());
  }
}

TEST_CASE("merging extents never narrows coverage", "[cache][fill]")
{
  auto const unset = chunk_fill::unset();
  auto const whole = chunk_fill::whole();

  CHECK(merge(unset, chunk_fill::prefix_of(3)) == chunk_fill::prefix_of(3));
  CHECK(merge(chunk_fill::prefix_of(3), unset) == chunk_fill::prefix_of(3));
  CHECK(merge(whole, chunk_fill::prefix_of(3)) == whole);
  CHECK(merge(chunk_fill::prefix_of(3), whole) == whole);

  // Same edge: the wider one wins, in either argument order.
  CHECK(merge(chunk_fill::prefix_of(3), chunk_fill::prefix_of(7)) == chunk_fill::prefix_of(7));
  CHECK(merge(chunk_fill::prefix_of(7), chunk_fill::prefix_of(3)) == chunk_fill::prefix_of(7));

  // Opposite edges together span the chunk, so the fold is full.  Conservative
  // — it over-reads — but it can never advertise a hole in the middle.
  CHECK(merge(chunk_fill::prefix_of(1), chunk_fill::suffix_of(1)) == whole);
}

TEST_CASE("the bytes read always contain the bytes claimed", "[cache][fill]")
{
  // The load-bearing safety property of partial fills: whatever fill_span tells
  // a loader to read must be a superset of everything covers() will later claim
  // is present.  Anything less is silent data corruption, so this is checked as
  // a property over random requests rather than a handful of examples.
  std::mt19937 rng(20260809);
  std::uniform_int_distribution<std::size_t> pos(0, 3 * CHUNK);

  for (int i = 0; i < 20000; ++i) {
    auto lo = OFF - CHUNK + pos(rng);
    auto hi = lo + 1 + pos(rng);

    auto const clamped_lo = std::max(lo, OFF);
    auto const clamped_hi = std::min(hi, OFF + CHUNK);
    if (clamped_lo >= clamped_hi) { continue; }

    auto const f          = needed_fill(OFF, CHUNK, lo, hi);
    auto const [slo, shi] = fill_span(f, OFF, CHUNK);

    // The request itself must be covered, and must lie inside what we read.
    REQUIRE(covers(f, OFF, CHUNK, clamped_lo, clamped_hi));
    REQUIRE(slo <= clamped_lo);
    REQUIRE(shi >= clamped_hi);

    // Reads stay page-aligned so they remain O_DIRECT-compatible.
    REQUIRE(slo % PAGE == 0);
    REQUIRE(shi % PAGE == 0);
    REQUIRE(slo >= OFF);
    REQUIRE(shi <= OFF + CHUNK);

    // The general form: ANY sub-range the extent claims must have been read.
    auto const a = OFF + (pos(rng) % CHUNK);
    auto const b = a + 1 + (pos(rng) % CHUNK);
    if (b <= OFF + CHUNK && covers(f, OFF, CHUNK, a, b)) {
      REQUIRE(slo <= a);
      REQUIRE(shi >= b);
    }
  }
}

// ===========================================================================
// chunk_state — the packed word
// ===========================================================================

TEST_CASE("chunk_state packs into one word and a chunk into a quarter line", "[cache][state]")
{
  CHECK(sizeof(chunk_state) == 8);
  CHECK(sizeof(cached_chunk) == 32);
}

TEST_CASE("a fresh chunk starts empty, unset and unsubscribed", "[cache][state]")
{
  chunk_state s;
  auto const snap = s.load();
  CHECK(snap.state() == chunk_state::empty);
  CHECK(snap.pins() == 0);
  CHECK(snap.subscribers() == 0);
  CHECK(snap.fill().is_unset());
  CHECK_FALSE(snap.is_reclaimable());
}

TEST_CASE("the chunk lifecycle advances only along legal edges", "[cache][state]")
{
  chunk_state s;

  CHECK_FALSE(s.mark_allocated());  // queued is the only way out of empty
  CHECK_FALSE(s.mark_cached());
  CHECK_FALSE(s.mark_evicting());

  REQUIRE(s.mark_queued());
  CHECK_FALSE(s.mark_queued());  // not idempotent: a second claimant must fail
  CHECK_FALSE(s.mark_evicting());

  REQUIRE(s.mark_allocated());
  CHECK(s.load().is_reclaimable());  // an allocated chunk holds a buffer

  chunk_fill got{};
  REQUIRE(s.take_loading(got));
  CHECK(got.is_unset());
  CHECK_FALSE(s.take_loading(got));  // a second loader must not claim it
  CHECK_FALSE(s.load().is_reclaimable());

  REQUIRE(s.mark_cached());
  CHECK(s.get_state() == chunk_state::cached);

  REQUIRE(s.mark_evicting());
  REQUIRE(s.mark_empty());
  CHECK(s.get_state() == chunk_state::empty);
}

TEST_CASE("a failed load reverts to allocated so the buffer can be retried", "[cache][state]")
{
  chunk_state s;
  make_allocated(s);

  chunk_fill got{};
  REQUIRE(s.take_loading_merging(chunk_fill::prefix_of(4), got));
  REQUIRE(s.mark_load_failed());

  CHECK(s.get_state() == chunk_state::allocated);
  // The extent survives; `allocated` is not readable, and the next loader
  // re-derives its span from whatever the extent then holds.
  CHECK(s.get_fill() == chunk_fill::prefix_of(4));
  CHECK(s.take_loading(got));
  CHECK(got == chunk_fill::prefix_of(4));
}

TEST_CASE("pins keep a chunk readable and block eviction", "[cache][state]")
{
  chunk_state s;
  make_cached(s, chunk_fill::whole());

  REQUIRE(s.acquire_read());
  CHECK(s.get_state() == chunk_state::in_use);
  CHECK(s.get_pin_count() == 1);
  CHECK_FALSE(s.mark_evicting());  // pinned

  REQUIRE(s.acquire_read());
  CHECK(s.get_pin_count() == 2);

  CHECK_FALSE(s.release_read());  // not the last reader
  CHECK(s.get_state() == chunk_state::in_use);
  CHECK(s.release_read());  // last one out
  CHECK(s.get_state() == chunk_state::cached);
  CHECK(s.mark_evicting());
}

TEST_CASE("try_pin_covering refuses a chunk that is not populated far enough", "[cache][state]")
{
  chunk_state s;
  make_cached(s, chunk_fill::prefix_of(4));  // only [OFF, OFF + 4 pages) is real

  SECTION("inside the populated prefix: hit")
  {
    REQUIRE(s.try_pin_covering(OFF, CHUNK, OFF, OFF + 4 * PAGE));
    CHECK(s.get_pin_count() == 1);
    CHECK(s.release_read());
  }

  SECTION("one byte past the prefix: miss, and no pin is left behind")
  {
    CHECK_FALSE(s.try_pin_covering(OFF, CHUNK, OFF, OFF + 4 * PAGE + 1));
    // The miss path must not perturb the chunk at all — that is what makes it
    // one relaxed load rather than a pin/unpin pair.
    auto const snap = s.load();
    CHECK(snap.state() == chunk_state::cached);
    CHECK(snap.pins() == 0);
  }

  SECTION("wholly outside the populated prefix: miss")
  {
    CHECK_FALSE(s.try_pin_covering(OFF, CHUNK, OFF + 8 * PAGE, OFF + 9 * PAGE));
  }

  SECTION("a chunk that is not resident is a miss regardless of extent")
  {
    chunk_state fresh;
    CHECK_FALSE(fresh.try_pin_covering(OFF, CHUNK, OFF, OFF + PAGE));
  }
}

TEST_CASE("merge_fill widens only a chunk that has not been loaded", "[cache][state]")
{
  SECTION("empty, queued and allocated all widen")
  {
    for (int stop = 0; stop < 3; ++stop) {
      chunk_state s;
      if (stop >= 1) { REQUIRE(s.mark_queued()); }
      if (stop >= 2) { REQUIRE(s.mark_allocated()); }
      CHECK(s.merge_fill(chunk_fill::prefix_of(2)));
      CHECK(s.merge_fill(chunk_fill::prefix_of(5)));
      CHECK(s.get_fill() == chunk_fill::prefix_of(5));
    }
  }

  SECTION("a loading chunk belongs to its loader")
  {
    chunk_state s;
    make_allocated(s);
    chunk_fill got{};
    REQUIRE(s.take_loading_merging(chunk_fill::prefix_of(2), got));
    CHECK_FALSE(s.merge_fill(chunk_fill::whole()));
    CHECK(s.get_fill() == chunk_fill::prefix_of(2));
  }

  SECTION("a cached chunk is never widened — that would advertise unwritten bytes")
  {
    chunk_state s;
    make_cached(s, chunk_fill::prefix_of(2));
    CHECK_FALSE(s.merge_fill(chunk_fill::whole()));
    CHECK(s.get_fill() == chunk_fill::prefix_of(2));
    // …and a reader needing more correctly misses instead of reading garbage.
    CHECK_FALSE(s.try_pin_covering(OFF, CHUNK, OFF, OFF + 3 * PAGE));
  }
}

TEST_CASE("claiming a chunk hands back the merged extent, not just the claimant's",
          "[cache][state]")
{
  // A prefetch queues a chunk for a wide fill; a demand read then claims it for
  // a narrow one.  The loader must be told to fill the WIDE extent, or the
  // prefetch's own consumer would later be told bytes are present that the
  // narrow read never wrote.
  chunk_state s;
  make_allocated(s);
  REQUIRE(s.merge_fill(chunk_fill::prefix_of(6)));

  chunk_fill got{};
  REQUIRE(s.take_loading_merging(chunk_fill::prefix_of(2), got));
  CHECK(got == chunk_fill::prefix_of(6));

  REQUIRE(s.mark_cached());
  CHECK(s.try_pin_covering(OFF, CHUNK, OFF, OFF + 6 * PAGE));
}

// ===========================================================================
// subscribers
// ===========================================================================

TEST_CASE("subscribers gate eviction and survive the whole lifecycle", "[cache][state]")
{
  chunk_state s;
  s.add_subscriber();
  s.add_subscriber();
  CHECK(s.get_subscribers() == 2);

  make_cached(s, chunk_fill::whole());
  CHECK(s.get_subscribers() == 2);  // untouched by every transition

  CHECK_FALSE(s.mark_evicting());  // two live requests still name it
  s.drop_subscriber();
  CHECK_FALSE(s.mark_evicting());  // one still does
  s.drop_subscriber();
  CHECK(s.mark_evicting());
}

TEST_CASE("the last-resort pass may evict a still-subscribed chunk but never a pinned one",
          "[cache][state]")
{
  chunk_state s;
  s.add_subscriber();
  make_cached(s, chunk_fill::whole());

  CHECK_FALSE(s.mark_evicting(/*only_unsubscribed=*/true));
  CHECK(s.mark_evicting(/*only_unsubscribed=*/false));
  REQUIRE(s.mark_empty());

  chunk_state pinned;
  pinned.add_subscriber();
  make_cached(pinned, chunk_fill::whole());
  REQUIRE(pinned.acquire_read());
  CHECK_FALSE(pinned.mark_evicting(/*only_unsubscribed=*/false));  // a live read always wins
}

TEST_CASE("mark_empty clears the extent but keeps the subscriber count", "[cache][state]")
{
  chunk_state s;
  make_cached(s, chunk_fill::prefix_of(3));
  REQUIRE(s.mark_evicting());

  // A request may have named the chunk again while it was in transit; it still
  // has to be able to hand that reference back.
  s.add_subscriber();
  REQUIRE(s.mark_empty());

  CHECK(s.get_fill().is_unset());  // NOT `whole` — a reclaimed chunk holds nothing
  CHECK(s.get_subscribers() == 1);
  CHECK_FALSE(s.try_pin_covering(OFF, CHUNK, OFF, OFF + PAGE));
}

TEST_CASE("the subscriber count clamps instead of borrowing into the state", "[cache][state]")
{
  chunk_state s;
  REQUIRE(s.mark_queued());

  s.drop_subscriber();  // underflow would borrow through the neighbouring fields
  s.drop_subscriber();

  CHECK(s.get_subscribers() == 0);
  CHECK(s.get_state() == chunk_state::queued);
  CHECK(s.mark_allocated());
}

// ===========================================================================
// concurrency
// ===========================================================================

TEST_CASE("a reader is never told bytes are present that the loader did not write",
          "[cache][state][concurrency]")
{
  // One loader fills only the extent it was handed and publishes; readers pin
  // whatever they can and verify every byte the chunk claims was actually
  // written.  Poison marks the bytes the loader must not be allowed to claim.
  constexpr std::uint8_t POISON  = 0xEE;
  constexpr std::uint8_t WRITTEN = 0x5A;

  std::atomic<bool> corrupted{false};
  std::atomic<long> concurrent_hits{0};

  for (int round = 0; round < 64; ++round) {
    auto const filled_pages = 1 + (round % 15);
    std::vector<std::uint8_t> buffer(CHUNK, POISON);
    chunk_state s;
    make_allocated(s);
    REQUIRE(s.merge_fill(needed_fill(OFF, CHUNK, OFF, OFF + filled_pages * PAGE)));

    std::atomic<bool> published{false};

    std::thread loader([&] {
      chunk_fill got{};
      if (!s.take_loading(got)) { return; }
      auto const [lo, hi] = fill_span(got, OFF, CHUNK);
      std::fill(buffer.begin() + static_cast<std::ptrdiff_t>(lo - OFF),
                buffer.begin() + static_cast<std::ptrdiff_t>(hi - OFF),
                WRITTEN);
      std::ignore = s.mark_cached();
      published.store(true);
    });

    // Readers race the loader.  Whether any individual attempt lands is timing —
    // what must hold is that every attempt that DOES land sees written bytes.
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
      readers.emplace_back([&, r] {
        int attempt = 0;
        while (!published.load() || attempt < 32) {
          auto const lo = OFF + static_cast<std::size_t>((r + attempt) % 16) * PAGE;
          auto const hi = std::min(lo + PAGE, OFF + CHUNK);
          ++attempt;
          if (!s.try_pin_covering(OFF, CHUNK, lo, hi)) { continue; }
          for (auto b = lo; b < hi; ++b) {
            if (buffer[b - OFF] != WRITTEN) { corrupted.store(true); }
          }
          concurrent_hits.fetch_add(1);
          std::ignore = s.release_read();
        }
      });
    }

    loader.join();
    for (auto& t : readers) {
      t.join();
    }

    REQUIRE_FALSE(corrupted.load());

    // Deterministic liveness: once published, the filled extent must be
    // readable and the rest must still be refused.
    REQUIRE(s.try_pin_covering(OFF, CHUNK, OFF, OFF + filled_pages * PAGE));
    for (std::size_t b = 0; b < filled_pages * PAGE; ++b) {
      REQUIRE(buffer[b] == WRITTEN);
    }
    std::ignore = s.release_read();

    if (filled_pages < 16) {
      REQUIRE_FALSE(s.try_pin_covering(OFF, CHUNK, OFF, OFF + (filled_pages + 1) * PAGE));
    }
  }

  // The concurrent phase is only meaningful if reads actually got through it at
  // least sometimes across the 64 rounds.
  CHECK(concurrent_hits.load() > 0);
}

TEST_CASE("concurrent pins and subscriber churn leave a consistent word",
          "[cache][state][concurrency]")
{
  chunk_state s;
  make_cached(s, chunk_fill::whole());

  constexpr int THREADS    = 8;
  constexpr int ITERATIONS = 2000;

  std::vector<std::thread> workers;
  for (int t = 0; t < THREADS; ++t) {
    workers.emplace_back([&, t] {
      for (int i = 0; i < ITERATIONS; ++i) {
        if (t % 2 == 0) {
          if (s.acquire_read()) { std::ignore = s.release_read(); }
        } else {
          s.add_subscriber();
          s.drop_subscriber();
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  auto const snap = s.load();
  CHECK(snap.state() == chunk_state::cached);
  CHECK(snap.pins() == 0);
  CHECK(snap.subscribers() == 0);
  CHECK(snap.fill() == chunk_fill::whole());
}

TEST_CASE("prepared cache completion publishes only its exact physical segment",
          "[cache][completion]")
{
  cached_chunk succeeded;
  cached_chunk failed;
  cached_chunk pending;

  auto make_loading = [](cached_chunk& chunk) {
    REQUIRE(chunk.state.mark_queued());
    REQUIRE(chunk.state.mark_allocated());
    chunk_fill fill;
    REQUIRE(chunk.state.take_loading_merging(chunk_fill::whole(), fill));
  };
  make_loading(succeeded);
  make_loading(failed);
  make_loading(pending);

  auto completion = std::make_shared<sirius::io::prepared_io_completion>(
    [](std::span<cached_chunk* const> completed, bool host_ok) noexcept {
      for (auto* chunk : completed) {
        std::ignore = host_ok ? chunk->state.mark_cached() : chunk->state.mark_load_failed();
      }
    });

  std::array<cached_chunk*, 1> first{&succeeded};
  (*completion)(first, true);
  CHECK(succeeded.state.get_state() == chunk_state::cached);
  CHECK(failed.state.get_state() == chunk_state::loading);
  CHECK(pending.state.get_state() == chunk_state::loading);

  std::array<cached_chunk*, 1> second{&failed};
  (*completion)(second, false);
  CHECK(succeeded.state.get_state() == chunk_state::cached);
  CHECK(failed.state.get_state() == chunk_state::allocated);
  CHECK(pending.state.get_state() == chunk_state::loading);

  // A device-copy failure reports host_ok=true: the host cache source remains
  // valid and is published even though the request coordinator reports error.
  std::array<cached_chunk*, 1> third{&pending};
  (*completion)(third, true);
  CHECK(pending.state.get_state() == chunk_state::cached);
}

TEST_CASE("cache callback lifetime holds teardown admission through callback ownership",
          "[cache][completion][lifetime]")
{
  sirius::exec::completion_controller inflight;
  std::atomic<bool> drained{false};
  std::atomic<bool> invoked{false};
  auto subscription = inflight.on_completion([&drained] { drained.store(true); });
  auto lifetime = std::make_shared<sirius::exec::completion_controller::slot>(inflight.acquire());

  auto completion = std::make_shared<sirius::io::prepared_io_completion>(
    [lifetime = std::move(lifetime), &invoked](std::span<cached_chunk* const>, bool) noexcept {
      std::ignore = lifetime;
      invoked.store(true);
    });

  inflight.close();
  CHECK_FALSE(drained.load());

  (*completion)({}, true);
  CHECK(invoked.load());
  CHECK_FALSE(drained.load());

  // The completion may be shared by several physical operations. Teardown is
  // released only when the final callback owner disappears, not after the
  // first invocation returns.
  completion.reset();
  CHECK(drained.load());
}
