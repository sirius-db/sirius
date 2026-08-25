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

#pragma once

// Shared cache entry types used by both prefetching_cache and prepared IO
// slices. Extracted here to keep the shared request contracts independent of
// the cache implementation.

#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace sirius::io::cache {

/**
 * @brief Stage of a scan's consumer-visible lifecycle.
 *
 * - @c none: the backend opted out of prefetching.
 * - @c initialized: the scan's ranges are known but no work is queued yet.
 * - @c queued: the ranges are queued with the prefetching layer.
 * - @c preparing: the IO for the ranges is being prepared.
 * - @c reading: the consumer is reading the ranges.
 * - @c disposed: the scan is cancelled or finished; its work can be dropped.
 *
 * Consumer-side only, and the order is load-bearing: callers ask questions like
 * `stage >= reading` (is the executor pulling these bytes itself?) and
 * `stage == disposed` (is this split finished?).  Read-ahead is not a stage
 * here -- it is producer-side work that runs alongside this progression, and
 * @c scan_info::prefetch_state tracks it separately.
 */
enum class scan_stage { none, initialized, queued, preparing, reading, disposed };

// ---------------------------------------------------------------------------
// Page-alignment helpers
// ---------------------------------------------------------------------------
//
// @p a must be a power of two (in practice @c io::IO_BLOCK_SIZE, the O_DIRECT
// page size).  Used to keep partial chunk fills O_DIRECT-compatible — never
// hardcode 4096 at call sites.

[[nodiscard]] constexpr std::size_t align_down(std::size_t x, std::size_t a) noexcept
{
  return x & ~(a - 1);
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t x, std::size_t a) noexcept
{
  return (x + a - 1) & ~(a - 1);
}

// ---------------------------------------------------------------------------
// buffer_pool — growable pool of pinned chunks
// ---------------------------------------------------------------------------
//
// Backed by a @c cucascade::memory::fixed_size_host_memory_resource.  Each
// grow step requests CHUNKS_PER_SLAB blocks from the upstream resource and
// appends the raw pointers to an internal free list.  Blocks are never
// returned to the upstream resource until the pool is destroyed; allocate()
// pops from the free list and deallocate() pushes back.
//
// The chunk size is taken from @c mr.get_block_size() — all cache layout
// arithmetic that needs the chunk size reads it from @c chunk_bytes().

class buffer_pool {
 public:
  /// @p initial_slabs slabs are allocated up-front from @p mr (clamped to
  /// @p max_slabs).  Default preserves the historical behaviour of warming
  /// the pool with up to 10 slabs at construction.
  buffer_pool(cucascade::memory::memory_reservation_manager& reservation_manager,
              double reservation_fraction_for_prefetching = 0.0,
              double max_prefetching_budget_fraction      = 0.0);

  ~buffer_pool();

  buffer_pool(buffer_pool const&)            = delete;
  buffer_pool& operator=(buffer_pool const&) = delete;

  /// Bulk-allocate up to @p n chunks, appending pointers to @p out.
  /// Returns the number actually allocated (may be < n if the pool is
  /// exhausted and cannot grow).
  std::vector<std::byte*> allocate_bulk(size_t n, int& numa_node);

  std::vector<std::byte*> allocate_bulk_from(size_t n, int numa_node);

  void deallocate_bulk(std::vector<std::byte*>&& out, int numa) noexcept;

  [[nodiscard]] size_t chunk_size() const noexcept { return _chunk_bytes; }

  [[nodiscard]] size_t total_allocated_bytes() const noexcept
  {
    return _n_allocated_chunks * _chunk_bytes;
  }

  [[nodiscard]] size_t total_allocated_chunks() const noexcept { return _n_allocated_chunks; }

  [[nodiscard]] size_t reservation_size_for_prefetching() const noexcept;

  [[nodiscard]] size_t max_allowed_budget_for_prefetching() const noexcept;

  [[nodiscard]] size_t max_system_wide_usage() const noexcept;

  [[nodiscard]] bool should_start_evicting() const noexcept;

 private:
  struct host_arena {
    int numa_id;
    std::unique_ptr<cucascade::memory::reservation> reservation;
    cucascade::memory::fixed_size_host_memory_resource* mr;
  };

  size_t _chunk_bytes{1};
  size_t _reserved_size{0};
  size_t _max_allowed_budget_for_prefetching{0};
  std::unordered_map<int, size_t> _numa_to_arena_index;
  std::vector<host_arena> _host_arenas;
  std::atomic<size_t> _n_allocated_chunks{0};
};

// ---------------------------------------------------------------------------
// chunk_fill — the populated extent of a chunk
// ---------------------------------------------------------------------------
//
// A chunk's staging buffer need not be filled end to end.  A read that only
// touches the head or tail of a chunk populates just that edge, so a query over
// a small file no longer pays a full chunk of IO to cache a few kilobytes.
//
// The extent is edge-anchored and measured in @c io::IO_BLOCK_SIZE pages, so
// every derived read stays O_DIRECT-compatible:
//
//   unset  -> nothing recorded yet (a fresh or freshly-reclaimed chunk)
//   full   -> the whole chunk is populated
//   prefix -> [chunk_off, chunk_off + pages * PAGE) is populated
//   suffix -> [chunk_off + chunk_bytes - pages * PAGE, chunk_off + chunk_bytes)
//
// @c unset is deliberately a distinct value from @c full: a single sentinel for
// both would make a freshly-created chunk read as "already populated", and a
// merge into it would silently drop the caller's desired extent.

struct chunk_fill {
  std::uint16_t pages{0};
  bool suffix{false};
  bool full{false};

  [[nodiscard]] static constexpr chunk_fill unset() noexcept { return {}; }
  [[nodiscard]] static constexpr chunk_fill whole() noexcept { return {0, false, true}; }
  [[nodiscard]] static constexpr chunk_fill prefix_of(std::uint16_t pages) noexcept
  {
    return {pages, false, false};
  }
  [[nodiscard]] static constexpr chunk_fill suffix_of(std::uint16_t pages) noexcept
  {
    return {pages, true, false};
  }

  [[nodiscard]] constexpr bool is_unset() const noexcept { return !full && pages == 0; }

  [[nodiscard]] friend constexpr bool operator==(chunk_fill, chunk_fill) noexcept = default;
};

/// Fold @p want into @p cur.  @c full wins over everything; two extents anchored
/// to opposite edges together span the chunk, so they also fold to @c full; two
/// on the same edge keep the wider one.
[[nodiscard]] constexpr chunk_fill merge(chunk_fill cur, chunk_fill want) noexcept
{
  if (cur.is_unset()) { return want; }
  if (want.is_unset()) { return cur; }
  if (cur.full || want.full) { return chunk_fill::whole(); }
  if (cur.suffix != want.suffix) { return chunk_fill::whole(); }
  return cur.pages >= want.pages ? cur : want;
}

/// True iff @p f guarantees the bytes [@p lo, @p hi) of the chunk at
/// @p chunk_off have been written.  An @c unset extent covers nothing.
[[nodiscard]] constexpr bool covers(
  chunk_fill f, std::size_t chunk_off, std::size_t chunk_bytes, std::size_t lo, std::size_t hi)
{
  if (f.full) { return true; }
  if (f.is_unset()) { return false; }
  auto const bytes = std::min(static_cast<std::size_t>(f.pages) * io::IO_BLOCK_SIZE, chunk_bytes);
  return f.suffix ? lo >= chunk_off + chunk_bytes - bytes : hi <= chunk_off + bytes;
}

/// The half-open file span that must be read to populate the chunk at
/// @p chunk_off to exactly the extent @p f advertises.  Deriving the span FROM
/// the extent (rather than from the request that motivated it) is what
/// guarantees the bytes read are exactly the bytes @c covers will later claim.
/// An @c unset extent conservatively yields the whole chunk.
[[nodiscard]] constexpr std::pair<std::size_t, std::size_t> fill_span(
  chunk_fill f, std::size_t chunk_off, std::size_t chunk_bytes) noexcept
{
  if (f.full || f.is_unset()) { return {chunk_off, chunk_off + chunk_bytes}; }
  auto const bytes = std::min(static_cast<std::size_t>(f.pages) * io::IO_BLOCK_SIZE, chunk_bytes);
  return f.suffix ? std::pair{chunk_off + chunk_bytes - bytes, chunk_off + chunk_bytes}
                  : std::pair{chunk_off, chunk_off + bytes};
}

/// The extent a request over [@p req_lo, @p req_hi) (NOT yet clamped to the
/// chunk) implies for the chunk at @p chunk_off.  Magnitudes are rounded out to
/// whole pages, so an interior read conservatively fills to the nearer chunk
/// edge — it over-reads the head or tail, never the whole chunk.
///
/// An extent is edge-anchored (see @ref chunk_fill), so a request touching
/// neither edge cannot be recorded as-is and has to widen to one of them.  Both
/// widenings are correct; this picks whichever fetches less, which for a request
/// near the head is the head.  Anchoring unconditionally to the tail — the
/// obvious reading of "starts inside the chunk, so it is a suffix" — costs the
/// whole rest of the chunk for a request sitting one page in.
[[nodiscard]] constexpr chunk_fill needed_fill(std::size_t chunk_off,
                                               std::size_t chunk_bytes,
                                               std::size_t req_lo,
                                               std::size_t req_hi) noexcept
{
  constexpr std::size_t page = io::IO_BLOCK_SIZE;
  auto const lo              = std::max(req_lo, chunk_off);
  auto const hi              = std::min(req_hi, chunk_off + chunk_bytes);
  if (lo >= hi) { return chunk_fill::unset(); }  // no overlap
  if (lo <= chunk_off && hi >= chunk_off + chunk_bytes) { return chunk_fill::whole(); }

  // Both shapes contain [lo, hi); the smaller one is the one worth reading.
  auto const prefix_bytes = std::min(align_up(hi - chunk_off, page), chunk_bytes);
  auto const suffix_bytes = std::min((chunk_off + chunk_bytes) - align_down(lo, page), chunk_bytes);
  bool const prefer_prefix = prefix_bytes <= suffix_bytes;
  auto const bytes         = prefer_prefix ? prefix_bytes : suffix_bytes;
  if (bytes >= chunk_bytes) { return chunk_fill::whole(); }
  auto const pages = static_cast<std::uint16_t>(bytes / page);
  return prefer_prefix ? chunk_fill::prefix_of(pages) : chunk_fill::suffix_of(pages);
}

// ---------------------------------------------------------------------------
// chunk_state — the whole per-chunk concurrency state, in one atomic word
// ---------------------------------------------------------------------------
//
// State, reader pins, the populated extent, and the live-subscriber count all
// live in a single atomic uint64_t:
//
//   bits [ 3: 0]  state        4b   empty|queued|allocated|loading|cached|in_use|evicting
//   bits [15: 4]  pins        12b   concurrent readers (max 4095)
//   bits [29:16]  fill_pages  14b   populated extent, in IO_BLOCK_SIZE pages
//   bit  [30]     fill_side    1b   0 = prefix from chunk start, 1 = suffix to chunk end
//   bit  [31]     fill_full    1b   whole chunk populated (overrides the two above)
//   bits [47:32]  subscribers 16b   live prefetch requests naming this chunk
//   bits [63:48]  spare
//
// Packing buys three things beyond size.  (1) Every transition is one CAS, so
// there is no TOCTOU window between reading the state and mutating the pins or
// the extent — the escape hatch a lock would need for the extent merge is just
// the CAS retry.  (2) A loader receives the extent it must fill out of the same
// CAS that claims the chunk, so it can never fill a different extent from the
// one it publishes.  (3) The eviction sweep answers "is this chunk reclaimable"
// with a single relaxed load instead of a locked read plus a second cache line.
//
// State machine — each row is the complete set of valid outbound transitions
// for that state.  Any other transition is rejected by the corresponding
// method's precondition (return value == false).
//
//   empty      ──mark_queued()──────────►  queued
//   queued     ──mark_allocated()───────►  allocated
//   allocated  ──take_loading()─────────►  loading
//   allocated  ──mark_evicting()────────►  evicting
//   loading    ──mark_cached()──────────►  cached
//   loading    ──mark_load_failed()─────►  allocated     (IO failure)
//   cached     ──mark_evicting()────────►  evicting
//   cached     ──acquire_read()─────────►  in_use(pin = 1)
//   in_use     ──acquire_read()─────────►  in_use(pin += 1)
//   in_use     ──release_read()─────────►  in_use(pin -= 1) | cached (when pin → 0)
//   evicting   ──mark_empty()───────────►  empty         (clears the extent)
//
// `empty` is the only state with no inbound transitions other than from
// `evicting` — once a chunk leaves `empty`, it can only return through the
// `evicting` reclamation path.  `evicting` is a one-way transit state.
//
// The subscriber count is orthogonal to the state machine: it is incremented
// when a request names the chunk and decremented when that request is retired,
// and it survives every state transition including `mark_empty()`.

class chunk_state {
 public:
  enum value : std::uint8_t {
    empty     = 0,
    queued    = 1,  ///< registered for prefetch, not yet given a buffer
    allocated = 2,  ///< buffer assigned, IO not yet dispatched
    loading   = 3,
    cached    = 4,
    in_use    = 5,
    evicting  = 6,
  };

  static constexpr std::uint32_t MAX_PINS        = (1U << 12) - 1;
  static constexpr std::uint32_t MAX_SUBSCRIBERS = (1U << 16) - 1;

  /// Widest extent the packed field can express, in pages.  The chunk size must
  /// stay under this many pages; @ref buffer_pool checks it at construction.
  [[nodiscard]] static constexpr std::uint32_t max_fill_pages() noexcept { return (1U << 14) - 1; }

  chunk_state() noexcept = default;

  chunk_state(chunk_state const&)            = delete;
  chunk_state& operator=(chunk_state const&) = delete;

  /// An immutable read of the whole word.  One relaxed load answers every
  /// question the eviction sweep asks, so the sweep never writes to the line.
  class snapshot {
   public:
    explicit constexpr snapshot(std::uint64_t w) noexcept : _w(w) {}

    [[nodiscard]] constexpr value state() const noexcept
    {
      return static_cast<value>(_w & STATE_MASK);
    }
    [[nodiscard]] constexpr std::uint32_t pins() const noexcept
    {
      return static_cast<std::uint32_t>((_w & PIN_MASK) >> PIN_SHIFT);
    }
    [[nodiscard]] constexpr std::uint32_t subscribers() const noexcept
    {
      return static_cast<std::uint32_t>((_w & SUB_MASK) >> SUB_SHIFT);
    }
    [[nodiscard]] constexpr chunk_fill fill() const noexcept { return decode(_w); }

    /// The chunk owns a staging buffer and no reader is holding it — i.e. it is
    /// a candidate for @ref mark_evicting.  Says nothing about subscribers; the
    /// evictor gates on those separately so its last-resort pass can ignore them.
    [[nodiscard]] constexpr bool is_reclaimable() const noexcept
    {
      auto const s = state();
      return (s == allocated || s == cached) && pins() == 0;
    }

   private:
    std::uint64_t _w;
  };

  [[nodiscard]] snapshot load() const noexcept
  {
    return snapshot(_w.load(std::memory_order_acquire));
  }

  [[nodiscard]] value get_state() const noexcept { return load().state(); }
  [[nodiscard]] std::uint32_t get_pin_count() const noexcept { return load().pins(); }
  [[nodiscard]] std::uint32_t get_subscribers() const noexcept { return load().subscribers(); }
  [[nodiscard]] chunk_fill get_fill() const noexcept { return load().fill(); }

  /// empty → queued.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_queued() noexcept { return transition(empty, queued); }

  /// queued → allocated.  Called by the allocator once it attaches a buffer.
  [[nodiscard]] bool mark_allocated() noexcept { return transition(queued, allocated); }

  /// loading → cached.  Publishes the loader's writes to subsequent readers:
  /// this is the release half of the pair @ref acquire_read completes.
  [[nodiscard]] bool mark_cached() noexcept { return transition(loading, cached); }

  /// loading → allocated (IO-failure revert).  The buffer stays attached so a
  /// later reader can retry the load without a fresh queue/allocate roundtrip.
  /// The recorded extent is left alone: `allocated` is not readable, and the
  /// next loader re-derives its fill span from whatever the extent then holds.
  [[nodiscard]] bool mark_load_failed() noexcept { return transition(loading, allocated); }

  /// allocated → loading, handing back the extent the loader must populate.
  /// Reading the extent out of the claiming CAS (rather than with a second,
  /// unsynchronized load) is what makes "bytes read" == "bytes advertised".
  [[nodiscard]] bool take_loading(chunk_fill& out) noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      if ((cur & (STATE_MASK | PIN_MASK)) != static_cast<std::uint64_t>(allocated)) {
        return false;
      }
      std::uint64_t const next = (cur & ~STATE_MASK) | static_cast<std::uint64_t>(loading);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        out = decode(cur);
        return true;
      }
    }
  }

  /// allocated → loading, first widening the recorded extent by @p want.
  /// @p out is the MERGED extent — a demand read that claims a chunk somebody
  /// else queued for a wider fill must honour the wider promise, or a reader
  /// waiting on it would later be told bytes are present that nobody wrote.
  [[nodiscard]] bool take_loading_merging(chunk_fill want, chunk_fill& out) noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      if ((cur & (STATE_MASK | PIN_MASK)) != static_cast<std::uint64_t>(allocated)) {
        return false;
      }
      chunk_fill const merged = merge(decode(cur), want);
      std::uint64_t const next =
        (cur & ~(STATE_MASK | FILL_MASK)) | static_cast<std::uint64_t>(loading) | encode(merged);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        out = merged;
        return true;
      }
    }
  }

  /// Widen the recorded extent by @p want, but only while the chunk is still
  /// pre-load.  A `loading` chunk is owned by its loader; a `cached` / `in_use`
  /// one is already populated to its current extent, and widening it would
  /// advertise bytes nobody wrote.  A request needing more than a cached chunk
  /// holds correctly MISSES via @ref try_pin_covering and reads for itself.
  ///
  /// Returns true if the recorded extent now covers @p want.
  bool merge_fill(chunk_fill want) noexcept
  {
    if (want.is_unset()) { return false; }
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      switch (static_cast<value>(cur & STATE_MASK)) {
        case empty:
        case queued:
        case allocated: break;
        default: return false;  // loading: owned by its loader; cached/in_use: already populated
      }
      std::uint64_t const next = (cur & ~FILL_MASK) | encode(merge(decode(cur), want));
      if (next == cur) { return true; }  // already covered — no atomic RMW at all
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
  }

  /// (cached | in_use) → in_use with pins += 1.
  [[nodiscard]] bool acquire_read() noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      if (!readable(cur)) { return false; }
      if (!pin_once(cur)) { return true; }
    }
  }

  /// (cached | in_use) → in_use with pins += 1, but only if the populated
  /// extent already covers [@p lo, @p hi).  A coverage miss costs one relaxed
  /// load and no atomic RMW — with partial fills in play that is the common
  /// case, and the pin/unpin pair it replaces was two.
  [[nodiscard]] bool try_pin_covering(std::size_t chunk_off,
                                      std::size_t chunk_bytes,
                                      std::size_t lo,
                                      std::size_t hi) noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      if (!readable(cur)) { return false; }
      if (!covers(decode(cur), chunk_off, chunk_bytes, lo, hi)) { return false; }
      if (!pin_once(cur)) { return true; }
    }
  }

  /// Decrement the pin count, returning to `cached` at zero.
  /// Returns true if this was the last reader.
  bool release_read() noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      auto const pins = static_cast<std::uint64_t>((cur & PIN_MASK) >> PIN_SHIFT);
      assert(static_cast<value>(cur & STATE_MASK) == in_use && pins > 0);
      std::uint64_t const left = pins - 1;
      std::uint64_t const next = (cur & ~(STATE_MASK | PIN_MASK)) |
                                 static_cast<std::uint64_t>(left == 0 ? cached : in_use) |
                                 (left << PIN_SHIFT);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return left == 0;
      }
    }
  }

  /// (allocated | cached) with no reader → evicting.  When @p only_unsubscribed
  /// the chunk must also have no live subscriber; the evictor clears that flag
  /// only for its last-resort pass, when nothing else can be freed.
  [[nodiscard]] bool mark_evicting(bool only_unsubscribed = true) noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      auto const st = static_cast<value>(cur & STATE_MASK);
      if (st != allocated && st != cached) { return false; }
      if ((cur & PIN_MASK) != 0) { return false; }
      if (only_unsubscribed && (cur & SUB_MASK) != 0) { return false; }
      std::uint64_t const next = (cur & ~STATE_MASK) | static_cast<std::uint64_t>(evicting);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
  }

  /// evicting → empty, clearing the populated extent in the same CAS so a
  /// reclaimed chunk can never advertise the previous tenant's bytes.
  /// Subscribers are preserved: a request may have named the chunk again while
  /// it was in transit, and it still has to be able to drop its reference.
  [[nodiscard]] bool mark_empty() noexcept { return transition(evicting, empty, FILL_MASK); }

  /// Count one more live request naming this chunk.  Saturates rather than
  /// wrapping — an overflowed count would make the chunk permanently evictable.
  void add_subscriber() noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_relaxed);
    for (;;) {
      auto const n = (cur & SUB_MASK) >> SUB_SHIFT;
      if (n >= MAX_SUBSCRIBERS) { return; }
      std::uint64_t const next = (cur & ~SUB_MASK) | ((n + 1) << SUB_SHIFT);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
      }
    }
  }

  /// Retire one request's reference.  Clamps at zero rather than borrowing into
  /// the neighbouring fields, so a double-drop degrades to a lost hint instead
  /// of corrupting the state machine.
  void drop_subscriber() noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_relaxed);
    for (;;) {
      auto const n = (cur & SUB_MASK) >> SUB_SHIFT;
      if (n == 0) { return; }
      std::uint64_t const next = (cur & ~SUB_MASK) | ((n - 1) << SUB_SHIFT);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
      }
    }
  }

 private:
  static constexpr std::uint64_t STATE_MASK = 0xFULL;
  static constexpr int PIN_SHIFT            = 4;
  static constexpr std::uint64_t PIN_MASK   = 0xFFFULL << PIN_SHIFT;
  static constexpr int PAGE_SHIFT           = 16;
  static constexpr std::uint64_t PAGE_MASK  = 0x3FFFULL << PAGE_SHIFT;
  static constexpr std::uint64_t SIDE_BIT   = 1ULL << 30;
  static constexpr std::uint64_t FULL_BIT   = 1ULL << 31;
  static constexpr std::uint64_t FILL_MASK  = PAGE_MASK | SIDE_BIT | FULL_BIT;
  static constexpr int SUB_SHIFT            = 32;
  static constexpr std::uint64_t SUB_MASK   = 0xFFFFULL << SUB_SHIFT;

  static constexpr std::uint64_t encode(chunk_fill f) noexcept
  {
    return (static_cast<std::uint64_t>(f.pages) << PAGE_SHIFT) | (f.suffix ? SIDE_BIT : 0) |
           (f.full ? FULL_BIT : 0);
  }

  static constexpr chunk_fill decode(std::uint64_t w) noexcept
  {
    return chunk_fill{static_cast<std::uint16_t>((w & PAGE_MASK) >> PAGE_SHIFT),
                      (w & SIDE_BIT) != 0,
                      (w & FULL_BIT) != 0};
  }

  static constexpr bool readable(std::uint64_t w) noexcept
  {
    auto const st = static_cast<value>(w & STATE_MASK);
    return st == cached || st == in_use;
  }

  /// One attempt at (cached | in_use) → in_use, pins += 1.  Returns false when
  /// the CAS succeeded (or the pin count is saturated, which cannot be retried);
  /// true means @p cur was refreshed and the caller must re-check its guards.
  bool pin_once(std::uint64_t& cur) noexcept
  {
    auto const pins = (cur & PIN_MASK) >> PIN_SHIFT;
    if (pins >= MAX_PINS) { return false; }
    std::uint64_t const next = (cur & ~(STATE_MASK | PIN_MASK)) |
                               static_cast<std::uint64_t>(in_use) | ((pins + 1) << PIN_SHIFT);
    return !_w.compare_exchange_weak(
      cur, next, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  /// Exact-precondition transition: requires state == @p from AND no reader
  /// pins, preserves the extent and the subscriber count, and additionally
  /// clears the bits in @p clear.
  bool transition(value from, value to, std::uint64_t clear = 0) noexcept
  {
    std::uint64_t cur = _w.load(std::memory_order_acquire);
    for (;;) {
      if ((cur & (STATE_MASK | PIN_MASK)) != static_cast<std::uint64_t>(from)) { return false; }
      std::uint64_t const next = (cur & ~(STATE_MASK | clear)) | static_cast<std::uint64_t>(to);
      if (_w.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
  }

  std::atomic<std::uint64_t> _w{0};  // == {empty, 0 pins, unset extent, 0 subscribers}
};

static_assert(sizeof(chunk_state) == 8, "chunk_state must stay a single 64-bit word");

// ---------------------------------------------------------------------------
// producer_stage — atomic state of the producer side of a scan
// ---------------------------------------------------------------------------
//
// Single atomic uint32_t holding the current state.  Forward transitions are
// *monotone-max*: `mark_x()` succeeds iff the current state is strictly earlier
// than x, and then CASes straight to x — intermediate stages may be skipped.
// It returns false when the state already is at or past x, so the state can
// never move backwards and a skipped stage cannot wedge the machine.
//
//   < queued    ──mark_queued()──►      queued
//   < preparing ──mark_preparing()──►   preparing
//   < prepared  ──mark_prepared()──►    prepared
//   < loading   ──mark_loading()──►     loading
//   < ready     ──mark_ready()──►       ready
//   any         ──mark_abandoned()──►   abandoned       (request dropped)
//
// The one exception is the backward IO-failure revert, which stays an exact
// precondition CAS:
//
//   loading     ──mark_load_failed()──► prepared        (IO failure revert)
//
// `preparing` and `loading` are the two wait points: waiters park in
// wait_for_prepared() / wait_till_not_loading() until the state moves on, and
// each reports whether its target was actually reached.  `abandoned` is terminal and
// exists so that every path which drops a request can leave the producer in a
// non-transient, notified state instead of stranding waiters.

class producer_stage {
 public:
  enum value : uint32_t {
    initialized = 0,
    queued      = 1,
    preparing   = 2,
    prepared    = 3,
    loading     = 4,
    ready       = 5,
    abandoned   = 6,  ///< terminal: the request was dropped before completing
  };

  producer_stage() noexcept = default;

  /// Current state.
  [[nodiscard]] value get() const noexcept
  {
    return static_cast<value>(_packed.load(std::memory_order_acquire));
  }

  /// → queued.  Returns false if the state is already at or past @c queued.
  [[nodiscard]] bool mark_queued() noexcept { return advance(queued); }

  /// → preparing.  Returns false if the state is already at or past @c preparing.
  [[nodiscard]] bool mark_preparing() noexcept { return advance(preparing); }

  /// → prepared.  Wakes threads parked in @c wait_for_prepared().  Returns
  /// false if the state is already at or past @c prepared.
  [[nodiscard]] bool mark_prepared() noexcept { return advance_and_notify(prepared); }

  /// → loading.  Returns false if the state is already at or past @c loading.
  [[nodiscard]] bool mark_loading() noexcept { return advance(loading); }

  /// → ready.  Wakes threads parked in @c wait_till_not_loading().  Returns
  /// false if the state is already at or past @c ready.
  [[nodiscard]] bool mark_ready() noexcept { return advance_and_notify(ready); }

  /// loading → prepared (IO-failure revert).  The only backward transition, so
  /// it keeps an exact precondition CAS and returns false from any other state.
  /// Wakes threads parked in @c wait_till_not_loading().
  [[nodiscard]] bool mark_load_failed() noexcept
  {
    auto expected = static_cast<uint32_t>(loading);
    bool ok       = _packed.compare_exchange_strong(
      expected, static_cast<uint32_t>(prepared), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// any → abandoned (the request was dropped).  Always succeeds and always
  /// wakes threads parked in @c wait_for_prepared() / @c wait_till_not_loading().
  void mark_abandoned() noexcept
  {
    _packed.exchange(static_cast<uint32_t>(abandoned), std::memory_order_acq_rel);
    _packed.notify_all();
  }

  /// Block while the state is @c preparing.  Returns true iff the request
  /// reached @c prepared (or beyond), false if it was abandoned.
  [[nodiscard]] bool wait_for_prepared() noexcept
  {
    auto const st = wait_while(preparing);
    return st >= prepared && st != abandoned;
  }

  /// Block until state >= @c prepared, regardless of which pre-prepared state
  /// (initialized, queued, preparing) it starts in.  Each intermediate state
  /// either returns immediately from @c _packed.wait (value already changed)
  /// or parks until @c mark_prepared / @c mark_abandoned notifies.
  [[nodiscard]] bool wait_until_prepared() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (cur < static_cast<uint32_t>(prepared)) {
      _packed.wait(cur, std::memory_order_relaxed);
      cur = _packed.load(std::memory_order_acquire);
    }
    return static_cast<value>(cur) != abandoned;
  }

  /// Block until the state stops being @c loading.
  ///
  /// The wait is over the loading window, not over readiness: a load is not a
  /// promise of success.  It can settle at @c ready, revert to @c prepared when
  /// the IO fails, or be cut short by @c abandoned -- so the caller is told
  /// which of those happened rather than just "done".
  ///
  /// @return true iff the load this call waited out reached @c ready.
  ///
  /// Only a load actually in flight is worth blocking on, so any other state
  /// returns false immediately rather than parking.  That includes @c ready:
  /// this reports on a load it witnessed finish, and a request already past
  /// @c loading has nothing left for the caller to wait out.
  [[nodiscard]] bool wait_till_not_loading() noexcept
  {
    if (get() != loading) { return false; }
    return wait_while(loading) == ready;
  }

 private:
  bool advance(value to) noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (cur < static_cast<uint32_t>(to)) {
      if (_packed.compare_exchange_weak(
            cur, static_cast<uint32_t>(to), std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  bool advance_and_notify(value to) noexcept
  {
    bool ok = advance(to);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  value wait_while(value st) noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (cur == static_cast<uint32_t>(st)) {
      _packed.wait(cur, std::memory_order_relaxed);
      cur = _packed.load(std::memory_order_acquire);
    }
    return static_cast<value>(cur);
  }

  std::atomic<uint32_t> _packed{static_cast<uint32_t>(initialized)};
};

// ---------------------------------------------------------------------------
// consumer_stage — atomic state of the consumer side of a scan
// ---------------------------------------------------------------------------
//
// Mirrors the consumer-visible values of @c scan_stage.  Forward transitions
// are *monotone-max*, exactly as in @c producer_stage: `mark_x()` succeeds iff
// the current state is strictly earlier than x and then CASes straight to x, so
// stages may be skipped but the state never moves backwards.  `disposed` is the
// last value, hence reachable from every state — cancellation can happen at any
// time.
//
//   < queued    ──mark_queued()──►     queued
//   < preparing ──mark_preparing()──►  preparing
//   < reading   ──mark_reading()──►    reading
//   any         ──mark_disposed()──►   disposed

class consumer_stage {
 public:
  enum value : uint32_t {
    initialized = 0,
    queued      = 1,
    preparing   = 2,
    reading     = 3,
    disposed    = 4,
  };

  consumer_stage() noexcept = default;

  /// Current state.
  [[nodiscard]] value get() const noexcept
  {
    return static_cast<value>(_packed.load(std::memory_order_acquire));
  }

  /// → queued.  Returns false if the state is already at or past @c queued.
  [[nodiscard]] bool mark_queued() noexcept { return advance(queued); }

  /// → preparing.  Returns false if the state is already at or past @c preparing.
  [[nodiscard]] bool mark_preparing() noexcept { return advance(preparing); }

  /// → reading.  Returns false if the state is already at or past @c reading.
  [[nodiscard]] bool mark_reading() noexcept { return advance(reading); }

  /// any → disposed (cancellation).  Always succeeds.
  void mark_disposed() noexcept
  {
    _packed.exchange(static_cast<uint32_t>(disposed), std::memory_order_acq_rel);
  }

  /// Advance to @p to.  Routes @c disposed to @ref mark_disposed (which always
  /// succeeds); otherwise monotone-max, returning false if already at or past.
  [[nodiscard]] bool mark(value to) noexcept
  {
    if (to == disposed) {
      mark_disposed();
      return true;
    }
    return advance(to);
  }

 private:
  bool advance(value to) noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (cur < static_cast<uint32_t>(to)) {
      if (_packed.compare_exchange_weak(
            cur, static_cast<uint32_t>(to), std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  std::atomic<uint32_t> _packed{static_cast<uint32_t>(initialized)};
};

/// Map a @ref scan_stage onto its @ref consumer_stage counterpart.
/// @c none has no counterpart and yields nullopt.
[[nodiscard]] inline std::optional<consumer_stage::value> to_consumer_stage(scan_stage s) noexcept
{
  switch (s) {
    case scan_stage::none: return std::nullopt;
    case scan_stage::initialized: return consumer_stage::initialized;
    case scan_stage::queued: return consumer_stage::queued;
    case scan_stage::preparing: return consumer_stage::preparing;
    case scan_stage::reading: return consumer_stage::reading;
    case scan_stage::disposed: return consumer_stage::disposed;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// cached_chunk — one chunk-aligned slot of a file
// ---------------------------------------------------------------------------
//
// Deliberately small and deliberately NOT cache-line aligned.  Chunks adjacent
// in a file are almost always walked by the same thread (the request that
// covers that range), so packing them two-per-half-line quarters the lines a
// range sweep touches; padding each chunk out to 64 bytes would defend against
// a contention pattern this workload does not produce.  All cross-thread state
// is in the single word, so there is nothing else to isolate.

struct cached_chunk {
  cached_chunk() noexcept = default;
  explicit cached_chunk(std::size_t off) noexcept : offset(off) {}

  chunk_state state;
  std::size_t offset{0};
  std::uint8_t* data{nullptr};
  std::int32_t numa_node{-1};
};

static_assert(sizeof(cached_chunk) == 32, "cached_chunk should stay 4 per cache line");

// ---------------------------------------------------------------------------
// chunk_arena — append-only, pointer-stable storage for cached_chunk
// ---------------------------------------------------------------------------
//
// Raw @c cached_chunk* escape into prefetch requests and outlive the call that
// produced them, so chunk addresses must never move.  Chunks are never removed:
// eviction resets one in place (buffer returned, state back to `empty`), which
// is why a plain bump allocator over stable slabs is enough — no free list.

class chunk_arena {
 public:
  static constexpr std::size_t SLAB_CHUNKS = 1024;  // 32 KiB per slab

  [[nodiscard]] cached_chunk* emplace(std::size_t offset)
  {
    if (_used == SLAB_CHUNKS) {
      _slabs.push_back(std::make_unique<slab>());
      _used = 0;
    }
    auto* c   = &(*_slabs.back())[_used++];
    c->offset = offset;
    return c;
  }

 private:
  using slab = std::array<cached_chunk, SLAB_CHUNKS>;

  std::vector<std::unique_ptr<slab>> _slabs;
  std::size_t _used{SLAB_CHUNKS};  // forces a slab on first use
};

/// Coverage requirement for @ref find_entry.
enum class coverage_policy {
  full,     // return the chunks only when they fully cover [offset, offset + size); else none
  partial,  // return every chunk overlapping [offset, offset + size), even if coverage is partial
};

/// Select the chunks of @p chunks (sorted by offset, non-overlapping, all
/// @p chunk_size bytes apart) that serve [@p offset, @p offset + @p size).
///
/// With @c coverage_policy::full the request must map onto a contiguous run of
/// present chunks or an empty vector is returned; with @c partial every
/// overlapping chunk is returned regardless of gaps.  This is POSITIONAL
/// coverage only — that the requested bytes are actually populated is a
/// separate question, answered by @c chunk_state::try_pin_covering.
///
/// Pure lookup: it does not mutate the chunks.
[[nodiscard]] inline std::vector<cached_chunk*> find_entry(std::span<cached_chunk* const> chunks,
                                                           std::size_t offset,
                                                           std::size_t size,
                                                           coverage_policy policy,
                                                           std::size_t chunk_size)
{
  if (size == 0) { return {}; }

  auto const first_chunk_off = (offset / chunk_size) * chunk_size;
  auto const last_chunk_off  = ((offset + size - 1) / chunk_size) * chunk_size;
  auto const expected_count  = (last_chunk_off - first_chunk_off) / chunk_size + 1;

  // Find the first chunk at/after the aligned start (chunks are sorted).
  auto const first_it = std::lower_bound(
    chunks.begin(), chunks.end(), first_chunk_off, [](cached_chunk* c, std::size_t v) {
      return c->offset < v;
    });

  std::vector<cached_chunk*> result;
  result.reserve(expected_count);

  if (policy == coverage_policy::full) {
    if (chunks.size() < expected_count) { return {}; }
    if (first_it == chunks.end() || (*first_it)->offset != first_chunk_off) { return {}; }

    auto const first_idx = static_cast<std::size_t>(first_it - chunks.begin());
    auto const last_idx  = first_idx + expected_count - 1;
    if (last_idx >= chunks.size() || chunks[last_idx]->offset != last_chunk_off) { return {}; }

    // Coverage confirmed by the invariant: sorted + non-overlapping + fixed-size
    // means consecutive chunks differ by exactly chunk_size, so the intermediates
    // are forced once the first and last are at the expected positions.
    result.assign(chunks.begin() + static_cast<std::ptrdiff_t>(first_idx),
                  chunks.begin() + static_cast<std::ptrdiff_t>(last_idx) + 1);
    return result;
  }

  for (auto it = first_it; it != chunks.end() && (*it)->offset <= last_chunk_off; ++it) {
    result.push_back(*it);
  }
  return result;
}

}  // namespace sirius::io::cache
