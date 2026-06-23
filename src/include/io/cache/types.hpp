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

// Shared cache entry types used by both prefetching_cache and the IO context
// virtual interface (device_read_async_io_using).  Extracted here to break the
// circular include between io_context.hpp and prefetching_cache.hpp.

#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace sirius::io::cache {

/**
 * @brief How the prefetching layer should behave on top of a given backend.
 *
 * - @c none: no prefetching.  Either the backend does not support vector host
 *   reads (so the prefetcher cannot batch range requests cheaply) or the
 *   backend explicitly opted out.
 * - @c immediate: prefill the cache ahead of consumer demand.
 * - @c opportunistic: read-ahead on demand — issue extra IO only when triggered by a
 *   consumer read.
 * - @c disposable: prefetching is temporary and can be discarded when no longer needed.
 */
enum class prefetching_stage { none, opportunistic, immediate, just_in_time, disposable };

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
  static constexpr uint32_t CHUNKS_PER_SLAB = 500;  // 500 chunks per slab

  /// @p initial_slabs slabs are allocated up-front from @p mr (clamped to
  /// @p max_slabs).  Default preserves the historical behaviour of warming
  /// the pool with up to 10 slabs at construction.
  buffer_pool(cucascade::memory::memory_reservation_manager& reservation_manager,
              uint32_t initial_slabs = 10);

  ~buffer_pool();

  buffer_pool(buffer_pool const&)            = delete;
  buffer_pool& operator=(buffer_pool const&) = delete;

  /// Bulk-allocate up to @p n chunks, appending pointers to @p out.
  /// Returns the number actually allocated (may be < n if the pool is
  /// exhausted and cannot grow).
  std::vector<std::byte*> allocate_bulk(size_t n, int& numa_node);

  std::vector<std::byte*> allocate_bulk_from(size_t n, int numa_node);

  void deallocate_bulk(std::vector<std::byte*>&& out, int numa) noexcept;

  [[nodiscard]] size_t chunk_bytes() const noexcept { return _chunk_bytes; }

  /// Number of chunks currently handed out (outstanding) across all arenas.
  [[nodiscard]] size_t total_chunks() const noexcept { return _n_allocated_chunks; }

  /// Aggregate chunk capacity reserved across all arenas.  Stable for the
  /// pool's lifetime; used to score memory pressure for eviction.
  [[nodiscard]] size_t capacity_chunks() const noexcept { return _capacity_chunks; }

 private:
  struct host_arena {
    int numa_id;
    std::unique_ptr<cucascade::memory::reservation> reservation;
    cucascade::memory::fixed_size_host_memory_resource* mr;
  };

  size_t _chunk_bytes;
  size_t _capacity_chunks{0};
  std::unordered_map<int, size_t> _numa_to_arena_index;
  std::vector<host_arena> _host_arenas;
  std::atomic<size_t> _n_allocated_chunks{0};
};

// ---------------------------------------------------------------------------
// entry_state — packed atomic state + pin_count
// ---------------------------------------------------------------------------
//
// Packs a 4-bit state enum and a 28-bit reader pin count into a single
// atomic uint32_t.  Every transition is a single CAS, which eliminates the
// TOCTOU race between checking state and modifying pin_count.
//
// State machine — each row is the complete set of valid outbound transitions
// for that state.  Any other transition is rejected by the corresponding
// method's precondition CAS (return value == false).
//
//   empty      ──mark_queued()──►       queued
//   queued     ──mark_allocated()──►    allocated
//   allocated  ──mark_loading()──►      loading
//   allocated  ──mark_evicting()──►     evicting
//   loading    ──mark_cached()──►       cached
//   loading    ──mark_loading_in_use()──►       in_use(pin = 1)
//   loading    ──mark_load_failed()──►    allocated        (IO failure)
//   cached     ──mark_evicting()──►     evicting
//   cached     ──acquire_read()──►      in_use(pin = 1)
//   in_use     ──acquire_read()──►      in_use(pin += 1)
//   in_use     ──release_read()──►      in_use(pin -= 1) | cached (when pin → 0)
//   evicting   ──mark_empty()──►        empty
//
// `empty` is the only state with no inbound transitions other than from
// `evicting` — once an entry leaves `empty`, it can only return through the
// `evicting` reclamation path.  `evicting` is a one-way transit state.
//
// `loading` is the only non-terminal state with a wait point: readers that
// observe `loading` park on wait_while_pending() until the IO settles to one
// of cached / in_use(1) / allocated.

class entry_state {
 public:
  enum value : uint8_t {
    empty     = 0,
    queued    = 1,  ///< registered for prefetch, not yet given chunks
    allocated = 2,  ///< chunks assigned, IO not yet dispatched
    loading   = 3,
    cached    = 4,
    in_use    = 5,
    evicting  = 6,
  };

  entry_state() noexcept = default;

  [[nodiscard]] value get_state() const noexcept
  {
    return unpack_state(_packed.load(std::memory_order_acquire));
  }

  [[nodiscard]] uint32_t get_pin_count() const noexcept
  {
    return unpack_pins(_packed.load(std::memory_order_acquire));
  }

  /// empty → queued.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_queued() noexcept
  {
    auto expected = pack(empty, 0);
    return _packed.compare_exchange_strong(expected, pack(queued, 0), std::memory_order_acq_rel);
  }

  /// queued → allocated.  Returns false on precondition mismatch.  Called by
  /// the allocator when it attaches chunks to a previously-queued entry.
  [[nodiscard]] bool mark_allocated() noexcept
  {
    auto expected = pack(queued, 0);
    return _packed.compare_exchange_strong(expected, pack(allocated, 0), std::memory_order_acq_rel);
  }

  /// loading → allocated (IO-failure revert).  Returns false on precondition
  /// mismatch.  Used by io_dispatch_loop's failure paths to revert an entry
  /// whose IO did not complete: the entry's chunks stay attached so a
  /// subsequent allocated-steal read can retry the load with a fresh
  /// request_context, instead of discarding the entry to `empty` and forcing
  /// the next reader through a fresh queue/allocate roundtrip.  Wakes any
  /// threads parked in @c wait_while_pending().
  [[nodiscard]] bool mark_load_failed() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok =
      _packed.compare_exchange_strong(expected, pack(allocated, 0), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// allocated → loading.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_loading() noexcept
  {
    auto expected = pack(allocated, 0);
    return _packed.compare_exchange_strong(expected, pack(loading, 0), std::memory_order_acq_rel);
  }

  /// loading → cached.  Returns false on precondition mismatch.  Wakes any
  /// threads parked in @c wait_while_pending().
  [[nodiscard]] bool mark_cached() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok = _packed.compare_exchange_strong(expected, pack(cached, 0), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  [[nodiscard]] bool mark_loading_in_use() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok = _packed.compare_exchange_strong(expected, pack(in_use, 1), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// allocated → evicting, or cached → evicting.  Returns false on
  /// precondition mismatch.  Both source states have pin_count == 0 by
  /// invariant (allocated is set with pin==0 by mark_allocated(); cached is
  /// only entered from in_use via release_read() when pin → 0), so the two
  /// strong-CAS attempts below cover every legal transition exactly.
  [[nodiscard]] bool mark_evicting() noexcept
  {
    auto expected_allocated = pack(allocated, 0);
    if (_packed.compare_exchange_strong(
          expected_allocated, pack(evicting, 0), std::memory_order_acq_rel)) {
      return true;
    }
    auto expected_cached = pack(cached, 0);
    return _packed.compare_exchange_strong(
      expected_cached, pack(evicting, 0), std::memory_order_acq_rel);
  }

  /// evicting → empty.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_empty() noexcept
  {
    auto expected = pack(evicting, 0);
    return _packed.compare_exchange_strong(expected, pack(empty, 0), std::memory_order_acq_rel);
  }

  /// Block while state is @c loading.  Returns when the state transitions to
  /// any other state (i.e. to cached, in_use(1), or allocated — the three
  /// outbound transitions from `loading`).
  void wait_while_pending() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (unpack_state(cur) == loading) {
      _packed.wait(cur, std::memory_order_relaxed);
      cur = _packed.load(std::memory_order_acquire);
    }
  }

  /// Park while state is @c loading, then acquire a read pin via the normal
  /// (cached | in_use) → in_use(pin++) path.  Returns true if a read pin was
  /// acquired (load completed successfully and the entry is now readable),
  /// false if the load reverted to @c allocated (IO failure) or the entry was
  /// otherwise made non-readable (evicted / drained) while we were parked.
  /// Safe to call from any state: if the state is already past @c loading on
  /// entry, the wait loop is skipped and we go straight to @c acquire_read().
  [[nodiscard]] bool acquire_read_after_loading() noexcept
  {
    wait_while_pending();
    return acquire_read();
  }

  /// (cached | in_use) → in_use with pin_count += 1.
  /// Returns false if the entry is not in a readable state.
  [[nodiscard]] bool acquire_read() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != cached && st != in_use) return false;
      auto pins = unpack_pins(cur);
      auto next = pack(in_use, pins + 1);
      if (_packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire))
        return true;
    }
  }

  /// Decrement pin_count.  If it reaches 0, transition in_use → cached.
  /// Returns true if this was the last reader.
  bool release_read() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    assert(unpack_state(cur) == in_use && unpack_pins(cur) > 0);
    while (true) {
      auto pins      = unpack_pins(cur);
      auto new_pins  = pins - 1;
      auto new_state = new_pins == 0 ? cached : in_use;
      auto next      = pack(new_state, new_pins);
      if (_packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire))
        return new_pins == 0;
    }
  }

 private:
  static constexpr uint32_t STATE_BITS = 4;
  static constexpr uint32_t STATE_MASK = (1U << STATE_BITS) - 1;
  static constexpr uint32_t PIN_SHIFT  = STATE_BITS;

  static constexpr uint32_t pack(value s, uint32_t pins) noexcept
  {
    return static_cast<uint32_t>(s) | (pins << PIN_SHIFT);
  }
  static constexpr value unpack_state(uint32_t v) noexcept
  {
    return static_cast<value>(v & STATE_MASK);
  }
  static constexpr uint32_t unpack_pins(uint32_t v) noexcept { return v >> PIN_SHIFT; }

  std::atomic<uint32_t> _packed{pack(empty, 0)};
};

struct alignas(64) chunk_lifecycle {
  std::atomic<uint64_t> packed{0};

  static constexpr uint64_t INSERT_MASK = 0xFFFFull;
  static constexpr uint64_t READ_SHIFT  = 16;
  static constexpr uint64_t READ_MASK   = 0xFFFFull << READ_SHIFT;
  static constexpr uint64_t TICK_SHIFT  = 32;
  static constexpr uint64_t TICK_MASK   = 0xFFFFFFFFull << TICK_SHIFT;
  static constexpr uint64_t FRESH_SCORE = 4;

  static constexpr uint64_t pack(uint32_t tick, uint16_t reads, uint16_t inserts) noexcept
  {
    return (uint64_t(tick) << TICK_SHIFT) | (uint64_t(reads) << READ_SHIFT) | uint64_t(inserts);
  }

  void on_request(uint32_t query_tick) noexcept
  {
    uint64_t cur = packed.load(std::memory_order_relaxed);
    for (;;) {
      auto cur_tick    = uint32_t(cur >> TICK_SHIFT);
      auto cur_reads   = uint16_t((cur >> READ_SHIFT) & 0xFFFFu);
      auto cur_inserts = uint16_t(cur & INSERT_MASK);

      uint64_t next;
      if (query_tick > cur_tick) {
        // New query — reset counters to reflect just this insert.
        next = pack(query_tick, 0, 1);
      } else {
        // Same tick (or stale tick — treat as same). Increment inserts,
        // saturate at uint16 max to avoid wrap.
        uint16_t new_inserts = cur_inserts == 0xFFFFu ? 0xFFFFu : cur_inserts + 1;
        next                 = pack(cur_tick, cur_reads, new_inserts);
      }

      if (packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed))
        return;
    }
  }

  void on_consume() noexcept
  {
    uint64_t cur = packed.load(std::memory_order_relaxed);
    for (;;) {
      auto cur_reads   = uint16_t((cur >> READ_SHIFT) & 0xFFFFu);
      auto cur_inserts = uint16_t(cur & INSERT_MASK);

      // Clamp: never read more than we promised.
      if (cur_reads >= cur_inserts) return;

      uint64_t next = (cur & ~READ_MASK) | (uint64_t(cur_reads + 1) << READ_SHIFT);

      if (packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed))
        return;
    }
  }

  struct snapshot {
    uint32_t tick;
    uint16_t reads;
    uint16_t inserts;

    [[nodiscard]] uint16_t eviction_tier(uint32_t query_tick) const noexcept
    {
      return query_tick > tick
               ? 0
               : (inserts > reads ? std::min<uint16_t>(inserts - reads, FRESH_SCORE) : 0);
    }
  };

  [[nodiscard]] snapshot load() const noexcept
  {
    uint64_t v = packed.load(std::memory_order_acquire);
    return {
      uint32_t(v >> TICK_SHIFT),
      uint16_t((v >> READ_SHIFT) & 0xFFFFu),
      uint16_t(v & INSERT_MASK),
    };
  }
};

// ---------------------------------------------------------------------------
// cache_entry — per-range metadata
// ---------------------------------------------------------------------------
//
// State transitions are managed by the entry_state class above.
// See entry_state's state machine diagram for the full picture.

struct alignas(64) cached_chunk {
  explicit cached_chunk(size_t off) : offset(off) {}

  std::size_t offset;
  uint8_t* data;
  int numa_node{-1};
  entry_state state;
  chunk_lifecycle lifecycle;
};

// Coverage requirement for find_entry.
enum class coverage_policy {
  full,     // return the chunks only when they fully cover [offset, offset + size); else none
  partial,  // return every chunk overlapping [offset, offset + size), even if coverage is partial
};

// A pointer-like handle to a cached_chunk: a raw pointer or a smart pointer
// (std::unique_ptr / std::shared_ptr), with or without const.  std::to_address
// yields the underlying cached_chunk* (const-qualified iff the handle is to a
// const cached_chunk).
template <class P>
concept cached_chunk_pointer = requires(const P& p) {
  { std::to_address(p) } -> std::convertible_to<const cached_chunk*>;
};

// Find the cached chunks of @p chunks (sorted by offset, non-overlapping,
// fixed-size) that serve the request [@p offset, @p offset + @p size).
//
// With coverage_policy::full the request must be wholly covered or an empty
// vector is returned; with coverage_policy::partial every overlapping chunk is
// returned regardless of gaps.  Pure lookup: it does not mutate the chunks
// (callers apply lifecycle side effects on the result).  Accepts any contiguous
// range (vector, span, array, …) of cached_chunk pointer handles (raw / shared /
// unique, const or not); the returned pointers preserve the handle's constness.
template <std::ranges::contiguous_range Chunks>
  requires cached_chunk_pointer<std::ranges::range_value_t<Chunks>>
[[nodiscard]] std::vector<
  decltype(std::to_address(std::declval<const std::ranges::range_value_t<Chunks>&>()))>
find_entry(const Chunks& chunks, std::size_t offset, std::size_t size, coverage_policy policy)
{
  using chunk_ptr_t =
    decltype(std::to_address(std::declval<const std::ranges::range_value_t<Chunks>&>()));
  constexpr std::size_t chunk_size = 1 << 20;  // must match buffer_pool's chunk size
  if (size == 0) return {};

  auto const first        = std::ranges::begin(chunks);
  auto const last         = std::ranges::end(chunks);
  const std::size_t count = std::ranges::size(chunks);

  // Align the request to chunk boundaries.
  const std::size_t first_chunk_off = (offset / chunk_size) * chunk_size;
  const std::size_t end_off         = offset + size;
  const std::size_t last_chunk_off  = ((end_off - 1) / chunk_size) * chunk_size;
  const std::size_t expected_count  = (last_chunk_off - first_chunk_off) / chunk_size + 1;

  // Find the first chunk at/after the aligned start (chunks are sorted).
  auto first_it = std::lower_bound(first, last, first_chunk_off, [](const auto& c, std::size_t v) {
    return std::to_address(c)->offset < v;
  });

  if (policy == coverage_policy::full) {
    if (count < expected_count) return {};
    if (first_it == last || std::to_address(*first_it)->offset != first_chunk_off) {
      return {};  // first chunk missing
    }

    // Check the last chunk is at the expected position.
    const auto first_idx       = static_cast<std::size_t>(std::distance(first, first_it));
    const std::size_t last_idx = first_idx + expected_count - 1;

    if (last_idx >= count) return {};
    if (std::to_address(first[last_idx])->offset != last_chunk_off) return {};

    // Coverage confirmed by the invariant: sorted + non-overlapping + fixed-size
    // means consecutive chunks differ by exactly chunk_size, so the intermediates
    // are forced once the first and last are at the expected positions.
    std::vector<chunk_ptr_t> result;
    result.reserve(expected_count);
    for (std::size_t i = 0; i < expected_count; ++i) {
      first[first_idx + i]
        ->lifecycle.on_consume();  // side effect: count this chunk as consumed for eviction scoring
      result.push_back(std::to_address(first[first_idx + i]));
    }
    return result;
  }

  // coverage_policy::partial: return every chunk overlapping the request range,
  // even when some are missing (the invariant means offsets in
  // [first_chunk_off, last_chunk_off] are exactly the overlapping chunks).
  std::vector<chunk_ptr_t> result;
  result.reserve(expected_count);
  for (auto it = first_it; it != last && std::to_address(*it)->offset <= last_chunk_off; ++it) {
    std::to_address(*it)->lifecycle.on_consume();
    result.push_back(std::to_address(*it));
  }
  return result;
}

}  // namespace sirius::io::cache
