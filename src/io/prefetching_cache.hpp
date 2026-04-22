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

#include "concurrentqueue.h"
#include "io/admission_control.hpp"
#include "io/types.hpp"

#include <array>
#include <atomic>
#include <cudf/io/datasource.hpp>
#include <future>
#include <map>
#include <memory>
#include <semaphore>
#include <shared_mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {

// ---------------------------------------------------------------------------
// buffer_pool — growable multi-slab pool of 1MB pinned chunks
// ---------------------------------------------------------------------------
//
// Manages one or more CUDA-pinned slabs.  Each slab is a contiguous
// allocation of CHUNKS_PER_SLAB * 1MB.  New slabs are allocated lazily when
// the pool is exhausted, up to max_slabs.
//
// allocate() returns a raw pointer to a 1MB region.
// deallocate() finds the owning slab via a sorted map of slab base addresses
// and returns the chunk to that slab's free list.

class buffer_pool {
public:
  static constexpr size_t CHUNK_BYTES = 1UL << 20; // 1MB
  static constexpr uint32_t CHUNKS_PER_SLAB = 500; // 500 chunks per slab
  static constexpr size_t SLAB_BYTES =
      static_cast<size_t>(CHUNKS_PER_SLAB) * CHUNK_BYTES;

  explicit buffer_pool(uint32_t max_slabs);
  ~buffer_pool();

  buffer_pool(buffer_pool const &) = delete;
  buffer_pool &operator=(buffer_pool const &) = delete;

  /// Allocate a single 1MB chunk.  Returns nullptr when all slabs are
  /// exhausted and no new slab can be allocated.
  std::byte *allocate();

  /// Bulk-allocate up to @p n chunks, appending pointers to @p out.
  /// Returns the number actually allocated (may be < n if pool is exhausted
  /// and cannot grow).  Uses try_dequeue_bulk internally to minimise
  /// per-chunk overhead.
  size_t allocate_bulk(size_t n, std::vector<std::byte *> &out);

  /// Return a chunk to the pool.
  void deallocate(std::byte *p);

  size_t capacity() const noexcept {
    return static_cast<size_t>(_total_chunks.load(std::memory_order_relaxed)) *
           CHUNK_BYTES;
  }
  uint32_t free_count() const noexcept {
    return _total_free.load(std::memory_order_relaxed);
  }
  uint32_t total_chunks() const noexcept {
    return _total_chunks.load(std::memory_order_relaxed);
  }

private:
  struct slab {
    std::byte *base{nullptr};
    moodycamel::ConcurrentQueue<std::byte *> free_chunks;
    std::atomic<uint32_t> free_count{0};
  };

  bool grow();
  slab *find_slab(std::byte *p);

  uint32_t _max_slabs;

  // Protected by _grow_mtx (exclusive for grow, shared for find_slab).
  mutable std::shared_mutex _grow_mtx;
  std::vector<std::unique_ptr<slab>> _slabs;
  std::map<std::byte *, slab *> _slab_map;

  std::atomic<uint32_t> _total_free{0};
  std::atomic<uint32_t> _total_chunks{0};
};

// ---------------------------------------------------------------------------
// entry_state — packed atomic state + pin_count
// ---------------------------------------------------------------------------
//
// Packs a 4-bit state enum and a 28-bit reader pin count into a single
// atomic uint32_t.  Every transition is a single CAS (or store), which
// eliminates the TOCTOU race between checking state and modifying pin_count.
//
// State machine:
//
//         try_start_queueing()     try_start_loading()
//   empty ─────────────────► queued ──────────────────► loading
//     ▲                                                   │
//     │ mark_evicted()                          mark_cached() │ mark_load_failed()
//     │                                                   │             │
//     │   ┌───────────────────────────────────────────────▼             ▼
//  evicting │                                          cached ◄────── empty
//     ▲    │                                         ▲      │
//     │    │ release_read                            │      │ try_acquire_read()
//     │    │ (last reader)                           │      ▼
//     │    │                                         │  in_use (pin_count ≥ 1)
//     │    │                                         │
//     └────┘ try_start_evicting()  (only from cached, pin_count==0)

class entry_state {
public:
  enum value : uint8_t {
    empty = 0,
    queued = 1,
    loading = 2,
    cached = 3,
    in_use = 4,
    evicting = 5
  };

  entry_state() noexcept = default;

  [[nodiscard]] value get_state() const noexcept {
    return unpack_state(_packed.load(std::memory_order_acquire));
  }

  [[nodiscard]] uint32_t get_pin_count() const noexcept {
    return unpack_pins(_packed.load(std::memory_order_acquire));
  }

  /// empty → queued.  Returns false if not empty.
  /// Called by insert() to claim responsibility for scheduling a load.
  bool try_start_queueing() noexcept {
    auto expected = pack(empty, 0);
    return _packed.compare_exchange_strong(expected, pack(queued, 0),
                                           std::memory_order_acq_rel);
  }

  /// queued → loading.  Returns false if not queued.
  /// Called by the worker when it picks up a work item.
  bool try_start_loading() noexcept {
    auto expected = pack(queued, 0);
    return _packed.compare_exchange_strong(expected, pack(loading, 0),
                                           std::memory_order_acq_rel);
  }

  /// loading → cached.  Caller must ensure state is loading.
  /// Wakes any readers parked in @c wait_while_loading().
  void mark_cached() noexcept {
    _packed.store(pack(cached, 0), std::memory_order_release);
    _packed.notify_all();
  }

  /// loading → empty.  IO failed, chunks already freed by caller.
  /// Wakes any readers parked in @c wait_while_loading().
  void mark_load_failed() noexcept {
    _packed.store(pack(empty, 0), std::memory_order_release);
    _packed.notify_all();
  }

  /// Block while state == loading.  Returns when the state transitions
  /// out of loading (either cached on success or empty on failure).
  void wait_while_loading() noexcept {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (unpack_state(cur) == loading) {
      _packed.wait(cur, std::memory_order_relaxed);
      cur = _packed.load(std::memory_order_acquire);
    }
  }

  /// (cached | in_use) → in_use with pin_count+1.
  /// Returns false if the entry is not in a readable state.
  bool try_acquire_read() noexcept {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != cached && st != in_use)
        return false;
      auto pins = unpack_pins(cur);
      auto next = pack(in_use, pins + 1);
      if (_packed.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                        std::memory_order_acquire))
        return true;
    }
  }

  /// Decrement pin_count.  If it reaches 0, transition in_use → cached.
  /// Returns true if this was the last reader.
  bool release_read() noexcept {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    assert(unpack_state(cur) == in_use && unpack_pins(cur) > 0);
    while (true) {
      auto pins = unpack_pins(cur);
      auto new_pins = pins - 1;
      auto new_state = new_pins == 0 ? cached : in_use;
      auto next = pack(new_state, new_pins);
      if (_packed.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                        std::memory_order_acquire))
        return new_pins == 0;
    }
  }

  /// cached (pin_count==0) → evicting.
  /// Returns false if state != cached or readers are present.
  bool try_start_evicting() noexcept {
    auto expected = pack(cached, 0);
    return _packed.compare_exchange_strong(expected, pack(evicting, 0),
                                           std::memory_order_acq_rel);
  }

  /// evicting → empty.  Caller must ensure state is evicting.
  void mark_evicted() noexcept {
    _packed.store(pack(empty, 0), std::memory_order_release);
  }

private:
  static constexpr uint32_t STATE_BITS = 4;
  static constexpr uint32_t STATE_MASK = (1U << STATE_BITS) - 1;
  static constexpr uint32_t PIN_SHIFT = STATE_BITS;

  static constexpr uint32_t pack(value s, uint32_t pins) noexcept {
    return static_cast<uint32_t>(s) | (pins << PIN_SHIFT);
  }
  static constexpr value unpack_state(uint32_t v) noexcept {
    return static_cast<value>(v & STATE_MASK);
  }
  static constexpr uint32_t unpack_pins(uint32_t v) noexcept {
    return v >> PIN_SHIFT;
  }

  std::atomic<uint32_t> _packed{pack(empty, 0)};
};

// ---------------------------------------------------------------------------
// cache_entry — per-range metadata
// ---------------------------------------------------------------------------
//
// State transitions are managed by the entry_state class above.
// See entry_state's state machine diagram for the full picture.

struct alignas(64) cache_entry {
  cudf::io::text::byte_range_info logical_range;
  cudf::io::text::byte_range_info physical_range;

  /// Pointers to 1MB chunks from buffer_pool backing this range.
  std::vector<std::byte *> chunks;

  /// Packed state + pin_count.  All state transitions go through this.
  entry_state state;

  /// Cumulative number of insert() calls that included this range.
  /// Monotonically increasing — never decremented.  Used by the evictor
  /// (with _cache_age) to compute bucket_idx.
  std::atomic<int64_t> n_total_request{0};

  /// _cache_age at the most recent insert of this range.
  std::atomic<uint64_t> request_ts{0};

  /// _cache_age at the most recent successful read (pinned_view
  /// construction).  0 by default — never-consumed entries have
  /// consumption_ts < request_ts.
  std::atomic<uint64_t> consumption_ts{0};

  /// Evictor-managed: index of the LRU bucket this entry is currently in,
  /// or -1 if not in any bucket.  Written only by the evictor thread.
  std::atomic<int> bucket_idx{-1};

  cache_entry(cudf::io::text::byte_range_info logical,
              cudf::io::text::byte_range_info physical)
      : logical_range(logical), physical_range(physical) {}

  /// True iff this entry is safe to evict under the current cache age:
  /// either it has been read at or after its most recent insert (consumed),
  /// or its most recent request was made in an earlier epoch (stale — the
  /// caller has already moved on and no longer needs this range).
  [[nodiscard]] bool
  is_consumed_or_stale(uint64_t cache_age) const noexcept {
    auto req = request_ts.load(std::memory_order_acquire);
    auto cons = consumption_ts.load(std::memory_order_acquire);
    return cons >= req || req < cache_age;
  }
};

// ---------------------------------------------------------------------------
// eviction queue messages
// ---------------------------------------------------------------------------
//
// The evictor reads from two separate queues:
//   - candidate queue: hints posted by unpin() when an entry becomes
//     evictable.  Silent (no wake) — candidates are drained on a periodic
//     poll or whenever the evictor wakes for a request.
//   - request queue: backpressure from the worker asking for free chunks.
//     Each push signals the evictor's semaphore.

struct eviction_candidate {
  std::weak_ptr<cache_entry> entry;
};

struct eviction_request {
  /// Promise resolves when the evictor has freed at least @c n_chunks_needed
  /// back to the buffer pool.  The worker then retries @c pool.allocate_bulk
  /// to grab them.
  std::promise<void> promise;
  size_t n_chunks_needed;
};

// ---------------------------------------------------------------------------
// pinned_view — RAII read guard with per-chunk access
// ---------------------------------------------------------------------------
//
// Acquires a read pin on the cache_entry on construction (via
// entry_state::try_acquire_read — a single atomic CAS that transitions
// cached → in_use with pin_count+1).  Releases on destruction (via
// entry_state::release_read).
//
// Because chunks are scattered in memory there is NO single contiguous
// span.  Instead the view exposes individual chunk spans via operator[].

class pinned_view {
public:
  pinned_view() = default;
  pinned_view(std::shared_ptr<cache_entry> entry,
              moodycamel::ConcurrentQueue<eviction_candidate> &candidate_queue);
  ~pinned_view();

  pinned_view(pinned_view &&o) noexcept;
  pinned_view &operator=(pinned_view &&o) noexcept;

  pinned_view(pinned_view const &) = delete;
  pinned_view &operator=(pinned_view const &) = delete;

  /// Number of 1MB chunks backing this range.
  [[nodiscard]] size_t num_chunks() const noexcept;

  /// Access chunk @p i (physical data). Full CHUNK_BYTES except
  /// possibly the last chunk which may be shorter.
  [[nodiscard]] std::span<const std::byte> operator[](size_t i) const noexcept;

  /// Logical range this view covers.
  [[nodiscard]] cudf::io::text::byte_range_info logical_range() const noexcept;

  /// Physical (O_DIRECT aligned) range.
  [[nodiscard]] cudf::io::text::byte_range_info physical_range() const noexcept;

  /// Logical size (what the user actually requested).
  [[nodiscard]] size_t size() const noexcept;

  /// Slice the cached data at logical [offset, offset+size) into a vector of
  /// non_owning_buffers, one per chunk boundary crossed.  The caller must
  /// ensure [offset, offset+size) lies within the entry's logical range.
  [[nodiscard]] std::vector<cudf::io::datasource::non_owning_buffer>
  slice(size_t offset, size_t size) const;

  explicit operator bool() const noexcept { return _entry != nullptr; }

private:
  void unpin();

  std::shared_ptr<cache_entry> _entry;
  moodycamel::ConcurrentQueue<eviction_candidate> *_candidate_queue{nullptr};
};

// ---------------------------------------------------------------------------
// prefetching_cache
// ---------------------------------------------------------------------------
//
// Locking hierarchy:
//   Level 0: _map_mtx          — protects _file_cache map
//   Level 1: file_entry::mtx   — protects one file's entry vector
//   (independent): cache_entry atomics — lock-free

class prefetching_cache {
public:
  /// @p inflight_budget_chunks caps the number of 1MB chunks the worker
  /// may have submitted to the IO backend at once.  The worker acquires N
  /// tokens before each dispatch and releases them in the completion
  /// callback.  Default ~2 GB worth of in-flight IO.
  ///
  /// @p io_ctx is a non-owning pointer; the owning ioctx must outlive this
  /// cache (the ioctx itself owns the cache via initialize_cache()).
  explicit prefetching_cache(buffer_pool &pool, sirius_ioctx *io_ctx,
                             size_t inflight_budget_chunks = 2048);
  ~prefetching_cache();

  prefetching_cache(prefetching_cache const &) = delete;
  prefetching_cache &operator=(prefetching_cache const &) = delete;

  /// Register ranges for a file and trigger background prefetch.
  /// Ranges must be sorted by offset.
  void insert(const sirius_io_object &obj,
              std::shared_ptr<sirius_io_object_metadata> metadata,
              const std::vector<cudf::io::text::byte_range_info> &ranges);

  /// Non-blocking read of a single range.
  /// Returns an empty pinned_view if the range is not cached or the cached
  /// entry does not fully cover [offset, offset+size).  Updates hit / miss
  /// counters (see summary()).
  [[nodiscard]] pinned_view read(const sirius_io_object &obj, size_t offset,
                                 size_t size);

  /// Non-blocking batch read. Returns one pinned_view per input range
  /// (empty if that range is not yet cached).
  [[nodiscard]] std::vector<pinned_view>
  read_ranges(const sirius_io_object &obj,
              const std::vector<cudf::io::text::byte_range_info> &ranges);

  /// Increment _cache_age by 1.  The evictor uses
  /// (n_total_read_request - _cache_age) to score entries into buckets.
  void refresh_cache();

  /// One-line human-readable state: hit / partial-miss / full-miss counts,
  /// pool utilisation, and pending chunks.
  [[nodiscard]] std::string summary() const;

private:
  // ---- work items dispatched through the queue ------------------------------

  struct prefetch_req {
    std::string file_key;
    const sirius_io_object *io_obj;
    std::vector<std::shared_ptr<cache_entry>> entries;
  };
  using work_item = prefetch_req;

  // ---- per-file state -------------------------------------------------------

  struct file_entry {
    mutable std::shared_mutex mtx; ///< Level-1 lock.
    const sirius_io_object *io_obj{nullptr};
    size_t file_size{0};
    std::shared_ptr<sirius_io_object_metadata> metadata;
    std::vector<std::shared_ptr<cache_entry>> entries; ///< Sorted by offset.
  };

  // ---- helpers --------------------------------------------------------------

  void worker_loop(std::stop_token stop);
  void evictor_loop(std::stop_token stop);
  void enqueue_work(work_item item);

  /// Release all chunks held by an entry back to the pool.
  void release_chunks(cache_entry &entry);

  /// Binary search for an entry whose logical range fully covers
  /// [offset, offset+size).  Returns nullptr on miss.  Updates
  /// _partial_miss_count / _full_miss_count based on the classification.
  /// The caller updates _hit_count on successful pin.
  std::shared_ptr<cache_entry>
  find_entry(const std::vector<std::shared_ptr<cache_entry>> &entries,
             size_t offset, size_t size);

  // ---- members (destruction order matters: worker joined first) -------------

  buffer_pool &_pool;
  sirius_ioctx *_io_ctx;
  std::atomic<size_t> _allocated_bytes{0};
  std::atomic<size_t> _pending_chunks{0};
  /// Monotonic age counter.  Advanced only by refresh_cache() — caller
  /// pulses it to age the cache (non-refreshed entries drift toward bucket 0).
  /// request_ts and consumption_ts on cache_entry snapshot this value.
  std::atomic<int64_t> _cache_age{0};

  // Observability counters (updated on every read() / read_ranges() entry).
  std::atomic<uint64_t> _hit_count{0};
  std::atomic<uint64_t> _partial_miss_count{0};
  std::atomic<uint64_t> _full_miss_count{0};

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  /// Evictor inputs.  Candidate pushes are silent; request pushes bump the
  /// semaphore.  Evictor polls with a timeout to still drain candidates
  /// when no requests arrive.
  moodycamel::ConcurrentQueue<eviction_candidate> _candidate_queue;
  moodycamel::ConcurrentQueue<eviction_request> _request_queue;
  std::counting_semaphore<> _request_sem{0};
  static constexpr auto EVICTOR_POLL_INTERVAL = std::chrono::milliseconds(50);

  /// IO in-flight budget: units == chunks.  Worker acquires a slot sized to
  /// the batch before dispatching IO; the slot is carried into the completion
  /// callback and releases on destruction.
  admission_control _inflight_budget;

  /// Tiered eviction buckets, managed exclusively by the evictor thread.
  /// Bucket 0 = coldest (lowest diff), bucket 4 = hottest.
  static constexpr size_t NUM_LRU_BUCKETS = 5;
  std::array<std::vector<std::weak_ptr<cache_entry>>, NUM_LRU_BUCKETS>
      _lru_buckets;

  moodycamel::ConcurrentQueue<work_item> _work_queue;
  std::atomic<uint64_t> _work_seq{0};

  std::jthread _evictor_thread;
  std::jthread _worker_thread;
};

} // namespace sirius::io
