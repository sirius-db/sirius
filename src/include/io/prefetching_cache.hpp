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

#include "io/admission_control.hpp"
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>

#include <cuda_runtime.h>

#include <concurrentqueue.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <array>
#include <atomic>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <semaphore>
#include <shared_mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {

class sirius_ioctx;

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

  buffer_pool(cucascade::memory::fixed_size_host_memory_resource& mr, uint32_t max_slabs);
  ~buffer_pool();

  buffer_pool(buffer_pool const&)            = delete;
  buffer_pool& operator=(buffer_pool const&) = delete;

  /// Allocate a single chunk.  Returns nullptr when the pool is exhausted
  /// and the upstream resource cannot supply a fresh slab.
  std::byte* allocate();

  /// Bulk-allocate up to @p n chunks, appending pointers to @p out.
  /// Returns the number actually allocated (may be < n if the pool is
  /// exhausted and cannot grow).
  size_t allocate_bulk(size_t n, std::vector<std::byte*>& out);

  void deallocate_bulk(std::vector<std::byte*>& out);

  /// Return a chunk to the pool.
  void deallocate(std::byte* p);

  [[nodiscard]] size_t chunk_bytes() const noexcept { return _chunk_bytes; }
  [[nodiscard]] size_t slab_bytes() const noexcept
  {
    return static_cast<size_t>(CHUNKS_PER_SLAB) * _chunk_bytes;
  }
  [[nodiscard]] size_t capacity() const noexcept
  {
    return static_cast<size_t>(_total_chunks.load(std::memory_order_relaxed)) * _chunk_bytes;
  }
  [[nodiscard]] uint32_t free_count() const noexcept
  {
    return _total_free.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint32_t total_chunks() const noexcept
  {
    return _total_chunks.load(std::memory_order_relaxed);
  }

 private:
  /// Pull one slab worth of blocks from the upstream resource and append
  /// them to @c _free_list.  Caller must hold @c _mtx.
  bool grow_locked();

  cucascade::memory::fixed_size_host_memory_resource& _mr;
  size_t _chunk_bytes;
  uint32_t _max_slabs;

  // Protects _allocations and _free_list.
  std::mutex _mtx;
  // Held to keep upstream blocks alive for the lifetime of the pool —
  // the multiple_blocks_allocation destructor is what returns blocks to
  // the resource, so we never drop these until the pool is destroyed.
  std::vector<cucascade::memory::fixed_multiple_blocks_allocation> _allocations;
  std::vector<std::byte*> _free_list;

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
//     │ mark_evicted()                          mark_cached() │
//     mark_load_failed() │                                                   │
//     │ │   ┌───────────────────────────────────────────────▼             ▼
//  evicting │                                          cached ◄────── empty
//     ▲    │                                         ▲      │
//     │    │ release_read                            │      │
//     try_acquire_read() │    │ (last reader)                           │ ▼ │
//     │                                         │  in_use (pin_count ≥ 1) │ │ │
//     └────┘ try_start_evicting()  (only from cached, pin_count==0)

class entry_state {
 public:
  enum value : uint8_t { empty = 0, queued = 1, loading = 2, cached = 3, in_use = 4, evicting = 5 };

  entry_state() noexcept = default;

  [[nodiscard]] value get_state() const noexcept
  {
    return unpack_state(_packed.load(std::memory_order_acquire));
  }

  [[nodiscard]] uint32_t get_pin_count() const noexcept
  {
    return unpack_pins(_packed.load(std::memory_order_acquire));
  }

  /// empty → queued.  Returns false if not empty.
  /// Called by insert() to claim responsibility for scheduling a load.
  bool try_start_queueing() noexcept
  {
    auto expected = pack(empty, 0);
    return _packed.compare_exchange_strong(expected, pack(queued, 0), std::memory_order_acq_rel);
  }

  /// (empty | queued) → loading.  Returns false if the entry is already
  /// loading / cached / in_use / evicting (no work to do).
  /// Called by the worker when it picks up an entry in a work_item.
  bool try_start_loading() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != empty && st != queued) return false;
      auto next = pack(loading, 0);
      if (_packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire))
        return true;
    }
  }

  /// queued → empty.  Returns false if not queued.
  /// Used by refresh_cache to drop stale pending prefetches and by the
  /// worker's abort path to release orphaned entries.
  bool try_cancel_queued() noexcept
  {
    auto expected = pack(queued, 0);
    return _packed.compare_exchange_strong(expected, pack(empty, 0), std::memory_order_acq_rel);
  }

  /// loading → cached.  Caller must ensure state is loading.
  /// Wakes any readers parked in @c wait_while_loading().
  void mark_cached() noexcept
  {
    _packed.store(pack(cached, 0), std::memory_order_release);
    _packed.notify_all();
  }

  /// loading → empty.  IO failed, chunks already freed by caller.
  /// Wakes any readers parked in @c wait_while_loading().
  void mark_load_failed() noexcept
  {
    _packed.store(pack(empty, 0), std::memory_order_release);
    _packed.notify_all();
  }

  /// Block while state == loading.  Returns when the state transitions
  /// out of loading (either cached on success or empty on failure).
  void wait_while_loading() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (unpack_state(cur) == loading) {
      _packed.wait(cur, std::memory_order_relaxed);
      cur = _packed.load(std::memory_order_acquire);
    }
  }

  /// (cached | in_use) → in_use with pin_count+1.
  /// Returns false if the entry is not in a readable state.
  bool try_acquire_read() noexcept
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

  /// cached (pin_count==0) → evicting.
  /// Returns false if state != cached or readers are present.
  bool try_start_evicting() noexcept
  {
    auto expected = pack(cached, 0);
    return _packed.compare_exchange_strong(expected, pack(evicting, 0), std::memory_order_acq_rel);
  }

  /// evicting → empty.  Caller must ensure state is evicting.
  void mark_evicted() noexcept { _packed.store(pack(empty, 0), std::memory_order_release); }

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

// ---------------------------------------------------------------------------
// cache_entry — per-range metadata
// ---------------------------------------------------------------------------
//
// State transitions are managed by the entry_state class above.
// See entry_state's state machine diagram for the full picture.

struct alignas(64) cache_entry {
  cudf::io::text::byte_range_info logical_range;
  cudf::io::text::byte_range_info physical_range;

  /// Size of each chunk in @c chunks (== buffer_pool::chunk_bytes() at the
  /// time of entry creation).  Stored on the entry so pinned_view doesn't
  /// need a pool reference to do chunk-index arithmetic.
  size_t chunk_bytes{0};

  /// Pointers to pinned chunks (each of size @c chunk_bytes) from buffer_pool
  /// backing this range.
  std::vector<std::byte*> chunks;

  /// Packed state + pin_count.  All state transitions go through this.
  entry_state state;

  /// Cumulative number of insert() calls that included this range.
  /// Monotonically increasing — never decremented.  Used for historical
  /// demand signals (LRU bucket scoring etc.).
  std::atomic<int64_t> n_total_request{0};

  /// Epoch-scoped outstanding-read counter.  Bumped per insert that lands
  /// in the current cache_age epoch (reset to 1 on epoch change).
  /// Decremented on every read terminal (hit or miss).  The worker skips
  /// a work_item entry when n_request &lt;= 0 — all expected readers have
  /// already resolved, so prefetch would be wasted.
  std::atomic<int> n_request{0};

  /// _cache_age at the most recent insert of this range.
  std::atomic<uint64_t> request_ts{0};

  /// _cache_age at the most recent successful read (pinned_view
  /// construction).  0 by default — never-consumed entries have
  /// consumption_ts < request_ts.
  std::atomic<uint64_t> consumption_ts{0};

  /// Set to true while this entry sits in the evictor's LRU list, cleared
  /// when the evictor pops it out (eviction or otherwise).  Only mutated
  /// by the evictor thread, so no synchronisation is required.
  bool in_lru{false};

  cache_entry(cudf::io::text::byte_range_info logical,
              cudf::io::text::byte_range_info physical,
              size_t chunk_bytes)
    : logical_range(logical), physical_range(physical), chunk_bytes(chunk_bytes)
  {
  }

  cache_entry(cache_entry const&)            = delete;
  cache_entry& operator=(cache_entry const&) = delete;

  /// True iff this entry is safe to evict under the current cache age:
  /// either it has been read at or after its most recent insert (consumed),
  /// or its most recent request was made in an earlier epoch (stale — the
  /// caller has already moved on and no longer needs this range).
  [[nodiscard]] bool is_consumed_or_stale(uint64_t cache_age) const noexcept
  {
    auto req  = request_ts.load(std::memory_order_acquire);
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
  /// @p stream is the caller's CUDA stream.  When non-null, the read pin
  /// is released via a host callback enqueued on this stream, so the entry
  /// stays in_use (and therefore non-evictable) until any cudaMemcpyAsync
  /// the caller submitted earlier has finished consuming the chunks.  When
  /// null, the read is released synchronously on destruction (host-only
  /// reads have no async work to wait on).
  pinned_view(std::shared_ptr<cache_entry> entry,
              duckdb_moodycamel::ConcurrentQueue<eviction_candidate>& candidate_queue,
              cudaStream_t stream);
  ~pinned_view();

  pinned_view(pinned_view&& o) noexcept;
  pinned_view& operator=(pinned_view&& o) noexcept;

  pinned_view(pinned_view const&)            = delete;
  pinned_view& operator=(pinned_view const&) = delete;

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
  [[nodiscard]] std::vector<cudf::io::datasource::non_owning_buffer> slice(size_t offset,
                                                                           size_t size) const;

  explicit operator bool() const noexcept { return _entry != nullptr; }

 private:
  void unpin();

  std::shared_ptr<cache_entry> _entry;
  duckdb_moodycamel::ConcurrentQueue<eviction_candidate>* _candidate_queue{nullptr};
  cudaStream_t _stream{nullptr};
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
  explicit prefetching_cache(buffer_pool& pool,
                             sirius_ioctx* io_ctx,
                             size_t inflight_budget_chunks = 2048);
  ~prefetching_cache();

  prefetching_cache(prefetching_cache const&)            = delete;
  prefetching_cache& operator=(prefetching_cache const&) = delete;

  /// Register ranges for a file and trigger background prefetch.
  /// Ranges must be sorted by offset.  The cache retains @p obj via
  /// @c shared_from_this() so the underlying file handles stay open
  /// until all pending prefetch work for this file has completed.
  /// @p obj must already be owned by a @c std::shared_ptr.
  void insert(sirius_io_object& obj,
              std::shared_ptr<sirius_io_object_metadata> metadata,
              const std::vector<cudf::io::text::byte_range_info>& ranges);

  /// Non-blocking read of a single range.
  /// Returns an empty pinned_view if the range is not cached or the cached
  /// entry does not fully cover [offset, offset+size).  Updates hit / miss
  /// counters (see summary()).  @p stream, when non-null, defers the
  /// returned pinned_view's read release until the stream reaches the
  /// release point — see @c pinned_view's ctor for the full contract.
  [[nodiscard]] pinned_view read(const sirius_io_object& obj,
                                 size_t offset,
                                 size_t size,
                                 cudaStream_t stream);

  /// Non-blocking batch read. Returns one pinned_view per input range
  /// (empty if that range is not yet cached).
  [[nodiscard]] std::vector<pinned_view> read_ranges(
    const sirius_io_object& obj,
    const std::vector<cudf::io::text::byte_range_info>& ranges,
    cudaStream_t stream);

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
    /// shared_ptr keeps the io_object (and thus its file handles) alive
    /// until the worker has issued IO for this request.
    std::shared_ptr<sirius_io_object> io_obj;
    std::vector<std::shared_ptr<cache_entry>> entries;
  };
  using work_item = prefetch_req;

  // ---- per-file state -------------------------------------------------------

  struct file_entry {
    mutable std::shared_mutex mtx;  ///< Level-1 lock.
    /// shared_ptr so the file stays open as long as this cache keeps
    /// entries for it — outliving the caller's datasource if necessary.
    std::shared_ptr<sirius_io_object> io_obj;
    size_t file_size{0};
    std::shared_ptr<sirius_io_object_metadata> metadata;
    std::vector<std::shared_ptr<cache_entry>> entries;  ///< Sorted by offset.
  };

  // ---- helpers --------------------------------------------------------------

  void worker_loop(std::stop_token stop);
  void evictor_loop(std::stop_token stop);
  void enqueue_work(work_item item);

  /// Release all chunks held by an entry back to the pool.
  void release_chunks(cache_entry& entry);

  // ---- LRU helpers (evictor-thread only; no locking) ----------------------

  /// Bucket an entry would land in given the current age:
  ///   clamp(4 + n_total_request - age, 0, N_LRU_BUCKETS-1)
  static int lru_bucket_for(cache_entry const& entry, int64_t age) noexcept;

  /// Insert @p entry at the tail of bucket @p bucket in @c _lru_list,
  /// patching separators so that buckets below @p bucket don't claim
  /// the newly inserted element.  Returns the new list iterator.
  std::list<cache_entry*>::iterator lru_insert(cache_entry* entry, int bucket);

  /// Erase the element at @p it from @c _lru_list, fixing any separator
  /// that pointed at @p it.  Returns the iterator following @p it.
  std::list<cache_entry*>::iterator lru_erase(std::list<cache_entry*>::iterator it);

  /// Shift bucket separators one notch: bucket @c i absorbs bucket @c i+1,
  /// and the topmost real separator drops to the very last element so
  /// bucket 4 shrinks to a single entry.  Safe to call repeatedly.
  void age_lru_buckets();

  /// Pull pending candidates off @c _candidate_queue and file each into
  /// its intended LRU bucket.  Entries already in @c _lru_list (in_lru
  /// == true) are skipped.
  void drain_candidates_into_lru();

  /// Binary search for an entry whose logical range fully covers
  /// [offset, offset+size).  Returns nullptr on miss.  Updates
  /// _partial_miss_count / _full_miss_count based on the classification.
  /// The caller updates _hit_count on successful pin.
  std::shared_ptr<cache_entry> find_entry(const std::vector<std::shared_ptr<cache_entry>>& entries,
                                          size_t offset,
                                          size_t size);

  // ---- members (destruction order matters: worker joined first) -------------

  buffer_pool& _pool;
  sirius_ioctx* _io_ctx;
  /// Monotonic age counter.  Advanced only by refresh_cache() — caller
  /// pulses it to age the cache (non-refreshed entries drift toward bucket 0).
  /// request_ts and consumption_ts on cache_entry snapshot this value.
  // Starts at 1 (not 0) so the default cache_entry::consumption_ts (0) is
  // strictly less than any real cache_age.  Lets the worker distinguish
  // "reader never touched" (cons == 0) from "reader resolved in this epoch"
  // (cons == _cache_age).
  std::atomic<int64_t> _cache_age{1};

  // Observability counters (updated on every read() / read_ranges() entry).
  std::atomic<uint64_t> _hit_count{0};
  std::atomic<uint64_t> _partial_miss_count{0};
  std::atomic<uint64_t> _full_miss_count{0};

  // Hits that had to wait on a loading entry (subset of _hit_count).
  std::atomic<uint64_t> _hit_after_wait{0};

  // Worker skipped an entry because the reader already resolved it in this
  // epoch (consumption_ts == _cache_age on a not-yet-cached entry).
  std::atomic<uint64_t> _worker_skipped_reader_resolved{0};

  // Cumulative evictions since construction: entries whose chunks were
  // released back to the pool (+ total chunks freed).
  std::atomic<uint64_t> _evicted_entries{0};
  std::atomic<uint64_t> _evicted_chunks{0};

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  /// Evictor inputs.  Candidate pushes are silent; request pushes bump the
  /// semaphore.  Evictor polls with a timeout to still drain candidates
  /// when no requests arrive.
  duckdb_moodycamel::ConcurrentQueue<eviction_candidate> _candidate_queue;
  duckdb_moodycamel::ConcurrentQueue<eviction_request> _request_queue;
  std::counting_semaphore<> _request_sem{0};
  static constexpr auto EVICTOR_POLL_INTERVAL = std::chrono::milliseconds(50);

  /// IO in-flight budget: units == chunks.  Worker acquires a slot sized to
  /// the batch before dispatching IO; the slot is carried into the completion
  /// callback and releases on destruction.
  admission_control _inflight_budget;

  // ---- LRU-with-buckets eviction list (evictor-thread only) ---------------
  //
  // The evictor drains the candidate queue into @c _lru_list, bucketing
  // entries by freshness: @c _lru_buckets[i] is the iterator one-past-the-
  // end of bucket @c i (so bucket @c i covers [bucket[i-1], bucket[i]), with
  // bucket 0 starting at @c _lru_list.begin()).  @c _lru_buckets[4] is kept
  // equal to @c _lru_list.end().
  //
  // Only the evictor thread touches these fields, so no locking is needed.
  static constexpr size_t N_LRU_BUCKETS = 5;
  std::list<cache_entry*> _lru_list;
  std::array<std::list<cache_entry*>::iterator, N_LRU_BUCKETS> _lru_buckets;
  /// Last @c _cache_age seen by the evictor.  When the live age has
  /// advanced past this, the evictor shifts bucket separators to age the
  /// LRU before the next eviction pass.
  int64_t _last_seen_age{0};

  duckdb_moodycamel::ConcurrentQueue<work_item> _work_queue;
  std::atomic<uint64_t> _work_seq{0};

  std::jthread _evictor_thread;
  std::jthread _worker_thread;
};

}  // namespace sirius::io
