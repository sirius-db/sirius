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

#include "blockingconcurrentqueue.h"
#include "exec/completion_controller.hpp"
#include "exec/invocable.hpp"
#include "exec/semi_future.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/config.hpp"
#include "io/cache/types.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <concurrentqueue.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sirius::io {
class ioctx;
class sirius_datasource;
}  // namespace sirius::io

namespace sirius::cuda {
class device_copy_batch;
}  // namespace sirius::cuda

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::io::cache {

/// True when every chunk of @p chunks owns a staging buffer, i.e. its
/// @c chunk_state has reached @c allocated and has not been reclaimed since.
/// This is the precondition of the producer's @c prepared stage.  One relaxed
/// load per chunk, so the sweep is cheap enough not to need a summary counter.
[[nodiscard]] inline bool all_chunks_have_buffers(const std::vector<cached_chunk*>& chunks) noexcept
{
  return std::ranges::all_of(chunks, [](const cached_chunk* c) {
    auto const s = c->state.get_state();
    return s >= chunk_state::allocated && s != chunk_state::evicting;
  });
}

/// One prefetch request: the two stage machines plus the chunk set they cover.
/// Held by value — the cache's queues and the owning @ref prefetching_handle
/// each carry a copy, so the stages outlive whichever side finishes first.
struct prefetch_request {
  std::shared_ptr<const io_object> obj;
  std::shared_ptr<producer_stage> producer;
  std::shared_ptr<consumer_stage> consumer;
  std::shared_ptr<const std::vector<cached_chunk*>> chunks;
  std::uint32_t timestamp{0};
  /// Preferred NUMA node for the staging buffers, derived from the requesting
  /// GPU's topology.  -1 means "no preference" (allocate from any arena).
  int preferred_numa{-1};

  /// False for the empty request the queues use as a wakeup sentinel.
  [[nodiscard]] explicit operator bool() const noexcept { return producer != nullptr; }

  /// True once the consumer has queued the scan and has not been disposed.
  [[nodiscard]] bool is_active() const noexcept
  {
    if (!consumer) { return false; }
    auto const s = consumer->get();
    return s >= consumer_stage::queued && s != consumer_stage::disposed;
  }

  /// True once the consumer is gone: no consumer machine, or it is disposed.
  /// This is the disposal check, and it stays meaningful after the IO has been
  /// issued -- which is what the evictor needs to hand a subscriber reference
  /// back, and what a request re-checks once it clears the rate limiter.
  [[nodiscard]] bool is_cancelled() const noexcept
  {
    return !consumer || consumer->get() == consumer_stage::disposed;
  }

  /// True when the readahead lost the race: the consumer has reached
  /// @c preparing or beyond -- so the executor is already pulling this split's
  /// bytes through itself -- while this request's IO has not started.  Issuing
  /// the prefetch now would only duplicate the read that is already happening,
  /// so the gates that decide whether to begin one turn it away here.
  ///
  /// Deliberately false once the producer reaches @c loading: at that point the
  /// IO is in flight and there is nothing left to call off, so the question
  /// stops being "is this worth starting" and becomes "is the consumer still
  /// there" -- which is @ref is_cancelled.
  [[nodiscard]] bool has_fallen_behind() const noexcept
  {
    if (!consumer || !producer) { return true; }
    return consumer->get() >= consumer_stage::preparing &&
           producer->get() < producer_stage::loading;
  }
};

/// A standing demand on the evictor: free at least @p bytes_to_free bytes of
/// staging memory, whether or not the pool has crossed its own pressure
/// threshold.
///
/// The evictor's own trigger is a fraction of what the pool holds, which is the
/// right rule when the cache is the only thing under pressure and the wrong one
/// when it is not: a caller that needs a specific amount back -- because its own
/// allocation just failed, or because it is about to make a large one -- knows a
/// number the cache cannot derive.  This carries that number.
///
/// A demand, not a guarantee: the evictor frees what it can reclaim and does not
/// report back.  Chunks a reader has pinned stay put, so a request for more than
/// is reclaimable simply frees everything reclaimable.
struct eviction_request {
  std::size_t bytes_to_free{0};
};

/// What the evictor's queue carries.  Two things reach it: prefetch requests,
/// handed over at creation so their chunks become eviction candidates once the
/// consumer is done, and explicit demands for memory back.  They are different
/// enough that a single struct would have to encode "which kind am I" in a
/// field, so the queue carries the variant and the loop visits it.
///
/// A default-constructed value holds an empty @ref prefetch_request, which is
/// the queue's wakeup sentinel -- see @ref prefetching_cache::evict_loop.
using cache_request = std::variant<prefetch_request, eviction_request>;

using request_queue_type = duckdb_moodycamel::BlockingConcurrentQueue<cache_request>;

class prefetching_handle {
 public:
  prefetching_handle() noexcept = default;
  /// Marks the consumer disposed so the evictor can reclaim the request.
  ~prefetching_handle();
  prefetching_handle(prefetching_handle const&)            = delete;
  prefetching_handle& operator=(prefetching_handle const&) = delete;

  prefetching_handle(prefetching_handle&& o) noexcept;
  prefetching_handle& operator=(prefetching_handle&& o) noexcept;

  /// Drive the consumer-side stage machine.
  void update(scan_stage stage) noexcept;

  [[nodiscard]] bool is_active() const noexcept;

  /// Producer-side state of this request, for the prefetch census.
  [[nodiscard]] producer_stage::value producer_state() const noexcept;

  /// True while this request's prefetch IO is in flight.  A read that lands now
  /// would duplicate IO the cache is already doing, so callers wait it out via
  /// @ref wait_until_ready instead of issuing their own.
  [[nodiscard]] bool is_prefetch_in_flight() const noexcept;

  /// True once the consumer reached @c scan_stage::reading — the executor is
  /// pulling this split's bytes through itself, so a prefetch started now would
  /// only duplicate that IO.
  [[nodiscard]] bool has_started_reading() const noexcept;

  /// Block until the prefetch IO settles.  Returns true iff it completed
  /// successfully (the chunks are now cache-resident); false if it failed or
  /// the request was abandoned.  No-op and returns false on an empty handle.
  [[nodiscard]] bool wait_until_ready() noexcept;

  /// Block until staging buffers have been allocated for every chunk in this
  /// request (producer state >= prepared).  Returns true iff preparation
  /// succeeded; false means the request was abandoned (e.g. the pool ran out of
  /// memory).  No-op and returns false on an empty handle.
  [[nodiscard]] bool wait_until_prepared() noexcept;

  /// The chunks of the underlying request.  Null when the handle is empty.
  [[nodiscard]] std::shared_ptr<const std::vector<cached_chunk*>> chunks() const noexcept;

  explicit operator bool() const noexcept;

 private:
  friend class prefetching_cache;

  explicit prefetching_handle(prefetch_request req) noexcept;

  prefetch_request _req;
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
  // The cache only accepts new prefetch requests through
  // sirius_datasource::fadvise — that's the single entry point for the
  // fadvise(scan_stage) protocol.  Friending the
  // datasource keeps insert() out of the public API while still letting
  // fadvise dispatch through it.
  friend class sirius::io::sirius_datasource;
  friend class prefetching_handle;

 public:
  using byte_range = cudf::io::text::byte_range_info;

  prefetching_cache(cucascade::memory::memory_reservation_manager& reservation_manager,
                    ioctx* io_ctx,
                    const config& cfg,
                    std::shared_ptr<const sirius::memory::topology_index> topology_index);
  ~prefetching_cache();

  prefetching_cache(prefetching_cache const&)            = delete;
  prefetching_cache& operator=(prefetching_cache const&) = delete;

  [[nodiscard]] bool is_armed() const noexcept { return _armed; }

  [[nodiscard]] std::size_t host_read(const io_object& obj,
                                      size_t offset,
                                      size_t size,
                                      uint8_t* dst,
                                      prefetching_handle* out_handle = nullptr);

  [[nodiscard]] exec::semi_future<std::size_t> host_read_async(
    const io_object& obj,
    size_t offset,
    size_t size,
    uint8_t* dst,
    prefetching_handle* out_handle = nullptr);

  [[nodiscard]] exec::semi_future<std::size_t> device_read_async(
    const io_object& obj,
    size_t offset,
    size_t size,
    uint8_t* device_ptr,
    rmm::cuda_stream_view stream,
    prefetching_handle* out_handle = nullptr);

  /// Vectored form of @ref device_read_async: each range is served from the
  /// cache where it is populated, loaded through the cache where it can be, and
  /// bounced through the backend otherwise — all in one dispatch.  Requires a
  /// backend with one of the two batch capabilities (both reactors have them);
  /// on any other it returns the backend's failed future without touching the
  /// cache.  Reports the bytes delivered, each range clamped to the object's
  /// end, resolving once the copies are enqueued on @p stream.
  [[nodiscard]] exec::semi_future<std::size_t> device_read_ranges_async(
    const io_object& obj,
    std::span<const slice> slices,
    rmm::cuda_stream_view stream,
    prefetching_handle* out_handle = nullptr);

  /// Vectored form of @ref device_read_async: each range is served from the
  /// cache where it is populated, loaded through the cache where it can be, and
  /// bounced through the backend otherwise — all in one dispatch.  Requires a
  /// backend with one of the two batch capabilities (both reactors have them);
  /// on any other it returns the backend's failed future without touching the
  /// cache.  Reports the bytes delivered, each range clamped to the object's
  /// end, resolving once the copies are enqueued on @p stream.
  [[nodiscard]] exec::semi_future<std::size_t> host_read_ranges_async(
    const io_object& obj, std::span<const slice> slices, prefetching_handle* out_handle = nullptr);

  /// Issue prefetch IO for @p handle's request.  @p on_done fires exactly once
  /// with the outcome — inline when no IO is issued, otherwise from the IO
  /// completion.  Returns whether IO was issued.
  bool prefetch(prefetching_handle& handle, exec::invocable<void(bool) noexcept> on_done);

  /// Bytes of staging memory the cache currently holds: every chunk buffer
  /// handed out by the pool and not yet reclaimed.  This is what an explicit
  /// @ref evict can act on -- the ceiling on how much it could ever free, and
  /// the number to re-read afterwards to see how much it did.
  ///
  /// A relaxed read of a counter other threads are moving, so it is a snapshot
  /// rather than a value to compute an exact target from.
  [[nodiscard]] std::size_t claimed_bytes() const noexcept;

  /// Ask the evictor to free at least @p bytes_to_free bytes of staging memory.
  ///
  /// Asynchronous and best-effort: this enqueues the demand and returns.  The
  /// evictor gets to it on its next round and frees what it can -- chunks a
  /// reader has pinned are not reclaimable, so a demand larger than what is
  /// reclaimable frees everything reclaimable and no more.  Poll
  /// @ref claimed_bytes to see the result.
  ///
  /// The point of it is that memory pressure is not always the cache's own: the
  /// evictor's built-in trigger fires on the pool's occupancy, which says
  /// nothing about a GPU allocation failing elsewhere.  This is how something
  /// that knows it needs host memory back says so.
  ///
  /// A zero request is a no-op, as is one on a cache that is not armed or is
  /// already shutting down.
  void evict(std::size_t bytes_to_free);

  [[nodiscard]] std::string summary() const;

  void prepare_for_query() noexcept;

  [[nodiscard]] uint32_t query_epoch() const noexcept
  {
    return _ticker.load(std::memory_order_relaxed);
  }

 private:
  struct cached_copy_retirement;

  [[nodiscard]] prefetching_handle initiate_prefetching_request(const io_object& obj,
                                                                std::span<const byte_range> ranges,
                                                                std::optional<int> gpu_id = {});

  /// Attach staging buffers to @p handle's request, so a following @ref prefetch
  /// has chunks it can claim: a chunk without a buffer cannot be taken for
  /// loading, and a prefetch over one issues no IO and settles `ready` having
  /// done nothing.
  ///
  /// @p wait_for_eviction lets the call wait for the evictor to hand chunks back
  /// rather than fail on a momentarily empty pool.  Callers that can afford to
  /// stall -- the readahead, whose whole job is to be ahead -- should; callers
  /// on a read path should not.
  ///
  /// Private, and reached only through @c sirius_datasource (which the readahead
  /// drives via @c scan_info::prepare_for_prefetching): preparing is part of the
  /// fadvise-owned request lifecycle, not something an arbitrary caller starts.
  ///
  /// @return false when the request could not be prepared -- the pool could not
  ///         satisfy it, the consumer has already moved past it, or somebody
  ///         else has already taken it past @c queued.  Only the first two leave
  ///         it @c abandoned.
  bool prepare(prefetching_handle& handle, bool wait_for_eviction);

  [[nodiscard]] bool prepare_request(prefetch_request& req, bool wait_for_eviction = false);

  /// Resolve cache positions handle-first, then against the file-wide entry.
  /// A partial lookup preserves every materialised position so the read planner
  /// can mix resident, loadable, and backend-staged pieces in one dispatch.
  [[nodiscard]] std::vector<cached_chunk*> ranges_in_cache(const io_object& obj,
                                                           size_t offset,
                                                           size_t size,
                                                           coverage_policy policy,
                                                           prefetching_handle* out_handle) const;

  struct file_entry {
    /// Materialise a chunk for every offset in @p incoming, fold the matching
    /// entry of @p desired into its populated extent, and count the calling
    /// request as a subscriber of each.  Returns the chunks in @p incoming
    /// order.  @p desired runs index-parallel to @p incoming.
    std::vector<cached_chunk*> update_and_get_chunks(std::span<const size_t> incoming,
                                                     std::span<const chunk_fill> desired);

    std::vector<cached_chunk*> fetch_chunks(std::size_t offset,
                                            std::size_t size,
                                            coverage_policy policy) const;

    /// Slot index covering byte @p off.  Callers must bounds-check against
    /// @c slots.size() — a read past EOF has no slot.
    [[nodiscard]] std::size_t slot_of(std::size_t off) const noexcept { return off / chunk_size; }

    mutable std::shared_mutex mtx;
    std::shared_ptr<const io_object> io_obj;
    /// Direct-mapped: slot i covers [i * chunk_size, (i + 1) * chunk_size);
    /// nullptr means that chunk has never been materialised.  Sized once at
    /// creation — exactly the capacity the old sorted vector reserved — so a
    /// lookup is an index instead of a search.
    std::vector<cached_chunk*> slots;
    chunk_arena arena;  ///< owns the chunks `slots` points at; append-only
    size_t file_size{0};
    size_t chunk_size{1};
  };

  /// Enqueue @p copies and keep their cache-hit pins alive until @p stream has
  /// passed them. The preallocated retirement also settles the cache-copy
  /// credit in the caller's grouped coordinator, so even an all-hit read's
  /// future waits for the asynchronous H2D work.
  void retire_pins_after_stream(rmm::cuda_stream_view stream,
                                sirius::cuda::device_copy_batch const& copies,
                                std::shared_ptr<cached_copy_retirement> retirement) noexcept;

  /// Pop every request still sitting in @p queue and retire its producer on
  /// @c producer_stage::abandoned, so a shutdown never leaves a request parked
  /// in a transient stage with waiters asleep on it.
  static void drain_and_abandon(request_queue_type& queue) noexcept;

  void evict_loop(const std::stop_token& st);

  file_entry& get_or_create_file_entry(const io_object& obj);

  const config _cfg;
  std::unique_ptr<buffer_pool> _pool;
  size_t _chunk_size = 1;

  ioctx* const _io_ctx;

  // Hardware GPU/NUMA topology index, shared from the scan_manager.  Used to
  // place prefetch staging buffers on the NUMA node closest to the target GPU.
  std::shared_ptr<const sirius::memory::topology_index> const _topology_index;

  bool const _armed;

  std::atomic<bool> _shutting_down{false};

  std::atomic<uint32_t> _ticker{0};  // see prefetch_stats::snapshot for layout

  // ---- Telemetry counters --------------------------------------------------
  // Global cumulative counts.  @c _last_reported snapshots them on every cache
  // refresh (prepare_for_query) so summary() can also report per-cycle deltas.
  struct counters {
    std::atomic<uint64_t> n_reads{0};    // device read requests served by the cache
    std::atomic<uint64_t> hits{0};       // chunks served from cache (read pin acquired, in_use)
    std::atomic<uint64_t> h2d{0};        // chunks (re)loaded via host->device IO (mark_loading)
    std::atomic<uint64_t> misses{0};     // chunks read fresh (missing / not yet usable)
    std::atomic<uint64_t> evictions{0};  // chunks evicted back to the pool
  };
  struct counters_snapshot {
    uint64_t n_reads{0};
    uint64_t hits{0};
    uint64_t h2d{0};
    uint64_t misses{0};
    uint64_t evictions{0};
  };

  counters _counters;
  counters_snapshot _last_reported;

  /// One slot per issued cache-backed IO, retained by its physical completion
  /// callback until no reactor can touch cache-owned chunk state. This is NOT
  /// a rate limit -- @c acquire
  /// never blocks, and how much read-ahead is in flight is the readahead
  /// manager's budget to set.  It exists because the completion writes through
  /// raw @c cached_chunk pointers into file entries this cache owns, so the
  /// destructor has to wait those completions out before letting them go.
  exec::completion_controller _inflight_io;
  std::mutex _inflight_io_mtx;

  [[nodiscard]] exec::completion_controller::slot acquire_inflight_io() noexcept;

  /// Block until every issued prefetch IO has run its completion.  Closes
  /// @ref _inflight_io, so it is a teardown step and not repeatable.
  void drain_inflight_io() noexcept;

  std::jthread _evictor_thread;
  request_queue_type _eviction_queue;
  std::stop_source _evictor_stop_source;
  bool _dispose_on_idle{false};

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  /// CUDA completion waits run here, never inside a CUDA stream callback. Each
  /// task is fully scheduled before its copies are published and retains an
  /// inflight cache slot through coordinator settlement.
  exec::static_thread_pool _io_cb_thread_pool{2, "io_cb"};
};

}  // namespace sirius::io::cache
