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
#include "cuda/device_copy_batch.hpp"
#include "exec/completion_controller.hpp"
#include "exec/invocable.hpp"
#include "exec/scoped_dispatcher.hpp"
#include "exec/semi_future.hpp"
#include "exec/stream_ordered_retirer.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/config.hpp"
#include "io/cache/types.hpp"

#include <concurrentqueue.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class ioctx;
class sirius_datasource;
}  // namespace sirius::io

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

using request_queue_type = duckdb_moodycamel::BlockingConcurrentQueue<prefetch_request>;

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

  /// Block until the prepare_loop has allocated staging buffers for every
  /// chunk in this request (producer state >= prepared).  Returns true iff
  /// preparation succeeded; false means the request was abandoned (e.g. the
  /// pool ran out of memory).  No-op and returns false on an empty handle.
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
    std::span<const io::io_device_range> ranges,
    rmm::cuda_stream_view stream,
    prefetching_handle* out_handle = nullptr);

  /// Issue prefetch IO for @p handle's request.  @p on_done fires exactly once
  /// with the outcome — inline when no IO is issued, otherwise from the IO
  /// completion.  Returns whether IO was issued.
  bool prefetch(prefetching_handle& handle, exec::invocable<void(bool) noexcept> on_done);

  [[nodiscard]] std::string summary() const;

  void prepare_for_query() noexcept;

  [[nodiscard]] uint32_t query_epoch() const noexcept
  {
    return _ticker.load(std::memory_order_relaxed);
  }

 private:
  [[nodiscard]] prefetching_handle insert(const io_object& obj,
                                          std::span<const byte_range> ranges,
                                          std::optional<int> gpu_id = {});

  [[nodiscard]] bool host_read_from_cache_only(
    const io_object& obj, size_t offset, size_t size, uint8_t* dst, prefetching_handle* out_handle);

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

  /// Pin every chunk of @p chunks over the part of @p range it covers and stage
  /// the copy to the range's device destination into @p copies, appending the
  /// pins to @p pinned — which the caller must release only after draining the
  /// stream the batch is issued on, since a released pin makes its chunk
  /// evictable mid-copy.  Returns false if the range went unserved: a chunk
  /// populated too little releases its own pins and stages nothing.
  ///
  /// Nothing reaches the driver here.  The caller owns @p copies across every
  /// range of the batch and issues them in one @c cudaMemcpyBatchAsync, so a
  /// scan's worth of column chunks costs one driver round-trip, not one each.
  [[nodiscard]] bool copy_range_from_cache(std::span<cached_chunk* const> chunks,
                                           const io::io_device_range& range,
                                           std::vector<cached_chunk*>& pinned,
                                           sirius::cuda::device_copy_batch& copies);

  /// How a batch of ranges resolves against the cache: chunks pinned for a
  /// direct copy, chunks claimed for loading, the staged copies for everything
  /// already resident, and the IO ranges covering everything that still has to
  /// be read.  Both chunk lists stay held until the caller retires them once the
  /// IO and the copies have drained.
  struct device_read_plan {
    std::vector<cached_chunk*> pinned;
    std::vector<cached_chunk*> loading;
    std::vector<io::io_host_device_range> io_ranges;
    /// Cache-resident copies for the whole batch, issued once by the caller.
    sirius::cuda::device_copy_batch copies;
    std::size_t served{0};
    std::size_t hits{0};
    std::size_t h2d{0};
    std::size_t misses{0};
  };

  /// Classify every chunk position of @p range into @p plan: stage what is
  /// already populated into @c plan.copies, claim what can be loaded and read it
  /// through the chunk's own buffer, and bounce the rest through the backend.
  void plan_device_range(std::span<cached_chunk* const> chunks,
                         const io::io_device_range& range,
                         device_read_plan& plan);

  /// Apply a device read's outcome to its chunks once @p stream has executed
  /// the copies: drop the read pins in @p pinned and publish (or fail) the
  /// chunks in @p loading.
  ///
  /// Deferred through @ref _retirer rather than waited on.  Both lists must
  /// stay untouched until the copies have drained — a released pin makes its
  /// chunk evictable, and its staging buffer reusable, while the copy engines
  /// are still reading it.
  ///
  /// @p host_ok — the host-side result of the read — is what decides whether
  /// the chunks in @p loading are published or failed.  `cached` is a claim
  /// about the pinned staging buffer, and only the host read can establish it;
  /// the stream's completion status describes the copy out of that buffer into
  /// the caller's device memory, which cannot invalidate its source.  A device
  /// fault is therefore reported, not folded into the chunk state.
  void retire_after_stream(rmm::cuda_stream_view stream,
                           std::vector<cached_chunk*>&& pinned,
                           std::vector<cached_chunk*>&& loading,
                           bool host_ok) noexcept;

  /// Pop every request still sitting in @p queue and retire its producer on
  /// @c producer_stage::abandoned, so a shutdown never leaves a request parked
  /// in a transient stage with waiters asleep on it.
  static void drain_and_abandon(request_queue_type& queue) noexcept;

  void prepare_loop(const std::stop_token& st);
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

  std::jthread _preparation_thread;
  request_queue_type _preparation_queue;
  std::stop_source _preparation_stop_source;

  /// One slot per issued prefetch IO, parked in that IO's completion so it
  /// drops when the completion runs.  This is NOT a rate limit -- @c acquire
  /// never blocks, and how much read-ahead is in flight is the readahead
  /// manager's budget to set.  It exists because the completion writes through
  /// raw @c cached_chunk pointers into file entries this cache owns, so the
  /// destructor has to wait those completions out before letting them go.
  exec::completion_controller _inflight_io;

  /// Block until every issued prefetch IO has run its completion.  Closes
  /// @ref _inflight_io, so it is a teardown step and not repeatable.
  void drain_inflight_io() noexcept;

  std::jthread _evictor_thread;
  request_queue_type _eviction_queue;
  std::stop_source _evictor_stop_source;
  bool _dispose_on_idle{false};

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  exec::static_thread_pool _io_cb_thread_pool{
    2, "io_cb"};  // single-threaded pool for IO completion callbacks
  exec::scoped_dispatcher _io_cb_dispatcher{_io_cb_thread_pool, 2};

  /// Retires each device read's chunk-state transitions when its stream's
  /// completion frontier passes the copies, instead of parking a thread on a
  /// per-read CUDA event.  Drained opportunistically by whoever is about to
  /// need a staging buffer (prepare_loop, evict_loop) or a cache hit (the read
  /// paths); nothing polls it in steady state.
  ///
  /// Declared last so it is destroyed first — its lanes hand staging buffers
  /// back to @ref _pool, which must still be alive.  @c ~prefetching_cache
  /// quiesces it explicitly as well, so the order is not load-bearing.
  exec::stream_ordered_retirer _retirer;
};

}  // namespace sirius::io::cache
