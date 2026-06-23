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

#include "io/cache/prefetching_cache.hpp"

#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/cache/types.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"
#include "memory/topology_index.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

namespace sirius::io::cache {

namespace {

using chunk_iter = std::vector<std::unique_ptr<cached_chunk>>::iterator;

// Locates the first chunk with offset >= `off` within [first, last) using a
// galloping (exponential) search seeded at `first`.
//
// The incoming offsets are sorted and clustered — successive queries are
// usually exactly one chunk apart — so we probe `first`, then `first+1`,
// `first+2`, `first+4`, ... until we overshoot, then binary-search the bounded
// window.  This collapses the common neighboring case to O(1) and keeps gaps at
// O(log gap) rather than O(log N) over the whole (20-40K-element) vector that a
// plain lower_bound(first, last, ...) would incur on every single offset.
chunk_iter gallop_lower_bound(chunk_iter first, chunk_iter last, size_t off)
{
  auto cmp = [](const std::unique_ptr<cached_chunk>& c, size_t v) { return c->offset < v; };

  if (first == last || (*first)->offset >= off) { return first; }

  auto probe  = first;  // invariant: (*probe)->offset < off
  size_t step = 1;
  while (true) {
    const auto remaining = static_cast<size_t>(last - probe);
    if (step >= remaining) {
      // `off` lies in (probe, last): bounded binary search of the tail.
      return std::lower_bound(probe + 1, last, off, cmp);
    }
    const auto hi = probe + static_cast<std::ptrdiff_t>(step);
    if ((*hi)->offset < off) {
      probe = hi;
      step <<= 1U;
    } else {
      // `off` lies in (probe, hi]: binary search that bounded window.
      return std::lower_bound(probe + 1, hi + 1, off, cmp);
    }
  }
}

}  // namespace

class prefetching_handle::prefetch_lifecycle_manager {
 public:
  explicit prefetch_lifecycle_manager(
    prefetching_cache::prefetch_request ctx,
    prefetching_cache::request_queue_type& eviction_queue,
    prefetching_cache::request_queue_type& prefetch_queue) noexcept
    : _eviction_queue(eviction_queue), _prefetch_queue(prefetch_queue)
  {
    if (ctx) {
      _user_state        = ctx->user_state;
      _prefetching_state = ctx->state;
      _ctx               = std::move(ctx);
    }
  }

  ~prefetch_lifecycle_manager() noexcept { evict(); }

  void activate() noexcept
  {
    prefetching_handle_state expected = prefetching_handle_state::idle;
    if (_user_state->compare_exchange_strong(expected, prefetching_handle_state::active)) {
      auto ctx = _ctx.lock();
      if (ctx and _prefetching_state->mark_loading()) { _prefetch_queue.enqueue(std::move(ctx)); }
    }
  }

  void cancel() noexcept { _user_state->store(prefetching_handle_state::cancelled); }

  [[nodiscard]] bool is_active() const noexcept
  {
    return _user_state->load(std::memory_order_acquire) == prefetching_handle_state::active;
  }

  [[nodiscard]] std::shared_ptr<prefetch_request_context> get_context() const noexcept
  {
    auto ctx = _ctx.lock();
    return ctx ? ctx : nullptr;
  }

  void evict() noexcept
  {
    cancel();
    auto ctx = _ctx.lock();
    if (ctx) { _eviction_queue.enqueue(std::move(ctx)); }
  }

 private:
  std::weak_ptr<prefetch_request_context> _ctx;
  std::shared_ptr<std::atomic<prefetching_handle_state>> _user_state;
  std::shared_ptr<entry_state> _prefetching_state;
  prefetching_cache::request_queue_type& _eviction_queue;
  prefetching_cache::request_queue_type& _prefetch_queue;
};

prefetching_handle::prefetching_handle() noexcept = default;
prefetching_handle::~prefetching_handle()         = default;

prefetching_handle::prefetching_handle(prefetching_handle&& o) noexcept            = default;
prefetching_handle& prefetching_handle::operator=(prefetching_handle&& o) noexcept = default;

void prefetching_handle::activate() noexcept
{
  if (_state) { _state->activate(); }
}

void prefetching_handle::cancel() noexcept
{
  if (_state) { _state->cancel(); }
}

bool prefetching_handle::is_active() const noexcept { return _state && _state->is_active(); }

std::shared_ptr<prefetch_request_context> prefetching_handle::get_context() const noexcept
{
  return _state ? _state->get_context() : nullptr;
}

prefetching_handle::prefetching_handle(std::unique_ptr<prefetch_lifecycle_manager> mgr) noexcept
  : _state(std::move(mgr))
{
}

prefetching_handle::operator bool() const noexcept { return _state != nullptr; }

std::vector<cached_chunk*> prefetching_cache::file_entry::update_and_get_chunks(
  std::span<size_t> incoming, uint32_t ticker)
{
  std::vector<cached_chunk*> result(incoming.size());

  // Phase 1: classify under shared lock — find which offsets already exist.
  // Track the indices of incoming items that need to be inserted.
  std::vector<size_t> missing_indices;  // indices into `incoming`/`result`
  {
    std::shared_lock lock(mtx);

    auto s           = chunks.begin();
    const auto s_end = chunks.end();

    for (size_t i = 0; i < incoming.size(); ++i) {
      const size_t off = incoming[i];
      s                = gallop_lower_bound(s, s_end, off);

      if (s != s_end && (*s)->offset == off) {
        s->get()->lifecycle.on_request(ticker);
        result[i] = s->get();  // existing
      } else {
        missing_indices.push_back(i);  // mark for insertion
      }
    }

    if (missing_indices.empty()) {
      return result;  // fast path: nothing to insert
    }
  }

  // Phase 2: upgrade to exclusive lock and insert missing chunks.
  // Another writer may have inserted some of our "missing" offsets in the
  // gap between unlocking and re-locking, so re-check each one.
  std::vector<std::unique_ptr<cached_chunk>> to_insert;
  to_insert.reserve(missing_indices.size());

  {
    std::unique_lock lock(mtx);

    auto s     = chunks.begin();
    auto s_end = chunks.end();

    for (size_t idx : missing_indices) {
      const size_t off = incoming[idx];
      s                = gallop_lower_bound(s, s_end, off);

      if (s != s_end && (*s)->offset == off) {
        result[idx] = s->get();  // someone else inserted it
      } else {
        auto chunk = std::make_unique<cached_chunk>(off);
        chunk->lifecycle.on_request(ticker);
        result[idx] = chunk.get();  // capture raw ptr before move
        to_insert.push_back(std::move(chunk));
      }
    }

    if (to_insert.empty()) {
      return result;  // all races lost, but result is filled
    }

    // Bulk merge: to_insert is sorted because missing_indices is in order
    // and incoming is sorted+unique.
    const auto mid = chunks.size();
    chunks.reserve(mid + to_insert.size());
    chunks.insert(chunks.end(),
                  std::make_move_iterator(to_insert.begin()),
                  std::make_move_iterator(to_insert.end()));
    std::inplace_merge(
      chunks.begin(),
      chunks.begin() + mid,
      chunks.end(),
      [](const std::unique_ptr<cached_chunk>& a, const std::unique_ptr<cached_chunk>& b) {
        return a->offset < b->offset;
      });
  }

  return result;
}

std::vector<cached_chunk*> prefetching_cache::file_entry::fetch_chunks(std::size_t offset,
                                                                       std::size_t size,
                                                                       coverage_policy policy) const
{
  std::shared_lock lock(mtx);

  auto result = find_entry(chunks, offset, size, policy);
  return result;
}

prefetching_cache::prefetching_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  sirius_ioctx* io_ctx,
  size_t inflight_budget_chunks,
  uint32_t buffer_pool_slabs,
  std::shared_ptr<const sirius::memory::topology_index> topology_index,
  bool dispose_after_use)
  : _pool(std::make_unique<buffer_pool>(reservation_manager, buffer_pool_slabs)),
    _io_ctx(io_ctx),
    _topology_index(std::move(topology_index)),
    _armed(true),
    _rate_limiter(inflight_budget_chunks),
    _dispose_after_use(dispose_after_use)
{
  _preparation_thread = std::jthread([this](const std::stop_token& st) { prepare_loop(st); },
                                     _preparation_stop_source.get_token());
  _prefetch_thread    = std::jthread([this](const std::stop_token& st) { prefetch_loop(st); },
                                  _prefetch_stop_source.get_token());
  _evictor_thread     = std::jthread([this](const std::stop_token& st) { evict_loop(st); },
                                 _evictor_stop_source.get_token());
}

prefetching_cache::~prefetching_cache()
{
  _shutting_down.store(true, std::memory_order_release);
  _preparation_stop_source.request_stop();
  _prefetch_stop_source.request_stop();
  _evictor_stop_source.request_stop();
}

// ===========================================================================
// insert
// ===========================================================================

prefetching_cache::file_entry& prefetching_cache::get_or_create_file_entry(
  const sirius_io_object& obj)
{
  const auto& key = obj.raw_file_cache_id();
  std::shared_lock lk(_map_mtx);
  auto it = _file_cache.find(key);
  if (it == _file_cache.end()) {
    lk.unlock();
    std::unique_lock ulk(_map_mtx);
    auto [new_it, inserted] = _file_cache.try_emplace(key, std::make_unique<file_entry>());
    it                      = new_it;
    if (inserted) {
      it->second->file_size = obj.size();
      it->second->io_obj    = obj.shared_from_this();
      it->second->chunks.reserve((obj.size() + _pool->chunk_bytes() - 1) / _pool->chunk_bytes());
    }
  }
  return *it->second;
}

prefetching_handle prefetching_cache::insert(const sirius_io_object& obj,
                                             std::span<const byte_range> ranges,
                                             std::optional<int> gpu_id)
{
  if (!_armed) { return prefetching_handle(nullptr); }

  const auto& key = obj.raw_file_cache_id();
  auto& file      = get_or_create_file_entry(obj);

  const size_t chunk_bytes = _pool->chunk_bytes();
  auto coalesced_ranges    = _io_ctx->align_and_coalesce(ranges, chunk_bytes);

  // Enumerate the chunk-aligned offsets covered by the coalesced ranges.  The
  // ranges come back aligned to chunk_bytes, sorted, and non-overlapping, so the
  // resulting offsets are sorted, unique, and chunk_bytes apart within each
  // range — the contract update_and_get_chunks' galloping search relies on.
  std::vector<size_t> chunk_offsets;  // sorted, unique, chunk-aligned
  size_t total_chunks = 0;
  for (const auto& r : coalesced_ranges) {
    total_chunks += (static_cast<size_t>(r.size()) + chunk_bytes - 1) / chunk_bytes;
  }
  chunk_offsets.reserve(total_chunks);
  for (const auto& r : coalesced_ranges) {
    const auto start = static_cast<size_t>(r.offset());
    const auto end   = start + static_cast<size_t>(r.size());
    for (size_t off = start; off < end; off += chunk_bytes) {
      chunk_offsets.push_back(off);
    }
  }

  auto chunks_to_fetch =
    file.update_and_get_chunks(chunk_offsets, _ticker.load(std::memory_order_relaxed));

  auto work    = std::make_shared<prefetch_request_context>(obj, _ticker.load());
  work->chunks = std::move(chunks_to_fetch);
  // Resolve the preferred NUMA node for staging buffers from the target GPU's
  // topology; -1 (no preference) when no GPU hint or the GPU is out of scope.
  if (gpu_id && _topology_index) { work->preferred_numa = _topology_index->numa_node_of(*gpu_id); }

  prefetching_handle handle(std::make_unique<prefetching_handle::prefetch_lifecycle_manager>(
    work, _eviction_queue, _prefetch_queue));
  _preparation_queue.enqueue(std::move(work));

  return std::move(handle);
}

bool prefetching_cache::host_read_from_cache_only(const sirius_io_object& obj,
                                                  size_t offset,
                                                  size_t size,
                                                  uint8_t* dst,
                                                  prefetching_handle* out_handle)
{
  if (size == 0) return true;

  std::vector<cached_chunk*> chunks;
  if (out_handle && *out_handle) {
    auto& requested_chunks = out_handle->get_context()->chunks;
    chunks                 = find_entry(requested_chunks, offset, size, coverage_policy::full);
  } else {
    file_entry* file = nullptr;
    {
      std::shared_lock lk(_map_mtx);
      auto it = _file_cache.find(obj.raw_file_cache_id());
      if (it != _file_cache.end()) { file = it->second.get(); }
    }
    if (file) { chunks = file->fetch_chunks(offset, size, coverage_policy::full); }
  }

  while (!chunks.empty()) {
    auto iter =
      std::ranges::find_if(chunks, [](cached_chunk* c) { return !c->state.acquire_read(); });

    if (iter != chunks.end()) {
      std::for_each(chunks.begin(), iter, [](cached_chunk* c) { c->state.release_read(); });
      break;
    }

    auto const end_offset = offset + size;
    auto const chunk_size = _pool->chunk_bytes();

    for (auto* chunk : chunks) {
      auto const chunk_begin = std::max(offset, chunk->offset);
      auto const chunk_end   = std::min(end_offset, chunk->offset + chunk_size);
      auto const copy_size   = chunk_end - chunk_begin;
      auto const src_offset  = chunk_begin - chunk->offset;
      auto const dst_offset  = chunk_begin - offset;

      std::memcpy(dst + dst_offset, chunk->data + src_offset, copy_size);
      chunk->state.release_read();
    }
    _counters.hits.fetch_add(chunks.size(), std::memory_order_relaxed);
    return true;
  }
  return false;
}

exec::semi_future<std::size_t> prefetching_cache::host_read_async(const sirius_io_object& obj,
                                                                  size_t offset,
                                                                  size_t size,
                                                                  uint8_t* dst,
                                                                  prefetching_handle* out_handle)
{
  bool status = host_read_from_cache_only(obj, offset, size, dst, out_handle);
  if (status) { return exec::make_semi_future<std::size_t>(size); }
  size_t n_chunks = (size + _pool->chunk_bytes() - 1) / _pool->chunk_bytes();
  _counters.misses.fetch_add(n_chunks, std::memory_order_relaxed);
  return _io_ctx->host_read_async_io(obj, offset, size, dst);
}

std::size_t prefetching_cache::host_read(const sirius_io_object& obj,
                                         size_t offset,
                                         size_t size,
                                         uint8_t* dst,
                                         prefetching_handle* out_handle)
{
  bool status = host_read_from_cache_only(obj, offset, size, dst, out_handle);
  if (status) { return size; }
  size_t n_chunks = (size + _pool->chunk_bytes() - 1) / _pool->chunk_bytes();
  _counters.misses.fetch_add(n_chunks, std::memory_order_relaxed);
  return _io_ctx->host_read_io(obj, offset, size, dst);
}

exec::semi_future<std::size_t> prefetching_cache::device_read_async(const sirius_io_object& obj,
                                                                    size_t offset,
                                                                    size_t size,
                                                                    uint8_t* dst,
                                                                    rmm::cuda_stream_view stream,
                                                                    prefetching_handle* out_handle)
{
  if (size == 0 || dst == nullptr) { return std::size_t{0}; }

  _counters.n_reads.fetch_add(1, std::memory_order_relaxed);

  coverage_policy policy =
    _io_ctx->supports_host_to_device_read() ? coverage_policy::partial : coverage_policy::full;

  size_t n_chunks = (size + _pool->chunk_bytes() - 1) / _pool->chunk_bytes();
  std::vector<cached_chunk*> chunks;
  chunks.reserve(n_chunks);
  if (out_handle && *out_handle) {
    auto& requested_chunks = out_handle->get_context()->chunks;
    chunks                 = find_entry(requested_chunks, offset, size, policy);
  } else {
    std::shared_lock lk(_map_mtx);
    auto it = _file_cache.find(obj.raw_file_cache_id());
    if (it != _file_cache.end()) { chunks = it->second->fetch_chunks(offset, size, policy); }
  }

  while (!chunks.empty()) {
    size_t const chunk_bytes     = _pool->chunk_bytes();
    size_t const first_chunk_off = (offset / chunk_bytes) * chunk_bytes;
    size_t const last_chunk_off  = ((offset + size - 1) / chunk_bytes) * chunk_bytes;

    // Classify every chunk-aligned position covering the request into one of:
    //   (1) already populated         -> acquire a read pin and copy it to the
    //       device now;
    //   (2) allocated but not loaded  -> take it `loading` and read
    //       file -> its own bounce buffer -> device, publishing it to the cache
    //       on success;
    //   (3) missing or busy           -> a gap (no chunk), or a chunk we could
    //       neither read-pin nor take for loading: read file -> an internal
    //       bounce slot -> device with a null host buffer, leaving the cache
    //       untouched.
    // Cases (2) and (3) are issued together through host_to_device_read_async_io.
    std::vector<cached_chunk*> cached_chunks;        // case 1
    std::vector<cached_chunk*> io_chunks;            // case 2
    std::vector<io::io_object_segment> io_segments;  // cases 2 + 3, in file order

    bool cache_while_reading_enabled = _io_ctx->supports_host_to_device_read();
    bool every_chunk_is_cached       = true;
    std::size_t hits                 = 0;
    std::size_t h2d                  = 0;
    std::size_t misses               = 0;
    size_t ci                        = 0;  // cursor into `chunks` (sorted by offset)
    for (size_t off = first_chunk_off; off <= last_chunk_off; off += chunk_bytes) {
      while (ci < chunks.size() && chunks[ci]->offset < off) {
        ++ci;
      }
      cached_chunk* c = (ci < chunks.size() && chunks[ci]->offset == off) ? chunks[ci] : nullptr;

      if (c != nullptr && c->state.acquire_read()) {
        cached_chunks.push_back(c);  // (1) hit
        hits++;
      } else {
        if (!cache_while_reading_enabled) {
          every_chunk_is_cached = false;
          break;  // (3) miss, but we can't do H2D IO, so fall back to direct device read
        }
        if (c != nullptr && c->state.mark_loading()) {
          assert(c->data != nullptr);
          io_chunks.push_back(c);  // (2) host-to-device load
          io_segments.emplace_back(off, chunk_bytes, c->data);
          h2d++;
        } else {
          io_segments.emplace_back(off, chunk_bytes, nullptr);  // (3) miss
          misses++;
        }
      }
    }

    // Without host-to-device IO we can only serve positions already in the cache.
    // If anything needs loading/bouncing, undo our marks and let the caller fall
    // back to a direct device read.
    if (!cache_while_reading_enabled && !every_chunk_is_cached) {
      std::ranges::for_each(cached_chunks, [](cached_chunk* c) { c->state.release_read(); });
      break;
    }

    _counters.hits.fetch_add(hits, std::memory_order_relaxed);
    _counters.h2d.fetch_add(h2d, std::memory_order_relaxed);
    _counters.misses.fetch_add(misses, std::memory_order_relaxed);

    // (1) copy the already-cached chunks straight to the device on `stream`.
    for (cached_chunk* c : cached_chunks) {
      size_t const copy_start = std::max(c->offset, offset);
      size_t const copy_end   = std::min(c->offset + chunk_bytes, offset + size);
      cudaMemcpyAsync(dst + (copy_start - offset),
                      c->data + (copy_start - c->offset),
                      copy_end - copy_start,
                      cudaMemcpyHostToDevice,
                      stream);
    }

    // (2)+(3): file -> (own bounce | internal bounce) -> device through the IO
    // context.  The future resolves once the H2D copies are *enqueued*; we then
    // synchronize the stream (covering both these copies and the case-(1) copies
    // above) before mutating chunk state, since releasing a read pin or publishing
    // loading -> cached makes a chunk evictable.  Run the continuation on the IO
    // callback pool, not inline on the reactor thread, because it blocks on
    // stream.synchronize().  When there are no IO segments (cache-only path) we
    // synthesize a ready future so both paths share one continuation; stream.synchronize()
    // is equivalent to the previous event.synchronize() since only case-(1) copies are
    // on the stream at that point.
    auto io_fut = io_segments.empty() ? exec::make_semi_future<size_t>(size)
                                      : _io_ctx->host_to_device_read_async_io(
                                          obj, io_segments, offset, size, dst, stream);
    return std::move(io_fut)
      .via(&_io_cb_dispatcher)
      .then_try([stream,
                 size,
                 read_pinned = std::move(cached_chunks),
                 loading     = std::move(io_chunks)](exec::try_t<size_t>&& res) mutable -> size_t {
        bool ok = !res.has_exception();

        std::exception_ptr cuda_exception = nullptr;
        try {
          stream.synchronize();
        } catch (...) {
          cuda_exception = std::current_exception();
          ok             = false;
        }

        // (1) drop the read pins; (2) publish loading -> cached on success, or
        // revert loading -> allocated on failure so a later read can retry.
        std::ranges::for_each(read_pinned, [](cached_chunk* c) { c->state.release_read(); });
        auto transition = ok ? &entry_state::mark_cached : &entry_state::mark_load_failed;
        std::ranges::for_each(loading, [transition](cached_chunk* c) { (c->state.*transition)(); });

        if (res.has_exception() || cuda_exception) {
          std::rethrow_exception(res.has_exception() ? std::move(res).exception() : cuda_exception);
        }
        return size;
      })
      .semi();
  }
  _counters.misses.fetch_add(n_chunks, std::memory_order_relaxed);
  return _io_ctx->device_read_async_io(obj, offset, size, dst, stream);
}

std::string prefetching_cache::summary() const
{
  // Global totals plus the deltas since the last refresh (the most recent
  // query cycle), reported separately.
  uint64_t const reads = _counters.n_reads.load(std::memory_order_relaxed);
  uint64_t const hits  = _counters.hits.load(std::memory_order_relaxed);
  uint64_t const h2d   = _counters.h2d.load(std::memory_order_relaxed);
  uint64_t const miss  = _counters.misses.load(std::memory_order_relaxed);
  uint64_t const evict = _counters.evictions.load(std::memory_order_relaxed);

  return fmt::format(
    "prefetching_cache: "
    "global[reads={} hits={} h2d={} miss={} evictions={}] "
    "last_cycle[reads={} hits={} h2d={} miss={} evictions={}]",
    reads,
    hits,
    h2d,
    miss,
    evict,
    reads - _last_reported.n_reads,
    hits - _last_reported.hits,
    h2d - _last_reported.h2d,
    miss - _last_reported.misses,
    evict - _last_reported.evictions);
}

void prefetching_cache::prepare_for_query(const sirius::planner::query& query) noexcept
{
  _ticker.fetch_add(1, std::memory_order_relaxed);

  // Snapshot the counters so the next summary() can report this cycle's deltas.
  _last_reported = {
    _counters.n_reads.load(std::memory_order_relaxed),
    _counters.hits.load(std::memory_order_relaxed),
    _counters.h2d.load(std::memory_order_relaxed),
    _counters.misses.load(std::memory_order_relaxed),
    _counters.evictions.load(std::memory_order_relaxed),
  };
}

// ===========================================================================

void prefetching_cache::prepare_loop(const std::stop_token& st)
{
  std::stop_callback cb(st, [this]() {
    spdlog::trace("prefetching_cache: prepare_loop received stop request, unblocking queue");
    _preparation_queue.enqueue(nullptr);  // unblock the worker if it's waiting on an empty queueue
  });

  while (!_shutting_down && !st.stop_requested()) {
    prefetch_request req = nullptr;
    _preparation_queue.wait_dequeue(req);
    if (req == nullptr) { continue; }  // spurious wakeup or shutdown

    if (req->is_cancelled()) { continue; }  // request was cancelled

    auto& chunks = req->chunks;

    // how many buffers we need to allocate from the pool to prepare this request?
    std::size_t n_chunks_needed = std::ranges::count_if(
      chunks, [](cached_chunk* c) { return c->state.get_state() == entry_state::empty; });

    // Allocate from the arena on the request's preferred NUMA node, falling
    // back to any other arena (allocate_bulk wraps around).  numa_allocated is
    // updated to the arena we actually drew from — the whole batch comes from a
    // single arena, so all chunks share that NUMA node.
    int numa_allocated = req->preferred_numa;
    auto buffers       = _pool->allocate_bulk(n_chunks_needed, numa_allocated);
    if (buffers.size() != n_chunks_needed) {
      // No single arena could satisfy the request.  Return whatever we got and
      // re-enqueue the work for a retry after the evictor frees some.
      // todo(amin): needs eviction and then backoff
      if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa_allocated); }
      _preparation_queue.enqueue(std::move(req));
      continue;
    }

    for (size_t i = 0; i < n_chunks_needed; ++i) {
      if (chunks[i]->state.mark_queued()) {
        chunks[i]->data      = reinterpret_cast<uint8_t*>(buffers[i]);
        chunks[i]->numa_node = numa_allocated;
        if (!chunks[i]->state.mark_allocated()) {
          spdlog::error(
            "prefetching_cache: chunk at offset {} was marked queued but failed to mark "
            "allocated",
            chunks[i]->offset);
        }
      }
    }

    std::ignore = req->state->mark_allocated();

    if (!_io_ctx->supports_vector_host_read() ||
        _io_ctx->preferred_prefetching_stage() == prefetching_stage::disposable) {
      // either the backend doesn't support scatter-gather reads or it prefers not to reuse
      // buffers for multiple reads.  In either case, we can skip the prefetching step and let the
      // read() path handle the IO directly into the caller's buffer.
      continue;
    }

    if (req->is_active() && !st.stop_requested()) { _prefetch_queue.enqueue(std::move(req)); }
  }
}

void prefetching_cache::prefetch_loop(const std::stop_token& st)
{
  std::stop_callback cb(st, [this]() {
    spdlog::trace("prefetching_cache: prefetch_loop received stop request, unblocking queue");
    _prefetch_queue.enqueue(nullptr);  // unblock the worker if it's waiting on an empty queueue
  });
  while (!_shutting_down && !st.stop_requested()) {
    prefetch_request req = nullptr;
    _prefetch_queue.wait_dequeue(req);
    if (req == nullptr || req->is_cancelled()) { continue; }

    auto& allocated_chunks = req->chunks;
    auto& io_obj           = req->obj;
    std::vector<io::io_object_segment> segments;

    segments.reserve(allocated_chunks.size());
    allocated_chunks.erase(std::remove_if(allocated_chunks.begin(),
                                          allocated_chunks.end(),
                                          [&](cached_chunk* c) {
                                            if (c->state.mark_loading()) {
                                              segments.emplace_back(
                                                c->offset, _pool->chunk_bytes(), c->data);
                                              return false;
                                            }
                                            return true;
                                          }),
                           allocated_chunks.end());

    std::ignore = req->state->mark_loading();

    // todo(amin): mark the cache_entries
    _io_ctx->host_read_ranges_async_io(*io_obj, segments)
      .via(&_io_cb_dispatcher)
      .then_try([chunks = std::move(allocated_chunks)](exec::try_t<size_t>&& res) mutable {
        auto transition =
          res.has_value() ? [](cached_chunk* c) { static_cast<void>(c->state.mark_cached()); }
                          : [](cached_chunk* c) { static_cast<void>(c->state.mark_load_failed()); };
        std::ranges::for_each(chunks, transition);
      });
  }
}

void prefetching_cache::evict_loop(const std::stop_token& st)
{
  std::stop_callback cb(st, [this]() {
    spdlog::trace("prefetching_cache: evict_loop received stop request, unblocking queue");
    _eviction_queue.enqueue(nullptr);  // unblock the worker if it's waiting on an empty queueue
  });

  // One queued prefetch request plus how many of its chunks have been reclaimed
  // so far.  The request is dropped once n_evicted reaches its chunk count.
  struct eviction_request {
    prefetch_request req;
    size_t n_evicted{0};
  };

  std::vector<eviction_request> eviction_batch;
  // Reclaimed buffers grouped by their origin NUMA node so each group can be
  // returned to the arena it came from.
  std::unordered_map<int, std::vector<std::byte*>> reclaim_by_numa;
  while (!_shutting_down && !st.stop_requested()) {
    prefetch_request req = nullptr;
    _eviction_queue.wait_dequeue(req);
    if (req == nullptr) { continue; }

    // Accumulate newly-queued requests into the persistent batch.  The batch is
    // NOT cleared each round: a request is dropped only once all of its chunks
    // have been evicted.  Chunks that are busy this round — or skipped once the
    // free target is met — therefore stay candidates for a later round instead
    // of being lost forever.
    eviction_batch.push_back({std::move(req), 0});
    while (_eviction_queue.try_dequeue(req)) {
      if (req != nullptr) { eviction_batch.push_back({std::move(req), 0}); }
    }

    // When disposing after use we reclaim everything; otherwise we only evict
    // under memory pressure and stop once enough chunks are free again.  Memory
    // pressure is scored as outstanding (handed-out) chunks against the pool's
    // aggregate reserved capacity.
    bool const dispose       = _dispose_after_use;
    size_t const capacity    = _pool->capacity_chunks();
    size_t const outstanding = _pool->total_chunks();
    bool const should_evict =
      dispose || (capacity > 0 && outstanding > (capacity * 3) / 4);  // <25% free
    if (!should_evict) { continue; }

    // Number of chunks we want to evict before stopping (unbounded when
    // disposing).  Aim to bring outstanding back down to ~25% of capacity.
    size_t const target_outstanding = capacity / 4;
    size_t const need =
      dispose ? std::numeric_limits<size_t>::max()
              : (outstanding > target_outstanding ? outstanding - target_outstanding : 0);

    auto const query_tick = static_cast<uint32_t>(_ticker.load(std::memory_order_relaxed));

    // Pass 1: in a single sweep, histogram the currently-evictable chunks
    // (cached/allocated with pin == 0) by demand tier.  This lets us pick the
    // exact tier bar up-front instead of re-sweeping the batch once per tier.
    std::array<size_t, chunk_lifecycle::FRESH_SCORE + 1> tier_count{};
    for (auto const& er : eviction_batch) {
      if (er.req == nullptr) { continue; }
      for (cached_chunk* c : er.req->chunks) {
        auto const s = c->state.get_state();
        if (s != entry_state::cached && s != entry_state::allocated) { continue; }
        ++tier_count[c->lifecycle.load().eviction_tier(query_tick)];
      }
    }

    // The cutoff is the lowest tier whose cumulative evictable count meets the
    // target (or the top tier if even all evictable chunks fall short).  We
    // evict everything strictly below the cutoff and just enough of the cutoff
    // tier to reach the target — keeping the most in-demand chunks resident.
    uint16_t cutoff   = chunk_lifecycle::FRESH_SCORE;
    size_t cumulative = 0;
    for (uint16_t t = 0; t <= chunk_lifecycle::FRESH_SCORE; ++t) {
      cumulative += tier_count[t];
      if (cumulative >= need) {
        cutoff = t;
        break;
      }
    }

    // Pass 2: a single sweep evicts every chunk below the cutoff tier, plus
    // chunks at the cutoff tier until the target is met.  Reclaimed buffers are
    // bucketed by their origin NUMA node so they go back to the right arena.
    for (auto& [_, buffers] : reclaim_by_numa) {
      buffers.clear();
    }
    size_t reclaimed = 0;
    for (auto& er : eviction_batch) {
      if (er.req == nullptr) { continue; }
      for (cached_chunk* c : er.req->chunks) {
        uint16_t const tier = c->lifecycle.load().eviction_tier(query_tick);
        if (tier > cutoff) { continue; }
        if (tier == cutoff && reclaimed >= need) { continue; }  // top bar satisfied
        // mark_evicting only succeeds from cached/allocated with pin == 0, so
        // in-use, loading, queued or already-evicted chunks are skipped.
        if (c->state.mark_evicting()) {
          reclaim_by_numa[c->numa_node].push_back(reinterpret_cast<std::byte*>(c->data));
          c->data = nullptr;
          static_cast<void>(c->state.mark_empty());
          ++reclaimed;
          ++er.n_evicted;
          _counters.evictions.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

    // Return each NUMA group to its own arena (origin-safe).
    for (auto& [numa, buffers] : reclaim_by_numa) {
      if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa); }
    }

    // Drop requests whose chunks have all been evicted; keep the rest for a
    // later round.
    std::erase_if(eviction_batch, [](eviction_request const& er) {
      return er.req == nullptr || er.n_evicted >= er.req->chunks.size();
    });
  }
}

}  // namespace sirius::io::cache
