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

#include "cucascade/cuda/event.hpp"
#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/cache/types.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "util/error_utils.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

namespace sirius::io::cache {

prefetching_handle::prefetching_handle(prefetch_request req) noexcept : _req(std::move(req)) {}

prefetching_handle::~prefetching_handle()
{
  if (_req.consumer) { _req.consumer->mark_disposed(); }
}

prefetching_handle::prefetching_handle(prefetching_handle&& o) noexcept : _req(std::move(o._req))
{
  o._req = {};
}

prefetching_handle& prefetching_handle::operator=(prefetching_handle&& o) noexcept
{
  if (this != &o) {
    if (_req.consumer) { _req.consumer->mark_disposed(); }
    _req   = std::move(o._req);
    o._req = {};
  }
  return *this;
}

void prefetching_handle::update(scan_stage stage) noexcept
{
  if (!_req.consumer) { return; }
  auto const mapped = to_consumer_stage(stage);
  if (!mapped) { return; }
  std::ignore = _req.consumer->mark(*mapped);
}

bool prefetching_handle::is_active() const noexcept { return _req.is_active(); }

bool prefetching_handle::wait_until_prepared() noexcept
{
  if (!_req.producer) { return false; }
  return _req.producer->wait_until_prepared();
}

std::shared_ptr<const std::vector<cached_chunk*>> prefetching_handle::chunks() const noexcept
{
  return _req.chunks;
}

prefetching_handle::operator bool() const noexcept { return static_cast<bool>(_req); }

std::vector<cached_chunk*> prefetching_cache::file_entry::update_and_get_chunks(
  std::span<const size_t> incoming, std::span<const chunk_fill> desired)
{
  assert(incoming.size() == desired.size());
  std::vector<cached_chunk*> result(incoming.size(), nullptr);

  // Claim an already-materialised chunk for this request: count the subscriber
  // and widen the extent it is expected to hold.  merge_fill is itself gated on
  // the chunk's state — it widens only a chunk that has not been loaded yet, so
  // a chunk already `cached` at a narrower extent keeps advertising exactly the
  // bytes somebody wrote, and a request needing more of it correctly misses.
  auto claim = [](cached_chunk* c, chunk_fill want) {
    c->state.add_subscriber();
    std::ignore = c->state.merge_fill(want);
  };

  // Phase 1: under the shared lock, take every slot that is already populated.
  // Only the offsets whose slot is still empty need the exclusive lock.
  std::vector<size_t> missing_indices;  // indices into `incoming`/`result`
  {
    std::shared_lock lock(mtx);
    for (size_t i = 0; i < incoming.size(); ++i) {
      auto const slot = slot_of(incoming[i]);
      if (slot >= slots.size()) { continue; }  // past EOF: leave result[i] null
      if (auto* c = slots[slot]) {
        claim(c, desired[i]);
        result[i] = c;
      } else {
        missing_indices.push_back(i);
      }
    }
    if (missing_indices.empty()) { return result; }
  }

  // Phase 2: materialise the missing chunks.  Another writer may have filled
  // some of our slots between unlocking and re-locking, so re-check each one.
  {
    std::unique_lock lock(mtx);
    for (size_t idx : missing_indices) {
      auto const slot = slot_of(incoming[idx]);
      auto*& entry    = slots[slot];
      if (entry == nullptr) { entry = arena.emplace(incoming[idx]); }
      claim(entry, desired[idx]);
      result[idx] = entry;
    }
  }

  return result;
}

std::vector<cached_chunk*> prefetching_cache::file_entry::fetch_chunks(std::size_t offset,
                                                                       std::size_t size,
                                                                       coverage_policy policy) const
{
  if (size == 0) { return {}; }

  auto const first_slot = slot_of(offset);
  auto const last_slot  = slot_of(offset + size - 1);

  std::vector<cached_chunk*> result;
  result.reserve(last_slot - first_slot + 1);

  std::shared_lock lock(mtx);
  if (last_slot >= slots.size()) { return {}; }  // request runs past EOF

  for (auto slot = first_slot; slot <= last_slot; ++slot) {
    auto* c = slots[slot];
    if (c == nullptr) {
      if (policy == coverage_policy::full) { return {}; }  // gap: no positional coverage
      continue;
    }
    result.push_back(c);
  }
  return result;
}

prefetching_cache::prefetching_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  ioctx* io_ctx,
  const config& cfg,
  std::shared_ptr<const sirius::memory::topology_index> topology_index)
  : _cfg(cfg),
    _pool(std::make_unique<buffer_pool>(
      reservation_manager, cfg.min_prefetching_budget_fraction, cfg.eviction_threshold_fraction)),
    _io_ctx(io_ctx),
    _topology_index(std::move(topology_index)),
    _armed(_io_ctx->can_use_prefetching_cache()),
    _rate_limiter(_cfg.inflight_io_chunk_budget)
{
  _preparation_thread = std::jthread([this](const std::stop_token& st) { prepare_loop(st); },
                                     _preparation_stop_source.get_token());
  _evictor_thread     = std::jthread([this](const std::stop_token& st) { evict_loop(st); },
                                 _evictor_stop_source.get_token());
  _chunk_size         = _pool->chunk_size();

  // chunk_state records a chunk's populated extent as a 14-bit page count, so a
  // chunk wider than that many pages could not express a partial fill.
  if (_chunk_size / io::IO_BLOCK_SIZE > chunk_state::max_fill_pages()) {
    SIRIUS_LOG_ERROR(
      "prefetching_cache: chunk size {} exceeds the {} bytes a chunk's populated extent can "
      "address; partial chunk fills will be recorded incorrectly",
      _chunk_size,
      static_cast<size_t>(chunk_state::max_fill_pages()) * io::IO_BLOCK_SIZE);
  }
}

prefetching_cache::~prefetching_cache()
{
  _shutting_down.store(true, std::memory_order_release);
  _preparation_stop_source.request_stop();
  _evictor_stop_source.request_stop();

  _rate_limiter.wait_for_all();
  _preparation_thread.join();
  _evictor_thread.join();
}

// ===========================================================================
// insert
// ===========================================================================

prefetching_cache::file_entry& prefetching_cache::get_or_create_file_entry(const io_object& obj)
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
      it->second->file_size  = obj.size();
      it->second->io_obj     = obj.shared_from_this();
      it->second->chunk_size = _chunk_size;
      // One slot per chunk-aligned position in the file — the same capacity the
      // sorted chunk vector used to reserve, but indexable instead of searchable.
      it->second->slots.assign((obj.size() + _chunk_size - 1) / _chunk_size, nullptr);
    }
  }
  return *it->second;
}

prefetching_handle prefetching_cache::insert(const io_object& obj,
                                             std::span<const byte_range> ranges,
                                             std::optional<int> gpu_id)
{
  if (!_armed) { return prefetching_handle(); }

  auto& file = get_or_create_file_entry(obj);

  const size_t chunk_bytes = _chunk_size;

  // Enumerate the chunk-aligned positions the requested ranges touch, together
  // with the extent each one is actually wanted for.  Derived from the ORIGINAL
  // ranges rather than a chunk-aligned coalesce of them: coalescing would pull
  // in whole chunks that no range touches, and since every chunk now issues its
  // own IO segment sized to its extent, those chunks would be pure over-read.
  std::vector<std::pair<size_t, chunk_fill>> wanted;
  for (const auto& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const lo = static_cast<size_t>(r.offset());
    auto const hi = lo + static_cast<size_t>(r.size());
    for (auto off = (lo / chunk_bytes) * chunk_bytes; off < hi; off += chunk_bytes) {
      wanted.emplace_back(off, needed_fill(off, chunk_bytes, lo, hi));
    }
  }
  std::ranges::sort(wanted, {}, [](auto const& w) { return w.first; });

  // Collapse to unique offsets, folding the extents of ranges that share a chunk.
  std::vector<size_t> chunk_offsets;  // sorted, unique, chunk-aligned
  std::vector<chunk_fill> desired;    // index-parallel with chunk_offsets
  chunk_offsets.reserve(wanted.size());
  desired.reserve(wanted.size());
  for (auto const& [off, fill] : wanted) {
    if (!chunk_offsets.empty() && chunk_offsets.back() == off) {
      desired.back() = merge(desired.back(), fill);
    } else {
      chunk_offsets.push_back(off);
      desired.push_back(fill);
    }
  }

  auto chunks_to_fetch = file.update_and_get_chunks(chunk_offsets, desired);
  std::erase(chunks_to_fetch, nullptr);  // offsets past EOF have no slot

  prefetch_request req;
  req.obj       = obj.shared_from_this();
  req.producer  = std::make_shared<producer_stage>();
  req.consumer  = std::make_shared<consumer_stage>();
  req.chunks    = std::make_shared<const std::vector<cached_chunk*>>(std::move(chunks_to_fetch));
  req.timestamp = _ticker.load(std::memory_order_relaxed);
  // Resolve the preferred NUMA node for staging buffers from the target GPU's
  // topology; -1 (no preference) when no GPU hint or the GPU is out of scope.
  if (gpu_id && _topology_index) { req.preferred_numa = _topology_index->numa_node_of(*gpu_id); }

  std::ignore = req.producer->mark_queued();
  _preparation_queue.enqueue(req);

  return prefetching_handle(std::move(req));
}

bool prefetching_cache::host_read_from_cache_only(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, prefetching_handle* out_handle)
{
  if (size == 0) return true;

  std::vector<cached_chunk*> chunks;
  if (out_handle && *out_handle) {
    if (auto requested = out_handle->chunks()) {
      chunks = find_entry(*requested, offset, size, coverage_policy::full, _chunk_size);
    }
  }
  if (chunks.empty()) {
    std::shared_lock lk(_map_mtx);
    auto it = _file_cache.find(obj.raw_file_cache_id());
    if (it != _file_cache.end()) {
      auto* file = it->second.get();
      lk.unlock();
      chunks = file->fetch_chunks(offset, size, coverage_policy::full);
    }
  }

  auto const end_offset = offset + size;
  auto const chunk_size = _chunk_size;

  while (!chunks.empty()) {
    // Pin each chunk and confirm in the same CAS that it is populated over the
    // sub-range this read needs.  A chunk we cannot pin — or one whose extent
    // does not cover the request — aborts the whole attempt: we unwind the pins
    // taken so far and let the caller do one fallback IO, never a partial copy.
    auto iter = std::ranges::find_if(chunks, [&](cached_chunk* c) {
      auto const lo = std::max(offset, c->offset);
      auto const hi = std::min(end_offset, c->offset + chunk_size);
      return !c->state.try_pin_covering(c->offset, chunk_size, lo, hi);
    });

    if (iter != chunks.end()) {
      std::for_each(chunks.begin(), iter, [](cached_chunk* c) { c->state.release_read(); });
      break;
    }

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

exec::semi_future<std::size_t> prefetching_cache::host_read_async(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, prefetching_handle* out_handle)
{
  bool status = host_read_from_cache_only(obj, offset, size, dst, out_handle);
  if (status) { return exec::make_semi_future<std::size_t>(size); }
  size_t n_chunks = (size + _chunk_size - 1) / _chunk_size;
  _counters.misses.fetch_add(n_chunks, std::memory_order_relaxed);
  return _io_ctx->host_read_async_io(obj, offset, size, dst);
}

std::size_t prefetching_cache::host_read(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, prefetching_handle* out_handle)
{
  bool status = host_read_from_cache_only(obj, offset, size, dst, out_handle);
  if (status) { return size; }
  size_t n_chunks = (size + _chunk_size - 1) / _chunk_size;
  _counters.misses.fetch_add(n_chunks, std::memory_order_relaxed);
  return _io_ctx->host_read_io(obj, offset, size, dst);
}

exec::semi_future<std::size_t> prefetching_cache::device_read_async(const io_object& obj,
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

  size_t n_chunks = (size + _chunk_size - 1) / _chunk_size;
  std::vector<cached_chunk*> chunks;
  chunks.reserve(n_chunks);
  if (out_handle && *out_handle) {
    if (auto requested = out_handle->chunks()) {
      chunks = find_entry(*requested, offset, size, policy, _chunk_size);
    }
  }
  if (chunks.empty()) {
    std::shared_lock lk(_map_mtx);
    auto it = _file_cache.find(obj.raw_file_cache_id());
    if (it != _file_cache.end()) {
      auto* file = it->second.get();
      lk.unlock();
      chunks = file->fetch_chunks(offset, size, policy);
    }
  }

  while (!chunks.empty()) {
    size_t const chunk_bytes     = _chunk_size;
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

      // The portion of this chunk the request actually needs, clamped to both
      // the request and the chunk.  Shared by the hit gate and the load span.
      size_t const need_lo = std::max(off, offset);
      size_t const need_hi = std::min(off + chunk_bytes, offset + size);

      // (1) Already populated over the bytes we need: pin it and copy.  The
      // coverage test rides along in the pinning CAS, so a chunk that is
      // resident but not populated far enough costs one load, not a pin/unpin.
      if (c != nullptr && c->state.try_pin_covering(off, chunk_bytes, need_lo, need_hi)) {
        cached_chunks.push_back(c);
        hits++;
        continue;
      }

      if (!cache_while_reading_enabled) {
        every_chunk_is_cached = false;
        break;  // (3) miss, but we can't do H2D IO, so fall back to direct device read
      }

      // (2) Claim the chunk and stage the read through its own buffer.  The
      // extent comes back out of the claiming CAS already widened by whatever a
      // queued prefetch asked for, and the IO span is derived FROM that extent —
      // so the bytes read are exactly the bytes the chunk will later advertise.
      // Only the needed edge is read, so a chunk touched at its head or tail no
      // longer costs a whole chunk of IO to cache.
      chunk_fill fill;
      if (c != nullptr &&
          c->state.take_loading_merging(needed_fill(off, chunk_bytes, need_lo, need_hi), fill)) {
        assert(c->data != nullptr);
        auto const [seg_lo, seg_hi] = fill_span(fill, off, chunk_bytes);
        io_chunks.push_back(c);  // host-to-device load into the cache buffer
        io_segments.emplace_back(seg_lo, seg_hi - seg_lo, c->data + (seg_lo - off));
        h2d++;
        continue;
      }

      // (3) busy or missing chunk: read just the needed, block-aligned span via
      // an internal bounce slot (null host buffer); do not touch the cache.
      size_t const seg_lo = align_down(need_lo, io::IO_BLOCK_SIZE);
      size_t const seg_hi = std::min(off + chunk_bytes, align_up(need_hi, io::IO_BLOCK_SIZE));
      io_segments.emplace_back(seg_lo, seg_hi - seg_lo, nullptr);
      misses++;
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

    auto device_id = rmm::get_current_cuda_device();

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
      .via(exec::inline_executor::instance())
      .then_try([this,
                 stream,
                 device_id,
                 size,
                 read_pinned = std::move(cached_chunks),
                 loading     = std::move(io_chunks)](exec::try_t<size_t>&& res) -> size_t {
        bool ok = !res.has_exception();

        rmm::cuda_set_device_raii guard(device_id);
        std::unique_ptr<cucascade::cuda::cuda_event> event;
        if (ok) {
          event = std::make_unique<cucascade::cuda::cuda_event>();
          event->record(stream);
        }

        _io_cb_dispatcher.enqueue([read_pinned = std::move(read_pinned),
                                   loading     = std::move(loading),
                                   event       = std::move(event),
                                   ok          = ok]() mutable {
          if (event) {
            SIRIUS_TRY_AND_LOG_EXCEPTION(
              event->synchronize(),
              "prefetching_cache: failed to synchronize CUDA stream after host-to-device copies");
          }

          std::ranges::for_each(read_pinned, [](cached_chunk* c) { c->state.release_read(); });
          auto transition = ok ? &chunk_state::mark_cached : &chunk_state::mark_load_failed;
          std::ranges::for_each(
            loading, [transition](cached_chunk* c) { std::ignore = (c->state.*transition)(); });
        });

        if (res.has_exception()) { std::rethrow_exception(std::move(res).exception()); }
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

  return std::format(
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

void prefetching_cache::prepare_for_query() noexcept
{
  SIRIUS_LOG_TRACE("prefetching_cache: summary of cache performance {}", summary());

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

void prefetching_cache::drain_and_abandon(request_queue_type& queue) noexcept
{
  prefetch_request req;
  while (queue.try_dequeue(req)) {
    if (req) { req.producer->mark_abandoned(); }
    req = {};
  }
}

void prefetching_cache::prepare_loop(const std::stop_token& st)
{
  std::stop_callback cb(st, [this]() {
    SIRIUS_LOG_TRACE("prefetching_cache: prepare_loop received stop request, unblocking queue");
    // unblock the worker if it's waiting on an empty queueue
    _preparation_queue.enqueue(prefetch_request{});
  });

  while (!_shutting_down && !st.stop_requested()) {
    prefetch_request req;
    _preparation_queue.wait_dequeue(req);
    if (!req) { continue; }  // spurious wakeup or shutdown

    if (req.is_cancelled() || !req.chunks) {
      req.producer->mark_abandoned();
      // Still route it to the evictor: insert() already counted this request as
      // a subscriber of every chunk it named, and only the evictor hands that
      // reference back.  Dropping the request here would pin those chunks for
      // the lifetime of the process.
      _eviction_queue.enqueue(std::move(req));
      continue;
    }

    std::ignore = req.producer->mark_preparing();

    auto const& chunks = *req.chunks;

    // how many buffers we need to allocate from the pool to prepare this request?
    std::size_t n_chunks_needed = std::ranges::count_if(
      chunks, [](cached_chunk* c) { return c->state.get_state() == chunk_state::empty; });

    // Allocate from the arena on the request's preferred NUMA node, falling
    // back to any other arena (allocate_bulk wraps around).  numa_allocated is
    // updated to the arena we actually drew from — the whole batch comes from a
    // single arena, so all chunks share that NUMA node.
    int numa_allocated = req.preferred_numa;
    auto buffers       = _pool->allocate_bulk(n_chunks_needed, numa_allocated);
    if (buffers.size() != n_chunks_needed) {
      // No single arena could satisfy the request.  Return whatever we got, ask
      // the evictor to free some, and retire this request on `abandoned` so no
      // waiter is left parked on the transient `preparing` stage.
      if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa_allocated); }
      req.producer->mark_abandoned();
      _eviction_queue.enqueue(prefetch_request{});  // sentinel: free some buffers
      _eviction_queue.enqueue(std::move(req));      // and take back our subscriptions
      continue;
    }

    for (auto* c : chunks) {
      if (buffers.empty()) { break; }
      if (c->state.mark_queued()) {
        auto* buffer = buffers.back();
        buffers.pop_back();
        c->data      = reinterpret_cast<uint8_t*>(buffer);
        c->numa_node = numa_allocated;
        if (!c->state.mark_allocated()) {
          buffers.push_back(buffer);  // return the buffer to the pool
          SIRIUS_LOG_ERROR(
            "prefetching_cache: chunk at offset {} was marked queued but failed to mark "
            "allocated",
            c->offset);
        }
      }
    }

    if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa_allocated); }

    if (!all_chunks_have_buffers(chunks)) {
      // Buffers were already attached above, so the request still has to reach
      // the evictor or those chunks are never reclaimed.
      _eviction_queue.enqueue(req);
      req.producer->mark_abandoned();
      continue;
    }
    std::ignore = req.producer->mark_prepared();

    // Register with the evictor eagerly; the evictor skips the request until
    // its consumer is disposed.
    _eviction_queue.enqueue(req);
  }

  drain_and_abandon(_preparation_queue);
}

bool prefetching_cache::prefetch(prefetching_handle& handle,
                                 exec::invocable<void(bool) noexcept> on_done)
{
  auto& req   = handle._req;
  auto settle = [&on_done](bool ok) {
    on_done(ok);
    return false;
  };

  if (!req || req.is_cancelled() || !req.chunks) { return settle(false); }
  if (!_io_ctx->supports_vector_host_read()) { return settle(false); }
  if (!req.producer->mark_loading()) { return settle(false); }

  std::vector<io::io_object_segment> segments;
  std::vector<cached_chunk*> claimed_chunks;
  segments.reserve(req.chunks->size());
  claimed_chunks.reserve(req.chunks->size());
  for (cached_chunk* c : *req.chunks) {
    // Claim the chunk and read back the extent it was queued for in one CAS,
    // then read exactly that span — not always the whole chunk.  A range that
    // only clips a chunk's head or tail costs one page-aligned edge of IO.
    chunk_fill fill;
    if (c->state.take_loading(fill)) {
      auto const [seg_lo, seg_hi] = fill_span(fill, c->offset, _chunk_size);
      claimed_chunks.push_back(c);
      segments.emplace_back(seg_lo, seg_hi - seg_lo, c->data + (seg_lo - c->offset));
    }
  }
  if (segments.empty()) {
    std::ignore = req.producer->mark_ready();
    return settle(true);
  }

  auto token = _rate_limiter.acquire(segments.size());

  if (req.is_cancelled()) {
    std::ranges::for_each(claimed_chunks,
                          [](cached_chunk* c) { std::ignore = c->state.mark_load_failed(); });
    std::ignore = req.producer->mark_load_failed();
    return settle(false);
  }

  _io_ctx->host_read_ranges_async_io(*req.obj, segments)
    .via(&_io_cb_dispatcher)
    .then_try([req,
               chunks = std::move(claimed_chunks),
               done   = std::move(on_done),
               _      = std::move(token)](exec::try_t<size_t>&& res) mutable {
      auto transition =
        res.has_value() ? &chunk_state::mark_cached : &chunk_state::mark_load_failed;
      std::ignore = res.has_value() ? req.producer->mark_ready() : req.producer->mark_load_failed();
      std::ranges::for_each(
        chunks, [transition](cached_chunk* c) { std::ignore = (c->state.*transition)(); });
      done(res.has_value());
    });
  return true;
}

void prefetching_cache::evict_loop(const std::stop_token& st)
{
  std::stop_callback cb(st, [this]() {
    SIRIUS_LOG_TRACE("prefetching_cache: evict_loop received stop request, unblocking queue");
    // unblock the worker if it's waiting on an empty queueue
    _eviction_queue.enqueue(prefetch_request{});
  });

  // One queued prefetch request plus whether its per-chunk subscriber
  // references have been handed back yet.
  struct eviction_request {
    prefetch_request req;
    bool released{false};
  };

  std::vector<eviction_request> eviction_batch;
  // Reclaimed buffers grouped by their origin NUMA node so each group can be
  // returned to the arena it came from.
  std::unordered_map<int, std::vector<std::byte*>> reclaim_by_numa;
  while (!_shutting_down && !st.stop_requested()) {
    prefetch_request req;
    _eviction_queue.wait_dequeue(req);

    // A null request is the "free some memory now" sentinel prepare_loop sends
    // when the pool cannot satisfy an allocation.  It has to survive into the
    // eviction decision below rather than be skipped past, or the back-pressure
    // signal does nothing and the pool never recovers.
    bool eviction_requested = false;
    auto absorb             = [&](prefetch_request&& r) {
      if (r) {
        eviction_batch.push_back({std::move(r), false});
      } else {
        eviction_requested = true;
      }
    };

    // Accumulate newly-queued requests into the persistent batch.  The batch is
    // NOT cleared each round: a request is retired only once every chunk it
    // named has been reclaimed or taken over, so chunks that are busy this
    // round stay candidates for a later one instead of being lost.
    absorb(std::move(req));
    while (_eviction_queue.try_dequeue(req)) {
      absorb(std::move(req));
    }

    if (_shutting_down || st.stop_requested()) { break; }

    // Hand back the subscriber reference of every request whose consumer is
    // gone — exactly once, which is what makes the count an accurate reference
    // count rather than a heuristic.  Pairing the decrement with the request's
    // retirement (and not with reads) is what lets a chunk be shared: a chunk
    // stops being protected only when its LAST subscriber goes away.
    for (auto& er : eviction_batch) {
      if (er.released || !er.req || !er.req.chunks || !er.req.is_cancelled()) { continue; }
      for (cached_chunk* c : *er.req.chunks) {
        c->state.drop_subscriber();
      }
      er.released = true;
    }

    // When disposing on idle we reclaim everything; otherwise we only evict
    // under memory pressure and stop once enough chunks are free again.  Memory
    // pressure is scored as outstanding (handed-out) chunks against the pool's
    // aggregate reserved capacity.
    bool const should_evict =
      _cfg.dispose_on_idle || eviction_requested || _pool->should_start_evicting();
    if (!should_evict) { continue; }

    size_t const need = _cfg.dispose_on_idle ? std::numeric_limits<size_t>::max()
                                             : _pool->total_allocated_chunks() * 0.25;

    for (auto& [_, buffers] : reclaim_by_numa) {
      buffers.clear();
    }

    // Pass 0 reclaims only chunks no live request is subscribed to.  If that
    // cannot meet the target, pass 1 sweeps again ignoring subscriptions: a
    // starved allocator is worse than a cache miss, and mark_evicting still
    // refuses any chunk a reader has pinned, so the fallback can never pull a
    // buffer out from under a live read — it only costs a future hit.
    size_t reclaimed = 0;
    for (int pass = 0; pass < 2 && reclaimed < need; ++pass) {
      bool const respect_subscribers = (pass == 0);
      for (auto& er : eviction_batch) {
        if (!er.released || !er.req || !er.req.chunks) { continue; }
        for (cached_chunk* c : *er.req.chunks) {
          if (reclaimed >= need) { break; }
          // A single relaxed load answers the whole question: does this chunk
          // hold a reclaimable buffer, and is anybody still subscribed to it?
          auto const snap = c->state.load();
          if (!snap.is_reclaimable()) { continue; }
          if (respect_subscribers && snap.subscribers() != 0) { continue; }
          if (!c->state.mark_evicting(respect_subscribers)) { continue; }
          reclaim_by_numa[c->numa_node].push_back(reinterpret_cast<std::byte*>(c->data));
          c->data     = nullptr;
          std::ignore = c->state.mark_empty();  // also clears the populated extent
          ++reclaimed;
          _counters.evictions.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

    // Return each NUMA group to its own arena (origin-safe).
    for (auto& [numa, buffers] : reclaim_by_numa) {
      if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa); }
    }

    // Retire a request once it has released its references and every chunk it
    // named is either reclaimed or now owned by somebody else.  Counting our
    // own evictions instead would strand every request that shares a chunk:
    // whoever loses the race to reclaim it could never reach its own count.
    std::erase_if(eviction_batch, [](eviction_request const& er) {
      if (!er.req || !er.req.chunks) { return true; }
      if (!er.released) { return false; }
      return std::ranges::all_of(*er.req.chunks, [](cached_chunk const* c) {
        auto const snap = c->state.load();
        return snap.state() == chunk_state::empty || snap.subscribers() != 0;
      });
    });
  }
}

}  // namespace sirius::io::cache
