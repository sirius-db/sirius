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

#include "io/prefetching_cache.hpp"

#include "ctrack.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace sirius::io {

// ===========================================================================
// buffer_pool
// ===========================================================================

buffer_pool::buffer_pool(uint32_t max_slabs) : _max_slabs(max_slabs) {}

buffer_pool::~buffer_pool() {
  for (auto &s : _slabs)
    cudaFreeHost(s->base);
}

bool buffer_pool::grow() {
  std::unique_lock lk(_grow_mtx);
  if (_slabs.size() >= _max_slabs)
    return false;

  void *raw = nullptr;
  // Portable so multi-GPU ioctx users can H2D-copy from these slabs on any
  // device's stream.
  auto err = cudaHostAlloc(&raw, SLAB_BYTES, cudaHostAllocPortable);
  if (err != cudaSuccess) {
    spdlog::warn("buffer_pool: cudaHostAlloc({:.0f}MB) failed: {}",
                 static_cast<double>(SLAB_BYTES) / (1024.0 * 1024.0),
                 cudaGetErrorString(err));
    return false;
  }

  auto s = std::make_unique<slab>();
  s->base = static_cast<std::byte *>(raw);
  s->free_count.store(CHUNKS_PER_SLAB, std::memory_order_relaxed);
  for (uint32_t i = 0; i < CHUNKS_PER_SLAB; ++i)
    s->free_chunks.enqueue(s->base + static_cast<size_t>(i) * CHUNK_BYTES);

  _slab_map[s->base] = s.get();
  _slabs.push_back(std::move(s));
  _total_chunks.fetch_add(CHUNKS_PER_SLAB, std::memory_order_relaxed);
  _total_free.fetch_add(CHUNKS_PER_SLAB, std::memory_order_relaxed);

  spdlog::debug("buffer_pool: allocated slab {} ({:.0f}MB)", _slabs.size(),
                static_cast<double>(SLAB_BYTES) / (1024.0 * 1024.0));
  return true;
}

buffer_pool::slab *buffer_pool::find_slab(std::byte *p) {
  std::shared_lock lk(_grow_mtx);
  auto it = _slab_map.upper_bound(p);
  if (it == _slab_map.begin())
    return nullptr;
  --it;
  if (p >= it->first && p < it->first + SLAB_BYTES)
    return it->second;
  return nullptr;
}

std::byte *buffer_pool::allocate() {
  // Try existing slabs, most recent first (likely to have free chunks).
  {
    std::shared_lock lk(_grow_mtx);
    for (auto it = _slabs.rbegin(); it != _slabs.rend(); ++it) {
      std::byte *p;
      if ((*it)->free_chunks.try_dequeue(p)) {
        (*it)->free_count.fetch_sub(1, std::memory_order_relaxed);
        _total_free.fetch_sub(1, std::memory_order_relaxed);
        return p;
      }
    }
  }

  // All slabs exhausted — try to grow.
  if (!grow())
    return nullptr;

  std::shared_lock lk(_grow_mtx);
  std::byte *p;
  if (_slabs.back()->free_chunks.try_dequeue(p)) {
    _slabs.back()->free_count.fetch_sub(1, std::memory_order_relaxed);
    _total_free.fetch_sub(1, std::memory_order_relaxed);
    return p;
  }
  return nullptr;
}

size_t buffer_pool::allocate_bulk(size_t n, std::vector<std::byte *> &out) {
  if (n == 0)
    return 0;

  // Scratch buffer for try_dequeue_bulk (stack-allocated for small n,
  // heap for large).
  constexpr size_t STACK_MAX = 64;
  std::byte *stack_buf[STACK_MAX];
  std::unique_ptr<std::byte *[]> heap_buf;
  std::byte **buf = stack_buf;
  if (n > STACK_MAX) {
    heap_buf = std::make_unique<std::byte *[]>(n);
    buf = heap_buf.get();
  }

  size_t remaining = n;
  size_t total_got = 0;

  auto drain_slabs = [&]() {
    std::shared_lock lk(_grow_mtx);
    for (auto it = _slabs.rbegin(); it != _slabs.rend() && remaining > 0;
         ++it) {
      auto got =
          (*it)->free_chunks.try_dequeue_bulk(buf + total_got, remaining);
      if (got > 0) {
        (*it)->free_count.fetch_sub(static_cast<uint32_t>(got),
                                    std::memory_order_relaxed);
        _total_free.fetch_sub(static_cast<uint32_t>(got),
                              std::memory_order_relaxed);
        total_got += got;
        remaining -= got;
      }
    }
  };

  drain_slabs();

  // If we still need more, grow and drain the new slab.
  while (remaining > 0) {
    if (!grow())
      break;
    drain_slabs();
  }

  out.insert(out.end(), buf, buf + total_got);
  return total_got;
}

void buffer_pool::deallocate(std::byte *p) {
  auto *s = find_slab(p);
  assert(s && "buffer_pool::deallocate: pointer does not belong to any slab");
  s->free_chunks.enqueue(p);
  s->free_count.fetch_add(1, std::memory_order_relaxed);
  _total_free.fetch_add(1, std::memory_order_relaxed);
}

// ===========================================================================
// pinned_view
// ===========================================================================

pinned_view::pinned_view(
    std::shared_ptr<cache_entry> entry,
    moodycamel::ConcurrentQueue<eviction_candidate> &candidate_queue)
    : _entry(nullptr), _candidate_queue(&candidate_queue) {
  if (!entry)
    return;
  if (!entry->state.try_acquire_read())
    return;
  _entry = std::move(entry);
}

pinned_view::~pinned_view() { unpin(); }

pinned_view::pinned_view(pinned_view &&o) noexcept
    : _entry(std::move(o._entry)), _candidate_queue(o._candidate_queue) {
  o._entry.reset();
}

pinned_view &pinned_view::operator=(pinned_view &&o) noexcept {
  if (this != &o) {
    unpin();
    _entry = std::move(o._entry);
    o._entry.reset();
    _candidate_queue = o._candidate_queue;
  }
  return *this;
}

void pinned_view::unpin() {
  if (!_entry)
    return;
  // release_read returns true only when this was the *last* active reader
  // (state transitioned in_use → cached).  That's exactly the edge that makes
  // the entry newly evictable, so post a candidate hint then and only then.
  if (_entry->state.release_read()) {
    // Silent post: no semaphore release.  The evictor drains candidates on
    // its next poll tick (EVICTOR_POLL_INTERVAL) or whenever a chunk request
    // wakes it — whichever comes first.
    _candidate_queue->enqueue({std::weak_ptr<cache_entry>(_entry)});
  }
  _entry.reset();
}

size_t pinned_view::num_chunks() const noexcept {
  return _entry ? _entry->chunks.size() : 0;
}

std::span<const std::byte> pinned_view::operator[](size_t i) const noexcept {
  if (!_entry || i >= _entry->chunks.size())
    return {};
  auto phys_size = static_cast<size_t>(_entry->physical_range.size());
  auto chunk_start = i * buffer_pool::CHUNK_BYTES;
  auto chunk_sz = std::min(buffer_pool::CHUNK_BYTES, phys_size - chunk_start);
  return {_entry->chunks[i], chunk_sz};
}

cudf::io::text::byte_range_info pinned_view::logical_range() const noexcept {
  if (!_entry)
    return {0, 0};
  return _entry->logical_range;
}

cudf::io::text::byte_range_info pinned_view::physical_range() const noexcept {
  if (!_entry)
    return {0, 0};
  return _entry->physical_range;
}

size_t pinned_view::size() const noexcept {
  return _entry ? static_cast<size_t>(_entry->logical_range.size()) : 0;
}

std::vector<cudf::io::datasource::non_owning_buffer>
pinned_view::slice(size_t offset, size_t size) const {
  std::vector<cudf::io::datasource::non_owning_buffer> result;
  if (!_entry || size == 0)
    return result;

  // Physical range starts at a potentially earlier (aligned) offset.
  // The delta tells us where logical byte 0 sits inside the physical buffer.
  auto phys_off = static_cast<size_t>(_entry->physical_range.offset());
  auto logical_off = static_cast<size_t>(_entry->logical_range.offset());
  auto phys_size = static_cast<size_t>(_entry->physical_range.size());

  // Convert logical [offset, offset+size) to physical byte position
  // within the chunked buffer.
  size_t phys_start = (offset - logical_off) + (logical_off - phys_off);
  size_t remaining = size;

  // Walk the chunks that span [phys_start, phys_start + size).
  size_t chunk_idx = phys_start / buffer_pool::CHUNK_BYTES;
  size_t off_in_chunk = phys_start % buffer_pool::CHUNK_BYTES;

  while (remaining > 0 && chunk_idx < _entry->chunks.size()) {
    auto chunk_avail = std::min(
        buffer_pool::CHUNK_BYTES - off_in_chunk,
        phys_size - chunk_idx * buffer_pool::CHUNK_BYTES - off_in_chunk);
    auto n = std::min(remaining, chunk_avail);
    auto *p = reinterpret_cast<uint8_t const *>(_entry->chunks[chunk_idx]) +
              off_in_chunk;

    // Coalesce with the previous slice if this chunk is contiguous with the
    // tail of the previous slice in the pinned host address space.  Adjacent
    // chunks within the same slab are virtually contiguous (the slab is one
    // cudaHostAlloc), which is the common case for a freshly-filled pool.
    if (!result.empty()) {
      auto const &last = result.back();
      if (last.data() + last.size() == p) {
        result.back() = cudf::io::datasource::non_owning_buffer(
            last.data(), last.size() + n);
        remaining -= n;
        ++chunk_idx;
        off_in_chunk = 0;
        continue;
      }
    }

    result.emplace_back(p, n);
    remaining -= n;
    ++chunk_idx;
    off_in_chunk = 0;
  }

  return result;
}

// ===========================================================================
// prefetching_cache — construction / destruction
// ===========================================================================

prefetching_cache::prefetching_cache(buffer_pool &pool, sirius_ioctx *io_ctx,
                                     size_t inflight_budget_chunks)
    : _pool(pool), _io_ctx(io_ctx), _inflight_budget(inflight_budget_chunks),
      _evictor_thread(
          [this](std::stop_token st) { evictor_loop(std::move(st)); }),
      _worker_thread(
          [this](std::stop_token st) { worker_loop(std::move(st)); }) {}

prefetching_cache::~prefetching_cache() {
  _worker_thread.request_stop();
  _evictor_thread.request_stop();
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_one();
  _request_sem.release();
}

void prefetching_cache::enqueue_work(work_item item) {
  _work_queue.enqueue(std::move(item));
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_one();
}

void prefetching_cache::release_chunks(cache_entry &entry) {
  for (auto *p : entry.chunks)
    _pool.deallocate(p);
  auto freed = entry.chunks.size() * buffer_pool::CHUNK_BYTES;
  _allocated_bytes.fetch_sub(freed, std::memory_order_relaxed);
  entry.chunks.clear();
}

// ===========================================================================
// find_entry — binary search + hit/miss classification
// ===========================================================================

std::shared_ptr<cache_entry> prefetching_cache::find_entry(
    const std::vector<std::shared_ptr<cache_entry>> &entries, size_t offset,
    size_t size) {
  // upper_bound: first entry whose offset > requested offset.  The candidate
  // is pos-1 (the last entry whose offset <= requested offset).
  auto pos = std::upper_bound(
      entries.begin(), entries.end(), offset, [](size_t off, auto const &e) {
        return off < static_cast<size_t>(e->logical_range.offset());
      });

  // No entry starts at or before our offset — nothing covers us.
  if (pos == entries.begin()) {
    _full_miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  --pos;
  auto entry_end = static_cast<size_t>((*pos)->logical_range.offset()) +
                   static_cast<size_t>((*pos)->logical_range.size());

  // Candidate ends before our offset — no overlap at all.
  if (entry_end <= offset) {
    _full_miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  // Candidate overlaps but doesn't fully contain the requested tail.
  if (offset + size > entry_end) {
    _partial_miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  return *pos;
}

// ===========================================================================
// insert
// ===========================================================================

void prefetching_cache::insert(
    const sirius_io_object &obj,
    std::shared_ptr<sirius_io_object_metadata> metadata,
    const std::vector<cudf::io::text::byte_range_info> &ranges) {
  CTRACK_NAME("cache::insert");

  assert(std::is_sorted(ranges.begin(), ranges.end(),
                        [](auto const &a, auto const &b) {
                          return a.offset() < b.offset();
                        }) &&
         "ranges must be sorted by offset");

  auto const &key = obj.raw_file_cache_id();
  auto file_size = obj.size();

  // _cache_age only advances via refresh_cache().  Sample it here so every
  // range stamped in this call shares the same request_ts — they belong to
  // the same epoch.
  auto tick = _cache_age.load(std::memory_order_relaxed);

  std::unique_lock map_lk(_map_mtx);
  auto [it, inserted] = _file_cache.try_emplace(key, nullptr);
  if (inserted)
    it->second = std::make_unique<file_entry>();
  auto &file = *it->second;

  std::unique_lock file_lk(file.mtx);
  map_lk.unlock();

  file.io_obj = &obj;
  file.file_size = file_size;
  file.metadata = std::move(metadata);

  // Entries whose load this call is responsible for scheduling.  Only entries
  // for which try_start_queueing() succeeded end up here, so there is no
  // chance of double-queueing the same entry.
  std::vector<std::shared_ptr<cache_entry>> new_entries;

  auto try_claim = [&](std::shared_ptr<cache_entry> const &e) {
    if (e->state.try_start_queueing())
      new_entries.push_back(e);
  };

  // Linear merge: both `file.entries` and `ranges` are sorted by offset.
  // Build a new sorted vector in one O(N + K) pass and swap it in, instead of
  // repeated O(N) vector::insert calls.
  std::vector<std::shared_ptr<cache_entry>> merged;
  merged.reserve(file.entries.size() + ranges.size());
  new_entries.reserve(ranges.size());

  auto ex_it = file.entries.begin();
  auto ex_end = file.entries.end();

  for (auto const &logical : ranges) {
    auto off = logical.offset();
    // Forward existing entries that sort before this range.
    while (ex_it != ex_end && (*ex_it)->logical_range.offset() < off) {
      merged.push_back(std::move(*ex_it));
      ++ex_it;
    }
    if (ex_it != ex_end && (*ex_it)->logical_range.offset() == off) {
      auto &existing = *ex_it;
      existing->n_total_request.fetch_add(1, std::memory_order_relaxed);
      existing->request_ts.store(tick, std::memory_order_release);
      try_claim(existing);
      merged.push_back(std::move(existing));
      ++ex_it;
    } else {
      auto physical = _io_ctx->compute_physical_range(logical, file_size);
      auto e = std::make_shared<cache_entry>(logical, physical);
      e->n_total_request.fetch_add(1, std::memory_order_relaxed);
      e->request_ts.store(tick, std::memory_order_release);
      try_claim(e);
      merged.push_back(std::move(e));
    }
  }
  // Forward any trailing existing entries.
  for (; ex_it != ex_end; ++ex_it)
    merged.push_back(std::move(*ex_it));

  file.entries = std::move(merged);

  // Account for chunks we're about to load.  Only claimed entries contribute;
  // duplicates whose queueing CAS lost are already being loaded by someone
  // else, so they don't double-count.
  for (auto const &e : new_entries) {
    _pending_chunks.fetch_add((static_cast<size_t>(e->physical_range.size()) +
                               buffer_pool::CHUNK_BYTES - 1) /
                                  buffer_pool::CHUNK_BYTES,
                              std::memory_order_relaxed);
  }

  file_lk.unlock();

  if (!new_entries.empty())
    enqueue_work(prefetch_req{key, &obj, std::move(new_entries)});
}

// ===========================================================================
// read — non-blocking, single range by offset
// ===========================================================================

pinned_view prefetching_cache::read(const sirius_io_object &obj, size_t offset,
                                    size_t size) {
  CTRACK_NAME("cache::read");
  auto const &key = obj.raw_file_cache_id();

  std::shared_lock map_lk(_map_mtx);
  auto it = _file_cache.find(key);
  if (it == _file_cache.end()) {
    _full_miss_count.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  auto &file = *it->second;

  std::shared_lock file_lk(file.mtx);
  map_lk.unlock();

  auto entry = find_entry(file.entries, offset, size);
  if (!entry)
    return {};

  // Dispatch on state:
  //   cached / in_use  → pin immediately (stamps consumption_ts).
  //   loading          → wait for the worker to resolve the load, then retry.
  //   queued / evicting / empty → return empty (caller falls back).
  while (true) {
    auto st = entry->state.get_state();
    if (st == entry_state::cached || st == entry_state::in_use) {
      pinned_view view{entry, _candidate_queue};
      if (view) {
        _hit_count.fetch_add(1, std::memory_order_relaxed);
        entry->consumption_ts.store(_cache_age.load(std::memory_order_relaxed),
                                    std::memory_order_release);
        return view;
      }
      // try_acquire_read lost a race; re-observe the state.
      continue;
    }
    if (st == entry_state::loading) {
      entry->state.wait_while_loading();
      continue;
    }
    _partial_miss_count.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
}

// ===========================================================================
// read_ranges — non-blocking batch read
// ===========================================================================

std::vector<pinned_view> prefetching_cache::read_ranges(
    const sirius_io_object &obj,
    const std::vector<cudf::io::text::byte_range_info> &ranges) {
  auto const &key = obj.raw_file_cache_id();

  std::shared_lock map_lk(_map_mtx);
  auto it = _file_cache.find(key);
  if (it == _file_cache.end()) {
    _full_miss_count.fetch_add(ranges.size(), std::memory_order_relaxed);
    return std::vector<pinned_view>(ranges.size());
  }
  auto &file = *it->second;

  std::shared_lock file_lk(file.mtx);
  map_lk.unlock();

  std::vector<pinned_view> result;
  result.reserve(ranges.size());

  for (auto const &r : ranges) {
    auto entry = find_entry(file.entries, static_cast<size_t>(r.offset()),
                            static_cast<size_t>(r.size()));
    if (!entry) {
      result.emplace_back();
      continue;
    }
    // Same dispatch as read(): pin on cached/in_use, wait on loading,
    // give up on queued/evicting/empty.
    pinned_view view;
    while (true) {
      auto st = entry->state.get_state();
      if (st == entry_state::cached || st == entry_state::in_use) {
        pinned_view v{entry, _candidate_queue};
        if (v) {
          _hit_count.fetch_add(1, std::memory_order_relaxed);
          entry->consumption_ts.store(
              _cache_age.load(std::memory_order_relaxed),
              std::memory_order_release);
          view = std::move(v);
        }
        if (view)
          break;
        continue;
      }
      if (st == entry_state::loading) {
        entry->state.wait_while_loading();
        continue;
      }
      _partial_miss_count.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    result.push_back(std::move(view));
  }
  return result;
}

void prefetching_cache::refresh_cache() {
  _cache_age.fetch_add(1, std::memory_order_relaxed);
}

std::string prefetching_cache::summary() const {
  auto hits = _hit_count.load(std::memory_order_relaxed);
  auto partial = _partial_miss_count.load(std::memory_order_relaxed);
  auto full = _full_miss_count.load(std::memory_order_relaxed);
  auto total = hits + partial + full;
  auto pct = [&](uint64_t n) {
    return total > 0
               ? (100.0 * static_cast<double>(n) / static_cast<double>(total))
               : 0.0;
  };
  return fmt::format(
      "prefetching_cache: {} reads ({} hit {:.1f}%, {} partial-miss {:.1f}%, "
      "{} full-miss {:.1f}%); pool {}/{} chunks free, {:.0f} MB allocated, "
      "{} chunks pending",
      total, hits, pct(hits), partial, pct(partial), full, pct(full),
      _pool.free_count(), _pool.total_chunks(),
      static_cast<double>(_allocated_bytes.load(std::memory_order_relaxed)) /
          (1024.0 * 1024.0),
      _pending_chunks.load(std::memory_order_relaxed));
}

// ===========================================================================
// evictor_loop
// ===========================================================================

void prefetching_cache::evictor_loop(std::stop_token stop) {
  std::stop_callback stop_cb(stop, [this] { _request_sem.release(); });

  // Raw bucket score before clamping.  Lower means colder.
  //   >= 0    → lives in some bucket [0, NUM-1] after clamp
  //   < -5    → extremely cold — drain_candidates evicts immediately
  auto compute_raw_bucket = [this](cache_entry const &e) -> int64_t {
    int64_t n = e.n_total_request.load(std::memory_order_relaxed);
    int64_t age = _cache_age.load(std::memory_order_relaxed);
    return static_cast<int64_t>(NUM_LRU_BUCKETS - 1) + n - age;
  };

  // Clamped bucket index for entries that are actually getting placed in a
  // bucket.  Non-(consumed_or_stale) entries get floor 1 so they're never
  // first to be evicted.
  auto compute_bucket_idx = [&](cache_entry const &e) -> int {
    uint64_t age =
        static_cast<uint64_t>(_cache_age.load(std::memory_order_relaxed));
    int64_t raw = compute_raw_bucket(e);
    int64_t floor_v =
        e.consumption_ts.load(std::memory_order_relaxed) == age ? 1 : -10;
    int64_t v = std::max(raw, floor_v);
    v = std::min(v, static_cast<int64_t>(NUM_LRU_BUCKETS - 1));
    return static_cast<int>(v);
  };

  // Evict one entry's chunks back to the pool.  Returns the number of chunks
  // freed, or 0 on race (someone pinned/evicted it concurrently).
  auto evict_to_pool = [this](cache_entry *entry) -> size_t {
    if (!entry->state.try_start_evicting())
      return 0;
    size_t n = entry->chunks.size();
    for (auto *p : entry->chunks)
      _pool.deallocate(p);
    entry->chunks.clear();
    _allocated_bytes.fetch_sub(n * buffer_pool::CHUNK_BYTES,
                               std::memory_order_relaxed);
    entry->bucket_idx.store(-1, std::memory_order_relaxed);
    entry->state.mark_evicted();
    return n;
  };

  // Drain candidates.  Extremely cold entries (raw bucket < -5) are evicted
  // on the spot; the rest are placed into their clamped bucket.  Returns the
  // number of chunks freed to the pool during this drain.
  auto drain_candidates = [&]() -> size_t {
    size_t freed = 0;
    eviction_candidate cand;
    while (_candidate_queue.try_dequeue(cand)) {
      auto entry = cand.entry.lock();
      if (!entry)
        continue;
      auto st = entry->state.get_state();
      if (st != entry_state::cached && st != entry_state::in_use)
        continue;
      if (entry->bucket_idx.load(std::memory_order_relaxed) != -1)
        continue;

      int64_t raw = compute_raw_bucket(*entry);
      uint64_t age =
          static_cast<uint64_t>(_cache_age.load(std::memory_order_relaxed));

      // Opportunistic immediate eviction for very cold consumed/stale
      // entries.  Avoids the queue→bucket→walk round-trip when the answer
      // is obvious.
      if (raw < -5 && entry->is_consumed_or_stale(age) &&
          st == entry_state::cached) {
        freed += evict_to_pool(entry.get());
        continue;
      }

      int idx = compute_bucket_idx(*entry);
      entry->bucket_idx.store(idx, std::memory_order_relaxed);
      _lru_buckets[static_cast<size_t>(idx)].push_back(entry);
    }
    return freed;
  };

  while (!stop.stop_requested()) {
    bool have_signal = _request_sem.try_acquire_for(EVICTOR_POLL_INTERVAL);

    size_t reclaimed = drain_candidates();

    if (!have_signal)
      continue;
    if (stop.stop_requested())
      break;

    eviction_request req;
    if (!_request_queue.try_dequeue(req))
      continue; // spurious wake (e.g., destructor release)

    CTRACK_NAME("cache::evictor_reclaim");

    size_t needed = req.n_chunks_needed;
    uint64_t age =
        static_cast<uint64_t>(_cache_age.load(std::memory_order_relaxed));

    // Walk buckets coldest-first.  Entries that are no longer evictable get
    // moved to the right bucket or dropped; evictable ones have their chunks
    // returned to the pool.
    for (size_t b = 0; b < NUM_LRU_BUCKETS && reclaimed < needed; ++b) {
      auto &bucket = _lru_buckets[b];
      while (!bucket.empty() && reclaimed < needed) {
        auto entry = bucket.back().lock();
        bucket.pop_back();

        if (!entry) {
          continue;
        }

        entry->bucket_idx.store(-1, std::memory_order_relaxed);

        if (entry->state.get_state() != entry_state::cached or
            !entry->is_consumed_or_stale(age)) {
          continue;
        }

        int new_idx = compute_raw_bucket(*entry);
        if (new_idx > b) {
          entry->bucket_idx.store(new_idx, std::memory_order_relaxed);
          _lru_buckets[static_cast<size_t>(new_idx)].push_back(entry);
          continue;
        }

        size_t n = evict_to_pool(entry.get());
        if (n == 0) {
          // Raced — leave in bucket, move on.
          continue;
        }
        reclaimed += n;
      }
    }

    if (reclaimed >= needed) {
      req.promise.set_value();
    } else {
      req.promise.set_exception(std::make_exception_ptr(
          std::runtime_error("eviction could not free enough chunks")));
    }
  }
}

// ===========================================================================
// worker_loop
// ===========================================================================

void prefetching_cache::worker_loop(std::stop_token stop) {
  std::stop_callback stop_cb(stop, [this] {
    _work_seq.fetch_add(1, std::memory_order_release);
    _work_seq.notify_one();
  });

  while (!stop.stop_requested()) {
    work_item item;
    if (!_work_queue.try_dequeue(item)) {
      auto seq = _work_seq.load(std::memory_order_acquire);
      if (!_work_queue.try_dequeue(item)) {
        _work_seq.wait(seq, std::memory_order_relaxed);
        continue;
      }
    }

    CTRACK_NAME("cache::worker_iter");

    // ---- prefetch_req ------------------------------------------------------
    std::vector<std::shared_ptr<cache_entry>> batch;
    bool pool_exhausted = false;

    for (auto const &entry : item.entries) {
      if (pool_exhausted)
        break;

      if (!entry->state.try_start_loading())
        continue;

      auto phys_size = static_cast<size_t>(entry->physical_range.size());
      auto n_chunks =
          (phys_size + buffer_pool::CHUNK_BYTES - 1) / buffer_pool::CHUNK_BYTES;
      _pending_chunks.fetch_sub(n_chunks, std::memory_order_relaxed);

      std::vector<std::byte *> ptrs;
      auto got = _pool.allocate_bulk(n_chunks, ptrs);
      if (got < n_chunks) {
        // Return partial allocation and ask evictor to free chunks back to
        // the pool.  The evictor resolves the promise once it has freed at
        // least n_chunks chunks; we then re-allocate from the pool.
        for (auto *p : ptrs)
          _pool.deallocate(p);
        ptrs.clear();

        eviction_request req;
        req.n_chunks_needed = n_chunks;
        auto fut = req.promise.get_future();
        _request_queue.enqueue(std::move(req));
        _request_sem.release();

        try {
          fut.get();
        } catch (...) {
          entry->state.mark_load_failed();
          spdlog::warn(
              "prefetching_cache: eviction could not free enough for {}",
              item.file_key);
          pool_exhausted = true;
          continue;
        }

        got = _pool.allocate_bulk(n_chunks, ptrs);
        if (got < n_chunks) {
          // Race: another allocator grabbed the freed chunks before us.
          for (auto *p : ptrs)
            _pool.deallocate(p);
          entry->state.mark_load_failed();
          pool_exhausted = true;
          continue;
        }
      }

      _allocated_bytes.fetch_add(ptrs.size() * buffer_pool::CHUNK_BYTES,
                                 std::memory_order_relaxed);
      entry->chunks = std::move(ptrs);
      batch.push_back(entry);
    }

    if (batch.empty())
      continue;

    // Total chunks in this batch — what we'll acquire from the in-flight
    // budget and release when IO completes.
    size_t batch_chunks = 0;
    for (auto const &e : batch)
      batch_chunks += e->chunks.size();

    // Acquire the budget before dispatching IO.  Blocks until the batch fits,
    // or returns a disengaged slot if shutdown was requested mid-wait.
    auto budget_slot = _inflight_budget.acquire(batch_chunks, stop);
    if (!budget_slot) {
      for (auto const &e : batch) {
        release_chunks(*e);
        e->state.mark_load_failed();
      }
      break;
    }

    // Build IO arguments: one sub-read per chunk.
    std::vector<cudf::io::text::byte_range_info> io_ranges;
    std::vector<cudf::host_span<std::byte>> io_dsts;

    for (auto const &e : batch) {
      auto phys_off = static_cast<size_t>(e->physical_range.offset());
      auto phys_size = static_cast<size_t>(e->physical_range.size());
      for (size_t i = 0; i < e->chunks.size(); ++i) {
        auto off = phys_off + i * buffer_pool::CHUNK_BYTES;
        auto sz = std::min(buffer_pool::CHUNK_BYTES,
                           phys_size - i * buffer_pool::CHUNK_BYTES);
        io_ranges.emplace_back(static_cast<int64_t>(off),
                               static_cast<int64_t>(sz));
        io_dsts.emplace_back(e->chunks[i], sz);
      }
    }

    // io_completion_handler is std::function (copy-constructible), so a
    // move-only slot cannot be captured by value.  Wrap it in a shared_ptr
    // whose destructor releases the budget when the callback is discarded.
    auto slot_holder =
        std::make_shared<admission_control::slot>(std::move(budget_slot));

    {
      CTRACK_NAME("cache::worker_dispatch_io");
      _io_ctx->host_read_ranges_async(
          const_cast<sirius_io_object &>(*item.io_obj), io_ranges, io_dsts,
          [this, batch = std::move(batch), slot = std::move(slot_holder),
           key = std::move(item.file_key)](size_t /*bytes*/,
                                           std::exception_ptr ep) {
            if (ep) {
              try {
                std::rethrow_exception(ep);
              } catch (std::exception const &ex) {
                spdlog::error("prefetching_cache: IO failed for {}: {}", key,
                              ex.what());
              }
              for (auto const &e : batch) {
                release_chunks(*e);
                e->state.mark_load_failed();
              }
            } else {
              for (auto const &e : batch)
                e->state.mark_cached();
            }
            // `slot` destructs here, returning budget to admission_control.
          });
    }
  }
}

} // namespace sirius::io
