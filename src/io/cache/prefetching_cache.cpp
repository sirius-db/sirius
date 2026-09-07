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

#include "cuda/device_copy_batch.hpp"
#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/cache/types.hpp"
#include "io/io_context.hpp"
#include "io/io_request.hpp"
#include "io/prefetch_census.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "util/error_utils.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <ctrack.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <format>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::io::cache {

namespace {
using size_terminal = exec::invocable<void(exec::try_t<std::size_t>&&) &&>;
}

struct prefetching_cache::cached_copy_retirement {
  std::vector<cached_chunk*> pins;
  std::shared_ptr<exec::completion_controller::slot> lifetime;
  std::shared_ptr<grouped_coordinator> coordinator;
  std::exception_ptr async_error;
  std::latch published{1};
  cudaEvent_t event{};
  cudaStream_t stream{};
  int device_id{-1};
  bool enqueue_ok{false};
  bool event_recorded{false};
  bool pins_released{false};

  ~cached_copy_retirement()
  {
    release_pins();
    if (event == nullptr) { return; }
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      std::ignore = cudaEventDestroy(event);
      event       = nullptr;
    } catch (...) {  // Best effort during prepublication rollback.
    }
  }

  void release_pins() noexcept
  {
    if (pins_released) { return; }
    for (auto* chunk : pins) {
      chunk->state.release_read();
    }
    pins_released = true;
  }

  void wait_and_finish() noexcept
  {
    published.wait();
    cudaError_t wait_status = cudaErrorUnknown;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      wait_status = event_recorded ? cudaEventSynchronize(event) : cudaStreamSynchronize(stream);
      if (event != nullptr) {
        std::ignore = cudaEventDestroy(event);
        event       = nullptr;
      }
    } catch (...) {
      wait_status = cudaErrorUnknown;
    }
    finish(enqueue_ok && wait_status == cudaSuccess);
  }

  void finish(bool copy_ok) noexcept
  {
    release_pins();
    if (copy_ok) {
      coordinator->on_complete();
    } else {
      coordinator->report_error(async_error);
    }
  }
};

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

producer_stage::value prefetching_handle::producer_state() const noexcept
{
  return _req.producer ? _req.producer->get() : producer_stage::initialized;
}

bool prefetching_handle::is_prefetch_in_flight() const noexcept
{
  return _req.producer && _req.producer->get() == producer_stage::loading;
}

bool prefetching_handle::has_started_reading() const noexcept
{
  return _req.consumer && _req.consumer->get() >= consumer_stage::reading;
}

bool prefetching_handle::wait_until_ready() noexcept
{
  if (!_req.producer) { return false; }
  return _req.producer->wait_till_not_loading();
}

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
    _armed(_io_ctx->can_use_prefetching_cache())
{
  _chunk_size          = _pool->chunk_size();
  auto const max_bytes = static_cast<size_t>(chunk_state::max_fill_pages()) * io::IO_BLOCK_SIZE;
  if (_chunk_size == 0) {
    throw std::invalid_argument("prefetching_cache: chunk size must be non-zero");
  }
  if (_chunk_size > max_bytes) {
    throw std::invalid_argument(
      std::format("prefetching_cache: chunk size {} exceeds the {}-byte packed fill limit",
                  _chunk_size,
                  max_bytes));
  }

  _evictor_thread = std::jthread([this](const std::stop_token& st) { evict_loop(st); },
                                 _evictor_stop_source.get_token());
}

exec::completion_controller::slot prefetching_cache::acquire_inflight_io() noexcept
{
  std::lock_guard lock(_inflight_io_mtx);
  if (_shutting_down.load(std::memory_order_acquire)) { return {}; }
  return _inflight_io.acquire();
}

void prefetching_cache::drain_inflight_io() noexcept
{
  std::latch drained{1};
  // Armed before close(), so both orderings land exactly once: with no IO
  // outstanding the callback runs inline on close(), otherwise the last slot to
  // drop fires it from its IO thread.
  auto subscription = _inflight_io.on_completion([&drained] { drained.count_down(); });
  {
    std::lock_guard lock(_inflight_io_mtx);
    _inflight_io.close();
  }
  drained.wait();
}

prefetching_cache::~prefetching_cache()
{
  _shutting_down.store(true, std::memory_order_release);
  _evictor_stop_source.request_stop();

  // The IO completions write into file entries this object owns, so they have
  // to have run before any of it is torn down.
  drain_inflight_io();
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
      auto const n_slots =
        obj.size() / _chunk_size + static_cast<size_t>(obj.size() % _chunk_size != 0);
      it->second->slots.assign(n_slots, nullptr);
    }
  }
  return *it->second;
}

prefetching_handle prefetching_cache::initiate_prefetching_request(
  const io_object& obj, std::span<const byte_range> ranges, std::optional<int> gpu_id)
{
  if (!_armed) { return prefetching_handle(); }

  auto& file = get_or_create_file_entry(obj);

  const size_t chunk_bytes = _chunk_size;

  // Normalise the request set to what the backend actually addresses BEFORE
  // mapping it onto chunks.
  //
  // Widen each range to the backend's alignment (a page for an O_DIRECT file, a
  // single byte for an object store), then fuse ranges close enough that
  // bridging the gap beats issuing a second request.  Doing it here, once, is
  // what stops a chunk being touched by several small ranges anchored to
  // opposite edges -- `merge` folds those to a whole-chunk fill, and the whole
  // chunk is then read to serve a few kilobytes.
  auto const alignment = std::max<size_t>(1, _io_ctx->min_alignment_requirement());
  auto const gap       = _io_ctx->merge_gap_size();

  std::vector<std::pair<size_t, size_t>> spans;  // [lo, hi), aligned and EOF-clamped
  spans.reserve(ranges.size());
  for (const auto& r : ranges) {
    if (r.offset() < 0 || r.size() <= 0) { continue; }
    auto const lo = static_cast<size_t>(r.offset());
    if (lo >= obj.size()) { continue; }

    auto const requested    = static_cast<size_t>(r.size());
    auto const logical_size = std::min(requested, obj.size() - lo);
    auto const logical_hi   = lo + logical_size;
    auto const aligned_lo   = lo - (lo % alignment);
    auto aligned_hi         = logical_hi;
    if (auto const remainder = aligned_hi % alignment; remainder != 0) {
      auto const delta = alignment - remainder;
      aligned_hi       = delta > obj.size() - aligned_hi ? obj.size() : aligned_hi + delta;
    }
    if (aligned_hi > aligned_lo) { spans.emplace_back(aligned_lo, aligned_hi); }
  }
  std::ranges::sort(spans);

  std::vector<std::pair<size_t, size_t>> merged;
  merged.reserve(spans.size());
  for (auto const& [lo, hi] : spans) {
    // `<=` so exactly-adjacent ranges fuse even at gap 0.
    if (!merged.empty() && (lo <= merged.back().second || lo - merged.back().second <= gap)) {
      merged.back().second = std::max(merged.back().second, hi);
    } else {
      merged.emplace_back(lo, hi);
    }
  }

  // Enumerate the chunk-aligned positions the merged ranges touch, together
  // with the extent each one is actually wanted for.  Still derived from the
  // ranges rather than a chunk-aligned coalesce of them: coalescing to chunk
  // granularity would pull in whole chunks that no range touches, and since
  // every chunk issues its own IO segment sized to its extent, those chunks
  // would be pure over-read.
  std::vector<std::pair<size_t, chunk_fill>> wanted;
  for (auto const& [lo, hi] : merged) {
    auto off = (lo / chunk_bytes) * chunk_bytes;
    while (off < hi) {
      wanted.emplace_back(off, needed_fill(off, chunk_bytes, lo, hi));
      if (chunk_bytes > std::numeric_limits<size_t>::max() - off) { break; }
      off += chunk_bytes;
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
  _eviction_queue.enqueue(cache_request{req});

  return prefetching_handle(std::move(req));
}

bool prefetching_cache::prepare_request(prefetch_request& req, bool wait_for_eviction)
{
  // Never allocate for a request the consumer has already moved past.  A request
  // is handed to the evictor when it is created, and once its consumer disposes
  // the evictor releases its subscriber references and retires it from the
  // batch -- so buffers attached after that point are named by nothing, and
  // nothing will ever reclaim them.  Abandon instead: terminal and notified, so
  // no waiter is stranded on the transient `preparing` stage.
  if (!req.chunks || req.has_fallen_behind()) {
    req.producer->mark_abandoned();
    return false;
  }

  if (!req.producer->mark_preparing()) { return false; }

  auto const& chunks = *req.chunks;

  std::size_t n_chunks_needed = std::ranges::count_if(
    chunks, [](cached_chunk* c) { return c->state.get_state() == chunk_state::empty; });

  int numa_allocated = req.preferred_numa;
  auto buffers       = _pool->allocate_bulk(n_chunks_needed, numa_allocated);
  if (buffers.size() != n_chunks_needed) {
    if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa_allocated); }
    req.producer->mark_abandoned();
    return false;
  }

  for (auto* c : chunks) {
    if (buffers.empty()) { break; }
    if (c->state.mark_queued()) {
      auto* buffer = buffers.back();
      buffers.pop_back();
      c->data      = reinterpret_cast<uint8_t*>(buffer);
      c->numa_node = numa_allocated;
      if (!c->state.mark_allocated()) { buffers.push_back(buffer); }
    }
  }

  if (!buffers.empty()) { _pool->deallocate_bulk(std::move(buffers), numa_allocated); }

  std::ignore = req.producer->mark_prepared();
  return true;
}

std::vector<cached_chunk*> prefetching_cache::ranges_in_cache(const io_object& obj,
                                                              size_t offset,
                                                              size_t size,
                                                              coverage_policy policy,
                                                              prefetching_handle* out_handle) const
{
  std::vector<cached_chunk*> chunks;
  if (out_handle && *out_handle) {
    if (auto requested = out_handle->chunks()) {
      chunks = find_entry(*requested, offset, size, policy, _chunk_size);
    }
  }
  if (!chunks.empty()) { return chunks; }

  std::shared_lock lk(_map_mtx);
  auto const it = _file_cache.find(obj.raw_file_cache_id());
  if (it == _file_cache.end()) { return {}; }
  auto* file = it->second.get();
  lk.unlock();
  return file->fetch_chunks(offset, size, policy);
}

exec::semi_future<std::size_t> prefetching_cache::host_read_async(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, prefetching_handle* out_handle)
{
  if (size == 0) { return exec::make_semi_future<std::size_t>(0); }
  if (dst == nullptr) {
    return exec::make_semi_future<std::size_t>(
      std::make_exception_ptr(std::invalid_argument("host read destination is null")));
  }
  if (offset >= obj.size()) { return exec::make_semi_future<std::size_t>(0); }
  size = std::min(size, obj.size() - offset);
  slice request{offset, size, dst};
  return host_read_ranges_async(obj, std::span<slice const>{&request, 1}, out_handle);
}

std::size_t prefetching_cache::host_read(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, prefetching_handle* out_handle)
{
  auto future = host_read_async(obj, offset, size, dst, out_handle);
  return std::move(future).get();
}

exec::semi_future<std::size_t> prefetching_cache::host_read_ranges_async(
  const io_object& obj, std::span<const slice> requests, prefetching_handle* out_handle)
{
  if (requests.empty()) { return exec::make_semi_future<std::size_t>(0); }

  struct hit_copy {
    cached_chunk* chunk;
    range requested;
    std::uint8_t* dst;
  };

  std::vector<cached_chunk*> claimed;
  std::vector<hit_copy> hits;
  try {
    std::vector<prepared_io_slice> prepared;
    std::size_t logical_bytes = 0;
    std::size_t n_hits        = 0;
    std::size_t n_loads       = 0;
    std::size_t n_misses      = 0;
    auto admission            = acquire_inflight_io();
    if (!admission) { throw std::runtime_error("prefetching_cache is shutting down"); }
    auto lifetime = std::make_shared<exec::completion_controller::slot>(std::move(admission));

    for (auto const& raw : requests) {
      if (raw.size() == 0) { continue; }
      if (raw.dst == nullptr) { throw std::invalid_argument("host readv destination is null"); }
      if (raw.offset() >= obj.size()) { continue; }
      auto const request_size = std::min(raw.size(), obj.size() - raw.offset());
      if (request_size > std::numeric_limits<std::size_t>::max() - logical_bytes) {
        throw std::overflow_error("host cache read byte count overflow");
      }
      logical_bytes += request_size;

      range const request_rng{raw.offset(), request_size};
      auto chunks = ranges_in_cache(
        obj, request_rng.offset, request_rng.size, coverage_policy::partial, out_handle);
      std::size_t ci       = 0;
      auto const first     = (request_rng.offset / _chunk_size) * _chunk_size;
      auto const last      = ((request_rng.end() - 1) / _chunk_size) * _chunk_size;
      auto const positions = ((last - first) / _chunk_size) + 1;

      for (std::size_t position = 0; position < positions; ++position) {
        auto const off = first + position * _chunk_size;
        while (ci < chunks.size() && chunks[ci]->offset < off) {
          ++ci;
        }
        cached_chunk* chunk =
          ci < chunks.size() && chunks[ci]->offset == off ? chunks[ci] : nullptr;
        range const needed = intersect(request_rng, range{off, _chunk_size});
        auto* piece_dst    = raw.dst + (needed.offset - request_rng.offset);

        if (chunk != nullptr &&
            chunk->state.try_pin_covering(off, _chunk_size, needed.offset, needed.end())) {
          hits.push_back(hit_copy{chunk, needed, piece_dst});
          ++n_hits;
          continue;
        }

        chunk_fill fill;
        if (chunk != nullptr && _io_ctx->supports_vector_host_read() &&
            chunk->state.take_loading_merging(
              needed_fill(off, _chunk_size, needed.offset, needed.end()), fill)) {
          claimed.push_back(chunk);
          auto completion = std::make_shared<prepared_io_completion>(
            [this, needed, piece_dst, lifetime](std::span<cached_chunk* const> completed,
                                                bool host_ok) noexcept {
              std::ignore = lifetime;
              if (!host_ok) {
                for (auto* current : completed) {
                  std::ignore = current->state.mark_load_failed();
                }
                return;
              }
              for (auto* current : completed) {
                auto const copy_rng = intersect(needed, range{current->offset, _chunk_size});
                if (!copy_rng.empty()) {
                  std::memcpy(piece_dst + (copy_rng.offset - needed.offset),
                              current->data + (copy_rng.offset - current->offset),
                              copy_rng.size);
                }
              }
              for (auto* current : completed) {
                std::ignore = current->state.mark_cached();
              }
            });
          prepared_io_slice io_slice{needed, host_buffer{std::vector<cached_chunk*>{chunk}}};
          io_slice.on_complete = std::move(completion);
          prepared.push_back(std::move(io_slice));
          ++n_loads;
          continue;
        }

        prepared.emplace_back(needed, host_buffer{piece_dst});
        ++n_misses;
      }
    }

    bool const has_backend = !prepared.empty();
    exec::semi_future<std::size_t> result_future;
    if (has_backend) {
      auto coordinator = std::make_shared<grouped_coordinator>(logical_bytes, 1);
      result_future    = coordinator->get_future();
      size_terminal terminal{[coordinator](exec::try_t<std::size_t>&& result) mutable noexcept {
        if (result.has_exception()) {
          coordinator->report_error(std::move(result).exception());
        } else {
          coordinator->on_complete();
        }
      }};

      auto io_future = _io_ctx->host_device_readv_async_io(obj, std::move(prepared));
      claimed.clear();
      std::move(io_future).install_callback(std::move(terminal));
    } else {
      claimed.clear();
    }

    // Dispatch first, then satisfy resident pieces while the backend is already
    // filling the misses. A read pin keeps each source stable for the memcpy.
    for (auto const& copy : hits) {
      std::memcpy(copy.dst,
                  copy.chunk->data + (copy.requested.offset - copy.chunk->offset),
                  copy.requested.size);
      copy.chunk->state.release_read();
    }
    hits.clear();

    _counters.hits.fetch_add(n_hits, std::memory_order_relaxed);
    _counters.h2d.fetch_add(n_loads, std::memory_order_relaxed);
    _counters.misses.fetch_add(n_misses, std::memory_order_relaxed);

    if (has_backend) { return std::move(result_future); }
    return exec::make_semi_future<std::size_t>(logical_bytes);
  } catch (...) {
    for (auto const& copy : hits) {
      copy.chunk->state.release_read();
    }
    for (auto* chunk : claimed) {
      std::ignore = chunk->state.mark_load_failed();
    }
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
}

exec::semi_future<std::size_t> prefetching_cache::device_read_async(const io_object& obj,
                                                                    size_t offset,
                                                                    size_t size,
                                                                    uint8_t* dst,
                                                                    rmm::cuda_stream_view stream,
                                                                    prefetching_handle* out_handle)
{
  CTRACK_NAME("cache::device_read_async");
  if (size == 0) { return exec::make_semi_future<std::size_t>(0); }
  if (dst == nullptr) {
    return exec::make_semi_future<std::size_t>(
      std::make_exception_ptr(std::invalid_argument("device read destination is null")));
  }
  if (offset >= obj.size()) { return exec::make_semi_future<std::size_t>(0); }
  size = std::min(size, obj.size() - offset);
  slice request{offset, size, dst};
  return device_read_ranges_async(obj, std::span<slice const>{&request, 1}, stream, out_handle);
}

exec::semi_future<std::size_t> prefetching_cache::device_read_ranges_async(
  const io_object& obj,
  std::span<const io::slice> requests,
  rmm::cuda_stream_view stream,
  prefetching_handle* out_handle)
{
  if (requests.empty()) { return exec::make_semi_future<std::size_t>(0); }

  bool const cache_while_reading = _io_ctx->supports_host_to_device_read();
  auto const device_id           = rmm::get_current_cuda_device();

  std::vector<cached_chunk*> pinned;
  std::vector<cached_chunk*> claimed;
  sirius::cuda::device_copy_batch cached_copies;
  std::vector<prepared_io_slice> prepared;
  std::size_t logical_bytes = 0;
  std::size_t reads         = 0;
  std::size_t hits          = 0;
  std::size_t loads         = 0;
  std::size_t misses        = 0;

  std::shared_ptr<exec::completion_controller::slot> lifetime;
  std::shared_ptr<prepared_io_completion> completion;
  std::shared_ptr<cached_copy_retirement> retirement;
  std::shared_ptr<grouped_coordinator> coordinator;
  exec::semi_future<std::size_t> result_future;
  size_terminal backend_terminal;
  bool has_backend{false};
  bool has_cached{false};

  try {
    auto admission = acquire_inflight_io();
    if (!admission) { throw std::runtime_error("prefetching_cache is shutting down"); }
    lifetime   = std::make_shared<exec::completion_controller::slot>(std::move(admission));
    completion = std::make_shared<prepared_io_completion>(
      [lifetime](std::span<cached_chunk* const> completed, bool host_ok) noexcept {
        std::ignore = lifetime;
        for (auto* chunk : completed) {
          std::ignore = host_ok ? chunk->state.mark_cached() : chunk->state.mark_load_failed();
        }
      });
    for (auto const& raw : requests) {
      if (raw.size() == 0) { continue; }
      if (raw.dst == nullptr) { throw std::invalid_argument("device readv destination is null"); }
      if (raw.offset() >= obj.size()) { continue; }
      auto const request_size = std::min(raw.size(), obj.size() - raw.offset());
      if (request_size > std::numeric_limits<std::size_t>::max() - logical_bytes) {
        throw std::overflow_error("device cache read byte count overflow");
      }
      logical_bytes += request_size;
      ++reads;

      range const request_rng{raw.offset(), request_size};
      auto chunks = ranges_in_cache(
        obj, request_rng.offset, request_rng.size, coverage_policy::partial, out_handle);
      std::size_t ci       = 0;
      auto const first     = (request_rng.offset / _chunk_size) * _chunk_size;
      auto const last      = ((request_rng.end() - 1) / _chunk_size) * _chunk_size;
      auto const positions = ((last - first) / _chunk_size) + 1;

      for (std::size_t position = 0; position < positions; ++position) {
        auto const off = first + position * _chunk_size;
        while (ci < chunks.size() && chunks[ci]->offset < off) {
          ++ci;
        }
        cached_chunk* chunk =
          ci < chunks.size() && chunks[ci]->offset == off ? chunks[ci] : nullptr;
        range const needed = intersect(request_rng, range{off, _chunk_size});
        auto* device_dst   = raw.dst + (needed.offset - request_rng.offset);

        prefetch_census::instance().bytes_logical.fetch_add(needed.size, std::memory_order_relaxed);
        if (chunk != nullptr &&
            chunk->state.try_pin_covering(off, _chunk_size, needed.offset, needed.end())) {
          pinned.push_back(chunk);
          cached_copies.add(device_dst, chunk->data + (needed.offset - chunk->offset), needed.size);
          prefetch_census::instance().bytes_hit.fetch_add(needed.size, std::memory_order_relaxed);
          ++hits;
          continue;
        }

        chunk_fill fill;
        if (cache_while_reading && chunk != nullptr &&
            chunk->state.take_loading_merging(
              needed_fill(off, _chunk_size, needed.offset, needed.end()), fill)) {
          claimed.push_back(chunk);
          prepared_io_slice io_slice{needed,
                                     host_buffer{std::vector<cached_chunk*>{chunk}},
                                     device_buffer{device_dst, stream, device_id.value()}};
          io_slice.on_complete = completion;
          prepared.push_back(std::move(io_slice));
          prefetch_census::instance().bytes_h2d.fetch_add(needed.size, std::memory_order_relaxed);
          ++loads;
          continue;
        }

        prepared.emplace_back(needed, device_buffer{device_dst, stream, device_id.value()});
        prefetch_census::instance().bytes_miss.fetch_add(needed.size, std::memory_order_relaxed);
        ++misses;
      }
    }

    has_backend = !prepared.empty();
    has_cached  = !pinned.empty();
    auto const task_count =
      static_cast<std::size_t>(has_backend) + static_cast<std::size_t>(has_cached);
    if (task_count != 0) {
      coordinator   = std::make_shared<grouped_coordinator>(logical_bytes, task_count);
      result_future = coordinator->get_future();
      if (has_backend) {
        backend_terminal =
          size_terminal{[coordinator](exec::try_t<std::size_t>&& result) mutable noexcept {
            if (result.has_exception()) {
              coordinator->report_error(std::move(result).exception());
            } else {
              coordinator->on_complete();
            }
          }};
      }

      if (has_cached) {
        retirement              = std::make_shared<cached_copy_retirement>();
        retirement->pins        = std::move(pinned);
        retirement->lifetime    = lifetime;
        retirement->coordinator = coordinator;
        retirement->async_error = std::make_exception_ptr(
          std::runtime_error("prefetching_cache: asynchronous cached host-to-device copy failed"));
        retirement->stream    = stream.value();
        retirement->device_id = device_id.value();

        auto const create_error =
          cudaEventCreateWithFlags(&retirement->event, cudaEventDisableTiming);
        if (create_error != cudaSuccess) {
          throw std::runtime_error(std::string("prefetching_cache: CUDA event creation failed: ") +
                                   cudaGetErrorString(create_error));
        }
        exec::invocable<void()> completion_task{
          [retirement]() noexcept { retirement->wait_and_finish(); }};
        _io_cb_thread_pool.schedule(std::move(completion_task));
      }
    }
  } catch (...) {
    for (auto* chunk : pinned) {
      chunk->state.release_read();
    }
    for (auto* chunk : claimed) {
      std::ignore = chunk->state.mark_load_failed();
    }
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }

  _counters.n_reads.fetch_add(reads, std::memory_order_relaxed);
  _counters.hits.fetch_add(hits, std::memory_order_relaxed);
  _counters.h2d.fetch_add(loads, std::memory_order_relaxed);
  _counters.misses.fetch_add(misses, std::memory_order_relaxed);

  if (has_backend) {
    auto io_future = _io_ctx->host_device_readv_async_io(obj, std::move(prepared));
    // From this point the prepared-slice callbacks own every loading transition.
    claimed.clear();
    std::move(io_future).install_callback(std::move(backend_terminal));
  } else {
    claimed.clear();
  }

  if (has_cached) { retire_pins_after_stream(stream, cached_copies, std::move(retirement)); }

  if (has_backend || has_cached) { return std::move(result_future); }
  return exec::make_semi_future<std::size_t>(logical_bytes);
}

void prefetching_cache::retire_pins_after_stream(
  rmm::cuda_stream_view stream,
  sirius::cuda::device_copy_batch const& copies,
  std::shared_ptr<cached_copy_retirement> retirement) noexcept
{
  CTRACK_NAME("cache::retire_pins_after_stream");
  cudaError_t enqueue_status = cudaErrorUnknown;
  try {
    enqueue_status = copies.enqueue(stream);
  } catch (...) {
    enqueue_status = cudaErrorUnknown;
  }
  retirement->enqueue_ok     = enqueue_status == cudaSuccess;
  auto const record_status   = cudaEventRecord(retirement->event, stream.value());
  retirement->event_recorded = record_status == cudaSuccess;
  retirement->published.count_down();
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

std::size_t prefetching_cache::claimed_bytes() const noexcept
{
  return _pool ? _pool->total_allocated_bytes() : 0;
}

void prefetching_cache::evict(std::size_t bytes_to_free)
{
  // Nothing to free, nothing holding memory, or a cache on its way down -- in
  // the last case the evictor is already reclaiming everything it can, and a
  // demand queued behind the stop sentinel would never be looked at.
  if (bytes_to_free == 0 || !_armed || _shutting_down.load(std::memory_order_relaxed)) { return; }
  _eviction_queue.enqueue(cache_request{eviction_request{bytes_to_free}});
}

void prefetching_cache::drain_and_abandon(request_queue_type& queue) noexcept
{
  cache_request entry;
  while (queue.try_dequeue(entry)) {
    // Only prefetch requests have anything to abandon: an eviction request owns
    // no stage machine and no waiter, so dropping it strands nobody.
    if (auto* req = std::get_if<prefetch_request>(&entry); req != nullptr && *req) {
      req->producer->mark_abandoned();
    }
    entry = {};
  }
}

bool prefetching_cache::prepare(prefetching_handle& handle, bool wait_for_eviction)
{
  if (!handle) { return false; }
  return prepare_request(handle._req, wait_for_eviction);
}

bool prefetching_cache::prefetch(prefetching_handle& handle,
                                 exec::invocable<void(bool) noexcept> on_done)
{
  CTRACK_NAME("cache::prefetch");
  auto& req   = handle._req;
  auto settle = [&on_done](bool ok) {
    on_done(ok);
    return false;
  };

  if (!req || req.has_fallen_behind() || !req.chunks) { return settle(false); }
  if (!_io_ctx->supports_vector_host_read()) { return settle(false); }
  // Read the stage before mark_loading advances it: a request that has not
  // reached `prepared` has no buffers attached yet, so the claim loop below
  // will find every chunk in empty/queued and read nothing.
  bool const was_prepared = req.producer->get() >= producer_stage::prepared;
  if (!req.producer->mark_loading()) { return settle(false); }

  std::vector<prepared_io_slice> prepared;
  std::vector<cached_chunk*> claimed_chunks;
  std::shared_ptr<exec::invocable<void(bool) noexcept>> done_owner;
  size_terminal terminal;
  auto fail_setup = [&](bool inflight_counted) noexcept {
    for (auto* chunk : claimed_chunks) {
      std::ignore = chunk->state.mark_load_failed();
    }
    std::ignore = req.producer->mark_load_failed();
    if (inflight_counted) {
      prefetch_census::instance().inflight_prefetches.fetch_sub(1, std::memory_order_relaxed);
    }
    if (done_owner != nullptr) {
      (*done_owner)(false);
    } else {
      on_done(false);
    }
    return false;
  };

  try {
    prepared.reserve(req.chunks->size());
    claimed_chunks.reserve(req.chunks->size());
    for (cached_chunk* c : *req.chunks) {
      // Claim the chunk and preserve the promised fill as this prefetch's logical
      // range. The reactor owns any further physical chunking and alignment.
      chunk_fill fill;
      if (c->state.take_loading(fill)) {
        auto const [seg_lo, seg_hi] = fill_span(fill, c->offset, _chunk_size);
        prefetch_census::instance().bytes_prefetch.fetch_add(seg_hi - seg_lo,
                                                             std::memory_order_relaxed);
        claimed_chunks.push_back(c);
        prepared.emplace_back(range{seg_lo, seg_hi - seg_lo},
                              host_buffer{std::vector<cached_chunk*>{c}});
      }
    }
  } catch (...) {
    return fail_setup(false);
  }
  if (prepared.empty()) {
    // Nothing was claimable.  Either the request genuinely had no IO left to do
    // (already cached / host-backed), or it was issued before anything prepared
    // it and no chunk had a buffer yet — in which case this `ready` is a lie the
    // reader will pay for, so the two are counted apart.
    auto& census = prefetch_census::instance();
    (was_prepared ? census.skipped_no_ranges : census.prefetch_unprepared)
      .fetch_add(1, std::memory_order_relaxed);
    std::ignore = req.producer->mark_ready();
    return settle(true);
  }
  prefetch_census::instance().prefetch_issued.fetch_add(1, std::memory_order_relaxed);

  if (req.is_cancelled()) { return fail_setup(false); }

  bool inflight_counted = false;
  try {
    prefetch_census::instance().inflight_prefetches.fetch_add(1, std::memory_order_relaxed);
    inflight_counted = true;
    auto admission   = acquire_inflight_io();
    if (!admission) { throw std::runtime_error("prefetching_cache is shutting down"); }
    auto lifetime   = std::make_shared<exec::completion_controller::slot>(std::move(admission));
    auto completion = std::make_shared<prepared_io_completion>(
      [lifetime](std::span<cached_chunk* const> completed, bool host_ok) noexcept {
        std::ignore = lifetime;
        for (auto* chunk : completed) {
          std::ignore = host_ok ? chunk->state.mark_cached() : chunk->state.mark_load_failed();
        }
      });
    done_owner = std::make_shared<exec::invocable<void(bool) noexcept>>(std::move(on_done));
    terminal   = size_terminal{[req, done_owner](exec::try_t<size_t>&& res) mutable noexcept {
      auto const ok = res.has_value();
      std::ignore   = ok ? req.producer->mark_ready() : req.producer->mark_load_failed();
      prefetch_census::instance().inflight_prefetches.fetch_sub(1, std::memory_order_relaxed);
      (*done_owner)(ok);
    }};
    for (auto& slice : prepared) {
      slice.on_complete = completion;
    }
  } catch (...) {
    return fail_setup(inflight_counted);
  }

  auto io_future = _io_ctx->host_device_readv_async_io(*req.obj, std::move(prepared));
  std::move(io_future).install_callback(std::move(terminal));
  return true;
}

void prefetching_cache::evict_loop(const std::stop_token& st)
{
  std::stop_callback cb(st, [this]() {
    SIRIUS_LOG_TRACE("prefetching_cache: evict_loop received stop request, unblocking queue");
    // unblock the worker if it's waiting on an empty queueue
    _eviction_queue.enqueue(cache_request{});
  });

  // One queued prefetch request plus whether its per-chunk subscriber
  // references have been handed back yet.
  struct tracked_request {
    prefetch_request req;
    bool released{false};
  };

  std::vector<tracked_request> eviction_batch;
  // Reclaimed buffers grouped by their origin NUMA node so each group can be
  // returned to the arena it came from.
  std::unordered_map<int, std::vector<std::byte*>> reclaim_by_numa;
  while (!_shutting_down && !st.stop_requested()) {
    cache_request entry;
    _eviction_queue.wait_dequeue(entry);

    // Bytes explicitly demanded this round, summed across every eviction
    // request absorbed below.  Summed rather than maxed: two callers each short
    // by their own amount are short by the total, and serving only the larger
    // leaves the other one still waiting.
    std::size_t requested_bytes = 0;
    bool eviction_requested     = false;
    auto absorb                 = [&](cache_request&& e) {
      std::visit(
        [&](auto&& r) {
          using T = std::decay_t<decltype(r)>;
          if constexpr (std::is_same_v<T, prefetch_request>) {
            // A falsy request is the wakeup sentinel and names no chunks, so
            // there is nothing to track -- it has already done its job by
            // getting the loop past wait_dequeue.
            if (r) { eviction_batch.push_back({std::move(r), false}); }
          } else {
            // An explicit demand has to survive into the eviction decision
            // below rather than be skipped past, or the back-pressure signal
            // does nothing and the pool never recovers.
            eviction_requested = true;
            requested_bytes += r.bytes_to_free;
          }
        },
        std::move(e));
    };

    // Accumulate newly-queued requests into the persistent batch.  The batch is
    // NOT cleared each round: a request is retired only once every chunk it
    // named has been reclaimed or taken over, so chunks that are busy this
    // round stay candidates for a later one instead of being lost.
    absorb(std::move(entry));
    while (_eviction_queue.try_dequeue(entry)) {
      absorb(std::move(entry));
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

    // Under pressure the target is a fraction of what the pool holds; an
    // explicit demand raises it to at least what was asked for.  A floor rather
    // than a replacement, so a demand that arrives while the pool is ALSO over
    // its own threshold does not talk the evictor down to the smaller of the
    // two -- both reasons to free memory are still true.
    size_t need = 0;
    if (_cfg.dispose_on_idle) {
      need = std::numeric_limits<size_t>::max();
    } else {
      if (_pool->should_start_evicting()) {
        need = static_cast<size_t>(static_cast<double>(_pool->total_allocated_chunks()) * 0.25);
      }
      // Rounded up: freeing whole chunks is the only granularity there is, so a
      // demand for part of one still costs the whole thing.
      auto const chunk_bytes = std::max<size_t>(_pool->chunk_size(), 1);
      need                   = std::max(need, (requested_bytes + chunk_bytes - 1) / chunk_bytes);
    }

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
    std::erase_if(eviction_batch, [](tracked_request const& er) {
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
