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

#include "io/rdma/cuobj_rdma_reactor.hpp"

#include "io/uri_parser.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::io::rdma {

namespace {

std::runtime_error not_implemented(std::string_view entry_point)
{
  return std::runtime_error("cuobj_rdma_reactor::" + std::string(entry_point) +
                            ": the S3 RDMA transport is not implemented yet (no RDMA client "
                            "configured)");
}

std::runtime_error short_read_error(size_t got, size_t expected)
{
  return std::runtime_error("cuobj_rdma_reactor: short read: got " + std::to_string(got) + " of " +
                            std::to_string(expected) + " bytes");
}

void throw_on_cuda_error(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuobj_rdma_reactor: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

size_t clipped_size(const cuobj_rdma_io_object& file, size_t offset, size_t size)
{
  const size_t file_size = file.size();
  if (offset >= file_size) { return 0; }
  return std::min(size, file_size - offset);
}

cuobj_rdma_reactor::config sanitized(cuobj_rdma_reactor::config cfg)
{
  if (cfg.max_inflight == 0) { cfg.max_inflight = 1; }
  if (cfg.arena_slot_size == 0) { cfg.arena_slot_size = 64UL << 10; }
  return cfg;
}

}  // namespace

cuobj_rdma_reactor::arena::~arena()
{
  if (base != nullptr) {
    if (registrar) { registrar->deregister_memory(base); }
    (void)cudaFree(base);
  }
}

cuobj_rdma_reactor::cuobj_rdma_reactor(std::shared_ptr<reactor_context> ctx)
  : _ctx(std::move(ctx)), _config(sanitized(_ctx->cfg()))
{
}

cuobj_rdma_reactor::~cuobj_rdma_reactor()
{
  try {
    shutdown();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
}

void cuobj_rdma_reactor::start()
{
  std::lock_guard lk{_mtx};
  if (_started || _joined) { return; }
  _started = true;
  _workers.reserve(_config.max_inflight);
  for (size_t i = 0; i < _config.max_inflight; ++i) {
    _workers.emplace_back([this] { worker_loop(); });
  }
}

void cuobj_rdma_reactor::shutdown()
{
  {
    std::unique_lock lk{_mtx};
    if (_joined) { return; }
    _stopping = true;
    _cv.notify_all();
    _drained_cv.wait(lk, [&] { return _queue.empty() && _active == 0; });
    _joined = true;
    _cv.notify_all();
  }
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();
}

void cuobj_rdma_reactor::interrupt() { _cv.notify_all(); }

bool cuobj_rdma_reactor::stopping()
{
  std::lock_guard lk{_mtx};
  return _stopping;
}

rdma_perf_snapshot cuobj_rdma_reactor::perf_snapshot() const noexcept
{
  rdma_perf_snapshot s;
  s.bytes_total      = _bytes_total.load(std::memory_order_relaxed);
  s.requests_total   = _requests_total.load(std::memory_order_relaxed);
  s.retries_total    = _retries_total.load(std::memory_order_relaxed);
  s.short_read_total = _short_read_total.load(std::memory_order_relaxed);
  s.error_total      = _error_total.load(std::memory_order_relaxed);
  s.slot_wait_total  = _slot_wait_total.load(std::memory_order_relaxed);
  s.flush_total      = _flush_total.load(std::memory_order_relaxed);
  s.inflight_peak    = _inflight_peak.load(std::memory_order_relaxed);
  return s;
}

size_t cuobj_rdma_reactor::get_with_retry(
  std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst)
{
  auto& client = *_ctx->client();
  for (size_t attempt = 1;; ++attempt) {
    try {
      const size_t n = client.get(bucket, key, offset, size, dst);
      if (n != size) {
        _short_read_total.fetch_add(1, std::memory_order_relaxed);
        throw short_read_error(n, size);
      }
      return n;
    } catch (...) {
      if (attempt >= _config.max_get_attempts || stopping()) { throw; }
      _retries_total.fetch_add(1, std::memory_order_relaxed);
      auto delay = _config.retry_backoff_base * attempt;
      if (_config.retry_jitter.count() > 0) {
        delay += std::chrono::milliseconds{
          static_cast<int64_t>(attempt * 1315423911U % (_config.retry_jitter.count() + 1))};
      }
      if (delay.count() > 0) { std::this_thread::sleep_for(delay); }
    }
  }
}

void cuobj_rdma_reactor::worker_loop()
{
  for (;;) {
    std::unique_ptr<cuobj_chunked_rx_request> chunk;
    {
      std::unique_lock lk{_mtx};
      _cv.wait(lk, [&] { return _stopping || !_queue.empty(); });
      if (_queue.empty()) {
        if (_stopping) { return; }
        continue;
      }
      chunk = std::move(_queue.front());
      _queue.pop_front();
      ++_active;
      if (auto peak = _inflight_peak.load(std::memory_order_relaxed); _active > peak) {
        _inflight_peak.store(_active, std::memory_order_relaxed);
      }
    }
    process_chunk(*chunk);
    {
      std::lock_guard lk{_mtx};
      --_active;
      if (_queue.empty() && _active == 0) { _drained_cv.notify_all(); }
    }
  }
}

void cuobj_rdma_reactor::process_chunk(cuobj_chunked_rx_request& chunk)
{
  _requests_total.fetch_add(1, std::memory_order_relaxed);
  try {
    if (!chunk.is_device) {
      const size_t n = get_with_retry(chunk.bucket, chunk.key, chunk.offset, chunk.size, chunk.dst);
      _bytes_total.fetch_add(n, std::memory_order_relaxed);
      chunk.manager->chunk_complete(n);
      return;
    }

    const int device =
      chunk.device_id >= 0 ? chunk.device_id : rmm::get_current_cuda_device().value();
    rmm::cuda_set_device_raii device_scope{rmm::cuda_device_id{device}};
    auto& ar = arena_for_device(device);
    int slot = ar.pool->try_acquire();
    if (slot == slot_pool::no_slot) {
      _slot_wait_total.fetch_add(1, std::memory_order_relaxed);
      slot = ar.pool->acquire();
    }
    struct slot_release {
      slot_pool* pool;
      int slot;
      ~slot_release() { pool->release(slot); }
    } release{ar.pool.get(), slot};

    uint8_t* slot_ptr = ar.base + static_cast<size_t>(slot) * _config.arena_slot_size;
    const size_t n    = get_with_retry(chunk.bucket, chunk.key, chunk.offset, chunk.size, slot_ptr);

    if (_ctx->flush_before_copy()) {
      throw_on_cuda_error(
        cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                           cudaFlushGPUDirectRDMAWritesToOwner),
        "GPUDirect writes flush failed");
      _flush_total.fetch_add(1, std::memory_order_relaxed);
    }
    throw_on_cuda_error(
      cudaMemcpyAsync(chunk.dst, slot_ptr, n, cudaMemcpyDeviceToDevice, chunk.stream.value()),
      "D2D copy enqueue failed");
    // Inline completion wait: the D2D of one slot is microseconds against the
    // blocking GET that preceded it, so the worker waits for its own copy and
    // recycles the slot immediately — no parked-copy machinery.
    cudaEvent_t event{};
    throw_on_cuda_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                        "event create failed");
    auto record_err = cudaEventRecord(event, chunk.stream.value());
    auto sync_err   = record_err == cudaSuccess ? cudaEventSynchronize(event) : record_err;
    (void)cudaEventDestroy(event);
    throw_on_cuda_error(sync_err, "D2D completion wait failed");

    _bytes_total.fetch_add(n, std::memory_order_relaxed);
    chunk.manager->chunk_complete(n);
  } catch (...) {
    _error_total.fetch_add(1, std::memory_order_relaxed);
    chunk.manager->report_error(std::current_exception());
  }
}

cuobj_rdma_reactor::arena& cuobj_rdma_reactor::arena_for_device(int device_id)
{
  std::lock_guard lk{_arena_mtx};
  auto it = _arenas.find(device_id);
  if (it != _arenas.end()) { return *it->second; }

  auto ar = std::make_unique<arena>();
  throw_on_cuda_error(
    cudaMalloc(reinterpret_cast<void**>(&ar->base), _config.max_inflight * _config.arena_slot_size),
    "landing arena allocation failed");
  ar->pool = std::make_unique<slot_pool>(_config.max_inflight);
  if (const auto& client = _ctx->client()) {
    client->register_memory(ar->base, _config.max_inflight * _config.arena_slot_size);
    ar->registrar = client;
  }
  return *_arenas.emplace(device_id, std::move(ar)).first->second;
}

cuobj_rdma_reactor::request_type_ptr cuobj_rdma_reactor::prep_host_rx_request(
  const config& /*cfg*/, const io_object_type& file, const io_object_segment& segment)
{
  const size_t n = clipped_size(file, segment.offset, segment.size);
  if (n == 0) { return request_type::create({}); }

  auto manager = std::make_shared<request_manager>(n, 1);
  std::vector<std::unique_ptr<cuobj_chunked_rx_request>> chunks;
  auto chunk     = std::make_unique<cuobj_chunked_rx_request>();
  chunk->bucket  = file.bucket();
  chunk->key     = file.key();
  chunk->offset  = segment.offset;
  chunk->size    = n;
  chunk->dst     = segment.data();
  chunk->manager = std::move(manager);
  chunks.push_back(std::move(chunk));
  return request_type::create(std::move(chunks));
}

cuobj_rdma_reactor::request_type_ptr cuobj_rdma_reactor::prep_device_rx_request(
  const config& cfg,
  const io_object_type& file,
  uint8_t* dst,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream,
  int device_id)
{
  const size_t total = clipped_size(file, offset, size);
  if (total == 0) { return request_type::create({}); }

  const size_t slot_size = cfg.arena_slot_size != 0 ? cfg.arena_slot_size : (64UL << 10);
  const size_t n_chunks  = (total + slot_size - 1) / slot_size;
  auto manager           = std::make_shared<request_manager>(total, n_chunks);

  std::vector<std::unique_ptr<cuobj_chunked_rx_request>> chunks;
  chunks.reserve(n_chunks);
  for (size_t i = 0; i < n_chunks; ++i) {
    const size_t delta = i * slot_size;
    auto chunk         = std::make_unique<cuobj_chunked_rx_request>();
    chunk->bucket      = file.bucket();
    chunk->key         = file.key();
    chunk->offset      = offset + delta;
    chunk->size        = std::min(slot_size, total - delta);
    chunk->dst         = dst + delta;
    chunk->is_device   = true;
    chunk->stream      = stream;
    chunk->device_id   = device_id;
    chunk->manager     = manager;
    chunks.push_back(std::move(chunk));
  }
  return request_type::create(std::move(chunks));
}

void cuobj_rdma_reactor::enqueue(request_type_ptr req)
{
  if (!req) { return; }
  auto chunks = req->get_all_chunks();
  if (chunks.empty()) { return; }

  std::exception_ptr failure;
  {
    std::lock_guard lk{_mtx};
    if (!_ctx->client()) {
      failure = std::make_exception_ptr(not_implemented("enqueue"));
    } else if (_stopping || _joined) {
      failure = std::make_exception_ptr(
        std::runtime_error("cuobj_rdma_reactor::enqueue: reactor is shut down"));
    } else if (!_started) {
      failure = std::make_exception_ptr(
        std::runtime_error("cuobj_rdma_reactor::enqueue: reactor is not started"));
    } else {
      for (auto& chunk : chunks) {
        _queue.push_back(std::move(chunk));
      }
    }
  }
  if (failure) {
    for (auto& chunk : chunks) {
      if (chunk) { chunk->manager->report_error(failure); }
    }
    return;
  }
  _cv.notify_all();
}

size_t cuobj_rdma_reactor::host_read(const io_object_type& file,
                                     size_t offset,
                                     size_t size,
                                     uint8_t* dst)
{
  if (!_ctx->client()) { throw not_implemented("host_read"); }
  const size_t n = clipped_size(file, offset, size);
  if (n == 0) { return 0; }
  _requests_total.fetch_add(1, std::memory_order_relaxed);
  try {
    const size_t got = get_with_retry(file.bucket(), file.key(), offset, n, dst);
    _bytes_total.fetch_add(got, std::memory_order_relaxed);
    return got;
  } catch (...) {
    _error_total.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
}

std::unique_ptr<cuobj_rdma_reactor::io_object_type> cuobj_rdma_reactor::create_io_object(
  std::string path)
{
  throw std::runtime_error("cuobj_rdma_reactor::create_io_object(" + path +
                           "): object creation needs the client-side HEAD and lives in "
                           "s3_rdma_ioctx::create_io_object");
}

bool cuobj_rdma_reactor::supports(std::string_view path)
{
  try {
    return parse(path).scheme == "s3";
  } catch (...) {
    return false;
  }
}

}  // namespace sirius::io::rdma
