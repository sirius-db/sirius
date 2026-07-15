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
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
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

/// Sticky context errors: the wait ended AND the CUDA context is dead — all
/// in-flight and future work on it is terminated, so nothing can write the
/// caller's dst anymore.  Per the CUDA contract these leave the process state
/// inconsistent; restart is the recovery (fail-stop is containment).
bool is_context_fatal(cudaError_t rc)
{
  switch (rc) {
    case cudaErrorIllegalAddress:
    case cudaErrorMisalignedAddress:
    case cudaErrorIllegalInstruction:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorHardwareStackError:
    case cudaErrorLaunchFailure: return true;
    default: return false;
  }
}

/// Codes meaning the sync call cannot have performed the wait at all — the
/// counterexamples to "an error return is a deferred report after the wait".
bool cannot_have_waited(cudaError_t rc)
{
  return rc == cudaErrorInvalidResourceHandle || rc == cudaErrorStreamCaptureUnsupported;
}

/// RAII owner of the delivery completion event (F01): destroy runs only when
/// create succeeded, and a destroy failure is logged without ever overriding
/// the delivery result.
struct event_guard {
  const cuda_delivery_ops& ops;
  cudaEvent_t handle{};
  bool created{false};

  explicit event_guard(const cuda_delivery_ops& delivery_ops) : ops(delivery_ops) {}
  event_guard(const event_guard&)            = delete;
  event_guard& operator=(const event_guard&) = delete;

  ~event_guard()
  {
    if (!created) { return; }
    cudaError_t rc = cudaSuccess;
    try {
      rc = ops.event_destroy(handle);
    } catch (...) {
      rc = cudaErrorUnknown;
    }
    if (rc != cudaSuccess) {
      SIRIUS_LOG_WARN(
        "cuobj_rdma_reactor: completion event destroy failed ({}); delivery result "
        "preserved",
        cudaGetErrorString(rc));
    }
  }
};

}  // namespace

void validate(const cuda_delivery_ops& ops)
{
  const bool complete = ops.event_create && ops.event_record && ops.event_synchronize &&
                        ops.event_destroy && ops.memcpy_async && ops.stream_synchronize &&
                        ops.device_synchronize;
  if (!complete) {
    throw std::invalid_argument(
      "cuda_delivery_ops: every delivery op must be callable (the defaults are the real CUDA "
      "runtime entry points; partial injections must keep the rest real)");
  }
}

cuobj_rdma_reactor::arena::~arena()
{
  if (base == nullptr) { return; }
  if (leaked) {
    return;
  }  // F01: deliberately leaked — deregistering/freeing
     // under an un-quiesced device is the use-after-free
  if (registrar) { registrar->deregister_memory(base); }
  (void)cudaFree(base);
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
  if (_state != reactor_state::created) { return; }
  _state = reactor_state::running;
  _workers.reserve(_config.max_inflight);
  for (size_t i = 0; i < _config.max_inflight; ++i) {
    _workers.emplace_back([this] { worker_loop(); });
  }
}

void cuobj_rdma_reactor::shutdown()
{
  {
    std::unique_lock lk{_mtx};
    if (_state == reactor_state::joined) { return; }
    _state = reactor_state::stopping;
    _cv.notify_all();
    _drained_cv.wait(lk, [&] { return _queue.empty() && _active == 0; });
    _state = reactor_state::joined;
    _cv.notify_all();
  }
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();

  // F01 fail-stop teardown probe: after a fatal delivery state the arenas may
  // still be the target of un-quiesced NIC/copy work.  Prove device
  // quiescence per arena or LEAK it — freeing under an un-quiesced device is
  // the use-after-free; leaking is safe.  (_first_fatal is stable here: its
  // only writers are the worker threads, all joined above.)
  if (_first_fatal) {
    std::lock_guard alk{_arena_mtx};
    for (auto& [device_id, ar] : _arenas) {
      if (!ar || ar->base == nullptr || ar->leaked) { continue; }
      cudaError_t rc = cudaSuccess;
      try {
        rmm::cuda_set_device_raii device_scope{rmm::cuda_device_id{device_id}};
        rc = _ctx->delivery_ops().device_synchronize();
      } catch (...) {
        rc = cudaErrorUnknown;
      }
      if (rc != cudaSuccess) {
        ar->leaked = true;
        _arena_leak_total.fetch_add(1, std::memory_order_relaxed);
        SIRIUS_LOG_ERROR(
          "cuobj_rdma_reactor: leaking the device {} landing arena at teardown — device "
          "quiescence could not be established after a fatal delivery state ({}); a fatal CUDA "
          "error is process-level: restart is the recovery",
          device_id,
          cudaGetErrorString(rc));
      }
    }
  }
}

void cuobj_rdma_reactor::interrupt() { _cv.notify_all(); }

bool cuobj_rdma_reactor::stopping()
{
  std::lock_guard lk{_mtx};
  return _state == reactor_state::failing || _state == reactor_state::failed ||
         _state == reactor_state::stopping || _state == reactor_state::joined;
}

bool cuobj_rdma_reactor::delivery_fatal()
{
  std::lock_guard lk{_mtx};
  return _first_fatal != nullptr;
}

std::string cuobj_rdma_reactor::first_fatal_message()
{
  std::lock_guard lk{_mtx};
  return _first_fatal_message;
}

void cuobj_rdma_reactor::enter_failed(std::exception_ptr fatal, std::string message)
{
  std::deque<std::unique_ptr<cuobj_chunked_rx_request>> drained;
  {
    std::lock_guard lk{_mtx};
    if (_state != reactor_state::running) { return; }  // exactly-once; shutdown keeps its own path
    _state               = reactor_state::failing;
    _first_fatal         = fatal;
    _first_fatal_message = std::move(message);
    drained.swap(_queue);
    _delivery_fatal_total.fetch_add(1, std::memory_order_relaxed);
    _cv.notify_all();
    _drained_cv.notify_all();
  }
  SIRIUS_LOG_ERROR(
    "cuobj_rdma_reactor: FATAL delivery state — {}; intake stopped, {} queued "
    "chunk(s) drained by error; a fatal CUDA error is process-level: restart is "
    "the recovery",
    _first_fatal_message,
    drained.size());
  for (auto& chunk : drained) {
    if (chunk && chunk->manager) { chunk->manager->report_error(fatal); }
  }
  {
    std::lock_guard lk{_mtx};
    if (_state == reactor_state::failing) { _state = reactor_state::failed; }
  }
}

void cuobj_rdma_reactor::quiesce_or_fail_stop(const cuobj_chunked_rx_request& chunk)
{
  _fallback_stream_sync_total.fetch_add(1, std::memory_order_relaxed);
  const auto& ops = _ctx->delivery_ops();

  // Stream rung.  An error return normally means the wait COMPLETED and a
  // deferred async error is being reported — that is still proof of
  // quiescence; only the cannot-have-waited codes leave it unproved.
  cudaError_t rc = cudaSuccess;
  bool waited    = true;
  try {
    rc     = ops.stream_synchronize(chunk.stream.value());
    waited = !cannot_have_waited(rc);
  } catch (...) {
    rc     = cudaErrorUnknown;
    waited = false;  // a throwing callable proves nothing
  }
  if (waited && !is_context_fatal(rc)) { return; }  // quiesced_recoverable
  if (waited) {                                     // quiesced_context_fatal
    enter_failed(std::make_exception_ptr(std::runtime_error(
                   std::string("cuobj_rdma_reactor: fatal delivery state (stream sync: ") +
                   cudaGetErrorString(rc) + ")")),
                 std::string("stream sync: ") + cudaGetErrorString(rc));
    return;
  }

  // Device rung: success proves quiescence device-wide; any error here means
  // dead-or-unknowable — there is no safe "report and continue" fourth path
  // under the bare-dst contract (F01 contract v3 §2).
  try {
    rc = ops.device_synchronize();
  } catch (...) {
    rc = cudaErrorUnknown;
  }
  if (rc == cudaSuccess) { return; }  // quiesced_recoverable via the device rung
  enter_failed(std::make_exception_ptr(std::runtime_error(
                 std::string("cuobj_rdma_reactor: fatal delivery state (device sync: ") +
                 cudaGetErrorString(rc) + ")")),
               std::string("device sync: ") + cudaGetErrorString(rc));
}

rdma_perf_snapshot cuobj_rdma_reactor::perf_snapshot() const noexcept
{
  rdma_perf_snapshot s;
  s.bytes_total                = _bytes_total.load(std::memory_order_relaxed);
  s.requests_total             = _requests_total.load(std::memory_order_relaxed);
  s.retries_total              = _retries_total.load(std::memory_order_relaxed);
  s.short_read_total           = _short_read_total.load(std::memory_order_relaxed);
  s.error_total                = _error_total.load(std::memory_order_relaxed);
  s.slot_wait_total            = _slot_wait_total.load(std::memory_order_relaxed);
  s.flush_total                = _flush_total.load(std::memory_order_relaxed);
  s.inflight_peak              = _inflight_peak.load(std::memory_order_relaxed);
  s.fallback_stream_sync_total = _fallback_stream_sync_total.load(std::memory_order_relaxed);
  s.delivery_fatal_total       = _delivery_fatal_total.load(std::memory_order_relaxed);
  s.arena_leak_total           = _arena_leak_total.load(std::memory_order_relaxed);
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
      _cv.wait(lk, [&] {
        return _state == reactor_state::stopping || _state == reactor_state::joined ||
               !_queue.empty();
      });
      if (_queue.empty()) {
        if (_state == reactor_state::stopping || _state == reactor_state::joined) { return; }
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

    // Owner-worker convergence (F01): after a fatal transition never start new
    // CUDA work on the dead context.  The owner resolves its own chunk here,
    // pre-enqueue — nothing is in flight on the slot, so the plain RAII
    // release below is safe.
    if (delivery_fatal()) {
      throw std::runtime_error("cuobj_rdma_reactor: chunk aborted after a fatal delivery state (" +
                               first_fatal_message() + ")");
    }

    const auto& ops = _ctx->delivery_ops();
    if (_ctx->flush_before_copy()) {
      throw_on_cuda_error(
        cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                           cudaFlushGPUDirectRDMAWritesToOwner),
        "GPUDirect writes flush failed");
      _flush_total.fetch_add(1, std::memory_order_relaxed);
    }

#ifndef NDEBUG
    {
      // Precondition (F01 contract v3 §2): RDMA device reads do not support
      // captured streams — a capturing stream makes the quiescence ladder's
      // sync calls unable to wait.
      cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(chunk.stream.value(), &capture) == cudaSuccess) {
        assert(capture == cudaStreamCaptureStatusNone &&
               "RDMA device reads do not support captured streams");
      }
    }
#endif

    // F01 discipline: the completion event exists BEFORE the enqueue, so a
    // create failure unwinds with nothing in flight (safe release,
    // recoverable).  Inline completion wait: the D2D of one slot is
    // microseconds against the blocking GET that preceded it — no parked-copy
    // machinery.
    event_guard event{ops};
    throw_on_cuda_error(ops.event_create(&event.handle, cudaEventDisableTiming),
                        "completion event create failed");
    event.created = true;
    throw_on_cuda_error(
      ops.memcpy_async(chunk.dst, slot_ptr, n, cudaMemcpyDeviceToDevice, chunk.stream.value()),
      "D2D copy enqueue failed");

    // Post-enqueue: from here the slot belongs to the stream and the caller's
    // future must stay unresolved until quiescence is proved — any failure
    // goes through the quiescence ladder before this frame unwinds (source
    // slot AND destination RMM UAF guards, F01).
    cudaError_t delivery_rc = cudaSuccess;
    std::string delivery_stage;
    try {
      delivery_rc    = ops.event_record(event.handle, chunk.stream.value());
      delivery_stage = "completion event record failed";
      if (delivery_rc == cudaSuccess) {
        delivery_rc    = ops.event_synchronize(event.handle);
        delivery_stage = "completion event wait failed";
      }
    } catch (...) {
      delivery_rc    = cudaErrorUnknown;
      delivery_stage = "completion event operation threw";
    }
    if (delivery_rc != cudaSuccess) {
      quiesce_or_fail_stop(chunk);
      throw std::runtime_error("cuobj_rdma_reactor: " + delivery_stage + ": " +
                               cudaGetErrorString(delivery_rc));
    }

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
    } else if (_state == reactor_state::failing || _state == reactor_state::failed) {
      // F01 fail-fast point: static prep_* cannot see reactor state, so a
      // fatal delivery state surfaces here, naming the first fatal error.
      failure = std::make_exception_ptr(
        std::runtime_error("cuobj_rdma_reactor::enqueue: reactor entered a fatal delivery state (" +
                           _first_fatal_message + ")"));
    } else if (_state == reactor_state::stopping || _state == reactor_state::joined) {
      failure = std::make_exception_ptr(
        std::runtime_error("cuobj_rdma_reactor::enqueue: reactor is shut down"));
    } else if (_state != reactor_state::running) {
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
