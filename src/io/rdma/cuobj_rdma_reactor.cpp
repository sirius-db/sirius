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

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace sirius::io::rdma {

namespace {

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

/// Sticky/context-fatal classification, driven by the single table
/// @c k_sticky_context_fatal_codes.  Consulted before the delivery boundary
/// only; past the boundary every failure is fatal regardless.
bool is_context_fatal(cudaError_t rc) noexcept
{
  for (const cudaError_t code : k_sticky_context_fatal_codes) {
    if (rc == code) { return true; }
  }
  return false;
}

/// RAII owner of the delivery completion event: destroy runs only when
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
      // Destroy-after-success is log-only; it must never fail the delivery.
      // The dtor is implicitly noexcept and the log path formats + calls a
      // sink (neither noexcept), so a throwing sink or a formatting bad_alloc
      // here would std::terminate and leave the future unresolved.  Swallow
      // everything: best-effort diagnostics, never fatal.
      try {
        SIRIUS_LOG_WARN(
          "cuobj_rdma_reactor: completion event destroy failed ({}); delivery result "
          "preserved",
          cudaGetErrorString(rc));
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
  }
};

std::string current_exception_message()
{
  try {
    throw;
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "non-standard exception";
  }
}

/// Monotonic peak update: plain load/store lets a smaller concurrent value
/// overwrite a larger one, under-reporting the peak.
void update_peak(std::atomic<uint64_t>& peak, uint64_t value) noexcept
{
  uint64_t prev = peak.load(std::memory_order_relaxed);
  while (value > prev && !peak.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {}
}

}  // namespace

void default_fatal_hook(const char* what, cudaError_t rc) noexcept
{
  // The hard guarantee is std::terminate in invoke_fatal; this diagnostic
  // must never delay it.  A stalled stderr consumer (full pipe) would block a
  // plain write() forever, so flip stderr to non-blocking for the duration —
  // a full pipe then yields EAGAIN instead of blocking.  We are terminating,
  // so the brief flag change on the shared fd is acceptable; restore it after.
  const int flags = ::fcntl(STDERR_FILENO, F_GETFL);
  if (flags != -1) { (void)::fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK); }
  const auto emit = [](const char* text) {
    if (text == nullptr) { return; }
    size_t len = 0;
    while (text[len] != '\0') {
      ++len;
    }
    (void)!::write(
      STDERR_FILENO, text, len);  // single non-blocking attempt; EAGAIN/partial dropped
  };
  emit("cuobj_rdma_reactor: process-fatal CUDA delivery failure: ");
  emit(what);
  emit(": ");
  emit(cudaGetErrorName(rc));
  emit("\n");
  if (flags != -1) { (void)::fcntl(STDERR_FILENO, F_SETFL, flags); }
}

[[noreturn]] void invoke_fatal(const cuda_delivery_ops& ops,
                               const char* what,
                               cudaError_t rc) noexcept
{
  try {
    if (ops.fatal_hook) { ops.fatal_hook(what, rc); }
  } catch (...) {  // NOLINT(bugprone-empty-catch) — a throwing hook dies the same way
  }
  std::terminate();
}

void validate(const cuda_delivery_ops& ops)
{
  const bool complete = ops.event_create && ops.event_record && ops.event_synchronize &&
                        ops.event_destroy && ops.memcpy_async && ops.flush &&
                        ops.stream_capture_query && ops.fatal_hook;
  if (!complete) {
    throw std::invalid_argument(
      "cuda_delivery_ops: every delivery op must be callable (the defaults are the real CUDA "
      "runtime entry points; partial injections must keep the rest real)");
  }
}

cuobj_rdma_reactor::config sanitized(cuobj_rdma_reactor::config cfg)
{
  if (cfg.max_inflight == 0) { cfg.max_inflight = 1; }
  if (cfg.arena_slot_size == 0) { cfg.arena_slot_size = 64UL << 10; }
  if (cfg.queue_cap.has_value()) {
    if (*cfg.queue_cap == 0) {
      throw std::invalid_argument("cuobj_rdma_reactor: queue_cap must be positive");
    }
  } else {
    if (cfg.max_inflight > std::numeric_limits<size_t>::max() / 4) {
      throw std::overflow_error(
        "cuobj_rdma_reactor: the derived queue_cap (4 x max_inflight) overflows");
    }
    cfg.queue_cap = 4 * cfg.max_inflight;
  }
  return cfg;
}

cuobj_rdma_reactor::reactor_context::reactor_context(config cfg,
                                                     rdma_transport_clients clients,
                                                     cuda_delivery_ops delivery)
  : _config(sanitized(cfg)),
    _clients(std::move(clients)),
    _delivery(std::move(delivery)),
    _gate(*_config.queue_cap)
{
  validate(_delivery);
}

cuobj_rdma_reactor::arena::~arena()
{
  if (leaked) {
    // Deliberately leaked after a fail-stop: deregistering or freeing under
    // an un-quiesced device is a use-after-free.  Release the registrar
    // session too, so ITS destructor cannot tear down the registration the
    // leak is meant to preserve; the base and pool are intentionally not
    // freed.
    (void)registrar.release();
    return;
  }
  if (base == nullptr) { return; }
  if (registered && registrar) { registrar->deregister_memory(base); }
  (void)cudaFree(base);
}

cuobj_rdma_reactor::cuobj_rdma_reactor(std::shared_ptr<reactor_context> ctx)
  : _ctx(std::move(ctx)), _config(_ctx->cfg())
{
  // Bound before the reactor is externally visible and before any worker
  // exists, so the bind is single-threaded and immutable.
  _ctx->gate().bind_arena_marker(&cuobj_rdma_reactor::mark_arenas_non_freeable, this);
}

cuobj_rdma_reactor::~cuobj_rdma_reactor()
{
  try {
    shutdown();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
}

void cuobj_rdma_reactor::mark_arenas_non_freeable(void* opaque) noexcept
{
  auto* self = static_cast<cuobj_rdma_reactor*>(opaque);
  std::lock_guard lk{self->_arena_mtx};
  for (auto& [device_id, ar] : self->_arenas) {
    if (ar && !ar->leaked) {
      ar->leaked = true;
      self->_arena_leak_total.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

void cuobj_rdma_reactor::start()
{
  std::unique_lock lk{_lifecycle_mtx};
  if (_started) { return; }
  _started = true;
  // Startup rendezvous: every worker acquires its own data session ON ITS
  // OWN THREAD (session state is per-worker by contract) and reports back
  // before this returns, so a capability failure surfaces synchronously
  // from start() instead of as a dead pool.
  _startup_reported = 0;
  _startup_error    = nullptr;
  _workers.reserve(_config.max_inflight);
  for (size_t i = 0; i < _config.max_inflight; ++i) {
    _workers.emplace_back([this] { worker_loop(); });
  }
  _startup_cv.wait(lk, [&] { return _startup_reported == _workers.size(); });
  if (_startup_error != nullptr) {
    auto error = _startup_error;
    lk.unlock();
    // Failed workers exited; close admission so the successful ones do too.
    auto batch = _ctx->gate().begin_close();
    if (batch.has_token()) {
      batch.error_complete_all(error);
      _ctx->gate().complete_drain(std::move(batch).take_token());
    }
    _ctx->gate().await_closed();
    for (auto& worker : _workers) {
      if (worker.joinable()) { worker.join(); }
    }
    _workers.clear();
    lk.lock();
    _joined = true;
    std::rethrow_exception(error);
  }
}

void cuobj_rdma_reactor::shutdown()
{
  std::unique_lock lk{_lifecycle_mtx};
  if (_joined) { return; }
  if (_closing) {
    // A joiner is already elected; block until it finishes.
    _joined_cv.wait(lk, [&] { return _joined; });
    return;
  }
  _closing = true;
  lk.unlock();

  auto& gate = _ctx->gate();
  auto batch = gate.begin_close();
  if (batch.has_token()) {
    batch.error_complete_all(
      std::make_exception_ptr(std::runtime_error("cuobj_rdma_reactor: transport closed")));
    gate.complete_drain(std::move(batch).take_token());
  }
  // Issued work publishes before this returns: permits, claim guards, and the
  // drain all resolve first.
  gate.await_closed();
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();

  lk.lock();
  _joined = true;
  _joined_cv.notify_all();
}

void cuobj_rdma_reactor::interrupt() {}

rdma_perf_snapshot cuobj_rdma_reactor::perf_snapshot() const noexcept
{
  const auto& gate = _ctx->gate();
  rdma_perf_snapshot s;
  s.bytes_total            = _bytes_total.load(std::memory_order_relaxed);
  s.requests_total         = _requests_total.load(std::memory_order_relaxed);
  s.retries_total          = _retries_total.load(std::memory_order_relaxed);
  s.short_read_total       = _short_read_total.load(std::memory_order_relaxed);
  s.error_total            = _error_total.load(std::memory_order_relaxed);
  s.slot_wait_total        = _slot_wait_total.load(std::memory_order_relaxed);
  s.flush_total            = _flush_total.load(std::memory_order_relaxed);
  s.inflight_peak          = _inflight_peak.load(std::memory_order_relaxed);
  s.envelope_wait_total    = gate.envelope_wait_total();
  s.envelope_wait_ns_total = gate.envelope_wait_ns_total();
  s.envelope_depth_peak    = gate.envelope_depth_peak();
  s.slots_in_use_peak      = _slots_in_use_peak.load(std::memory_order_relaxed);
  s.fail_stop_total        = gate.fail_stop_total();
  s.arena_leak_total       = _arena_leak_total.load(std::memory_order_relaxed);
  return s;
}

size_t cuobj_rdma_reactor::control_transfer(const cuobj_chunked_rx_request& chunk)
{
  // Host-plane bounded retry (contract §6 frozen v1 policy): the control
  // service layer above the one-attempt s3_control_client retries transient
  // failures — a transport fault, HTTP 5xx, or a short read — up to
  // k_host_max_attempts with a linear backoff.  A 4xx other than 416 is
  // permanent.  This is the HOST plane only; the device plane never retries,
  // so retries_total (the device tripwire) is never touched here.
  constexpr int k_host_max_attempts             = 3;
  constexpr std::chrono::milliseconds k_backoff = std::chrono::milliseconds{5};
  std::string last_error;
  for (int attempt = 1;; ++attempt) {
    auto result =
      _ctx->clients().control->range_get(*chunk.route, chunk.offset, chunk.size, chunk.dst);
    const long status = result.outcome.http_status;
    bool retriable    = false;
    if (!result.outcome.transport_ok()) {
      last_error = "control-plane transfer failed: " + result.outcome.transport_error;
      retriable  = true;
    } else if (status == 416) {
      // The request was clipped to the object's HEAD-reported size, so a
      // Range-Not-Satisfiable means the object shrank since HEAD — an
      // append-only (contract §2.6) violation, not a benign empty read.
      // Permanent; the client seam still reports 416 verbatim, the reactor
      // treats it as an error here.
      throw std::runtime_error(
        "cuobj_rdma_reactor: control-plane transfer -> HTTP 416 (object changed since HEAD; "
        "keys must be append-only)");
    } else if (status >= 500) {
      last_error = "control-plane transfer -> HTTP " + std::to_string(status);
      retriable  = true;
    } else if (status != 200 && status != 206) {
      throw std::runtime_error("cuobj_rdma_reactor: control-plane transfer -> HTTP " +
                               std::to_string(status));  // 4xx: permanent
    } else if (result.delivered_bytes != chunk.size) {
      _short_read_total.fetch_add(1, std::memory_order_relaxed);
      last_error = "short read: got " + std::to_string(result.delivered_bytes) + " of " +
                   std::to_string(chunk.size) + " bytes";
      retriable = true;
    } else {
      return result.delivered_bytes;  // exact success
    }
    // Do not keep retrying (or sleeping) once admission is closed — a
    // shutdown or a device fail-stop has begun and this host read should
    // abort promptly rather than delay teardown.
    if (!retriable || attempt >= k_host_max_attempts || _ctx->gate().terminal()) {
      throw std::runtime_error("cuobj_rdma_reactor: " + last_error);
    }
    std::this_thread::sleep_for(k_backoff * attempt);
  }
}

size_t cuobj_rdma_reactor::data_transfer(const cuobj_chunked_rx_request& chunk,
                                         rdma_data_session& session,
                                         void* dst,
                                         bool& fail_stop_failure)
{
  auto result = session.get(*chunk.route, chunk.offset, chunk.size, dst);
  switch (result.commit) {
    case data_commit_state::not_sent:
      // Provably never left the process: this chunk fails, the transport
      // does not.  (fail_stop_failure stays false.)
      throw std::runtime_error(
        redact_rdma_tokens("cuobj_rdma_reactor: data GET not issued: " + result.transport_error));
    case data_commit_state::sent_unknown:
      fail_stop_failure = true;
      throw std::runtime_error(redact_rdma_tokens("cuobj_rdma_reactor: data GET outcome unknown: " +
                                                  result.transport_error));
    case data_commit_state::completed: break;
  }
  // Completion authority: ALL three legs must hold before the size report.
  // Status is checked before the byte count so a completed-but-error-status
  // result is reported as its HTTP status rather than misattributed as a
  // short read (and its short-read counter left untouched).
  fail_stop_failure = true;  // any invalid completion poisons the transport
  if (result.reply_tag.empty() || !_ctx->clients().tag_predicate(result.reply_tag)) {
    throw std::runtime_error(result.reply_tag.empty()
                               ? "cuobj_rdma_reactor: data completion reply tag missing"
                               : "cuobj_rdma_reactor: data completion reply tag rejected");
  }
  if (result.http_status != 200 && result.http_status != 206) {
    throw std::runtime_error(redact_rdma_tokens(
      "cuobj_rdma_reactor: data completion HTTP status " + std::to_string(result.http_status) +
      (result.transport_error.empty() ? "" : ": " + result.transport_error)));
  }
  if (result.delivered_bytes != chunk.size) {
    if (result.delivered_bytes < chunk.size) {
      _short_read_total.fetch_add(1, std::memory_order_relaxed);
    }
    throw std::runtime_error(
      "cuobj_rdma_reactor: short completion: " + std::to_string(result.delivered_bytes) + " of " +
      std::to_string(chunk.size) + " bytes");
  }
  fail_stop_failure = false;
  return result.delivered_bytes;
}

void cuobj_rdma_reactor::worker_loop()
{
  std::unique_ptr<rdma_data_session> session;
  {
    std::exception_ptr acquire_error;
    try {
      session = _ctx->clients().data_sessions->acquire();
      if (!session) { throw std::runtime_error("the data-session factory returned no session"); }
    } catch (...) {
      acquire_error = std::make_exception_ptr(std::runtime_error(
        "cuobj_rdma_reactor: data-session acquisition failed: " + current_exception_message()));
    }
    std::lock_guard lk{_lifecycle_mtx};
    ++_startup_reported;
    if (acquire_error != nullptr && _startup_error == nullptr) { _startup_error = acquire_error; }
    _startup_cv.notify_all();
    if (acquire_error != nullptr) { return; }
  }
  auto& gate = _ctx->gate();
  for (;;) {
    auto claimed = gate.claim();
    if (!claimed.has_value()) { return; }  // admission closed: the exit signal
    update_peak(_inflight_peak, _inflight.fetch_add(1, std::memory_order_relaxed) + 1);
    process_claimed(std::move(*claimed), *session);
    _inflight.fetch_sub(1, std::memory_order_relaxed);
  }
}

void cuobj_rdma_reactor::process_claimed(admission_gate::claimed_chunk claimed_arg,
                                         rdma_data_session& session)
{
  auto& gate = _ctx->gate();
  // Declaration order carries the teardown ordering (members are destroyed
  // in reverse): `claimed` is destroyed FIRST, publishing the request's
  // outcome by dropping the manager reference; then the permit releases its
  // outstanding-work count; the arena slot is released LAST — after a
  // fail-stop's transition, arena marking, and error report have all run,
  // so a slot whose remote write state is unknown is never handed to
  // another worker mid-transition.
  struct slot_holder {
    cuobj_rdma_reactor* self{nullptr};
    slot_pool* pool{nullptr};
    int slot{slot_pool::no_slot};
    ~slot_holder()
    {
      if (pool == nullptr) { return; }
      pool->release(slot);
      self->_slots_in_use.fetch_sub(1, std::memory_order_relaxed);
    }
  } held_slot;
  std::optional<admission_gate::admission_permit> permit;
  std::optional<admission_gate::claimed_chunk> claimed{std::move(claimed_arg)};
  bool fail_stop_failure = false;
  try {
    const auto& chunk = claimed->chunk();
    if (!chunk.is_device) {
      permit.emplace(gate.acquire_get(std::move(*claimed)));
      const size_t n = control_transfer(chunk);
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
    update_peak(_slots_in_use_peak, _slots_in_use.fetch_add(1, std::memory_order_relaxed) + 1);
    // Member-wise on purpose: assigning a braced temporary would run the
    // temporary's destructor and release the slot immediately.
    held_slot.self = this;
    held_slot.pool = ar.pool.get();
    held_slot.slot = slot;

    const auto& ops = _ctx->delivery_ops();

    // Captured-stream validation runs before the RDMA GET, in release builds
    // too, so a doomed request makes no remote side effect.  A sticky probe
    // result is context health, not slot lifetime: process-fatal even though
    // nothing is in flight.
    {
      cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
      const cudaError_t capture_rc    = ops.stream_capture_query(chunk.stream.value(), &capture);
      if (capture_rc != cudaSuccess) {
        if (is_context_fatal(capture_rc)) {
          invoke_fatal(ops, "stream capture query failed", capture_rc);
        }
        throw_on_cuda_error(capture_rc, "stream capture query failed");
      }
      if (capture != cudaStreamCaptureStatusNone) {
        throw std::runtime_error(
          "cuobj_rdma_reactor: RDMA device reads do not support captured streams");
      }
    }

    uint8_t* slot_ptr = ar.base + static_cast<size_t>(slot) * _config.arena_slot_size;
    permit.emplace(gate.acquire_get(std::move(*claimed)));
    const size_t n = data_transfer(chunk, session, slot_ptr, fail_stop_failure);

    // A fatal latched by another worker while our GET was in flight: never
    // start new CUDA work on the dead context.  The owning worker resolves
    // its own chunk here, before any CUDA work is enqueued; nothing is in
    // flight on the slot, so the plain RAII release below is safe.
    if (gate.first_fatal() != nullptr) {
      throw std::runtime_error("cuobj_rdma_reactor: chunk aborted after a fatal delivery state");
    }

    if (_ctx->flush_before_copy()) {
      const cudaError_t flush_rc = ops.flush();
      if (flush_rc != cudaSuccess) {
        if (is_context_fatal(flush_rc)) {
          invoke_fatal(ops, "GPUDirect writes flush failed", flush_rc);
        }
        throw_on_cuda_error(flush_rc, "GPUDirect writes flush failed");
      }
      _flush_total.fetch_add(1, std::memory_order_relaxed);
    }

    // The completion event exists before the enqueue, so a create failure
    // unwinds with nothing in flight (safe release, recoverable).  The
    // completion wait is inline: the D2D copy of one slot takes microseconds
    // against the blocking GET that preceded it, so no parked-copy machinery
    // is needed.
    event_guard event{ops};
    const cudaError_t create_rc = ops.event_create(&event.handle, cudaEventDisableTiming);
    if (create_rc != cudaSuccess) {
      if (is_context_fatal(create_rc)) {
        invoke_fatal(ops, "completion event create failed", create_rc);
      }
      throw_on_cuda_error(create_rc, "completion event create failed");
    }
    event.created = true;
    // ---- Delivery boundary: the first memcpy_async call (safety contract:
    // experimental/s3-rdma-transport-design.md, Section 3).  From here the
    // only returning path is an event wait that reports cudaSuccess; every
    // other outcome (error return or exception from the memcpy, the record,
    // or the wait) is process-fatal through invoke_fatal.  No unwinding
    // runs: the slot_holder and event_guard destructors above never fire,
    // no future resolves, and the arena is never freed.  Only static
    // literals reach the hook (no throwing formatting on this path).
    try {
      cudaError_t delivery_rc =
        ops.memcpy_async(chunk.dst, slot_ptr, n, cudaMemcpyDeviceToDevice, chunk.stream.value());
      if (delivery_rc != cudaSuccess) { invoke_fatal(ops, "D2D copy enqueue failed", delivery_rc); }
      delivery_rc = ops.event_record(event.handle, chunk.stream.value());
      if (delivery_rc != cudaSuccess) {
        invoke_fatal(ops, "completion event record failed", delivery_rc);
      }
      delivery_rc = ops.event_synchronize(event.handle);
      if (delivery_rc != cudaSuccess) {
        invoke_fatal(ops, "completion event wait failed", delivery_rc);
      }
    } catch (...) {
      invoke_fatal(ops, "post-boundary CUDA delivery operation threw", cudaErrorUnknown);
    }

    _bytes_total.fetch_add(n, std::memory_order_relaxed);
    chunk.manager->chunk_complete(n);
  } catch (...) {
    _error_total.fetch_add(1, std::memory_order_relaxed);
    auto error = std::current_exception();
    if (fail_stop_failure) {
      // One-shot: an ambiguous or invalid data-plane completion poisons the
      // transport rather than retrying.  Close admission and mark the arenas
      // FIRST — before any best-effort diagnostic — so a blocking log sink
      // cannot delay the cutoff and let another worker take a permit or issue
      // a GET after the fatal.  The transition marks the arenas; queued work
      // drains by error right here; the diagnostic is token-redacted already.
      auto batch = gate.fail_stop(error);
      if (batch.has_token()) {
        batch.error_complete_all(error);
        gate.complete_drain(std::move(batch).take_token());
      }
      try {
        SIRIUS_LOG_ERROR("cuobj_rdma_reactor: data-plane fail-stop: {}",
                         current_exception_message());
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
    // Publication before release: report the failing chunk while the claim
    // guard / get permit is still held (their destructors run after this
    // frame, claimed first).
    claimed->report_error(error);
  }
}

cuobj_rdma_reactor::arena& cuobj_rdma_reactor::arena_for_device(int device_id)
{
  {
    std::lock_guard lk{_arena_mtx};
    if (auto it = _arenas.find(device_id); it != _arenas.end()) { return *it->second; }
  }
  // Creation only: refuse when admission is closed, and hold the gate's
  // creation scope across the insert so the arena is either present at a
  // fail-stop's marking point or never created (lock order gate -> arena).
  auto scope = _ctx->gate().enter_creation();
  std::lock_guard lk{_arena_mtx};
  if (auto it = _arenas.find(device_id); it != _arenas.end()) { return *it->second; }

  auto ar = std::make_unique<arena>();
  throw_on_cuda_error(
    cudaMalloc(reinterpret_cast<void**>(&ar->base), _config.max_inflight * _config.arena_slot_size),
    "landing arena allocation failed");
  ar->pool = std::make_unique<slot_pool>(_config.max_inflight);
  // The arena owns a dedicated session for its registration lifetime:
  // worker sessions die when workers exit, but deregistration happens at
  // arena teardown, after the join.  A factory that yields no session is a
  // capability failure, not a null dereference.
  ar->registrar = _ctx->clients().data_sessions->acquire();
  if (!ar->registrar) {
    throw std::runtime_error(
      "cuobj_rdma_reactor: the data-session factory returned no session for arena registration");
  }
  ar->registrar->register_memory(ar->base, _config.max_inflight * _config.arena_slot_size);
  ar->registered = true;  // only after register_memory returns (a throw leaves it false)
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
  chunk->route   = std::make_shared<const rx_route>(rx_route{file.bucket(), file.key()});
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

  // One descriptor for the WHOLE logical request; the admission gate
  // materializes slot-sized chunks lazily at claim time.  The manager's
  // fan-in count must match that chunking exactly.
  const size_t slot_size = cfg.arena_slot_size != 0 ? cfg.arena_slot_size : (64UL << 10);
  const size_t n_chunks  = (total + slot_size - 1) / slot_size;
  auto manager           = std::make_shared<request_manager>(total, n_chunks);

  std::vector<std::unique_ptr<cuobj_chunked_rx_request>> chunks;
  auto whole       = std::make_unique<cuobj_chunked_rx_request>();
  whole->route     = std::make_shared<const rx_route>(rx_route{file.bucket(), file.key()});
  whole->offset    = offset;
  whole->size      = total;
  whole->dst       = dst;
  whole->is_device = true;
  whole->stream    = stream;
  whole->device_id = device_id;
  whole->manager   = std::move(manager);
  chunks.push_back(std::move(whole));
  return request_type::create(std::move(chunks));
}

void cuobj_rdma_reactor::enqueue(request_type_ptr req)
{
  if (!req) { return; }
  auto chunks = req->get_all_chunks();
  if (chunks.empty()) { return; }

  std::exception_ptr failure;
  {
    auto& gate = _ctx->gate();
    try {
      bool admitted = false;
      for (auto& descriptor : chunks) {
        if (!descriptor) { continue; }
        admission_gate::envelope env{
          std::string{},
          std::string{},
          descriptor->offset,
          descriptor->size,
          descriptor->dst,
          descriptor->is_device,
          descriptor->stream,
          descriptor->device_id,
          descriptor->manager,
          descriptor->is_device ? _config.arena_slot_size : descriptor->size};
        gate.submit(descriptor->route,
                    std::move(env));  // may block at the cap; throws once terminal
        admitted = true;
      }
      if (admitted) { _requests_total.fetch_add(1, std::memory_order_relaxed); }
    } catch (...) {
      failure = std::current_exception();
    }
  }
  if (failure) {
    for (auto& descriptor : chunks) {
      if (descriptor && descriptor->manager) { descriptor->manager->report_error(failure); }
    }
  }
}

size_t cuobj_rdma_reactor::host_read(const io_object_type& file,
                                     size_t offset,
                                     size_t size,
                                     uint8_t* dst)
{
  const size_t n = clipped_size(file, offset, size);
  if (n == 0) { return 0; }

  auto manager = std::make_shared<request_manager>(n, 1);
  auto future  = manager->get_future();
  auto route   = std::make_shared<const rx_route>(rx_route{file.bucket(), file.key()});
  admission_gate::envelope env{std::string{},
                               std::string{},
                               offset,
                               n,
                               dst,
                               /*is_device=*/false,
                               rmm::cuda_stream_view{},
                               /*device_id=*/-1,
                               std::move(manager),
                               /*slot_bytes=*/n};
  _ctx->gate().submit(std::move(route), std::move(env));
  _requests_total.fetch_add(1, std::memory_order_relaxed);
  return std::move(future).get();
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
