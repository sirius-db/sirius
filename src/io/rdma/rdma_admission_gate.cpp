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

#include "io/rdma/rdma_admission_gate.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace sirius::io::rdma {

namespace {

constexpr const char* k_closed_message = "s3_rdma admission_gate: transport closed";

}  // namespace

// ---- envelope_ring ---------------------------------------------------------

admission_gate::envelope_ring::envelope_ring(size_t capacity) : _capacity(capacity)
{
  // The size computation must not wrap: a wrapped byte count would allocate
  // less storage than the ring later placement-constructs into.
  if (capacity > std::numeric_limits<size_t>::max() / sizeof(queued_request)) {
    throw std::overflow_error("admission_gate: queue_cap storage size overflows");
  }
  // The one allocation, before the gate is published; every later slot write
  // is a placement construction into this storage.
  _storage = std::make_unique<std::byte[]>(capacity * sizeof(queued_request));
}

admission_gate::envelope_ring::envelope_ring(envelope_ring&& other) noexcept
  : _storage(std::move(other._storage)),
    _capacity(other._capacity),
    _head(other._head),
    _count(other._count)
{
  other._capacity = 0;
  other._head     = 0;
  other._count    = 0;
}

admission_gate::envelope_ring& admission_gate::envelope_ring::operator=(
  envelope_ring&& other) noexcept
{
  if (this != &other) {
    destroy_all();
    _storage        = std::move(other._storage);
    _capacity       = other._capacity;
    _head           = other._head;
    _count          = other._count;
    other._capacity = 0;
    other._head     = 0;
    other._count    = 0;
  }
  return *this;
}

admission_gate::envelope_ring::~envelope_ring() { destroy_all(); }

admission_gate::queued_request* admission_gate::envelope_ring::slot(size_t logical_index) noexcept
{
  const size_t physical = (_head + logical_index) % _capacity;
  return std::launder(
    reinterpret_cast<queued_request*>(_storage.get() + physical * sizeof(queued_request)));
}

void admission_gate::envelope_ring::emplace(queued_request&& request) noexcept(
  std::is_nothrow_move_constructible_v<queued_request>)
{
  assert(_count < _capacity);
  ::new (
    static_cast<void*>(_storage.get() + ((_head + _count) % _capacity) * sizeof(queued_request)))
    queued_request(std::move(request));
  ++_count;
}

admission_gate::queued_request& admission_gate::envelope_ring::front() noexcept
{
  assert(_count > 0);
  return *slot(0);
}

void admission_gate::envelope_ring::pop_front() noexcept
{
  assert(_count > 0);
  slot(0)->~queued_request();
  _head = (_head + 1) % _capacity;
  --_count;
}

void admission_gate::envelope_ring::destroy_all() noexcept
{
  while (_count > 0) {
    pop_front();
  }
}

// ---- claimed_chunk ---------------------------------------------------------

admission_gate::claimed_chunk::claimed_chunk(admission_gate* gate,
                                             cuobj_chunked_rx_request chunk) noexcept
  : _gate(gate), _chunk(std::move(chunk)), _engaged(true)
{
}

admission_gate::claimed_chunk::claimed_chunk(claimed_chunk&& other) noexcept
  : _gate(other._gate), _chunk(std::move(other._chunk)), _engaged(other._engaged)
{
  other._gate    = nullptr;
  other._engaged = false;
}

admission_gate::claimed_chunk& admission_gate::claimed_chunk::operator=(
  claimed_chunk&& other) noexcept
{
  if (this != &other) {
    _chunk.manager.reset();
    if (_gate != nullptr && _engaged) { _gate->release_claim_guard(); }
    _gate          = other._gate;
    _chunk         = std::move(other._chunk);
    _engaged       = other._engaged;
    other._gate    = nullptr;
    other._engaged = false;
  }
  return *this;
}

admission_gate::claimed_chunk::~claimed_chunk()
{
  // Drop the manager reference FIRST: when this holds the last reference,
  // the request's future publishes here — before the guard release below can
  // let await_closed return.  Aborts stay observable: report_error, publish,
  // then the outstanding-work count drops.
  _chunk.manager.reset();
  if (_gate != nullptr && _engaged) { _gate->release_claim_guard(); }
}

void admission_gate::claimed_chunk::report_error(std::exception_ptr error) noexcept
{
  if (!_chunk.manager) { return; }
  try {
    _chunk.manager->report_error(error);
  } catch (...) {  // NOLINT(bugprone-empty-catch) — first-error-wins latch; never propagate
  }
}

// ---- admission_permit ------------------------------------------------------

admission_gate::admission_permit::admission_permit(admission_gate* gate, kind which) noexcept
  : _gate(gate), _kind(which)
{
}

admission_gate::admission_permit::admission_permit(admission_permit&& other) noexcept
  : _gate(other._gate), _kind(other._kind)
{
  other._gate = nullptr;
}

admission_gate::admission_permit& admission_gate::admission_permit::operator=(
  admission_permit&& other) noexcept
{
  if (this != &other) {
    if (_gate != nullptr) { _gate->release_permit(_kind); }
    _gate       = other._gate;
    _kind       = other._kind;
    other._gate = nullptr;
  }
  return *this;
}

admission_gate::admission_permit::~admission_permit()
{
  if (_gate != nullptr) { _gate->release_permit(_kind); }
}

// ---- drain_batch -----------------------------------------------------------

void admission_gate::drain_batch::error_complete_all(std::exception_ptr error) noexcept
{
  while (!_ring.empty()) {
    auto& env = _ring.front().env;
    if (env.manager) {
      try {
        env.manager->report_error(error);
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
    // Releasing the envelope drops its manager reference, publishing the
    // request's failed future for every chunk that was never generated.
    _ring.pop_front();
  }
}

admission_gate::drain_token admission_gate::drain_batch::take_token() && noexcept
{
  assert(_has_token && "take_token on a tokenless (loser) drain_batch");
  _has_token = false;
  drain_token token;
  token._valid = true;
  return token;
}

// ---- admission_gate --------------------------------------------------------

admission_gate::admission_gate(size_t queue_cap) : _queue(queue_cap)
{
  if (queue_cap == 0) { throw std::invalid_argument("admission_gate: queue_cap must be positive"); }
}

admission_gate::~admission_gate() = default;

void admission_gate::bind_arena_marker(arena_marker fn, void* ctx) noexcept
{
  std::lock_guard lk{_mtx};
  assert(_marker == nullptr && "the arena marker binds exactly once");
  if (_marker == nullptr) {
    _marker     = fn;
    _marker_ctx = ctx;
  }
}

void admission_gate::throw_terminal_locked() const
{
  if (_first_fatal) { std::rethrow_exception(_first_fatal); }
  throw std::runtime_error(k_closed_message);
}

admission_gate::admission_permit admission_gate::acquire_control()
{
  std::lock_guard lk{_mtx};
  if (_admission_closed) { throw_terminal_locked(); }
  ++_control_permits;
  return admission_permit{this, admission_permit::kind::control};
}

admission_gate::creation_scope admission_gate::enter_creation()
{
  std::unique_lock lk{_mtx};
  if (_admission_closed) { throw_terminal_locked(); }
  return creation_scope{std::move(lk)};
}

void admission_gate::submit(envelope e)
{
  // Convenience form: derive the route here, before the terminal check and
  // reservation, so a failed allocation rejects the submit with no gate
  // state touched.
  auto route = std::make_shared<const rx_route>(rx_route{std::move(e.bucket), std::move(e.key)});
  submit(std::move(route), std::move(e));
}

void admission_gate::submit(std::shared_ptr<const rx_route> route, envelope e)
{
  // Checked before the lock: a null route would otherwise surface as a GET-
  // time null dereference in a release build.
  if (route == nullptr) {
    throw std::invalid_argument("admission_gate::submit: route must not be null");
  }
  std::unique_lock lk{_mtx};
  if (_admission_closed) { throw_terminal_locked(); }
  if (_queue.size() >= _queue.capacity()) {
    _envelope_wait_total.fetch_add(1, std::memory_order_relaxed);
    const auto wait_start = std::chrono::steady_clock::now();
    _submit_cv.wait(lk, [&] { return _admission_closed || _queue.size() < _queue.capacity(); });
    _envelope_wait_ns_total.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - wait_start)
                              .count()),
      std::memory_order_relaxed);
    // A terminal wake releases the wait without ever reserving a slot.
    if (_admission_closed) { throw_terminal_locked(); }
  }
  queued_request request{std::move(route), std::move(e)};
  static_assert(noexcept(_queue.emplace(std::move(request))),
                "the post-reservation commit must have no throw point");
  _queue.emplace(std::move(request));  // cannot throw: empty pre-allocated slot + nothrow move
  const auto depth = static_cast<uint64_t>(_queue.size());
  if (depth > _envelope_depth_peak.load(std::memory_order_relaxed)) {
    _envelope_depth_peak.store(depth, std::memory_order_relaxed);
  }
  _claim_cv.notify_one();
}

std::optional<admission_gate::claimed_chunk> admission_gate::claim()
{
  std::unique_lock lk{_mtx};
  _claim_cv.wait(lk, [&] { return _admission_closed || !_queue.empty(); });
  if (_admission_closed || _queue.empty()) { return std::nullopt; }

  queued_request& front = _queue.front();
  envelope& env         = front.env;
  assert(env.slot_bytes > 0);
  const size_t delta = _front_cursor * env.slot_bytes;
  cuobj_chunked_rx_request chunk;
  chunk.route     = front.route;  // shared, pre-built at submit: no allocation here
  chunk.offset    = env.offset + delta;
  chunk.size      = std::min(env.slot_bytes, env.size - delta);
  chunk.dst       = env.dst + delta;
  chunk.is_device = env.is_device;
  chunk.stream    = env.stream;
  chunk.device_id = env.device_id;
  chunk.manager   = env.manager;
  ++_front_cursor;
  if (delta + chunk.size >= env.size) {
    // The front envelope's last chunk: the queue depth drops here, so cap
    // waiters get their slot.
    _queue.pop_front();
    _front_cursor = 0;
    _submit_cv.notify_one();
  }
  if (!_queue.empty()) {
    // Baton pass: more chunks remain claimable (this envelope or the next),
    // and submit only woke ONE worker per envelope.  Without this, a large
    // request would be served by a single worker and never reach the
    // max_inflight concurrency the pool exists for.
    _claim_cv.notify_one();
  }
  ++_outstanding_data;  // the claim guard
  return claimed_chunk{this, std::move(chunk)};
}

admission_gate::admission_permit admission_gate::acquire_get(claimed_chunk&& c)
{
  std::lock_guard lk{_mtx};
  // On the throwing path @p c is untouched: the caller still owns the
  // claimed chunk and reports its error before the guard drops.
  if (_admission_closed) { throw_terminal_locked(); }
  c._engaged = false;  // guard transfers into the permit; the count is continuous
  return admission_permit{this, admission_permit::kind::data};
}

admission_gate::drain_batch admission_gate::transition(std::exception_ptr fatal) noexcept
{
  drain_batch batch;
  std::lock_guard lk{_mtx};
  bool latched = false;
  if (fatal && !_first_fatal) {
    _first_fatal = std::move(fatal);
    _fail_stop_total.fetch_add(1, std::memory_order_relaxed);
    latched = true;
  }
  _admission_closed = true;
  if (!_queue_detached) {
    _queue_detached = true;
    _front_cursor   = 0;
    if (!_queue.empty()) {
      // O(1) takeover of the ring backing: no allocation, no per-envelope
      // move.  The queue stays detached forever — admission is closed, so no
      // replacement storage is ever needed.
      batch    = drain_batch{std::move(_queue), /*token=*/true};
      _drained = false;
    }
  }
  if (latched && _marker != nullptr) {
    // Marking precedes publication (the mutex release): the marker takes the
    // arena mutex under the held gate mutex — lock order gate → arena.
    _marker(_marker_ctx);
  }
  _submit_cv.notify_all();
  _claim_cv.notify_all();
  _closed_cv.notify_all();
  return batch;
}

admission_gate::drain_batch admission_gate::fail_stop(std::exception_ptr fatal) noexcept
{
  return transition(std::move(fatal));
}

admission_gate::drain_batch admission_gate::begin_close() noexcept { return transition(nullptr); }

void admission_gate::complete_drain(drain_token&& token) noexcept
{
  if (!token._valid) { return; }
  token._valid = false;
  std::lock_guard lk{_mtx};
  _drained = true;
  _closed_cv.notify_all();
}

void admission_gate::await_closed()
{
  std::unique_lock lk{_mtx};
  _closed_cv.wait(lk, [&] {
    return _admission_closed && _control_permits == 0 && _outstanding_data == 0 && _drained;
  });
}

bool admission_gate::terminal() const noexcept
{
  std::lock_guard lk{_mtx};
  return _admission_closed;
}

std::exception_ptr admission_gate::first_fatal() const noexcept
{
  std::lock_guard lk{_mtx};
  return _first_fatal;
}

uint64_t admission_gate::fail_stop_total() const noexcept
{
  return _fail_stop_total.load(std::memory_order_relaxed);
}

uint64_t admission_gate::envelope_wait_total() const noexcept
{
  return _envelope_wait_total.load(std::memory_order_relaxed);
}

uint64_t admission_gate::envelope_wait_ns_total() const noexcept
{
  return _envelope_wait_ns_total.load(std::memory_order_relaxed);
}

uint64_t admission_gate::envelope_depth_peak() const noexcept
{
  return _envelope_depth_peak.load(std::memory_order_relaxed);
}

void admission_gate::release_claim_guard() noexcept
{
  std::lock_guard lk{_mtx};
  assert(_outstanding_data > 0);
  --_outstanding_data;
  _closed_cv.notify_all();
}

void admission_gate::release_permit(admission_permit::kind which) noexcept
{
  std::lock_guard lk{_mtx};
  if (which == admission_permit::kind::control) {
    assert(_control_permits > 0);
    --_control_permits;
  } else {
    assert(_outstanding_data > 0);
    --_outstanding_data;
  }
  _closed_cv.notify_all();
}

}  // namespace sirius::io::rdma
