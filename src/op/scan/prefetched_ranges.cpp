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

#include "op/scan/prefetched_ranges.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::op::scan {

namespace {

/// Binds @p device into @p bind (left empty when the switch fails).  Destructors use it: a failed
/// device switch must not terminate the process, and the frees that follow are still attempted.
void try_bind_device(std::optional<rmm::cuda_set_device_raii>& bind, int device) noexcept
{
  try {
    bind.emplace(rmm::cuda_device_id{device});
  } catch (...) {
    bind.reset();
  }
}

std::exception_ptr make_cuda_failure(char const* what, cudaError_t status) noexcept
{
  try {
    return std::make_exception_ptr(
      std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)));
  } catch (...) {
    return std::current_exception();
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// device_stream_pool
//===----------------------------------------------------------------------===//

std::shared_ptr<device_stream_pool> device_stream_pool::create(int device, std::size_t n)
{
  rmm::cuda_set_device_raii bind{rmm::cuda_device_id{device}};
  auto pool     = std::shared_ptr<device_stream_pool>(new device_stream_pool);
  pool->_device = device;
  pool->_streams.reserve(std::max<std::size_t>(n, 1));
  for (std::size_t i = 0; i < std::max<std::size_t>(n, 1); ++i) {
    pool->_streams.push_back(
      std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking));
  }
  return pool;
}

device_stream_pool::lease device_stream_pool::acquire() const
{
  auto const i = _next.fetch_add(1, std::memory_order_relaxed) % _streams.size();
  return lease{.owner = shared_from_this(), .view = ::cuda::stream_ref{_streams[i]->value()}};
}

device_stream_pool::~device_stream_pool()
{
  if (_streams.empty()) { return; }
  // A stream belongs to the device it was created on; destroy each there, after the work still
  // queued on it (stream-ordered frees of the buffers it allocated) has finished.
  std::optional<rmm::cuda_set_device_raii> bind;
  try_bind_device(bind, _device);
  for (auto& stream : _streams) {
    stream->synchronize_no_throw();
  }
  _streams.clear();
}

//===----------------------------------------------------------------------===//
// prefetched_ranges
//===----------------------------------------------------------------------===//

prefetched_ranges::prefetched_ranges(int device_id,
                                     device_stream_pool::lease producer,
                                     std::vector<device_ranges> ranges,
                                     std::size_t logical_bytes,
                                     std::size_t reserved_bytes,
                                     claim_hook on_claim,
                                     release_hook on_release,
                                     std::shared_ptr<std::mutex> allocation_mutex) noexcept
  : _device_id(device_id),
    _producer(std::move(producer)),
    _allocation_mutex(std::move(allocation_mutex)),
    _total_bytes(logical_bytes),
    _reserved_bytes(reserved_bytes),
    _on_claim(std::move(on_claim)),
    _on_release(std::move(on_release)),
    _ranges(std::move(ranges))
{
  // The event must belong to the buffers' device.  A failure is parked for set_ready() so the
  // constructor can stay noexcept (the buffers must be freed device bound, not by an unwind).
  std::optional<rmm::cuda_set_device_raii> bind;
  try_bind_device(bind, _device_id);
  if (!bind) {
    _event_error =
      make_cuda_failure("device prefetch could not bind its device", cudaErrorInvalidDevice);
    return;
  }
  if (auto const status = cudaEventCreateWithFlags(&_ready_event, cudaEventDisableTiming);
      status != cudaSuccess) {
    _ready_event = nullptr;
    _event_error = make_cuda_failure("device prefetch ready event creation failed", status);
    (void)cudaGetLastError();
  }
}

prefetched_ranges::~prefetched_ranges()
{
  claim();
  if (!_released.load(std::memory_order_acquire)) {
    auto const reason = _wrong_device.load(std::memory_order_acquire) ? release_reason::wrong_device
                        : current() == state::failed                  ? release_reason::failed
                                                                      : release_reason::unconsumed;
    fire_release(reason);
  }
  // Frees are stream-ordered on each buffer's stream -- the producer stream, still alive because
  // _producer is destroyed after this body -- and belong to the buffers' device.
  std::optional<rmm::cuda_set_device_raii> bind;
  try_bind_device(bind, _device_id);
  if (!_ranges.empty()) {
    // Only the untaken producer-stream frees need the allocation mutex (a taken prefetch has
    // moved its buffers out and frees nothing here).  The hooks above ran outside it.
    std::unique_lock<std::mutex> allocation_lock;
    if (_allocation_mutex) { allocation_lock = std::unique_lock<std::mutex>(*_allocation_mutex); }
    _ranges.clear();
  }
  if (_ready_event != nullptr) {
    (void)cudaEventDestroy(_ready_event);
    _ready_event = nullptr;
  }
}

bool prefetched_ranges::settle(state next, std::exception_ptr error, bool abandoned) noexcept
{
  {
    std::lock_guard lock(_mutex);
    if (_state != state::pending) { return false; }
    if (abandoned) {
      // Building the message allocates, and this is noexcept: under memory pressure keep the
      // allocation failure itself as the recorded error, but still settle and wake the waiter.
      try {
        error = std::make_exception_ptr(
          std::runtime_error("device prefetch abandoned: consumer overtook the readahead"));
      } catch (...) {
        error = std::current_exception();
      }
    }
    _state     = next;
    _failure   = std::move(error);
    _abandoned = abandoned;
  }
  _cv.notify_all();
  return true;
}

bool prefetched_ranges::set_ready() noexcept
{
  if (current() != state::pending) { return false; }
  if (_event_error) { return settle(state::failed, _event_error, false); }
  // Capture everything enqueued on the producer stream so far -- the allocation and any stream
  // work ahead of this point -- so take() can order the consumer's stream after it.
  std::optional<rmm::cuda_set_device_raii> bind;
  try_bind_device(bind, _device_id);
  if (!bind) {
    return settle(
      state::failed,
      make_cuda_failure("device prefetch could not bind its device", cudaErrorInvalidDevice),
      false);
  }
  if (auto const status = cudaEventRecord(_ready_event, _producer.view.get());
      status != cudaSuccess) {
    (void)cudaGetLastError();
    return settle(
      state::failed, make_cuda_failure("device prefetch ready event record failed", status), false);
  }
  return settle(state::ready, nullptr, false);
}

bool prefetched_ranges::set_failed(std::exception_ptr error) noexcept
{
  return settle(state::failed, std::move(error), false);
}

bool prefetched_ranges::abandon() noexcept { return settle(state::failed, nullptr, true); }

prefetched_ranges::state prefetched_ranges::wait_for(std::chrono::milliseconds timeout) const
{
  auto const start = std::chrono::steady_clock::now();
  state seen{state::pending};
  {
    std::unique_lock lock(_mutex);
    _cv.wait_for(lock, timeout, [this] { return _state != state::pending; });
    seen = _state;
  }
  auto const waited = std::chrono::steady_clock::now() - start;
  _wait_ns.fetch_add(static_cast<std::uint64_t>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
                     std::memory_order_relaxed);
  return seen;
}

prefetched_ranges::state prefetched_ranges::current() const noexcept
{
  std::lock_guard lock(_mutex);
  return _state;
}

std::exception_ptr prefetched_ranges::failure() const noexcept
{
  std::lock_guard lock(_mutex);
  return _failure;
}

bool prefetched_ranges::abandoned() const noexcept
{
  std::lock_guard lock(_mutex);
  return _abandoned;
}

void prefetched_ranges::flag_wrong_device() noexcept
{
  _wrong_device.store(true, std::memory_order_release);
}

void prefetched_ranges::claim() noexcept
{
  if (_claimed.exchange(true, std::memory_order_acq_rel)) { return; }
  if (!_on_claim) { return; }
  _on_claim();
  // Drop the hook's captures (the budget) now rather than with this object.
  _on_claim = nullptr;
}

std::optional<std::vector<prefetched_ranges::device_ranges>> prefetched_ranges::take(
  ::cuda::stream_ref consumer_stream)
{
  // The returned buffers are freed on the caller's current device: refuse any other.
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device != _device_id) {
    flag_wrong_device();
    return std::nullopt;
  }
  std::vector<device_ranges> out;
  {
    std::lock_guard lock(_mutex);
    if (_state != state::ready || _taken || _wrong_device.load(std::memory_order_acquire)) {
      return std::nullopt;
    }
    // Order the consumer's stream after every producer-stream write BEFORE touching the buffers:
    // the rebind below only retargets the free, it does not order the decode after the fill.
    if (cudaStreamWaitEvent(consumer_stream.get(), _ready_event, 0) != cudaSuccess) {
      (void)cudaGetLastError();
      return std::nullopt;
    }
    _taken = true;
    // The consumer frees these after its decode: rebinding makes that free trail the decode
    // kernels on the consumer's stream instead of racing them on the producer stream.
    for (auto& entry : _ranges) {
      for (auto& buffer : entry.buffers) {
        buffer.set_stream(consumer_stream);
      }
    }
    out = std::move(_ranges);
    _ranges.clear();
  }
  claim();
  fire_release(release_reason::consumed);
  return out;
}

std::span<prefetched_ranges::device_ranges const> prefetched_ranges::peek() const noexcept
{
  std::lock_guard lock(_mutex);
  if (_state != state::ready || _taken || _wrong_device.load(std::memory_order_acquire)) {
    return {};
  }
  return _ranges;
}

bool prefetched_ranges::taken() const noexcept
{
  std::lock_guard lock(_mutex);
  return _taken;
}

std::size_t prefetched_ranges::entry_count() const noexcept
{
  std::lock_guard lock(_mutex);
  return _ranges.size();
}

void prefetched_ranges::fire_release(release_reason reason) noexcept
{
  if (_released.exchange(true, std::memory_order_acq_rel)) { return; }
  if (!_on_release) { return; }
  _on_release(release_event{.reason         = reason,
                            .bytes          = _total_bytes,
                            .wait_ns        = _wait_ns.load(std::memory_order_relaxed),
                            .reserved_bytes = _reserved_bytes});
  // Drop the hook's captures (the per-device window) now rather than with this object.
  _on_release = nullptr;
}

}  // namespace sirius::op::scan
