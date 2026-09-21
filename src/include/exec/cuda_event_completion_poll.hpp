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

#include "exec/invocable.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace sirius::exec {

// Historical name: stream callbacks, not CUDA events, publish each stream's
// completed prefix. Host callers explicitly drain ownership-bearing functions;
// the registry creates no worker. See the public types below for lifetime rules.
inline constexpr std::size_t cacheline_v    = 64;
inline constexpr std::uint64_t no_pending_v = ~std::uint64_t{0};
// Runs on a host drainer with the boundary's CUDA status. Keep it short and
// noexcept. Reentrant drain() is allowed, but waiting for retirement or calling
// quiesce()/fail_all() on the same lane/registry would deadlock.
using retire_fn = invocable<void(cudaError_t) noexcept>;

struct alignas(cacheline_v) completion_slot {
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> first_failed{no_pending_v};
  std::atomic<cudaError_t> first_error{cudaSuccess};
};
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<cudaError_t>::is_always_lock_free);

namespace detail {
// Deterministic error/lifetime tests can substitute this small runtime seam.
// An alternative must obey CUDA's exactly-once, in-stream callback ordering.
struct completion_cuda_api {
  decltype(&cudaStreamAddCallback) add_callback = &cudaStreamAddCallback;
  decltype(&cudaStreamQuery) query              = &cudaStreamQuery;
  decltype(&cudaStreamSynchronize) synchronize  = &cudaStreamSynchronize;
};

inline void CUDART_CB bump_ticket(cudaStream_t, cudaError_t status, void* data) noexcept
{
  auto& slot        = *static_cast<completion_slot*>(data);
  const auto ticket = slot.completed.load(std::memory_order_relaxed) + 1;
  // Only ordered CUDA callbacks write this state. API errors on the host must
  // never manufacture a failed/completed device frontier.
  if (status != cudaSuccess && slot.first_failed.load(std::memory_order_relaxed) == no_pending_v) {
    slot.first_error.store(status, std::memory_order_relaxed);
    slot.first_failed.store(ticket, std::memory_order_release);
  }
  slot.completed.store(ticket, std::memory_order_release);  // last access to slot
}

struct pending_entry {
  std::uint64_t ticket;
  retire_fn fn;
};

inline void completion_backoff(std::chrono::microseconds& delay) noexcept
{
  std::this_thread::sleep_for(delay);
  delay = std::min(std::chrono::microseconds{256}, delay * 2);
}
}  // namespace detail

class retire_lane;
/**
 * Exclusive submission scope for a borrowed stream. Holds the lane's producer
 * lock until destruction, even after commit(); use only from the owning thread.
 * The lane must outlive this object. Stage captures before launching work, then
 * commit after the last launch. Destruction commits automatically but cannot
 * report errors, so explicitly commit when the result matters.
 */
class [[nodiscard]] submission {
 public:
  submission(submission&&)                 = delete;
  submission& operator=(submission&&)      = delete;
  submission(submission const&)            = delete;
  submission& operator=(submission const&) = delete;
  ~submission();

  // Borrowed handle for work enqueued inside this scope, before commit().
  [[nodiscard]] cudaStream_t stream() const noexcept;

  // Takes ownership of fn's captures. Stage BEFORE enqueuing device work,
  // including construction of fn: either can allocate/throw. Throws
  // invalid_argument for an empty fn, or logic_error after commit().
  void on_retire(retire_fn fn);
  // Publishes one boundary for all staged functions, without allocating or
  // invoking them inline. Idempotent; an empty submission installs no callback.
  // Installation failure closes the lane and synchronizes the stream, which
  // can block. A fallback sync error takes precedence in the return value;
  // lane.enqueue_error() retains the cause. Unconfirmed captures remain queued;
  // successful recovery makes them drainable with the installation error.
  [[nodiscard]] cudaError_t commit() noexcept;

 private:
  friend class retire_lane;
  submission(retire_lane& lane, std::unique_lock<std::mutex> lock) noexcept
    : lane_(&lane), lock_(std::move(lock))
  {
  }
  retire_lane* lane_;
  std::unique_lock<std::mutex> lock_;
  std::list<detail::pending_entry> staged_;
  bool committed_{false};
  cudaError_t result_{cudaSuccess};
};

/**
 * Owns completion state and queued captures for one borrowed CUDA stream.
 * Producers serialize through begin(); concurrent drainers invoke functions in
 * FIFO order, including capture destruction. No cross-lane ordering is implied.
 * Stop producers and externally serialize lifecycle calls (detach, quiesce,
 * fail_all); drains may overlap them. All calls and submissions must end before
 * destruction, which quiesces and terminates if unresolved captures remain.
 * Keep the stream alive until detach(), and the lane alive until CUDA callbacks
 * finish. A failed synchronization does not establish either completion.
 */
class retire_lane {
 public:
  explicit retire_lane(cudaStream_t stream,
                       detail::completion_cuda_api api     = {},
                       std::atomic<std::uint64_t>* retired = nullptr) noexcept
    : stream_(stream), api_(api), registry_retired_(retired)
  {
  }
  ~retire_lane()
  {
    if (!idle()) { (void)quiesce(); }
    if (!idle()) { std::terminate(); }
  }
  retire_lane(retire_lane const&)            = delete;
  retire_lane& operator=(retire_lane const&) = delete;

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

  // Blocks on another submission/lifecycle call. Throws logic_error if detached
  // or closed by callback-installation failure. The returned scope owns the lock.
  [[nodiscard]] submission begin()
  {
    std::unique_lock lock(submit_m_);
    if (detached_ || closed_) {
      throw std::logic_error("submission on detached or failed completion lane");
    }
    return submission{*this, std::move(lock)};
  }

  // Invokes ready functions inline and returns their count, without allocating
  // or waiting for device progress/another drainer. May briefly take the queue
  // mutex. A busy drainer returns zero so it cannot stall a registry scan.
  // Ownership spans invocation and capture destruction to preserve FIFO.
  std::size_t drain() noexcept
  {
    const auto done = completed();
    if (done < oldest_hint_.load(std::memory_order_relaxed)) { return 0; }
    if (draining_.test_and_set(std::memory_order_acquire)) { return 0; }
    const auto count = drain_owned(done, false, cudaSuccess);
    release_drainer();
    return count;
  }

  // Advisory lane-local ticket, or no_pending_v. May be stale and excludes the
  // active drainer's batch; never compare ticket values across lanes.
  [[nodiscard]] std::uint64_t oldest_pending_hint() const noexcept
  {
    return oldest_hint_.load(std::memory_order_relaxed);
  }

  // Snapshot under the queue mutex, including actively retiring functions.
  // no_pending_v means no queued/active functions (not uncommitted submissions).
  [[nodiscard]] std::uint64_t oldest_pending_locked() noexcept
  {
    std::lock_guard lock(pending_m_);
    return std::min(active_ticket_, pending_.empty() ? no_pending_v : pending_.front().ticket);
  }

  // Snapshot of committed host retirement, including release of captures.
  [[nodiscard]] bool idle() noexcept { return oldest_pending_locked() == no_pending_v; }

  // Sticky error snapshots; observing an error never authorizes retirement.
  [[nodiscard]] bool faulted() const noexcept { return fault_error() != cudaSuccess; }

  // First delivered device error, otherwise the first observed host API error.
  [[nodiscard]] cudaError_t fault_error() const noexcept
  {
    if (slot_.first_failed.load(std::memory_order_acquire) != no_pending_v) {
      return slot_.first_error.load(std::memory_order_relaxed);
    }
    return host_error_.load(std::memory_order_acquire);
  }

  // Callback-installation error, even if the fallback synchronization also fails.
  [[nodiscard]] cudaError_t enqueue_error() const noexcept
  {
    return enqueue_error_.load(std::memory_order_acquire);
  }

  // Blocks until this lane's target boundary is delivered/recovered or a health
  // error is observed. Does not invoke retire functions; success does not imply
  // host retirement or a successful device status. Registry waiting never uses
  // this: a stalled stream must not hide progress in another lane.
  cudaError_t wait_for(std::uint64_t target) noexcept
  {
    auto delay = std::chrono::microseconds{4};
    for (;;) {
      if (completed() >= target) { return cudaSuccess; }
      if (auto error = poll_health(); error != cudaSuccess) { return error; }
      detail::completion_backoff(delay);
    }
  }

  // Queries CUDA unless a submitter holds the lock. Success may mean the query
  // was skipped, not that work completed. Returns InvalidResourceHandle after
  // detach; other query errors are remembered without advancing completion.
  cudaError_t poll_health() noexcept
  {
    // Serializes CUDA API access with detach's return.
    std::unique_lock lock(submit_m_, std::try_to_lock);
    if (!lock.owns_lock()) { return cudaSuccess; }  // a submitter must not stall all lanes
    if (detached_) { return cudaErrorInvalidResourceHandle; }
    const auto result = api_.query(stream_);
    if (result == cudaSuccess || result == cudaErrorNotReady) {
      return host_error_.load(std::memory_order_acquire);
    }
    remember_host_error(result);
    return result;
  }

  // Fences in-flight CUDA API calls and permanently rejects submissions. Call
  // before external stream destruction. Does not stop callbacks or prove device
  // completion: cudaStreamDestroy is asynchronous, so it is not that proof either.
  void detach() noexcept
  {
    std::lock_guard lock(submit_m_);
    detached_ = true;
  }

  // Snapshot under the submission mutex; can block behind an active submission.
  [[nodiscard]] bool detached() const noexcept
  {
    std::lock_guard lock(submit_m_);
    return detached_;
  }

  // Caller explicitly asserts ALL device work AND CUDA callback delivery have
  // stopped. A query/sync error or stream destruction alone is insufficient.
  // Terminal and permanent: closes/detaches, never reuses or resets a frontier.
  // Waits for any active drainer, then invokes remaining functions inline and
  // returns their count. cudaSuccess is normalized to cudaErrorUnknown.
  std::size_t fail_all(cudaError_t error) noexcept
  {
    {
      std::lock_guard lock(submit_m_);
      detached_ = true;
      closed_   = true;
    }
    claim_drainer();
    const auto count =
      drain_owned(no_pending_v, true, error == cudaSuccess ? cudaErrorUnknown : error);
    release_drainer();
    return count;
  }

  // Caller stopped producers. Synchronizes the borrowed stream unless detached,
  // waits for an active drainer, then invokes confirmed functions inline. Returns
  // the sync error, otherwise cudaErrorNotReady if unconfirmed captures remain.
  // Errors retain captures for external recovery; success means host retirement
  // is complete, not that every function received cudaSuccess.
  [[nodiscard]] cudaError_t quiesce() noexcept
  {
    cudaError_t result = cudaSuccess;
    {
      std::lock_guard lock(submit_m_);
      if (!detached_) {
        result = api_.synchronize(stream_);
        if (result != cudaSuccess) { remember_host_error(result); }
        if (result == cudaSuccess && enqueue_error_.load() != cudaSuccess) {
          recovered_.store(submitted_, std::memory_order_release);
        }
      }
    }
    claim_drainer();
    drain_owned(completed(), false, cudaSuccess);
    release_drainer();
    if (result != cudaSuccess) { return result; }
    return idle() ? cudaSuccess : cudaErrorNotReady;
  }

 private:
  friend class submission;
  std::uint64_t completed() const noexcept
  {
    return std::max(slot_.completed.load(std::memory_order_acquire),
                    recovered_.load(std::memory_order_acquire));
  }
  void remember_host_error(cudaError_t error) noexcept
  {
    auto expected = cudaSuccess;
    host_error_.compare_exchange_strong(expected, error, std::memory_order_release);
  }
  void claim_drainer() noexcept
  {
    while (draining_.test_and_set(std::memory_order_acquire)) {
      draining_.wait(true, std::memory_order_relaxed);
    }
  }
  void release_drainer() noexcept
  {
    draining_.clear(std::memory_order_release);
    draining_.notify_all();
  }
  std::size_t drain_owned(std::uint64_t done, bool force, cudaError_t error) noexcept
  {
    std::list<detail::pending_entry> ready;
    {
      std::lock_guard lock(pending_m_);
      auto end = pending_.begin();
      while (end != pending_.end() && end->ticket <= done) {
        ++end;
      }
      // Splice existing nodes: no drain-time allocation.
      ready.splice(ready.end(), pending_, pending_.begin(), end);
      active_ticket_ = ready.empty() ? no_pending_v : ready.front().ticket;
      oldest_hint_.store(pending_.empty() ? no_pending_v : pending_.front().ticket,
                         std::memory_order_relaxed);
    }
    const auto bad           = slot_.first_failed.load(std::memory_order_acquire);
    const auto device_error  = slot_.first_error.load(std::memory_order_relaxed);
    const auto install_error = enqueue_error_.load(std::memory_order_acquire);
    for (auto& entry : ready) {
      auto status = entry.ticket >= bad ? device_error : cudaSuccess;
      if (entry.ticket == failed_enqueue_ticket_.load(std::memory_order_relaxed)) {
        status = install_error;
      }
      entry.fn(force ? error : status);
    }
    const auto count = ready.size();
    ready.clear();  // release captures before publishing host retirement
    if (registry_retired_ && count) {
      registry_retired_->fetch_add(count, std::memory_order_release);
    }
    {
      std::lock_guard lock(pending_m_);
      active_ticket_ = no_pending_v;
    }
    return count;
  }
  cudaError_t publish(std::list<detail::pending_entry>& staged) noexcept
  {
    if (staged.empty()) { return cudaSuccess; }
    if (submitted_ == no_pending_v - 1) { std::terminate(); }  // never wrap
    const auto ticket = ++submitted_;
    for (auto& entry : staged) {
      entry.ticket = ticket;
    }
    {
      std::lock_guard lock(pending_m_);
      pending_.splice(pending_.end(), staged);
      oldest_hint_.store(pending_.front().ticket, std::memory_order_relaxed);
    }
    // CUDA plans to deprecate/remove cudaStreamAddCallback, and stream capture
    // is unsupported. Unlike cudaLaunchHostFunc, it delivers device errors;
    // retirement needs that error boundary. The callback only publishes atomics:
    // it never calls CUDA, waits, or invokes user code.
    const auto error = api_.add_callback(stream_, &detail::bump_ticket, &slot_, 0);
    if (error == cudaSuccess) { return error; }
    // A missing callback permanently closes the lane. Only successful sync
    // can make that ticket safe to drain. Preserve queue order and ownership.
    closed_ = true;
    enqueue_error_.store(error, std::memory_order_release);
    failed_enqueue_ticket_.store(ticket, std::memory_order_relaxed);
    const auto sync_error = api_.synchronize(stream_);
    remember_host_error(sync_error == cudaSuccess ? error : sync_error);
    if (sync_error == cudaSuccess) { recovered_.store(ticket, std::memory_order_release); }
    return sync_error == cudaSuccess ? error : sync_error;
  }

  completion_slot slot_;
  cudaStream_t stream_;
  detail::completion_cuda_api api_;
  std::atomic<std::uint64_t>* registry_retired_;
  alignas(cacheline_v) mutable std::mutex submit_m_;
  std::uint64_t submitted_{0};
  bool detached_{false};
  bool closed_{false};
  std::atomic<cudaError_t> host_error_{cudaSuccess};
  std::atomic<cudaError_t> enqueue_error_{cudaSuccess};
  std::atomic<std::uint64_t> failed_enqueue_ticket_{no_pending_v};
  std::atomic<std::uint64_t> recovered_{0};
  alignas(cacheline_v) std::mutex pending_m_;
  std::list<detail::pending_entry> pending_;
  std::uint64_t active_ticket_{no_pending_v};
  std::atomic<std::uint64_t> oldest_hint_{no_pending_v};
  std::atomic_flag draining_ = ATOMIC_FLAG_INIT;
};

inline cudaStream_t submission::stream() const noexcept { return lane_->stream_; }
inline void submission::on_retire(retire_fn fn)
{
  if (committed_) { throw std::logic_error("on_retire after commit"); }
  if (!fn) { throw std::invalid_argument("empty completion retire function"); }
  staged_.push_back({0, std::move(fn)});
}
inline cudaError_t submission::commit() noexcept
{
  if (!committed_) {
    committed_ = true;
    result_    = lane_->publish(staged_);
  }
  return result_;
}
inline submission::~submission()
{
  if (!committed_) { (void)commit(); }
}

/**
 * Owns stable retirement lanes for explicit streams on one CUDA device; streams
 * remain caller-owned. No worker is created. Multiple completion waiters and
 * opportunistic request/eviction drainers may retire concurrently; user functions
 * run inline on those callers, with FIFO ordering only within each lane.
 * Stop producers/registration and serialize lifecycle operations externally.
 * Draining may overlap quiesce(), but all public calls must end before destruction.
 * Detach before destroying streams and keep this registry alive until callbacks
 * finish, or use fail_all() after externally proving device work AND callback
 * delivery have stopped. Destruction quiesces and terminates on unresolved work.
 */
class cuda_event_completion_poll {
 public:
  static constexpr std::size_t max_lanes_v = 256;
  explicit cuda_event_completion_poll(detail::completion_cuda_api api = {}) noexcept : api_(api) {}
  ~cuda_event_completion_poll()
  {
    (void)quiesce();
    const auto count = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
      if (!lanes_[i]->idle()) { std::terminate(); }
    }
  }
  cuda_event_completion_poll(cuda_event_completion_poll const&)            = delete;
  cuda_event_completion_poll& operator=(cuda_event_completion_poll const&) = delete;
  // Registration is cold; rmm::cuda_stream_view converts to cudaStream_t.
  // Use explicit streams, one registry per device. A per-thread default
  // stream has no stable identity across submitting threads.
  // Thread-safe with registration/draining. The returned reference remains valid
  // until registry destruction. Throws logic_error after detach, or bad_alloc on
  // allocation failure/when max_lanes_v distinct streams are already registered.
  retire_lane& lane_for(cudaStream_t stream)
  {
    std::lock_guard lock(reg_m_);
    if (detached_) { throw std::logic_error("registration after completion registry detach"); }
    const auto count = count_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < count; ++i) {
      if (lanes_[i]->stream() == stream) { return *lanes_[i]; }
    }
    if (count == max_lanes_v) { throw std::bad_alloc{}; }
    lanes_[count] = std::make_unique<retire_lane>(stream, api_, &retired_);
    count_.store(count + 1, std::memory_order_release);
    return *lanes_[count];
  }
  // Closes registration, then fences each lane's CUDA API calls/submissions.
  // May block behind submitters. Does not establish completion (see retire_lane).
  void detach() noexcept
  {
    std::size_t count;
    {
      std::lock_guard lock(reg_m_);
      detached_ = true;  // closes registration before snapshotting the lane set
      count     = count_.load(std::memory_order_acquire);
    }
    for (std::size_t i = 0; i < count; ++i) {
      lanes_[i]->detach();
    }
  }
  // Invokes ready functions inline across all lanes and returns their count.
  // Never waits for device progress or another drainer; busy lanes are skipped.
  // Safe with concurrent producers, registration, waiters, and drainers.
  std::size_t drain_all() noexcept
  {
    std::size_t retired = 0;
    const auto count    = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
      retired += lanes_[i]->drain();
    }
    return retired;
  }
  // made: this or another drainer retired functions; none: no committed work;
  // undelivered: a health error prevents confirming pending completion.
  enum class progress { made, none, undelivered };
  // Blocks with backoff while scanning/draining every lane. Multiple waiters are
  // supported. none is only a snapshot; undelivered is NOT permission to release
  // captures. Neither value precludes another caller's subsequent progress.
  progress wait_for_progress() noexcept
  {
    const auto before = retired_.load(std::memory_order_acquire);
    auto delay        = std::chrono::microseconds{4};
    for (;;) {
      if (drain_all() || retired_.load(std::memory_order_acquire) != before) {
        return progress::made;
      }
      bool pending     = false;
      bool unhealthy   = false;
      const auto count = count_.load(std::memory_order_acquire);
      for (std::size_t i = 0; i < count; ++i) {
        if (!lanes_[i]->idle()) {
          pending = true;
          unhealthy |= lanes_[i]->poll_health() != cudaSuccess;
        }
      }
      // Callbacks could have arrived during health queries.
      if (drain_all() || retired_.load(std::memory_order_acquire) != before) {
        return progress::made;
      }
      if (!pending) { return progress::none; }
      if (unhealthy) { return progress::undelivered; }
      detail::completion_backoff(delay);
    }
  }
  // Terminal cleanup under retire_lane::fail_all's external stop precondition.
  // Detaches, waits for active drainers, and invokes all remaining functions
  // inline with error (cudaSuccess becomes cudaErrorUnknown); returns their count.
  std::size_t fail_all(cudaError_t error) noexcept
  {
    detach();
    std::size_t retired = 0;
    const auto count    = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
      retired += lanes_[i]->fail_all(error);
    }
    return retired;
  }
  // Retries a caller-owned resource acquisition while draining/waiting for host
  // retirement. Returns the first truthy value, or a final attempt when no
  // progress can be confirmed; never force-retires on error. try_fn must return
  // a non-reference value contextually convertible to bool (e.g. unique_ptr).
  // May block, invokes retire functions inline, and propagates try_fn exceptions.
  template <class TryFn>
    requires(!std::is_reference_v<std::invoke_result_t<TryFn&>>) &&
            requires(TryFn& fn, std::remove_cv_t<std::invoke_result_t<TryFn&>>& result) {
              fn();
              result ? true : false;
            }
  auto acquire(TryFn&& try_fn) -> std::invoke_result_t<TryFn&>
  {
    if (auto result = try_fn()) { return result; }
    drain_all();
    if (auto result = try_fn()) { return result; }
    while (wait_for_progress() == progress::made) {
      if (auto result = try_fn()) { return result; }
    }
    return try_fn();  // never force-retire after a query error
  }
  // Stop registration/producers and externally serialize lifecycle calls. Blocks
  // through every lane's quiesce(), including active user functions; returns the
  // first non-success result. Failed lanes retain unconfirmed captures. Concurrent
  // drains are allowed. Success means all committed captures were released.
  [[nodiscard]] cudaError_t quiesce() noexcept
  {
    cudaError_t first = cudaSuccess;
    const auto count  = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
      const auto error = lanes_[i]->quiesce();
      if (first == cudaSuccess) { first = error; }
    }
    return first;
  }

 private:
  detail::completion_cuda_api api_;
  // Must outlive lane destructor-time retirement.
  std::atomic<std::uint64_t> retired_{0};
  std::array<std::unique_ptr<retire_lane>, max_lanes_v> lanes_{};
  std::atomic<std::size_t> count_{0};
  std::mutex reg_m_;
  bool detached_{false};
};

}  // namespace sirius::exec
