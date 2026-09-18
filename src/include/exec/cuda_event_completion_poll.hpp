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

#include <cuda_runtime.h>

#include <absl/functional/any_invocable.h>

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

// Historical name: this uses stream callbacks, not CUDA events. One frontier
// describes each stream's completed prefix. No worker is created; retire
// functions run inline on drain/acquire/quiesce callers. Scheduling belongs to
// the consumer (normally one on-demand pump for its 8-16 streams).
//
// Contract:
// * Enqueue work while its submission holds the lane lock. Stage ownership-
//   bearing functions BEFORE launching work: on_retire and construction of
//   its argument may allocate/throw; commit and drain allocate nothing.
// * Retirement is FIFO per lane, including concurrent drainers. Functions
//   must be noexcept and short. They may call nonblocking drain, but must not
//   wait for retirement or call quiesce/fail_all on this registry.
// * Stop producers before quiesce, terminal cleanup, or destruction. Serialize
//   lifecycle operations externally. Draining may overlap quiesce, which waits
//   for active retire functions. Every public call must end before destruction.
// * Streams are borrowed. detach BEFORE external stream destruction: it fences
//   CUDA API calls and rejects submissions, but does NOT establish completion.
//   Keep this object alive until callbacks finish, or call fail_all only after
//   device work AND callback delivery have stopped. cudaStreamDestroy returns
//   asynchronously and is NOT such a proof.
// * Failed quiescence retains unconfirmed captures. Destruction with unresolved
//   work terminates rather than freeing live buffers or CUDA callback state.
//
// cudaStreamAddCallback is slated for eventual deprecation/removal and is
// unsupported in stream capture. Unlike cudaLaunchHostFunc, it delivers device
// errors. Its callback only publishes atomics: no CUDA, waits, or user code.
inline constexpr std::size_t cacheline_v    = 64;
inline constexpr std::uint64_t no_pending_v = ~std::uint64_t{0};
using retire_fn                             = absl::AnyInvocable<void(cudaError_t) noexcept>;

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
class [[nodiscard]] submission {
 public:
  submission(submission&&)                 = delete;
  submission& operator=(submission&&)      = delete;
  submission(submission const&)            = delete;
  submission& operator=(submission const&) = delete;
  ~submission();

  [[nodiscard]] cudaStream_t stream() const noexcept;

  // Stage before enqueuing device work, including construction of fn.
  void on_retire(retire_fn fn);
  // Idempotent. Installation failure closes the lane. A fallback sync error
  // takes precedence in the return value; enqueue_error() retains the cause.
  // Unconfirmed functions remain queued; successful recovery makes them
  // drainable with the installation error (never inline under the submit lock).
  cudaError_t commit() noexcept;

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
    if (!idle()) { quiesce(); }
    if (!idle()) { std::terminate(); }
  }
  retire_lane(retire_lane const&)            = delete;
  retire_lane& operator=(retire_lane const&) = delete;

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

  [[nodiscard]] submission begin()
  {
    std::unique_lock lock(submit_m_);
    if (detached_ || closed_) {
      throw std::logic_error("submission on detached or failed completion lane");
    }
    return submission{*this, std::move(lock)};
  }

  // A busy drainer must not stall a registry scan. Ownership spans invocation,
  // not merely removal from the queue, to preserve FIFO across callers.
  std::size_t drain() noexcept
  {
    const auto done = completed();
    if (done < oldest_hint_.load(std::memory_order_relaxed)) { return 0; }
    if (draining_.test_and_set(std::memory_order_acquire)) { return 0; }
    const auto count = drain_owned(done, false, cudaSuccess);
    draining_.clear(std::memory_order_release);
    return count;
  }

  [[nodiscard]] std::uint64_t oldest_pending_hint() const noexcept
  {
    return oldest_hint_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t oldest_pending_locked() noexcept
  {
    std::lock_guard lock(pending_m_);
    return std::min(active_ticket_, pending_.empty() ? no_pending_v : pending_.front().ticket);
  }

  [[nodiscard]] bool idle() noexcept { return oldest_pending_locked() == no_pending_v; }

  [[nodiscard]] bool faulted() const noexcept { return fault_error() != cudaSuccess; }

  [[nodiscard]] cudaError_t fault_error() const noexcept
  {
    if (slot_.first_failed.load(std::memory_order_acquire) != no_pending_v) {
      return slot_.first_error.load(std::memory_order_relaxed);
    }
    return host_error_.load(std::memory_order_acquire);
  }

  [[nodiscard]] cudaError_t enqueue_error() const noexcept
  {
    return enqueue_error_.load(std::memory_order_acquire);
  }

  // Lane-local utility. Registry waiting deliberately never calls this.
  cudaError_t wait_for(std::uint64_t target) noexcept
  {
    auto delay = std::chrono::microseconds{4};
    for (;;) {
      if (completed() >= target) { return cudaSuccess; }
      if (auto error = poll_health(); error != cudaSuccess) { return error; }
      detail::completion_backoff(delay);
    }
  }

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

  void detach() noexcept
  {
    std::lock_guard lock(submit_m_);
    detached_ = true;
  }

  [[nodiscard]] bool detached() const noexcept
  {
    std::lock_guard lock(submit_m_);
    return detached_;
  }

  // Caller explicitly asserts ALL device work AND CUDA callback delivery have
  // stopped. A query/sync error or stream destruction alone is insufficient.
  // Terminal and permanent: closes/detaches, never reuses or resets a frontier.
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
    draining_.clear(std::memory_order_release);
    return count;
  }

  // Caller stopped producers. An error does not prove completion; only drain
  // delivered boundaries and retain everything else for external recovery.
  cudaError_t quiesce() noexcept
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
    draining_.clear(std::memory_order_release);
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
      std::this_thread::yield();
    }
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
  if (!committed_) { commit(); }
}

class cuda_event_completion_poll {
 public:
  static constexpr std::size_t max_lanes_v = 256;
  explicit cuda_event_completion_poll(detail::completion_cuda_api api = {}) noexcept : api_(api) {}
  ~cuda_event_completion_poll()
  {
    quiesce();
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
  std::size_t drain_all() noexcept
  {
    std::size_t retired = 0;
    const auto count    = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
      retired += lanes_[i]->drain();
    }
    return retired;
  }
  enum class progress { made, none, undelivered };
  // undelivered means waiting cannot confirm completion, NOT safe to release.
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
  template <class TryFn>
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
  // Externally serialize with registration, submission, and lifecycle calls.
  cudaError_t quiesce() noexcept
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
