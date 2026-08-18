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
//
// stream_ordered_retirer.hpp
//
// Defer host-side work (buffer recycling, chunk state transitions, waiter
// wakeups) until a stream's completion frontier passes a given point --
// without a completion thread, without one CUDA event per submission, and
// without a shared task queue that one busy stream can flood.
//
// MODEL
//   Each stream gets a lane holding:
//     - a monotonic ticket counter, incremented once per logical submission
//     - a `completed` word advanced by the stream itself, one per ticket
//     - a FIFO of (ticket, retire_fn)
//
//   Work on a stream completes in submission order, so `completed >= t`
//   implies every ticket <= t is done. Retirement is a prefix pop, and one
//   64-bit load tells you everything about the whole lane.
//
//   Nothing polls in steady state. `drain()` runs on whoever is about to need
//   a resource:
//
//       pinned_buffer* acquire() {
//         if (auto* b = free.try_pop()) return b;
//         retirer.drain_all();                 // one relaxed load per lane
//         if (auto* b = free.try_pop()) return b;
//         return retirer.acquire([&]{ return free.try_pop(); });  // backpressure
//       }
//
// WHY cudaStreamAddCallback AND NOT cudaLaunchHostFunc
//   Two reasons, and the second is the important one:
//
//   1. A host function enqueued behind a faulting operation never executes.
//      Its ticket strands, the frontier freezes, and every retire_fn behind it
//      is lost -- including the ones that would move chunks out of `loading`
//      and unpark readers. A device fault becomes a silent hang. Callbacks
//      always fire.
//
//   2. Callbacks receive the stream's `cudaError_t`. Without it, retirement
//      can only act on the *host-side* result of the read, so a faulted H2D
//      copy would be marked cached with garbage in the device buffer. There
//      is no way to detect that with a host function. retire_fn therefore
//      takes the completion status, and the cached/failed decision is made
//      with it.
//
//   Cost: cudaStreamAddCallback is deprecated, and -- verify against your
//   toolkit -- is not permitted during stream capture. If you later want to
//   capture the copy batch into a CUDA graph, this is the thing that blocks
//   it, not the deprecation.
//
// INVARIANTS (violating any of these is a use-after-free, not a slowdown)
//   1. Ticket assignment order must equal stream-enqueue order. `submission`
//      holds the lane's submit lock across your launches to enforce this.
//      Prefer one lane per submitting thread; then it is uncontended.
//   2. Every allocated ticket lands exactly one callback. `completed` counts
//      callbacks, so a ticket without one makes the frontier lag forever.
//   3. The pending entry is published before the callback is enqueued, so a
//      drainer can never see `completed >= t` with no entry.
//   4. retire_fn runs outside all lane locks. It may take your freelist lock,
//      CAS chunk control words, and unpark ParkingLot waiters.
//
// USAGE
//     auto& lane = retirer.lane_for(stream);       // once, at stream setup
//     ...
//     {
//       auto sub = lane.begin();                   // takes submit lock
//       for (auto& r : ranges) cudaMemcpyAsync(..., sub.stream());
//
//       sub.on_retire([pinned  = std::move(pinned),
//                      loading = std::move(loading),
//                      host_ok = ok](cudaError_t status) noexcept {
//         const bool ok = host_ok && status == cudaSuccess;
//         ...                                      // ONE fn for the whole batch
//       });
//
//       if (auto e = sub.commit(); e != cudaSuccess) { /* already recovered */ }
//     }
//
#include "exec/invocable.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::exec {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Cache-line size, used to pad and align away false sharing.
//
// Pinned rather than taken from std::hardware_destructive_interference_size on
// purpose.  That constant varies with -mtune, and this one appears in `alignas`
// on a type (completion_slot) and on members of a class defined in this header,
// so two translation units compiled with different tuning would disagree about
// layout and sizeof -- an ODR violation, not merely a missed optimisation.
// This is precisely what GCC's -Winterference-size warns about, and its own
// advice is to use a constant you define.
//
// 64 is the line size on both host architectures Sirius builds for: x86-64 and
// aarch64 (including Grace).
inline constexpr std::size_t cacheline_v = 64;

// Receives the stream's completion status for the batch it was staged with.
// Must be noexcept: these run on the allocation path.
//
// @c exec::invocable (absl::AnyInvocable) rather than std::move_only_function:
// the project targets C++20, and this is the move-only callable the rest of
// the pipeline and future primitives already use.
using retire_fn = invocable<void(cudaError_t) noexcept>;

inline constexpr std::uint64_t no_pending_v = ~std::uint64_t{0};

// ---------------------------------------------------------------------------
// completion_slot -- the only state the stream touches
// ---------------------------------------------------------------------------

struct alignas(cacheline_v) completion_slot {
  // Counts callbacks, one per ticket. Monotonic, so
  // "completed >= t  =>  ticket t is done".
  std::atomic<std::uint64_t> completed{0};

  // Lowest ticket that completed with an error, or no_pending_v. Device errors
  // are sticky and terminal, so every ticket from here on is failed -- one
  // number describes the whole tail.
  std::atomic<std::uint64_t> first_failed{no_pending_v};
  std::atomic<cudaError_t> first_error{cudaSuccess};
};

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "completion_slot must be lock-free; it is written from a CUDA callback");

namespace detail {

// Runs on a driver callback thread, in stream order. Must not: call any CUDA
// API, block, or allocate. Three relaxed stores at worst, one release RMW.
inline void CUDART_CB bump_ticket(cudaStream_t, cudaError_t status, void* user_data) noexcept
{
  auto* slot = static_cast<completion_slot*>(user_data);

  if (status != cudaSuccess) {
    // Record the failure BEFORE advancing the frontier. A drainer that sees
    // this ticket completed must also see that it failed -- publishing in the
    // other order lets it retire a faulted batch as a success.
    //
    // Callbacks are FIFO on this stream and are the only writer of completed,
    // so `completed + 1` is this callback's ticket and the first failing
    // callback holds the lowest failing ticket. CAS-once is therefore correct.
    slot->first_error.store(status, std::memory_order_relaxed);
    std::uint64_t none = no_pending_v;
    slot->first_failed.compare_exchange_strong(none,
                                               slot->completed.load(std::memory_order_relaxed) + 1,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed);
  }

  slot->completed.fetch_add(1, std::memory_order_release);  // publishes the above
}

struct pending_entry {
  std::uint64_t ticket;
  retire_fn fn;
};

}  // namespace detail

class retire_lane;

// ---------------------------------------------------------------------------
// submission -- RAII scope that pins ticket order to stream-enqueue order
// ---------------------------------------------------------------------------

class [[nodiscard]] submission {
 public:
  // Non-movable on purpose. A moved-from submission keeps lane_ non-null and
  // committed_ false while its unique_lock has released submit_m_, so its
  // destructor would commit *unlocked*: a racing ++submitted_ plus a callback
  // with no pending entries, driving `completed` ahead of the tickets and
  // retiring buffers that are still in flight.
  //
  // begin() still works: `return submission{...}` is a prvalue, so C++17
  // guaranteed elision constructs it in place with no move.
  submission(submission&&)                 = delete;
  submission& operator=(submission&&)      = delete;
  submission(const submission&)            = delete;
  submission& operator=(const submission&) = delete;

  ~submission();

  // Launch all of this batch's work on this stream, inside this scope.
  [[nodiscard]] cudaStream_t stream() const noexcept;

  // Stage work to run once everything launched in this scope has completed.
  // The callable receives the stream's completion status for the batch.
  //
  // May be called more than once; all staged fns share one ticket and one
  // callback. Prefer one fn covering the whole batch -- that is the difference
  // between 100 wakeups and 1.
  void on_retire(retire_fn f);

  // Publishes the ticket and enqueues the callback. After this returns, do not
  // enqueue further work on this stream for this batch.
  //
  // On failure the lane has already synchronized the stream and run the staged
  // fns inline with the error, so the caller's buffers and chunk states are
  // resolved either way; the error is returned for logging and propagation.
  cudaError_t commit() noexcept;

 private:
  friend class retire_lane;
  submission(retire_lane& lane, std::unique_lock<std::mutex> lk) noexcept
    : lane_(&lane), lk_(std::move(lk))
  {
  }

  retire_lane* lane_;
  std::unique_lock<std::mutex> lk_;
  std::vector<retire_fn> staged_;
  bool committed_{false};
};

// ---------------------------------------------------------------------------
// retire_lane -- one per stream
// ---------------------------------------------------------------------------

class retire_lane {
 public:
  explicit retire_lane(cudaStream_t s) noexcept : stream_(s) {}

  ~retire_lane() { quiesce(); }

  retire_lane(const retire_lane&)            = delete;
  retire_lane& operator=(const retire_lane&) = delete;

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

  // Opens a submission scope. Blocks only if another thread is mid-submission
  // on this same stream -- which is why one lane per submitting thread is the
  // recommended topology.
  [[nodiscard]] submission begin() { return submission{*this, std::unique_lock{submit_m_}}; }

  // Retires every entry whose ticket has completed, handing each its status.
  // Returns how many ran. Safe to call from any thread, including concurrently
  // with submitters.
  std::size_t drain() noexcept
  {
    const std::uint64_t done = slot_.completed.load(std::memory_order_acquire);

    // Advisory fast path: one relaxed load, no lock, no cache-line handoff.
    // A stale-high hint only postpones retirement to the next drain; the
    // blocking path uses oldest_pending_locked() instead.
    if (done < oldest_hint_.load(std::memory_order_relaxed)) { return 0; }

    // Ordered after the acquire above, so a failure published before the
    // frontier moved is visible here.
    const std::uint64_t bad = slot_.first_failed.load(std::memory_order_relaxed);
    const cudaError_t err   = slot_.first_error.load(std::memory_order_relaxed);

    std::vector<detail::pending_entry> ready;
    {
      std::lock_guard g(pending_m_);
      while (!pending_.empty() && pending_.front().ticket <= done) {
        ready.push_back(std::move(pending_.front()));
        pending_.pop_front();
      }
      refresh_hint_locked();
    }

    // Outside every lane lock: these take freelist locks, CAS chunk control
    // words, and unpark waiters.
    for (auto& e : ready) {
      e.fn(e.ticket >= bad ? err : cudaSuccess);
    }
    return ready.size();
  }

  // Advisory. May read stale, which is fine for *skipping* a drain but not for
  // concluding there is nothing left to wait on.
  [[nodiscard]] std::uint64_t oldest_pending_hint() const noexcept
  {
    return oldest_hint_.load(std::memory_order_relaxed);
  }

  // Authoritative: oldest ticket not yet retired, or no_pending_v if idle.
  [[nodiscard]] std::uint64_t oldest_pending_locked() noexcept
  {
    std::lock_guard g(pending_m_);
    return pending_.empty() ? no_pending_v : pending_.front().ticket;
  }

  [[nodiscard]] bool idle() noexcept { return oldest_pending_locked() == no_pending_v; }

  [[nodiscard]] bool faulted() const noexcept
  {
    return slot_.first_failed.load(std::memory_order_acquire) != no_pending_v;
  }

  [[nodiscard]] cudaError_t fault_error() const noexcept
  {
    return slot_.first_error.load(std::memory_order_acquire);
  }

  // Blocks until `completed >= target`. Backpressure only -- never the steady
  // state. Does not drain; call drain() after.
  //
  // A device fault does not strand us: the callback fires with the error, the
  // frontier advances, and drain() retires the batch as failed. poll_health()
  // is only an escape hatch for the case where the context is torn down from
  // under the lane and no callback can be delivered at all.
  cudaError_t wait_for(std::uint64_t target) noexcept
  {
    for (int i = 0; i < 64; ++i) {  // brief spin: the frontier is usually close
      if (slot_.completed.load(std::memory_order_acquire) >= target) { return cudaSuccess; }
      std::this_thread::yield();
    }

    // Sleep granularity here is irrelevant next to the H2D copy we are waiting
    // on, and polling is what keeps the escape hatch reachable.
    auto delay               = std::chrono::microseconds{4};
    constexpr auto max_delay = std::chrono::microseconds{256};
    for (;;) {
      if (slot_.completed.load(std::memory_order_acquire) >= target) { return cudaSuccess; }
      if (const cudaError_t e = poll_health(); e != cudaSuccess) { return e; }
      std::this_thread::sleep_for(delay);
      delay = std::min(max_delay, delay * 2);
    }
  }

  // cudaSuccess if the stream is alive (idle or busy); the sticky error
  // otherwise. Sticky errors are context-wide, so any lane detects the fault --
  // including lanes that never submitted the bad work.
  cudaError_t poll_health() noexcept
  {
    if (detached_.load(std::memory_order_acquire)) { return cudaErrorInvalidResourceHandle; }
    const cudaError_t q = cudaStreamQuery(stream_);
    if (q == cudaSuccess || q == cudaErrorNotReady) { return cudaSuccess; }
    mark_faulted(q);
    return q;
  }

  // Give up the stream handle: after this the lane makes no CUDA calls at all.
  //
  // For owners that do not control stream lifetime. A lane holds a raw
  // cudaStream_t, and nothing stops the stream's owner from destroying it
  // first -- at which point quiesce()'s cudaStreamSynchronize and
  // poll_health()'s cudaStreamQuery are calls on a dangling handle, which
  // segfaults inside the driver rather than returning an error.
  //
  // Retirement does not need the stream: drain() reads the frontier the
  // callbacks already published, and anything still outstanding is reported as
  // unconfirmed instead of waited on. Detaching is therefore safe precisely
  // when the stream's owner has already ensured the work finished -- which it
  // had to, in order to destroy the stream.
  void detach() noexcept { detached_.store(true, std::memory_order_release); }

  [[nodiscard]] bool detached() const noexcept { return detached_.load(std::memory_order_acquire); }

  // Retires every pending entry unconditionally with `e`. Terminal recovery
  // for the case where callbacks cannot be delivered; the normal fault path
  // does not need it, because callbacks still fire.
  //
  // ORDERING: these fns hand pinned buffers back to the pool. Call this only
  // once the device is stopped -- after the context is destroyed / reset -- so
  // nothing can still be DMAing into them.
  std::size_t fail_all(cudaError_t e) noexcept
  {
    std::vector<detail::pending_entry> ready;
    {
      std::lock_guard g(pending_m_);
      while (!pending_.empty()) {
        ready.push_back(std::move(pending_.front()));
        pending_.pop_front();
      }
      refresh_hint_locked();
    }
    for (auto& p : ready) {
      p.fn(e);
    }
    return ready.size();
  }

  // Synchronize the stream and retire everything. Use on shutdown and after a
  // fault. Must not be called from inside a submission scope on this lane: it
  // takes submit_m_ so no ticket can be allocated while the counter is resynced.
  cudaError_t quiesce() noexcept
  {
    std::lock_guard submit_g(submit_m_);

    if (detached_.load(std::memory_order_acquire)) {
      // No CUDA calls: the handle may already be dead. Entries whose callbacks
      // did fire still retire with their true status; the rest are reported
      // unconfirmed, because without the stream there is no way to learn it.
      drain();
      const std::uint64_t bad = slot_.first_failed.load(std::memory_order_acquire);
      fail_all(bad == no_pending_v ? cudaErrorInvalidResourceHandle
                                   : slot_.first_error.load(std::memory_order_relaxed));
      slot_.completed.store(submitted_, std::memory_order_release);
      return cudaSuccess;
    }

    const cudaError_t sync_err = cudaStreamSynchronize(stream_);
    if (sync_err != cudaSuccess) { mark_faulted(sync_err); }

    // Callbacks fire even on fault, so drain() normally has already emptied
    // this and each entry got its true status. Anything left could not be
    // delivered; give it the lane's error.
    drain();

    const std::uint64_t bad = slot_.first_failed.load(std::memory_order_acquire);
    fail_all(bad == no_pending_v ? sync_err : slot_.first_error.load(std::memory_order_relaxed));

    // Republish the frontier so the lane is reusable if the caller rebuilds
    // the stream. Safe: everything is retired and no ticket can be in flight.
    slot_.completed.store(submitted_, std::memory_order_release);
    return sync_err;
  }

 private:
  friend class submission;

  void mark_faulted(cudaError_t e) noexcept
  {
    slot_.first_error.store(e, std::memory_order_relaxed);
    std::uint64_t none = no_pending_v;
    slot_.first_failed.compare_exchange_strong(none,
                                               slot_.completed.load(std::memory_order_relaxed) + 1,
                                               std::memory_order_release,
                                               std::memory_order_relaxed);
  }

  void refresh_hint_locked() noexcept
  {
    oldest_hint_.store(pending_.empty() ? no_pending_v : pending_.front().ticket,
                       std::memory_order_relaxed);
  }

  // Called by submission::commit() with submit_m_ held.
  //
  // INVARIANT: every allocated ticket lands exactly one callback. `completed`
  // counts callbacks, so a ticket without one makes it lag `submitted_`
  // forever -- every later batch then retires only when the *next* batch
  // completes, and the backpressure path (which by definition stops
  // submitting) waits on a frontier that can never advance. Every exit from
  // this function must leave the invariant intact.
  cudaError_t publish(std::vector<retire_fn>& staged) noexcept
  {
    // Nothing staged: no ticket, no callback, no stream stall. Nobody can be
    // waiting on a frontier that carries no retirement work.
    if (staged.empty()) { return cudaSuccess; }

    const std::uint64_t t = ++submitted_;  // guarded by submit_m_

    // Publish before enqueueing: a drainer must never see completed >= t with
    // no entry to retire.
    {
      std::lock_guard g(pending_m_);
      for (auto& f : staged) {
        pending_.push_back({t, std::move(f)});
      }
      refresh_hint_locked();
    }

    // flags must be 0.
    const cudaError_t e = cudaStreamAddCallback(stream_, &detail::bump_ticket, &slot_, 0);
    if (e != cudaSuccess) { unwind_failed_enqueue(t, e); }
    return e;
  }

  // Ticket `t` will never complete. Called with submit_m_ held, so nobody
  // appended after us and no drainer can have taken these (completed < t).
  // Give the ticket back first, then sync and retire inline with the error.
  void unwind_failed_enqueue(std::uint64_t t, cudaError_t e) noexcept
  {
    std::vector<detail::pending_entry> orphans;
    {
      std::lock_guard g(pending_m_);
      while (!pending_.empty() && pending_.back().ticket == t) {
        orphans.push_back(std::move(pending_.back()));
        pending_.pop_back();
      }
      refresh_hint_locked();
    }
    --submitted_;  // restores the ticket/callback invariant

    cudaStreamSynchronize(stream_);
    for (auto it = orphans.rbegin(); it != orphans.rend(); ++it) {
      it->fn(e);
    }  // original order
  }

  // --- hot: written by the callback thread, read by every drainer ----------
  completion_slot slot_{};

  cudaStream_t stream_;

  // --- submitter side ------------------------------------------------------
  alignas(cacheline_v) std::mutex submit_m_;
  std::uint64_t submitted_{0};  // guarded by submit_m_

  // --- queue side ----------------------------------------------------------
  alignas(cacheline_v) std::mutex pending_m_;
  std::deque<detail::pending_entry> pending_;             // guarded by pending_m_
  std::atomic<std::uint64_t> oldest_hint_{no_pending_v};  // advisory

  // Set once, never cleared: the stream handle is no longer safe to touch.
  std::atomic<bool> detached_{false};
};

// ---------------------------------------------------------------------------
// submission out-of-line definitions
// ---------------------------------------------------------------------------

inline cudaStream_t submission::stream() const noexcept { return lane_->stream_; }

inline void submission::on_retire(retire_fn f) { staged_.push_back(std::move(f)); }

inline cudaError_t submission::commit() noexcept
{
  if (committed_) { return cudaSuccess; }
  committed_ = true;
  return lane_->publish(staged_);
}

inline submission::~submission()
{
  // Abandoning a scope that already launched work would recycle buffers the
  // copy engines are still reading. Commit anyway; publish() recovers.
  if (!committed_ && lane_ != nullptr) { commit(); }
}

// ---------------------------------------------------------------------------
// stream_ordered_retirer -- append-only registry over lanes
// ---------------------------------------------------------------------------

class stream_ordered_retirer {
 public:
  static constexpr std::size_t max_lanes_v = 256;

  stream_ordered_retirer() noexcept = default;
  ~stream_ordered_retirer() { quiesce(); }

  stream_ordered_retirer(const stream_ordered_retirer&)            = delete;
  stream_ordered_retirer& operator=(const stream_ordered_retirer&) = delete;

  // Registers a stream. Call once at stream setup and keep the reference --
  // this takes a lock and does a linear scan; it is not a hot-path lookup.
  retire_lane& lane_for(cudaStream_t s)
  {
    {
      const std::size_t n = count_.load(std::memory_order_acquire);
      for (std::size_t i = 0; i < n; ++i) {
        if (lanes_[i]->stream() == s) { return *lanes_[i]; }
      }
    }
    std::lock_guard g(reg_m_);
    const std::size_t n = count_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < n; ++i) {
      if (lanes_[i]->stream() == s) { return *lanes_[i]; }
    }
    if (n == max_lanes_v) { throw std::bad_alloc{}; }
    owned_.push_back(std::make_unique<retire_lane>(s));
    lanes_[n] = owned_.back().get();
    // Registering after detach should not happen, but a lane that could still
    // reach for a stream handle would undo the whole point of detaching.
    if (detached_.load(std::memory_order_acquire)) { lanes_[n]->detach(); }
    count_.store(n + 1, std::memory_order_release);  // publish last
    return *lanes_[n];
  }

  // Give up every stream handle; see retire_lane::detach(). Call this before
  // teardown when the streams belong to somebody else -- retirement continues
  // to work, it just stops waiting on streams it can no longer trust.
  void detach() noexcept
  {
    detached_.store(true, std::memory_order_release);
    const std::size_t n = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
      lanes_[i]->detach();
    }
  }

  // One relaxed load per lane in the common "nothing ready" case.
  std::size_t drain_all() noexcept
  {
    std::size_t retired = 0;
    const std::size_t n = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
      retired += lanes_[i]->drain();
    }
    return retired;
  }

  enum class progress {
    made,        // the frontier advanced; retry
    none,        // nothing outstanding anywhere; waiting would deadlock
    undelivered  // callbacks cannot be delivered; nothing will ever complete
  };

  // Blocks on the globally oldest outstanding ticket.
  progress wait_for_progress() noexcept
  {
    retire_lane* target  = nullptr;
    std::uint64_t oldest = no_pending_v;

    // Authoritative read: a stale hint here would report "nothing outstanding"
    // and turn a wait into a spurious allocation failure. We are about to
    // block anyway, so N uncontended lane locks cost nothing.
    const std::size_t n = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t t = lanes_[i]->oldest_pending_locked();
      if (t != no_pending_v && (target == nullptr || t < oldest)) {
        oldest = t;
        target = lanes_[i];
      }
    }
    if (target == nullptr) { return progress::none; }

    // A device fault is NOT this branch: the callback still fires, the
    // frontier advances, and drain() retires the batch as failed. This only
    // triggers if the context went away entirely.
    if (target->wait_for(oldest) != cudaSuccess) { return progress::undelivered; }

    target->drain();
    return progress::made;
  }

  // Terminal recovery across every lane. See retire_lane::fail_all() for the
  // ordering constraint: stop the device first.
  std::size_t fail_all(cudaError_t e) noexcept
  {
    std::size_t n_retired = 0;
    const std::size_t n   = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
      n_retired += lanes_[i]->fail_all(e);
    }
    return n_retired;
  }

  // Retry `try_fn` -- typically a freelist pop -- draining between attempts and
  // blocking only when there is nothing left to reclaim without waiting.
  // Returns try_fn's falsy value rather than hanging when the resource can
  // never become available.
  template <class TryFn>
  auto acquire(TryFn&& try_fn) -> std::invoke_result_t<TryFn&>
  {
    if (auto r = try_fn()) { return r; }
    if (drain_all() != 0) {
      if (auto r = try_fn()) { return r; }
    }
    for (;;) {
      switch (wait_for_progress()) {
        case progress::made:
          if (auto r = try_fn()) { return r; }
          break;
        case progress::none: return try_fn();
        case progress::undelivered:
          // Unblock every parked reader with a failure status, then let the
          // caller see the allocation fail. Hanging here would be worse.
          fail_all(cudaErrorUnknown);
          return try_fn();
      }
    }
  }

  cudaError_t quiesce() noexcept
  {
    cudaError_t first   = cudaSuccess;
    const std::size_t n = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
      const cudaError_t e = lanes_[i]->quiesce();
      if (first == cudaSuccess) { first = e; }
    }
    return first;
  }

 private:
  // Append-only: lookups are lock-free, registration is not.
  std::array<retire_lane*, max_lanes_v> lanes_{};
  std::atomic<std::size_t> count_{0};
  std::mutex reg_m_;
  std::vector<std::unique_ptr<retire_lane>> owned_;
  std::atomic<bool> detached_{false};
};

}  // namespace sirius::exec
