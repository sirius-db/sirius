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

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op::scan {

/// A small pool of non-blocking CUDA streams on one GPU, owned per scan manager.  The readahead
/// tier allocates device prefetches (and records their ready events) on streams leased from it.
/// A @ref lease shares ownership of the pool, so the streams outlive in-flight IO and untaken
/// buffers whichever side lets go first.  Destroying the pool binds @ref device, synchronizes
/// every stream without throwing, and destroys them.
class device_stream_pool : public std::enable_shared_from_this<device_stream_pool> {
 public:
  static constexpr std::size_t default_stream_count = 4;

  /// A stream borrowed from the pool, plus the share of the pool that keeps it alive.
  struct lease {
    std::shared_ptr<device_stream_pool const> owner;
    ::cuda::stream_ref view;
  };

  /// Creates @p n (at least one) non-blocking streams on @p device (bound for the call).
  [[nodiscard]] static std::shared_ptr<device_stream_pool> create(
    int device, std::size_t n = default_stream_count);

  /// Round-robin lease.  Thread-safe.
  [[nodiscard]] lease acquire() const;

  [[nodiscard]] int device() const noexcept { return _device; }
  [[nodiscard]] std::size_t size() const noexcept { return _streams.size(); }

  ~device_stream_pool();
  device_stream_pool(device_stream_pool const&)            = delete;
  device_stream_pool& operator=(device_stream_pool const&) = delete;

 private:
  device_stream_pool() = default;

  int _device{-1};
  std::vector<std::unique_ptr<rmm::cuda_stream>> _streams;  ///< destroyed with @ref _device bound
  mutable std::atomic<std::size_t> _next{0};
};

/// Device-resident copies of one split's hint ranges, read ahead of demand by the readahead's
/// device tier.  Entry k mirrors scan_info::fadvise_hints()[k]; buffer j of entry k holds range j
/// of that hint (the column-chunk order the hybrid scan reader wants).  Shared between the split's
/// slot and the IO completion that fills it.
class prefetched_ranges {
 public:
  enum class state : int { pending = 0, ready = 1, failed = 2 };

  /// Why the buffers left this object -- reported once to the release hook.
  enum class release_reason : std::uint8_t { consumed, unconsumed, failed, wrong_device };

  struct release_event {
    release_reason reason;
    std::size_t bytes;              ///< logical bytes (total_bytes())
    std::uint64_t wait_ns;          ///< time consumers spent in wait_for() on this object
    std::size_t reserved_bytes{0};  ///< bytes reserved for this prefetch (reserved_bytes())
  };

  /// Both hooks must be noexcept and touch only atomics: they may run on any thread -- an IO
  /// completion, the evictor, a pipeline thread, the readahead worker -- and must own what they
  /// touch (shared_ptr), because they can outlive whoever installed them.
  ///
  /// The claim hook gives back the prefetch's budget reservation (and nothing else: it frees no
  /// buffer).  It fires at most once, from claim().
  using claim_hook = sirius::exec::invocable<void() noexcept>;
  /// The release hook reports where the buffers ended up.  It fires exactly once.
  using release_hook = sirius::exec::invocable<void(release_event) noexcept>;

  struct device_ranges {
    std::vector<rmm::device_buffer> buffers;                   ///< one per hint range, hint order
    std::vector<cudf::device_span<std::uint8_t const>> spans;  ///< parallel to buffers
  };

  /// @p ranges were allocated on @p producer.view() (device @p device_id).  @p logical_bytes is
  /// the data size, @p reserved_bytes what was reserved for it.  @p on_claim fires once (claim(),
  /// take() or the destructor); @p on_release fires exactly once, after the claim: from take()
  /// (consumed) or from the destructor (any other reason).
  ///
  /// Creates the ready event on @p device_id.  A failure to create it is not thrown: it surfaces
  /// as the failure of set_ready().
  prefetched_ranges(int device_id,
                    device_stream_pool::lease producer,
                    std::vector<device_ranges> ranges,
                    std::size_t logical_bytes,
                    std::size_t reserved_bytes,
                    claim_hook on_claim,
                    release_hook on_release,
                    std::shared_ptr<std::mutex> allocation_mutex = {}) noexcept;

  /// Claims, fires the release hook if take() did not, then frees whatever take() did not move
  /// out (stream-ordered on the allocation stream, device bound).  Must not run while an IO may
  /// still write the buffers: the producer keeps a shared_ptr until its last read settled.
  ~prefetched_ranges();

  prefetched_ranges(prefetched_ranges const&)            = delete;
  prefetched_ranges& operator=(prefetched_ranges const&) = delete;

  // ---- producer side (readahead worker / IO completion thread; noexcept, host-only) ----
  //
  // Each transition leaves pending at most once and returns whether this call made it.  The
  // caller must hold a reference (shared_ptr) for the duration of the call: a waiter woken by the
  // transition may drop its own reference right away.  The producer and the waiter both do.

  /// pending -> ready; idempotent.  Call only once every IO future into the buffers has settled:
  /// records the ready event on the producer stream (device bound), which orders the consumer's
  /// stream after the allocation and any stream work enqueued before this call.  If the event
  /// cannot be created or recorded the transition is to failed instead (the return value is still
  /// whether this call left pending).
  bool set_ready() noexcept;
  /// pending -> failed; keeps the first error.
  bool set_failed(std::exception_ptr error) noexcept;
  /// pending -> failed with "device prefetch abandoned: consumer overtook the readahead" and the
  /// abandoned flag, so a consumer already waiting wakes at once and reads for itself.
  bool abandon() noexcept;

  // ---- consumer side (pipeline thread) ----

  /// Blocks until the state leaves pending or @p timeout passes; returns the state seen and
  /// accumulates the time spent for the release event.  The caller must hold a reference
  /// (shared_ptr) while it waits.
  [[nodiscard]] state wait_for(std::chrono::milliseconds timeout) const;
  [[nodiscard]] state current() const noexcept;
  [[nodiscard]] std::exception_ptr failure() const noexcept;
  [[nodiscard]] bool abandoned() const noexcept;

  /// The consumer decodes on another GPU than the buffers live on: makes the destructor report
  /// wrong_device.  take() refuses after this.
  void flag_wrong_device() noexcept;

  /// Release the budget reservation without freeing anything; idempotent.  take() and the
  /// destructor call it too.
  void claim() noexcept;
  [[nodiscard]] bool claimed() const noexcept { return _claimed.load(std::memory_order_acquire); }

  /// Move the buffers out for a decode on @p consumer_stream.  Only when ready, exactly once, and
  /// not flagged wrong_device (otherwise nullopt).  The consumer stream first waits on the ready
  /// event (so it orders after every producer-stream write), and only then is every buffer's
  /// deallocation stream rebound to it.  Claims, then fires the release hook with reason
  /// consumed.  The returned spans stay valid while the returned buffers live.
  ///
  /// Device contract: the calling thread's current device must be device_id() both here and when
  /// the returned buffers are destroyed -- rmm::device_buffer's destructor frees on the current
  /// device without binding its own.  Check device_id() first and flag_wrong_device() on a
  /// mismatch; as a guard, a call made on another current device flags wrong_device itself and
  /// returns nullopt.  A failure to enqueue the wait also returns nullopt (nothing is moved).
  [[nodiscard]] std::optional<std::vector<device_ranges>> take(::cuda::stream_ref consumer_stream);
  /// The entries take() would move out, without taking them (the hook does not fire): empty
  /// unless ready, untaken and not flagged wrong_device.  For the single consumer, which may
  /// check them before committing to take(); valid until take() or destruction.
  [[nodiscard]] std::span<device_ranges const> peek() const noexcept;
  [[nodiscard]] bool taken() const noexcept;

  /// Logical bytes of the prefetched data.
  [[nodiscard]] std::size_t total_bytes() const noexcept { return _total_bytes; }
  /// Bytes reserved against the device budget for this prefetch.
  [[nodiscard]] std::size_t reserved_bytes() const noexcept { return _reserved_bytes; }
  [[nodiscard]] int device_id() const noexcept { return _device_id; }
  /// Entries still held: the hint count until take() moves them out, 0 after.
  [[nodiscard]] std::size_t entry_count() const noexcept;

 private:
  /// Exactly once (the _released exchange decides who reports).
  void fire_release(release_reason reason) noexcept;
  /// Leaves pending for @p next under the mutex; false when the state had already left it.
  /// With @p abandoned the abandon error is built here, only when the transition happens.
  bool settle(state next, std::exception_ptr error, bool abandoned) noexcept;

  int _device_id;
  /// Declared BEFORE _ranges, so it is destroyed after them: an untaken buffer's stream-ordered
  /// free is enqueued on a stream that still exists.
  device_stream_pool::lease _producer;
  /// The issuing GPU's allocation mutex, shared: it serializes the per-stream reservation tracker
  /// attach/reset around an issuer's allocations, and the destructor frees the untaken producer
  /// buffers under it, so a free never races another issuer's attach/reset on the same stream.
  /// Strong, so it outlives the GPU state the buffers' hooks captured.  Null: no serialization.
  std::shared_ptr<std::mutex> _allocation_mutex;
  std::size_t _total_bytes;
  std::size_t _reserved_bytes;
  claim_hook _on_claim;
  release_hook _on_release;
  /// Recorded on the producer stream by set_ready(); the consumer stream waits on it in take().
  cudaEvent_t _ready_event{nullptr};
  std::exception_ptr _event_error;  ///< why _ready_event could not be created, if it could not
  std::atomic<bool> _claimed{false};
  std::atomic<bool> _released{false};
  std::atomic<bool> _wrong_device{false};
  mutable std::atomic<std::uint64_t> _wait_ns{0};
  mutable std::mutex _mutex;
  mutable std::condition_variable _cv;
  state _state{state::pending};
  bool _abandoned{false};
  bool _taken{false};
  std::exception_ptr _failure;
  std::vector<device_ranges> _ranges;  ///< emptied by take(); declared LAST
};

}  // namespace sirius::op::scan
