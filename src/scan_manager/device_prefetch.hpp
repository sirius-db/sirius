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
#include "op/scan/prefetched_ranges.hpp"

#include <cucascade/memory/config.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace cucascade::memory {
class memory_space;
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::op::scan {
class scan_info;
}  // namespace sirius::op::scan

namespace sirius::scan_manager {

/// How one device-readahead attempt resolved.
struct device_readahead_result {
  std::size_t bytes{0};      ///< logical bytes requested (== prefetched_ranges::total_bytes())
  bool ok{false};            ///< every read settled with the expected byte count
  std::exception_ptr error;  ///< first error when !ok
};

/// Why an attempt did (or did not) issue.  Everything but @c issued leaves nothing installed on
/// the split and never calls @c on_done: the readahead worker falls back to the host tier.
enum class device_readahead_issue : std::uint8_t {
  issued,       ///< reads are in flight; on_done fires exactly once (possibly already did)
  unsupported,  ///< the split cannot consume device copies
  no_ranges,    ///< nothing to read (no hints / zero bytes)
  fell_behind,  ///< the consumer reached the split first (before or right after install)
  too_large,    ///< bytes_reserved exceeds the GPU's whole speculative budget
  pressure,     ///< budget exhausted, usage ceiling reached, or a demand reservation is admitting
  memory,       ///< the reservation was refused or the allocation hit OOM
  error,        ///< a host-side failure before anything was submitted (logged)
};

/// One GPU's shared speculative-readahead state, shared by every query running on that GPU and
/// owned for the whole runtime by @ref device_readahead_registry.
///
/// It rations a byte budget (a fraction of the GPU's configured capacity) across concurrent
/// queries, keeps post-admission consumption under a usage ceiling (the lesser of the threshold
/// and the downgrade stop, so speculation never pushes the space into eviction), and carries the
/// demand-admission guard the executor raises while it secures a real reservation so the readahead
/// cannot refill memory a demand read is waiting for.
///
/// Held by @c shared_ptr and @c enable_shared_from_this so that a budget hook captured by an
/// in-flight prefetch keeps the state alive past the registry: an asynchronous buffer may still
/// give its token back after the registry (and every readahead manager) is gone.  Every piece is
/// thread-safe and touches only atomics or short critical sections.
class device_readahead_gpu : public std::enable_shared_from_this<device_readahead_gpu> {
 public:
  /// @p capacity_bytes is the GPU's configured memory capacity; @p budget_bytes the speculative
  /// byte budget (0 means this GPU never speculates, so it is not registered); @p ceiling_bytes
  /// caps post-admission consumption.
  device_readahead_gpu(cucascade::memory::memory_space& space,
                       std::size_t capacity_bytes,
                       std::size_t budget_bytes,
                       std::size_t ceiling_bytes);

  [[nodiscard]] int device_id() const noexcept;
  [[nodiscard]] cucascade::memory::memory_space& space() const noexcept { return _space; }
  [[nodiscard]] std::size_t budget_bytes() const noexcept { return _budget_bytes; }
  [[nodiscard]] std::size_t ceiling_bytes() const noexcept { return _ceiling_bytes; }

  /// Serializes demand-guard entry, budget reservation and slot install, so a shed can never miss
  /// a just-installed allocation.  The issuer holds it around admission + install; the executor
  /// holds it around demand-guard entry.
  [[nodiscard]] std::mutex& allocation_mutex() noexcept { return *_alloc_mutex; }
  /// Shared owner of the same mutex.  A prefetch holds it so its destructor can free the untaken
  /// producer buffers under it (serialized with an issuer's per-stream tracker attach/reset) even
  /// after this state is gone.  Never lock it while dropping the last reference to a prefetch.
  [[nodiscard]] std::shared_ptr<std::mutex> const& allocation_mutex_owner() const noexcept
  {
    return _alloc_mutex;
  }

  /// Charge @p bytes to the speculative budget.  Refuses (returns false, nothing charged) when a
  /// demand reservation is admitting, the budget is exhausted, or the projected consumption would
  /// pass the ceiling.  Call under @ref allocation_mutex so the check and the charge are one step.
  [[nodiscard]] bool try_reserve(std::size_t bytes) noexcept;
  /// Give @p bytes of the speculative budget back.  The budget token is released exactly once, at
  /// claim or at waste, by prefetched_ranges::claim().
  void release(std::size_t bytes) noexcept;
  [[nodiscard]] std::size_t reserved() const noexcept
  {
    return _reserved.load(std::memory_order_relaxed);
  }

  /// Raise/lower the per-GPU demand-admission guard.  While it is up the issuer refuses, so an
  /// admitting demand read is never raced by fresh speculation on its GPU.
  void enter_demand_guard() noexcept { _demand_guard.fetch_add(1, std::memory_order_acq_rel); }
  void exit_demand_guard() noexcept { _demand_guard.fetch_sub(1, std::memory_order_acq_rel); }
  [[nodiscard]] bool demand_active() const noexcept
  {
    return _demand_guard.load(std::memory_order_acquire) > 0;
  }

  /// Remember an installed, still-unclaimed prefetch so a demand read under pressure can shed it.
  /// Call under @ref allocation_mutex (with the install).  May throw (host allocation): the
  /// issuer's slot guard unwinds the install.
  void register_slot(std::shared_ptr<op::scan::scan_info> const& split,
                     std::shared_ptr<op::scan::prefetched_ranges> const& ranges);

  /// Abandon and discard every installed prefetch that has not been claimed by a consumer, so the
  /// memory they reserved is released on their in-flight IO's completion.  Returns how many were
  /// shed -- every prefetch actually taken from its split, ready or still pending -- so a caller
  /// that retries after relief sees it.  A claimed prefetch (the executor is about to decode it)
  /// is left alone.  The producer's
  /// fan-in keeps each shed buffer alive until its real IO settles.
  std::size_t shed_unclaimed() noexcept;

 private:
  cucascade::memory::memory_space& _space;
  std::size_t _capacity_bytes;
  std::size_t _budget_bytes;
  std::size_t _ceiling_bytes;
  std::atomic<std::size_t> _reserved{0};
  std::atomic<int> _demand_guard{0};
  std::shared_ptr<std::mutex> _alloc_mutex;

  struct slot {
    std::weak_ptr<op::scan::scan_info> split;
    std::weak_ptr<op::scan::prefetched_ranges> ranges;
  };
  std::mutex _slots_mutex;
  std::vector<slot> _slots;
};

/// Runtime-wide registry of per-GPU device-readahead state, one record per GPU that has a positive
/// budget.  Owned by @ref sirius_scan_manager and torn down after every per-query readahead
/// manager has stopped; its per-GPU state outlives it through the budget hooks of any still
/// in-flight prefetch.
class device_readahead_registry {
 public:
  /// Builds a record for every GPU @ref cucascade::memory::gpu_memory_space_config in @p configs
  /// with a positive capacity.  @p budget_fraction and @p threshold_fraction are the two public
  /// knobs (finite, in [0, 1]); the budget is that fraction of the GPU's configured capacity and
  /// the ceiling is @c min(threshold x capacity, downgrade_stop_threshold).  A GPU whose budget
  /// rounds to zero, or has no live memory space, is left out.  With @p budget_fraction zero or
  /// @p configs empty the registry is disabled (@ref enabled() false).
  device_readahead_registry(cucascade::memory::memory_reservation_manager& manager,
                            double budget_fraction,
                            double threshold_fraction,
                            std::span<const cucascade::memory::memory_space_config> configs = {});

  /// Whether any GPU speculates (a positive budget on at least one space).
  [[nodiscard]] bool enabled() const noexcept { return !_gpus.empty(); }

  /// The record for @p device_id, or null when that GPU does not speculate.
  [[nodiscard]] device_readahead_gpu* gpu(int device_id) noexcept;

  /// Upper bound on stream-pool groups the registry will ever create, across all GPUs (an
  /// internal constant, not a knob).  At the default of 4 streams per group that is 128 streams.
  static constexpr std::size_t max_pool_groups = 32;

  /// Borrow a private stream pool ("group") for @p device_id, or null when none can be had (the
  /// cap of @ref max_pool_groups is reached with every group busy, or the streams could not be
  /// created) -- the caller then takes the host tier.
  ///
  /// A group is exclusive to one borrower at a time: concurrent readahead managers always get
  /// distinct pools, so no two of them share a stream.  A group is reused once the registry is
  /// its sole owner again, i.e. the borrower AND every prefetch buffer leased from it are gone;
  /// groups are never destroyed before then, so the same fixed stream handles serve every query.
  ///
  /// Why retained instead of created per query: the cache's completion poll keeps the raw handles
  /// of every stream it has seen (in a bounded set of lanes) and cannot detach one, so a stream
  /// destroyed at query end would be polled afterwards.  Lifetime contract: this registry (and so
  /// its groups) must be destroyed only after the caches have been shut down -- which
  /// sirius_scan_manager's destructor guarantees by shutting every cache down before its members.
  [[nodiscard]] std::shared_ptr<op::scan::device_stream_pool> acquire_pool(int device_id);

 private:
  std::map<int, std::shared_ptr<device_readahead_gpu>> _gpus;
  std::mutex _pools_mutex;
  std::map<int, std::vector<std::shared_ptr<op::scan::device_stream_pool>>> _pools;
  std::size_t _pool_count{0};
};

/// Read one @p split's hint ranges straight into device buffers on @p gpu, on a @p stream leased
/// from the calling readahead manager's own pool.
///
/// On @c issued the split carries a pending @c prefetched_ranges, its budget token and shed-slot
/// are accounted, and @p on_done runs exactly once (on an IO thread, or inline before this
/// returns).  Every other result leaves nothing installed and never calls @p on_done; the worker
/// then takes the host tier.  Never blocks on memory or IO (the one exception: a read whose
/// completion callback could not be installed is awaited inline so its buffers outlive it).  Host
/// side failures are reported, never thrown.  The caller keeps @p split alive for the call.
[[nodiscard]] device_readahead_issue issue_device_readahead(
  device_readahead_gpu& gpu,
  op::scan::device_stream_pool::lease stream,
  std::shared_ptr<op::scan::scan_info> const& split,
  sirius::exec::invocable<void(device_readahead_result) noexcept> on_done) noexcept;

}  // namespace sirius::scan_manager
