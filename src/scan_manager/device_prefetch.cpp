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

#include "scan_manager/device_prefetch.hpp"

#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"
#include "op/scan/gpu_ingestible_types.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::scan_manager {

namespace {

using op::scan::prefetched_ranges;
using op::scan::scan_info;
using read_result   = sirius::exec::try_t<std::size_t>;
using read_callback = sirius::exec::invocable<void(read_result&&) &&>;
using done_callback = sirius::exec::invocable<void(device_readahead_result) noexcept>;

/// An exception_ptr holding a std::runtime_error, or whatever constructing it threw.
std::exception_ptr runtime_error_ptr(char const* what) noexcept
{
  try {
    return std::make_exception_ptr(std::runtime_error(what));
  } catch (...) {
    return std::current_exception();
  }
}

/// Blocks until @p f settles and returns its result; never throws.  @p f must be valid.
read_result settle_inline(sirius::exec::semi_future<std::size_t>&& f) noexcept
{
  try {
    return std::move(f).get_try();
  } catch (...) {
    return read_result{std::current_exception()};
  }
}

/// Fan-in over one attempt's per-hint reads.  Settles the prefetch and reports once every hint
/// and the issuer's guard have arrived.  Shared by every read callback, so it outlives the
/// issue() call; everything here may run on any thread (an IO completion, a pipeline thread
/// draining the completion poll, the worker inline), so it is noexcept and blocks only on its own
/// mutex.
struct completion {
  completion(std::vector<std::size_t> expected, std::size_t logical, done_callback done)
    : _remaining(expected.size() + 1),
      _expected_bytes(std::move(expected)),
      _bytes_logical(logical),
      _on_done(std::move(done))
  {
  }

  /// Hint @p k settled with @p result.  A short count is a failure: the backend clamps a read at
  /// end of file and still succeeds with the bytes it found.
  void arrive(std::size_t k, read_result&& result) noexcept
  {
    if (result.has_exception()) {
      record(std::move(result).exception());
    } else if (!result.has_value()) {
      record_failure("device read settled without a result");
    } else if (result.value() != _expected_bytes[k]) {
      try {
        record_failure("short device read: expected " + std::to_string(_expected_bytes[k]) +
                       " bytes, got " + std::to_string(result.value()));
      } catch (...) {
        record(std::current_exception());
      }
    }
    settle();
  }

  /// Drops the issuer's guard once every hint was submitted (or settled).
  void release_guard() noexcept { settle(); }

  std::shared_ptr<prefetched_ranges> keep_alive;  ///< until the last read settled

 private:
  void record(std::exception_ptr error) noexcept
  {
    std::lock_guard lock(_mutex);
    if (!_first_error) { _first_error = std::move(error); }
  }

  void record_failure(std::string const& what) noexcept
  {
    try {
      record(std::make_exception_ptr(std::runtime_error(what)));
    } catch (...) {
      record(std::current_exception());
    }
  }

  void settle() noexcept
  {
    if (_remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
    std::exception_ptr error;
    {
      std::lock_guard lock(_mutex);
      error = _first_error;
    }
    auto ranges = std::move(keep_alive);
    if (ranges) {
      if (error) {
        std::ignore = ranges->set_failed(error);
      } else {
        std::ignore = ranges->set_ready();
      }
    }
    // Before on_done: once it ran, the fan-in no longer keeps the buffers alive.
    ranges.reset();
    if (_on_done) {
      _on_done(device_readahead_result{.bytes = _bytes_logical, .ok = !error, .error = error});
      _on_done = nullptr;
    }
  }

  std::atomic<std::size_t> _remaining;
  std::vector<std::size_t> _expected_bytes;
  std::size_t _bytes_logical;
  done_callback _on_done;
  std::mutex _mutex;
  std::exception_ptr _first_error;
};

/// Abandons an installed attempt on every exit before its reads were submitted, so no consumer can
/// wait on a prefetch nobody will settle.
class slot_guard {
 public:
  slot_guard(scan_info& split, prefetched_ranges& ranges) noexcept : _split(split), _ranges(ranges)
  {
  }
  ~slot_guard()
  {
    if (_armed) {
      std::ignore = _ranges.abandon();
      _split.discard_prefetched_ranges();
    }
  }
  slot_guard(slot_guard const&)            = delete;
  slot_guard& operator=(slot_guard const&) = delete;

  void disarm() noexcept { _armed = false; }

 private:
  scan_info& _split;
  prefetched_ranges& _ranges;
  bool _armed{true};
};

/// Where a split's IO goes, for log lines.
std::string path_of(scan_info::fadvise_entry const& hint)
{
  return hint.datasource ? std::string{hint.datasource->get_io_object().object_path()}
                         : std::string{"<none>"};
}

/// issue_device_readahead() without the catch-all: may throw before anything was submitted.
device_readahead_issue try_issue(device_readahead_gpu& gpu,
                                 op::scan::device_stream_pool::lease& lease,
                                 std::shared_ptr<scan_info> const& split,
                                 done_callback& on_done)
{
  // 1. What the split would read: per hint the logical bytes, and the reservation, which is what
  //    the allocator will track (every buffer rounded up to the allocation alignment).
  if (!split || !split->supports_device_prefetch()) { return device_readahead_issue::unsupported; }
  auto const hints = split->fadvise_hints();
  if (hints.empty()) { return device_readahead_issue::no_ranges; }
  std::vector<std::size_t> expected_bytes;
  std::size_t bytes_logical  = 0;
  std::size_t bytes_reserved = 0;
  try {
    expected_bytes.reserve(hints.size());
  } catch (std::bad_alloc const&) {
    return device_readahead_issue::error;
  }
  for (auto const& hint : hints) {
    if (!hint.datasource) { return device_readahead_issue::unsupported; }
    std::size_t hint_bytes = 0;
    for (auto const& range : hint.ranges) {
      if (range.size() < 0) { return device_readahead_issue::unsupported; }
      auto const size = static_cast<std::size_t>(range.size());
      // Every sum and the alignment round-up are checked before they can wrap: a split this large
      // is refused as too large, long before an allocation or a read.
      constexpr auto max_size  = std::numeric_limits<std::size_t>::max();
      constexpr auto alignment = static_cast<std::size_t>(rmm::CUDA_ALLOCATION_ALIGNMENT);
      if (size > max_size - (alignment - 1)) { return device_readahead_issue::too_large; }
      auto const aligned = rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      if (size > max_size - hint_bytes || aligned > max_size - bytes_reserved) {
        return device_readahead_issue::too_large;
      }
      hint_bytes += size;
      bytes_reserved += aligned;
    }
    if (hint_bytes > std::numeric_limits<std::size_t>::max() - bytes_logical) {
      return device_readahead_issue::too_large;
    }
    expected_bytes.push_back(hint_bytes);
    bytes_logical += hint_bytes;
  }
  if (bytes_logical == 0) { return device_readahead_issue::no_ranges; }

  // 2. A split larger than this GPU's whole speculative budget could never be admitted: refuse.
  if (bytes_reserved > gpu.budget_bytes()) { return device_readahead_issue::too_large; }

  // 3. The consumer got here first: its own read is already under way.
  if (split->has_fallen_behind()) { return device_readahead_issue::fell_behind; }

  // 4. Allocation, reads and every free below belong to this GPU.  Declared first so it outlives
  //    every buffer and the objects that may hold the last reference to them.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu.device_id()}};
  ::cuda::stream_ref const stream = lease.view;
  rmm::cuda_stream_view const alloc_stream{stream.get()};

  // Declared BEFORE alloc_lock (step 5), deliberately: the last reference to the prefetch is
  // dropped on every setup-failure exit, and its destructor takes the allocation mutex to free the
  // untaken buffers.  Destroyed after alloc_lock is released, it can never self-deadlock.
  std::shared_ptr<prefetched_ranges> ranges;

  // 5. Admission and install are serialized against demand-guard entry and shedding, so a shed
  //    never misses a just-installed allocation.
  std::unique_lock alloc_lock{gpu.allocation_mutex()};

  // 6. Charge the speculative budget (demand guard, downgrade zone, budget and usage ceiling all
  //    checked here).  Refunded on every exit until the ranges object takes over the token.
  if (!gpu.try_reserve(bytes_reserved)) { return device_readahead_issue::pressure; }
  bool token_charged   = true;
  absl::Cleanup refund = [&] {
    if (token_charged) { gpu.release(bytes_reserved); }
  };

  // 7. Reserve the real memory and allocate one buffer per hint range under it, attached to this
  //    stream only for the allocations; the allocated bytes stay counted until a consumer frees
  //    them.
  auto reservation = gpu.space().make_reservation_or_null(bytes_reserved);
  if (!reservation) { return device_readahead_issue::memory; }

  std::vector<prefetched_ranges::device_ranges> entries;
  {
    auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
    if (allocator == nullptr ||
        !allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
      SIRIUS_LOG_WARN(
        "[readahead] could not attach the device prefetch reservation on GPU {}; skipping",
        gpu.device_id());
      return device_readahead_issue::error;  // not a memory shortage
    }
    absl::Cleanup detach = [allocator, s = stream] { allocator->reset_stream_reservation(s); };
    try {
      entries.resize(hints.size());
      for (std::size_t k = 0; k < hints.size(); ++k) {
        auto& buffers = entries[k].buffers;
        buffers.reserve(hints[k].ranges.size());
        for (auto const& range : hints[k].ranges) {
          buffers.emplace_back(static_cast<std::size_t>(range.size()),
                               alloc_stream,
                               gpu.space().get_default_allocator());
        }
      }
    } catch (rmm::out_of_memory const&) {
      return device_readahead_issue::memory;
    } catch (std::bad_alloc const&) {
      return device_readahead_issue::memory;
    }
  }

  // 8. Build everything that can throw BEFORE the slot is installed: once a consumer can see the
  //    prefetch, the only way out without issuing is the slot guard's abandon.
  std::shared_ptr<completion> fan_in;
  std::vector<std::vector<sirius::io::slice>> slices;
  std::vector<read_callback> callbacks;
  try {
    for (auto& entry : entries) {
      entry.spans.reserve(entry.buffers.size());
      for (auto const& buffer : entry.buffers) {
        entry.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
      }
    }
    fan_in =
      std::make_shared<completion>(std::move(expected_bytes), bytes_logical, std::move(on_done));
    // one vectored read per hint; zero-size ranges read nothing and get no slice
    slices.resize(hints.size());
    for (std::size_t k = 0; k < hints.size(); ++k) {
      auto const& ranges_k = hints[k].ranges;
      slices[k].reserve(ranges_k.size());
      for (std::size_t j = 0; j < ranges_k.size(); ++j) {
        if (ranges_k[j].size() == 0) { continue; }
        slices[k].emplace_back(static_cast<std::size_t>(ranges_k[j].offset()),
                               static_cast<std::size_t>(ranges_k[j].size()),
                               static_cast<std::uint8_t*>(entries[k].buffers[j].data()));
      }
    }
    callbacks.reserve(hints.size());
    for (std::size_t k = 0; k < hints.size(); ++k) {
      callbacks.emplace_back([c = fan_in, k](read_result&& r) mutable noexcept {
        c->arrive(k, std::move(r));
        c.reset();
      });
    }
    // The budget token lives on the prefetched_ranges from here: its claim() (at the executor's
    // claim, at take(), or in the destructor) gives it back exactly once.
    // Capture a strong reference to the per-GPU state, so the token can be returned even after the
    // registry (and every readahead manager) is gone while this prefetch is still in flight.
    auto on_claim = [state = gpu.shared_from_this(), bytes_reserved]() noexcept {
      state->release(bytes_reserved);
    };
    prefetched_ranges::release_hook on_release = [](prefetched_ranges::release_event) noexcept {};
    ranges             = std::make_shared<prefetched_ranges>(gpu.device_id(),
                                                 std::move(lease),
                                                 std::move(entries),
                                                 bytes_logical,
                                                 bytes_reserved,
                                                 std::move(on_claim),
                                                 std::move(on_release),
                                                 gpu.allocation_mutex_owner());
    token_charged      = false;  // the ranges object owns the refund now
    fan_in->keep_alive = ranges;
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN(
      "[readahead] device prefetch for '{}' not issued: {}", path_of(hints[0]), e.what());
    return device_readahead_issue::error;
  } catch (...) {
    SIRIUS_LOG_WARN("[readahead] device prefetch for '{}' not issued: unknown exception",
                    path_of(hints[0]));
    return device_readahead_issue::error;
  }

  // 9. Install.  From here every non-issued exit goes through the guard, which abandons the slot.
  if (!split->set_prefetched_ranges(ranges)) {
    SIRIUS_LOG_WARN("[readahead] device prefetch for '{}' not issued: slot occupied or claimed",
                    path_of(hints[0]));
    return device_readahead_issue::error;  // ranges destructor refunds the token
  }
  slot_guard guard{*split, *ranges};
  gpu.register_slot(split, ranges);

  // 10. The consumer may have overtaken (or the executor claimed) while we allocated.
  if (split->has_fallen_behind() || split->device_readahead_claimed()) {
    return device_readahead_issue::fell_behind;  // guard abandons; destructor refunds
  }

  // 11. Admission is done; let demand reads and sheds proceed while the IO runs.
  alloc_lock.unlock();

  // 12. Submit one vectored device read per hint.  A submitted read is never arrived early: its
  //     buffers must outlive it (the fan-in keeps them alive).
  for (std::size_t k = 0; k < hints.size(); ++k) {
    if (slices[k].empty()) {
      fan_in->arrive(k, read_result{std::size_t{0}});
      continue;
    }
    auto const& ds = *hints[k].datasource;
    sirius::exec::semi_future<std::size_t> f;
    try {
      auto io_ctx = ds.io_ctx();
      auto* cache = io_ctx->uses_prefetching_cache() ? io_ctx->cache() : nullptr;
      if (cache != nullptr) {
        f = cache->device_read_ranges_async(ds.get_io_object(), slices[k], stream, nullptr);
      } else {
        f = io_ctx->device_readv_async_io(ds.get_io_object(), slices[k], stream);
      }
    } catch (...) {
      fan_in->arrive(k, read_result{std::current_exception()});
      continue;
    }
    if (!f.valid()) {
      fan_in->arrive(k, read_result{runtime_error_ptr("device read returned no future")});
      continue;
    }
    try {
      std::move(f).install_callback(std::move(callbacks[k]));
    } catch (...) {
      // The IO is submitted.  install_callback resets the future's state only after the callback
      // is stored, so a still-valid future means the callback is NOT installed: wait inline and
      // arrive with the real result.  An invalid future means the callback IS installed.
      try {
        SIRIUS_LOG_ERROR(
          "[readahead] install_callback failed for hint {} on GPU {}; waiting for the read inline",
          k,
          gpu.device_id());
      } catch (...) {
      }
      if (f.valid()) {  // NOLINT(bugprone-use-after-move)
        fan_in->arrive(k, settle_inline(std::move(f)));
      }
    }
  }

  // 13. Issued: the fan-in owns the outcome from here.
  guard.disarm();
  fan_in->release_guard();
  return device_readahead_issue::issued;
}

}  // namespace

//===----------------------------------------------------------------------===//
// device_readahead_gpu
//===----------------------------------------------------------------------===//

device_readahead_gpu::device_readahead_gpu(cucascade::memory::memory_space& space,
                                           std::size_t capacity_bytes,
                                           std::size_t budget_bytes,
                                           std::size_t ceiling_bytes)
  : _space(space),
    _capacity_bytes(capacity_bytes),
    _budget_bytes(budget_bytes),
    _ceiling_bytes(ceiling_bytes),
    _alloc_mutex(std::make_shared<std::mutex>())
{
}

int device_readahead_gpu::device_id() const noexcept { return _space.get_device_id(); }

bool device_readahead_gpu::try_reserve(std::size_t bytes) noexcept
{
  // A demand read is admitting on this GPU: never refill memory it may be waiting for.
  if (demand_active()) { return false; }
  // Every comparison is a subtraction against a checked bound, so nothing here can wrap.
  auto const current = _reserved.load(std::memory_order_relaxed);
  if (current > _budget_bytes || bytes > _budget_bytes - current) { return false; }
  // Projected consumption (reservations + allocations by anyone, measured against the configured
  // capacity: get_available_memory() is capacity - the reservation/allocation counter) must stay
  // under the ceiling after this split lands.  The ceiling is already the lesser of the threshold
  // and the downgrade stop, so speculation never pushes the space into eviction.
  auto const available = _space.get_available_memory();
  auto const consumed  = _capacity_bytes > available ? _capacity_bytes - available : std::size_t{0};
  // Projected consumption must stay strictly below the ceiling: at it (or above) is refused.
  if (consumed >= _ceiling_bytes || bytes >= _ceiling_bytes - consumed) { return false; }
  _reserved.fetch_add(bytes, std::memory_order_relaxed);
  return true;
}

void device_readahead_gpu::release(std::size_t bytes) noexcept
{
  _reserved.fetch_sub(bytes, std::memory_order_relaxed);
}

void device_readahead_gpu::register_slot(std::shared_ptr<op::scan::scan_info> const& split,
                                         std::shared_ptr<op::scan::prefetched_ranges> const& ranges)
{
  std::lock_guard lock(_slots_mutex);
  // Drop records whose split or prefetch is already gone, so the list tracks only live slots.
  std::erase_if(_slots, [](slot const& s) { return s.split.expired() || s.ranges.expired(); });
  _slots.push_back(slot{split, ranges});
}

std::size_t device_readahead_gpu::shed_unclaimed() noexcept
{
  std::vector<slot> live;
  {
    std::lock_guard lock(_slots_mutex);
    live.swap(_slots);
  }
  std::size_t shed = 0;
  for (auto const& s : live) {
    auto split  = s.split.lock();
    auto ranges = s.ranges.lock();
    if (!split || !ranges) { continue; }
    // One atomic step under the split's slot mutex: a claimed split (the executor is about to
    // decode it) or a slot that no longer holds this prefetch yields nothing; otherwise the split
    // is marked permanently claimed (no reinstall) and the slot's reference is ours.
    auto taken = split->shed_unclaimed_prefetch(ranges);
    if (!taken) { continue; }
    // Outside the lock: abandon so a waiting consumer wakes and reads for itself (a no-op for a
    // prefetch that was already ready or failed), then drop the reference.  The producer's fan-in
    // still holds the buffers until the in-flight IO settles; a ready prefetch frees here.
    std::ignore = taken->abandon();
    taken.reset();
    ++shed;
  }
  return shed;
}

//===----------------------------------------------------------------------===//
// device_readahead_registry
//===----------------------------------------------------------------------===//

device_readahead_registry::device_readahead_registry(
  cucascade::memory::memory_reservation_manager& manager,
  double budget_fraction,
  double threshold_fraction,
  std::span<const cucascade::memory::memory_space_config> configs)
{
  auto const clamp01 = [](double v) {
    if (!std::isfinite(v) || v <= 0.0) { return 0.0; }
    return v > 1.0 ? 1.0 : v;
  };
  budget_fraction    = clamp01(budget_fraction);
  threshold_fraction = clamp01(threshold_fraction);
  if (budget_fraction <= 0.0) { return; }  // disabled: empty registry

  for (auto const& config : configs) {
    auto const* gpu_cfg = std::get_if<cucascade::memory::gpu_memory_space_config>(&config);
    if (gpu_cfg == nullptr) { continue; }
    auto const capacity = gpu_cfg->memory_capacity;
    if (capacity == 0) { continue; }
    auto const budget = static_cast<std::size_t>(budget_fraction * static_cast<double>(capacity));
    if (budget == 0) { continue; }
    // The ceiling is the lesser of the configured threshold and the downgrade stop, so speculation
    // stays below where the downgrade executor would start reclaiming.
    auto const threshold_bytes =
      static_cast<std::size_t>(threshold_fraction * static_cast<double>(capacity));
    auto const ceiling = std::min(threshold_bytes, gpu_cfg->downgrade_stop_threshold());
    // The issuer needs a mutable space handle to make reservations.
    auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, gpu_cfg->device_id);
    if (space == nullptr) { continue; }
    _gpus.emplace(gpu_cfg->device_id,
                  std::make_shared<device_readahead_gpu>(*space, capacity, budget, ceiling));
  }
}

device_readahead_gpu* device_readahead_registry::gpu(int device_id) noexcept
{
  auto it = _gpus.find(device_id);
  return it == _gpus.end() ? nullptr : it->second.get();
}

std::shared_ptr<op::scan::device_stream_pool> device_readahead_registry::acquire_pool(int device_id)
{
  std::lock_guard lock(_pools_mutex);
  try {
    auto& groups = _pools[device_id];
    // The registry is the sole owner exactly when no manager and no buffer lease holds the
    // group.  Nobody can copy a handle they do not have, so the count cannot rise under the lock.
    for (auto const& pool : groups) {
      if (pool.use_count() == 1) { return pool; }
    }
    if (_pool_count >= max_pool_groups) { return nullptr; }
    groups.push_back(op::scan::device_stream_pool::create(device_id));
    ++_pool_count;
    return groups.back();
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN(
      "[readahead] no device prefetch stream pool on GPU {}: {}", device_id, e.what());
  } catch (...) {
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// issue_device_readahead
//===----------------------------------------------------------------------===//

device_readahead_issue issue_device_readahead(
  device_readahead_gpu& gpu,
  op::scan::device_stream_pool::lease stream,
  std::shared_ptr<op::scan::scan_info> const& split,
  sirius::exec::invocable<void(device_readahead_result) noexcept> on_done) noexcept
{
  try {
    return try_issue(gpu, stream, split, on_done);
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN(
      "[readahead] device prefetch on GPU {} not issued: {}", gpu.device_id(), e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[readahead] device prefetch on GPU {} not issued: unknown exception",
                    gpu.device_id());
  }
  return device_readahead_issue::error;
}

}  // namespace sirius::scan_manager

namespace sirius::op::scan {

// Defined here (not in the header) so gpu_ingestible_types.hpp need not see the full
// prefetched_ranges type: this translation unit already includes it.
void scan_info::claim_device_readahead() noexcept
{
  std::shared_ptr<prefetched_ranges> installed;
  {
    std::lock_guard lock(_prefetched_mutex);
    if (_device_readahead_claimed.load(std::memory_order_acquire)) { return; }
    _device_readahead_claimed.store(true, std::memory_order_release);
    installed = _prefetched;
  }
  // Return the speculative budget token before the demand reservation; the buffers stay alive for
  // the decode.  claim() is idempotent, so a later take()/destructor does not double-release.
  if (installed) { installed->claim(); }
}

}  // namespace sirius::op::scan
