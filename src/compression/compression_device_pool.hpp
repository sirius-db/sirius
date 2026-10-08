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

#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::compression {

/**
 * @brief A device arena reserved for spill-compression working memory.
 *
 * The spill encoder allocates from the same RMM pool as the query, and it runs
 * precisely when that pool is exhausted — a downgrade only happens because the
 * GPU is full. Sharing is therefore circular: the allocation that would relieve
 * the pressure is the one certain to fail, and an `rmm::out_of_memory` there
 * latches `spill_compression_suppressed` for the rest of the episode.
 *
 * Measured on q3/SF1000 with no arena: compression latched off and back on 11
 * times in one query while the monitor issued 111,641 downgrade requests, and
 * the query had to be killed. With a 4 GiB arena the same query ran in 44.3 s,
 * level with an untouched baseline.
 *
 * The arena is a fixed allocation taken once at startup and never grown, so it
 * cannot compete with the query later. It is a *partition* of the device, not
 * extra memory: sirius_config::resolve_hardware() subtracts it from the GPU
 * memory space's capacity (see sirius_config::carve_compression_arena), so the
 * query pool and the arena together stay inside the configured usage limit.
 * Sizing it is still a cliff, not a gradient — at 1 GiB (too small for the
 * concurrent encodes) the same query failed outright and fell back to DuckDB.
 */

/**
 * @brief Outstanding-bytes counter with a resettable high-water mark.
 *
 * Backs the arena's always-on usage accounting (compression_device_pool_used_bytes,
 * compression_device_pool_peak_bytes). A class of its own so the reset semantics
 * can be unit-tested without installing an arena, which is process-wide and
 * permanent.
 *
 * Relaxed atomics throughout: a reset racing an allocation can miss that
 * allocation's contribution to the new window's peak. It is a diagnostic, read
 * once per query, so that is the right trade against a lock on every allocation.
 */
class usage_high_water_mark {
 public:
  void on_allocate(std::size_t bytes) noexcept
  {
    const auto now = _used.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak      = _peak.load(std::memory_order_relaxed);
    while (now > peak && !_peak.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {}
  }

  void on_deallocate(std::size_t bytes) noexcept
  {
    _used.fetch_sub(bytes, std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t used() const noexcept { return _used.load(std::memory_order_relaxed); }
  [[nodiscard]] std::size_t peak() const noexcept { return _peak.load(std::memory_order_relaxed); }

  /// Start a new measurement window: the peak restarts from what is outstanding
  /// right now (memory still held from the previous window counts, a peak that
  /// has since drained does not). Returns the previous window's peak.
  std::size_t reset_peak() noexcept
  {
    return _peak.exchange(_used.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

 private:
  std::atomic<std::size_t> _used{0};
  std::atomic<std::size_t> _peak{0};
};

/// Allocate the arena on @p device_id (the current device when negative).
/// @p bytes == 0 installs none: the compress path then allocates from the current
/// device resource -- the GPU memory space's reservation-aware adaptor -- under a
/// per-encode reservation (soft, or strict with
/// compression.spill_encode_strict_reservation; see compression_converters.cpp,
/// scoped_encode_reservation). Idempotent; returns false when the arena could not
/// be reserved, leaving the no-arena path in effect.
bool init_compression_device_pool(std::size_t bytes, int device_id = -1);

/// The resource spill compression allocates its transients from: the arena when
/// one is installed, else the current device resource (the query's pool, charged
/// to the calling thread's encode reservation when one is attached).
///
/// Only the explicit allocations go here. cuDF's internal temporaries and the few
/// Simpatico scratch buffers created without an mr (dictionary encode, the
/// dictionary rep's key-chars copy) always use the current device resource, so
/// with an arena they land in the query pool unreserved; without one they are
/// charged to the same thread reservation as everything else.
rmm::device_async_resource_ref compression_device_mr();

/// True when an arena is installed.
bool compression_device_pool_enabled() noexcept;

/// Configured arena size in bytes; 0 when disabled.
std::size_t compression_device_pool_bytes() noexcept;

/// Bytes currently allocated from the arena (0 when disabled). Counts what the
/// encoders hold, not the pool's fragmentation, so "nearly full" by this measure
/// is a lower bound on how full the arena really is.
std::size_t compression_device_pool_used_bytes() noexcept;

/// High-water mark of compression_device_pool_used_bytes() since the last
/// compression_device_pool_reset_peak() (0 when disabled). Always on, independent
/// of SIRIUS_COMPRESSION_ALLOC_STATS: it is what tells how much of the arena the
/// encoders actually needed, i.e. whether the carve-out could be smaller.
std::size_t compression_device_pool_peak_bytes() noexcept;

/// Restart the high-water mark (SiriusContext calls this at each query begin).
/// Process-wide: with concurrent queries the window spans all of them.
void compression_device_pool_reset_peak() noexcept;

// ── No-arena mode: encode reservations ───────────────────────────────────────
//
// Without an arena the encoder allocates from the GPU memory space itself, under
// a cuCascade reservation taken per encode (compression_converters.cpp,
// scoped_encode_reservation). These counters are that mode's equivalent of the
// arena's usage and peak: how much the encodes reserved, how much of it they
// used, and how often a reservation could not be had.

/// Per-window (per query) totals of the encode reservations.
struct encode_reservation_stats {
  /// Reservations granted and declined (not grantable, too little headroom, or
  /// the thread already held one) since the window began.
  std::uint64_t granted  = 0;
  std::uint64_t declined = 0;
  /// Bytes held by in-flight encode reservations right now, and the window's
  /// high-water mark of that sum -- what the encodes took from the query's
  /// budget at worst.
  std::size_t outstanding_reserved      = 0;
  std::size_t peak_outstanding_reserved = 0;
  /// Largest single reservation, and the largest amount any one encode actually
  /// allocated under its reservation (its tracker's peak). The ratio of the two
  /// says whether `spill_encode_reserve_fraction` is sized right.
  std::size_t largest_reserved = 0;
  std::size_t largest_used     = 0;
};

void note_encode_reservation_granted(std::size_t reserved_bytes) noexcept;
void note_encode_reservation_released(std::size_t reserved_bytes, std::size_t peak_used) noexcept;
void note_encode_reservation_declined() noexcept;
[[nodiscard]] encode_reservation_stats read_encode_reservation_stats() noexcept;
/// Restart the window (SiriusContext calls this at each query begin).
void reset_encode_reservation_window() noexcept;

}  // namespace sirius::compression
