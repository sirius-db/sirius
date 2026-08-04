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

#include <cstddef>
#include <cstdint>
#include <functional>

namespace cucascade {
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius {

namespace scan_manager {
class sirius_scan_manager;
}  // namespace scan_manager

namespace memory {

/**
 * Guard rails around blocking GPU memory reservations (engine fix for the
 * reservation blocking-wait livelock: a demand that no amount of downgrading
 * can ever satisfy used to wait forever with zero diagnostics).
 *
 * Background: gpu-tier pinned tables are allocated straight from a memory
 * space's default allocator with no reservation, so they permanently occupy
 * the adaptor's allocated-bytes budget, and they live in the scan manager's
 * pinned-entry registry — in no data repository — so the downgrade executor
 * can never evict them. A blocking reservation whose demand exceeds
 * (reservation limit − pinned bytes) therefore can NEVER be satisfied:
 * cucascade's memory_space::make_reservation waits on its notification
 * channel forever with zero diagnostics.
 *
 * This module provides:
 *  1. the conservative satisfiability arithmetic (pure, unit-testable),
 *  2. a provider hook through which the pipeline executor can ask "how many
 *     unevictable gpu-tier pinned bytes sit on this memory space" without
 *     being coupled to the scan manager, and
 *  3. an RAII wait scope that makes any outstanding blocking reservation
 *     wait visible via a periodic INFO log line (a lazily-created watchdog
 *     thread that is parked on a condition variable — zero cost — whenever
 *     no wait is outstanding).
 */

//===----------------------------------------------------------------------===//
// Satisfiability arithmetic (pure)
//===----------------------------------------------------------------------===//

/// Largest full reservation that can ever be granted on a space whose
/// reservation limit is @p reservation_limit while @p unevictable_bytes of it
/// are permanently occupied (gpu-tier pins). Saturates at 0.
[[nodiscard]] constexpr std::size_t max_satisfiable_reservation(
  std::size_t reservation_limit, std::size_t unevictable_bytes) noexcept
{
  return reservation_limit > unevictable_bytes ? reservation_limit - unevictable_bytes : 0;
}

/// Conservative check: true iff a FULL reservation of @p demand bytes can
/// never succeed on the space, no matter how much reclaimable memory the
/// downgrade executor frees. Deliberately conservative — transient pressure
/// (other tasks' reservations) never triggers it, only permanently-pinned
/// bytes do.
[[nodiscard]] constexpr bool reservation_is_unsatisfiable(std::size_t demand,
                                                          std::size_t reservation_limit,
                                                          std::size_t unevictable_bytes) noexcept
{
  return demand > max_satisfiable_reservation(reservation_limit, unevictable_bytes);
}

//===----------------------------------------------------------------------===//
// Unevictable-pinned-bytes provider hook
//===----------------------------------------------------------------------===//

/// Returns the unevictable (gpu-tier pinned) bytes resident on @p space.
using unevictable_bytes_provider =
  std::function<std::size_t(const cucascade::memory::memory_space* space)>;

/// Register @p provider under @p owner (an opaque identity token; typically
/// the SiriusContext). Re-registering the same owner replaces its provider.
void register_unevictable_bytes_provider(const void* owner, unevictable_bytes_provider provider);

/// Remove the provider registered under @p owner. No-op when absent.
void unregister_unevictable_bytes_provider(const void* owner) noexcept;

/// Sum of all registered providers' answers for @p space. Returns 0 when no
/// provider is registered; a throwing provider contributes 0 (logged).
[[nodiscard]] std::size_t unevictable_pinned_bytes(
  const cucascade::memory::memory_space* space) noexcept;

//===----------------------------------------------------------------------===//
// Scan-manager walk (the canonical provider implementation)
//===----------------------------------------------------------------------===//

/// Total device bytes of all GPU-tier pinned entries resident on @p space,
/// recomputed on demand from @p mgr's pinned-entry registry (the ground truth
/// — no stateful mirror that could drift across merge/replace/unpin).
/// Compressed device chunks report their compressed payload bytes
/// (get_size_in_bytes), a slight undercount of the true footprint (alignment
/// padding) — conservative in the safe direction for the fail-fast check.
///
/// Thread safety: pin/unpin is serialized against query execution windows
/// (the same discipline visit_pinned_entries relies on), so calls made while
/// queries execute read a stable registry.
[[nodiscard]] std::size_t gpu_tier_pinned_bytes(
  const sirius::scan_manager::sirius_scan_manager& mgr,
  const cucascade::memory::memory_space* space);

//===----------------------------------------------------------------------===//
// Reservation-wait visibility
//===----------------------------------------------------------------------===//

/// RAII registration of one outstanding blocking reservation wait. While any
/// scope is alive, a watchdog thread emits one INFO line per outstanding wait
/// roughly every 10 seconds (requested bytes, elapsed, available/limit/pinned
/// bytes of the space). The watchdog thread is created lazily on the first
/// scope ever constructed and parks indefinitely on a condition variable when
/// no scope is alive — it costs nothing when nothing waits.
class reservation_wait_scope {
 public:
  reservation_wait_scope(const cucascade::memory::memory_space* space,
                         std::size_t requested_bytes,
                         std::uint64_t pipeline_id,
                         std::uint64_t task_id);
  ~reservation_wait_scope();

  reservation_wait_scope(const reservation_wait_scope&)            = delete;
  reservation_wait_scope& operator=(const reservation_wait_scope&) = delete;
  reservation_wait_scope(reservation_wait_scope&&)                 = delete;
  reservation_wait_scope& operator=(reservation_wait_scope&&)      = delete;

 private:
  std::uint64_t _id;
};

}  // namespace memory
}  // namespace sirius
