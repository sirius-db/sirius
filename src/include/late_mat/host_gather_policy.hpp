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

// Which way to read a HOST-tier pin, decided by measuring this machine's link
// (env gate: SIRIUS_EXP_LATE_MAT).
//
// Two routes produce the same values. IN PLACE reads only the selected rows,
// scattered, straight out of pinned host memory over unified addressing. STAGING
// copies the whole chunk to the device in bulk and gathers there. Which one wins
// depends on how much of the chunk the selection actually wants, and on how badly
// the link punishes scattered reads relative to a bulk copy:
//
//   in-place cost  ~ selected rows,  at the link's SCATTERED bandwidth
//   staging cost   ~ all rows,       at the link's BULK bandwidth
//
// On a coherent CPU-GPU link the two bandwidths are close, so in-place wins at
// every density and the crossover sits at or above 1.0. On PCIe a scattered read
// pulls a whole transaction per element, so the crossover falls to a few percent
// and staging wins for anything denser.
//
// THAT RATIO IS NOT INFERABLE FROM THE DEVICE NAME OR FROM BULK BANDWIDTH ALONE.
// A link can be fast in bulk and still cacheline-granular for scattered reads,
// which is exactly the combination that makes an in-place gather lose. So both
// numbers are MEASURED here, once per process, and the crossover is read off the
// measurements rather than interpolated from other machines.

#include <cstdint>
#include <optional>

namespace sirius::late_mat {

/// What the probe concluded about this machine's host-to-device link.
struct host_gather_policy {
  /// Fraction of a pinned table's rows below which reading in place beats
  /// staging. 0.0 means staging always wins (also the fail-closed answer when
  /// the probe could not run); 1.0 means in place always wins.
  double max_inplace_density = 0.0;
  /// How much dearer a host materialization is than the equivalent device
  /// gather, which is what the deferral value floor has to be scaled by since
  /// that floor was calibrated against a device gather.
  std::int64_t cost_multiplier = 64;
  /// False when the probe failed and the conservative defaults above are in use.
  bool measured = false;
  /// Bulk host-to-device bandwidth, bytes/s. Diagnostics only.
  double bulk_bytes_per_second = 0.0;
  /// Useful scattered-read bandwidth at full density, bytes/s. Diagnostics only.
  double inplace_bytes_per_second = 0.0;
};

/// The policy for the current device, measured on first call and shared after.
///
/// The probe costs one small pinned allocation and a handful of kernel launches
/// (tens of milliseconds, once per process). Any CUDA failure inside it is
/// swallowed and reported as the conservative policy above: a lost optimization,
/// never a wrong route.
[[nodiscard]] host_gather_policy const& measured_host_gather_policy();

/// Whether @p selected_rows out of @p total_rows should be read in place.
///
/// SIRIUS_EXP_LATE_MAT_HOST_INPLACE_MAX_DENSITY overrides the measured
/// crossover; setting it to 1 forces in place and 0 forces staging, which is how
/// a route can be pinned for a measurement or a test.
[[nodiscard]] bool prefer_inplace_host_gather(std::int64_t selected_rows, std::int64_t total_rows);

/// How many host-tier materializations have taken each route.
///
/// A correctness assertion passes whichever route ran, so a test that means to
/// exercise one has to be able to see which one it got. Monotonic for the life
/// of the process.
struct host_gather_route_counts {
  std::uint64_t inplace = 0;
  std::uint64_t staged  = 0;
};

[[nodiscard]] host_gather_route_counts host_gather_routes_taken();

/// Record that a materialization took a route. Called by the materializer.
void note_host_gather_route(bool inplace);

/// Pin the route, or release it back to the measured policy with `std::nullopt`.
///
/// The env override is parsed once per process, so it cannot exercise both
/// routes in one test binary; this can. Affects every subsequent host-tier
/// materialization on every thread, so a test that sets it must reset it.
void force_host_gather_route(std::optional<bool> inplace);

/// The value-floor multiplier for a host-tier bundle, measured unless
/// SIRIUS_EXP_LATE_MAT_HOST_COST_MULTIPLIER overrides it.
[[nodiscard]] std::int64_t host_tier_cost_multiplier();

}  // namespace sirius::late_mat
