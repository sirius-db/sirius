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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace sirius::op {
class sirius_dynamic_filter;
class sirius_dynamic_filter_set;
}  // namespace sirius::op

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// dynamic_filter_gate
//===----------------------------------------------------------------------===//
/// @brief Per-scan selectivity gate for post-decode dynamic filters.
///
/// Used by @ref apply_dynamic_filters_gated_view. The first applied non-empty batch decides:
/// filters that keep more than 25% of rows are disabled for the scan; selective filters stay active.
/// If more filters publish after a disable decision, the gate re-arms and measures once more.
/// Concurrent batches may both measure during re-arm; that only costs redundant work.
class dynamic_filter_gate {
 public:
  /// True when a gated apply would do work now: at least one filter has published, and the gate is
  /// active or re-armed by a later publish after disable.
  [[nodiscard]] bool applicable(sirius::op::sirius_dynamic_filter_set const& filters) const;

  /// Record one split's keep ratio (@p rows_after / @p rows_before) for an apply that observed
  /// @p observed_filter_count published filters (snapshot taken before computing the mask, so a
  /// concurrent publish at worst triggers one extra re-measurement). Empty splits are no-ops;
  /// an active gate never demotes; a disabled gate re-decides only when the apply saw more
  /// filters than the disabling one did.
  void record_keep_ratio(std::size_t rows_before,
                         std::size_t rows_after,
                         std::size_t observed_filter_count);

  //===--------------------------------------------------------------------===//
  // Per-filter marginal usefulness
  //===--------------------------------------------------------------------===//
  // The scan-level gate above decides whether applying anything is worth it; these decide whether
  // one filter still earns its mask. The apply cascades filters most-selective-first and records
  // each filter's marginal keep ratio (its drop on the rows surviving the filters before it). A
  // filter whose marginal keep exceeds the skip threshold is dropped from later splits. Skipping is
  // safe: the join is authoritative.

  /// Marginal keep ratio recorded for @p filter, or nullopt while unmeasured.
  [[nodiscard]] std::optional<double> filter_keep_ratio(
    sirius::op::sirius_dynamic_filter const* filter) const;

  /// Record @p filter's marginal keep ratio. First measurement wins; later calls are no-ops
  /// (later splits see survivor sets whose composition depends on cascade order — the first
  /// measurement is the stable one).
  void record_filter_keep_ratio(sirius::op::sirius_dynamic_filter const* filter, double kept);

  /// True when @p kept marks a filter as not worth its per-split mask kernel.
  [[nodiscard]] static constexpr bool filter_skippable(double kept) noexcept
  { return kept > k_filter_skip_keep_threshold; }

 private:
  enum class state { unknown, active, disabled };

  /// A filter that keeps more than this fraction of the rows it sees prunes too little to repay
  /// its per-split mask kernel, so it is dropped from later splits.
  static constexpr double k_filter_skip_keep_threshold = 0.5;

  std::atomic<state> _state{state::unknown};

  /// Filter count observed by the apply whose ratio last decided @c _state. Read together with
  /// @c _state without joint atomicity: a torn read at worst causes one extra measurement.
  std::atomic<std::size_t> _decided_filter_count{0};

  /// First-measured marginal keep ratio per filter. Mutex-guarded: touched once per (split,
  /// filter) on the apply slow path, never on the zero-copy fast path.
  mutable std::mutex _filter_ratios_mu;
  std::unordered_map<sirius::op::sirius_dynamic_filter const*, double> _filter_keep_ratios;
};

}  // namespace sirius::op::scan
