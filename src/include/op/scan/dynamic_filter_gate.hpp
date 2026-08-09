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
struct dynamic_filter_snapshot;
}  // namespace sirius::op

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// dynamic_filter_gate
//===----------------------------------------------------------------------===//
/// @brief Per-scan selectivity gate for post-decode dynamic filters.
///
/// `apply_dynamic_filters_gated_view()` uses the first applicable non-empty batch to decide whether
/// post-decode filtering earns its cost. A keep ratio above `keep_threshold` disables filtering;
/// otherwise the gate becomes permanently active. If the change signal grows after a disable
/// decision, the gate permits one new measurement. Decision updates are serialized so an older
/// concurrent batch cannot overwrite an active decision.
///
/// One monotonic re-arm marker per instance, and one marker domain per instance, fixed by the
/// instance's owner: a scan or endpoint gate only ever observes channel generations (snapshot
/// overloads; append-only set callers delegate through them), while the Top-N sink prefilter gate
/// only ever observes its coordinator's boundary-update count. Mixing domains on one instance is
/// a programming error enforced by call-site discipline, not runtime tagging.
class dynamic_filter_gate {
 public:
  /// Default fraction of retained rows above which the scan's post-decode filtering is disabled
  /// (a filter this unselective does not repay its per-split mask kernel).
  static constexpr double k_default_keep_threshold = 0.9;

  /// @param keep_threshold Disable the scan's filtering once a measured split keeps more than this
  ///                       fraction of its rows. In [0, 1]; 1.0 never disables.
  explicit dynamic_filter_gate(double keep_threshold = k_default_keep_threshold)
    : _keep_threshold(keep_threshold)
  {
  }

  /// Channel-free variant of @ref applicable for callers whose growth signal is a monotonic
  /// update count instead of a filter set -- the Top-N sink prefilter passes its coordinator's
  /// boundary-update count. Work is due when the count is nonzero and the gate is active or the
  /// count has grown beyond the measurement that disabled it.
  [[nodiscard]] bool applicable(std::size_t observed_update_count) const;

  /// Generation-aware applicability: work is due when @p snap holds filters and the gate is
  /// active or the channel generation has advanced past the disabling decision's generation
  /// (replacement never grows `filter_count`, so generation is the scan-side change signal).
  [[nodiscard]] bool applicable(sirius::op::dynamic_filter_snapshot const& snap) const;

  /// Record one split's keep ratio (@p rows_after / @p rows_before) against @p observed_marker,
  /// the instance's change-signal value observed before computing the mask (channel generation
  /// for scan/endpoint gates, boundary-update count for the prefilter gate; `std::size_t`
  /// callers convert losslessly). A concurrent grower causes at most one extra re-measurement.
  /// Empty splits are no-ops; an active gate never demotes; a disabled gate re-decides only when
  /// the apply observed a newer marker than the disabling one did.
  void record_keep_ratio(std::size_t rows_before,
                         std::size_t rows_after,
                         std::uint64_t observed_marker);

  //===--------------------------------------------------------------------===//
  // Per-filter marginal usefulness
  //===--------------------------------------------------------------------===//
  // Membership filters also record their marginal keep ratio after earlier masks. A skippable
  // verdict omits the filter from every later split permanently; a selective reading goes stale
  // on channel growth and is remeasured.

  /// Marginal keep ratio for @p filter, or `std::nullopt` when unmeasured. A skippable ratio is
  /// returned at any channel size; a selective one only when measured against at least
  /// @p observed_filter_count filters, else `std::nullopt` so the caller remeasures.
  [[nodiscard]] std::optional<double> filter_keep_ratio(
    sirius::op::sirius_dynamic_filter const* filter, std::size_t observed_filter_count) const;

  /// Record @p filter's marginal keep ratio against a channel of @p observed_filter_count filters.
  /// The first measurement at one channel size wins; a larger channel size supersedes it. Only a
  /// measurement already in flight can supersede a skippable verdict, because
  /// @ref filter_keep_ratio never invalidates one.
  void record_filter_keep_ratio(sirius::op::sirius_dynamic_filter const* filter,
                                double kept,
                                std::size_t observed_filter_count);

  /// True when @p kept marks a filter as not worth its per-split mask kernel.
  [[nodiscard]] static constexpr bool filter_skippable(double kept) noexcept
  {
    return kept > k_filter_skip_keep_threshold;
  }

 private:
  enum class state { unknown, active, disabled };

  /// A filter that keeps more than this fraction of the rows it sees prunes too little to repay
  /// its per-split mask kernel, so it is dropped from later splits.
  static constexpr double k_filter_skip_keep_threshold = 0.5;

  /// Keep ratio above which the scan-level gate disables. Set at construction from config.
  double _keep_threshold;

  std::atomic<state> _state{state::unknown};

  /// Change-signal marker observed by the apply whose ratio last decided @c _state (one domain
  /// per instance -- see the class comment). Read together with @c _state without joint
  /// atomicity: a torn read at worst causes one extra measurement.
  std::atomic<std::uint64_t> _decided_marker{0};

  /// Serializes the rare decision update after a mask has already been computed. Applicability
  /// remains lock-free; the lock only makes the state/count transition honor ACTIVE's terminal
  /// contract when batches from different filter generations finish concurrently.
  std::mutex _decision_mu;

  /// One filter's marginal keep ratio and the channel size it was measured against. The size is
  /// carried so channel growth can invalidate a selective reading; a skippable one is permanent —
  /// see @ref filter_keep_ratio.
  struct filter_measurement {
    double kept                       = 1.0;
    std::size_t observed_filter_count = 0;
  };

  /// Marginal keep ratio per filter, each with the channel size it was measured against.
  /// Mutex-guarded: touched once per (split, filter) on the apply slow path, never on the
  /// zero-copy fast path.
  mutable std::mutex _filter_ratios_mu;
  std::unordered_map<sirius::op::sirius_dynamic_filter const*, filter_measurement>
    _filter_keep_ratios;
};

}  // namespace sirius::op::scan
