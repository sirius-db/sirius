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

namespace sirius::op {
struct dynamic_filter_stats;
}  // namespace sirius::op

namespace sirius::op::scan {

/// @brief Per-scan runtime gate for the parquet reader's dynamic-filter merge (WI-0b).
///
/// Whether merging dynamic filters into `reader_options::set_filter` pays is a property of the
/// data -- row groups the merged predicate excludes are the only work uniquely saved at the
/// reader -- so the decision is made at runtime from the reader's own accounting: the
/// `cudf::io::table_metadata` returned by each split's `read_parquet` (row groups in versus
/// remaining after the statistics/bloom stages). One instance per `parquet_gpu_ingestible`,
/// shared by that scan's concurrent split tasks; per-execution because plans are per-execution
/// (main doc, "Execution-scoped state").
///
/// State machine: `measuring` (initial) merges and samples every split; any sample with a pruned
/// row group makes the gate `active`, terminally -- boundaries only tighten, so observed pruning
/// cannot go stale. `k_disable_after_barren_splits` samples with zero pruning make it `disabled`:
/// the dynamic merge is skipped until the channel generation reaches the re-arm point, which
/// starts one generation past the disable and doubles on every barren re-measurement (1, 2, 4,
/// ...), bounding re-measurement to O(log G) while still catching a boundary that tightens into
/// usefulness. Only the dynamic merge is gated; static WHERE pushdown, membership masks,
/// endpoints, and the sink prefilter are untouched (main doc, "The siting rule is necessary but
/// not sufficient: the reader path needs a runtime gate").
///
/// Evidence rules: the caller records a sample only when the merge actually added dynamic
/// conjuncts and the reader reported accounting -- a pinned-cache-served scan (no reader), a
/// device without replicas, the zero-row fallback split, and a filterless split contribute no
/// evidence, never "zero pruning". A barren sample recorded while disabled with a generation
/// below the re-arm point is a pre-decision straggler and consumes no re-arm budget; a sample
/// with pruning activates from any state and generation (monotone tightening makes older
/// evidence hold a fortiori). `applicable()` is lock-free; `record_sample()` serializes the rare
/// decision, exactly like `dynamic_filter_gate`'s decision mutex.
class reader_pruning_gate {
 public:
  enum class state : std::uint8_t { measuring, active, disabled };

  /// Barren (zero-pruning) samples required before the merge is disabled. Four bounds the
  /// adversary's residual to ~4 merges plus O(log G) re-measurements out of thousands of splits,
  /// while the clustered winner activates on its first measured split; one sample would be
  /// hostage to a single boundary-straddling split, and more buys nothing the backoff re-arm
  /// does not already recover.
  static constexpr std::uint64_t k_disable_after_barren_splits = 4;

  /// First re-arm gap after a disable, in channel generations; doubles on every barren
  /// re-measurement (the 1st/2nd/4th/8th schedule of the design doc).
  static constexpr std::uint64_t k_initial_rearm_generation_gap = 1;

  /// Whether this split's dynamic merge is due. Lock-free; callers pass the channel's advisory
  /// `generation()` (predicates are still built only from a coherent snapshot).
  [[nodiscard]] bool applicable(std::uint64_t channel_generation) const noexcept
  {
    auto const s = _state.load(std::memory_order_relaxed);
    if (s != state::disabled) { return true; }
    return channel_generation >= _rearm_at.load(std::memory_order_relaxed);
  }

  /// Record one merged split's reader accounting against the snapshot generation the merge used.
  /// No-ops (no evidence) when @p row_groups_considered is zero or @p row_groups_remaining
  /// exceeds it. Increments `reader_gate_*` counters on @p stats when non-null:
  /// measurements/considered/pruned per recorded sample, `rearmed` per permitted re-measurement
  /// taken while disabled, `disabled` per transition into the disabled state (re-disables
  /// included). Delivery-time only -- see the counter block comment.
  void record_sample(std::size_t row_groups_considered,
                     std::size_t row_groups_remaining,
                     std::uint64_t observed_generation,
                     sirius::op::dynamic_filter_stats* stats);

  /// Current state, for tests.
  [[nodiscard]] state current_state() const noexcept
  {
    return _state.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<state> _state{state::measuring};
  /// Channel generation at which a disabled gate permits its next measurement.
  std::atomic<std::uint64_t> _rearm_at{0};

  /// Serializes the rare decision after a read has completed. Applicability stays lock-free.
  std::mutex _decision_mu;
  std::uint64_t _barren_samples = 0;                               // guarded by _decision_mu
  std::uint64_t _rearm_gap      = k_initial_rearm_generation_gap;  // guarded by _decision_mu
};

}  // namespace sirius::op::scan
