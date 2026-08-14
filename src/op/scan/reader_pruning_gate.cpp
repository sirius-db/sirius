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

#include <log/logging.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/scan/reader_pruning_gate.hpp>

#include <mutex>

namespace sirius::op::scan {

void reader_pruning_gate::record_sample(std::size_t row_groups_considered,
                                        std::size_t row_groups_remaining,
                                        std::uint64_t observed_generation,
                                        sirius::op::dynamic_filter_stats* stats)
{
  if (row_groups_considered == 0 || row_groups_remaining > row_groups_considered) {
    return;  // no evidence
  }
  auto const pruned = row_groups_considered - row_groups_remaining;

  std::scoped_lock lock(_decision_mu);
  auto const s = _state.load(std::memory_order_relaxed);
  bool const was_rearm =
    s == state::disabled && observed_generation >= _rearm_at.load(std::memory_order_relaxed);
  if (s == state::disabled && !was_rearm && pruned == 0) {
    return;  // pre-decision straggler: consumes no re-arm budget
  }

  if (stats != nullptr) {
    stats->reader_gate_measurements.fetch_add(1, std::memory_order_relaxed);
    stats->reader_gate_row_groups_considered.fetch_add(row_groups_considered,
                                                       std::memory_order_relaxed);
    stats->reader_gate_row_groups_pruned.fetch_add(pruned, std::memory_order_relaxed);
    if (was_rearm) { stats->reader_gate_rearmed.fetch_add(1, std::memory_order_relaxed); }
  }

  if (pruned > 0) {  // terminal success, from any state and generation
    if (s != state::active) {
      SIRIUS_LOG_DEBUG("[reader_pruning_gate] pruned {}/{} row groups (generation {}) -> active.",
                       pruned,
                       row_groups_considered,
                       observed_generation);
    }
    _state.store(state::active, std::memory_order_relaxed);
    return;
  }
  if (s == state::active) { return; }  // terminal; never demotes
  if (s == state::measuring) {
    if (++_barren_samples >= k_disable_after_barren_splits) {
      _rearm_gap = k_initial_rearm_generation_gap;
      _rearm_at.store(observed_generation + _rearm_gap, std::memory_order_relaxed);
      _state.store(state::disabled, std::memory_order_relaxed);
      if (stats != nullptr) { stats->reader_gate_disabled.fetch_add(1, std::memory_order_relaxed); }
      SIRIUS_LOG_DEBUG(
        "[reader_pruning_gate] {} barren samples -> disabled; re-arm at generation {}.",
        _barren_samples,
        observed_generation + _rearm_gap);
    }
    return;
  }
  // disabled && was_rearm && barren: double the gap and re-disable.
  _rearm_gap *= 2;
  _rearm_at.store(observed_generation + _rearm_gap, std::memory_order_relaxed);
  if (stats != nullptr) { stats->reader_gate_disabled.fetch_add(1, std::memory_order_relaxed); }
  SIRIUS_LOG_DEBUG(
    "[reader_pruning_gate] barren re-measurement at generation {} -> re-disabled; re-arm at "
    "generation {}.",
    observed_generation,
    observed_generation + _rearm_gap);
}

}  // namespace sirius::op::scan
