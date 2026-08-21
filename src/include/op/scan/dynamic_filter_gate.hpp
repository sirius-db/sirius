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

/**
 * @brief Per-scan gate for post-decode dynamic filters
 *
 * The first apply that produces a mask trains it. A selective result stays active; an unselective
 * result waits for channel growth. Updates are serialized.
 */
class dynamic_filter_gate {
 public:
  static constexpr double k_default_keep_threshold = 0.9;

  explicit dynamic_filter_gate(double keep_threshold = k_default_keep_threshold)
    : _keep_threshold(keep_threshold)
  {
  }

  /// True when filters exist and the gate is active or due for retraining.
  [[nodiscard]] bool applicable(sirius::op::sirius_dynamic_filter_set const& filters) const;

  /// Updates from one split; empty splits do not train, and disabled gates wait for channel growth.
  void record_keep_ratio(std::size_t rows_before,
                         std::size_t rows_after,
                         std::size_t observed_filter_count);

  // Marginal ratios measure each filter on rows surviving earlier masks.

  /// Returns null when absent or stale; a skippable verdict never becomes stale.
  [[nodiscard]] std::optional<double> filter_keep_ratio(
    sirius::op::sirius_dynamic_filter const* filter, std::size_t observed_filter_count) const;

  /// Updates only for a larger observed filter count; equal or older measurements are ignored.
  /// A newer measurement may overwrite a skippable verdict — a deliberate allowance for splits
  /// already in flight, not a bug.
  void record_filter_keep_ratio(sirius::op::sirius_dynamic_filter const* filter,
                                double kept,
                                std::size_t observed_filter_count);

  [[nodiscard]] static constexpr bool filter_skippable(double kept) noexcept
  {
    return kept > k_filter_skip_keep_threshold;
  }

 private:
  enum class state { unknown, active, disabled };

  // Drop a filter after it keeps more than half of its input.
  static constexpr double k_filter_skip_keep_threshold = 0.5;

  double _keep_threshold;

  std::atomic<state> _state{state::unknown};

  // Relaxed reads may trigger redundant measurements; decision updates are serialized.
  std::atomic<std::size_t> _decided_filter_count{0};

  // Serializes state/count decisions across concurrent splits.
  std::mutex _decision_mu;

  struct filter_measurement {
    double kept                       = 1.0;
    std::size_t observed_filter_count = 0;
  };

  mutable std::mutex _filter_ratios_mu;
  std::unordered_map<sirius::op::sirius_dynamic_filter const*, filter_measurement>
    _filter_keep_ratios;
};

}  // namespace sirius::op::scan
