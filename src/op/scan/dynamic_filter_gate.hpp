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
#include <span>
#include <unordered_map>

namespace sirius::op {
class sirius_dynamic_filter;
class dynamic_filter_snapshot;
}  // namespace sirius::op

namespace sirius::op::scan {

namespace detail {
struct gate_test_access;
}  // namespace detail

/**
 * @brief Nonowning identity of one filter binding in an owned endpoint channel for measurement
 * bookkeeping.
 */
struct bound_filter_key {
  /// The consumer's channel keeps the filter alive.
  sirius::op::sirius_dynamic_filter const* filter;
  /// The same filter can apply to multiple output ordinals.
  std::size_t output_ordinal;
  [[nodiscard]] bool operator==(bound_filter_key const&) const = default;
};

/**
 * @brief Hash function for `bound_filter_key` to be used to map bound filter keys to filter
 * measurements.
 */
struct bound_filter_key_hash {
  [[nodiscard]] std::size_t operator()(bound_filter_key key) const noexcept
  {
    auto const hash = std::hash<sirius::op::sirius_dynamic_filter const*>{}(key.filter);
    return hash ^ (key.output_ordinal + 0x9e3779b9U + (hash << 6) + (hash >> 2));
  }
};

struct gate_selection {
  bool applicable;
  bool needs_combined_sample;
};

/**
 * @brief Per-scan gate for post-decode dynamic filters
 *
 * An eligible clean combined sample trains it. A selective result stays active; an unselective
 * result waits for channel growth. Updates are serialized.
 */
class dynamic_filter_gate {
 public:
  static constexpr double k_default_keep_threshold = 0.9;

  explicit dynamic_filter_gate(double keep_threshold = k_default_keep_threshold)
    : _keep_threshold(keep_threshold)
  {
  }

  /// Updates from one split; empty splits do not train, and disabled gates wait for channel growth.
  void record_keep_ratio(std::size_t rows_before,
                         std::size_t rows_after,
                         std::size_t observed_filter_count) noexcept;

  // Marginal ratios measure each filter on rows surviving earlier masks.

  /// Updates only for a larger observed filter count; equal or older measurements are ignored.
  /// A newer measurement may overwrite a skippable verdict — a deliberate allowance for splits
  /// already in flight, not a bug.
  void record_filter_keep_ratio(bound_filter_key filter,
                                double kept,
                                std::size_t observed_filter_count) noexcept;

  /// Captures 1) applicability,
  ///          2) whether a combined sample is needed, and
  ///          3) the current per-binding keep ratios
  /// while holding the necessary locks.
  ///
  /// The caller allocates one estimate slot per key before calling. Applicable means filters exist
  /// and the gate is active or due for retraining.
  [[nodiscard]] gate_selection capture(sirius::op::dynamic_filter_snapshot const& filters,
                                       std::span<bound_filter_key const> keys,
                                       std::span<std::optional<double>> estimates) const;

  [[nodiscard]] static constexpr bool filter_skippable(double kept) noexcept
  {
    return kept > k_filter_skip_keep_threshold;
  }

 private:
  friend struct detail::gate_test_access;

  enum class state { UNKNOWN, ACTIVE, DISABLED };

  // Drop a filter after it keeps more than half of its input.
  static constexpr double k_filter_skip_keep_threshold = 0.5;

  double _keep_threshold;

  std::atomic<state> _state{state::UNKNOWN};

  // Relaxed reads may trigger redundant measurements; decision updates are serialized.
  std::atomic<std::size_t> _decided_filter_count{0};

  // Serializes state/count decisions across concurrent batches.
  mutable std::mutex _decision_mu;

  struct filter_measurement {
    double kept                       = 1.0;
    std::size_t observed_filter_count = 0;
  };

  mutable std::mutex _filter_ratios_mu;

  // Maps each bound filter key to its most recent measurement.
  std::unordered_map<bound_filter_key, filter_measurement, bound_filter_key_hash>
    _filter_keep_ratios;

  /// Returns null when absent or stale; a skippable verdict never becomes stale. The caller holds
  /// `_filter_ratios_mu`.
  [[nodiscard]] std::optional<double> current_keep_ratio(bound_filter_key filter,
                                                         std::size_t observed_filter_count) const;
};

namespace detail {
/**
 * @brief Test access to gate decisions that production reads only through
 * `dynamic_filter_gate::capture`
 */
struct gate_test_access {
  [[nodiscard]] static bool applicable(dynamic_filter_gate const& gate,
                                       sirius::op::dynamic_filter_snapshot const& filters);
  [[nodiscard]] static bool needs_combined_sample(
    dynamic_filter_gate const& gate, sirius::op::dynamic_filter_snapshot const& filters);
  [[nodiscard]] static std::optional<double> filter_keep_ratio(dynamic_filter_gate const& gate,
                                                               bound_filter_key filter,
                                                               std::size_t observed_filter_count);
};
}  // namespace detail

}  // namespace sirius::op::scan
