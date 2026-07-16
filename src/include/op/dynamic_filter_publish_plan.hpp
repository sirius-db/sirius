/*
 * Copyright 2025, Sirius Contributors.
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

/**
 * @file
 * @brief The immutable value used by the legacy runtime publication path.
 *
 * C1a-2a leaves `JoinFilterPushdownInfo` together with this plan as the sole runtime publication
 * authority. The canonical planner sidecar is declared separately in
 * `dynamic_filter_publish_plan_builder.hpp`.
 */

// sirius
#include <op/dynamic_filter_replica_space.hpp>

// cudf
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::op {

class sirius_dynamic_filter_set;

/// @brief Immutable plan-time description consumed by the legacy runtime publisher.
class dynamic_filter_publish_plan final {
 public:
  struct probe_target {
    /// Existing scan-created delivery channel.
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    /// One entry per DuckDB filter ordinal; each value indexes the target scan's `column_ids`.
    std::vector<std::size_t> probe_col_idx;
    /// Storage type corresponding to each `probe_col_idx` entry.
    std::vector<cudf::data_type> probe_col_type;
  };

  /// Default coverage cutoff; a key is skipped when estimated coverage is at least this value.
  static constexpr double k_default_domain_coverage_threshold = 0.9;

  dynamic_filter_publish_plan() = default;
  dynamic_filter_publish_plan(
    std::vector<probe_target> probe_targets,
    bool emit_zone_map_filters,
    std::vector<std::size_t> build_key_domain_cardinalities,
    std::vector<dynamic_filter_replica_space> replica_spaces,
    double domain_coverage_threshold = k_default_domain_coverage_threshold);

  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  [[nodiscard]] bool emit_zone_map_filters() const noexcept { return _emit_zone_map_filters; }
  /// Per DuckDB filter ordinal, aligned with `JoinFilterPushdownInfo::join_condition`: the best
  /// available cardinality estimate for the base table to which the build key traces, or zero when
  /// untraceable, which disables the coverage gate for that ordinal.
  [[nodiscard]] std::vector<std::size_t> const& build_key_domain_cardinalities() const noexcept
  {
    return _build_key_domain_cardinalities;
  }
  [[nodiscard]] std::vector<dynamic_filter_replica_space> const& replica_spaces() const noexcept
  {
    return _replica_spaces;
  }
  [[nodiscard]] double domain_coverage_threshold() const noexcept
  {
    return _domain_coverage_threshold;
  }

 private:
  std::vector<probe_target> _probe_targets;
  bool _emit_zone_map_filters = false;
  std::vector<std::size_t> _build_key_domain_cardinalities;
  double _domain_coverage_threshold = k_default_domain_coverage_threshold;
  /// Non-owning GPU/HOST placements. See dynamic_filter_replica_space for the lifetime contract.
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};

}  // namespace sirius::op
