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

#include "op/dynamic_filter_replica_space.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::op {

class sirius_dynamic_filter_set;

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan
//===----------------------------------------------------------------------===//
/// @brief Immutable plan-time description of one hash join's dynamic-filter publication.
///
/// The planner owns routing and placement decisions. The runtime publisher consumes this value but
/// cannot mutate its targets, policy, or device set after operator construction. Replica placements
/// cover every active GPU space, each paired with its planned HOST staging space. The build GPU's
/// space is included because it sources filter construction and the remote transfers, not because a
/// second copy is made there; only other GPUs receive replicas. Their owner follows the lifetime
/// contract on @ref dynamic_filter_replica_space.
class dynamic_filter_publish_plan final {
 public:
  struct probe_target {
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    std::vector<std::size_t> probe_col_idx;
    std::vector<cudf::data_type> probe_col_type;
  };

  /// Default fraction of a key's domain a build may cover and still publish that key's filters.
  static constexpr double k_default_domain_coverage_threshold = 0.9;

  /// Default bound on the exact hash IN-list's estimated cuco-set size as a fraction of the
  /// smallest probe-GPU L2 cache; see operator_params::dynamic_filter_inlist_max_l2_fraction for
  /// the full semantics.
  static constexpr double k_default_inlist_max_l2_fraction = 0.125;

  dynamic_filter_publish_plan() = default;
  /// The plan performs no domain validation on @p inlist_max_l2_fraction: both configuration
  /// surfaces already enforce the [0, 1] domain, and tests may legitimately construct out-of-domain
  /// plans.
  dynamic_filter_publish_plan(
    std::vector<probe_target> probe_targets,
    bool emit_zone_map_filters,
    std::vector<std::size_t> build_key_domain_cardinalities,
    std::vector<dynamic_filter_replica_space> replica_spaces,
    double domain_coverage_threshold = k_default_domain_coverage_threshold,
    double inlist_max_l2_fraction    = k_default_inlist_max_l2_fraction);

  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  [[nodiscard]] bool emit_zone_map_filters() const noexcept { return _emit_zone_map_filters; }
  /// Per pushed key, aligned with the pushdown info's join_condition: the unfiltered cardinality
  /// of the base table the build key traces to, or 0 when untraceable (coverage gates off).
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
  [[nodiscard]] double inlist_max_l2_fraction() const noexcept { return _inlist_max_l2_fraction; }

  /// \brief Drop replica targets on GPUs outside @p admitted_gpu_ids. An empty list means
  /// "no subset" and leaves the plan untouched. See
  /// sirius_pipeline_converter::restrict_dynamic_filter_replicas for why this is needed.
  void restrict_replicas_to(std::vector<int> const& admitted_gpu_ids);

 private:
  std::vector<probe_target> _probe_targets;
  bool _emit_zone_map_filters = false;
  std::vector<std::size_t> _build_key_domain_cardinalities;
  double _domain_coverage_threshold = k_default_domain_coverage_threshold;
  double _inlist_max_l2_fraction    = k_default_inlist_max_l2_fraction;
  /// Non-owning GPU/HOST placements. See @ref dynamic_filter_replica_space for the lifetime
  /// contract.
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};

}  // namespace sirius::op
