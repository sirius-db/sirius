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

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op {

class sirius_dynamic_filter_set;

/**
 * @brief Shape captured before computed keys are materialized
 *
 * Admission must not infer it from rewritten conditions.
 */
enum class dynamic_filter_key_shape : std::uint8_t { direct, cast, computed };

struct dynamic_filter_condition_shape {
  dynamic_filter_key_shape probe = dynamic_filter_key_shape::direct;
  dynamic_filter_key_shape build = dynamic_filter_key_shape::direct;

  [[nodiscard]] bool operator==(dynamic_filter_condition_shape const&) const = default;
};

/**
 * @brief Consumer route: scans accept zone maps; direct routes are membership-only
 *
 * Channel push ordinals are always in the target's output schema.
 */
enum class dynamic_filter_route_class : std::uint8_t { scan, direct };

struct dynamic_filter_publication_policy {
  static constexpr double k_default_domain_coverage_threshold = 0.9;
  static constexpr double k_default_inlist_max_l2_fraction    = 0.125;

  bool emit_zone_map_filters       = false;
  double domain_coverage_threshold = k_default_domain_coverage_threshold;
  double inlist_max_l2_fraction    = k_default_inlist_max_l2_fraction;
};

/**
 * @brief Publication metadata for one hash join
 *
 * Pipeline setup may narrow replicas before execution; the plan is immutable during execution.
 */
class dynamic_filter_publish_plan final {
 public:
  struct admitted_key {
    // Original planner order; never index equality-reordered physical-join arrays with this.
    std::size_t planner_condition_index = 0;
    cudf::size_type build_key_ordinal   = 0;
    // Discovery starts here; remapping may make this differ from channel_push_ordinal.
    cudf::size_type probe_key_ordinal = 0;
    cudf::data_type storage_type{cudf::type_id::EMPTY};
    // EMPTY means the probe has no cuDF representation.
    cudf::data_type probe_storage_type{cudf::type_id::EMPTY};
    dynamic_filter_condition_shape key_shape{};
    // Exact base-table row bound; 0 disables coverage gating.
    std::size_t build_key_domain_cardinality = 0;
    // Required before row count can represent key-domain coverage.
    bool build_key_proven_unique = false;

    [[nodiscard]] bool operator==(admitted_key const&) const = default;
  };

  struct key_binding {
    std::size_t admitted_key_index = 0;
    // Target consumer output ordinal.
    std::size_t channel_push_ordinal = 0;
    // EMPTY suppresses zone maps; direct routes require an exact INT32 or INT64 match.
    cudf::data_type probe_storage_type{cudf::type_id::EMPTY};

    [[nodiscard]] bool operator==(key_binding const&) const = default;
  };

  struct probe_target {
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    dynamic_filter_route_class route_class = dynamic_filter_route_class::scan;
    bool accepts_zone_map_filters          = true;
    std::vector<key_binding> key_bindings;
  };

  dynamic_filter_publish_plan() = default;

  /**
   * @brief Validates and stores publication metadata
   *
   * Replica spaces are sorted and deduplicated. Target ordinals must already be valid in the
   * consumer schema.
   *
   * @throw std::invalid_argument for invalid key, target, binding, or replica metadata
   */
  dynamic_filter_publish_plan(std::vector<admitted_key> admitted_keys,
                              std::vector<probe_target> probe_targets,
                              std::vector<dynamic_filter_replica_space> replica_spaces,
                              dynamic_filter_publication_policy policy = {});

  // A target enables the plan even when no key was admitted.
  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  [[nodiscard]] std::vector<admitted_key> const& admitted_keys() const noexcept
  {
    return _admitted_keys;
  }
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  [[nodiscard]] bool emit_zone_map_filters() const noexcept
  {
    return _policy.emit_zone_map_filters;
  }
  [[nodiscard]] std::vector<dynamic_filter_replica_space> const& replica_spaces() const noexcept
  {
    return _replica_spaces;
  }
  [[nodiscard]] bool has_replica_on_device(int gpu_device_id) const noexcept;
  [[nodiscard]] double domain_coverage_threshold() const noexcept
  {
    return _policy.domain_coverage_threshold;
  }
  [[nodiscard]] double inlist_max_l2_fraction() const noexcept
  {
    return _policy.inlist_max_l2_fraction;
  }

  /**
   * @brief Retains replicas on the listed GPUs
   *
   * An empty list is a no-op. If no replicas remain, the plan is disabled and targets are cleared.
   */
  void restrict_replicas_to(std::vector<int> const& admitted_gpu_ids);

 private:
  std::vector<admitted_key> _admitted_keys;
  std::vector<probe_target> _probe_targets;
  dynamic_filter_publication_policy _policy{};
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};

}  // namespace sirius::op
