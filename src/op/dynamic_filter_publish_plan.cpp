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

#include "op/dynamic_filter_publish_plan.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius::op {

dynamic_filter_publish_plan::dynamic_filter_publish_plan(
  std::vector<probe_target> probe_targets,
  bool emit_zone_map_filters,
  std::vector<std::size_t> build_key_domain_cardinalities,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  double domain_coverage_threshold)
  : _probe_targets(std::move(probe_targets)),
    _emit_zone_map_filters(emit_zone_map_filters),
    _build_key_domain_cardinalities(std::move(build_key_domain_cardinalities)),
    _replica_spaces(std::move(replica_spaces)),
    _domain_coverage_threshold(domain_coverage_threshold)
{
  if (!_probe_targets.empty() && _replica_spaces.empty()) {
    throw std::invalid_argument(
      "[dynamic_filter_publish_plan] An enabled dynamic-filter publish plan requires at least one "
      "GPU memory space");
  }
  for (auto const& target : _replica_spaces) {
    if (target.get_gpu_space().get_tier() != cucascade::memory::Tier::GPU) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] A dynamic-filter replica target must be a GPU memory space");
    }
    if (target.get_host_staging_space().get_tier() != cucascade::memory::Tier::HOST) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] Dynamic-filter staging requires a HOST memory space");
    }
  }
  auto const device_less = [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() < rhs.get_gpu_space().get_device_id();
  };
  auto const same_device = [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() == rhs.get_gpu_space().get_device_id();
  };
  // Ensure no duplicated device ids
  std::sort(_replica_spaces.begin(), _replica_spaces.end(), device_less);
  _replica_spaces.erase(std::unique(_replica_spaces.begin(), _replica_spaces.end(), same_device),
                        _replica_spaces.end());
}

}  // namespace sirius::op
