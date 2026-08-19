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

#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include "log/logging.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius::op {

dynamic_filter_publish_plan::dynamic_filter_publish_plan(
  std::vector<admitted_key> admitted_keys,
  std::vector<probe_target> probe_targets,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  dynamic_filter_publication_policy policy)
  : _admitted_keys(std::move(admitted_keys)),
    _probe_targets(std::move(probe_targets)),
    _policy(policy),
    _replica_spaces(std::move(replica_spaces))
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
  std::sort(_replica_spaces.begin(), _replica_spaces.end(), device_less);
  _replica_spaces.erase(std::unique(_replica_spaces.begin(), _replica_spaces.end(), same_device),
                        _replica_spaces.end());

  std::vector<std::size_t> condition_indexes;
  condition_indexes.reserve(_admitted_keys.size());
  for (auto const& key : _admitted_keys) {
    if (key.build_key_ordinal < 0) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] An admitted key has a negative build ordinal (upstream "
        "index-conversion defect)");
    }
    if (key.storage_type.id() == cudf::type_id::EMPTY) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] An admitted key has an EMPTY storage type (admission must "
        "not admit a key whose type it cannot represent)");
    }
    condition_indexes.push_back(key.planner_condition_index);
  }
  std::ranges::sort(condition_indexes);
  if (std::ranges::adjacent_find(condition_indexes) != condition_indexes.end()) {
    throw std::invalid_argument(
      "[dynamic_filter_publish_plan] Admitted keys must name distinct planner conditions");
  }

  for (auto const& target : _probe_targets) {
    if (!target.filter_set) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] A probe target requires an endpoint channel");
    }
    if (target.route_class == dynamic_filter_route_class::direct &&
        target.accepts_zone_map_filters) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] A join-edge endpoint is membership-only and cannot accept "
        "zone-map filters");
    }
    std::vector<std::size_t> bound_keys;
    bound_keys.reserve(target.key_bindings.size());
    for (auto const& binding : target.key_bindings) {
      if (binding.admitted_key_index >= _admitted_keys.size()) {
        throw std::invalid_argument(
          "[dynamic_filter_publish_plan] A key binding references an admitted key that does not "
          "exist");
      }
      if (target.route_class == dynamic_filter_route_class::direct) {
        auto const& key = _admitted_keys[binding.admitted_key_index];
        if ((binding.probe_storage_type.id() != cudf::type_id::INT32 &&
             binding.probe_storage_type.id() != cudf::type_id::INT64) ||
            binding.probe_storage_type != key.storage_type) {
          throw std::invalid_argument(
            "[dynamic_filter_publish_plan] A join-edge endpoint binding requires an INT32/INT64 "
            "probe storage type equal to the admitted key's build storage type");
        }
      }
      bound_keys.push_back(binding.admitted_key_index);
    }
    // Channel ordinals may repeat, but each admitted key may bind once per target.
    std::ranges::sort(bound_keys);
    if (std::ranges::adjacent_find(bound_keys) != bound_keys.end()) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] A probe target may bind each admitted key at most once");
    }
  }
}

void dynamic_filter_publish_plan::restrict_replicas_to(std::vector<int> const& admitted_gpu_ids)
{
  if (admitted_gpu_ids.empty()) { return; }
  std::erase_if(_replica_spaces, [&](dynamic_filter_replica_space const& target) {
    auto const gpu_id = target.get_gpu_space().get_device_id();
    return std::find(admitted_gpu_ids.begin(), admitted_gpu_ids.end(), gpu_id) ==
           admitted_gpu_ids.end();
  });
  if (_replica_spaces.empty() && !_probe_targets.empty()) {
    SIRIUS_LOG_WARN(
      "[dynamic_filter_publish_plan] The admitted GPU set holds none of this plan's replica "
      "GPUs; disabling dynamic-filter publication for this join ({} probe target(s) dropped).",
      _probe_targets.size());
    _probe_targets.clear();
  }
}

bool dynamic_filter_publish_plan::has_replica_on_device(int gpu_device_id) const noexcept
{
  return std::ranges::any_of(_replica_spaces, [gpu_device_id](auto const& target) {
    return target.get_gpu_space().get_device_id() == gpu_device_id;
  });
}

}  // namespace sirius::op
