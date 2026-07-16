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

// sirius
#include <op/dynamic_filter_publish_plan.hpp>
#include <op/dynamic_filter_publish_plan_builder.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/assert.hpp>

// cucascade
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <stdexcept>
#ifdef D_ASSERT_IS_ENABLED
#include <unordered_set>
#endif
#include <utility>

namespace sirius::op {

namespace {

#ifdef D_ASSERT_IS_ENABLED
template <class T>
bool all_unique(std::vector<T> values)
{
  std::sort(values.begin(), values.end());
  return std::adjacent_find(values.begin(), values.end()) == values.end();
}
#endif

}  // namespace

dynamic_filter_publish_plan::dynamic_filter_publish_plan(
  std::vector<probe_target> probe_targets,
  bool emit_zone_map_filters,
  std::vector<std::size_t> build_key_domain_cardinalities,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  double domain_coverage_threshold)
  : _probe_targets(std::move(probe_targets)),
    _emit_zone_map_filters(emit_zone_map_filters),
    _build_key_domain_cardinalities(std::move(build_key_domain_cardinalities)),
    _domain_coverage_threshold(domain_coverage_threshold),
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
  // Keep at most one replica placement per GPU device.
  std::sort(_replica_spaces.begin(), _replica_spaces.end(), device_less);
  _replica_spaces.erase(std::unique(_replica_spaces.begin(), _replica_spaces.end(), same_device),
                        _replica_spaces.end());
}

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan_builder
//===----------------------------------------------------------------------===//

dynamic_filter_publish_plan_builder::dynamic_filter_publish_plan_builder(descriptor draft)
  : _publication_plan_id(draft.publication_plan_id),
    _build_subtree_has_filter_hint(draft.build_subtree_has_filter_hint),
    _emit_zone_map_filters(draft.emit_zone_map_filters),
    _domain_coverage_threshold(draft.domain_coverage_threshold),
    _replica_spaces(std::move(draft.replica_spaces)),
    _key_candidates(std::move(draft.key_candidates))
{
#ifdef D_ASSERT_IS_ENABLED
  D_ASSERT(_domain_coverage_threshold > 0.0);
  D_ASSERT(_publication_plan_id.is_valid());
  D_ASSERT(!_key_candidates.empty());
  D_ASSERT(draft.join_condition_count > 0);
  std::vector<std::size_t> condition_indexes;
  for (std::size_t j = 0; j < _key_candidates.size(); ++j) {
    D_ASSERT(_key_candidates[j].duckdb_ordinal.value == j);
    D_ASSERT(_key_candidates[j].condition_index.value < draft.join_condition_count);
    condition_indexes.push_back(_key_candidates[j].condition_index.value);
  }
  D_ASSERT(all_unique(std::move(condition_indexes)));
  std::vector<std::uint32_t> target_ids;
  std::unordered_set<sirius_dynamic_filter_set const*> channel_objects;
  for (auto const& target : draft.scan_targets) {
    D_ASSERT(target.target_id.is_valid());
    D_ASSERT(target.channel_id.is_valid());
    D_ASSERT(target.channel != nullptr);
    D_ASSERT(target.probe_col_idx.size() == _key_candidates.size());
    D_ASSERT(target.probe_col_type.size() == _key_candidates.size());
    auto const distinct_channel = channel_objects.insert(target.channel.get()).second;
    D_ASSERT(distinct_channel);
    target_ids.push_back(target.target_id.value());
  }
  D_ASSERT(all_unique(std::move(target_ids)));
  std::vector<int> replica_devices;
  for (auto const& replica : _replica_spaces) {
    D_ASSERT(replica.get_gpu_space().get_tier() == cucascade::memory::Tier::GPU);
    D_ASSERT(replica.get_host_staging_space().get_tier() == cucascade::memory::Tier::HOST);
    replica_devices.push_back(replica.get_gpu_space().get_device_id());
  }
  D_ASSERT(all_unique(std::move(replica_devices)));
#endif

  _scan_targets.reserve(draft.scan_targets.size());
  _scan_target_channels.reserve(draft.scan_targets.size());
  for (auto& target : draft.scan_targets) {
    _scan_targets.push_back(dynamic_filter_planning_scan_target{target.target_id,
                                                                target.channel_id,
                                                                std::move(target.probe_col_idx),
                                                                std::move(target.probe_col_type)});
    _scan_target_channels.push_back(std::move(target.channel));
  }
}

void dynamic_filter_publish_plan_builder::resolve_keys(
  std::vector<dynamic_filter_key_decision> decisions,
  std::vector<dynamic_filter_key_plan> resolved_keys,
  std::size_t build_input_column_count)
{
  if (_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] resolve_keys called twice on one builder");
  }

#ifdef D_ASSERT_IS_ENABLED
  D_ASSERT(decisions.size() == _key_candidates.size());
  std::size_t admitted = 0;
  for (std::size_t j = 0; j < decisions.size(); ++j) {
    if (decisions[j] != dynamic_filter_key_decision::admitted) { continue; }
    auto const& candidate = _key_candidates[j];
    D_ASSERT(candidate.is_equality);
    D_ASSERT(candidate.has_direct_uncast_keys);
    D_ASSERT(candidate.has_supported_membership_type);
    D_ASSERT(!_scan_targets.empty());
    D_ASSERT(admitted < resolved_keys.size());
    auto const& key = resolved_keys[admitted];
    D_ASSERT(key.ordinal.value == admitted);
    D_ASSERT(key.duckdb_ordinal == candidate.duckdb_ordinal);
    D_ASSERT(key.condition_index == candidate.condition_index);
    D_ASSERT(key.build_batch_column_index < build_input_column_count);
    ++admitted;
  }
  D_ASSERT(admitted == resolved_keys.size());
#endif

  // Built into a local first, so planning_view() can never observe a partially resolved builder.
  std::vector<dynamic_filter_planning_ordinal_view> ordinal_records;
  ordinal_records.reserve(_key_candidates.size());
  std::size_t next_key = 0;
  for (std::size_t j = 0; j < _key_candidates.size(); ++j) {
    dynamic_filter_planning_ordinal_view v{};
    v.duckdb_ordinal  = _key_candidates[j].duckdb_ordinal;
    v.condition_index = _key_candidates[j].condition_index;
    v.decision        = decisions[j];
    if (v.decision == dynamic_filter_key_decision::admitted) {
      v.admitted_key = resolved_keys[next_key];
      ++next_key;
    }
    ordinal_records.push_back(v);
  }

  // Commit; nothing below throws.
  _decisions                = std::move(decisions);
  _resolved_keys            = std::move(resolved_keys);
  _build_input_column_count = build_input_column_count;
  _ordinal_records          = std::move(ordinal_records);
  _keys_resolved            = true;
}

dynamic_filter_planning_view dynamic_filter_publish_plan_builder::planning_view() const
{
  if (!_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] planning_view before resolve_keys");
  }
  dynamic_filter_planning_view view{};
  view.publication_plan_id           = _publication_plan_id;
  view.build_subtree_has_filter_hint = _build_subtree_has_filter_hint;
  view.enabled =
    !_scan_targets.empty() && std::any_of(_decisions.begin(), _decisions.end(), [](auto decision) {
      return decision == dynamic_filter_key_decision::admitted;
    });
  view.scan_targets      = _scan_targets;
  view.by_duckdb_ordinal = ordinal_records();
  return view;
}

}  // namespace sirius::op
