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

// cucascade
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
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
  // Ensure no duplicated device ids
  std::sort(_replica_spaces.begin(), _replica_spaces.end(), device_less);
  _replica_spaces.erase(std::unique(_replica_spaces.begin(), _replica_spaces.end(), same_device),
                        _replica_spaces.end());
}

dynamic_filter_publish_plan_builder::dynamic_filter_publish_plan_builder(
  dynamic_filter_publication_plan_id publication_plan_id,
  bool wired,
  std::vector<scan_target_draft> scan_targets,
  bool emit_zone_map_filters,
  double domain_coverage_threshold,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  std::vector<dynamic_filter_key_candidate> key_candidates)
  : _publication_plan_id(publication_plan_id),
    _wired(wired),
    _scan_targets(std::move(scan_targets)),
    _emit_zone_map_filters(emit_zone_map_filters),
    _domain_coverage_threshold(domain_coverage_threshold),
    _replica_spaces(std::move(replica_spaces)),
    _key_candidates(std::move(key_candidates))
{
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
  _decisions                = std::move(decisions);
  _resolved_keys            = std::move(resolved_keys);
  _build_input_column_count = build_input_column_count;
  _keys_resolved            = true;
}

dynamic_filter_planning_view dynamic_filter_publish_plan_builder::planning_view() const
{
  if (!_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] planning_view before resolve_keys");
  }
  if (_planning_view_storage.empty() && !_key_candidates.empty()) {
    _planning_view_storage.reserve(_key_candidates.size());
    std::size_t next_key = 0;
    for (std::size_t j = 0; j < _key_candidates.size(); ++j) {
      dynamic_filter_planning_ordinal_view v{};
      v.duckdb_ordinal = _key_candidates[j].duckdb_ordinal;
      v.decision = j < _decisions.size() ? _decisions[j] : dynamic_filter_key_decision::unresolved;
      if (v.decision == dynamic_filter_key_decision::admitted && next_key < _resolved_keys.size()) {
        v.admitted_key = _resolved_keys[next_key];
        v.build_type   = _resolved_keys[next_key].build_type;
        ++next_key;
      }
      _planning_view_storage.push_back(v);
    }
  }
  dynamic_filter_planning_view view{};
  view.publication_plan_id = _publication_plan_id;
  view.wired               = _wired;
  view.enabled             = !_scan_targets.empty() && !_resolved_keys.empty();
  view.by_duckdb_ordinal   = _planning_view_storage;
  return view;
}

std::shared_ptr<dynamic_filter_publish_plan const> dynamic_filter_publish_plan_builder::finalize()
  const
{
  auto const fail = [this](char const* what) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] publication plan {} failed final validation: {}",
      _publication_plan_id.value,
      what);
  };

  if (!_keys_resolved) { fail("keys were never resolved"); }

  // 1. decision count equals candidate count (identity match is positional by construction).
  if (_decisions.size() != _key_candidates.size()) { fail("decision count != candidate count"); }
  // 2./4. every admitted decision has exactly one key; key count == admitted-decision count.
  auto const admitted_count = static_cast<std::size_t>(
    std::count(_decisions.begin(), _decisions.end(), dynamic_filter_key_decision::admitted));
  if (_resolved_keys.size() != admitted_count) {
    fail("resolved key count != admitted decision count");
  }
  // 3. rejected decisions have no Sirius ordinal — enforced structurally: keys exist only in
  //    _resolved_keys, and (5.) their compact ordinals must be unique and contiguous from zero,
  //    in candidate order.
  {
    auto next_key = std::size_t{0};
    for (std::size_t j = 0; j < _decisions.size(); ++j) {
      if (_decisions[j] != dynamic_filter_key_decision::admitted) { continue; }
      auto const& key = _resolved_keys[next_key];
      if (key.ordinal.value != next_key) {
        fail("Sirius key ordinals are not contiguous in candidate order");
      }
      // 6./7. provenance bijection: the key names exactly its candidate's ordinal spaces.
      if (key.duckdb_ordinal != _key_candidates[j].duckdb_ordinal ||
          key.condition_index != _key_candidates[j].condition_index) {
        fail("admitted key does not match its candidate's ordinals");
      }
      // 8. each build column is inside the captured build input width.
      if (key.build_column_index >= _build_input_column_count) {
        fail("admitted key's build column exceeds the build input width");
      }
      ++next_key;
    }
  }
  // 6. DuckDB ordinals unique and below key count (candidate side).
  {
    std::vector<bool> seen(_key_candidates.size(), false);
    for (auto const& cand : _key_candidates) {
      auto const j = static_cast<std::size_t>(cand.duckdb_ordinal.value);
      if (j >= _key_candidates.size() || seen[j]) {
        fail("duplicate or out-of-range DuckDB ordinal");
      }
      seen[j] = true;
    }
  }
  // 9. full-arity scan target column/type vectors equal DuckDB key count.
  for (auto const& target : _scan_targets) {
    if (target.probe_col_idx.size() != _key_candidates.size() ||
        target.probe_col_type.size() != _key_candidates.size()) {
      fail("scan target arity != DuckDB key count");
    }
  }
  // 10. enabled plans: nonzero IDs, unique target IDs, non-null channels.
  bool const enabled = !_scan_targets.empty() && !_resolved_keys.empty();
  if (!_publication_plan_id.is_valid()) { fail("publication plan ID is zero"); }
  {
    std::vector<dynamic_filter_target_id> ids;
    for (auto const& target : _scan_targets) {
      if (!target.target_id.is_valid() || !target.channel_id.is_valid()) {
        fail("target or channel ID is zero");
      }
      if (target.channel == nullptr) { fail("scan target has a null channel"); }
      ids.push_back(target.target_id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) { fail("duplicate target IDs"); }
  }
  // 11. replica spaces: reuse the existing ctor validation (GPU/HOST tier + unique devices) by
  //     construction below — dynamic_filter_publish_plan's constructor still enforces it.
  // 12. disabled plans contain no live target but are still built and installed.

  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  if (enabled) {
    targets.reserve(_scan_targets.size());
    for (auto const& draft : _scan_targets) {
      targets.push_back(dynamic_filter_publish_plan::probe_target{
        draft.channel, draft.probe_col_idx, draft.probe_col_type});
    }
  }
  // Domain evidence is null in C1a-2: all-zero cardinalities keep the coverage gates exactly off,
  // byte-for-byte matching the dead pre-C1a-2 walk's runtime effect.
  return std::make_shared<dynamic_filter_publish_plan const>(
    std::move(targets),
    _emit_zone_map_filters,
    std::vector<std::size_t>(_key_candidates.size(), 0),
    _replica_spaces,
    _domain_coverage_threshold);
}

}  // namespace sirius::op
