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

#include "op/dynamic_filter/dynamic_filter_publisher.hpp"

#include "log/logging.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op {

namespace {
// Size exact filters for the smallest probe-device L2; return 0 if unavailable.
std::size_t device_l2_cache_bytes(
  std::span<dynamic_filter_replica_space const> replica_spaces) noexcept
{
  if (replica_spaces.empty()) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) { return 0; }
    int l2 = 0;
    return cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, current) == cudaSuccess && l2 > 0
             ? static_cast<std::size_t>(l2)
             : 0;
  }

  std::size_t minimum = std::numeric_limits<std::size_t>::max();
  for (auto const& target : replica_spaces) {
    auto const device_id = target.get_gpu_space().get_device_id();
    int l2               = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device_id) != cudaSuccess || l2 <= 0) {
      return 0;
    }
    minimum = std::min(minimum, static_cast<std::size_t>(l2));
  }
  return minimum == std::numeric_limits<std::size_t>::max() ? 0 : minimum;
}
}  // namespace

dynamic_filter_publication_outcome publish_dynamic_filters(dynamic_filter_publish_plan const& plan,
                                                           cudf::table_view const& build_view,
                                                           rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::push_build_side"};
  assert(plan.enabled());
  dynamic_filter_publication_outcome outcome;

  if (build_view.num_rows() == 0) {
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] Skipping dynamic filter push: empty build table.");
    return outcome;
  }

  auto target_accepts_filters = [](dynamic_filter_publish_plan::probe_target const& tgt) {
    return tgt.filter_set && tgt.filter_set->accepting_filters();
  };
  auto const& probe_targets = plan.probe_targets();
  if (std::none_of(probe_targets.begin(), probe_targets.end(), target_accepts_filters)) {
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] Skipping dynamic filter push: all target scans drained.");
    outcome.skipped_targets_drained = 1;
    return outcome;
  }

  auto const& admitted_keys = plan.admitted_keys();

  std::vector<char> key_bound(admitted_keys.size(), 0);
  for (auto const& target : probe_targets) {
    for (auto const& binding : target.key_bindings) {
      key_bound[binding.admitted_key_index] = 1;
    }
  }

  int source_device = -1;
  if (cudaGetDevice(&source_device) != cudaSuccess) {
    throw std::runtime_error(
      "[publish_dynamic_filters] Dynamic-filter publisher could not identify its source GPU");
  }
  auto const source_space =
    std::find_if(plan.replica_spaces().begin(),
                 plan.replica_spaces().end(),
                 [source_device](auto const& target) {
                   return target.get_gpu_space().get_device_id() == source_device;
                 });
  if (source_space == plan.replica_spaces().end()) {
    throw std::logic_error(
      "[publish_dynamic_filters] Dynamic-filter source GPU is absent from the immutable publish "
      "plan");
  }
  auto const allocator_ref = source_space->get_gpu_space().get_default_allocator();
  auto const build_rows    = static_cast<std::size_t>(build_view.num_rows());
  auto const l2_bytes      = device_l2_cache_bytes(plan.replica_spaces());

  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_zone_map(admitted_keys.size());
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_membership(admitted_keys.size());
  std::vector<cudf::data_type> per_key_build_type(admitted_keys.size(),
                                                  cudf::data_type{cudf::type_id::EMPTY});

  for (std::size_t admitted_key_index = 0; admitted_key_index < admitted_keys.size();
       ++admitted_key_index) {
    if (key_bound[admitted_key_index] == 0) { continue; }
    ++outcome.keys_considered;
    auto const& admitted_key = admitted_keys[admitted_key_index];

    auto const key_domain = admitted_key.build_key_domain_cardinality;
    if (key_domain > 0) {
      ++outcome.keys_with_known_domain;
      if (build_rows > key_domain) { ++outcome.keys_build_exceeded_domain; }
    }
    if (domain_coverage_gate_fires(build_rows,
                                   key_domain,
                                   admitted_key.build_key_proven_unique,
                                   plan.domain_coverage_threshold())) {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] publish gate: key {}: build {} rows cover {:.2f} of key "
        "domain (~{} rows) -> skip key.",
        admitted_key_index,
        build_view.num_rows(),
        static_cast<double>(build_rows) / static_cast<double>(key_domain),
        key_domain);
      ++outcome.keys_skipped_domain_gate;
      continue;
    }

    if (admitted_key.build_key_ordinal >= build_view.num_columns()) {
      throw std::logic_error(
        "[publish_dynamic_filters] An admitted key's build ordinal lies outside the runtime build "
        "table");
    }
    auto const& col = build_view.column(admitted_key.build_key_ordinal);
    if (col.type() != admitted_key.storage_type) {
      SIRIUS_LOG_WARN(
        "[sirius_physical_hash_join] dynamic filter key {}: skipped (plan recorded type id {} but "
        "build column {} carries type id {}).",
        admitted_key_index,
        static_cast<int32_t>(admitted_key.storage_type.id()),
        admitted_key.build_key_ordinal,
        static_cast<int32_t>(col.type().id()));
      ++outcome.keys_skipped_type_mismatch;
      continue;
    }
    per_key_build_type[admitted_key_index] = col.type();

    if (plan.emit_zone_map_filters() &&
        sirius::op::sirius_dynamic_zone_map_filter::supports(col.type())) {
      nvtx3::scoped_range vr{"dynfilter::build_zone_map"};
      auto min_s = cudf::reduce(col,
                                *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
                                col.type(),
                                stream,
                                allocator_ref);
      auto max_s = cudf::reduce(col,
                                *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
                                col.type(),
                                stream,
                                allocator_ref);
      if (min_s && max_s && min_s->is_valid(stream) && max_s->is_valid(stream)) {
        std::vector<sirius::op::zone_map_entry> zones;
        zones.push_back({std::move(min_s), std::move(max_s)});
        per_key_zone_map[admitted_key_index] =
          std::make_shared<sirius::op::sirius_dynamic_zone_map_filter>(
            std::move(zones), true, true);
      }
    }

    auto const set_bytes =
      sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(build_rows, col.type());
    auto const bloom_bytes = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(build_rows);

    auto const chosen = choose_membership_filter(
      {.build_rows               = build_rows,
       .l2_cache_bytes           = l2_bytes,
       .estimated_hash_set_bytes = set_bytes,
       .inlist_max_l2_fraction   = plan.inlist_max_l2_fraction(),
       .supports_small_in_list   = sirius::op::sirius_dynamic_small_in_list_filter::supports(col),
       .supports_hash_in_list    = sirius::op::sirius_dynamic_in_list_filter::supports(col),
       .supports_bloom           = sirius::op::sirius_dynamic_bloom_filter::supports(col.type())});

    char const* choice = "none";
    switch (chosen) {
      case membership_filter_kind::small_in_list: {
        nvtx3::scoped_range vr{"dynfilter::build_small_in_list"};
        per_key_membership[admitted_key_index] =
          std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
            col, stream, allocator_ref);
        choice = "small_in_list";
        break;
      }
      case membership_filter_kind::hash_in_list: {
        nvtx3::scoped_range vr{"dynfilter::build_in_list"};
        per_key_membership[admitted_key_index] =
          std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(col, stream, allocator_ref);
        choice = "in_list";
        break;
      }
      case membership_filter_kind::bloom: {
        nvtx3::scoped_range vr{"dynfilter::build_bloom"};
        per_key_membership[admitted_key_index] =
          std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(col, stream, allocator_ref);
        choice = "bloom";
        break;
      }
      case membership_filter_kind::none: break;
    }
    if (per_key_membership[admitted_key_index]) { ++outcome.membership_filters_built; }
    if (per_key_zone_map[admitted_key_index]) { ++outcome.zone_map_filters_built; }
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] dynamic filter key {}: build_rows={} zone_map={} membership: "
      "in_list_set={}B bloom={}B L2={}B inlist_max_l2_fraction={} -> {}",
      admitted_key_index,
      build_rows,
      per_key_zone_map[admitted_key_index] ? "yes" : "no",
      set_bytes,
      bloom_bytes,
      l2_bytes,
      plan.inlist_max_l2_fraction(),
      choice);
  }

  // Finish construction and replication before publishing to independent consumer streams.
  auto const built = [](auto const& f) { return static_cast<bool>(f); };
  if (std::any_of(per_key_membership.begin(), per_key_membership.end(), built) ||
      std::any_of(per_key_zone_map.begin(), per_key_zone_map.end(), built)) {
    stream.synchronize();

    nvtx3::scoped_range replicate_range{"dynfilter::replicate_devices"};
    auto replicate = [&plan](std::shared_ptr<sirius_dynamic_filter> const& filter) {
      if (!filter) { return; }
      auto* replicable = dynamic_cast<sirius_device_replicable*>(filter.get());
      if (replicable == nullptr) {
        throw std::logic_error(
          "[publish_dynamic_filters] A published device-backed dynamic filter must implement "
          "sirius_device_replicable");
      }
      replicable->replicate_to_devices(plan.replica_spaces());
    };
    for (auto const& filter : per_key_zone_map) {
      replicate(filter);
    }
    for (auto const& filter : per_key_membership) {
      replicate(filter);
    }
  }

  std::size_t total_pushed   = 0;
  std::size_t active_targets = 0;
  for (auto const& tgt : probe_targets) {
    if (!target_accepts_filters(tgt)) { continue; }
    ++active_targets;
    ++outcome.active_targets;

    for (auto const& binding : tgt.key_bindings) {
      assert(binding.admitted_key_index < admitted_keys.size());
      auto const& zone_map = per_key_zone_map[binding.admitted_key_index];
      if (zone_map && tgt.accepts_zone_map_filters &&
          binding.probe_storage_type == per_key_build_type[binding.admitted_key_index] &&
          tgt.filter_set->push_filter(binding.channel_push_ordinal, zone_map)) {
        ++total_pushed;
      }
      auto const& membership = per_key_membership[binding.admitted_key_index];
      if (membership && tgt.filter_set->push_filter(binding.channel_push_ordinal, membership)) {
        ++total_pushed;
      }
    }
  }
  SIRIUS_LOG_INFO(
    "[sirius_physical_hash_join] Pushed {} dynamic filter(s) across {} active target(s) "
    "of {} wired target(s) ({} build rows, {} bound keys of {} admitted).",
    total_pushed,
    active_targets,
    probe_targets.size(),
    build_view.num_rows(),
    outcome.keys_considered,
    admitted_keys.size());
  outcome.filters_pushed = total_pushed;
  return outcome;
}

}  // namespace sirius::op
