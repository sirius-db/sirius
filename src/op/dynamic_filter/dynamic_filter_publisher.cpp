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
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op {

namespace {
// Minimum L2 cache size across every GPU that may probe the filter (0 if any query fails). A single
// filter kind is replicated to all probe devices, so the exact set must fit the smallest cache.
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
  outcome.keys_considered   = admitted_keys.size();

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

  // Build up to 2 complementary filters per admitted key:
  //  1) a zone-map (read-time ROW-GROUP pruning, the only path that cuts scan I/O)
  //  2) a post-decode membership filter: linear-scan IN-list for a tiny build, otherwise a
  //     hash-set IN-list or Bloom filter chosen by L2-cache fit.
  // The two ride different consumer paths and compose; either may be absent for a key. All three
  // vectors are indexed by admitted-key index.
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_zone_map(admitted_keys.size());
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_membership(admitted_keys.size());
  std::vector<cudf::data_type> per_key_build_type(admitted_keys.size(),
                                                  cudf::data_type{cudf::type_id::EMPTY});

  // Read a numeric scalar to host as a double for the zone-map range-coverage gate. Returns nullopt
  // for non-numeric keys (the gate is then skipped and the zone-map is published).
  auto scalar_to_double = [stream](cudf::scalar const& s) -> std::optional<double> {
    switch (s.type().id()) {
      case cudf::type_id::INT8:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::int8_t> const&>(s).value(stream));
      case cudf::type_id::INT16:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::int16_t> const&>(s).value(stream));
      case cudf::type_id::INT32:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::int32_t> const&>(s).value(stream));
      case cudf::type_id::INT64:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::int64_t> const&>(s).value(stream));
      case cudf::type_id::UINT8:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::uint8_t> const&>(s).value(stream));
      case cudf::type_id::UINT16:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::uint16_t> const&>(s).value(stream));
      case cudf::type_id::UINT32:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::uint32_t> const&>(s).value(stream));
      case cudf::type_id::UINT64:
        return static_cast<double>(
          static_cast<cudf::numeric_scalar<std::uint64_t> const&>(s).value(stream));
      default: return std::nullopt;
    }
  };

  for (std::size_t admitted_key_index = 0; admitted_key_index < admitted_keys.size();
       ++admitted_key_index) {
    auto const& admitted_key = admitted_keys[admitted_key_index];

    // Skip domain-covering keys before paying to build a membership structure; their filters keep
    // most probe rows, and the consumer-side gate remains the runtime backstop.
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

    // Validate the plan/runtime key mapping in every build: a silently drifted ordinal or type
    // could construct a filter from the wrong same-typed column and remove valid probe rows, so
    // an inconsistency fails this publication attempt loudly (the caller records FAILED) instead
    // of passing as a successful filter. Unreachable for consistent planner output.
    if (admitted_key.build_key_ordinal >= build_view.num_columns()) {
      throw std::logic_error(
        "[publish_dynamic_filters] An admitted key's build ordinal lies outside the runtime build "
        "table");
    }
    auto const& col = build_view.column(admitted_key.build_key_ordinal);
    if (col.type() != admitted_key.storage_type) {
      // Plan-time and runtime type derivation disagree. Skip the key rather than fail the query:
      // dynamic filters are advisory, the join remains authoritative, and this check cannot
      // detect the wrong-column case that would actually remove valid rows.
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

    // (1) Zone-map -- read-time row-group pruning. This only helps when build keys are
    // correlatively clustered with the filter column(s), so it is off by default (TPC-H keys are
    // scattered).
    if (plan.emit_zone_map_filters()) {
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
        // Range-coverage publication gate (the zone-map analogue of the cardinality gate above):
        // a [min,max] spanning most of the build key's domain prunes nothing, so skip it.
        // Inactive until base-column value-range evidence exists: the key's domain evidence is a
        // row count, and dividing this gate's value span by it would over-fire on sparse integer
        // keys, so the gate receives a domain of 0 and never fires.
        auto const zone_map_range_domain = std::size_t{0};
        bool publish_zone_map            = true;
        if (zone_map_range_domain > 0) {
          auto const lo = scalar_to_double(*min_s);
          auto const hi = scalar_to_double(*max_s);
          if (lo && hi) {
            auto const coverage = (*hi - *lo + 1.0) / static_cast<double>(zone_map_range_domain);
            if (zone_map_range_gate_fires(
                  *lo, *hi, zone_map_range_domain, plan.domain_coverage_threshold())) {
              SIRIUS_LOG_DEBUG(
                "[sirius_physical_hash_join] zone-map key {}: skipped (range [{},{}] covers {:.2f} "
                "of key domain ~{}).",
                admitted_key_index,
                *lo,
                *hi,
                coverage,
                zone_map_range_domain);
              publish_zone_map = false;
            }
          }
        }
        if (publish_zone_map) {
          std::vector<sirius::op::zone_map_entry> zones;
          zones.push_back({std::move(min_s), std::move(max_s)});
          per_key_zone_map[admitted_key_index] =
            std::make_shared<sirius::op::sirius_dynamic_zone_map_filter>(
              std::move(zones), true, true);
        }
      }
    }

    // (2) Membership filter -- post-decode. Prefer, in order:
    //  - A. the exact IN-list with a brute-force scan for a very small key set (no hash build);
    //  - B. the hash-based IN-list when its cuco set fits the device L2 cache;
    //  - C. otherwise the Bloom filter whenever the key type supports it;
    // `none` only when the key type has no membership support (anything other than INT32/INT64).
    auto const set_bytes =
      sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(build_rows, col.type());
    auto const bloom_bytes = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(build_rows);

    // Decide, then construct: the choice depends only on counts, sizes, and capability answers,
    // so it lives in dynamic_filter_source_policy.hpp and is testable without a device.
    auto const chosen = choose_membership_filter(
      {.build_rows               = build_rows,
       .l2_cache_bytes           = l2_bytes,
       .estimated_hash_set_bytes = set_bytes,
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
      "in_list_set={}B bloom={}B L2={}B -> {}",
      admitted_key_index,
      build_rows,
      per_key_zone_map[admitted_key_index] ? "yes" : "no",
      set_bytes,
      bloom_bytes,
      l2_bytes,
      choice);
  }

  // Publish is cross-stream: consumers probe these structures from their own task streams the
  // moment push_filter lands, with no event ordering back to local producer `stream`.
  auto const built = [](auto const& f) { return static_cast<bool>(f); };
  if (std::any_of(per_key_membership.begin(), per_key_membership.end(), built) ||
      std::any_of(per_key_zone_map.begin(), per_key_zone_map.end(), built)) {
    stream.synchronize();

    // Build each structure only on the producer. Remote GPUs receive raw needles, finished
    // static-set slots, Bloom words, or exact zone bounds. Replication completes before the filter
    // is published, so consumers never wait and never observe a cross-device pointer.
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

  // Fan out across probe targets, sparsely: each target receives exactly its bound keys' filters
  // at its own channel push ordinals.
  std::size_t total_pushed   = 0;
  std::size_t active_targets = 0;
  for (auto const& tgt : probe_targets) {
    if (!target_accepts_filters(tgt)) { continue; }
    ++active_targets;
    ++outcome.active_targets;

    for (auto const& binding : tgt.key_bindings) {
      assert(binding.admitted_key_index < admitted_keys.size());  // plan-constructor invariant
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
    "of {} wired target(s) ({} build rows, {} keys).",
    total_pushed,
    active_targets,
    probe_targets.size(),
    build_view.num_rows(),
    admitted_keys.size());
  outcome.filters_pushed = total_pushed;
  return outcome;
}

}  // namespace sirius::op
