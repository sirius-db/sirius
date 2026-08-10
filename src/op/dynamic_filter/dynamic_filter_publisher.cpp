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
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
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

  // Publication constructs a filter only for a key some target binds. Admission records legality
  // beyond consumption -- a legal key for which the discovery walk bound no target (scan or
  // join-edge) stays in the array as a fact -- and such a key must cost no filter construction.
  // Binding indexes are constructor-validated, so no bounds check is needed here.
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
  auto const bloom_bytes   = sirius_dynamic_bloom_filter::estimated_bytes(build_rows);

  // Build up to two complementary filters per admitted key:
  //  1) a zone-map (read-time ROW-GROUP pruning, the only path that cuts scan I/O)
  //  2) a post-decode membership filter: linear-scan IN-list for a tiny build, otherwise a
  //     hash-set IN-list or Bloom filter chosen by L2-cache fit.
  // The two ride different consumer paths and compose; either may be absent for a key. All three
  // vectors are indexed by admitted-key index.
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_zone_map(admitted_keys.size());
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_membership(admitted_keys.size());
  std::vector<char> per_key_bloom_candidate(admitted_keys.size(), 0);
  std::vector<cudf::data_type> per_key_build_type(admitted_keys.size(),
                                                  cudf::data_type{cudf::type_id::EMPTY});

  for (std::size_t admitted_key_index = 0; admitted_key_index < admitted_keys.size();
       ++admitted_key_index) {
    if (key_bound[admitted_key_index] == 0) { continue; }
    ++outcome.keys_considered;
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

    // An out-of-range ordinal is a plan invariant failure. A type mismatch remains an advisory
    // per-key skip because the authoritative join still checks the result.
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

    // Zone maps can prune Parquet row groups or apply as row masks on native scans.
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
        // Publish one zone map for every valid min/max reduction. The range gate is inactive
        // because no value-range domain is available.
        std::vector<sirius::op::zone_map_entry> zones;
        zones.push_back({std::move(min_s), std::move(max_s)});
        per_key_zone_map[admitted_key_index] =
          std::make_shared<sirius::op::sirius_dynamic_zone_map_filter>(
            std::move(zones), true, true);
      }
    }

    // Membership filters apply after decode. choose_membership_filter() prefers
    // sirius_dynamic_small_in_list_filter for a supported small build,
    // sirius_dynamic_in_list_filter when its set fits L2, and sirius_dynamic_bloom_filter for the
    // remaining supported keys. It returns none when no representation supports the key.
    auto const set_bytes =
      sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(build_rows, col.type());

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
        per_key_bloom_candidate[admitted_key_index] = 1;
        choice                                      = "bloom";
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

  auto const bloom_candidate_count = static_cast<std::size_t>(
    std::count(per_key_bloom_candidate.begin(), per_key_bloom_candidate.end(), 1));
  if (bloom_budget_allows(bloom_bytes, bloom_candidate_count, plan.max_bloom_bytes_per_gpu())) {
    nvtx3::scoped_range vr{"dynfilter::build_bloom"};
    for (std::size_t key_index = 0; key_index < per_key_bloom_candidate.size(); ++key_index) {
      if (per_key_bloom_candidate[key_index] == 0) { continue; }
      auto const& column = build_view.column(admitted_keys[key_index].build_key_ordinal);
      per_key_membership[key_index] =
        std::make_shared<sirius_dynamic_bloom_filter>(column, stream, allocator_ref);
      ++outcome.membership_filters_built;
    }
  } else {
    outcome.keys_skipped_bloom_size_gate += bloom_candidate_count;
    SIRIUS_LOG_INFO(
      "[sirius_physical_hash_join] Skipping {} Bloom filter candidate(s): each would use {} "
      "allocator-accounted bytes against the {}-byte aggregate per-GPU budget.",
      bloom_candidate_count,
      bloom_bytes,
      plan.max_bloom_bytes_per_gpu());
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

struct dynamic_filter_accumulator::impl {
  struct device_partial {
    std::mutex mutex;
    std::vector<std::shared_ptr<sirius_dynamic_bloom_filter>> filters;
  };

  class build_type_mismatch : public std::invalid_argument {
   public:
    build_type_mismatch()
      : std::invalid_argument("a contribution build-key type does not match the plan")
    {
    }
  };

  dynamic_filter_publish_plan const& plan;
  std::size_t build_rows;
  std::unordered_set<std::uint64_t> expected_ids;
  std::unordered_set<std::uint64_t> in_flight_ids;
  std::unordered_set<std::uint64_t> completed_ids;
  std::vector<char> active_keys;
  std::vector<cudf::data_type> build_types;
  std::map<int, std::unique_ptr<device_partial>> partials;
  dynamic_filter_publication_outcome outcome;
  detail::dynamic_filter_accumulator_test_hooks test_hooks;
  mutable std::mutex mutex;
  bool is_complete = false;
  bool is_aborted  = false;

  impl(dynamic_filter_publish_plan const& plan,
       std::size_t build_rows,
       std::vector<std::uint64_t> expected_batch_ids,
       detail::dynamic_filter_accumulator_test_hooks test_hooks)
    : plan(plan),
      build_rows(build_rows),
      expected_ids(expected_batch_ids.begin(), expected_batch_ids.end()),
      active_keys(plan.admitted_keys().size(), 0),
      build_types(plan.admitted_keys().size(), cudf::data_type{cudf::type_id::EMPTY}),
      test_hooks(std::move(test_hooks))
  {
    if (!plan.enabled() || expected_batch_ids.empty() ||
        expected_ids.size() != expected_batch_ids.size()) {
      throw std::invalid_argument(
        "[dynamic_filter_accumulator] enabled plan and unique non-empty expected IDs required.");
    }

    for (auto const& space : plan.replica_spaces()) {
      partials.emplace(space.get_gpu_space().get_device_id(), std::make_unique<device_partial>());
    }

    std::vector<char> key_bound(plan.admitted_keys().size(), 0);
    for (auto const& target : plan.probe_targets()) {
      for (auto const& binding : target.key_bindings) {
        key_bound[binding.admitted_key_index] = 1;
      }
    }
    for (std::size_t key_index = 0; key_index < plan.admitted_keys().size(); ++key_index) {
      if (key_bound[key_index] == 0) { continue; }
      ++outcome.keys_considered;
      auto const& key   = plan.admitted_keys()[key_index];
      auto const domain = key.build_key_domain_cardinality;
      if (domain > 0) {
        ++outcome.keys_with_known_domain;
        if (build_rows > domain) { ++outcome.keys_build_exceeded_domain; }
      }
      if (domain_coverage_gate_fires(
            build_rows, domain, key.build_key_proven_unique, plan.domain_coverage_threshold())) {
        ++outcome.keys_skipped_domain_gate;
        continue;
      }
      if (!sirius_dynamic_bloom_filter::supports(key.storage_type)) { continue; }
      active_keys[key_index] = 1;
      build_types[key_index] = key.storage_type;
    }

    auto const active_bloom_keys =
      static_cast<std::size_t>(std::count(active_keys.begin(), active_keys.end(), 1));
    auto const bloom_bytes = sirius_dynamic_bloom_filter::estimated_bytes(build_rows);
    if (!bloom_budget_allows(bloom_bytes, active_bloom_keys, plan.max_bloom_bytes_per_gpu())) {
      std::fill(active_keys.begin(), active_keys.end(), 0);
      std::fill(build_types.begin(), build_types.end(), cudf::data_type{cudf::type_id::EMPTY});
      outcome.keys_skipped_bloom_size_gate += active_bloom_keys;
      SIRIUS_LOG_INFO(
        "[dynamic_filter_accumulator] Skipping {} global Bloom filter candidate(s): each would "
        "use {} allocator-accounted bytes against the {}-byte aggregate per-GPU budget.",
        active_bloom_keys,
        bloom_bytes,
        plan.max_bloom_bytes_per_gpu());
    }
  }

  [[nodiscard]] dynamic_filter_replica_space const* replica_space(int device_id) const noexcept
  {
    auto const it = std::find_if(
      plan.replica_spaces().begin(), plan.replica_spaces().end(), [device_id](auto const& space) {
        return space.get_gpu_space().get_device_id() == device_id;
      });
    return it == plan.replica_spaces().end() ? nullptr : &*it;
  }

  [[nodiscard]] dynamic_filter_accumulation_result aborted_result() const
  {
    return {.state = dynamic_filter_accumulation_result::status::aborted, .publication = outcome};
  }

  [[nodiscard]] dynamic_filter_accumulation_result abort_locked(std::string_view reason)
  {
    if (!is_aborted && !is_complete) {
      is_aborted = true;
      SIRIUS_LOG_WARN("[dynamic_filter_accumulator] publication aborted: {}", reason);
    }
    return aborted_result();
  }

  static void synchronize_after_failure(rmm::cuda_stream_view stream,
                                        std::string_view operation) noexcept
  {
    try {
      stream.synchronize();
    } catch (std::exception const& error) {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_accumulator] {} failure drain also failed: {}", operation, error.what());
    } catch (...) {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_accumulator] {} failure drain also failed with an unknown error.",
        operation);
    }
  }

  [[nodiscard]] dynamic_filter_accumulation_result publish_locked()
  {
    auto target_accepts_filters = [](dynamic_filter_publish_plan::probe_target const& target) {
      return target.filter_set && target.filter_set->accepting_filters();
    };
    if (std::none_of(
          plan.probe_targets().begin(), plan.probe_targets().end(), target_accepts_filters)) {
      outcome.skipped_targets_drained = 1;
      is_complete                     = true;
      return {.state       = dynamic_filter_accumulation_result::status::published,
              .publication = outcome};
    }

    auto const& root_space = plan.replica_spaces().front();
    auto const root_device = root_space.get_gpu_space().get_device_id();
    rmm::cuda_set_device_raii root_guard{rmm::cuda_device_id{root_device}};
    auto const root_stream = root_space.get_gpu_space().acquire_stream();
    auto& root_filters     = partials.at(root_device)->filters;
    root_filters.resize(active_keys.size());

    try {
      for (std::size_t key_index = 0; key_index < active_keys.size(); ++key_index) {
        if (active_keys[key_index] == 0) { continue; }
        if (!root_filters[key_index]) {
          root_filters[key_index] = std::make_shared<sirius_dynamic_bloom_filter>(
            build_types[key_index],
            build_rows,
            root_stream,
            root_space.get_gpu_space().get_default_allocator());
        }
        for (auto const& [device_id, partial] : partials) {
          if (device_id == root_device || key_index >= partial->filters.size() ||
              !partial->filters[key_index]) {
            continue;
          }
          auto const* source_space = replica_space(device_id);
          if (source_space == nullptr) {
            throw std::logic_error("a contributing GPU is absent from the immutable replica plan");
          }
          root_filters[key_index]->merge_from(
            *partial->filters[key_index], *source_space, root_space, root_stream);
        }
      }
      root_stream.synchronize();
    } catch (...) {
      synchronize_after_failure(root_stream, "root reduction");
      throw;
    }

    for (std::size_t key_index = 0; key_index < active_keys.size(); ++key_index) {
      if (active_keys[key_index] != 0 && root_filters[key_index]) {
        root_filters[key_index]->release_reduction_scratch();
      }
    }
    for (auto& [device_id, partial] : partials) {
      if (device_id != root_device) { partial->filters.clear(); }
    }

    for (std::size_t key_index = 0; key_index < active_keys.size(); ++key_index) {
      if (active_keys[key_index] == 0 || !root_filters[key_index]) { continue; }
      if (test_hooks.strict_replicate) {
        test_hooks.strict_replicate(*root_filters[key_index], plan.replica_spaces());
      } else {
        root_filters[key_index]->replicate_to_devices_strict(plan.replica_spaces());
      }
      ++outcome.membership_filters_built;
    }

    for (auto const& target : plan.probe_targets()) {
      if (!target_accepts_filters(target)) { continue; }
      ++outcome.active_targets;
      for (auto const& binding : target.key_bindings) {
        auto const key_index = binding.admitted_key_index;
        if (key_index >= root_filters.size() || active_keys[key_index] == 0 ||
            !root_filters[key_index]) {
          continue;
        }
        if (target.filter_set->push_filter(binding.channel_push_ordinal, root_filters[key_index])) {
          ++outcome.filters_pushed;
        }
      }
    }

    SIRIUS_LOG_INFO(
      "[dynamic_filter_accumulator] published {} global Bloom filter(s) across {} active "
      "target(s) after {} exact build contribution(s), {} build rows.",
      outcome.membership_filters_built,
      outcome.active_targets,
      completed_ids.size(),
      build_rows);
    is_complete = true;
    return {.state = dynamic_filter_accumulation_result::status::published, .publication = outcome};
  }

  [[nodiscard]] dynamic_filter_accumulation_result contribute(std::uint64_t batch_id,
                                                              cudf::table_view const& build_view,
                                                              rmm::cuda_stream_view stream)
  {
    int device_id = -1;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
      std::scoped_lock coordinator_lock(mutex);
      return abort_locked("could not identify the contribution GPU");
    }

    device_partial* partial = nullptr;
    {
      std::scoped_lock coordinator_lock(mutex);
      if (!expected_ids.contains(batch_id)) {
        return abort_locked("received an unknown build batch ID");
      }
      if (is_aborted) { return aborted_result(); }
      if (is_complete || completed_ids.contains(batch_id) || in_flight_ids.contains(batch_id)) {
        return {.state       = dynamic_filter_accumulation_result::status::duplicate,
                .publication = outcome};
      }
      auto const it = partials.find(device_id);
      if (it == partials.end()) {
        return abort_locked("the contribution GPU is absent from the immutable replica plan");
      }
      in_flight_ids.insert(batch_id);
      partial = it->second.get();
    }

    try {
      if (test_hooks.after_id_claim) { test_hooks.after_id_claim(batch_id); }

      {
        std::scoped_lock partial_lock(partial->mutex);
        for (std::size_t key_index = 0; key_index < active_keys.size(); ++key_index) {
          if (active_keys[key_index] == 0) { continue; }
          auto const& key = plan.admitted_keys()[key_index];
          if (key.build_key_ordinal >= build_view.num_columns()) {
            throw std::invalid_argument(
              "an admitted build-key ordinal is outside a contribution table");
          }
          if (build_view.column(key.build_key_ordinal).type() != build_types[key_index]) {
            throw build_type_mismatch{};
          }
        }

        partial->filters.resize(active_keys.size());
        for (std::size_t key_index = 0; key_index < active_keys.size(); ++key_index) {
          if (active_keys[key_index] == 0) { continue; }
          auto const column = build_view.column(plan.admitted_keys()[key_index].build_key_ordinal);
          if (!partial->filters[key_index]) {
            auto const& space         = *replica_space(device_id);
            auto const durable_stream = space.get_gpu_space().acquire_stream();
            auto filter               = std::make_shared<sirius_dynamic_bloom_filter>(
              build_types[key_index],
              build_rows,
              durable_stream,
              space.get_gpu_space().get_default_allocator());
            // CUCO retains the construction stream for deallocation; finish its initial clear on
            // this durable stream before inserting on the task stream.
            durable_stream.synchronize();
            partial->filters[key_index] = std::move(filter);
          }
          partial->filters[key_index]->add(column, stream);
        }
        stream.synchronize();
      }

      if (test_hooks.after_insert_sync) { test_hooks.after_insert_sync(batch_id); }
    } catch (build_type_mismatch const& error) {
      std::scoped_lock coordinator_lock(mutex);
      in_flight_ids.erase(batch_id);
      if (!is_aborted) { ++outcome.keys_skipped_type_mismatch; }
      return abort_locked(error.what());
    } catch (std::exception const& error) {
      synchronize_after_failure(stream, "build contribution");
      std::scoped_lock coordinator_lock(mutex);
      in_flight_ids.erase(batch_id);
      return abort_locked(error.what());
    } catch (...) {
      synchronize_after_failure(stream, "build contribution");
      std::scoped_lock coordinator_lock(mutex);
      in_flight_ids.erase(batch_id);
      return abort_locked("unknown accumulation failure");
    }

    std::scoped_lock coordinator_lock(mutex);
    in_flight_ids.erase(batch_id);
    if (is_aborted) { return aborted_result(); }
    completed_ids.insert(batch_id);
    if (completed_ids.size() != expected_ids.size()) {
      return {.state = dynamic_filter_accumulation_result::status::pending, .publication = outcome};
    }
    try {
      return publish_locked();
    } catch (std::exception const& error) {
      return abort_locked(error.what());
    } catch (...) {
      return abort_locked("unknown publication failure");
    }
  }
};

dynamic_filter_accumulator::dynamic_filter_accumulator(
  dynamic_filter_publish_plan const& plan,
  std::size_t build_rows,
  std::vector<std::uint64_t> expected_batch_ids)
  : dynamic_filter_accumulator(plan, build_rows, std::move(expected_batch_ids), {})
{
}

dynamic_filter_accumulator::dynamic_filter_accumulator(
  dynamic_filter_publish_plan const& plan,
  std::size_t build_rows,
  std::vector<std::uint64_t> expected_batch_ids,
  detail::dynamic_filter_accumulator_test_hooks test_hooks)
  : _impl(std::make_unique<impl>(
      plan, build_rows, std::move(expected_batch_ids), std::move(test_hooks)))
{
}

dynamic_filter_accumulator::~dynamic_filter_accumulator() = default;

dynamic_filter_accumulation_result dynamic_filter_accumulator::contribute(
  std::uint64_t batch_id, cudf::table_view const& build_view, rmm::cuda_stream_view stream)
{
  return _impl->contribute(batch_id, build_view, stream);
}

std::optional<dynamic_filter_publication_outcome>
dynamic_filter_accumulator::abort_if_incomplete() noexcept
{
  std::scoped_lock lock(_impl->mutex);
  if (_impl->is_complete || _impl->is_aborted) { return std::nullopt; }
  _impl->is_aborted = true;
  SIRIUS_LOG_WARN(
    "[dynamic_filter_accumulator] incomplete build snapshot closed without publication.");
  return _impl->outcome;
}

bool dynamic_filter_accumulator::complete() const noexcept
{
  std::scoped_lock lock(_impl->mutex);
  return _impl->is_complete;
}

bool dynamic_filter_accumulator::aborted() const noexcept
{
  std::scoped_lock lock(_impl->mutex);
  return _impl->is_aborted;
}

}  // namespace sirius::op
