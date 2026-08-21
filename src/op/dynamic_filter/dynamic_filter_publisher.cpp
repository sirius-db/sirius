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

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

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
#include <utility>
#include <vector>

namespace sirius::op {

std::optional<complete_build_snapshot> complete_build_snapshot::try_create(
  std::uint64_t total_rows, std::vector<std::uint64_t> batch_ids, std::size_t partition_count)
{
  if (partition_count <= 1 || batch_ids.empty() || !std::in_range<std::size_t>(total_rows)) {
    return std::nullopt;
  }

  std::unordered_set<std::uint64_t> unique_ids;
  unique_ids.reserve(batch_ids.size());
  for (auto const batch_id : batch_ids) {
    if (!unique_ids.insert(batch_id).second) { return std::nullopt; }
  }

  return complete_build_snapshot{
    static_cast<std::size_t>(total_rows), std::move(batch_ids), partition_count};
}

complete_build_snapshot::complete_build_snapshot(complete_build_snapshot&& other) noexcept
  : _total_rows(std::exchange(other._total_rows, 0)),
    _batch_ids(std::move(other._batch_ids)),
    _partition_count(std::exchange(other._partition_count, 0))
{
}

complete_build_snapshot& complete_build_snapshot::operator=(
  complete_build_snapshot&& other) noexcept
{
  if (this == &other) { return *this; }
  _total_rows      = std::exchange(other._total_rows, 0);
  _batch_ids       = std::move(other._batch_ids);
  _partition_count = std::exchange(other._partition_count, 0);
  return *this;
}

std::optional<complete_build_snapshot> detail::try_summarize_complete_build(
  std::span<complete_build_batch_summary const> batches, std::size_t partition_count)
{
  std::uint64_t total_rows = 0;
  std::vector<std::uint64_t> batch_ids;
  batch_ids.reserve(batches.size());
  for (auto const& batch : batches) {
    if (total_rows > std::numeric_limits<std::uint64_t>::max() - batch.rows) {
      return std::nullopt;
    }
    total_rows += batch.rows;
    batch_ids.push_back(batch.batch_id);
  }
  return complete_build_snapshot::try_create(total_rows, std::move(batch_ids), partition_count);
}

namespace {
template <typename Function>
void invoke_noexcept(Function&& function) noexcept
{
  try {
    std::forward<Function>(function)();
  } catch (...) {
  }
}

void synchronize_after_failure(rmm::cuda_stream_view stream, std::string_view operation) noexcept
{
  try {
    stream.synchronize();
  } catch (std::exception const& error) {
    invoke_noexcept([&] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publisher] {} failure drain also failed: {}", operation, error.what());
    });
  } catch (...) {
    invoke_noexcept([&] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publisher] {} failure drain also failed with an unknown error.",
        operation);
    });
  }
}

class failure_stream_guard {
 public:
  failure_stream_guard(rmm::cuda_stream_view stream, std::string_view operation) noexcept
    : _stream(stream), _operation(operation)
  {
  }

  ~failure_stream_guard() noexcept
  {
    if (_active) { synchronize_after_failure(_stream, _operation); }
  }

  void dismiss() noexcept { _active = false; }

 private:
  rmm::cuda_stream_view _stream;
  std::string_view _operation;
  bool _active{true};
};

void fold_dynamic_filter_outcome(dynamic_filter_stats& stats,
                                 dynamic_filter_publication_outcome const& outcome) noexcept
{
  auto const relaxed = std::memory_order_relaxed;
  stats.keys_considered.fetch_add(outcome.keys_considered, relaxed);
  stats.keys_with_known_domain.fetch_add(outcome.keys_with_known_domain, relaxed);
  stats.keys_skipped_domain_gate.fetch_add(outcome.keys_skipped_domain_gate, relaxed);
  stats.keys_skipped_bloom_size_gate.fetch_add(outcome.keys_skipped_bloom_size_gate, relaxed);
  stats.keys_skipped_type_mismatch.fetch_add(outcome.keys_skipped_type_mismatch, relaxed);
  stats.keys_build_exceeded_domain.fetch_add(outcome.keys_build_exceeded_domain, relaxed);
  stats.membership_filters_built.fetch_add(outcome.membership_filters_built, relaxed);
  stats.zone_map_filters_built.fetch_add(outcome.zone_map_filters_built, relaxed);
  stats.publications_skipped_targets_drained.fetch_add(outcome.skipped_targets_drained, relaxed);
  stats.filters_pushed.fetch_add(outcome.filters_pushed, relaxed);
}

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

dynamic_filter_publication_outcome publish_dynamic_filters_impl(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  rmm::cuda_stream_view stream,
  std::function<void(std::size_t)> const* before_key)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::push_build_side"};
  assert(plan.enabled());
  dynamic_filter_publication_outcome outcome;

  if (build_view.num_rows() == 0) {
    invoke_noexcept([] {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] Skipping dynamic filter push: empty build table.");
    });
    return outcome;
  }

  auto target_accepts_filters = [](dynamic_filter_publish_plan::probe_target const& tgt) {
    return tgt.filter_set && tgt.filter_set->accepting_filters();
  };
  auto const& probe_targets = plan.probe_targets();
  if (std::none_of(probe_targets.begin(), probe_targets.end(), target_accepts_filters)) {
    invoke_noexcept([] {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] Skipping dynamic filter push: all target scans drained.");
    });
    outcome.skipped_targets_drained = 1;
    return outcome;
  }

  failure_stream_guard failure_guard{stream, "one-shot publication"};

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
  auto const bloom_bytes   = sirius_dynamic_bloom_filter::estimated_bytes(build_rows);

  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_zone_map(admitted_keys.size());
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_membership(admitted_keys.size());
  std::vector<char> per_key_bloom_candidate(admitted_keys.size(), 0);
  std::vector<cudf::data_type> per_key_build_type(admitted_keys.size(),
                                                  cudf::data_type{cudf::type_id::EMPTY});

  for (std::size_t admitted_key_index = 0; admitted_key_index < admitted_keys.size();
       ++admitted_key_index) {
    if (key_bound[admitted_key_index] == 0) { continue; }
    if (before_key != nullptr && *before_key) { (*before_key)(admitted_key_index); }
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
      invoke_noexcept([&] {
        SIRIUS_LOG_DEBUG(
          "[sirius_physical_hash_join] publish gate: key {}: build {} rows cover {:.2f} of key "
          "domain (~{} rows) -> skip key.",
          admitted_key_index,
          build_view.num_rows(),
          static_cast<double>(build_rows) / static_cast<double>(key_domain),
          key_domain);
      });
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
      invoke_noexcept([&] {
        SIRIUS_LOG_WARN(
          "[sirius_physical_hash_join] dynamic filter key {}: skipped (plan recorded type id {} "
          "but build column {} carries type id {}).",
          admitted_key_index,
          static_cast<int32_t>(admitted_key.storage_type.id()),
          admitted_key.build_key_ordinal,
          static_cast<int32_t>(col.type().id()));
      });
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
        // Deferred: construction happens after the loop, under the aggregate per-GPU budget.
        per_key_bloom_candidate[admitted_key_index] = 1;
        choice                                      = "bloom";
        break;
      }
      case membership_filter_kind::none: break;
    }
    if (per_key_membership[admitted_key_index]) { ++outcome.membership_filters_built; }
    if (per_key_zone_map[admitted_key_index]) { ++outcome.zone_map_filters_built; }
    invoke_noexcept([&] {
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
    });
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
    invoke_noexcept([&] {
      SIRIUS_LOG_INFO(
        "[sirius_physical_hash_join] Skipping {} Bloom filter candidate(s): each would use {} "
        "allocator-accounted bytes against the {}-byte aggregate per-GPU budget.",
        bloom_candidate_count,
        bloom_bytes,
        plan.max_bloom_bytes_per_gpu());
    });
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
  outcome.filters_pushed = total_pushed;
  failure_guard.dismiss();
  invoke_noexcept([&] {
    SIRIUS_LOG_INFO(
      "[sirius_physical_hash_join] Pushed {} dynamic filter(s) across {} active target(s) "
      "of {} wired target(s) ({} build rows, {} bound keys of {} admitted).",
      total_pushed,
      active_targets,
      probe_targets.size(),
      build_view.num_rows(),
      outcome.keys_considered,
      admitted_keys.size());
  });
  return outcome;
}
}  // namespace

dynamic_filter_publication_outcome publish_dynamic_filters(dynamic_filter_publish_plan const& plan,
                                                           cudf::table_view const& build_view,
                                                           rmm::cuda_stream_view stream)
{
  return publish_dynamic_filters_impl(plan, build_view, stream, nullptr);
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
  bool is_complete             = false;
  bool is_aborted              = false;
  int published_root_device_id = -1;

  impl(dynamic_filter_publish_plan const& plan,
       complete_build_snapshot snapshot,
       detail::dynamic_filter_accumulator_test_hooks test_hooks)
    : plan(plan),
      build_rows(snapshot.total_rows()),
      expected_ids(snapshot.batch_ids().begin(), snapshot.batch_ids().end()),
      active_keys(plan.admitted_keys().size(), 0),
      build_types(plan.admitted_keys().size(), cudf::data_type{cudf::type_id::EMPTY}),
      test_hooks(std::move(test_hooks))
  {
    if (!plan.enabled() || !snapshot.valid()) {
      throw std::invalid_argument(
        "[dynamic_filter_accumulator] enabled plan and valid snapshot required.");
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
      if (build_rows == 0) { continue; }
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
      invoke_noexcept([&] {
        SIRIUS_LOG_INFO(
          "[dynamic_filter_accumulator] Skipping {} global Bloom filter candidate(s): each would "
          "use {} allocator-accounted bytes against the {}-byte aggregate per-GPU budget.",
          active_bloom_keys,
          bloom_bytes,
          plan.max_bloom_bytes_per_gpu());
      });
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

  [[nodiscard]] dynamic_filter_accumulation_result completed_result(
    dynamic_filter_accumulation_result::status state) const
  {
    return {.state                    = state,
            .publication              = outcome,
            .exact_contribution_count = completed_ids.size(),
            .global_build_rows        = build_rows,
            .root_device_id           = published_root_device_id};
  }

  [[nodiscard]] std::optional<dynamic_filter_accumulation_result> terminal_result_locked() const
  {
    if (is_complete) {
      return completed_result(dynamic_filter_accumulation_result::status::duplicate);
    }
    if (is_aborted) { return aborted_result(); }
    return std::nullopt;
  }

  [[nodiscard]] dynamic_filter_accumulation_result abort_locked(std::string_view reason)
  {
    if (!is_aborted && !is_complete) {
      is_aborted = true;
      invoke_noexcept(
        [&] { SIRIUS_LOG_WARN("[dynamic_filter_accumulator] publication aborted: {}", reason); });
    }
    return aborted_result();
  }

  [[nodiscard]] dynamic_filter_accumulation_result publish_locked(int root_device)
  {
    auto const* root_space = replica_space(root_device);
    if (root_space == nullptr) {
      throw std::logic_error("the final contribution GPU is absent from the replica plan");
    }

    auto target_accepts_filters = [](dynamic_filter_publish_plan::probe_target const& target) {
      return target.filter_set && target.filter_set->accepting_filters();
    };
    if (std::none_of(
          plan.probe_targets().begin(), plan.probe_targets().end(), target_accepts_filters)) {
      outcome.skipped_targets_drained = 1;
      published_root_device_id        = root_device;
      is_complete                     = true;
      return completed_result(dynamic_filter_accumulation_result::status::published);
    }

    rmm::cuda_set_device_raii root_guard{rmm::cuda_device_id{root_device}};
    auto const root_stream = root_space->get_gpu_space().acquire_stream();
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
            root_space->get_gpu_space().get_default_allocator());
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
            *partial->filters[key_index], *source_space, *root_space, root_stream);
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

    published_root_device_id = root_device;
    is_complete              = true;
    invoke_noexcept([&] {
      SIRIUS_LOG_INFO(
        "[dynamic_filter_accumulator] published {} global Bloom filter(s) across {} active "
        "target(s) after {} exact build contribution(s), {} build rows, root GPU {}.",
        outcome.membership_filters_built,
        outcome.active_targets,
        completed_ids.size(),
        build_rows,
        root_device);
    });
    return completed_result(dynamic_filter_accumulation_result::status::published);
  }

  [[nodiscard]] dynamic_filter_accumulation_result contribute(std::uint64_t batch_id,
                                                              cudf::table_view const& build_view,
                                                              rmm::cuda_stream_view stream)
  {
    {
      std::scoped_lock coordinator_lock(mutex);
      if (auto terminal = terminal_result_locked()) { return *terminal; }
    }

    int device_id = -1;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
      std::scoped_lock coordinator_lock(mutex);
      if (auto terminal = terminal_result_locked()) { return *terminal; }
      return abort_locked("could not identify the contribution GPU");
    }

    device_partial* partial = nullptr;
    {
      std::scoped_lock coordinator_lock(mutex);
      if (auto terminal = terminal_result_locked()) { return *terminal; }
      if (!expected_ids.contains(batch_id)) {
        return abort_locked("received an unknown build batch ID");
      }
      if (completed_ids.contains(batch_id) || in_flight_ids.contains(batch_id)) {
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
      // gpu_pipeline_task owns a reservation on this host thread's current GPU when reservation
      // tracking is per thread. Keeping the final contributor as the source avoids attaching a
      // second reservation to that same adaptor during strict replica fan-out.
      return publish_locked(device_id);
    } catch (std::exception const& error) {
      return abort_locked(error.what());
    } catch (...) {
      return abort_locked("unknown publication failure");
    }
  }
};

dynamic_filter_accumulator::dynamic_filter_accumulator(dynamic_filter_publish_plan const& plan,
                                                       complete_build_snapshot snapshot)
  : dynamic_filter_accumulator(plan, std::move(snapshot), {})
{
}

dynamic_filter_accumulator::dynamic_filter_accumulator(
  dynamic_filter_publish_plan const& plan,
  complete_build_snapshot snapshot,
  detail::dynamic_filter_accumulator_test_hooks test_hooks)
  : _impl(std::make_unique<impl>(plan, std::move(snapshot), std::move(test_hooks)))
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
  std::unique_lock lock(_impl->mutex);
  if (_impl->is_complete || _impl->is_aborted) { return std::nullopt; }
  _impl->is_aborted  = true;
  auto const outcome = _impl->outcome;
  lock.unlock();
  invoke_noexcept([] {
    SIRIUS_LOG_WARN(
      "[dynamic_filter_accumulator] incomplete build snapshot closed without publication.");
  });
  return outcome;
}

dynamic_filter_accumulation_result dynamic_filter_accumulator::abort_or_get_terminal() noexcept
{
  bool newly_aborted = false;
  dynamic_filter_accumulation_result result;
  {
    std::scoped_lock lock(_impl->mutex);
    if (_impl->is_complete) {
      result = _impl->completed_result(dynamic_filter_accumulation_result::status::published);
    } else {
      newly_aborted     = !_impl->is_aborted;
      _impl->is_aborted = true;
      result            = {.state       = dynamic_filter_accumulation_result::status::aborted,
                           .publication = _impl->outcome};
    }
  }
  if (newly_aborted) {
    invoke_noexcept([] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_accumulator] incomplete build snapshot closed without publication.");
    });
  }
  return result;
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

dynamic_filter_publication_session::dynamic_filter_publication_session(
  dynamic_filter_publish_plan const& plan, dynamic_filter_stats* stats, bool enable_multi_partition)
  : dynamic_filter_publication_session(plan, stats, enable_multi_partition, {})
{
}

dynamic_filter_publication_session::dynamic_filter_publication_session(
  dynamic_filter_publish_plan const& plan,
  dynamic_filter_stats* stats,
  bool enable_multi_partition,
  detail::dynamic_filter_publication_session_test_hooks test_hooks)
  : _plan(plan),
    _stats(stats),
    _enable_multi_partition(enable_multi_partition),
    _test_hooks(std::move(test_hooks))
{
  if (_stats != nullptr && enabled()) {
    _stats->producers_enabled.fetch_add(1, std::memory_order_relaxed);
  }
}

bool dynamic_filter_publication_session::is_open() const noexcept
{
  std::scoped_lock lock(_mutex);
  return _state == state::open;
}

void dynamic_filter_publication_session::commit_terminal_locked(
  state terminal, dynamic_filter_publication_outcome const& outcome) noexcept
{
  assert(terminal == state::finished || terminal == state::failed);
  _state = terminal;
  _accumulator.reset();
  if (_stats == nullptr) { return; }

  fold_dynamic_filter_outcome(*_stats, outcome);
  if (terminal == state::finished) {
    _stats->publications_finished.fetch_add(1, std::memory_order_relaxed);
  } else {
    _stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
  }
}

void dynamic_filter_publication_session::commit_accumulation_terminal_locked(
  state terminal,
  dynamic_filter_accumulation_result const& result,
  std::uint64_t join_operator_id) noexcept
{
  commit_terminal_locked(terminal, result.publication);
  if (_stats == nullptr || terminal != state::finished ||
      result.state != dynamic_filter_accumulation_result::status::published) {
    return;
  }
  _stats->record_global_accumulator_completion(join_operator_id,
                                               result.exact_contribution_count,
                                               result.root_device_id,
                                               result.global_build_rows,
                                               result.publication.membership_filters_built,
                                               result.publication.active_targets,
                                               result.publication.filters_pushed);
}

bool dynamic_filter_publication_session::try_arm(complete_build_snapshot snapshot)
{
  if (!wants_multi_partition() || !snapshot.valid()) { return false; }

  std::unique_lock lock(_mutex);
  if (_state != state::open) { return false; }

  if (_stats != nullptr) { _stats->publication_attempts.fetch_add(1, std::memory_order_relaxed); }
  auto const expected_count = snapshot.batch_ids().size();
  auto const build_rows     = snapshot.total_rows();
  auto const partitions     = snapshot.partition_count();
  try {
    _accumulator = std::make_shared<dynamic_filter_accumulator>(
      _plan, std::move(snapshot), std::move(_test_hooks.accumulator));
    _state = state::accumulating;
    lock.unlock();
    invoke_noexcept([&] {
      SIRIUS_LOG_DEBUG(
        "[dynamic_filter_publication_session] armed global dynamic Bloom for {} exact build "
        "batch(es), {} rows, {} partition(s).",
        expected_count,
        build_rows,
        partitions);
    });
    return true;
  } catch (std::exception const& error) {
    _accumulator.reset();
    _state = state::failed;
    if (_stats != nullptr) { _stats->publications_failed.fetch_add(1, std::memory_order_relaxed); }
    lock.unlock();
    invoke_noexcept([&] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publication_session] global dynamic Bloom disabled for this join: {}",
        error.what());
    });
    return false;
  } catch (...) {
    _accumulator.reset();
    _state = state::failed;
    if (_stats != nullptr) { _stats->publications_failed.fetch_add(1, std::memory_order_relaxed); }
    lock.unlock();
    invoke_noexcept([] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publication_session] global dynamic Bloom disabled by an unknown error.");
    });
    return false;
  }
}

void dynamic_filter_publication_session::contribute(std::uint64_t join_operator_id,
                                                    std::uint64_t batch_id,
                                                    cudf::table_view const& build_view,
                                                    rmm::cuda_stream_view stream) noexcept
{
  std::shared_ptr<dynamic_filter_accumulator> accumulator;
  {
    std::scoped_lock lock(_mutex);
    if (_state != state::accumulating || !_accumulator) { return; }
    accumulator = _accumulator;
  }

  dynamic_filter_accumulation_result result;
  auto commit_terminal_after_exception = [&] {
    auto const terminal = accumulator->abort_or_get_terminal();
    std::scoped_lock lock(_mutex);
    if (_state != state::accumulating) { return; }
    commit_accumulation_terminal_locked(
      terminal.state == dynamic_filter_accumulation_result::status::published ? state::finished
                                                                              : state::failed,
      terminal,
      join_operator_id);
  };
  try {
    result = accumulator->contribute(batch_id, build_view, stream);
    if (_test_hooks.after_accumulation_result) {
      _test_hooks.after_accumulation_result(result.state);
    }
  } catch (std::exception const& error) {
    commit_terminal_after_exception();
    invoke_noexcept([&] {
      SIRIUS_LOG_WARN("[dynamic_filter_publication_session] unexpected contribution failure: {}",
                      error.what());
    });
    return;
  } catch (...) {
    commit_terminal_after_exception();
    invoke_noexcept([] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publication_session] unexpected unknown contribution failure.");
    });
    return;
  }

  std::scoped_lock lock(_mutex);
  if (_state != state::accumulating) { return; }
  if (result.state == dynamic_filter_accumulation_result::status::published) {
    commit_accumulation_terminal_locked(state::finished, result, join_operator_id);
  } else if (result.state == dynamic_filter_accumulation_result::status::aborted) {
    commit_accumulation_terminal_locked(state::failed, result, join_operator_id);
  }
}

void dynamic_filter_publication_session::publish_one_shot(cudf::table_view const& complete_build,
                                                          rmm::cuda_stream_view stream)
{
  {
    std::scoped_lock lock(_mutex);
    if (_state != state::open || !enabled()) { return; }
    _state = state::publishing;
    if (_stats != nullptr) { _stats->publication_attempts.fetch_add(1, std::memory_order_relaxed); }
  }

  try {
    auto const outcome =
      publish_dynamic_filters_impl(_plan, complete_build, stream, &_test_hooks.before_one_shot_key);
    {
      std::scoped_lock lock(_mutex);
      if (_state == state::publishing) { commit_terminal_locked(state::finished, outcome); }
    }
    invoke_noexcept([&] {
      SIRIUS_LOG_DEBUG(
        "[dynamic_filter_publication_session] publication: {} key(s) considered, {} skipped "
        "(domain gate), {} skipped (Bloom size gate), {} skipped (type mismatch), {} membership + "
        "{} zone-map built, {} filter(s) pushed across {} active target(s).",
        outcome.keys_considered,
        outcome.keys_skipped_domain_gate,
        outcome.keys_skipped_bloom_size_gate,
        outcome.keys_skipped_type_mismatch,
        outcome.membership_filters_built,
        outcome.zone_map_filters_built,
        outcome.filters_pushed,
        outcome.active_targets);
    });
  } catch (rmm::out_of_memory const& oom) {
    // Dynamic filters are optional; device OOM fails publication without failing the query.
    // Terminal failed, never reopened: retrying a sibling delivery under the same memory
    // pressure is the storm this catch exists to avoid (the no-usable-source skip path leaves
    // the session open instead).
    {
      std::scoped_lock lock(_mutex);
      if (_state == state::publishing) {
        commit_terminal_locked(state::failed, dynamic_filter_publication_outcome{});
      }
    }
    invoke_noexcept([&] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publication_session] one-shot publication hit device memory exhaustion; "
        "continuing without filters: {}",
        oom.what());
    });
  } catch (...) {
    std::scoped_lock lock(_mutex);
    if (_state == state::publishing) {
      commit_terminal_locked(state::failed, dynamic_filter_publication_outcome{});
    }
    throw;
  }
}

void dynamic_filter_publication_session::finalize_or_abort() noexcept
{
  std::shared_ptr<dynamic_filter_accumulator> accumulator;
  {
    std::scoped_lock lock(_mutex);
    if (_state == state::open) {
      _state = state::closed;
      return;
    }
    if (_state != state::accumulating || !_accumulator) { return; }
    accumulator = _accumulator;
  }

  auto const outcome = accumulator->abort_if_incomplete();
  if (!outcome) { return; }

  bool committed = false;
  {
    std::scoped_lock lock(_mutex);
    if (_state == state::accumulating) {
      commit_terminal_locked(state::failed, *outcome);
      committed = true;
    }
  }
  if (committed) {
    invoke_noexcept([] {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_publication_session] global dynamic Bloom closed with a missing build "
        "contribution; no filter was published.");
    });
  }
}

void dynamic_filter_publication_session::record_source_not_resident() noexcept
{
  std::scoped_lock lock(_mutex);
  if (_state != state::open || _stats == nullptr) { return; }
  _stats->publications_skipped_source_not_resident.fetch_add(1, std::memory_order_relaxed);
}

void dynamic_filter_publication_session::record_build_not_whole() noexcept
{
  std::scoped_lock lock(_mutex);
  if (_state != state::open || _stats == nullptr) { return; }
  _stats->publications_skipped_build_not_whole.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace sirius::op
