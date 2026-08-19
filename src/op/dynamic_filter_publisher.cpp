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

#include "op/dynamic_filter_publisher.hpp"

#include "cudf/aggregation.hpp"
#include "cudf/reduction.hpp"
#include "cudf/scalar/scalar.hpp"
#include "cudf/types.hpp"
#include "log/logging.hpp"
#include "op/sirius_dynamic_filter.hpp"

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

void dynamic_filter_publisher::publish(cudf::table_view const& build_view,
                                       rmm::cuda_stream_view stream) const
{
  nvtx3::scoped_range nvtx_range{"dynfilter::push_build_side"};
  assert(_plan.enabled());

  if (build_view.num_rows() == 0) {
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] Skipping dynamic filter push: empty build table.");
    return;
  }

  auto target_accepts_filters = [](dynamic_filter_publish_plan::probe_target const& tgt) {
    return tgt.filter_set && tgt.filter_set->accepting_filters();
  };
  auto const& probe_targets = _plan.probe_targets();
  if (std::none_of(probe_targets.begin(), probe_targets.end(), target_accepts_filters)) {
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] Skipping dynamic filter push: all target scans drained.");
    return;
  }

  auto const& key_domains = _plan.build_key_domain_cardinalities();

  int source_device = -1;
  if (cudaGetDevice(&source_device) != cudaSuccess) {
    throw std::runtime_error(
      "[dynamic_filter_publisher::publish] Dynamic-filter publisher could not identify its source "
      "GPU");
  }
  auto const source_space =
    std::find_if(_plan.replica_spaces().begin(),
                 _plan.replica_spaces().end(),
                 [source_device](auto const& target) {
                   return target.get_gpu_space().get_device_id() == source_device;
                 });
  if (source_space == _plan.replica_spaces().end()) {
    throw std::logic_error(
      "[dynamic_filter_publisher::publish] Dynamic-filter source GPU is absent from the immutable "
      "publish plan");
  }
  auto const allocator_ref = source_space->get_gpu_space().get_default_allocator();
  auto const build_rows    = static_cast<std::size_t>(build_view.num_rows());
  auto const l2_bytes      = device_l2_cache_bytes(_plan.replica_spaces());

  // Build up to 2 complementary filters per join key:
  //  1) a zone-map (read-time ROW-GROUP pruning, the only path that cuts scan I/O)
  //  2) a post-decode membership filter: linear-scan IN-list for a tiny build, otherwise a
  //     hash-set IN-list or Bloom filter chosen by L2-cache fit.
  // The two ride different consumer paths and compose; either may be absent for a key.
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_zone_map(
    _filter_pushdown.join_condition.size());
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_membership(
    _filter_pushdown.join_condition.size());
  std::vector<cudf::data_type> per_key_build_type(_filter_pushdown.join_condition.size(),
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

  for (std::size_t k = 0; k < _filter_pushdown.join_condition.size(); ++k) {
    auto const cond_idx = _filter_pushdown.join_condition[k];
    // Skip cast keys to ensure type-equivalent build/probe base keys are used for the dynamic
    // filter. We can repair this constraint later.
    if (cond_idx < _key_casts.size() &&
        (_key_casts[cond_idx].cast_right || _key_casts[cond_idx].cast_left)) {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] dynamic filter key {}: skipped (cast on build key "
        "cond_idx={}).",
        k,
        cond_idx);
      continue;
    }
    if (cond_idx >= _right_key_col_indices.size()) { continue; }

    // Skip domain-covering keys before paying to build a membership structure; their filters keep
    // most probe rows, and the consumer-side gate remains the runtime backstop.
    auto const key_domain = k < key_domains.size() ? key_domains[k] : 0;
    if (key_domain > 0) {
      auto const covered =
        static_cast<double>(build_view.num_rows()) / static_cast<double>(key_domain);
      if (covered >= _plan.domain_coverage_threshold()) {
        SIRIUS_LOG_DEBUG(
          "[sirius_physical_hash_join] publish gate: key {}: build {} rows cover {:.2f} of key "
          "domain (~{} rows) -> skip key.",
          k,
          build_view.num_rows(),
          covered,
          key_domain);
        continue;
      }
    }

    auto const build_col_idx = _right_key_col_indices[cond_idx];
    auto const& col          = build_view.column(build_col_idx);
    per_key_build_type[k]    = col.type();

    // (1) Zone-map — read-time row-group pruning. This only helps when build keys are correlatively
    // clustered with the filter column(s), so it is off by default (TPC-H keys are scattered).
    if (_plan.emit_zone_map_filters()) {
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
        bool publish_zone_map = true;
        if (key_domain > 0) {
          auto const lo = scalar_to_double(*min_s);
          auto const hi = scalar_to_double(*max_s);
          if (lo && hi) {
            auto const coverage = (*hi - *lo + 1.0) / static_cast<double>(key_domain);
            if (coverage >= _plan.domain_coverage_threshold()) {
              SIRIUS_LOG_DEBUG(
                "[sirius_physical_hash_join] zone-map key {}: skipped (range [{},{}] covers {:.2f} "
                "of key domain ~{}).",
                k,
                *lo,
                *hi,
                coverage,
                key_domain);
              publish_zone_map = false;
            }
          }
        }
        if (publish_zone_map) {
          std::vector<sirius::op::zone_map_entry> zones;
          zones.push_back({std::move(min_s), std::move(max_s)});
          per_key_zone_map[k] = std::make_shared<sirius::op::sirius_dynamic_zone_map_filter>(
            std::move(zones), true, true);
        }
      }
    }

    // (2) Membership filter — post-decode. Prefer, in order:
    //  - A. the exact IN-list with a brute-force scan for a very small key set (no hash build);
    //  - B. the hash-based IN-list when its cuco set fits the device L2 cache and stays within
    //       the plan's inlist_max_l2_fraction of it. The fraction bounds residency, not
    //       capacity: competing with the streaming probe traffic, the set stops being
    //       cache-resident well before it reaches L2 capacity (measured, its probe cost is flat
    //       below the bound and degrades steadily beyond), while the smaller-but-inexact Bloom
    //       probes >= 2.2x faster at every hash-set size — so the exact filter is kept only
    //       where exactness costs the least;
    //  - C. otherwise the Bloom filter whenever the key type supports it;
    // `none` only when the key type has no membership support (anything other than INT32/INT64).
    auto const set_bytes =
      sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(build_rows, col.type());
    auto const bloom_bytes  = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(build_rows);
    char const* choice      = "none";
    bool const in_list_fits = l2_bytes > 0 &&
                              sirius::op::sirius_dynamic_in_list_filter::supports(col) &&
                              set_bytes <= l2_bytes;
    bool const bloom_supported = sirius::op::sirius_dynamic_bloom_filter::supports(col.type());
    // `in_list_fits` implies l2_bytes > 0, so the fraction threshold is well-defined wherever it
    // is evaluated; with no device L2 info (l2_bytes == 0) the legacy fit rule above has already
    // resolved the choice. A key type without Bloom support keeps any fitting IN-list.
    bool const prefer_in_list =
      in_list_fits &&
      (!bloom_supported || static_cast<double>(set_bytes) <=
                             _plan.inlist_max_l2_fraction() * static_cast<double>(l2_bytes));
    if (sirius::op::sirius_dynamic_small_in_list_filter::supports(col)) {
      nvtx3::scoped_range vr{"dynfilter::build_small_in_list"};
      per_key_membership[k] = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
        col, stream, allocator_ref);
      choice = "small_in_list";
    } else if (prefer_in_list) {
      nvtx3::scoped_range vr{"dynfilter::build_in_list"};
      per_key_membership[k] =
        std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(col, stream, allocator_ref);
      choice = "in_list";
    } else if (sirius::op::sirius_dynamic_bloom_filter::supports(col.type())) {
      nvtx3::scoped_range vr{"dynfilter::build_bloom"};
      per_key_membership[k] =
        std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(col, stream, allocator_ref);
      choice = "bloom";
    }
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] dynamic filter key {}: build_rows={} zone_map={} membership: "
      "in_list_set={}B bloom={}B L2={}B inlist_max_l2_fraction={} -> {}",
      k,
      build_rows,
      per_key_zone_map[k] ? "yes" : "no",
      set_bytes,
      bloom_bytes,
      l2_bytes,
      _plan.inlist_max_l2_fraction(),
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
    auto replicate = [this](std::shared_ptr<sirius_dynamic_filter> const& filter) {
      if (!filter) { return; }
      auto* replicable = dynamic_cast<sirius_device_replicable*>(filter.get());
      if (replicable == nullptr) {
        throw std::logic_error(
          "[dynamic_filter_publisher::publish] A published device-backed dynamic filter must "
          "implement sirius_device_replicable");
      }
      replicable->replicate_to_devices(_plan.replica_spaces());
    };
    for (auto const& filter : per_key_zone_map) {
      replicate(filter);
    }
    for (auto const& filter : per_key_membership) {
      replicate(filter);
    }
  }

  // Fan out across probe targets
  std::size_t total_pushed   = 0;
  std::size_t active_targets = 0;
  for (auto const& tgt : probe_targets) {
    if (!target_accepts_filters(tgt)) { continue; }
    ++active_targets;

    if (tgt.probe_col_idx.size() != per_key_membership.size()) {
      SIRIUS_LOG_WARN(
        "[sirius_physical_hash_join] dynamic-filter column mismatch (probe_col_idx={} keys={}); "
        "skipping target to preserve correctness.",
        tgt.probe_col_idx.size(),
        per_key_membership.size());
      continue;
    }
    for (std::size_t k = 0; k < per_key_membership.size(); ++k) {
      if (per_key_zone_map[k] && k < tgt.probe_col_type.size() &&
          tgt.probe_col_type[k] == per_key_build_type[k] &&
          tgt.filter_set->push_filter(tgt.probe_col_idx[k], per_key_zone_map[k])) {
        ++total_pushed;
      }
      if (per_key_membership[k] &&
          tgt.filter_set->push_filter(tgt.probe_col_idx[k], per_key_membership[k])) {
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
    _filter_pushdown.join_condition.size());
}

}  // namespace sirius::op
