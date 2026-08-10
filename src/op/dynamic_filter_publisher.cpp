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

#include <rmm/cuda_device.hpp>

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

bool target_accepts_filters(dynamic_filter_publish_plan::probe_target const& tgt)
{
  return tgt.filter_set && tgt.filter_set->accepting_filters();
}

// Replicate every built filter to all planned device spaces. The producer stream must already be
// drained: consumers probe replicas from their own streams the moment push_filter lands, with no
// event ordering back to the producer.
void replicate_filters(
  dynamic_filter_publish_plan const& plan,
  std::vector<std::shared_ptr<sirius_dynamic_filter>> const& per_key_zone_map,
  std::vector<std::shared_ptr<sirius_dynamic_filter>> const& per_key_membership)
{
  nvtx3::scoped_range replicate_range{"dynfilter::replicate_devices"};
  auto replicate = [&plan](std::shared_ptr<sirius_dynamic_filter> const& filter) {
    if (!filter) { return; }
    auto* replicable = dynamic_cast<sirius_device_replicable*>(filter.get());
    if (replicable == nullptr) {
      throw std::logic_error(
        "[dynamic_filter_publisher] A published device-backed dynamic filter must "
        "implement sirius_device_replicable");
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

struct fan_out_result {
  std::size_t total_pushed   = 0;
  std::size_t active_targets = 0;
};

// Push every built filter into each still-accepting probe target's channel.
fan_out_result fan_out_filters(
  dynamic_filter_publish_plan const& plan,
  std::vector<std::shared_ptr<sirius_dynamic_filter>> const& per_key_zone_map,
  std::vector<std::shared_ptr<sirius_dynamic_filter>> const& per_key_membership,
  std::vector<cudf::data_type> const& per_key_build_type)
{
  fan_out_result result;
  for (auto const& tgt : plan.probe_targets()) {
    if (!target_accepts_filters(tgt)) { continue; }
    ++result.active_targets;

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
        ++result.total_pushed;
      }
      if (per_key_membership[k] &&
          tgt.filter_set->push_filter(tgt.probe_col_idx[k], per_key_membership[k])) {
        ++result.total_pushed;
      }
    }
  }
  return result;
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
    //  - B. the hash-based IN-list when its cuco set fits the device L2 cache;
    //  - C. otherwise the Bloom filter whenever the key type supports it;
    // `none` only when the key type has no membership support (anything other than INT32/INT64).
    auto const set_bytes =
      sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(build_rows, col.type());
    auto const bloom_bytes  = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(build_rows);
    char const* choice      = "none";
    bool const in_list_fits = l2_bytes > 0 &&
                              sirius::op::sirius_dynamic_in_list_filter::supports(col) &&
                              set_bytes <= l2_bytes;
    if (sirius::op::sirius_dynamic_small_in_list_filter::supports(col)) {
      nvtx3::scoped_range vr{"dynfilter::build_small_in_list"};
      per_key_membership[k] = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
        col, stream, allocator_ref);
      choice = "small_in_list";
    } else if (in_list_fits) {
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
      "in_list_set={}B bloom={}B L2={}B -> {}",
      k,
      build_rows,
      per_key_zone_map[k] ? "yes" : "no",
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
    replicate_filters(_plan, per_key_zone_map, per_key_membership);
  }

  auto const pushed =
    fan_out_filters(_plan, per_key_zone_map, per_key_membership, per_key_build_type);
  SIRIUS_LOG_INFO(
    "[sirius_physical_hash_join] Pushed {} dynamic filter(s) across {} active target(s) "
    "of {} wired target(s) ({} build rows, {} keys).",
    pushed.total_pushed,
    pushed.active_targets,
    probe_targets.size(),
    build_view.num_rows(),
    _filter_pushdown.join_condition.size());
}

//===----------------------------------------------------------------------===//
// dynamic_filter_accumulator
//===----------------------------------------------------------------------===//

dynamic_filter_accumulator::dynamic_filter_accumulator(
  duckdb::JoinFilterPushdownInfo const& filter_pushdown,
  dynamic_filter_publish_plan const& plan,
  std::vector<sirius_physical_hash_join::key_cast_info> const& key_casts,
  std::vector<cudf::size_type> const& right_key_col_indices,
  std::size_t estimated_build_rows,
  std::size_t expected_batches)
  : _filter_pushdown(filter_pushdown),
    _plan(plan),
    _key_casts(key_casts),
    _right_key_col_indices(right_key_col_indices),
    _estimated_build_rows(estimated_build_rows),
    _expected_batches(expected_batches),
    _source_device(plan.replica_spaces().empty()
                     ? -1
                     : plan.replica_spaces().front().get_gpu_space().get_device_id())
{
  assert(plan.enabled());
  if (_source_device < 0) {
    throw std::logic_error(
      "[dynamic_filter_accumulator] publish plan has no replica space to source construction");
  }
  if (_expected_batches == 0) {
    throw std::invalid_argument("[dynamic_filter_accumulator] expected_batches must be > 0");
  }
  _keys.resize(_filter_pushdown.join_condition.size());
}

dynamic_filter_accumulator::~dynamic_filter_accumulator()
{
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{_source_device}};
  for (auto* event : _events) {
    (void)cudaEventDestroy(event);
  }
}

bool dynamic_filter_accumulator::finished() const noexcept { return _finished; }

void dynamic_filter_accumulator::begin(cudf::table_view const& build_view,
                                       rmm::cuda_stream_view stream)
{
  _begun = true;

  // All targets may already have drained (their scans finished before the build sized) — then
  // nothing built here could ever be consumed, so leave every key ineligible.
  auto const& probe_targets = _plan.probe_targets();
  if (std::none_of(probe_targets.begin(), probe_targets.end(), target_accepts_filters)) {
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] multi-batch dynamic filter: all target scans drained before "
      "accumulation began; publishing nothing.");
    return;
  }

  auto const& key_domains  = _plan.build_key_domain_cardinalities();
  auto const allocator_ref = _plan.replica_spaces().front().get_gpu_space().get_default_allocator();

  for (std::size_t k = 0; k < _filter_pushdown.join_condition.size(); ++k) {
    auto const cond_idx = _filter_pushdown.join_condition[k];
    // Same gates as the one-shot publisher: cast keys are skipped to keep build/probe key
    // representations type-equivalent, and domain-covering keys are not worth a filter. The
    // coverage pre-gate uses the row *estimate*; finish() re-checks with the exact count.
    if (cond_idx < _key_casts.size() &&
        (_key_casts[cond_idx].cast_right || _key_casts[cond_idx].cast_left)) {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] multi-batch dynamic filter key {}: skipped (cast on build "
        "key cond_idx={}).",
        k,
        cond_idx);
      continue;
    }
    if (cond_idx >= _right_key_col_indices.size()) { continue; }
    auto const key_domain = k < key_domains.size() ? key_domains[k] : 0;
    if (key_domain > 0) {
      auto const covered =
        static_cast<double>(_estimated_build_rows) / static_cast<double>(key_domain);
      if (covered >= _plan.domain_coverage_threshold()) {
        SIRIUS_LOG_DEBUG(
          "[sirius_physical_hash_join] multi-batch publish gate: key {}: ~{} build rows cover "
          "{:.2f} of key domain (~{} rows) -> skip key.",
          k,
          _estimated_build_rows,
          covered,
          key_domain);
        continue;
      }
    }

    auto const build_col_idx = _right_key_col_indices[cond_idx];
    auto const& col          = build_view.column(build_col_idx);
    if (!sirius_dynamic_bloom_filter::supports(col.type())) {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] multi-batch dynamic filter key {}: skipped (type has no "
        "Bloom support; incremental construction is Bloom-only).",
        k);
      continue;
    }

    nvtx3::scoped_range vr{"dynfilter::begin_multibatch_bloom"};
    _keys[k].build_type = col.type();
    _keys[k].bloom      = std::make_shared<sirius_dynamic_bloom_filter>(
      col.type(), _estimated_build_rows, stream, allocator_ref);
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] multi-batch dynamic filter key {}: Bloom sized for ~{} keys "
      "({}B), fed by {} expected batches.",
      k,
      _estimated_build_rows,
      sirius_dynamic_bloom_filter::estimated_bytes(_estimated_build_rows),
      _expected_batches);
  }
}

void dynamic_filter_accumulator::record_event(rmm::cuda_stream_view stream)
{
  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    throw std::runtime_error("[dynamic_filter_accumulator] failed to create ordering event");
  }
  if (cudaEventRecord(event, stream.value()) != cudaSuccess) {
    (void)cudaEventDestroy(event);
    throw std::runtime_error("[dynamic_filter_accumulator] failed to record ordering event");
  }
  _events.push_back(event);
}

bool dynamic_filter_accumulator::add(cudf::table_view const& build_view,
                                     rmm::cuda_stream_view stream)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_finished) {
    SIRIUS_LOG_WARN(
      "[dynamic_filter_accumulator] build batch arrived after publication ({} expected); "
      "ignoring.",
      _expected_batches);
    return false;
  }

  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{_source_device}};
  if (!_begun) { begin(build_view, stream); }

  for (std::size_t k = 0; k < _keys.size(); ++k) {
    if (!_keys[k].bloom) { continue; }
    auto const build_col_idx = _right_key_col_indices[_filter_pushdown.join_condition[k]];
    // add_keys orders nothing across streams itself; each contributing stream is instead captured
    // in _events below, and finish() orders the publishing stream after all of them.
    _keys[k].bloom->add_keys(build_view.column(build_col_idx), stream);
  }
  _accumulated_rows += static_cast<std::size_t>(build_view.num_rows());
  record_event(stream);
  ++_accumulated_batches;

  if (_accumulated_batches < _expected_batches) { return false; }
  finish(stream);
  return true;
}

void dynamic_filter_accumulator::finish(rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::publish_multibatch"};
  _finished = true;

  // Order the publishing stream after every contributing add(), then drain it: publication is
  // cross-stream with no event ordering to consumers (same contract as the one-shot publisher).
  for (auto* event : _events) {
    if (cudaStreamWaitEvent(stream.value(), event, 0) != cudaSuccess) {
      throw std::runtime_error("[dynamic_filter_accumulator] failed to wait on ordering event");
    }
  }

  // Exact-row re-check of the domain-coverage gate. The Bloom build cost is already sunk, but a
  // domain-covering filter keeps ~everything and only costs the probe side — drop it.
  auto const& key_domains = _plan.build_key_domain_cardinalities();
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_membership(_keys.size());
  std::vector<std::shared_ptr<sirius_dynamic_filter>> per_key_zone_map(_keys.size());
  std::vector<cudf::data_type> per_key_build_type(_keys.size(),
                                                  cudf::data_type{cudf::type_id::EMPTY});
  std::size_t built = 0;
  for (std::size_t k = 0; k < _keys.size(); ++k) {
    if (!_keys[k].bloom) { continue; }
    auto const key_domain = k < key_domains.size() ? key_domains[k] : 0;
    if (key_domain > 0) {
      auto const covered = static_cast<double>(_accumulated_rows) / static_cast<double>(key_domain);
      if (covered >= _plan.domain_coverage_threshold()) {
        SIRIUS_LOG_DEBUG(
          "[sirius_physical_hash_join] multi-batch publish gate: key {}: {} build rows cover "
          "{:.2f} of key domain (~{} rows) -> drop built filter.",
          k,
          _accumulated_rows,
          covered,
          key_domain);
        continue;
      }
    }
    per_key_membership[k] = _keys[k].bloom;
    per_key_build_type[k] = _keys[k].build_type;
    _published_membership.emplace_back(k, per_key_membership[k]);
    ++built;
  }
  if (built == 0) {
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_hash_join] multi-batch dynamic filter: no key survived the publication "
      "gates ({} batches, {} rows).",
      _accumulated_batches,
      _accumulated_rows);
    return;
  }

  stream.synchronize();
  replicate_filters(_plan, per_key_zone_map, per_key_membership);
  auto const pushed =
    fan_out_filters(_plan, per_key_zone_map, per_key_membership, per_key_build_type);
  SIRIUS_LOG_INFO(
    "[sirius_physical_hash_join] Pushed {} dynamic filter(s) across {} active target(s) of {} "
    "wired target(s) (multi-batch: {} build rows over {} batches, {} keys).",
    pushed.total_pushed,
    pushed.active_targets,
    _plan.probe_targets().size(),
    _accumulated_rows,
    _accumulated_batches,
    _filter_pushdown.join_condition.size());
}

}  // namespace sirius::op
