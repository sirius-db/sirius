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

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op {

//===----------dynamic_filter_publication_session::state----------===//
struct dynamic_filter_publication_session::state {
  enum class phase {
    open,        // A producer may still register on the plan's channels
    publishing,  // A producer has observed a whole-build delivery and is publishing filters on it
                 // (exclusively)
    terminal  // The session has completed, cancelled, or failed; no further producer registration
  };

  explicit state(dynamic_filter_publish_plan value, dynamic_filter_stats* sink)
    : plan(std::move(value)), stats(sink)
  {
    // For each target channel in the plan, register a producer and retain the handle
    producers.reserve(plan.probe_targets().size());
    channels.reserve(plan.probe_targets().size());
    for (auto const& target : plan.probe_targets()) {
      std::vector<std::size_t> columns;
      columns.reserve(target.key_bindings.size());
      for (auto const& binding : target.key_bindings) {
        columns.push_back(binding.channel_push_ordinal);
      }
      producers.push_back(target.filter_set->register_producer(std::move(columns)));
      channels.push_back(target.filter_set);
    }
    if (stats && plan.enabled()) {
      stats->producers_enabled.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void seal() noexcept
  {
    if (sealed) { return; }
    for (auto const& channel : channels) {
      channel->freeze_registration();
    }
    sealed = true;
  }

  void complete(sirius_dynamic_filter_set::completion result, bool account_attempt = false) noexcept
  {
    if (current == phase::terminal) { return; }
    for (auto const& producer : producers) {
      producer.finish(result);
    }
    current = phase::terminal;
    if (!stats || !account_attempt) { return; }
    if (result == sirius_dynamic_filter_set::completion::published ||
        result == sirius_dynamic_filter_set::completion::skipped) {
      stats->publications_finished.fetch_add(1, std::memory_order_relaxed);
    } else {
      // result == completion::cancelled or completion::failed
      stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void record(dynamic_filter_publication_outcome const& outcome) noexcept
  {
    if (!stats) { return; }
    auto const relaxed = std::memory_order_relaxed;
    stats->keys_considered.fetch_add(outcome.keys_considered, relaxed);
    stats->keys_with_known_domain.fetch_add(outcome.keys_with_known_domain, relaxed);
    stats->keys_skipped_domain_gate.fetch_add(outcome.keys_skipped_domain_gate, relaxed);
    stats->keys_skipped_type_mismatch.fetch_add(outcome.keys_skipped_type_mismatch, relaxed);
    stats->keys_build_exceeded_domain.fetch_add(outcome.keys_build_exceeded_domain, relaxed);
    stats->membership_filters_built.fetch_add(outcome.membership_filters_built, relaxed);
    stats->zone_map_filters_built.fetch_add(outcome.zone_map_filters_built, relaxed);
    stats->publications_skipped_targets_drained.fetch_add(outcome.skipped_targets_drained, relaxed);
    stats->filters_pushed.fetch_add(outcome.filters_pushed, relaxed);
  }

  std::mutex mutex;
  dynamic_filter_publish_plan plan;
  dynamic_filter_stats* stats;
  std::vector<sirius_dynamic_filter_set::producer> producers;
  std::vector<std::shared_ptr<sirius_dynamic_filter_set>> channels;
  phase current       = phase::open;
  bool sealed         = false;
  bool input_closed   = false;
  bool cancelled      = false;
  bool fanout_started = false;
};

//===----------dynamic_filter_publication_session----------===//
dynamic_filter_publication_session::dynamic_filter_publication_session(
  dynamic_filter_publish_plan plan, dynamic_filter_stats* stats)
  : _state(std::make_shared<state>(std::move(plan), stats))
{
}

dynamic_filter_publication_session::~dynamic_filter_publication_session() { cancel(); }

dynamic_filter_publish_plan const& dynamic_filter_publication_session::plan() const noexcept
{
  return _state->plan;
}

void dynamic_filter_publication_session::restrict_replicas_to(
  std::vector<int> const& admitted_gpu_ids)
{
  std::scoped_lock lock(_state->mutex);
  if (_state->sealed) {
    throw std::logic_error(
      "[dynamic_filter_publication_session::restrict_replicas_to] replica placement cannot change "
      "during execution");
  }
  _state->plan.restrict_replicas_to(admitted_gpu_ids);
  if (!_state->plan.enabled()) { _state->complete(sirius_dynamic_filter_set::completion::skipped); }
}

void dynamic_filter_publication_session::seal_plan() noexcept
{
  std::scoped_lock lock(_state->mutex);
  _state->seal();
}

void dynamic_filter_publication_session::finish_input() noexcept
{
  auto operation = _state;
  std::scoped_lock lock(operation->mutex);
  operation->seal();
  operation->input_closed = true;
  if (operation->current == state::phase::open) {
    operation->complete(sirius_dynamic_filter_set::completion::skipped);
  }
}

void dynamic_filter_publication_session::cancel() noexcept
{
  auto operation = _state;
  std::scoped_lock lock(operation->mutex);
  operation->input_closed = true;
  operation->cancelled    = true;
  if (operation->current == state::phase::open) {
    operation->complete(sirius_dynamic_filter_set::completion::cancelled);
  }
}

void dynamic_filter_publication_session::observe_whole_build(
  complete_build_delivery const& delivery, std::function<void()> const& deposit)
{
  auto operation = _state;
  bool claimed   = false;
  {
    std::scoped_lock lock(operation->mutex);
    operation->seal();
    if (operation->current == state::phase::open && operation->plan.enabled() && delivery._batch) {
      operation->current = state::phase::publishing;
      claimed            = true;
      if (operation->stats) {
        operation->stats->publication_attempts.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  if (!claimed) {
    deposit();
    return;
  }

  // Repository delivery is mandatory, so its allocation failures must not fail open.
  auto source = [&] {
    try {
      auto pinned = delivery._batch->to_read_only();
      deposit();
      return pinned;
    } catch (...) {
      std::scoped_lock lock(operation->mutex);
      operation->complete(sirius_dynamic_filter_set::completion::failed, true);
      throw;
    }
  }();

  try {
    nvtx_scoped_range range{"dynfilter::publish_hook"};
    auto* space = source.get_data() ? source.get_memory_space() : nullptr;
    if (!space || source.get_current_tier() != cucascade::memory::Tier::GPU ||
        !operation->plan.has_replica_on_device(space->get_device_id())) {
      std::scoped_lock lock(operation->mutex);
      if (operation->stats) {
        operation->stats->publications_skipped_source_not_resident.fetch_add(
          1, std::memory_order_relaxed);
      }
      if (operation->input_closed) {
        operation->complete(operation->cancelled ? sirius_dynamic_filter_set::completion::cancelled
                                                 : sirius_dynamic_filter_set::completion::skipped);
      } else {
        operation->current = state::phase::open;
      }
      return;
    }

    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space->get_device_id()}};
    ::cuda::stream_ref stream = space->acquire_stream();
    auto const writer         = source.get_writer_event();
    auto const ready =
      writer ? cudaStreamWaitEvent(stream.get(), writer, 0) : cudaDeviceSynchronize();
    if (ready != cudaSuccess) {
      throw std::runtime_error(
        std::string(
          "[dynamic_filter_publication_session::observe_whole_build] source readiness failed: ") +
        cudaGetErrorString(ready));
    }

    dynamic_filter_publication_outcome outcome;
    try {
      outcome = publish_dynamic_filters(operation->plan,
                                        sirius::get_cudf_table_view(source),
                                        stream,
                                        operation->producers,
                                        [&operation] {
                                          std::scoped_lock lock(operation->mutex);
                                          if (operation->cancelled) { return false; }
                                          operation->fanout_started = true;
                                          return true;
                                        });
    } catch (...) {
      // The source pin must outlive every accepted read, including a partially built filter.
      auto error = std::current_exception();
      stream.sync();
      std::rethrow_exception(error);
    }

    std::scoped_lock lock(operation->mutex);
    operation->record(outcome);
    auto const result = operation->cancelled && !operation->fanout_started
                          ? sirius_dynamic_filter_set::completion::cancelled
                        : outcome.filters_pushed != 0
                          ? sirius_dynamic_filter_set::completion::published
                          : sirius_dynamic_filter_set::completion::skipped;
    operation->complete(result, true);
  } catch (rmm::out_of_memory const& error) {
    {
      std::scoped_lock lock(operation->mutex);
      operation->complete(sirius_dynamic_filter_set::completion::failed, true);
    }
    SIRIUS_LOG_WARN(
      "[publish_dynamic_filters] publication exhausted device memory; continuing without filters: "
      "{}",
      error.what());
  } catch (...) {
    std::scoped_lock lock(operation->mutex);
    operation->complete(sirius_dynamic_filter_set::completion::failed, true);
    throw;
  }
}

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

dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  ::cuda::stream_ref stream,
  std::span<sirius_dynamic_filter_set::producer const> producers,
  std::function<bool()> const& before_fanout)
{
  nvtx_scoped_range nvtx_range{"dynfilter::push_build_side"};
  assert(plan.enabled());
  if (producers.size() != plan.probe_targets().size()) {
    throw std::invalid_argument(
      "[publish_dynamic_filters] publication requires one right per target");
  }
  dynamic_filter_publication_outcome outcome;

  if (build_view.num_rows() == 0) {
    SIRIUS_LOG_DEBUG("[publish_dynamic_filters] Skipping dynamic filter push: empty build table.");
    return outcome;
  }

  auto target_accepts_filters = [](dynamic_filter_publish_plan::probe_target const& tgt) {
    return tgt.filter_set && tgt.filter_set->accepting_filters();
  };
  auto const& probe_targets = plan.probe_targets();
  if (std::none_of(probe_targets.begin(), probe_targets.end(), target_accepts_filters)) {
    SIRIUS_LOG_DEBUG(
      "[publish_dynamic_filters] Skipping dynamic filter push: all target scans drained.");
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

  try {
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
          "[publish_dynamic_filters] publish gate: key {}: build {} rows cover {:.2f} of key "
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
          "[publish_dynamic_filters] An admitted key's build ordinal lies outside the runtime "
          "build "
          "table");
      }
      auto const& col = build_view.column(admitted_key.build_key_ordinal);
      if (col.type() != admitted_key.storage_type) {
        SIRIUS_LOG_WARN(
          "[publish_dynamic_filters] dynamic filter key {}: skipped (plan recorded type id {} "
          "but "
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
        nvtx_scoped_range vr{"dynfilter::build_zone_map"};
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
         .supports_bloom = sirius::op::sirius_dynamic_bloom_filter::supports(col.type())});

      char const* choice = "none";
      switch (chosen) {
        case membership_filter_kind::small_in_list: {
          nvtx_scoped_range vr{"dynfilter::build_small_in_list"};
          per_key_membership[admitted_key_index] =
            std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
              col, stream, allocator_ref);
          choice = "small_in_list";
          break;
        }
        case membership_filter_kind::hash_in_list: {
          nvtx_scoped_range vr{"dynfilter::build_in_list"};
          per_key_membership[admitted_key_index] =
            std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(col, stream, allocator_ref);
          choice = "in_list";
          break;
        }
        case membership_filter_kind::bloom: {
          nvtx_scoped_range vr{"dynfilter::build_bloom"};
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
        "[publish_dynamic_filters] dynamic filter key {}: build_rows={} zone_map={} membership: "
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
      stream.sync();

      nvtx_scoped_range replicate_range{"dynfilter::replicate_devices"};
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

    // Permission to publish is granted by before_fanout(). This allows the caller to cancel
    // publication after construction and replication, but before the first push to any target
    // channel, providing a mechanism to separate replication from publication.
    if (before_fanout && !before_fanout()) { return outcome; }

    std::size_t total_pushed   = 0;
    std::size_t active_targets = 0;
    for (std::size_t target_index = 0; target_index < probe_targets.size(); ++target_index) {
      auto const& tgt = probe_targets[target_index];
      if (!target_accepts_filters(tgt)) { continue; }
      ++active_targets;
      ++outcome.active_targets;

      for (auto const& binding : tgt.key_bindings) {
        assert(binding.admitted_key_index < admitted_keys.size());
        auto const& zone_map = per_key_zone_map[binding.admitted_key_index];
        if (zone_map && tgt.accepts_zone_map_filters &&
            binding.probe_storage_type == per_key_build_type[binding.admitted_key_index] &&
            producers[target_index].push_filter(binding.channel_push_ordinal, zone_map)) {
          ++total_pushed;
        }
        auto const& membership = per_key_membership[binding.admitted_key_index];
        if (membership &&
            producers[target_index].push_filter(binding.channel_push_ordinal, membership)) {
          ++total_pushed;
        }
      }
    }
    SIRIUS_LOG_INFO(
      "[publish_dynamic_filters] dynamic-filter publication: pushed {} dynamic filter(s) "
      "across {} active target(s) of {} wired target(s) ({} build rows, {} bound keys of {} "
      "admitted).",
      total_pushed,
      active_targets,
      probe_targets.size(),
      build_view.num_rows(),
      outcome.keys_considered,
      admitted_keys.size());
    outcome.filters_pushed = total_pushed;
    return outcome;
  } catch (...) {
    // Retire accepted work before the outer filter owners are destroyed during unwinding.
    auto error = std::current_exception();
    stream.sync();
    std::rethrow_exception(error);
  }
}

}  // namespace sirius::op
