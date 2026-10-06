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
#include "helper/numeric_narrowing.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

using completion = sirius_dynamic_filter_set::completion;

/// A field of the publication counters; a reason an attempt ended is counted by setting it to one.
using counter_field = std::uint64_t dynamic_filter_stats_snapshot::*;

enum class accumulation_failure : std::uint8_t { ADMISSION, TRANSIENT, ERROR, LEAK };

accumulation_failure classify_failure(std::exception_ptr error) noexcept
{
  try {
    std::rethrow_exception(std::move(error));
  } catch (detail::unjoined_gpu_work const&) {
    return accumulation_failure::LEAK;
  } catch (detail::accumulation_cuda_error const& e) {
    return e.transient_launch_failure() ? accumulation_failure::TRANSIENT
                                        : accumulation_failure::ERROR;
  } catch (std::bad_alloc const&) {
    return accumulation_failure::ADMISSION;
  } catch (...) {
    return accumulation_failure::ERROR;
  }
}

bool recoverable(accumulation_failure kind) noexcept
{ return kind == accumulation_failure::ADMISSION || kind == accumulation_failure::TRANSIENT; }

void count_failure(dynamic_filter_stats_snapshot& outcome, accumulation_failure kind) noexcept
{
  switch (kind) {
    case accumulation_failure::ADMISSION: outcome.accumulations_skipped_admission = 1; break;
    case accumulation_failure::TRANSIENT: outcome.accumulations_skipped_transient = 1; break;
    case accumulation_failure::LEAK: outcome.accumulation_storage_leaks = 1; [[fallthrough]];
    case accumulation_failure::ERROR: outcome.accumulations_skipped_error = 1; break;
  }
}

/**
 * @brief The probe storage type of every binding, per admitted key of @p plan; a key that no
 * binding reads has none.
 */
std::vector<std::vector<cudf::data_type>> binding_probe_types(
  dynamic_filter_publish_plan const& plan)
{
  std::vector<std::vector<cudf::data_type>> probe_types(plan.admitted_keys().size());
  for (auto const& target : plan.probe_targets()) {
    for (auto const& binding : target.key_bindings) {
      probe_types[binding.admitted_key_index].push_back(binding.probe_storage_type);
    }
  }
  return probe_types;
}

/**
 * @brief Counts a bound @p key of a build with @p build_rows rows in `keys_considered`,
 * `keys_with_known_domain` and `keys_build_exceeded_domain`.
 *
 * @return Whether the domain coverage gate skips the key; the caller counts the skip where its own
 * checks order it
 */
bool consider_bound_key(dynamic_filter_stats_snapshot& counts,
                        dynamic_filter_publish_plan const& plan,
                        dynamic_filter_publish_plan::admitted_key const& key,
                        std::size_t build_rows) noexcept
{
  ++counts.keys_considered;
  auto const key_domain = key.build_key_domain_cardinality;
  if (key_domain != 0) {
    ++counts.keys_with_known_domain;
    if (build_rows > key_domain) { ++counts.keys_build_exceeded_domain; }
  }
  return domain_coverage_gate_fires(
    build_rows, key_domain, key.build_key_proven_unique, plan.domain_coverage_threshold());
}

/**
 * @brief The filters one admitted key publishes; a null filter publishes nothing.
 */
struct key_filters {
  std::shared_ptr<sirius_dynamic_filter> zone_map;
  /// The type the zone map bounds carry: the key's recorded storage type.
  cudf::data_type zone_map_type{cudf::type_id::EMPTY};
  std::shared_ptr<sirius_dynamic_filter> membership;
  /// The domain the membership filter was built over.
  std::optional<membership_key_domain> membership_domain;
};

struct fanout_counts {
  std::size_t active_targets      = 0;
  std::size_t pushed              = 0;
  std::size_t incompatible_probes = 0;
};

/**
 * @brief Pushes @p filters, indexed by admitted key, to every binding of every target of @p plan
 * that still accepts filters; the fan-out shared by whole-build and accumulated publication.
 *
 * A zone map lowers to literals of its bound type against the probe column, so it goes only to a
 * binding of a zone-map-accepting target that probes at exactly that type (an EMPTY probe type
 * suppresses it). A membership filter converts per element, so it serves any probe carrier its key
 * domain accepts (`membership_probe_compatible`, the rule `compute_mask` applies); a binding whose
 * recorded probe type the domain cannot read would decline every batch, so it gets no filter, while
 * an EMPTY probe type (no cuDF mapping recorded) is left to the runtime check. Every push succeeds
 * or fails without allocating, so a fan-out cannot stop part way.
 *
 * @param on_first_push Invoked once, when a channel first accepts a filter
 */
template <std::invocable OnFirstPush>
fanout_counts fan_out(dynamic_filter_publish_plan const& plan,
                      std::span<sirius_dynamic_filter_set::producer const> producers,
                      std::span<key_filters const> filters,
                      OnFirstPush&& on_first_push) noexcept
{
  fanout_counts counts;
  auto const push = [&](sirius_dynamic_filter_set::producer const& producer,
                        std::size_t ordinal,
                        std::shared_ptr<sirius_dynamic_filter> const& filter) noexcept {
    if (!producer.push_filter(ordinal, filter)) { return; }
    if (counts.pushed++ == 0) { std::invoke(on_first_push); }
  };
  auto const& probe_targets = plan.probe_targets();
  for (std::size_t target_index = 0; target_index < probe_targets.size(); ++target_index) {
    auto const& target = probe_targets[target_index];
    if (!target.filter_set || !target.filter_set->accepting_filters()) { continue; }
    ++counts.active_targets;
    auto const& producer = producers[target_index];
    for (auto const& binding : target.key_bindings) {
      auto const& key = filters[binding.admitted_key_index];
      if (key.zone_map && target.accepts_zone_map_filters &&
          binding.probe_storage_type == key.zone_map_type) {
        push(producer, binding.channel_push_ordinal, key.zone_map);
      }
      if (!key.membership) { continue; }
      if (binding.probe_storage_type.id() != cudf::type_id::EMPTY &&
          key.membership_domain.has_value() &&
          !membership_probe_compatible(*key.membership_domain, binding.probe_storage_type)) {
        sirius::log::log_noexcept([&] {
          SIRIUS_LOG_DEBUG(
            "[publish_dynamic_filters] dynamic filter key {}: membership filter not pushed to "
            "channel ordinal {}: probe storage type id {} is not a carrier of the key family.",
            binding.admitted_key_index,
            binding.channel_push_ordinal,
            static_cast<int32_t>(binding.probe_storage_type.id()));
        });
        ++counts.incompatible_probes;
        continue;
      }
      push(producer, binding.channel_push_ordinal, key.membership);
    }
  }
  return counts;
}

}  // namespace

//===----------------------------------------------------------------------===//
// dynamic_filter_publication_session::state
//===----------------------------------------------------------------------===//
struct dynamic_filter_publication_session::state {
  /**
   * @brief The phases of a dynamic filter publication session:
   * Healthy:
   * OPEN -> COLLECTING -> PUBLISHING -> TERMINAL
   *  |                        ^
   *  -------------------------|
   * Corrupt: any state may transition to TERMINAL.
   */
  enum class phase {
    OPEN,        // A producer may still register on the plan's channels
    COLLECTING,  // An accumulation accepts contributions from certified build batches
    PUBLISHING,  // One whole-build delivery or the final contributor's task owns publication
    TERMINAL  // The session has completed, cancelled, or failed; no further producer registration
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

  /**
   * @brief Finishes every producer with @p result and enters TERMINAL, once; with
   * `account_attempt`, also counts the claimed attempt as finished or failed. The caller holds the
   * mutex.
   */
  void complete(completion result, bool account_attempt = false) noexcept
  {
    if (current == phase::TERMINAL) { return; }
    for (auto const& producer : producers) {
      producer.finish(result);
    }
    current = phase::TERMINAL;
    if (!stats || !account_attempt) { return; }
    if (result == completion::PUBLISHED || result == completion::SKIPPED) {
      stats->publications_finished.fetch_add(1, std::memory_order_relaxed);
    } else {
      // result == completion::CANCELLED or completion::FAILED
      stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void record(dynamic_filter_stats_snapshot const& counts) noexcept
  {
    if (stats) { stats->add(counts); }
  }

  [[nodiscard]] bool targets_accepting() const noexcept
  {
    return std::ranges::any_of(channels,
                               [](auto const& channel) { return channel->accepting_filters(); });
  }

  /**
   * @brief Ends the attempt with @p result unless it already has one. The caller holds the mutex.
   */
  void end_attempt(completion result) noexcept
  {
    if (!accumulation_result) { accumulation_result = result; }
    input_closed = true;
  }

  /**
   * @brief Counts @p reason in the attempt's outcome, then ends the attempt with @p result. The
   * caller holds the mutex.
   */
  void end_attempt(completion result, counter_field reason) noexcept
  {
    outcome.*reason = 1;
    end_attempt(result);
  }

  /**
   * @brief Locks the mutex, then ends the attempt as skipped for @p reason.
   */
  void skip_locked(counter_field reason) noexcept
  {
    std::scoped_lock lock(mutex);
    end_attempt(completion::SKIPPED, reason);
  }

  /**
   * @brief Ends the attempt if cancellation or drained targets forbid a fan-out, and returns
   * whether it did. The caller holds the mutex.
   */
  [[nodiscard]] bool end_if_fanout_forbidden() noexcept
  {
    if (cancelled) {
      end_attempt(completion::CANCELLED);
      return true;
    }
    if (!targets_accepting()) {
      end_attempt(completion::SKIPPED,
                  &dynamic_filter_stats_snapshot::publications_skipped_targets_drained);
      return true;
    }
    return false;
  }

  /**
   * @brief Completes an ended accumulation once no operation is active, recording its outcome.
   */
  void settle() noexcept
  {
    std::optional<completion> settled;
    {
      std::scoped_lock lock(mutex);
      if (inventory && current != phase::TERMINAL && active_operations == 0 &&
          accumulation_result) {
        settled = accumulation_result;
        record(outcome);
        complete(*settled, true);
      }
    }
    if (settled) {
      sirius::log::log_noexcept([&] {
        SIRIUS_LOG_DEBUG("[dynamic_filter_publication_session] accumulation settled: outcome={}",
                         static_cast<int>(*settled));
      });
    }
  }

  /**
   * @brief Marks one contribution or publication active while claimed, and settles the session when
   * it ends. Every contribute() and publish_if_final() holds an active_operation guard. The last
   * activer operation settles the session, transitioning it to TERMINAL. This is deliberately
   * distinct from deciding the outcome: the decision (via end_attempt(result)) records the
   * accumulation_result, and then settle() ensures the producers finish.
   *
   * Transitions the session PUBLISHING -> TERMINAL when the last active operation ends.
   */
  struct active_operation {
    state& owner;
    bool claimed = false;

    explicit active_operation(state& value) noexcept : owner{value} {}
    active_operation(active_operation const&)            = delete;
    active_operation& operator=(active_operation const&) = delete;

    ~active_operation() noexcept
    {
      if (claimed) {
        std::scoped_lock lock(owner.mutex);
        --owner.active_operations;
      }
      owner.settle();
    }
  };

  /**
   * @brief Reduces and replicates the partials, then fans out. Runs on the task that owes the
   * publication.
   */
  void publish(cucascade::memory::memory_space const& task_space, ::cuda::stream_ref stream)
  {
    try {
      {
        std::scoped_lock lock(mutex);
        if (accumulation_result || end_if_fanout_forbidden()) { return; }
      }
      // The reduction runs on the task's stream, so the task's GPU is the root.
      if (task_space.get_tier() != cucascade::memory::Tier::GPU) {
        throw detail::accumulation_invariant_error{"publication task space must be a GPU space"};
      }
      auto const filters = builder->finish(rmm::cuda_device_id{task_space.get_device_id()},
                                           stream,
                                           task_space.get_default_allocator());
      if (!filters) {
        skip_locked(&dynamic_filter_stats_snapshot::accumulations_skipped_admission);
        return;
      }
      {
        std::scoped_lock lock(mutex);
        outcome.membership_filters_built = filters->size();
        if (end_if_fanout_forbidden()) { return; }
      }
      std::vector<key_filters> per_key(plan.admitted_keys().size());
      for (std::size_t index = 0; index < active_keys.size(); ++index) {
        auto const& filter    = (*filters)[index];
        auto& key             = per_key[active_keys[index]];
        key.membership        = filter;
        key.membership_domain = filter->domain();
      }
      std::optional<std::chrono::steady_clock::duration> latency;
      auto const counts = fan_out(plan, producers, per_key, [&] {
        latency = std::chrono::steady_clock::now() - final_commit;
        nvtx3::mark_in<nvtx_domain>("dynfilter::accum::visible");
      });
      sirius::log::log_noexcept([&] {
        SIRIUS_LOG_INFO(
          "[dynamic_filter_publication_session] accumulated Bloom published: {} filter(s) pushed "
          "to {} target(s), {:.1f} ms after the final contribution.",
          counts.pushed,
          counts.active_targets,
          latency ? std::chrono::duration<double, std::milli>(*latency).count() : 0.0);
      });
      bool const published = counts.pushed != 0;
      std::scoped_lock lock(mutex);
      outcome.filters_pushed                      = counts.pushed;
      outcome.bindings_skipped_incompatible_probe = counts.incompatible_probes;
      outcome.accumulation_publications_finished  = published ? 1 : 0;
      if (latency) {
        outcome.accumulation_publication_latency_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(*latency).count());
      }
      end_attempt(published ? completion::PUBLISHED : completion::SKIPPED);
    } catch (...) {
      // Nothing was pushed: a fan-out cannot fail.
      auto const error = std::current_exception();
      auto const kind  = classify_failure(error);
      {
        std::scoped_lock lock(mutex);
        count_failure(outcome, kind);
        end_attempt(recoverable(kind) ? completion::SKIPPED : completion::FAILED);
      }
      // Optional OOM must not escape into gpu_pipeline_task's reschedule handler after the pending
      // publication was consumed. The builder has joined accepted work or raised a fatal cleanup
      // error.
      if (!recoverable(kind)) { std::rethrow_exception(error); }
    }
  }

  /**
   * @brief Ends an owed publication that its task never ran (the query failed before the task
   * returned or was retried). The caller holds the mutex.
   */
  void abandon_pending_publication() noexcept
  {
    if (!std::exchange(pending_publication, std::nullopt).has_value()) { return; }
    if (cancelled) {
      end_attempt(completion::CANCELLED);
    } else {
      if (!accumulation_result) { outcome.accumulations_abandoned = 1; }
      end_attempt(completion::SKIPPED);
    }
  }

  std::mutex mutex;
  dynamic_filter_publish_plan plan;
  dynamic_filter_stats* stats;
  std::vector<sirius_dynamic_filter_set::producer> producers;
  std::vector<std::shared_ptr<sirius_dynamic_filter_set>> channels;
  // Holds a value from the start of an accumulation on.
  std::optional<complete_build_inventory> inventory;
  std::vector<bool> claimed;  // One entry per inventory batch, in inventory order
  std::vector<std::size_t> active_keys;
  std::optional<detail::accumulated_bloom_builder> builder;
  dynamic_filter_stats_snapshot outcome;  // The accumulation's counts, recorded when it settles
  std::optional<completion> accumulation_result;
  std::chrono::steady_clock::time_point final_commit{};
  std::size_t active_operations = 0;
  // The ID of the batch whose contribution completed the inventory, while its task's publication is
  // owed and has not run.
  std::optional<std::uint64_t> pending_publication;
  phase current       = phase::OPEN;
  bool sealed         = false;
  bool input_closed   = false;
  bool cancelled      = false;
  bool fanout_started = false;
};

//===----------------------------------------------------------------------===//
// dynamic_filter_publication_session
//===----------------------------------------------------------------------===//
void dynamic_filter_publication_session::decline_accumulation() noexcept
{
  auto& operation = *_state;
  bool declined   = false;
  {
    std::scoped_lock lock(operation.mutex);
    if (operation.current == state::phase::COLLECTING && !operation.accumulation_result) {
      operation.end_attempt(completion::SKIPPED,
                            &dynamic_filter_stats_snapshot::accumulations_skipped_inventory);
      declined = true;
    }
  }
  if (declined) {
    sirius::log::log_noexcept([] {
      SIRIUS_LOG_INFO(
        "[dynamic_filter_publication_session] accumulated Bloom declined: "
        "a task input is not exactly one certified batch.");
    });
  }
  operation.settle();
}

void dynamic_filter_publication_session::release_retired_storage() noexcept
{
  auto& operation = *_state;
  std::optional<detail::accumulated_bloom_builder> released;
  {
    std::scoped_lock lock(operation.mutex);
    if (operation.current == state::phase::TERMINAL && operation.active_operations == 0 &&
        operation.builder && operation.builder->releasable_here()) {
      released = std::move(operation.builder);
      operation.builder.reset();
    }
  }
}

void dynamic_filter_publication_session::finalize_input() noexcept
{
  finish_input();
  release_retired_storage();
}

dynamic_filter_publication_session::dynamic_filter_publication_session(
  dynamic_filter_publish_plan plan, dynamic_filter_stats* stats)
  : _state(std::make_unique<state>(std::move(plan), stats))
{
}

dynamic_filter_publication_session::~dynamic_filter_publication_session() { cancel(); }

dynamic_filter_publish_plan const& dynamic_filter_publication_session::plan() const noexcept
{ return _state->plan; }

void dynamic_filter_publication_session::restrict_replicas_to(
  std::vector<int> const& admitted_gpu_ids)
{
  auto& operation = *_state;
  std::scoped_lock lock(operation.mutex);
  if (operation.sealed) {
    throw std::logic_error(
      "[dynamic_filter_publication_session::restrict_replicas_to] replica placement cannot change "
      "during execution");
  }
  operation.plan.restrict_replicas_to(admitted_gpu_ids);
  if (!operation.plan.enabled()) { operation.complete(completion::SKIPPED); }
}

void dynamic_filter_publication_session::seal_plan() noexcept
{
  std::scoped_lock lock(_state->mutex);
  _state->seal();
}

bool dynamic_filter_publication_session::try_begin_accumulation(
  std::optional<complete_build_inventory> inventory)
{
  auto& operation = *_state;
  nvtx_scoped_range range{"dynfilter::accum::begin"};
  dynamic_filter_stats_snapshot selection;
  // Every exit, including a failure, records the selection's counts.
  absl::Cleanup const record_selection = [&] { operation.record(selection); };
  try {
    {
      std::scoped_lock lock(operation.mutex);
      if (operation.current != state::phase::OPEN || operation.input_closed ||
          !operation.plan.multi_partition_enabled()) {
        return false;
      }
      // Replica placement is final from here on, so the plan can be read without the mutex.
      operation.seal();
    }
    if (!inventory) {
      selection.accumulations_skipped_inventory = 1;
      return false;
    }
    auto const& plan          = operation.plan;
    auto const& admitted_keys = plan.admitted_keys();
    auto const probe_types    = binding_probe_types(plan);
    auto const rows           = inventory->total_rows();
    std::vector<detail::accumulated_bloom_builder::key> keys;
    std::vector<std::size_t> active_keys;
    for (std::size_t index = 0; index < probe_types.size(); ++index) {
      auto const& bindings = probe_types[index];
      if (bindings.empty()) { continue; }
      auto const& key  = admitted_keys[index];
      bool const gated = consider_bound_key(selection, plan, key, rows);
      if (key.build_key_ordinal < 0 || inventory->consistent_type_at(static_cast<std::size_t>(
                                         key.build_key_ordinal)) != key.storage_type) {
        ++selection.keys_skipped_type_mismatch;
        continue;
      }
      if (gated) {
        ++selection.keys_skipped_domain_gate;
        continue;
      }
      if (!detail::accumulated_bloom_builder::supports(key.storage_type)) {
        ++selection.keys_skipped_bloom_unsupported;
        continue;
      }
      // The fan-out's rule, applied before any array is allocated: a key whose domain can read
      // none of its bindings' probe types would publish only filters that decline every batch.
      auto const domain   = classify_membership_key(key.storage_type);
      auto const readable = [&](cudf::data_type probe) {
        return probe.id() == cudf::type_id::EMPTY ||
               (domain && membership_probe_compatible(*domain, probe));
      };
      if (std::ranges::none_of(bindings, readable)) {
        selection.bindings_skipped_incompatible_probe += bindings.size();
        continue;
      }
      if (rows == 0) { continue; }
      active_keys.push_back(index);
      keys.push_back({key.build_key_ordinal, key.storage_type});
    }
    if (keys.empty()) { return false; }
    auto const geometry = detail::accumulated_bloom_geometry::try_create(
      rows, keys.size(), plan.max_bloom_bytes_per_gpu());
    if (!geometry) {
      selection.keys_skipped_bloom_size_gate = keys.size();
      auto const required                    = detail::accumulated_bloom_geometry::try_create(
        rows, keys.size(), std::numeric_limits<std::uint64_t>::max());
      SIRIUS_LOG_INFO(
        "[dynamic_filter_publication_session] accumulated Bloom refused: per-GPU cap, rows={}, "
        "active_keys={}, required_bytes={}, cap={}",
        rows,
        keys.size(),
        required ? std::to_string(required->arrays_bytes) : "overflow",
        plan.max_bloom_bytes_per_gpu());
      return false;
    }

    auto const& replicas = plan.replica_spaces();
    if (replicas.empty()) {
      selection.accumulations_skipped_admission = 1;
      return false;
    }
    std::vector<int> replica_devices;
    replica_devices.reserve(replicas.size());
    for (auto const& replica : replicas) {
      replica_devices.push_back(replica.get_gpu_space().get_device_id());
    }
    if (auto const broken = find_pair_without_peer_dma(replica_devices)) {
      SIRIUS_LOG_INFO(
        "[dynamic_filter_publication_session] accumulated Bloom skipped: no working peer DMA "
        "from GPU {} to GPU {}.",
        broken->first,
        broken->second);
      selection.accumulations_skipped_admission = 1;
      return false;
    }

    // Allocation happens outside the mutex; the builder is installed only if nothing closed the
    // session meanwhile.
    auto built = detail::accumulated_bloom_builder::try_create(keys, *geometry, replicas);
    if (!built) {
      SIRIUS_LOG_DEBUG(
        "[dynamic_filter_publication_session] accumulated Bloom skipped: partial-array lease "
        "refused.");
      selection.accumulations_skipped_admission = 1;
      return false;
    }
    auto const expected = inventory->batches().size();
    std::vector<bool> claimed(expected, false);
    {
      std::scoped_lock lock(operation.mutex);
      if (operation.current != state::phase::OPEN || operation.input_closed) {
        return false;  // `built` is released here, on the untracked creator thread.
      }
      operation.claimed = std::move(claimed);
      operation.inventory.emplace(std::move(*inventory));
      operation.active_keys = std::move(active_keys);
      operation.builder     = std::move(built);
      operation.current     = state::phase::COLLECTING;
      // Start counters are visible at once; the attempt's outcome is recorded when it settles.
      selection.accumulations_started               = 1;
      selection.accumulation_expected_contributions = expected;
      if (operation.stats) {
        operation.stats->publication_attempts.fetch_add(1, std::memory_order_relaxed);
      }
    }
    sirius::log::log_noexcept([&] {
      SIRIUS_LOG_INFO(
        "[dynamic_filter_publication_session] accumulated Bloom started: {} key(s), {} batch(es), "
        "{} array bytes per GPU on {} GPU(s), {}-byte transfer chunks.",
        keys.size(),
        expected,
        geometry->arrays_bytes,
        replicas.size(),
        geometry->chunk_bytes);
    });
    return true;
  } catch (...) {
    auto const error = std::current_exception();
    auto const kind  = classify_failure(error);
    count_failure(selection, kind);
    if (!recoverable(kind)) { std::rethrow_exception(error); }
    return false;
  }
}

bool dynamic_filter_publication_session::accumulation_claimed() const noexcept
{
  std::scoped_lock lock(_state->mutex);
  return _state->inventory.has_value();
}

void dynamic_filter_publication_session::contribute(std::uint64_t original_id,
                                                    cucascade::read_only_data_batch const& source,
                                                    ::cuda::stream_ref stream)
{
  auto& operation = *_state;
  state::active_operation active{operation};
  try {
    auto const* gpu   = dynamic_cast<cucascade::gpu_table_representation const*>(source.get_data());
    auto const* space = source.get_memory_space();
    std::optional<cudf::table_view> view;
    {
      std::scoped_lock lock(operation.mutex);
      if (!operation.inventory) { return; }
      auto const* entry = operation.inventory->find(original_id);
      auto const index = entry != nullptr
                           ? static_cast<std::size_t>(entry - operation.inventory->batches().data())
                           : 0;
      // Retry identity precedes representation validation: a completed batch can have been
      // spilled or cloned since its first contribution.
      if (entry != nullptr && operation.claimed[index]) {
        if (operation.current != state::phase::TERMINAL) {
          ++operation.outcome.accumulation_duplicate_contributions;
        } else if (operation.stats) {
          operation.stats->accumulation_duplicate_contributions.fetch_add(
            1, std::memory_order_relaxed);
        }
        return;
      }
      // While collecting, a closed input or cancellation has already ended the attempt.
      if (operation.current != state::phase::COLLECTING || operation.accumulation_result) {
        return;
      }
      if (!operation.targets_accepting()) {
        operation.end_attempt(completion::SKIPPED,
                              &dynamic_filter_stats_snapshot::publications_skipped_targets_drained);
        return;
      }
      // Built only after the retry and phase checks: a duplicate must not allocate on the host.
      if (gpu != nullptr) { view.emplace(gpu->get_table_view()); }
      // A late batch without rows is not in the inventory and adds no key.
      if (entry == nullptr && view && view->num_rows() == 0) { return; }
      if (entry != nullptr) { operation.claimed[index] = true; }
      bool const valid = entry != nullptr && view && space != nullptr &&
                         space->get_tier() == cucascade::memory::Tier::GPU &&
                         std::cmp_equal(view->num_rows(), entry->rows) &&
                         std::ranges::all_of(operation.active_keys, [&](std::size_t key_index) {
                           auto const& key = operation.plan.admitted_keys()[key_index];
                           return key.build_key_ordinal >= 0 &&
                                  key.build_key_ordinal < view->num_columns() &&
                                  view->column(key.build_key_ordinal).type() == key.storage_type;
                         });
      if (!valid) {
        operation.end_attempt(completion::SKIPPED,
                              &dynamic_filter_stats_snapshot::accumulations_skipped_inventory);
        return;
      }
      ++operation.active_operations;
      active.claimed = true;
    }
    if (!operation.builder->enqueue_add(
          *view, rmm::cuda_device_id{space->get_device_id()}, stream)) {
      operation.skip_locked(&dynamic_filter_stats_snapshot::accumulations_skipped_admission);
      return;
    }
    std::scoped_lock lock(operation.mutex);
    if (++operation.outcome.accumulation_completed_contributions == operation.claimed.size() &&
        !operation.accumulation_result) {
      operation.current             = state::phase::PUBLISHING;
      operation.pending_publication = original_id;
      operation.final_commit        = std::chrono::steady_clock::now();
      nvtx3::mark_in<nvtx_domain>("dynfilter::accum::final_commit");
    }
  } catch (...) {
    auto const error = std::current_exception();
    auto const kind  = classify_failure(error);
    {
      std::scoped_lock lock(operation.mutex);
      count_failure(operation.outcome, kind);
      operation.end_attempt(recoverable(kind) ? completion::SKIPPED : completion::FAILED);
    }
    if (!recoverable(kind)) { std::rethrow_exception(error); }
  }
}

void dynamic_filter_publication_session::publish_if_final(
  std::uint64_t original_id,
  cucascade::memory::memory_space const& task_space,
  ::cuda::stream_ref stream)
{
  auto& operation = *_state;
  state::active_operation active{operation};
  {
    std::scoped_lock lock(operation.mutex);
    if (operation.pending_publication != original_id || operation.accumulation_result) { return; }
    operation.pending_publication.reset();
    ++operation.active_operations;
    active.claimed = true;
  }
  operation.publish(task_space, stream);
}

void dynamic_filter_publication_session::observe_whole_build(
  complete_build_delivery const& delivery, std::function<void()> const& deposit)
{
  auto& operation = *_state;
  bool claimed    = false;
  {
    std::scoped_lock lock(operation.mutex);
    operation.seal();
    if (operation.current == state::phase::OPEN && operation.plan.enabled() && delivery._batch) {
      operation.current = state::phase::PUBLISHING;
      claimed           = true;
      if (operation.stats) {
        operation.stats->publication_attempts.fetch_add(1, std::memory_order_relaxed);
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
      std::scoped_lock lock(operation.mutex);
      operation.complete(completion::FAILED, true);
      throw;
    }
  }();

  try {
    nvtx_scoped_range range{"dynfilter::publish_hook"};
    auto* space = source.get_data() ? source.get_memory_space() : nullptr;
    if (!space || source.get_current_tier() != cucascade::memory::Tier::GPU ||
        !operation.plan.has_replica_on_device(space->get_device_id())) {
      std::scoped_lock lock(operation.mutex);
      if (operation.stats) {
        operation.stats->publications_skipped_source_not_resident.fetch_add(
          1, std::memory_order_relaxed);
      }
      if (operation.input_closed) {
        operation.complete(operation.cancelled ? completion::CANCELLED : completion::SKIPPED);
      } else {
        operation.current = state::phase::OPEN;
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
      outcome = publish_dynamic_filters(operation.plan,
                                        sirius::get_cudf_table_view(source),
                                        stream,
                                        operation.producers,
                                        [&operation] {
                                          std::scoped_lock lock(operation.mutex);
                                          if (operation.cancelled) { return false; }
                                          operation.fanout_started = true;
                                          return true;
                                        });
    } catch (...) {
      // The source pin must outlive every accepted read, including a partially built filter.
      auto error = std::current_exception();
      stream.sync();
      std::rethrow_exception(error);
    }

    std::scoped_lock lock(operation.mutex);
    operation.record(outcome);
    auto const result = operation.cancelled && !operation.fanout_started ? completion::CANCELLED
                        : outcome.filters_pushed != 0                    ? completion::PUBLISHED
                                                                         : completion::SKIPPED;
    operation.complete(result, true);
  } catch (rmm::out_of_memory const& error) {
    {
      std::scoped_lock lock(operation.mutex);
      operation.complete(completion::FAILED, true);
    }
    SIRIUS_LOG_WARN(
      "[publish_dynamic_filters] publication exhausted device memory; continuing without filters: "
      "{}",
      error.what());
  } catch (...) {
    std::scoped_lock lock(operation.mutex);
    operation.complete(completion::FAILED, true);
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

// Converts a zone-map bound to the recorded storage type through a one-row column, so DATE bounds
// take the same INT32 tunnel (cast_through_rep) the build column would.
std::unique_ptr<cudf::scalar> restore_scalar(cudf::scalar const& bound,
                                             cudf::data_type target,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const one      = cudf::make_column_from_scalar(bound, 1, stream, mr);
  auto const restored = sirius::cast_through_rep(one->view(), target, stream, mr);
  return cudf::get_element(restored->view(), 0, stream, mr);
}
}  // namespace

// 4. Whole-build publication: build the filters from one complete build, publish them to the
// probe targets, and record the outcome.
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
    outcome.publications_skipped_targets_drained = 1;
    return outcome;
  }

  auto const& admitted_keys = plan.admitted_keys();
  auto const probe_types    = binding_probe_types(plan);

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

  std::vector<key_filters> per_key(admitted_keys.size());
  // Build columns restored from a narrowed carrier; filters copy from them asynchronously on
  // `stream`, so they must outlive the synchronize below.
  std::vector<std::unique_ptr<cudf::column>> restored_build_columns;

  try {
    for (std::size_t admitted_key_index = 0; admitted_key_index < admitted_keys.size();
         ++admitted_key_index) {
      if (probe_types[admitted_key_index].empty()) { continue; }
      auto const& admitted_key = admitted_keys[admitted_key_index];
      auto& filters            = per_key[admitted_key_index];
      if (consider_bound_key(outcome, plan, admitted_key, build_rows)) {
        auto const key_domain = admitted_key.build_key_domain_cardinality;
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
      // The build column may arrive at a narrower carrier than the plan recorded: compressed
      // materialization casts a pinned column to the narrowest carrier its values fit, and that
      // carrier is restorable to the recorded type without changing any value. Any other
      // disagreement is a type-derivation bug and skips the key; the join stays authoritative.
      cudf::column_view col   = build_view.column(admitted_key.build_key_ordinal);
      auto const arrived_type = col.type();
      if (arrived_type != admitted_key.storage_type) {
        if (!sirius::can_restore_to(arrived_type, admitted_key.storage_type)) {
          SIRIUS_LOG_WARN(
            "[publish_dynamic_filters] dynamic filter key {}: skipped (plan recorded type id {} "
            "but build column {} carries type id {}).",
            admitted_key_index,
            static_cast<int32_t>(admitted_key.storage_type.id()),
            admitted_key.build_key_ordinal,
            static_cast<int32_t>(arrived_type.id()));
          ++outcome.keys_skipped_type_mismatch;
          continue;
        }
        // One rule for every key family. The membership filters classify on the runtime column,
        // so they are built at the carrier when it lands the key in the same family as the
        // recorded type (an INT16 carrier of a BIGINT key, a DECIMAL32 carrier of a DECIMAL64
        // key): every probe the recorded type accepts is comparable to the carrier-sized set and
        // range-checks into it, and no widened build copy is made. A carrier that changes the
        // family (a DATE stored as INT16 classifies as a signed integer, and a signed-integer set
        // would decline the native TIMESTAMP_DAYS probe) is restored to the recorded type first;
        // the cast is small (the build side) and costs no set width.
        if (!membership_same_family(admitted_key.storage_type, arrived_type)) {
          restored_build_columns.push_back(
            sirius::cast_through_rep(col, admitted_key.storage_type, stream, allocator_ref));
          col = restored_build_columns.back()->view();
          SIRIUS_LOG_DEBUG(
            "[publish_dynamic_filters] dynamic filter key {}: build column {} arrives at carrier "
            "type id {} outside the recorded type's key family; restored to type id {} for "
            "publication.",
            admitted_key_index,
            admitted_key.build_key_ordinal,
            static_cast<int32_t>(arrived_type.id()),
            static_cast<int32_t>(admitted_key.storage_type.id()));
        } else {
          SIRIUS_LOG_DEBUG(
            "[publish_dynamic_filters] dynamic filter key {}: build column {} arrives at "
            "narrowed carrier type id {} (plan recorded type id {}); building filters at the "
            "carrier.",
            admitted_key_index,
            admitted_key.build_key_ordinal,
            static_cast<int32_t>(arrived_type.id()),
            static_cast<int32_t>(admitted_key.storage_type.id()));
        }
      }

      // Zone-map bounds are lowered to AST literals compared against the consumer's column, which
      // carries the recorded storage type once decoded, so the bounds are reduced at whatever
      // carrier the build arrived at and then restored to the recorded type: two scalars, not the
      // whole column, so a narrowed build publishes the same zone map a native one does.
      filters.zone_map_type = admitted_key.storage_type;
      if (plan.emit_zone_map_filters() &&
          sirius::op::sirius_dynamic_zone_map_filter::supports(admitted_key.storage_type)) {
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
          if (col.type() != admitted_key.storage_type) {
            min_s = restore_scalar(*min_s, admitted_key.storage_type, stream, allocator_ref);
            max_s = restore_scalar(*max_s, admitted_key.storage_type, stream, allocator_ref);
          }
          std::vector<sirius::op::zone_map_entry> zones;
          zones.push_back({std::move(min_s), std::move(max_s)});
          filters.zone_map = std::make_shared<sirius::op::sirius_dynamic_zone_map_filter>(
            std::move(zones), true, true);
        }
      }
      filters.membership_domain = classify_membership_key(col.type());

      // Every membership filter compacts null build keys out (they match nothing under the join's
      // null_equality::UNEQUAL), so the representation is sized and chosen on the valid rows.
      auto const valid_rows = build_rows - static_cast<std::size_t>(col.null_count());
      auto const set_bytes =
        sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(valid_rows, col.type());
      auto const bloom_bytes = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(valid_rows);

      // The type gates below are necessary, not sufficient: a DECIMAL128 key sits on the int64 rep
      // only when its unscaled build values fit, which is a property of this build, not the type.
      // One min/max reduction here spares every filter's supports() from re-deriving it; an
      // unfitting build declines membership for the key while the zone map (exact at DECIMAL128)
      // still publishes.
      bool const fits_rep = membership_key_supported(col.type()) &&
                            membership_build_fits_rep(col, stream, allocator_ref);
      if (membership_key_supported(col.type()) && !fits_rep) {
        SIRIUS_LOG_DEBUG(
          "[publish_dynamic_filters] dynamic filter key {}: build values exceed the membership "
          "key rep (DECIMAL128 outside int64); membership filters declined.",
          admitted_key_index);
      }

      auto const chosen = choose_membership_filter(
        {.build_rows               = valid_rows,
         .l2_cache_bytes           = l2_bytes,
         .estimated_hash_set_bytes = set_bytes,
         .inlist_max_l2_fraction   = plan.inlist_max_l2_fraction(),
         .supports_small_in_list =
           fits_rep && sirius::op::sirius_dynamic_small_in_list_filter::supports(col),
         .supports_hash_in_list =
           fits_rep && sirius::op::sirius_dynamic_in_list_filter::supports(col),
         .supports_bloom =
           fits_rep && sirius::op::sirius_dynamic_bloom_filter::supports(col.type())});

      char const* choice = "none";
      switch (chosen) {
        case membership_filter_kind::small_in_list: {
          nvtx_scoped_range vr{"dynfilter::build_small_in_list"};
          filters.membership = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
            col, stream, allocator_ref);
          choice = "small_in_list";
          break;
        }
        case membership_filter_kind::hash_in_list: {
          nvtx_scoped_range vr{"dynfilter::build_in_list"};
          filters.membership =
            std::make_shared<sirius::op::sirius_dynamic_in_list_filter>(col, stream, allocator_ref);
          choice = "in_list";
          break;
        }
        case membership_filter_kind::bloom: {
          nvtx_scoped_range vr{"dynfilter::build_bloom"};
          filters.membership =
            std::make_shared<sirius::op::sirius_dynamic_bloom_filter>(col, stream, allocator_ref);
          choice = "bloom";
          break;
        }
        case membership_filter_kind::none: break;
      }
      if (filters.membership) { ++outcome.membership_filters_built; }
      if (filters.zone_map) { ++outcome.zone_map_filters_built; }
      SIRIUS_LOG_DEBUG(
        "[publish_dynamic_filters] dynamic filter key {}: build_rows={} (valid={}) zone_map={} "
        "membership: in_list_set={}B bloom={}B L2={}B inlist_max_l2_fraction={} -> {}",
        admitted_key_index,
        build_rows,
        valid_rows,
        filters.zone_map ? "yes" : "no",
        set_bytes,
        bloom_bytes,
        l2_bytes,
        plan.inlist_max_l2_fraction(),
        choice);
    }

    // Finish construction and replication before publishing to independent consumer streams.
    if (std::ranges::any_of(per_key, [](key_filters const& filters) {
          return filters.membership || filters.zone_map;
        })) {
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
      for (auto const& filters : per_key) {
        replicate(filters.zone_map);
      }
      for (auto const& filters : per_key) {
        replicate(filters.membership);
      }
    }

    // Permission to publish is granted by before_fanout(). This allows the caller to cancel
    // publication after construction and replication, but before the first push to any target
    // channel, providing a mechanism to separate replication from publication.
    if (before_fanout && !before_fanout()) { return outcome; }

    auto const counts = fan_out(plan, producers, per_key, [] {});
    SIRIUS_LOG_INFO(
      "[publish_dynamic_filters] dynamic-filter publication: pushed {} dynamic filter(s) "
      "across {} active target(s) of {} wired target(s) ({} build rows, {} bound keys of {} "
      "admitted).",
      counts.pushed,
      counts.active_targets,
      probe_targets.size(),
      build_view.num_rows(),
      outcome.keys_considered,
      admitted_keys.size());
    outcome.active_targets                      = counts.active_targets;
    outcome.filters_pushed                      = counts.pushed;
    outcome.bindings_skipped_incompatible_probe = counts.incompatible_probes;
    return outcome;
  } catch (...) {
    // Retire accepted work before the outer filter owners are destroyed during unwinding.
    auto error = std::current_exception();
    stream.sync();
    std::rethrow_exception(error);
  }
}

void dynamic_filter_publication_session::finish_input() noexcept
{
  auto& operation = *_state;
  {
    std::scoped_lock lock(operation.mutex);
    operation.seal();
    operation.input_closed = true;
    if (operation.current == state::phase::OPEN) {
      operation.complete(completion::SKIPPED);
    } else if (operation.current == state::phase::COLLECTING && !operation.accumulation_result) {
      operation.end_attempt(completion::SKIPPED,
                            &dynamic_filter_stats_snapshot::accumulations_incomplete);
    }
    operation.abandon_pending_publication();
  }
  operation.settle();
}

void dynamic_filter_publication_session::cancel() noexcept
{
  auto& operation = *_state;
  {
    std::scoped_lock lock(operation.mutex);
    operation.input_closed = true;
    operation.cancelled    = true;
    if (operation.current == state::phase::OPEN) {
      operation.complete(completion::CANCELLED);
    } else if (operation.current == state::phase::COLLECTING) {
      operation.end_attempt(completion::CANCELLED);
    }
    operation.abandon_pending_publication();
  }
  operation.settle();
}

}  // namespace sirius::op
