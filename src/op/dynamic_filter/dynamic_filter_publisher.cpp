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
#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
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

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

/** @brief The type of failure that occurred during accumulation. */
enum class masked_failure : std::uint8_t { ERROR, TRANSIENT, LEAK };

/**
 * @brief Logs the exception in flight and classifies it for the accumulation counters.
 *
 * @pre Called from a catch handler
 */
masked_failure mask_current_exception(char const* where) noexcept
{
  auto kind = masked_failure::ERROR;
  try {
    try {
      throw;
    } catch (detail::unjoined_gpu_work const& e) {
      kind = masked_failure::LEAK;
      SIRIUS_LOG_ERROR(
        "[{}] accumulated Bloom storage leaked after a failed host join: {}", where, e.what());
    } catch (detail::accumulation_cuda_error const& e) {
      kind = e.transient_launch_failure() ? masked_failure::TRANSIENT : masked_failure::ERROR;
      SIRIUS_LOG_WARN("[{}] accumulated Bloom skipped after a CUDA error: {}", where, e.what());
    } catch (std::logic_error const& e) {
      SIRIUS_LOG_ERROR(
        "[{}] accumulated Bloom skipped after an invariant failure: {}", where, e.what());
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("[{}] accumulated Bloom skipped: {}", where, e.what());
    } catch (...) {
      SIRIUS_LOG_WARN("[{}] accumulated Bloom skipped after a non-standard exception", where);
    }
  } catch (...) {  // Logging failed; the classification stands.
  }
  return kind;
}

void count_failure(dynamic_filter_publication_outcome& outcome, masked_failure kind) noexcept
{
  switch (kind) {
    case masked_failure::TRANSIENT: outcome.accumulations_skipped_transient = 1; break;
    case masked_failure::LEAK:
      outcome.accumulation_storage_leaks  = 1;
      outcome.accumulations_skipped_error = 1;
      break;
    case masked_failure::ERROR: outcome.accumulations_skipped_error = 1; break;
  }
}

char const* describe(accumulation_decline reason) noexcept
{
  switch (reason) {
    case accumulation_decline::CONTRIBUTION_UNACCOUNTABLE:
      return "a task input is not exactly one certified batch";
  }
  return "unknown reason";
}

}  // namespace

//===----------------------------------------------------------------------===//
// dynamic_filter_publication_session::state
//===----------------------------------------------------------------------===//
struct dynamic_filter_publication_session::state {
  enum class phase {
    OPEN,        // A producer may still register on the plan's channels
    COLLECTING,  // An accumulation accepts contributions from certified build batches
    PUBLISHING,  // One whole-build delivery or the publishing job owns publication (exclusively)
    TERMINAL  // The session has completed, cancelled, or failed; no further producer registration
  };

  enum class contribution : std::uint8_t { PENDING, IN_FLIGHT, COMPLETED };

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
    if (current == phase::TERMINAL) { return; }
    for (auto const& producer : producers) {
      producer.finish(result);
    }
    current = phase::TERMINAL;
    if (!stats || !account_attempt) { return; }
    count_terminal(result);
  }

  void count_terminal(sirius_dynamic_filter_set::completion result) noexcept
  {
    if (!stats) { return; }
    if (result == sirius_dynamic_filter_set::completion::published ||
        result == sirius_dynamic_filter_set::completion::skipped) {
      stats->publications_finished.fetch_add(1, std::memory_order_relaxed);
    } else {
      // result == completion::cancelled or completion::failed
      stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
    }
  }

  /**
   * @brief Adds @p value to the shared counters. The caller holds the mutex, as for every other
   * outcome update.
   */
  void record(dynamic_filter_publication_outcome const& value) noexcept
  {
    if (!stats) { return; }
    auto const relaxed = std::memory_order_relaxed;
    stats->accumulations_started.fetch_add(value.accumulations_started, relaxed);
    stats->accumulation_expected_contributions.fetch_add(value.accumulation_expected_contributions,
                                                         relaxed);
    stats->accumulation_completed_contributions.fetch_add(
      value.accumulation_completed_contributions, relaxed);
    stats->accumulation_duplicate_contributions.fetch_add(
      value.accumulation_duplicate_contributions, relaxed);
    stats->accumulations_skipped_inventory.fetch_add(value.accumulations_skipped_inventory,
                                                     relaxed);
    stats->accumulations_skipped_admission.fetch_add(value.accumulations_skipped_admission,
                                                     relaxed);
    stats->accumulations_skipped_error.fetch_add(value.accumulations_skipped_error, relaxed);
    stats->accumulations_skipped_transient.fetch_add(value.accumulations_skipped_transient,
                                                     relaxed);
    stats->accumulations_incomplete.fetch_add(value.accumulations_incomplete, relaxed);
    stats->accumulations_abandoned.fetch_add(value.accumulations_abandoned, relaxed);
    stats->accumulation_storage_leaks.fetch_add(value.accumulation_storage_leaks, relaxed);
    stats->accumulation_publications_finished.fetch_add(value.accumulation_publications_finished,
                                                        relaxed);
    stats->accumulation_publication_latency_ns.fetch_add(value.accumulation_publication_latency_ns,
                                                         relaxed);
    stats->keys_skipped_bloom_size_gate.fetch_add(value.keys_skipped_bloom_size_gate, relaxed);
    stats->keys_skipped_bloom_unsupported.fetch_add(value.keys_skipped_bloom_unsupported, relaxed);
    stats->keys_considered.fetch_add(value.keys_considered, relaxed);
    stats->keys_with_known_domain.fetch_add(value.keys_with_known_domain, relaxed);
    stats->keys_skipped_domain_gate.fetch_add(value.keys_skipped_domain_gate, relaxed);
    stats->keys_skipped_type_mismatch.fetch_add(value.keys_skipped_type_mismatch, relaxed);
    stats->keys_build_exceeded_domain.fetch_add(value.keys_build_exceeded_domain, relaxed);
    stats->membership_filters_built.fetch_add(value.membership_filters_built, relaxed);
    stats->zone_map_filters_built.fetch_add(value.zone_map_filters_built, relaxed);
    stats->publications_skipped_targets_drained.fetch_add(value.skipped_targets_drained, relaxed);
    stats->filters_pushed.fetch_add(value.filters_pushed, relaxed);
  }

  [[nodiscard]] bool targets_accepting() const noexcept
  {
    return std::ranges::any_of(channels,
                               [](auto const& channel) { return channel->accepting_filters(); });
  }

  /**
   * @brief Ends the attempt with @p result unless it already has one. The caller holds the mutex.
   */
  void end_attempt(sirius_dynamic_filter_set::completion result) noexcept
  {
    if (!accumulation_result) { accumulation_result = result; }
    input_closed = true;
  }

  /**
   * @brief Moves an accumulation to terminal once its result is known and nothing is in flight,
   * then releases a retired builder if this thread may.
   *
   * @return Whether a retired builder still awaits release on another thread
   */
  bool settle() noexcept
  {
    std::optional<detail::accumulated_bloom_builder> released;
    std::optional<sirius_dynamic_filter_set::completion> finished;
    bool pending = false;
    {
      std::scoped_lock lock(mutex);
      if (accumulated && current != phase::TERMINAL && active_operations == 0 &&
          accumulation_result) {
        current  = phase::TERMINAL;
        finished = accumulation_result;
        record(outcome);
        count_terminal(*finished);
        retiring = std::move(builder);
        builder.reset();
      }
      if (retiring && retiring->releasable_here()) {
        released = std::move(retiring);
        retiring.reset();
      }
      pending = retiring.has_value();
    }
    if (finished) {
      for (auto const& producer : producers) {
        producer.finish(*finished);
      }
    }
    if (released) {
      nvtx_scoped_range range{"dynfilter::accum::retire"};
      released.reset();
    }
    return pending;
  }

  /**
   * @brief The publishing job's body: reduce and replicate the partials, then fan out.
   */
  void publish(::cuda::stream_ref stream) noexcept
  {
    using completion = sirius_dynamic_filter_set::completion;
    try {
      {
        std::scoped_lock lock(mutex);
        if (accumulation_result) { return; }
        if (cancelled) {
          end_attempt(completion::cancelled);
          return;
        }
        if (!targets_accepting()) {
          outcome.skipped_targets_drained = 1;
          end_attempt(completion::skipped);
          return;
        }
      }
      int device = -1;
      if (auto const status = cudaGetDevice(&device); status != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error(std::string{"cudaGetDevice: "} + cudaGetErrorString(status));
      }
      auto filters = builder->finish(rmm::cuda_device_id{device}, stream);
      if (!filters) {
        SIRIUS_LOG_DEBUG(
          "[dynamic_filter_publication_session] accumulated Bloom skipped: scratch lease refused.");
        std::scoped_lock lock(mutex);
        outcome.accumulations_skipped_admission = 1;
        end_attempt(completion::skipped);
        return;
      }
      {
        std::scoped_lock lock(mutex);
        if (cancelled) {
          end_attempt(completion::cancelled);
          return;
        }
        if (!targets_accepting()) {
          outcome.skipped_targets_drained = 1;
          end_attempt(completion::skipped);
          return;
        }
        fanout_started = true;
      }
      std::size_t pushed         = 0;
      std::size_t active_targets = 0;
      std::optional<std::chrono::steady_clock::duration> latency;
      for (std::size_t target_index = 0; target_index < plan.probe_targets().size();
           ++target_index) {
        auto const& target = plan.probe_targets()[target_index];
        if (!target.filter_set->accepting_filters()) { continue; }
        ++active_targets;
        for (auto const& binding : target.key_bindings) {
          auto const key = std::ranges::find(active_keys, binding.admitted_key_index);
          if (key == active_keys.end()) { continue; }
          auto const filter_index = static_cast<std::size_t>(key - active_keys.begin());
          if (!latency) {
            latency = std::chrono::steady_clock::now() - final_commit;
            nvtx3::mark_in<nvtx_domain>("dynfilter::accum::visible");
          }
          if (producers[target_index].push_filter(binding.channel_push_ordinal,
                                                  (*filters)[filter_index])) {
            ++pushed;
          }
        }
      }
      std::scoped_lock lock(mutex);
      outcome.membership_filters_built           = filters->size();
      outcome.filters_pushed                     = pushed;
      outcome.active_targets                     = active_targets;
      outcome.accumulation_publications_finished = pushed != 0 ? 1 : 0;
      if (latency) {
        outcome.accumulation_publication_latency_ns = static_cast<std::size_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(*latency).count());
      }
      end_attempt(pushed != 0 ? completion::published : completion::skipped);
    } catch (...) {
      auto const kind = mask_current_exception("dynamic_filter_publication_session::publish");
      std::scoped_lock lock(mutex);
      count_failure(outcome, kind);
      end_attempt(completion::failed);
    }
  }

  /**
   * @brief Consumes a job: the publishing job publishes when invoked and abandons otherwise; every
   * job settles.
   */
  static void consume(std::shared_ptr<state> operation,
                      std::optional<::cuda::stream_ref> stream) noexcept
  {
    bool publishes = false;
    {
      std::scoped_lock lock(operation->mutex);
      publishes = std::exchange(operation->publishing_share, false);
    }
    if (publishes) {
      if (stream) {
        operation->publish(*stream);
      } else {
        std::scoped_lock lock(operation->mutex);
        if (operation->cancelled) {
          operation->end_attempt(sirius_dynamic_filter_set::completion::cancelled);
        } else {
          if (!operation->accumulation_result) { operation->outcome.accumulations_abandoned = 1; }
          operation->end_attempt(sirius_dynamic_filter_set::completion::skipped);
        }
      }
      std::scoped_lock lock(operation->mutex);
      --operation->active_operations;
    }
    (void)operation->settle();
  }

  std::mutex mutex;
  dynamic_filter_publish_plan plan;
  dynamic_filter_stats* stats;
  std::vector<sirius_dynamic_filter_set::producer> producers;
  std::vector<std::shared_ptr<sirius_dynamic_filter_set>> channels;
  std::optional<complete_build_inventory> inventory;
  std::vector<contribution> journal;  // One entry per inventory batch, in inventory order
  std::vector<std::size_t> active_keys;
  std::optional<detail::accumulated_bloom_builder> builder;
  // A builder whose release would credit the settling thread's tracker; released by a later settle.
  std::optional<detail::accumulated_bloom_builder> retiring;
  dynamic_filter_publication_outcome outcome;
  std::optional<sirius_dynamic_filter_set::completion> accumulation_result;
  std::chrono::steady_clock::time_point final_commit{};
  std::size_t active_operations = 0;  // Claimed contributions plus the outstanding publishing job
  bool accumulated              = false;
  bool publishing_share         = false;  // The publishing job exists and has not been consumed
  phase current                 = phase::OPEN;
  bool sealed                   = false;
  bool input_closed             = false;
  bool cancelled                = false;
  bool fanout_started           = false;
};

//===----------------------------------------------------------------------===//
// dynamic_filter_publication_session::accumulation_job
//===----------------------------------------------------------------------===//
dynamic_filter_publication_session::accumulation_job
dynamic_filter_publication_session::decline_accumulation(accumulation_decline reason) noexcept
{
  auto operation = _state;
  bool declined  = false;
  {
    std::scoped_lock lock(operation->mutex);
    if (operation->current == state::phase::COLLECTING && !operation->accumulation_result) {
      operation->outcome.accumulations_skipped_inventory = 1;
      operation->end_attempt(sirius_dynamic_filter_set::completion::skipped);
      declined = true;
    }
  }
  if (declined) {
    try {
      SIRIUS_LOG_INFO("[dynamic_filter_publication_session] accumulated Bloom declined: {}.",
                      describe(reason));
    } catch (...) {  // The decline stands without its log line.
    }
  }
  if (operation->settle()) { return accumulation_job{std::move(operation)}; }
  return {};
}

dynamic_filter_publication_session::accumulation_job::~accumulation_job()
{
  if (_state) { state::consume(std::move(_state), std::nullopt); }
}

dynamic_filter_publication_session::accumulation_job::accumulation_job(
  accumulation_job&& other) noexcept
  : _state(std::move(other._state))
{
}

dynamic_filter_publication_session::accumulation_job&
dynamic_filter_publication_session::accumulation_job::operator=(accumulation_job&& other) noexcept
{
  if (this != &other) {
    if (_state) { state::consume(std::move(_state), std::nullopt); }
    _state = std::move(other._state);
  }
  return *this;
}

void dynamic_filter_publication_session::accumulation_job::operator()(
  ::cuda::stream_ref stream) && noexcept
{
  if (_state) { state::consume(std::move(_state), stream); }
}

//===----------------------------------------------------------------------===//
// dynamic_filter_publication_session
//===----------------------------------------------------------------------===//
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

// 1. Start accumulation
bool dynamic_filter_publication_session::try_begin_accumulation(
  std::optional<complete_build_inventory> inventory) noexcept
{
  auto operation = _state;
  nvtx_scoped_range range{"dynfilter::accum::begin"};
  dynamic_filter_publication_outcome selection;
  try {
    std::vector<detail::accumulated_bloom_builder::key> keys;
    std::vector<std::size_t> active_keys;
    std::optional<detail::accumulated_bloom_geometry> geometry;
    std::size_t expected = 0;
    {
      std::scoped_lock lock(operation->mutex);
      if (operation->current != state::phase::OPEN || operation->input_closed ||
          !operation->plan.multi_partition_enabled()) {
        return false;
      }
      // Replica placement is final from here on, so the plan can be read without the mutex.
      operation->seal();
      if (!inventory || !inventory->valid()) {
        selection.accumulations_skipped_inventory = 1;
        operation->record(selection);
        return false;
      }
      std::vector<bool> bound(operation->plan.admitted_keys().size(), false);
      for (auto const& target : operation->plan.probe_targets()) {
        for (auto const& binding : target.key_bindings) {
          bound[binding.admitted_key_index] = true;
        }
      }
      auto const rows = inventory->total_rows();
      for (std::size_t index = 0; index < bound.size(); ++index) {
        if (!bound[index]) { continue; }
        auto const& key = operation->plan.admitted_keys()[index];
        ++selection.keys_considered;
        if (key.build_key_domain_cardinality != 0) {
          ++selection.keys_with_known_domain;
          if (rows > key.build_key_domain_cardinality) { ++selection.keys_build_exceeded_domain; }
        }
        if (key.build_key_ordinal < 0 ||
            static_cast<std::size_t>(key.build_key_ordinal) >= inventory->schema().size() ||
            inventory->schema()[key.build_key_ordinal].type != key.storage_type) {
          ++selection.keys_skipped_type_mismatch;
          continue;
        }
        if (domain_coverage_gate_fires(rows,
                                       key.build_key_domain_cardinality,
                                       key.build_key_proven_unique,
                                       operation->plan.domain_coverage_threshold())) {
          ++selection.keys_skipped_domain_gate;
          continue;
        }
        if (!sirius_dynamic_bloom_filter::supports(key.storage_type)) {
          ++selection.keys_skipped_bloom_unsupported;
          continue;
        }
        if (rows == 0) { continue; }
        active_keys.push_back(index);
        keys.push_back({key.build_key_ordinal, key.storage_type});
      }
      if (!keys.empty()) {
        geometry = detail::accumulated_bloom_geometry::try_create(
          rows, keys.size(), operation->plan.max_bloom_bytes_per_gpu());
        if (!geometry) { selection.keys_skipped_bloom_size_gate = keys.size(); }
      }
      if (!geometry) {
        operation->record(selection);
        return false;
      }
    }

    auto const& replicas = operation->plan.replica_spaces();
    if (replicas.empty()) {
      selection.accumulations_skipped_admission = 1;
      std::scoped_lock lock(operation->mutex);
      operation->record(selection);
      return false;
    }
    for (auto const& from : replicas) {
      for (auto const& to : replicas) {
        auto const source      = from.get_gpu_space().get_device_id();
        auto const destination = to.get_gpu_space().get_device_id();
        if (source != destination &&
            !cucascade::memory::probe_peer_dma_works(source, destination)) {
          SIRIUS_LOG_INFO(
            "[dynamic_filter_publication_session] accumulated Bloom skipped: no working peer DMA "
            "from GPU {} to GPU {}.",
            source,
            destination);
          selection.accumulations_skipped_admission = 1;
          std::scoped_lock lock(operation->mutex);
          operation->record(selection);
          return false;
        }
      }
    }

    // Allocation happens outside the mutex; the builder is installed only if nothing closed the
    // session meanwhile.
    auto built = detail::accumulated_bloom_builder::try_create(keys, *geometry, replicas);
    if (!built) {
      SIRIUS_LOG_DEBUG(
        "[dynamic_filter_publication_session] accumulated Bloom skipped: partial-array lease "
        "refused.");
      selection.accumulations_skipped_admission = 1;
      std::scoped_lock lock(operation->mutex);
      operation->record(selection);
      return false;
    }
    {
      std::scoped_lock lock(operation->mutex);
      if (operation->current != state::phase::OPEN || operation->input_closed) {
        operation->record(selection);
        return false;  // `built` is released here, on the untracked creator thread.
      }
      expected = inventory->batches().size();
      operation->journal.assign(expected, state::contribution::PENDING);
      operation->inventory.emplace(std::move(*inventory));
      operation->active_keys = std::move(active_keys);
      operation->builder     = std::move(built);
      operation->accumulated = true;
      operation->current     = state::phase::COLLECTING;
      // Start counters are visible at once; the attempt's outcome is recorded when it settles.
      selection.accumulations_started               = 1;
      selection.accumulation_expected_contributions = expected;
      operation->record(selection);
      if (operation->stats) {
        operation->stats->publication_attempts.fetch_add(1, std::memory_order_relaxed);
      }
    }
    SIRIUS_LOG_INFO(
      "[dynamic_filter_publication_session] accumulated Bloom started: {} key(s), {} batch(es), {} "
      "array bytes per GPU on {} GPU(s), {}-byte transfer chunks.",
      keys.size(),
      expected,
      geometry->arrays_bytes,
      replicas.size(),
      geometry->chunk_bytes);
    return true;
  } catch (...) {
    count_failure(selection, mask_current_exception("dynamic_filter_publication_session::begin"));
    std::scoped_lock lock(operation->mutex);
    operation->record(selection);
    return false;
  }
}

bool dynamic_filter_publication_session::accumulation_claimed() const noexcept
{
  std::scoped_lock lock(_state->mutex);
  return _state->accumulated;
}

// 2. Contribute a batch to the accumulation
dynamic_filter_publication_session::accumulation_job dynamic_filter_publication_session::contribute(
  std::uint64_t original_id,
  cucascade::read_only_data_batch const& source,
  ::cuda::stream_ref stream) noexcept
{
  auto operation    = _state;
  std::size_t index = 0;
  bool claimed      = false;
  try {
    {
      std::scoped_lock lock(operation->mutex);
      if (!operation->accumulated) { return {}; }
    }
    auto const* gpu = dynamic_cast<cucascade::gpu_table_representation const*>(source.get_data());
    std::optional<cudf::table_view> view;
    if (gpu != nullptr) { view.emplace(gpu->get_table_view()); }
    auto const* space = source.get_memory_space();
    {
      std::scoped_lock lock(operation->mutex);
      auto const* entry = operation->inventory->find(original_id);
      // A batch without rows that arrived after certification (`build_arrival_ledger` admits it)
      // adds no key.
      if (entry == nullptr && view && view->num_rows() == 0) { return {}; }
      if (entry != nullptr) {
        index = static_cast<std::size_t>(entry - operation->inventory->batches().data());
        if (operation->journal[index] != state::contribution::PENDING) {
          // A retry of an already claimed batch: its keys are, or are being, inserted. After the
          // attempt settled, its outcome is already recorded, so the count goes straight to the
          // stats.
          if (operation->current != state::phase::TERMINAL) {
            ++operation->outcome.accumulation_duplicate_contributions;
          } else if (operation->stats) {
            operation->stats->accumulation_duplicate_contributions.fetch_add(
              1, std::memory_order_relaxed);
          }
          return {};
        }
      }
      if (operation->current != state::phase::COLLECTING || operation->accumulation_result) {
        return {};
      }
      bool const valid =
        entry != nullptr && view && space != nullptr &&
        space->get_tier() == cucascade::memory::Tier::GPU &&
        static_cast<std::uint64_t>(view->num_rows()) == entry->rows &&
        operation->inventory->schema_matches(*view) &&
        operation->builder->has_partial(rmm::cuda_device_id{space->get_device_id()});
      if (!valid) {
        throw std::logic_error(
          "[dynamic_filter_publication_session::contribute] the contribution does not match its "
          "certified batch or has no partial on its GPU");
      }
      if (!operation->targets_accepting()) {
        operation->outcome.skipped_targets_drained = 1;
        operation->end_attempt(sirius_dynamic_filter_set::completion::skipped);
      } else {
        operation->journal[index] = state::contribution::IN_FLIGHT;
        ++operation->active_operations;
        claimed = true;
      }
    }
    if (claimed) {
      operation->builder->enqueue_add(*view, rmm::cuda_device_id{space->get_device_id()}, stream);
      std::scoped_lock lock(operation->mutex);
      operation->journal[index] = state::contribution::COMPLETED;
      claimed                   = false;
      if (++operation->outcome.accumulation_completed_contributions == operation->journal.size() &&
          !operation->accumulation_result) {
        // The final commit happens after every enqueue_add returned, so publication sees all
        // inserts. The publishing job keeps this operation's share.
        operation->current          = state::phase::PUBLISHING;
        operation->publishing_share = true;
        operation->final_commit     = std::chrono::steady_clock::now();
        nvtx3::mark_in<nvtx_domain>("dynfilter::accum::final_commit");
        return accumulation_job{std::move(operation)};
      }
      --operation->active_operations;
    }
  } catch (...) {
    auto const kind = mask_current_exception("dynamic_filter_publication_session::contribute");
    std::scoped_lock lock(operation->mutex);
    if (claimed) { --operation->active_operations; }
    if (operation->current == state::phase::COLLECTING) {
      count_failure(operation->outcome, kind);
      operation->end_attempt(sirius_dynamic_filter_set::completion::failed);
    }
  }
  if (operation->settle()) { return accumulation_job{std::move(operation)}; }
  return {};
}

void dynamic_filter_publication_session::observe_whole_build(
  complete_build_delivery const& delivery, std::function<void()> const& deposit)
{
  auto operation = _state;
  bool claimed   = false;
  {
    std::scoped_lock lock(operation->mutex);
    operation->seal();
    if (operation->current == state::phase::OPEN && operation->plan.enabled() && delivery._batch) {
      operation->current = state::phase::PUBLISHING;
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
        operation->current = state::phase::OPEN;
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

// 3. Publish the accumulated filters to the probe targets, if any, and record the outcome.
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

// 4. Settle the session after all contributions have been made and the input is closed, or the
// accumulation has been cancelled.
void dynamic_filter_publication_session::finish_input() noexcept
{
  auto operation = _state;
  {
    std::scoped_lock lock(operation->mutex);
    operation->seal();
    operation->input_closed = true;
    if (operation->current == state::phase::OPEN) {
      operation->complete(sirius_dynamic_filter_set::completion::skipped);
    } else if (operation->current == state::phase::COLLECTING && !operation->accumulation_result &&
               std::ranges::find(operation->journal, state::contribution::PENDING) !=
                 operation->journal.end()) {
      operation->outcome.accumulations_incomplete = 1;
      operation->end_attempt(sirius_dynamic_filter_set::completion::skipped);
    }
  }
  (void)operation->settle();
}

void dynamic_filter_publication_session::cancel() noexcept
{
  auto operation = _state;
  {
    std::scoped_lock lock(operation->mutex);
    operation->input_closed = true;
    operation->cancelled    = true;
    if (operation->current == state::phase::OPEN) {
      operation->complete(sirius_dynamic_filter_set::completion::cancelled);
    } else if (operation->current == state::phase::COLLECTING) {
      operation->end_attempt(sirius_dynamic_filter_set::completion::cancelled);
    }
  }
  (void)operation->settle();
}

}  // namespace sirius::op
