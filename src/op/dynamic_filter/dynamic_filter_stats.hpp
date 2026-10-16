/*
 * Copyright 2026, Sirius Contributors.
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

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::op {

/**
 * @brief The dynamic-filter publication and application counters, each held as a @p Counter
 *
 * `dynamic_filter_stats` holds them as atomics for the lifetime of a connection.
 * `dynamic_filter_stats_snapshot` holds plain values, both for a copy of those atomics and for the
 * counts that one publication attempt adds to them (`dynamic_filter_publication_outcome`).
 * `dynamic_filter_counter_fields` lists every field once for both conversions.
 *
 * `producers_enabled` counts plan construction, not execution. Each claim increments
 * `publication_attempts` and exactly one of `publications_finished`, `publications_failed`, or
 * `publications_skipped_source_not_resident`; a source skip may reopen the claim window.
 * `publications_skipped_build_not_whole` is counted once per join rather than once per delivery.
 * `bindings_skipped_incompatible_probe` counts bindings whose recorded probe type no membership
 * adapter can read, which would decline every batch. `applications_skipped_other_device` counts
 * batches a `sirius::op::scan::dynamic_filter_consumer` passed through unfiltered because the batch
 * or its prepared program belongs to a device other than the current one.
 *
 * Accumulation counters: `accumulations_started` counts attempts that allocated their partial
 * arrays. `accumulations_skipped_inventory` counts builds whose input could not be certified and
 * attempts ended because a task input could not be matched to exactly one certified batch.
 * `accumulations_skipped_admission` counts allocation or admission refusals before any filter is
 * published, including initialization leases, publication scratch, missing peer DMA, and a
 * contribution or publication GPU without an admitted partial. `accumulations_skipped_error` counts
 * unexpected or fatal failures before they propagate, and `accumulations_skipped_transient` counts
 * retryable kernel-launch errors. `accumulations_incomplete` counts inputs that closed before every
 * batch contributed, `accumulations_abandoned` publications owed by a task that never ran them
 * before input closed in an uncancelled session, and `accumulation_publications_finished` fan-outs
 * that pushed at least one filter. `accumulation_storage_leaks` counts partial storage deliberately
 * leaked because GPU work might still use it after a failed host join.
 * `accumulation_publication_latency_ns` sums the time from final contribution to first push for
 * every attempt that published a filter.
 */
template <typename Counter>
struct dynamic_filter_counters {
  Counter accumulations_started{};
  Counter accumulation_expected_contributions{};
  Counter accumulation_completed_contributions{};
  Counter accumulation_duplicate_contributions{};
  Counter accumulations_skipped_inventory{};
  Counter accumulations_skipped_admission{};
  Counter accumulations_skipped_error{};
  Counter accumulations_skipped_transient{};
  Counter accumulations_incomplete{};
  Counter accumulations_abandoned{};
  Counter accumulation_storage_leaks{};
  Counter accumulation_publications_finished{};
  Counter accumulation_publication_latency_ns{};
  Counter keys_skipped_bloom_size_gate{};
  Counter keys_skipped_bloom_unsupported{};
  Counter producers_enabled{};

  Counter keys_considered{};
  Counter keys_with_known_domain{};
  Counter keys_skipped_domain_gate{};
  Counter keys_skipped_type_mismatch{};
  Counter keys_build_exceeded_domain{};
  Counter membership_filters_built{};
  Counter zone_map_filters_built{};

  Counter publication_attempts{};
  Counter publications_finished{};
  Counter publications_failed{};
  Counter publications_skipped_source_not_resident{};
  Counter publications_skipped_build_not_whole{};
  Counter publications_skipped_targets_drained{};
  Counter filters_pushed{};

  Counter bindings_skipped_incompatible_probe{};
  Counter applications_skipped_other_device{};
};

/**
 * @brief Every field of `dynamic_filter_counters<Counter>`, in declaration order
 */
template <typename Counter>
inline constexpr std::array dynamic_filter_counter_fields{
  &dynamic_filter_counters<Counter>::accumulations_started,
  &dynamic_filter_counters<Counter>::accumulation_expected_contributions,
  &dynamic_filter_counters<Counter>::accumulation_completed_contributions,
  &dynamic_filter_counters<Counter>::accumulation_duplicate_contributions,
  &dynamic_filter_counters<Counter>::accumulations_skipped_inventory,
  &dynamic_filter_counters<Counter>::accumulations_skipped_admission,
  &dynamic_filter_counters<Counter>::accumulations_skipped_error,
  &dynamic_filter_counters<Counter>::accumulations_skipped_transient,
  &dynamic_filter_counters<Counter>::accumulations_incomplete,
  &dynamic_filter_counters<Counter>::accumulations_abandoned,
  &dynamic_filter_counters<Counter>::accumulation_storage_leaks,
  &dynamic_filter_counters<Counter>::accumulation_publications_finished,
  &dynamic_filter_counters<Counter>::accumulation_publication_latency_ns,
  &dynamic_filter_counters<Counter>::keys_skipped_bloom_size_gate,
  &dynamic_filter_counters<Counter>::keys_skipped_bloom_unsupported,
  &dynamic_filter_counters<Counter>::producers_enabled,
  &dynamic_filter_counters<Counter>::keys_considered,
  &dynamic_filter_counters<Counter>::keys_with_known_domain,
  &dynamic_filter_counters<Counter>::keys_skipped_domain_gate,
  &dynamic_filter_counters<Counter>::keys_skipped_type_mismatch,
  &dynamic_filter_counters<Counter>::keys_build_exceeded_domain,
  &dynamic_filter_counters<Counter>::membership_filters_built,
  &dynamic_filter_counters<Counter>::zone_map_filters_built,
  &dynamic_filter_counters<Counter>::publication_attempts,
  &dynamic_filter_counters<Counter>::publications_finished,
  &dynamic_filter_counters<Counter>::publications_failed,
  &dynamic_filter_counters<Counter>::publications_skipped_source_not_resident,
  &dynamic_filter_counters<Counter>::publications_skipped_build_not_whole,
  &dynamic_filter_counters<Counter>::publications_skipped_targets_drained,
  &dynamic_filter_counters<Counter>::filters_pushed,
  &dynamic_filter_counters<Counter>::bindings_skipped_incompatible_probe,
  &dynamic_filter_counters<Counter>::applications_skipped_other_device};

static_assert(sizeof(dynamic_filter_counters<std::uint64_t>) ==
                dynamic_filter_counter_fields<std::uint64_t>.size() * sizeof(std::uint64_t),
              "dynamic_filter_counter_fields must list every counter");

/**
 * @brief Relaxed, copyable snapshot of `dynamic_filter_stats`, or the counts one publication adds
 *
 * Individual fields are coherent; cross-field identities require no concurrent updates.
 */
using dynamic_filter_stats_snapshot = dynamic_filter_counters<std::uint64_t>;

/**
 * @brief Connection-lifetime dynamic-filter counters owned by `SiriusContext`
 */
struct dynamic_filter_stats final : dynamic_filter_counters<std::atomic<std::uint64_t>> {
  /**
   * @brief Loads every counter; the loads are relaxed and are not atomic across fields.
   */
  [[nodiscard]] dynamic_filter_stats_snapshot snapshot() const noexcept
  {
    dynamic_filter_stats_snapshot result;
    for (std::size_t index = 0; index < shared_fields.size(); ++index) {
      result.*plain_fields[index] = (this->*shared_fields[index]).load(std::memory_order_relaxed);
    }
    return result;
  }

  /**
   * @brief Adds every counter of @p delta with relaxed increments.
   */
  void add(dynamic_filter_stats_snapshot const& delta) noexcept
  {
    for (std::size_t index = 0; index < shared_fields.size(); ++index) {
      (this->*shared_fields[index])
        .fetch_add(delta.*plain_fields[index], std::memory_order_relaxed);
    }
  }

 private:
  static constexpr auto const& shared_fields =
    dynamic_filter_counter_fields<std::atomic<std::uint64_t>>;
  static constexpr auto const& plain_fields = dynamic_filter_counter_fields<std::uint64_t>;
};

}  // namespace sirius::op
