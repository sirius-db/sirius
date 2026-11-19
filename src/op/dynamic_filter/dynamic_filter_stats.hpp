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
 * @brief The dynamic-filter publication and application counters, each held as a `Counter`
 *
 * `dynamic_filter_stats` holds them as atomics for the lifetime of a connection.
 * `dynamic_filter_stats_snapshot` holds plain values, both for a copy of those atomics and for the
 * counts that one publication attempt adds to them. `dynamic_filter_counter_fields` lists every
 * field once for both conversions.
 *
 * CONSTRUCTION COUNTERS, set by `dynamic_filter_publication_session`, `plan_construction`, and
 * `build_filter_construction`. The `accumulations_skipped_*`, `accumulations_incomplete`, and
 * `accumulations_abandoned` counters each name why a started attempt ended without publishing. A
 * Decline ends the session before any attempt starts and is counted by reason:
 * `accumulations_declined_*`, or `accumulations_skipped_inventory` for a wrong input shape or an
 * inventory mismatch:
 * - `accumulations_started`: an attempt whose `begin` leased its Construction storage, including
 *   its Bloom partial arrays, and began collecting Contributions.
 * - `accumulation_expected_contributions`: one certified batch of a started attempt's Complete
 *   Build Inventory, each of which must contribute once.
 * - `accumulation_completed_contributions`: one Contribution whose keys were added to every
 *   accumulator.
 * - `accumulation_duplicate_contributions`: a Contribution of a batch already claimed (a retry, or
 *   a batch spilled or cloned since its first Contribution), which adds nothing.
 * - `accumulations_skipped_inventory`: a build whose input had the wrong shape or whose arrival
 *   ledger disagreed with its repository, or an attempt ended because a task input could not be
 *   matched to exactly one certified batch or its batch did not match what the inventory recorded.
 * - `accumulations_declined_pipeline_barrier`: a Decline because an ORDER BY producer's PIPELINE
 *   barrier holds the build input.
 * - `accumulations_declined_port_directive`: a Decline because a late-materialization directive
 *   defers input columns.
 * - `accumulations_declined_unreadable_arrival`: a Decline because an arrival's rows or types
 *   cannot be read.
 * - `accumulations_declined_compressed_arrival`: a Decline because an arrival is compressed.
 * - `accumulations_skipped_admission`: an attempt ended by an allocation or Memory Admission
 *   refusal before any filter is published, including initialization leases, publication workspace,
 *   and a contribution or publication GPU without an admitted partial.
 * - `accumulations_skipped_error`: an attempt ended by an unexpected or fatal failure, counted
 *   before the failure propagates.
 * - `accumulations_skipped_transient`: an attempt ended by a retryable kernel-launch error.
 * - `accumulations_incomplete`: an attempt whose input closed before every batch contributed.
 * - `accumulations_abandoned`: a publication owed by a task that never ran it before input closed
 *   in an uncancelled session.
 * - `accumulation_storage_leaks`: an attempt whose storage was deliberately leaked because GPU work
 *   might still use it after a failed host join; it also counts in `accumulations_skipped_error`.
 * - `accumulation_publications_finished`: a fan-out that pushed at least one filter.
 * - `accumulation_publications_host_staged`: such a fan-out whose replicas were staged through HOST
 *   memory instead of copied between GPU pools.
 * - `accumulation_publication_latency_ns`: one nanosecond from the Publishing Task's last
 *   Contribution to the first push, summed over every attempt that published a filter.
 * - `keys_skipped_bloom_size_gate`: a planned Bloom key whose membership was dropped because the
 *   plan's Bloom arrays together exceed `max_bloom_bytes_per_gpu` and no fixed Bloom size is set.
 * - `keys_skipped_shadowed`: a Shadowed Key, planned without a membership filter because no
 *   Consumer would select it over the filters already on its Channels.
 * - `keys_skipped_shadowed_late`: a key of a Construction abandoned before it finished because
 *   every one of its keys had become a Shadowed Key since `begin`: a stronger filter was published
 *   while the build was partitioned. Non-zero means the two joins' publication order flipped.
 * - `keys_sized_on_total_rows`: a key whose null count was unknown, so its Bloom filter was sized
 *   on every row and no exact IN-list was considered.
 * - `filters_skipped_construction_invariant`: an exact filter dropped because staging found more
 *   valid rows than the inventory recorded.
 * - `filters_skipped_construction_admission`: a filter dropped because the Publishing GPU refused
 *   the memory to build it.
 * - `producers_enabled`: a `dynamic_filter_publication_session` created over an enabled plan; it
 *   counts plan construction, not execution.
 *
 * KEY COUNTERS, set by `plan_construction` for each Admitted Key with at least one Key Binding:
 * - `keys_considered`: such a key, judged once per attempt.
 * - `keys_with_known_domain`: such a key whose build key domain has a known cardinality.
 * - `keys_skipped_domain_gate`: such a key the Domain Gate skipped.
 * - `keys_skipped_type_mismatch`: such a key whose build column is absent from some batch or
 *   arrived in a Carrier Type that does not equal or restore to its Recorded Type.
 * - `keys_build_exceeded_domain`: such a key whose build has more rows than its known domain
 *   cardinality.
 * - `membership_filters_built`: one Membership Filter that Construction built, whether or not it
 *   was pushed.
 * - `zone_map_filters_built`: one Zone Map that Construction built, whether or not it was pushed.
 *
 * PUBLICATION COUNTERS. Each `begin` increments `publication_attempts` and, once the attempt ends,
 * exactly one of `publications_finished` or `publications_failed`:
 * - `publication_attempts`: a `begin` on an open session, including one whose attempt ends at once.
 * - `publications_finished`: an attempt that ended PUBLISHED or SKIPPED.
 * - `publications_failed`: an attempt that ended CANCELLED or FAILED.
 * - `publications_skipped_targets_drained`: an attempt that found every Channel closed; under
 *   `dynamic_filter_publication_policy::publish_when_empty_or_drained` it is counted but the
 *   attempt goes on.
 * - `filters_pushed`: one filter pushed to one Channel's Producer.
 *
 * APPLICATION COUNTERS. The cached placement counters describe successful complete scan payloads,
 * not attempted invocations, and each such scan counts at most one of `cached_placements`, a
 * decline, or `cached_bypassed_transfer`:
 * - `bindings_skipped_incompatible_probe`: a Key Binding whose recorded probe type no membership
 *   adapter can read, which would decline every batch, so it gets no Membership Filter.
 * - `applications_skipped_other_device`: a batch a `sirius::op::scan::dynamic_filter_consumer`
 *   passed through unfiltered because the batch or its prepared program belongs to a device other
 *   than the current one.
 * - `cached_placements`: a scan whose cached checkpoint ran at least one membership step and placed
 *   its selected rows.
 * - `cached_declined_no_layout`: a scan of a borrowed resident batch whose cached view the scan
 *   could not describe.
 * - `cached_declined_no_membership`: a scan whose checkpoint ran no membership step: the batch was
 *   empty, the Usefulness Gate was inactive, no published filter could probe it, or every probe
 *   declined, even when a static residual still selected the rows.
 * - `cached_bypassed_transfer`: a scan whose owned table, transferred or stolen by
 *   `prepare_for_processing`, bypassed the cached checkpoint.
 * - `cached_disabled`: a scan of a borrowed resident batch that skipped the cached checkpoint
 *   because its consumer has cached placement disabled.
 * - `cached_payload_bytes`: one byte of a placement's selected table before normalization,
 *   excluding its separately counted position selection.
 * - `cached_position_selections`: a placement that also kept its surviving rows' positions, which a
 *   deferred output needs.
 */
template <typename Counter>
struct dynamic_filter_counters {
  Counter accumulations_started{};
  Counter accumulation_expected_contributions{};
  Counter accumulation_completed_contributions{};
  Counter accumulation_duplicate_contributions{};
  Counter accumulations_skipped_inventory{};
  Counter accumulations_declined_pipeline_barrier{};
  Counter accumulations_declined_port_directive{};
  Counter accumulations_declined_unreadable_arrival{};
  Counter accumulations_declined_compressed_arrival{};
  Counter accumulations_skipped_admission{};
  Counter accumulations_skipped_error{};
  Counter accumulations_skipped_transient{};
  Counter accumulations_incomplete{};
  Counter accumulations_abandoned{};
  Counter accumulation_storage_leaks{};
  Counter accumulation_publications_finished{};
  Counter accumulation_publications_host_staged{};
  Counter accumulation_publication_latency_ns{};
  Counter keys_skipped_bloom_size_gate{};
  Counter keys_skipped_shadowed{};
  Counter keys_skipped_shadowed_late{};
  Counter keys_sized_on_total_rows{};
  Counter filters_skipped_construction_invariant{};
  Counter filters_skipped_construction_admission{};
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
  Counter publications_skipped_targets_drained{};
  Counter filters_pushed{};

  Counter bindings_skipped_incompatible_probe{};
  Counter applications_skipped_other_device{};
  Counter cached_placements{};
  Counter cached_declined_no_layout{};
  Counter cached_declined_no_membership{};
  Counter cached_bypassed_transfer{};
  Counter cached_disabled{};
  Counter cached_payload_bytes{};
  Counter cached_position_selections{};
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
  &dynamic_filter_counters<Counter>::accumulations_declined_pipeline_barrier,
  &dynamic_filter_counters<Counter>::accumulations_declined_port_directive,
  &dynamic_filter_counters<Counter>::accumulations_declined_unreadable_arrival,
  &dynamic_filter_counters<Counter>::accumulations_declined_compressed_arrival,
  &dynamic_filter_counters<Counter>::accumulations_skipped_admission,
  &dynamic_filter_counters<Counter>::accumulations_skipped_error,
  &dynamic_filter_counters<Counter>::accumulations_skipped_transient,
  &dynamic_filter_counters<Counter>::accumulations_incomplete,
  &dynamic_filter_counters<Counter>::accumulations_abandoned,
  &dynamic_filter_counters<Counter>::accumulation_storage_leaks,
  &dynamic_filter_counters<Counter>::accumulation_publications_finished,
  &dynamic_filter_counters<Counter>::accumulation_publications_host_staged,
  &dynamic_filter_counters<Counter>::accumulation_publication_latency_ns,
  &dynamic_filter_counters<Counter>::keys_skipped_bloom_size_gate,
  &dynamic_filter_counters<Counter>::keys_skipped_shadowed,
  &dynamic_filter_counters<Counter>::keys_skipped_shadowed_late,
  &dynamic_filter_counters<Counter>::keys_sized_on_total_rows,
  &dynamic_filter_counters<Counter>::filters_skipped_construction_invariant,
  &dynamic_filter_counters<Counter>::filters_skipped_construction_admission,
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
  &dynamic_filter_counters<Counter>::publications_skipped_targets_drained,
  &dynamic_filter_counters<Counter>::filters_pushed,
  &dynamic_filter_counters<Counter>::bindings_skipped_incompatible_probe,
  &dynamic_filter_counters<Counter>::applications_skipped_other_device,
  &dynamic_filter_counters<Counter>::cached_placements,
  &dynamic_filter_counters<Counter>::cached_declined_no_layout,
  &dynamic_filter_counters<Counter>::cached_declined_no_membership,
  &dynamic_filter_counters<Counter>::cached_bypassed_transfer,
  &dynamic_filter_counters<Counter>::cached_disabled,
  &dynamic_filter_counters<Counter>::cached_payload_bytes,
  &dynamic_filter_counters<Counter>::cached_position_selections};

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
   * @brief Adds every nonzero counter of @p delta with relaxed increments.
   */
  void add(dynamic_filter_stats_snapshot const& delta) noexcept
  {
    for (std::size_t index = 0; index < shared_fields.size(); ++index) {
      if (auto const value = delta.*plain_fields[index]; value != 0) {
        (this->*shared_fields[index]).fetch_add(value, std::memory_order_relaxed);
      }
    }
  }

 private:
  static constexpr auto const& shared_fields =
    dynamic_filter_counter_fields<std::atomic<std::uint64_t>>;
  static constexpr auto const& plain_fields = dynamic_filter_counter_fields<std::uint64_t>;
};

}  // namespace sirius::op
