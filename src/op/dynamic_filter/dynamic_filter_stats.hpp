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

#include <atomic>
#include <cstdint>

namespace sirius::op {

/**
 * @brief Relaxed, copyable snapshot of @ref dynamic_filter_stats
 *
 * Individual fields are coherent; cross-field identities require no concurrent updates.
 */
struct dynamic_filter_stats_snapshot {
  std::uint64_t accumulations_started                = 0;
  std::uint64_t accumulation_expected_contributions  = 0;
  std::uint64_t accumulation_completed_contributions = 0;
  std::uint64_t accumulation_duplicate_contributions = 0;
  std::uint64_t accumulations_skipped_inventory      = 0;
  std::uint64_t accumulations_skipped_admission      = 0;
  std::uint64_t accumulations_skipped_error          = 0;
  std::uint64_t accumulations_skipped_transient      = 0;
  std::uint64_t accumulations_incomplete             = 0;
  std::uint64_t accumulations_abandoned              = 0;
  std::uint64_t accumulation_storage_leaks           = 0;
  std::uint64_t accumulation_publications_finished   = 0;
  std::uint64_t accumulation_publication_latency_ns  = 0;
  std::uint64_t keys_skipped_bloom_size_gate         = 0;
  std::uint64_t keys_skipped_bloom_unsupported       = 0;
  std::uint64_t producers_enabled                    = 0;

  std::uint64_t keys_considered            = 0;
  std::uint64_t keys_with_known_domain     = 0;
  std::uint64_t keys_skipped_domain_gate   = 0;
  std::uint64_t keys_skipped_type_mismatch = 0;
  std::uint64_t keys_build_exceeded_domain = 0;
  std::uint64_t membership_filters_built   = 0;
  std::uint64_t zone_map_filters_built     = 0;

  std::uint64_t publication_attempts                     = 0;
  std::uint64_t publications_finished                    = 0;
  std::uint64_t publications_failed                      = 0;
  std::uint64_t publications_skipped_source_not_resident = 0;
  std::uint64_t publications_skipped_build_not_whole     = 0;
  std::uint64_t publications_skipped_targets_drained     = 0;
  std::uint64_t filters_pushed                           = 0;

  std::uint64_t bindings_skipped_incompatible_probe = 0;
};

/**
 * @brief Connection-lifetime publication counters owned by `SiriusContext`
 *
 * Accumulation counters: `accumulations_started` counts attempts that allocated their partial
 * arrays. `accumulations_skipped_inventory` counts builds whose input could not be certified and
 * attempts ended because a task input could not be matched to exactly one certified batch.
 * `accumulations_skipped_admission` counts refused leases (at the start or for publication scratch)
 * and missing peer DMA. `accumulations_skipped_error` counts masked exceptions and
 * `accumulations_skipped_transient` retryable kernel-launch errors. `accumulations_incomplete`
 * counts inputs that closed before every batch contributed, `accumulations_abandoned` publishing
 * jobs destroyed uninvoked in an uncancelled session, and `accumulation_publications_finished`
 * attempts that pushed a filter. `accumulation_storage_leaks` counts partial storage deliberately
 * leaked because GPU work might still use it after a failed host join.
 * `accumulation_publication_latency_ns` sums the time from each final contribution to its first
 * push.
 *
 * `producers_enabled` counts plan construction, not execution. Each claim increments
 * `publication_attempts` and exactly one of `publications_finished`, `publications_failed`, or
 * `publications_skipped_source_not_resident`; a source skip may reopen the claim window.
 */
struct dynamic_filter_stats {
  std::atomic<std::uint64_t> accumulations_started{0};
  std::atomic<std::uint64_t> accumulation_expected_contributions{0};
  std::atomic<std::uint64_t> accumulation_completed_contributions{0};
  std::atomic<std::uint64_t> accumulation_duplicate_contributions{0};
  std::atomic<std::uint64_t> accumulations_skipped_inventory{0};
  std::atomic<std::uint64_t> accumulations_skipped_admission{0};
  std::atomic<std::uint64_t> accumulations_skipped_error{0};
  std::atomic<std::uint64_t> accumulations_skipped_transient{0};
  std::atomic<std::uint64_t> accumulations_incomplete{0};
  std::atomic<std::uint64_t> accumulations_abandoned{0};
  std::atomic<std::uint64_t> accumulation_storage_leaks{0};
  std::atomic<std::uint64_t> accumulation_publications_finished{0};
  std::atomic<std::uint64_t> accumulation_publication_latency_ns{0};
  std::atomic<std::uint64_t> keys_skipped_bloom_size_gate{0};
  std::atomic<std::uint64_t> keys_skipped_bloom_unsupported{0};
  std::atomic<std::uint64_t> producers_enabled{0};

  std::atomic<std::uint64_t> keys_considered{0};
  std::atomic<std::uint64_t> keys_with_known_domain{0};
  std::atomic<std::uint64_t> keys_skipped_domain_gate{0};
  std::atomic<std::uint64_t> keys_skipped_type_mismatch{0};
  std::atomic<std::uint64_t> keys_build_exceeded_domain{0};
  std::atomic<std::uint64_t> membership_filters_built{0};
  std::atomic<std::uint64_t> zone_map_filters_built{0};

  std::atomic<std::uint64_t> publication_attempts{0};
  std::atomic<std::uint64_t> publications_finished{0};
  std::atomic<std::uint64_t> publications_failed{0};
  std::atomic<std::uint64_t> publications_skipped_source_not_resident{0};
  // Counted once per join rather than once per delivery.
  std::atomic<std::uint64_t> publications_skipped_build_not_whole{0};
  std::atomic<std::uint64_t> publications_skipped_targets_drained{0};
  std::atomic<std::uint64_t> filters_pushed{0};

  // Bindings whose recorded probe type no membership adapter can read (would decline every batch).
  std::atomic<std::uint64_t> bindings_skipped_incompatible_probe{0};

  // Snapshot loads are relaxed and are not atomic across fields.
  [[nodiscard]] dynamic_filter_stats_snapshot snapshot() const noexcept
  {
    return dynamic_filter_stats_snapshot{
      .accumulations_started = accumulations_started.load(std::memory_order_relaxed),
      .accumulation_expected_contributions =
        accumulation_expected_contributions.load(std::memory_order_relaxed),
      .accumulation_completed_contributions =
        accumulation_completed_contributions.load(std::memory_order_relaxed),
      .accumulation_duplicate_contributions =
        accumulation_duplicate_contributions.load(std::memory_order_relaxed),
      .accumulations_skipped_inventory =
        accumulations_skipped_inventory.load(std::memory_order_relaxed),
      .accumulations_skipped_admission =
        accumulations_skipped_admission.load(std::memory_order_relaxed),
      .accumulations_skipped_error = accumulations_skipped_error.load(std::memory_order_relaxed),
      .accumulations_skipped_transient =
        accumulations_skipped_transient.load(std::memory_order_relaxed),
      .accumulations_incomplete   = accumulations_incomplete.load(std::memory_order_relaxed),
      .accumulations_abandoned    = accumulations_abandoned.load(std::memory_order_relaxed),
      .accumulation_storage_leaks = accumulation_storage_leaks.load(std::memory_order_relaxed),
      .accumulation_publications_finished =
        accumulation_publications_finished.load(std::memory_order_relaxed),
      .accumulation_publication_latency_ns =
        accumulation_publication_latency_ns.load(std::memory_order_relaxed),
      .keys_skipped_bloom_size_gate = keys_skipped_bloom_size_gate.load(std::memory_order_relaxed),
      .keys_skipped_bloom_unsupported =
        keys_skipped_bloom_unsupported.load(std::memory_order_relaxed),
      .producers_enabled          = producers_enabled.load(std::memory_order_relaxed),
      .keys_considered            = keys_considered.load(std::memory_order_relaxed),
      .keys_with_known_domain     = keys_with_known_domain.load(std::memory_order_relaxed),
      .keys_skipped_domain_gate   = keys_skipped_domain_gate.load(std::memory_order_relaxed),
      .keys_skipped_type_mismatch = keys_skipped_type_mismatch.load(std::memory_order_relaxed),
      .keys_build_exceeded_domain = keys_build_exceeded_domain.load(std::memory_order_relaxed),
      .membership_filters_built   = membership_filters_built.load(std::memory_order_relaxed),
      .zone_map_filters_built     = zone_map_filters_built.load(std::memory_order_relaxed),
      .publication_attempts       = publication_attempts.load(std::memory_order_relaxed),
      .publications_finished      = publications_finished.load(std::memory_order_relaxed),
      .publications_failed        = publications_failed.load(std::memory_order_relaxed),
      .publications_skipped_source_not_resident =
        publications_skipped_source_not_resident.load(std::memory_order_relaxed),
      .publications_skipped_build_not_whole =
        publications_skipped_build_not_whole.load(std::memory_order_relaxed),
      .publications_skipped_targets_drained =
        publications_skipped_targets_drained.load(std::memory_order_relaxed),
      .filters_pushed = filters_pushed.load(std::memory_order_relaxed),
      .bindings_skipped_incompatible_probe =
        bindings_skipped_incompatible_probe.load(std::memory_order_relaxed)};
  }
};

}  // namespace sirius::op
