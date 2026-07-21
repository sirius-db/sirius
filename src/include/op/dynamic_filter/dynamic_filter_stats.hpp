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
 * @brief Plain copyable snapshot of @ref dynamic_filter_stats
 *
 * Field meanings are documented once, on the atomic aggregate. Each field is individually coherent
 * at any time; cross-field comparisons are meaningful only after the query completes, when
 * publication tasks have quiesced.
 */
struct dynamic_filter_stats_snapshot {
  std::uint64_t keys_considered            = 0;
  std::uint64_t keys_with_known_domain     = 0;
  std::uint64_t keys_skipped_domain_gate   = 0;
  std::uint64_t keys_skipped_type_mismatch = 0;
  std::uint64_t keys_build_exceeded_domain = 0;
  std::uint64_t membership_filters_built   = 0;
  std::uint64_t zone_map_filters_built     = 0;

  std::uint64_t producers_enabled                        = 0;
  std::uint64_t publication_attempts                     = 0;
  std::uint64_t publications_finished                    = 0;
  std::uint64_t publications_failed                      = 0;
  std::uint64_t publications_skipped_source_not_resident = 0;
  std::uint64_t publications_skipped_targets_drained     = 0;
  std::uint64_t filters_pushed                           = 0;
};

/**
 * @brief Process-lifetime dynamic-filter publication counters, owned by `SiriusContext`
 *
 * `sirius_physical_hash_join` folds each `dynamic_filter_publication_outcome` into this sink
 * through a non-owning pointer handed to it at construction by `sirius_plan_comparison_join`. The
 * owning `SiriusContext` outlives every plan built during a query -- the same lifetime contract as
 * `dynamic_filter_replica_space`.
 *
 * The fields split into two families, and tests must respect the split. The first family records
 * deterministic policy decisions: for a fixed query, settings, and data they are reproducible
 * run-to-run, and they are the assertion anchors for gate regressions and the rollback evidence.
 * The second family records opportunistic delivery, which races with probe-side draining; assert
 * it only as deltas or directions, never as equality anchors.
 */
struct dynamic_filter_stats {
  // Deterministic policy decisions
  std::atomic<std::uint64_t> keys_considered{0};  ///< Admitted keys walked by publication attempts
  std::atomic<std::uint64_t> keys_with_known_domain{0};  ///< Keys carrying a nonzero domain -- the
                                                         ///< gate-liveness (anti-inertness) signal
  std::atomic<std::uint64_t> keys_skipped_domain_gate{0};    ///< Coverage gate fired
  std::atomic<std::uint64_t> keys_skipped_type_mismatch{0};  ///< Plan/runtime type disagreement
  std::atomic<std::uint64_t> keys_build_exceeded_domain{0};  ///< Direct Invariant-N violation
                                                             ///< count: build rows above the
                                                             ///< domain bound
  std::atomic<std::uint64_t> membership_filters_built{0};    ///< Constructed, before delivery
  std::atomic<std::uint64_t> zone_map_filters_built{0};      ///< Constructed, before delivery

  // Opportunistic delivery
  std::atomic<std::uint64_t> producers_enabled{0};     ///< Joins constructed with an enabled plan
  std::atomic<std::uint64_t> publication_attempts{0};  ///< OPEN -> PUBLISHING claims
  std::atomic<std::uint64_t> publications_finished{0};
  std::atomic<std::uint64_t> publications_failed{0};
  /// Build batch was not GPU-resident at delivery; publication skipped without claiming
  std::atomic<std::uint64_t> publications_skipped_source_not_resident{0};
  /// Attempt hit the all-targets-drained early return; no deterministic counter moved for it
  std::atomic<std::uint64_t> publications_skipped_targets_drained{0};
  std::atomic<std::uint64_t> filters_pushed{0};  ///< Accepted pushes; drain-dependent

  [[nodiscard]] dynamic_filter_stats_snapshot snapshot() const noexcept
  {
    return dynamic_filter_stats_snapshot{
      .keys_considered            = keys_considered.load(std::memory_order_relaxed),
      .keys_with_known_domain     = keys_with_known_domain.load(std::memory_order_relaxed),
      .keys_skipped_domain_gate   = keys_skipped_domain_gate.load(std::memory_order_relaxed),
      .keys_skipped_type_mismatch = keys_skipped_type_mismatch.load(std::memory_order_relaxed),
      .keys_build_exceeded_domain = keys_build_exceeded_domain.load(std::memory_order_relaxed),
      .membership_filters_built   = membership_filters_built.load(std::memory_order_relaxed),
      .zone_map_filters_built     = zone_map_filters_built.load(std::memory_order_relaxed),
      .producers_enabled          = producers_enabled.load(std::memory_order_relaxed),
      .publication_attempts       = publication_attempts.load(std::memory_order_relaxed),
      .publications_finished      = publications_finished.load(std::memory_order_relaxed),
      .publications_failed        = publications_failed.load(std::memory_order_relaxed),
      .publications_skipped_source_not_resident =
        publications_skipped_source_not_resident.load(std::memory_order_relaxed),
      .publications_skipped_targets_drained =
        publications_skipped_targets_drained.load(std::memory_order_relaxed),
      .filters_pushed = filters_pushed.load(std::memory_order_relaxed)};
  }
};

}  // namespace sirius::op
