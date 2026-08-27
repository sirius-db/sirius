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
  std::uint64_t producers_enabled = 0;

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
};

/**
 * @brief Connection-lifetime publication counters owned by `SiriusContext`
 *
 * `producers_enabled` counts plan construction, not execution. Each claim increments
 * `publication_attempts` and exactly one of `publications_finished`, `publications_failed`, or
 * `publications_skipped_source_not_resident`; a source skip may reopen the claim window.
 */
struct dynamic_filter_stats {
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

  // Snapshot loads are relaxed and are not atomic across fields.
  [[nodiscard]] dynamic_filter_stats_snapshot snapshot() const noexcept
  {
    return dynamic_filter_stats_snapshot{
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
      .filters_pushed = filters_pushed.load(std::memory_order_relaxed)};
  }
};

}  // namespace sirius::op
