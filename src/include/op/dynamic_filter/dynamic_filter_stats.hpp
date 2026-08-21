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
#include <cstdint>
#include <mutex>
#include <string_view>
#include <vector>

namespace sirius::op {

struct named_dynamic_filter_stat {
  std::string_view name;
  std::uint64_t value;
};

enum class dynamic_filter_event_kind : std::uint8_t { global_accumulator_completion };

/**
 * @brief One append-only global-accumulator completion record
 */
struct dynamic_filter_event_record {
  std::uint64_t event_id = 0;
  dynamic_filter_event_kind kind{dynamic_filter_event_kind::global_accumulator_completion};
  std::uint64_t join_operator_id         = 0;
  std::uint64_t exact_contribution_count = 0;
  int device_id                          = -1;
  std::uint64_t build_rows               = 0;
  std::uint64_t filters_built            = 0;
  std::uint64_t active_targets           = 0;
  std::uint64_t filters_pushed           = 0;
};

struct dynamic_filter_event_snapshot {
  std::uint64_t last_event_id = 0;
  std::vector<dynamic_filter_event_record> records;
};

/**
 * @brief Relaxed, copyable snapshot of @ref dynamic_filter_stats
 *
 * Individual fields are coherent; cross-field identities require no concurrent updates.
 */
struct dynamic_filter_stats_snapshot {
  std::uint64_t producers_enabled = 0;

  std::uint64_t keys_considered              = 0;
  std::uint64_t keys_with_known_domain       = 0;
  std::uint64_t keys_skipped_domain_gate     = 0;
  std::uint64_t keys_skipped_bloom_size_gate = 0;
  std::uint64_t keys_skipped_type_mismatch   = 0;
  std::uint64_t keys_build_exceeded_domain   = 0;
  std::uint64_t membership_filters_built     = 0;
  std::uint64_t zone_map_filters_built       = 0;

  std::uint64_t publication_attempts                     = 0;
  std::uint64_t publications_finished                    = 0;
  std::uint64_t publications_failed                      = 0;
  std::uint64_t publications_skipped_source_not_resident = 0;
  std::uint64_t publications_skipped_build_not_whole     = 0;
  std::uint64_t publications_skipped_targets_drained     = 0;
  std::uint64_t filters_pushed                           = 0;

  /**
   * @brief Stable names and values for every cumulative field
   *
   * The SQL observability bridge and benchmark harness enumerate this list instead of maintaining
   * a second hand-written counter schema. Adding a counter above must therefore also add it here.
   */
  [[nodiscard]] auto named_values() const noexcept
  {
    return std::to_array<named_dynamic_filter_stat>(
      {{"producers_enabled", producers_enabled},
       {"keys_considered", keys_considered},
       {"keys_with_known_domain", keys_with_known_domain},
       {"keys_skipped_domain_gate", keys_skipped_domain_gate},
       {"keys_skipped_bloom_size_gate", keys_skipped_bloom_size_gate},
       {"keys_skipped_type_mismatch", keys_skipped_type_mismatch},
       {"keys_build_exceeded_domain", keys_build_exceeded_domain},
       {"membership_filters_built", membership_filters_built},
       {"zone_map_filters_built", zone_map_filters_built},
       {"publication_attempts", publication_attempts},
       {"publications_finished", publications_finished},
       {"publications_failed", publications_failed},
       {"publications_skipped_source_not_resident", publications_skipped_source_not_resident},
       {"publications_skipped_build_not_whole", publications_skipped_build_not_whole},
       {"publications_skipped_targets_drained", publications_skipped_targets_drained},
       {"filters_pushed", filters_pushed}});
  }
};

/**
 * @brief Connection-lifetime publication counters owned by `SiriusContext`
 *
 * `producers_enabled` counts plan construction, not execution. Each session claim increments
 * `publication_attempts` and exactly one of `publications_finished` or `publications_failed`; a
 * non-resident source increments `publications_skipped_source_not_resident` without claiming the
 * session, so broadcast deliveries may repeat it.
 */
struct dynamic_filter_stats {
  std::atomic<std::uint64_t> producers_enabled{0};

  std::atomic<std::uint64_t> keys_considered{0};
  std::atomic<std::uint64_t> keys_with_known_domain{0};
  std::atomic<std::uint64_t> keys_skipped_domain_gate{0};
  std::atomic<std::uint64_t> keys_skipped_bloom_size_gate{0};
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
      .producers_enabled            = producers_enabled.load(std::memory_order_relaxed),
      .keys_considered              = keys_considered.load(std::memory_order_relaxed),
      .keys_with_known_domain       = keys_with_known_domain.load(std::memory_order_relaxed),
      .keys_skipped_domain_gate     = keys_skipped_domain_gate.load(std::memory_order_relaxed),
      .keys_skipped_bloom_size_gate = keys_skipped_bloom_size_gate.load(std::memory_order_relaxed),
      .keys_skipped_type_mismatch   = keys_skipped_type_mismatch.load(std::memory_order_relaxed),
      .keys_build_exceeded_domain   = keys_build_exceeded_domain.load(std::memory_order_relaxed),
      .membership_filters_built     = membership_filters_built.load(std::memory_order_relaxed),
      .zone_map_filters_built       = zone_map_filters_built.load(std::memory_order_relaxed),
      .publication_attempts         = publication_attempts.load(std::memory_order_relaxed),
      .publications_finished        = publications_finished.load(std::memory_order_relaxed),
      .publications_failed          = publications_failed.load(std::memory_order_relaxed),
      .publications_skipped_source_not_resident =
        publications_skipped_source_not_resident.load(std::memory_order_relaxed),
      .publications_skipped_build_not_whole =
        publications_skipped_build_not_whole.load(std::memory_order_relaxed),
      .publications_skipped_targets_drained =
        publications_skipped_targets_drained.load(std::memory_order_relaxed),
      .filters_pushed = filters_pushed.load(std::memory_order_relaxed)};
  }

  /** Record one successfully completed exact-ID global accumulator. */
  void record_global_accumulator_completion(std::uint64_t join_operator_id,
                                            std::uint64_t exact_contribution_count,
                                            int root_device_id,
                                            std::uint64_t build_rows,
                                            std::uint64_t filters_built,
                                            std::uint64_t active_targets,
                                            std::uint64_t accepted_pushes) noexcept
  {
    dynamic_filter_event_record record{.join_operator_id         = join_operator_id,
                                       .exact_contribution_count = exact_contribution_count,
                                       .device_id                = root_device_id,
                                       .build_rows               = build_rows,
                                       .filters_built            = filters_built,
                                       .active_targets           = active_targets,
                                       .filters_pushed           = accepted_pushes};
    try {
      std::scoped_lock lock(event_mutex_);
      record.event_id = next_event_id_;
      event_records_.push_back(record);
      ++next_event_id_;
    } catch (...) {
      // Dynamic-filter telemetry is best effort and must never fail a query. Advancing the ID only
      // after push_back succeeds keeps every visible snapshot contiguous.
    }
  }

  /** Return a coherent copy of the append-only event journal and its high-water mark. */
  [[nodiscard]] dynamic_filter_event_snapshot event_snapshot() const
  {
    std::scoped_lock lock(event_mutex_);
    return {.last_event_id = next_event_id_ - 1, .records = event_records_};
  }

 private:
  mutable std::mutex event_mutex_;
  std::vector<dynamic_filter_event_record> event_records_;
  std::uint64_t next_event_id_ = 1;
};

}  // namespace sirius::op
