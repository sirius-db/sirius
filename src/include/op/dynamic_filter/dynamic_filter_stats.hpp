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
 * @brief Plain copyable snapshot of @ref dynamic_filter_stats
 *
 * Field meanings are documented once, on the atomic aggregate. Each field is individually coherent
 * at any time; cross-field comparisons are meaningful only while no publisher is updating the
 * counters.
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
 * @brief Connection-lifetime dynamic-filter publication counters, owned by `SiriusContext`
 *
 * `dynamic_filter_publication_session` folds each terminal `dynamic_filter_publication_outcome`
 * into this sink through a non-owning pointer supplied by `sirius_plan_comparison_join`. The owning
 * `SiriusContext` outlives every plan built during a query -- the same lifetime contract as
 * `dynamic_filter_replica_space`.
 *
 * The fields have three timing classes.
 *
 * `producers_enabled` is a plan-time fact. `dynamic_filter_publication_session` increments it when
 * constructed with an enabled `dynamic_filter_publish_plan`, before execution begins. It counts
 * plan constructions, not executed producers: the transparent path builds the physical plan once at
 * prepare and again at execution, so a single query contributes twice per producing join. Compare
 * it across runs or use it as a direction, never as the left side of an accounting identity.
 *
 * The key and filter counters record policy decisions for attempts that reach per-key processing.
 * One-shot source-residency and all-targets-drained returns occur before that processing.
 * Accumulated attempts evaluate policy when armed and may therefore record those counters together
 * with a later all-targets-drained completion.
 *
 * The publication and delivery counters may vary with probe-side draining and target lifetime. Each
 * atomic is coherent independently; `snapshot()` does not provide a transactionally consistent view
 * across fields.
 *
 * Session counters describe publication claims rather than disjoint build deliveries.
 * `publication_attempts` begins when one-shot publication claims the session or accumulated-claim
 * initialization begins, and exactly one finished or failed terminal counter follows. Outside that
 * lifecycle, an otherwise eligible whole-build delivery that is not GPU resident increments
 * `publications_skipped_source_not_resident` without claiming the session, so broadcast deliveries
 * may repeat it. The caller latches `publications_skipped_build_not_whole` once per join.
 * Deliveries after the session closes are counted nowhere.
 */
struct dynamic_filter_stats {
  // Plan-time fact
  std::atomic<std::uint64_t> producers_enabled{0};  ///< Joins constructed with an enabled plan

  // Deterministic policy decisions
  std::atomic<std::uint64_t> keys_considered{0};  ///< Bound admitted keys walked by publication
                                                  ///< attempts
  std::atomic<std::uint64_t> keys_with_known_domain{0};        ///< Keys carrying a nonzero domain
  std::atomic<std::uint64_t> keys_skipped_domain_gate{0};      ///< Coverage gate fired
  std::atomic<std::uint64_t> keys_skipped_bloom_size_gate{0};  ///< Bloom budget gate fired
  std::atomic<std::uint64_t> keys_skipped_type_mismatch{0};    ///< Plan/runtime type disagreement
  std::atomic<std::uint64_t> keys_build_exceeded_domain{0};    ///< Build row count exceeded the
                                                               ///< recorded domain bound
  std::atomic<std::uint64_t> membership_filters_built{0};      ///< Constructed, before delivery
  std::atomic<std::uint64_t> zone_map_filters_built{0};        ///< Constructed, before delivery

  // Opportunistic delivery
  std::atomic<std::uint64_t> publication_attempts{
    0};  ///< OPEN claim initialization, including accumulator construction failure
  std::atomic<std::uint64_t> publications_finished{0};
  std::atomic<std::uint64_t> publications_failed{0};
  /// Build batch was not GPU-resident at delivery; publication skipped without claiming
  std::atomic<std::uint64_t> publications_skipped_source_not_resident{0};
  /// First build delivery for a wired join that is still OPEN but lacks a whole-build batch;
  /// counted once per join.
  std::atomic<std::uint64_t> publications_skipped_build_not_whole{0};
  /// Attempt found every target drained; accumulated attempts may retain arm-time policy counters
  std::atomic<std::uint64_t> publications_skipped_targets_drained{0};
  std::atomic<std::uint64_t> filters_pushed{0};  ///< Accepted pushes; drain-dependent

  /**
   * @brief Read every counter with relaxed ordering
   *
   * @return A copyable, non-transactional snapshot
   */
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
