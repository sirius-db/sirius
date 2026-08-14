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
 * at any time; cross-field comparisons are meaningful only while no publisher is updating the
 * counters.
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

  std::uint64_t top_n_producers_eligible       = 0;
  std::uint64_t top_n_producers_rejected       = 0;
  std::uint64_t top_n_producers_first_key_only = 0;
  std::uint64_t top_n_offers                   = 0;
  std::uint64_t top_n_offers_not_tighter       = 0;
  std::uint64_t top_n_offers_unsupported       = 0;
  std::uint64_t top_n_prefilter_rows_in        = 0;
  std::uint64_t top_n_prefilter_rows_out       = 0;
  std::uint64_t top_n_prefilter_disabled       = 0;

  std::uint64_t top_n_first_key_scan_targets          = 0;
  std::uint64_t top_n_lex_scan_targets                = 0;
  std::uint64_t top_n_first_key_endpoint_sites_placed = 0;
  std::uint64_t top_n_lex_endpoint_sites_placed       = 0;
  std::uint64_t top_n_sites_skipped_no_work_saved     = 0;
  std::uint64_t top_n_first_key_subsumed_by_lex       = 0;
  std::uint64_t top_n_revisions_published             = 0;
  std::uint64_t top_n_lex_filters_pushed              = 0;
  std::uint64_t top_n_first_key_filters_pushed        = 0;
  std::uint64_t top_n_revisions_failed                = 0;
  std::uint64_t top_n_revisions_stale                 = 0;
  std::uint64_t top_n_revisions_ignored               = 0;

  std::uint64_t top_n_group_producers_eligible = 0;
  std::uint64_t top_n_group_producers_rejected = 0;
  std::uint64_t top_n_group_offers             = 0;
  std::uint64_t top_n_group_witness_set_full   = 0;
  std::uint64_t top_n_group_prefilter_rows_in  = 0;
  std::uint64_t top_n_group_prefilter_rows_out = 0;

  std::uint64_t post_decode_apply_rows_in  = 0;
  std::uint64_t post_decode_apply_rows_out = 0;

  std::uint64_t reader_gate_row_groups_considered = 0;
  std::uint64_t reader_gate_row_groups_pruned     = 0;
  std::uint64_t reader_gate_measurements          = 0;
  std::uint64_t reader_gate_disabled              = 0;
  std::uint64_t reader_gate_rearmed               = 0;
  std::uint64_t reader_gate_merges_skipped        = 0;
};

/**
 * @brief Connection-lifetime dynamic-filter publication counters, owned by `SiriusContext`
 *
 * `sirius_physical_hash_join` folds each `dynamic_filter_publication_outcome` into this sink
 * through a non-owning pointer handed to it at construction by `sirius_plan_comparison_join`. The
 * owning `SiriusContext` outlives every plan built during a query -- the same lifetime contract as
 * `dynamic_filter_replica_space`.
 *
 * The fields have three timing classes.
 *
 * `producers_enabled` is a plan-time fact. `sirius_physical_hash_join` increments it when
 * constructed with an enabled `dynamic_filter_publish_plan`, before execution begins. It counts
 * plan constructions, not executed producers: the transparent path builds the physical plan once at
 * prepare and again at execution, so a single query contributes twice per producing join. Compare
 * it across runs or use it as a direction, never as the left side of an accounting identity.
 *
 * The key and filter counters record policy decisions for attempts that reach per-key processing. A
 * source-residency or all-targets-drained return occurs earlier and does not increment them.
 *
 * The publication and delivery counters may vary with probe-side draining and target lifetime. Each
 * atomic is coherent independently; `snapshot()` does not provide a transactionally consistent view
 * across fields.
 *
 * The delivery counters classify *build deliveries*, not joins, and they are disjoint but not
 * exhaustive. A delivery that claims the window goes on to either attempt publication or report its
 * source not resident. The first delivery of a wired join that cannot claim increments
 * `publications_skipped_build_not_whole`; that counter is latched, so the same join's later
 * non-claiming deliveries are counted nowhere, and neither is a delivery arriving after the window
 * has closed. Only the not-whole counter is latched -- the others can fire repeatedly for one join,
 * because a broadcast build delivers one batch per GPU.
 */
struct dynamic_filter_stats {
  // Plan-time fact
  std::atomic<std::uint64_t> producers_enabled{0};  ///< Joins constructed with an enabled plan

  // Deterministic policy decisions
  std::atomic<std::uint64_t> keys_considered{0};  ///< Bound admitted keys walked by publication
                                                  ///< attempts
  std::atomic<std::uint64_t> keys_with_known_domain{0};      ///< Keys carrying a nonzero domain
  std::atomic<std::uint64_t> keys_skipped_domain_gate{0};    ///< Coverage gate fired
  std::atomic<std::uint64_t> keys_skipped_type_mismatch{0};  ///< Plan/runtime type disagreement
  std::atomic<std::uint64_t> keys_build_exceeded_domain{0};  ///< Build row count exceeded the
                                                             ///< recorded domain bound
  std::atomic<std::uint64_t> membership_filters_built{0};    ///< Constructed, before delivery
  std::atomic<std::uint64_t> zone_map_filters_built{0};      ///< Constructed, before delivery

  // Opportunistic delivery
  std::atomic<std::uint64_t> publication_attempts{0};  ///< OPEN -> PUBLISHING claims
  std::atomic<std::uint64_t> publications_finished{0};
  std::atomic<std::uint64_t> publications_failed{0};
  /// Build batch was not GPU-resident at delivery; publication skipped without claiming
  std::atomic<std::uint64_t> publications_skipped_source_not_resident{0};
  /// Wired join the upstream PARTITION never reported a whole build for, so its one-shot
  /// publication window can never claim: probe-driven sizing, a hash-partitioned multi-partition
  /// build (a broadcast build is whole on every partition), or no build-side CONCAT to fold.
  /// Counted once per join, at the first build delivery that observes the condition.
  std::atomic<std::uint64_t> publications_skipped_build_not_whole{0};
  /// Attempt hit the all-targets-drained early return; no deterministic counter moved for it
  std::atomic<std::uint64_t> publications_skipped_targets_drained{0};
  std::atomic<std::uint64_t> filters_pushed{0};  ///< Accepted pushes; drain-dependent

  // --- Top-N refinement (Stage 1) ---
  // The producer counters are plan-time facts like producers_enabled (the transparent path
  // constructs the plan twice per query); the offer and prefilter counters are delivery-time and
  // batch-arrival dependent, so tests assert them as deltas or directions only.
  std::atomic<std::uint64_t> top_n_producers_eligible{0};  ///< Plan-time fact, like
                                                           ///< producers_enabled
  std::atomic<std::uint64_t> top_n_producers_rejected{0};  ///< Failed eligibility (keys/shape/K)
  std::atomic<std::uint64_t> top_n_producers_first_key_only{0};  ///< Tail key type degraded LEX
                                                                 ///< away
  std::atomic<std::uint64_t> top_n_offers{0};                    ///< Witness offers reaching the
                                                                 ///< coordinator
  std::atomic<std::uint64_t> top_n_offers_not_tighter{0};        ///< Lost the lexicographic compare
  std::atomic<std::uint64_t> top_n_offers_unsupported{0};        ///< Null first boundary component
  std::atomic<std::uint64_t> top_n_prefilter_rows_in{0};   ///< Rows entering measured prefilters
  std::atomic<std::uint64_t> top_n_prefilter_rows_out{0};  ///< Rows surviving measured prefilters
  std::atomic<std::uint64_t> top_n_prefilter_disabled{0};  ///< Keep-ratio disable observations;
                                                           ///< concurrent measurements may record
                                                           ///< one decision more than once

  // --- Top-N publication and endpoints (Stage 4); per layer where meaningful ---
  // The scan-target, endpoint-site, and subsumption counters are plan-time facts like
  // producers_enabled; the revision and push counters are delivery-time and race scan starts by
  // design, so tests assert them as deltas or directions only.
  std::atomic<std::uint64_t> top_n_first_key_scan_targets{0};           ///< Plan-time
  std::atomic<std::uint64_t> top_n_lex_scan_targets{0};                 ///< Plan-time; all keys
                                                                        ///< one scan
  std::atomic<std::uint64_t> top_n_first_key_endpoint_sites_placed{0};  ///< Plan-time
  /// Plan-time; a sited endpoint carrying the full-tuple predicate, one per arrive-together site
  std::atomic<std::uint64_t> top_n_lex_endpoint_sites_placed{0};
  /// Plan-time; the siting rule declined -- the target neither reads less because of the predicate
  /// nor shields per-row work from it, so it would only repeat the sink prefilter's own pass.
  /// Covers endpoints and post-decode-only scan binds alike. Also counts the type-admission
  /// refusal (`boundary_key_matches_site_type` mismatch) -- deliberately shared: type-preserving
  /// hops make that refusal unconstructible through a built plan, so it is pinned at the
  /// pure-function level rather than given a counter no plan can move.
  std::atomic<std::uint64_t> top_n_sites_skipped_no_work_saved{0};
  std::atomic<std::uint64_t> top_n_first_key_subsumed_by_lex{0};  ///< Plan-time; dedup fired
  std::atomic<std::uint64_t> top_n_revisions_published{0};        ///< Boundary updates fanned out
  std::atomic<std::uint64_t> top_n_lex_filters_pushed{0};
  std::atomic<std::uint64_t> top_n_first_key_filters_pushed{0};
  std::atomic<std::uint64_t> top_n_revisions_failed{0};  ///< Replica failure; old revision retained
  std::atomic<std::uint64_t> top_n_revisions_stale{0};
  /// Publishes refused because the slot's primary or a referenced ordinal is ignored
  std::atomic<std::uint64_t> top_n_revisions_ignored{0};

  // --- Top-N group-key producer (Stage 5) ---
  // The producer counters are plan-time facts; the offer, witness-set, and prefilter counters are
  // delivery-time and batch-arrival dependent, so tests assert them as deltas or directions only.
  std::atomic<std::uint64_t> top_n_group_producers_eligible{0};  ///< Plan-time fact
  std::atomic<std::uint64_t> top_n_group_producers_rejected{0};  ///< Aggregate-output key, filter,
                                                                 ///< or K/shape refusal
  std::atomic<std::uint64_t> top_n_group_offers{0};              ///< Distinct-key offers merged
  std::atomic<std::uint64_t> top_n_group_witness_set_full{0};    ///< Boundary first became defined
  std::atomic<std::uint64_t> top_n_group_prefilter_rows_in{0};
  std::atomic<std::uint64_t> top_n_group_prefilter_rows_out{0};

  // --- Post-decode consumer (pinned-serve flip) ---
  // Delivery-time: sirius_physical_dynamic_filter adds one batch's rows when its gated apply
  // produced a result table (at least one filter computed a mask or compaction dropped rows;
  // gate-declined and no-applicable-filter batches count nowhere). Covers every provenance
  // (scan_route, join_edge, top_n_endpoint), every capability, and both modes -- a test that
  // must isolate one capability uses a channel carrying only that capability. Batch arrival
  // races publication by design, so tests assert deltas or directions only.
  std::atomic<std::uint64_t> post_decode_apply_rows_in{0};
  std::atomic<std::uint64_t> post_decode_apply_rows_out{0};

  // --- Parquet reader pruning gate (WI-0b) ---
  // All delivery-time: recorded inside parquet_gpu_ingestible::materialize_metadata_to_table,
  // which runs only for the executing plan -- the transparent path's prepare-time plan owns a
  // separate ingestible that never materializes, so unlike the plan-time top_n_* facts these
  // never double. considered/pruned copy the reader's own accounting (cudf::io::table_metadata)
  // for splits whose reader AST carried merged dynamic conjuncts; measurements counts those
  // samples; disabled counts transitions into the disabled state (re-disables included); rearmed
  // counts backoff-permitted re-measurements taken while disabled; merges_skipped counts splits
  // whose dynamic merge the gate skipped. A pinned-cache-served scan runs no reader and moves
  // none of these. Batch/publication timing is racy by design, so tests assert deltas or
  // directions only.
  std::atomic<std::uint64_t> reader_gate_row_groups_considered{0};
  std::atomic<std::uint64_t> reader_gate_row_groups_pruned{0};
  std::atomic<std::uint64_t> reader_gate_measurements{0};
  std::atomic<std::uint64_t> reader_gate_disabled{0};
  std::atomic<std::uint64_t> reader_gate_rearmed{0};
  std::atomic<std::uint64_t> reader_gate_merges_skipped{0};

  /**
   * @brief Read every counter with relaxed ordering
   *
   * @return A copyable, non-transactional snapshot
   */
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
      .filters_pushed           = filters_pushed.load(std::memory_order_relaxed),
      .top_n_producers_eligible = top_n_producers_eligible.load(std::memory_order_relaxed),
      .top_n_producers_rejected = top_n_producers_rejected.load(std::memory_order_relaxed),
      .top_n_producers_first_key_only =
        top_n_producers_first_key_only.load(std::memory_order_relaxed),
      .top_n_offers                 = top_n_offers.load(std::memory_order_relaxed),
      .top_n_offers_not_tighter     = top_n_offers_not_tighter.load(std::memory_order_relaxed),
      .top_n_offers_unsupported     = top_n_offers_unsupported.load(std::memory_order_relaxed),
      .top_n_prefilter_rows_in      = top_n_prefilter_rows_in.load(std::memory_order_relaxed),
      .top_n_prefilter_rows_out     = top_n_prefilter_rows_out.load(std::memory_order_relaxed),
      .top_n_prefilter_disabled     = top_n_prefilter_disabled.load(std::memory_order_relaxed),
      .top_n_first_key_scan_targets = top_n_first_key_scan_targets.load(std::memory_order_relaxed),
      .top_n_lex_scan_targets       = top_n_lex_scan_targets.load(std::memory_order_relaxed),
      .top_n_first_key_endpoint_sites_placed =
        top_n_first_key_endpoint_sites_placed.load(std::memory_order_relaxed),
      .top_n_lex_endpoint_sites_placed =
        top_n_lex_endpoint_sites_placed.load(std::memory_order_relaxed),
      .top_n_sites_skipped_no_work_saved =
        top_n_sites_skipped_no_work_saved.load(std::memory_order_relaxed),
      .top_n_first_key_subsumed_by_lex =
        top_n_first_key_subsumed_by_lex.load(std::memory_order_relaxed),
      .top_n_revisions_published = top_n_revisions_published.load(std::memory_order_relaxed),
      .top_n_lex_filters_pushed  = top_n_lex_filters_pushed.load(std::memory_order_relaxed),
      .top_n_first_key_filters_pushed =
        top_n_first_key_filters_pushed.load(std::memory_order_relaxed),
      .top_n_revisions_failed  = top_n_revisions_failed.load(std::memory_order_relaxed),
      .top_n_revisions_stale   = top_n_revisions_stale.load(std::memory_order_relaxed),
      .top_n_revisions_ignored = top_n_revisions_ignored.load(std::memory_order_relaxed),
      .top_n_group_producers_eligible =
        top_n_group_producers_eligible.load(std::memory_order_relaxed),
      .top_n_group_producers_rejected =
        top_n_group_producers_rejected.load(std::memory_order_relaxed),
      .top_n_group_offers           = top_n_group_offers.load(std::memory_order_relaxed),
      .top_n_group_witness_set_full = top_n_group_witness_set_full.load(std::memory_order_relaxed),
      .top_n_group_prefilter_rows_in =
        top_n_group_prefilter_rows_in.load(std::memory_order_relaxed),
      .top_n_group_prefilter_rows_out =
        top_n_group_prefilter_rows_out.load(std::memory_order_relaxed),
      .post_decode_apply_rows_in  = post_decode_apply_rows_in.load(std::memory_order_relaxed),
      .post_decode_apply_rows_out = post_decode_apply_rows_out.load(std::memory_order_relaxed),
      .reader_gate_row_groups_considered =
        reader_gate_row_groups_considered.load(std::memory_order_relaxed),
      .reader_gate_row_groups_pruned =
        reader_gate_row_groups_pruned.load(std::memory_order_relaxed),
      .reader_gate_measurements   = reader_gate_measurements.load(std::memory_order_relaxed),
      .reader_gate_disabled       = reader_gate_disabled.load(std::memory_order_relaxed),
      .reader_gate_rearmed        = reader_gate_rearmed.load(std::memory_order_relaxed),
      .reader_gate_merges_skipped = reader_gate_merges_skipped.load(std::memory_order_relaxed)};
  }
};

}  // namespace sirius::op
