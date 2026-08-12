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

#include "op/dynamic_filter/exact_host_scalar.hpp"
#include "op/dynamic_filter/top_n_dynamic_filter_publish_plan.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::op {

struct dynamic_filter_stats;

/**
 * @brief Result of offering one local Top-N boundary to the coordinator
 */
enum class threshold_offer_result {
  ACCEPTED_FOR_PUBLICATION,  ///< Tightest so far; this call owns the publisher loop (Stage 4)
  COALESCED,                 ///< Tightest so far; an active publisher will flush it (Stage 4)
  NOT_TIGHTER,           ///< Boundary does not lexicographically strengthen the tightest; ignored
  NO_ACCEPTING_TARGET,   ///< Tightened `tightest_seen` only; no channel target exists/accepts
  UNSUPPORTED_BOUNDARY,  ///< Null first component or otherwise unpublishable Kth-row tuple
  REJECTED_STATE         ///< Coordinator is FINISHING, FINISHED, or CANCELLED
};

/**
 * @brief One local result's proof of a legal boundary
 *
 * Created only by the `sirius_physical_top_n::execute` seam after the stream-ordered
 * device-to-host copies of every key column of row `K - 1` have completed (see
 * `docs/super-sirius/dynamic-filters-top-n.md`, "Witness handoff"). The witness co-owns the
 * result batch so the K retained rows cannot be released while an asynchronous publication still
 * refers to them.
 */
struct top_n_threshold_witness {
  exact_host_key_tuple boundary;                     ///< Completed exact host Kth-row key tuple
  std::shared_ptr<cucascade::data_batch> witnesses;  ///< The K-row local result, kept alive
};

/**
 * @brief One batch's best distinct ORDER-BY key values, host-extracted
 *
 * Sorted best-first, deduplicated, at most K entries. Unlike @ref top_n_threshold_witness this
 * carries no batch handle and needs none: its producer completes every device-to-host copy before
 * constructing it, and those copies read the producer's own extraction table rather than the input
 * batch, so the witness holds completed host values and no device reference at all. The durability
 * rule that binds @ref top_n_threshold_witness is therefore moot here -- it still binds any future
 * producer whose witness does refer to device memory.
 */
struct top_n_distinct_key_witness {
  std::vector<exact_host_key_tuple> best_keys;
};

/**
 * @brief Execution-owned Top-N threshold policy: monotonic boundary, revisions, publication
 *
 * One coordinator per Top-N producer per execution, shared by the local and merge operators via
 * `std::shared_ptr`. Holds checked K, per-key semantics, the tightest host boundary tuple, and
 * metrics; the pending-candidate, revision, and publisher-loop state arrive with Stage 4
 * publication. Tightness is `exact_host_key_tuple::lex_compare` over the full tuple. It does not
 * discover targets, inspect DuckDB metadata, schedule scans, or decide final output.
 *
 * Threading: `offer`, `tightest_boundary`, `finish`, and `cancel` are thread-safe. Host
 * comparison happens under one short internal mutex; filter construction and replication (Stage
 * 4) run outside it with at most one publisher loop active. All state is execution-scoped and
 * starts empty (main doc, "Execution-scoped state").
 */
class top_n_threshold_coordinator final {
 public:
  /**
   * @brief Construct with frozen semantics from the planner
   *
   * @param[in] k Checked `limit + offset`
   * @param[in] keys Complete ORDER BY semantics, in key order; `keys[0]` is the first-key layer's
   * key
   * @param[in] lex_admitted True when every key's type is admitted -- enables the LEX layer and
   * the lexicographic prefilter; false degrades both to the first-key comparison
   * @param[in] stats Non-owning counter sink outliving every plan built during a query, exactly
   * as `sirius_physical_hash_join` receives it; nullable
   */
  top_n_threshold_coordinator(std::size_t k,
                              std::vector<top_n_key_semantics> keys,
                              bool lex_admitted,
                              dynamic_filter_stats* stats = nullptr,
                              top_n_producer_kind kind    = top_n_producer_kind::ROW);

  /**
   * @brief Install the frozen publication plan, giving this coordinator real targets
   *
   * Plan-time only, before any offer: the planner calls this once after discovery froze the
   * targets and replica spaces. Without it the coordinator stays target-free and every offer
   * returns `NO_ACCEPTING_TARGET` (the Stage-1 self-consumption-only producer).
   */
  void set_publish_plan(top_n_dynamic_filter_publish_plan plan);

  top_n_threshold_coordinator(top_n_threshold_coordinator const&)            = delete;
  top_n_threshold_coordinator& operator=(top_n_threshold_coordinator const&) = delete;

  /**
   * @brief Offer a K-witness boundary; monotonically tightens `tightest_seen`
   *
   * Stage 1 semantics: tighten the shared boundary for the sink prefilter and return
   * `NO_ACCEPTING_TARGET`. Stage 4 adds the publisher loop and the remaining results.
   */
  threshold_offer_result offer(top_n_threshold_witness witness);

  /**
   * @brief GROUP_KEY mode only: merge a batch's distinct keys into the bounded witness set
   *
   * Union by key *value* under the coordinator mutex, truncated to the K best; two tasks
   * witnessing the same grouping key must count once, or the set would claim K proven groups from
   * fewer and the K-distinct proof would not hold. The boundary is the set's Kth element, defined
   * only once the set is full, and tightens monotonically afterwards because adding candidates can
   * only improve the Kth best. Sub-K batches contribute, unlike row-mode offers. A Kth element
   * whose first component is null is `UNSUPPORTED_BOUNDARY`, the same conservative rule row mode
   * applies, and leaves the witness set untouched so a later key can still define a boundary.
   *
   * @throw std::logic_error when called on a ROW-mode coordinator
   */
  threshold_offer_result offer(top_n_distinct_key_witness witness);

  /// Which witness discipline this coordinator runs; fixed at construction.
  [[nodiscard]] top_n_producer_kind kind() const noexcept { return _kind; }

  /**
   * @brief Whether the published first-key bound admits rows tying the boundary
   *
   * A group-key producer is always inclusive: its boundary is the Kth-best *grouping key*, so a
   * row whose key equals it belongs to a group that is in the answer, and dropping it would lower
   * that group's aggregates. A row producer is inclusive only when a later key can still order a
   * key-zero tie, i.e. when it has more than one key.
   */
  [[nodiscard]] bool first_key_inclusive() const noexcept
  {
    return _kind == top_n_producer_kind::GROUP_KEY || _keys.size() > 1;
  }

  /**
   * @brief Whether the published full-tuple predicate admits rows tying the whole boundary
   *
   * Strict for a row producer -- full-tuple peers are interchangeable once K witnesses exist --
   * and inclusive for a group-key producer, whose ties are never interchangeable at any key count.
   */
  [[nodiscard]] bool lex_inclusive() const noexcept
  {
    return _kind == top_n_producer_kind::GROUP_KEY;
  }

  /**
   * @brief Current tightest boundary tuple for sink self-consumption, or empty before K witnesses
   *
   * A mutex-guarded host copy; deliberately not a channel read. The sink prefilter builds the
   * strict LEX predicate from it (the degraded inclusive first-key comparison when
   * `lex_admitted` is false) with task-local scalars. Staleness is safe -- a stale boundary
   * prunes less, never more.
   */
  [[nodiscard]] std::optional<exact_host_key_tuple> tightest_boundary() const;

  /**
   * @brief Monotonic count of accepted boundary tightenings
   *
   * The sink prefilter passes it to `dynamic_filter_gate` as the observed count, so a tightened
   * boundary re-arms exactly one keep-ratio measurement through the gate's existing growth rule.
   */
  [[nodiscard]] std::size_t boundary_update_count() const noexcept;

  /// Frozen checked `limit + offset`.
  [[nodiscard]] std::size_t k() const noexcept { return _k; }
  /// Frozen per-key semantics, in ORDER BY order.
  [[nodiscard]] std::span<top_n_key_semantics const> keys() const noexcept { return _keys; }
  /// Whether every key's type is admitted (full LEX prefilter) or only key zero (degraded).
  [[nodiscard]] bool lex_admitted() const noexcept { return _lex_admitted; }

  /// Fold one measured sink-prefilter batch into this producer kind's prefilter row counters:
  /// `top_n_prefilter_rows_*` for `ROW`, `top_n_group_prefilter_rows_*` for `GROUP_KEY`.
  void record_prefilter(std::size_t rows_in, std::size_t rows_out) noexcept;
  /// Count one keep-ratio disable decision of the sink prefilter.
  void record_prefilter_disabled() noexcept;

  /**
   * @brief Synchronous producer-side drain called by merge/finalization
   *
   * Transitions OPEN -> FINISHING, rejects later offers, joins/starts the publisher until pending
   * work is empty (Stage 4), then transitions to FINISHED. Idempotent and safe at any point after
   * construction. Never called or awaited by consumers.
   */
  void finish();

  /**
   * @brief Reject further offers and publication; safe under teardown ordering
   */
  void cancel() noexcept;

 private:
  enum class state { open, finishing, finished, cancelled };

  void bump(std::atomic<std::uint64_t> dynamic_filter_stats::* counter) const noexcept;

  /**
   * @brief Drain pending candidates until none remain, publishing one revision each
   *
   * Runs with no lock held except around the pending/revision handoff. Exactly one caller owns
   * the loop at a time; the pending-empty check and the `_publisher_active = false` handoff share
   * one critical section, so an offer arriving after that transition takes ownership itself and
   * no candidate is ever left without a publisher.
   */
  void publisher_loop();

  /**
   * @brief Build, replicate, and install one revision of every planned layer
   *
   * All-or-nothing for construction and replication: if any filter or any device replica fails,
   * nothing is installed and the previous revision stays visible everywhere. Per-target
   * best-effort for closure: a drained channel returns `CLOSED` and is skipped while the other
   * targets still receive this revision.
   */
  void publish_revision(exact_host_key_tuple const& boundary, std::uint64_t revision);

  std::size_t const _k;
  std::vector<top_n_key_semantics> const _keys;
  bool const _lex_admitted;
  dynamic_filter_stats* const _stats;
  top_n_producer_kind const _kind;

  /// Lock-free mirror of the accepted-tightening count backing @ref boundary_update_count.
  std::atomic<std::size_t> _boundary_updates{0};

  mutable std::mutex _mu;
  std::condition_variable _publisher_idle;  ///< Signalled when the publisher loop goes quiescent
  state _state = state::open;
  std::optional<exact_host_key_tuple> _tightest_seen;

  /// Frozen at plan time by @ref set_publish_plan, which refuses to replace it once any offer
  /// could have occurred; @ref publish_revision therefore reads it without the lock. Empty
  /// targets means self-consumption only.
  top_n_dynamic_filter_publish_plan _plan;

  /// Latched per target when its channel reports CLOSED. Written only by the single active
  /// publisher loop, so it needs no separate guard.
  std::vector<bool> _target_closed;

  /**
   * @brief GROUP_KEY mode: the K best distinct grouping keys witnessed so far
   *
   * Ordered best-first and deduplicated by value; never longer than K. Guarded by @c _mu.
   */
  std::vector<exact_host_key_tuple> _distinct_keys;
  bool _witness_set_full = false;  ///< Latched when the set first reaches K

  /// Tightest candidate awaiting publication, and whether a publisher loop owns the drain.
  std::optional<exact_host_key_tuple> _pending;
  bool _publisher_active       = false;
  std::uint64_t _next_revision = 1;
};

}  // namespace sirius::op
