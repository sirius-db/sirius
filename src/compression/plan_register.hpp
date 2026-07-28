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

#include <cucascade/data/data_repository.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::compression {

/**
 * @brief Thread-safe registry for Simpatico column-compression plan DSL strings.
 *
 * Keys are "<table_name>::<column_name>" strings. Returns single-column plan DSL
 * blocks as strings. The global singleton is accessed via plan_register::global().
 *
 * Usage pattern:
 *   - Set a default plan (applies to all columns that lack a specific entry).
 *   - Optionally override per (table, column) pairs.
 *   - Call resolve_table_plan to get the full N-column plan DSL for compression.
 */
class plan_register {
 public:
  static plan_register& global();

  // ── Per-table whole-plan entries (input-table compression) ─────────────────

  /**
   * @brief Store the complete multi-column Simpatico plan DSL for @p table_name.
   *
   * The DSL must be the "---"-separated string that compress_with_plan expects
   * (one block per column, in schema order).  Overwrites any previous entry.
   */
  void set_table_plan(const std::string& table_name, std::string full_plan_dsl);

  /// Remove the whole-table plan entry for @p table_name.
  void clear_table_plan(const std::string& table_name);

  /**
   * @brief Return the whole-table plan DSL for @p table_name, or nullopt if none.
   *
   * Does not fall back to per-column entries or the default plan — this is
   * the direct input-table lookup used by PinTableFunction.
   */
  [[nodiscard]] std::optional<std::string> resolve_table_plan(const std::string& table_name) const;

  // ── Per-column entries (used by explore / manual overrides) ─────

  /// Register (or overwrite) a single-column plan for a specific (table, column) pair.
  void set_plan(const std::string& table_name,
                const std::string& column_name,
                std::string plan_dsl);

  /// Remove the per-(table, column) plan override if present.
  void clear_plan(const std::string& table_name, const std::string& column_name);

  // ── Spill-path plan entries (keyed by shared_data_repository*) ──────────
  //
  // One entry per query-graph edge (operator output port). The repo pointer is
  // stable for the lifetime of a query and uniquely identifies the output schema
  // + data distribution. Plans are discovered lazily on first spill via
  // simpatico::explore_column_compression and reused for later batches.
  //
  // An entry also remembers whether compression turned out to be *worth it* for
  // that edge. When a compressed batch misses the size threshold the entry is
  // marked unviable, so later batches skip the (futile) compress attempt
  // entirely instead of paying for it and discarding the result every time.
  // The verdict is not permanent: after `replan_after_uses` batches the entry
  // expires and the edge is explored afresh, which also re-tests an edge that
  // was previously marked unviable.

  /// Cached spill-compression state for a single column of one edge.
  ///
  /// Compressibility is a property of a column, not of a batch: a wide output
  /// commonly mixes columns that shrink 10x with ones that do not compress at
  /// all. Verdicts are therefore tracked per column, so one incompressible
  /// column neither disables its well-compressing neighbours nor keeps costing
  /// a compress attempt on every batch.
  struct column_plan_state {
    std::string dsl;    ///< single-column plan DSL for this column
    bool viable{true};  ///< false once this column proved not worth compressing

    /// Consecutive hard failures since this column's last real verdict. As at
    /// edge level, an error is not evidence about the data, so a column is only
    /// written off once the failures prove durable.
    std::uint32_t consecutive_errors{0};

    // Explorer-reported characteristics of `dsl` when it was adopted. Compared
    // against a later exploration to decide whether the new plan is materially
    // different — see set_spill_plan.
    double compression_ratio{1.0};
    double compress_gbps{0.0};
    double decompress_gbps{0.0};
  };

  /// A plan the explorer produced for one column, with the measurements that
  /// justify it. Offered to set_spill_plan, which decides whether adopting it
  /// over the cached plan is worthwhile.
  struct column_plan_candidate {
    std::string dsl;
    double compression_ratio{1.0};
    double compress_gbps{0.0};
    double decompress_gbps{0.0};
  };

  /// Cached spill-compression state for one query-graph edge.
  struct spill_plan_state {
    /// One entry per source column, in schema order.
    std::vector<column_plan_state> columns;

    std::uint64_t uses{0};  ///< spill attempts since this entry was installed

    /// Effective re-explore interval for this edge. 0 = follow the configured
    /// `spill_replan_after_uses`; non-zero once adaptive backoff has moved it.
    /// The schedule is per edge — batches arrive per edge, so all its columns
    /// are re-explored together.
    std::uint64_t replan_interval{0};

    // Bookkeeping describing the re-explore that installed this entry, consumed
    // by conclude_spill_attempt() to decide whether to back off.
    bool from_replan{false};           ///< this entry replaced an earlier one
    bool plan_changed{false};          ///< ...and at least one column's DSL differs
    std::size_t prev_viable_count{0};  ///< ...and how many of its columns were viable

    /// Columns currently worth compressing.
    [[nodiscard]] std::size_t viable_count() const
    {
      std::size_t n = 0;
      for (auto const& c : columns) {
        if (c.viable) { ++n; }
      }
      return n;
    }
  };

  /// What the spill path should do for an edge.
  enum class spill_plan_verdict {
    explore,  ///< no usable entry (absent or expired) — run the explorer
    use,      ///< compress the columns in `columns` that are still viable
    skip,     ///< no column is worth compressing — spill uncompressed
  };

  struct spill_plan_decision {
    spill_plan_verdict verdict{spill_plan_verdict::explore};
    /// Per-column state, set when verdict == use. Columns whose `viable` is
    /// false should be stored with a passthrough plan rather than compressed.
    std::vector<column_plan_state> columns;
  };

  /**
   * @brief Decide what the spill path should do for @p repo.
   *
   * @param replan_after_uses  Expire the entry once it has been used this many
   *                           times, forcing a fresh explore (0 = never expire).
   *                           Used only while the entry is on the configured
   *                           schedule; once adaptive backoff has moved the
   *                           entry's own interval, that takes precedence.
   *
   * An expired entry yields `explore` regardless of its previous verdict, so a
   * plan that stopped paying off — or an edge wrongly judged unviable from an
   * unrepresentative early batch — is reconsidered.
   */
  [[nodiscard]] spill_plan_decision decide_spill_plan(const cucascade::shared_data_repository* repo,
                                                      std::uint64_t replan_after_uses) const;

  /**
   * @brief Offer freshly explored per-column plans for @p repo (schema order).
   *
   * @param change_threshold  relative change in compression ratio or in either
   *                          throughput below which a candidate is considered
   *                          equivalent to the cached plan (e.g. 0.2 = 20%).
   *
   * With no cached entry every candidate is adopted, viable, with the use count
   * at zero.
   *
   * Replacing an entry, each column is decided on its own. The explorer is a
   * beam search over a large space and readily returns a *differently spelled*
   * plan that performs the same; adopting those would churn the cache and — worse
   * — register as a change, resetting the replan backoff and locking the edge
   * into re-exploring forever. So a candidate is only adopted when its ratio or
   * one of its throughputs differs from the cached plan's by more than
   * @p change_threshold. An adopted column resets to viable with a clear error
   * streak; a column that keeps its cached plan keeps its verdict too, since an
   * equivalent plan will not compress any better than the one already judged.
   *
   * Only genuinely adopted columns mark the entry as changed for
   * conclude_spill_attempt(), so an all-equivalent re-explore backs off.
   */
  void set_spill_plan(const cucascade::shared_data_repository* repo,
                      std::vector<column_plan_candidate> candidates,
                      double change_threshold);

  /// How a spill attempt ended.
  enum class spill_attempt_outcome {
    compressed,    ///< the compressed form was kept
    not_worth_it,  ///< measured: compressed size missed the threshold
    failed,        ///< errored out — possibly transient (e.g. OOM under pressure)
  };

  /**
   * @brief Record how a spill attempt for @p repo turned out, per column.
   *
   * @param per_column       one outcome per column, in schema order. Empty means
   *                         the attempt died before any column could be judged,
   *                         and is treated as `failed` for every column.
   * @param base_interval    the configured `spill_replan_after_uses`.
   * @param error_tolerance  consecutive `failed` outcomes to absorb before
   *                         writing a column off (minimum 1).
   *
   * `compressed` and `not_worth_it` are *measurements* and take effect at once:
   * they set that column's viability and clear its error streak.
   *
   * `failed` is not a measurement — compression runs under memory pressure, so an
   * exception is as likely to be a transient allocation failure as a real verdict
   * on the data. It only increments that column's error streak, leaving viability
   * untouched, until @p error_tolerance consecutive failures make it durable.
   * Without this a single transient OOM would disable compression for a whole
   * replan interval and stretch that interval further.
   *
   * When at least one column was measured, this also adapts the edge's replan
   * interval. Re-exploring costs a beam search per column, so it should only stay
   * frequent while it is paying off:
   *
   *   - the cycle produced a *working change* (plans changed, or more columns are
   *     viable than before, and at least one column compresses) → reset the
   *     interval to @p base_interval and keep checking on schedule;
   *   - anything else — the explorer returned the same plans, or nothing
   *     compresses → double the interval, so a stable or stubbornly
   *     incompressible edge stops paying for explores it learns nothing from.
   *
   * Call exactly once per attempt that actually tried to compress (not for a
   * skipped edge, which made no attempt to judge).
   */
  void conclude_spill_attempt(const cucascade::shared_data_repository* repo,
                              std::span<const spill_attempt_outcome> per_column,
                              std::uint64_t base_interval,
                              std::uint32_t error_tolerance);

  /// Count one spill attempt against @p repo's entry. Call exactly once per
  /// attempt, including attempts that were skipped or that failed — otherwise a
  /// skipped edge would never accumulate uses and never be re-explored.
  void note_spill_plan_use(const cucascade::shared_data_repository* repo);

  /// Remove the spill entry for @p repo.
  void clear_spill_plan(const cucascade::shared_data_repository* repo);

  /// Return the raw spill state for @p repo, or nullopt if none. Mainly for tests
  /// and diagnostics; the spill path itself uses decide_spill_plan().
  [[nodiscard]] std::optional<spill_plan_state> resolve_spill_plan(
    const cucascade::shared_data_repository* repo) const;

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /// Remove all entries (table-level, per-column, and spill-path).
  void clear_all();

 private:
  mutable std::shared_mutex _mutex;
  std::unordered_map<std::string, std::string> _table_plans;  // table_name → full multi-col DSL
  std::unordered_map<std::string, std::string> _col_plans;    // "table::column" → single-col DSL
  // repo* → per-edge spill state; keyed by pointer (stable within a query)
  std::unordered_map<const cucascade::shared_data_repository*, spill_plan_state> _spill_plans;
};

/**
 * @brief Select the plan blocks for a pinned column subset.
 *
 * A whole-table plan DSL has one "---"-separated block per full-table column, in
 * schema order. When a pin caches only some columns, @p column_indices gives the
 * full-table index of each pinned column (in the pinned/materialized order); this
 * returns a DSL with just those blocks, in that order, so it lines up 1:1 with the
 * pinned table for compress_with_plan. Returns nullopt if any index is out of
 * range (the plan does not cover a pinned column), so the caller pins uncompressed.
 */
[[nodiscard]] std::optional<std::string> select_plan_blocks(
  const std::string& full_plan_dsl, const std::vector<std::size_t>& column_indices);

}  // namespace sirius::compression
