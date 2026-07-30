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
#include <string_view>
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

  // ── Offline plan measurements ──────────────────────────────────────────────

  /**
   * @brief Measurements recorded alongside an offline column plan.
   *
   * The generator writes these as `#` comments above each block:
   *
   *     # column: l_orderkey  dtype: i64  ratio: 20.711x  depth: 2
   *     # comp: 531.34 GB/s  decomp: 626.76 GB/s
   *
   * simpatico::split_plan_dsl strips every comment line, so the numbers are lost
   * the moment a plan is turned into blocks. They are parsed out separately at
   * load time and kept index-aligned with those blocks.
   */
  struct plan_metrics {
    double compression_ratio{0.0};
    double compress_gbps{0.0};
    double decompress_gbps{0.0};
  };

  /**
   * @brief Thresholds a plan must clear to be worth compressing eagerly.
   *
   * Unlike the spill path — which compresses because it has to — an eager
   * consumer chooses to spend GPU time, so it only pays for a plan that is both
   * fast and worth it. Compress *and* decompress are gated: the data is written
   * once and read back at most once, so a plan that is cheap to read but slow to
   * write is no bargain on the critical path.
   */
  struct plan_quality_gate {
    double min_ratio{3.0};
    double min_compress_gbps{250.0};
    double min_decompress_gbps{250.0};

    [[nodiscard]] bool admits(const plan_metrics& m) const
    {
      return m.compression_ratio > min_ratio && m.compress_gbps > min_compress_gbps &&
             m.decompress_gbps > min_decompress_gbps;
    }
  };

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

  /**
   * @brief Measurements for @p table_name's plan block at @p column_index.
   *
   * Indices share the space `select_plan_blocks` uses — one entry per non-empty
   * block, in schema order — so a `spill_column_origin::table_column_index` looks
   * up both the plan and its metrics. Returns nullopt when the table has no plan,
   * the index is out of range, or that block carried no measurement comments
   * (hand-written plans typically do not).
   */
  [[nodiscard]] std::optional<plan_metrics> resolve_plan_metrics(const std::string& table_name,
                                                                 std::size_t column_index) const;

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

  /// Where a spilled column came from in the base tables, as resolved by the
  /// planner. Deliberately a plain string/index pair rather than
  /// `sirius::planner::column_origin`, so the compression layer does not pull in
  /// DuckDB planner headers; the wiring site converts.
  struct spill_column_origin {
    std::string table_name;
    std::size_t table_column_index{0};
  };

  /// Per-column origins for an edge, nullopt where the column is computed.
  using spill_column_origins = std::vector<std::optional<spill_column_origin>>;

  /// Record where @p repo's columns came from. Called once at plan-wiring time.
  void set_spill_column_origins(const cucascade::shared_data_repository* repo,
                                spill_column_origins origins);

  /// Origins for @p repo, or nullopt when none were resolved.
  [[nodiscard]] std::optional<spill_column_origins> resolve_spill_column_origins(
    const cucascade::shared_data_repository* repo) const;

  /**
   * @brief Per-column plans for @p repo taken from the offline table plans.
   *
   * For each column with a known origin, looks up that table's plan and extracts
   * the block for its column index. Columns with no origin, or whose table has no
   * plan loaded, come back as nullopt for the caller to handle (store raw, or
   * explore).
   *
   * Returns nullopt when nothing at all could be seeded, so the caller can fall
   * back to exploring rather than compressing everything as passthrough.
   */
  [[nodiscard]] std::optional<std::vector<std::optional<std::string>>> seed_plans_from_lineage(
    const cucascade::shared_data_repository* repo, std::size_t expected_columns) const;

  /// One column of an edge worth compressing eagerly, with the plan to use and
  /// the base-table measurements that justified admitting it.
  struct output_plan_selection {
    std::size_t column_index{0};
    std::string dsl;
    plan_metrics metrics;

    /// True when the plan's ratio depends on the base table's row *order* — any
    /// cascade containing `delta`. The offline ratio was measured on the sorted
    /// base table; a task output has been through joins and hash partitioning,
    /// so that ordering is gone and the plan may compress far worse than its
    /// recorded ratio. The caller admits these but must verify the ratio it
    /// actually achieves on the first batch (see conclude_spill_attempt) rather
    /// than trusting `metrics`.
    bool order_dependent{false};
  };

  /**
   * @brief Columns of @p repo whose base-table plan clears @p gate.
   *
   * Resolves each column through lineage to its base-table plan, then admits only
   * those whose recorded measurements pass. Columns with no origin (aggregate
   * results, computed expressions), no loaded table plan, or no measurement
   * comments are skipped — an unmeasured plan cannot be shown to clear the bar,
   * and eager compression is opt-in per column rather than best-effort.
   *
   * Returns an empty vector when nothing qualifies.
   */
  [[nodiscard]] std::vector<output_plan_selection> select_output_plans(
    const cucascade::shared_data_repository* repo,
    std::size_t expected_columns,
    const plan_quality_gate& gate) const;

  // ── Eager output compression (keyed by shared_data_repository*) ─────────────
  //
  // Decided once per edge, on its first output batch, and then reused. Unlike
  // the spill path there is no explorer here: a column is compressed only when
  // lineage offers a plan whose *measured* characteristics clear the gate, so an
  // edge with no qualifying column costs one lookup and nothing more.

  /// Per-column plan for eager output compression; nullopt means store raw.
  using output_column_plans = std::vector<std::optional<std::string>>;

  /**
   * @brief Plans to compress @p repo's output with, deciding on first call.
   *
   * The first call runs @ref select_output_plans and caches the result; later
   * calls return the cached decision, minus any column since written off by
   * @ref conclude_output_attempt. Returns nullopt when no column qualifies (or
   * all have been written off), which is the common case and must stay cheap.
   */
  [[nodiscard]] std::optional<output_column_plans> decide_output_plan(
    const cucascade::shared_data_repository* repo,
    std::size_t expected_columns,
    const plan_quality_gate& gate);

  /**
   * @brief Record what compression actually achieved on @p repo's output.
   *
   * @p achieved_ratios is per column, 0 for columns that were stored raw. A
   * column whose achieved ratio misses `gate.min_ratio` is written off, so later
   * batches on this edge store it raw.
   *
   * This is what makes admitting an order-dependent (delta) plan safe. Its
   * recorded ratio was measured on the sorted base table; a task output has been
   * through joins and hash partitioning, so the only way to know whether the
   * ordering survived is to compress one batch and look. A plan that does not
   * deliver is dropped after a single wasted pass rather than on every batch.
   */
  void conclude_output_attempt(const cucascade::shared_data_repository* repo,
                               std::span<const double> achieved_ratios,
                               const plan_quality_gate& gate);

  /// Test seam: the cached per-column decision for @p repo, if one was made.
  [[nodiscard]] std::optional<output_column_plans> resolve_output_plan(
    const cucascade::shared_data_repository* repo) const;

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
    /// One entry per source column, in schema order. Empty when no exploration
    /// has yet succeeded for this edge.
    std::vector<column_plan_state> columns;

    /// Consecutive failed *explorations* for this edge.
    ///
    /// Exploration allocates on the GPU, and the spill path runs it precisely
    /// when the GPU is full, so it can fail repeatedly. It fails before any
    /// per-column state exists, so the streak lives here rather than on a
    /// column: without it there is nothing to record against and every later
    /// spill re-runs the whole beam search.
    std::uint32_t explore_failures{0};

    /// Set once `explore_failures` reaches the configured tolerance: stop asking
    /// for an exploration until this entry expires.
    bool explore_exhausted{false};

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

  /**
   * @brief Record that exploring @p repo's columns failed.
   *
   * Creates the entry if there is none, so the streak has somewhere to live —
   * exploration fails before any per-column state exists. Once
   * @p error_tolerance consecutive explorations have failed, decide_spill_plan()
   * returns `skip` for the edge instead of asking for another one, until the
   * entry expires and the edge is retried on the normal replan schedule.
   */
  void note_spill_explore_failure(const cucascade::shared_data_repository* repo,
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

  /**
   * @brief Drop every per-query spill entry: cached plans, verdicts and origins.
   *
   * Must run at query end. Spill state is keyed by `shared_data_repository*`, and
   * those repositories are destroyed between queries — so without this the maps
   * grow without bound holding entries keyed by freed pointers, and a repository
   * later allocated at a recycled address inherits state belonging to an
   * unrelated edge.
   *
   * Leaves the offline table plans (`set_table_plan`) alone: those come from
   * `input_plan_dir` at startup rather than from a query, and are what the
   * lineage seeding reads.
   */
  void clear_spill_state();

  /// Remove all entries (table-level, per-column, and spill-path).
  void clear_all();

 private:
  mutable std::shared_mutex _mutex;
  std::unordered_map<std::string, std::string> _table_plans;  // table_name → full multi-col DSL
  // table_name → per-block measurements, index-aligned with select_plan_blocks.
  // Populated by set_table_plan; empty where a plan carried no `#` metric lines.
  std::unordered_map<std::string, std::vector<std::optional<plan_metrics>>> _table_plan_metrics;
  std::unordered_map<std::string, std::string> _col_plans;  // "table::column" → single-col DSL
  // repo* → per-edge spill state; keyed by pointer (stable within a query)
  std::unordered_map<const cucascade::shared_data_repository*, spill_plan_state> _spill_plans;
  // repo* → per-column base-table origins, recorded once at plan-wiring time.
  std::unordered_map<const cucascade::shared_data_repository*, spill_column_origins> _spill_origins;

  /// Cached eager-output decision for one edge.
  struct output_edge_state {
    output_column_plans columns;  ///< nullopt entry = store this column raw
    bool any_viable{false};       ///< false once every column has been written off
  };
  std::unordered_map<const cucascade::shared_data_repository*, output_edge_state> _output_plans;
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

/**
 * @brief Parse the per-block `#` measurement comments out of a whole-table plan.
 *
 * Mirrors simpatico::split_plan_dsl's block splitting exactly — same "---"
 * separators, same dropping of blocks with no DSL lines — so entry i describes
 * block i as select_plan_blocks numbers them. A block whose comments are missing
 * or unparsable yields nullopt rather than shifting its neighbours.
 *
 * Kept here rather than in Simpatico because the comments are a convention of the
 * plan *generator*, not part of the DSL the parser accepts.
 */
[[nodiscard]] std::vector<std::optional<plan_register::plan_metrics>> parse_plan_metrics(
  std::string_view full_plan_dsl);

}  // namespace sirius::compression
