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

  /// Cached spill-compression state for one query-graph edge.
  struct spill_plan_state {
    std::string dsl;        ///< multi-column plan DSL for this edge
    bool viable{true};      ///< false once compression was judged not worth it
    std::uint64_t uses{0};  ///< spill attempts since this entry was installed
  };

  /// What the spill path should do for an edge.
  enum class spill_plan_verdict {
    explore,  ///< no usable entry (absent or expired) — run the explorer
    use,      ///< compress with the returned DSL
    skip,     ///< compression is not worth it here — spill uncompressed
  };

  struct spill_plan_decision {
    spill_plan_verdict verdict{spill_plan_verdict::explore};
    std::string dsl;  ///< set when verdict == use
  };

  /**
   * @brief Decide what the spill path should do for @p repo.
   *
   * @param replan_after_uses  Expire the entry once it has been used this many
   *                           times, forcing a fresh explore (0 = never expire).
   *
   * An expired entry yields `explore` regardless of its previous verdict, so a
   * plan that stopped paying off — or an edge wrongly judged unviable from an
   * unrepresentative early batch — is reconsidered.
   */
  [[nodiscard]] spill_plan_decision decide_spill_plan(const cucascade::shared_data_repository* repo,
                                                      std::uint64_t replan_after_uses) const;

  /// Install a freshly explored plan for @p repo (resets viability and use count).
  void set_spill_plan(const cucascade::shared_data_repository* repo, std::string plan_dsl);

  /// Record that compression is not worth it for @p repo. Keeps the DSL and the
  /// use count, so the entry still expires on schedule and gets re-explored.
  void mark_spill_plan_unviable(const cucascade::shared_data_repository* repo);

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
