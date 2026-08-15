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

#include <cstddef>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::compression {

/**
 * @brief Thread-safe registry for Simpatico column-compression plan DSL strings.
 *
 * Table-level keys are caller-supplied identity strings. The registry is one
 * PROCESS-GLOBAL singleton (plan_register::global()), so callers must key by a
 * UNIQUE table identity, never a bare table name: the pin path keys by the
 * pinned entry's resolved cache identity (duckdb catalog.schema.table, or the
 * canonicalized parquet file set) via cache_entry_info::compression_plan_key()
 * — register E5, where two ATTACHed databases with same-named tables collided
 * on the bare name and one pinned with the other database's plan DSL.
 * Per-column keys are "<table_key>::<column_name>" strings.
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
   * @brief Store the complete multi-column Simpatico plan DSL for @p table_key.
   *
   * The DSL must be the "---"-separated string that compress_with_plan expects
   * (one block per column, in schema order).  Overwrites any previous entry.
   */
  void set_table_plan(const std::string& table_key, std::string full_plan_dsl);

  /// Remove the whole-table plan entry for @p table_key.
  void clear_table_plan(const std::string& table_key);

  /**
   * @brief Return the whole-table plan DSL for @p table_key, or nullopt if none.
   *
   * Does not fall back to per-column entries or the default plan — this is
   * the direct input-table lookup used by PinTableFunction.
   */
  [[nodiscard]] std::optional<std::string> resolve_table_plan(const std::string& table_key) const;

  /**
   * @brief Atomic lookup-or-populate for the whole-table plan (register E5).
   *
   * Under ONE exclusive critical section: return the existing non-empty plan
   * for @p table_key, or invoke @p loader and — when it yields a non-empty
   * DSL — store and return it (a nullopt/empty result stores nothing, so a
   * later call retries the load). This replaces the racy
   * resolve-miss-load-set-resolve sequence: two concurrent callers can never
   * both miss and overwrite each other, and every caller returns the value
   * that actually ended up registered.
   *
   * @p loader runs while the registry lock is held (plan files are small local
   * reads; exactly-once loading is worth the brief exclusion). It must not
   * call back into this plan_register.
   */
  std::optional<std::string> get_or_load_table_plan(
    const std::string& table_key, const std::function<std::optional<std::string>()>& loader);

  // ── Per-column entries (used by explore / manual overrides) ─────

  /// Register (or overwrite) a single-column plan for a specific (table, column) pair.
  void set_plan(const std::string& table_name,
                const std::string& column_name,
                std::string plan_dsl);

  /// Remove the per-(table, column) plan override if present.
  void clear_plan(const std::string& table_name, const std::string& column_name);

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /// Remove all entries (table-level and per-column).
  void clear_all();

 private:
  mutable std::shared_mutex _mutex;
  std::unordered_map<std::string, std::string> _table_plans;  // table_name → full multi-col DSL
  std::unordered_map<std::string, std::string> _col_plans;    // "table::column" → single-col DSL
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
