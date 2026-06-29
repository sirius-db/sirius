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

  // ── Per-column entries (used by Phase 4 explore / manual overrides) ─────

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

}  // namespace sirius::compression
