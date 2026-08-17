/*
 * Copyright 2025, Sirius Contributors.
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

#include "duckdb/common/vector.hpp"
#include "helper/logical_type.hpp"

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace sirius::op {

class surrogate_deferral_store;

/**
 * @file groupby_surrogate_deferral.hpp
 * @brief Surrogate-key group-by (late string materialization) -- the plan-time instruction
 * values shared between the deferral join and the group-by merge.
 *
 * When every STRING group key of a HASH_GROUP_BY is a pure pass-through of one side of a single
 * upstream INNER hash join ("the deferral join"), the planner replaces those keys in the join's
 * output with a compact numeric surrogate: the first deferred slot of each side carries a BIGINT
 * row id (the join gather-map value plus a per-source base offset) and the remaining deferred
 * slots carry constant TINYINT dummies. Column COUNT and POSITIONS are unchanged everywhere
 * between the deferral join and the group-by, so intermediate operators only see a type change
 * at those slots. The deferral join retains read-only handles on its deferred-side input batches
 * (see `op/groupby_surrogate_store.hpp`); MERGE_GROUP_BY materializes the strings from them
 * after aggregation and restores the original output schema, so everything downstream of the
 * merge is untouched.
 *
 * Correctness: the surrogate REFINES the original key tuple (each source row has exactly one
 * tuple), so partial sums compose. Grouping by surrogate can only differ from grouping by the
 * tuple when two distinct source rows carry an identical FULL tuple. The merge therefore takes
 * the "fast path" (no re-group) only when an EXACT distinct_count over the non-deferred key
 * columns equals the merged row count -- which proves all tuples are distinct -- and otherwise
 * gathers the strings and re-groups by the full original tuple. The upstream PARTITION hashes
 * only the non-deferred key slots, so rows with equal real keys (a superset of equal-tuple rows)
 * always meet in the same merge task, making the per-task check and re-group globally sound.
 *
 * Index vocabulary used throughout the surrogate module:
 *  - **slot** -- index into the group-by's output key list (`rowid_key_slot`,
 *    `real_key_slots`);
 *  - **pos** -- position within one join side's output-column list (`rowid_out_pos`,
 *    `dummy_out_pos`);
 *  - **col** -- column index into a child/source schema (`source_col`, `key_col_indices`).
 */

/// One side of the deferral join. Every per-side API in the surrogate module takes this enum;
/// no call site passes a bare bool for a side.
enum class join_side : std::uint8_t { left = 0, right = 1 };

/// The side's lowercase name ("left" / "right") for log and error messages.
[[nodiscard]] constexpr std::string_view to_string(join_side side) noexcept
{
  return side == join_side::left ? "left" : "right";
}

/// @brief Immutable plan-time instruction for the deferral join: which output slots of each
/// side to synthesize instead of gathering.
///
/// Built and installed by the surrogate planner pass
/// (`sirius::planner::apply_groupby_surrogate_keys`) via
/// `sirius_physical_hash_join::install_surrogate_emit`. It co-owns the
/// `surrogate_deferral_store` with the group-by side's `surrogate_restore_plan` and dies with
/// the physical plan.
class surrogate_emit_plan final {
 public:
  /// One join side's synthesized output slots. Positions index into that side's output-column
  /// list (lhs_output_columns / rhs_output_columns), not the join's full output.
  class side_plan final {
   public:
    /// @throws sirius::internal_exception when `rowid_out_pos` is negative or any dummy
    /// position is negative, duplicated, or equal to `rowid_out_pos`.
    side_plan(cudf::size_type rowid_out_pos, std::vector<cudf::size_type> dummy_out_pos);

    /// Output position that carries the BIGINT rowid.
    [[nodiscard]] cudf::size_type rowid_out_pos() const noexcept { return _rowid_out_pos; }
    /// Output positions emitted as constant TINYINT dummies.
    [[nodiscard]] std::vector<cudf::size_type> const& dummy_out_pos() const noexcept
    {
      return _dummy_out_pos;
    }

   private:
    cudf::size_type _rowid_out_pos;
    std::vector<cudf::size_type> _dummy_out_pos;
  };

  /// @throws sirius::internal_exception when `store` is null or neither side is present.
  surrogate_emit_plan(std::optional<side_plan> left,
                      std::optional<side_plan> right,
                      std::shared_ptr<surrogate_deferral_store> store);

  [[nodiscard]] std::optional<side_plan> const& side(join_side side) const noexcept
  {
    return side == join_side::left ? _left : _right;
  }
  /// The retention store shared with the group-by merge; non-null by construction. Mutable at
  /// runtime (reserve/commit) while the plan itself stays immutable.
  [[nodiscard]] surrogate_deferral_store& store() const noexcept { return *_store; }

 private:
  std::optional<side_plan> _left;
  std::optional<side_plan> _right;
  std::shared_ptr<surrogate_deferral_store> _store;
};

/// @brief Immutable plan-time instruction for HASH_GROUP_BY / MERGE_GROUP_BY finalization:
/// which key slots to restore from which retained sources, and how.
///
/// Installed on the HASH_GROUP_BY by the surrogate planner pass
/// (`sirius_physical_grouped_aggregate::install_surrogate_restore`);
/// `sirius_physical_grouped_aggregate_merge` acquires it through its clone-from-aggregate
/// constructor. Co-owns the `surrogate_deferral_store` with the join's `surrogate_emit_plan`.
class surrogate_restore_plan final {
 public:
  /// One deferred key slot: where it sits in the group-by output, which column of the retained
  /// source batches holds its strings, and the logical type to restore.
  struct restored_key {
    int key_slot;                        ///< group-by output key slot to restore
    cudf::size_type source_col;          ///< column in the retained source batches
    sirius::logical_type original_type;  ///< original (string) logical type
  };

  /// One group of deferred key slots sharing a single rowid (one per deferral-join side). Keys
  /// are kept in ascending key-slot order (frozen restore order).
  class restore_group final {
   public:
    /// @throws sirius::internal_exception when `keys` is empty or `rowid_key_slot` is not the
    /// `key_slot` of any element of `keys`.
    restore_group(join_side side, int rowid_key_slot, std::vector<restored_key> keys);

    /// Which side of the store to gather from.
    [[nodiscard]] join_side side() const noexcept { return _side; }
    /// Output key slot holding the BIGINT rowid.
    [[nodiscard]] int rowid_key_slot() const noexcept { return _rowid_key_slot; }
    /// The key slots to restore (including the rowid slot).
    [[nodiscard]] std::vector<restored_key> const& keys() const noexcept { return _keys; }

   private:
    join_side _side;
    int _rowid_key_slot;
    std::vector<restored_key> _keys;
  };

  /// @throws sirius::internal_exception when `store` is null, `groups` is empty,
  /// `real_key_slots` is empty (the ">= 1 real key" planner gate, restated as a type
  /// invariant), or the restored and real key slots are not disjoint or fall outside
  /// `original_output_types`.
  surrogate_restore_plan(std::shared_ptr<surrogate_deferral_store> store,
                         std::vector<restore_group> groups,
                         std::vector<int> real_key_slots,
                         duckdb::vector<sirius::logical_type> original_output_types,
                         bool allow_unique_fastpath);

  /// The retention store shared with the deferral join; non-null by construction.
  [[nodiscard]] std::shared_ptr<surrogate_deferral_store> const& store() const noexcept
  {
    return _store;
  }
  [[nodiscard]] std::vector<restore_group> const& groups() const noexcept { return _groups; }
  /// Non-deferred, non-dummy key slots: partition hashing subset and uniqueness-check columns.
  [[nodiscard]] std::vector<int> const& real_key_slots() const noexcept { return _real_key_slots; }
  /// The group-by's original output schema (keys with STRING types restored), which the merge
  /// redeclares and reconstructs.
  [[nodiscard]] duckdb::vector<sirius::logical_type> const& original_output_types() const noexcept
  {
    return _original_output_types;
  }
  /// Permission (knob snapshot) to take the no-re-group fast path when the exact distinct
  /// check proves tuple distinctness.
  [[nodiscard]] bool allow_unique_fastpath() const noexcept { return _allow_unique_fastpath; }

 private:
  std::shared_ptr<surrogate_deferral_store> _store;
  std::vector<restore_group> _groups;
  std::vector<int> _real_key_slots;
  duckdb::vector<sirius::logical_type> _original_output_types;
  bool _allow_unique_fastpath;
};

/// @brief Rewrite `physical_types` so every deferred key slot of `plan` declares its restored
/// native cuDF carrier.
///
/// The HASH_GROUP_BY's physical sidecar carries the rowid/dummy carrier types at the deferred
/// key slots, but MERGE_GROUP_BY finalizes back to the original schema -- its sidecar must
/// declare the restored native carriers. Called by the physical plan generator's
/// `wrap_hash_group_by` on the merge's sidecar. Slots beyond `physical_types` are ignored.
void restore_deferred_carriers(surrogate_restore_plan const& plan,
                               std::vector<cudf::data_type>& physical_types);

}  // namespace sirius::op
