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

#include <cucascade/data/data_batch.hpp>

#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Surrogate-key group-by (late string materialization) — shared plan/runtime state.
 *
 * When every STRING group key of a HASH_GROUP_BY is a pure pass-through of one side of a single
 * upstream INNER hash join ("the deferral join"), the planner replaces those keys in the join's
 * output with a compact numeric surrogate: the first deferred slot of each side carries a BIGINT
 * row id (the join gather-map value plus a per-registration base offset) and the remaining
 * deferred slots carry constant TINYINT dummies. Column COUNT and POSITIONS are unchanged
 * everywhere between the deferral join and the group-by, so intermediate operators only see a
 * type change at those slots. The deferral join retains read-only handles on its deferred-side
 * input batches; MERGE_GROUP_BY materializes the strings from them after aggregation and
 * restores the original output schema, so everything downstream of the merge is untouched.
 *
 * Correctness: the surrogate REFINES the original key tuple (each source row has exactly one
 * tuple), so partial sums compose. Grouping by surrogate can only differ from grouping by the
 * tuple when two distinct source rows carry an identical FULL tuple. The merge therefore takes
 * the "fast path" (no re-group) only when an EXACT distinct_count over the non-deferred key
 * columns equals the merged row count — which proves all tuples are distinct — and otherwise
 * gathers the strings and re-groups by the full original tuple. The upstream PARTITION hashes
 * only the non-deferred key slots, so rows with equal real keys (a superset of equal-tuple rows)
 * always meet in the same merge task, making the per-task check and re-group globally sound.
 */
struct surrogate_deferral_store {
  /// One registered source: a read-only handle on a deferral-join input batch whose rows are
  /// addressed by rowids in [base, base + rows). Bases are absolute and monotonically assigned,
  /// so a stale registration (e.g. from a retried task whose output was discarded) wastes memory
  /// but never corrupts addressing.
  struct source {
    int64_t base;
    cudf::size_type rows;
    ::cucascade::read_only_data_batch batch;
  };

  std::mutex mutex;
  std::vector<source> left_sources;
  std::vector<source> right_sources;
  int64_t left_next_base  = 0;
  int64_t right_next_base = 0;

  /// Register a source batch for one side; returns the assigned base offset.
  int64_t register_source(bool is_left, ::cucascade::read_only_data_batch batch,
                          cudf::size_type rows)
  {
    std::lock_guard<std::mutex> lg(mutex);
    auto& next    = is_left ? left_next_base : right_next_base;
    auto& sources = is_left ? left_sources : right_sources;
    int64_t base  = next;
    next += rows;
    // Finalization gathers with an INT32 cudf gather map; refuse address spaces that overflow it
    // instead of computing garbage. (The planner gates on estimated cardinality; this is the
    // hard runtime backstop.)
    if (next > std::numeric_limits<cudf::size_type>::max()) {
      throw std::runtime_error(
        "groupby_surrogate_keys: deferred string source exceeds int32 row addressing; disable "
        "the groupby_surrogate_keys setting for this query");
    }
    sources.push_back(source{base, rows, std::move(batch)});
    return base;
  }

  /// Snapshot one side's sources ordered by base (they are appended in base order already).
  std::vector<source> const& sources_for(bool is_left) const
  {
    return is_left ? left_sources : right_sources;
  }
};

/// Plan-time instructions for the deferral join: which output slots of each side to synthesize
/// instead of gathering. Positions index into that side's output column list
/// (lhs_output_columns / rhs_output_columns), not the join's full output.
struct surrogate_join_emit {
  struct side {
    /// Output position (within the side's output columns) that carries the BIGINT rowid.
    cudf::size_type rowid_out_pos = -1;
    /// Output positions (within the side's output columns) emitted as constant TINYINT dummies.
    std::vector<cudf::size_type> dummy_out_pos;
  };
  std::optional<side> left;
  std::optional<side> right;
  std::shared_ptr<surrogate_deferral_store> store;
};

/// Plan-time instructions for HASH_GROUP_BY / MERGE_GROUP_BY finalization.
struct surrogate_groupby_spec {
  /// One group of deferred key slots sharing a single rowid (one per deferral-join side).
  struct restore_group {
    bool from_left = true;             ///< which side of the store to gather from
    int rowid_key_slot = -1;           ///< output key slot holding the BIGINT rowid
    std::vector<int> restore_key_slots;             ///< key slots to restore (incl. rowid slot)
    std::vector<cudf::size_type> source_input_cols; ///< parallel: column in the retained batches
    std::vector<sirius::logical_type> restored_types;  ///< parallel: original logical types
  };

  std::shared_ptr<surrogate_deferral_store> store;
  std::vector<restore_group> groups;
  /// Non-deferred, non-dummy key slots: partition hashing subset and uniqueness-check columns.
  std::vector<int> real_key_slots;
  /// The group-by's original output schema (keys with STRING types restored), which the merge
  /// re-declares and reconstructs.
  duckdb::vector<sirius::logical_type> original_output_types;
  /// Take the no-re-group fast path when the exact distinct check proves tuple distinctness.
  bool unique_fastpath = true;
};

}  // namespace op
}  // namespace sirius
