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
#include <string>
#include <utility>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Surrogate-key group-by (late string materialization) — shared plan/runtime state.
 *
 * When every STRING group key of a HASH_GROUP_BY is a pure pass-through of one side of a single
 * upstream INNER hash join ("the deferral join"), the planner replaces those keys in the join's
 * output with a compact numeric surrogate: the first deferred slot of each side carries a BIGINT
 * row id (the join gather-map value plus a per-source base offset) and the remaining deferred
 * slots carry constant TINYINT dummies. Column COUNT and POSITIONS are unchanged everywhere
 * between the deferral join and the group-by, so intermediate operators only see a type change
 * at those slots. The deferral join retains read-only handles on its deferred-side input batches;
 * MERGE_GROUP_BY materializes the strings from them after aggregation and restores the original
 * output schema, so everything downstream of the merge is untouched.
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

/// Retention store shared between the deferral join and the group-by merge.
///
/// Address-space invariants (relied on by the merge's finalize gather):
///  - each SOURCE BATCH occupies exactly one contiguous range [base, base + rows), assigned once
///    per batch id (`reserve` is idempotent per id, so task retries and BUILD_PROBE probe tasks
///    sharing one build table reuse the same range and emit identical rowids for the same row);
///  - ranges are contiguous and entries are kept in base order, so concatenating the committed
///    sources in entry order reproduces the absolute rowid address space exactly.
///
/// Retention protocol (OOM-downgrade friendly): `reserve` takes NO accessor — a task that fails
/// after reserving leaves the source batch downgradable and only names an address range (reused
/// on retry via the id dedupe). The pinning `read_only_data_batch` accessor is attached by
/// `commit` only after the task's output was produced successfully. `snapshot` therefore
/// requires every reserved range to be committed: an uncommitted range can only belong to a
/// batch whose task never succeeded, in which case the query has already failed before any
/// merge finalize could run.
class surrogate_deferral_store {
 public:
  struct source_view {
    int64_t base;
    cudf::size_type rows;
    ::cucascade::read_only_data_batch batch;
  };

  /// Reserve (or look up) the address range for `batch_id` on one side; returns its base.
  /// Throws when the total address space would exceed int32 row addressing (checked before any
  /// state is mutated), or when `rows` disagrees with an existing reservation for the same id.
  int64_t reserve(bool is_left, uint64_t batch_id, cudf::size_type rows)
  {
    std::lock_guard<std::mutex> lg(mutex_);
    auto& side = is_left ? left_ : right_;
    for (auto const& e : side.entries) {
      if (e.batch_id == batch_id) {
        if (e.rows != rows) {
          throw std::runtime_error("surrogate_deferral_store::reserve: batch " +
                                   std::to_string(batch_id) +
                                   " re-reserved with a different row count (" +
                                   std::to_string(e.rows) + " vs " + std::to_string(rows) + ")");
        }
        return e.base;
      }
    }
    if (side.next_base > static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) -
                           static_cast<int64_t>(rows)) {
      // Finalization gathers with an INT32 cudf gather map; refuse address spaces that overflow
      // it instead of computing garbage. (The planner declines on estimated cardinality; this is
      // the hard runtime backstop.)
      throw std::runtime_error(
        "groupby_surrogate_keys: deferred string source exceeds int32 row addressing; disable "
        "the groupby_surrogate_keys setting for this query");
    }
    int64_t const base = side.next_base;
    side.next_base += rows;
    side.entries.push_back(entry{batch_id, base, rows, std::nullopt});
    return base;
  }

  /// Attach the retaining read-only accessor for a previously reserved batch. Idempotent; call
  /// only after the task's output was produced successfully so failed tasks never pin sources.
  void commit(bool is_left, uint64_t batch_id, ::cucascade::read_only_data_batch batch)
  {
    std::lock_guard<std::mutex> lg(mutex_);
    auto& side = is_left ? left_ : right_;
    for (auto& e : side.entries) {
      if (e.batch_id == batch_id) {
        if (!e.batch) { e.batch = std::move(batch); }
        return;
      }
    }
    throw std::runtime_error("surrogate_deferral_store::commit: batch " + std::to_string(batch_id) +
                             " was never reserved");
  }

  /// Snapshot one side's committed sources in base order. Throws if a reserved range was never
  /// committed (see the retention protocol above — unreachable in a successfully-running query).
  std::vector<source_view> snapshot(bool is_left) const
  {
    std::lock_guard<std::mutex> lg(mutex_);
    auto const& side = is_left ? left_ : right_;
    std::vector<source_view> out;
    out.reserve(side.entries.size());
    for (auto const& e : side.entries) {
      if (!e.batch) {
        throw std::runtime_error("surrogate_deferral_store::snapshot: reserved source batch " +
                                 std::to_string(e.batch_id) +
                                 " was never committed (its producing task cannot have succeeded)");
      }
      out.push_back(source_view{e.base, e.rows, *e.batch});
    }
    return out;
  }

  /// Drop every retained accessor (called once all merge finalizes are done, from the merge
  /// operator's finalize hook). Returns {source count, retained bytes} for observability.
  std::pair<std::size_t, std::size_t> release()
  {
    std::lock_guard<std::mutex> lg(mutex_);
    std::size_t count = 0;
    std::size_t bytes = 0;
    for (auto* side : {&left_, &right_}) {
      for (auto& e : side->entries) {
        if (e.batch) {
          ++count;
          if (auto const* data = e.batch->get_data(); data != nullptr) {
            bytes += data->get_size_in_bytes();
          }
          e.batch.reset();
        }
      }
    }
    return {count, bytes};
  }

 private:
  struct entry {
    uint64_t batch_id;
    int64_t base;
    cudf::size_type rows;
    std::optional<::cucascade::read_only_data_batch> batch;  ///< set by commit only
  };
  struct side_state {
    std::vector<entry> entries;  ///< base-ordered by construction
    int64_t next_base = 0;
  };

  mutable std::mutex mutex_;
  side_state left_;
  side_state right_;
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
    bool from_left     = true;                         ///< which side of the store to gather from
    int rowid_key_slot = -1;                           ///< output key slot holding the BIGINT rowid
    std::vector<int> restore_key_slots;                ///< key slots to restore (incl. rowid slot)
    std::vector<cudf::size_type> source_input_cols;    ///< parallel: column in the retained batches
    std::vector<sirius::logical_type> restored_types;  ///< parallel: original logical types
  };

  std::shared_ptr<surrogate_deferral_store> store;
  std::vector<restore_group> groups;
  /// Non-deferred, non-dummy key slots: partition hashing subset and uniqueness-check columns.
  std::vector<int> real_key_slots;
  /// The group-by's original output schema (keys with STRING types restored), which the merge
  /// redeclares and reconstructs.
  duckdb::vector<sirius::logical_type> original_output_types;
  /// Take the no-re-group fast path when the exact distinct check proves tuple distinctness.
  bool unique_fastpath = true;
};

}  // namespace op
}  // namespace sirius
