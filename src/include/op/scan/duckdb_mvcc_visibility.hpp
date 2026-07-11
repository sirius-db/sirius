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

#include <duckdb/common/typedefs.hpp>
#include <duckdb/transaction/transaction_data.hpp>
#include <scan_manager/duckdb_mvcc_metadata.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace duckdb {
class ClientContext;
class DataTable;
class RowGroup;
}  // namespace duckdb

namespace sirius::op::scan {

/**
 * @brief One row group's contribution to one cached chunk's keep-mask.
 *
 * The pointed-to RowGroup is owned by the table's segment tree and only
 * removed by checkpoint compaction (suppressed while a duckdb table is
 * pinned), so the raw pointer is stable for the query's prepare/serve window.
 */
struct mvcc_row_group_slice {
  duckdb::RowGroup* row_group{nullptr};
  duckdb::idx_t row_group_start{0};  ///< absolute rowid of the RG's first row
  std::size_t chunk_offset{0};       ///< where its rows land in the chunk's mask
  std::size_t row_count{0};          ///< rows covered by the pin (last RG clamped)
  bool has_version_state{false};     ///< capture's non-loading dirty probe (skip signal)
};

/**
 * @brief Per-query visibility work plan over a pinned entry's cached prefix.
 *
 * Slot i of @ref mvcc_row_groups holds the slices for cached chunk i,
 * paralleling base_row_count_per_chunk[i]. Holds copies
 * and tree-owned pointers only — no pinned_entry references escape into it.
 */
struct mvcc_visibility_plan {
  explicit mvcc_visibility_plan(duckdb::TransactionData transaction) : transaction(transaction) {}

  duckdb::TransactionData transaction;  ///< query txn on the pinned table's own database
  std::vector<std::vector<mvcc_row_group_slice>> mvcc_row_groups;
  std::vector<bool> chunk_has_version_state;  ///< OR of each chunk's slice probes

  /// True when any chunk has version state — the mask job's do-anything-at-all
  /// signal. False = every mask slot stays null; no allocation, no tasks.
  [[nodiscard]] bool any_version_state() const;
};

/**
 * @brief Build the visibility plan for one pinned duckdb table. SERIAL —
 *        prepare/query thread only (the ClientContext discipline of
 *        prepare_duckdb_native_walk).
 *
 * Captures the query transaction with BOTH ids off the table's own database
 * (so the query's own uncommitted deletes are honored) and walks the row
 * groups covering the pinned prefix [0, metadata.n_cache()), clamping the
 * last row group to pin coverage (its live count grows under appends). The
 * same pass records each row group's non-loading dirty probe via
 * RowGroup::GetPartitionStats (count_type APPROXIMATE <=> version state):
 * safe because a delete visible to any active snapshot always has version
 * state present — delete markers persist until checkpoint compaction
 * (suppressed while pinned), and checkpointed tombstones show as unloaded
 * deletes. Conservative in two ways that cost a walk but never a wrong mask:
 * append-only version info probes dirty, and a RowVersionManager created by
 * this session's own writes stays attached forever — only freshly-loaded
 * (ATTACH-then-pin) tables take the zero-cost clean path.
 *
 * Throws std::runtime_error on invariants validated at pin time
 * (validate_duckdb_pin_chunk): start_time < v_base (a re-pin raced
 * plan->prepare), non-contiguous or short row-group coverage, a chunk
 * boundary off a row-group start, or a slice offset not 32-row aligned (the
 * lock-free bit fill requires task ranges to never share a mask word).
 */
mvcc_visibility_plan capture_mvcc_visibility_plan(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  scan_manager::duckdb_mvcc_metadata const& metadata);

/**
 * @brief Fill the keep-mask bits for @p slices (all belonging to ONE chunk).
 *
 * Worker-safe with plain stores: concurrent calls over disjoint slice lists
 * write disjoint words (slice offsets are 32-row aligned, asserted at
 * capture). Writes exactly bits [chunk_offset, chunk_offset + row_count) per
 * slice, 1 = keep; a slice without version state bulk-sets ones with no MVCC
 * calls or locks, otherwise each 2048-row vector goes through
 * RowGroup::GetSelVector. Whole-word writes never collide with a later
 * writer (a partial vector ends its slice; a partial slice ends the table),
 * and padding bits past the covered range are don't-care.
 *
 * @return true when any row in the covered range was dropped.
 */
bool fill_keep_mask_for_row_groups(std::span<mvcc_row_group_slice const> slices,
                                   duckdb::TransactionData transaction,
                                   std::span<std::uint32_t> chunk_mask_words);

/**
 * @brief True when any row group intersecting rowids [0, row_prefix) carries
 *        an in-memory update chain (UpdateSegment) on any of
 *        @p storage_column_indices.
 *
 * Update chains version VALUES in place, invisibly to GetSelVector, so the
 * cached base (and the on-disk image) would serve stale data. SERIAL — may
 * lazily load column data.
 */
bool any_update_chains(duckdb::DataTable& storage,
                       std::span<duckdb::storage_t const> storage_column_indices,
                       std::size_t row_prefix);

/// Verdict of check_native_read_mvcc_state: whether the MVCC-blind disk-native
/// read of a table is exact for a given snapshot, and if not, why.
enum class native_read_mvcc_state : std::uint8_t {
  exact,               ///< raw bytes are visibility-exact for this snapshot
  has_update_chains,   ///< a scanned column carries in-memory UpdateSegment values
  has_invisible_rows,  ///< tombstones or in-flight inserts at this snapshot
};

/**
 * @brief One row-group walk deciding whether the disk-native read is exact
 *        for @p transaction, checking update chains and row visibility
 *        together. Early-out on the first finding.
 *
 * Per row group, tiered cheap→expensive: (1) UpdateSegment chains on the
 * scanned columns — update chains version VALUES in place, invisibly to any
 * row mask; (2) the non-loading version-state probe — clean row groups are
 * provably exact and skip ahead; (3) probe-dirty row groups only pay DuckDB's
 * own RowGroup::GetVisibleRowCount(transaction) == physical count. Tier 3's
 * precision matters because the probe is conservative the wrong way for
 * unpinned tables — a RowVersionManager created by this session's own writes
 * stays attached after every row became globally visible, and refusing on the
 * probe alone would push every same-session-written table off the GPU
 * permanently. SERIAL — prepare/query thread only.
 *
 * Does NOT see transaction-local appends (they live in LocalStorage, outside
 * the table's segment tree) — callers guard those separately.
 */
native_read_mvcc_state check_native_read_mvcc_state(
  duckdb::DataTable& storage,
  std::span<duckdb::storage_t const> scanned_columns,
  duckdb::TransactionData transaction);

}  // namespace sirius::op::scan
