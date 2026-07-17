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

#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/common/typedefs.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/transaction/transaction_data.hpp>
#include <helper/logical_type.hpp>
#include <op/scan/duckdb_native_metadata.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace duckdb {
class BufferManager;
class ClientContext;
class ColumnSegment;
class DataTable;
class RowGroup;
}  // namespace duckdb

namespace sirius::op::scan {

/**
 * @brief One segment's contribution to the insert delta of one row group.
 *
 * The delta is positional — physical rows [n_cache, n_total) — and a segment
 * joins it regardless of its type; only the BYTE PATH branches:
 *  - TRANSIENT (small committed appends): in-memory only. A prepare-time task
 *    pins the segment's buffer and memcpys [copy_src_offset, +bytes_size)
 *    into cuda-pinned staging (slab_offset within the row group's slab); the
 *    decode descriptor gets a host_ptr. Always UNCOMPRESSED (asserted).
 *  - PERSISTENT (bulk appends, optimistically flushed by MergeStorage):
 *    checkpoint-shaped blocks in the .db file, compressed like a checkpoint.
 *    The descriptor keeps block_id/block_offset and rides the decoder's
 *    existing file-read lane; no prepare-time byte work.
 *
 * `segment_start` is rebased to the row group's DELTA slice (absolute rowid
 * `rg.row_group_start + segment_start`); segments never straddle the pin
 * boundary (appends start exactly at the persistent tail), so only the tail
 * clamp against the n_total snapshot ever shortens one.
 */
struct insert_delta_segment {
  duckdb::ColumnSegment* segment{nullptr};  ///< Pin target for the transient copy task
  bool is_transient{false};
  bool is_validity{false};
  duckdb::block_id_t block_id{-1};                    ///< persistent only
  std::vector<duckdb::block_id_t> additional_blocks;  ///< persistent only (overflow blocks)
  duckdb::idx_t block_offset{0};                      ///< persistent only
  duckdb::idx_t segment_start{0};                     ///< rebased to the delta slice
  duckdb::idx_t segment_count{0};                     ///< covered delta rows
  duckdb::CompressionType compression{duckdb::CompressionType::COMPRESSION_AUTO};
  std::optional<std::uint32_t> max_string_length;  ///< varchar data segments only
  std::size_t bytes_size{0};  ///< staging extent (transient) / payload upper bound (persistent)
  std::size_t copy_src_offset{0};  ///< transient: byte offset into the pinned buffer
  std::size_t slab_offset{0};      ///< transient: byte offset within the row group's staging slab
};

/// Per-column delta segments, mirroring duckdb_column_metadata's data/validity split.
struct insert_delta_column {
  duckdb::idx_t column_id{0};
  bool is_varchar{false};
  std::vector<insert_delta_segment> data_segments;
  std::vector<insert_delta_segment> validity_segments;
};

/**
 * @brief One row group's slice of the insert delta.
 *
 * `k_offset` is the number of this row group's rows below n_cache. With
 * current DuckDB append semantics it is always 0 — post-checkpoint appends
 * open a FRESH row group (RowGroupAppendMode::REQUIRE_NEW), and pins start
 * checkpoint-clean, so n_cache lands on a row-group boundary. The k > 0
 * handling (boundary-vector intersection in the visibility walk, rebased
 * segment math) is kept as defensive generality against tail-growing append
 * variants.
 */
struct insert_delta_row_group {
  duckdb::RowGroup* row_group{nullptr};  ///< tree-owned; stable while pinned (no checkpoints)
  duckdb::idx_t row_group_index{0};
  std::size_t row_group_start{0};  ///< absolute rowid of the FIRST DELTA row (rg start + k_offset)
  std::size_t k_offset{0};
  std::size_t row_count{0};  ///< delta rows covered (tail-clamped to the n_total snapshot)
  bool has_version_state{false};
  std::vector<insert_delta_column> columns;  ///< parallel to the capture's column list
  std::size_t decoded_bytes_budget{0};
  std::vector<std::size_t> varchar_bytes_per_col;  ///< parallel to columns; 0 for non-varchar
  std::size_t transient_staging_bytes{0};          ///< pinned slab bytes this row group needs
};

/**
 * @brief Serial capture of everything the insert-delta job needs for one
 *        pinned entry: the query transaction, the n_total snapshot, and the
 *        per-row-group segment plan over [n_cache, n_total).
 *
 * Holds tree-owned pointers (RowGroup / ColumnSegment) — stable for the
 * query's prepare window because checkpoints are suppressed while pinned and
 * committed segments are never rewritten in place.
 */
struct insert_delta_plan {
  duckdb::TransactionData transaction{
    duckdb::TransactionData(duckdb::transaction_t{0}, duckdb::transaction_t{0})};
  duckdb::BufferManager* buffer_manager{nullptr};
  std::size_t n_cache{0};
  std::size_t n_total{0};  ///< GetTotalRows() snapshot; all walks clamp to it
  std::vector<insert_delta_row_group> row_groups;
  /// Capture-time partition stats, carried onto delta splits so CONSTANT
  /// segment decode never calls GetPartitionStats off the query thread.
  std::shared_ptr<duckdb::vector<duckdb::PartitionStatistics>> partition_stats;

  [[nodiscard]] bool empty() const { return row_groups.empty(); }
  [[nodiscard]] std::size_t delta_rows() const { return n_total - n_cache; }
};

/**
 * @brief Build the insert-delta plan for one pinned duckdb table. SERIAL —
 *        prepare/query thread only (ClientContext discipline; takes segment
 *        tree locks while walking).
 *
 * Walks the row groups overlapping [n_cache, GetTotalRows()), branching per
 * SEGMENT on segment_type (see insert_delta_segment). Throws on states the
 * plan-time guards should have excluded or that violate the pin contract:
 * a compressed TRANSIENT segment (a checkpoint ran while pinned), a varchar
 * segment at/over the overflow-block limit, an unsupported persistent codec,
 * an ARRAY column (declined at plan time in v1), or non-StandardColumnData
 * storage.
 *
 * @param storage_column_indices The union of the querying operators' storage
 *        columns (superset staging; per-operator splits are cut later).
 * @param column_types Parallel to @p storage_column_indices.
 */
insert_delta_plan capture_insert_delta_plan(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  std::size_t n_cache,
  std::span<duckdb::storage_t const> storage_column_indices,
  std::span<sirius::logical_type const> column_types);

/**
 * @brief Visibility counting pass for one delta row group — task-safe
 *        (GetSelVector serializes on the row group's own version lock).
 *
 * Sets one bit per VISIBLE delta row into @p mask_words at
 * @p mask_bit_offset + (row - first delta row); the caller pre-zeroes the
 * words (single writer per bundle mask, so no alignment constraints beyond
 * not sharing words across concurrent bundles).
 *
 * @return the number of visible delta rows.
 */
std::size_t count_visible_delta_rows(insert_delta_row_group const& rg,
                                     duckdb::TransactionData transaction,
                                     std::span<std::uint32_t> mask_words,
                                     std::size_t mask_bit_offset);

/**
 * @brief Copy one delta row group's TRANSIENT segment bytes into its staging
 *        slab — task-safe. Pins each segment's buffer just long enough to
 *        memcpy [copy_src_offset, +bytes_size) to slab_base + slab_offset;
 *        no DuckDB handle outlives the call. Persistent segments need no
 *        byte work (the decoder's file lane reads them at materialize).
 */
void copy_delta_row_group(insert_delta_row_group const& rg,
                          duckdb::BufferManager& buffer_manager,
                          std::uint8_t* slab_base);

}  // namespace sirius::op::scan
