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

/// One segment's contribution to a row group's insert delta.
///
/// Only the byte path depends on the segment type. Transient segments live in
/// memory and must be uncompressed; a prepare task copies
/// [copy_src_offset, +bytes_size) into cuda-pinned staging at slab_offset and
/// the decoder reads that host pointer. Persistent segments are bulk appends
/// that MergeStorage optimistically flushed as checkpoint-shaped blocks; the
/// decoder reads them from the .db file by block_id/block_offset and nothing
/// is staged.
///
/// segment_start is relative to the row group's delta slice. Appends begin at
/// the persistent tail, so a segment never straddles the pin boundary; only
/// the clamp to the n_total snapshot can shorten one.
struct insert_delta_segment {
  duckdb::ColumnSegment* segment{nullptr};  ///< Pin target for the transient copy task
  bool is_transient{false};
  bool is_validity{false};
  bool all_null{false};  ///< CONSTANT validity marked all-NULL by its stats; blockless
  /// CONSTANT data segments: capture-time snapshot of the segment's own stats
  /// (the constant value); see duckdb_segment_descriptor::segment_stats.
  std::shared_ptr<duckdb::BaseStatistics> segment_stats;
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

/// One column's delta segments, split into data and validity.
struct insert_delta_column {
  duckdb::idx_t column_id{0};
  bool is_varchar{false};
  std::vector<insert_delta_segment> data_segments;
  std::vector<insert_delta_segment> validity_segments;
};

/// One row group's slice of the insert delta.
///
/// k_offset counts this row group's rows below n_cache. It is always 0 today:
/// pins start checkpoint-clean and post-checkpoint appends open a fresh row
/// group (RowGroupAppendMode::REQUIRE_NEW), so n_cache lands on a row-group
/// boundary. The k > 0 handling stays in case an append path ever grows an
/// existing row group's tail.
struct insert_delta_row_group {
  duckdb::RowGroup* row_group{nullptr};  ///< tree-owned; stable while pinned (no checkpoints)
  duckdb::idx_t row_group_index{0};
  std::size_t row_group_start{0};  ///< absolute rowid of the first delta row
  std::size_t k_offset{0};
  std::size_t row_count{0};  ///< delta rows covered (tail-clamped to the n_total snapshot)
  bool has_version_state{false};
  std::vector<insert_delta_column> columns;  ///< parallel to the capture's column list
  std::size_t decoded_bytes_budget{0};
  std::vector<std::size_t> varchar_bytes_per_col;  ///< parallel to columns; 0 for non-varchar
  std::size_t transient_staging_bytes{0};          ///< pinned slab bytes this row group needs
};

/// Everything the insert-delta job needs for one pinned entry: the query
/// transaction, the n_total snapshot, and the per-row-group segment plan for
/// physical rows [n_cache, n_total).
///
/// Holds tree-owned RowGroup and ColumnSegment pointers. They stay valid for
/// the prepare window because checkpoints are suppressed while pinned and
/// committed segments are never rewritten in place.
struct insert_delta_plan {
  duckdb::TransactionData transaction{
    duckdb::TransactionData(duckdb::transaction_t{0}, duckdb::transaction_t{0})};
  duckdb::BufferManager* buffer_manager{nullptr};
  std::size_t n_cache{0};
  std::size_t n_total{0};  ///< GetTotalRows() snapshot; all walks clamp to it
  std::vector<insert_delta_row_group> row_groups;

  [[nodiscard]] bool empty() const { return row_groups.empty(); }
  [[nodiscard]] std::size_t delta_rows() const { return n_total - n_cache; }
};

/**
 * Build the insert-delta plan for one pinned duckdb table. Serial: call from
 * the prepare/query thread only, since it uses the ClientContext and takes
 * segment tree locks.
 *
 * Walks the row groups overlapping [n_cache, GetTotalRows()). Throws on
 * states the plan-time guards should have excluded or that break the pin
 * contract: a compressed transient segment, a varchar segment at the
 * overflow-block limit, an unsupported persistent codec, an ARRAY column, or
 * non-StandardColumnData storage.
 *
 * @param storage_column_indices Union of the querying operators' storage
 *        columns; per-operator splits are cut from it later.
 * @param column_types Parallel to storage_column_indices.
 */
insert_delta_plan capture_insert_delta_plan(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  std::size_t n_cache,
  std::span<duckdb::storage_t const> storage_column_indices,
  std::span<sirius::logical_type const> column_types);

/**
 * Count one delta row group's visible rows, setting one bit per visible row
 * at mask_bit_offset + (row - first delta row). Task-safe: GetSelVector
 * serializes on the row group's version lock.
 *
 * The caller pre-zeroes mask_words. Concurrent callers must not share mask
 * words.
 *
 * @return Number of visible delta rows.
 */
std::size_t count_visible_delta_rows(insert_delta_row_group const& rg,
                                     duckdb::TransactionData transaction,
                                     std::span<std::uint32_t> mask_words,
                                     std::size_t mask_bit_offset);

/**
 * Copy one delta row group's transient segment bytes into its staging slab.
 * Task-safe. Each segment's buffer is pinned only for the memcpy; no DuckDB
 * handle outlives the call. Persistent segments are skipped; the decoder
 * reads them from the file.
 */
void copy_delta_row_group(insert_delta_row_group const& rg,
                          duckdb::BufferManager& buffer_manager,
                          std::uint8_t* slab_base);

}  // namespace sirius::op::scan
