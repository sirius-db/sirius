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

#include "helper/logical_type.hpp"

#include <cudf/types.hpp>

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

/// Synthetic rowid columns carry no `storage_idx`.
struct projected_column {
  duckdb::StorageIndex storage_idx;
  bool is_rowid = false;
};

/// Mirrors `duckdb::ColumnSegmentInfo` with the compression string resolved
/// to the enum.
struct duckdb_segment_descriptor {
  /// -1 for blockless layouts (e.g. Constant segments).
  duckdb::block_id_t block_id;
  /// Overflow blocks for variable-width payloads (FSST tables, etc.).
  std::vector<duckdb::block_id_t> additional_blocks;
  duckdb::idx_t block_offset;
  /// Row offset within the row group.
  duckdb::idx_t segment_start;
  duckdb::idx_t segment_count;
  duckdb::CompressionType compression;
  /// Parsed from `ColumnSegmentInfo::segment_stats`. nullopt on validity
  /// and non-VARCHAR segments. Every VARCHAR segment in a viable walk
  /// carries Some; Some(0) is the legal all-empty-row-group case.
  std::optional<std::uint32_t> max_string_length;
  /// Byte size of this segment's main-block payload. Excludes
  /// `additional_blocks`; 0 when `block_id < 0`.
  std::size_t bytes_size = 0;
};

struct duckdb_column_metadata {
  duckdb::idx_t column_id;
  /// Sorted by `segment_start` ascending. Empty when `is_rowid`.
  std::vector<duckdb_segment_descriptor> data_segments;
  /// Sorted by `segment_start` ascending. Empty when there is no validity
  /// column or when `is_rowid`.
  std::vector<duckdb_segment_descriptor> validity_segments;
  bool is_rowid = false;
};

struct duckdb_row_group_metadata {
  duckdb::idx_t row_group_index;
  /// Absolute row index of the row group's first row.
  duckdb::idx_t row_group_start;
  duckdb::idx_t row_count;
  /// Parallel to the walker's `projected_cols` argument.
  std::vector<duckdb_column_metadata> columns;
  std::size_t decoded_bytes_budget = 0;
  /// Parallel to `columns`. For varchar columns: Σ(seg.segment_count ×
  /// *seg.max_string_length) — the upper bound used against the cudf int32
  /// chars threshold. 0 for non-varchar columns.
  std::vector<std::size_t> varchar_bytes_per_col;
};

/// Default-mode cudf strings columns use int32 offsets;
/// `make_offsets_child_column` throws `std::overflow_error` ("Size of output
/// exceeds the column size limit") when total chars per strings column
/// `>= std::numeric_limits<cudf::size_type>::max()` unless
/// `LIBCUDF_LARGE_STRINGS_ENABLED` is set. Sirius does not opt in and its
/// strings-decode kernels (`gpu_decode_strings.cu`) are hard-coded to
/// int32 offsets, so the walker refuses any row group whose per-column
/// varchar upper bound hits this threshold.
constexpr std::size_t kCudfInt32StringsThreshold =
  static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());

/// Exposed for direct unit-testing of the codec-rejection logic without
/// going through DuckDB's codec selection (which is hard to drive into
/// unsupported codecs in a test).
bool is_supported_data_compression(duckdb::CompressionType c);
bool is_supported_validity_compression(duckdb::CompressionType c);

//===----------------------------------------------------------------------===//
// Two-phase metadata walk:
//  - prepare_duckdb_native_walk(): one-time setup.
//  - walk_duckdb_native_row_group_range(): the per-range segment walk; ranges
//    are walked concurrently.
//===----------------------------------------------------------------------===//

/// @brief Read PartitionStatistics, validate projected types and partition
/// `row_start`, and capture the row-group count, block size, and a storage
/// primary index -> projected column index map. No per-segment IO.
/// PartitionStatistics touches ClientContext / LocalStorage, which are not
/// thread-safe, so this runs before the concurrent range walks.
struct duckdb_native_walk_plan {
  duckdb::DataTable* storage     = nullptr;
  duckdb::ClientContext* context = nullptr;

  const std::vector<projected_column>* projected_cols      = nullptr;
  const std::vector<sirius::logical_type>* projected_types = nullptr;
  std::unordered_map<duckdb::idx_t, std::size_t>
    projected_lookup;  /// Map storage primary index -> projected column index (rowid cols
                       /// excluded).

  //===----------RowGroupCollection data----------===//
  std::size_t n_row_groups = 0;
  std::size_t block_size   = 0;

  //===----------PartitionStatistics data: vector across row groups----------===//
  std::vector<duckdb::idx_t> row_group_start;  ///< Absolute first-row index per row group
                                               ///< (PartitionStatistics::row_start).
  std::vector<duckdb::idx_t>
    row_count;  ///< Rows per row group (PartitionStatistics::count); 0 where unavailable.

  //===----------Row-group pruning inputs----------===//
  /// Per-row-group handles. Each range walk reads its own row groups' column
  /// statistics through them while pruning. Ranges are disjoint, so a handle is
  /// only ever touched by one worker.
  std::vector<duckdb::shared_ptr<duckdb::PartitionRowGroup>> partition_row_groups;
  /// Pushed-down filters and their storage-index mapping. Both non-null and
  /// non-empty enables statistics pruning in the range walk.
  const duckdb::TableFilterSet* table_filters           = nullptr;
  const duckdb::vector<duckdb::ColumnIndex>* column_ids = nullptr;

  //===----------Error Handling----------===//
  // `viable == false` (with reason) for:
  //  - unsupported projected types, or
  //  - an invalid partition `row_start`.
  bool viable = false;
  std::string viability_failure_reason;
};
duckdb_native_walk_plan prepare_duckdb_native_walk(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types,
  const duckdb::TableFilterSet* table_filters           = nullptr,
  const duckdb::vector<duckdb::ColumnIndex>* column_ids = nullptr);

/// @brief Walk the projected-column segment metadata for row groups [rg_begin, rg_end)
/// of `plan`, filling per-row-group
///  - `decoded_bytes_budget`,
///  - `varchar_bytes_per_col`, and
///  - segment `bytes_size`.
/// When `plan` carries pushed-down filters, row groups a filter proves can hold
/// no matching rows are dropped from `row_groups`.
struct duckdb_native_row_group_range {
  //===----------ColumnSegmentInfo data----------===//
  std::vector<duckdb_row_group_metadata>
    row_groups;  ///< rg metadata ranging over [rg_begin, rg_end)

  //===----------Filter-statistics pruning counters----------===//
  std::size_t pruned_row_groups    = 0;  ///< Row groups dropped by a pushed-down filter.
  std::size_t pruned_decoded_bytes = 0;  ///< Sum of their decoded-byte budgets.

  //===----------Error Handling----------===//
  //`viable == false` (with reason) on the first
  //  - unsupported segment compression, or
  //  - absent/over-threshold varchar stat.
  // `row_groups` is then partial and must not be consumed.
  bool viable = true;
  std::string viability_failure_reason;
};
duckdb_native_row_group_range walk_duckdb_native_row_group_range(
  const duckdb_native_walk_plan& plan, std::size_t rg_begin, std::size_t rg_end);

}  // namespace sirius::op::scan
