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
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {

struct scan_plan;  // op/scan/scan_plan.hpp — walker takes it by const ref

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
  /// Parallel to the plan's `data_columns`.
  std::vector<duckdb_column_metadata> columns;
  std::size_t decoded_bytes_budget = 0;
  /// Parallel to `columns`. For varchar columns: Σ(seg.segment_count ×
  /// *seg.max_string_length) — the upper bound used against the cudf int32
  /// chars threshold. 0 for non-varchar columns. Populated by the walker so
  /// downstream partitioning never re-walks segments.
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

/// When `viable` is false the walker bailed at the first unsupported
/// segment or type; `row_groups` is partial and must not be consumed.
struct duckdb_native_metadata {
  std::vector<duckdb_row_group_metadata> row_groups;
  bool viable = false;
  std::string viability_failure_reason;
  /// Row-group filter-statistics pruning counters (0 when no filters / pruning disabled).
  std::size_t pruned_row_groups = 0;  ///< Number of row groups pruned.
  /// Sum of decoded-byte budgets of pruned row groups.
  std::size_t pruned_decoded_bytes = 0;
};

/// Metadata-only walk of `storage` via `DataTable::GetColumnSegmentInfo`
/// and `GetPartitionStats`. Pins no blocks and reads no bytes.
///
/// Returns `viable = false` with a populated `viability_failure_reason`
/// on the first unsupported segment or type.
///
/// Caller is responsible for the operator-level escape gates
/// (`dynamic_filters`, sample options, virtual columns, type pushdown)
/// on the originating `LogicalGet`.
///
/// @note When both `table_filters` and `column_ids` are non-null, row group pruning is applied
duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  scan_plan const& plan,
  const duckdb::TableFilterSet* table_filters           = nullptr,
  const duckdb::vector<duckdb::ColumnIndex>* column_ids = nullptr);

/// Exposed for direct unit-testing of the codec-rejection logic without
/// going through DuckDB's codec selection (which is hard to drive into
/// unsupported codecs in a test).
bool is_supported_data_compression(duckdb::CompressionType c);
bool is_supported_validity_compression(duckdb::CompressionType c);

}  // namespace sirius::op::scan
