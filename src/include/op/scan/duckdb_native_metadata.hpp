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

#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
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
  /// Per-segment max-string-length parsed from DuckDB's
  /// `ColumnSegmentInfo::segment_stats` text blob. nullopt for validity
  /// segments and non-VARCHAR data segments. After a viable walk, every
  /// VARCHAR data segment is guaranteed Some(...); Some(0) is legal data
  /// (every value in this segment is ""). Drives the GPU kernel's
  /// chars-region sizing and short/long-string routing, and the
  /// partitioner's per-column varchar-byte cap.
  std::optional<std::uint32_t> max_string_length;
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
};

/// When `viable` is false the walker bailed at the first unsupported
/// segment or type; `row_groups` is partial and must not be consumed.
struct duckdb_native_metadata {
  std::vector<duckdb_row_group_metadata> row_groups;
  bool viable = false;
  std::string viability_failure_reason;
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
duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types);

/// Exposed for direct unit-testing of the codec-rejection logic without
/// going through DuckDB's codec selection (which is hard to drive into
/// unsupported codecs in a test).
bool is_supported_data_compression(duckdb::CompressionType c);
bool is_supported_validity_compression(duckdb::CompressionType c);

}  // namespace sirius::op::scan
