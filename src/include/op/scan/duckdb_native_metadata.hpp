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
#include <string>
#include <vector>

namespace sirius::op::scan {

/// One projected column. Synthetic rowid columns carry no `storage_idx` —
/// the walker emits a per-row-group sentinel and the scan operator
/// synthesises the BIGINT range.
struct projected_column {
  duckdb::StorageIndex storage_idx;
  bool is_rowid = false;
};

/// Mirrors the relevant fields of `duckdb::ColumnSegmentInfo`
/// (table_storage_info.hpp:19), resolved into engine types.
struct duckdb_segment_descriptor {
  /// May be -1 for blockless layouts (e.g. Constant segments in stats).
  duckdb::block_id_t block_id;
  /// Additional blocks for variable-width payloads (FSST tables, etc.).
  std::vector<duckdb::block_id_t> additional_blocks;
  duckdb::idx_t block_offset;
  /// First row of this segment relative to the row group.
  duckdb::idx_t segment_start;
  duckdb::idx_t segment_count;
  duckdb::CompressionType compression;
  /// 0 = no stat advertised. Dictionary-family VARCHAR codecs are refused
  /// in that case (need it to size pre-decode buffers); Uncompressed
  /// VARCHAR accepts 0 and triggers the row group's
  /// `decoded_bytes_budget_is_lower_bound`.
  std::uint32_t max_string_length = 0;
};

struct duckdb_column_metadata {
  duckdb::idx_t column_id;
  /// Ordered by `segment_start` ascending. Empty when `is_rowid`.
  std::vector<duckdb_segment_descriptor> data_segments;
  /// Ordered by `segment_start` ascending. Empty when there is no validity
  /// column or when `is_rowid`.
  std::vector<duckdb_segment_descriptor> validity_segments;
  bool is_rowid = false;
};

struct duckdb_row_group_metadata {
  duckdb::idx_t row_group_index;
  /// Absolute row index of the row group's first row within the table;
  /// drives rowid synthesis.
  duckdb::idx_t row_group_start;
  duckdb::idx_t row_count;
  /// Parallel to the walker's `projected_cols` argument.
  std::vector<duckdb_column_metadata> columns;
  std::size_t decoded_bytes_budget = 0;
  /// Set when the budget used `VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES` for
  /// at least one column. Consumers must treat the budget as a soft lower
  /// bound — the fallback may over- or under-shoot the actual decoded
  /// bytes.
  bool decoded_bytes_budget_is_lower_bound = false;
};

/// Per-row VARCHAR fallback when no max-string-length stat is advertised.
/// Sized at the threshold above which DuckDB's storage almost always
/// picks a dictionary-family codec (which carries the stat); using this
/// flips the row group's `decoded_bytes_budget_is_lower_bound`.
inline constexpr std::uint32_t VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES = 256;

/// When `viable` is false the walker bailed on the first unsupported
/// segment / type — `row_groups` may be partially populated and must
/// not be consumed.
struct duckdb_native_metadata {
  std::vector<duckdb_row_group_metadata> row_groups;
  bool viable = false;
  std::string viability_failure_reason;
};

/// Metadata-only walker over `storage`'s segment trees, via DuckDB's public
/// `DataTable::GetColumnSegmentInfo` + `GetPartitionStats`. Reads no bytes
/// and pins no blocks; I/O + Roaring host-decode are PR #9's job.
///
/// On the first viability violation returns `viable = false` and a
/// populated `viability_failure_reason`. The accept/refuse lists for data
/// compression, validity compression, and logical type live in
/// `is_supported_data_compression` / `is_supported_validity_compression` /
/// `is_supported_logical_type` in the .cpp.
///
/// Operator-level escape gates (`dynamic_filters`, sample options, virtual
/// columns, type pushdown rewrites) live on the `LogicalGet` and are the
/// caller's responsibility to check before invoking the walker.
duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types);

}  // namespace sirius::op::scan
