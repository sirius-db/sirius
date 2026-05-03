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

/// One projected column. Either a real storage column, identified by
/// `storage_idx`, or a synthetic rowid column when `is_rowid` is true. Rowid
/// columns have no segments — the walker emits per-row-group rowid_range
/// sentinels instead, and the downstream scan operator materialises them as
/// BIGINT via `thrust::sequence`.
struct projected_column {
  duckdb::StorageIndex storage_idx;
  bool is_rowid = false;
};

/// Description of a single on-disk segment, sufficient to drive H2D + GPU
/// decode without re-walking metadata. Mirrors the relevant fields of
/// `duckdb::ColumnSegmentInfo` (public struct, table_storage_info.hpp:19),
/// resolved into engine types.
struct duckdb_segment_descriptor {
  /// Block carrying the segment payload. May be -1 for blockless layouts
  /// (e.g. Constant segments stored entirely in stats).
  duckdb::block_id_t block_id;
  /// Additional blocks for variable-width payloads (FSST tables, etc.).
  std::vector<duckdb::block_id_t> additional_blocks;
  duckdb::idx_t block_offset;
  /// First row of this segment relative to the row group.
  duckdb::idx_t segment_start;
  duckdb::idx_t segment_count;
  duckdb::CompressionType compression;
  /// VARCHAR-only stats hint used for pre-decode buffer sizing. 0 means
  /// the segment did not advertise a max length. The walker refuses
  /// dictionary-family VARCHAR codecs (Dictionary / FSST / DICT_FSST) when
  /// this would be 0; for Uncompressed VARCHAR a 0 reaches the consumer,
  /// which must size dynamically (and the row group's
  /// `decoded_bytes_budget_is_lower_bound` will be set).
  std::uint32_t max_string_length = 0;
};

/// Per row-group view of one projected column.
struct duckdb_column_metadata {
  duckdb::idx_t column_id;
  /// Empty when `is_rowid`. Ordered by `segment_start` ascending.
  std::vector<duckdb_segment_descriptor> data_segments;
  /// Empty when there is no validity column (a few non-standard columns) or
  /// when `is_rowid`. Ordered by `segment_start` ascending.
  std::vector<duckdb_segment_descriptor> validity_segments;
  /// True if this projected entry is a synthetic rowid column. The data /
  /// validity descriptors are unused; the consumer materialises a BIGINT
  /// range `[row_group_start, row_group_start + row_count)` on the GPU.
  bool is_rowid = false;
};

struct duckdb_row_group_metadata {
  duckdb::idx_t row_group_index;
  /// Absolute row index of the row group's first row within the table.
  /// Drives rowid synthesis and external row-id reporting.
  duckdb::idx_t row_group_start;
  duckdb::idx_t row_count;
  /// One entry per projected column, parallel to the walker's
  /// `projected_cols` argument.
  std::vector<duckdb_column_metadata> columns;
  /// Sum of decoded byte budget across projected columns. Drives split-batch
  /// sizing in the upstream split provider.
  std::size_t decoded_bytes_budget = 0;
  /// True when at least one VARCHAR column in this row group fell back to
  /// the @c VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES per-row upper bound because
  /// the row group did not advertise a max-string-length stat (only reachable
  /// today for Uncompressed VARCHAR — dictionary-family codecs are refused
  /// up-front when the stat is missing). Consumers using the budget for
  /// split-batch sizing should treat it as a soft lower bound when this is
  /// true, since the fallback can either over- or under-shoot the actual
  /// decoded bytes by a wide margin.
  bool decoded_bytes_budget_is_lower_bound = false;
};

/// Per-row upper bound used by `decoded_bytes_budget` when a VARCHAR row
/// group has no advertised max-string-length stat. Picked as a defensible
/// "wide-but-not-absurd" string size; matches the threshold above which
/// DuckDB's storage path almost always picks a dictionary-family codec
/// (which has the stat). When this is used, the row group's
/// `decoded_bytes_budget_is_lower_bound` is set true.
inline constexpr std::uint32_t VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES = 256;

/// Output of `walk_duckdb_native_metadata`. When `viable` is false the
/// walker bails on the first unsupported segment / type and `row_groups`
/// holds whatever it had completed up to that point — callers must not
/// consume the partial result.
struct partitioned_duckdb_native_metadata {
  std::vector<duckdb_row_group_metadata> row_groups;
  bool viable = false;
  /// Empty when `viable`. Diagnostic string identifying which segment or
  /// type caused the bail-out.
  std::string viability_failure_reason;
};

/// Walk the segment trees of `projected_cols` across every row group of
/// `storage` via DuckDB's public `DataTable::GetColumnSegmentInfo` and
/// `DataTable::GetPartitionStats` APIs.
///
/// The walker is metadata-only: it does not pin blocks, read bytes, or
/// host-decode any payload. The downstream scan operator (PR #9) is
/// responsible for I/O via the sirius_io substrate, including any host-side
/// decode of validity codecs that can't be GPU-dispatched directly (e.g.
/// Roaring).
///
/// The walker enforces the following viability constraints; on first
/// violation it returns with `viable = false` and a populated
/// `viability_failure_reason`:
///   - data segments: compression must be GPU-dispatchable
///     (Uncompressed / Constant / RLE / Dictionary / BitPacking / FSST /
///     DICT_FSST / ALP / ALPRD); VARCHAR codecs additionally require
///     a non-zero max-string-length stat
///   - validity segments: compression must be Uncompressed / Empty Validity
///     / Constant (effectively all-valid) / Roaring (host-decoded later)
///   - logical types: only types Sirius has a GPU decode path for are
///     accepted. Refused: HUGEINT / UHUGEINT (no 128-bit integer decode),
///     DECIMAL with precision > 18 (DECIMAL128 storage), STRUCT, LIST
///     (nested types, deferred), and the placeholder INVALID / SQLNULL
///     sentinels. Every other `sirius::type_id` is accepted.
///
/// Operator-level escape gates (`dynamic_filters`, sample options, virtual
/// columns, type pushdown rewrites) live on the `LogicalGet` and are the
/// caller's responsibility to check before invoking the walker.
partitioned_duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types);

}  // namespace sirius::op::scan
