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

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <op/scan/scan_plan.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

/// Per-file metadata scan output. Provides everything the caller needs to
/// produce row_group_slice instances and accumulate per-row-group bytes for
/// budget-based bundling.
struct file_metadata_scan_result {
  /// Parsed parquet metadata. Held by shared_ptr so it can be passed verbatim
  /// into one or more row_group_slice instances that reference this file.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  /// Datasource for this parquet file. The caller can hold this for the
  /// lifetime of any task that reads from the file, or drop it if the
  /// downstream scan_data does not need it.
  std::shared_ptr<cudf::io::datasource> datasource;
  /// Row group indices selected after metadata-stats-based filter pruning,
  /// in file order. Empty if every row group was pruned (or the file had
  /// none to begin with).
  std::vector<cudf::size_type> selected_row_group_indices;
  /// Parquet column-chunk indices selected by @ref scan_plan projection
  /// (resolved by name against this file's schema). Empty when the scan is
  /// non-projected — the caller treats every chunk in the row group as
  /// selected.
  std::vector<std::size_t> selected_chunk_indices;
  /// Subset of @ref selected_chunk_indices marked as pure-filter (read for
  /// filter evaluation, not part of the output). The caller should exclude
  /// these from the uncompressed-byte accounting that drives budget-based
  /// bundling.
  std::unordered_set<std::size_t> pure_filter_chunk_indices;
};

/**
 * @brief Scan one parquet file's footer + metadata and resolve projection /
 *        row-group filtering. Does not partition the row groups into slices —
 *        the caller bundles them under whatever budget policy fits its scan
 *        type.
 *
 * Filter-pushdown gating is derived from @p reader_options: row-group stats
 * pruning runs iff @c reader_options.get_filter().has_value(). The caller is
 * expected to install (or omit) the AST filter via @c set_filter before
 * calling this helper.
 *
 * @param file_path             Parquet file to scan.
 * @param plan                  Canonical scan plan; drives projection /
 *                              pure-filter accounting.
 * @param reader_options        Reader options carrying any AST filter
 *                              already installed via @c set_filter.
 * @param data_column_names     Projected data-column names in @c scan_plan
 *                              D-order. Pass once per call site rather than
 *                              rebuilding per file (the caller typically
 *                              hoists @c plan.data_column_names() out of its
 *                              file loop).
 * @param pure_filter_positions Pure-filter D-positions from @c
 *                              plan.pure_filter_batch_positions(). Hoisted
 *                              for the same reason.
 * @param stream                Stream used by row-group stats pruning.
 *
 * @throws std::runtime_error if a projected column from @p plan is not
 *                            found in the file.
 *
 * @note A second per-file metadata scan still lives in
 *       @c parquet_scan_task_global_state::initialize_from_files (legacy
 *       task-based path); consolidating that caller is gated on legacy
 *       parquet_scan_task removal and is intentionally out of scope for the
 *       initial extraction.
 */
file_metadata_scan_result scan_parquet_file_metadata(
  std::string const& file_path,
  scan_plan const& plan,
  cudf::io::parquet_reader_options const& reader_options,
  std::vector<std::string> const& data_column_names,
  std::unordered_set<std::size_t> const& pure_filter_positions,
  rmm::cuda_stream_view stream);

/// Per-row-group byte accounting for budget-based partition bundling.
struct row_group_bytes {
  /// Sum of total_uncompressed_size over selected (non-pure-filter) chunks.
  std::size_t uncompressed_bytes;
  /// Sum of total_compressed_size over all selected chunks (pure-filter
  /// chunks count toward compressed-byte reservation but not toward the
  /// uncompressed-byte budget).
  std::size_t compressed_bytes;
};

/**
 * @brief Compute the byte contribution of a single row group, honouring
 *        @p plan projection and pure-filter exclusion.
 *
 * For non-projected scans (@c plan.is_projected() == false) every column
 * chunk contributes; @p selected_chunk_indices and @p pure_filter_chunk_indices
 * are ignored.
 *
 * For projected scans only the chunks named in @p selected_chunk_indices
 * contribute; chunks in @p pure_filter_chunk_indices are excluded from the
 * uncompressed total.
 */
row_group_bytes compute_row_group_bytes(
  cudf::io::parquet::RowGroup const& row_group,
  scan_plan const& plan,
  std::vector<std::size_t> const& selected_chunk_indices,
  std::unordered_set<std::size_t> const& pure_filter_chunk_indices);

}  // namespace sirius::op::scan
