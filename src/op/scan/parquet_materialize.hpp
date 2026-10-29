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

// Turning a set of parquet row-group slices into a cudf table, by whichever
// route the backend serving them prefers.
//
// Kept in a neutral header (like row_group_metadata.hpp) so the scan operator
// and the io benchmarks can share it without either depending on the other's
// headers -- parquet_gpu_ingestible.hpp pulls in DuckDB, which a benchmark has
// no business including.

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cucascade/cudf/datasource.hpp>

#include <memory>
#include <span>
#include <vector>

namespace sirius::op::scan {

/// One file's contribution to a split: where to read it from, its parsed
/// footer, and which row groups are wanted.  Mirrors the fields of
/// @c row_group_slice that materialization actually needs, without dragging in
/// the estimate/accounting ones.
struct parquet_source {
  std::shared_ptr<cucascade::io::datasource> datasource;
  std::shared_ptr<cudf::io::parquet::FileMetaData const> metadata;
  std::vector<cudf::size_type> row_group_indices;
};

/// Column-chunk byte ranges a read fetches for @p row_group_indices, honoring
/// @p options' column projection.  Empty when there are no row groups.
[[nodiscard]] std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& row_group_indices);

/// @ref column_chunk_ranges over many row-group subsets, keeping the reader of
/// the last file and reader options it was asked about.
///
/// Building a reader copies the whole footer, so a file split into many
/// batches costs one copy instead of one per batch. The kept reader holds that
/// copy until the next file, or until @ref clear.
class column_chunk_range_cache {
 public:
  /// Same result as @ref column_chunk_ranges for @p metadata, @p options and
  /// @p row_group_indices.
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> ranges(
    std::shared_ptr<cudf::io::parquet::FileMetaData const> const& metadata,
    std::shared_ptr<cudf::io::parquet_reader_options const> const& options,
    std::vector<cudf::size_type> const& row_group_indices);

  /// Release the kept reader and its footer copy.
  void clear() noexcept;

 private:
  std::shared_ptr<cudf::io::parquet::FileMetaData const> _metadata;
  std::shared_ptr<cudf::io::parquet_reader_options const> _options;
  std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> _reader;
};

/// Materialize @p sources into one table.
///
/// Takes one of two routes, picked from what the backend says it wants:
///
///   bulk    - when every source's backend reports @c prefers_bulk_io().  Each
///             source's column chunks are read in a single vectored device
///             request straight into their own device buffers, and the table is
///             decoded from those buffers by the hybrid scan reader —
///             @c hybrid_scan_reader for one source, @c hybrid_scan_multifile
///             for several.  One round trip per file for the whole split
///             instead of one per chunk, which is what makes this worth doing
///             against an object store.
///
///   general - otherwise.  cudf::io::read_parquet over the datasources and
///             their pre-parsed footers, which reads as it decodes.
///
/// Both routes honor @p options' row filter: @c materialize_all_columns applies
/// it at row level exactly as @c read_parquet does, so the route never changes
/// which rows come back.  With the reader options Sirius uses, both routes
/// also refuse sources whose schemas differ.
///
/// @param ranges  column-chunk ranges, one vector per entry of @p sources and in
///                the same order.  Only read on the bulk route; any source whose
///                entry is missing or empty has its ranges derived from the
///                metadata.  Pass an empty span to derive them all.
[[nodiscard]] std::unique_ptr<cudf::table> materialize_parquet(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/// Whether @c materialize_parquet would take the bulk route for @p sources and
/// @p options.  Exposed so a caller can decide *before* paying to build the
/// ranges.
[[nodiscard]] bool prefers_bulk_materialize(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options) noexcept;

/// Throw unless every entry of @p sources has the same footer schema as the
/// first, compared as cudf::io::read_parquet compares several sources.  The
/// multi-file hybrid scan reader skips that check, so the bulk route calls
/// this first.  Each source must have non-null metadata.
void require_same_parquet_schema(std::span<parquet_source const> sources);

}  // namespace sirius::op::scan
