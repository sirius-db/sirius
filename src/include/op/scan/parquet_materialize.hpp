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

#include "io/sirius_datasource.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <span>
#include <vector>

namespace sirius::op::scan {

/// One file's contribution to a split: where to read it from, its parsed
/// footer, and which row groups are wanted.  Mirrors the fields of
/// @c row_group_slice that materialization actually needs, without dragging in
/// the estimate/accounting ones.
struct parquet_source {
  std::shared_ptr<io::sirius_datasource> datasource;
  std::shared_ptr<cudf::io::parquet::FileMetaData const> metadata;
  std::vector<cudf::size_type> row_group_indices;
};

/// Column-chunk byte ranges a read fetches for @p row_group_indices, honoring
/// @p options' column projection.  Empty when there are no row groups.
[[nodiscard]] std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& row_group_indices);

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
/// The bulk route cannot apply a row filter.  cudf's hybrid scan evaluates a
/// predicate through a different sequence entirely -- build a row mask,
/// materialize the filter columns to narrow it, then materialize the payload
/// columns under it -- and @c materialize_all_columns is the shortcut that skips
/// all of that.  Handing it filtered options would silently return unfiltered
/// rows, so a filter on @p options forces the general route.
///
/// @param ranges  column-chunk ranges, one vector per entry of @p sources and in
///                the same order.  Only read on the bulk route; any source whose
///                entry is missing or empty has its ranges derived from the
///                metadata.  Pass an empty span to derive them all.
[[nodiscard]] std::unique_ptr<cudf::table> materialize_parquet(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/// Whether @c materialize_parquet would take the bulk route for @p sources and
/// @p options.  Exposed so a caller can decide *before* paying to build the
/// ranges.  Answers false for filtered options -- see @c materialize_parquet.
[[nodiscard]] bool prefers_bulk_materialize(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options) noexcept;

}  // namespace sirius::op::scan
