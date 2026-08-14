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

#include "op/scan/parquet_materialize.hpp"

#include "io/io_context.hpp"
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <ctrack.hpp>

#include <utility>

namespace sirius::op::scan {

using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& row_group_indices)
{
  if (row_group_indices.empty()) { return {}; }
  hybrid_scan_reader reader(metadata, options);
  return reader.all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size()),
    options);
}

bool prefers_bulk_materialize(std::span<parquet_source const> sources,
                              cudf::io::parquet_reader_options const& options) noexcept
{
  // A filter is disqualifying, not merely unsupported: materialize_all_columns
  // ignores one rather than rejecting it, so taking this route with filtered
  // options would hand back every row and claim it was filtered.  Enforced here
  // rather than left to each caller to remember.
  if (options.get_filter().has_value()) { return false; }
  // One source only: the bulk route reads through a single datasource's vectored
  // request, and there is no batched form spanning several files.
  if (sources.size() != 1) { return false; }
  auto const& src = sources.front();
  if (!src.datasource || !src.metadata || src.row_group_indices.empty()) { return false; }
  return src.datasource->prefers_bulk_io();
}

namespace {

/// Bulk route: one vectored device read for every column chunk of the split,
/// then decode straight out of those device buffers.
std::unique_ptr<cudf::table> materialize_bulk(
  parquet_source const& src,
  cudf::io::parquet_reader_options const& options,
  std::span<cudf::io::text::byte_range_info const> ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CTRACK_NAME("materialize_parquet::bulk");

  // Derive the ranges when the caller did not already have them.  A caller that
  // fadvised this split does, and passing them avoids re-walking the footer.
  std::vector<cudf::io::text::byte_range_info> owned;
  if (ranges.empty()) {
    owned  = column_chunk_ranges(*src.metadata, options, src.row_group_indices);
    ranges = owned;
  }

  // One device buffer per column chunk, allocated on `stream`.  They stay alive
  // for the whole function; their stream-ordered deallocation at scope exit is
  // ordered behind the decode below, so no synchronize is needed here.
  std::vector<rmm::device_buffer> buffers;
  buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()), stream, mr);
  }

  std::vector<io::io_device_range> reads;
  reads.reserve(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    reads.push_back(io::io_device_range{static_cast<std::size_t>(ranges[i].offset()),
                                        static_cast<std::size_t>(ranges[i].size()),
                                        static_cast<std::uint8_t*>(buffers[i].data())});
  }

  {
    CTRACK_NAME("materialize_parquet::bulk::read");
    std::ignore = src.datasource->device_read_ranges_async(reads, stream).get();
  }

  std::vector<cudf::device_span<std::uint8_t const>> spans;
  spans.reserve(buffers.size());
  for (auto const& buf : buffers) {
    spans.emplace_back(static_cast<std::uint8_t const*>(buf.data()), buf.size());
  }

  CTRACK_NAME("materialize_parquet::bulk::decode");
  hybrid_scan_reader reader(*src.metadata, options);
  auto result = reader.materialize_all_columns(
    cudf::host_span<cudf::size_type const>(src.row_group_indices.data(),
                                           src.row_group_indices.size()),
    cudf::host_span<cudf::device_span<std::uint8_t const> const>(spans.data(), spans.size()),
    options,
    stream,
    mr);
  return std::move(result.tbl);
}

/// General route: let cudf read as it decodes, over whatever sources there are.
std::unique_ptr<cudf::table> materialize_general(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CTRACK_NAME("materialize_parquet::general");

  std::vector<std::unique_ptr<cudf::io::datasource>> cudf_sources;
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  cudf_sources.reserve(sources.size());
  metadatas.reserve(sources.size());
  rg_per_src.reserve(sources.size());

  for (auto const& src : sources) {
    cudf_sources.push_back(cudf::io::datasource::create(src.datasource.get()));
    metadatas.push_back(*src.metadata);
    rg_per_src.push_back(src.row_group_indices);
  }

  auto opts = options;
  opts.set_row_groups(std::move(rg_per_src));

  auto [table, _] =
    cudf::io::read_parquet(std::move(cudf_sources), std::move(metadatas), opts, stream, mr);
  return std::move(table);
}

}  // namespace

std::unique_ptr<cudf::table> materialize_parquet(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<cudf::io::text::byte_range_info const> ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (sources.empty()) { return nullptr; }
  if (prefers_bulk_materialize(sources, options)) {
    return materialize_bulk(sources.front(), options, ranges, stream, mr);
  }
  return materialize_general(sources, options, stream, mr);
}

}  // namespace sirius::op::scan
