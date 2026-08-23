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
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <ctrack.hpp>

#include <algorithm>
#include <iterator>
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
  if (sources.empty()) { return false; }
  // Every source has to qualify: the decode consumes one flattened chunk-data
  // span covering all of them, so a single file that cannot be read this way
  // takes the whole split to the general route.
  return std::all_of(sources.begin(), sources.end(), [](parquet_source const& src) {
    return src.datasource && src.metadata && !src.row_group_indices.empty() &&
           src.datasource->prefers_bulk_io();
  });
}

namespace {

/// One source's column chunks, resident on the device.
struct fetched_chunks {
  /// One buffer per column chunk.  They must outlive the decode; their
  /// stream-ordered deallocation is ordered behind it, so no synchronize is
  /// needed at the hand-off.
  std::vector<rmm::device_buffer> buffers;
  /// One span per chunk, in the order the reader enumerated the ranges -- which
  /// is the order @c materialize_all_columns expects its chunk data in.
  std::vector<cudf::device_span<std::uint8_t const>> spans;
};

/// Read every column chunk of @p src into device memory in ONE vectored
/// request, so a split costs one round trip per file rather than one per chunk.
/// That is the whole point of the bulk route against an object store, where
/// every extra request is another round trip.
fetched_chunks fetch_chunks(parquet_source const& src,
                            cudf::io::parquet_reader_options const& options,
                            std::span<cudf::io::text::byte_range_info const> ranges,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  // Derive the ranges when the caller did not already have them.  A caller that
  // fadvised this split does, and passing them avoids re-walking the footer.
  std::vector<cudf::io::text::byte_range_info> owned;
  if (ranges.empty()) {
    owned  = column_chunk_ranges(*src.metadata, options, src.row_group_indices);
    ranges = owned;
  }

  fetched_chunks out;
  out.buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    out.buffers.emplace_back(static_cast<std::size_t>(range.size()), stream, mr);
  }

  std::vector<io::slice> reads;
  reads.reserve(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    reads.emplace_back(static_cast<std::size_t>(ranges[i].offset()),
                       static_cast<std::size_t>(ranges[i].size()),
                       static_cast<std::uint8_t*>(out.buffers[i].data()));
  }

  {
    CTRACK_NAME("materialize_parquet::bulk::read");
    std::ignore = src.datasource->device_read_ranges_async(reads, stream).get();
  }

  out.spans.reserve(out.buffers.size());
  for (auto const& buf : out.buffers) {
    out.spans.emplace_back(static_cast<std::uint8_t const*>(buf.data()), buf.size());
  }
  return out;
}

/// Ranges for source @p i, or an empty span when the caller supplied none.
std::span<cudf::io::text::byte_range_info const> ranges_for(
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges, std::size_t i)
{
  return i < ranges.size() ? std::span<cudf::io::text::byte_range_info const>(ranges[i])
                           : std::span<cudf::io::text::byte_range_info const>{};
}

/// Bulk route: read every source's column chunks, then decode straight out of
/// those device buffers.
std::unique_ptr<cudf::table> materialize_bulk(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CTRACK_NAME("materialize_parquet::bulk");

  if (sources.size() == 1) {
    // Single file: the single-source reader, whose chunk data is flat.
    auto const& src   = sources.front();
    auto const chunks = fetch_chunks(src, options, ranges_for(ranges, 0), stream, mr);

    CTRACK_NAME("materialize_parquet::bulk::decode");
    hybrid_scan_reader reader(*src.metadata, options);
    auto result =
      reader.materialize_all_columns(cudf::host_span<cudf::size_type const>(
                                       src.row_group_indices.data(), src.row_group_indices.size()),
                                     cudf::host_span<cudf::device_span<std::uint8_t const> const>(
                                       chunks.spans.data(), chunks.spans.size()),
                                     options,
                                     stream,
                                     mr);
    return std::move(result.tbl);
  }

  // Several files in one split: the multi-file reader takes the row groups per
  // source and wants its chunk data flattened in source order, with each
  // source's chunks in the same row-group / column order the single-file reader
  // produces.  Fetching per source and appending in that order is exactly that
  // layout -- and it keeps one vectored request per file, which is the unit the
  // IO backend can coalesce (ranges from different files share no offsets).
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::device_span<std::uint8_t const>> spans;
  metadatas.reserve(sources.size());
  rg_per_src.reserve(sources.size());

  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto chunks = fetch_chunks(sources[i], options, ranges_for(ranges, i), stream, mr);
    spans.insert(spans.end(), chunks.spans.begin(), chunks.spans.end());
    // Moving a device_buffer keeps its device pointer, so the spans stay valid;
    // the buffers just have to outlive the decode.
    std::move(chunks.buffers.begin(), chunks.buffers.end(), std::back_inserter(buffers));
    metadatas.push_back(*sources[i].metadata);
    rg_per_src.push_back(sources[i].row_group_indices);
  }

  CTRACK_NAME("materialize_parquet::bulk::decode");
  cudf::io::parquet::experimental::hybrid_scan_multifile reader(
    cudf::host_span<cudf::io::parquet::FileMetaData const>(metadatas.data(), metadatas.size()),
    options);
  auto result = reader.materialize_all_columns(
    cudf::host_span<std::vector<cudf::size_type> const>(rg_per_src.data(), rg_per_src.size()),
    cudf::host_span<cudf::device_span<std::uint8_t const> const>(spans.data(), spans.size()),
    options,
    stream,
    mr);
  return std::move(result.tbl);
}

/// General route: let cudf read as it decodes, over whatever sources there are.
std::unique_ptr<cudf::table> materialize_general(std::span<parquet_source const> sources,
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
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (sources.empty()) { return nullptr; }
  if (prefers_bulk_materialize(sources, options)) {
    return materialize_bulk(sources, options, ranges, stream, mr);
  }
  return materialize_general(sources, options, stream, mr);
}

}  // namespace sirius::op::scan
