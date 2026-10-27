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

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
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

std::vector<cudf::io::text::byte_range_info> column_chunk_range_cache::ranges(
  std::shared_ptr<cudf::io::parquet::FileMetaData const> const& metadata,
  std::shared_ptr<cudf::io::parquet_reader_options const> const& options,
  std::vector<cudf::size_type> const& row_group_indices)
{
  if (row_group_indices.empty()) { return {}; }
  if (!_reader || _metadata != metadata || _options != options) {
    _reader.reset();
    _reader   = std::make_unique<hybrid_scan_reader>(*metadata, *options);
    _metadata = metadata;
    _options  = options;
  }
  return _reader->all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size()),
    *options);
}

void column_chunk_range_cache::clear() noexcept
{
  _reader.reset();
  _metadata.reset();
  _options.reset();
}

bool prefers_bulk_materialize(std::span<parquet_source const> sources,
                              cudf::io::parquet_reader_options const& options) noexcept
{
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

char const* physical_type_name(cudf::io::parquet::Type type)
{
  switch (type) {
    case cudf::io::parquet::Type::BOOLEAN: return "BOOLEAN";
    case cudf::io::parquet::Type::INT32: return "INT32";
    case cudf::io::parquet::Type::INT64: return "INT64";
    case cudf::io::parquet::Type::INT96: return "INT96";
    case cudf::io::parquet::Type::FLOAT: return "FLOAT";
    case cudf::io::parquet::Type::DOUBLE: return "DOUBLE";
    case cudf::io::parquet::Type::BYTE_ARRAY: return "BYTE_ARRAY";
    case cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY: return "FIXED_LEN_BYTE_ARRAY";
    default: return "group";
  }
}

std::string describe(cudf::io::parquet::SchemaElement const& element)
{
  return "'" + element.name + "' " + physical_type_name(element.type);
}

std::string path_of(parquet_source const& src)
{
  return src.datasource ? src.datasource->get_io_object().object_path() : std::string{"<unknown>"};
}

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
                            ::cuda::stream_ref stream,
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

/// Throw @ref prefetched_mismatch unless @p spans are exactly the chunks @p expected describes:
/// one span per range, in order, each of the range's size.
void require_spans_match(std::span<cudf::io::text::byte_range_info const> expected,
                         std::span<cudf::device_span<std::uint8_t const> const> spans,
                         std::string const& what)
{
  if (expected.size() != spans.size()) {
    throw prefetched_mismatch(
      "[parquet_materialize] device spans do not match the reader's "
      "column chunks for " +
      what + ": expected " + std::to_string(expected.size()) + " chunks, got " +
      std::to_string(spans.size()));
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (static_cast<std::size_t>(expected[i].size()) != spans[i].size()) {
      throw prefetched_mismatch(
        "[parquet_materialize] device spans do not match the reader's "
        "column chunks for " +
        what + ": chunk " + std::to_string(i) + " is " + std::to_string(spans[i].size()) +
        " bytes, expected " + std::to_string(expected[i].size()));
    }
  }
}

/// Throw @ref prefetched_mismatch unless @p claimed -- the file ranges the spans were read from,
/// when the caller knows them -- start where @p expected says.  Sizes alone let a permutation of
/// equal-size chunks through.  An empty @p claimed is not checked.
void require_offsets_match(std::span<cudf::io::text::byte_range_info const> expected,
                           std::span<cudf::io::text::byte_range_info const> claimed,
                           std::string const& what)
{
  if (claimed.empty()) { return; }
  if (claimed.size() != expected.size()) {
    throw prefetched_mismatch(
      "[parquet_materialize] device spans do not match the reader's "
      "column chunks for " +
      what + ": read from " + std::to_string(claimed.size()) + " ranges, expected " +
      std::to_string(expected.size()));
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (claimed[i].offset() != expected[i].offset() || claimed[i].size() != expected[i].size()) {
      throw prefetched_mismatch(
        "[parquet_materialize] device spans do not match the reader's "
        "column chunks for " +
        what + ": chunk " + std::to_string(i) + " was read from offset " +
        std::to_string(claimed[i].offset()) + ", expected " + std::to_string(expected[i].offset()));
    }
  }
}

/// Decode @p sources from their column chunks already on the device (@p per_source: one span
/// list per source, in @p sources order; @p ranges: the file ranges they were read from, per
/// source, where known).  The spans are first checked against the column chunks the reader built
/// for this decode enumerates under @p options, so chunks staged for other options (another
/// projection, a filter column outside it) are refused before any decode work instead of being
/// misread.
std::unique_ptr<cudf::table> decode_from_device(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  std::span<prefetched_chunk_spans const> per_source,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (sources.size() == 1) {
    // Single file: the single-source reader, whose chunk data is flat.
    auto const& src   = sources.front();
    auto const spans  = per_source.front();
    auto const groups = cudf::host_span<cudf::size_type const>(src.row_group_indices.data(),
                                                               src.row_group_indices.size());
    hybrid_scan_reader reader(*src.metadata, options);
    auto const expected = reader.all_column_chunks_byte_ranges(groups, options);
    auto const what     = "'" + path_of(src) + "'";
    require_spans_match(expected, spans, what);
    require_offsets_match(expected, ranges_for(ranges, 0), what);

    auto result = reader.materialize_all_columns(
      groups,
      cudf::host_span<cudf::device_span<std::uint8_t const> const>(spans.data(), spans.size()),
      options,
      stream,
      mr);
    return std::move(result.tbl);
  }

  require_same_parquet_schema(sources);

  // Several files in one split: the multi-file reader takes the row groups per source and wants
  // its chunk data flattened in source order, with each source's chunks in the same row-group /
  // column order the single-file reader produces.
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  std::vector<cudf::device_span<std::uint8_t const>> spans;
  metadatas.reserve(sources.size());
  rg_per_src.reserve(sources.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    metadatas.push_back(*sources[i].metadata);
    rg_per_src.push_back(sources[i].row_group_indices);
    spans.insert(spans.end(), per_source[i].begin(), per_source[i].end());
  }

  cudf::io::parquet::experimental::hybrid_scan_multifile reader(
    cudf::host_span<cudf::io::parquet::FileMetaData const>(metadatas.data(), metadatas.size()),
    options);
  auto const groups =
    cudf::host_span<std::vector<cudf::size_type> const>(rg_per_src.data(), rg_per_src.size());
  auto const [expected, source_of] = reader.all_column_chunks_byte_ranges(groups, options);
  require_spans_match(expected, spans, std::to_string(sources.size()) + " sources");
  // Equal totals could still hide a chunk attributed to the wrong file.
  std::size_t k = 0;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto const first = k;
    for (std::size_t j = 0; j < per_source[i].size(); ++j, ++k) {
      if (static_cast<std::size_t>(source_of[k]) != i) {
        throw prefetched_mismatch(
          "[parquet_materialize] device spans do not match the reader's "
          "column chunks for '" +
          path_of(sources[i]) + "': chunk " + std::to_string(k) + " belongs to source " +
          std::to_string(source_of[k]));
      }
    }
    require_offsets_match(std::span(expected).subspan(first, per_source[i].size()),
                          ranges_for(ranges, i),
                          "'" + path_of(sources[i]) + "'");
  }

  auto result = reader.materialize_all_columns(
    groups,
    cudf::host_span<cudf::device_span<std::uint8_t const> const>(spans.data(), spans.size()),
    options,
    stream,
    mr);
  return std::move(result.tbl);
}

/// Bulk route: read every source's column chunks, then decode straight out of
/// those device buffers.
std::unique_ptr<cudf::table> materialize_bulk(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (sources.size() == 1) {
    // Single file: the single-source reader, whose chunk data is flat.
    auto const& src   = sources.front();
    auto const chunks = fetch_chunks(src, options, ranges_for(ranges, 0), stream, mr);

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

  require_same_parquet_schema(sources);

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
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
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
  if (std::all_of(
        rg_per_src.begin(), rg_per_src.end(), [](auto const& groups) { return groups.empty(); })) {
    opts.set_num_rows(0);
  } else {
    opts.set_row_groups(std::move(rg_per_src));
  }

  auto [table, _] =
    cudf::io::read_parquet(std::move(cudf_sources), std::move(metadatas), opts, stream, mr);
  return std::move(table);
}

}  // namespace

void require_same_parquet_schema(std::span<parquet_source const> sources)
{
  if (sources.size() < 2) { return; }
  auto const& first = sources.front().metadata->schema;
  for (auto const& src : sources.subspan(1)) {
    auto const& schema = src.metadata->schema;
    if (schema == first) { continue; }
    std::string const prefix = "[parquet_materialize] All sources must have the same schema: '" +
                               path_of(sources.front()) + "' and '" + path_of(src) + "' ";
    if (schema.size() != first.size()) {
      throw std::runtime_error(prefix + "have " + std::to_string(first.size()) + " and " +
                               std::to_string(schema.size()) +
                               " schema elements (including the root)");
    }
    auto const [lhs, rhs] = std::mismatch(first.begin(), first.end(), schema.begin(), schema.end());
    std::string detail    = describe(*lhs) + " vs " + describe(*rhs);
    if (lhs->name == rhs->name && lhs->type == rhs->type) {
      detail += "; they differ in annotation, width or nesting";
    }
    throw std::runtime_error(prefix + "differ at schema element " +
                             std::to_string(lhs - first.begin()) + " (" + detail + ")");
  }
}

bool can_materialize_from_prefetched(
  std::span<parquet_source const> sources,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  std::span<prefetched_chunk_spans const> prefetched) noexcept
{
  if (sources.empty() || prefetched.size() != sources.size()) { return false; }
  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto const& src  = sources[i];
    auto const spans = prefetched[i];
    if (!src.metadata || src.row_group_indices.empty() || spans.empty()) { return false; }
    auto const expected = ranges_for(ranges, i);
    if (expected.empty()) { continue; }
    if (expected.size() != spans.size()) { return false; }
    for (std::size_t j = 0; j < spans.size(); ++j) {
      if (static_cast<std::size_t>(expected[j].size()) != spans[j].size()) { return false; }
    }
  }
  return true;
}

std::unique_ptr<cudf::table> materialize_parquet(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  std::span<prefetched_chunk_spans const> prefetched,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (prefetched.empty()) { return materialize_parquet(sources, options, ranges, stream, mr); }
  if (sources.empty()) { return nullptr; }
  if (!can_materialize_from_prefetched(sources, ranges, prefetched)) {
    throw std::invalid_argument(
      "[parquet_materialize] prefetched device spans need one non-empty entry per source, one "
      "span per column-chunk range");
  }
  return decode_from_device(sources, options, ranges, prefetched, stream, mr);
}

std::unique_ptr<cudf::table> materialize_parquet(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (sources.empty()) { return nullptr; }
  if (prefers_bulk_materialize(sources, options)) {
    return materialize_bulk(sources, options, ranges, stream, mr);
  }
  return materialize_general(sources, options, stream, mr);
}

}  // namespace sirius::op::scan
