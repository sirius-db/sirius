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

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/io/types.hpp>

#include <algorithm>
#include <future>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
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

/// Cap unjoined file reads per materialization; demand misses can compete with
/// read-ahead for staging.
constexpr std::size_t max_pending_fetches = 32;

/// One source's vectored read, issued and not yet joined.
struct pending_fetch {
  pending_fetch()                           = default;
  pending_fetch(pending_fetch&&)            = default;
  pending_fetch& operator=(pending_fetch&&) = delete;

  fetched_chunks chunks;
  std::vector<cucascade::io::slice> reads;
  /// Bytes the ranges ask for; a backend clips a range that passes the end of its object.
  std::size_t requested = 0;
  /// Declared last: destroying it waits for the read, before the slices and
  /// the buffers it writes to are freed.  Assignment would free them first.
  std::future<std::size_t> read;
};
static_assert(std::is_nothrow_move_constructible_v<pending_fetch>);

/// Start one logical vectored read of every column chunk of @p src into device
/// memory.  The backend may serve it from cache or split it into physical reads.
pending_fetch begin_fetch(parquet_source const& src,
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

  pending_fetch out;
  for (auto const& range : ranges) {
    auto const size = static_cast<std::size_t>(range.size());
    if (size > std::numeric_limits<std::size_t>::max() - out.requested) {
      throw std::overflow_error("Parquet column data: requested byte count overflows");
    }
    out.requested += size;
  }
  out.chunks.buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    out.chunks.buffers.emplace_back(static_cast<std::size_t>(range.size()), stream, mr);
  }

  out.reads.reserve(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    out.reads.emplace_back(static_cast<std::size_t>(ranges[i].offset()),
                           static_cast<std::size_t>(ranges[i].size()),
                           static_cast<std::uint8_t*>(out.chunks.buffers[i].data()));
  }

  out.read = src.datasource->device_read_ranges_async(out.reads, stream);
  return out;
}

/// Wait for @p fetch's read.  Throws unless it returned every requested byte:
/// the rest of a clipped range's buffer was never written.
void join_exact(pending_fetch& fetch)
{
  auto const read = fetch.read.get();
  if (read != fetch.requested) {
    throw std::runtime_error("Parquet column data: read " + std::to_string(read) + " of " +
                             std::to_string(fetch.requested) + " requested bytes");
  }
}

fetched_chunks finish_fetch(pending_fetch fetch)
{
  join_exact(fetch);

  auto out = std::move(fetch.chunks);
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
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (sources.size() == 1) {
    // Single file: the single-source reader, whose chunk data is flat.
    auto const& src   = sources.front();
    auto const chunks = finish_fetch(begin_fetch(src, options, ranges_for(ranges, 0), stream, mr));

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

  // Reserved before the first read is issued: taking ownership of an issued
  // read cannot throw, so its result is never dropped unread.
  std::vector<pending_fetch> fetches;
  fetches.reserve(sources.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    try {
      for (auto j = fetches.size(); j < sources.size() && j - i < max_pending_fetches; ++j) {
        fetches.push_back(begin_fetch(sources[j], options, ranges_for(ranges, j), stream, mr));
      }
    } catch (...) {
      // A failed read of an earlier file comes first: reading the files one at
      // a time would have stopped there.
      for (auto j = i; j < fetches.size(); ++j) {
        join_exact(fetches[j]);
      }
      throw;
    }
    auto chunks = finish_fetch(std::move(fetches[i]));
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
