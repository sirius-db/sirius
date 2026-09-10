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

#include "simpatico_file_ingest.hpp"

#include <api/simpatico_codegen.hpp>
#include <codegen/plan/leaf_desc.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sirius {

namespace {

/// First guess at how many bytes of a .hpln are header. Wide enough that a normal table parses in
/// one read; the loop below handles the rest.
constexpr std::uint64_t kHeaderProbeBytes = 1u << 20;  // 1 MiB
/// Past this a "header" is not plausible and a malformed file is the likelier explanation than a
/// very wide schema, so stop rather than read the whole object looking for one.
constexpr std::uint64_t kHeaderProbeMax = 64u << 20;  // 64 MiB
/// Tail read that should contain the trailer and the whole postscript in one go.
constexpr std::uint64_t kTailProbeBytes = 64u << 10;  // 64 KiB

/// A .hpln opened far enough to know where every chunk is.
struct hpln_layout {
  std::vector<simpatico::hpln_segment_ref> segs;
  std::vector<simpatico::hpln_chunk_ref> chunks;
  /// False for a file written before the trailer existed, whose extent can only be found by
  /// parsing a speculative prefix. Such a file is always one chunk.
  bool located = false;
};

std::optional<simpatico::hpln_segment_ref> find_segment(hpln_layout const& layout,
                                                        simpatico::hpln_segment kind)
{
  for (auto const& sg : layout.segs) {
    if (sg.kind == kind) { return sg; }
  }
  return std::nullopt;
}

/// Locate the segments and resolve the chunk directory of an open file.
///
/// One tail read locates everything: the trailer names the postscript, the postscript names every
/// segment, and the chunk directory subdivides them. That is what makes the format cheap to open
/// over an object store, where a request costs more than the bytes it moves
/// (CHUNK_SKIPPING_PLAN.md 7.6).
hpln_layout locate_hpln(hpln_source& src, std::string const& path)
{
  auto const file_size = src.size();
  hpln_layout out;
  std::uint64_t need = 0;
  auto tail          = src.read_tail(kTailProbeBytes, "trailer");
  auto err           = simpatico::read_hpln_postscript(tail, file_size, out.segs, &need);
  if (!err.empty() && need > tail.size() && need <= file_size) {
    tail = src.read_tail(need, "trailer");  // exact re-read, never a second guess
    err  = simpatico::read_hpln_postscript(tail, file_size, out.segs, nullptr);
  }
  if (!err.empty()) {
    out.segs.clear();
    return out;  // pre-trailer file; the caller falls back to parsing from the front
  }
  out.located = true;

  // A directory is authoritative when present. Without one the file predates chunking and is a
  // single chunk covering the whole header and payload segments -- unknown kinds being skipped is
  // exactly what makes that additive.
  if (auto const dir = find_segment(out, simpatico::hpln_segment::chunk_directory)) {
    auto const derr = simpatico::unpack_hpln_chunk_directory(
      src.read_range(dir->offset, dir->bytes, "chunk directory"), out.chunks);
    if (!derr.empty()) { throw std::runtime_error("[hpln] '" + path + "': " + derr); }
    if (out.chunks.empty()) {
      throw std::runtime_error("[hpln] '" + path + "': chunk directory names no chunks");
    }
  } else {
    auto const hdr = find_segment(out, simpatico::hpln_segment::header);
    auto const pay = find_segment(out, simpatico::hpln_segment::payload);
    if (!hdr) { throw std::runtime_error("[hpln] '" + path + "': no header segment"); }
    out.chunks.push_back({hdr->offset,
                          hdr->bytes,
                          pay ? pay->offset : hdr->offset + hdr->bytes,
                          pay ? pay->bytes : 0,
                          0});
  }

  for (auto const& c : out.chunks) {
    if (c.header_offset + c.header_bytes > file_size ||
        c.payload_offset + c.payload_bytes > file_size) {
      throw std::runtime_error("[hpln] '" + path + "' is truncated: a chunk runs past the file");
    }
  }
  return out;
}

/// True when two chunks describe the same columns. Row counts may differ -- chunks are sized by
/// the writer, not by the schema.
bool same_schema(simpatico::hpln_schema const& a, simpatico::hpln_schema const& b)
{
  if (a.columns.size() != b.columns.size()) { return false; }
  for (std::size_t i = 0; i < a.columns.size(); ++i) {
    if (a.columns[i].name != b.columns[i].name ||
        a.columns[i].dtype_tag != b.columns[i].dtype_tag ||
        a.columns[i].scale != b.columns[i].scale) {
      return false;
    }
  }
  return true;
}

/// Parse every chunk's structural header, in one pass over the contiguous header region.
///
/// This is the payoff of the segregated layout: N chunks cost one sequential read of metadata
/// rather than a seek per chunk. It also validates that the chunks agree, which has to happen
/// HERE -- discovering mid-scan that chunk 7 has a different column order means a batch that
/// cannot be concatenated, or worse one that can but is wrong.
void describe_chunks(hpln_source& src,
                     std::string const& path,
                     hpln_layout& layout,
                     hpln_io_policy const& policy,
                     std::vector<std::vector<std::uint8_t>>& out_headers,
                     simpatico::hpln_schema& out_schema)
{
  out_headers.clear();
  out_headers.resize(layout.chunks.size());

  // Every chunk header in ONE planned read. The layout puts them contiguously, so the coalescer
  // turns N chunks into one request rather than N -- which is the difference between a bind
  // costing one round trip and costing one per chunk.
  std::vector<hpln_extent> extents;
  extents.reserve(layout.chunks.size());
  for (std::size_t i = 0; i < layout.chunks.size(); ++i) {
    auto const& chunk = layout.chunks[i];
    out_headers[i].resize(static_cast<std::size_t>(chunk.header_bytes));
    if (chunk.header_bytes == 0) { continue; }
    hpln_extent e;
    e.offset = chunk.header_offset;
    e.dst.push_back({out_headers[i].data(), chunk.header_bytes});
    extents.push_back(std::move(e));
  }
  src.read_extents(std::move(extents), policy, "header segment");

  for (std::size_t i = 0; i < layout.chunks.size(); ++i) {
    auto& chunk = layout.chunks[i];
    simpatico::hpln_schema schema;
    auto const err = simpatico::describe_compressed_table_header(out_headers[i], schema);
    if (!err.empty()) {
      throw std::runtime_error("[hpln] '" + path + "' chunk " + std::to_string(i) +
                               " header does not parse: " + err);
    }
    // The directory and the header must agree about the chunk's extent. They are written from the
    // same numbers, so a disagreement is corruption -- and a payload read at a wrong offset does
    // not fault, it returns a neighbour's bytes as values.
    if (schema.header_bytes != chunk.header_bytes || schema.payload_bytes > chunk.payload_bytes) {
      throw std::runtime_error("[hpln] '" + path + "' chunk " + std::to_string(i) +
                               ": the directory and the header disagree about its extent");
    }
    if (i == 0) {
      out_schema = schema;
    } else if (!same_schema(out_schema, schema)) {
      throw std::runtime_error("[hpln] '" + path + "' chunk " + std::to_string(i) +
                               " has a different schema from chunk 0; a file's chunks must all "
                               "describe the same table");
    }
    chunk.num_rows = schema.columns.empty() ? 0 : schema.columns.front().num_rows;
  }
}

/// Allocate a chunk's pinned blocks and describe the read that fills them.
///
/// Allocating and reading are separated so a batch of chunks can be planned as ONE set of
/// requests: the payloads of consecutive chunks are adjacent in the file, so the coalescer fuses
/// them into large sequential reads instead of one request per chunk (and per block).
std::shared_ptr<pinned_compressed_blob> allocate_chunk(std::vector<std::uint8_t> header,
                                                       simpatico::hpln_chunk_ref const& chunk,
                                                       cucascade::memory::memory_space& host_space,
                                                       std::vector<hpln_extent>& out_extents)
{
  auto* host_mr = host_space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("[hpln ingest] target host space has no host memory resource");
  }

  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->header        = std::move(header);
  auto const nb       = chunk.payload_bytes;
  auto payload_res    = host_space.make_reservation_or_null(nb);
  blob->payload       = host_mr->allocate_multiple_blocks(nb, payload_res.get());
  blob->payload_bytes = nb;
  if (nb == 0) { return blob; }

  // Straight into the pinned blocks -- the allocation is not contiguous, but the file range is,
  // so it is one extent scattered across blocks rather than one read per block. Going through an
  // intermediate host buffer would double both the copy and the footprint.
  hpln_extent e;
  e.offset           = chunk.payload_offset;
  auto const block   = blob->payload->block_size();
  std::uint64_t done = 0;
  std::size_t idx    = 0;
  while (done < nb) {
    auto const n = std::min<std::uint64_t>(block, nb - done);
    e.dst.push_back({reinterpret_cast<std::uint8_t*>(blob->payload->at(idx).data()), n});
    done += n;
    ++idx;
  }
  out_extents.push_back(std::move(e));
  return blob;
}

/// Open @p path for reading, through @p options's io_context when it has one.
std::unique_ptr<hpln_source> open_hpln(std::string const& path,
                                       char const* who,
                                       hpln_open_options const& options)
{
  return open_hpln_source(path, options.io_ctx, who);
}

/// Publish what the transport did, for a caller that asked.
void report(hpln_source const& src, hpln_open_options const& options)
{
  if (options.stats != nullptr) { *options.stats = src.stats(); }
}

}  // namespace

std::vector<ingested_hpln_chunk> read_hpln_chunks_into_pinned(
  std::string const& path,
  cucascade::memory::memory_space& host_space,
  std::span<const std::size_t> chunk_ids,
  hpln_open_options const& options)
{
  auto src    = open_hpln(path, "hpln ingest", options);
  auto layout = locate_hpln(*src, path);
  if (!layout.located) {
    throw std::runtime_error("[hpln ingest] '" + path +
                             "' predates the trailer and has no chunk directory; it can only be "
                             "read as a single chunk");
  }
  std::vector<std::vector<std::uint8_t>> headers;
  simpatico::hpln_schema schema;
  describe_chunks(*src, path, layout, options.policy, headers, schema);

  // Allocate every requested chunk first, then fill them all in one planned read: consecutive
  // chunks are adjacent in the payload region, so the batch collapses to a few large sequential
  // requests rather than one per chunk (7.7). No chunk outside `chunk_ids` is touched.
  std::vector<ingested_hpln_chunk> out;
  std::vector<hpln_extent> extents;
  out.reserve(chunk_ids.size());
  extents.reserve(chunk_ids.size());
  // A chunk named twice is staged ONCE and served to both positions. The blob is read-only from
  // here on, so sharing it is not just cheaper -- reading the same range into two buffers would
  // mean two overlapping requests, which the read planner refuses precisely because overlapping
  // destinations are how a range silently lands in the wrong place.
  std::unordered_map<std::size_t, std::shared_ptr<pinned_compressed_blob>> staged;
  for (auto const id : chunk_ids) {
    if (id >= layout.chunks.size()) {
      throw std::runtime_error("[hpln ingest] '" + path + "': chunk " + std::to_string(id) +
                               " is out of range for a file with " +
                               std::to_string(layout.chunks.size()) + " chunks");
    }
    auto it = staged.find(id);
    if (it == staged.end()) {
      it = staged.emplace(id, allocate_chunk(headers[id], layout.chunks[id], host_space, extents))
             .first;
    }
    out.push_back({it->second, layout.chunks[id].num_rows});
  }
  src->read_extents(std::move(extents), options.policy, "chunk payload");
  report(*src, options);
  return out;
}

ingested_hpln read_hpln_into_pinned(std::string const& path,
                                    cucascade::memory::memory_space& host_space,
                                    hpln_open_options const& options)
{
  auto src             = open_hpln(path, "hpln ingest", options);
  auto const file_size = src->size();
  auto layout          = locate_hpln(*src, path);

  ingested_hpln out;
  std::vector<std::uint8_t> header;
  if (layout.located) {
    if (layout.chunks.size() != 1) {
      throw std::runtime_error("[hpln ingest] '" + path + "' holds " +
                               std::to_string(layout.chunks.size()) +
                               " chunks; read_hpln_into_pinned serves a single chunk only");
    }
    std::vector<std::vector<std::uint8_t>> headers;
    describe_chunks(*src, path, layout, options.policy, headers, out.schema);
    header = std::move(headers.front());
  } else {
    // Pre-trailer file: grow a speculative prefix until the header parses, which is the round trip
    // the trailer exists to avoid.
    std::vector<std::uint8_t> prefix;
    std::string err;
    for (std::uint64_t want = kHeaderProbeBytes;; want *= 2) {
      prefix = src->read_prefix(want, "header");
      err    = simpatico::describe_compressed_table_header(prefix, out.schema);
      if (err.empty()) { break; }
      if (prefix.size() >= file_size || want >= kHeaderProbeMax) {
        throw std::runtime_error("[hpln ingest] '" + path + "' is not a readable .hpln: " + err);
      }
    }
    if (out.schema.header_bytes + out.schema.payload_bytes > file_size) {
      throw std::runtime_error("[hpln ingest] '" + path + "' is truncated: header says " +
                               std::to_string(out.schema.header_bytes + out.schema.payload_bytes) +
                               " bytes, file has " + std::to_string(file_size));
    }
    layout.chunks.push_back({0,
                             out.schema.header_bytes,
                             out.schema.header_bytes,
                             out.schema.payload_bytes,
                             out.schema.columns.empty() ? 0 : out.schema.columns.front().num_rows});
    header.assign(prefix.begin(),
                  prefix.begin() + static_cast<std::ptrdiff_t>(out.schema.header_bytes));
  }

  std::vector<hpln_extent> extents;
  out.blob = allocate_chunk(std::move(header), layout.chunks.front(), host_space, extents);
  src->read_extents(std::move(extents), options.policy, "chunk payload");

  // Zone maps, if the file carries them. A file written before the segment existed, or one whose
  // segment does not decode, simply serves unpruned -- so this never fails the ingest.
  for (auto const& sg : layout.segs) {
    if (sg.bytes == 0) { continue; }
    // Unknown segment kinds are skipped, which is what makes them additive.
    if (sg.kind == simpatico::hpln_segment::zone_maps) {
      std::string zerr;
      out.group_bounds = scan_manager::group_bounds_arena::unpack(
        src->read_range(sg.offset, sg.bytes, "zone map segment"), &zerr);
      if (!zerr.empty()) {
        SIRIUS_LOG_WARN(
          "[hpln ingest] '{}': zone maps unreadable ({}); serving unpruned", path, zerr);
      }
    } else if (sg.kind == simpatico::hpln_segment::logical_types) {
      std::string terr;
      out.column_types =
        unpack_logical_types(src->read_range(sg.offset, sg.bytes, "logical types segment"), &terr);
      if (!terr.empty()) {
        SIRIUS_LOG_WARN(
          "[hpln ingest] '{}': logical types unreadable ({}); the caller has only "
          "the cuDF physical types",
          path,
          terr);
      }
    }
  }

  SIRIUS_LOG_DEBUG("[hpln ingest] '{}': {} columns, {} rows, header {} B, payload {} B, {}",
                   path,
                   out.schema.columns.size(),
                   out.schema.columns.empty() ? 0 : out.schema.columns.front().num_rows,
                   out.schema.header_bytes,
                   out.schema.payload_bytes,
                   out.group_bounds.empty() ? "no zone maps" : "zone maps present");
  report(*src, options);
  return out;
}

namespace {

// Stable wire tags for the engine's logical types. OUR numbering, not duckdb::LogicalTypeId's,
// so a DuckDB upgrade that renumbers its enum cannot silently reinterpret existing files.
enum class type_tag : std::uint8_t {
  unknown = 0,
  boolean,
  i8,
  i16,
  i32,
  i64,
  u8,
  u16,
  u32,
  u64,
  f32,
  f64,
  decimal,
  varchar,
  date,
  time,
  timestamp,
  timestamp_tz,
  blob
};

type_tag tag_of(duckdb::LogicalType const& t)
{
  switch (t.id()) {
    case duckdb::LogicalTypeId::BOOLEAN: return type_tag::boolean;
    case duckdb::LogicalTypeId::TINYINT: return type_tag::i8;
    case duckdb::LogicalTypeId::SMALLINT: return type_tag::i16;
    case duckdb::LogicalTypeId::INTEGER: return type_tag::i32;
    case duckdb::LogicalTypeId::BIGINT: return type_tag::i64;
    case duckdb::LogicalTypeId::UTINYINT: return type_tag::u8;
    case duckdb::LogicalTypeId::USMALLINT: return type_tag::u16;
    case duckdb::LogicalTypeId::UINTEGER: return type_tag::u32;
    case duckdb::LogicalTypeId::UBIGINT: return type_tag::u64;
    case duckdb::LogicalTypeId::FLOAT: return type_tag::f32;
    case duckdb::LogicalTypeId::DOUBLE: return type_tag::f64;
    case duckdb::LogicalTypeId::DECIMAL: return type_tag::decimal;
    case duckdb::LogicalTypeId::VARCHAR: return type_tag::varchar;
    case duckdb::LogicalTypeId::DATE: return type_tag::date;
    case duckdb::LogicalTypeId::TIME: return type_tag::time;
    case duckdb::LogicalTypeId::TIMESTAMP: return type_tag::timestamp;
    case duckdb::LogicalTypeId::TIMESTAMP_TZ: return type_tag::timestamp_tz;
    case duckdb::LogicalTypeId::BLOB: return type_tag::blob;
    default: return type_tag::unknown;
  }
}

duckdb::LogicalType type_of(type_tag tag, std::uint8_t width, std::uint8_t scale)
{
  using L = duckdb::LogicalTypeId;
  switch (tag) {
    case type_tag::boolean: return duckdb::LogicalType(L::BOOLEAN);
    case type_tag::i8: return duckdb::LogicalType(L::TINYINT);
    case type_tag::i16: return duckdb::LogicalType(L::SMALLINT);
    case type_tag::i32: return duckdb::LogicalType(L::INTEGER);
    case type_tag::i64: return duckdb::LogicalType(L::BIGINT);
    case type_tag::u8: return duckdb::LogicalType(L::UTINYINT);
    case type_tag::u16: return duckdb::LogicalType(L::USMALLINT);
    case type_tag::u32: return duckdb::LogicalType(L::UINTEGER);
    case type_tag::u64: return duckdb::LogicalType(L::UBIGINT);
    case type_tag::f32: return duckdb::LogicalType(L::FLOAT);
    case type_tag::f64: return duckdb::LogicalType(L::DOUBLE);
    // Precision is the field cuDF cannot carry: it tracks scale only, so DECIMAL(12,2) and
    // DECIMAL(18,2) are indistinguishable once a table is compressed. This is why the segment
    // exists.
    case type_tag::decimal: return duckdb::LogicalType::DECIMAL(width, scale);
    case type_tag::varchar: return duckdb::LogicalType(L::VARCHAR);
    case type_tag::date: return duckdb::LogicalType(L::DATE);
    case type_tag::time: return duckdb::LogicalType(L::TIME);
    case type_tag::timestamp: return duckdb::LogicalType(L::TIMESTAMP);
    case type_tag::timestamp_tz: return duckdb::LogicalType(L::TIMESTAMP_TZ);
    case type_tag::blob: return duckdb::LogicalType(L::BLOB);
    default: return duckdb::LogicalType(L::SQLNULL);
  }
}

/// cuDF physical type -> the closest DuckDB type, for files with no `logical_types` segment.
/// Deliberately approximate: DECIMAL precision is set to the carrier's maximum because cuDF does
/// not record the declared one, which is the loss the logical_types segment exists to prevent.
duckdb::LogicalType duckdb_type_for_cudf(cudf::data_type dtype, std::int32_t scale)
{
  using L = duckdb::LogicalTypeId;
  switch (dtype.id()) {
    case cudf::type_id::INT8: return duckdb::LogicalType(L::TINYINT);
    case cudf::type_id::INT16: return duckdb::LogicalType(L::SMALLINT);
    case cudf::type_id::INT32: return duckdb::LogicalType(L::INTEGER);
    case cudf::type_id::INT64: return duckdb::LogicalType(L::BIGINT);
    case cudf::type_id::UINT8: return duckdb::LogicalType(L::UTINYINT);
    case cudf::type_id::UINT16: return duckdb::LogicalType(L::USMALLINT);
    case cudf::type_id::UINT32: return duckdb::LogicalType(L::UINTEGER);
    case cudf::type_id::UINT64: return duckdb::LogicalType(L::UBIGINT);
    case cudf::type_id::FLOAT32: return duckdb::LogicalType(L::FLOAT);
    case cudf::type_id::FLOAT64: return duckdb::LogicalType(L::DOUBLE);
    case cudf::type_id::STRING: return duckdb::LogicalType(L::VARCHAR);
    case cudf::type_id::DECIMAL32: return duckdb::LogicalType::DECIMAL(9, -scale);
    case cudf::type_id::DECIMAL64: return duckdb::LogicalType::DECIMAL(18, -scale);
    case cudf::type_id::DECIMAL128: return duckdb::LogicalType::DECIMAL(38, -scale);
    case cudf::type_id::TIMESTAMP_DAYS: return duckdb::LogicalType(L::DATE);
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return duckdb::LogicalType(L::TIMESTAMP);
    default: return duckdb::LogicalType(L::SQLNULL);
  }
}

constexpr std::uint16_t kLogicalTypesWireVersion = 1;

}  // namespace

std::vector<std::uint8_t> pack_logical_types(duckdb::vector<duckdb::LogicalType> const& types)
{
  std::vector<std::uint8_t> out;
  auto put16 = [&](std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
  };
  put16(kLogicalTypesWireVersion);
  put16(static_cast<std::uint16_t>(types.size()));
  for (auto const& t : types) {
    auto const tag = tag_of(t);
    out.push_back(static_cast<std::uint8_t>(tag));
    if (tag == type_tag::decimal) {
      out.push_back(duckdb::DecimalType::GetWidth(t));
      out.push_back(duckdb::DecimalType::GetScale(t));
    } else {
      out.push_back(0);
      out.push_back(0);
    }
  }
  return out;
}

duckdb::vector<duckdb::LogicalType> unpack_logical_types(std::span<const std::uint8_t> bytes,
                                                         std::string* error)
{
  auto fail = [&](char const* why) {
    if (error) *error = why;
    return duckdb::vector<duckdb::LogicalType>{};
  };
  if (bytes.size() < 4) return fail("logical-types segment: truncated preamble");
  auto const version = static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
  auto const n       = static_cast<std::uint16_t>(bytes[2] | (bytes[3] << 8));
  if (version != kLogicalTypesWireVersion) {
    return fail("logical-types segment: unsupported version");
  }
  if (bytes.size() < 4u + 3u * n) return fail("logical-types segment: truncated column table");
  duckdb::vector<duckdb::LogicalType> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto const* p = bytes.data() + 4 + 3 * i;
    out.push_back(type_of(static_cast<type_tag>(p[0]), p[1], p[2]));
  }
  return out;
}

hpln_bind_schema read_hpln_schema(std::string const& path, hpln_open_options const& options)
{
  // Reuses the ingest reader for locating and parsing, but stops before any payload is staged:
  // binding a query must not move data. Every chunk's header is parsed -- one sequential read of
  // the segregated metadata region -- both to sum the row counts the optimizer wants and to
  // refuse a file whose chunks disagree here rather than mid-scan.
  auto src             = open_hpln(path, "hpln schema", options);
  auto const file_size = src->size();
  auto layout          = locate_hpln(*src, path);

  simpatico::hpln_schema header_schema;
  duckdb::vector<duckdb::LogicalType> declared;
  std::vector<std::int64_t> chunk_rows;
  scan_manager::group_bounds_arena bounds;

  if (layout.located) {
    std::vector<std::vector<std::uint8_t>> headers;
    describe_chunks(*src, path, layout, options.policy, headers, header_schema);
    for (auto const& c : layout.chunks) {
      chunk_rows.push_back(c.num_rows);
    }
    if (auto const sg = find_segment(layout, simpatico::hpln_segment::logical_types);
        sg && sg->bytes > 0) {
      declared = unpack_logical_types(
        src->read_range(sg->offset, sg->bytes, "logical types segment"), nullptr);
    }
    // The zone maps are read HERE rather than at scan time because a bind is where a scan learns
    // what it may skip: the walk has to decide a chunk's fate before it emits a split for it, and
    // the bounds are one small segregated read whatever the file's size.
    if (auto const sg = find_segment(layout, simpatico::hpln_segment::zone_maps);
        sg && sg->bytes > 0) {
      std::string zerr;
      bounds = scan_manager::group_bounds_arena::unpack(
        src->read_range(sg->offset, sg->bytes, "zone map segment"), &zerr);
      if (!zerr.empty()) {
        SIRIUS_LOG_WARN(
          "[hpln schema] '{}': zone maps unreadable ({}); binding unpruned", path, zerr);
      }
    }
  } else {
    // Pre-trailer file: parse the header from the front, growing the prefix as needed.
    std::string herr;
    for (std::uint64_t want = kHeaderProbeBytes;; want *= 2) {
      auto const prefix = src->read_prefix(want, "header");
      herr              = simpatico::describe_compressed_table_header(prefix, header_schema);
      if (herr.empty()) { break; }
      if (prefix.size() >= file_size || want >= kHeaderProbeMax) {
        throw std::runtime_error("[hpln schema] '" + path + "' is not a readable .hpln: " + herr);
      }
    }
    chunk_rows.push_back(header_schema.columns.empty() ? 0
                                                       : header_schema.columns.front().num_rows);
  }

  hpln_bind_schema out;
  out.chunk_rows = std::move(chunk_rows);
  // A capture whose chunk count disagrees with the file's is evidence of nothing, and pruning on
  // it would drop the wrong chunks -- so it is discarded rather than used positionally.
  if (!bounds.empty() && bounds.chunk_count() == out.chunk_rows.size()) {
    out.group_bounds = std::move(bounds);
  } else if (!bounds.empty()) {
    SIRIUS_LOG_WARN(
      "[hpln schema] '{}': zone maps describe {} chunks but the file has {}; binding unpruned",
      path,
      bounds.chunk_count(),
      out.chunk_rows.size());
  }
  out.num_rows = std::accumulate(out.chunk_rows.begin(), out.chunk_rows.end(), std::int64_t{0});
  out.names.reserve(header_schema.columns.size());
  out.types.reserve(header_schema.columns.size());
  out.physical_types.reserve(header_schema.columns.size());
  for (std::size_t i = 0; i < header_schema.columns.size(); ++i) {
    auto const& c = header_schema.columns[i];
    out.names.push_back(c.name.empty() ? "column" + std::to_string(i) : c.name);
    out.physical_types.push_back(simpatico::tag_to_dtype(c.dtype_tag));
    if (i < declared.size() && declared[i].id() != duckdb::LogicalTypeId::SQLNULL) {
      out.types.push_back(declared[i]);
    } else {
      // Fall back to the cuDF physical type. Lossy by construction -- see 7.9 -- but a file
      // without declared types should still bind rather than refuse.
      out.types.push_back(duckdb_type_for_cudf(simpatico::tag_to_dtype(c.dtype_tag), c.scale));
    }
  }
  report(*src, options);
  return out;
}

std::string write_tables_to_hpln(std::vector<cudf::table_view> const& tables,
                                 duckdb::vector<duckdb::LogicalType> const& column_types,
                                 std::vector<std::string> const& column_names,
                                 std::string const& plan_dsl,
                                 std::size_t group_rows,
                                 std::string const& path,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  if (tables.empty()) { return "[hpln export] no chunks to write"; }

  std::vector<simpatico::compressed_table> chunks;
  chunks.reserve(tables.size());
  std::vector<scan_manager::chunk_group_stats> per_chunk;
  for (auto const& t : tables) {
    try {
      chunks.push_back(simpatico::compress_with_plan(t, plan_dsl, stream, mr, column_names));
    } catch (std::exception const& e) {
      return std::string("[hpln export] compression failed: ") + e.what();
    }
    // Statistics come from the DECODED values, which is why they have to be computed here rather
    // than recovered later: once the file holds compressed bytes the bounds are gone until someone
    // decodes them, and the whole point of carrying them is to avoid that.
    if (group_rows > 0 && !column_types.empty()) {
      per_chunk.push_back(
        scan_manager::compute_pinned_group_stats(t, column_types, group_rows, stream, mr));
    }
  }

  // Every chunk's bounds, in chunk order, in one arena -- so a pruning reader indexes it by the
  // same chunk id the directory uses.
  std::vector<std::uint8_t> packed;
  if (!per_chunk.empty()) {
    packed = scan_manager::group_bounds_arena::from_capture(column_types, per_chunk).pack();
  }

  // The engine's types always travel: without them a reader has only cuDF physical types and
  // cannot reconstruct DECIMAL precision or nullability, so the file would not be self-describing.
  auto const type_bytes = pack_logical_types(column_types);
  std::vector<simpatico::hpln_extra_segment> extra;
  if (!type_bytes.empty()) {
    extra.push_back({simpatico::hpln_segment::logical_types, type_bytes});
  }
  if (!packed.empty()) { extra.push_back({simpatico::hpln_segment::zone_maps, packed}); }

  std::vector<simpatico::compressed_table const*> refs;
  refs.reserve(chunks.size());
  for (auto const& c : chunks) {
    refs.push_back(&c);
  }
  return simpatico::write_compressed_tables(refs, path, stream, extra);
}

std::string write_table_to_hpln(cudf::table_view const& table,
                                duckdb::vector<duckdb::LogicalType> const& column_types,
                                std::vector<std::string> const& column_names,
                                std::string const& plan_dsl,
                                std::size_t group_rows,
                                std::string const& path,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  return write_tables_to_hpln(std::vector<cudf::table_view>{table},
                              column_types,
                              column_names,
                              plan_dsl,
                              group_rows,
                              path,
                              stream,
                              mr);
}

}  // namespace sirius
