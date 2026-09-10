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
#include <stdexcept>
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

std::vector<std::uint8_t> read_prefix(std::ifstream& f, std::uint64_t want, std::uint64_t file_size)
{
  auto const n = static_cast<std::size_t>(std::min(want, file_size));
  std::vector<std::uint8_t> buf(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  if (!f) { throw std::runtime_error("[hpln ingest] short read while reading the header"); }
  return buf;
}

std::vector<std::uint8_t> read_tail(std::ifstream& f, std::uint64_t want, std::uint64_t file_size)
{
  auto const n = static_cast<std::size_t>(std::min(want, file_size));
  std::vector<std::uint8_t> buf(n);
  f.seekg(static_cast<std::streamoff>(file_size - n));
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  if (!f) { throw std::runtime_error("[hpln ingest] short read while reading the trailer"); }
  return buf;
}

}  // namespace

ingested_hpln read_hpln_into_pinned(std::string const& path,
                                    cucascade::memory::memory_space& host_space)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  auto const file_size = static_cast<std::uint64_t>(fs::file_size(path, ec));
  if (ec) { throw std::runtime_error("[hpln ingest] cannot stat '" + path + "': " + ec.message()); }

  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("[hpln ingest] cannot open '" + path + "'"); }

  // Preferred path: the trailer says exactly where the header is, so one tail read locates
  // everything. Falls back to growing a speculative prefix for pre-v7 files, which is what the
  // trailer exists to avoid -- over a network the guess costs a round trip every time it is short.
  ingested_hpln out;
  std::vector<std::uint8_t> prefix;
  std::uint64_t header_at = 0;
  bool located            = false;
  {
    std::vector<simpatico::hpln_segment_ref> segs;
    std::uint64_t need = 0;
    auto tail          = read_tail(f, kTailProbeBytes, file_size);
    auto err           = simpatico::read_hpln_postscript(tail, file_size, segs, &need);
    if (!err.empty() && need > tail.size() && need <= file_size) {
      tail = read_tail(f, need, file_size);  // exact re-read, never a second guess
      err  = simpatico::read_hpln_postscript(tail, file_size, segs, nullptr);
    }
    if (err.empty()) {
      for (auto const& sg : segs) {
        if (sg.kind != simpatico::hpln_segment::header) { continue; }
        prefix.resize(static_cast<std::size_t>(sg.bytes));
        f.seekg(static_cast<std::streamoff>(sg.offset));
        f.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(sg.bytes));
        if (!f) { throw std::runtime_error("[hpln ingest] short read of the header segment"); }
        header_at = sg.offset;
        located   = true;
        break;
      }
    }
  }
  if (located) {
    auto const err = simpatico::describe_compressed_table_header(prefix, out.schema);
    if (!err.empty()) {
      throw std::runtime_error("[hpln ingest] '" + path +
                               "' header segment does not parse: " + err);
    }
  } else {
    std::string err;
    for (std::uint64_t want = kHeaderProbeBytes;; want *= 2) {
      prefix = read_prefix(f, want, file_size);
      err    = simpatico::describe_compressed_table_header(prefix, out.schema);
      if (err.empty()) { break; }
      if (prefix.size() >= file_size || want >= kHeaderProbeMax) {
        throw std::runtime_error("[hpln ingest] '" + path + "' is not a readable .hpln: " + err);
      }
    }
  }

  auto const header_bytes  = out.schema.header_bytes;
  auto const payload_bytes = out.schema.payload_bytes;
  if (header_at + header_bytes + payload_bytes > file_size) {
    throw std::runtime_error("[hpln ingest] '" + path + "' is truncated: header says " +
                             std::to_string(header_at + header_bytes + payload_bytes) +
                             " bytes, file has " + std::to_string(file_size));
  }

  auto* host_mr = host_space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("[hpln ingest] target host space has no host memory resource");
  }

  out.blob = std::make_shared<pinned_compressed_blob>();
  out.blob->header.assign(prefix.begin(),
                          prefix.begin() + static_cast<std::ptrdiff_t>(header_bytes));
  auto payload_res        = host_space.make_reservation_or_null(payload_bytes);
  out.blob->payload       = host_mr->allocate_multiple_blocks(payload_bytes, payload_res.get());
  out.blob->payload_bytes = payload_bytes;

  // Straight into the pinned blocks, one block at a time -- the allocation is not contiguous, and
  // going through an intermediate host buffer would double both the copy and the footprint.
  f.seekg(static_cast<std::streamoff>(header_at + header_bytes));
  auto const block   = out.blob->payload->block_size();
  std::uint64_t done = 0;
  std::size_t idx    = 0;
  while (done < payload_bytes) {
    auto const n = static_cast<std::size_t>(std::min<std::uint64_t>(block, payload_bytes - done));
    f.read(reinterpret_cast<char*>(out.blob->payload->at(idx).data()),
           static_cast<std::streamsize>(n));
    if (!f) {
      throw std::runtime_error("[hpln ingest] short read at payload offset " +
                               std::to_string(done));
    }
    done += n;
    ++idx;
  }

  // Zone maps, if the file carries them. A file written before the segment existed, or one whose
  // segment does not decode, simply serves unpruned -- so this never fails the ingest.
  if (located) {
    std::vector<simpatico::hpln_segment_ref> segs;
    auto tail          = read_tail(f, kTailProbeBytes, file_size);
    std::uint64_t need = 0;
    auto err           = simpatico::read_hpln_postscript(tail, file_size, segs, &need);
    if (!err.empty() && need > tail.size() && need <= file_size) {
      tail = read_tail(f, need, file_size);
      err  = simpatico::read_hpln_postscript(tail, file_size, segs, nullptr);
    }
    auto segment_bytes = [&](simpatico::hpln_segment_ref const& sg) {
      std::vector<std::uint8_t> buf(static_cast<std::size_t>(sg.bytes));
      f.seekg(static_cast<std::streamoff>(sg.offset));
      f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sg.bytes));
      if (!f) {
        f.clear();
        buf.clear();
      }
      return buf;
    };
    for (auto const& sg : segs) {
      if (sg.bytes == 0) { continue; }
      // Unknown segment kinds are skipped, which is what makes them additive.
      if (sg.kind == simpatico::hpln_segment::zone_maps) {
        std::string zerr;
        out.group_bounds = scan_manager::group_bounds_arena::unpack(segment_bytes(sg), &zerr);
        if (!zerr.empty()) {
          SIRIUS_LOG_WARN(
            "[hpln ingest] '{}': zone maps unreadable ({}); serving unpruned", path, zerr);
        }
      } else if (sg.kind == simpatico::hpln_segment::logical_types) {
        std::string terr;
        out.column_types = unpack_logical_types(segment_bytes(sg), &terr);
        if (!terr.empty()) {
          SIRIUS_LOG_WARN(
            "[hpln ingest] '{}': logical types unreadable ({}); the caller has only "
            "the cuDF physical types",
            path,
            terr);
        }
      }
    }
  }

  SIRIUS_LOG_DEBUG("[hpln ingest] '{}': {} columns, {} rows, header {} B, payload {} B, {}",
                   path,
                   out.schema.columns.size(),
                   out.schema.columns.empty() ? 0 : out.schema.columns.front().num_rows,
                   header_bytes,
                   payload_bytes,
                   out.group_bounds.empty() ? "no zone maps" : "zone maps present");
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

hpln_bind_schema read_hpln_schema(std::string const& path)
{
  // Reuses the ingest reader for locating and parsing, but stops before any payload is staged:
  // binding a query must not move data.
  namespace fs = std::filesystem;
  std::error_code ec;
  auto const file_size = static_cast<std::uint64_t>(fs::file_size(path, ec));
  if (ec) { throw std::runtime_error("[hpln schema] cannot stat '" + path + "': " + ec.message()); }
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("[hpln schema] cannot open '" + path + "'"); }

  simpatico::hpln_schema header_schema;
  duckdb::vector<duckdb::LogicalType> declared;
  std::vector<simpatico::hpln_segment_ref> segs;
  std::uint64_t need = 0;
  auto tail          = read_tail(f, kTailProbeBytes, file_size);
  auto err           = simpatico::read_hpln_postscript(tail, file_size, segs, &need);
  if (!err.empty() && need > tail.size() && need <= file_size) {
    tail = read_tail(f, need, file_size);
    err  = simpatico::read_hpln_postscript(tail, file_size, segs, nullptr);
  }

  auto read_seg = [&](simpatico::hpln_segment_ref const& sg) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(sg.bytes));
    f.seekg(static_cast<std::streamoff>(sg.offset));
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sg.bytes));
    if (!f) {
      f.clear();
      buf.clear();
    }
    return buf;
  };

  if (err.empty()) {
    for (auto const& sg : segs) {
      if (sg.bytes == 0) { continue; }
      if (sg.kind == simpatico::hpln_segment::header) {
        auto const bytes = read_seg(sg);
        auto const herr  = simpatico::describe_compressed_table_header(bytes, header_schema);
        if (!herr.empty()) { throw std::runtime_error("[hpln schema] '" + path + "': " + herr); }
      } else if (sg.kind == simpatico::hpln_segment::logical_types) {
        declared = unpack_logical_types(read_seg(sg), nullptr);
      }
    }
  } else {
    // Pre-trailer file: parse the header from the front, growing the prefix as needed.
    std::string herr;
    for (std::uint64_t want = kHeaderProbeBytes;; want *= 2) {
      auto const prefix = read_prefix(f, want, file_size);
      herr              = simpatico::describe_compressed_table_header(prefix, header_schema);
      if (herr.empty()) { break; }
      if (prefix.size() >= file_size || want >= kHeaderProbeMax) {
        throw std::runtime_error("[hpln schema] '" + path + "' is not a readable .hpln: " + herr);
      }
    }
  }

  hpln_bind_schema out;
  out.num_rows = header_schema.columns.empty() ? 0 : header_schema.columns.front().num_rows;
  out.names.reserve(header_schema.columns.size());
  out.types.reserve(header_schema.columns.size());
  for (std::size_t i = 0; i < header_schema.columns.size(); ++i) {
    auto const& c = header_schema.columns[i];
    out.names.push_back(c.name.empty() ? "column" + std::to_string(i) : c.name);
    if (i < declared.size() && declared[i].id() != duckdb::LogicalTypeId::SQLNULL) {
      out.types.push_back(declared[i]);
    } else {
      // Fall back to the cuDF physical type. Lossy by construction -- see 7.9 -- but a file
      // without declared types should still bind rather than refuse.
      out.types.push_back(duckdb_type_for_cudf(simpatico::tag_to_dtype(c.dtype_tag), c.scale));
    }
  }
  return out;
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
  simpatico::compressed_table ct;
  try {
    ct = simpatico::compress_with_plan(table, plan_dsl, stream, mr, column_names);
  } catch (std::exception const& e) {
    return std::string("[hpln export] compression failed: ") + e.what();
  }

  // Statistics come from the DECODED values, which is why they have to be computed here rather
  // than recovered later: once the file holds compressed bytes the bounds are gone until someone
  // decodes them, and the whole point of carrying them is to avoid that.
  std::vector<std::uint8_t> packed;
  if (group_rows > 0 && !column_types.empty()) {
    auto stats =
      scan_manager::compute_pinned_group_stats(table, column_types, group_rows, stream, mr);
    std::vector<scan_manager::chunk_group_stats> per_chunk;
    per_chunk.push_back(std::move(stats));
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
  return simpatico::write_compressed_table(ct, path, stream, extra);
}

}  // namespace sirius
