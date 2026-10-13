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
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
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
  /// What the file says its own segments hash to. Empty for a file written before the segment
  /// existed, which reads exactly as it always did -- nothing is verified, and nothing is
  /// refused for lacking a checksum.
  std::vector<simpatico::hpln_checksum_entry> checksums;
  /// False for a file written before the trailer existed, whose extent can only be found by
  /// parsing a speculative prefix. Such a file is always one chunk.
  bool located = false;
};

/// A located file plus every chunk's header: what bind parks in @ref hpln_metadata so a split's
/// read can skip both. Immutable once built.
struct hpln_ingest_layout {
  hpln_layout layout;
  std::vector<std::vector<std::uint8_t>> headers;
};

std::optional<simpatico::hpln_segment_ref> find_segment(hpln_layout const& layout,
                                                        simpatico::hpln_segment kind)
{
  for (auto const& sg : layout.segs) {
    if (sg.kind == kind) { return sg; }
  }
  return std::nullopt;
}

/// Name a segment as an error message should: `payload` alone is ambiguous in a file of chunks.
std::string segment_name(simpatico::hpln_segment kind, std::uint32_t index)
{
  switch (kind) {
    case simpatico::hpln_segment::header: return "header of chunk " + std::to_string(index);
    case simpatico::hpln_segment::payload: return "payload of chunk " + std::to_string(index);
    case simpatico::hpln_segment::zone_maps: return "zone_maps segment";
    case simpatico::hpln_segment::logical_types: return "logical_types segment";
    case simpatico::hpln_segment::chunk_directory: return "chunk_directory segment";
    case simpatico::hpln_segment::checksums: return "checksums segment";
  }
  return "segment kind " + std::to_string(static_cast<int>(kind));
}

/// Check @p crc against what the file recorded for (@p kind, @p index).
///
/// A region the checksum table does not cover passes: that is a file written before checksums
/// existed, and refusing it would be a version bump wearing another hat. A region it DOES cover
/// and disagrees with throws -- never a warning, never a repair. Wrong bytes that decode are the
/// failure this exists to convert into an error.
void verify_crc(hpln_layout const& layout,
                std::string const& path,
                simpatico::hpln_segment kind,
                std::uint32_t index,
                std::uint64_t bytes,
                std::uint32_t crc)
{
  for (auto const& e : layout.checksums) {
    if (e.kind != kind || e.index != index) { continue; }
    if (e.bytes != bytes) {
      throw std::runtime_error("[hpln] '" + path + "': " + segment_name(kind, index) + " is " +
                               std::to_string(bytes) + " bytes but the checksum table records " +
                               std::to_string(e.bytes) + "; the file is corrupt");
    }
    if (e.crc != crc) {
      throw std::runtime_error("[hpln] '" + path + "': " + segment_name(kind, index) +
                               " fails its CRC32C (" + std::to_string(crc) + " read, " +
                               std::to_string(e.crc) + " recorded); the file is corrupt");
    }
    return;
  }
}

void verify_bytes(hpln_layout const& layout,
                  std::string const& path,
                  simpatico::hpln_segment kind,
                  std::uint32_t index,
                  std::span<const std::uint8_t> bytes)
{
  if (layout.checksums.empty()) { return; }
  verify_crc(layout, path, kind, index, bytes.size(), simpatico::hpln_crc32c(bytes));
}

/// Read a metadata segment and check it against the file's own record of it.
std::vector<std::uint8_t> read_verified_segment(hpln_source& src,
                                                hpln_layout const& layout,
                                                std::string const& path,
                                                simpatico::hpln_segment_ref const& sg,
                                                char const* what)
{
  auto bytes = src.read_range(sg.offset, sg.bytes, what);
  verify_bytes(layout, path, sg.kind, 0, bytes);
  return bytes;
}

/// Locate the segments and resolve the chunk directory of an open file.
///
/// One tail read locates everything: the trailer names the postscript, the postscript names every
/// segment, and the chunk directory subdivides them. That is what makes the format cheap to open
/// over an object store, where a request costs more than the bytes it moves.
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

  // The trailing segments -- the directory and the checksum table -- are almost always inside the
  // tail that was just read, because they are written last. Slicing them out of it keeps opening
  // a file at ONE request, which is the property the whole trailer design exists for; a fresh
  // read_range per segment would quietly make it three.
  auto const tail_start = file_size - tail.size();
  auto const read_region =
    [&](std::uint64_t off, std::uint64_t bytes, char const* what) -> std::vector<std::uint8_t> {
    if (off >= tail_start && off + bytes <= file_size) {
      auto const* at = tail.data() + (off - tail_start);
      return std::vector<std::uint8_t>(at, at + bytes);
    }
    return src.read_range(off, bytes, what);
  };

  // A directory is authoritative when present. Without one the file predates chunking and is a
  // single chunk covering the whole header and payload segments -- unknown kinds being skipped is
  // exactly what makes that additive.
  // Before anything is trusted: the checksum table, so every segment read below is checked as it
  // is read. A file that carries none reads exactly as it did before checksums existed.
  if (auto const sums = find_segment(out, simpatico::hpln_segment::checksums);
      sums && sums->bytes > 0) {
    auto const serr = simpatico::unpack_hpln_checksums(
      read_region(sums->offset, sums->bytes, "checksum table"), out.checksums);
    if (!serr.empty()) { throw std::runtime_error("[hpln] '" + path + "': " + serr); }
  }

  if (auto const dir = find_segment(out, simpatico::hpln_segment::chunk_directory)) {
    auto const dir_bytes = read_region(dir->offset, dir->bytes, "chunk directory");
    verify_bytes(out, path, dir->kind, 0, dir_bytes);
    auto const derr = simpatico::unpack_hpln_chunk_directory(dir_bytes, out.chunks);
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
    // Checked before it is parsed: a corrupt header does not usually fail to parse, it parses
    // into offsets that read a neighbour's bytes as values.
    verify_bytes(
      layout, path, simpatico::hpln_segment::header, static_cast<std::uint32_t>(i), out_headers[i]);
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
    } else if (same_schema(out_schema, schema)) {
      // Nullability is per chunk and same_schema deliberately ignores it: a column with nulls in
      // only some chunks is one nullable column, not a schema disagreement. OR it in, so a caller
      // that sees "no nulls" can rely on it for the whole file rather than for chunk 0.
      for (std::size_t c = 0; c < out_schema.columns.size(); ++c) {
        out_schema.columns[c].has_nulls |= schema.columns[c].has_nulls;
        out_schema.columns[c].null_count += schema.columns[c].null_count;
      }
    } else {
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

/// Segments of @p blocks covering `[offset, offset + size)`, for one ranged read's destination.
///
/// The pinned allocation is a list of fixed-size blocks, so a byte range that is contiguous in the
/// FILE is not contiguous in memory; the reader takes a scatter list per extent, which is exactly
/// this.
void append_block_segments(
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& blocks,
  std::uint64_t offset,
  std::uint64_t size,
  hpln_extent& out)
{
  auto const block = blocks.block_size();
  while (size > 0) {
    auto const idx    = static_cast<std::size_t>(offset / block);
    auto const within = offset % block;
    auto const n      = std::min<std::uint64_t>(block - within, size);
    out.dst.push_back({reinterpret_cast<std::uint8_t*>(blocks.at(idx).data()) + within, n});
    offset += n;
    size -= n;
  }
}

/// Stage only @p columns of @p chunk, as a chunk that genuinely HAS only those columns.
///
/// The alternative -- stage everything and describe a subset -- is what pin_table used to refuse,
/// and rightly: every consumer indexes a chunk by the entry's column position, so a chunk holding
/// 16 columns behind an entry declaring 4 reads the wrong ones. So the header is rebuilt to
/// describe exactly @p columns and the payload is gathered to match, which makes the result an
/// ordinary narrower chunk that nothing downstream has to know about.
std::shared_ptr<pinned_compressed_blob> allocate_chunk_columns(
  std::vector<std::uint8_t> const& header,
  simpatico::hpln_chunk_ref const& chunk,
  std::span<const std::size_t> columns,
  cucascade::memory::memory_space& host_space,
  std::vector<hpln_extent>& out_extents,
  std::string const& path)
{
  std::vector<std::uint8_t> subset_header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  if (auto const err = simpatico::build_column_subset_header(
        header, columns, subset_header, gather, &payload_bytes);
      !err.empty()) {
    throw std::runtime_error("[hpln ingest] '" + path + "': " + err);
  }

  auto* host_mr = host_space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("[hpln ingest] target host space has no host memory resource");
  }
  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->header        = std::move(subset_header);
  auto payload_res    = host_space.make_reservation_or_null(payload_bytes);
  blob->payload       = host_mr->allocate_multiple_blocks(payload_bytes, payload_res.get());
  blob->payload_bytes = payload_bytes;
  if (payload_bytes == 0) { return blob; }

  // One extent per gather range. The ranges are already coalesced where a column's buffers are
  // adjacent in the file -- which the writer makes true for every column, since it assigns payload
  // offsets column by column -- so a subset of k columns is about k requests, not one per buffer.
  for (auto const& g : gather) {
    hpln_extent e;
    e.offset = chunk.payload_offset + g.src_offset;
    append_block_segments(*blob->payload, g.dst_offset, g.size, e);
    out_extents.push_back(std::move(e));
  }
  return blob;
}

/// Stage @p chunk narrowed to @p columns and/or @p decode_chunks, reading only the bytes that
/// survive BOTH.
///
/// The difference from doing this after the read -- which is what the scan used to do, and what a
/// host pin still does because its bytes are already in memory -- is the whole point: a chunk that
/// survives whole-chunk pruning but is 92% ruled out by its group bounds was still being read in
/// full. Over a network that is the request that costs.
///
/// Sizing a bitpack `packed` buffer's surviving chunks needs its chunk_count/chunk_bits, which
/// live in the PAYLOAD, so this reads those small buffers first and the narrowed payload second.
/// That is one extra round of small reads to avoid a large one.
///
/// Returns false (leaving @p out_blob null) when the narrowing is refused -- a column that is not
/// chunk-addressable -- so the caller can fall back to staging the chunk whole. Refusing is always
/// correct, just less selective.
bool allocate_chunk_narrowed(std::vector<std::uint8_t> const& header,
                             simpatico::hpln_chunk_ref const& chunk,
                             std::span<const std::size_t> columns,
                             std::span<const std::uint32_t> decode_chunks,
                             hpln_source* src,
                             hpln_io_policy const& policy,
                             cucascade::memory::memory_space* host_space,
                             std::vector<hpln_extent>& out_extents,
                             std::shared_ptr<pinned_compressed_blob>& out_blob,
                             hpln_sizing_cache* cache               = nullptr,
                             std::vector<hpln_file_range>* plan_out = nullptr,
                             planned_hpln_chunk* plan_chunk         = nullptr)
{
  out_blob.reset();

  // Step one: the columns. Produces a header describing only them and the ranges, in FILE space,
  // that back it.
  std::vector<std::uint8_t> header_after_columns;
  std::vector<simpatico::gather_range> gather_columns;
  std::span<const std::uint8_t> working_header = header;
  if (!columns.empty()) {
    if (auto const err = simpatico::build_column_subset_header(
          header, columns, header_after_columns, gather_columns, nullptr);
        !err.empty()) {
      return false;
    }
    working_header = header_after_columns;
  }

  // Step two: the rows, against whatever the first step left. Its ranges are in the FIRST step's
  // destination space, which is why they are composed rather than used directly.
  std::vector<std::uint8_t> narrowed_header;
  std::vector<simpatico::gather_range> gather_rows;
  std::uint64_t payload_bytes = 0;
  std::vector<std::uint8_t> subsetted;
  // Reads the small sizing buffers out of the file, translating from the first step's space when
  // there is one.
  simpatico::payload_host_read_fn read_payload =
    [&](std::uint64_t off, std::uint64_t size, void* dst) {
      if (size == 0) { return true; }
      std::vector<simpatico::gather_range> want{{off, size, 0}};
      auto const ranges =
        gather_columns.empty() ? want : simpatico::compose_gathers(gather_columns, want);
      if (ranges.empty()) { return false; }
      for (auto const& r : ranges) {
        auto const file_off = chunk.payload_offset + r.src_offset;
        if (cache != nullptr) {
          auto const hit = cache->bytes.find({file_off, r.size});
          if (hit != cache->bytes.end()) {
            std::memcpy(static_cast<std::uint8_t*>(dst) + r.dst_offset, hit->second.data(), r.size);
            continue;
          }
          if (cache->collect) {
            cache->missing.insert({file_off, r.size});
            return false;
          }
        }
        // Planning must never block on a read: a range nobody fetched ahead simply means the
        // chunk cannot be planned, and the caller does not hint it.
        if (src == nullptr || plan_out != nullptr || plan_chunk != nullptr) { return false; }
        auto const bytes = src->read_range(file_off, r.size, "sizing");
        if (bytes.size() != r.size) { return false; }
        std::memcpy(static_cast<std::uint8_t*>(dst) + r.dst_offset, bytes.data(), r.size);
      }
      return true;
    };
  auto const subset_err = simpatico::build_chunk_subset_header(working_header,
                                                               decode_chunks,
                                                               read_payload,
                                                               narrowed_header,
                                                               gather_rows,
                                                               /*max_gap_bytes=*/0,
                                                               &payload_bytes,
                                                               &subsetted);
  // A collecting pass only wanted the ranges asked for; its header is built on stand-in answers.
  if (cache != nullptr && cache->collect) { return false; }
  if (!subset_err.empty()) { return false; }
  // A column emitted WHOLE beside compacted ones has a different row count, and the two cannot be
  // one cudf::table. Every column here is one this scan reads, so any refusal refuses the chunk.
  for (auto const flag : subsetted) {
    if (flag == 0) { return false; }
  }

  auto const file_ranges = simpatico::compose_gathers(gather_columns, gather_rows);
  if (file_ranges.empty() && payload_bytes > 0) { return false; }

  if (plan_out != nullptr) {
    for (auto const& g : file_ranges) {
      plan_out->push_back({chunk.payload_offset + g.src_offset, g.size});
    }
    return true;
  }
  if (plan_chunk != nullptr) {
    plan_chunk->header        = std::move(narrowed_header);
    plan_chunk->payload_bytes = payload_bytes;
    plan_chunk->file_map.reserve(file_ranges.size());
    for (auto const& g : file_ranges) {
      plan_chunk->file_map.push_back({chunk.payload_offset + g.src_offset, g.size, g.dst_offset});
    }
    return true;
  }

  auto* host_mr = host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("[hpln ingest] target host space has no host memory resource");
  }
  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->header        = std::move(narrowed_header);
  auto payload_res    = host_space->make_reservation_or_null(payload_bytes);
  blob->payload       = host_mr->allocate_multiple_blocks(payload_bytes, payload_res.get());
  blob->payload_bytes = payload_bytes;
  for (auto const& g : file_ranges) {
    hpln_extent e;
    e.offset = chunk.payload_offset + g.src_offset;
    append_block_segments(*blob->payload, g.dst_offset, g.size, e);
    out_extents.push_back(std::move(e));
  }
  static_cast<void>(policy);
  out_blob = std::move(blob);
  return true;
}

/// Fetch every sizing buffer the narrowed chunks in @p chunk_ids will ask for, as planned batches
/// rather than one blocking read each. See @ref hpln_sizing_cache.
void prefetch_sizing_reads(std::vector<std::vector<std::uint8_t>> const& headers,
                           hpln_layout const& layout,
                           std::span<const std::size_t> chunk_ids,
                           std::span<const std::size_t> columns,
                           std::span<const std::vector<std::uint32_t>> decode_chunks,
                           hpln_source& src,
                           hpln_io_policy const& policy,
                           hpln_sizing_cache& cache)
{
  // A column with several bitpacked leaves asks for the later ones only after the earlier ones
  // resolve, so one pass is not enough: measured on TPC-H q3, stopping after one left the rest to
  // the blocking path and cost more than the extra pass (plan phase 17 ms -> 460 ms a split). One
  // round per level, so a handful at most; the cap only guards against a header that never stops
  // asking.
  constexpr int max_rounds = 8;
  cache.collect            = true;
  for (int round = 0; round < max_rounds; ++round) {
    auto const t_round = std::chrono::steady_clock::now();
    cache.missing.clear();
    std::set<std::size_t> seen;
    for (std::size_t at = 0; at < chunk_ids.size(); ++at) {
      auto const id = chunk_ids[at];
      // A chunk named twice is staged once, with its first occurrence's rows.
      if (id >= layout.chunks.size() || !seen.insert(id).second) { continue; }
      if (at >= decode_chunks.size() || decode_chunks[at].empty()) { continue; }
      std::vector<hpln_extent> unused;
      std::shared_ptr<pinned_compressed_blob> blob;
      allocate_chunk_narrowed(headers[id],
                              layout.chunks[id],
                              columns,
                              decode_chunks[at],
                              &src,
                              policy,
                              nullptr,
                              unused,
                              blob,
                              &cache);
    }
    auto const t_collected = std::chrono::steady_clock::now();
    if (cache.missing.empty()) {
      SIRIUS_LOG_DEBUG("[hpln sizing] round {}: collect {:.1f} ms, nothing missing",
                       round,
                       std::chrono::duration<double, std::milli>(t_collected - t_round).count());
      break;
    }
    std::uint64_t missing_bytes = 0;
    for (auto const& [off, size] : cache.missing) {
      missing_bytes += size;
    }
    std::vector<hpln_extent> extents;
    extents.reserve(cache.missing.size());
    for (auto const& [off, size] : cache.missing) {
      // std::map nodes do not move, so the pointer stays valid as the map grows.
      auto& buf = cache.bytes[{off, size}];
      buf.resize(size);
      hpln_extent e;
      e.offset = off;
      e.dst.push_back({buf.data(), size});
      extents.push_back(std::move(e));
    }
    auto const n_missing = extents.size();
    src.read_extents(std::move(extents), policy, "sizing");
    SIRIUS_LOG_DEBUG(
      "[hpln sizing] round {}: collect {:.1f} ms, read {} ranges / {} B in {:.1f} ms",
      round,
      std::chrono::duration<double, std::milli>(t_collected - t_round).count(),
      n_missing,
      missing_bytes,
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_collected)
        .count());
  }
  cache.collect = false;
  cache.missing.clear();
}

/// Open @p path for reading, through @p options's io_context when it has one.
std::unique_ptr<hpln_source> open_hpln(std::string const& path,
                                       char const* who,
                                       hpln_open_options const& options)
{
  return open_hpln_source(path, options.io_ctx, who, options.datasource);
}

/// CRC32C over a staged chunk payload, walking the pinned blocks in file order.
///
/// The payload is not contiguous in memory -- it lands in the host allocator's blocks -- but it
/// IS contiguous in the file, so the running checksum has to visit the blocks in exactly the
/// order allocate_chunk filled them.
std::uint32_t crc_of_staged_payload(pinned_compressed_blob const& blob)
{
  std::uint32_t crc = 0;
  auto const nb     = blob.payload_bytes;
  if (nb == 0 || !blob.payload) { return simpatico::hpln_crc32c({}, crc); }
  auto const block   = blob.payload->block_size();
  std::uint64_t done = 0;
  std::size_t idx    = 0;
  while (done < nb) {
    auto const n = std::min<std::uint64_t>(block, nb - done);
    crc =
      simpatico::hpln_crc32c({reinterpret_cast<std::uint8_t const*>(blob.payload->at(idx).data()),
                              static_cast<std::size_t>(n)},
                             crc);
    done += n;
    ++idx;
  }
  return crc;
}

/// Publish what the transport did, for a caller that asked.
void report(hpln_source const& src, hpln_open_options const& options)
{
  if (options.stats != nullptr) { *options.stats = src.stats(); }
}

}  // namespace

bool hpln_verify_payload_default()
{
  // Read once: this is consulted per open, and the answer cannot change within a process.
  static bool const enabled = [] {
    char const* v = std::getenv("SIRIUS_HPLN_VERIFY_PAYLOAD");
    return v != nullptr && *v != '\0' && *v != '0';
  }();
  return enabled;
}

std::vector<ingested_hpln_chunk> read_hpln_chunks_into_pinned(
  std::string const& path,
  cucascade::memory::memory_space& host_space,
  std::span<const std::size_t> chunk_ids,
  hpln_open_options const& options,
  std::span<const std::size_t> columns,
  std::span<const std::vector<std::uint32_t>> decode_chunks)
{
  using clock        = std::chrono::steady_clock;
  auto const t_enter = clock::now();
  auto const ms      = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto src          = open_hpln(path, "hpln ingest", options);
  auto const policy = options.policy.value_or(src->default_policy());
  // A bind of this file already located it and read every chunk's header; reuse that rather than
  // paying for both again on every split. Falls back to doing it here when no bind parked them
  // (a transport with no metadata store, or a file read without a prior bind).
  std::shared_ptr<hpln_ingest_layout const> cached_layout;
  if (auto const parked = src->metadata()) {
    if (auto const* hm = dynamic_cast<hpln_metadata const*>(parked.get())) {
      cached_layout = std::static_pointer_cast<hpln_ingest_layout const>(hm->ingest_layout());
    }
  }
  hpln_ingest_layout owned_layout;
  hpln_ingest_layout const* ingest_layout = cached_layout.get();
  if (ingest_layout == nullptr) {
    owned_layout.layout = locate_hpln(*src, path);
    if (!owned_layout.layout.located) {
      throw std::runtime_error("[hpln ingest] '" + path +
                               "' predates the trailer and has no chunk directory; it can only "
                               "be read as a single chunk");
    }
  }
  auto const t_located = clock::now();
  if (ingest_layout == nullptr) {
    simpatico::hpln_schema schema;
    describe_chunks(*src, path, owned_layout.layout, policy, owned_layout.headers, schema);
    ingest_layout = &owned_layout;
  }
  auto const& layout   = ingest_layout->layout;
  auto const& headers  = ingest_layout->headers;
  auto const t_headers = clock::now();

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
  std::unordered_map<std::size_t, bool> narrowed_ids;
  // A scan hands in the sizing buffers it already fetched for every split at once, and they are
  // used as they are -- this read inserts nothing, so tasks can share the cache unlocked. Without
  // one, fetch this read's own.
  hpln_sizing_cache local_sizing;
  hpln_sizing_cache* sizing = options.sizing.get();
  if (sizing == nullptr) {
    sizing = &local_sizing;
    if (std::ranges::any_of(decode_chunks, [](auto const& d) { return !d.empty(); })) {
      prefetch_sizing_reads(
        headers, layout, chunk_ids, columns, decode_chunks, *src, policy, *sizing);
    }
  }
  auto const t_sized = clock::now();
  std::size_t at     = 0;
  for (auto const id : chunk_ids) {
    if (id >= layout.chunks.size()) {
      throw std::runtime_error("[hpln ingest] '" + path + "': chunk " + std::to_string(id) +
                               " is out of range for a file with " +
                               std::to_string(layout.chunks.size()) + " chunks");
    }
    auto it = staged.find(id);
    if (it == staged.end()) {
      std::shared_ptr<pinned_compressed_blob> blob;
      bool narrowed = false;
      // Row narrowing is positional with chunk_ids, not with the file's chunks: it says what the
      // CALLER wants of this split's i-th chunk.
      auto const rows = at < decode_chunks.size()
                          ? std::span<const std::uint32_t>{decode_chunks[at]}
                          : std::span<const std::uint32_t>{};
      if (!rows.empty()) {
        narrowed = allocate_chunk_narrowed(headers[id],
                                           layout.chunks[id],
                                           columns,
                                           rows,
                                           src.get(),
                                           policy,
                                           &host_space,
                                           extents,
                                           blob,
                                           sizing);
      }
      if (!blob) {
        blob = columns.empty()
                 ? allocate_chunk(headers[id], layout.chunks[id], host_space, extents)
                 : allocate_chunk_columns(
                     headers[id], layout.chunks[id], columns, host_space, extents, path);
      }
      it               = staged.emplace(id, std::move(blob)).first;
      narrowed_ids[id] = narrowed;
    }
    // A narrowed blob holds only the surviving decode chunks' rows, so the row count has to
    // follow the bytes -- a consumer sizing anything by the chunk's own count would be wrong.
    std::int64_t rows_in_blob = layout.chunks[id].num_rows;
    if (narrowed_ids[id]) {
      constexpr std::size_t decode_chunk_rows = static_cast<std::size_t>(::codegen::kChunkSize);
      auto const chunk_rows = static_cast<std::size_t>(layout.chunks[id].num_rows);
      std::size_t kept      = 0;
      for (auto const d : decode_chunks[at]) {
        auto const begin = static_cast<std::size_t>(d) * decode_chunk_rows;
        if (begin >= chunk_rows) { continue; }
        kept += std::min(decode_chunk_rows, chunk_rows - begin);
      }
      rows_in_blob = static_cast<std::int64_t>(kept);
    }
    out.push_back({it->second, rows_in_blob, narrowed_ids[id]});
    ++at;
  }
  auto const t_planned = clock::now();
  src->read_extents(std::move(extents), policy, "chunk payload");
  auto const t_read = clock::now();
  // A narrowed read cannot be checksum-verified: the recorded CRC covers a chunk's WHOLE payload,
  // and this read deliberately did not fetch all of it.
  auto const any_narrowed =
    std::ranges::any_of(narrowed_ids, [](auto const& kv) { return kv.second; });
  if (options.verify_payload && columns.empty() && !any_narrowed && !layout.checksums.empty()) {
    // After the read, before the blob is handed to anyone: a caller that got a chunk back has
    // already been told it is intact.
    for (auto const& [id, blob] : staged) {
      verify_crc(layout,
                 path,
                 simpatico::hpln_segment::payload,
                 static_cast<std::uint32_t>(id),
                 blob->payload_bytes,
                 crc_of_staged_payload(*blob));
    }
  }
  // What one split's staging actually cost the storage, split from what it asked for. A .hpln
  // read is planned (coalesced, split, bridged) and re-locates the file every call, so "bytes the
  // payload needed" and "bytes the device served" are different numbers and only the second is
  // paid for.
  auto const st = src->stats();
  SIRIUS_LOG_DEBUG(
    "[hpln ingest] '{}' via {}: {} chunks of {}, {} requests, {} extents, {} B read for "
    "{} B wanted; ms: open+locate {:.1f}, headers {:.1f}, sizing {:.1f}, plan {:.1f}, payload "
    "{:.1f}, total {:.1f}",
    path,
    st.transport,
    chunk_ids.size(),
    layout.chunks.size(),
    st.requests,
    st.extents,
    st.bytes_read,
    st.bytes_wanted,
    ms(t_enter, t_located),
    ms(t_located, t_headers),
    ms(t_headers, t_sized),
    ms(t_sized, t_planned),
    ms(t_planned, t_read),
    ms(t_enter, clock::now()));
  report(*src, options);
  return out;
}

//===----------------------------------------------------------------------===//
// read planning
//===----------------------------------------------------------------------===//
struct hpln_read_planner::impl {
  std::string path;
  hpln_open_options options;
  std::vector<std::size_t> columns;
  std::unique_ptr<hpln_source> src;
  hpln_io_policy policy;
  std::shared_ptr<hpln_ingest_layout const> layout;
  std::shared_ptr<hpln_sizing_cache> sizing = std::make_shared<hpln_sizing_cache>();
  /// The file's shared sizing buffers, when a bind parked metadata to hold them.
  std::shared_ptr<hpln_sizing_store> store;
};

hpln_read_planner::hpln_read_planner(std::string path,
                                     hpln_open_options options,
                                     std::vector<std::size_t> columns)
  : _impl(std::make_unique<impl>())
{
  _impl->path    = std::move(path);
  _impl->options = std::move(options);
  _impl->columns = std::move(columns);
  _impl->src     = open_hpln(_impl->path, "hpln plan", _impl->options);
  _impl->policy  = _impl->options.policy.value_or(_impl->src->default_policy());
  // The layout a bind parked, when there is one -- the same reuse a split's read makes.
  if (auto const parked = _impl->src->metadata()) {
    if (auto const* hm = dynamic_cast<hpln_metadata const*>(parked.get())) {
      _impl->layout = std::static_pointer_cast<hpln_ingest_layout const>(hm->ingest_layout());
      _impl->store  = hm->sizing_store();
    }
  }
  if (!_impl->layout) {
    auto fresh    = std::make_shared<hpln_ingest_layout>();
    fresh->layout = locate_hpln(*_impl->src, _impl->path);
    if (!fresh->layout.located) {
      throw std::runtime_error("[hpln plan] '" + _impl->path +
                               "' predates the trailer and has no chunk directory");
    }
    simpatico::hpln_schema schema;
    describe_chunks(*_impl->src, _impl->path, fresh->layout, _impl->policy, fresh->headers, schema);
    _impl->layout = std::move(fresh);
  }
}

hpln_read_planner::~hpln_read_planner() = default;

std::shared_ptr<hpln_sizing_cache> const& hpln_read_planner::sizing() const noexcept
{
  return _impl->sizing;
}

void hpln_read_planner::prefetch_sizing(std::span<const std::size_t> chunk_ids,
                                        std::span<const std::vector<std::uint32_t>> decode_chunks)
{
  if (!std::ranges::any_of(decode_chunks, [](auto const& d) { return !d.empty(); })) { return; }
  auto const t0       = std::chrono::steady_clock::now();
  auto const& layout  = _impl->layout->layout;
  auto const& headers = _impl->layout->headers;

  // Every sizing range the narrowed chunks need, listed in ONE parse per chunk header -- not
  // discovered by building trial subset headers round after round (a column's chunk_bits used to
  // wait a whole round behind its chunk_count). Keyed exactly as the narrowing read looks them up:
  // the file range each payload range composes to through the column subset.
  std::set<std::pair<std::uint64_t, std::uint64_t>> wanted;
  std::set<std::size_t> seen;
  std::vector<simpatico::gather_range> listed;
  for (std::size_t at = 0; at < chunk_ids.size() && at < decode_chunks.size(); ++at) {
    auto const id = chunk_ids[at];
    if (id >= layout.chunks.size() || decode_chunks[at].empty() || !seen.insert(id).second) {
      continue;
    }
    auto const& chunk = layout.chunks[id];
    std::vector<std::uint8_t> header_after_columns;
    std::vector<simpatico::gather_range> gather_columns;
    std::span<const std::uint8_t> working = headers[id];
    if (!_impl->columns.empty()) {
      if (!simpatico::build_column_subset_header(
             headers[id], _impl->columns, header_after_columns, gather_columns, nullptr)
             .empty()) {
        continue;
      }
      working = header_after_columns;
    }
    if (!simpatico::list_chunk_sizing_ranges(working, listed).empty()) { continue; }
    for (auto const& r : listed) {
      std::vector<simpatico::gather_range> want{{r.src_offset, r.size, 0}};
      auto const pieces =
        gather_columns.empty() ? want : simpatico::compose_gathers(gather_columns, want);
      for (auto const& piece : pieces) {
        wanted.emplace(chunk.payload_offset + piece.src_offset, piece.size);
      }
    }
  }
  auto const t_listed = std::chrono::steady_clock::now();

  // What earlier scans of this file already fetched.
  auto const base = _impl->store ? _impl->store->snapshot() : nullptr;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> missing;
  for (auto const& key : wanted) {
    if (!base || !base->bytes.contains(key)) { missing.push_back(key); }
  }
  if (missing.empty() && base) {
    // Read-only from here on, as every reader of a sizing cache already treats it.
    _impl->sizing = std::const_pointer_cast<hpln_sizing_cache>(base);
  } else {
    auto next = std::make_shared<hpln_sizing_cache>();
    if (base) { next->bytes = base->bytes; }
    std::vector<hpln_extent> extents;
    extents.reserve(missing.size());
    std::uint64_t missing_bytes = 0;
    for (auto const& key : missing) {
      auto& buf = next->bytes[key];  // std::map nodes do not move as the map grows
      buf.resize(key.second);
      missing_bytes += key.second;
      hpln_extent e;
      e.offset = key.first;
      e.dst.push_back({buf.data(), key.second});
      extents.push_back(std::move(e));
    }
    if (!extents.empty()) { _impl->src->read_extents(std::move(extents), _impl->policy, "sizing"); }
    _impl->sizing =
      _impl->store
        ? std::const_pointer_cast<hpln_sizing_cache>(_impl->store->publish(std::move(next)))
        : std::move(next);
    SIRIUS_LOG_DEBUG(
      "[hpln sizing] '{}': fetched {} ranges / {} B", _impl->path, missing.size(), missing_bytes);
  }
  SIRIUS_LOG_DEBUG(
    "[hpln sizing] '{}': {} ranges wanted, {} cached; list {:.1f} ms, total {:.1f} ms",
    _impl->path,
    wanted.size(),
    wanted.size() - missing.size(),
    std::chrono::duration<double, std::milli>(t_listed - t0).count(),
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

std::vector<hpln_file_range> hpln_read_planner::ranges(
  std::size_t chunk_id, std::span<const std::uint32_t> decode_chunks) const
{
  auto const& layout  = _impl->layout->layout;
  auto const& headers = _impl->layout->headers;
  if (chunk_id >= layout.chunks.size()) { return {}; }
  auto const& chunk = layout.chunks[chunk_id];

  // The same three cases the read walks through, in the same order: narrowed to rows, narrowed to
  // columns, or the whole chunk.
  std::vector<hpln_file_range> out;
  if (!decode_chunks.empty()) {
    std::vector<hpln_extent> unused;
    std::shared_ptr<pinned_compressed_blob> blob;
    if (!allocate_chunk_narrowed(headers[chunk_id],
                                 chunk,
                                 _impl->columns,
                                 decode_chunks,
                                 nullptr,
                                 _impl->policy,
                                 nullptr,
                                 unused,
                                 blob,
                                 _impl->sizing.get(),
                                 &out)) {
      // Not knowable without a read (or refused): leave it unhinted rather than guess.
      return {};
    }
  } else if (_impl->columns.empty()) {
    if (chunk.payload_bytes > 0) { out.push_back({chunk.payload_offset, chunk.payload_bytes}); }
  } else {
    std::vector<std::uint8_t> subset_header;
    std::vector<simpatico::gather_range> gather;
    std::uint64_t payload_bytes = 0;
    if (!simpatico::build_column_subset_header(
           headers[chunk_id], _impl->columns, subset_header, gather, &payload_bytes)
           .empty()) {
      return {};
    }
    for (auto const& g : gather) {
      out.push_back({chunk.payload_offset + g.src_offset, g.size});
    }
  }

  std::ranges::sort(out, {}, &hpln_file_range::offset);
  std::vector<hpln_file_range> merged;
  for (auto const& r : out) {
    if (r.size == 0) { continue; }
    if (!merged.empty() && r.offset <= merged.back().offset + merged.back().size) {
      auto const end     = std::max(merged.back().offset + merged.back().size, r.offset + r.size);
      merged.back().size = end - merged.back().offset;
    } else {
      merged.push_back(r);
    }
  }
  return merged;
}

std::optional<planned_hpln_chunk> hpln_read_planner::plan_chunk(
  std::size_t chunk_id, std::span<const std::uint32_t> decode_chunks) const
{
  auto const& layout  = _impl->layout->layout;
  auto const& headers = _impl->layout->headers;
  if (chunk_id >= layout.chunks.size()) { return std::nullopt; }
  auto const& chunk = layout.chunks[chunk_id];

  // The same three cases the read walks through, in the same order.
  planned_hpln_chunk plan;
  plan.num_rows = chunk.num_rows;
  if (!decode_chunks.empty()) {
    std::vector<hpln_extent> unused;
    std::shared_ptr<pinned_compressed_blob> blob;
    if (!allocate_chunk_narrowed(headers[chunk_id],
                                 chunk,
                                 _impl->columns,
                                 decode_chunks,
                                 nullptr,
                                 _impl->policy,
                                 nullptr,
                                 unused,
                                 blob,
                                 _impl->sizing.get(),
                                 nullptr,
                                 &plan)) {
      return std::nullopt;
    }
    constexpr std::size_t decode_chunk_rows = static_cast<std::size_t>(::codegen::kChunkSize);
    auto const chunk_rows                   = static_cast<std::size_t>(chunk.num_rows);
    std::size_t kept                        = 0;
    for (auto const d : decode_chunks) {
      auto const begin = static_cast<std::size_t>(d) * decode_chunk_rows;
      if (begin >= chunk_rows) { continue; }
      kept += std::min(decode_chunk_rows, chunk_rows - begin);
    }
    plan.num_rows      = static_cast<std::int64_t>(kept);
    plan.rows_narrowed = true;
  } else if (_impl->columns.empty()) {
    plan.header        = headers[chunk_id];
    plan.payload_bytes = chunk.payload_bytes;
    if (chunk.payload_bytes > 0) {
      plan.file_map.push_back({chunk.payload_offset, chunk.payload_bytes, 0});
    }
  } else {
    std::vector<simpatico::gather_range> gather;
    if (!simpatico::build_column_subset_header(
           headers[chunk_id], _impl->columns, plan.header, gather, &plan.payload_bytes)
           .empty()) {
      return std::nullopt;
    }
    plan.file_map.reserve(gather.size());
    for (auto const& g : gather) {
      plan.file_map.push_back({chunk.payload_offset + g.src_offset, g.size, g.dst_offset});
    }
  }
  std::ranges::sort(plan.file_map, {}, &simpatico::gather_range::dst_offset);
  return plan;
}

ingested_hpln read_hpln_into_pinned(std::string const& path,
                                    cucascade::memory::memory_space& host_space,
                                    hpln_open_options const& options)
{
  auto src             = open_hpln(path, "hpln ingest", options);
  auto const policy    = options.policy.value_or(src->default_policy());
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
    describe_chunks(*src, path, layout, policy, headers, out.schema);
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
  src->read_extents(std::move(extents), policy, "chunk payload");
  if (options.verify_payload && !layout.checksums.empty()) {
    verify_crc(layout,
               path,
               simpatico::hpln_segment::payload,
               0,
               out.blob->payload_bytes,
               crc_of_staged_payload(*out.blob));
  }

  // Zone maps, if the file carries them. A file written before the segment existed, or one whose
  // segment does not decode, simply serves unpruned -- so this never fails the ingest.
  for (auto const& sg : layout.segs) {
    if (sg.bytes == 0) { continue; }
    // Unknown segment kinds are skipped, which is what makes them additive.
    if (sg.kind == simpatico::hpln_segment::zone_maps) {
      std::string zerr;
      out.group_bounds = scan_manager::group_bounds_arena::unpack(
        read_verified_segment(*src, layout, path, sg, "zone map segment"), &zerr);
      if (!zerr.empty()) {
        SIRIUS_LOG_WARN(
          "[hpln ingest] '{}': zone maps unreadable ({}); serving unpruned", path, zerr);
      }
    } else if (sg.kind == simpatico::hpln_segment::logical_types) {
      std::string terr;
      out.column_types = unpack_logical_types(
        read_verified_segment(*src, layout, path, sg, "logical types segment"), &terr);
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

/// Re-attach @p scale to a fixed-point @p dtype. A non-decimal type carries no scale and is
/// returned unchanged, so this is safe to apply to every column.
cudf::data_type dtype_with_scale(cudf::data_type dtype, std::int32_t scale)
{
  switch (dtype.id()) {
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return cudf::data_type{dtype.id(), scale};
    default: return dtype;
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

namespace {

/// The uncached parse. @ref read_hpln_schema and @ref read_hpln_schema_shared differ only in what
/// they do around it.
hpln_bind_schema parse_hpln_schema(std::string const& path,
                                   hpln_open_options const& options,
                                   hpln_source& src,
                                   std::shared_ptr<void const>* ingest_out = nullptr);

}  // namespace

std::shared_ptr<hpln_bind_schema const> read_hpln_schema_shared(std::string const& path,
                                                                hpln_open_options const& options)
{
  auto src = open_hpln(path, "hpln schema", options);
  if (auto cached = src->metadata()) {
    if (auto const* hm = dynamic_cast<hpln_metadata const*>(cached.get()); hm && hm->schema()) {
      return hm->schema();
    }
  }
  std::shared_ptr<void const> ingest;
  auto schema =
    std::make_shared<hpln_bind_schema const>(parse_hpln_schema(path, options, *src, &ingest));
  // Best effort: a transport with nowhere to park it simply re-parses next time, which is slower
  // and not wrong. Racing binds of the same file both parse and the last one wins -- the entries
  // are equal, so there is nothing to reconcile.
  src->store_metadata(std::make_shared<hpln_metadata>(schema, std::move(ingest)));
  return schema;
}

hpln_bind_schema read_hpln_schema(std::string const& path, hpln_open_options const& options)
{
  auto src = open_hpln(path, "hpln schema", options);
  return parse_hpln_schema(path, options, *src);
}

namespace {

hpln_bind_schema parse_hpln_schema(std::string const& path,
                                   hpln_open_options const& options,
                                   hpln_source& src,
                                   std::shared_ptr<void const>* ingest_out)
{
  // Reuses the ingest reader for locating and parsing, but stops before any payload is staged:
  // binding a query must not move data. Every chunk's header is parsed -- one sequential read of
  // the segregated metadata region -- both to sum the row counts the optimizer wants and to
  // refuse a file whose chunks disagree here rather than mid-scan.
  auto const file_size = src.size();
  auto layout          = locate_hpln(src, path);

  simpatico::hpln_schema header_schema;
  duckdb::vector<duckdb::LogicalType> declared;
  std::vector<std::int64_t> chunk_rows;
  scan_manager::group_bounds_arena bounds;

  if (layout.located) {
    std::vector<std::vector<std::uint8_t>> headers;
    describe_chunks(
      src, path, layout, options.policy.value_or(src.default_policy()), headers, header_schema);
    for (auto const& c : layout.chunks) {
      chunk_rows.push_back(c.num_rows);
    }
    if (auto const sg = find_segment(layout, simpatico::hpln_segment::logical_types);
        sg && sg->bytes > 0) {
      declared = unpack_logical_types(
        read_verified_segment(src, layout, path, *sg, "logical types segment"), nullptr);
    }
    // The zone maps are read HERE rather than at scan time because a bind is where a scan learns
    // what it may skip: the walk has to decide a chunk's fate before it emits a split for it, and
    // the bounds are one small segregated read whatever the file's size.
    if (auto const sg = find_segment(layout, simpatico::hpln_segment::zone_maps);
        sg && sg->bytes > 0) {
      std::string zerr;
      bounds = scan_manager::group_bounds_arena::unpack(
        read_verified_segment(src, layout, path, *sg, "zone map segment"), &zerr);
      if (!zerr.empty()) {
        SIRIUS_LOG_WARN(
          "[hpln schema] '{}': zone maps unreadable ({}); binding unpruned", path, zerr);
      }
    }
    if (ingest_out != nullptr) {
      *ingest_out =
        std::make_shared<hpln_ingest_layout const>(hpln_ingest_layout{layout, std::move(headers)});
    }
  } else {
    // Pre-trailer file: parse the header from the front, growing the prefix as needed.
    std::string herr;
    for (std::uint64_t want = kHeaderProbeBytes;; want *= 2) {
      auto const prefix = src.read_prefix(want, "header");
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
    // The tag alone does not carry a fixed-point column's scale -- the header stores it beside the
    // tag, and the decode rebuilds the column as `{id, scale}`. Reproduce that here: a carrier
    // that says DECIMAL64 scale 0 where the decode produces DECIMAL64 scale -2 is a different
    // cudf::data_type, and the plan generator's carrier check rejects the column outright.
    out.physical_types.push_back(dtype_with_scale(simpatico::tag_to_dtype(c.dtype_tag), c.scale));
    out.column_has_nulls.push_back(c.has_nulls);
    // The column headers and the zone-map segment are written together but are separate records,
    // and only the headers are authoritative about nulls. If they disagree, believe the headers:
    // an arena that claims a nullable column has no nulls would let a group holding a NULL be
    // pruned by a predicate the NULL cannot satisfy, and the row would silently disappear.
    if (c.has_nulls) { out.group_bounds.mark_column_nullable(i); }
    if (i < declared.size() && declared[i].id() != duckdb::LogicalTypeId::SQLNULL) {
      out.types.push_back(declared[i]);
    } else {
      // Fall back to the cuDF physical type. Lossy by construction -- see 7.9 -- but a file
      // without declared types should still bind rather than refuse.
      out.types.push_back(duckdb_type_for_cudf(simpatico::tag_to_dtype(c.dtype_tag), c.scale));
    }
  }
  report(src, options);
  return out;
}

}  // namespace

struct hpln_table_writer::impl {
  duckdb::vector<duckdb::LogicalType> column_types;
  std::vector<std::string> column_names;
  std::string plan_dsl;
  std::size_t group_rows;
  simpatico::hpln_stream_writer out;
  // Every chunk's per-group bounds. Retained rather than written per chunk because the arena is
  // one segment over the whole file, indexed by the same chunk id the directory uses. This is the
  // only thing that grows with the file: ~187 MB of host memory for SF1000 lineitem at G=8192,
  // against the ~214 GB of payload it describes.
  std::vector<scan_manager::chunk_group_stats> per_chunk;

  impl(std::string path,
       duckdb::vector<duckdb::LogicalType> types,
       std::vector<std::string> names,
       std::string dsl,
       std::size_t groups)
    : column_types(std::move(types)),
      column_names(std::move(names)),
      plan_dsl(std::move(dsl)),
      group_rows(groups),
      out(std::move(path))
  {
  }
};

hpln_table_writer::hpln_table_writer(std::string path,
                                     duckdb::vector<duckdb::LogicalType> column_types,
                                     std::vector<std::string> column_names,
                                     std::string plan_dsl,
                                     std::size_t group_rows)
  : _impl(std::make_unique<impl>(std::move(path),
                                 std::move(column_types),
                                 std::move(column_names),
                                 std::move(plan_dsl),
                                 group_rows))
{
}

hpln_table_writer::~hpln_table_writer() = default;

std::size_t hpln_table_writer::chunks_written() const noexcept
{
  return _impl->out.chunks_appended();
}

std::string hpln_table_writer::append(cudf::table_view const& table,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto& st = *_impl;
  simpatico::compressed_table compressed;
  try {
    compressed = simpatico::compress_with_plan(table, st.plan_dsl, stream, mr, st.column_names);
  } catch (std::exception const& e) {
    return std::string("[hpln export] compression failed: ") + e.what();
  }
  // Statistics come from the DECODED values, which is why they have to be computed here rather
  // than recovered later: once the file holds compressed bytes the bounds are gone until someone
  // decodes them, and the whole point of carrying them is to avoid that.
  if (st.group_rows > 0 && !st.column_types.empty()) {
    st.per_chunk.push_back(
      scan_manager::compute_pinned_group_stats(table, st.column_types, st.group_rows, stream, mr));
  }
  return st.out.append(compressed, stream);
}

std::string hpln_table_writer::finish()
{
  auto& st = *_impl;
  // Every chunk's bounds, in chunk order, in one arena -- so a pruning reader indexes it by the
  // same chunk id the directory uses.
  std::vector<std::uint8_t> packed;
  if (!st.per_chunk.empty()) {
    packed = scan_manager::group_bounds_arena::from_capture(st.column_types, st.per_chunk).pack();
  }

  // The engine's types always travel: without them a reader has only cuDF physical types and
  // cannot reconstruct DECIMAL precision or nullability, so the file would not be self-describing.
  auto const type_bytes = pack_logical_types(st.column_types);
  std::vector<simpatico::hpln_extra_segment> extra;
  if (!type_bytes.empty()) {
    extra.push_back({simpatico::hpln_segment::logical_types, type_bytes});
  }
  if (!packed.empty()) { extra.push_back({simpatico::hpln_segment::zone_maps, packed}); }
  return st.out.finish(extra);
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

  // The whole-table form is the streaming one with every chunk already in hand: one writer, one
  // layout, one set of rules about where the bounds come from.
  hpln_table_writer writer(path, column_types, column_names, plan_dsl, group_rows);
  for (auto const& t : tables) {
    if (auto err = writer.append(t, stream, mr); !err.empty()) { return err; }
  }
  return writer.finish();
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
