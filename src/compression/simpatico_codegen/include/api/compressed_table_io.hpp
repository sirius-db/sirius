// SPDX-License-Identifier: Apache-2.0
#pragma once

// C++-native .hpln writer/reader for compressed_table.
//
// The plan tree is serialized structurally (a node array); the read path
// rebuilds it directly and attaches each rep to its (node, slot). No DSL text
// is stored — render it from the tree on demand with render_plan_tree.
//
// File layout:
//   [Binary header]
//     "HPLN" (4 bytes)
//     version (uint8)
//     num_cols (uint16 LE)
//     per column:
//       name_len (uint16 LE) + name bytes   [0 = no name]
//       dtype_tag (uint8)                   [decoded column type]
//       num_rows (int64 LE)
//       num_nodes (uint16 LE)
//       per node:
//         op_len (uint16 LE) + op bytes
//         is_bitjoin (uint8); if 1: output_tag (uint8), num_inputs (uint16 LE),
//           per input: src_node (uint32 LE), channel (str16),
//                      has_range (uint8) [+ hi (uint32 LE), lo (uint32 LE)]
//         num_edges (uint16 LE), per edge: channel (str16) + child (uint32 LE)
//         num_outputs (uint16 LE), per output: name (str16)
//       num_leaves (uint16 LE)
//       per leaf:
//         node_index (uint32 LE)
//         slot (int32 LE)                   [-1 = node's own rep, else output port]
//         kind (uint8)                      [OpId]
//         type_tag (uint8)                  [decoded element type]
//         meta_kind (uint8)  0=none 1=alp_rd 2=ans 3=bitcomp 4=cascaded 5=snappy 6=lz4 7=deflate
//         meta bytes (variable per meta_kind; see push_meta)
//         num_bufs (uint8)
//         per buffer:
//           name (str16) + buf_type_tag (uint8) + size_bytes (uint64 LE) + payload_offset (uint64
//           LE)
//   [Payload]  — all buffer bytes concatenated in write order, copied D→H
//
// A file may hold N chunks, each an independently compressed table with its own header. The
// headers are then written contiguously ahead of every payload and located by a `chunk_directory`
// segment; see write_compressed_tables.

#include "api/simpatico_codegen.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simpatico {

// ─── Self-locating files: trailer + postscript ──────────────────────────────
//
// The original layout was [header][payload] with nothing saying where the header ends, so a
// reader could not locate anything without parsing the header, and could not parse the header
// without already holding it -- over a network that is a speculative read and a re-read.
//
// A file now ends with a fixed trailer pointing at a postscript, which is a locator table for
// segments:
//
//     [header][payload][segment...][postscript][trailer]
//
// Read the last few KB and you know where everything is, in one round trip. New segment kinds are
// ADDITIVE and a reader skips kinds it does not know, so null masks (in flight separately) or the
// group->byte table of CHUNK_SKIPPING_PLAN.md 6.1 can be added without another format break.
// Same shape as Vortex's postscript, for the same reasons.

enum class hpln_segment : std::uint16_t {
  header    = 1,  ///< the structural header parse_hpln_header/describe_... consume
  payload   = 2,  ///< every leaf buffer, concatenated, at the offsets the header declares
  zone_maps = 3,  ///< per-column, per-group min/max for pruning
  /// The ENGINE's logical schema, positional with the header's columns. The header carries cuDF
  /// physical types, which cannot express DECIMAL precision, nullability, or a timestamp's time
  /// zone -- a pin gets those from memory, a FILE has nowhere else to get them.
  logical_types = 4,
  /// Where each CHUNK's header and payload live. Written for every file, including a
  /// single-chunk one, so the read path has one shape. A reader that finds no directory treats
  /// the file as a single chunk spanning the `header` and `payload` segments -- which is what
  /// every file written before this segment existed is.
  chunk_directory = 5,
};

/// One chunk's extent within a multi-chunk .hpln.
///
/// A chunk is compressed independently, so it needs its own structural header -- but the headers
/// are written CONTIGUOUSLY, ahead of every payload (see CHUNK_SKIPPING_PLAN.md 7.5), so a reader
/// gets all of a file's metadata in one sequential read rather than a seek per chunk. That is
/// what makes the trailer's single tail read pay off over a network, where request count is what
/// costs (7.6). Offsets are absolute within the file.
struct hpln_chunk_ref {
  std::uint64_t header_offset  = 0;
  std::uint64_t header_bytes   = 0;
  std::uint64_t payload_offset = 0;
  std::uint64_t payload_bytes  = 0;
  /// Rows the chunk decodes to. An ingesting reader has to report split sizes before it decodes
  /// anything, and the row count is otherwise only reachable by parsing the chunk's header.
  std::int64_t num_rows = 0;
};

/// Serialize a chunk directory into the bytes a @c hpln_segment::chunk_directory segment carries.
[[nodiscard]] std::vector<std::uint8_t> pack_hpln_chunk_directory(
  std::span<const hpln_chunk_ref> chunks);

/// Inverse of @ref pack_hpln_chunk_directory. Returns an empty string on success.
///
/// Unlike zone maps, a malformed directory is NOT recoverable by serving unpruned: it says where
/// the data is, so a caller must fail rather than fall back.
[[nodiscard]] std::string unpack_hpln_chunk_directory(std::span<const std::uint8_t> bytes,
                                                      std::vector<hpln_chunk_ref>& out);

struct hpln_segment_ref {
  hpln_segment kind{};
  std::uint64_t offset = 0;
  std::uint64_t bytes  = 0;
};

/// An extra segment to append when writing. The bytes are opaque here; the KIND is what a reader
/// dispatches on.
struct hpln_extra_segment {
  hpln_segment kind{};
  std::span<const std::uint8_t> bytes;
};

inline constexpr std::size_t kHplnTrailerBytes  = 16;
inline constexpr std::uint16_t kHplnFileVersion = 7;

/// Locate a .hpln's segments from a TAIL of the file.
///
/// @p tail must be the last `tail.size()` bytes of a file of @p file_size bytes. When the tail is
/// long enough to hold the trailer and the postscript, @p out is filled and an empty string
/// returned. When it is not, @p need_bytes receives a sufficient tail length so a remote reader
/// can re-read exactly once instead of guessing -- which is the whole point of the trailer.
///
/// A file written before the trailer existed has no magic and is reported as such; the caller can
/// fall back to parsing [header][payload] from the front.
std::string read_hpln_postscript(std::span<const std::uint8_t> tail,
                                 std::uint64_t file_size,
                                 std::vector<hpln_segment_ref>& out,
                                 std::uint64_t* need_bytes = nullptr);

/// Write a compressed_table to *path*.
/// Returns an empty string on success; a human-readable error message otherwise.
std::string write_compressed_table(compressed_table const& table,
                                   std::string const& path,
                                   rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                   std::span<const hpln_extra_segment> extra = {});

/// Write @p tables to *path* as one multi-chunk file, in the order given.
///
/// The layout is SEGREGATED: every chunk's structural header first, contiguously, then every
/// chunk's payload, then the extra segments and a `chunk_directory` locating both regions per
/// chunk. Interleaving [hdr][pay][hdr][pay] would be simpler to write and would cost a seek per
/// chunk to read the metadata of, which is the access pattern this format exists to avoid.
///
/// The chunks are NOT checked against each other here -- this layer has no notion of a table
/// schema beyond one chunk. A reader assembling them into one table must validate that they
/// agree.
std::string write_compressed_tables(std::span<compressed_table const* const> tables,
                                    std::string const& path,
                                    rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                    std::span<const hpln_extra_segment> extra = {});

/// Read a compressed_table from *path*.
/// On failure writes an error to *error_out (if non-null) and returns an empty
/// compressed_table.
compressed_table read_compressed_table(
  std::string const& path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  std::string* error_out            = nullptr);

// ─── In-memory (pinned host) serialization ──────────────────────────────────
//
// Splits the .hpln stream into its two natural regions so the caller can keep
// the (large) payload in its own store — e.g. cuCascade pinned host memory —
// while the (small) structural header stays a flat byte vector that can be re-parsed
// quickly during decompression.

/// One payload buffer to be staged out of device memory by the caller.
/// `device_ptr` is borrowed from the source compressed_table and stays valid
/// only until that table is destroyed.
struct payload_buffer_ref {
  std::uint64_t offset     = 0;        ///< byte offset assigned within the payload region
  const void* device_ptr   = nullptr;  ///< source device bytes
  std::uint64_t size_bytes = 0;
  /// Bytes the leaf column occupies once reconstructed (decoded element count times
  /// element width, plus bitpack gather slop) — always >= size_bytes. A caller that
  /// places reconstructed leaves in its own contiguous slab must reserve this many
  /// bytes per leaf, not size_bytes: a decode kernel reads/writes the full column,
  /// so a slice sized only to the compressed bytes would run past its end.
  std::uint64_t alloc_bytes = 0;
  /// Index of the table column this buffer belongs to; summing `size_bytes` over
  /// the buffers sharing an index gives that column's compressed footprint.
  std::uint64_t column_index = 0;
};

/// Build the .hpln header for @p table into @p out_header and enumerate every
/// payload buffer (in header order, matching describe()) into @p out_buffers with
/// dense byte offsets; @p out_payload_bytes receives the total payload size. No
/// bytes are copied — the caller stages each buffer from its `device_ptr` into
/// its own payload store, then reconstructs later via
/// read_compressed_table_from_memory. Returns an empty string on success.
std::string build_compressed_table_header(
  compressed_table const& table,
  std::vector<std::uint8_t>& out_header,
  std::vector<payload_buffer_ref>& out_buffers,
  std::uint64_t& out_payload_bytes,
  rmm::cuda_stream_view stream = cudf::get_default_stream());

/// Copies @p size bytes of the external payload at logical @p offset into the
/// pre-allocated device buffer @p dst_device, enqueued on @p stream.
using payload_fetch_fn = std::function<void(
  std::uint64_t offset, std::size_t size, void* dst_device, rmm::cuda_stream_view stream)>;

/// Reconstruct a compressed_table from a header produced by
/// build_compressed_table_header plus a payload accessor. Re-parses the plan
/// tree from @p header and pulls each leaf buffer's bytes into device memory via
/// @p fetch. On failure writes an error to @p error_out (if non-null) and returns
/// an empty compressed_table.
/// @p leaf_mr, when set, allocates the enumerated leaf buffers (the columns whose
/// bytes @p fetch fills) — letting the caller place them in a dedicated arena/slab
/// (e.g. one contiguous device buffer per pinned chunk). Codec decode scratch is
/// always allocated from @p mr, so it never disturbs leaf placement. Defaults to
/// @p mr (leaves and scratch share one resource, the original behavior).
compressed_table read_compressed_table_from_memory(
  std::span<const std::uint8_t> header,
  payload_fetch_fn const& fetch,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  std::string* error_out            = nullptr,
  std::optional<rmm::device_async_resource_ref> leaf_mr = std::nullopt);

/// Like @ref read_compressed_table_from_memory but reconstructs only the columns
/// in @p selected_columns (indices into the full table's column order, in the
/// order the caller wants them). Only those columns' payload buffers are fetched,
/// so serving a projection of a wide pin does not pull every column to the GPU.
/// Out-of-range indices write an error to @p error_out and return an empty table.
compressed_table read_compressed_table_subset_from_memory(
  std::span<const std::uint8_t> header,
  payload_fetch_fn const& fetch,
  std::span<const std::size_t> selected_columns,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  std::string* error_out            = nullptr);

// ─── Reading a header without reading the data ──────────────────────────────
//
// An INGEST path needs the schema and the byte budget before it moves any payload: which columns
// exist, how many rows, how much space to allocate. read_compressed_table() cannot answer that --
// it reads the WHOLE file into host memory and reconstructs a device-resident table, which is the
// opposite of what a reader wants when the file is large or remote.

/// One column as the header describes it. No payload is touched and no GPU is involved.
struct hpln_column_desc {
  std::string name;
  std::uint8_t dtype_tag         = 0;  ///< decoded column type; see tag_to_dtype()
  std::int32_t scale             = 0;  ///< decimal scale, 0 otherwise
  std::int64_t num_rows          = 0;
  std::uint64_t compressed_bytes = 0;  ///< sum of this column's leaf buffers
};

struct hpln_schema {
  std::vector<hpln_column_desc> columns;
  /// Bytes the structural header occupies. In a .hpln FILE the payload starts here — which a
  /// reader can only discover by parsing, since the format carries no length prefix or footer
  /// (see CHUNK_SKIPPING_PLAN.md 7.5). A reader over a network therefore has to read a
  /// speculative prefix and re-read if it was too short.
  std::uint64_t header_bytes = 0;
  /// End of the payload region, i.e. max(payload_offset + size_bytes) over every buffer.
  std::uint64_t payload_bytes = 0;
};

/// Parse ONLY the structural header of @p header into @p out.
///
/// @p header may be a prefix of a larger file or buffer; everything past the header is ignored.
/// Returns an empty string on success, or a human-readable error -- including the truncation case,
/// which is how a caller learns its speculative prefix was too short.
std::string describe_compressed_table_header(std::span<const std::uint8_t> header,
                                             hpln_schema& out);

// ─── Serving a subset of a column's 1024-row chunks ─────────────────────────
//
// A compressed column can be served as a compacted table containing only some of its 1024-row
// chunks, because bitpack's `bp_offsets` -- where each chunk's bits start inside `packed` -- is
// not stored: it is scanned at decode time from the `chunk_count`/`chunk_bits` that were loaded
// (src/bridge/offsets_cumsum.cu). Hand the reader a header whose declared sizes are the compacted
// ones, plus a fetch that yields the surviving chunks' bytes concatenated in order, and an
// ordinary unmodified full decode produces exactly the surviving rows. Nothing on the read side
// changes.

/// One contiguous copy from the original payload into the compacted payload.
struct gather_range {
  std::uint64_t src_offset = 0;  ///< offset in the ORIGINAL payload
  std::uint64_t size       = 0;
  std::uint64_t dst_offset = 0;  ///< offset in the COMPACTED payload the new header describes
};

/// Reads @p size bytes of the original payload at @p offset into host memory at @p dst.
/// Returns false if the range cannot be served. Used only to read the small per-chunk metadata
/// buffers that size a bitpack `packed` buffer; on a host pin this is a memcpy.
using payload_host_read_fn =
  std::function<bool(std::uint64_t offset, std::uint64_t size, void* dst)>;

/// Synthesize a header describing only @p surviving_chunks of every column of @p header, together
/// with the ranges to gather from the original payload into the compacted one.
///
/// @p surviving_chunks are batch-local 1024-row chunk ids, strictly ascending. The result is fed
/// to read_compressed_table_from_memory unchanged, with a @ref payload_fetch_fn backed by
/// @p out_gather; the reader needs no knowledge that a subset is in play.
///
/// A column that cannot be subsetted -- a whole_column buffer such as a snappy root, an
/// unclassified operator, a nested node whose length differs from the column's, or missing sizing
/// metadata -- is emitted WHOLE (all its buffers at full size). Fetching whole is always correct
/// for that COLUMN, so this never fails for a well-formed table; an error string is returned only
/// for genuinely malformed input.
///
/// It is not automatically correct for the TABLE, and a caller must not ignore it: a subsetted
/// column and a whole one have different row counts, and assembling them into one cudf::table
/// throws (or, in a consumer that does not check, silently misaligns rows). @p
/// out_column_subsetted reports the outcome per column, in header column order, so a caller can
/// fall back to fetching the whole chunk when any column it actually reads was emitted whole.
///
/// @p read_payload supplies the values of a bitpack leaf's chunk_count/chunk_bits, which live in
/// the payload and are the only way to size its `packed` chunks. A null or failing reader demotes
/// that column to whole rather than failing.
/// @p max_gap_bytes bridges small holes between kept ranges into one range (see
/// simpatico::append_coalesced): moving a few pruned bytes beats paying for another request.
/// @p out_payload_bytes, when non-null, receives the compacted payload's total size, which can
/// exceed the gathered bytes (a bitpack `packed` buffer's decode guard words).
/// @p out_column_subsetted, when non-null, receives one entry per column of @p header: 1 when the
/// column was compacted to the surviving chunks, 0 when it was emitted whole.
std::string build_chunk_subset_header(std::span<const std::uint8_t> header,
                                      std::span<const std::uint32_t> surviving_chunks,
                                      payload_host_read_fn const& read_payload,
                                      std::vector<std::uint8_t>& out_header,
                                      std::vector<gather_range>& out_gather,
                                      std::uint64_t max_gap_bytes                     = 0,
                                      std::uint64_t* out_payload_bytes                = nullptr,
                                      std::vector<std::uint8_t>* out_column_subsetted = nullptr);

}  // namespace simpatico
