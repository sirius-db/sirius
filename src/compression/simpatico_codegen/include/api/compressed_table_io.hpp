// SPDX-License-Identifier: Apache-2.0
#pragma once

// C++-native .hpln v9 writer/reader for compressed_table.
//
// The plan tree is serialized structurally (a node array); the read path
// rebuilds it directly and attaches each rep to its (node, slot). No DSL text
// is stored — render it from the tree on demand with render_plan_tree.
//
// File layout (v9):
//   [Binary header]
//     "HPLN" (4 bytes)
//     version = 9 (uint8)
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

#include "api/simpatico_codegen.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace simpatico {

/// Write a compressed_table to *path*.
/// Returns an empty string on success; a human-readable error message otherwise.
std::string write_compressed_table(compressed_table const& table,
                                   std::string const& path,
                                   rmm::cuda_stream_view stream = cudf::get_default_stream());

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
// Splits the .hpln v9 stream into its two natural regions so the caller can keep
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
};

/// Build the .hpln v9 header for @p table into @p out_header and enumerate every
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
compressed_table read_compressed_table_from_memory(
  std::span<const std::uint8_t> header,
  payload_fetch_fn const& fetch,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  std::string* error_out            = nullptr);

}  // namespace simpatico
