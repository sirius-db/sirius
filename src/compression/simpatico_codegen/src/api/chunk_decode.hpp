// SPDX-License-Identifier: Apache-2.0
#pragma once

// Chunk-span decode of one compressed column straight from its .hpln payload.
//
// The whole-table read path (read_compressed_table_from_memory + decompress)
// rebuilds every leaf on the device and decodes all rows. This API instead
// decodes a contiguous span of 1024-row chunks of ONE column into caller-owned
// destination buffers, fetching only the payload bytes that span needs. It is
// what a row-range reader over a compressed batch is built on.
//
// Usage:
//   auto desc   = std::make_shared<const hpln_description>(describe_hpln_header(header));
//   compressed_column_cursor cursor{desc, column, payload, stream, mr};
//   if (cursor.supported()) decompress_column_chunks(cursor, c0, c1, skip, take, dst, stream);
//
// The .hpln header layout is documented at the top of api/compressed_table_io.hpp.

#include "codegen/plan/leaf_desc.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simpatico {

/// Rows per chunk of the chunked encodings (bitpack, delta, rle, ...). Equals codegen::kChunkSize.
inline constexpr std::int64_t kChunkRows = 1024;

// ─── Parsed .hpln header ────────────────────────────────────────────────────

/// One payload buffer of a leaf: where its bytes live within the payload region.
struct hpln_buffer_range {
  std::string name;                 ///< channel name, e.g. "chunk_min", "packed"
  std::uint8_t type_tag{0};         ///< element type tag of the buffer (see kTypeTags)
  std::uint64_t size{0};            ///< size in bytes
  std::uint64_t payload_offset{0};  ///< byte offset from the start of the payload region
};

/// One compressed leaf (one rep slot of the plan tree); mirrors leaf_desc without device pointers.
struct hpln_leaf_description {
  std::uint32_t node_index{0};      ///< owning node, index into hpln_column_description::nodes
  std::int32_t slot{kSelfRepSlot};  ///< kSelfRepSlot = the node's own rep, else an output port
  OpId kind{OpId::Unknown};
  std::uint8_t type_tag{0};  ///< decoded element type tag
  leaf_meta_v meta{leaf_meta::none{}};
  std::vector<hpln_buffer_range> buffers;  ///< in the rep's named_channels() order
};

/// One plan-tree node. Bitjoin parameters (inputs, bit ranges) are not surfaced: a column whose
/// plan contains a bitjoin is not chunk-decodable.
struct hpln_node_description {
  std::string op;
  bool is_bitjoin{false};
  std::vector<std::pair<std::string, std::uint32_t>> edges;  ///< (channel, child node index)
  std::vector<std::string> outputs;                          ///< output port names
};

enum class hpln_validity : std::uint8_t { all_valid = 0, all_null = 1, mask = 2 };

/// One column of the compressed table.
struct hpln_column_description {
  std::optional<std::string> name;
  cudf::data_type dtype{};  ///< decoded column type
  std::int64_t num_rows{0};
  hpln_validity validity{hpln_validity::all_valid};
  std::int64_t null_count{0};                ///< 0 when validity == all_valid
  std::uint64_t validity_size{0};            ///< mask bytes; 0 unless validity == mask
  std::uint64_t validity_payload_offset{0};  ///< mask offset from the start of the payload region
  std::vector<hpln_node_description> nodes;  ///< nodes[0] is the plan root
  std::vector<hpln_leaf_description> leaves;
};

/// A parsed .hpln structural header.
struct hpln_description {
  std::uint8_t version{0};
  std::vector<hpln_column_description> columns;
  /// Size of the payload region. For a parsed header: the sum of all buffer sizes, validity masks
  /// included. For a relocated device-batch description: the extent from the lowest buffer to the
  /// end of the last, gaps included.
  std::uint64_t payload_bytes{0};
};

/// Parses a .hpln structural header (no payload bytes needed).
/// @throws std::runtime_error on malformed input.
hpln_description describe_hpln_header(std::span<const std::byte> header);

// ─── Payload access ─────────────────────────────────────────────────────────

/// Where the compressed payload bytes live. Implementations: device-resident (zero copy), pinned
/// host blocks, disk file (Sirius provides these).
class payload_source {
 public:
  virtual ~payload_source() = default;

  /// Stream-ordered copy of payload bytes [offset, offset + size) into device memory at dst.
  virtual void read_to_device(std::uint64_t offset,
                              std::uint64_t size,
                              void* dst,
                              rmm::cuda_stream_view stream) = 0;

  /// Base device pointer of the payload if it is already device resident (offsets added by the
  /// caller), else nullptr.
  [[nodiscard]] virtual const std::byte* device_base() const noexcept { return nullptr; }
};

// ─── Destination ────────────────────────────────────────────────────────────

/// Destination buffers of one column. Same layout semantics as cucascade::column_destination:
/// rows are appended at dst_row_offset; STRING offsets are rebased to dst_byte_offset and
/// take + 1 of them are written; null bits are written at bit dst_row_offset + i and every other
/// bit of the mask is kept.
struct column_write_target {
  void* data{nullptr};                     ///< fixed-width values, or STRING chars
  std::size_t data_capacity{0};            ///< bytes available at data
  cudf::bitmask_type* null_mask{nullptr};  ///< nullptr ⇒ the decoded range must have no nulls
  void* offsets{nullptr};                  ///< STRING offsets; nullptr for fixed-width columns
  std::uint8_t offset_width{4};            ///< bytes per destination offset: 4 or 8
  cudf::size_type dst_row_offset{0};
  std::size_t dst_byte_offset{0};  ///< STRING: chars already written at data
};

/// Thrown when a column (or a requested range of it) cannot be chunk-decoded; the caller falls
/// back to a whole-table decode.
class unsupported_chunk_decode : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ─── Cursor + decode ────────────────────────────────────────────────────────

/// Per-(caller, batch, column) decode state: chunk-offset prefixes, dictionary keys, parsed ANS
/// frame headers, grow-only scratch. Payload bytes staged for a decode call are freed when the
/// call returns (dictionary keys excepted: they stay staged for the cursor's life). Stored values
/// and chars are read straight into the destination and ANS chars pass through about 1 MiB at a
/// time, so neither is staged whole. Not thread-safe. One stream and one memory resource
/// for the cursor's life: its device buffers are allocated, used and freed on the stream of its
/// decode calls, so a caller that changes stream (or resource) drops the cursor and builds a new
/// one, as the Sirius row-range reader does.
class compressed_column_cursor {
 public:
  /// @param description Parsed header of the batch; shared with the caller.
  /// @param column      Index into description->columns.
  /// @param payload     Source of the batch's payload bytes.
  /// @param stream      Unused: nothing is staged at construction; decode calls take their own.
  /// @param mr          Allocates the staged bytes and the decoder state; must outlive the
  ///                    cursor.
  compressed_column_cursor(std::shared_ptr<const hpln_description> description,
                           std::size_t column,
                           std::shared_ptr<payload_source> payload,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);
  ~compressed_column_cursor();
  compressed_column_cursor(compressed_column_cursor&&) noexcept;
  compressed_column_cursor& operator=(compressed_column_cursor&&) noexcept;

  /// false ⇒ decompress_column_chunks throws unsupported_chunk_decode for this column.
  [[nodiscard]] bool supported() const noexcept;
  [[nodiscard]] const hpln_column_description& column() const noexcept;
  /// Device bytes currently held by the cursor (staged payload + decoder state).
  [[nodiscard]] std::size_t device_bytes_held() const noexcept;

  struct impl;

 private:
  std::unique_ptr<impl> _impl;
  friend cudf::size_type decompress_column_chunks(compressed_column_cursor&,
                                                  cudf::size_type,
                                                  cudf::size_type,
                                                  cudf::size_type,
                                                  cudf::size_type,
                                                  column_write_target const&,
                                                  rmm::cuda_stream_view);
};

/// Decodes chunks [chunk_begin, chunk_end) of the cursor's column, skipping skip_head rows of the
/// first chunk and writing at most `take` rows into dst (values/offsets/chars and validity).
/// Returns rows written: == take for fixed width; may be < take for STRING when the chars do not
/// fit dst.data_capacity - dst.dst_byte_offset (0 if even one row does not fit).
/// Requires chunk_begin * kChunkRows + skip_head + take <= min(chunk_end * kChunkRows, num_rows).
/// @throws unsupported_chunk_decode if !cursor.supported(), or if the range has nulls and
///         dst.null_mask == nullptr.
cudf::size_type decompress_column_chunks(compressed_column_cursor& cursor,
                                         cudf::size_type chunk_begin,
                                         cudf::size_type chunk_end,
                                         cudf::size_type skip_head,
                                         cudf::size_type take,
                                         column_write_target const& dst,
                                         rmm::cuda_stream_view stream);

// ─── Direct leaf reads (for the chunk decoders; not part of the public API) ─

namespace chunk {

class leaf_access;

/// Device pointer to bytes [offset, offset + size) of buffer `buffer` of leaf `leaf` when they
/// are served without staging: in place for the cursor's access over a device-resident payload,
/// and whatever device_bytes returns for any other leaf_access. nullptr when the cursor's access
/// would have to stage them (the caller then reads what it needs with read_leaf_bytes), and for
/// size == 0.
const std::byte* unstaged_leaf_bytes(leaf_access& leaves,
                                     std::size_t leaf,
                                     std::size_t buffer,
                                     std::uint64_t offset,
                                     std::uint64_t size,
                                     rmm::cuda_stream_view stream);

/// Stream-ordered copy of bytes [offset, offset + size) of buffer `buffer` of leaf `leaf` into
/// device memory at dst. The cursor's access reads them from its payload source straight into
/// dst (device to device for a device-resident payload or bytes already staged), so no staging
/// buffer is allocated; any other leaf_access copies from its device_bytes.
void read_leaf_bytes(leaf_access& leaves,
                     std::size_t leaf,
                     std::size_t buffer,
                     std::uint64_t offset,
                     std::uint64_t size,
                     void* dst,
                     rmm::cuda_stream_view stream);

}  // namespace chunk

}  // namespace simpatico
