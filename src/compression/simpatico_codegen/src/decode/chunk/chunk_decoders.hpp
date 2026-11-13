// SPDX-License-Identifier: Apache-2.0
#pragma once

// Internal interface between the chunk-decode cursor (api/chunk_decode.cpp) and the
// per-encoding chunk decoders under decode/chunk/. Not part of the public API.

#include "api/chunk_decode.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace simpatico::chunk {

/// One column's compressed bytes, staged on demand. What a decode call stages is freed when that
/// call returns; the buffers of a dictionary node's own leaves (the keys) are the exception and
/// stay staged for the cursor's life. A device-resident payload is served in place.
class leaf_access {
 public:
  virtual ~leaf_access() = default;

  [[nodiscard]] virtual const hpln_column_description& column() const noexcept = 0;

  /// Device pointer to bytes [offset, offset + size) of buffer `buffer` of leaf `leaf`; valid
  /// until the current decode call returns (until the cursor dies for a dictionary node's own
  /// leaf buffers).
  virtual const std::byte* device_bytes(std::size_t leaf,
                                        std::size_t buffer,
                                        std::uint64_t offset,
                                        std::uint64_t size,
                                        rmm::cuda_stream_view stream) = 0;

  /// Synchronous host copy of the same bytes (small per-chunk metadata only). Waits for the
  /// stream and stages nothing: keep it off the per-call path.
  virtual void host_bytes(std::size_t leaf,
                          std::size_t buffer,
                          std::uint64_t offset,
                          std::uint64_t size,
                          void* dst,
                          rmm::cuda_stream_view stream) = 0;

  /// Device pointer to validity-mask bytes [offset, offset + size).
  virtual const std::byte* validity_bytes(std::uint64_t offset,
                                          std::uint64_t size,
                                          rmm::cuda_stream_view stream) = 0;

  /// Resource for decoder state and scratch.
  [[nodiscard]] virtual rmm::device_async_resource_ref mr() const noexcept = 0;
};

/// The rows one decode call covers: chunks [chunk_begin, chunk_end), minus the first skip_head
/// rows of chunk_begin, at most `take` rows.
struct chunk_span {
  std::int64_t chunk_begin{0};
  std::int64_t chunk_end{0};
  std::int64_t skip_head{0};
  std::int64_t take{0};

  /// Column row index of the first row decoded.
  [[nodiscard]] std::int64_t first_row() const noexcept
  {
    return chunk_begin * kChunkRows + skip_head;
  }
};

/// Decoder of one column's value plan. Stateful across calls (prefixes, keys, ANS headers).
/// Writes values / offsets / chars only — never validity. Returns rows written.
class chunk_decoder {
 public:
  virtual ~chunk_decoder() = default;

  virtual std::int64_t decode(leaf_access& leaves,
                              chunk_span const& span,
                              column_write_target const& dst,
                              rmm::cuda_stream_view stream) = 0;

  /// Device bytes of decoder-owned state, the state of the primitives it uses included (counted
  /// in the cursor's device_bytes_held()).
  [[nodiscard]] virtual std::size_t state_bytes() const noexcept = 0;
};

/// The shape of a column's value plan, as far as chunk decode distinguishes them.
enum class value_plan : std::uint8_t {
  bitpack,
  delta_bitpack,
  delta_rle_bitpack,
  dictionary,
  str_split,
  stored,
  unsupported
};

/// Classifies a column's plan tree; `unsupported` for anything no decoder below handles.
value_plan classify_value_plan(hpln_column_description const& column);

std::unique_ptr<chunk_decoder> make_bitpack_decoder(hpln_column_description const& column);
std::unique_ptr<chunk_decoder> make_delta_decoder(hpln_column_description const& column,
                                                  bool with_rle);
std::unique_ptr<chunk_decoder> make_dictionary_decoder(hpln_column_description const& column);
std::unique_ptr<chunk_decoder> make_str_split_decoder(hpln_column_description const& column);
std::unique_ptr<chunk_decoder> make_stored_decoder(hpln_column_description const& column);

/// Writes validity bits of rows [row_begin, row_begin + rows) to dst.null_mask at
/// dst.dst_row_offset; all_valid ⇒ 1s; all_null ⇒ 0s.
/// @throws unsupported_chunk_decode if dst.null_mask == nullptr and the range has nulls.
void write_validity(leaf_access& leaves,
                    std::int64_t row_begin,
                    std::int64_t rows,
                    column_write_target const& dst,
                    rmm::cuda_stream_view stream);

// ─── Plan-tree lookups ──────────────────────────────────────────────────────

/// A buffer of the column: indices into column.leaves and leaf.buffers.
struct buffer_ref {
  std::size_t leaf{0};
  std::size_t buffer{0};
};

/// The four buffers of one bitpack node.
struct bitpack_locator {
  buffer_ref chunk_min, chunk_count, chunk_bits, packed;
  cudf::data_type value_type{};  ///< the leaf's decoded element type
};

/// Index of the node reached from node `from` via edge `channel`; an empty `channel` names the
/// plan root (node 0) and ignores `from`. nullopt if there is no such edge or node. The caller
/// checks the returned node's op.
[[nodiscard]] inline std::optional<std::uint32_t> find_child_node(
  hpln_column_description const& column, std::uint32_t from, std::string_view channel)
{
  if (channel.empty()) {
    return column.nodes.empty() ? std::nullopt : std::optional<std::uint32_t>{0};
  }
  if (from >= column.nodes.size()) { return std::nullopt; }
  for (auto const& [edge_channel, child] : column.nodes[from].edges) {
    if (edge_channel == channel) {
      return child < column.nodes.size() ? std::optional<std::uint32_t>{child} : std::nullopt;
    }
  }
  return std::nullopt;
}

/// Locates the buffers of the bitpack node `node_index`: its own-rep Bitpack leaf with all of
/// chunk_min / chunk_count / chunk_bits / packed. nullopt if the node has no such leaf.
[[nodiscard]] inline std::optional<bitpack_locator> find_bitpack(
  hpln_column_description const& column, std::uint32_t node_index)
{
  for (std::size_t l = 0; l < column.leaves.size(); ++l) {
    auto const& leaf = column.leaves[l];
    if (leaf.node_index != node_index || leaf.slot != kSelfRepSlot || leaf.kind != OpId::Bitpack) {
      continue;
    }
    bitpack_locator loc;
    loc.value_type = tag_to_dtype(leaf.type_tag);
    unsigned found = 0;
    for (std::size_t b = 0; b < leaf.buffers.size(); ++b) {
      auto const& name = leaf.buffers[b].name;
      buffer_ref const ref{l, b};
      if (name == "chunk_min") {
        loc.chunk_min = ref;
        found |= 1U;
      } else if (name == "chunk_count") {
        loc.chunk_count = ref;
        found |= 2U;
      } else if (name == "chunk_bits") {
        loc.chunk_bits = ref;
        found |= 4U;
      } else if (name == "packed") {
        loc.packed = ref;
        found |= 8U;
      }
    }
    if (found == 15U) { return loc; }
    return std::nullopt;
  }
  return std::nullopt;
}

// ─── Shared primitives ──────────────────────────────────────────────────────

/// Bitpack primitive: unpacks chunks [chunk_begin, chunk_end) of a bitpack node to int64
/// values (chunk_min + unpacked), writing (chunk_end - chunk_begin) * kChunkRows (last chunk
/// shorter) values at out (device). Caches the host prefix of chunk byte offsets in `state`
/// (create with make_bitpack_state; reuse across calls).
struct bitpack_state;
std::unique_ptr<bitpack_state, void (*)(bitpack_state*)> make_bitpack_state(
  bitpack_locator const& locator);
void unpack_bitpack_chunks(leaf_access& leaves,
                           bitpack_state& state,
                           std::int64_t chunk_begin,
                           std::int64_t chunk_end,
                           std::int64_t* out,
                           rmm::cuda_stream_view stream);
/// Device bytes `state` holds between calls.
[[nodiscard]] std::size_t bitpack_state_bytes(bitpack_state const& state) noexcept;

/// Delta primitive: decodes chunks [chunk_begin, chunk_end) of a delta(->rle)->bitpack subtree
/// rooted at `delta_node` to int64 values at out (device).
struct delta_state;
std::unique_ptr<delta_state, void (*)(delta_state*)> make_delta_state(
  hpln_column_description const& column, std::uint32_t delta_node);
void decode_delta_chunks(leaf_access& leaves,
                         delta_state& state,
                         std::int64_t chunk_begin,
                         std::int64_t chunk_end,
                         std::int64_t* out,
                         rmm::cuda_stream_view stream);
/// Device bytes `state` holds between calls, its bitpack children included.
[[nodiscard]] std::size_t delta_state_bytes(delta_state const& state) noexcept;

/// ANS primitive: writes uncompressed bytes [byte_begin, byte_end) of the ANS leaf buffer `ans`
/// to dst (device), decoding only the 64 KiB blocks that cover the range.
struct ans_state;
std::unique_ptr<ans_state, void (*)(ans_state*)> make_ans_state(
  hpln_column_description const& column, buffer_ref ans);
void decode_ans_range(leaf_access& leaves,
                      ans_state& state,
                      std::uint64_t byte_begin,
                      std::uint64_t byte_end,
                      void* dst,
                      rmm::cuda_stream_view stream);
/// Device bytes `state` holds between calls.
[[nodiscard]] std::size_t ans_state_bytes(ans_state const& state) noexcept;

// ─── Staging, exposed for tests ─────────────────────────────────────────────

/// The leaf_access a cursor over `column` of `description` hands to its decoder.
std::unique_ptr<leaf_access> make_staged_leaf_access(
  std::shared_ptr<const hpln_description> description,
  std::size_t column,
  std::shared_ptr<payload_source> payload,
  rmm::device_async_resource_ref mr);
/// Device bytes currently staged by an access made by make_staged_leaf_access.
[[nodiscard]] std::size_t staged_bytes_held(leaf_access const& access);
/// What decompress_column_chunks does to its cursor's access at the end of every call.
void release_transient_staging(leaf_access& access);

}  // namespace simpatico::chunk
