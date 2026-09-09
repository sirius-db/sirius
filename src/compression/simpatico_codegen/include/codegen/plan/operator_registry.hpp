// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simpatico {

struct compressor;  // defined in representation.hpp

// Stable identity of an operator *kind*. Doubles as the on-disk leaf tag
// (serialised as a uint8_t in the .hpln format). Values for leaf ops are
// fixed; structural pre-processing ops (Bitextract, StrSplit) have values
// that will never appear on disk and may be renumbered freely.
enum class OpId : std::uint8_t {
  // Leaf ops — values are stable wire-format tags (do not reorder/renumber).
  Delta          = 1,
  Rle            = 2,
  Dictionary     = 3,
  Bitpack        = 4,
  Identity       = 5,
  For            = 6,
  Snappy         = 7,
  Deflate        = 8,
  Lz4            = 9,
  Ans            = 10,
  Bitcomp        = 11,
  NvcompCascaded = 12,
  Alp            = 13,
  AlpRd          = 14,
  Zigzag         = 15,
  // Structural / pre-processing ops — not serialised as leaf tags.
  Bitextract = 16,
  StrSplit   = 17,
  // Sentinel for unknown/unrecognised kinds read from file.
  Unknown = 255,
};

// Canonical position of a channel within its operator's fixed channel set.
using ChannelId = std::uint8_t;

// How a channel is laid out with respect to the 1024-row decode chunk, which is what decides
// whether a subset of chunks can be served without the rest of the column.
//
// Fetching only some chunks of a column is only possible if the result is still a VALID column:
// the decode must see metadata whose entries correspond one-to-one with the bulk bytes it was
// given. Bitpack makes that work because `bp_offsets` is not stored — it is scanned from the
// `chunk_count`/`chunk_bits` that were loaded (src/bridge/offsets_cumsum.cu) — so a compacted
// metadata array plus the matching packed bytes decodes correctly on its own.
enum class ChannelLayout : std::uint8_t {
  /// One entry per 1024-row chunk, in chunk order. A subset is formed by keeping the surviving
  /// entries: fixed stride, so the byte range of any chunk is arithmetic.
  per_chunk_metadata,
  /// Fixed bytes per ROW, so chunk c starts at (rows before c) * element size. Derivable from
  /// the row count alone — no operator-specific knowledge, no metadata read.
  bulk_fixed_stride,
  /// Variable bytes per chunk, derivable only from that operator's own per-chunk metadata
  /// (bitpack's packed, sized by chunk_count x chunk_bits) or from an out-of-band
  /// group-to-byte table. The ONLY layout that needs operator-specific arithmetic.
  bulk_variable,
  /// No per-chunk structure at all: a codec's opaque output with its own internal chunking. Always
  /// fetched whole, and its bytes depend on WHICH rows are in the column, so a column whose bulk
  /// channel is opaque cannot be partially fetched at all.
  whole_column,
  /// Column-wide STATE, not rows: a dictionary's keys. Fetched whole like @c whole_column, but
  /// unlike it the bytes do not depend on which rows are served, so the column can still be
  /// subsetted — the keys stay valid for any subset of the indices that reference them.
  ///
  /// The distinction between this and @c whole_column is the whole reason a dictionary-encoded
  /// string column can be chunk-addressed while an lz4 one cannot.
  column_state,
};

// Layout of the PERSISTED BUFFER named @p buffer within @p id, or nullopt when the operator
// persists no such buffer.
//
// Deliberately keyed on buffer names, not on OperatorInfo::channels: those are the operator's
// output PORTS (delta's "differences" names the edge to its child), whereas what a fetch has to
// address is what actually lands in the payload (delta persists "delta_first"). The two coincide
// for bitpack and diverge for every preprocessing op.
[[nodiscard]] std::optional<ChannelLayout> buffer_layout(OpId id, std::string_view buffer);

// True when a node of type @p id can be served as a subset of its 1024-row chunks: it persists at
// least one buffer, every persisted buffer is classified, and none is whole_column
// (column_state buffers are fine — see ChannelLayout). False means "fetch it whole", which is
// always correct and merely forgoes the saving.
[[nodiscard]] bool supports_chunk_subset(OpId id);

// True when the values @p id produces on output channel @p channel are column-wide STATE rather
// than one value per row of the op's own output.
//
// This is about the EDGE, not the buffer: whatever compresses `dictionary.keys_offsets` — a
// bitpack, a delta feeding an rle, an ans — produces buffers of its own that look perfectly
// row-indexed on their own grid, and are not the column's rows at all. A caller serving a subset
// of a column's chunks must fetch that whole subtree whole, and must not mistake its differing
// length for a reason to refuse the column. @p channel may be a dotted path; the last component
// is the port name.
[[nodiscard]] bool channel_is_column_state(OpId id, std::string_view channel);

// One row of the operator registry: the single source of truth tying an
// operator's DSL/diagnostic name, canonical output-channel order, and
// classification together. make_compressor / reconstruct_representation /
// all_compressor_names / the fusion + explorer predicates / DSL channel
// canonicalisation all derive from this table instead of maintaining their
// own scattered name lists.
struct OperatorInfo {
  OpId id;
  std::string_view name;  // canonical DSL / diagnostic name
  // Canonical channel order, mirroring the rep's named_channels(). Empty for
  // variable-arity ops (dictionary, bitextract) whose channels are not fixed.
  std::vector<std::string> channels;
  bool explorable;     // surfaced by all_compressor_names() (false = excluded, e.g. identity)
  bool terminal;       // emits an opaque byte payload that ends a cascade branch
  bool preprocessing;  // reshapes data without necessarily shrinking it
  bool codegen;        // inverted by the JIT codegen decode path, not a rep's decompress()
  // The buffers this operator PERSISTS, and how each sits relative to the 1024-row decode chunk.
  // Independent of `channels` above, which names output ports; see buffer_layout(). Empty means
  // "not classified", which makes the operator ineligible for chunk-subset serving.
  std::vector<std::pair<std::string_view, ChannelLayout>> persisted_buffers;
};

// The full registry, in catalog order.
std::vector<OperatorInfo> const& operator_registry();

// Registry row for an OpId.
OperatorInfo const& op_info(OpId id);

// Resolve a DSL compressor name (including parameterised suffix forms) to its
// OpId, or nullopt if unrecognised.
std::optional<OpId> op_id_from_name(std::string const& name);

// Canonical output-channel order for `id`, or nullptr for variable-arity ops.
std::vector<std::string> const* canonical_channels(OpId id);

// Canonical index of `channel` within op `id`'s channel set, or nullopt.
std::optional<ChannelId> channel_id(OpId id, std::string_view channel);

// Operator names the explorer / sweep attempt, in catalog order. bitextract
// expands to its float aliases; non-explorable ops (identity, nvcomp_cascaded)
// are excluded.
std::vector<std::string> const& all_compressor_names();

// Terminal ops emit an opaque byte payload that ends a cascade branch.
bool is_terminal_compressor(std::string const& name);

// Preprocessing (transform) ops reshape data without necessarily shrinking it.
bool is_preprocessing_compressor(std::string const& name);

/// Resolve a DSL compressor name to a compressor instance, or nullptr if
/// the name is unknown. The fused ops (delta / rle / bitpack / for / zigzag) are
/// not here — they go through the JIT codegen encoder, not a compressor factory.
///
/// Factory-created names (including parameterised suffix forms):
///   identity, dictionary, str_split, alp, alp_rd, ans, snappy, lz4, deflate,
///   bitcomp[_default|_sparse], nvcomp_cascaded[_<opts>], bitextract_<spec>.
std::unique_ptr<compressor> make_compressor(std::string const& name);

}  // namespace simpatico
