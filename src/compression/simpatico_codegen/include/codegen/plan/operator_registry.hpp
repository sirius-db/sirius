// SPDX-License-Identifier: Apache-2.0
#ifndef SIMPATICO_OPERATOR_REGISTRY_HPP
#define SIMPATICO_OPERATOR_REGISTRY_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simpatico {

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
  // Names surfaced by all_compressor_names() for this op (usually {name};
  // bitextract expands to its float aliases). Empty = excluded from the catalog.
  std::vector<std::string> catalog_names;
  bool terminal;       // emits an opaque byte payload that ends a cascade branch
  bool preprocessing;  // reshapes data without necessarily shrinking it
  bool codegen;        // inverted by the JIT codegen decode path, not a rep's decompress()
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

}  // namespace simpatico

#endif  // SIMPATICO_OPERATOR_REGISTRY_HPP
