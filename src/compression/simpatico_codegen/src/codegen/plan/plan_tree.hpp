// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/bit_spec.hpp"  // bitfield_spec
#include "codegen/plan/leaf_desc.hpp"
#include "codegen/plan/operator_registry.hpp"  // ChannelId
#include "codegen/plan/plan_dsl.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simpatico {

using NodeId = std::uint32_t;

// Structural identity of a dataflow value: the (producing node, output port) it
// leaves on. Decode keys and resolves values by this, never by a path string.
// Node 0 port 0 is the column input.
struct ValueId {
  NodeId node       = 0;
  ChannelId channel = 0;
};

inline bool operator==(ValueId a, ValueId b) { return a.node == b.node && a.channel == b.channel; }
inline bool operator!=(ValueId a, ValueId b) { return !(a == b); }

// Pack a ValueId into a unique 64-bit key (channel is a uint8, so node<<8 leaves
// no room for collision). Shared by the decode memo and the compress walk maps.
inline constexpr std::uint64_t value_id_key(ValueId v) noexcept
{
  return (static_cast<std::uint64_t>(v.node) << 8) | v.channel;
}

// Hasher so ValueId can key an unordered_map/set (the compress walk's columns /
// reprs / refcount tables).
struct ValueIdHash {
  std::size_t operator()(ValueId v) const noexcept
  {
    return static_cast<std::size_t>(value_id_key(v));
  }
};

struct PlanEdge {
  std::string channel;
  NodeId child = 0;
};

struct bitextract_attrs {
  std::vector<bitfield_spec> fields;
  cudf::data_type output_type{cudf::type_id::EMPTY};
};

struct bitjoin_input_ref {
  NodeId node = 0;
  std::string channel;
  std::optional<bit_range> range;
};

struct bitjoin_attrs {
  std::vector<bitjoin_input_ref> inputs;
  cudf::data_type output_type{cudf::type_id::EMPTY};
};

struct op_attrs {
  std::optional<bitextract_attrs> bitextract;
  std::optional<bitjoin_attrs> bitjoin;
};

struct PlanNode {
  std::string op;
  std::vector<PlanEdge> children;
  op_attrs attrs;

  // Producer metadata (populated by plan_tree_from_steps for decode). The
  // output_paths string keys are the node.channels buffer contract shared by
  // encode and decode.
  std::vector<std::string> output_names;  // port names in step order
  std::vector<std::string> output_paths;  // dotted paths this op produces

  // Structural identity of each value this op consumes, in field order: the
  // (node, port) each input is produced on. Both compress and decode resolve
  // cross-node values against this, so neither has to match DSL path strings.
  // For a single-input op this holds one entry; for a bitjoin, one per field.
  // Populated by compute_input_sources from the edge wiring + bitjoin attrs.
  std::vector<ValueId> input_sources;

  // Node-owned compressed representations. Two storage slots cover every shape:
  //
  //   * `rep` — this op's own single representation. Set for all JIT-fused ops
  //     (Delta/Rle/Bitpack/For/Zigzag).
  //
  //   * `channels` — terminal OUTPUT representations (identity / RawFused),
  //     keyed by their DSL output PATH. Set for a producing node's non-consumed
  //     outputs — e.g. `for`'s `references` channel or the RawFused `values`
  //     passthrough of an Rle or the `deltas` passthrough of a For.
  std::unique_ptr<compressed_representation> rep;
  std::unordered_map<std::string, std::unique_ptr<compressed_representation>> channels;

  // Per-node decode metadata populated by the compress walk from
  // rep->describe_meta(). Used by reconstruct_representation to supply
  // information (e.g. uncompressed_size for ANS/Bitcomp) that cannot be
  // recovered from the stored channel buffers alone.
  leaf_meta_v meta{leaf_meta::none{}};
};

struct PlanTree {
  std::vector<PlanNode> nodes;
};

std::optional<PlanTree> plan_tree_from_dsl(std::string_view dsl, std::string* error_out = nullptr);

// Build the canonical PlanTree from the parsed plan steps.
std::optional<PlanTree> plan_tree_from_steps(std::vector<plan_step> const& steps,
                                             std::string* error_out = nullptr);

// Populate every node's input_sources from the tree's edge wiring (and bitjoin
// attrs, which carry input order). Called after the tree is built from the DSL
// or deserialized, so decode has structural value identity without path
// strings.
void compute_input_sources(PlanTree& tree);

std::string dotted_label(PlanTree const& tree, NodeId node);

// Render a PlanTree back to plan DSL text (the inverse of plan_tree_from_dsl).
// Paths are derived structurally from the edge wiring, so this works on a tree
// deserialized from disk (which carries no DSL). Round-trips through
// parse_plan_dsl to an equivalent tree.
std::string render_plan_tree(PlanTree const& tree);

}  // namespace simpatico
