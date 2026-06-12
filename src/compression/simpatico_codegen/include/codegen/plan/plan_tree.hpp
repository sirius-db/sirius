// SPDX-License-Identifier: Apache-2.0
#ifndef CODEGEN_PLAN_TREE_HPP
#define CODEGEN_PLAN_TREE_HPP

#include "codegen/plan/leaf_desc.hpp"
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

struct compressed_representation;

using NodeId = std::uint32_t;

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

  // Producer metadata (populated by plan_tree_from_steps for decode).
  std::string input_path;                 // DSL path this op consumes (empty for bitjoin)
  std::vector<std::string> output_names;  // port names in step order
  std::vector<std::string> output_paths;  // dotted paths this op produces

  // Consumer metadata (populated by plan_tree_from_steps for the compress walk).
  // The DSL paths this op consumes and their per-input source bit ranges. For a
  // single-input op this is {input_path}; for a bitjoin it is the full field
  // list. Lets the forward compress walk run an op straight from its node.
  std::vector<std::string> input_paths;
  std::vector<std::optional<bit_range>> input_ranges;

  // Node-owned compressed representations. Two storage slots cover every shape:
  //
  //   * `rep` — this op's own single representation, keyed by its INPUT path
  //     (`rep_path`). Set for the fused ops (Delta/Rle/Bitpack) and terminal
  //     ops consumed by nothing downstream (e.g. `input -> for`).
  //
  //   * `channels` — terminal OUTPUT representations (identity / RawFused),
  //     keyed by their DSL output PATH. Set for a producing node's non-consumed
  //     outputs — e.g. `for`'s {references, reference_offsets} channels or the
  //     RawFused `values` passthrough of an Rle.
  std::unique_ptr<compressed_representation> rep;
  std::string rep_path;
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

// Maps each DSL dotted path to the node that produces it and the channel
// (output-port) name under which it is produced. This is the inverse of the
// tree's named-edge wiring, used when populating node reps during native
// tree-based compression (each produced channel's rep attaches to
// `node[path]` under `channel[path]`). The reserved root path "input" maps to
// node 0 with channel "input".
struct PlanPathMap {
  std::unordered_map<std::string, NodeId> node;
  std::unordered_map<std::string, std::string> channel;
};

std::optional<PlanTree> plan_tree_from_dsl(std::string_view dsl,
                                           std::string* error_out = nullptr,
                                           PlanPathMap* path_map  = nullptr);

// Build the canonical PlanTree from the parsed plan steps (synthetic identity
// drains are skipped — they have no node). Populates `path_map` (producer node
// + channel per output path) when non-null.
std::optional<PlanTree> plan_tree_from_steps(std::vector<plan_step> const& steps,
                                             std::string* error_out = nullptr,
                                             PlanPathMap* path_map  = nullptr);

std::string dotted_label(PlanTree const& tree, NodeId node);

}  // namespace simpatico

#endif  // CODEGEN_PLAN_TREE_HPP
