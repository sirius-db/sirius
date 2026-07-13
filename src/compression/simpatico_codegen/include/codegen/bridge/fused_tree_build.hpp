// SPDX-License-Identifier: Apache-2.0
#ifndef CODEGEN_BRIDGE_FUSED_TREE_BUILD_HPP
#define CODEGEN_BRIDGE_FUSED_TREE_BUILD_HPP

#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/plan_tree.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simpatico {

// ---------------------------------------------------------------------------
// Shared structural builder: PlanTree fusable region -> jit::FusedTree.
//
// Both the encode and decode bridges need the SAME runtime FusedTree shape for
// a fused region; the only difference is what each side does per node afterwards
// (encode produces device buffers + reps; decode binds device buffers). This
// builder owns the one piece they share — the tree shape and the DFS-preorder
// node-id assignment — so the two sides can never drift on structure or id
// order.
//
// The build mirrors the fusion accept rules (see plan/fusion.hpp). Fusable ops
// are bitpack, zigzag, delta, rle, and for:
//   * bitpack is a leaf (its outgoing edges are entropy tails, ignored).
//   * delta/rle/for recurse into each declared channel when it feeds a codegen
//     op, else synthesize a Raw passthrough leaf (channel drained through a
//     synthetic identity leaf, left terminal, or feeding a non-fusable op); rle
//     keeps both its "runs" and "values" children.
//   * zigzag fuses its "zigzag" child only when present and codegen; otherwise
//     it carries no child.
//
// Children are keyed by their DSL channel name directly ("differences", "runs",
// "values", "deltas", "zigzag") with no translation, and stored in a std::map so
// iteration is lex-sorted — the same order jit::dfs_nodes() assigns node ids, so
// `preorder` index == rendered kernel node_id.
// ---------------------------------------------------------------------------

// One built FusedTree node mapped back to its PlanTree origin, in DFS-preorder
// (index == jit node_id). For a real op node `plan_node` is the PlanTree node
// id. For a synthesized Raw passthrough leaf, `is_raw_passthrough` is true,
// `parent_rle` is the owning parent node, `parent_op` is the parent's op name
// ("rle", "for", "delta") and `parent_channel` is the exact DSL channel name
// being materialized ("values", "runs", "deltas", "differences").
struct FusedNodeOrigin {
  codegen::jit::FusedTree* node = nullptr;
  NodeId plan_node              = 0;
  bool is_raw_passthrough       = false;
  NodeId parent_rle             = 0;  // owning parent's PlanTree NodeId
  std::string parent_op;              // owning parent's op name
  std::string parent_channel;         // channel being materialized by this Raw leaf
};

struct BuiltFusedTree {
  std::shared_ptr<codegen::jit::FusedTree> tree;
  std::vector<FusedNodeOrigin> preorder;  // index == jit node_id
};

// Build the FusedTree for the fusable region rooted at `root`. Returns
// std::nullopt if `root` is not the root of a valid fusable region.
std::optional<BuiltFusedTree> build_fused_tree(PlanTree const& tree, NodeId root);

}  // namespace simpatico

#endif  // CODEGEN_BRIDGE_FUSED_TREE_BUILD_HPP
