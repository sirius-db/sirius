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
// a fused region; the only difference is FusedTree::fixed_stride on Bitpack
// nodes (true for the OverAllocate encode layout, false for the Compact decode
// layout) and what each side does per node afterwards (encode produces device
// buffers + reps; decode binds device buffers). This builder owns the one piece
// they share — the tree shape and the DFS-preorder node-id assignment — so the
// two sides can never drift on structure or id order.
//
// The build mirrors the fusion accept rules (see plan/fusion.hpp):
//   * bitpack -> Bitpack leaf (its outgoing edges are entropy tails, ignored).
//   * delta   -> Delta with one child keyed "values" (DSL "differences").
//   * rle     -> Rle with children "runs" (required, fusable) and "values"
//                (fusable child, OR a synthesized Raw passthrough leaf when the
//                tree has no "values" edge — the DSL drained it through a
//                synthetic identity leaf that plan_tree_from_steps dropped).
//
// Child keys use the renderer's convention (Delta's value port is keyed
// "values"), and children are stored in a std::map so iteration is lex-sorted —
// the same order jit::dfs_nodes() assigns node ids, so `preorder` index ==
// rendered kernel node_id.
// ---------------------------------------------------------------------------

// One built FusedTree node mapped back to its PlanTree origin, in DFS-preorder
// (index == jit node_id). For a real op node `plan_node` is the PlanTree node
// id. For a synthesized Raw passthrough leaf, `is_raw_passthrough` is true,
// `parent_rle` is the owning parent node (rle or for), and `parent_op` is the
// parent's op name ("rle" or "for") so the encode/decode bridges know which
// channel to bind: "values" for RLE, "deltas" for FOR.
struct FusedNodeOrigin {
  codegen::jit::FusedTree* node = nullptr;
  NodeId plan_node              = 0;
  bool is_raw_passthrough       = false;
  NodeId parent_rle             = 0;  // owning parent's PlanTree NodeId
  std::string parent_op;              // owning parent's op name ("rle" or "for")
};

struct BuiltFusedTree {
  std::shared_ptr<codegen::jit::FusedTree> tree;
  std::vector<FusedNodeOrigin> preorder;  // index == jit node_id
};

// Build the FusedTree for the fusable region rooted at `root`. `fixed_stride`
// sets FusedTree::fixed_stride on every Bitpack node. Returns std::nullopt if
// `root` is not the root of a valid fusable region.
std::optional<BuiltFusedTree> build_fused_tree(PlanTree const& tree,
                                               NodeId root,
                                               bool fixed_stride);

}  // namespace simpatico

#endif  // CODEGEN_BRIDGE_FUSED_TREE_BUILD_HPP
