// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"

#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/plan_interpreter.hpp"

namespace simpatico {
namespace {

namespace jit = codegen::jit;
using codegen::OpKind;

PlanEdge const* find_edge(PlanNode const& node, std::string const& channel)
{
  for (auto const& e : node.children) {
    if (e.channel == channel) return &e;
  }
  return nullptr;
}

// Recursively build the FusedTree for the fusable region rooted at `nid`,
// appending each created node to `out.preorder` in DFS-preorder (parent before
// children; children visited in ascending child-key order so the index matches
// jit::dfs_nodes()). Returns the built node, or nullptr if the region is
// invalid.
std::shared_ptr<jit::FusedTree> build_rec(PlanTree const& tree, NodeId nid, BuiltFusedTree& out)
{
  PlanNode const& node = tree.nodes[nid];

  if (node.op == "bitpack") {
    auto t = jit::FusedTree::make(OpKind::Bitpack);
    out.preorder.push_back({t.get(), nid, false, 0});
    return t;
  }

  if (node.op == "zigzag") {
    PlanEdge const* zz = find_edge(node, "zigzag");
    auto t             = jit::FusedTree::make(OpKind::Zigzag);
    out.preorder.push_back({t.get(), nid, false, 0});
    if (zz != nullptr && is_codegen_compressor(tree.nodes[zz->child].op)) {
      auto child = build_rec(tree, zz->child, out);
      if (!child) return nullptr;
      // Child keyed "zigzag" everywhere — the DSL port, the fused rep channel,
      // and the renderer all agree, so there is no name translation.
      t->children.emplace("zigzag", std::move(child));
    }
    return t;
  }

  if (node.op == "delta") {
    PlanEdge const* diff = find_edge(node, "differences");
    auto t               = jit::FusedTree::make(OpKind::Delta);
    out.preorder.push_back({t.get(), nid, false, 0});
    std::shared_ptr<jit::FusedTree> diff_child;
    if (diff != nullptr && is_codegen_compressor(tree.nodes[diff->child].op)) {
      diff_child = build_rec(tree, diff->child, out);
      if (!diff_child) return nullptr;
    } else {
      diff_child = jit::FusedTree::make(OpKind::Raw);
      out.preorder.push_back({diff_child.get(), 0, true, nid, "delta", "differences"});
    }
    t->children.emplace("differences", std::move(diff_child));
    return t;
  }

  if (node.op == "rle") {
    PlanEdge const* runs   = find_edge(node, "runs");
    PlanEdge const* values = find_edge(node, "values");
    auto t                 = jit::FusedTree::make(OpKind::Rle);
    out.preorder.push_back({t.get(), nid, false, 0});
    // Children appended in ascending child-key order: "runs" < "values".

    // runs — dual-mode: fuse a codegen child, or synthesize Raw passthrough.
    // Like values, a missing edge (terminal run-count storage) is valid.
    std::shared_ptr<jit::FusedTree> runs_child;
    if (runs != nullptr && is_codegen_compressor(tree.nodes[runs->child].op)) {
      runs_child = build_rec(tree, runs->child, out);
      if (!runs_child) return nullptr;
    } else {
      // Materialize run-counts (compact layout, int32 element size). The
      // downstream non-fused op entropy-codes the run-count stream.
      runs_child = jit::FusedTree::make(OpKind::Raw);
      out.preorder.push_back({runs_child.get(), 0, true, nid, "rle", "runs"});
    }
    t->children.emplace("runs", std::move(runs_child));

    // values — dual-mode: fuse a codegen child, or synthesize Raw passthrough.
    std::shared_ptr<jit::FusedTree> values_child;
    if (values != nullptr && is_codegen_compressor(tree.nodes[values->child].op)) {
      values_child = build_rec(tree, values->child, out);
      if (!values_child) return nullptr;
    } else {
      // Raw passthrough: either no values edge (DSL drained it through a
      // synthetic identity leaf) or the edge feeds a non-fusable op.
      values_child = jit::FusedTree::make(OpKind::Raw);
      out.preorder.push_back({values_child.get(), 0, true, nid, "rle", "values"});
    }
    t->children.emplace("values", std::move(values_child));
    return t;
  }

  if (node.op == "for") {
    // Dual-mode: fuse a codegen deltas child, or synthesize Raw passthrough.
    PlanEdge const* deltas = find_edge(node, "deltas");
    auto t                 = jit::FusedTree::make(OpKind::For);
    out.preorder.push_back({t.get(), nid, false, 0});
    std::shared_ptr<jit::FusedTree> deltas_child;
    if (deltas != nullptr && is_codegen_compressor(tree.nodes[deltas->child].op)) {
      deltas_child = build_rec(tree, deltas->child, out);
      if (!deltas_child) return nullptr;
    } else {
      // Materialize deltas (fixed-stride storage). Downstream non-fused op
      // entropy-codes the channel bytes, or terminal (no consumer).
      deltas_child = jit::FusedTree::make(OpKind::Raw);
      out.preorder.push_back({deltas_child.get(), 0, true, nid, "for", "deltas"});
    }
    t->children.emplace("deltas", std::move(deltas_child));
    return t;
  }

  return nullptr;  // non-fusable op
}

}  // namespace

std::optional<BuiltFusedTree> build_fused_tree(PlanTree const& tree, NodeId root)
{
  if (root >= tree.nodes.size()) return std::nullopt;
  BuiltFusedTree out;
  auto t = build_rec(tree, root, out);
  if (!t) return std::nullopt;
  out.tree = std::move(t);
  return out;
}

}  // namespace simpatico
