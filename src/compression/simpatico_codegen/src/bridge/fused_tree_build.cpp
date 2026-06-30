// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"

#include "codegen/tree.hpp"

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
std::shared_ptr<jit::FusedTree> build_rec(PlanTree const& tree,
                                          NodeId nid,
                                          bool fixed_stride,
                                          BuiltFusedTree& out)
{
  PlanNode const& node = tree.nodes[nid];

  if (node.op == "bitpack") {
    auto t          = jit::FusedTree::make(OpKind::Bitpack);
    t->fixed_stride = fixed_stride;
    out.preorder.push_back({t.get(), nid, false, 0});
    return t;
  }

  if (node.op == "zigzag") {
    // Leaf: ZigZag stores its transformed stream to the `zigzag` channel
    // (its outgoing edge, if any, is an entropy tail on that channel —
    // ignored here, exactly like Bitpack's packed tail).
    auto t = jit::FusedTree::make(OpKind::Zigzag);
    out.preorder.push_back({t.get(), nid, false, 0});
    return t;
  }

  if (node.op == "delta") {
    PlanEdge const* diff = find_edge(node, "differences");
    if (diff == nullptr) return nullptr;  // value child required (no raw form)
    auto t = jit::FusedTree::make(OpKind::Delta);
    out.preorder.push_back({t.get(), nid, false, 0});
    auto child = build_rec(tree, diff->child, fixed_stride, out);
    if (!child) return nullptr;
    // Delta's single child is keyed "differences" everywhere — the DSL port,
    // the fused rep channel, and the renderer all agree, so there is no name
    // translation.
    t->children.emplace("differences", std::move(child));
    return t;
  }

  if (node.op == "rle") {
    PlanEdge const* runs = find_edge(node, "runs");
    if (runs == nullptr) return nullptr;  // runs child required + fusable
    PlanEdge const* values = find_edge(node, "values");
    auto t                 = jit::FusedTree::make(OpKind::Rle);
    out.preorder.push_back({t.get(), nid, false, 0});
    // Children appended in ascending child-key order: "runs" < "values".
    auto runs_child = build_rec(tree, runs->child, fixed_stride, out);
    if (!runs_child) return nullptr;
    t->children.emplace("runs", std::move(runs_child));

    std::shared_ptr<jit::FusedTree> values_child;
    if (values != nullptr) {
      values_child = build_rec(tree, values->child, fixed_stride, out);
      if (!values_child) return nullptr;
    } else {
      // Raw passthrough: the rle "values" channel has no tree node (the DSL
      // drained it through a synthetic identity leaf). Synthesize the Raw leaf.
      values_child = jit::FusedTree::make(OpKind::Raw);
      out.preorder.push_back({values_child.get(), 0, true, nid, "rle"});
    }
    t->children.emplace("values", std::move(values_child));
    return t;
  }

  if (node.op == "for") {
    PlanEdge const* deltas = find_edge(node, "deltas");
    auto t                 = jit::FusedTree::make(OpKind::For);
    out.preorder.push_back({t.get(), nid, false, 0});
    std::shared_ptr<jit::FusedTree> deltas_child;
    if (deltas != nullptr) {
      deltas_child = build_rec(tree, deltas->child, fixed_stride, out);
      if (!deltas_child) return nullptr;
    } else {
      // No downstream deltas consumer: Raw passthrough (fixed-stride storage).
      // Mirrors RLE's values Raw passthrough but uses "deltas" port.
      deltas_child = jit::FusedTree::make(OpKind::Raw);
      out.preorder.push_back({deltas_child.get(), 0, true, nid, "for"});
    }
    t->children.emplace("deltas", std::move(deltas_child));
    return t;
  }

  return nullptr;  // non-fusable op
}

}  // namespace

std::optional<BuiltFusedTree> build_fused_tree(PlanTree const& tree, NodeId root, bool fixed_stride)
{
  if (root >= tree.nodes.size()) return std::nullopt;
  BuiltFusedTree out;
  auto t = build_rec(tree, root, fixed_stride, out);
  if (!t) return std::nullopt;
  out.tree = std::move(t);
  return out;
}

}  // namespace simpatico
