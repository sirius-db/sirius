// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/plan_interpreter.hpp"

namespace simpatico {

bool is_codegen_compressor(std::string const& op)
{
  return op == "delta" || op == "rle" || op == "bitpack" || op == "for" || op == "zigzag";
}

// Returns true iff `channel` is a JIT-recursive child channel of `parent_op`
// — one that build_rec actually recurses into and renders inline inside the
// same kernel.  Boundary-output channels (for.references, bitpack.packed,
// bitpack.chunk_min, …) are NOT fused child channels; their downstream ops
// are independent fused regions decoded separately.
static bool is_fused_child_channel(std::string const& parent_op, std::string const& channel)
{
  if (parent_op == "delta") return channel == "differences";
  if (parent_op == "rle") return channel == "runs" || channel == "values";
  if (parent_op == "for") return channel == "deltas";
  if (parent_op == "zigzag") return channel == "zigzag";
  return false;  // bitpack is a leaf — no fused children
}

bool is_fusion_interior(NodeId nid, PlanTree const& tree)
{
  if (nid >= tree.nodes.size()) return false;
  if (!is_codegen_compressor(tree.nodes[nid].op)) return false;
  for (NodeId i = 0; i < tree.nodes.size(); ++i) {
    if (!is_codegen_compressor(tree.nodes[i].op)) continue;
    for (auto const& e : tree.nodes[i].children) {
      if (e.child == nid && is_fused_child_channel(tree.nodes[i].op, e.channel)) return true;
    }
  }
  return false;
}

}  // namespace simpatico
