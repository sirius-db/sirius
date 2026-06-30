// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/plan_interpreter.hpp"

namespace simpatico {

bool is_codegen_compressor(std::string const& op)
{
  return op == "delta" || op == "rle" || op == "bitpack" || op == "for" || op == "zigzag";
}

bool is_fusion_interior(NodeId nid, PlanTree const& tree)
{
  if (nid >= tree.nodes.size()) return false;
  if (!is_codegen_compressor(tree.nodes[nid].op)) return false;
  for (NodeId i = 0; i < tree.nodes.size(); ++i) {
    if (!is_codegen_compressor(tree.nodes[i].op)) continue;
    for (auto const& e : tree.nodes[i].children) {
      if (e.child == nid) return true;
    }
  }
  return false;
}

}  // namespace simpatico
