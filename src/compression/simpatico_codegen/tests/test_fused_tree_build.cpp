// SPDX-License-Identifier: Apache-2.0
//
// Structural tests for the shared PlanTree -> jit::FusedTree builder
// (bridge/fused_tree_build.{hpp,cpp}). Host-only (no GPU): asserts tree shape,
// DFS-preorder node-id ordering, and the rle raw passthrough leaf.

#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/plan_tree.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

using codegen::OpKind;

void expect(bool cond, char const* msg)
{
  if (!cond) throw std::runtime_error(msg);
}

simpatico::PlanTree build_tree(std::string const& dsl)
{
  std::string err;
  auto t = simpatico::plan_tree_from_dsl(dsl, &err);
  expect(t.has_value(), err.empty() ? "plan_tree_from_dsl failed" : err.c_str());
  return std::move(*t);
}

simpatico::NodeId node_for_op(simpatico::PlanTree const& t, std::string const& op)
{
  for (simpatico::NodeId i = 0; i < t.nodes.size(); ++i) {
    if (t.nodes[i].op == op) return i;
  }
  return static_cast<simpatico::NodeId>(t.nodes.size());
}

void test_cascade_shape_and_preorder()
{
  auto t = build_tree(
    "input -> delta -> differences\n"
    "delta.differences -> rle -> values, runs\n"
    "delta.differences.values -> bitpack\n"
    "delta.differences.runs -> bitpack\n");
  simpatico::NodeId delta = node_for_op(t, "delta");

  auto built = simpatico::build_fused_tree(t, delta);
  expect(built.has_value(), "build_fused_tree returned nullopt for valid cascade");

  auto const& root = *built->tree;
  expect(root.op == OpKind::Delta, "root op Delta");
  auto rle_it = root.children.find("differences");
  expect(rle_it != root.children.end(), "delta has 'differences' child");
  auto const& rle = *rle_it->second;
  expect(rle.op == OpKind::Rle, "delta child is Rle");
  expect(rle.children.count("runs") == 1 && rle.children.count("values") == 1,
         "rle has runs+values children");
  expect(rle.children.at("runs")->op == OpKind::Bitpack, "runs child Bitpack");
  expect(rle.children.at("values")->op == OpKind::Bitpack, "values child Bitpack");

  // Preorder: delta, rle, bitpack(runs), bitpack(values) — children lex order.
  expect(built->preorder.size() == 4, "4 nodes in preorder");
  expect(built->preorder[0].node->op == OpKind::Delta, "preorder[0] Delta");
  expect(built->preorder[1].node->op == OpKind::Rle, "preorder[1] Rle");
  expect(built->preorder[2].node->op == OpKind::Bitpack, "preorder[2] Bitpack");
  expect(built->preorder[3].node->op == OpKind::Bitpack, "preorder[3] Bitpack");
  // preorder[2] is the "runs" bitpack, preorder[3] the "values" bitpack.
  expect(built->preorder[2].node == rle.children.at("runs").get(), "preorder[2] == runs bitpack");
  expect(built->preorder[3].node == rle.children.at("values").get(),
         "preorder[3] == values bitpack");
  for (auto const& o : built->preorder) {
    expect(!o.is_raw_passthrough, "no raw passthrough in cascade");
  }
}

void test_lone_bitpack()
{
  auto t               = build_tree("input -> bitpack\n");
  simpatico::NodeId bp = node_for_op(t, "bitpack");
  auto built           = simpatico::build_fused_tree(t, bp);
  expect(built.has_value(), "lone bitpack builds");
  expect(built->tree->op == OpKind::Bitpack, "root Bitpack");
  expect(built->preorder.size() == 1, "single preorder node");
}

void test_rle_raw_values_passthrough()
{
  auto t = build_tree(
    "input -> rle -> values, runs\n"
    "rle.runs -> bitpack\n");
  simpatico::NodeId rle = node_for_op(t, "rle");
  auto built            = simpatico::build_fused_tree(t, rle);
  expect(built.has_value(), "rle+raw-values builds");

  auto const& root = *built->tree;
  expect(root.op == OpKind::Rle, "root Rle");
  expect(root.children.at("runs")->op == OpKind::Bitpack, "runs Bitpack");
  auto const& vals = *root.children.at("values");
  expect(vals.op == OpKind::Raw, "values is Raw passthrough leaf");
  expect(vals.is_leaf(), "Raw leaf has no children");

  // Preorder: rle, bitpack(runs), raw(values).
  expect(built->preorder.size() == 3, "3 preorder nodes");
  expect(built->preorder[2].is_raw_passthrough, "preorder[2] is raw passthrough");
  expect(built->preorder[2].parent_rle == rle, "raw passthrough parent is rle");
}

// delta with a non-fusable (ans) child on the differences channel now builds
// a valid tree via the new dual-mode Raw passthrough, just like FOR's deltas.
void test_delta_raw_differences_passthrough()
{
  auto t = build_tree(
    "input -> delta -> differences\n"
    "delta.differences -> ans\n");
  simpatico::NodeId delta = node_for_op(t, "delta");
  auto built              = simpatico::build_fused_tree(t, delta);
  expect(built.has_value(), "delta with ans child -> valid Raw passthrough tree");

  auto const& root = *built->tree;
  expect(root.op == OpKind::Delta, "root is Delta");
  auto diff_it = root.children.find("differences");
  expect(diff_it != root.children.end(), "delta has 'differences' child");
  expect(diff_it->second->op == OpKind::Raw, "differences child is Raw passthrough");
  expect(diff_it->second->is_leaf(), "Raw passthrough has no children");

  // Preorder: delta, raw(differences).
  expect(built->preorder.size() == 2, "2 preorder nodes");
  expect(!built->preorder[0].is_raw_passthrough, "preorder[0] is real Delta node");
  expect(built->preorder[1].is_raw_passthrough, "preorder[1] is raw passthrough");
  expect(built->preorder[1].parent_rle == delta, "raw passthrough parent is delta");
  expect(built->preorder[1].parent_channel == "differences",
         "raw passthrough channel is 'differences'");
}

}  // namespace

int main()
{
  try {
    test_cascade_shape_and_preorder();
    test_lone_bitpack();
    test_rle_raw_values_passthrough();
    test_delta_raw_differences_passthrough();
    std::printf("test_fused_tree_build: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_fused_tree_build: FAIL: %s\n", e.what());
    return 1;
  }
}
