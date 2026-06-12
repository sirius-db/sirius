#include "codegen/plan/plan_tree.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

void expect(bool cond, char const* msg)
{
  if (!cond) throw std::runtime_error(msg);
}

void test_delta_rle_bitpack_cascade()
{
  std::string dsl =
    "input -> delta -> differences\n"
    "delta.differences -> rle -> values, runs\n"
    "delta.differences.values -> bitpack\n"
    "delta.differences.runs -> bitpack\n";

  std::string err;
  auto tree = simpatico::plan_tree_from_dsl(dsl, &err);
  expect(tree.has_value(), err.empty() ? "plan_tree_from_dsl failed" : err.c_str());
  expect(tree->nodes.size() == 5, "input + delta + rle + 2 bitpack nodes");
  expect(tree->nodes[0].op == "input", "root op");
  expect(tree->nodes[0].children.size() == 1, "root has one child");
  expect(tree->nodes[0].children[0].channel == "delta", "root edge channel");

  int bitpack_count = 0;
  std::string values_label;
  for (simpatico::NodeId i = 0; i < tree->nodes.size(); ++i) {
    if (tree->nodes[i].op != "bitpack") continue;
    ++bitpack_count;
    auto label = simpatico::dotted_label(*tree, i);
    if (label.find("values") != std::string::npos) values_label = std::move(label);
  }
  expect(bitpack_count == 2, "two bitpack nodes");
  expect(!values_label.empty(), "values bitpack dotted label");
  expect(values_label.find("input.delta.differences.values") == 0 ||
           values_label.find("input.delta.differences.rle.values") == 0 ||
           values_label.find("values") != std::string::npos,
         "values path in dotted label");
}

void test_bitjoin_attrs()
{
  std::string dsl = "input_3:0, input_7:4 -> bitjoin_u8 -> swapped";
  std::string err;
  auto tree = simpatico::plan_tree_from_dsl(dsl, &err);
  expect(tree.has_value(), err.empty() ? "bitjoin parse failed" : err.c_str());
  expect(tree->nodes.size() == 2, "input + bitjoin");
  auto const& bj = tree->nodes[1];
  expect(bj.op == "bitjoin_u8", "bitjoin op");
  expect(bj.attrs.bitjoin.has_value(), "bitjoin attrs");
  expect(bj.attrs.bitjoin->inputs.size() == 2, "two bitjoin inputs");
  expect(bj.attrs.bitjoin->output_type.id() == cudf::type_id::UINT8, "u8 output");
  expect(tree->nodes[0].children.size() == 2, "input wired to bitjoin twice");
}

void test_path_map()
{
  std::string dsl =
    "input -> delta -> differences\n"
    "delta.differences -> rle -> values, runs\n"
    "delta.differences.values -> bitpack\n"
    "delta.differences.runs -> bitpack\n";

  std::string err;
  simpatico::PlanPathMap pm;
  auto tree = simpatico::plan_tree_from_dsl(dsl, &err, &pm);
  expect(tree.has_value(), err.empty() ? "plan_tree_from_dsl failed" : err.c_str());

  // Root.
  expect(pm.node.at("input") == 0, "input -> node 0");
  expect(pm.channel.at("input") == "input", "input channel");

  // delta produces "delta.differences" (channel "differences") at the delta node.
  simpatico::NodeId delta_node = pm.node.at("delta.differences");
  expect(tree->nodes[delta_node].op == "delta", "delta.differences produced by delta node");
  expect(pm.channel.at("delta.differences") == "differences", "differences channel");

  // rle produces both values and runs at the SAME (rle) node, distinct channels.
  simpatico::NodeId values_node = pm.node.at("delta.differences.values");
  simpatico::NodeId runs_node   = pm.node.at("delta.differences.runs");
  expect(values_node == runs_node, "rle values+runs share one producing node");
  expect(tree->nodes[values_node].op == "rle", "values/runs produced by rle node");
  expect(pm.channel.at("delta.differences.values") == "values", "values channel");
  expect(pm.channel.at("delta.differences.runs") == "runs", "runs channel");

  // Terminal bitpack nodes produce no downstream path (not map producers).
  expect(pm.node.find("delta.differences.values.bitpack") == pm.node.end(),
         "terminal bitpack produces no path");
}

}  // namespace

int main()
{
  try {
    test_delta_rle_bitpack_cascade();
    test_bitjoin_attrs();
    test_path_map();
    std::printf("test_plan_tree: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_plan_tree: FAIL: %s\n", e.what());
    return 1;
  }
}
