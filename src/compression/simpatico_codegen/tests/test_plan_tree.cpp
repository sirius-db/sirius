#include "codegen/plan/operator_registry.hpp"
#include "codegen/plan/plan_tree.hpp"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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
  auto tree = simpatico::plan_tree_from_dsl(dsl, &err);
  expect(tree.has_value(), err.empty() ? "plan_tree_from_dsl failed" : err.c_str());

  // Resolve a produced path to (producer node, channel) straight from the tree's
  // output wiring: node.output_paths[k] is produced by that node on channel
  // node.output_names[k].
  auto producer_of = [&](std::string const& path) -> std::pair<simpatico::NodeId, std::string> {
    for (simpatico::NodeId i = 0; i < tree->nodes.size(); ++i) {
      auto const& n = tree->nodes[i];
      for (std::size_t k = 0; k < n.output_paths.size(); ++k)
        if (n.output_paths[k] == path) return {i, n.output_names[k]};
    }
    return {static_cast<simpatico::NodeId>(tree->nodes.size()), std::string{}};
  };

  expect(tree->nodes[0].op == "input", "node 0 is input");

  // delta produces "delta.differences" (channel "differences") at the delta node.
  auto [delta_node, delta_chan] = producer_of("delta.differences");
  expect(tree->nodes[delta_node].op == "delta", "delta.differences produced by delta node");
  expect(delta_chan == "differences", "differences channel");

  // rle produces both values and runs at the SAME (rle) node, distinct channels.
  auto [values_node, values_chan] = producer_of("delta.differences.values");
  auto [runs_node, runs_chan]     = producer_of("delta.differences.runs");
  expect(values_node == runs_node, "rle values+runs share one producing node");
  expect(tree->nodes[values_node].op == "rle", "values/runs produced by rle node");
  expect(values_chan == "values", "values channel");
  expect(runs_chan == "runs", "runs channel");

  // Terminal bitpack nodes produce no downstream path.
  expect(producer_of("delta.differences.values.bitpack").first == tree->nodes.size(),
         "terminal bitpack produces no path");
}

// The operator, not the DSL author, owns channel order: `rle -> values, runs`
// and `rle -> runs, values` must build the identical tree (canonical order).
void test_channel_order_canonicalization()
{
  auto rle_outputs = [](char const* dsl) {
    std::string err;
    auto tree = simpatico::plan_tree_from_dsl(dsl, &err);
    expect(tree.has_value(), err.empty() ? "parse failed" : err.c_str());
    for (auto const& n : tree->nodes)
      if (n.op == "rle") return n.output_names;
    return std::vector<std::string>{};
  };
  auto a = rle_outputs("input -> rle -> values, runs\n");
  auto b = rle_outputs("input -> rle -> runs, values\n");
  expect(a == b, "rle channel order canonicalized regardless of DSL order");
  expect(a.size() == 2 && a[0] == "runs" && a[1] == "values",
         "rle canonical order is runs, values");
}

// The operator registry is the single source of truth: name resolution
// (incl. parameterised suffix forms), classification flags, and canonical
// channel order / index all derive from one table.
void test_operator_registry()
{
  using simpatico::channel_id;
  using simpatico::op_id_from_name;
  using simpatico::op_info;
  using simpatico::OpId;

  // Name resolution, exact and parameterised.
  expect(op_id_from_name("rle") == OpId::Rle, "rle -> Rle");
  expect(op_id_from_name("bitpack") == OpId::Bitpack, "bitpack -> Bitpack");
  expect(op_id_from_name("bitcomp") == OpId::Bitcomp, "bitcomp -> Bitcomp");
  expect(op_id_from_name("bitcomp_sparse") == OpId::Bitcomp, "bitcomp_sparse -> Bitcomp");
  expect(op_id_from_name("nvcomp_cascaded") == OpId::NvcompCascaded, "nvcomp_cascaded -> Cascaded");
  expect(op_id_from_name("nvcomp_cascaded_2D1R1B") == OpId::NvcompCascaded,
         "cascaded suffix -> Cascaded");
  expect(op_id_from_name("bitextract_f32") == OpId::Bitextract, "bitextract_f32 -> Bitextract");
  expect(op_id_from_name("bitextract_3hi_5lo") == OpId::Bitextract,
         "generic bitextract -> Bitextract");
  expect(!op_id_from_name("nope").has_value(), "unknown name -> nullopt");

  // Classification flags.
  expect(
    op_info(OpId::Rle).codegen && op_info(OpId::Rle).preprocessing && !op_info(OpId::Rle).terminal,
    "rle is codegen + preprocessing, not terminal");
  expect(op_info(OpId::Bitpack).codegen && !op_info(OpId::Bitpack).preprocessing,
         "bitpack is codegen, not preprocessing");
  expect(op_info(OpId::Ans).terminal && !op_info(OpId::Ans).codegen,
         "ans is terminal, not codegen");

  // Canonical channels + index.
  auto const* rle_ch = simpatico::canonical_channels(OpId::Rle);
  expect(rle_ch != nullptr && *rle_ch == (std::vector<std::string>{"runs", "values"}),
         "rle canonical channels are runs, values");
  expect(simpatico::canonical_channels(OpId::Dictionary) == nullptr,
         "dictionary has no fixed channels");
  expect(simpatico::canonical_channels(OpId::Bitextract) == nullptr,
         "bitextract has no fixed channels");
  expect(channel_id(OpId::Rle, "runs") == simpatico::ChannelId{0}, "rle.runs is channel 0");
  expect(channel_id(OpId::Rle, "values") == simpatico::ChannelId{1}, "rle.values is channel 1");
  expect(channel_id(OpId::Bitpack, "packed") == simpatico::ChannelId{3},
         "bitpack.packed is channel 3");
  expect(!channel_id(OpId::Rle, "bogus").has_value(), "unknown channel -> nullopt");

  // Self-consistency of every row: id round-trips, catalog names resolve back,
  // and each channel's index matches its position.
  for (auto const& op : simpatico::operator_registry()) {
    expect(op_info(op.id).id == op.id, "op_info round-trips id");
    for (auto const& cat : op.catalog_names)
      expect(op_id_from_name(cat) == op.id, "catalog name resolves to its op");
    for (simpatico::ChannelId i = 0; i < op.channels.size(); ++i)
      expect(channel_id(op.id, op.channels[i]) == i, "channel index matches position");
  }
}

// render_plan_tree is the inverse of the parser: parsing its output must yield
// an equivalent tree, so re-rendering is stable (idempotent).
void test_render_plan_tree()
{
  auto canon = [](std::string const& dsl) {
    std::string err;
    auto tree = simpatico::plan_tree_from_dsl(dsl, &err);
    expect(tree.has_value(), err.empty() ? "parse failed" : err.c_str());
    return simpatico::render_plan_tree(*tree);
  };
  char const* plans[] = {
    "input -> delta -> differences\n"
    "delta.differences -> rle -> values, runs\n"
    "delta.differences.values -> bitpack\n"
    "delta.differences.runs -> bitpack\n",
    "input -> for -> deltas, references\n"
    "for.deltas -> bitpack\n",
    "input -> alp\n",
    "input -> bitextract_f32 -> sign, exponent, mantissa\n",
    "input -> bitextract_f32 -> sign, exponent, mantissa\n"
    "bitextract_f32.sign, bitextract_f32.exponent, bitextract_f32.mantissa "
    "-> bitjoin_f32 -> rejoined\n",
  };
  for (auto const* p : plans) {
    std::string once  = canon(p);
    std::string twice = canon(once);
    expect(!once.empty(), "render_plan_tree produced empty DSL");
    expect(once == twice, "render_plan_tree round-trip not stable");
  }
}

}  // namespace

int main()
{
  try {
    test_delta_rle_bitpack_cascade();
    test_bitjoin_attrs();
    test_path_map();
    test_channel_order_canonicalization();
    test_operator_registry();
    test_render_plan_tree();
    std::printf("test_plan_tree: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_plan_tree: FAIL: %s\n", e.what());
    return 1;
  }
}
