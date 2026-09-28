// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/plan_tree.hpp"

#include <deque>
#include <unordered_map>

namespace simpatico {
namespace {

bool is_bitjoin_compressor(std::string const& compressor)
{
  return compressor.rfind("bitjoin", 0) == 0;
}

bool is_bitextract_compressor(std::string const& compressor)
{
  return compressor.rfind("bitextract", 0) == 0;
}

std::string bitjoin_spec_suffix(std::string const& compressor)
{
  constexpr std::string_view prefix = "bitjoin_";
  if (compressor.size() > prefix.size() && compressor.compare(0, prefix.size(), prefix) == 0) {
    return compressor.substr(prefix.size());
  }
  return {};
}

}  // namespace

std::optional<PlanTree> plan_tree_from_dsl(std::string_view dsl, std::string* error_out)
{
  std::vector<plan_step> steps;
  if (!parse_plan_dsl(dsl, &steps, error_out)) return std::nullopt;
  return plan_tree_from_steps(steps, error_out);
}

std::optional<PlanTree> plan_tree_from_steps(std::vector<plan_step> const& steps,
                                             std::string* error_out)
{
  PlanTree tree;
  tree.nodes.push_back(PlanNode{.op = "input"});

  std::unordered_map<std::string, NodeId> path_to_node;
  std::unordered_map<std::string, std::string> path_to_channel;
  path_to_node.emplace("input", 0);

  for (std::size_t step_idx = 0; step_idx < steps.size(); ++step_idx) {
    auto const& step = steps[step_idx];

    NodeId const nid = static_cast<NodeId>(tree.nodes.size());
    PlanNode node;
    node.op = step.compressor;

    if (is_bitextract_compressor(step.compressor)) {
      if (auto suffix = strip_bitextract_prefix(step.compressor)) {
        auto spec = parse_bitextract_spec(*suffix);
        if (!spec.fields.empty()) {
          node.attrs.bitextract = bitextract_attrs{spec.fields, spec.output_type};
        }
      }
    }

    if (is_bitjoin_compressor(step.compressor)) {
      auto spec = parse_bitjoin_spec(bitjoin_spec_suffix(step.compressor));
      bitjoin_attrs bj;
      bj.output_type = spec.output_type;
      bj.inputs.reserve(step.input_paths.size());
      for (size_t i = 0; i < step.input_paths.size(); ++i) {
        auto const& path = step.input_paths[i];
        auto it          = path_to_node.find(path);
        if (it == path_to_node.end()) {
          if (error_out) {
            *error_out = "plan_tree: unknown input path '" + path + "' for bitjoin";
          }
          return std::nullopt;
        }
        std::string channel = (path == "input") ? "input" : path_to_channel.at(path);
        bj.inputs.push_back(bitjoin_input_ref{it->second, channel, step.input_ranges[i]});
        tree.nodes[it->second].children.push_back(PlanEdge{channel, nid});
      }
      node.attrs.bitjoin = std::move(bj);
    } else if (step.input_paths.size() == 1) {
      auto const& parent_path = step.input_paths[0];
      auto pit                = path_to_node.find(parent_path);
      if (pit == path_to_node.end()) {
        if (error_out) { *error_out = "plan_tree: unknown input path '" + parent_path + "'"; }
        return std::nullopt;
      }
      std::string channel = (parent_path == "input") ? bitextract_canonical_name(step.compressor)
                                                     : path_to_channel.at(parent_path);
      tree.nodes[pit->second].children.push_back(PlanEdge{channel, nid});
    } else {
      if (error_out) {
        *error_out = "plan_tree: multi-input step is not bitjoin: " + step.compressor;
      }
      return std::nullopt;
    }

    node.output_names = step.output_names;
    node.output_paths = step.output_paths;

    tree.nodes.push_back(std::move(node));

    for (size_t i = 0; i < step.output_paths.size(); ++i) {
      path_to_node[step.output_paths[i]]    = nid;
      path_to_channel[step.output_paths[i]] = step.output_names[i];
    }
  }

  compute_input_sources(tree);

  if (error_out) error_out->clear();
  return tree;
}

namespace {

// Output-port index of `channel` on producing node `node`. Node 0 (input) has
// no named outputs; its single value is port 0.
ChannelId port_of(PlanTree const& tree, NodeId node, std::string const& channel)
{
  if (node != 0) {
    auto const& names = tree.nodes[node].output_names;
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == channel) return static_cast<ChannelId>(i);
    }
  }
  return 0;
}

}  // namespace

void compute_input_sources(PlanTree& tree)
{
  // A bitjoin lists its inputs (with order) in its attrs; every other op has a
  // single in-edge, found by scanning parents for the edge that targets it.
  for (NodeId nid = 0; nid < tree.nodes.size(); ++nid) {
    PlanNode& node = tree.nodes[nid];
    node.input_sources.clear();
    if (node.attrs.bitjoin.has_value()) {
      for (auto const& ref : node.attrs.bitjoin->inputs) {
        node.input_sources.push_back(ValueId{ref.node, port_of(tree, ref.node, ref.channel)});
      }
      continue;
    }
    for (NodeId parent = 0; parent < tree.nodes.size(); ++parent) {
      for (auto const& e : tree.nodes[parent].children) {
        if (e.child == nid) {
          node.input_sources.push_back(ValueId{parent, port_of(tree, parent, e.channel)});
        }
      }
    }
  }
}

namespace {

// DSL dotted path of the value produced at (v.node, v.channel), matching
// parse_plan_dsl's output-path scheme: the root input is "input"; a single-
// input op namespaces its outputs under its own input path (or, when fed by
// the root, under its canonical name); a bitjoin namespaces under its op name.
std::string value_dsl_path(PlanTree const& tree, ValueId v)
{
  if (v.node == 0 || v.node >= tree.nodes.size()) return "input";
  PlanNode const& p = tree.nodes[v.node];
  std::string base;
  if (p.attrs.bitjoin.has_value()) {
    base = p.op;
  } else {
    ValueId const src = p.input_sources.empty() ? ValueId{0, 0} : p.input_sources.front();
    base = (src.node == 0) ? bitextract_canonical_name(p.op) : value_dsl_path(tree, src);
  }
  std::string const chan =
    v.channel < p.output_names.size() ? p.output_names[v.channel] : std::string{};
  if (base.empty()) return chan;
  return chan.empty() ? base : base + "." + chan;
}

}  // namespace

std::string render_plan_tree(PlanTree const& tree)
{
  std::vector<plan_step> steps;
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    PlanNode const& node = tree.nodes[nid];
    plan_step s;
    s.compressor = node.op;
    if (node.attrs.bitjoin.has_value()) {
      auto const& bj = *node.attrs.bitjoin;
      for (std::size_t j = 0; j < node.input_sources.size(); ++j) {
        s.input_paths.push_back(value_dsl_path(tree, node.input_sources[j]));
        s.input_ranges.push_back(j < bj.inputs.size() ? bj.inputs[j].range : std::nullopt);
      }
    } else {
      ValueId const src = node.input_sources.empty() ? ValueId{0, 0} : node.input_sources.front();
      s.input_paths.push_back(value_dsl_path(tree, src));
      s.input_ranges.push_back(std::nullopt);
    }
    s.output_names = node.output_names;
    s.output_ranges.assign(node.output_names.size(), std::nullopt);
    steps.push_back(std::move(s));
  }
  return render_plan_steps(steps);
}

std::string dotted_label(PlanTree const& tree, NodeId node)
{
  if (node >= tree.nodes.size()) return {};
  if (node == 0) return "input";

  std::unordered_map<NodeId, std::string> labels;
  labels[0] = "input";
  std::deque<NodeId> q{0};
  while (!q.empty()) {
    NodeId const id = q.front();
    q.pop_front();
    if (id == node) return labels[id];
    for (auto const& edge : tree.nodes[id].children) {
      if (labels.find(edge.child) != labels.end()) continue;
      labels[edge.child] = labels[id] + "." + edge.channel;
      q.push_back(edge.child);
    }
  }
  return {};
}

}  // namespace simpatico
