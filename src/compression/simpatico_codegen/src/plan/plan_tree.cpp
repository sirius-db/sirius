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

std::optional<PlanTree> plan_tree_from_dsl(std::string_view dsl,
                                           std::string* error_out,
                                           PlanPathMap* path_map)
{
  std::vector<plan_step> steps;
  if (!parse_plan_dsl(dsl, &steps, error_out)) return std::nullopt;
  return plan_tree_from_steps(steps, error_out, path_map);
}

std::optional<PlanTree> plan_tree_from_steps(std::vector<plan_step> const& steps,
                                             std::string* error_out,
                                             PlanPathMap* path_map)
{
  PlanTree tree;
  tree.nodes.push_back(PlanNode{.op = "input"});

  std::unordered_map<std::string, NodeId> path_to_node;
  std::unordered_map<std::string, std::string> path_to_channel;
  path_to_node.emplace("input", 0);

  for (std::size_t step_idx = 0; step_idx < steps.size(); ++step_idx) {
    auto const& step = steps[step_idx];
    if (step.synthetic) continue;

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
    } else if (is_bitjoin_compressor(step.compressor)) {
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

    if (is_bitjoin_compressor(step.compressor)) {
      node.input_path.clear();
    } else if (step.input_paths.size() == 1) {
      node.input_path = step.input_paths[0];
    }
    node.output_names = step.output_names;
    node.output_paths = step.output_paths;
    node.input_paths  = step.input_paths;
    node.input_ranges = step.input_ranges;

    tree.nodes.push_back(std::move(node));

    for (size_t i = 0; i < step.output_paths.size(); ++i) {
      path_to_node[step.output_paths[i]]    = nid;
      path_to_channel[step.output_paths[i]] = step.output_names[i];
    }
  }

  if (path_map) {
    path_map->node    = path_to_node;
    path_map->channel = path_to_channel;
    path_map->channel.emplace("input", "input");
  }

  if (error_out) error_out->clear();
  return tree;
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
