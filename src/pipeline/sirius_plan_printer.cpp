/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pipeline/sirius_plan_printer.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace sirius::pipeline {

namespace {

/// Maximum recursion depth for tree building to prevent stack overflow from cycles.
constexpr size_t MAX_TREE_DEPTH = 100;

/// A node in the 2D render grid representing a pipeline.
struct render_node {
  size_t pipeline_id;
  std::vector<std::string> content_lines;  // operator chain as multi-line text
  size_t width;                            // width of this node's box in characters

  // Connection annotation to parent (port name + barrier type)
  std::string connection_label;  // e.g., "build [FULL]" or "default [PIPELINE]"
};

/// A 2D grid of optional render nodes: grid[y][x].
struct render_tree {
  size_t width;   // number of columns
  size_t height;  // number of rows
  // grid[y][x] = optional node at that position
  std::vector<std::vector<std::optional<render_node>>> grid;
  // Map: (y, x) -> list of child x positions in row y+1
  // Used to draw horizontal branch connectors for multi-child parents
  std::map<std::pair<size_t, size_t>, std::vector<size_t>> children_map;
};

/// Get the width (columns) and height (rows) of the subtree rooted at a pipeline.
/// Already-visited pipelines are treated as leaf nodes (reference-only).
void get_tree_width_height(const sirius_pipeline& pipeline,
                           size_t& width,
                           size_t& height,
                           size_t depth_limit,
                           std::set<size_t>& visited)
{
  bool is_duplicate = visited.count(pipeline.get_pipeline_id()) > 0;
  visited.insert(pipeline.get_pipeline_id());

  if (depth_limit == 0 || pipeline.dependencies.empty() || is_duplicate) {
    width  = 1;
    height = 1;
    return;
  }
  width  = 0;
  height = 0;
  for (auto& dep : pipeline.dependencies) {
    size_t child_w = 0;
    size_t child_h = 0;
    get_tree_width_height(*dep, child_w, child_h, depth_limit - 1, visited);
    width += child_w;
    height = std::max(height, child_h);
  }
  height++;
}

/// Recursively place nodes into the 2D grid.
/// Already-visited pipelines render as compact reference nodes instead of full subtrees.
void place_node(render_tree& tree,
                const sirius_pipeline& pipeline,
                const sirius_pipeline* parent,
                size_t x,
                size_t y,
                size_t depth_limit,
                std::set<size_t>& visited)
{
  bool is_duplicate = visited.count(pipeline.get_pipeline_id()) > 0;
  visited.insert(pipeline.get_pipeline_id());

  render_node node;
  node.pipeline_id = pipeline.get_pipeline_id();

  // Determine connection label: scan THIS pipeline's sink next_port_after_sink
  // to find the port that delivers data to the parent (consumer) pipeline.
  // The parent's input operator is operators[0] if any, otherwise its sink.
  if (parent) {
    auto child_sink       = pipeline.get_sink();
    auto parent_ops       = parent->get_operators();
    auto* parent_input_op = !parent_ops.empty()
                              ? &parent_ops[0].get()
                              : (parent->get_sink() ? parent->get_sink().get() : nullptr);
    if (child_sink && parent_input_op) {
      for (auto& np : child_sink->get_next_ports_after_sink()) {
        if (np.next_operator != parent_input_op) { continue; }
        auto* port = np.next_operator->get_port(np.next_operator_port_name);
        if (!port) { continue; }
        node.connection_label = std::string(np.next_operator_port_name) + " [" +
                                sirius_plan_printer::barrier_type_to_string(port->type) + "]";
        break;
      }
    }
  }

  if (is_duplicate) {
    // Render a compact reference node instead of the full subtree
    node.content_lines.push_back("(-> Pipeline #" + std::to_string(pipeline.get_pipeline_id()) +
                                 ")");
    node.width      = node.content_lines[0].size() + 4;
    tree.grid[y][x] = std::move(node);
    return;
  }

  // Build content lines bottom-to-top: sink first, then operators (reversed), then source.
  // Query plans are read from the bottom, so source should be at the bottom of the box.
  auto source = pipeline.get_source();
  auto sink   = pipeline.get_sink();
  if (sink && sink != source) {
    node.content_lines.push_back(sirius_plan_printer::format_operator_with_id(*sink));
    for (auto& detail : sirius_plan_printer::get_operator_detail_lines(*sink)) {
      node.content_lines.push_back(detail);
    }
  }
  auto ops = pipeline.get_operators();
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    auto& op_ref = it->get();
    if (source && &op_ref == source.get()) { continue; }
    if (sink && &op_ref == sink.get()) { continue; }
    node.content_lines.push_back(sirius_plan_printer::format_operator_with_id(op_ref));
    for (auto& detail : sirius_plan_printer::get_operator_detail_lines(op_ref)) {
      node.content_lines.push_back(detail);
    }
  }
  if (source) {
    node.content_lines.push_back(sirius_plan_printer::format_operator_with_id(*source));
    for (auto& detail : sirius_plan_printer::get_operator_detail_lines(*source)) {
      node.content_lines.push_back(detail);
    }
  }

  // Append input-port memory-barrier annotations at the bottom of the box
  // (data flows in from the source side). For pipelines whose dependent
  // entry point is the sink (no operators[]), ports live on sink instead.
  // const_cast: get_port_ids/get_port are not marked const, but the read is
  // logically const (just inspecting registered ports).
  auto ops_for_input = pipeline.get_operators();
  auto* input_op_const =
    !ops_for_input.empty() ? &ops_for_input[0].get() : (sink ? sink.get() : nullptr);
  auto* input_op = const_cast<op::sirius_physical_operator*>(input_op_const);
  if (input_op) {
    for (auto port_id : input_op->get_port_ids()) {
      auto* port = input_op->get_port(port_id);
      if (!port) { continue; }
      std::string src_label = port->src_pipeline
                                ? "(#" + std::to_string(port->src_pipeline->get_pipeline_id()) + ")"
                                : "";
      node.content_lines.push_back("Input" + src_label + ": " +
                                   sirius_plan_printer::barrier_type_to_string(port->type));
    }
  }

  // Add pipeline ID as first content line
  node.content_lines.insert(node.content_lines.begin(),
                            "Pipeline #" + std::to_string(pipeline.get_pipeline_id()));

  // Calculate node width: max content line width + 4 (for "| " and " |" border padding)
  node.width = 0;
  for (auto& l : node.content_lines) {
    node.width = std::max(node.width, l.size());
  }
  node.width += 4;  // border padding

  tree.grid[y][x] = std::move(node);

  // Recurse into dependencies and record child column positions
  if (depth_limit > 0) {
    size_t child_x = x;
    std::vector<size_t> child_cols;
    for (auto& dep : pipeline.dependencies) {
      size_t child_w                = 0;
      size_t child_h                = 0;
      std::set<size_t> size_visited = visited;
      get_tree_width_height(*dep, child_w, child_h, depth_limit - 1, size_visited);
      child_cols.push_back(child_x);
      place_node(tree, *dep, &pipeline, child_x, y + 1, depth_limit - 1, visited);
      child_x += child_w;
    }
    if (!child_cols.empty()) { tree.children_map[{y, x}] = std::move(child_cols); }
  }
}

/// Repeat a string n times.
std::string repeat_string(const std::string& s, size_t n)
{
  std::string result;
  result.reserve(s.size() * n);
  for (size_t i = 0; i < n; i++) {
    result += s;
  }
  return result;
}

/// Render the top border of a box: ┌──...──┐
std::string render_top_border(const plan_printer_config& config, size_t inner_width)
{
  std::string result;
  result += config.LTCORNER;
  result += repeat_string(config.HORIZONTAL, inner_width);
  result += config.RTCORNER;
  return result;
}

/// Render the bottom border of a box: └──...──┘ or └──┬──┘ if has_children
std::string render_bottom_border(const plan_printer_config& config,
                                 size_t inner_width,
                                 bool has_children)
{
  std::string result;
  result += config.LDCORNER;
  if (has_children && inner_width >= 2) {
    size_t left_side  = (inner_width - 1) / 2;
    size_t right_side = inner_width - 1 - left_side;
    result += repeat_string(config.HORIZONTAL, left_side);
    result += config.TMIDDLE;
    result += repeat_string(config.HORIZONTAL, right_side);
  } else {
    result += repeat_string(config.HORIZONTAL, inner_width);
  }
  result += config.RDCORNER;
  return result;
}

/// Render a content line inside a box: │ content │
/// Pads content to inner_width characters.
std::string render_content_line(const plan_printer_config& config,
                                const std::string& content,
                                size_t inner_width)
{
  std::string result;
  result += config.VERTICAL;
  result += " ";
  result += content;
  // Pad to inner_width (the content area is inner_width - 2 for the " " padding on each side)
  if (content.size() + 2 < inner_width) {
    result += std::string(inner_width - content.size() - 2, ' ');
  }
  result += " ";
  result += config.VERTICAL;
  return result;
}

}  // unnamed namespace

// ============================================================================
// sirius_plan_printer implementation
// ============================================================================

sirius_plan_printer::sirius_plan_printer(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines)
  : pipelines_(pipelines)
{
}

std::string sirius_plan_printer::barrier_type_to_string(op::MemoryBarrierType type)
{
  switch (type) {
    case op::MemoryBarrierType::FULL: return "FULL";
    case op::MemoryBarrierType::PARTIAL: return "PARTIAL";
    case op::MemoryBarrierType::PIPELINE: return "PIPELINE";
    default: return "UNKNOWN";
  }
}

std::string sirius_plan_printer::format_operator_with_id(const op::sirius_physical_operator& op)
{
  return op.get_name() + " (id=" + std::to_string(op.get_operator_id()) + ")";
}

std::vector<std::string> sirius_plan_printer::get_operator_detail_lines(
  const op::sirius_physical_operator& op)
{
  std::vector<std::string> lines;

  // Join type annotation ("  type: INNER")
  if (op.type == op::SiriusPhysicalOperatorType::HASH_JOIN) {
    lines.push_back("  type: " +
                    duckdb::JoinTypeToString(op.Cast<op::sirius_physical_hash_join>().join_type));
  } else if (op.type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
    lines.push_back("  type: " + duckdb::JoinTypeToString(
                                   op.Cast<op::sirius_physical_nested_loop_join>().join_type));
  }

  // Scan function name annotation ("  scan: seq_scan")
  std::string scan_name;
  switch (op.type) {
    case op::SiriusPhysicalOperatorType::TABLE_SCAN:
      scan_name = op.Cast<op::sirius_physical_table_scan>().function.name;
      break;
    default: break;
  }
  if (!scan_name.empty()) { lines.push_back("  scan: " + scan_name); }

  return lines;
}

std::string sirius_plan_printer::build_operator_chain(const sirius_pipeline& pipeline)
{
  std::ostringstream ss;
  auto source = pipeline.get_source();
  auto sink   = pipeline.get_sink();
  if (source) { ss << format_operator_with_id(*source); }
  auto ops = pipeline.get_operators();
  for (auto& op_ref : ops) {
    if (source && &op_ref.get() == source.get()) { continue; }
    if (sink && &op_ref.get() == sink.get()) { continue; }
    ss << "\n" << format_operator_with_id(op_ref.get());
  }
  if (sink && sink != source) { ss << "\n" << format_operator_with_id(*sink); }
  return ss.str();
}

duckdb::shared_ptr<sirius_pipeline> sirius_plan_printer::find_root_pipeline() const
{
  for (auto& p : pipelines_) {
    auto sink = p->get_sink();
    if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) { return p; }
  }
  // Fallback: last pipeline
  if (!pipelines_.empty()) { return pipelines_.back(); }
  return nullptr;
}

// ============================================================================
// render_pipelines: compact one-line-per-pipeline view
// ============================================================================

std::string sirius_plan_printer::render_pipelines() const
{
  std::ostringstream ss;
  ss << "=== Pipeline Overview ===\n";

  for (auto& pipeline : pipelines_) {
    if (!pipeline) { continue; }
    ss << "Pipeline #" << pipeline->get_pipeline_id() << ": ";

    // Build arrow-separated operator chain with IDs
    auto source = pipeline->get_source();
    auto sink   = pipeline->get_sink();
    if (source) { ss << format_operator_with_id(*source); }
    auto ops = pipeline->get_operators();
    for (auto& op_ref : ops) {
      if (source && &op_ref.get() == source.get()) { continue; }
      if (sink && &op_ref.get() == sink.get()) { continue; }
      ss << " -> " << format_operator_with_id(op_ref.get());
    }
    if (sink && sink != source) { ss << " -> " << format_operator_with_id(*sink); }
    ss << "\n";

    // Show input ports on this pipeline's first executable operator (mirrors Output section).
    // For pipelines whose dependent end is the sink (no explicit operators), ports live on sink.
    auto* input_op = !ops.empty() ? &ops[0].get() : (sink ? sink.get() : nullptr);
    if (input_op) {
      for (auto port_id : input_op->get_port_ids()) {
        auto* port = input_op->get_port(port_id);
        if (!port) { continue; }
        ss << "  Input: ";
        if (port->src_pipeline) {
          ss << "<- Pipeline #" << port->src_pipeline->get_pipeline_id();
        } else {
          ss << "<- (no source pipeline)";
        }
        ss << " [on: " << input_op->get_name() << " (id=" << input_op->get_operator_id() << ")"
           << ", port: " << port_id << ", barrier: " << barrier_type_to_string(port->type) << "]\n";
      }
    }

    // Show dependencies
    if (!pipeline->dependencies.empty()) {
      ss << "  Dependencies:";
      for (size_t i = 0; i < pipeline->dependencies.size(); i++) {
        if (i > 0) { ss << ","; }
        ss << " Pipeline #" << pipeline->dependencies[i]->get_pipeline_id();
      }
      ss << "\n";
    }

    // Show output connections from sink
    if (sink) {
      auto& next_ports = sink->get_next_ports_after_sink();
      for (auto& np : next_ports) {
        if (!np.next_operator) { continue; }
        ss << "  Output: -> " << np.next_operator->get_name();
        ss << " [port: " << np.next_operator_port_name;
        auto* port = np.next_operator->get_port(np.next_operator_port_name);
        if (port) { ss << ", barrier: " << barrier_type_to_string(port->type); }
        ss << "]\n";
      }
    }
  }

  return ss.str();
}

// ============================================================================
// render_dag: 2D grid tree with box-drawing characters
// ============================================================================

std::string sirius_plan_printer::render_dag() const
{
  if (pipelines_.empty()) { return "=== Query Plan DAG ===\n(empty)\n"; }

  auto root = find_root_pipeline();
  if (!root) { return "=== Query Plan DAG ===\n(no root pipeline found)\n"; }

  // Step 1: Compute tree dimensions (with visited set to avoid counting shared subtrees twice)
  size_t tree_width  = 0;
  size_t tree_height = 0;
  std::set<size_t> size_visited;
  get_tree_width_height(*root, tree_width, tree_height, MAX_TREE_DEPTH, size_visited);

  // Step 2: Create 2D grid and place nodes (with visited set to deduplicate shared pipelines)
  render_tree tree;
  tree.width  = tree_width;
  tree.height = tree_height;
  tree.grid.resize(tree_height, std::vector<std::optional<render_node>>(tree_width));
  std::set<size_t> place_visited;
  place_node(tree, *root, nullptr, 0, 0, MAX_TREE_DEPTH, place_visited);

  // Step 3: Compute node_render_width (uniform width for all nodes)
  size_t node_render_width = config_.minimum_render_width;
  for (size_t y = 0; y < tree.height; y++) {
    for (size_t x = 0; x < tree.width; x++) {
      if (tree.grid[y][x].has_value()) {
        node_render_width = std::max(node_render_width, tree.grid[y][x]->width);
      }
    }
  }

  // Dynamic width adjustment: shrink if tree is too wide
  while (tree.width * node_render_width > config_.maximum_render_width) {
    if (node_render_width <= config_.minimum_render_width + 2) { break; }
    node_render_width -= 2;
  }

  // The inner width is node_render_width - 2 (minus the two vertical border chars)
  size_t inner_width = node_render_width - 2;

  // The connector center must match the ┬ position in render_bottom_border:
  //   └ (1 char) + left_side horizontal chars, where left_side = (inner_width - 1) / 2
  size_t connector_center = 1 + (inner_width - 1) / 2;

  // Step 4: Render row by row
  std::ostringstream ss;
  ss << "=== Query Plan DAG ===\n";

  for (size_t y = 0; y < tree.height; y++) {
    // Determine max content lines in this row
    size_t max_content_lines = 0;
    for (size_t x = 0; x < tree.width; x++) {
      if (tree.grid[y][x].has_value()) {
        max_content_lines = std::max(max_content_lines, tree.grid[y][x]->content_lines.size());
      }
    }

    // --- Top border layer ---
    for (size_t x = 0; x < tree.width; x++) {
      if (tree.grid[y][x].has_value()) {
        ss << render_top_border(config_, inner_width);
      } else {
        ss << std::string(node_render_width, ' ');
      }
    }
    ss << "\n";

    // --- Content layers ---
    for (size_t line_idx = 0; line_idx < max_content_lines; line_idx++) {
      for (size_t x = 0; x < tree.width; x++) {
        if (tree.grid[y][x].has_value() && line_idx < tree.grid[y][x]->content_lines.size()) {
          ss << render_content_line(config_, tree.grid[y][x]->content_lines[line_idx], inner_width);
        } else if (tree.grid[y][x].has_value()) {
          // Node exists but fewer content lines -- render empty content
          ss << render_content_line(config_, "", inner_width);
        } else {
          ss << std::string(node_render_width, ' ');
        }
      }
      ss << "\n";
    }

    // --- Bottom border layer ---
    for (size_t x = 0; x < tree.width; x++) {
      if (tree.grid[y][x].has_value()) {
        bool has_children = false;
        // Check if this node has children (nodes in y+1 that it placed)
        if (y + 1 < tree.height) {
          // Find the pipeline for this node
          for (auto& p : pipelines_) {
            if (p->get_pipeline_id() == tree.grid[y][x]->pipeline_id) {
              has_children = !p->dependencies.empty();
              break;
            }
          }
        }
        ss << render_bottom_border(config_, inner_width, has_children);
      } else {
        ss << std::string(node_render_width, ' ');
      }
    }
    ss << "\n";

    // --- Connection layer (between rows) ---
    if (y + 1 < tree.height) {
      bool has_connections = false;
      for (size_t x = 0; x < tree.width; x++) {
        if (tree.grid[y + 1][x].has_value()) {
          has_connections = true;
          break;
        }
      }

      if (has_connections) {
        // Map each column to its parent's column (for multi-child parents only).
        // Columns not in any multi-child branch are left unmapped.
        std::map<size_t, size_t> col_to_parent;  // child_col -> parent_col
        bool has_any_branch = false;
        for (size_t px = 0; px < tree.width; px++) {
          auto it = tree.children_map.find({y, px});
          if (it != tree.children_map.end() && it->second.size() > 1) {
            has_any_branch = true;
            size_t min_col = *std::min_element(it->second.begin(), it->second.end());
            size_t max_col = *std::max_element(it->second.begin(), it->second.end());
            for (size_t c = min_col; c <= max_col; c++) {
              col_to_parent[c] = px;
            }
          }
        }

        if (has_any_branch) {
          // Render branch line — each parent's branch is independent
          for (size_t x = 0; x < tree.width; x++) {
            auto owner_it = col_to_parent.find(x);
            if (owner_it != col_to_parent.end()) {
              size_t parent_col = owner_it->second;
              auto& children    = tree.children_map[{y, parent_col}];
              size_t min_col    = *std::min_element(children.begin(), children.end());
              size_t max_col    = *std::max_element(children.begin(), children.end());
              std::set<size_t> child_set(children.begin(), children.end());

              bool is_child     = child_set.count(x) > 0;
              bool is_leftmost  = (x == min_col);
              bool is_rightmost = (x == max_col);

              // Left half: horizontal if not leftmost, spaces if leftmost
              std::string left_half;
              if (!is_leftmost) {
                for (size_t i = 0; i < connector_center; i++) {
                  left_half += config_.HORIZONTAL;
                }
              } else {
                left_half = std::string(connector_center, ' ');
              }

              // Center character
              std::string center_char;
              if (is_child) {
                if (is_leftmost && is_rightmost) {
                  center_char = config_.VERTICAL;
                } else if (is_leftmost) {
                  center_char = config_.LTCORNER;
                } else if (is_rightmost) {
                  center_char = config_.RTCORNER;
                } else {
                  center_char = config_.TMIDDLE;
                }
              } else {
                center_char = config_.HORIZONTAL;
              }

              // Right half: horizontal if not rightmost, spaces if rightmost
              std::string right_half;
              size_t right_len = (connector_center + 1 < node_render_width)
                                   ? node_render_width - connector_center - 1
                                   : 0;
              if (!is_rightmost) {
                for (size_t i = 0; i < right_len; i++) {
                  right_half += config_.HORIZONTAL;
                }
              } else {
                right_half = std::string(right_len, ' ');
              }

              ss << left_half << center_char << right_half;
            } else if (tree.grid[y + 1][x].has_value()) {
              // Single-child — just │
              std::string left_pad(connector_center, ' ');
              size_t right_len = (connector_center + 1 < node_render_width)
                                   ? node_render_width - connector_center - 1
                                   : 0;
              ss << left_pad << config_.VERTICAL << std::string(right_len, ' ');
            } else {
              ss << std::string(node_render_width, ' ');
            }
          }
          ss << "\n";

          // Render vertical drops from each junction to child boxes
          for (size_t x = 0; x < tree.width; x++) {
            if (tree.grid[y + 1][x].has_value()) {
              std::string left_pad(connector_center, ' ');
              size_t right_len = (connector_center + 1 < node_render_width)
                                   ? node_render_width - connector_center - 1
                                   : 0;
              ss << left_pad << config_.VERTICAL << std::string(right_len, ' ');
            } else {
              ss << std::string(node_render_width, ' ');
            }
          }
          ss << "\n";
        } else {
          // No multi-child parents — simple vertical connectors
          for (size_t x = 0; x < tree.width; x++) {
            if (tree.grid[y + 1][x].has_value()) {
              std::string left_pad(connector_center, ' ');
              size_t right_len = (connector_center + 1 < node_render_width)
                                   ? node_render_width - connector_center - 1
                                   : 0;
              ss << left_pad << config_.VERTICAL << std::string(right_len, ' ');
            } else {
              ss << std::string(node_render_width, ' ');
            }
          }
          ss << "\n";
        }
      }
    }
  }

  return ss.str();
}

// ============================================================================
// render: combined view
// ============================================================================

std::string sirius_plan_printer::render() const { return render_pipelines() + "\n" + render_dag(); }

}  // namespace sirius::pipeline
