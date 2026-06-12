// SPDX-License-Identifier: Apache-2.0
#ifndef SIMPATICO_PLAN_DSL_HPP
#define SIMPATICO_PLAN_DSL_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simpatico {

// (hi, lo) source-bit range parsed from a `<path>_<hi>:<lo>` token.
using bit_range = std::pair<uint32_t, uint32_t>;

struct plan_step {
  std::vector<std::string> input_paths;  // one or more input column paths (bitjoin uses >1)
  // Per-input source bit range (parallel to input_paths). nullopt = default (bits [n_bits-1:0]).
  std::vector<std::optional<bit_range>> input_ranges;
  std::string compressor;
  std::vector<std::string> output_names;  // raw names from DSL line (e.g. values, runs)
  std::vector<std::string> output_paths;  // fully-qualified paths derived from input_paths
  std::vector<std::optional<bit_range>> output_ranges;  // parallel to output_names
  // True when `parse_plan_dsl` injected this step to drain an unconsumed
  // output path (always `identity` with no outputs). Distinguishes those
  // auto-generated leaves from user-written leaf-only steps like
  // `input -> lc_sp_speed`, which must round-trip through the renderer.
  bool synthetic = false;
};

/// Parse line-based DSL into a flat list of plan steps (with synthetic identity
/// drains appended for unconsumed output paths). Returns false on error;
/// error_out (if non-null) gets a message. This is a parse-time intermediate;
/// the canonical runtime structure is the PlanTree (plan_tree_from_steps).
bool parse_plan_dsl(std::string_view dsl, std::vector<plan_step>* out, std::string* error_out);

/// Render parsed plan steps back to DSL text (`input -> compressor -> outputs`
/// per line). Skips the synthetic identity leaf steps that `parse_plan_dsl`
/// appends. Output re-parses with `parse_plan_dsl` to an equivalent step list
/// (modulo whitespace/comments).
std::string render_plan_steps(std::vector<plan_step> const& steps);

}  // namespace simpatico

#endif  // SIMPATICO_PLAN_DSL_HPP
