// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/plan_dsl.hpp"

#include "codegen/plan/operator_registry.hpp"  // op_id_from_name, canonical_channels
#include "codegen/plan/representation.hpp"     // bitextract_canonical_name, canonicalize_path

#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <sstream>
#include <vector>

namespace simpatico {
namespace {

// Fixed output-channel order per operator — the operator, not the DSL author,
// owns the channel set/order. Sourced from the operator registry; nullptr for
// variable-arity ops (bitextract's user-named fields, dictionary's mode-
// dependent channels), which are left untouched.
std::vector<std::string> const* canonical_output_channels(std::string const& op)
{
  auto id = op_id_from_name(op);
  return id ? canonical_channels(*id) : nullptr;
}

// Reorder a step's outputs into canonical order (channels not in `canon` are
// kept, appended in their original order). Pure reordering: decode matches
// channels by name, so this only makes the DSL author's ordering irrelevant —
// `rle -> values, runs` and `rle -> runs, values` build identical trees.
void canonicalize_output_order(std::vector<std::string> const& canon, plan_step& step)
{
  std::vector<std::string> names;
  std::vector<std::optional<bit_range>> ranges;
  names.reserve(step.output_names.size());
  ranges.reserve(step.output_names.size());
  std::vector<bool> taken(step.output_names.size(), false);
  for (auto const& c : canon) {
    for (std::size_t i = 0; i < step.output_names.size(); ++i) {
      if (!taken[i] && step.output_names[i] == c) {
        names.push_back(step.output_names[i]);
        ranges.push_back(step.output_ranges[i]);
        taken[i] = true;
      }
    }
  }
  for (std::size_t i = 0; i < step.output_names.size(); ++i) {
    if (!taken[i]) {
      names.push_back(step.output_names[i]);
      ranges.push_back(step.output_ranges[i]);
    }
  }
  step.output_names  = std::move(names);
  step.output_ranges = std::move(ranges);
}

bool parse_uint(std::string_view s, uint32_t* out)
{
  auto const* end = s.data() + s.size();
  auto [ptr, ec]  = std::from_chars(s.data(), end, *out);
  return ec == std::errc{} && ptr == end;
}

// Parses a token of the form `<path>` or `<path>_<hi>:<lo>`.
// On success returns true; `path_out` receives the bare path and `range_out` receives
// the parsed bit range (nullopt if the token has no `:`).
// Returns false if the token contains `:` but does not match the expected pattern.
bool split_path_and_range(std::string_view tok,
                          std::string* path_out,
                          std::optional<bit_range>* range_out)
{
  size_t colon = tok.rfind(':');
  if (colon == std::string_view::npos) {
    *path_out  = std::string(tok);
    *range_out = std::nullopt;
    return true;
  }
  std::string_view prefix = tok.substr(0, colon);
  size_t under            = prefix.rfind('_');
  if (under == std::string_view::npos || under == 0) return false;
  uint32_t hi = 0, lo = 0;
  if (!parse_uint(tok.substr(under + 1, colon - under - 1), &hi)) return false;
  if (!parse_uint(tok.substr(colon + 1), &lo)) return false;
  *path_out  = std::string(tok.substr(0, under));
  *range_out = bit_range{hi, lo};
  return true;
}

std::string trim_copy(std::string_view s)
{
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
    ++start;
  }
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
    --end;
  }
  return std::string(s.substr(start, end - start));
}

std::vector<std::string> split_by_arrow(std::string_view line)
{
  std::vector<std::string> parts;
  size_t pos = 0;
  while (pos < line.size()) {
    size_t next = line.find("->", pos);
    if (next == std::string_view::npos) {
      parts.push_back(trim_copy(line.substr(pos)));
      break;
    }
    parts.push_back(trim_copy(line.substr(pos, next - pos)));
    pos = next + 2;
  }
  return parts;
}

std::vector<std::string> split_csv(std::string_view s)
{
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t next = s.find(',', pos);
    if (next == std::string_view::npos) {
      std::string item = trim_copy(s.substr(pos));
      if (!item.empty()) { out.push_back(std::move(item)); }
      break;
    }
    std::string item = trim_copy(s.substr(pos, next - pos));
    if (!item.empty()) { out.push_back(std::move(item)); }
    pos = next + 1;
  }
  return out;
}

bool is_root_input(std::string_view path) { return path == "input"; }

// Parse a comma-separated list of `<path>[_<hi>:<lo>]` tokens into parallel
// path / range vectors. On error, populates `*error_out` with a message including
// `line_num` and `what` (e.g. "input" / "output") and returns false.
bool parse_token_list(std::string_view csv,
                      char const* what,
                      size_t line_num,
                      std::vector<std::string>* paths_out,
                      std::vector<std::optional<bit_range>>* ranges_out,
                      std::string* error_out)
{
  auto raw = split_csv(csv);
  paths_out->reserve(raw.size());
  ranges_out->reserve(raw.size());
  for (auto const& tok : raw) {
    std::string path;
    std::optional<bit_range> rng;
    if (!split_path_and_range(tok, &path, &rng) || path.empty()) {
      if (error_out) {
        *error_out =
          "line " + std::to_string(line_num) + ": malformed " + what + " token '" + tok + "'";
      }
      return false;
    }
    // Normalise bitextract path namespaces so typed and untyped references
    // resolve to the same path.
    paths_out->push_back(canonicalize_path(path));
    ranges_out->push_back(rng);
  }
  return true;
}

}  // namespace

bool parse_plan_dsl(std::string_view dsl, std::vector<plan_step>* out, std::string* error_out)
{
  if (out == nullptr) {
    if (error_out) { *error_out = "plan steps output pointer is null"; }
    return false;
  }

  std::vector<plan_step> result;
  // Tracks input paths already claimed by a step, for duplicate detection (each
  // input path may be consumed by at most one step).
  std::unordered_set<std::string> seen_inputs;
  std::istringstream stream{std::string(dsl)};
  std::string raw_line;
  size_t line_num = 0;

  while (std::getline(stream, raw_line)) {
    ++line_num;
    if (!raw_line.empty() && raw_line.back() == '\r') { raw_line.pop_back(); }

    std::string_view line_view(raw_line);
    size_t comment_pos = line_view.find('#');
    if (comment_pos != std::string_view::npos) { line_view = line_view.substr(0, comment_pos); }

    std::string line = trim_copy(line_view);
    if (line.empty()) { continue; }

    auto parts = split_by_arrow(line);
    if (parts.size() < 2 || parts.size() > 3) {
      if (error_out) {
        *error_out =
          "line " + std::to_string(line_num) + ": expected 2 or 3 sections separated by '->'";
      }
      return false;
    }

    plan_step step;
    // parts[0] may be a comma-separated list of input paths (for bitjoin)
    if (!parse_token_list(
          parts[0], "input", line_num, &step.input_paths, &step.input_ranges, error_out)) {
      return false;
    }
    step.compressor = parts[1];
    if (step.input_paths.empty() || step.compressor.empty()) {
      if (error_out) {
        *error_out = "line " + std::to_string(line_num) + ": input paths or compressor is empty";
      }
      return false;
    }

    if (parts.size() == 3) {
      if (!parse_token_list(
            parts[2], "output", line_num, &step.output_names, &step.output_ranges, error_out)) {
        return false;
      }
      if (step.output_names.empty()) {
        if (error_out) {
          *error_out = "line " + std::to_string(line_num) + ": outputs section is empty";
        }
        return false;
      }
      if (auto const* canon = canonical_output_channels(step.compressor)) {
        canonicalize_output_order(*canon, step);
      }
    }

    // Check for duplicate input paths (each input path may only appear in one step)
    for (auto const& ipath : step.input_paths) {
      if (seen_inputs.find(ipath) != seen_inputs.end()) {
        if (error_out) {
          *error_out =
            "line " + std::to_string(line_num) + ": duplicate step for input path '" + ipath + "'";
        }
        return false;
      }
    }

    // Derive output paths from a base path. Bitextract operators contribute
    // their *canonical* name (without optional type prefix) so the path
    // namespace is invariant under auto-injected types.
    std::string base_path;
    if (step.input_paths.size() == 1) {
      base_path = step.input_paths[0];
      if (is_root_input(step.input_paths[0])) {
        base_path = bitextract_canonical_name(step.compressor);
      }
    } else {
      // Multi-input (bitjoin): use compressor name as the namespace for outputs
      base_path = step.compressor;
    }

    for (auto const& output : step.output_names) {
      if (output.empty()) { continue; }
      if (base_path.empty()) {
        step.output_paths.push_back(output);
      } else {
        step.output_paths.push_back(base_path + "." + output);
      }
    }

    // Claim all input paths so later lines can detect duplicates.
    for (auto const& ipath : step.input_paths) {
      seen_inputs.insert(ipath);
    }

    result.push_back(std::move(step));
  }

  *out = std::move(result);
  if (error_out) { error_out->clear(); }
  return true;
}

namespace {

void render_token_list(std::ostringstream& out,
                       std::vector<std::string> const& names,
                       std::vector<std::optional<bit_range>> const& ranges)
{
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out << ", ";
    out << names[i];
    if (i < ranges.size() && ranges[i].has_value()) {
      out << '_' << ranges[i]->first << ':' << ranges[i]->second;
    }
  }
}

}  // namespace

std::string render_plan_steps(std::vector<plan_step> const& steps)
{
  std::ostringstream out;
  for (auto const& step : steps) {
    render_token_list(out, step.input_paths, step.input_ranges);
    out << " -> " << step.compressor;
    if (!step.output_names.empty()) {
      out << " -> ";
      render_token_list(out, step.output_names, step.output_ranges);
    }
    out << '\n';
  }
  return out.str();
}

}  // namespace simpatico
