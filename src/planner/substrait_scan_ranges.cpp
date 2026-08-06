/*
 * Copyright 2026, Sirius Contributors.
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

#include "planner/substrait_scan_ranges.hpp"

#include "sirius/exception.hpp"
#include "substrait/plan.pb.h"

namespace sirius::planner {

std::vector<scan_byte_range> scan_byte_ranges_state::claim(const std::string& path)
{
  auto it = _ranges.find(path);
  if (it == _ranges.end()) { return {}; }
  if (!_claimed.insert(path).second) {
    throw sirius::invalid_input_exception(
      "two scans in one plan read byte ranges of '{}'; the ranges cannot be attributed to "
      "either without double-reading",
      path);
  }
  return it->second;
}

void scan_byte_ranges_state::assert_all_consumed() const
{
  for (const auto& [path, ranges] : _ranges) {
    if (_claimed.count(path) == 0) {
      throw sirius::invalid_input_exception(
        "the plan carries {} byte range(s) for '{}' that no scan consumed; executing anyway "
        "would read whole files and duplicate rows across splits",
        ranges.size(),
        path);
    }
  }
}

namespace {

void collect_from_rel(const substrait::Rel& rel,
                      std::map<std::string, std::vector<scan_byte_range>>& out);

void collect_from_read(const substrait::ReadRel& read,
                       std::map<std::string, std::vector<scan_byte_range>>& out)
{
  if (!read.has_local_files()) { return; }
  for (const auto& item : read.local_files().items()) {
    if (item.start() == 0 && item.length() == 0) { continue; }  // whole file
    if (!item.has_uri_file()) {
      throw sirius::invalid_input_exception(
        "a byte-ranged LocalFiles item uses a path type other than uri_file; its range "
        "cannot be attributed to a file");
    }
    out[item.uri_file()].emplace_back(item.start(), item.length());
  }
}

void collect_from_rel(const substrait::Rel& rel,
                      std::map<std::string, std::vector<scan_byte_range>>& out)
{
  switch (rel.rel_type_case()) {
    case substrait::Rel::kRead: collect_from_read(rel.read(), out); return;
    case substrait::Rel::kFilter: collect_from_rel(rel.filter().input(), out); return;
    case substrait::Rel::kFetch: collect_from_rel(rel.fetch().input(), out); return;
    case substrait::Rel::kAggregate: collect_from_rel(rel.aggregate().input(), out); return;
    case substrait::Rel::kSort: collect_from_rel(rel.sort().input(), out); return;
    case substrait::Rel::kProject: collect_from_rel(rel.project().input(), out); return;
    case substrait::Rel::kJoin:
      collect_from_rel(rel.join().left(), out);
      collect_from_rel(rel.join().right(), out);
      return;
    case substrait::Rel::kCross:
      collect_from_rel(rel.cross().left(), out);
      collect_from_rel(rel.cross().right(), out);
      return;
    case substrait::Rel::kSet:
      for (const auto& input : rel.set().inputs()) {
        collect_from_rel(input, out);
      }
      return;
    case substrait::Rel::REL_TYPE_NOT_SET: return;
    default:
      // Skipping an unknown rel could hide a ranged read inside it, and a hidden ranged
      // read degrades to a whole-file scan that duplicates rows across splits.
      throw sirius::invalid_input_exception(
        "byte-range extraction does not know Substrait rel type {}; refusing to guess "
        "whether it hides a ranged read",
        static_cast<int>(rel.rel_type_case()));
  }
}

}  // namespace

std::map<std::string, std::vector<scan_byte_range>> extract_scan_byte_ranges(
  const std::string& plan_bytes)
{
  substrait::Plan plan;
  if (!plan.ParseFromString(plan_bytes)) {
    throw sirius::invalid_input_exception(
      "failed to parse the Substrait plan while extracting scan byte ranges");
  }
  std::map<std::string, std::vector<scan_byte_range>> out;
  for (const auto& plan_rel : plan.relations()) {
    if (plan_rel.has_root() && plan_rel.root().has_input()) {
      collect_from_rel(plan_rel.root().input(), out);
    } else if (plan_rel.has_rel()) {
      collect_from_rel(plan_rel.rel(), out);
    }
  }
  return out;
}

}  // namespace sirius::planner
