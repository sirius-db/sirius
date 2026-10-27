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

#include <google/protobuf/message.h>

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace sirius::planner {

namespace {

/// `file:` scheme variants (`file:/p`, `file:///p`, `file://localhost/p`) all name the local
/// filesystem; the plan may carry one form (the FE's) while DuckDB's bind reports another (or
/// the plain path). One canonical form — the plain path — keyed at insert AND lookup is what
/// keeps a registered range from being silently unclaimed over a spelling difference.
std::string canonical_scan_path(const std::string& path)
{
  constexpr std::string_view kScheme = "file:";
  if (path.rfind(kScheme, 0) != 0) { return path; }
  auto rest = path.substr(kScheme.size());
  if (rest.rfind("//", 0) == 0) {
    // Drop the authority (empty or localhost — anything else never reaches this planner).
    auto const slash = rest.find('/', 2);
    if (slash == std::string::npos) { return path; }
    rest = rest.substr(slash);
  }
  return rest;
}

}  // namespace

bool scan_byte_ranges_state::has(const std::string& path) const
{
  return _ranges.count(canonical_scan_path(path)) > 0;
}

std::vector<scan_byte_range> scan_byte_ranges_state::claim(const std::string& path)
{
  auto it = _ranges.find(canonical_scan_path(path));
  if (it == _ranges.end()) { return {}; }
  if (!_claimed.insert(it->first).second) {
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

/// A non-`file` URI scheme (`s3://`, `gs://`, `hdfs://`, ...). Byte ranges are only defined for
/// local parquet files; remote reads have their own footer and prefetch lifecycle.
// The protobuf bundled with the Substrait reader lives under duckdb::.
namespace pb = duckdb::google::protobuf;

bool has_remote_scheme(const std::string& canonical_path)
{
  return canonical_path.find("://") != std::string::npos;
}

/// What one ReadRel says about one file: read whole, and/or these ranges.
struct read_items {
  bool whole = false;
  std::vector<scan_byte_range> ranges;
};

/// Every ReadRel's LocalFiles items, grouped per canonical path. Visited by protobuf reflection,
/// so a read nested under any rel type (or inside an expression) is seen without an exhaustive
/// rel-type switch, and a plan with no ranges is never refused for its shape.
void visit_reads(const pb::Message& message, std::vector<std::map<std::string, read_items>>& reads)
{
  if (auto const* read = dynamic_cast<const substrait::ReadRel*>(&message);
      read != nullptr && read->has_local_files()) {
    auto& files = reads.emplace_back();
    for (const auto& item : read->local_files().items()) {
      bool const ranged = item.start() != 0 || item.length() != 0;
      if (!item.has_uri_file()) {
        if (ranged) {
          throw sirius::invalid_input_exception(
            "a byte-ranged LocalFiles item uses a path type other than uri_file; its range "
            "cannot be attributed to a file");
        }
        continue;  // a whole-directory/glob read; its files cannot be named here
      }
      auto& entry = files[canonical_scan_path(item.uri_file())];
      if (!ranged) {
        entry.whole = true;
        continue;
      }
      if (item.length() > std::numeric_limits<std::uint64_t>::max() - item.start()) {
        throw sirius::invalid_input_exception(
          "byte range ({}, {}) of '{}' overflows a 64-bit offset",
          item.start(),
          item.length(),
          item.uri_file());
      }
      entry.ranges.emplace_back(item.start(), item.length());
    }
  }
  auto const* reflection = message.GetReflection();
  std::vector<const pb::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (auto const* field : fields) {
    if (field->cpp_type() != pb::FieldDescriptor::CPPTYPE_MESSAGE) { continue; }
    if (field->is_repeated()) {
      for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
        visit_reads(reflection->GetRepeatedMessage(message, field, i), reads);
      }
    } else {
      visit_reads(reflection->GetMessage(message, field), reads);
    }
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
  std::vector<std::map<std::string, read_items>> reads;
  visit_reads(plan, reads);

  // Ranges are claimed per path after DuckDB binds the plan, so a path's ranges can be
  // attributed to its read only when exactly one ReadRel names that path. With two, the
  // optimizer may drop one read (its ranges then land on the other) or merge both into one
  // shared subplan (one claim, rows returned twice).
  std::map<std::string, std::size_t> reads_per_path;
  for (const auto& files : reads) {
    for (const auto& [path, items] : files) {
      ++reads_per_path[path];
    }
  }
  std::map<std::string, std::vector<scan_byte_range>> out;
  for (const auto& files : reads) {
    for (const auto& [path, items] : files) {
      if (items.ranges.empty()) { continue; }
      if (has_remote_scheme(path)) {
        throw sirius::invalid_input_exception(
          "byte-range splits are only supported on local parquet files, not '{}'", path);
      }
      if (items.whole) {
        throw sirius::invalid_input_exception(
          "a read names '{}' both whole and by byte range; one scan cannot read a file both ways",
          path);
      }
      if (reads_per_path[path] > 1) {
        throw sirius::invalid_input_exception(
          "'{}' is read by byte range in a plan that reads it {} times; the ranges cannot be "
          "attributed to one read",
          path,
          reads_per_path[path]);
      }
      out[path] = items.ranges;
    }
  }
  return out;
}

}  // namespace sirius::planner
