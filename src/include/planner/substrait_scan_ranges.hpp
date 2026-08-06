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

#pragma once

#include <duckdb/main/client_context_state.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sirius::planner {

using scan_byte_range = std::pair<std::uint64_t, std::uint64_t>;

/**
 * @brief Per-plan parquet byte ranges, carried out-of-band from the Substrait plan.
 *
 * A distributed split scan encodes each split as `FileOrFiles.start/length` in the plan's
 * LocalFiles read. DuckDB's Substrait consumer and its `parquet_scan` binding have no byte-range
 * parameter, so the range cannot ride the relation tree: `lower_substrait` extracts every range
 * from the plan bytes into this state, and `build_parquet_table_info` claims them when it lifts
 * the scan's bind data into the GPU ingestible.
 *
 * The failure mode this state's discipline prevents is silent: a range that is emitted but not
 * applied degrades to a whole-file read, and N splits of one file then read every row N times.
 * Hence claims are per-file and single-shot (a second claim throws — two scans sharing a ranged
 * file would double-read), and `assert_all_consumed` runs after plan generation so an unclaimed
 * range is a loud error. `lower_substrait` replaces the state on every plan, so a stale registry
 * can never leak ranges into the next statement.
 */
class scan_byte_ranges_state : public duckdb::ClientContextState {
 public:
  static constexpr const char* kStateKey = "sirius_scan_byte_ranges";

  explicit scan_byte_ranges_state(std::map<std::string, std::vector<scan_byte_range>> ranges)
    : _ranges(std::move(ranges))
  {
  }

  [[nodiscard]] bool has(const std::string& path) const { return _ranges.count(path) > 0; }

  /// Claims every range registered for `path`. Single-shot per file.
  /// @throws sirius::invalid_input_exception on a second claim of the same file.
  [[nodiscard]] std::vector<scan_byte_range> claim(const std::string& path);

  /// @throws sirius::invalid_input_exception naming any file whose ranges were never claimed —
  /// the scan that should honor them would otherwise silently read whole files.
  void assert_all_consumed() const;

 private:
  std::map<std::string, std::vector<scan_byte_range>> _ranges;
  std::set<std::string> _claimed;
};

/**
 * @brief Extracts `path -> byte ranges` from every LocalFiles read in a Substrait plan.
 *
 * Only items carrying a real range (`start`/`length` not both zero) are returned; a whole-file
 * item contributes nothing. The relation walk is exhaustive over the rel types this engine can
 * receive and THROWS on an unknown one: skipping it could hide a ranged read, and a hidden
 * ranged read silently duplicates rows.
 */
[[nodiscard]] std::map<std::string, std::vector<scan_byte_range>> extract_scan_byte_ranges(
  const std::string& plan_bytes);

}  // namespace sirius::planner
