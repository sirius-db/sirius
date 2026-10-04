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
#include "scan_manager/preparation_ledger.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
// For deferred results:
// Positions use the charged allocation itself; no separately allocated vector or shadow charge.
// Publish only shared_ptr<const iceberg_delete_set>. The backing outlives every consumer.
struct iceberg_delete_set {
  explicit iceberg_delete_set(std::string file) : data_file(std::move(file)) {}
  // Legacy positions alias the completed payload; they are outside the preparation ledger.
  iceberg_delete_set(std::string file, std::shared_ptr<std::vector<int64_t> const> positions_owner)
    : data_file(std::move(file)), legacy_positions_(std::move(positions_owner))
  {
    if (!legacy_positions_) throw std::invalid_argument("legacy delete positions require backing");
    positions = *legacy_positions_;
  }
  iceberg_delete_set(std::string file,
                     scan_manager::charged_block backing,
                     uint64_t count,
                     uint64_t sources)
    : data_file(std::move(file)), source_dv_count(sources), backing_(std::move(backing))
  {
    if (count && (!backing_ || !backing_.retained()))
      throw std::invalid_argument("delete positions require retained backing");
    if (count > backing_.size() / sizeof(int64_t))
      throw std::invalid_argument("delete positions exceed charged backing");
    if (count && reinterpret_cast<uintptr_t>(backing_.data()) % alignof(int64_t))
      throw std::invalid_argument("delete positions backing is misaligned");
    positions = {reinterpret_cast<int64_t const*>(backing_.data()), static_cast<size_t>(count)};
    if (!std::is_sorted(positions.begin(), positions.end()) ||
        std::adjacent_find(positions.begin(), positions.end()) != positions.end() ||
        (!positions.empty() && positions.front() < 0))
      throw std::invalid_argument("delete positions must be sorted, unique and nonnegative");
  }
  std::string data_file;
  std::span<int64_t const> positions;
  uint64_t source_dv_count = 0;

 private:
  scan_manager::charged_block backing_;
  std::shared_ptr<std::vector<int64_t> const> legacy_positions_;
};
using iceberg_delete_sets =
  std::unordered_map<std::string, std::shared_ptr<iceberg_delete_set const>>;
}  // namespace sirius::op::scan
