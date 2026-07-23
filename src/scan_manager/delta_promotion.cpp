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

#include "scan_manager/delta_promotion.hpp"

#include <algorithm>
#include <utility>

namespace sirius::scan_manager {

bool promotion_sink::try_begin_capture(std::string const& entry_name, duckdb::idx_t first_row_group)
{
  std::lock_guard<std::mutex> guard(mutex_);
  return seen_.emplace(entry_name, first_row_group).second;
}

void promotion_sink::add(std::string const& entry_name, promotion_captured_slice slice)
{
  std::lock_guard<std::mutex> guard(mutex_);
  captures_[entry_name].slices.push_back(std::move(slice));
}

void promotion_sink::record_skip(std::string const& entry_name, std::string reason)
{
  std::lock_guard<std::mutex> guard(mutex_);
  captures_[entry_name].last_skip_reason = std::move(reason);
}

bool promotion_sink::empty() const
{
  std::lock_guard<std::mutex> guard(mutex_);
  return captures_.empty();
}

std::unordered_map<std::string, promotion_sink::entry_capture> promotion_sink::take_all()
{
  std::lock_guard<std::mutex> guard(mutex_);
  auto out = std::move(captures_);
  captures_.clear();
  seen_.clear();
  return out;
}

std::vector<promotion_captured_slice> select_promotion_prefix(
  std::vector<promotion_captured_slice> slices,
  std::size_t n_cache,
  std::vector<promotion_captured_slice>& dropped)
{
  std::sort(slices.begin(), slices.end(), [](auto const& a, auto const& b) {
    return a.first_rowid < b.first_rowid;
  });

  std::vector<promotion_captured_slice> selected;
  std::size_t expected = n_cache;
  for (auto& slice : slices) {
    // Advance the base prefix only by a slice that begins exactly where it
    // currently ends. Once a gap appears every later slice starts even higher
    // (sorted), so nothing else can match — the base stays one unbroken run.
    if (slice.first_rowid == expected && slice.row_count != 0) {
      expected += slice.row_count;
      selected.push_back(std::move(slice));
    } else {
      dropped.push_back(std::move(slice));
    }
  }
  return selected;
}

}  // namespace sirius::scan_manager
