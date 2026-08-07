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

#include "io/cache/types.hpp"
#include "planner/query_index.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
class scan_info;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/// Per-query readahead bookkeeping for GPU scans.  Seeded from the query's
/// prefetching order, then driven by the scans as they advance.
class readahead_scan_manager {
 public:
  readahead_scan_manager()                                         = default;
  readahead_scan_manager(readahead_scan_manager const&)            = delete;
  readahead_scan_manager& operator=(readahead_scan_manager const&) = delete;

  /// Seed the per-operator readahead order for @p query.
  void prepare_for_query(const sirius::planner::query& query);

  /// Record a scan task under the operator that produced it.
  void register_scan_task(std::weak_ptr<op::scan::scan_info> task, std::size_t operator_id);

  /// Advance the stage reported for @p operator_id.
  void update(std::size_t operator_id, io::cache::scan_stage stage);

  void reset();

 private:
  struct operator_state {
    planner::prefetching_mode mode{planner::prefetching_mode::pipeline};
    std::size_t branch_id{0};
    std::size_t count{0};
    io::cache::scan_stage stage{io::cache::scan_stage::none};
    std::vector<std::weak_ptr<op::scan::scan_info>> tasks;
  };

  mutable std::mutex _mutex;
  std::unordered_map<std::size_t, operator_state> _by_operator;
};

}  // namespace sirius::scan_manager
