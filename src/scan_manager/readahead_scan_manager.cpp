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

#include "scan_manager/readahead_scan_manager.hpp"

#include "op/sirius_physical_operator.hpp"
#include "planner/query_index.hpp"

#include <algorithm>
#include <utility>

namespace sirius::scan_manager {

void readahead_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  auto index = planner::query_index::build_index(query, planner::build_index_options{});
  if (!index) { return; }

  std::lock_guard lock{_mutex};
  for (auto const& step : index->prefetching_orders()) {
    if (step.scan == nullptr || !step.scan->has_operator_id()) { continue; }
    auto& state     = _by_operator[step.scan->get_operator_id()];
    state.mode      = step.mode;
    state.branch_id = step.branch_id;
    state.count     = step.count;
  }
}

void readahead_scan_manager::register_scan_task(std::weak_ptr<op::scan::scan_info> task,
                                                std::size_t operator_id)
{
  std::lock_guard lock{_mutex};
  _by_operator[operator_id].tasks.push_back(std::move(task));
}

void readahead_scan_manager::update(std::size_t operator_id, io::cache::scan_stage stage)
{
  std::lock_guard lock{_mutex};
  auto& state = _by_operator[operator_id];
  state.stage = stage;
  auto& tasks = state.tasks;
  tasks.erase(
    std::remove_if(tasks.begin(),
                   tasks.end(),
                   [](std::weak_ptr<op::scan::scan_info> const& t) { return t.expired(); }),
    tasks.end());
}

void readahead_scan_manager::reset()
{
  std::lock_guard lock{_mutex};
  _by_operator.clear();
}

}  // namespace sirius::scan_manager
