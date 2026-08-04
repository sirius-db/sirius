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

#include "scan_manager/prefetching_state_manager.hpp"

#include <stdexcept>

namespace sirius::scan_manager {

prefetching_state_manager::prefetching_state_manager(config cfg) noexcept : _cfg(cfg)
{
  throw std::logic_error("prefetching_state_manager::prefetching_state_manager: not implemented");
}

void prefetching_state_manager::prepare_for_query(const sirius::planner::query& query) noexcept
{
  throw std::logic_error("prefetching_state_manager::prepare_for_query: not implemented");
}

void prefetching_state_manager::clean_up() noexcept
{
  // IMPLEMENTATION NOTE: this method is noexcept but has to log summary(), which builds a
  // std::string and can throw. Letting that escape would call std::terminate on the query
  // teardown path, so the real body must be shaped as:
  //     try { SIRIUS_LOG_TRACE("prefetching_state_manager: {}", summary()); } catch (...) {}
  // Losing one diagnostic line is always preferable to aborting the process during cleanup.
  throw std::logic_error("prefetching_state_manager::clean_up: not implemented");
}

sirius::query_id_t prefetching_state_manager::query_id() const noexcept
{
  throw std::logic_error("prefetching_state_manager::query_id: not implemented");
}

void prefetching_state_manager::update(io::cache::prefetching_stage site) noexcept
{
  throw std::logic_error("prefetching_state_manager::update: not implemented");
}

void prefetching_state_manager::on_input_created() noexcept
{
  throw std::logic_error("prefetching_state_manager::on_input_created: not implemented");
}

void prefetching_state_manager::on_input_disposed() noexcept
{
  throw std::logic_error("prefetching_state_manager::on_input_disposed: not implemented");
}

void prefetching_state_manager::on_task_queue_depleted() noexcept
{
  // IMPLEMENTATION NOTE: noexcept. The bounded split_connector::prefetch_if walk this will drive
  // acquires a mutex and runs a caller-supplied predicate, both of which can throw; contain them
  // with try/catch(...) here rather than propagating out onto the scheduler management thread.
  throw std::logic_error("prefetching_state_manager::on_task_queue_depleted: not implemented");
}

void prefetching_state_manager::on_task_not_created(const op::sirius_physical_operator* requested,
                                                    creator::request_type kind) noexcept
{
  // IMPLEMENTATION NOTE: same try/catch(...) requirement as on_task_queue_depleted, and stricter
  // still — this runs on the single task-creation thread, outside any try block, so an escaping
  // exception would silently end all task creation.
  throw std::logic_error("prefetching_state_manager::on_task_not_created: not implemented");
}

prefetching_state_manager::counters_snapshot prefetching_state_manager::snapshot() const noexcept
{
  throw std::logic_error("prefetching_state_manager::snapshot: not implemented");
}

std::string prefetching_state_manager::summary() const
{
  throw std::logic_error("prefetching_state_manager::summary: not implemented");
}

const prefetching_state_manager::config& prefetching_state_manager::get_config() const noexcept
{
  throw std::logic_error("prefetching_state_manager::get_config: not implemented");
}

}  // namespace sirius::scan_manager
