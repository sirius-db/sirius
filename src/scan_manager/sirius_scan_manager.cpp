/*
 * Copyright 2025, Sirius Contributors.
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

#include "scan_manager/sirius_scan_manager.hpp"

#include "log/logging.hpp"
#include "planner/query.hpp"

namespace sirius::scan_manager {

sirius_scan_manager::sirius_scan_manager(exec::thread_pool_config config)
  : _config(std::move(config))
{
}

sirius_scan_manager::~sirius_scan_manager() { stop(); }

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={}",
                   query.get_pipelines().size());
}

void sirius_scan_manager::start()
{
  if (_thread_pool) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
}

void sirius_scan_manager::stop()
{
  if (!_thread_pool) { return; }
  _thread_pool->stop();
  _thread_pool.reset();
}

}  // namespace sirius::scan_manager
