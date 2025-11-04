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

#pragma once

// sirius
#include <parallel/task_executor.hpp>
#include <scan/duckdb_scan_task_queue.hpp>

namespace sirius::parallel {

/**
 * @brief A task executor for duckdb scan tasks.
 *
 * This class extends the generic itask_executor simply by instantiating it with a
 * duckdb_scan_task_queue.
 *
 */
class duckdb_scan_executor : public itask_executor {
 public:
  explicit duckdb_scan_executor(task_executor_config config)
    : itask_executor(make_unique<duckdb_scan_task_queue>(), config)
  {
  }

  int get_num_threads() const { return _config.num_threads; }
};

}  // namespace sirius::parallel