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

#include "config.hpp"
#include "memory/config.hpp"
#include "parallel/config.hpp"

#include <filesystem>

namespace sirius {

namespace config {
struct configuration_setter;
}  // namespace config

struct sirius_config {
  sirius_config()  = default;
  ~sirius_config() = default;

  void load_from_file(const std::filesystem::path& config_path);

  [[nodiscard]] size_t get_task_creator_thread_count() const noexcept
  {
    return _task_creator_thread_count;
  }

  [[nodiscard]] const std::vector<cucascade::memory::memory_space_config>&
  get_memory_space_configs() const noexcept;

  [[nodiscard]] const parallel::task_executor_config& get_gpu_pipeline_executor_config()
    const noexcept;

  [[nodiscard]] const parallel::task_executor_config& get_downgrade_executor_config()
    const noexcept;

  [[nodiscard]] const parallel::task_executor_config& get_duckdb_scan_executor_config()
    const noexcept;

 private:
  std::vector<cucascade::memory::memory_space_config> _memory_space_configs;
  parallel::task_executor_config _gpu_pipeline_executor_config{.num_threads    = 4,
                                                               .retry_on_error = true};
  parallel::task_executor_config _downgrade_executor_config{.num_threads    = 4,
                                                            .retry_on_error = false};
  parallel::task_executor_config _duckdb_scan_executor_config{.num_threads    = 4,
                                                              .retry_on_error = false};
  size_t _task_creator_thread_count = 4;
};

}  // namespace sirius
