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
#include "exec/config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <filesystem>

namespace sirius {

namespace config {
struct configuration_setter;
}  // namespace config

struct sirius_config {
  sirius_config();
  ~sirius_config() = default;

  void load_from_file(const std::filesystem::path& config_path);

  [[nodiscard]] const cucascade::memory::system_topology_info& get_hw_topology() const noexcept
  {
    return _hw_topology;
  }

  [[nodiscard]] const std::vector<cucascade::memory::memory_space_config>&
  get_memory_space_configs() const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_task_creator_config() const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_gpu_pipeline_executor_config() const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_downgrade_executor_config() const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_duckdb_scan_executor_config() const noexcept;

  [[nodiscard]] bool is_scan_caching_enabled() const noexcept { return enable_scan_caching_; }

 private:
  cucascade::memory::system_topology_info _hw_topology;
  std::vector<cucascade::memory::memory_space_config> _memory_space_configs;
  exec::thread_pool_config _task_creator_config{.num_threads        = 2,
                                                .thread_name_prefix = "task_creator"};
  exec::thread_pool_config _gpu_pipeline_executor_config{.num_threads        = 4,
                                                         .thread_name_prefix = "gpu_pipeline"};
  exec::thread_pool_config _downgrade_executor_config{.num_threads        = 4,
                                                      .thread_name_prefix = "downgrade"};
  exec::thread_pool_config _duckdb_scan_executor_config{.num_threads        = 4,
                                                        .thread_name_prefix = "duckdb_scan"};
  bool enable_scan_caching_ = false;
};

}  // namespace sirius
