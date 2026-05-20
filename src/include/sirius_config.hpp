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
#include "op/scan/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <filesystem>

namespace sirius {

namespace config {

constexpr uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE       = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_SCAN_TASK_VARCHAR_SIZE     = 256LL;
constexpr uint64_t DEFAULT_HASH_PARTITION_BYTES       = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_CONCAT_BATCH_BYTES         = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_MAX_BUILD_HASH_TABLE_BYTES = 500ULL * 1024 * 1024;  // 500 MB

}  // namespace config

/// Parameters controlling operator-level resource sizing.
/// These can be set via the .yaml file under the sirius.operator_params section
/// or overridden at runtime using DuckDB SET commands.
struct operator_params {
  /// Target batch size (bytes) for DuckDB scan tasks.
  uint64_t scan_task_batch_size = config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  /// Default size estimate (bytes) for VARCHAR columns when computing rows per batch.
  uint64_t default_scan_task_varchar_size = config::DEFAULT_SCAN_TASK_VARCHAR_SIZE;

  /// Maximum bytes per sort partition (0 = auto: 33% of available GPU memory).
  uint64_t max_sort_partition_bytes = 0;

  /// Target size (bytes) per hash partition for joins and group-bys.
  uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES;

  /// Target size (bytes) for the concat operator output batch.
  uint64_t concat_batch_bytes = config::DEFAULT_CONCAT_BATCH_BYTES;

  /// Maximum build-side bytes for switching to BUILD_PROBE join mode.
  /// May be larger than concat_batch_bytes; build-side batches will be concatenated if needed.
  uint64_t max_build_hash_table_bytes = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES;
};

struct sirius_config {
  sirius_config();
  ~sirius_config() = default;

  void load_from_file(const std::filesystem::path& config_path);
  void apply_defaults();

  [[nodiscard]] const cucascade::memory::system_topology_info& get_hw_topology() const noexcept
  {
    return _hw_topology;
  }

  [[nodiscard]] const std::vector<cucascade::memory::memory_space_config>&
  get_memory_space_configs() const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_task_creator_config() const noexcept;

  [[nodiscard]] const scan_manager::scan_manager_config& get_scan_manager_config() const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_gpu_pipeline_executor_config() const noexcept;

  [[nodiscard]] const exec::downgrade_executor_config& get_downgrade_executor_config()
    const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_duckdb_scan_executor_config() const noexcept;

  [[nodiscard]] bool is_scan_caching_enabled() const noexcept
  {
    return _scan_executor_config.cache != op::scan::cache_level::NONE;
  }

  [[nodiscard]] op::scan::cache_level get_cache_level() const noexcept
  {
    return _scan_executor_config.cache;
  }

  void set_cache_level(op::scan::cache_level level) noexcept
  {
    _scan_executor_config.cache = level;
  }

  [[nodiscard]] const operator_params& get_operator_params() const noexcept
  {
    return _operator_params;
  }

  [[nodiscard]] operator_params& get_operator_params() noexcept { return _operator_params; }

 private:
  cucascade::memory::system_topology_info _hw_topology{.num_gpus = 1};
  std::vector<cucascade::memory::memory_space_config> _memory_space_configs;
  exec::thread_pool_config _task_creator_config{.num_threads        = 2,
                                                .thread_name_prefix = "task_creator"};
  scan_manager::scan_manager_config _scan_manager_config{};
  exec::thread_pool_config _gpu_pipeline_executor_config{.num_threads        = 4,
                                                         .thread_name_prefix = "gpu_pipeline"};
  exec::downgrade_executor_config _downgrade_executor_config;
  op::scan::scan_executor_config _scan_executor_config;
  operator_params _operator_params;
};

}  // namespace sirius
