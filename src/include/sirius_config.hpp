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
#include "exec/inspectable_mpsc.hpp"
#include "io/object_store_config.hpp"
#include "op/scan/config.hpp"
#include "scan_manager/config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <filesystem>
#include <string>

namespace sirius {

namespace config {

constexpr uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE       = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_SCAN_TASK_VARCHAR_SIZE     = 256LL;
constexpr uint64_t DEFAULT_HASH_PARTITION_BYTES       = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_CONCAT_BATCH_BYTES         = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_SORT_SAMPLE_BYTES          = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_MAX_BUILD_HASH_TABLE_BYTES = 500ULL * 1024 * 1024;  // 500 MB

/// Fraction of available GPU memory used per sort partition when max_sort_partition_bytes is 0.
constexpr double DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION = 0.33;

}  // namespace config

/// Parameters controlling operator-level resource sizing.
/// These can be set via the .yaml file under the sirius.operator_params section
/// or overridden at runtime using DuckDB SET commands.
struct operator_params {
  /// Target batch size (bytes) for DuckDB scan tasks.
  uint64_t scan_task_batch_size = config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  /// Default size estimate (bytes) for VARCHAR columns when computing rows per batch.
  uint64_t default_scan_task_varchar_size = config::DEFAULT_SCAN_TASK_VARCHAR_SIZE;

  /// Maximum bytes per sort partition (0 = auto based on max_sort_partition_memory_fraction).
  uint64_t max_sort_partition_bytes = 0;

  /// Fraction of available GPU memory per sort partition when max_sort_partition_bytes is 0.
  double max_sort_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;

  /// Target size (bytes) per hash partition for joins and group-bys.
  uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES;

  /// Target size (bytes) for the concat operator output batch.
  uint64_t concat_batch_bytes = config::DEFAULT_CONCAT_BATCH_BYTES;

  /// Target size (bytes) of data to sample before computing sort partition boundaries.
  uint64_t sort_sample_bytes = config::DEFAULT_SORT_SAMPLE_BYTES;

  /// Maximum build-side bytes for switching to BUILD_PROBE join mode.
  /// May be larger than concat_batch_bytes; build-side batches will be concatenated if needed.
  uint64_t max_build_hash_table_bytes = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES;
};

struct telemetry_config {
  bool enable_quent{false};
  std::string output_directory{"telemetry_data"};
  std::string engine_name{"siriusDB"};
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

  /// Overwrite the stored scan_manager_config. SiriusContext::initialize() uses
  /// this to persist the S3 backend it materialized from object_store_config,
  /// so a later get_config() reflects the actual scan_manager wiring.
  void set_scan_manager_config(scan_manager::scan_manager_config config) noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_gpu_pipeline_executor_config() const noexcept;

  [[nodiscard]] const exec::downgrade_executor_config& get_downgrade_executor_config()
    const noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_duckdb_scan_executor_config() const noexcept;

  /// Pop ordering for the task_scheduler's pipeline-level task queue. See
  /// exec::queue_ordering for semantics. Defaults to FIFO (legacy behavior).
  [[nodiscard]] exec::queue_ordering get_task_queue_ordering() const noexcept
  {
    return _task_queue_ordering;
  }

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

  [[nodiscard]] const telemetry_config& get_telemetry_config() const noexcept
  {
    return _telemetry_config;
  }

  /// Object-store backend credentials + endpoint. Empty fields disable the
  /// S3 backend; SiriusContext::initialize() reads this to populate
  /// scan_manager_config::s3_config before constructing the scan_manager.
  /// Direct member access (no getter/setter) to keep the test fixture and
  /// future SET-handler wiring simple — both sides write into this struct
  /// and SiriusContext consumes it at initialize() time.
  sirius::io::object_store_config object_store_config{};

 private:
  /// When @c _memory_space_configs contains more than one GPU memory space,
  /// force @c _scan_manager_config.use_sirius_datasource to true (sirius
  /// datasource is required for multi-GPU IO routing). Emits a WARNING when
  /// the override takes effect. Called from the end of @ref load_from_file.
  void enforce_sirius_datasource_for_multi_gpu();

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
  telemetry_config _telemetry_config;
  exec::queue_ordering _task_queue_ordering{exec::queue_ordering::FIFO};
};

}  // namespace sirius
