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

/// Row-count ratio gate for switching STANDARD-mode MARK joins to cudf::mark_join (build on the
/// left/output side) instead of cudf::filtered_join (build on the right side). mark_join only
/// wins when the left side is much smaller than the right (probe) side.
///
/// Provenance: a standalone microbenchmark (see issue #510) compared filtered_join (build right,
/// probe left) vs. mark_join (build left, probe right) — including the BOOL8 scatter that
/// resolve_mark_join_result performs — across many left/right size ratios on an NVIDIA L4. The
/// scatter cost was negligible and identical for both. filtered_join won at or near parity and
/// only lost once the right side was roughly >= 3-4x the left side, i.e. when mark_join's build
/// side (left) was substantially smaller. We default to 8.0 (well above the measured ~3-4x
/// crossover) so the switch only triggers when it is a clear win, leaving headroom for the fact
/// that the crossover is hardware- and workload-dependent. Recalibrate per GPU; set to 0 to
/// disable (always use filtered_join).
constexpr double DEFAULT_MARK_JOIN_BUILD_SWITCH_RATIO = 8.0;

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

  /// For STANDARD-mode MARK joins: build the hash table on the left/output side via
  /// cudf::mark_join (instead of on the right side via filtered_join) when the right (probe)
  /// side has at least this many times more rows than the left side. mark_join only wins when
  /// the left side is substantially smaller; the crossover is hardware-dependent (~3-4x on an
  /// L4 in the issue #510 microbenchmark, defaulted higher to stay conservative). Set to 0 to
  /// disable (always use filtered_join).
  double mark_join_build_switch_ratio = config::DEFAULT_MARK_JOIN_BUILD_SWITCH_RATIO;

  /// Wire dynamic table-filter pushdown: an eligible BUILD_PROBE hash-join build publishes a
  /// runtime membership filter (IN-list / Bloom, chosen by L2-cache fit) into the probe-side scan,
  /// which applies it post-decode to drop non-matching rows before the join. On by default; the
  /// master switch for the feature.
  bool enable_dynamic_filter_pushdown = true;

  /// Additionally emit a runtime zone-map (build-key [min,max]) alongside the membership filter,
  /// for READ-time row-group pruning at the probe scan. Off by default and requires
  /// enable_dynamic_filter_pushdown: on TPC-H-shaped joins DuckDB's static transitive-predicate
  /// pushdown already prunes range-derivable builds, and scattered keys prune nothing, so the
  /// zone-map only pays off on clustered-keyset joins whose narrow key range is runtime-determined.
  bool enable_dynamic_zone_map_filter = false;

  /// Skip publishing a key's dynamic filters when the build covers at least this fraction of the
  /// key's domain (rows gate and zone-map range gate). Values >= 1.0 effectively disable the gate.
  double dynamic_filter_domain_coverage_threshold = 0.9;

  /// Consumer-side scan gate: disable a scan's post-decode dynamic filtering once a measured split
  /// keeps more than this fraction of its rows (too unselective to repay the mask kernel). In
  /// [0, 1]; 1.0 keeps filtering always on.
  double dynamic_filter_keep_threshold = 0.9;

  /// Zone-map pruning of pinned-table chunks at cache-serve time: skip cached chunks whose pin-time
  /// min/max statistics prove the scan's pushed-down filter matches no rows. Gates BOTH the
  /// pin-time statistics capture and the serve-side survivor plan: a table pinned while the flag is
  /// off carries no zone maps and cannot prune until re-pinned with the flag on.
  bool enable_pinned_zone_map_pruning = true;
};

struct telemetry_config {
  bool enable_quent{true};
  std::string output_directory{"telemetry_data"};
  std::string engine_name{"siriusDB"};
};

/// Parameters controlling Simpatico compression for pin_table(tier=>'host').
/// These settings apply exclusively to cached input-table pinning and have no
/// effect on spill-path compression (Phase 3).
struct compression_config {
  /// When true, pin_table(tier=>'host') attempts to compress each chunk with
  /// Simpatico before storing it in host memory. Falls back to uncompressed
  /// host storage when no plan file is found for a table or compression fails.
  bool enable_pin_table_compression{false};

  /// Minimum chunk size (uncompressed bytes) below which compression is
  /// skipped and the chunk is stored uncompressed.  0 = no threshold.
  std::size_t min_batch_size_bytes{1ULL * 1024 * 1024};  // 1 MiB

  /// Maximum compressed footprint, as a fraction of the batch's original device
  /// size, for the compressed form to be kept.  When the compressed header +
  /// payload exceeds this fraction of the original (i.e. compression saved too
  /// little), the compressed data is discarded and the batch is pinned
  /// uncompressed.  Default 0.95 (require at least a 5% saving).
  double max_compressed_fraction{0.95};

  /// Directory containing per-table Simpatico plan files for input-table
  /// compression.  Each file is named "<table_name>.<ext>" (any extension);
  /// its contents are the multi-column plan DSL (columns separated by "---"
  /// lines) passed verbatim to simpatico::compress_with_plan.  If no file
  /// exists for a table, that table is pinned uncompressed regardless of the
  /// enable flag.  Empty string = feature disabled.
  std::string input_plan_dir{};
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

  /// Overwrite the stored scan_manager_config. Allows callers (e.g.
  /// SiriusContext::initialize()) to persist runtime-derived wiring so a later
  /// get_scan_manager_config() reflects the actual scan_manager state.
  void set_scan_manager_config(scan_manager::scan_manager_config config) noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_gpu_pipeline_executor_config() const noexcept;

  [[nodiscard]] const exec::downgrade_executor_config& get_downgrade_executor_config()
    const noexcept;

  /// Pop ordering for the task_scheduler's pipeline-level task queue. See
  /// exec::queue_ordering for semantics. Defaults to FIFO (legacy behavior).
  [[nodiscard]] exec::queue_ordering get_task_queue_ordering() const noexcept
  {
    return _task_queue_ordering;
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

  [[nodiscard]] const compression_config& get_compression_config() const noexcept
  {
    return _compression_config;
  }

  [[nodiscard]] compression_config& get_compression_config() noexcept
  {
    return _compression_config;
  }

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
  operator_params _operator_params;
  telemetry_config _telemetry_config;
  compression_config _compression_config;
  exec::queue_ordering _task_queue_ordering{exec::queue_ordering::FIFO};
};

}  // namespace sirius
