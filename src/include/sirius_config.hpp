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
#include "creator/config.hpp"
#include "exec/config.hpp"
#include "scan_manager/config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cmath>
#include <filesystem>
#include <string>

namespace sirius {

namespace config {

/// Static fallback for operator batch/partition sizing, used when no GPU is
/// visible; the per-operator alias constants below keep it as their last-resort
/// value for unwired construction paths.
constexpr uint64_t DEFAULT_BATCH_SIZE = 800ULL * 1024 * 1024;  // 800 MiB

/// Shared operator batch default: 2.5% of the smallest visible GPU's total memory,
/// clamped to [512 MiB, 5 GiB]; DEFAULT_BATCH_SIZE when no GPU is visible. Queried
/// once per process (memoized). operator_params derives its batch members from this,
/// so every default-constructed instance agrees. When YAML explicitly configures an
/// effective GPU capacity, sirius_config narrows the shared defaults from the resolved
/// memory-space configs before applying explicit operator_params overrides.
uint64_t derived_default_batch_size();

constexpr uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE       = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_HASH_PARTITION_BYTES       = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_CONCAT_BATCH_BYTES         = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_SORT_SAMPLE_BYTES          = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_MAX_BUILD_HASH_TABLE_BYTES = 2 * DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_MAX_BROADCAST_JOIN_SIZE    = 256ULL * 1024 * 1024;  // 256 MiB

/// Multi-GPU small-table threshold, charged per GPU. A partition-sizing consumer (hash join,
/// merge_group_by) keeps inputs below `num_gpus * this` on a single GPU (one partition) to avoid
/// cross-device overhead; above it, the multi-GPU floor of `num_gpus` partitions kicks in.
constexpr uint64_t PARTITION_SMALL_TABLE_BYTES_PER_GPU = 16ULL * 1024 * 1024;  // 16 MB

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

struct valid_domain_coverage_threshold {
  [[nodiscard]] bool operator()(double value) const noexcept
  {
    return std::isfinite(value) && value > 0.0;
  }
  [[nodiscard]] static constexpr char const* description() noexcept
  {
    return "must be finite and greater than 0.0";
  }
};

/// Test build-key uniqueness at runtime when the planner could not prove it statically.
///
/// cudf's general hash join probes twice — a count pass to size the output, then a retrieve pass —
/// while cudf::distinct_hash_join probes once, because a distinct build bounds the output by the
/// probe row count. Sirius already implements both, but the distinct path is gated on a *proof* of
/// uniqueness, which only a declared PRIMARY KEY on a catalog table can supply. The runtime test is
/// one cudf::distinct_count pass over the build keys, taken only in BUILD_PROBE mode.
///
/// Temporarily off by default (issue #1600): cudf::distinct_count can hit a cuCollections bug
/// (NVIDIA/cuCollections#834) on some key distributions. Re-enable once the fix ships in libcudf.
constexpr bool DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE = false;

}  // namespace config

/// Operator parameters shared between planning and execution.
/// User-tunable members can be set under sirius.operator_params in YAML or overridden with
/// DuckDB SET commands; engine-owned query policy is noted below.
struct operator_params {
  /// Engine-owned query policy. The user-facing setting defaults to enabled, but an unwired
  /// execution context stays fail-closed until the engine snapshots the connection value.
  bool like_swar_fastpath = false;

  /// Target batch size (bytes) for DuckDB scan tasks.
  uint64_t scan_task_batch_size = config::derived_default_batch_size();

  /// Maximum bytes per sort partition (0 = auto based on max_sort_partition_memory_fraction).
  uint64_t max_sort_partition_bytes = 0;

  /// Fraction of available GPU memory per sort partition when max_sort_partition_bytes is 0.
  double max_sort_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;

  /// Target size (bytes) per hash partition for joins and group-bys.
  uint64_t hash_partition_bytes = config::derived_default_batch_size();

  /// Target size (bytes) for the concat operator output batch.
  uint64_t concat_batch_bytes = config::derived_default_batch_size();

  /// Target size (bytes) of data to sample before computing sort partition boundaries.
  uint64_t sort_sample_bytes = config::derived_default_batch_size();

  /// Maximum build-side bytes for switching to BUILD_PROBE join mode: 2x the shared
  /// batch default. May be larger than concat_batch_bytes; build-side batches will be
  /// concatenated if needed.
  uint64_t max_build_hash_table_bytes = 2 * config::derived_default_batch_size();

  /// Maximum build-side bytes for a broadcast join. A build below this size is eligible to be
  /// replicated to every GPU (instead of hash-partitioning across GPUs) when the probe side is
  /// large relative to the build side. See compute_hash_join_partition_strategy.
  uint64_t max_broadcast_join_size = config::DEFAULT_MAX_BROADCAST_JOIN_SIZE;

  /// For STANDARD-mode MARK joins: build the hash table on the left/output side via
  /// cudf::mark_join (instead of on the right side via filtered_join) when the right (probe)
  /// side has at least this many times more rows than the left side. mark_join only wins when
  /// the left side is substantially smaller; the crossover is hardware-dependent (~3-4x on an
  /// L4 in the issue #510 microbenchmark, defaulted higher to stay conservative). Set to 0 to
  /// disable (always use filtered_join).
  double mark_join_build_switch_ratio = config::DEFAULT_MARK_JOIN_BUILD_SWITCH_RATIO;

  /// Engine-owned policy: when enabled and the planner could not prove build-key uniqueness, test
  /// it at runtime (one cudf::distinct_count pass over the build keys) and take the single-pass
  /// cudf::distinct_hash_join when the keys are distinct. Temporarily disabled pending issue #1600.
  /// BUILD_PROBE mode only, INNER/LEFT equality joins with null-unequal semantics. Tests retain
  /// direct programmatic control of this field to exercise both implementations. See
  /// DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE.
  bool enable_runtime_distinct_build_probe = config::DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE;

  /// Enable dynamic filters for eligible hash joins.
  bool enable_dynamic_filter = true;

  /// Emit build-key min/max filters in addition to membership filters.
  bool enable_dynamic_zone_map_filter = false;

  /// Skip a proven-unique key when its complete build meets this known-domain coverage. Values
  /// above 1 disable the gate.
  double dynamic_filter_domain_coverage_threshold = 0.9;

  /// Hash-IN-list size limit as a fraction of the smallest known probe-GPU L2, in [0, 1]. Larger
  /// sets use Bloom; unknown L2 makes the hash IN-list ineligible.
  double dynamic_filter_inlist_max_l2_fraction = 0.125;

  /// Disable post-decode filtering above this measured keep ratio. Values are in [0, 1]; 1 keeps
  /// the scan-level gate active.
  double dynamic_filter_keep_threshold = 0.9;

  /// Zone-map pruning of pinned-table chunks at cache-serve time: skip cached chunks whose pin-time
  /// min/max statistics prove the scan's pushed-down filter matches no rows. Gates BOTH the
  /// pin-time statistics capture and the serve-side survivor plan: a table pinned while the flag is
  /// off carries no zone maps and cannot prune until re-pinned with the flag on.
  bool enable_pinned_zone_map_pruning = true;

  /// Store eligible integer and fixed-point DECIMAL columns in carriers selected from exact
  /// per-chunk bounds during pinning. Matching pinned scans derive targets from recorded storage
  /// metadata; other scans use native carriers. Logical types remain unchanged, and type-sensitive
  /// boundaries restore native carriers.
  bool enable_compressed_materialization = true;

  /// Admission-time GPU allocation: target bytes of projected scan output per GPU.
  /// At query start, the engine estimates total scan output bytes from the plan's
  /// estimated_cardinality × per-column width, then assigns
  /// ceil(total_bytes / admission_bytes_per_gpu) GPUs, clamped to [1, active_gpu_count].
  /// 0 disables dynamic estimation and falls back to topology.gpus_per_query.
  uint64_t admission_bytes_per_gpu = 0;

  /// Bytes assumed per variable-width column (VARCHAR, LIST, etc.) when computing
  /// per-row byte estimates for admission. Only used when admission_bytes_per_gpu > 0.
  uint64_t avg_variable_column_bytes = 32;
};

struct telemetry_config {
  bool enable_quent{true};
  /// Emit per-batch placement telemetry (Batch FSM + MemoryTier usages).
  /// Roughly doubles telemetry volume; no-op when enable_quent is false.
  bool enable_batch_events{true};
  std::string exporter{"ndjson"};
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
  /// little), the compressed data is discarded and the uncompressed batch is used.
  /// Must be finite and non-negative. Values above 1 deliberately allow compressed
  /// representations that expand relative to the original batch.
  //  Default 0.75 (that coincides with a 1.33x compression ratio).
  double max_compressed_fraction{0.75};

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

  [[nodiscard]] const creator::task_creator_config& get_task_creator_config() const noexcept;

  [[nodiscard]] const scan_manager::scan_manager_config& get_scan_manager_config() const noexcept;

  /// The read path's caching configuration (the
  /// @c sirius.executor.scan_manager.cache YAML block).  Stored inside the
  /// scan_manager config, which is its only consumer, so the two can never
  /// disagree.
  [[nodiscard]] const io::cache::config& get_cache_config() const noexcept
  {
    return _scan_manager_config.cache;
  }

  /// Overwrite the stored scan_manager_config. Allows callers (e.g.
  /// SiriusContext::initialize()) to persist runtime-derived wiring so a later
  /// get_scan_manager_config() reflects the actual scan_manager state.
  void set_scan_manager_config(scan_manager::scan_manager_config config) noexcept;

  [[nodiscard]] const exec::thread_pool_config& get_gpu_pipeline_executor_config() const noexcept;

  [[nodiscard]] const exec::downgrade_executor_config& get_downgrade_executor_config()
    const noexcept;

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

  /// How many GPUs to allocate per query. 0 = use all active GPUs (default).
  /// Limits each query to the first @c gpus_per_query entries of the sorted
  /// active-GPU list; the rest are left available for future concurrent queries.
  [[nodiscard]] int gpus_per_query() const noexcept { return _gpus_per_query; }

 private:
  /// When @c _memory_space_configs contains more than one GPU memory space,
  /// force @c _scan_manager_config.backend to @c io_backend::sirius (the
  /// sirius backend is required for multi-GPU IO routing). Emits a WARNING when
  /// the override takes effect. Called from the end of @ref load_from_file.
  void enforce_sirius_backend_for_multi_gpu();

  /// Re-default @c _scan_manager_config.uring.n_max_concurrent_scans to the
  /// CONFIGURED pipeline pool size. The struct default can only use the
  /// compile-time thread count, so resizing the pipeline in config would
  /// otherwise leave the readahead budget behind. Called from the end of
  /// @ref load_from_file; an explicit config value is left alone.
  void derive_uring_scan_budget();

  /// Re-default @c _scan_manager_config.rest.n_max_concurrent_scans to a
  /// multiple of the configured pipeline pool size. Object-store reads are
  /// latency-bound, so the readahead needs several splits in flight per pipeline
  /// thread to stay ahead of demand. Called from the end of @ref load_from_file;
  /// an explicit config value is left alone.
  void derive_rest_scan_budget();

  cucascade::memory::system_topology_info _hw_topology{.num_gpus = 1};
  int _gpus_per_query = 0;
  std::vector<cucascade::memory::memory_space_config> _memory_space_configs;
  creator::task_creator_config _task_creator_config;
  scan_manager::scan_manager_config _scan_manager_config{};
  exec::thread_pool_config _gpu_pipeline_executor_config{
    .num_threads = exec::default_gpu_pipeline_num_threads, .thread_name_prefix = "gpu_pipeline"};
  exec::downgrade_executor_config _downgrade_executor_config;
  operator_params _operator_params;
  telemetry_config _telemetry_config;
  compression_config _compression_config;
};

}  // namespace sirius
