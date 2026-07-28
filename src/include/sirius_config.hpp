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

#include <filesystem>
#include <string>

namespace sirius {

namespace config {

constexpr uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE       = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_HASH_PARTITION_BYTES       = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_CONCAT_BATCH_BYTES         = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_SORT_SAMPLE_BYTES          = 512ULL * 1024 * 1024;  // 512 MB
constexpr uint64_t DEFAULT_MAX_BUILD_HASH_TABLE_BYTES = 500ULL * 1024 * 1024;  // 500 MB
constexpr uint64_t DEFAULT_MAX_BROADCAST_JOIN_SIZE    = 256ULL * 1024 * 1024;  // 256 MB

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

}  // namespace config

/// Parameters controlling operator-level resource sizing.
/// These can be set via the .yaml file under the sirius.operator_params section
/// or overridden at runtime using DuckDB SET commands.
struct operator_params {
  /// Target batch size (bytes) for DuckDB scan tasks.
  uint64_t scan_task_batch_size = config::DEFAULT_SCAN_TASK_BATCH_SIZE;

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

  /// Wire dynamic table-filter pushdown: an eligible BUILD_PROBE hash-join build publishes a raw
  /// exact IN-list for 1..12 supported build rows, otherwise a hash IN-list if it fits the smallest
  /// probe-GPU L2, or a Bloom, into the probe-side scan. The scan applies membership post-decode to
  /// drop non-matching rows before the join. On by default; the master switch for the feature.
  bool enable_dynamic_filter_pushdown = true;

  /// Additionally emit a runtime zone-map (build-key [min,max]) alongside the membership filter,
  /// for READ-time row-group pruning on parquet scans; duckdb-native scans apply it row-wise
  /// post-decode instead. Off by default and requires enable_dynamic_filter_pushdown: on
  /// TPC-H-shaped joins DuckDB's static transitive-predicate pushdown already prunes
  /// range-derivable builds, and scattered keys prune nothing, so the zone-map only pays off on
  /// clustered-keyset joins whose narrow key range is runtime-determined.
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
  /// Emit per-batch placement telemetry (Batch FSM + MemoryTier usages).
  /// Roughly doubles telemetry volume; no-op when enable_quent is false.
  bool enable_batch_events{true};
  std::string output_directory{"telemetry_data"};
  std::string engine_name{"siriusDB"};
};

/// Parameters controlling Simpatico compression.
///
/// Two independent paths are configured here:
///   - **pin** — caching an input table via pin_table(), gated by
///     `enable_pin_table_compression`
///   - **spill** — compressing batches downgraded off the GPU, gated by
///     `enable_spill_compression`
///
/// Fields named `pin_*` / `spill_*` affect only that path; the rest
/// (`max_compressed_fraction`, `column_threads`) apply to both.
struct compression_config {
  /// When true, pin_table(tier=>'host') attempts to compress each chunk with
  /// Simpatico before storing it in host memory. Falls back to uncompressed
  /// host storage when no plan file is found for a table or compression fails.
  bool enable_pin_table_compression{false};

  /// Minimum chunk size (uncompressed bytes) below which pin compression is
  /// skipped and the chunk is stored uncompressed.  0 = no threshold.
  /// Pin path only — the spill path compresses regardless of batch size.
  std::size_t min_batch_size_bytes{1ULL * 1024 * 1024};  // 1 MiB

  /// Maximum compressed footprint, as a fraction of the batch's original device
  /// size, for the compressed form to be kept.  When the compressed header +
  /// payload exceeds this fraction of the original (i.e. compression saved too
  /// little), the compressed data is discarded and the uncompressed batch is used.
  //  Default 0.75 (that coincides with a 1.33x compression ratio).
  ///
  /// Applies to BOTH paths: a pin chunk is stored uncompressed, and a spilled
  /// batch is downgraded uncompressed, when it fails this test.
  double max_compressed_fraction{0.75};

  /// Directory containing per-table Simpatico plan files for input-table
  /// compression.  Each file is named "<table_name>.<ext>" (any extension);
  /// its contents are the multi-column plan DSL (columns separated by "---"
  /// lines) passed verbatim to simpatico::compress_with_plan.  If no file
  /// exists for a table, that table is pinned uncompressed regardless of the
  /// enable flag.  Empty string = feature disabled.
  std::string input_plan_dir{};

  /// Degree of column-parallelism for Simpatico (de)compression: simpatico fans
  /// a table's columns across this many worker threads/streams (one column per
  /// stream). <=1 = sequential (single-stream). Capped at the column count per
  /// call. Applies to both the pin-time compress path and the scan-time
  /// decompress converters.
  int column_threads{4};

  // ── Spill-path compression (Phase 3) ──────────────────────────────────────

  /// When true, GPU→HOST and GPU→DISK downgrades compress the batch with
  /// Simpatico before writing to host/disk memory. Falls back to uncompressed
  /// on any compression failure.
  bool enable_spill_compression{false};

  /// Beam width for the per-column explorer that runs on first spill from a
  /// given operator output. Smaller values are faster but find less optimal
  /// plans. Default 20 is a fast-path setting; the full default (100) is
  /// better for offline profiling.
  uint32_t spill_explore_beam_width{20};

  /// Per-column byte cap for the spill-path explorer. Columns larger than this
  /// are explored on a trimmed prefix so that the beam search stays within
  /// device memory. Default 256 MiB.
  std::size_t spill_explore_max_bytes{256ULL * 1024 * 1024};

  /// Re-run the explorer for an operator output edge after its cached plan has
  /// been used this many times. 0 = never re-explore (cache the first plan for
  /// the rest of the query).
  ///
  /// Two things expire on this schedule: a plan that no longer suits the data
  /// (distributions drift as a query progresses), and the "compression is not
  /// worth it here" verdict recorded when a batch misses
  /// `max_compressed_fraction`. Without expiry that verdict is permanent, so an
  /// unrepresentative first batch could disable compression for an edge for the
  /// whole query. The default amortizes one explore over many batches, which is
  /// a small fraction of spill cost while still self-correcting.
  std::uint64_t spill_replan_after_uses{128};

  /// Consecutive compression *errors* on one edge to absorb before treating the
  /// edge as not worth compressing. Minimum 1 (write off on the first error).
  ///
  /// Distinct from `max_compressed_fraction`, which is a measurement and applies
  /// immediately. Compression runs under memory pressure, so an exception is as
  /// likely to be a transient allocation failure as a real signal about the data;
  /// writing the edge off on the first one would disable compression for a whole
  /// replan interval — and stretch that interval — over a passing blip.
  std::uint32_t spill_error_tolerance{3};

  /// Relative change in a column's compression ratio or in either of its
  /// throughputs below which a re-explored plan counts as equivalent to the
  /// cached one, and the cached plan is kept. 0.2 = 20%.
  ///
  /// The explorer is a beam search over a large space and readily returns a
  /// differently spelled plan that performs the same. Adopting those churns the
  /// cache and, worse, registers as a change — which resets the replan backoff
  /// and locks the edge into re-exploring for the rest of the query. Set to 0 to
  /// adopt every re-explored plan.
  double spill_replan_change_threshold{0.20};

  /// Row prefix the spill-path explorer runs on (0 = the whole column).
  ///
  /// The beam search allocates for hundreds of trial encodes, and on the spill
  /// path it runs exactly when the GPU is out of memory — on full columns it
  /// mostly throws bad_alloc, costing the full search and yielding no plan.
  /// Sampling bounds both allocation and search time. Note the explorer's own
  /// caveat: a row prefix picks markedly worse plans for sorted/monotonic
  /// columns, whose best cascade exploits global structure.
  std::size_t spill_explore_sample_rows{65536};
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

 private:
  /// When @c _memory_space_configs contains more than one GPU memory space,
  /// force @c _scan_manager_config.use_sirius_datasource to true (sirius
  /// datasource is required for multi-GPU IO routing). Emits a WARNING when
  /// the override takes effect. Called from the end of @ref load_from_file.
  void enforce_sirius_datasource_for_multi_gpu();

  cucascade::memory::system_topology_info _hw_topology{.num_gpus = 1};
  std::vector<cucascade::memory::memory_space_config> _memory_space_configs;
  creator::task_creator_config _task_creator_config;
  scan_manager::scan_manager_config _scan_manager_config{};
  exec::thread_pool_config _gpu_pipeline_executor_config{.num_threads        = 4,
                                                         .thread_name_prefix = "gpu_pipeline"};
  exec::downgrade_executor_config _downgrade_executor_config;
  operator_params _operator_params;
  telemetry_config _telemetry_config;
  compression_config _compression_config;
};

}  // namespace sirius
