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

/// Test build-key uniqueness at runtime when the planner could not prove it statically.
///
/// cudf's general hash join probes twice — a count pass to size the output, then a retrieve pass —
/// while cudf::distinct_hash_join probes once, because a distinct build bounds the output by the
/// probe row count. Sirius already implements both, but the distinct path is gated on a *proof* of
/// uniqueness, which only a declared PRIMARY KEY on a catalog table can supply. The runtime test is
/// one cudf::distinct_count pass over the build keys, taken only in BUILD_PROBE mode.
constexpr bool DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE = true;

}  // namespace config

/// Parameters controlling operator-level resource sizing.
/// These can be set via the .yaml file under the sirius.operator_params section
/// or overridden at runtime using DuckDB SET commands.
struct operator_params {
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

  /// When the planner could not prove build-key uniqueness, test it at runtime (one
  /// cudf::distinct_count pass over the build keys) and take the single-pass
  /// cudf::distinct_hash_join instead of the two-pass general path when the keys are in fact
  /// distinct. BUILD_PROBE mode only, INNER/LEFT equality joins with null-unequal semantics. See
  /// DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE.
  bool enable_runtime_distinct_build_probe = config::DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE;

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

/// Parameters controlling Simpatico compression.
///
/// Two independent paths are configured here:
///   - **pin** — caching an input table via pin_table(), gated by
///     `enable_pin_table_compression`
///   - **spill** — compressing batches downgraded off the GPU, gated by
///     `enable_spill_compression`
///
/// Fields named `pin_*` / `spill_*` affect only that path; the rest
/// (`max_compressed_fraction`) apply to both.
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
  /// Must be finite and non-negative. Values above 1 deliberately allow compressed
  /// representations that expand relative to the original batch.
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

  // ── Eager task-output compression ─────────────────────────────────────────

  /// When true, a finished task's output batch is compressed on the GPU before
  /// publication, for those columns whose base-table plan (reached through
  /// column lineage) is measurably both fast and high-ratio. Falls back to
  /// publishing uncompressed on any failure or when nothing qualifies.
  ///
  /// Distinct from spill compression in intent: that one compresses because it
  /// must, this one only when the offline measurements say the GPU time is worth
  /// it. An edge with no qualifying column costs one lookup per query.
  bool enable_output_compression{false};

  /// Minimum recorded compression ratio for a column's plan to be used eagerly.
  ///
  /// Also re-checked against the ratio actually *achieved* on the first batch, so
  /// a plan whose offline ratio does not survive the operator output — notably a
  /// delta cascade, whose base-table ratio comes from sorted storage that a join
  /// or hash partition has destroyed — is dropped after one wasted pass.
  double output_compression_min_ratio{3.0};

  /// Minimum recorded compress throughput (GB/s) for a plan to be used eagerly.
  ///
  /// Gated separately from decompression because output is written once and read
  /// back at most once, so encode speed is on the critical path. Note the shipped
  /// SF1000 plans were Pareto-picked for *decompress* only, so this is the gate
  /// that actually binds: at 250 GB/s it admits 13 of 53 TPC-H columns, and every
  /// column it rejects with a good ratio is rejected on compress speed.
  double output_compression_min_compress_gbps{250.0};

  /// Minimum recorded decompress throughput (GB/s) for a plan to be used eagerly.
  double output_compression_min_decompress_gbps{250.0};

  /// Smallest output batch worth compressing eagerly.
  ///
  /// Compressing a batch costs a roughly fixed amount regardless of its size —
  /// a per-column, per-plan-node `cudaStreamSynchronize` (compress.cpp, needed
  /// because variable-output codecs report their size from device memory), plus
  /// the blob staging. The SF100 sweep measured ~2.95 ms per batch against
  /// ~30 us of actual codec work for a 13.4 MiB batch, i.e. ~1-2% of the codecs'
  /// rated throughput: below some size a batch simply cannot repay the setup.
  ///
  /// Separate from `min_batch_size_bytes` (the pin path's threshold) because the
  /// two pay different fixed costs and run under different pressure.
  std::size_t output_compression_min_batch_bytes{64ULL * 1024 * 1024};

  /// Smallest batch worth compressing on the spill path.
  ///
  /// The same fixed per-batch cost applies here, but the spill path cannot choose
  /// its batch sizes: they are whatever the operators produced, and shrinking
  /// operator batch limits to relieve GPU pressure shrinks spill batches with
  /// them. Measured at SF1000 with 500 MB operator batches, spill batches landed
  /// around 500 KB and a downgrade request moved 1.06 GB across 79 of them in
  /// 14.1 s — 71.7 MB/s, against 9,056 MB/s for the same request uncompressed.
  /// Below this size the setup cost dominates so heavily that compressing is
  /// worse than spilling raw, however good the ratio.
  std::size_t spill_min_batch_bytes{64ULL * 1024 * 1024};
  /// Device memory reserved exclusively for spill-compression transients.
  /// 0 (the default) keeps the encoder allocating from the query's pool.
  ///
  /// Sharing that pool is circular: a downgrade happens *because* the pool is
  /// full, so the encode that would relieve the pressure is the one allocation
  /// certain to fail. Measured on q3/SF1000 with no arena, compression latched
  /// off and on 11 times while the monitor issued 111,641 downgrade requests.
  ///
  /// This is a partition of the device, not extra memory: reserving N bytes
  /// requires lowering `memory.gpu.usage_limit_fraction` by the same N. Undersizing
  /// is a cliff rather than a gradient — at 1 GiB, too small for the concurrent
  /// encodes, the same query failed outright. Size it for
  /// `downgrade.num_threads` concurrent encodes of the largest spill batch.
  std::size_t device_pool_bytes{0};

  /// Free each uncompressed source column as soon as it has been encoded, instead
  /// of holding the whole batch until the converter returns.
  ///
  /// This is the memory the spill exists to reclaim, and releasing it during the
  /// encode rather than after is what lets the compression arena be small: per
  /// encode the device carries one column's source instead of the batch's.
  ///
  /// Opt-in because it forfeits the fall-back. Once a column has been freed the
  /// batch cannot be spilled uncompressed any more, so the encode must run to
  /// completion — a column that cannot even be stored raw (identity, no codec
  /// scratch) becomes fatal for the batch rather than a decline. Requires
  /// ownership of the table; batches viewing externally-owned memory are skipped.
  bool spill_release_columns_early{false};

  /// Fraction of a batch's uncompressed size reserved on the device for encode
  /// working memory when no compression arena is configured. See
  /// spill_context.hpp::encode_reserve_fraction.
  double spill_encode_reserve_fraction{0.5};

  /// Decline spill compression when free device memory is below this fraction of
  /// capacity and no arena is configured. See
  /// spill_context.hpp::encode_min_headroom_fraction.
  double spill_encode_min_headroom_fraction{0.10};

  /// When true, the downgrade executor may satisfy a request by compressing
  /// batches in place on the device, instead of spilling them to host/disk.
  ///
  /// Independent of `enable_output_compression`: that one compresses task output
  /// speculatively at the sink, this one only when a downgrade request has proven
  /// the memory is needed. They share the plan-quality gate but are separate
  /// policies and are measured separately.
  bool enable_device_compression_downgrade{false};
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

  /// How many GPUs to allocate per query. 0 = use all active GPUs (default).
  /// Limits each query to the first @c gpus_per_query entries of the sorted
  /// active-GPU list; the rest are left available for future concurrent queries.
  [[nodiscard]] int gpus_per_query() const noexcept { return _gpus_per_query; }

 private:
  /// When @c _memory_space_configs contains more than one GPU memory space,
  /// force @c _scan_manager_config.use_sirius_datasource to true (sirius
  /// datasource is required for multi-GPU IO routing). Emits a WARNING when
  /// the override takes effect. Called from the end of @ref load_from_file.
  void enforce_sirius_datasource_for_multi_gpu();

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
