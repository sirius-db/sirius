# Configuration

This document covers Super Sirius configuration: the `sirius_config` class, operator parameters, thread pool settings, and DuckDB SET variables.

## `sirius_config`

**File:** `src/include/sirius_config.hpp`

The `sirius_config` class loads configuration from a YAML file or uses built-in defaults. It provides:

- Hardware topology (GPU count, NUMA layout)
- Memory space configurations (GPU, Host, Disk)
- Thread pool configs for all executor types
- Operator parameters (batch sizes, limits)
- Telemetry options

### Config File Resolution

Sirius searches for a config file in this order:

1. **`SIRIUS_CONFIG_FILE`** environment variable — explicit path
2. **`./sirius.yaml`** — current working directory
3. **`~/.sirius/sirius.yaml`** — user's home directory

If no config file is found, Sirius initializes with built-in defaults (95% GPU memory, 90% of each NUMA node's RAM as pinned host memory).

### `SIRIUS_DISABLE`

Set `SIRIUS_DISABLE=1` to prevent Super Sirius from initializing. This is **required** when using the legacy code path (`gpu_buffer_init`/`gpu_processing`), because Super Sirius claims most GPU and pinned host memory on startup, leaving insufficient memory for the legacy buffer manager. It is also useful for CPU-only benchmarks.

```bash
export SIRIUS_DISABLE=1
```

### Byte Suffixes

Any integer config value that represents bytes supports human-readable suffixes:

| Suffix | Base | Example | Bytes |
|--------|------|---------|-------|
| `K`, `KB` | 1000 | `500K` | 500,000 |
| `Ki`, `KiB` | 1024 | `500Ki` | 512,000 |
| `M`, `MB` | 1000² | `512M` | 512,000,000 |
| `Mi`, `MiB` | 1024² | `512Mi` | 536,870,912 |
| `G`, `GB` | 1000³ | `8G` | 8,000,000,000 |
| `Gi`, `GiB` | 1024³ | `8Gi` | 8,589,934,592 |
| `T`, `TB` | 1000⁴ | `1T` | 1,000,000,000,000 |
| `Ti`, `TiB` | 1024⁴ | `1Ti` | 1,099,511,627,776 |

Fractional values are supported (e.g. `1.5Gi`). Follows the [Kubernetes resource units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory) convention.

```yaml
sirius:
  memory:
    host:
      capacity_bytes: 64Gi       # 68,719,476,736 bytes
      block_size: 1Mi            # 1,048,576 bytes
  operator_params:
    scan_task_batch_size: 512Mi  # 536,870,912 bytes
```

### Loading (C++ API)

```cpp
sirius_config config;
config.load_from_file("/path/to/config.yaml");  // Optional
```

### Example Config File

```yaml
sirius:
  topology: { num_gpus: 1 }
  memory:
    gpu:
      usage_limit_fraction: 0.95
      reservation_limit_fraction: 1.0
      downgrade_trigger_fraction: 0.8
      downgrade_stop_fraction: 0.6
    host: { capacity_bytes: 25GB, initial_number_pools: 50, pool_size: 512, block_size: 1048576 }
    disk: { disk_id: 0, capacity_bytes: 1000000000000, downgrade_root_dirs: "/tmp/sirius_disk_memory" }
  executor:
    scan_manager: { num_threads: 4, use_sirius_datasource: true, uring_n_reactors: 1, enable_prefetch_cache: false }
    pipeline:     { num_threads: 4 }
    downgrade:    { num_threads: 1 }
    task_creator: { num_threads: 1 }
  operator_params:
    scan_task_batch_size:       805306368   # 768 MiB
    max_sort_partition_bytes:   0           # 0 = auto (33% GPU memory)
    hash_partition_bytes:       805306368   # 768 MiB
    concat_batch_bytes:         805306368   # 768 MiB
    max_build_hash_table_bytes: 805306368   # 768 MiB
    enable_dynamic_filter: true    # scan and join-edge runtime filters
    enable_dynamic_zone_map_filter: false  # optional parquet-read/native-post-decode min/max
    dynamic_filter_domain_coverage_threshold: 0.9  # skip all filters for keys meeting known-domain coverage
    dynamic_filter_inlist_max_l2_fraction: 0.125  # hash-IN-list fraction of known probe-GPU L2 (0 = Bloom for non-small keys; 1.0 = full L2)
    dynamic_filter_keep_threshold: 0.9  # disable a scan's filtering when a split keeps > this fraction
    enable_pinned_zone_map_pruning: true  # capture and use per-chunk stats for pinned tables
  telemetry:
    enable_quent: true
    output_directory: telemetry_data
    engine_name: siriusDB
```

## Memory Configuration

Sirius uses cuCascade for tiered memory management across GPU, Host (pinned), and Disk tiers. The `memory` section provides a high-level interface that maps to cuCascade's underlying memory space configs.

### Topology

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_gpus` | int | all visible GPUs | Non-negative number of GPUs to use. Defaults to every GPU visible to topology discovery (honors `CUDA_VISIBLE_DEVICES`); `0` also means auto. Mutually exclusive with `gpu_ids`. |
| `gpu_ids` | list of int | — | Non-empty list of unique, non-negative GPU device IDs. Mutually exclusive with `num_gpus`. |
| `gpus_per_query` | int | `0` (all) | How many GPUs each query is allocated at admission time. `0` uses all active GPUs. Values exceeding the active GPU count are clamped to the full set. |

### GPU Memory (`sirius.memory.gpu`)

Controls how much GPU VRAM Sirius claims and when it starts evicting data to host memory.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `usage_limit_fraction` | double | 0.95 | Fraction of total VRAM to use as Sirius's GPU memory capacity. The remaining 5% is left for the CUDA runtime, cuDF temporaries, and other GPU consumers. Mutually exclusive with `usage_limit_bytes`; configuration loading rejects both when both values are non-null. |
| `usage_limit_bytes` | bytes | — | Absolute VRAM limit. Mutually exclusive with `usage_limit_fraction`; configuration loading rejects both when both values are non-null. |
| `reservation_limit_fraction` | double | 1.0 | Fraction of the GPU capacity (set by `usage_limit_*`) that can be reserved by pipeline tasks. Reservations are acquired before task execution and prevent overcommit. Mutually exclusive with `reservation_limit_bytes`; configuration loading rejects both when both values are non-null. |
| `reservation_limit_bytes` | bytes | — | Absolute reservation limit. Mutually exclusive with `reservation_limit_fraction`; configuration loading rejects both when both values are non-null. |
| `downgrade_trigger_fraction` | double (0,1] | 0.8 | Start evicting GPU-resident data to host when reserved memory exceeds this fraction of capacity. Must be greater than `downgrade_stop_fraction`. |
| `downgrade_stop_fraction` | double (0,1] | 0.6 | Stop evicting when reserved memory drops to this fraction of capacity. Must be less than `downgrade_trigger_fraction`; configuration loading rejects an invalid pair. |

The high-level GPU path keeps per-stream reservation tracking off. The
diagnostic control remains available only through the explicit low-level
`sirius.space.gpu[].per_stream_reservation` replacement surface.

### Host Memory (`sirius.memory.host`)

Controls pinned host memory pools. One pool group is created per NUMA node (auto-detected from hardware topology).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `capacity_fraction` | double (0,1] | 0.9 | Pinned host memory capacity as a fraction of **each backing NUMA node's total RAM** (read from `/sys/devices/system/node/node<id>/meminfo`). Mutually exclusive with `capacity_bytes`; configuration loading rejects both when both values are non-null. Initialization fails if a node's capacity cannot be determined. |
| `capacity_bytes` | bytes | — | Pinned host memory capacity **per NUMA node** as absolute bytes, allocated with `cudaMallocHost`. Mutually exclusive with `capacity_fraction`; configuration loading rejects both when both values are non-null. |
| `reservation_limit_fraction` | double | 1.0 | Fraction of host capacity that can be reserved. Mutually exclusive with `reservation_limit_bytes`; configuration loading rejects both when both values are non-null. |
| `reservation_limit_bytes` | bytes | — | Absolute reservation limit. Mutually exclusive with `reservation_limit_fraction`; configuration loading rejects both when both values are non-null. |
| `downgrade_trigger_fraction` | double (0,1] | 0.9 | Start evicting host-resident data to disk when reserved memory exceeds this fraction. Must be greater than `downgrade_stop_fraction`. Eviction also requires a configured `downgrade_root_dirs`; without one the downgrade executor logs a warning and skips. |
| `downgrade_stop_fraction` | double (0,1] | 0.8 | Stop evicting when reserved memory drops to this fraction. Must be less than `downgrade_trigger_fraction`; configuration loading rejects an invalid pair. |
| `block_size` | bytes | 1Mi | Size of each allocation block in the pool. Larger blocks reduce allocation overhead but waste memory on small allocations. |
| `pool_size` | int | 128 | Number of blocks per pool. Total pool capacity = `block_size × pool_size`. |
| `initial_number_pools` | int | 4 | Number of pools pre-allocated at startup. Additional pools are created on demand. Initial host footprint = `block_size × pool_size × initial_number_pools`. |

### Disk Memory (`sirius.memory.disk`)

Controls the disk spill tier. Data evicted from host memory is written here. Disk spilling is **disabled by default** (empty `downgrade_root_dirs`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `disk_id` | int | 0 | Identifier for the disk space. |
| `capacity_bytes` | bytes | 1Ti | Maximum disk space for spill files. |
| `downgrade_root_dirs` | string | "" | Directory path for spill files. **Must be set** to enable disk spilling. Use a fast local mount (NVMe preferred). |

### Input-table compression (`sirius.compression`)

These settings control optional Simpatico compression when
`pin_table(tier=>'host')` caches input tables. Compression requires both
`enable_pin_table_compression: true` and a matching plan in `input_plan_dir`;
otherwise the table is pinned uncompressed.

| Key | DuckDB setting | Type | Default | Description |
|-----|----------------|------|---------|-------------|
| `enable_pin_table_compression` | `pin_table_compression` | bool | false | Attempt planned compression while pinning input tables to host memory. |
| `min_batch_size_bytes` | `pin_table_compression_min_batch_size_bytes` | bytes | 1Mi | Skip compression below this uncompressed batch size. `0` disables the size gate. |
| `max_compressed_fraction` | `pin_table_compression_max_compressed_fraction` | finite double >= 0 | 0.75 | Keep a compressed representation only at or below this fraction of the original batch size. `0` retains none; values above `1` deliberately permit expansion, primarily for testing encodability. |
| `input_plan_dir` | `pin_table_input_compression_plan_dir` | string | "" | Directory of per-table Simpatico plan files. An empty path leaves compression inactive. |

The YAML loader and DuckDB `SET` surface reject negative, NaN, and infinite
`max_compressed_fraction` values instead of silently changing retention behavior.

### How Downgrade Thresholds Work

Each memory tier uses a trigger/stop threshold pair to control data eviction:

```
  0%             downgrade_stop    downgrade_trigger     reservation_limit
  |─────────────────|─────────────────|────────────────────|───── capacity
       normal           hysteresis         evicting           denied
```

- Below `downgrade_stop`: normal operation, no eviction
- Between `stop` and `trigger`: no new evictions start, but in-flight evictions finish
- Above `downgrade_trigger`: actively evict data to the next lower tier
- Above `reservation_limit`: new reservations are denied (triggers OOM retry)

The gap between `trigger` and `stop` prevents oscillation — without it, evicting one batch could drop below trigger, then the next allocation re-triggers eviction.
Configuration loading enforces `0 < downgrade_stop_fraction < downgrade_trigger_fraction <= 1`
for both the high-level and low-level GPU and host surfaces. Missing or explicit-null
members retain their surface defaults before the pair is validated.

### Low-Level Explicit Memory Spaces (`sirius.space`) — advanced

> **Mutually exclusive with configured `sirius.memory` sub-blocks.** `sirius.space` is a low-level
> alternative that declares individual memory spaces directly, bypassing the high-level
> `sirius.memory` configurator. If any `sirius.space.{gpu,host,disk}` list is non-empty, the
> configuration loader rejects a simultaneous non-null `sirius.memory.{gpu,host,disk}` sub-block
> instead of silently ignoring it. A null memory sub-block is absent; a non-null mapping selects
> the high-level path even when the mapping is empty or its individual values are null. Empty
> `sirius.space` lists do not select the low-level path. This path is generally reserved for
> **low-level developer testing** — e.g. hand-placing spaces on specific devices/NUMA nodes. Most
> users should use `sirius.memory` instead.
>
> Note the key names differ from `sirius.memory`: capacities are `memory_capacity` (not
> `capacity_bytes`), and the GPU/host tiers key on `device_id` / `numa_id`.

Each tier is a **YAML sequence** of space configs. Byte fields accept suffixes; fractions are `[0,1]`.

#### `sirius.space.gpu[]` — one entry per GPU memory space

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device_id` | int | 0 | CUDA device the space lives on. |
| `per_stream_reservation` | bool | false | Track reservations per CUDA stream (forced false unless set). |
| `reservation_limit_fraction` | double [0,1] | space default | Fraction of the space that may be reserved. |
| `downgrade_trigger_fraction` | double (0,1] | space default | Start evicting above this fraction. Must be greater than `downgrade_stop_fraction`. |
| `downgrade_stop_fraction` | double (0,1] | space default | Stop evicting below this fraction. Must be less than `downgrade_trigger_fraction`. |
| `memory_capacity` | bytes | space default | Absolute capacity of the space. |

#### `sirius.space.host[]` — one entry per host (pinned) memory space

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `numa_id` | int | 0 | NUMA node the pinned pool is bound to. |
| `reservation_limit_fraction` | double [0,1] | space default | Fraction of the space that may be reserved. |
| `downgrade_trigger_fraction` | double (0,1] | space default | Start evicting above this fraction. Must be greater than `downgrade_stop_fraction`. |
| `downgrade_stop_fraction` | double (0,1] | space default | Stop evicting below this fraction. Must be less than `downgrade_trigger_fraction`. |
| `memory_capacity` | bytes | space default | Absolute capacity of the space. |
| `block_size` | bytes | pool default | Allocation block size. |
| `pool_size` | int | pool default | Blocks per pool. |
| `initial_number_pools` | int | pool default | Pools pre-allocated at startup. |

#### `sirius.space.disk[]` — one entry per disk spill space

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `disk_id` | int | 0 | Identifier for the disk space. |
| `mount_path` | string | "" | Spill directory. |
| `memory_capacity` | bytes | space default | Maximum disk space for spill files. |

```yaml
sirius:
  # NOTE: do NOT also configure sirius.memory — the loader rejects both paths together.
  space:
    gpu:
      - { device_id: 0, memory_capacity: 40Gi, reservation_limit_fraction: 0.9 }
    host:
      - { numa_id: 0, memory_capacity: 32Gi, block_size: 1Mi, pool_size: 128 }
    disk:
      - { disk_id: 0, mount_path: /tmp/sirius_disk_memory, memory_capacity: 1Ti }
```

## Executor Configuration

**Files:** `src/include/exec/config.hpp`, `src/include/creator/config.hpp`

The `sirius.executor` block configures the thread pools and task scheduling. It has four
per-pool sub-blocks: `task_creator`, `pipeline`, `downgrade`, and `scan_manager`. The
`scan_manager` sub-block is large and is documented in
[Scan Manager & IO Configuration](#scan-manager--io-configuration) below.

The thread-pool sub-blocks use these common keys; pipeline affinity is the
hardware-derived exception described below:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_threads` | int (**> 0**) | per pool (below) | Worker threads in the pool. |
| `cpu_affinity` | list of int | — | Cores to pin task-creator and downgrade threads. GPU pipeline affinity is derived from the selected GPU's CPU topology. |

### `sirius.executor.task_creator`

Thread pool (default `num_threads: 1`). Task creation
policy and within-branch priority are internal: Sirius currently creates tasks
on demand and prioritizes source-side pipelines first. The former
`sirius.executor.task_creator.strategy` and
`sirius.executor.task_creator.priority_order` keys have been removed;
configurations that still contain either key must delete it.

### `sirius.executor.pipeline`

Thread pool only (default `num_threads: 4`). No extra keys.

### `sirius.executor.downgrade`

Thread pool (default `num_threads: 1`) plus:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `monitor_period` | int (ms) | 10 | Period of the memory-pressure monitor loop. Set to `0` to disable the monitor loop entirely. |

## Scan Manager & IO Configuration

**Files:** `src/include/scan_manager/config.hpp`, `src/include/io/uring/config.hpp`, `src/include/io/rest/config.hpp`, `src/include/io/cache/config.hpp`, `src/include/io/object_store_config.hpp`

The `sirius.executor.scan_manager` block configures the scan-metadata thread pool and the Sirius IO layer that feeds the GPU scan operators.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_threads` | int (**> 2**) | remaining cores (min 4) | Threads in the scan-manager pool that run metadata tasks. Defaults to every core left after the other default pools (1 downgrade + 1 task_creator + 4 pipeline + 1 uring reactor), with a floor of 4. Rejected unless strictly greater than 2 (i.e. minimum 3). |
| `cpu_affinity` | list of int | — | Cores to pin scan-manager threads to. |
| `use_sirius_datasource` | bool | true | Route reads through the Sirius `io_uring` datasource. When false, the kvikio fallback is used (single-GPU only; multi-GPU requires the Sirius datasource). |
| `uring_n_reactors` | int (**> 0**) | 1 | Number of io_uring reactor threads for local-disk reads. |
| `rest_n_reactors` | int (**> 0**) | 2 | Number of REST reactor threads for object-store (`s3://`) reads. |
| `enable_prefetch_cache` | bool | false | Attach the pinned-memory prefetching cache in front of the backend. |

Five optional nested sub-configs tune the individual backends, caches, and the memory prefetcher:

### `scan_manager.memory_prefetcher` — background host→GPU upload of pinned-cache scan splits (`scan_manager/config.hpp`)

Overlaps the host→GPU upload of queued pinned-cache scan splits with compute:
worker threads walk the pending splits in scan execution order and convert
resident batches to GPU tier ahead of task creation, gated on GPU memory
headroom (see `scan_manager/memory_prefetcher.hpp`). Disabled by default;
single-GPU configurations only (logs a warning and disables itself otherwise).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | bool | false | Master switch for the prefetcher. |
| `num_threads` | int (**> 0**) | 2 | Prefetch worker threads; each drives one in-flight batch conversion on its own stream. |
| `min_free_fraction` | double [0,1] | 0.4 | Keep at least this fraction of the GPU space free after each prefetch; conversions (and their reservations) are only attempted above this floor. |
| `poll_interval_ms` | int (**> 0**) | 2 | Worker sweep interval while waiting for headroom / new splits. |
| `drain_quiet_ms` | int (ms) | 100 | A connector counts as actively draining (and is skipped) until this long passes since its last pop. Must exceed the scan's inter-pop interval. |

### `scan_manager.local` — io_uring backend (`io/uring/config.hpp`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_odirect` | bool | true | Use `O_DIRECT` for local-disk reads. |
| `max_n_chunks` | int | 1 | Max contiguous file segments fused into one vectored read. |

### `scan_manager.rest` — REST / S3 backend (`io/rest/config.hpp`)

TLS verification policy and the CA bundle are configured only under
`scan_manager.object_store`. The REST reactor consumes those values so signing
and transport use one trust policy; there are no separate REST YAML controls.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `request_timeout_s` | int (seconds) | 30 | Whole-request timeout and presigned-URL TTL (0 = no limit). |
| `max_connections` | int | 16 | Max concurrent in-flight connections per reactor. |
| `chunk_size` | bytes | 8Mi | Target bytes per ranged GET (scatter/device-staging paths). |
| `max_n_chunks` | int | 16 | Max file-adjacent segments fused into one scatter GET. |
| `max_read_split` | int | 16 | Max parallel ranged GETs for one contiguous host read (reads < 2 MiB stay a single GET). |
| `upkeep_interval_ms` | int (ms) | 15000 | Idle-connection keepalive interval (`curl_easy_upkeep`; 0 disables). |
| `conn_max_age_s` | int (seconds) | 20 | Max age curl may reuse a pooled connection (`CURLOPT_MAXAGE_CONN`; 0 = curl default). |
| `retry_backoff_base_ms` | int (ms) | 50 | Base backoff between retries. |
| `retry_jitter_ms` | int (ms) | 50 | Random jitter added to retry backoff. |
| `max_retry_attempts` | int | 10 | Retry attempts for transient errors. |
| `max_auth_retry_attempts` | int | 3 | Retry attempts for HTTP 403 (expired presigned URL). Kept low so a genuine AccessDenied fails fast. |
| `honor_retry_after` | bool | true | Respect the server's `Retry-After` header. |
| `perf_instrumentation` | bool | false | Record per-chunk micro-timings (chunk_get, queue_wait, ttfb, h2d) into perf counters. |
| `footer_probe_bytes` | bytes | 512Ki | Suffix-range window for the parquet footer probe. Must cover the footer, so err large. |
| `list_max_matches` | int | 100000 | Cap on files a glob/listing may accumulate (throws "narrow the glob prefix", never truncates). |
| `list_max_scanned` | int | 1000000 | Cap on objects a LIST sweep may scan across pages (throws, never truncates). |

### `scan_manager.cache` — prefetching cache (`io/cache/config.hpp`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `inflight_io_chunk_budget` | int (**> 0**) | 2048 | Max in-flight IO chunks (enforced by admission control). |
| `eviction_threshold_fraction` | double [0,1] | 0.6 | Start evicting when the pool fills to this fraction. |
| `min_prefetching_budget_fraction` | double [0,1] | 0.05 | Floor of the budget reserved for prefetching. |
| `dispose_after_use` | bool | false | Discard chunks immediately after use. |

### `scan_manager.object_store` — S3 credentials & endpoint (`io/object_store_config.hpp`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `endpoint` | string | "" | S3 endpoint URL. |
| `region` | string | "" | AWS region. |
| `access_key` / `secret_key` | string | "" | Static credentials. |
| `session_token` | string | "" | STS session token for temporary credentials. |
| `signing_mode` | enum: `presigned`, `header` | `presigned` | SigV4 form: `presigned` (auth in the URL query string) or `header` (`Authorization` + `x-amz-*` headers). Values are lowercase. |
| `s3_transport` | enum: `auto`, `http`, `https`, `rdma` | `auto` | Transport selection. Values are lowercase; `https` is an alias for `http`. `auto` lets the backend choose from the URI scheme and endpoint. |
| `ca_bundle_path` | string | "" | Sole YAML source for the REST endpoint's PEM CA bundle. |
| `tls_verify` | bool | true | Sole YAML source for REST endpoint certificate verification. |

## Operator Parameters

**File:** `src/include/sirius_config.hpp` — `operator_params` struct

The four batch/partition sizes (`scan_task_batch_size`, `hash_partition_bytes`,
`concat_batch_bytes`, `sort_sample_bytes`) share one built-in default. Without an
explicit GPU capacity in YAML, it is computed at startup as
`clamp(smallest visible physical GPU memory / 40, 512 MiB, 5 GiB)` (800 MiB when no
GPU is visible). When the active YAML memory path explicitly caps GPU capacity, the
default is narrowed to
`min(physical-memory default, max(1 byte, smallest resolved GPU capacity / 40))`.
`max_build_hash_table_bytes` defaults to **2× that batch default**. Explicit
`operator_params` values are applied afterward and still override each value
individually.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scan_task_batch_size` | Shared physical/effective GPU batch default described above | Target batch size for DuckDB scan tasks; must be greater than zero |
| `enable_compressed_materialization` | true | Store eligible integer and fixed-point DECIMAL values in value-preserving narrower carriers when exact pin-time bounds permit it; restore native carriers at type-sensitive boundaries. |
| `max_sort_partition_bytes` | 0 (auto) | Max bytes per sort partition. Auto = 33% of GPU memory. |
| `hash_partition_bytes` | Shared physical/effective GPU batch default described above | Target partition size for hash joins and group-bys; must be greater than zero |
| `concat_batch_bytes` | Shared physical/effective GPU batch default described above | Target output batch size for CONCAT operator |
| `sort_sample_bytes` | Shared physical/effective GPU batch default described above | Bytes sampled before computing sort partition boundaries |
| `max_build_hash_table_bytes` | 2× batch default | Max build-side size for BUILD_PROBE join mode |
| `max_broadcast_join_size` | 256 MiB | Max build-side size eligible for a broadcast join. A build below this size is replicated to every GPU (instead of hash-partitioned) when it is tiny, or when the DuckDB-estimated probe-to-build row ratio is at least `num_gpus * 1.25`. |
| `max_sort_partition_memory_fraction` | 0.33 | Fraction of GPU memory per sort partition when `max_sort_partition_bytes` is 0 |
| `mark_join_build_switch_ratio` | 8.0 | For STANDARD MARK joins, build on the smaller (left) side when `right_rows >= ratio * left_rows` (0 disables) |
| `enable_dynamic_filter` | true | Enable runtime filters for eligible hash joins. Plan-time wiring admits keys by join type (not join mode); at delivery, any join whose build side arrives as one whole batch publishes — a single-partition or broadcast `BUILD_PROBE` build, and a single-partition `STANDARD`/`MIXED_JOIN` build on the same terms. Targets may be probe scans or join-edge endpoints. An eligible build selects a raw exact IN-list for 1–12 supported build rows, otherwise a hash IN-list within the L2 budget or a Bloom. |
| `enable_dynamic_zone_map_filter` | false | Publish build-key min/max filters in addition to membership filters. Parquet scans use them for row-group pruning; duckdb-native scans apply them post-decode. Requires `enable_dynamic_filter`; intended for clustered-keyset workloads. |
| `dynamic_filter_domain_coverage_threshold` | 0.9 | Positive finite threshold. Before constructing either a membership filter or zone map, skip the key when the complete build covers at least this fraction of the key's unfiltered base-table row bound. Applies only to build keys proven unique in their base relation, with evidence from DuckDB-native scans. Values above 1.0 disable the gate; exactly 1.0 fires only at full coverage. |
| `dynamic_filter_inlist_max_l2_fraction` | 0.125 | Finite threshold in [0, 1]: maximum estimated cuco-set size for the exact hash IN-list, as a fraction of the smallest probe-GPU L2. Larger sets use Bloom when supported. For keys not handled by the raw IN-list, 0 selects Bloom when supported, while 1.0 reproduces the legacy L2-fit rule only when L2 size is known. If L2 size is unknown, the hash IN-list is ineligible and selection falls back to Bloom or no membership filter. The 0.125 default comes from a GB300 residency sweep: hash-set probe cost is flat below ~0.28 of L2 and degrades beyond it, while Bloom was at least 2.2x faster at every swept set size. |
| `dynamic_filter_keep_threshold` | 0.9 | Finite threshold in [0, 1] for disabling post-decode filtering once a measured split keeps more than this fraction of its rows; 1.0 keeps filtering always on. |
| `enable_pinned_zone_map_pruning` | true | Capture per-chunk min/max statistics while pinning and use them to skip cached chunks that cannot match a scan filter. |
| `admission_bytes_per_gpu` | 0 (off) | Target projected scan-output bytes per GPU. At admission the engine estimates a query's total scan output and takes the smallest GPU subset that keeps each GPU under this figure, bounded by `topology.gpus_per_query`. `0` disables the estimate, leaving the allocation to `topology.gpus_per_query` alone. |
| `avg_variable_column_bytes` | 32 | Per-row width assumed for variable-width columns (VARCHAR, LIST, STRUCT, ARRAY) when estimating scan output. Fixed-width columns use their real carrier width. Only consulted when `admission_bytes_per_gpu` is non-zero. |

**Note:** `admission_bytes_per_gpu` is a parallelism dial, not a memory budget. Peak GPU residency is bounded by partition sizing (`hash_partition_bytes` and the batch settings), not by the admitted GPU count — a query on fewer GPUs processes more partitions sequentially at roughly unchanged peak memory, trading wall-clock for freed devices. Tune it against how much of the fleet a query should occupy, not against VRAM.

**Note:** `max_build_hash_table_bytes` can be larger than `concat_batch_bytes`. When it is, the partition operator configures CONCAT to concatenate all batches, enabling the more efficient BUILD_PROBE join mode for larger build sides. Other joins (STANDARD, MIXED) still use `concat_batch_bytes` as the batch size threshold.

Runtime distinct-build probing is an internal join policy, not a user configuration choice. It is
temporarily disabled while the cuCollections defect tracked in #1600 remains unresolved. The
engine retains the guarded single-pass `cudf::distinct_hash_join` path for policy-controlled use
after that dependency is fixed.

## Telemetry

```yaml
sirius:
  telemetry:
    enable_quent: true
    output_directory: telemetry_data
    engine_name: siriusDB
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_quent` | bool | true | Emit Quent telemetry using the configured exporter. When false, telemetry uses the noop exporter. |
| `exporter` | string | `ndjson` | Quent filesystem exporter: `ndjson`, `msgpack`, or `postcard`. |
| `output_directory` | non-empty string | `telemetry_data` | Directory for Quent telemetry files. |
| `engine_name` | non-empty string | `siriusDB` | Engine name reported in engine-level telemetry. |

Per-query labels are configured separately from YAML. They can be set with the
`sirius_set_query_label` SQL function or inline with the `query_label` named
parameter on `gpu_execution(...)`:

```sql
-- Applies to the next Sirius query, including transparent plain-SQL execution.
CALL sirius_set_query_label('tpch_q1_iter1');
SELECT *
FROM lineitem
WHERE l_orderkey < 100;

-- Inline label for an explicit gpu_execution call.
CALL gpu_execution(
  'SELECT * FROM lineitem WHERE l_orderkey < 100',
  query_label = 'tpch_q1_iter1'
);
```

`sirius_set_query_label` is consumed once by the next Sirius query. For explicit
`gpu_execution(...)`, an inline `query_label` parameter takes precedence over a
pending label set with `sirius_set_query_label`.

### Generating Query Telemetry

To generate telemetry from Sirius queries, update the Sirius config file used by
the query run and enable Quent export:

```yaml
sirius:
  telemetry:
    enable_quent: true
    output_directory: telemetry_data
    engine_name: siriusDB
```

Load that config through the normal config resolution path, usually by setting
`SIRIUS_CONFIG_FILE=/path/to/sirius.yaml`. Any Sirius query run with
`enable_quent: true` writes Quent ndjson files into `output_directory` by default. Set
`exporter: postcard` for compact benchmark or CI telemetry.

For TPC-H Parquet runs, the helper script runs queries and labels each
`(query, iteration)` pair before executing it:

```bash
pixi run -- ./test/tpch_performance/run_tpch_parquet_and_generate_telemetry.sh \
  --iterations 1 \
  --parquet-dir /data/tpch/sf100/p16/zstd-8/ \
  100
```

Pass `--config <path>` to use a full Sirius config. If no query numbers are
provided, the script runs all 22 TPC-H queries.

Start the Quent analyzer server over the same telemetry directory to view the
captured telemetry:

```bash
pixi run quent
```

The `quent` Pixi task defaults to `telemetry_data`, and runs the telemetry
server with the UI enabled. If the config uses a different `output_directory`,
pass that path as the task argument:

```bash
pixi run quent /path/to/telemetry_data
```

Then open `http://localhost:8080` and select the captured Sirius engine/query.

## Thread Pool Configuration

Summary of the four executor thread pools. See [Executor Configuration](#executor-configuration)
and [Scan Manager & IO Configuration](#scan-manager--io-configuration) for the full YAML keys and
per-pool extras.

| Pool | YAML block | Default Threads | Thread Name Prefix | Purpose |
|------|-----------|----------------|-------------------|---------|
| `task_creator` | `executor.task_creator` | 1 | `task_creator` | Task creation from scheduling requests |
| `gpu_pipeline_executor` | `executor.pipeline` | 4 | `gpu_pipeline` | GPU pipeline task execution |
| `downgrade_executor` | `executor.downgrade` | 1 | `downgrade` | Data tier migration (GPU→Host) |
| `scan_manager` | `executor.scan_manager` | remaining cores (min 4) | `scan_manager` | Scan metadata production + IO reactor management |

The task-creator, downgrade, and scan-manager pools support optional CPU affinity lists
(`cpu_affinity`) for core pinning. GPU pipeline affinity is derived per executor from the selected
GPU's CPU topology. `num_threads` must be `> 0` for every pool except `scan_manager`, which requires
`> 2`.

## DuckDB SET Variables

Registered in `src/sirius_extension.cpp`. These can be changed at runtime:

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `sirius_log_backend` | `spdlog` | Log sink: `spdlog`, `duckdb`, or `noop` |
| `sirius_log_level` | `info` | Log level: trace, debug, info, warn, error (`spdlog` only) |
| `sirius_log_dir` | `log` | Log output directory (`spdlog` only) |
| `sirius_log_flush_seconds` | 3 | Log flush interval in seconds (`spdlog` only) |

- **`spdlog`** (default): writes the daily-rotated `<sirius_log_dir>/sirius.log`, honouring
  `sirius_log_level` and `sirius_log_flush_seconds`.
- **`duckdb`**: routes logs into DuckDB's own logging under the `Sirius` log type; enable and
  read them with `CALL enable_logging(); SELECT * FROM duckdb_logs WHERE type = 'Sirius';`.
  Level and enable are governed by DuckDB, not `sirius_log_level`.
- **`noop`**: discards all logs.

These can also be set at load via the `SIRIUS_LOG_BACKEND`, `SIRIUS_LOG_DIR`, and
`SIRIUS_LOG_LEVEL` environment variables.

### Expression Evaluation

**File:** `src/include/expression_evaluator/expression_evaluator_strategy.hpp`

| Variable | Default | Description |
|----------|---------|-------------|
| `expression_evaluator_strategy` | `ast_interpret` | Expression evaluator strategy: `materialize`, `ast_interpret`, or `ast_jit` |

`expression_executor_strategy` remains registered as a deprecated compatibility alias for
`expression_evaluator_strategy`; new configuration should use the evaluator name.

### Scan

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_compressed_materialization` | true | Keep eligible integer and fixed-point DECIMAL values in narrower physical carriers until a native semantic boundary. |

The scan-task batch target is derived from effective GPU capacity. Advanced
benchmark and test envelopes may still override `scan_task_batch_size` in YAML
under `sirius.operator_params`, where it must remain greater than zero, but it
is not a normal session setting.

`enable_compressed_materialization` is also accepted in YAML under `sirius.operator_params`.
At pin time, exact per-chunk bounds select stored carriers. At query planning, a matching pinned
entry's recorded column-storage metadata determines the scan carriers; Parquet footer and DuckDB
catalog statistics are not consulted for this decision. Unpinned scans stay native, and exact
runtime bounds guard any narrowing cast required after a plan becomes stale. Changing the setting
does not rewrite an existing pinned entry; cached scans normalize its stored carriers for the
current query. See [Compressed Materialization](compressed-materialization.md) for eligibility,
restoration boundaries, and cache-reservation behavior.

```sql
SET enable_compressed_materialization = false;
```

### Pipeline / Operator

| Variable | Default | Description |
|----------|---------|-------------|
| `max_sort_partition_bytes` | 0 (auto) | Max sort partition bytes |
| `max_sort_partition_memory_fraction` | 0.33 | Auto sort-partition fraction when `max_sort_partition_bytes` is 0 |
| `hash_partition_bytes` | Shared physical/effective GPU batch default | Hash partition target size; must be greater than zero |
| `sort_sample_bytes` | Shared physical/effective GPU batch default | Bytes sampled before computing sort boundaries |
| `max_build_hash_table_bytes` | 2× batch default | Max build-side hash table bytes |
| `max_broadcast_join_size` | 256 MiB | Max build-side size eligible for a broadcast join |
| `mark_join_build_switch_ratio` | 8.0 | STANDARD MARK join build-side switch ratio (0 disables) |

Eligible GROUP BY and TOP_N merge pipelines are fused automatically. This is an engine-owned plan
policy rather than a user configuration choice; see
[Merge fusion](physical-plan-generation.md#merge-fusion).

The CONCAT output-batch target is derived from effective GPU capacity. Advanced
benchmark and test envelopes may still override `concat_batch_bytes` in YAML
under `sirius.operator_params`, but it is not a normal session setting.

Runtime distinct-build probing is also engine-owned and is temporarily disabled pending #1600.

### GPU Admission

| Variable | Default | Description |
|----------|---------|-------------|
| `admission_bytes_per_gpu` | 0 (off) | Target projected scan-output bytes per GPU; `0` disables the estimate, leaving the allocation to `topology.gpus_per_query` |
| `avg_variable_column_bytes` | 32 | Per-row width assumed for variable-width columns when estimating scan output; must be greater than zero |

Both take effect from the next query, since admission runs per query. `topology.gpus_per_query`
is YAML-only — it is read once when the GPU memory spaces are configured.

```sql
SET admission_bytes_per_gpu = 34359738368;  -- 32 GiB
```

### Dynamic Filters

Dynamic membership-filter pushdown is automatic and enabled by default. The
clustered-keyset zone-map path is automatic-off by default because it does not
repay its row-level cost on scattered keys. Advanced benchmark and diagnosis
envelopes can override either behavior in YAML under `sirius.operator_params`.

`enable_dynamic_filter` replaces the former `enable_dynamic_filter_pushdown` and the temporary
`enable_dynamic_filter_sip`; the old keys are not aliased — a YAML file still naming them is
rejected as unknown, and the old `SET` variables no longer exist.

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_dynamic_filter` | true | Enable runtime filters for eligible hash joins. Plan-time wiring admits keys by join type (not join mode); at delivery, any join whose build side arrives as one whole batch publishes — a single-partition or broadcast `BUILD_PROBE` build, and a single-partition `STANDARD`/`MIXED_JOIN` build on the same terms. Targets may be probe scans or join-edge endpoints. An eligible build selects a raw exact IN-list for 1–12 supported build rows, otherwise a hash IN-list within the L2 budget or a Bloom. |
| `enable_dynamic_zone_map_filter` | false | Publish build-key min/max filters in addition to membership filters. Parquet scans use them for row-group pruning; duckdb-native scans apply them post-decode. Requires `enable_dynamic_filter`; intended for clustered-keyset workloads. |
| `dynamic_filter_domain_coverage_threshold` | 0.9 | Positive finite threshold. Before constructing either a membership filter or zone map, skip the key when the complete build covers at least this fraction of the key's unfiltered base-table row bound. Applies only to build keys proven unique in their base relation, with evidence from DuckDB-native scans. Values above 1.0 disable the gate; exactly 1.0 fires only at full coverage. |
| `dynamic_filter_inlist_max_l2_fraction` | 0.125 | Finite threshold in [0, 1]: maximum estimated cuco-set size for the exact hash IN-list, as a fraction of the smallest probe-GPU L2. Larger sets use Bloom when supported. For keys not handled by the raw IN-list, 0 selects Bloom when supported, while 1.0 reproduces the legacy L2-fit rule only when L2 size is known. If L2 size is unknown, the hash IN-list is ineligible and selection falls back to Bloom or no membership filter. The 0.125 default comes from a GB300 residency sweep: hash-set probe cost is flat below ~0.28 of L2 and degrades beyond it, while Bloom was at least 2.2x faster at every swept set size. |
| `dynamic_filter_keep_threshold` | 0.9 | Finite threshold in [0, 1] for disabling post-decode filtering once a measured split keeps more than this fraction of its rows; 1.0 keeps filtering always on. |

The direct DuckDB session overrides are registered only when the process
explicitly enables Sirius test options; they are not part of the normal user
surface.

### Pinned Tables

Pinned-table zone-map capture and pruning are automatic. The advanced YAML escape hatch is under
`sirius.operator_params` for benchmark and diagnosis envelopes; it is not a normal session choice.

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_pinned_zone_map_pruning` | true | Capture pinned-chunk zone maps at pin time and use them to prune cached scans. |

Setting the YAML value to `false` before startup avoids the extra GPU reductions and creates a
statless entry. Enabling it later does not add statistics to that entry; re-pin the table with
the setting enabled. See
[Pinned-table zone maps](scan.md#zone-maps) for supported types, pruning, and re-pin behavior.

The direct DuckDB `SET` override is registered only when the process explicitly enables Sirius
test options; it is not part of the normal user surface.

### Pinned-Table Compression (Simpatico)

| Variable | Default | Description |
|----------|---------|-------------|
| `pin_table_compression` | false | Enable Simpatico compression for `pin_table(tier=>'host')` chunks. |
| `pin_table_input_compression_plan_dir` | (empty) | Directory of per-table Simpatico plan files (`<table_name>.<ext>`, multi-column plan DSL). Tables with no matching file are pinned uncompressed. No effect on spill compression. |
| `pin_table_compression_min_batch_size_bytes` | 1 MiB | Minimum uncompressed batch size below which pin-table compression is skipped. |
| `pin_table_compression_max_compressed_fraction` | 0.75 | Discard the compressed form and pin uncompressed when the compressed size exceeds this fraction of the original (compression saved too little). |

Both size gates are evaluated on the narrowed table when compressed materialization is active.
See [Compressed Pinning](compressed-pinning.md) for tier selection, plan authoring, and results.

### Transparent Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `gpu_execution` | true | Enable transparent GPU execution for plain SQL queries. When enabled, supported queries are automatically intercepted and executed on the GPU. When disabled, only explicit `CALL gpu_execution('...')` uses the GPU. |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_duckdb_fallback` | true | Fall back to DuckDB CPU execution on Sirius errors. Gates both plan-time fallback (unsupported operator/type) and runtime fallback (GPU execution failure) on the transparent path, plus the legacy `CALL gpu_execution(...)` path. Set to `false` to surface Sirius errors instead of falling back. |
| `enable_regex_jit_impl` | true | Use JIT regex implementation |


## Legacy Config Flags

### Legacy-release DuckDB settings

The following settings only control the legacy `gpu_processing` path. Sirius registers them
when built with `ENABLE_LEGACY_SIRIUS=ON`, including the `legacy-release` preset used by
`make legacy-release`. Normal builds omit them from `duckdb_settings()` and reject attempts to
`SET` them.

| Variable | Default | Description |
|----------|---------|-------------|
| `use_pin_memory` | true | Use pinned memory for legacy CPU↔GPU transfers |
| `use_pin_memory_for_caching` | false | Use pinned memory for the legacy scan cache |
| `use_cudf_expr` | true | Use cuDF in the legacy expression executor |
| `use_custom_top_n` | true | Use the legacy custom top-N kernel |
| `use_opt_table_scan` | true | Use the legacy optimized table scan |
| `opt_table_scan_num_streams` | 8 | CUDA streams used by the legacy optimized scan |
| `opt_table_scan_memcpy_size` | 64 MiB | Copy chunk size used by the legacy optimized scan |
| `print_gpu_table_max_rows` | 1000 | Maximum rows rendered by the legacy GPU-table printer |
| `enable_fallback_check` | false | Enable legacy fallback validation |
| `modified_pipeline` | false | Enable legacy modified-pipeline scheduling |

### Static flags

**File:** `src/include/config.hpp`

Static constants from `namespace duckdb::Config` (used by legacy Sirius) and `namespace sirius::Config`:

| Flag | Value | Namespace |
|------|-------|-----------|
| `USE_PIN_MEM_FOR_CPU_PROCESSING` | true | `duckdb::Config` |
| `USE_PIN_MEM_FOR_CACHING` | false | `duckdb::Config` |
| `USE_CUDF_EXPR` | true | `duckdb::Config` |
| `ENABLE_DUCKDB_FALLBACK` | true | `duckdb::Config` |
| `NUM_GPU_EXECUTOR_THREADS` | 2 | `sirius::Config` |
| `NUM_PIPELINE_EXECUTOR_THREADS` | 1 | `sirius::Config` |
| `NUM_GPU` | 1 | `sirius::Config` |

These are compile-time defaults. Runtime configuration via `sirius_config` and DuckDB SET variables takes precedence.

## Key Files

| File | Purpose |
|------|---------|
| `src/include/sirius_config.hpp` | Config class, operator_params, thread pool configs |
| `src/include/config.hpp` | Legacy config flags |
| `src/sirius_extension.cpp` | SET variable registration |
| `src/include/scan_manager/config.hpp` | Scan manager config (thread pool, IO reactors, prefetch cache, object store) |
| `src/include/io/uring/config.hpp`, `io/rest/config.hpp`, `io/cache/config.hpp`, `io/object_store_config.hpp` | Per-backend IO / cache / object-store sub-configs |

## Tuned profile: GB300, TPC-H SF1000 host-pinned

Measured 2026-07-31 (72-core GB300, 256 GB HBM, 3 hot iterations, results
byte-identical to defaults). Relative to the stock config the following
deltas are worth **-13% suite hot time** (20.8 s → 18.1 s; q9 -19%):

```yaml
sirius:
  memory:
    host:
      block_size: 64Mi       # default 1Mi: ~5000 copy segments per 5 GB batch
      pool_size: 8           # keep block_size x pool_size at 512Mi per pool
  executor:
    scan_manager:
      memory_prefetcher:     # requires the memory prefetcher (PR #1181)
        enable: true
        num_threads: 3       # 2 is the swept default; 3 buys q21 ~3% under 8 pipeline threads
    pipeline:
      num_threads: 8         # default 4; cliff at 12 (q1/q6 regress)
```

Attribution: host `block_size` 1 Mi → 64 Mi removes per-segment submission
overhead in batched host→GPU copies (~11 ms of every 39 ms five-GB
conversion); sweep 16-64 Mi if small-host-allocation fragmentation is a
concern. `pipeline.num_threads` 4 → 8 helps task-parallel aggregation
queries (q1 -16%, q12 -14%). The prefetcher block overlaps pinned-cache
uploads with compute (see `scan_manager.memory_prefetcher` above). Numbers
include the cuCascade all-valid null-mask conversion fix; without it,
expect roughly half the converter-side gain.

The knobs that did NOT help on this hardware, all measured full-suite:
prefetcher `min_free_fraction` below 0.4, prefetcher threads above 3,
pipeline threads above 8. The H2D interconnect sustains ~350-380 GB/s
regardless of stream count; past these settings the link, not scheduling,
is the bound.
