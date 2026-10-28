#Configuration

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

### Concurrent query admission

`sirius.max_concurrent_queries` is a positive startup integer, default **1**, shared by
connections to one DatabaseInstance. It bounds initialization through retirement. Additional
queries wait in a cancellable FIFO queue. The former
`sirius.executor.scan_manager.max_concurrent_queries` key is a deprecated alias; conflicting
values fail configuration loading. See [Concurrent queries](concurrent-queries.md) for the
maintenance, error and multi-GPU contracts.

### `SIRIUS_DISABLE`

Set `SIRIUS_DISABLE=1` to prevent Super Sirius from initializing. Useful for CPU-only benchmarks, since Super Sirius claims most GPU and pinned host memory on startup.

```bash
export SIRIUS_DISABLE=1
```

These related modes have different behavior:

| Mode | Query execution | NVTX/Quent behavior |
|------|-----------------|---------------------|
| `SIRIUS_DISABLE=1` | Ordinary SQL runs in DuckDB; Super Sirius is never attempted | Sirius does not create its Quent context or configure automatic NVTX injection |
| `SET gpu_execution = false` | Ordinary SQL runs in DuckDB for that connection | The process-wide Sirius runtime remains initialized and, when Quent is enabled, NVTX injection remains armed because the setting is reversible and other connections may use Sirius; DuckDB emits no NVTX events unless DuckDB or another loaded component explicitly calls NVTX |
| Automatic CPU fallback | Sirius first rejects the GPU plan or fails during GPU execution, then DuckDB executes the CPU plan | NVTX remains active so any Sirius/libcudf work before the fallback is retained; the DuckDB portion emits events only if the executing code calls NVTX |

Use `SIRIUS_DISABLE=1` for a pure DuckDB CPU baseline in a process that contains the Sirius extension. Use `SET gpu_execution = false` when the initialized Sirius runtime must remain available to turn back on later.

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
  max_concurrent_queries: 1
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
    scan_manager:
      num_threads: 4
      backend: native
      uring_n_reactors: 4
      cache: { mode: none, eviction: lru }
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
    enable_dynamic_filter_multi_partition: true   # Bloom accumulation for multi-partition builds
    max_dynamic_filter_bloom_bytes_per_gpu: 256Mi  # accumulated Bloom arrays across keys, per GPU
    enable_dynamic_zone_map_filter: false  # optional parquet-read/native-post-decode min/max
    dynamic_filter_domain_coverage_threshold: 0.9  # skip all filters for keys meeting known-domain coverage
    dynamic_filter_inlist_max_l2_fraction: 0.125  # hash-IN-list fraction of known probe-GPU L2 (0 = Bloom for non-small keys; 1.0 = full L2)
    dynamic_filter_keep_threshold: 0.9  # disable a scan's filtering when a split keeps > this fraction
    enable_pinned_zone_map_pruning: true  # capture and use per-chunk stats for pinned tables
    enable_runtime_size_estimation: false  # project total port input from upstream ratios
    enable_decimal_sum_stats_bound: true  # decide decimal and BIGINT SUM width from statistics at plan time
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
| `downgrade_trigger_fraction` | double (0,1] | 0.9 | Start HOST memory reclamation above this fraction. Must be greater than `downgrade_stop_fraction`. File-cache reclaim needs no DISK tier and only frees cache memory above the prefetch reservation; spilling HOST repository data requires configured `downgrade_root_dirs`. |
| `downgrade_stop_fraction` | double (0,1] | 0.8 | Stop evicting when reserved memory drops to this fraction. Must be less than `downgrade_trigger_fraction`; configuration loading rejects an invalid pair. |
| `block_size` | bytes | 1Mi | Size of each allocation block in the pool. Larger blocks reduce allocation overhead but waste memory on small allocations. |
| `pool_size` | int | 128 | Number of blocks per pool. Total pool capacity = `block_size × pool_size`. |
| `initial_number_pools` | int | 4 | Number of pools pre-allocated at startup. Additional pools are created on demand. Initial host footprint = `block_size × pool_size × initial_number_pools`. |

### Disk Memory (`sirius.memory.disk`)

Controls the disk spill tier. Data evicted from host memory is written here. Disk spilling is **disabled by default** (empty `downgrade_root_dirs`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `disk_id` | int | 0 | Identifier for the disk space. |
| `capacity_bytes` | bytes | 1Ti | Capacity bound for reservations / in-flight conversions, not a quota on stored spill files. The filesystem must have enough free space. |
| `downgrade_root_dirs` | string | "" | Directory path for spill files. **Must be set** to enable disk spilling. Use a fast local mount (NVMe preferred). |

Raw and compressed `.hpln` spills use the DISK space's shared `cucascade::io::ioctx`.
The default is adopted lazily on first conversion and shared process-wide through a weak
cache. Automatic selection prefers **real GDS via cuFile**, then **usable io_uring**, then
**cuFile POSIX/staging** if io_uring is unavailable or forbidden. The context owns its
private staging and reactor teardown; see [DISK I/O and ownership](memory-management.md#disk-io-and-ownership).

The existing `disk_id`, `capacity_bytes`, and `downgrade_root_dirs` keys and spill-file
layouts are unchanged. There are no additional DISK-backend or per-mount reactor YAML
settings, and no backend registration is required. `sirius.executor.scan_manager.backend`,
reactor counts and cache settings configure independent scan contexts, not this spill
default. The removed kvikIO backend is not available for reads or writes.

### Input-table compression (`sirius.compression`)

These settings control optional Simpatico compression when
`pin_table` caches input tables. Prefer the per-call
`CALL pin_table(..., compression => true)` argument to request compression for a specific pin.
The settings below supply the default for calls that omit the argument and configure plan lookup
and retention gates. Compression also requires a matching plan in `input_plan_dir`; otherwise the
table is pinned uncompressed.

| Key | DuckDB setting | Type | Default | Description |
|-----|----------------|------|---------|-------------|
| `enable_pin_table_compression` | `pin_table_compression` | bool | false | Default for `pin_table` calls that omit the `compression` argument. |
| `min_batch_size_bytes` | `pin_table_compression_min_batch_size_bytes` | bytes | 1Mi | Skip compression below this uncompressed batch size. `0` disables the size gate. |
| `max_compressed_fraction` | `pin_table_compression_max_compressed_fraction` | finite double >= 0 | 0.75 | Keep a compressed representation only at or below this fraction of the original batch size. `0` retains none; values above `1` deliberately permit expansion, primarily for testing encodability. |
| `input_plan_dir` | `pin_table_input_compression_plan_dir` | string | "" | Directory of per-table Simpatico plan files. An empty path leaves compression inactive. |

The YAML loader and DuckDB `SET` surface reject negative, NaN, and infinite
`max_compressed_fraction` values instead of silently changing retention behavior.

### Spill compression (`sirius.compression`)

With spill compression on, a GPU batch the downgrade executor moves to the host or disk tier is
stored Simpatico-compressed and decoded on its way back to the GPU. Plans are chosen per column
from its type (see [Memory Management](memory-management.md#spill-compression)); no plan files are
involved. A batch that does not qualify, saves too little, or fails to encode spills
uncompressed. Compressed DISK spills use the same per-space ioctx as raw spills, with
unchanged `.hpln` layout. This I/O transport change does not change the settings below or
the process-wide SQL `spill_compression` semantics.

| Key | DuckDB setting | Type | Default | Description |
|-----|----------------|------|---------|-------------|
| `enable_spill_compression` | `spill_compression` | bool | false | Compress spilled batches. The setting applies to the whole process. |
| `spill_min_batch_bytes` | — | bytes | 64Mi | Spill batches smaller than this uncompressed. |
| `spill_max_compressed_fraction` | — | finite double >= 0 | 0.75 | A spilled batch stays compressed only at or below this fraction of its device size. Independent of the input-table `max_compressed_fraction`. |
| `spill_encode_reserve_fraction` | — | double in [0, 1] | 0.5 | Encode only when this fraction of the batch's size could be reserved in its GPU memory space; otherwise the batch spills uncompressed. `0` skips the check. |

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
> `sirius.memory` configurator. If any `sirius.space.{
  gpu, host, disk}` list is non-empty, the
> configuration loader rejects a simultaneous non-null `sirius.memory.{
  gpu, host, disk}` sub-block
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
| `memory_capacity` | bytes | space default | Reservation / in-flight conversion bound, not a stored-file quota. |

These spaces use the same [lazy DISK I/O default](#disk-memory-siriusmemorydisk),
independent of the scan contexts; the low-level YAML surface adds no I/O selection knobs.

```yaml
sirius:
#NOTE : do NOT also configure sirius.memory — the loader rejects both paths together.
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

HOST downgrade executors synchronously reclaim idle staging chunks from the scan manager's
default local ioctx's `fs_cache` before spilling repository batches or queued tasks. If a byte-target request
is still short after those workers finish, a forced pass reclaims additional unpinned,
non-loading chunks up to the remaining deficit, even if a future scan subscribes to them.
Only bytes above the cache's prefetch reservation (`scan_manager.cache.min_prefetching_budget_fraction`
of each HOST space) relieve HOST pressure: the executor reclaims and credits only those bytes,
from its own NUMA space. Chunks inside the reservation are kept. Reclaim performs no disk writes
and needs no DISK tier. Without DISK, a HOST executor whose cache holds nothing above the
reservation backs off like any other non-viable executor. GPU executors, independent DISK spill
contexts and routed remote ioctx caches do not participate. This adds no configuration keys.

## Scan Manager & IO Configuration

**Files:** `src/scan_manager/config.hpp`, `src/sirius_config.cpp` (YAML parsing); the IO sub-configs are cuCascade's, under `cucascade/include/cucascade/io/`: `uring/config.hpp`, `rest/config.hpp`, `cufile/config.hpp`, `tls/config.hpp`, `cache/config.hpp`, `object_store_config.hpp`

The `sirius.executor.scan_manager` block configures the scan-metadata thread pool and the cuCascade IO layer that feeds the GPU scan operators.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_threads` | int (**> 0**) | remaining cores (min 4) | Metadata-producer worker threads. |
| `cpu_affinity` | list of int | — | Cores to pin scan-manager threads to. |
| `backend` | enum: `native`, `cufile` | `native` | IO backend for reads. `native` uses cuCascade's native IO stack: io_uring for local paths and native REST (HTTP/1.1 over io_uring, with kTLS or user-space TLS) for `s3://`; `cufile` uses cuFile with compatibility/POSIX fallback for local regular files and keeps REST for `s3://`. S3 listing and globs always use REST. cuFile is single-GPU only: a multi-GPU configuration forces `native` with a warning. `sirius` remains a deprecated alias of `native`. `backend: kvikio` and any `scan_manager.kvikio` block are rejected with an error to use `native`; they are not accepted as aliases. Values are lowercase. |
| `uring_n_reactors` | int (**> 0**) | 4 | Number of uring reactors for local-disk reads. Each reactor is one worker thread with its own io_uring ring and request queue, and allocates its own pinned staging from the host memory space when the IO context starts: 64 MiB of whole host blocks (at most 64 slots, at least one block). |
| `rest_n_reactors` | int (**>= 0**) | 2 | Number of REST reactors for object-store (`s3://`) reads, each one worker thread with its own io_uring ring and HTTP/1.1 connection pool. A total for the process, never per GPU. `0` sizes the pool from the egress NIC (`rest.interface`, else the default route's interface): when the NIC's queue IRQs are steered to a strict subset of its NUMA node's cores, one reactor per distinct IRQ CPU, each pinned to that CPU (`rest.pin_to_irq_cpus`) at the highest scheduling priority the process may set; otherwise 2 unpinned reactors. Pinning never widens the process's inherited CPU affinity. |
| `cufile_n_reactors` | int (**> 0**) | 4 | Number of cuFile reactors for local reads, each with one worker thread. Single-GPU only. |
| `max_readahead_scans` | int | — (unset) | Scans the readahead may keep in flight, and the switch that runs it at all. See below. |
| `readahead_strategy` | enum: `eager`, `opportunistic` | — (unset) | When the readahead issues. Unset takes the serving backend's own preference: `eager` for object-store (REST) reads, `opportunistic` for local (uring) ones. Values are lowercase. |

Caching itself is configured in the [`cache`](#scan_managercache--read-path-caching-iocacheconfighpp)
sub-config below. cuFile cannot use `fs_cache` for scan readahead.
Neither `cache.mode: cucs` nor an explicit positive `max_readahead_scans`
creates a readahead manager without an eligible advertised backend. This also
excludes empty queries, cuFile-only queries and requested pinned caches that failed to build.
The readahead policies below apply only to eligible backends; routed REST reads retain their own policy.

`max_readahead_scans` has three states:

| Value | Effect |
|-------|--------|
| unset (default) | Defers to `cache`: with `mode` other than `none`, the budget is the backend's own `n_max_concurrent_scans`; with `mode: none` the readahead does not run. |
| `0` | The readahead does not run, whatever the cache mode. |
| `n > 0` | An eligible advertised backend runs readahead with a budget of `n`, whatever the cache mode. |

`readahead_strategy` works the same way, deferring to the backend rather than to the cache:

| Value | Effect |
|-------|--------|
| unset (default) | The serving backend's preference — `eager` for REST, `opportunistic` for uring. |
| `eager` | Every wake-up fills every free slot in the budget. An object-store round trip is dead time no matter what else is running, so only queue depth hides it. |
| `opportunistic` | One prefetch each time the executor deploys a task that is *not* a scan, i.e. only while the device is not already busy with the executor's own reads. A local device read competes with those, so issuing one mid-scan reorders the queue rather than adding throughput. |

When the readahead ends up `opportunistic` (either way), an *unset* `max_readahead_scans` schedules
against the pipeline pool's width rather than the backend's depth — one prefetch per non-scan
deployment is only useful while a pipeline thread could still pick up another scan. An explicit
`max_readahead_scans` wins over that substitution.

A query's local policy is selected from the serving backend publishing the widest budget.
A runtime budget additionally accounts for every backend touched by each split across all
queries. Demand reads borrow immediately; their debt pauses speculative prefetch. Events are
routed by query ID. A failed cache allocation gets one eviction retry before speculation yields.

Six optional nested sub-configs tune the individual backends, the cache, and the memory prefetcher:

### `scan_manager.uring` — io_uring backend (`io/uring/config.hpp`)

There are no static chunk-size or `readv`-fusion knobs. The worker chooses physical
operation sizes dynamically from backlog pressure, available slots, and the pinned block size.

```yaml
sirius:
  executor:
    scan_manager:
      uring:
        slices_per_pass: 4
        range_batch_slices: 8
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_max_concurrent_scans` | int | 0 | Readahead budget the local backend publishes, which an unset `max_readahead_scans` defers to (under `opportunistic`, the default for local reads, a positive value switches the readahead on and the pipeline width sets the budget). `0` (Sirius's default, and cuCascade's) keeps the readahead off for local reads, since a local NVMe read competes with the executor's own reads rather than hiding a round trip. An explicit `max_readahead_scans` overrides it. |
| `slices_per_pass` | int (**0..64**) | 4 | Most slices of the active request a reactor turns into physical reads per loop pass before it waits for a completion. A reactor works through one request at a time, in queue order; the default `4` keeps a single many-slice request (a whole-split prefetch, a wide demand read) deep — up to 4 slices per pass, bounded by the reactor's free staging slots — without letting one pass claim every free slot. `0` = no cap: keep going while every planned read finds a free staging slot (each reactor has at most 64). `1` expands one slice per pass, which holds such a request to a depth of 1-2 per reactor. |
| `range_batch_slices` | int (**>= 0**) | 8 | Most slices one queue entry holds when a host-only multi-range read (a whole-split prefetch, the DuckDB-native decoder's column-chunk set) is dispatched to the uring reactors: a local read does not prefer bulk I/O, so the read is queued as several entries of at most this many slices instead of one per reactor. Reads with a device destination are never split. `0` = no split. |

Sirius rejects a negative `range_batch_slices` and a `slices_per_pass` outside 0..64 when it loads the config, naming the
key. cuCascade does not re-check `slices_per_pass`: a value above 64 would simply behave as no cap, since a reactor has at most 64 staging slots.

`uring.prefetch_reactors` is deprecated: the reactor priority tiers replaced prefetch-reactor partitioning, so the key is
accepted and ignored with a warning, and will be removed.

### `scan_manager.rest` — REST / S3 backend (`io/rest/config.hpp`)

The native REST backend speaks HTTP/1.1 over io_uring; it does not use libcurl.
HTTPS endpoints use kernel TLS (kTLS) when the kernel accepts the session keys
and cuCascade's user-space TLS record layer otherwise. It requires Linux 6.1+ with
usable io_uring, for plain HTTP endpoints too: a container seccomp policy or
`kernel.io_uring_disabled` makes S3 unavailable, and there is no alternate
transport. kTLS additionally needs the `tls` kernel module; without it,
`tls.ktls: prefer` (the default) falls back to user space.

TLS trust is supplied by the resolved `object_store_config`: its
`ca_bundle_path` and `tls_verify` are applied to transport as well as signing.
Set a private CA with `ca_bundle_path` through `sirius_config::set_object_store_config()`.
A selected SQL secret keeps that path, but a secret cannot carry a CA itself.
Alternatively, set `NIX_SSL_CERT_FILE` / `SSL_CERT_FILE` / `SSL_CERT_DIR` in the process
environment; these apply when `ca_bundle_path` is empty. A usable `NIX_SSL_CERT_FILE`
takes precedence over `SSL_CERT_FILE`. With neither an explicit CA nor usable
process trust settings, trust falls back to the OpenSSL default store and system bundles.
Certificate verification is controlled by `tls_verify`, or by `VERIFY_SSL` on a selected
`SIRIUS_S3` secret. `scan_manager.object_store` and
`rest.ca_bundle_path` / `rest.tls_verify` are not YAML-loadable and are rejected.
The `rest.tls` block below controls protocol/offload policy only, not trust.

Sirius sets no transport defaults of its own: every default below is cuCascade's
`rest::config` default (the one exception is `n_max_concurrent_scans`). Each key is
range-checked when the config loads, and the loaded block is then checked with
cuCascade's own `rest::config::validate()`, so cross-field rules (`queue_depth`
against `max_connections`, `max_segment_bytes` against `min_segment_bytes`, the TLS
session-cache rule) fail at load time, naming the YAML key. Sirius hands the loaded
`rest::config` to cuCascade inside a `rest_reactor::reactor_context`, which is
non-copyable and shared by the pool's reactors through a `shared_ptr`.

**Requests and retries**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_max_concurrent_scans` | int (**>= 0**) | max(8, 2 × `pipeline.num_threads`) | Readahead budget the REST backend publishes, which an unset `max_readahead_scans` defers to. Unset derives it from the pipeline pool, since several splits per pipeline thread give an object-store prefetch time to land; an explicit value wins. |
| `request_timeout_s` | int (seconds, **>= 0**) | 30 | Whole-exchange timeout of one control-plane request (HEAD / LIST / footer probe / warm-up); retries can extend a call's total time. Also the base of the presigned-URL TTL (this value plus 60 s). `0` = no limit on control requests, and presigned URLs then live 300 s. Data GETs use the stall detector instead. |
| `stall_speed_limit_bytes` | int (bytes/s, **>= 0**) | 65536 | Stall detector for data GETs: a transfer that received fewer than `stall_speed_limit_bytes` × `stall_time_s` bytes during the last `stall_time_s` seconds fails and is retried. `0` disables the detector. |
| `stall_time_s` | int (seconds, **>= 0**) | 30 | Window of the stall detector; `0` disables it. |
| `retry_backoff_base_ms` | int (ms, **>= 0**) | 50 | Base of the exponential backoff between retries. |
| `retry_jitter_ms` | int (ms, **>= 0**) | 50 | Random jitter added to each retry backoff. |
| `max_retry_attempts` | int (**>= 1**) | 10 | Attempt budget (the first attempt included) for transient transport failures and HTTP 408/429/500/502/503/504; `1` = no retry. |
| `max_auth_retry_attempts` | int (**>= 1**) | 3 | Async data attempt budget for HTTP 403 (an expired presigned URL); each attempt re-authorizes. Kept low so a genuine AccessDenied fails fast. Control requests treat 403 as terminal. |
| `honor_retry_after` | bool | true | Wait a numeric `Retry-After` instead of the backoff. |
| `footer_probe_bytes` | bytes (**<= 1Gi**) | 512Ki | Suffix-range window for the parquet footer probe. Must cover the footer, so err large. |
| `list_max_matches` | int (**> 0**) | 100000 | Cap on files a glob/listing may accumulate (throws "narrow the glob prefix", never truncates). |
| `list_max_scanned` | int (**> 0**) | 1000000 | Cap on objects a LIST sweep may scan across pages (throws, never truncates). |

**Physical GET sizing**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_segment_bytes` | bytes (**> 0**) | 4Mi | Lower bound of the adaptive physical GET size. |
| `max_segment_bytes` | bytes (**>= `min_segment_bytes`**) | 16Mi | Upper bound of the adaptive physical GET size. |
| `merge_max_gap` | bytes | 512Ki | Logical cache/read-planner coalescing hint: the largest gap between two ranges still planned as one. `0` merges only adjacent ranges. It does not set physical GET segments. |

Each reactor sizes a physical GET as its queued backlog divided by its free
connections, clamped to [`min_segment_bytes`, `max_segment_bytes`]. Fragmented cache
fills are grouped only at whole cache-chunk boundaries so a chunk is never published
before all of its bytes arrive. Device reads stage through pinned CuCascade blocks that
stay owned through retries, the asynchronous H2D copy and its CUDA completion event,
so staging is proportional to the device work in flight.

**Connection pool** (libcurl parity)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_connections` | int (**1..4096**) | 64 | Keep-alive connections per reactor (one in-flight request each), not a process-wide cap. Past the point where the link is full, more connections only split the bandwidth into thinner streams; add reactors (`rest_n_reactors`) instead. |
| `conn_max_age_s` | int (seconds, **>= 0**) | 20 | Idle cutoff of a pooled connection (curl's `CURLOPT_MAXAGE_CONN`): a connection idle for longer is closed and replaced when a request needs its slot, never in a sweep. Also the per-bucket rate limit of the query-start warm-up (60 s when `0`). `0` = no idle cutoff. |
| `upkeep_http1` | bool | false | Send idle upkeep: while a reactor is idle, one zero-key `ListObjectsV2` per established connection every `upkeep_interval_ms`, staggered across the interval, so the endpoint does not idle-close the pool. Off by default, as libcurl's upkeep is a no-op on HTTP/1.1. |
| `upkeep_interval_ms` | int (ms, **>= 0**) | 15000 | Upkeep period; only acts with `upkeep_http1`. `0` disables upkeep. |
| `connect_timeout_ms` | int (ms, **> 0**) | 3000 | Bound of one connection's TCP connect plus TLS handshake (curl's `CONNECTTIMEOUT`). |
| `dns_ttl_s` | int (seconds, **> 0**) | 60 | DNS answer lifetime. Answers are refreshed ahead of expiry on a resolver thread shared by the pool, so a reactor never blocks on a lookup. IP literals bypass the resolver. |
| `dns_accumulate` | bool | false | Union fresh answers with the previous ones (aged out after 5 minutes) instead of replacing them, so connections spread over more front-end addresses over time. |
| `dns_max_addresses` | int (**1..4096**) | 64 | Addresses kept per host. |
| `keepalive_idle_s` | int (seconds, **> 0**) | 60 | TCP keepalive (always on): idle time before the first probe. |
| `keepalive_interval_s` | int (seconds, **> 0**) | 60 | TCP keepalive: time between probes. |
| `keepalive_count` | int (**>= 0**) | 0 | TCP keepalive: probes before the connection is dropped; `0` = kernel default. |
| `min_mtu` | bytes (**0..65607**) | 0 | Reject connections whose effective MTU is below this many bytes. `0` = off. See below. |
| `mtu_reject_max_attempts` | int (**>= 1**) | 8 | Consecutive `min_mtu` rejections of one request after which the connection is accepted anyway. |
| `close_with_rst` | bool | true | Close every socket with an RST (`SO_LINGER {on, 0}`), so none lingers in `TIME_WAIT` / `FIN_WAIT`. |

At query start Sirius warms the REST pool of each bucket the query reads: every reactor
sends one zero-key `ListObjectsV2` (`list-type=2&max-keys=0`) per idle connection slot,
each on a fresh connection, so TCP, TLS and HTTP are validated and the pool is hot before
the reads arrive. Warm-up runs at most once per `conn_max_age_s` per bucket; its
failures are logged at debug level and never fail a read. Nothing else is sent while
idle unless `upkeep_http1` is on, and nothing reconnects en masse (not after warm-up,
not at a query's end, not on a DNS refresh).

`min_mtu` exists for jumbo-frame networks, where some connections of a pool may
negotiate small TCP segments, which costs per-byte decryption time. Right after the TCP
connect, the effective MTU is `TCP_MAXSEG` + 40 (IPv4) or 60 (IPv6) header bytes, + 12
with TCP timestamps; a connection below `min_mtu` is reset and the next DNS address is
tried, up to `mtu_reject_max_attempts` times per request, after which the connection is
accepted. The REST counters (`rest_ioctx::stats()`) report `mtu_rejects`,
`mtu_fallbacks` and the smallest and largest MSS observed.

**Transport and placement**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `receive_buffer_bytes` | bytes (**16Ki..4Mi**) | 256Ki | Per-connection header / scratch buffer and the cap of one direct body receive. |
| `queue_depth` | int (**>= 2 × `max_connections`, <= 32768**) | 1024 | Submission-queue entries of each reactor's io_uring ring (one primary and one cancel entry per connection). Raise it with `max_connections` above 512. |
| `wait_mode` | enum: `epoll`, `uring` | `epoll` | How an idle reactor blocks: `epoll_pwait2` over its wake-up eventfd, retry timer and ring fd, or one `io_uring_submit_and_wait_timeout` per pass. |
| `iowq_max_unbound` | int (**>= 0**) | 0 | io-wq unbound-worker cap of each reactor's ring (used by kTLS receives placed on io-wq); `0` = kernel default. |
| `fan_out` | int (**>= 1**) | 2 | Number of least-backlogged reactors one read is split across. With a positive `rest_n_reactors`, a value above it is rejected unless it is the default 2, which is clamped to a one-reactor pool; with `rest_n_reactors: 0` it is clamped to the NIC-sized pool. |
| `interface` | string | "" | Egress interface whose queue IRQs size the pool when `rest_n_reactors: 0` and place (pin) its reactors; empty = the default route's interface. |
| `pin_to_irq_cpus` | bool | true | Pin reactors to the NIC's IRQ CPUs, and raise their priority, when the IRQs are steered to a strict subset of the NIC node's cores. With a positive `rest_n_reactors` the configured count is pinned round-robin over those CPUs. |

The remaining `rest::config` fields — `max_recv_iovecs`, `recv_waitall`, `sqpoll`,
`single_issuer`, `defer_taskrun`, `register_ring_fd`, `iowq_max_bound`,
`copy_poll_interval`, `hold_slot_during_copy`, `so_rcvbuf`, `so_sndbuf`, `busy_poll_us`,
`tcp_user_timeout_ms`, `footer_resolve_max_inflight` and `footer_resolve_stash_budget` —
are not YAML-loadable and keep their cuCascade struct defaults.

#### `scan_manager.rest.tls` — TLS protocol and offload policy (`io/tls/config.hpp`)

All enum values are lowercase. Certificate/hostname verification stays enabled by
default; configure trust through `object_store_config`, not this block.
`verify`, `ca_file`, `ca_dir`, and `ca_blob` under `rest.tls` are rejected.
The error for `verify` points to `VERIFY_SSL` / `tls_verify`. The errors for `ca_*`
point to `ca_bundle_path` or the process trust environment.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ktls` | enum: `prefer`, `require`, `disable` | `prefer` | Try kernel TLS then fall back to user space (counted as a kTLS fallback); `require` fails the connection if kTLS cannot be installed; `disable` never requests it. |
| `user_space_rx` | enum: `record_layer`, `library` | `record_layer` | Built-in AES-GCM receive record layer or the TLS library's receive path. Unsupported suites use the library path. |
| `min_version` | enum: `tls1.2`, `tls1.3` | `tls1.2` | Minimum TLS version. |
| `max_version` | enum: `tls1.2`, `tls1.3`, `auto` | `auto` | Automatic maximum considers kTLS and kernel re-key support, plus `allow_tls13_without_rekey`; with kTLS disabled, automatic permits TLS 1.3. Must not be below the minimum. |
| `recv_placement` | enum: `automatic`, `inline`, `iowq` | `automatic` | Where kTLS body decryption runs: io-wq on EC2 (by DMI vendor) and inline elsewhere by default. `iowq` is bounded by `rest.iowq_max_unbound`. |
| `expect_no_pad` | bool | true | Request the TLS 1.3 kTLS no-padding receive optimization. |
| `session_cache` | bool | false | Enable the client TLS session cache, shared by the reactors of a pool; useful for stores that resume sessions. It is effective for TLS 1.2 sessions and for TLS 1.3 sessions of `user_space_rx: library` connections only: kTLS and the built-in record layer consume TLS 1.3 session tickets without handing them to the TLS library. |
| `session_cache_per_endpoint` | int (**>= 0**; **> 0** with `session_cache: true`) | 256 | Maximum cached TLS sessions per endpoint when the cache is enabled. |
| `max_connection_bytes` | bytes | 64Gi | Retire a connection at check-in after this many bytes, limiting KeyUpdate exposure. `0` disables the bound. |
| `ciphersuites` | string | TLS 1.3 AES-128/256-GCM | TLS 1.3 suites in OpenSSL syntax (`TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384` by default). |
| `cipher_list` | string | TLS 1.2 ECDHE + AES-128/256-GCM | TLS 1.2 ciphers in OpenSSL syntax; the default includes ECDSA and RSA authentication. |
| `groups` | string | "" | Key-exchange groups in library syntax; empty keeps the library default. |
| `sni_override` | string | "" | Send this SNI name and verify the certificate against it instead of the endpoint host. Does not disable verification. |
| `allow_tls13_without_rekey` | bool | true | Permit TLS 1.3 on kernels without kTLS TLS 1.3 re-key support; a KeyUpdate then requires reconnection. |

The YAML loader intentionally leaves the TLS fields `alpn_http11`, `client_cert_file`,
`client_key_file` and `keylog_env` at their cuCascade struct defaults; they are not
YAML-loadable.

For example, tune transport independently of the credentials supplied by a secret
or the programmatic object-store config — here a NIC-sized, IRQ-pinned pool on a
jumbo-frame network:

```yaml
sirius:
  executor:
    scan_manager:
      backend: native
      rest_n_reactors: 0        # one pinned reactor per NIC IRQ CPU when IRQs are steered
      rest:
        interface: eth0
        max_connections: 64
        fan_out: 2
        min_segment_bytes: 4Mi
        max_segment_bytes: 16Mi
        min_mtu: 8000
        connect_timeout_ms: 3000
        tls:
          ktls: prefer
          user_space_rx: record_layer
          recv_placement: automatic
```

### `scan_manager.cufile` — local cuFile reads (`io/cufile/config.hpp`)

Select cuFile with the existing `backend` key. The default remains `native`.
This backend reads local regular files only; `s3://` reads, listing and globs still use
native REST. The existing multi-GPU guard forces `backend: native` with a warning
when more than one GPU is configured; this does not add multi-GPU cuFile placement.

```yaml
sirius:
  topology:
    num_gpus: 1
  executor:
    scan_manager:
      backend: cufile
      cufile_n_reactors: 4
      cache: { mode: none }
      cufile:
        max_inflight: 32
        max_op_bytes: 64MiB
        gds_threshold: 256KiB
        use_odirect: true
        compat_mode: auto
        poll_interval_us: 200
        low_service_interval: 4
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_inflight` | int (**> 0**) | 32 | Per-reactor bound on asynchronous items awaiting CUDA events (stream gates and fallback H2D copies), not concurrent `cuFileRead` calls. |
| `max_op_bytes` | bytes (**>= 4KiB**) | 64MiB | Largest synchronous `cuFileRead` issued by a worker; larger device reads are chunked. |
| `gds_threshold` | bytes | 256KiB | Device reads below this size use POSIX reads and pinned H2D staging rather than cuFile. `0` is legal. |
| `use_odirect` | bool | true | Open the direct-I/O descriptor needed for cuFile. `false` skips cuFile registration and uses buffered POSIX reads; it conflicts with `compat_mode: off`. Independent of `cache.mode`. |
| `compat_mode` | enum: `auto`, `on`, `off` | `auto` | Per-file policy: `auto` tries cuFile and permits POSIX fallback, `on` forces POSIX, `off` requires a usable cuFile handle and fails the open rather than falling back if it cannot be registered. Values are lowercase. |
| `poll_interval_us` | int (**>= 0**, µs) | 200 | Idle polling interval while fallback CUDA copies await completion; not a delay before dispatching ready work. |
| `low_service_interval` | int (**>= 0**) | 4 | High-priority requests started while low work is waiting before granting the low tier a turn. `0` restores strict priority. |

cuFile cannot fill fragmented pinned-cache destinations, so it never arms `fs_cache`
or scan readahead, **even with `cache.mode: cucs`**. Neither `max_readahead_scans` nor
`readahead_strategy` enables it. These are read-only settings; no write or
per-transfer buffer-registration controls are exposed.
`cufile.n_max_concurrent_scans` is intentionally not a YAML setting: cuFile cannot
publish a usable cache/readahead budget. Tune read concurrency with
`cufile_n_reactors` and `cufile.max_inflight` instead.

**Capability and fallback.** cuFile availability depends on the runtime library, GPU,
filesystem/mount, storage stack, alignment and cuFile configuration. A per-path
capability report combines that evidence with direct-open/handle-registration results;
`direct_candidate` is advisory, **not proof of direct DMA**. Unknown evidence is not
proof of unsupported storage. Successful registration (including `compat_mode: off`)
does not prove that cuFile itself avoided compatibility mode. With `auto`, reads can
remain correct through host/POSIX fallback on a compatibility-only host, including a
host without `nvidia-fs`; conversely, absence of `nvidia-fs` alone does not rule out
supported P2PDMA. Small reads and unaligned tails can use host staging. An explicit
cuFile context initialization failure is reported, not silently rerouted to another backend.

**cuFile logging.** `libcufile` can create `cufile.log` in the working directory.
Configure its `logging.level` and `logging.dir` in a cuFile JSON file, selected with
`CUFILE_ENV_PATH_JSON` **before starting the process / first cuFile initialization**
(otherwise cuFile uses `/etc/cufile.json` and its defaults). For example, save this as
`/absolute/path/to/cufile.json`:

```json
{"logging": {"level": "ERROR", "dir": "/tmp"}}
```

```bash
export CUFILE_ENV_PATH_JSON=/absolute/path/to/cufile.json
```

Use a more verbose level such as `TRACE` only when diagnosing transfer mode; verbose
logging can affect performance. This is cuFile configuration, not a Sirius YAML setting.

### `scan_manager.cache` — read-path caching (`io/cache/config.hpp`)

Everything about read-path caching lives in this one block — rather than a mode on the
scan manager and the cache's tunables in a sibling block.

```yaml
sirius:
  executor:
    scan_manager:
      cache:
        mode: cucs
        eviction: lru
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | enum: `none`, `os`, `cucs` | `none` | Which cache the read path goes through. `sirius` is a deprecated alias of `cucs`, still accepted. Values are lowercase. |
| `eviction` | enum: `idle`, `lru` | `lru` | What retires an idle chunk from the prefetching cache. Only meaningful under `mode: cucs`. Values are lowercase. |
| `eviction_threshold_fraction` | double [0,1] | 0.8 | Start evicting when pinned host-memory allocations (by any user) reach this fraction of capacity. |
| `min_prefetching_budget_fraction` | double [0,1] | 0.05 | Floor of the pool reserved for prefetching. |

For the native local backend, `mode: none` bypasses every cache (`O_DIRECT`, no
prefetching cache), `mode: os` reads through the kernel page cache, and `mode: cucs`
reads `O_DIRECT` into cuCascade's pinned prefetching cache (`fs_cache`). REST can also
use `fs_cache` under `cucs`. cuFile never attaches `fs_cache` or runs scan
readahead, regardless of this mode; cuFile's direct-open policy is set by
`cufile.use_odirect`, not derived from the cache mode.

`eviction: lru` keeps idle chunks for reuse and sweeps them, in request-creation order, once
pinned host allocations pass `eviction_threshold_fraction`; `eviction: idle` drops each chunk as soon as
it goes idle, making the cache a prefetch staging area sized for the reads in flight rather
than for reuse.

Those two knobs derive the settings below, which are therefore **not** individually settable from YAML:

| Derived setting | Derived from |
|-----------------|--------------|
| `uring.use_odirect` | `mode` — true for everything but `os` |
| whether the prefetching cache is armed | `mode` — only under `cucs`, and only for a backend that supports `fs_cache` (not cuFile) |
| `dispose_on_idle` | `eviction` — true under `idle` |
| the readahead's default budget | `mode` — the readahead does not run under `none`; see [`max_readahead_scans`](#scan-manager--io-configuration) |

### `scan_manager.object_store` — programmatic S3 config (`io/object_store_config.hpp`)

This is a C++ configuration reference, **not a YAML block**.
`scan_manager.object_store` in YAML is rejected: supply
`cucascade::io::object_store_config` through
`sirius_config::set_object_store_config()`, or use
`CREATE SECRET (TYPE SIRIUS_S3, ...)`. A matching SQL secret takes precedence
over this fallback; an invalid matching secret is an error, not a reason to fall
back. See [S3 authorization](scan.md#s3--object-store-backend).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint` | string | "" | S3 endpoint URL, including `http://` or `https://`. Input paths remain `s3://bucket/key`; raw HTTP(S) input URLs are unsupported. |
| `region` | string | "" | AWS signing region. |
| `access_key` / `secret_key` | string | "" | Static credentials. |
| `session_token` | string | "" | STS session token for temporary credentials. |
| `signing_mode` | enum: `presigned`, `header` | `presigned` | SigV4 form: auth in the URL query or in signed headers. |
| `s3_transport` | enum: `auto`, `http`, `https`, `rdma` | `auto` | Transport selection; `https` aliases `http`. HTTP versus HTTPS comes from the endpoint scheme. |
| `ca_bundle_path` | string | "" | PEM CA bundle for the REST endpoint; empty uses the process trust environment, then the TLS provider's default trust. |
| `tls_verify` | bool | true | Certificate and hostname verification. Disable only for development endpoints. Applied with the CA bundle on top of the REST TLS policy. |

### `sirius.executor.scan_manager.preparation`

These settings are loaded at initialization. Metadata jobs default to a limit of 4096; DV jobs have no configured count cap by default. `0` disables either job limit. `coordinator_workers: 0` selects dedicated coordinator threads; other count limits must be positive integers. Control limits default to the scan worker count, N (`scan_manager.num_threads`); `lookahead_batches` defaults to 2. Timing values accept suffixes such as `20ms` or `1s`; bare integers are milliseconds.

| Key | Default | Description |
|---|---|---|
| `max_inflight_metadata_jobs` | 4096 | Maximum submitted metadata preparation jobs across the query, including queued work. `0` means unlimited. Completion returns the job slot. |
| `max_inflight_dv_jobs` | unlimited | Maximum DV preparation jobs holding temporary-memory permits, including queued work. Omitted or `0` imposes no configured count cap. Admission may grant fewer permits when HOST memory is insufficient. |
| `lookahead_batches` | 2 | Per-scan preparation target in multiples of `scan_task_batch_size`, using decoded read-column bytes, including predicate-only columns. Results, retained coalescer input, and published batches stay charged until downstream takes the batch. Already submitted jobs may overshoot the target. |
| `native_reader_workers` | 2 | Native-reader worker threads per scan manager, started lazily at the first native reservation. Positive and independent of metadata workers and GPU count. |
| `max_native_readers` | 64 | Reader slots shared by all attempts within that manager. Must be at least `native_reader_workers`; an attempt reserves its full group before any reader starts. |
| `coordinator_workers` | 1 | Shared coordinator workers per scan manager, started lazily. `0` uses one dedicated coordinator per reader attempt. Metadata-only attempts still run coordination on the query owner. |
| `max_control_work` | N | Maximum coordinator actions per iteration. |
| `drain_quantum` | N | Maximum coalescing work per result advancement. |
| `partial_batch` | enabled | Partial-publication policy. An omitted or empty block uses the defaults below. |
| `partial_batch.min_fill_percent` | 50 | Minimum retained size as a percentage of the normal batch target for early publication. Integer from 1 to 100. Uses the coalescer's batch-size accounting. |
| `partial_batch.min_residence` | 50 ms | Minimum age before early publication. Also requires the size threshold and no batch from this scan waiting for downstream consumption. Must be positive. |
| `partial_batch.max_residence` | 2 s | Maximum wait for more metadata, even below the size threshold or with a batch waiting downstream. Must be at least `min_residence`. Output backpressure and scheduling can delay publication. |
| `interrupt_check_interval` | 10 ms | Maximum wait between interruption checks while preparation is idle. Must be positive. Draining an active read can take longer. |

Both residence times start when the first input enters an empty accumulator; later arrivals do not reset them. Normal full-batch, compatibility-boundary, final-flush, and window-progress publication still apply. Downstream consumption means taking the batch from the connector, not completing GPU execution. To express a fixed time limit, set both residence values to the same duration.

A file with a DV currently reads its footer and prepares its DV in the same job, so it uses a metadata job slot and a DV permit. The metadata limit does not size the DV reservation: admission reserves retained results plus K temporary workspaces, where K is bounded by the DV unit count, `scan_manager.num_threads`, and the DV limit when nonzero. Insufficient HOST memory may reduce K further. Neither setting changes the worker thread count. The previous `max_inflight_jobs` key is no longer accepted; use `max_inflight_metadata_jobs`.

```yaml
sirius:
  executor:
    scan_manager:
      preparation:
        max_inflight_metadata_jobs: 4096
        max_inflight_dv_jobs: 0
        coordinator_workers: 1
        native_reader_workers: 2
        max_native_readers: 64
        lookahead_batches: 2
        max_control_work: 4
        drain_quantum: 4
        partial_batch:
          min_fill_percent: 50
          min_residence: 50ms
          max_residence: 2s
        interrupt_check_interval: 5ms
```

### `scan_manager.memory_prefetcher` — background host→GPU upload of pinned-cache scan splits (`scan_manager/config.hpp`)

Overlaps the host→GPU upload of queued pinned-cache scan splits with compute:
worker threads walk the pending splits in scan execution order and convert
resident batches to GPU tier ahead of task execution, gated on GPU memory
headroom (see `scan_manager/memory_prefetcher.hpp`). Disabled by default. Background memory
prefetch runs only when the query has exactly one eligible GPU after admission filtering.
Workers use that device's reservations and exclusive stream leases, releasing leases before
parking. With multiple eligible GPUs, scan tasks perform their own uploads on their assigned
devices; independent background placement could conflict with a task's already selected device.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | bool | false | Master switch for the prefetcher. |
| `num_threads` | int (**> 0**) | 2 | Prefetch worker threads; each acquires an exclusive stream lease for a sweep and returns it before parking. |
| `min_free_fraction` | double [0,1] | 0.4 | Keep at least this fraction of the GPU space free after each prefetch; conversions (and their reservations) are only attempted above this floor. |
| `poll_interval_ms` | int (**> 0**) | 2 | Worker sweep interval while waiting for headroom / new splits. |
| `drain_quiet_ms` | int (ms) | 100 | A connector counts as actively draining until this long passes since its last pop. One worker may prefetch it tail-first, leaving four batches at the head for scan tasks; other workers skip it. Must exceed the scan's inter-pop interval. |

### Programmatic S3 credentials

S3 credentials are not YAML-loadable. Use `CREATE SECRET (TYPE SIRIUS_S3, ...)` for SQL-managed credentials, or set an in-memory `io::object_store_config` through `sirius_config::set_object_store_config()` before initializing the scan manager. A matching secret takes precedence over this programmatic fallback. YAML containing `sirius.executor.scan_manager.object_store` is rejected.

## Operator Parameters

**File:** `src/include/sirius_config.hpp` — `operator_params` struct

The five batch/partition sizes (`scan_task_batch_size`, `hash_partition_bytes`,
`concat_batch_bytes`, `cross_join_task_bytes`, `sort_sample_bytes`) share one built-in default. Without an
explicit GPU capacity in YAML, it is computed at startup as
`clamp(smallest visible physical GPU memory / 40, 512 MiB, 5 GiB)` (800 MiB when no
GPU is visible). When the active YAML memory path explicitly caps GPU capacity, the
default is narrowed to
`min(physical-memory default, max(1 byte, smallest resolved GPU capacity / 40))`.
`max_build_hash_table_bytes` defaults to **8× the startup-resolved
`hash_partition_bytes`**, including an explicit YAML partition size.
`max_build_probe_resident_bytes` defaults independently to **8× the batch default**.
An explicit value for either budget overrides its default. `SET hash_partition_bytes`
does not re-derive either budget; `RESET` restores each registered startup default.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scan_task_batch_size` | Shared physical/effective GPU batch default described above | Target batch size for DuckDB scan tasks; must be greater than zero |
| `enable_compressed_materialization` | true | Store eligible integer and fixed-point DECIMAL values in value-preserving narrower carriers when exact pin-time bounds permit it; restore native carriers at type-sensitive boundaries. |
| `max_sort_partition_bytes` | 0 (auto) | Max bytes per sort partition. Auto = 33% of GPU memory. |
| `hash_partition_bytes` | Shared physical/effective GPU batch default described above | Target partition size for hash joins and group-bys; must be greater than zero |
| `concat_batch_bytes` | Shared physical/effective GPU batch default described above | Target output batch size for CONCAT operator |
| `cross_join_task_bytes` | Shared physical/effective GPU batch default described above | Target output size of each cross product task; a larger pair of input batches is split into tasks over row ranges of the left batch |
| `sort_sample_bytes` | Shared physical/effective GPU batch default described above | Bytes sampled before computing sort partition boundaries |
| `max_build_hash_table_bytes` | 8× startup-resolved `hash_partition_bytes` | Max per-GPU estimated BUILD_PROBE hash-table bytes (measured build rows x 16 B; unavailable row counts decline admission). 0 disables BUILD_PROBE. |
| `max_build_probe_resident_bytes` | 8× batch default | Max per-GPU decoded folded build-side payload a BUILD_PROBE join may keep GPU-resident (un-spillable) for the whole probe stream. 0 disables BUILD_PROBE. |
| `max_broadcast_join_size` | 256 MiB | Max build-side size eligible for a broadcast join. A build below this size is replicated to every GPU (instead of hash-partitioned) when it is tiny, or when the DuckDB-estimated probe-to-build row ratio is at least `num_gpus * 1.25`. |
| `max_sort_partition_memory_fraction` | 0.33 | Fraction of GPU memory per sort partition when `max_sort_partition_bytes` is 0 |
| `mark_join_build_switch_ratio` | 8.0 | For STANDARD MARK joins, build on the smaller (left) side when `right_rows >= ratio * left_rows` (0 disables) |
| `enable_dynamic_filter` | true | Enable runtime filters for eligible hash joins. Plan-time wiring admits keys by join type (not join mode); at delivery, any join whose build side arrives as one whole batch publishes — a single-partition or broadcast `BUILD_PROBE` build, and a single-partition `STANDARD`/`MIXED_JOIN` build on the same terms. Targets may be probe scans or join-edge endpoints. An eligible build selects a raw exact IN-list for 1–12 supported build rows, otherwise a hash IN-list within the L2 budget or a Bloom. |
| `enable_dynamic_zone_map_filter` | false | Publish build-key min/max filters in addition to membership filters. Parquet scans use them for row-group pruning; duckdb-native scans apply them post-decode. Requires `enable_dynamic_filter`; intended for clustered-keyset workloads. |
| `dynamic_filter_domain_coverage_threshold` | 0.9 | Positive finite threshold. Before constructing either a membership filter or zone map, skip the key when the complete build covers at least this fraction of the key's unfiltered base-table row bound. Applies only to build keys proven unique in their base relation, with evidence from DuckDB-native scans. Values above 1.0 disable the gate; exactly 1.0 fires only at full coverage. |
| `dynamic_filter_inlist_max_l2_fraction` | 0.125 | Finite threshold in [0, 1]: maximum estimated cuco-set size for the exact hash IN-list, as a fraction of the smallest probe-GPU L2. Larger sets use Bloom when supported. For keys not handled by the raw IN-list, 0 selects Bloom when supported, while 1.0 reproduces the legacy L2-fit rule only when L2 size is known. If L2 size is unknown, the hash IN-list is ineligible and selection falls back to Bloom or no membership filter. The 0.125 default comes from a GB300 residency sweep: hash-set probe cost is flat below ~0.28 of L2 and degrades beyond it, while Bloom was at least 2.2x faster at every swept set size. |
| `enable_runtime_distinct_build_probe` | true | For `BUILD_PROBE` INNER/LEFT equality joins whose build-key uniqueness the planner could not prove, test distinctness at runtime (one `cudf::distinct_count` pass over the cached build, dimension-scale builds only) and take the single-pass `cudf::distinct_hash_join` instead of the general two-pass join when the keys are distinct. |
| `enable_dense_count_join` | true | Fuse `COUNT(col \| *) GROUP BY <preserved-side join key>` over a LEFT/RIGHT integer equi-join into the `DENSE_COUNT_JOIN` operator (TPC-H q13 shape): a direct-address count histogram over the preserved key domain replaces the join build, output materialization, and re-aggregation. |
| `dense_count_join_max_bytes` | 2 GiB | Cap on `DENSE_COUNT_JOIN`'s combined direct-address histogram footprint; a key domain too wide for the budget takes the operator's exact sparse (eager-aggregation) strategy. Must be greater than zero. |
| `dynamic_filter_keep_threshold` | 0.9 | Finite threshold in [0, 1] for disabling post-decode filtering once a measured split keeps more than this fraction of its rows; 1.0 keeps filtering always on. |
| `enable_pinned_zone_map_pruning` | true | Capture per-chunk min/max statistics while pinning and use them to skip cached chunks that cannot match a scan filter. |
| `admission_bytes_per_gpu` | 0 (off) | Target projected scan-output bytes per GPU. At admission the engine estimates a query's total scan output and takes the smallest GPU subset that keeps each GPU under this figure, bounded by `topology.gpus_per_query`. `0` disables the estimate, leaving the allocation to `topology.gpus_per_query` alone. |
| `avg_variable_column_bytes` | 32 | Per-row width assumed for variable-width columns (VARCHAR, LIST, STRUCT, ARRAY) when estimating scan output. Fixed-width columns use their real carrier width. Only consulted when `admission_bytes_per_gpu` is non-zero. |
| `enable_runtime_size_estimation` | false | Size grouped-aggregation partitions from projected input, allowing a partial ingress barrier. |
| `enable_decimal_sum_stats_bound` | true | Decide at plan time, from base-table statistics, whether a DECIMAL32/DECIMAL64 SUM or AVG input, or the BIGINT input of a SUM or AVG that DuckDB accumulates in HUGEINT, must be widened before aggregation, or whether a DECIMAL128 SUM input can be summed as DECIMAL64. An input whose magnitude bound the planner can prove skips the per-batch min/max scan and its stream synchronization; unproven inputs are still measured per batch. |

**Note:** `admission_bytes_per_gpu` is a parallelism dial, not a memory budget. Peak GPU residency is bounded by partition sizing (`hash_partition_bytes` and the batch settings), not by the admitted GPU count — a query on fewer GPUs processes more partitions sequentially at roughly unchanged peak memory, trading wall-clock for freed devices. Tune it against how much of the fleet a query should occupy, not against VRAM.

`max_build_probe_resident_bytes` is connection-local: `SET` and `RESET` affect only the
issuing connection, and `SET GLOBAL` is rejected. Planning captures its value from the
immutable query options; existing plans retain their admission budget. `RESET` restores
the registered default (including a YAML override).

**Note:** `max_build_probe_resident_bytes` can be larger than `concat_batch_bytes`. When the partition operator selects BUILD_PROBE, or folds a single-partition build whole for dynamic-filter publication (both admissions bounded by `max_build_probe_resident_bytes`), it configures CONCAT to concatenate all build batches into one. Other joins (STANDARD, MIXED) still use `concat_batch_bytes` as the batch size threshold.

BUILD_PROBE admission requires measured build rows and a full-build count within `cudf::size_type`'s row limit, so even a completely skewed partition can be concatenated. MARK joins follow the same admission rules as SEMI joins and can use partitioned STANDARD execution. Hash-table costs exclude cuco capacity rounding and auxiliary allocations; hash-partitioned admissions use rounded-up averages that do not bound skew. Resident accounting uses decoded bytes even for compressed inputs. Folding can temporarily hold both input and output, and concurrent joins can exhaust available memory despite individual admission budgets.

Runtime distinct-build probing is an internal join policy, not a user configuration choice. It is
temporarily disabled while the cuCollections defect tracked in #1600 remains unresolved. The
engine retains the guarded single-pass `cudf::distinct_hash_join` path for policy-controlled use
after that dependency is fixed.

## Telemetry

```yaml
sirius:
  telemetry:
    enable_quent: true
    enable_nvtx: false
    output_directory: telemetry_data
    engine_name: siriusDB
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_quent` | bool | true | Emit Quent telemetry using the configured exporter. When false, telemetry uses the noop exporter. |
| `enable_nvtx` | bool | false | Capture NVTX ranges from Sirius and dependency images such as libcudf into Quent. No-op when `enable_quent` is false. |
| `exporter` | string | `ndjson` | Quent filesystem exporter; only `ndjson` is supported. |
| `output_directory` | non-empty string | `telemetry_data` | Directory for Quent telemetry files. |
| `engine_name` | non-empty string | `siriusDB` | Engine name reported in engine-level telemetry. |
| `nvtx_injection_lib` | string | empty | Optional NVTX injection-library override. Normally unnecessary: a loadable Sirius uses its own DSO, while a Sirius-enabled DuckDB executable resolves the initializer from itself. `NVTX_INJECTION64_PATH` takes precedence. Used only when `enable_nvtx` is true. |

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
`enable_quent: true` writes Quent ndjson files into `output_directory` as NDJSON.

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
| `scan_manager` | `executor.scan_manager` | remaining cores (min 4) | `scan_manager` | Scan metadata production (the IO reactor threads are separate: `uring_n_reactors`, `rest_n_reactors`, `cufile_n_reactors`) |

The task-creator, downgrade, and scan-manager pools support optional CPU affinity lists
(`cpu_affinity`) for core pinning. GPU pipeline affinity is derived per executor from the selected
GPU's CPU topology. `num_threads` must be `> 0` for every pool.

## DuckDB Settings Changed by Sirius

Loading Sirius changes the database-wide default of one DuckDB setting. A session `SET` still wins, and a value the database already carries explicitly (from the connection config or a `SET GLOBAL` before `LOAD`) is left alone.

| Setting | DuckDB default | Default with Sirius | Description |
|---------|----------------|---------------------|-------------|
| `delim_join_as_cte` | `true` | `false` | Whether DuckDB rewrites the delim joins it creates while decorrelating subqueries into materialized CTEs |

DuckDB v2 rewrites decorrelated subqueries (correlated `EXISTS`, `NOT EXISTS`, `IN` and scalar subqueries) into materialized CTEs by default. Sirius does not yet plan those shapes well: at TPC-H SF1000, Q20, Q21 and Q22 ran 1.2-3.5x slower than with explicit delim joins, because Q21's `NOT EXISTS` became an anti join that hashes every late `lineitem` row and Q20/Q22 lost their dynamic filters behind mark joins. Sirius therefore keeps the explicit `LEFT_DELIM_JOIN` / `RIGHT_DELIM_JOIN` plans. This is a stopgap until Sirius handles the CTE plan shapes, and DuckDB already marks the setting as deprecated, so it must be revisited when DuckDB removes it. To get DuckDB's CTE plans in one session:

```sql
SET delim_join_as_cte = true;
```

## DuckDB SET Variables

Registered in `src/sirius_extension.cpp`. The `operator_params` setters, expression settings, and
pin-compression settings are connection-local overrides of the defaults registered for the database;
GLOBAL writes are rejected and RESET restores the registered default. Each execution uses one
immutable operator-options snapshot. Logging updates serialize runtime-wide. Hardware decompression
requires startup configuration.

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `sirius_log_backend` | `spdlog` | Log sink: `spdlog`, `duckdb`, or `noop` |
| `sirius_log_level` | `info` | Log level: trace, debug, info, warn, error, critical, off (`spdlog` only) |
| `sirius_log_dir` | `log` | Log output directory (`spdlog` only) |
| `sirius_log_flush_seconds` | 3 | Log flush interval in seconds; 0 disables periodic flushing and negative values are rejected (`spdlog` only) |

- **`spdlog`** (default): writes the daily-rotated `<sirius_log_dir>/sirius.log`, honouring
  `sirius_log_level` and `sirius_log_flush_seconds`.
- **`duckdb`**: routes logs into DuckDB's own logging under the `Sirius` log type; enable and
  read them with `CALL enable_logging(); SELECT * FROM duckdb_logs WHERE type = 'Sirius';`.
  Level and enable are governed by DuckDB, not `sirius_log_level`.
- **`noop`**: discards all logs.

These can also be set at load via the `SIRIUS_LOG_BACKEND`, `SIRIUS_LOG_DIR`, and
`SIRIUS_LOG_LEVEL` environment variables.

The embedded FFI engine reads these variables when it constructs a `sirius::ffi::Context`.
If none is set, it leaves the current log sink unchanged. If only `SIRIUS_LOG_LEVEL` is set,
the default `spdlog` backend writes to `./log/sirius.log`; if only `SIRIUS_LOG_DIR` is set,
the default `spdlog` backend writes there. An unknown backend leaves the current sink in
place. With `spdlog`, an unknown level falls back to `info` and emits a warning. The `duckdb`
backend needs a DuckDB database when the sink is installed, so selecting it in the FFI path
does not create a sink; this path has no SQL or settings entry point to enable or read DuckDB
logging. If the selected sink cannot be constructed, such as when the log directory is not
writable, context construction throws and restores the previous `Config::LOG_*` values while
leaving the previous sink active.

### Expression Evaluation

**File:** `src/expression_evaluator/expression_evaluator_strategy.hpp`

| Variable | Default | Description |
|----------|---------|-------------|
| `expression_evaluator_strategy` | `ast_interpret` | Expression evaluator strategy: `materialize`, `ast_interpret`, or `ast_jit` |
| `enable_regex_jit_impl` | true | Enable the specialized JIT implementation for supported regex patterns |

These settings are connection-local: use `SET` or `SET SESSION`; `SET GLOBAL` is
rejected. `RESET` restores the registered default. Query initialization captures both
settings in an immutable snapshot, so later changes do not affect an existing query.
Evaluators retain their own policy values rather than reading mutable global defaults.

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
| `max_build_hash_table_bytes` | 8× startup-resolved `hash_partition_bytes` | Max per-GPU estimated BUILD_PROBE hash-table bytes (measured build rows × 16 B) |
| `max_build_probe_resident_bytes` | 8× batch default | Max per-GPU decoded folded build payload kept GPU-resident by a BUILD_PROBE join |
| `max_broadcast_join_size` | 256 MiB | Max build-side size eligible for a broadcast join |
| `mark_join_build_switch_ratio` | 8.0 | STANDARD MARK join build-side switch ratio (0 disables) |
| `enable_dense_count_join` | true | Enable the fused count-over-outer-join operator; accepted only as a strict boolean under `sirius.operator_params`. |

Eligible GROUP BY and TOP_N merge pipelines are fused automatically. This is an engine-owned plan
policy rather than a user configuration choice; see
[Merge fusion](physical-plan-generation.md#merge-fusion).

The CONCAT output-batch target is derived from effective GPU capacity. Advanced
benchmark and test envelopes may still override `concat_batch_bytes` in YAML
under `sirius.operator_params`, but it is not a normal session setting.

Runtime distinct-build probing is also engine-owned and is temporarily disabled pending #1600.

Dense count-join is enabled by default and can be disabled with
`sirius.operator_params.enable_dense_count_join: false`. Both inputs are FULL barriers and are
hash-partitioned on the join key, so one partition of each input plus its workspace must fit one
GPU task. Its histogram budget is engine-owned and applies per partition task.

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
| `enable_dynamic_filter_multi_partition` | true | Accumulate a Bloom filter for a non-broadcast HASH build with more than one partition. Requires `enable_dynamic_filter`, a build fed through a FULL barrier (an `ORDER BY`-fed build never accumulates), and working peer DMA between every pair of replica GPUs. |
| `max_dynamic_filter_bloom_bytes_per_gpu` | 256 MiB | Cap on the accumulated Bloom arrays of one join on each GPU: the active keys times the array size rounded up to 256 bytes. Must be greater than zero: YAML loading and `SET` reject zero, and `enable_dynamic_filter_multi_partition` is the only switch that disables accumulation. A cap smaller than the required arrays records a size refusal. Whole-build publication is unaffected. |
| `dynamic_filter_domain_coverage_threshold` | 0.9 | Positive finite threshold. Before constructing either a membership filter or zone map, skip the key when the complete build covers at least this fraction of the key's unfiltered base-table row bound. Applies only to build keys proven unique in their base relation, with evidence from DuckDB-native scans. Values above 1.0 disable the gate; exactly 1.0 fires only at full coverage. |
| `dynamic_filter_inlist_max_l2_fraction` | 0.125 | Finite threshold in [0, 1]: maximum estimated cuco-set size for the exact hash IN-list, as a fraction of the smallest probe-GPU L2. Larger sets use Bloom when supported. For keys not handled by the raw IN-list, 0 selects Bloom when supported, while 1.0 reproduces the legacy L2-fit rule only when L2 size is known. If L2 size is unknown, the hash IN-list is ineligible and selection falls back to Bloom or no membership filter. The 0.125 default comes from a GB300 residency sweep: hash-set probe cost is flat below ~0.28 of L2 and degrades beyond it, while Bloom was at least 2.2x faster at every swept set size. |
| `dynamic_filter_keep_threshold` | 0.9 | Finite threshold in [0, 1] for disabling post-decode filtering once a measured split keeps more than this fraction of its rows; 1.0 keeps filtering always on. |

Accumulation holds each GPU's arrays, at most the cap, until safe release after build tasks drain; published filters can remain until query teardown. The arrays are not spillable. Publication also allocates short-lived scratch on the final PARTITION task's allocator: at most two transfer chunks (8 MiB each at most) per contributing non-root GPU. The task's ignore-limit policy may admit that scratch beyond its reservation against global capacity. Initialization admission failure or safely recoverable scratch OOM skips the optional filter. The multi-partition switch and byte cap are YAML settings; like the other switches in this section, their `SET` variables exist only when the process starts with `SIRIUS_ENABLE_TEST_OPTIONS=1`.

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

Prefer the per-call `pin_table(..., compression => true/false)` argument for a specific pin. The
variables below supply the default for calls that omit the argument and configure plan lookup and
retention gates.

| Variable | Default | Description |
|----------|---------|-------------|
| `pin_table_compression` | false | Default for `pin_table` calls that omit the `compression` argument. |
| `pin_table_input_compression_plan_dir` | (empty) | Directory of per-table Simpatico plan files (`<table_name>.<ext>`, multi-column plan DSL). Tables with no matching file are pinned uncompressed. No effect on spill compression. |
| `pin_table_compression_min_batch_size_bytes` | 1 MiB | Minimum uncompressed batch size below which pin-table compression is skipped. |
| `pin_table_compression_max_compressed_fraction` | 0.75 | Discard the compressed form and pin uncompressed when the compressed size exceeds this fraction of the original (compression saved too little). Spill compression uses `spill_max_compressed_fraction` instead. |

Both size gates are evaluated on the narrowed table when compressed materialization is active.
See [Compressed Pinning](compressed-pinning.md) for tier selection, plan authoring, and results.

### Spill Compression (Simpatico)

| Variable | Default | Description |
|----------|---------|-------------|
| `spill_compression` | false | Store batches the downgrade executor spills to the host or disk tier Simpatico-compressed. Applies to the whole process, not just the session. See [Spill compression](#spill-compression-siriuscompression) for the YAML-only gates. |

### Late Materialization (experimental)

Off by default, and gated by environment variables rather than `sirius_config`/`SET` — this is
still an experimental optimization, not a normal session choice.

| Variable | Default | Description |
|----------|---------|-------------|
| `SIRIUS_EXP_LATE_MAT` | off | The master gate. Carries a pin-order rowid instead of scanned values for a GPU-tier pinned table, restoring them at the far end. |
| `SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS` | off | Columns the pin-time uniqueness probe observes: `all`, a name list, or `none`. Required for a group-by-rowid ride or a rider to be admissible at all. |

Five further `SIRIUS_EXP_LATE_MAT_*` knobs tune the deferral floors and the count-on-deferred
path; see [Late Materialization](late-materialization.md#turning-it-on-experimental) for the full
gate table, the mechanism, and results.

### Runtime Data Size Estimation

Runtime estimation projects the total bytes that will reach an input port from measured upstream
ratios. Its current consumer is the grouped-aggregation `PARTITION`, which can overlap with its
producer after an estimate is available. The disabled path retains the original `FULL` barrier.

The partition logs its sizing basis (`projected`, `upstream-complete`, or `measured`) and final
error. See [Data Size Estimation](data-size-estimation.md) for the design.

```sql
SET enable_runtime_size_estimation = true;   -- off by default
```

### Transparent Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `gpu_execution` | true | Enable transparent GPU execution for plain SQL queries. When enabled, supported queries are automatically intercepted and executed on the GPU. When disabled, only explicit `CALL gpu_execution('...')` uses the GPU. |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_duckdb_fallback` | true | Fall back to DuckDB CPU execution on Sirius errors. Gates both plan-time fallback (unsupported operator/type) and runtime fallback (GPU execution failure) on the transparent path, plus explicit `CALL gpu_execution(...)`. Set to `false` to surface Sirius errors instead of falling back. |
| `enable_regex_jit_impl` | true | Use JIT regex implementation |
| `like_swar_fastpath` | true | Dispatch `%lit1%lit2%...%` LIKE/NOT LIKE patterns to the SWAR digram fast-path kernel instead of `cudf::strings::like` |


## Compile-Time Config Flags

### Static flags

**File:** `src/include/config.hpp`

Static constants from `namespace duckdb::Config` and `namespace sirius::Config`:

| Flag | Value | Namespace |
|------|-------|-----------|
| `NUM_GPU_EXECUTOR_THREADS` | 2 | `sirius::Config` |
| `NUM_PIPELINE_EXECUTOR_THREADS` | 1 | `sirius::Config` |
| `NUM_GPU` | 1 | `sirius::Config` |

These are compile-time defaults. Runtime configuration via `sirius_config` and DuckDB SET variables takes precedence.

## Key Files

| File | Purpose |
|------|---------|
| `src/sirius_config.hpp` | Config class, operator_params, thread pool configs |
| `src/config.hpp` | Static config flags |
| `src/sirius_extension.cpp` | SET variable registration |
| `src/scan_manager/config.hpp` | Scan manager config (thread pool, IO reactors, readahead, object store) |
| `cucascade/include/cucascade/io/cache/config.hpp` | Read-path caching config (`scan_manager.cache`: mode, eviction policy, prefetching-cache tunables) |
| `cucascade/include/cucascade/io/uring/config.hpp`, `io/rest/config.hpp`, `io/tls/config.hpp`, `io/object_store_config.hpp` | Per-backend IO / TLS / object-store sub-configs (cuCascade) |

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
concern.  Each uring reactor stages through whole host blocks under a fixed
64 MiB budget, so at `block_size: 64Mi` it pins exactly one block per reactor
and even a small device miss occupies that whole block for the duration of
its I/O. On the REST backend physical GET sizing is adaptive within
`rest.min_segment_bytes` / `rest.max_segment_bytes`, and fragmented cache fills
respect whole-chunk boundaries. Data GETs are bounded by a stall detector
(`stall_speed_limit_bytes` / `stall_time_s`) rather than by a whole-transfer
deadline, so a large transfer does not time out merely because a link is slow.
`pipeline.num_threads` 4 → 8 helps task-parallel aggregation
queries (q1 -16%, q12 -14%). The prefetcher block overlaps pinned-cache
uploads with compute (see `scan_manager.memory_prefetcher` above). Numbers
include the cuCascade all-valid null-mask conversion fix; without it,
expect roughly half the converter-side gain.

The knobs that did NOT help on this hardware, all measured full-suite:
prefetcher `min_free_fraction` below 0.4, prefetcher threads above 3,
pipeline threads above 8. The H2D interconnect sustains ~350-380 GB/s
regardless of stream count; past these settings the link, not scheduling,
is the bound.
