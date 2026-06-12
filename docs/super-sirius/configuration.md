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

If no config file is found, Sirius initializes with built-in defaults (95% GPU memory, 8 GB pinned host memory per NUMA node).

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
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.9
      reservation_limit_fraction: 1.0
      downgrade_trigger_fraction: 0.8
      downgrade_stop_fraction: 0.6
    host:
      capacity_bytes: 439Gi
      initial_number_pools: 785
      pool_size: 512
      block_size: 1Mi
    disk:
      disk_id: 0
      capacity_bytes: 1Ti
      downgrade_root_dirs: "/mnt/nvme/sirius_spill"
  executor:
    pipeline:
      num_threads: 4
      thread_name_prefix: "sirius_pipeline_executor"
    downgrade:
      num_threads: 1
      thread_name_prefix: "sirius_downgrade_executor"
      monitor_period_ms: 10
    duckdb_scan:
      num_threads: 4
      thread_name_prefix: "sirius_scan_executor"
      cache: "parquet"
    task_creator:
      num_threads: 2
  operator_params:
    scan_task_batch_size: 5Gi
    default_scan_task_varchar_size: 256
    max_sort_partition_bytes: 0          # 0 = auto (33% GPU memory)
    hash_partition_bytes: 5Gi
    concat_batch_bytes: 5Gi
    max_build_hash_table_bytes: 500Mi
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
| `num_gpus` | int | 1 | Number of GPUs to use. Mutually exclusive with `gpu_ids`. |
| `gpu_ids` | list of int | — | Explicit GPU device IDs. Mutually exclusive with `num_gpus`. |

### GPU Memory (`sirius.memory.gpu`)

Controls how much GPU VRAM Sirius claims and when it starts evicting data to host memory.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `usage_limit_fraction` | double | 0.95 | Fraction of total VRAM to use as Sirius's GPU memory capacity. The remaining 5% is left for the CUDA runtime, cuDF temporaries, and other GPU consumers. |
| `usage_limit_bytes` | bytes | — | Absolute VRAM limit. Mutually exclusive with `usage_limit_fraction`. |
| `reservation_limit_fraction` | double | 0.9 | Fraction of the GPU capacity (set by `usage_limit_*`) that can be reserved by pipeline tasks. Reservations are acquired before task execution and prevent overcommit. |
| `reservation_limit_bytes` | bytes | — | Absolute reservation limit. Mutually exclusive with `reservation_limit_fraction`. |
| `downgrade_trigger_fraction` | double | 1.0 | Start evicting GPU-resident data to host when reserved memory exceeds this fraction of capacity. At the default of 1.0, downgrading only triggers when the GPU is fully reserved. |
| `downgrade_stop_fraction` | double | 0.7 | Stop evicting when reserved memory drops to this fraction of capacity. The gap between trigger and stop prevents oscillation. |
| `track_per_stream_reservation` | bool | false | Track memory reservations per CUDA stream instead of globally. Useful for debugging per-task memory usage. |

### Host Memory (`sirius.memory.host`)

Controls pinned host memory pools. One pool group is created per NUMA node (auto-detected from hardware topology).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `capacity_bytes` | bytes | 8Gi | Pinned host memory capacity **per NUMA node**. This memory is allocated at startup using `cudaMallocHost`. |
| `reservation_limit_fraction` | double | 0.9 | Fraction of host capacity that can be reserved. |
| `reservation_limit_bytes` | bytes | — | Absolute reservation limit. Mutually exclusive with `reservation_limit_fraction`. |
| `downgrade_trigger_fraction` | double | 0.8 | Start evicting host-resident data to disk when reserved memory exceeds this fraction. |
| `downgrade_stop_fraction` | double | 0.7 | Stop evicting when reserved memory drops to this fraction. |
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

## Operator Parameters

**File:** `src/include/sirius_config.hpp` — `operator_params` struct

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scan_task_batch_size` | 512 MB | Target batch size for DuckDB scan tasks |
| `default_scan_task_varchar_size` | 256 B | Estimated size per VARCHAR value for row count estimation |
| `max_sort_partition_bytes` | 0 (auto) | Max bytes per sort partition. Auto = 33% of GPU memory. |
| `hash_partition_bytes` | 512 MB | Target partition size for hash joins and group-bys |
| `concat_batch_bytes` | 512 MB | Target output batch size for CONCAT operator |
| `max_build_hash_table_bytes` | 500 MB | Max build-side size for BUILD_PROBE join mode |

**Note:** `max_build_hash_table_bytes` can be larger than `concat_batch_bytes`. When it is, the partition operator configures CONCAT to concatenate all batches, enabling the more efficient BUILD_PROBE join mode for larger build sides. Other joins (STANDARD, MIXED) still use `concat_batch_bytes` as the batch size threshold.

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
| `enable_quent` | bool | true | Emit Quent telemetry using the ndjson exporter. When false, telemetry uses the noop exporter. |
| `output_directory` | string | `telemetry_data` | Directory for Quent ndjson files. |
| `engine_name` | string | `siriusDB` | Engine name reported in engine-level telemetry. |

Per-query labels are configured separately with `CALL sirius_set_query_label(...)` SQL function or the `query_label` named parameter on `gpu_execution(...)`.

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
`enable_quent: true` writes Quent ndjson files into `output_directory`.

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
pixi run cargo run --manifest-path rust/Cargo.toml \
  -p sirius-telemetry-server \
  --features ui \
  -- \
  --output-dir telemetry_data
```

Then open `http://localhost:8080` and select the captured Sirius engine/query.
If the config uses a different `output_directory`, pass that path to
`--output-dir`.

## Thread Pool Configuration

| Pool | Default Threads | Thread Name Prefix | Purpose |
|------|----------------|-------------------|---------|
| `task_creator` | 2 | `task_creator` | Task creation from scheduling requests |
| `gpu_pipeline_executor` | 4 | `gpu_pipeline` | GPU pipeline task execution |
| `downgrade_executor` | 4 | `downgrade` | Data tier migration (GPU→Host) |
| `duckdb_scan_executor` | 4 | `scan_executor` | Scan task execution (DuckDB/Parquet) |

Each pool supports optional CPU affinity lists for core pinning.

## DuckDB SET Variables

Registered in `src/sirius_extension.cpp`. These can be changed at runtime:

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `sirius_log_level` | `info` | Log level: trace, debug, info, warn, error |
| `sirius_log_dir` | `log` | Log output directory |
| `sirius_log_flush_seconds` | 5 | Log flush interval |

### Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `use_pin_memory` | true | Use pinned memory for CPU↔GPU transfers |
| `use_pin_memory_for_caching` | false | Use pinned memory for scan caching |

### Expression Evaluation

| Variable | Default | Description |
|----------|---------|-------------|
| `use_cudf_expr` | true | Use cuDF-based expression evaluation |
| `use_custom_top_n` | false | Use custom top-N implementation |

### Scan

| Variable | Default | Description |
|----------|---------|-------------|
| `use_opt_table_scan` | - | Enable optimized table scan |
| `opt_table_scan_num_streams` | - | Number of CUDA streams for optimized scan |
| `opt_table_scan_memcpy_size` | - | Memcpy size for optimized scan |
| `scan_cache_level` | `NONE` | Scan caching level: `NONE`, `PARQUET`, `TABLE_HOST`, `TABLE_GPU` |
| `scan_task_batch_size` | 512 MB | Target scan batch size |
| `default_scan_task_varchar_size` | 256 | VARCHAR size estimate |

### Pipeline / Operator

| Variable | Default | Description |
|----------|---------|-------------|
| `modified_pipeline` | - | Enable modified pipeline execution |
| `max_sort_partition_bytes` | 0 (auto) | Max sort partition bytes |
| `hash_partition_bytes` | 512 MB | Hash partition target size |
| `concat_batch_bytes` | 512 MB | CONCAT output batch size |
| `max_build_hash_table_bytes` | 500 MB | Max build-side hash table bytes |

### Transparent Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `gpu_execution` | true | Enable transparent GPU execution for plain SQL queries. When enabled, supported queries are automatically intercepted and executed on the GPU. When disabled, only explicit `CALL gpu_execution('...')` uses the GPU. |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `print_gpu_table_max_rows` | - | Max rows to print in debug output |
| `enable_fallback_check` | - | Enable fallback validation |
| `enable_duckdb_fallback` | true | Fall back to DuckDB CPU on Sirius errors. Matches the legacy `gpu_processing` path. Set to `false` to surface Sirius errors instead of silently falling back. |
| `enable_regex_jit_impl` | - | Use JIT regex implementation |

## Legacy Config Flags

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
| `src/include/op/scan/config.hpp` | Scan executor config, cache_level enum |
