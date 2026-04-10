# Configuration

This document covers Super Sirius configuration: the `sirius_config` class, operator parameters, thread pool settings, and DuckDB SET variables.

## `sirius_config`

**File:** `src/include/sirius_config.hpp`

The `sirius_config` class loads configuration from a YAML file or uses built-in defaults. It provides:

- Hardware topology (GPU count, NUMA layout)
- Memory space configurations (GPU, Host, Disk)
- Thread pool configs for all executor types
- Operator parameters (batch sizes, limits)

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
      capacity_bytes: 471200000000
      initial_number_pools: 785
      pool_size: 512
      block_size: 1048576
  executor:
    pipeline:
      num_threads: 4
      thread_name_prefix: "sirius_pipeline_executor"
    downgrade:
      num_threads: 1
      thread_name_prefix: "sirius_downgrade_executor"
    duckdb_scan:
      num_threads: 4
      thread_name_prefix: "sirius_scan_executor"
      cache: "parquet"
    task_creator:
      num_threads: 2
  operator_params:
    scan_task_batch_size: 5368709120     # 5 GB
    default_scan_task_varchar_size: 256
    max_sort_partition_bytes: 0          # 0 = auto (33% GPU memory)
    hash_partition_bytes: 5368709120     # 5 GB
    concat_batch_bytes: 5368709120       # 5 GB
```

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

**Validation:** `validate_and_fix()` ensures `max_build_hash_table_bytes < concat_batch_bytes`.

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

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `print_gpu_table_max_rows` | - | Max rows to print in debug output |
| `enable_fallback_check` | - | Enable fallback validation |
| `enable_duckdb_fallback` | false | Fall back to DuckDB CPU on Sirius errors |
| `enable_regex_jit_impl` | - | Use JIT regex implementation |

## Legacy Config Flags

**File:** `src/include/config.hpp`

Static constants from `namespace duckdb::Config` (used by legacy Sirius) and `namespace sirius::Config`:

| Flag | Value | Namespace |
|------|-------|-----------|
| `USE_PIN_MEM_FOR_CPU_PROCESSING` | true | `duckdb::Config` |
| `USE_PIN_MEM_FOR_CACHING` | false | `duckdb::Config` |
| `USE_CUDF_EXPR` | true | `duckdb::Config` |
| `ENABLE_DUCKDB_FALLBACK` | false | `duckdb::Config` |
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
