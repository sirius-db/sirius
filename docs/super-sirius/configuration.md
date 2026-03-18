# Configuration

This document covers Super Sirius configuration: the `sirius_config` class, operator parameters, thread pool settings, and DuckDB SET variables.

## `sirius_config`

**File:** `src/include/sirius_config.hpp`

The `sirius_config` class loads configuration from a `.cfg` file (libconfig++ format) or uses defaults. It provides:

- Hardware topology (GPU count, NUMA layout)
- Memory space configurations (GPU, Host, Disk)
- Thread pool configs for all executor types
- Operator parameters (batch sizes, limits)

### Loading

```cpp
sirius_config config;
config.load_from_file("/path/to/config.cfg");  // Optional
```

If no config file is provided, all parameters use defaults.

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
