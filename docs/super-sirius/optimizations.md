# Optimizations

This document catalogs Super Sirius performance optimizations by category. Each entry includes the PR reference, motivation, mechanism, code path, and configuration (if applicable).

## Pipeline-Level Optimizations

### Adaptive Partition Count (PR #371)

**Motivation:** Fixed partition counts waste resources on small datasets and under-partition large ones.

**Mechanism:** `determine_num_partitions()` computes partition count from actual input data size:
```
total_bytes = sum of all batch sizes from input repository
num_partitions = max(1, total_bytes / hash_partition_bytes)
```

**Code path:** `src/op/sirius_physical_partition.cpp` — `determine_num_partitions()`

**Config:** `hash_partition_bytes` (default: 512 MB)

### Drain and Restart Task Creator (PR #479)

**Motivation:** During pipeline executor drain (e.g., for error recovery or pipeline transitions), in-flight task creation must be safely completed before operator destruction.

**Mechanism:** `drain_pending_tasks()` drains the task creation queue via `_task_creation_queue.drain()` and waits for in-flight task creation lambdas via `_kiosk.wait_all()`.

**Code path:** `src/creator/task_creator.cpp` — `drain_pending_tasks()`

### 4-Phase Sort Pipeline (Architectural)

**Motivation:** Sorting datasets larger than GPU memory requires distributed sorting with dynamic partition boundaries.

**Mechanism:** ORDER_BY is split into 4 pipeline phases:
1. **ORDER_BY**: Local sort of each batch
2. **SORT_SAMPLE**: Sample N batches, compute P-1 partition boundary rows
3. **SORT_PARTITION**: Range-partition data using boundaries
4. **MERGE_SORT**: Multi-way merge of pre-sorted partitions via `cudf::merge_order_by()`

**Code path:**
- `src/sirius_engine.cpp` — `initialize_internal()` (pipeline splitting)
- `src/op/sirius_physical_sort_sample.cpp` — boundary computation
- `src/op/sirius_physical_sort_partition.cpp` — range partitioning
- `src/op/sirius_physical_merge_sort.cpp` — multi-way merge

**Config:** `max_sort_partition_bytes` (default: auto, 33% of GPU memory)

## Operator-Level Optimizations

### Adaptive Join BUILD_PROBE Mode (PR #423)

**Motivation:** For small build-side datasets, building the hash table once and probing many times is more efficient than the standard multi-partition Cartesian product approach.

**Mechanism:** `update_join_exec_mode()` switches to BUILD_PROBE mode when:
- Only 1 partition
- Build-side data < `max_build_hash_table_bytes`

In BUILD_PROBE mode, the first task builds a `cudf::hash_join` hash table and caches it. Subsequent tasks only probe.

**Code path:** `src/op/sirius_physical_hash_join.cpp` — `update_join_exec_mode()`

**Config:** `max_build_hash_table_bytes` (default: 500 MB)

### COUNT DISTINCT Optimization (PR #414)

**Motivation:** Exact COUNT(DISTINCT) requires expensive deduplication.

**Mechanism:** Uses cuDF's `COLLECT_SET` aggregation for distinct value collection, with `MERGE_SETS` in the merge phase. For multi-column DISTINCT, synthesizes struct columns from multiple input columns.

**Code path:**
- `src/op/aggregate/gpu_aggregate_impl.cpp` — `cudf::approx_distinct_count` usage
- `src/op/aggregate/aggregate_op_util.cpp` — `has_count_distinct` flag

## Memory Optimizations

### Memory-Pressure-Driven Downgrade (PR #368)

**Motivation:** GPU memory can be exhausted during complex queries with many concurrent pipelines.

**Mechanism:** Downgrade executor monitors GPU memory space every ~10ms. When `downgrade_trigger_fraction` is exceeded, `run_downgrade_pass()` selects candidates:
1. Partitioned repositories first, sorted by data size descending
2. Non-active partitions before active ones
3. Last-to-first partition iteration

Data is moved from GPU to HOST tier via converter registry.

**Code path:** `src/downgrade/downgrade_executor.cpp` — `monitor_loop()`, `run_downgrade_pass()`

**Config:** `downgrade_trigger_fraction` (default: 1.0 for GPU, 0.8 for Host), `downgrade_stop_fraction` (default: 0.7)

### OOM Retry Mechanism (PR #364)

**Motivation:** Transient GPU OOM can occur when multiple tasks compete for memory.

**Mechanism:** Operators throw `oom_reschedule_exception` carrying intermediate results and resume index. The GPU executor catches this and:
1. Preserves intermediate operator data
2. Creates a rescheduled task starting from the failure point
3. Retries up to 10 times with 5ms backoff

**Code path:**
- `src/include/pipeline/oom_reschedule_exception.hpp` — exception class
- `src/pipeline/gpu_pipeline_executor.cpp` — retry logic in `manager_loop()`

### Memory Pool Defragmentation (PR #378, #452)

**Motivation:** CUDA memory pools can become fragmented, causing allocation failures even with sufficient free memory.

**Mechanism:** On allocation failure, `defragmenter_oom_policy`:
1. Checks fragmentation via `cudaMemPoolGetAttribute()` (reserved vs. used)
2. If `reserved > used + 10× requested`: pool is fragmented
3. Trims pool with `cudaMemPoolTrimTo()` to release free blocks to driver
4. Retries allocation

**Code path:** `src/memory/defragmenter_oom_policy.cpp`

### Pinned Host Memory Caching (PR #437)

**Motivation:** Standard host memory requires page-locking for GPU transfers, which is expensive.

**Mechanism:** `small_pinned_host_memory_resource` maintains pre-allocated pinned memory pools with NUMA affinity. Used for GPU↔CPU transfers and scan output caching.

**Code path:** cuCascade `cucascade/src/memory/small_pinned_host_memory_resource.cpp`, integrated in `src/include/sirius_context.hpp`

**Config:** Memory manager settings in `sirius.cfg` (cuCascade)

## Scan Optimizations

### Scan Output Caching (PR #340)

**Motivation:** Repeated queries on the same data waste time re-scanning from storage.

**Mechanism:** Four caching levels:
- `NONE` — no caching
- `PARQUET` — cache raw compressed Parquet bytes in host memory
- `TABLE_HOST` — cache decoded table in host memory
- `TABLE_GPU` — cache decoded table in GPU memory (fastest warm runs)

Query hash matching detects cache hits. On cache hit (PRELOAD mode), data is loaded from cache with shallow cloning for zero-copy sharing.

**Code path:**
- `src/include/op/scan/config.hpp` — `cache_level` enum
- `src/op/scan/duckdb_scan_executor.cpp` — cache/preload logic

**Config:** `scan_cache_level` SET variable

### Skip File I/O from Cache (PR #455)

**Motivation:** Cached Parquet data should avoid redundant file I/O.

**Mechanism:** `prefetched_data_source` implements `cudf::io::datasource`:
- `cache_ranges` coalesces adjacent byte ranges from cached Parquet files
- `host_read()` satisfies reads from cache via `get_ranges()`, falling back to file I/O only on cache miss
- `device_read()` uses `cudaMemcpyBatchAsync()` (CUDA 13+) for efficient multi-span H2D copies with NUMA/device locality hints
- Tracks `bytes_read_from_cache` vs `bytes_read_from_fallback` for monitoring

**Code path:**
- `src/op/scan/cached_ranges.cpp` — byte range coalescing
- `src/op/scan/prefetched_data_source.cpp` — cached datasource
