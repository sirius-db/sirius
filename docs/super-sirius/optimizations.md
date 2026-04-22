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

### Distinct Hash Join (PR #558)

**Motivation:** `cudf::hash_join` does not exploit build-side uniqueness, performing unnecessary work for 1:1 joins.

**Mechanism:** When build-side keys are proven unique via logical plan analysis, Sirius uses `cudf::distinct_hash_join` instead of `cudf::hash_join` in BUILD_PROBE mode. `prove_unique_columns()` walks the DuckDB logical plan subtree and detects uniqueness from:
- PRIMARY KEY constraints on `LogicalGet`
- GROUP BY columns on `LogicalAggregate`
- Propagation through `LogicalFilter`, `LogicalOrder`, `LogicalLimit`, `LogicalTopN`, `LogicalProjection`, `LogicalComparisonJoin`

Only applies to INNER and LEFT joins with pure equality conditions (excludes IS NOT DISTINCT FROM due to `null_equality::UNEQUAL` semantics).

**Code path:** `src/planner/sirius_plan_comparison_join.cpp` — `prove_unique_columns()`, `src/op/sirius_physical_hash_join.cpp` — distinct hash table construction

### Scan Scheduling Tuning (PR #507)

**Motivation:** Eagerly depleting all scan sources at query startup wastes GPU memory on multi-scan plans (e.g., joins with two scanned tables).

**Mechanism:** Two changes:
1. At query startup, at most 2 scans are scheduled initially
2. In `task_creator::manager_loop`, scan exhaustion (continuous creation) only runs when `_num_scans_in_plan == 1`. For 2+ scans, the `get_next_task_hint()` topology-driven mechanism controls task creation instead.

**Code path:** `src/creator/task_creator.cpp` — `manager_loop()`, `src/pipeline/pipeline_executor.cpp` — `schedule_next_scan_tasks()`

**Config:** `max_build_hash_table_bytes` (default: 500 MB) — now independent from `concat_batch_bytes`, enabling larger build sides in BUILD_PROBE mode without affecting other joins.

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

### Adaptive Memory Reservation Estimation (PR #473)

**Motivation:** Fixed-multiplier memory reservation estimates cause either over-reservation (wasting GPU memory) or under-reservation (triggering OOM retries).

**Mechanism:** Each GPU pipeline maintains a `pipeline_memory_history` — a thread-safe ring buffer of up to 64 `task_memory_record` entries recording `estimated_bytes`, `peak_memory_bytes`, and `output_bytes`. `estimate_peak_memory()` computes a weighted average of historical `peak/estimated` ratios, where records with similar estimation bases are weighted higher using a log-ratio distance function. Failed tasks (OOM) ratchet up the estimate by keeping the maximum observed peak for a given input size.

**Code path:**
- `src/include/pipeline/pipeline_memory_history.hpp` — history ring buffer and estimation
- `src/pipeline/gpu_pipeline_task.cpp` — `get_estimated_reservation_size()`

### Downgrade Request Pattern (PR #579)

**Motivation:** The previous downgrade retry loop over-freed memory and caused contention between concurrent downgrade requests competing for the same batches.

**Mechanism:** `request_downgrade(target_bytes, predicate)` enqueues a `downgrade_request` struct onto an MPMC queue. A single processing thread handles requests sequentially, collecting `weak_ptr` candidates up to `target_bytes`, dispatching them to a thread pool one-by-one, and evaluating the caller-supplied `predicate` after each completion. The predicate defines "done" (e.g., "memory reservation succeeded") — no retry loop, no over-freeing.

**Code path:** `src/downgrade/downgrade_executor.cpp` — `request_downgrade()`, `process_request()`

### Pinned Host Memory Caching (PR #437)

**Motivation:** Standard host memory requires page-locking for GPU transfers, which is expensive.

**Mechanism:** `small_pinned_host_memory_resource` maintains pre-allocated pinned memory pools with NUMA affinity. Used for GPU↔CPU transfers and scan output caching.

**Code path:** cuCascade `cucascade/src/memory/small_pinned_host_memory_resource.cpp`, integrated in `src/include/sirius_context.hpp`

**Config:** Memory manager settings in `sirius.yaml` (see [Configuration](configuration.md))

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

### Row Group Pruning with Filter Pushdown (PR #363)

**Motivation:** Scanning all row groups wastes I/O bandwidth when filter predicates can eliminate entire groups.

**Mechanism:** When `gpu_expression_translator` successfully converts DuckDB `TableFilterSet` filters into a cuDF AST:
1. `filter_row_groups_with_stats()` uses Parquet column min/max statistics to discard row groups that cannot match the filter — before I/O
2. The AST is set on `parquet_reader_options` via `set_filter()`, pushing filtering into the cuDF reader
3. `TABLE_SCAN` is set to passthrough (no GPU expression evaluation needed)

If translation fails, filtering falls back to `gpu_expression_executor` on the decoded batch.

**Code path:**
- `src/op/scan/scan_utils.cpp` — `convert_table_filters_to_expression()`, `filter_row_groups_with_stats()`
- `src/op/scan/parquet_scan_task.cpp` — filter integration in global state initialization

### Batch Coalescing for Small Files (PR #503)

**Motivation:** Many small Parquet files each produce a tiny GPU batch, causing high per-task scheduling and kernel launch overhead.

**Mechanism:** `sirius_physical_table_scan::get_next_task_input_data()` accumulates batches until `accumulated_bytes >= scan_task_batch_size` OR `batch_count >= 32`. When multiple batches are present, `execute()` calls `cudf::concatenate()` to produce a single fused table before filtering/projecting.

**Code path:** `src/op/sirius_physical_table_scan.cpp` — `get_next_task_input_data()`, `execute()`

**Config:** `scan_task_batch_size` (default: 512 MB)

### Two-Pipeline Metadata Scan (PR #571)

**Motivation:** Synchronous metadata parsing in `parquet_scan_task_global_state` constructor blocks all pipeline tasks until all file footers are read.

**Mechanism:** Extracts metadata work into a dedicated first pipeline:
- **Pipeline 1** (`PARQUET_METADATA_SCAN` → `GPU_PARQUET_SCAN`): Parses Parquet footers (up to 8 files per task), translates AST filters, computes row-group partitions
- **Pipeline 2** (`GPU_PARQUET_SCAN` as source): Serves self-contained `parquet_scan_data` tasks, each calling `cudf::io::read_parquet`

**Code path:**
- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` — metadata scan operator
- `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` — GPU parquet scan operator

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
