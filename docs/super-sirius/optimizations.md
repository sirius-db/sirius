# Optimizations

This document catalogs Super Sirius performance optimizations by category. Each entry includes the PR reference, motivation, mechanism, code path, and configuration (if applicable).

## Pipeline-Level Optimizations

### Adaptive Partition Count (PR #371)

**Motivation:** Fixed partition counts waste resources on small datasets and under-partition large ones.

**Mechanism:** `determine_num_partitions()` computes partition count from actual input data size:
```
total_bytes = sum of all batch sizes from input repository
num_partitions = max(1, ceil(total_bytes / hash_partition_bytes))
```

**Code path:** `src/op/sirius_physical_partition.cpp` — `determine_num_partitions()`

**Config:** `hash_partition_bytes` (default: 512 MB)

### Drain and Restart Task Creator (PR #479)

**Motivation:** During pipeline executor drain (e.g., for error recovery or pipeline transitions), in-flight task creation must be safely completed before operator destruction.

**Mechanism:** `drain_pending_tasks()` drains the task creation queue via `_task_creation_queue.drain()` and waits for in-flight task creation lambdas via `_kiosk.wait_all()`.

**Code path:** `src/creator/task_creator.cpp` — `drain_pending_tasks()`

### 3-Phase Sort Pipeline (PR #866)

**Motivation:** Sorting datasets larger than GPU memory requires distributed sorting with dynamic partition boundaries. SORT_SAMPLE and SORT_PARTITION are tightly coupled (the partition operator reads boundaries directly from the sample operator via `_sample_op`), so colocating them in one pipeline eliminates an unnecessary repository hop and scheduling overhead.

**Mechanism:** ORDER_BY is split into 3 pipeline phases:
1. **ORDER_BY**: Local sort of each batch
2. **SORT_SAMPLE + SORT_PARTITION**: Sample N batches to compute boundaries, then range-partition — both run back-to-back in the same `gpu_pipeline_task`
3. **MERGE_SORT**: Multi-way merge of pre-sorted partitions via `cudf::merge_order_by()`

**Code path:**
- `src/planner/sirius_physical_plan_generator.cpp` — `wrap_order_by()` (plan-time sort chain insertion)
- `src/op/sirius_physical_sort_sample.cpp` — boundary computation
- `src/op/sirius_physical_sort_partition.cpp` — range partitioning
- `src/op/sirius_physical_merge_sort.cpp` — multi-way merge

**Config:** `max_sort_partition_bytes` (default: auto, 33% of GPU memory)

### SORT_SAMPLE Byte-Based Merge Sampling (PRs #876, #886)

**Motivation:** Multi-partition sorting samples batches to compute partition boundaries. A fixed sample batch count does not scale with variable batch sizes, single-batch task input contradicted the multi-batch sample the hint waited for, and concatenating + fully re-sorting the sample wasted work because upstream ORDER_BY already emits locally sorted batches. Scheduling boundary computation with a CAS election also let losing tasks waste GPU work.

**Mechanism:** `sirius_physical_sort_sample` overrides `get_next_task_input_data()` so the boundary task receives the full multi-batch sample the hint waited for. Sampling is byte-based: it pulls batches until `sort_sample_bytes` is reached (or upstream finishes), then merges the already-sorted sample batches with `gpu_merge_impl::merge_order_by` and computes boundaries from the merged run — no concatenate-and-full-sort. Boundary scheduling uses an explicit `NOT_DONE -> SCHEDULED -> DONE` state machine: `get_next_task_input_data()` claims the sample and moves to `SCHEDULED`, `get_next_task_hint()` returns `nullopt` while `SCHEDULED` so no duplicate boundary task is created, `execute()` moves to `DONE`, and OOM resets to `NOT_DONE` for retry. After boundaries are computed the operator falls back to single-batch passthrough.

**Code path:** `src/op/sirius_physical_sort_sample.cpp` — `get_next_task_input_data()`, `get_next_task_hint()`, `execute()`; `src/planner/sirius_physical_plan_generator.cpp` — wiring `sort_sample_bytes` into SORT_SAMPLE

**Config:** `sort_sample_bytes` (default: 512 MB), settable via YAML and the `sort_sample_bytes` SET option

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

**Code path:** `src/creator/task_creator.cpp` — `manager_loop()`, `src/pipeline/task_scheduler.cpp` — `schedule_next_scan_tasks()`

**Config:** `max_build_hash_table_bytes` (default: 500 MB) — now independent from `concat_batch_bytes`, enabling larger build sides in BUILD_PROBE mode without affecting other joins.

### Zero-Copy Projection Passthrough (PR #991)

**Motivation:** A projection that simply re-references input columns (`SELECT a, c, a`) previously deep-copied every output column on device via the expression evaluator's BOUND_REF path, even though the data already lived on the GPU.

**Mechanism:** `sirius_physical_projection::execute()` classifies each `select_list` entry as a pure passthrough (`sirius::ast::reference`) or an expression to evaluate, then takes one of three paths per batch:
1. **All evaluated:** owned `cudf::table` (unchanged).
2. **All passthrough:** output is a `cudf::table_view` over the input columns, wrapped as a view-backed batch whose owner is the input's `read_only_data_batch` lock — **zero device copies**.
3. **Mixed:** only the non-passthrough entries are evaluated; the output view mixes evaluated columns with input columns, jointly owned by the evaluated table (`shared_ptr<cudf::table>`) and the input lock.

Only the entries needing evaluation are handed to the evaluator (its `std::vector<sirius::ast::node const*>` constructor), so passthrough columns are never materialized.

**Code path:** `src/op/sirius_physical_projection.cpp` — `execute()`; `src/include/data/data_batch_utils.hpp` — `make_data_batch_from_view()`; `src/include/expression_evaluator/expression_evaluator.hpp` — subset constructor.

### Adaptive MARK Join Build Side (PR #924)

**Motivation:** Equality-only MARK joins build the hash table on the right (filter) side and probe with the left. When the right (probe) side is much larger than the left (output) side, building on the smaller left side is faster.

**Mechanism:** `sirius_physical_hash_join` keeps both paths and picks the one that hashes the smaller side, producing identical output (all left rows plus a BOOL8 mark column):
- `cudf::filtered_join` builds on the right and returns probe-side (left) match indices — used when the right side is not much larger than the left.
- `cudf::mark_join` builds on the left and returns build-side (left) indices — used when `right_rows >= mark_join_build_switch_ratio * left_rows`.

Both feed the same `resolve_mark_join_result()`, which scatters the match indices into the BOOL8 mark column, so only the hashed side changes. Setting the ratio to `0` disables the switch and always builds on the right.

**Code path:** `src/op/sirius_physical_hash_join.cpp` — `make_left_mark_join()`, `make_right_filtered_join_ptr()`, `resolve_mark_join_result()`, and the ratio gate in the MARK branch

**Config:** `mark_join_build_switch_ratio` (default: 8.0; `0` disables), settable via YAML and the `mark_join_build_switch_ratio` SET option

### Projection Folding (PR #909)

**Motivation:** Building a Sirius physical plan from DuckDB's logical plan inserts PROJECTION operators that DuckDB's optimizer never sees — for filter `projection_map` reordering, hoisted aggregate child/filter expressions, table-scan filter columns, and the user's SELECT list. Each PROJECTION is a full GPU pipeline stage that runs `expression_evaluator` over every batch and materializes an intermediate batch, so stacked projections evaluate expressions more than once.

**Mechanism:** All planner projection creation goes through `push_projection()`, and a single `fold_adjacent_projections()` pass over the finished plan tree collapses any `PROJECTION -> PROJECTION` chain into one projection. Folding composes the two select lists with the AST helpers `visit_references()` (find which child outputs the outer list reads) and `substitute_references()` (rewrite outer references in terms of the inner projection's expressions), so the merged projection produces the same columns in one expression-evaluation stage.

**Code path:**
- `src/planner/sirius_plan_projection_utils.cpp` — `push_projection()`, `fold_adjacent_projections()`
- `src/planner/sirius_physical_plan_generator.cpp` — post-pass invocation of `fold_adjacent_projections()`
- `src/expression/ast/reference_utils.cpp` — `visit_references()`, `substitute_references()`

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

**Mechanism:** `request_downgrade(predicate)` enqueues a `downgrade_request` struct onto an MPMC queue. A single processing thread handles requests sequentially, lazily fetching candidates from data repositories, then task queues, dispatching them to a thread pool one-by-one via `convertible_data::convert()`, and evaluating the caller-supplied `predicate` after each completion. The predicate defines "done" (e.g., "memory reservation succeeded") -- no retry loop, no over-freeing.

**Code path:** `src/downgrade/downgrade_executor.cpp` -- `request_downgrade()`, `processing_loop()`

### Pinned Host Memory Caching (PR #437)

**Motivation:** Standard host memory requires page-locking for GPU transfers, which is expensive.

**Mechanism:** `small_pinned_host_memory_resource` maintains pre-allocated pinned memory pools with NUMA affinity. Used for GPU↔CPU transfers and scan output caching.

**Code path:** cuCascade `cucascade/src/memory/small_pinned_host_memory_resource.cpp`, integrated in `src/include/sirius_context.hpp`

**Config:** Memory manager settings in `sirius.yaml` (see [Configuration](configuration.md))

## Scan Optimizations

### Row Group Pruning with Filter Pushdown (PR #363)

**Motivation:** Scanning all row groups wastes I/O bandwidth when filter predicates can eliminate entire groups.

**Mechanism:** When `gpu_expression_translator` successfully converts DuckDB `TableFilterSet` filters into a cuDF AST:
1. `filter_row_groups_with_stats()` uses Parquet column min/max statistics to discard row groups that cannot match the filter — before I/O
2. The AST is set on `parquet_reader_options` via `set_filter()`, pushing filtering into the cuDF reader
3. `TABLE_SCAN` is set to passthrough (no GPU expression evaluation needed)

If translation fails, filtering falls back to `expression_evaluator` on the decoded batch.

**Code path:**
- `src/op/scan/scan_utils.cpp` — `convert_table_filters_to_expression()`, `filter_row_groups_with_stats()`
- `src/op/scan/parquet_gpu_ingestible.cpp` — filter translation + row-group pruning in the per-file metadata task

### Batch Coalescing for Small Files (PR #503)

**Motivation:** Many small Parquet files each produce a tiny GPU batch, causing high per-task scheduling and kernel launch overhead.

**Mechanism:** `sirius_physical_table_scan::get_next_task_input_data()` accumulates batches until `accumulated_bytes >= scan_task_batch_size` OR `batch_count >= 32`. When multiple batches are present, `execute()` calls `cudf::concatenate()` to produce a single fused table before filtering/projecting.

**Code path:** `src/op/sirius_physical_table_scan.cpp` — `get_next_task_input_data()`, `execute()`

**Config:** `scan_task_batch_size` (default: 512 MB)

### Asynchronous Parquet Metadata via Scan Manager (PRs #571, #620, #731)

**Motivation:** Synchronous metadata parsing on the GPU pipeline thread blocks all pipeline tasks until file footers are read, AST filters are translated, and row-group partitions are computed.

**Mechanism:** A dedicated `sirius_scan_manager` runs alongside the GPU executors and owns a thread pool that drives a `split_provider` per GPU scan operator. Each provider composes the format's `gpu_ingestible`, which parses footers (one file per metadata task), translates AST filters, and prunes row groups; a per-query sequencer coalesces the parsed metadata into splits and pushes them into each operator's `split_connector`. The GPU scan operator's `get_next_task_input_data()` blocks on the connector and returns each split as it arrives, so consumer scheduling is decoupled from production order. Metadata tasks run on the shared pool so per-query memory pressure stays bounded.

**Code path:**
- `src/scan_manager/sirius_scan_manager.cpp` — manager thread pool, provider registry, per-query sequencer
- `src/op/scan/parquet_gpu_ingestible.cpp` — footer parsing, AST filter translation, row-group pruning
- `src/scan_manager/split_connector.cpp` — blocking queue between the sequencer and the operator

### Multifile Parquet Splits (PR #738)

**Motivation:** Many small parquet files each yielding a tiny GPU batch causes per-task scheduling and kernel-launch overhead to dominate scan throughput.

**Mechanism:** The `parquet_batch_coalescer` coalesces row-group slices from multiple parquet files into a single split when the bundled files share identical hive-partition values (so synthesized partition columns remain scalar) and the same reader-pushdown decision. Decoded bytes accumulate across files; a split is emitted once the total reaches `approximate_batch_size` or partition values change. The downstream `cudf::io::read_parquet` reads from all bundled slices in one invocation.

**Code path:** `src/op/scan/parquet_gpu_ingestible.cpp` — `parquet_batch_coalescer`

**Config:** `scan_task_batch_size` (default: 512 MB) is forwarded as `approximate_batch_size` to the coalescer.

### Sirius IO + Prefetching Cache (PR #675)

**Motivation:** Repeated parquet reads pay full file-system cost on every query. A pinned-memory cache between the file and cuDF's parquet reader can serve subsequent reads at H2D-copy speed without re-reading from disk.

**Mechanism:** `sirius::io` provides a `cudf::io::datasource` (`sirius_datasource`) backed by io_uring reactors and an optional pinned-memory `prefetching_cache`. The cache hit path issues `cudaMemcpyAsync` from pinned host memory directly to device; the miss path falls through to backend I/O, which uses `O_DIRECT` reads through pinned bounce slots and round-robin dispatch across reactor threads. A packed atomic state machine (4-bit state + 28-bit pin count in one `atomic<uint32_t>`) eliminates TOCTOU between readability checks and pin acquisition. Eviction is driven by a tiered LRU score; admission control caps concurrent in-flight chunks to keep memory bounded.

**Code path:**
- `src/io/sirius_datasource.cpp` — `cudf::io::datasource` implementation
- `src/io/cache/prefetching_cache.cpp` — chunk cache, worker, evictor, buffer pool
- `src/io/uring/uring_reactor.cpp` — io_uring backend reactor
- `src/exec/admission_control.cpp` — RAII budget enforcement

### DuckDB-Native Scan Metadata Walk (PRs #868, #895, #936, #900)

**Motivation:** The DuckDB-native GPU scan's per-row-group metadata walk dominates cold-query runtime at small scale factors. It issues roughly one cold block read per (row group × projected column), and the original walk both read metadata for every column and parsed DuckDB's stringified `GetColumnSegmentInfo` output, which builds and re-parses compression / column-path / `statistics.ToString()` blobs.

**Mechanism:** The walk is structured for minimal, parallel, typed metadata access with statistics pruning:
1. **Projected-column-only, typed walk (#868, #936):** `walk_duckdb_native_row_group_range()` walks the DuckDB segment trees directly for only the projected columns, reading typed `block_id` / compression / row counts / validity-child / max-string-length per segment instead of calling `GetColumnSegmentInfo` and re-parsing strings.
2. **Stats pruning (#900):** `prepare_duckdb_native_walk()` evaluates DuckDB's own `TableFilter::CheckStatistics` against each row group's per-column statistics and drops any row group a pushed-down filter proves `FILTER_ALWAYS_FALSE` before it is staged, copied to the GPU, or decoded; an all-pruned table routes to DuckDB CPU up front.
3. **Parallel range walk + early decode (#895):** `prepare_duckdb_native_walk()` runs as a cheap serial pre-step (partition stats, type-viability gate, row-group count) with no per-segment I/O; the row groups are sliced into ranges of `SIRIUS_METADATA_PARSE_CHUNK` groups, and the scan-manager pool walks the ranges in parallel so cold segment reads for different ranges overlap. The batch coalescer packs parsed ranges into cap-sized batches that decode while later ranges are still being parsed.

**Code path:**
- `src/op/scan/duckdb_native_metadata.cpp` — `prepare_duckdb_native_walk()`, `walk_duckdb_native_row_group_range()`, `mark_row_groups_pruned_by_filter_stats()`
- `src/op/scan/duckdb_native_gpu_ingestible.cpp` — parse-range slicing, per-range walk thunks, and the `duckdb_native_batch_coalescer`

**Config:** `SIRIUS_METADATA_PARSE_CHUNK` (row groups per parallel parse range, default 8)

### DuckDB-Native Async Coalesced Reads (PR #849)

**Motivation:** The DuckDB-native decoder issues many small segment reads. Synchronous `host_read()` calls bypass the datasource backend in favor of direct `pread()`, serializing I/O and inflating the request count per split.

**Mechanism:** The decoder coalesces file-adjacent segment reads — bridging the small per-block header gaps up to a `coalesce_max_gap` derived from the block header size — into large sequential ranges, then issues them as one batch via `sirius_ioctx::host_read_ranges_async_io()` into pinned host blocks. Each coalesced range maps to a contiguous destination span, and the decoder issues bulk asynchronous H2D memcpy into aligned device memory. This cuts read requests per split several-fold and raises read throughput, especially on warm runs.

**Code path:** `src/op/scan/duckdb_native_decoder.cpp` — range coalescing and `host_read_ranges_async_io()` dispatch

### Async S3 / REST Reactor Backend (PR #859)

**Motivation:** A per-request serial S3 backend staged each chunk as GET → H2D copy → `cudaStreamSynchronize`, serializing the GPU stream once per chunk and leaving request latency unhidden — costly when the reader issues many small ranged reads over high-RTT links.

**Mechanism:** The remote read path is an asynchronous, concurrent reactor that plugs into the same `templated_ioctx<Reactor>` abstraction as the local io_uring backend, so the backend-agnostic machinery (sync→async bridge, completion aggregation, per-request fan-out, device chunking) is shared rather than reimplemented. A device read issues async ranged GETs into a bounded pinned host staging block, then `cudaMemcpyAsync` H2D, detecting completion by polling a per-chunk `cudaEvent` (`cudaEventQuery`) rather than synchronizing the stream — so the GET window and copy window overlap and up to `max_connections` chunks are in flight while host staging stays O(`max_connections`).

**Code path:**
- `src/io/rest/rest_reactor.cpp`, `src/io/rest/rest_ioctx.cpp` — async REST/S3 reactor over the shared `templated_ioctx` base
- `src/include/io/templated_ioctx.hpp` — backend-agnostic async machinery shared with the io_uring path

**Config:** `object_store` config (endpoint / region / credentials / signing mode) under `executor.scan_manager`

### Zero-Copy from Pinned and Cached Tables (PR #881)

**Motivation:** When scan input is an already-resident pinned/cached host table, copying the data again on consumption is wasted work.

**Mechanism:** The pinned-table scan path moves host-pinned data to the GPU during prepare-for-processing (so the work is attributed correctly and bounded by the per-task reservation) and avoids copying out of cached tables when the batch can be consumed directly. A synchronization point after prepare-for-processing makes the prepare cost observable in logs and Quent.

**Code path:**
- `src/scan_manager/sirius_scan_manager.cpp` — cached-entry assignment (`try_assign_cached_entries`) and zero-copy resident-batch handoff
- `src/pipeline/gpu_pipeline_task.cpp` — prepare-for-processing synchronization
- `src/op/scan/sirius_gpu_scan_operator.cpp` — resident (cached) input path

### Partial-Read Prefetching Cache (PR #997)

**Motivation:** A read often overlaps the cache only partially. Treating a partial overlap as a miss re-reads bytes already pinned in the cache, and a prefetch-only cache never warms itself from ordinary reads.

**Mechanism:** The prefetching cache is chunk-granular (a fixed chunk size backed by a cuCascade pinned pool) and serves partial reads from cache while populating the cache on read, OS-page-cache style:
- On `device_read_async`, cached chunks are copied straight to the device buffer; the remaining chunks complete the read using pinned chunks in the cache. When the backend supports host-to-device reads, the coverage policy is `partial` so only the uncached chunks hit the backend.
- A read whose chunks are not yet cached can populate those chunk buffers as it reads (interior chunks are cached; block-aligned partial head/tail boundary chunks are read through an internal bounce slot and not cached), so a subsequent read of the same range is a hit.
- Chunk readiness uses a packed atomic state machine; admission control caps concurrent in-flight chunks and a tiered-LRU evictor returns chunks to the pool. Asynchronous results are delivered through the `exec::semi_future` primitive, which can be waited on or connected to an executor callback.

**Code path:**
- `src/io/cache/prefetching_cache.cpp` — `device_read_async()`, partial-read + populate-on-read, evictor
- `src/io/io_context.cpp` — `sirius_ioctx` cache integration and coverage policy
- `src/include/exec/semi_future.hpp` — async I/O completion primitive

**Config:** `enable_prefetch_cache` and the `cache` sub-config under `executor.scan_manager`

### Load-Balanced Scan Batch Coalescing (PR #997)

**Motivation:** Scan splits arrive sized by metadata-parse completion rather than by the configured batch size, so batches could come out smaller than requested, and multi-GPU runs need scan inputs spread across devices. Opportunistic prefetch hints issued in metadata-completion order also rob the head-of-line pipeline of lead time.

**Mechanism:** The scan manager owns a `load_balancing_scan_batch_coalescer`. Each scan pipeline registers a per-pipeline slot holding a `batch_coalescer`, a `split_connector`, and a `balancing_strategy`. A single sequencer task drains the slots in registration (execution) order: it coalesces each pipeline's splits to the requested batch size — independent of the configured max batch count — chooses a device via the balancing strategy, issues `fadvise(opportunistic)` / prefetch hints, and pushes the placed split onto the connector, advancing to the next pipeline only after the current one closes. The `balancing_strategy` interface stamps a `preferred_device_id` on each split (which the task creator honors); `round_robin_strategy` hands out GPUs via an atomic cursor and is the default.

**Code path:**
- `src/scan_manager/load_balancing_scan_batch_coalescer.cpp` — per-pipeline slots, sequencer loop, coalesce + place + push
- `src/include/scan_manager/balancing_strategy.hpp`, `src/scan_manager/round_robin_strategy.cpp` — device-distribution interface and default
- `src/include/scan_manager/split_connector.hpp` — blocking queue between the coalescer and the scan operator

**Config:** `scan_task_batch_size` (default: 512 MB) is the requested coalesced batch size; `executor.scan_manager` sets the thread pool and reactor counts

### Zone-Map Pruning on Pinned Chunks (PR #1154)

**Motivation:** Warm scans previously replayed every pinned chunk, including chunks that a filter
could not match. HOST-pinned chunks also paid an unnecessary H2D copy before being filtered out.

**Mechanism:** At pin time, Sirius runs one min/max reduction for each supported column in every
decoded chunk and stores the results with the pinned entry. This happens before GPU storage or
HOST conversion, so both tiers use the same capture path.

At cache-serve time, static pushed-down filters are checked with DuckDB's `CheckStatistics`.
Chunks proven unable to match are omitted from the cached scan plan; on the HOST tier this also
avoids their H2D copies. Runtime dynamic filters do not participate in this chunk-level pruning.

Unsupported types or filters and missing statistics keep the chunk. If all chunks are pruned,
chunk 0 is retained as a sentinel and emptied by the GPU filter so the pipeline still completes.

**Code path:** `src/pin_table.cpp` captures the statistics;
`src/scan_manager/pinned_chunk_stats.cpp` owns the statistics and safety checks; and
`src/scan_manager/sirius_scan_manager.cpp` builds and serves the survivor plan.

**Config:** `enable_pinned_zone_map_pruning` (default: `true`) is available through YAML and
DuckDB `SET`. It gates both capture and pruning; entries pinned while it is disabled remain
statless until re-pinned. See [Pinned-table zone maps](scan.md#zone-maps) for limitations.
