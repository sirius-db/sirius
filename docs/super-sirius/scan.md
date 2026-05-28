# Scan Subsystem

This document covers the scan subsystem end-to-end: how data enters Super Sirius from storage through two scan paths, the scan executor, caching, and prefetched data sources.

## Overview

Super Sirius supports four scan paths:

| Path | Operator | Use Case | Data Flow |
|------|----------|----------|-----------|
| **DuckDB Scan** | `DUCKDB_SCAN` | General DuckDB-managed tables | DuckDB table function → column builders → `host_data_representation` |
| **Parquet Scan** | `PARQUET_SCAN` | Legacy direct Parquet reading | Parquet byte ranges → `host_parquet_representation` |
| **GPU Parquet Scan** | `GPU_PARQUET_SCAN` | Parquet file reading via the scan manager | `sirius_scan_manager` produces `parquet_scan_data` splits → GPU read |
| **Iceberg Scan** | `ICEBERG_SCAN` | Apache Iceberg V1/V2/V3 tables | Parquet scan + GPU-accelerated delete filters |

The DuckDB and legacy parquet paths funnel through `duckdb_scan_executor` and the data-repository infrastructure. The GPU parquet path is driven by `sirius_scan_manager` and a dedicated `split_provider` per scan operator (see [Scan Manager](#scan-manager)).

## Scan Operators

### `sirius_physical_table_scan`
**File:** `src/include/op/sirius_physical_table_scan.hpp`

Base scan operator wrapping a DuckDB table function. During pipeline construction (`initialize_internal()`), it is converted to either DUCKDB_SCAN or PARQUET_SCAN based on the table function bind data.

Key members:
- `function` — DuckDB `TableFunction`
- `bind_data` — function binding info
- `column_ids` — which columns to scan
- `projection_ids` — projection optimization
- `table_filters` — predicate pushdown filters
- `scanned_types` — types of scanned columns (constructed from column IDs)

### `sirius_physical_duckdb_scan`
**File:** `src/include/op/sirius_physical_duckdb_scan.hpp`

Sequential scan using DuckDB's execution engine. Tracks an atomic `exhausted` flag. The `scanned_types` vector defines the column types for building output batches.

### `sirius_physical_parquet_scan`
**File:** `src/include/op/sirius_physical_parquet_scan.hpp`

Direct Parquet file scan. Maintains:
- `scanned_ids` — mapping of projection IDs to file column indices
- `has_more_partitions` — atomic flag for pipeline completion
- Row groups are partitioned by `approximate_batch_size` in the global state

### `sirius_physical_iceberg_scan`
**File:** `src/include/op/sirius_physical_iceberg_scan.hpp`

Iceberg table scan. Inherits from `sirius_physical_parquet_scan`. Holds delete file lists (`positional_delete_files`, `equality_delete_files`) and routes through the GPU parquet scan pipeline with a post-convert delete filter hook. See [Iceberg Scan](#iceberg-scan) below.

### `sirius_gpu_parquet_scan_operator` — `GPU_PARQUET_SCAN`
**File:** `src/include/op/scan/sirius_gpu_parquet_scan_operator.hpp`

Source operator for parquet scans. Carries a `parquet_scan_info` populated by the pipeline converter from the DuckDB bind data (file paths, projected column ids, table filters, hive partition indices, target batch size). The operator owns a bound `split_connector`; `get_next_task_input_data()` blocks inside `split_connector::get_next_split()` until either a `parquet_scan_data` is available or the connector is closed and drained.

`execute(input_data)` runs per task: it calls `cudf::io::read_parquet` over the row-group slices in the split, optionally applies a fallback DuckDB expression filter when AST translation failed, prunes pure-filter columns, and assembles the output table according to the `scan_plan` (data columns, hive-partition synthesis, output ordering).

## Scan Manager

**Files:** `src/include/scan_manager/sirius_scan_manager.hpp`, `src/scan_manager/sirius_scan_manager.cpp`, `src/include/scan_manager/split_provider.hpp`, `src/include/scan_manager/split_connector.hpp`

`sirius_scan_manager` owns a configurable thread pool and is responsible for producing the input splits consumed by every `GPU_PARQUET_SCAN` source operator. It runs alongside the GPU pipeline executors and is independent from the data repository / port machinery used between intermediate operators.

### Components

| Component | File | Role |
|-----------|------|------|
| `sirius_scan_manager` | `scan_manager/sirius_scan_manager.{hpp,cpp}` | Owns thread pool, holds providers and pinned-table entries, drives provider execution |
| `split_provider` | `scan_manager/split_provider.hpp` | Abstract producer of `operator_data` splits |
| `parquet_split_provider` | `scan_manager/parquet_split_provider.{hpp,cpp}` | Provider that parses parquet metadata and emits `parquet_scan_data` per row-group partition |
| `cached_split_provider` | `scan_manager/cached_split_provider.{hpp,cpp}` | Provider that emits zero-copy splits over pinned columns (see [Pinned Tables](#pinned-tables)) |
| `split_connector` | `scan_manager/split_connector.hpp` | Lock-protected blocking queue between the provider and the operator |

### Lifecycle

1. **Plan stage:** during pipeline conversion, `split_parquet_scan_source` packages the DuckDB bind data into a `parquet_scan_info` and constructs a `sirius_gpu_parquet_scan_operator` carrying that info. The operator is inserted at `operators[0]` of the pipeline; no separate metadata pipeline is created.
2. **Per-query preparation:** `sirius_scan_manager::prepare_for_query(query)` walks the plan, picks each parquet scan source, and calls `create_provider_for(op)`. The factory returns either a `cached_split_provider` (when a pinned entry matches the operator's file paths) or a `parquet_split_provider`. A fresh `split_connector` is bound to the operator and the provider is stored in the manager's map.
3. **Execution:** a driver thread runs providers **sequentially** in registration order. Each provider's `start(pool, connector)` schedules work onto the manager's thread pool and returns a future; the driver waits on it before starting the next provider. Inside the operator, `get_next_task_input_data()` blocks on `split_connector::get_next_split()` and returns each split as it arrives, so consumer-side scheduling is decoupled from production order.
4. **Teardown:** the provider closes the connector on every termination path (success, exception); on synchronous failure the manager closes it as a safety net. Once closed and drained, the operator's `all_ports_empty()` returns true and `get_next_task_hint()` returns `nullopt`.

### `parquet_split_provider`

`start()` iterates over the file list in `_max_file_processed`-sized batches (default 8). Each batch:

1. Fetches parquet footers via `cudf::io::parquet::fetch_footer_to_host()`.
2. Translates the cached DuckDB filter expression into a cuDF AST on a task-local CUDA stream (filter translation is deferred from construction so each task gets its own stream).
3. Prunes row groups by min/max statistics (`filter_row_groups_with_stats`).
4. Bundles row-group slices into partitions of approximately `approximate_batch_size` uncompressed bytes. Multiple files with identical hive-partition values may be coalesced into a single partition (see [Multifile Bundling](#multifile-bundling)).
5. Pushes one `parquet_scan_data` per partition into the connector. Each split carries the row-group slices, the reader options, the AST filter (if translated), and a `shared_ptr<const scan_plan>`.

### `scan_plan`

**File:** `src/include/op/scan/scan_plan.hpp`, `src/op/scan/scan_plan.cpp`

`scan_plan` is the canonical description of what a scan reads, how it assembles output, and how filters map between index spaces. It is constructed once per provider and shared (immutably) with every emitted split.

```cpp
struct scan_plan {
  std::vector<data_column>           data_columns;       // columns read from parquet, in batch order (D)
  std::vector<partition_column>      partition_columns;  // hive-injected columns (name, type, primary index)
  std::vector<output_entry>          output_layout;      // one entry per output column, in DuckDB order
  std::vector<std::optional<size_t>> batch_position_by_column_id;  // C → D map
  std::unordered_set<size_t>         partition_primary_indices;    // for filter-skip
};
```

Three index spaces appear in the parquet path:

- **P (primary index)** — DuckDB schema position
- **C (column-ids position)** — index into the scan's `column_ids` list
- **D (batch position)** — column position in the cuDF reader output (post-hive-removal)

`output_layout` is walked once in `execute()` to produce the final table: `DATA(k)` entries `std::move` from the read batch at position k, `PARTITION(k)` entries synthesize a scalar-backed column from the hive partition value. Pure-filter data columns (read but not output) fall out of scope and free.

For `SELECT *` with no partitions and no pure-filter columns, `build_inject_fn()` returns `nullptr` and the operator forwards the reader output unchanged — no permute, no copy. `SELECT count(*)` short-circuits the same way (output_layout empty) so the count aggregation sees a 0-column table without a synthesized 0-column reshape.

### Multifile Bundling

When many small parquet files each yield a small batch, scheduling and kernel-launch overhead dominates. `parquet_split_provider` coalesces row-group slices from **multiple files** into a single split as long as the bundled files share identical hive-partition values (so the synthesized partition columns remain scalar). `accum.total_uncompressed_bytes` accumulates across files; a split is emitted once it exceeds `approximate_batch_size` or partition values change. The downstream `cudf::io::read_parquet` call reads from all bundled files in one invocation.

### Column Mapping

Parquet column-chunk order is not guaranteed to match DuckDB's logical column order. `parquet_split_provider` builds a name-based DuckDB→parquet mapping via `parquet_schema_mapping::leaf_indices_for_column(schema, column_name)`, which walks the parquet schema's `path_in_schema` (case-insensitive, mirroring DuckDB).

For nested types (`STRUCT`, `LIST`), one DuckDB column maps to multiple parquet leaf chunks; the mapping returns all leaves under the top-level column name. The cuDF parquet reader, given a top-level column name, materializes the nested `cudf::column` natively without post-read reassembly.

## Pinned Tables

**Files:** `src/include/pin_table.hpp`, `src/pin_table.cpp`, `src/include/scan_manager/cached_split_provider.hpp`

The `pin_table` table function lets users pre-load a parquet table's columns into GPU memory (or, in the future, host memory) so subsequent scans of the same path bypass file I/O entirely.

```sql
CALL pin_table('/path/to/lineitem.parquet',
               name = 'lineitem',
               tier = 'gpu',
               cols = ['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_shipdate']);

-- Subsequent reads of the same path are served from the pinned columns.
SELECT SUM(l_extendedprice * l_quantity)
  FROM read_parquet('/path/to/lineitem.parquet')
  WHERE l_shipdate >= DATE '1994-01-01';

CALL unpin_table('lineitem');
```

`tier` accepts `gpu` or `host`; only `gpu` is supported today (a `host` argument throws `NotImplementedException` and will be added later).

A `pinned_entry` stores the column projection, resolved file paths, per-column chunk vectors, and the memory space the columns reside in. When `prepare_for_query` runs, the scan manager matches the operator's `parquet_scan_info::file_paths` against pinned entries; on a hit, it constructs a `cached_split_provider` (zero-copy view-backed `data_batch` per chunk) instead of a `parquet_split_provider`. The cached provider forwards the same `scan_plan` and filter expression as the parquet path, so the operator's `execute()` is unchanged. When `needs_output_assembly(*plan)` is false (identity layout), the cached batch is forwarded straight through with no permute or prune.

`insert_pinned_entry` supports re-pinning: if an entry exists with the same row count, only new columns are merged in (duplicates dropped); a different row count drops and replaces the entry.

## DuckDB Scan Task

**File:** `src/op/scan/duckdb_scan_task.cpp`, `src/include/op/scan/duckdb_scan_task.hpp`

### Global State

`duckdb_scan_task_global_state`:
- Manages DuckDB table function global state
- Thread-safe shared state across all scan tasks
- `MaxThreads()` — maximum concurrent scan threads
- `is_source_drained()` — checks if all local states are complete
- Filters are NOT passed to DuckDB (applied by `sirius_physical_table_scan` instead)

### Local State

`duckdb_scan_task_local_state`:
- Maintains per-task state with target batch size (`DEFAULT_SCAN_TASK_BATCH_SIZE`)
- **Column builder** — nested struct managing memory for individual columns:
  - Fixed-width types: data array + validity mask
  - VARCHAR: offset array + data + validity mask
  - Uses `multiple_blocks_allocation_accessor` for writing
  - 8-byte aligned column starts
- Row estimation based on:
  - Actual width of fixed-width types
  - Default VARCHAR width (`DEFAULT_SCAN_TASK_VARCHAR_SIZE`)
  - Per-column validity mask bits (1 bit per row, rounded up)

### Execution Flow

```
1. get_next_chunk() → fetch from DuckDB table function
2. chunk_fits() → check if data fits in pre-allocated buffers
3. process_chunk() → write chunk into column builders
4. Repeat until target batch size reached or source drained
5. Build host_data_representation from column builders
6. If scan incomplete: create next scan task (self-scheduling)
```

## Parquet Scan Task

**File:** `src/op/scan/parquet_scan_task.cpp`, `src/include/op/scan/parquet_scan_task.hpp`

### Global State

`parquet_scan_task_global_state`:
- Reads Parquet footers at construction via `cudf::io::parquet::fetch_footer_to_host()`
- Extracts file paths, sizes, footer offsets from DuckDB `MultiFileBindData`
- Computes compressed/uncompressed byte sizes per row group
- Partitions row groups into scan tasks: groups by accumulated uncompressed bytes where each partition ≈ target batch size
- Atomic counter `_next_rg_partition` for lock-free task scheduling
- Supports rebinding for cache reuse across query re-executions
- Only supports flat schemas (no nested columns)

### Local State

`parquet_scan_task_local_state`:
- Stores file index and row group indices for this task
- Reserves both compressed bytes (for allocation) and uncompressed bytes (metadata)

### Execution Flow

```
1. Construct byte ranges covering:
   - Parquet header (4 bytes PAR1 magic)
   - Column chunk byte ranges for selected row groups
   - Parquet footer + trailer
2. Allocate into chunked host memory
3. Read byte ranges asynchronously via host_read_async()
4. Wait for all read futures
5. Build host_parquet_representation wrapping:
   - Cached allocation, hybrid scan reader, reader options
   - Row group indices, byte ranges, file metadata
6. Optional materialization: if enabled, decompress Parquet → GPU table → host table
7. Wrap in cached_*_representation if caching enabled
```

## Scan Executor

**File:** `src/op/scan/duckdb_scan_executor.cpp`, `src/include/op/scan/duckdb_scan_executor.hpp`

### Thread Model

- **Manager thread**: consumes scan tasks from queue, acquires kiosk tickets, submits to thread pool
- **Worker thread pool** (default: 4 threads): executes scan tasks concurrently
- **CUDA stream pool**: exclusive streams per thread for async Host→Device transfers

### Manager Loop

```
while running:
    1. kiosk.acquire()              -- wait for worker availability
    2. task_queue.pop() or:
       - Try non-blocking pop first
       - If empty: submit scan task request to pipeline executor
       - Then blocking pop
    3. For parquet scans: acquire HOST memory reservation
    4. Dispatch to thread pool:
       a. Acquire CUDA stream
       b. get_scan_output() — applies caching logic
       c. scan_task->publish_output() — store to data repository
       d. Schedule output consumers via task_creator
```

### Cache Handling

`get_scan_output()` applies caching logic based on mode:

| Mode | Behavior |
|------|----------|
| CACHE (first run) | Execute scan, save result to cache |
| PRELOAD (cache hit) | Load from cache, clone if needed |

Cloning logic:
- `TABLE_GPU` cache level: return original (GPU-resident, no copy)
- Other levels: clone batch to avoid sharing cache data
- Parquet: `shallow_clone()` — increments refcount, zero-copy

## Caching Mechanism

**File:** `src/include/op/scan/config.hpp`

Four caching levels control scan result persistence:

### `NONE` (default)
No caching. Full scan on every query. Minimal memory overhead.

### `PARQUET`
Cache raw compressed Parquet bytes in host memory. Stored as `cached_host_parquet_representation`. Decompression happens on each re-execution. Smallest memory footprint for parquet scans.

### `TABLE_HOST`
Cache decoded (decompressed) table in host memory. Stored as `cached_host_data_representation`. Avoids decompression cost on re-execution. Medium memory usage.

### `TABLE_GPU`
Cache decoded table in GPU memory. Fastest — no data movement needed for GPU execution. Highest memory cost.

## Data Representations

### `host_data_representation`
Fixed-width columnar data in host memory. Created directly by `duckdb_scan_task` from DuckDB chunks via column builders.

### `host_parquet_representation`
Raw Parquet bytes in host memory with deferred decompression. Contains:
- `multiple_blocks_allocation` — byte chunks
- `hybrid_scan_reader` — cuDF reader for metadata + decoding
- Byte ranges and row group indices
- File metadata (size, footer offset)

### `cached_shared_representation<T>`
**File:** `src/include/data/cached_data_representation.hpp`

Template wrapper for caching any `idata_representation` type:
- `clone(stream)` — deep copy for unique batches
- `shallow_clone()` — reference-counted copy for cache hits
- `get_representation()` — access underlying shared representation

Specializations:
- `cached_host_parquet_representation = cached_shared_representation<host_parquet_representation>`
- `cached_host_data_representation = cached_shared_representation<host_data_representation>`

## Prefetched Data Source

**File:** `src/op/scan/prefetched_data_source.cpp`, `src/include/op/scan/prefetched_data_source.hpp`

Implements `cudf::io::datasource` interface for cached Parquet data.

### `cache_ranges`
**File:** `src/op/scan/cached_ranges.cpp`

Stores sorted, non-overlapping byte ranges with packed buffers:
- Coalesces adjacent ranges to minimize lookups
- Binary search for `get_ranges(offset, size)` — returns spans covering requested bytes
- Returns `nullopt` if query crosses range boundary (not in cache)
- Supports NUMA-aware hints (`device_id`, `numa_id`) for batch copy optimization

### `host_read()`
Delegates to `cache_ranges::get_ranges()`. If cached, copies spans via memcpy. If not cached, falls back to the original datasource. Tracks `bytes_read_from_cache` vs `bytes_read_from_fallback` atomically.

### `device_read()`
Enqueues async Host→Device copies:

**CUDA 13+ path:**
```cpp
cudaMemcpyBatchAsync()  // Efficient multi-span batched copies
```
Sets `cudaMemcpyAttributes` with NUMA/device locality hints for optimal placement.

**CUDA <13 fallback:**
```cpp
// Per-span cudaMemcpyAsync()
for (auto& span : spans) {
    cudaMemcpyAsync(dst, span.data, span.size, H2D, stream);
}
```

### `device_read_async()`
Uses deferred lambda with CUDA event synchronization:
1. Records `cuda_event_guard` after async copies
2. Returns future that syncs the event on `get()`

## Sirius IO Subsystem

**Files:** `src/include/io/`, `src/io/`

`sirius::io` is a `cudf::io::datasource`-compatible I/O stack designed for high-throughput parquet reading. It is built around io_uring reactors and a pinned-memory prefetching cache, with a pluggable backend seam so additional backends (e.g. cuFile) can be added without changing the cache or the datasource layer.

### Architecture

| Component | File | Role |
|-----------|------|------|
| `sirius_datasource` | `io/sirius_datasource.{hpp,cpp}` | `cudf::io::datasource` implementation; `supports_device_read() = true`; delegates every read to the bound `sirius_ioctx` |
| `sirius_ioctx` | `io/types.hpp` | Abstract shared context owning the optional `prefetching_cache` and the reactor pool. `device_read{,_async}` consults the cache and falls through to backend I/O on miss |
| `templated_ioctx<Reactor>` | `io/templated_ioctx.hpp` | Generic ioctx implementation: request splitting, aligned 1 MiB chunking, round-robin dispatch across reactors, sync/async adapters |
| `uring_reactor` / `uring_ioctx` | `io/uring/` | Concrete io_uring backend. One thread per reactor, `O_DIRECT` device reads through pinned bounce slots, buffered host reads on the same ring |
| `prefetching_cache` | `io/prefetching_cache.{hpp,cpp}` | Pinned-memory chunk cache with lock-free per-entry state machine, background worker, evictor threads, and tiered LRU buckets |
| `buffer_pool` | `io/prefetching_cache.cpp` | Growable multi-slab pool of 1 MiB pinned chunks |
| `admission_control` | `io/admission_control.{hpp,cpp}` | RAII slot handed out against a fixed in-flight budget (default 2 GiB worth of chunks) |

### Backend Seam

Two C++20 concepts define the plug-in contract:

- `io_object_c<O, Handle>` — derives from `sirius_io_object`, exposes `host_handle()` / `device_handle()` of type `Handle`.
- `io_reactor_c<R>` — associated types (`native_handle_type`, `io_object_type`, `device_read_req_type`, `host_read_req_type`) plus operations: `enqueue_bulk`, `host_read`, `host_read_async`, `shutdown`, static `align_to_physical`.

A new backend is: a custom `io_object` + reactor + `templated_ioctx<your_reactor>`. `uring_ioctx = templated_ioctx<uring_reactor>` is the first instantiation.

### Cache Seam

`sirius_ioctx::device_read{,_async}` is non-virtual: it consults `_cache` and falls through to pure-virtual `device_read_io{,_async}`. Backends never see the cache; the cache never sees the backend. `supports_device_read()` stays `true` even when the cache serves the read because the final copy is still `cudaMemcpyAsync` from pinned host memory to device.

### Cache Internals

- **Packed atomic state machine.** `entry_state` encodes a 4-bit state enum and a 28-bit pin count in a single `atomic<uint32_t>`. Every transition is one CAS, closing the TOCTOU gap between "is this entry readable?" and "bump the pin count." Readers park in `wait_while_loading()` via `atomic::wait` and are woken by `notify_all()` on completion.
- **Request aggregation via `request_context`.** One logical read fans out into N chunk sub-requests; each sub-request decrements `pending`; the last one fires the user's completion handler. Error reporting is single-writer (`failed.exchange`) so partial failures don't race.
- **Batch dispatch, amortized wakes.** `templated_ioctx::enqueue_device_read` groups chunks per-reactor with a rotating round-robin start and dispatches one `enqueue_bulk` per non-empty group, collapsing N wake-notifies to at most M (reactor count).
- **Multi-GPU safe.** `device_read_req` carries the caller's `device_id`; reactor threads `cudaSetDevice()` before issuing the H2D copy. Bounce slabs are `cudaHostAllocPortable` so they're reachable from any CUDA context.
- **Evictor as backpressure service.** When `buffer_pool.allocate_bulk` can't satisfy the worker, the worker posts an `eviction_request` (promise + chunk count) and blocks on the future. The evictor walks LRU buckets coldest-first, returns chunks to the pool, then resolves the promise. Pool exhaustion is never a silent failure.
- **Tiered LRU with age drift.** Five buckets; `refresh_cache()` is the caller's input to the aging signal. Score is `(NUM_BUCKETS-1) + n_total_request − cache_age`, clamped. Never-consumed entries get a floor of 1 to avoid being first out; raw score `< -5` is evicted on the spot during candidate drain.
- **Admission control deadlock escape.** A request larger than the total budget is granted the full budget when no other slots are outstanding, so oversized reads make progress instead of waiting forever.

### Constants (in `io/types.hpp`)

| Name | Value | Role |
|------|-------|------|
| `CHUNK_SIZE` | 1 MiB | Bounce-buffer / cache chunk size |
| `NUM_CHUNKS` | 32 | Bounce slots per reactor |
| `IO_BLOCK_SIZE` | 4096 | `O_DIRECT` alignment |
| `CHUNKS_PER_SLAB` | 500 | Pinned chunks per `buffer_pool` slab |

## Row Group Pruning

When filter pushdown is enabled and the `gpu_expression_translator` successfully converts DuckDB `TableFilterSet` filters into a cuDF AST, two optimizations activate:

1. **Row group statistics pruning:** `parquet_split_provider::run_batch` calls `filter_row_groups_with_stats()` on each fetched footer; row groups whose Parquet column min/max statistics cannot match the filter are dropped before any read is scheduled. Pure hive-partition filters are dropped during plan construction since hive columns aren't in the parquet file.

2. **Reader-level filter pushdown:** The cuDF AST is set on `parquet_reader_options` via `set_filter()`, so cuDF applies the filter inside `read_parquet`. The `TABLE_SCAN` operator is set to passthrough (`passthrough = true`) since filtering is already done by the reader.

If AST translation fails (e.g., unsupported expression types), `GPU_PARQUET_SCAN`'s `execute()` runs the cached DuckDB filter expression through `gpu_expression_executor` on the decoded batch.

**Filter translation path:** `TableFilterSet` → `convert_table_filters_to_expression()` (skips `OPTIONAL_FILTER`, `IS_NOT_NULL`, and partition-column filters) → `gpu_expression_translator` → cuDF AST tree.

## Batch Coalescing

When many small files each produce a tiny GPU batch, per-task scheduling and kernel-launch overhead dominates. Two mechanisms address this depending on the scan path:

1. **Multifile bundling in `parquet_split_provider`** (GPU_PARQUET_SCAN): a single split may bundle row-group slices from multiple parquet files when those files share the same hive-partition values, up to `approximate_batch_size` uncompressed bytes. The downstream `cudf::io::read_parquet` call reads the bundled slices in one invocation.

2. **Post-read coalescing in `sirius_physical_table_scan`** (DUCKDB_SCAN): `get_next_task_input_data()` pops batches in a loop until `accumulated_bytes >= scan_task_batch_size` OR `batch_count >= 32`, returning a `pipelineable_operator_data` wrapping the accumulated batches. `execute()` then calls `cudf::concatenate()` before filtering/projecting.

When the GPU parquet scan applies filter+projection via the cuDF reader (passthrough mode), `TABLE_SCAN` skips concatenation — only the DuckDB-source code path goes through post-read coalescing.

## Iceberg Scan

**File:** `src/include/op/sirius_physical_iceberg_scan.hpp`, `src/op/scan/iceberg_scan_task.cpp`

`sirius_physical_iceberg_scan` inherits from `sirius_physical_parquet_scan` and adds support for Iceberg V1, V2, and V3 tables.

### Supported Iceberg Features

| Version | Feature | Implementation |
|---------|---------|---------------|
| V1 | Append-only (no deletes) | Identical to plain parquet scan |
| V2 | Positional deletes | `positional_delete_filter`: binary-searches sorted row positions, builds boolean mask, applies `cudf::apply_boolean_mask` |
| V2 | Equality deletes (heterogeneous) | A `vector<EqualityDeleteGroup>` carries one `cudf::distinct_hash_join` per distinct key schema; each group is sequence-scoped so it only applies to data files written before its sequence number |
| V3 | Deletion vectors | Read from PUFFIN files via `puffin_reader`; the resulting bitmap drives the same boolean-mask filter as positional deletes |
| V2/V3 | Schema evolution | Per-file projection detects which columns are present in each parquet file; missing columns are injected as typed NULL columns post-read |
| V2/V3 | Snapshot time-travel | `snapshot_from_id` is forwarded to DuckDB's `iceberg_metadata()` so the manifest matches the requested snapshot |
| V2/V3 | Partition evolution | Per-file inject function decides whether each column comes from parquet data, from the file path (hive-style), or is NULL — handles tables whose partition scheme changed across snapshots |

### Architecture

- `iceberg_scan_task_global_state` inherits from `parquet_scan_task_global_state`. Its `build_delete_pipeline()` reads delete files (manifest discovery delegates to DuckDB's `iceberg_metadata()`; the custom Avro reader in `iceberg_avro_reader.cpp` is the fallback for V3 deletion-vector PUFFIN files) and installs a composed `iceberg_delete_pipeline` as a `post_convert_fn_t` hook.
- The `post_convert_fn_t` hook fires after each row-group batch is decompressed to a `cudf::table`, applying all delete filters in-place with zero `cudaMemcpy D2H` in the hot path.
- Equality-delete key columns not in the user's projection are force-projected at read time, then stripped via zero-copy `release()` + truncate after all filters run, so downstream operators see only the requested columns.
- Equality deletes apply only to data files whose sequence number is less than the delete group's, mirroring Iceberg's snapshot semantics.

## Complete Scan Flow

```mermaid
graph TD
    TS[sirius_physical_table_scan] -->|"convert"| GPS[GPU_PARQUET_SCAN]
    TS -->|"or"| DS[DUCKDB_SCAN]
    TS -->|"or"| IS[ICEBERG_SCAN]

    SM[sirius_scan_manager] -->|"build provider"| PSP[parquet_split_provider]
    SM -->|"on pinned-table hit"| CSP[cached_split_provider]

    PSP -->|"push split"| SC["split_connector<br/>(per operator)"]
    CSP -->|"push split"| SC
    SC -->|"get_next_split()"| GPS

    DS -->|"task_creator.schedule()"| TC[task_creator]
    IS -->|"task_creator.schedule()"| TC
    GPS -->|"task_creator.schedule()"| TC

    TC -->|"create task"| DST[duckdb_scan_task]
    TC -->|"create task"| GPT["gpu_pipeline_task<br/>(reads parquet_scan_data)"]
    TC -->|"create task"| IST["iceberg_scan_task<br/>(parquet + delete filters)"]

    DST -->|"dispatch"| SE[duckdb_scan_executor]
    IST -->|"dispatch"| SE

    SE -->|"execute on worker"| DST2[DuckDB table function → column builders → host_data_representation]
    SE -->|"execute on worker"| IST2["Parquet reads → GPU delete filters → host_parquet_representation"]
    GPT -->|"execute on GPU executor"| GPS2["cudf::io::read_parquet → scan_plan assembly → gpu_table_representation"]

    DST2 -->|"publish"| DR[data_repository]
    IST2 -->|"publish"| DR
    GPS2 -->|"to next operator"| DR

    DR -->|"consumed by"| GPT2[gpu_pipeline_task]
```

## Key Files

| File | Purpose |
|------|---------|
| `src/op/scan/duckdb_scan_task.cpp` | DuckDB scan implementation |
| `src/include/op/scan/duckdb_scan_task.hpp` | DuckDB scan task, column builders |
| `src/op/scan/parquet_scan_task.cpp` | Legacy parquet scan implementation |
| `src/include/op/scan/parquet_scan_task.hpp` | Legacy parquet scan task, row group partitioning |
| `src/op/scan/duckdb_scan_executor.cpp` | Scan executor manager loop |
| `src/include/op/scan/duckdb_scan_executor.hpp` | Scan executor interface |
| `src/op/scan/prefetched_data_source.cpp` | Legacy cached datasource for cuDF |
| `src/include/op/scan/prefetched_data_source.hpp` | Legacy cached datasource interface |
| `src/op/scan/cached_ranges.cpp` | Byte range coalescing and lookup |
| `src/include/op/scan/cached_ranges.hpp` | Cache range structure |
| `src/include/op/scan/config.hpp` | Scan config, cache_level enum |
| `src/include/op/scan/sirius_gpu_parquet_scan_operator.hpp` | GPU parquet scan source operator |
| `src/include/op/scan/parquet_scan_info.hpp` | Bind data parked on the scan operator for the scan manager |
| `src/include/op/scan/parquet_scan_operator_data.hpp` | `parquet_scan_data` split type emitted by providers |
| `src/include/op/scan/scan_plan.hpp` | Index-space mapping (P/C/D), output layout, partition injection |
| `src/include/op/scan/parquet_schema_mapping.hpp` | Name-based DuckDB→parquet column resolution |
| `src/include/scan_manager/sirius_scan_manager.hpp` | Scan manager: thread pool, providers, pinned-table registry |
| `src/include/scan_manager/split_provider.hpp` | Abstract split producer |
| `src/include/scan_manager/parquet_split_provider.hpp` | Parquet metadata-driven split provider |
| `src/include/scan_manager/cached_split_provider.hpp` | Pinned-column split provider |
| `src/include/scan_manager/split_connector.hpp` | Blocking queue between provider and operator |
| `src/include/pin_table.hpp` / `src/pin_table.cpp` | `pin_table` / `unpin_table` table-function bindings |
| `src/include/op/sirius_physical_iceberg_scan.hpp` | Iceberg scan operator |
| `src/include/op/scan/iceberg_scan_task.hpp` | Iceberg scan task with delete filters |
| `src/include/op/scan/iceberg_metadata_reader.hpp` | Iceberg manifest reader (DuckDB `iceberg_metadata()` + Avro fallback) |
| `src/include/op/scan/puffin_reader.hpp` | V3 deletion-vector PUFFIN reader |
| `src/op/scan/scan_utils.cpp` | Row group pruning, filter expression conversion |
| `src/include/data/cached_data_representation.hpp` | Cached data wrappers |
