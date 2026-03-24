# Operators

This document covers all Super Sirius physical operators, organized by category.

## Base Class

**File:** `src/include/op/sirius_physical_operator.hpp`

`sirius_physical_operator` is the base class for every operator.

### Pipeline Model

After pipeline finalization (see [Physical Plan Generation — Pipeline Finalization](physical-plan-generation.md#pipeline-finalization)), a pipeline's `operators` list contains **all** operators from first to last. `source` and `sink` are aliases:
- `source` = `operators[0]` (first operator)
- `sink` = last operator in the list

During task execution:
1. `compute_task()` iterates over **every** operator in `operators`, calling `execute()` on each
2. `publish_output()` then calls `sink()` on the last operator to push results to downstream ports

An operator's position in a pipeline is determined by `sirius_engine::initialize_internal()`. Many blocking operators appear as both the source (first) of one pipeline and the sink (last) of another — they accumulate data as a sink, then emit results as a source. See the [Operator Summary Table](#operator-summary-table) for the full per-operator breakdown.

### Key Methods

| Method | Purpose |
|--------|---------|
| `execute(input_data, stream)` | Called on **every** operator during `compute_task()` |
| `sink(output_data, stream)` | Called on the **last** operator after `compute_task()` to push results downstream |
| `is_source()` | Whether this operator can produce data (has scan state or owns accumulated data) |
| `is_sink()` | Whether this operator has a `sink()` implementation for pushing data to downstream ports |
| `get_next_task_hint()` | Checks port readiness, returns `READY` or `WAITING_FOR_INPUT_DATA` |
| `get_next_task_input_data()` | Pops one data batch from each input port |
| `can_create_more_tasks()` / `has_processed_all_tasks()` | Signals task exhaustion |

See [Task Creator](task-creator.md) for per-operator overrides.

## Scan Operators

These operators produce data for pipelines. See [Scan](scan.md) for in-depth coverage.

### `sirius_physical_table_scan` — `TABLE_SCAN`
**File:** `src/include/op/sirius_physical_table_scan.hpp`

Base scan operator wrapping a DuckDB table function. Stores column IDs, projection IDs, and optional table filters for predicate pushdown. During pipeline construction, converted to either DUCKDB_SCAN or PARQUET_SCAN.

### `sirius_physical_duckdb_scan` — `DUCKDB_SCAN`
**File:** `src/include/op/sirius_physical_duckdb_scan.hpp`

Sequential scan using DuckDB's execution engine. Accumulates chunks into fixed-size batches via column builders. Tracks an atomic `exhausted` flag for pipeline completion.

### `sirius_physical_parquet_scan` — `PARQUET_SCAN`
**File:** `src/include/op/sirius_physical_parquet_scan.hpp`

Direct Parquet file scan. Reads column-chunk byte ranges and optionally materializes (decompresses) to table format. Tracks `has_more_partitions` atomic flag. Row groups are partitioned by `approximate_batch_size`.

### `sirius_physical_dummy_scan` — `DUMMY_SCAN`
**File:** `src/include/op/sirius_physical_dummy_scan.hpp`

Generates a single empty row for constant queries (e.g., `SELECT 1+2`).

### `sirius_physical_column_data_scan` — `COLUMN_DATA_SCAN` / `CTE_SCAN` / `DELIM_SCAN`
**File:** `src/include/op/sirius_physical_column_data_scan.hpp`

Scans a pre-materialized `ColumnDataCollection`. Used for CTE results, correlated subquery intermediates, and expression-generated data.

## Streaming Operators

These operators process data in a single pass without buffering.

### `sirius_physical_filter` — `FILTER`
**File:** `src/include/op/sirius_physical_filter.hpp`

Applies a predicate expression to filter rows.

- **GPU execution:** `GpuExpressionExecutor::select(batch, stream)` — evaluates the boolean expression and compacts rows using cuDF filtering
- **Key members:** `expression` (filter predicate)

### `sirius_physical_projection` — `PROJECTION`
**File:** `src/include/op/sirius_physical_projection.hpp`

Evaluates a list of expressions to produce output columns.

- **GPU execution:** `GpuExpressionExecutor::execute(batch, stream)` — evaluates each expression, producing a new table with projected columns
- **Key members:** `select_list` (output expressions)

### `sirius_physical_streaming_limit` — `STREAMING_LIMIT`
**File:** `src/include/op/sirius_physical_limit.hpp`

Implements LIMIT/OFFSET using atomic counters for parallel execution.

- **Key members:** `_remaining_offset` (atomic), `_remaining_limit` (atomic), `_limit_exhausted` (atomic)
- **Mechanism:** Each task atomically claims a portion of the remaining limit via `claim()`. When the limit is exhausted, the pipeline terminates early.

## Blocking Operators

These operators buffer input before producing output. They are both sinks and sources.

### `sirius_physical_hash_join` — `HASH_JOIN`
**File:** `src/include/op/sirius_physical_hash_join.hpp`, `src/op/sirius_physical_hash_join.cpp`

Three execution modes:

| Mode | When Used | cuDF API |
|------|-----------|----------|
| `STANDARD` | Default, multi-partition Cartesian product | `cudf::inner_join()`, `cudf::left_join()`, etc. |
| `BUILD_PROBE` | Small build side (< `max_build_hash_table_bytes`, 1 partition) | `cudf::hash_join` (build once, probe many) |
| `MIXED_JOIN` | Equality + inequality conditions on disjoint columns | `cudf::mixed_join()` with cuDF AST |

Build/probe state machine for BUILD_PROBE mode:
```mermaid
stateDiagram-v2
    direction LR
    NOT_BUILT --> SCHEDULING
    SCHEDULING --> SCHEDULED
    SCHEDULED --> BUILT
    BUILT --> DESTROYED
```

Key members:
- `conditions` — join predicates (equality and inequality)
- `join_type` — INNER, LEFT, RIGHT, OUTER
- `_hash_table` — cached `cudf::hash_join` (BUILD_PROBE mode)
- `_build_table` — materialized build-side data batch
- `key_casts` — type alignment info for hash key matching
- `unique_build_keys` / `unique_probe_keys` — cardinality hints

Supported join types: INNER, LEFT, RIGHT, OUTER via `cudf::inner_join()`, `cudf::left_join()`, `cudf::full_outer_join()`.

### `sirius_physical_nested_loop_join` — `NESTED_LOOP_JOIN`
**File:** `src/include/op/sirius_physical_nested_loop_join.hpp`

Fallback for joins not supported by cuDF hash join (pure inequality conditions). Uses `PhysicalNestedLoopJoin::IsSupported()` to validate.

### `sirius_physical_order` — `ORDER_BY`
**File:** `src/include/op/sirius_physical_order.hpp`

Local sort of each data batch.

- **GPU execution:** `gpu_order_impl::local_order_by()` using `cudf::order_by()`
- **Key members:** `orders` (sort keys with ASC/DESC and null ordering), `projections` (output columns), `is_index_sort`

### `sirius_physical_top_n` — `TOP_N`
**File:** `src/include/op/sirius_physical_top_n.hpp`

Combined ORDER + LIMIT: sorts and returns only the top N rows.

- **Key members:** `orders`, `limit`, `offset`, `dynamic_filter`

### `sirius_physical_ungrouped_aggregate` — `UNGROUPED_AGGREGATE`
**File:** `src/include/op/sirius_physical_ungrouped_aggregate.hpp`

Aggregate without GROUP BY (e.g., `SELECT COUNT(*), SUM(x) FROM t`).

- **GPU execution:** `gpu_aggregate_impl::local_ungrouped_aggregate()` using `cudf::reduce()`
- **Supported:** MIN, MAX, SUM, COUNT_ALL, COUNT_VALID

### `sirius_physical_grouped_aggregate` — `HASH_GROUP_BY`
**File:** `src/include/op/sirius_physical_grouped_aggregate.hpp`

Hash-based GROUP BY.

- **GPU execution:** `gpu_aggregate_impl::local_grouped_aggregate()` using `cudf::groupby()`
- **AVG handling:** Decomposed into SUM + COUNT_VALID via `AggregateSlot`
- **COUNT(DISTINCT):** Implemented via `COLLECT_SET` aggregation with struct column synthesis
- **Key members:** `group_idx`, `cudf_aggregates`, `cudf_aggregate_idx`, `aggregate_slots`, `has_avg`, `has_count_distinct`

## Pipeline Breakers (Sirius-Specific)

These operators are injected during pipeline splitting. They don't map to DuckDB logical operators.

### `sirius_physical_partition` — `PARTITION`
**File:** `src/include/op/sirius_physical_partition.hpp`

Repartitions data into N buckets based on partition keys.

- **Modes:** `HASH` (most common), `RANGE`, `EVENLY`, `CUSTOM`, `NONE`
- **Adaptive count:** `determine_num_partitions()` computes N from actual input data size and `hash_partition_bytes` config
- **Sibling coordination:** Build-side partition determines count; probe-side waits for the result
- **Key members:** `_partition_keys`, `_partition_type`, `_num_partitions`, `_is_build`, `_sibling_partition_op`

### `sirius_physical_concat` — `CONCAT`
**File:** `src/include/op/sirius_physical_concat.hpp`

Reassembles partitioned data back into a linear stream. Behavior depends on join type:

- `_concat_all = true` (LEFT/ANTI/OUTER joins): waits for all data before emitting
- `_concat_all = false` (INNER joins): emits tasks when byte threshold (`_concat_batch_bytes`) is met

### `sirius_physical_sort_sample` — `SORT_SAMPLE`
**File:** `src/include/op/sirius_physical_sort_sample.hpp`

Samples N input batches to compute P-1 partition boundary rows for range partitioning.

- On first execution: concatenate samples, sort, compute boundaries, set `_boundaries_computed`
- On subsequent executions: pass through data unchanged
- Custom `get_next_task_hint()`: waits for N batches before returning READY

### `sirius_physical_sort_partition` — `SORT_PARTITION`
**File:** `src/include/op/sirius_physical_sort_partition.hpp`

Range-partitions data according to boundaries computed by SORT_SAMPLE. Links to the sample operator via `_sample_op`.

### `sirius_physical_merge_sort` — `MERGE_SORT`
**File:** `src/include/op/sirius_physical_merge_sort.hpp`

Merges pre-sorted partitions using `gpu_merge_impl::merge_order_by()` (multi-way merge via cuDF).

- Custom `get_next_task_input_data()`: drains all batches from one partition per call
- Tracks `_current_partition_index` atomically under mutex

### `sirius_physical_grouped_aggregate_merge` — `MERGE_GROUP_BY`
**File:** `src/include/op/sirius_physical_grouped_aggregate_merge.hpp`

Merges grouped aggregate results from multiple partitions. Drains one partition per task, similar to MERGE_SORT.

### `sirius_physical_ungrouped_aggregate_merge` — `MERGE_AGGREGATE`
**File:** `src/include/op/sirius_physical_ungrouped_aggregate_merge.hpp`

Merges ungrouped aggregate results from multiple partitions.

### `sirius_physical_top_n_merge` — `MERGE_TOP_N`
**File:** `src/include/op/sirius_physical_top_n_merge.hpp`

Merges local top-N results from multiple partitions.

## CTE / Delim Join Operators

### `sirius_physical_cte` — `CTE`
**File:** `src/include/op/sirius_physical_cte.hpp`

Materializes Common Table Expression results into a `ColumnDataCollection` for later scanning by CTE_SCAN operators.

- **Key members:** `working_table`, `cte_scans`, `ctename`, `table_index`

### `sirius_physical_left_delim_join` — `LEFT_DELIM_JOIN`
### `sirius_physical_right_delim_join` — `RIGHT_DELIM_JOIN`
**File:** `src/include/op/sirius_physical_delim_join.hpp`

Handle correlated subqueries via duplicate elimination. Wrap an inner join (hash or nested loop) and embed a `sirius_physical_grouped_aggregate` for DISTINCT on duplicate-eliminated columns.

- `join` — the actual join operator
- `distinct` — embedded aggregate for duplicate elimination
- `delim_scans` — downstream scan operators that receive the deduplicated data

### `sirius_physical_partition_consumer_operator`
**File:** `src/include/op/sirius_physical_partition_consumer_operator.hpp`

Base interface for operators that consume partitioned data. Provides `push_data_batch_partitioned(port_id, batch, partition_idx)`.

## Result Operators

### `sirius_physical_result_collector` / `sirius_physical_materialized_collector` — `RESULT_COLLECTOR`
**File:** `src/include/op/sirius_physical_result_collector.hpp`

Final sink that materializes query results into a `ColumnDataCollection`. The GPU executor checks for this operator type to determine query completion.

### `sirius_physical_empty_result` — `EMPTY_RESULT`
**File:** `src/include/op/sirius_physical_empty_result.hpp`

Returns an empty result set for queries with contradicted filters.

## Operator Summary Table

After pipeline finalization, `source` and `sink` are just aliases for the first and last operator in the `operators` list. All operators have `execute()` called during `compute_task()`; only the last operator additionally has `sink()` called via `publish_output()`.

| Operator | Category | GPU Method |
|----------|----------|-----------|
| DUCKDB_SCAN | Scan | DuckDB table function |
| PARQUET_SCAN | Scan | Direct Parquet reading |
| DUMMY_SCAN | Scan | Generates 1 row |
| COLUMN_DATA_SCAN | Scan | Reads ColumnDataCollection |
| FILTER | Relational | `GpuExpressionExecutor::select()` |
| PROJECTION | Relational | `GpuExpressionExecutor::execute()` |
| STREAMING_LIMIT | Relational | Atomic claim-based |
| ORDER_BY | Sort | `gpu_order_impl::local_order_by()` |
| TOP_N | Sort | Order + limit |
| SORT_SAMPLE | Sort | Sample + boundary computation |
| SORT_PARTITION | Sort | Range partition by boundaries |
| MERGE_SORT | Sort | `gpu_merge_impl::merge_order_by()` |
| UNGROUPED_AGGREGATE | Agg | `gpu_aggregate_impl::local_ungrouped_aggregate()` |
| HASH_GROUP_BY | Agg | `gpu_aggregate_impl::local_grouped_aggregate()` |
| MERGE_AGGREGATE | Agg | Merge ungrouped partitions |
| MERGE_GROUP_BY | Agg | Merge grouped partitions |
| HASH_JOIN | Join | `cudf::{inner,left,right,outer}_join()` |
| NESTED_LOOP_JOIN | Join | Fallback nested loops |
| LEFT_DELIM_JOIN | Join | Correlated subquery wrapper |
| RIGHT_DELIM_JOIN | Join | Correlated subquery wrapper |
| PARTITION | Pipeline | Hash/range partitioning |
| CONCAT | Pipeline | Partition reassembly |
| MERGE_TOP_N | Pipeline | Merge per-partition top-N |
| CTE | CTE | Materialize to ColumnDataCollection |
| RESULT_COLLECTOR | Result | Final result materialization |
| EMPTY_RESULT | Result | Empty result set |
