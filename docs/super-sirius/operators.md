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

Base scan operator wrapping a DuckDB table function. Stores column IDs, projection IDs, and optional table filters for predicate pushdown. It exists only as the plan-time carrier: during plan generation it is rewritten into a `GPU_SCAN` source (see below).

### `sirius_gpu_scan_operator` — `GPU_SCAN`
**File:** `src/include/op/scan/sirius_gpu_scan_operator.hpp`

Unified GPU scan source operator for reading table data from storage. It carries no format-specific code: it pulls pre-built splits off a `split_connector` and delegates per-split materialization to an installed `gpu_ingestible`, one implementation per source format (`parquet_gpu_ingestible` for Parquet, `duckdb_native_gpu_ingestible` for DuckDB-native `.duckdb` tables, `cached_parquet_gpu_ingestible` for pinned-cache hits).

The pipeline converter rewrites a DuckDB parquet or DuckDB-native table scan into a `GPU_SCAN` source: it lowers the bind data into the appropriate `ingestible_table_info`, builds the `gpu_ingestible`, and inserts the operator at `operators[0]` of the pipeline. Before a query runs, `sirius_scan_manager` prepares scan-side state — matching pinned-cache entries or building a `split_provider` over each operator's ingestible — and drives metadata production, split coalescing, and per-GPU balancing, pushing splits onto each operator's `split_connector`. `execute()` calls `gpu_ingestible::materialize_table` and, when a split carries filter/projection info, `gpu_ingestible::post_filter_and_project`.

See [Scan](scan.md) for the full scan subsystem (scan manager, `gpu_ingestible`, pinned-table caching, and the IO layer).

### `sirius_physical_streaming_source` — `STREAMING_SOURCE`
**File:** `src/include/op/sirius_physical_streaming_source.hpp`

Source operator that marks the bottom boundary of an intermediate pipeline fragment. It pulls
`exchange_batch_handle` records (batch-id + size) from a bounded `exec::exchange_channel`, resolves
each handle via a `cucascade::shared_data_repository`, and publishes the batch into the pipeline
as a `pipelineable_operator_data`. Used only when a fragment's input arrives from another node
over exchange; a leaf fragment keeps its normal `GPU_SCAN` source.

Key design invariants:
- The channel carries **handles**, not `shared_ptr`s — the repository owns the batch so queued
  items remain spill-visible to the downgrade executor.
- Engine workers use `try_pop` only (non-blocking); `push`/`pop` are provided for the wrapper/test side.
- EOS is **close-then-drain**: `close()` forbids new pushes; queued handles stay poppable;
  `drained()` (= `closed() && empty()`) is the terminal predicate.
- `execute()` is a pure pass-through (COLUMN_DATA_SCAN shape — no GPU work).
- `no_history_peak_memory_estimate()` returns `stats.bytes` (no extra allocation).

Hint table:

| Channel state | `get_next_task_hint()` |
|---|---|
| non-empty (open or closed) | `READY{this}` |
| open, empty | `WAITING{nullptr}` — re-armable by the session on push (#839) |
| closed && drained | `std::nullopt` — EOS |

`all_ports_empty()` is overridden to `_input_channel->drained()`, driving both the task-creation
loop guard and the port-less source pipeline-finish predicate.

Channel close notifies the pipeline (`update_pipeline_status(false)`, via a weak pipeline
reference wired in `set_pipeline`), so an empty or late-closed stream still finishes its
pipeline — and re-arms downstream consumers — even when no task is left in flight.

**Producer contract**: register the incoming batch in the input repository (`add_data_batch`) *first*,
then push the handle. The session (#839) owns edge-triggered re-scheduling; the plan generator (#838)
owns channel wiring.

**Backpressure (open integration requirement for #839)**: `try_pop()` frees channel item/byte
capacity at task-creation time, but the popped batches move into the unbounded task-scheduler
queue — the channel bound therefore does not bound total outstanding data. When #839 wires the
session, task creation must be gated on in-flight work (e.g. counting via the channel's `on_pop`
hook and the task-completion path) so a fast producer cannot accumulate an arbitrarily large GPU
backlog behind a nominally bounded channel.

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

- **GPU execution:** `expression_evaluator::select(batch)` — evaluates the boolean expression and compacts rows using cuDF filtering
- **Key members:** `expression` (filter predicate)

### `sirius_physical_projection` — `PROJECTION`
**File:** `src/include/op/sirius_physical_projection.hpp`

Evaluates a list of expressions to produce output columns.

- **GPU execution:** the operator classifies each `select_list` entry as either a pure column passthrough (a `sirius::ast::reference` / BOUND_REF) or an expression that must be evaluated, then takes one of three paths per input batch:
  - **All evaluated:** `expression_evaluator::evaluate()` produces an owned `cudf::table` of new columns.
  - **All passthrough:** the output is a zero-copy `cudf::table_view` over the input columns. The output batch is a view-backed `gpu_table_representation` (see [data management](data-management.md)) whose owner is the input's `read_only_data_batch` lock, which keeps the source columns alive and read-only-pinned for the output's lifetime — no device copies.
  - **Mixed:** only the non-passthrough entries are evaluated; the output view mixes the freshly-evaluated columns with the input's passthrough columns, owned jointly by the evaluated table and the input lock.

  Only the entries that need evaluation are passed to the expression evaluator (via its `std::vector<sirius::ast::node const*>` constructor). See [expression evaluator](expression-executor.md).
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
| `BUILD_PROBE` | Single partition, small build side (< `max_build_hash_table_bytes`) foldable to one batch | `cudf::hash_join`, `cudf::distinct_hash_join`, or `cudf::filtered_join` — built once, probed many times |
| `MIXED_JOIN` | Equality + inequality conditions on disjoint columns | `cudf::mixed_join()` with cuDF AST |

`update_join_exec_mode()` selects BUILD_PROBE when there is one partition, the build side fits and folds to a single batch, and the join type is not SEMI, ANTI, or RIGHT (these stay in STANDARD mode). INNER, LEFT, OUTER, and MARK joins are all eligible.

#### MARK joins
A MARK join emits every left row plus a `BOOL8` mark column indicating whether each left row had a match. Both build strategies funnel through `resolve_mark_join_result`, which scatters left-row match indices into the mark column.

- **STANDARD mode (adaptive build side):** by default the filter (right) side is built into a `cudf::filtered_join` and probed with the left, whose `semi_join` yields left-row match indices. When the right (probe) side is much larger than the left (output) side, Sirius instead builds the smaller left side into a `cudf::mark_join` and probes with the right. The switch is gated by `mark_join_build_switch_ratio` (build on left when `right_rows >= ratio * left_rows`; `0` disables). Both paths produce identical output.
- **BUILD_PROBE mode:** a single `cudf::filtered_join` is built once on the right (filter) keys and persisted as `_filtered_table`; each streamed left probe batch calls `semi_join` against it, reusing the hash table across probes.

#### Distinct Hash Join Optimization
For INNER/LEFT joins in BUILD_PROBE mode, when the build-side keys are proven unique, Sirius uses `cudf::distinct_hash_join` instead of `cudf::hash_join`. This optimization applies when:
- Join type is INNER or LEFT
- Build-side keys are proven unique via logical plan analysis (`prove_unique_columns()` in `src/planner/sirius_plan_comparison_join.cpp`)

Uniqueness is detected by walking the DuckDB logical plan:
- **PRIMARY KEY** on `LogicalGet` (with column mapping through `projection_ids`)
- **GROUP BY** uniqueness on `LogicalAggregate`
- Propagates through `LogicalFilter`, `LogicalOrder`, `LogicalLimit`, `LogicalTopN`, and `LogicalProjection`

Only PRIMARY KEY is considered (not plain UNIQUE) due to NULL handling semantics with `null_equality::UNEQUAL`. IS NOT DISTINCT FROM joins are excluded since they require `null_equality::EQUAL`.

Build/probe state machine for BUILD_PROBE mode:
```mermaid
stateDiagram-v2
    direction LR
    NOT_BUILT --> SCHEDULING
    SCHEDULING --> SCHEDULED
    SCHEDULED --> BUILT
    BUILT --> DESTROYED
```

When `get_next_task_hint()` is called after the operator is already finished, it returns `std::nullopt` (no new tasks needed).

Key members:
- `conditions` — join predicates (equality and inequality)
- `join_type` — INNER, LEFT, RIGHT, OUTER, MARK
- `_hash_table` — cached `cudf::hash_join` (BUILD_PROBE mode)
- `_distinct_hash_table` — cached `cudf::distinct_hash_join`, used instead of `_hash_table` when build keys are proven unique
- `_filtered_table` — reusable build-on-right `cudf::filtered_join` for MARK joins in BUILD_PROBE mode
- `_build_table` — materialized build-side data batch
- `key_casts` — type alignment info for hash key matching
- `unique_build_keys` / `unique_probe_keys` — cardinality hints (used to select distinct vs standard hash join)
- `mark_join_build_switch_ratio` — threshold for adaptively building a STANDARD MARK join on the smaller (left) side

Supported join types: INNER, LEFT, RIGHT, OUTER, MARK via `cudf::inner_join()`, `cudf::left_join()`, `cudf::full_outer_join()`, `cudf::filtered_join`, and `cudf::mark_join`.

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

Combined ORDER + LIMIT: selects and sorts the top N rows.

- **GPU execution:** Two-step process: `cudf::top_k_order()` selects top-N row indices, then `cudf::sort_by_key()` sorts the gathered rows to ensure deterministic output ordering- **Key members:** `orders`, `limit`, `offset`, `dynamic_filter`

### `sirius_physical_ungrouped_aggregate` — `UNGROUPED_AGGREGATE`
**File:** `src/include/op/sirius_physical_ungrouped_aggregate.hpp`

Aggregate without GROUP BY (e.g., `SELECT COUNT(*), SUM(x) FROM t`).

- **GPU execution:** `gpu_aggregate_impl::local_ungrouped_aggregate()` using `cudf::reduce()`
- **Supported:** SUM, MIN, MAX, COUNT (of valid values), COUNT(*), AVG, FIRST
- **AVG handling:** Decomposed into SUM + COUNT and finalized on-device. `make_avg_column()` divides the single-row merged sum/count columns with `cudf::binary_operation` — DECIMAL output divides directly in fixed point to preserve precision, while non-DECIMAL output casts both operands to FLOAT64 and divides. This keeps AVG off the host `long double` path, avoiding both the device→host sync and the precision loss of decimal round-trips.
- **DECIMAL overflow handling:** DECIMAL SUM casts to a wider type before reduction — DECIMAL32→DECIMAL64, DECIMAL64→DECIMAL128 — to prevent overflow
- **BIGINT SUM fallback:** BIGINT (INT64) SUM falls back to CPU execution because GPU lacks INT128 accumulator support. Without this, silent overflow produces incorrect results. BIGINT arithmetic operations (ADD, SUB, MUL) also fall back to CPU for the same reason.

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
- **Sibling coordination:** Build-side partition normally determines the shared count. For RIGHT-family hash joins other than `RIGHT_DELIM_JOIN`, the retained probe side determines it instead.
- **Key members:** `_partition_keys`, `_partition_type`, `_num_partitions`, `_is_build`, `_drives_partition_count`, `_sibling_partition_op`

### `sirius_physical_concat` — `CONCAT`
**File:** `src/include/op/sirius_physical_concat.hpp`

Reassembles partitioned data back into a linear stream. Behavior depends on join type:

- `_concat_all = true` (LEFT/ANTI/OUTER joins): waits for all data before emitting
- `_concat_all = false` (INNER joins): emits tasks when byte threshold (`_concat_batch_bytes`) is met

### `sirius_physical_sort_sample` — `SORT_SAMPLE`
**File:** `src/include/op/sirius_physical_sort_sample.hpp`

Samples input batches to compute P-1 partition boundary rows for range partitioning. Sampling is byte-based: it accumulates batches until `sort_sample_bytes` worth of input is available, rather than a fixed batch count.

Boundary computation follows an explicit `BoundaryState` lifecycle: `NOT_DONE → SCHEDULED → DONE`.
- `get_next_task_hint()` waits in `NOT_DONE` until enough sample bytes are available (or the upstream finishes), then signals READY; it returns `std::nullopt` while `SCHEDULED` so at most one boundary task is in flight.
- `get_next_task_input_data()` (overridden) claims the accumulated sample batches and moves the state to `SCHEDULED`, handing them to `execute()` as a single multi-batch input.
- `execute()` merges the pre-sorted sample batches, computes the boundaries, and moves to `DONE`. If a GPU allocation throws (e.g. OOM), the state stays `SCHEDULED` and the rescheduled task retries with the same input, preventing a duplicate boundary task.
- Once `DONE`, the operator falls back to default scheduling and passes through remaining batches unchanged.

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
| GPU_SCAN | Scan | Unified GPU scan source served by `sirius_scan_manager` via a per-format `gpu_ingestible` |
| STREAMING_SOURCE | Scan | Exchange-input source; pulls batch handles from `exchange_channel`, resolves via `shared_data_repository` |
| DUMMY_SCAN | Scan | Generates 1 row |
| COLUMN_DATA_SCAN | Scan | Reads ColumnDataCollection |
| FILTER | Relational | `expression_evaluator::select()` |
| PROJECTION | Relational | `expression_evaluator::evaluate()` |
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
| HASH_JOIN | Join | `cudf::{inner,left,right,outer}_join()`, `cudf::distinct_hash_join`, or `cudf::{filtered,mark}_join` (MARK) |
| NESTED_LOOP_JOIN | Join | Fallback nested loops |
| LEFT_DELIM_JOIN | Join | Correlated subquery wrapper |
| RIGHT_DELIM_JOIN | Join | Correlated subquery wrapper |
| PARTITION | Pipeline | Hash/range partitioning |
| CONCAT | Pipeline | Partition reassembly |
| MERGE_TOP_N | Pipeline | Merge per-partition top-N |
| CTE | CTE | Materialize to ColumnDataCollection |
| RESULT_COLLECTOR | Result | Final result materialization |
| EMPTY_RESULT | Result | Empty result set |
