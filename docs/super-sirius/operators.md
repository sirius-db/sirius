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
| `is_sink()` | Whether this operator has a `sink()` implementation for pushing data to downstream ports — true for unconditional sinks, or when the tree parent is a sink parent (PARTITION, RIGHT_DELIM_JOIN, DENSE_COUNT_JOIN) |
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

Unified GPU scan source operator for reading table data from storage. It carries no format-specific code: it pulls pre-built splits off a `split_connector` and delegates per-split materialization to an installed `gpu_ingestible`, one implementation per source format (`parquet_gpu_ingestible` for Parquet, `duckdb_native_gpu_ingestible` for DuckDB-native `.duckdb` tables); pinned-cache hits are served by the scan manager's `cached_databatch_provider`.

The pipeline converter rewrites a DuckDB parquet or DuckDB-native table scan into a `GPU_SCAN` source: it lowers the bind data into the appropriate `ingestible_table_info`, builds the `gpu_ingestible`, and inserts the operator at `operators[0]` of the pipeline. Before a query runs, `sirius_scan_manager` prepares scan-side state — matching pinned-cache entries or building a `split_provider` over each operator's ingestible — and drives metadata production, split coalescing, and per-GPU balancing, pushing splits onto each operator's `split_connector`. `execute()` calls `gpu_ingestible::materialize_table` and, when a split carries filter/projection info, `gpu_ingestible::post_filter_and_project`.

See [Scan](scan.md) for the full scan subsystem (scan manager, `gpu_ingestible`, pinned-table caching, and the IO layer).

### `sirius_physical_streaming_source` — `STREAMING_SOURCE`
**File:** `src/include/op/sirius_physical_streaming_source.hpp`

Source operator that marks the bottom boundary of an intermediate pipeline fragment. Producers
call `push(batch)` / `close_input(sender_id)`; the operator publishes each queued
`cucascade::data_batch` into the pipeline as a `pipelineable_operator_data`, one per task. Used
only when a fragment's input arrives from another node over exchange; a leaf fragment keeps its
normal `GPU_SCAN` source.

Unlike a `GPU_SCAN` it does not own its input — batches arrive at runtime — so an empty queue means
*wait*, not *exhausted*. It holds one `exec::batch_stream`: a borrowed
`cucascade::shared_data_repository` (**the queue** — batches sit there until a task claims one,
spill-visible to the downgrade executor when the caller registered that repository with the memory
manager) bound to the stream state (sender-aware end-of-stream, the availability classification,
the `on_data` notification, and the producer-error plane).

Key design invariants (full S1–S5 / P1–P4 in [Streaming Sessions](streaming-sessions.md#contracts-s1-s5)):
- Batches cross natively, in their current tier — no Arrow, no forced GPU upgrade.
- EOS is **sender-aware**: `close_input(sender)` is idempotent per sender, and the stream ends
  only once every *expected* sender has closed. An unexpected sender id is a defined error.
- **S1** — every batch lands in the repository before `on_data` announces it; `push()` returns
  false once the stream is terminal.
- **S2** — `fail_input` wakes a parked consumer (on_data / P4 for the engine path) for the rethrow.
- **S3** — an errored stream never reports `END_OF_STREAM` / `drained()`; exit is the rethrow.
- **S4** — `try_pull` / task input rethrows before popping queued batches.
- **S5** — `wait` then pull is not atomic; blocking loops must re-check.
- `execute()` is a pure pass-through (COLUMN_DATA_SCAN shape — no GPU work).
- `no_history_peak_memory_estimate()` returns `stats.bytes` (no extra allocation).

Hint state machine (contrast with `GPU_SCAN`: `READY{this} → drain blocking condvar → nullopt`):

```
WAITING{nullptr} ◄──── classify()=WAITING (open, empty)
      │                      ▲
      │ request dropped       │ try_pull() drains; stream still open
      │                      │
      └──► push() fires on_data ──► creator->schedule(head) ──► READY{this}
                                                                      │
                                              classify()=HAS_DATA ───┘
                                              get_next_task_input_data() pops one batch
                                              (or rethrows on error — S4)

classify()=END_OF_STREAM ──► nullopt ──► on_end_of_stream ──► pipeline finishes
```

| Stream state | `get_next_task_hint()` |
|---|---|
| `HAS_DATA` (queued or errored, open or ended) | `READY{this}` |
| `WAITING` (open, empty) | `WAITING{nullptr}` — the next `push()` re-nominates the head |
| `END_OF_STREAM` | `std::nullopt` |

`all_ports_empty()` is `stream.drained()` — a clean end only: every sender closed, queue empty, no
pending error. It drives both the task-creation loop guard and the port-less source
pipeline-finish predicate. Emptiness is evaluated under the stream's own lock — otherwise a batch
admitted between the check and the lock could be reported as end-of-stream and dropped.

**The live wake-up.** The head is scheduled once by `start_query()` and task completion only
nominates downstream consumers, so `on_data` — fired by every successful `push()` — is the only
thing that brings a dropped source back. It is deliberately not one-shot; `batch_stream::set_on_data`
explains why.

End-of-stream separately notifies the pipeline (`update_pipeline_status(false)`, via a weak
pipeline reference wired in `set_pipeline`), so an empty or late-closed stream still finishes its
pipeline — and re-arms downstream consumers — even when no task is left in flight.

**No channel-level backpressure.** Producers push into the repository and the downgrade executor
relieves memory pressure. Sirius has no upward "stop producing" signal today (hints are only
`READY` / `WAITING` / nothing), so the intended lever is per-fragment priority, not a bounded
queue.

### `sirius_physical_streaming_sink` — `STREAMING_SINK`
**File:** `src/include/op/sirius_physical_streaming_sink.hpp`

Terminal operator of a streaming fragment: every batch the pipeline produces is pushed into one
`exec::batch_stream` and exposed to an external consumer via `pull()` / `wait()` / `drained()`.

Where the source stands where no table exists, the sink stands where no `RESULT_COLLECTOR` exists:
the fragment's output travels through an `exec::batch_stream` over a caller-supplied output
repository rather than into a query result. It is shaped
like `RESULT_COLLECTOR` (appended to `current` in `build_pipelines`, set as the meta-pipeline sink)
so the executor places it at `operators[last]` and drives its `on_finalize_operator()` — which is
the only route to end-of-stream for the output stream.

Lifecycle:

```
sink(batch) ──► batch_stream[i].push(batch)     (per task, per destination partition)
      │
on_finalize_operator()                           (pipeline finish — the only EOS path)
      │
batch_stream[i].close(PIPELINE_SENDER)           (for every output stream i)
      │
consumers see END_OF_STREAM via wait(i) / drained(i)
```

Key design facts:
- **One sender, one finalize.** The pipeline feeding this sink is its single expected sender
  (`PIPELINE_SENDER = 0`). `on_finalize_operator()` calls `close(PIPELINE_SENDER)` on the output
  `exec::batch_stream`, which is what makes it terminal. Without `build_pipelines` placing the sink in
  `operators`, `on_finalize_operator()` is never called and every consumer blocked in `wait()`
  hangs forever with no error visible.
- **No self-nomination.** Unlike the source, the sink does not wire an `on_data` hook: its consumer
  is an external thread blocking in `wait()`, not an engine task needing re-nomination.
- **Native tier.** Batches are pushed in whatever tier they arrived — no Arrow, no forced GPU
  upgrade. A queued batch stays spillable in the repository until pulled.
- **execute() is a pass-through** (same shape as `RESULT_COLLECTOR`): it hands the batches back so
  `publish_output()` can deliver them to `sink()`. The base implementation drops them.
- **Partitioned variant (N destinations).** The second constructor takes N repositories and a
  `partition_spec` (`mode`, key columns + optional casts, validated at construction).
  `partition_mode::hash` GPU-hash-partitions each input batch and routes slice *i* into
  `_outputs[i]`, skipping empty slices. `partition_mode::broadcast` replicates every batch to all
  N outputs: output 0 gets the original handle, outputs 1..N−1 get independent `clone()`s with
  fresh `batch_id`s. Each destination has its own `exec::batch_stream` and EOS reaches all of them
  on a single `on_finalize_operator()` call. When N = 1 the spec is ignored and a native push
  bypasses routing entirely. `no_history_peak_memory_estimate()` stays 0 for N = 1 and becomes 2×
  the input for N > 1: hash_partition holds a reordered copy + slices; broadcast holds the
  original + N−1 clones simultaneously.

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
| `BUILD_PROBE` | Up to one partition per GPU, per-partition build side (< `max_build_hash_table_bytes`) foldable to one batch | `cudf::hash_join`, `cudf::distinct_hash_join`, or `cudf::filtered_join` — built once per partition, probed many times |
| `MIXED_JOIN` | Equality + inequality conditions on disjoint columns | `cudf::mixed_join()` with cuDF AST |

Partition count, broadcast, and BUILD_PROBE are decided together by the free function `compute_hash_join_partition_strategy()` (member wrapper: `get_partition_strategy()`), which returns a single `partition_strategy {num_partitions, broadcast, build_probe}`. BUILD_PROBE is selected when `num_partitions <= num_gpus` (one hash table per partition, at most one partition per GPU — this reduces to the historical single-partition rule when `num_gpus == 1`), the per-GPU build side fits `max_build_hash_table_bytes` and folds to a single batch, and the join is not RIGHT-family (`RIGHT`, `RIGHT_SEMI`, `RIGHT_ANTI`), `MIXED_JOIN`, or full `OUTER`. INNER, LEFT, MARK, SEMI, and ANTI joins are eligible (SEMI/ANTI/MARK build a persistent `cudf::filtered_join` on the right and stream left probe batches); MARK force-enables BUILD_PROBE even above the hash-table budget, since it has no STANDARD path. Full outer is excluded because BUILD_PROBE streams probe batches and calls `full_join` per batch, which would re-emit unmatched build rows on every batch (and, under broadcast/partitioning, on every GPU) with no global accumulation — full outer joins use the STANDARD path. For a broadcast join the **full** replicated build size is charged against `max_build_hash_table_bytes` (each GPU builds the entire table); a hash-partitioned build charges the per-partition average.

By execution time every equality-condition side is a plain column reference: a complex equality-key expression is materialized into a real column by a planner-inserted projection below the join (`materialize_expression_join_keys()`), because PARTITION — the first key consumer — hashes by column index and cannot evaluate expressions. Inequality sides stay inline as the mixed-join AST predicate.

**NULL comparison for keys.** `compare_nulls()` picks the `null_equality` flag threaded through every cuDF join call (including the build helpers and `filtered_join`/`mark_join`): `EQUAL` only when *every* equi-key is `IS NOT DISTINCT FROM`, `UNEQUAL` otherwise (any plain `=`, delim joins, and MARK). A null-safe key *mixed* with plain `=` is instead routed through MIXED_JOIN (see below).

**Broadcast small build tables.** On multi-GPU, a build side is a broadcast candidate when it is small (`< small_table_bytes`), or when it fits `max_broadcast_join_size` and the estimated probe-to-build ratio is at least `num_gpus * 1.25`; MARK joins are forced broadcast whenever `num_gpus > 1` (see [MARK joins](#mark-joins)). Instead of routing the whole build to one GPU, the PARTITION operator proposes `num_gpus` partitions and *replicates* the small build table to every GPU (the `_broadcast` flag), so each GPU builds its own hash table and joins its local probe rows. The build sink deposits the build batch into every slot; the probe sink routes each batch to the slot for its current GPU (`slot_for_device`). Once the probe side finishes, any slot that received replicated build data but no probe rows is discarded (`discard_build_only_slots_if_probe_complete`). Right-family / mixed joins reject BUILD_PROBE and fall back to the normal partition count.

#### Partial-barrier scheduling (STANDARD / MIXED_JOIN)

The join's own `build` and `default`/probe input ports are **PARTIAL** barriers (owned by the HASH_JOIN `input_port_for` / `input_barrier_for` hooks for `CONCAT → HASH_JOIN` edges). Batches arrive progressively on each side, and `get_next_task_hint` / `get_next_task_input_data` schedule per-partition **build × probe** cross-product pairs as they become available (state tracked in `partition_cross_schedule`, guarded by `op_state_mutex`), freeing each batch once it has been paired with every batch of a finished opposite side. This mirrors how BUILD_PROBE already streams its probe side; BUILD_PROBE is unaffected by the port barrier because it overrides its hint regardless.

Joining each build batch against each probe batch and unioning the results is only *inherently* correct for INNER. For every other join type, `sirius_physical_concat` folds the side that must be seen whole into a **single batch** (an implicit full barrier), so the streamed side is joined against that one folded batch — correct for all types. `refresh_cross_schedule` asserts this "whole side stays one batch" invariant and throws if a concat regression ever violates it.

**Empty-opposite side (orphan tasks).** When one side is a genuinely empty table, its concat emits *zero* batches. The surviving side's batches then have nothing to pair with, so once both producers finish, `next_cross_schedule_orphan` claims each survivor and `get_next_task_input_data` **synthesizes an empty opposite batch** for it (`sirius::make_empty_table` from the absent child's output types, built on the survivor's device) and returns an ordinary two-batch task. `execute()` then runs the normal join dispatch unchanged, so cuDF NULL-pads the survivor for LEFT/RIGHT/OUTER/ANTI and yields zero rows for INNER/SEMI. Popping the survivor also drains the repository, so the pipeline completes instead of hanging. FULL OUTER and RIGHT-family joins are the reachable cases (they always run STANDARD); small-build INNER/LEFT/SEMI/ANTI take BUILD_PROBE, which handles an empty side separately.

The following table summarizes, per join type, what concat folds, which side streams under this scheme, and the partition / broadcast / mode support:

| JoinType | concat fold (whole side) | Streams multi-batch | batch×batch + union | Multi-partition | Broadcast | Supported modes |
|---|---|---|---|---|---|---|
| INNER | neither | both sides | ✅ inherent | ✅ natural count | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| LEFT | build → 1 | probe | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| SEMI | build → 1 | probe | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| ANTI | build → 1 | probe | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| MARK | build resident (via BUILD_PROBE; `_concat_all = false`) | probe (BUILD_PROBE) | n/a — forced BUILD_PROBE | ❌ clamped to 1 (single-GPU) | ✅ forced on multi-GPU | BUILD_PROBE only |
| OUTER (FULL) | both → 1 | neither (stays 1×1) | ✅ degenerate | ✅ natural count | ❌ | STANDARD, MIXED_JOIN† |
| RIGHT | probe → 1 | build | ✅ | ✅ (probe-driven) | ❌ | STANDARD, MIXED_JOIN† |
| RIGHT_SEMI | probe → 1 | build | ✅ | ✅ | ❌ | STANDARD, MIXED_JOIN† |
| RIGHT_ANTI | probe → 1 | build | ✅ | ✅ | ❌ | STANDARD, MIXED_JOIN† |
| SINGLE | — | — | — | — | — | unsupported (throws) |

† MIXED_JOIN applies to any of these types when the join carries both equality **and** inequality conditions (MARK + MIXED is rejected at construction). "Streams multi-batch" describes the STANDARD/MIXED partial-barrier behavior; BUILD_PROBE already streamed its probe side. The whole-side fold is chosen in the `sirius_physical_concat` constructor from the downstream join type (`_concat_all`); partition / broadcast / mode eligibility is decided in `compute_hash_join_partition_strategy`.

#### MARK joins
A MARK join emits every left row plus a `BOOL8` mark column with SQL three-valued semantics: **true** for a match, **false** for a non-match when the probe key is non-NULL and the build side has no NULL key, and **NULL** otherwise (NULL probe key, or no match while a NULL build key exists — the row *might* have matched it). Both build strategies funnel through `resolve_mark_join_result`, which scatters left-row match indices into the mark column and attaches the null mask: when the build side has a NULL key, validity equals the match flags (`cudf::bools_to_mask`); otherwise validity is the probe keys' row validity (`cudf::bitmask_and`). Build-key nullness is recorded at build time in `_build_has_null` — an atomic that is join-wide because MARK is forced single-partition (or broadcast on multi-GPU, precisely so `_build_has_null` is globally consistent).

- **STANDARD mode (adaptive build side):** by default the filter (right) side is built into a `cudf::filtered_join` and probed with the left, whose `semi_join` yields left-row match indices. When the right (probe) side is much larger than the left (output) side, Sirius instead builds the smaller left side into a `cudf::mark_join` and probes with the right. The switch is gated by `mark_join_build_switch_ratio` (build on left when `right_rows >= ratio * left_rows`; `0` disables). Both paths produce identical output.
- **BUILD_PROBE mode:** a `cudf::filtered_join` is built once per partition slot on the right (filter) keys and persisted in that slot's build state; each streamed left probe batch calls `semi_join` against it, reusing the hash table across probes.

#### Distinct Hash Join Optimization
For INNER/LEFT joins in BUILD_PROBE mode, when the build-side keys are proven unique, Sirius uses `cudf::distinct_hash_join` instead of `cudf::hash_join`. This optimization applies when:
- Join type is INNER or LEFT
- Build-side keys are proven unique via logical plan analysis (`prove_unique_columns()` in `src/planner/sirius_plan_comparison_join.cpp`)

Uniqueness is detected by walking the DuckDB logical plan:
- **PRIMARY KEY** on `LogicalGet` (with column mapping through `projection_ids`)
- **GROUP BY** uniqueness on `LogicalAggregate`
- Propagates through `LogicalFilter`, `LogicalOrder`, `LogicalLimit`, `LogicalTopN`, and `LogicalProjection`

Only PRIMARY KEY is considered (not plain UNIQUE) due to NULL handling semantics with `null_equality::UNEQUAL`. IS NOT DISTINCT FROM joins are excluded since they require `null_equality::EQUAL`.

Build/probe state machine for BUILD_PROBE mode (one instance per partition slot, driven by `build_probe_action`):
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
- `conditions` — join predicates (equality and inequality; equality sides are always plain column references by execution time)
- `join_type` — INNER, LEFT, RIGHT, OUTER, MARK
- `_partition_build_states` — one `per_partition_build_state` per BUILD_PROBE partition slot, holding that slot's cached `hash_table` (`cudf::hash_join`), `distinct_hash_table` (used instead when build keys are proven unique), `filtered_table` (`cudf::filtered_join` for SEMI/ANTI/MARK), materialized `build_table`, `device_id`, and per-slot atomic `build_state`
- `_build_has_null` — join-wide atomic recording whether any build key was NULL (drives the MARK three-valued mask)
- `key_casts` — type alignment info for hash key matching
- `unique_build_keys` / `unique_probe_keys` — cardinality hints (used to select distinct vs standard hash join)
- `mark_join_build_switch_ratio` — threshold for adaptively building a STANDARD MARK join on the smaller (left) side

Supported join types: INNER, LEFT, RIGHT, OUTER, MARK via `cudf::inner_join()`, `cudf::left_join()`, `cudf::full_outer_join()`, `cudf::filtered_join`, and `cudf::mark_join`.

### `sirius_physical_nested_loop_join` — `NESTED_LOOP_JOIN`
**File:** `src/include/op/sirius_physical_nested_loop_join.hpp`

Fallback for joins not supported by cuDF hash join (pure inequality conditions). Uses `PhysicalNestedLoopJoin::IsSupported()` to validate.

Conditional MARK joins produce the same three-valued mark as the hash join, via a two-semi-join scheme in its own `resolve_mark_join_result`: one `conditional_left_semi_join` on the predicate itself yields the *matched* set, and a second on `predicate IS NOT FALSE` — each comparison rewritten as `cᵢ OR IS_NULL(left) OR IS_NULL(right)` with Kleene `NULL_LOGICAL_OR` — yields the *maybe* set. A row's mark is true if matched, NULL if in the maybe set but not matched, false otherwise. Null-safe (`IS [NOT] DISTINCT FROM`) conjuncts skip the `IS_NULL` tainting since they are never NULL-valued; `distinct_from` is lowered as `NOT(NULL_EQUAL(l, r))`.

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
- **AVG handling:** Decomposed into SUM + COUNT and finalized on-device. `make_avg_column()` divides the single-row merged sum/count columns with `cudf::binary_operation` — DECIMAL output divides directly in fixed point to preserve precision, while non-DECIMAL output casts both operands to FLOAT64 and divides. This keeps AVG off the host `long double` path, avoiding both the device→host sync and the precision loss of decimal round-trips. The denominator is the count of *non-null* values (matching SUM, which skips NULLs), computed with a NULL-excluding COUNT reduction; when the column has no NULLs the row count is used directly.
- **DECIMAL overflow handling:** DECIMAL SUM casts to a wider type before reduction — DECIMAL32→DECIMAL64, DECIMAL64→DECIMAL128 — to prevent overflow
- **BIGINT SUM fallback:** BIGINT (INT64) SUM falls back to CPU execution because GPU lacks INT128 accumulator support. Without this, silent overflow produces incorrect results. BIGINT arithmetic operations (ADD, SUB, MUL) also fall back to CPU for the same reason.

### `sirius_physical_grouped_aggregate` — `HASH_GROUP_BY`
**File:** `src/include/op/sirius_physical_grouped_aggregate.hpp`

Hash-based GROUP BY.

- **GPU execution:** `gpu_aggregate_impl::local_grouped_aggregate()` using `cudf::groupby()`
- **AVG handling:** Decomposed into SUM + COUNT_VALID via `AggregateSlot`
- **COUNT(DISTINCT):** Implemented via `COLLECT_SET` aggregation with struct column synthesis
- **Label-encoded group keys:** when a COLLECT_SET aggregation is present, the input has at least 1,048,576 rows, the non-nested key is not already a single null-free fixed-width column, and an HLL estimate puts group cardinality below 1% of rows, distinct key rows are sorted lexicographically with nulls last and installed as the build side of a `cudf::distinct_hash_join`. Probing the original rows yields their dense sorted-key indices, so cuDF's `stable_sorted_order` takes its single-column INT32 radix path; representative keys are recovered by a gather at group cardinality. A single nullable or variable-width key can therefore qualify, as can a multi-column key. Non-fatal label construction failures fall back to the original key columns, and an active label path bypasses STRING dictionary encoding.
- **Key members:** `group_idx`, `cudf_aggregates`, `cudf_aggregate_idx`, `aggregate_slots`, `has_avg`, `has_count_distinct`

### `sirius_physical_dense_count_join` — `DENSE_COUNT_JOIN`
**File:** `src/include/op/sirius_physical_dense_count_join.hpp`, `src/op/sirius_physical_dense_count_join.cpp`, kernels in `src/cuda/dense_count_join_impl.cu`

Fuses eligible `COUNT(col | *) GROUP BY key` over a preserved-side outer equi-join, replacing the
partitioned join and aggregate fragment. Children are normalized as [preserved, counted].

Both inputs are FULL barriers and are hash-partitioned on the join key, and the operator runs one
task per partition. Execution uses direct-address histograms or exact sparse aggregation;
ineligible and disabled plans retain the standard path. See [Configuration](configuration.md) and
[Fused Dense Count Join](optimizations.md#fused-dense-count-join-pr-1606) for the value rule, the
dense-versus-sparse gate and the dense fast paths.

DENSE_COUNT_JOIN is a sink parent: each direct child reports `is_sink()` and terminates its own
pipeline through the generic `build_pipelines()` protocol, so scans, streaming chains, joins,
aggregates and sorts can feed either input; the counted producer is built first. Delim-join,
materialized-CTE and delim/CTE scan roots are declined at plan time because their output does not
arrive through a child-owned pipeline, and a delim join is declined at any depth of an input
because the operator's task hint can poll a MARK hash join inside the delim subtree before its
sizing partitions have run.

## Pipeline Breakers (Sirius-Specific)

These operators are injected during pipeline splitting. They don't map to DuckDB logical operators.

### `sirius_physical_partition` — `PARTITION`
**File:** `src/include/op/sirius_physical_partition.hpp`

Repartitions data into N buckets based on partition keys. Partition keys are always plain column indices — a hash-join equality key that is a complex expression has already been materialized into a real column by a planner-inserted projection (`materialize_expression_join_keys()`), because PARTITION hashes by column index and cannot evaluate expressions.

- **Modes:** `HASH` (most common), `RANGE`, `EVENLY`, `CUSTOM`, `NONE`
- **Adaptive count:** the downstream consumer's `get_partition_strategy()` computes N from input data size and `hash_partition_bytes` (`natural_num_partitions()`). Fixed on the first task and never revised — later batches would otherwise land in slots computed under a different modulus.
- **Sibling coordination:** Build-side partition normally determines the shared count. For RIGHT-family hash joins other than `RIGHT_DELIM_JOIN`, the retained probe side determines it instead.
- **Projected sizing (aggregate fanout):** the group-by partition sizes itself from a *projected* total rather than the bytes on hand, via `estimate_port_total_input_bytes()` (see [data management](data-management.md#runtime-data-size-estimation)). That is what lets its ingress be a `PARTIAL` barrier — `resolve_barrier` relaxes that edge only when `enable_runtime_size_estimation` is set on the partition, so with the setting off the ingress stays `FULL` as it always was. While the edge *is* relaxed, `get_next_task_hint()` withholds tasks until a projection is actually available, reproducing the `FULL` wait. On finalize the operator logs projected-vs-actual bytes at INFO so the projection's accuracy can be checked from a query log.
- **Key members:** `_partition_keys`, `_partition_type`, `_num_partitions`, `_is_build`, `_drives_partition_count`, `_sibling_partition_op`, `_size_estimate`

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
| STREAMING_SOURCE | Scan | Exchange-input source; drains an `exec::batch_stream` |
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
| DENSE_COUNT_JOIN | Join+Agg | Fused dense/sparse COUNT over an outer join |
| NESTED_LOOP_JOIN | Join | Fallback nested loops |
| LEFT_DELIM_JOIN | Join | Correlated subquery wrapper |
| RIGHT_DELIM_JOIN | Join | Correlated subquery wrapper |
| PARTITION | Pipeline | Hash/range partitioning |
| CONCAT | Pipeline | Partition reassembly |
| MERGE_TOP_N | Pipeline | Merge per-partition top-N |
| CTE | CTE | Materialize to ColumnDataCollection |
| RESULT_COLLECTOR | Result | Final result materialization |
| EMPTY_RESULT | Result | Empty result set |
| STREAMING_SINK | Result | Terminal operator of a streaming fragment |
