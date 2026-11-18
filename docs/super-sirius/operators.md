# Operators

This document covers all Super Sirius physical operators, organized by category.

## Base Class

**File:** `src/op/sirius_physical_operator.hpp`

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

## Output ordering

Every concrete physical operator implements the pure virtual `output_order_preservation()`.
This declaration describes its local effect on logical order, independently of its pipeline role:

- `FIXED_ORDER`: establishes a global output sequence.
- `NO_ORDER`: does not preserve a global input sequence.
- `INSERTION_ORDER`: preserves the sequence supplied by its input; it does not establish one.

`operator.output_order()` resolves this property over the physical plan subtree. Defining and
order-destroying operators return their declaration directly. Preserving unary operators inherit
their child's resolved order, even if they are pipeline sources. The traversal uses `get_children()`
so collectors exposing a referenced plan participate. CTE nodes follow the consumer child, ignoring
the materialization child. Other preserving multi-input shapes are rejected rather than combining
unrelated sequences by child visitation order.

An `INSERTION_ORDER` leaf retains that result: its input sequence is external to this tree.
This is not proof that runtime batches carry positions. A planner must establish that external
contract before choosing an ordered consumer; it must not silently treat this value as `NO_ORDER`.
The analysis does not consult DuckDB's insertion-order setting or repository runtime state.

Use this analysis after the child's GPU operator expansion. `ORDER_BY` and `TOP_N` execute local
per-batch sorting; their expanded global result is established by `MERGE_SORT` and `MERGE_TOP_N`.
The planner can query `limit.children[0]->output_order()` before choosing a LIMIT implementation
and its pipeline boundary. The planner chooses `UNORDERED_STREAMING_LIMIT` only for a resolved `NO_ORDER`;
`FIXED_ORDER` and unresolved `INSERTION_ORDER` retain `STREAMING_LIMIT` and its full barrier.
An inline `REPLICATE` also retains that barrier: it can emit several batches per task, and
publication separates them for LIMIT's one-batch claims.

| Operator | `output_order_preservation()` |
|---|---|
| MERGE_SORT, MERGE_TOP_N | `FIXED_ORDER` |
| ORDER_BY (local sort), TOP_N (local top-N), SORT_SAMPLE, SORT_PARTITION | `NO_ORDER` |
| PARTITION, CONCAT, UNION | `NO_ORDER` |
| WINDOW, REPLICATE | `NO_ORDER` |
| HASH_JOIN, NESTED_LOOP_JOIN, DENSE_COUNT_JOIN, LEFT_DELIM_JOIN, RIGHT_DELIM_JOIN | `NO_ORDER` |
| HASH_GROUP_BY, UNGROUPED_AGGREGATE, MERGE_GROUP_BY, MERGE_AGGREGATE | `NO_ORDER` |
| GPU_SCAN, UNORDERED_STREAMING_LIMIT | `NO_ORDER` |
| PROJECTION, FILTER, DYNAMIC_FILTER, BUILD_FILTER, STREAMING_LIMIT | `INSERTION_ORDER` |
| PASSTHROUGH_SINK, CTE, materialized RESULT_COLLECTOR | `INSERTION_ORDER` |
| TABLE_SCAN, COLUMN_DATA_SCAN (including CTE_SCAN / DELIM_SCAN), GPU_VALUES | `INSERTION_ORDER` |
| DUMMY_SCAN, EMPTY_RESULT | `INSERTION_ORDER` |
| STREAMING_SOURCE, STREAMING_SINK (all routing modes) | `NO_ORDER` |

Abstract intermediates (partition consumer, delim join, result collector) leave the declaration
pure virtual so concrete implementations must choose. Test-only structural placeholders use an
explicit `NO_ORDER` test double. A subclass of a concrete operator can inherit its declared
behavior; C++ pure virtual methods enforce the decision at the first concrete implementation,
not anew for every descendant. Override the declaration when a specialization changes ordering.

## Scan Operators

These operators produce data for pipelines. See [Scan](scan.md) for in-depth coverage.

### `sirius_physical_table_scan` — `TABLE_SCAN`
**File:** `src/op/sirius_physical_table_scan.hpp`

Base scan operator wrapping a DuckDB table function. Stores column IDs, projection IDs, and optional table filters for predicate pushdown. It exists only as the plan-time carrier: during plan generation it is rewritten into a `GPU_SCAN` source (see below).

### `sirius_gpu_scan_operator` — `GPU_SCAN`
**File:** `src/op/scan/sirius_gpu_scan_operator.hpp`

Unified GPU scan source operator for reading table data from storage. It carries no format-specific code: it pulls pre-built splits off a `split_connector` and delegates per-split materialization to an installed `gpu_ingestible`, one implementation per source format (`parquet_gpu_ingestible` for Parquet, `duckdb_native_gpu_ingestible` for DuckDB-native `.duckdb` tables); pinned-cache hits are served by the scan manager's `cached_databatch_provider`.

The pipeline converter rewrites a DuckDB parquet or DuckDB-native table scan into a `GPU_SCAN` source: it lowers the bind data into the appropriate `ingestible_table_info`, builds the `gpu_ingestible`, and inserts the operator at `operators[0]` of the pipeline. Before a query runs, `sirius_scan_manager` prepares scan-side state — matching pinned-cache entries or building a `split_provider` over each operator's ingestible — and drives metadata production, split coalescing, and per-GPU balancing, pushing splits onto each operator's `split_connector`. `execute()` calls `gpu_ingestible::materialize_table` and, when a split carries filter/projection info, `gpu_ingestible::post_filter_and_project`.

See [Scan](scan.md) for the full scan subsystem (scan manager, `gpu_ingestible`, pinned-table caching, and the IO layer).

### `sirius_physical_streaming_source` — `STREAMING_SOURCE`
**File:** `src/op/sirius_physical_streaming_source.hpp`

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
**File:** `src/op/sirius_physical_streaming_sink.hpp`

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
**File:** `src/op/sirius_physical_dummy_scan.hpp`

Generates a single empty row for constant queries (e.g., `SELECT 1+2`).

### `sirius_physical_column_data_scan` — `COLUMN_DATA_SCAN` / `CTE_SCAN` / `DELIM_SCAN`
**File:** `src/op/sirius_physical_column_data_scan.hpp`

Scans a pre-materialized `ColumnDataCollection`. Used for CTE results, correlated subquery intermediates, and expression-generated data.

## Streaming Operators

These operators process data in a single pass without buffering.

### `sirius_physical_filter` — `FILTER`
**File:** `src/op/sirius_physical_filter.hpp`

Applies a predicate expression to filter rows.

- **GPU execution:** `expression_evaluator::select(batch)` — evaluates the boolean expression and compacts rows using cuDF filtering
- **Key members:** `expression` (filter predicate)
- **Batch ordering:** Tasks may execute in any order and forward the input batch position unchanged.

### `sirius_physical_projection` — `PROJECTION`
**File:** `src/op/sirius_physical_projection.hpp`

Evaluates a list of expressions to produce output columns.

- **GPU execution:** the operator classifies each `select_list` entry as either a pure column passthrough (a `sirius::ast::reference` / BOUND_REF) or an expression that must be evaluated, then takes one of three paths per input batch:
  - **All evaluated:** `expression_evaluator::evaluate()` produces an owned `cudf::table` of new columns.
  - **All passthrough:** the output is a zero-copy `cudf::table_view` over the input columns. The output batch is a view-backed `gpu_table_representation` (see [data management](data-management.md)) whose owner is the input's `read_only_data_batch` lock, which keeps the source columns alive and read-only-pinned for the output's lifetime — no device copies.
  - **Mixed:** only the non-passthrough entries are evaluated; the output view mixes the freshly-evaluated columns with the input's passthrough columns, owned jointly by the evaluated table and the input lock.

  Only the entries that need evaluation are passed to the expression evaluator (via its `std::vector<sirius::ast::node const*>` constructor). See [expression evaluator](expression-executor.md).
- **Key members:** `select_list` (output expressions)
- **Batch ordering:** Tasks may execute in any order and forward the input batch position unchanged.

### `sirius_physical_unordered_streaming_limit` — `UNORDERED_STREAMING_LIMIT`
**File:** `src/op/sirius_physical_limit.hpp`

Used only when the expanded child's `output_order()` proves `NO_ORDER`. Runs inline without an
extra pipeline boundary. Inside `execute()`, an RAII `claim_request` reserves offset/limit rows
under a mutex using the row count after upstream filters. The mutex is released during GPU work.
After constructing and synchronizing the output, execution approves the request. Exceptions
before approval refund only that request's reserved offset and limit; retries make a fresh claim.
No batch IDs or per-input claim history are needed. A retry may select different rows, which is
valid for this unordered input.

`is_limit_exhausted()` stops further task creation only when the limit is consumed, no pending
claim can refund offset or limit rows, and at least one output has been approved. Zero-row claims
do not delay exhaustion, even if they keep overlapping on an open stream. Both approval and
rollback update this condition. The pipeline waits for all already-created tasks (including retries) before
completing. The executor's post-return diagnostics do not perform retryable GPU allocations;
changes to its retry boundary must preserve approval after all retryable work. LIMIT 0 permits an
initial task to emit an empty batch for downstream aggregates. This does not cancel independent
upstream pipelines or scan I/O. Outputs carry no positions.

### `sirius_physical_streaming_limit` — `STREAMING_LIMIT`
**File:** `src/op/sirius_physical_limit.hpp`

Implements LIMIT/OFFSET with a FULL input barrier and serialized slice claims during task creation.

- **Key members:** `_remaining_offset` and `_remaining_limit` (`size_t` counters), guarded by `_claim_mutex`.
- **Mechanism:** After the producer finishes, `get_next_task_input_data()` consumes ordered input by ascending existing position, or unordered input in FIFO order. It snapshots existing positions for ordered input, reads the batch row count with `get_batch_rows_and_bytes()`, and claims an immutable slice under `_claim_mutex`. Unknown row counts are errors. The resulting `limiting_operator_data` carries the slice and optional position; execution and retries apply that slice without claiming again.
- **Completion:** Every input batch is consumed, but offset-only and trailing unused batches are drained during task creation rather than sent to GPU tasks. Spilled trailing batches are released without restoring them to GPU memory. Already-created tasks and retries still finish normally. Both ordered and unordered input wait for the complete producer output before LIMIT starts consuming it.
- **Empty output:** If no input slice selects rows (including LIMIT 0 or an offset past all input), one task emits a 0-row batch at the retained input's position, so a downstream ungrouped aggregate still returns its row. Other empty slices need no task. DuckDB plans an uncorrelated `EXISTS` as a count over `LIMIT 1`.

### `sirius_physical_replicate` — `REPLICATE`
**File:** `src/op/sirius_physical_replicate.hpp`

Repeats each input row by a count column, then drops that column. The count is always the last input column; row `i` appears `count[i]` times, copies adjacent, in input order. The operator does not know what the count means. Today only the `EXCEPT ALL` / `INTERSECT ALL` lowering builds it (see [set operation planning](physical-plan-generation.md#set-operation-planning)).

- **GPU execution:** `plan_slices` in `gpu_replicate_impl` (`src/op/replicate/gpu_replicate_impl.cu`) takes an `INT64` prefix sum of the counts and, when byte accounting is enabled, a byte prefix weighted by `cudf::row_bit_count`. It limits each slice to `max_rows` rows and uses `max_bytes` as a byte target, which the final row may cross. `materialize` searches the count prefix for each output position to build a gather map, then copies the slice with `cudf::gather`. The planner limits each slice to `INT32_MAX` rows.
- **Output batches:** one per slice, on the input batch's memory space. An input with no copies gives one empty batch, so downstream operators still see the schema.
- **Key members:** `_output_limits` (`gpu_replicate_impl::limits`). The planner sets `max_rows` to `INT32_MAX` and `max_bytes` to the `concat_batch_bytes` target, reading 0 as no byte accounting (`SIZE_MAX`).
- **Errors:** a null or negative count, a non-integer count column, or a batch of the wrong width throws `sirius::internal_exception`. These are planner bugs, not reasons to fall back.
- **Memory:** one `execute` holds the whole expansion of its input batches, so the limits shape output batches but do not bound peak memory. A key with very many copies is still one task.

## Blocking Operators

These operators buffer input before producing output. They are both sinks and sources.

### `sirius_physical_hash_join` — `HASH_JOIN`
**File:** `src/op/sirius_physical_hash_join.hpp`, `src/op/sirius_physical_hash_join.cpp`

Three execution modes:

| Mode | When Used | cuDF API |
|------|-----------|----------|
| `STANDARD` | Default, multi-partition Cartesian product | `cudf::inner_join()`, `cudf::left_join()`, etc. |
| `BUILD_PROBE` | Up to one partition per GPU; per-GPU estimated hash-table bytes (measured build rows × 16 B) under `max_build_hash_table_bytes` AND per-GPU decoded folded build payload under `max_build_probe_resident_bytes`, build foldable to one batch | `cudf::hash_join`, `cudf::distinct_hash_join`, or `cudf::filtered_join` — built once per partition, probed many times |
| `MIXED_JOIN` | Equality plus either inequality conditions or a null-safe `IS NOT DISTINCT FROM` key (mixed with a plain `=`), on disjoint columns | `cudf::mixed_join()` with cuDF AST |

Partition count, broadcast, and BUILD_PROBE are decided together by the free function `compute_hash_join_partition_strategy()` (member wrapper: `get_partition_strategy()`), which returns a single `partition_strategy {num_partitions, broadcast, build_probe, placement}`; `placement` names the GPU each partition runs on (see [multi-GPU architecture](multi-gpu-architecture.md#partition-device-pin-for-cuco-backed-operators)). BUILD_PROBE is selected when `num_partitions <= num_gpus` (one hash table per partition, at most one partition per GPU — this reduces to the historical single-partition rule when `num_gpus == 1`), the per-GPU estimated hash table fits `max_build_hash_table_bytes`, the decoded build payload fits `max_build_probe_resident_bytes`, all build rows are measured and the full row count fits `cudf::size_type`, and the build folds to a single batch, and the join is not RIGHT-family (`RIGHT`, `RIGHT_SEMI`, `RIGHT_ANTI`), `MIXED_JOIN`, or full `OUTER`. INNER, LEFT, MARK, SEMI, and ANTI joins are eligible (SEMI/ANTI/MARK build a persistent `cudf::filtered_join` on the right and stream left probe batches); MARK follows the SEMI rule (see [MARK joins](#mark-joins) for how its global NULL rule survives partitioning). Full outer is excluded because BUILD_PROBE streams probe batches and calls `full_join` per batch, which would re-emit unmatched build rows on every batch (and, under broadcast/partitioning, on every GPU) with no global accumulation — full outer joins use the STANDARD path. For a broadcast join the **full** replicated build rows and decoded bytes are charged against the two budgets (each GPU builds the entire table); a hash-partitioned build charges rounded-up per-partition averages.

By execution time every equality-condition side is a plain column reference: a complex equality-key expression is materialized into a real column by a planner-inserted projection below the join (`materialize_expression_join_keys()`), because PARTITION — the first key consumer — hashes by column index and cannot evaluate expressions. Inequality sides stay inline as the mixed-join AST predicate.

**NULL comparison for keys.** `compare_nulls()` picks the `null_equality` flag threaded through every cuDF join call (including the build helpers and `filtered_join`/`mark_join`): `EQUAL` only when *every* equi-key is `IS NOT DISTINCT FROM`, `UNEQUAL` otherwise (any plain `=`, delim joins, and MARK). A null-safe key *mixed* with plain `=` is instead routed through MIXED_JOIN (see below).

Before physical lowering, catalog `NOT NULL` constraints can prove null-safe keys equivalent to ordinary equality for matching rows. Normalization changes all null-safe equality keys together only when every key has a proven non-null operand; it uses no data statistics and excludes MARK joins. This can admit ordinary equality dynamic filters without changing the join's matching behavior. See [dynamic-filter null handling](dynamic-filters.md#nulls).

**Nullable mixed SEMI/ANTI residuals.** These joins, including RIGHT_SEMI/RIGHT_ANTI, retain equality-key hashing when a build residual contains NULL. cuDF 26.10 fixes mixed-join build deduplication when a conditional comparator uses the equality table's nullability. Only residual conditions enter the AST; hash-key casts continue through normal key preparation.

**Broadcast small build tables.** On multi-GPU, a build side is a broadcast candidate when it is small (`< small_table_bytes`), or when it fits `max_broadcast_join_size` and the estimated probe-to-build ratio is at least `num_gpus * 1.25`. Instead of routing the whole build to one GPU, the join places one partition on each GPU (`partition_placement::one_per_device`, so `placement.num_partitions()` equals the GPU count) and PARTITION *replicates* the small build table to every slot (the `_broadcast` flag), so each GPU builds its own hash table and joins its local probe rows. The build sink deposits the build batch into every slot; the probe sink routes each batch to the slot placed on its current GPU (`partition_placement::first_partition_for_device`). Once the probe side finishes, any slot that received replicated build data but no probe rows is discarded (`discard_build_only_slots_if_probe_complete`). Right-family / mixed joins reject BUILD_PROBE and fall back to the normal partition count.

#### Partial-barrier scheduling (STANDARD / MIXED_JOIN)

The join's own `build` and `default`/probe input ports are **PARTIAL** barriers (owned by the HASH_JOIN `input_port_for` / `input_barrier_for` hooks for `CONCAT → HASH_JOIN` edges). Batches arrive progressively on each side, and `get_next_task_hint` / `get_next_task_input_data` schedule per-partition **build × probe** cross-product pairs as they become available (state tracked in `partition_cross_schedule`, guarded by `op_state_mutex`), freeing each batch once it has been paired with every batch of a finished opposite side. This mirrors how BUILD_PROBE already streams its probe side; BUILD_PROBE is unaffected by the port barrier because it overrides its hint regardless.

Joining each build batch against each probe batch and unioning the results is only *inherently* correct for INNER. For every other join type, `sirius_physical_concat` folds the side that must be seen whole into a **single batch** (an implicit full barrier), so the streamed side is joined against that one folded batch — correct for all types. `refresh_cross_schedule` asserts this "whole side stays one batch" invariant and throws if a concat regression ever violates it.

**Empty-opposite side (orphan tasks).** When one side is a genuinely empty table, its concat emits *zero* batches. The surviving side's batches then have nothing to pair with, so once both producers finish, `next_cross_schedule_orphan` claims each survivor and `get_next_task_input_data` **synthesizes an empty opposite batch** for it (`sirius::make_empty_table` from the absent child's output types, built on the survivor's device) and returns an ordinary two-batch task. `execute()` then runs the normal join dispatch unchanged, so cuDF NULL-pads the survivor for LEFT/RIGHT/OUTER/ANTI and yields zero rows for INNER/SEMI. Popping the survivor also drains the repository, so the pipeline completes instead of hanging. FULL OUTER and RIGHT-family joins are the reachable cases (they always run STANDARD); small-build INNER/LEFT/SEMI/ANTI take BUILD_PROBE, which handles an empty side separately.

For BUILD_PROBE, a closed zero-batch probe reclaims every never-probed build slot. The unused GPU build is released on its owning device; HOST/DISK builds are released directly without uploading them or constructing a hash table. The join can finish without emitting any batch. Downstream ungrouped aggregate finalization owns the empty-input result. MARK pre-sizing and empty-build semantics continue through the normal MARK paths.


The following table summarizes, per join type, what concat folds, which side streams under this scheme, and the partition / broadcast / mode support:

| JoinType | concat fold (whole side) | Streams multi-batch | batch×batch + union | Multi-partition | Broadcast | Supported modes |
|---|---|---|---|---|---|---|
| INNER | neither | both sides | ✅ inherent | ✅ natural count | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| LEFT | build → 1 | probe | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| SEMI | build → 1 | probe | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| ANTI | build → 1 | probe | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE, MIXED_JOIN† |
| MARK | build → 1 | probe (pairs wait for the whole build) | ✅ | ✅ | ✅ (small build) | STANDARD, BUILD_PROBE |
| OUTER (FULL) | both → 1 | neither (stays 1×1) | ✅ degenerate | ✅ natural count | ❌ | STANDARD, MIXED_JOIN† |
| RIGHT | probe → 1 | build | ✅ | ✅ (probe-driven) | ❌ | STANDARD, MIXED_JOIN† |
| RIGHT_SEMI | probe → 1 | build | ✅ | ✅ | ❌ | STANDARD, MIXED_JOIN† |
| RIGHT_ANTI | probe → 1 | build | ✅ | ✅ | ❌ | STANDARD, MIXED_JOIN† |
| SINGLE | — | — | — | — | — | unsupported (throws) |

† MIXED_JOIN applies to any of these types when the join carries equality conditions **plus** either an inequality condition or a null-safe `IS NOT DISTINCT FROM` key mixed with a plain `=` (the null-safe key is routed to the AST predicate as `NULL_EQUAL`, since cuDF's single `null_equality` flag can't give `=` and null-safe keys opposite semantics). MARK is never routed to MIXED_JOIN: MARK + inequality is rejected at construction, and a MARK + null-safe key stays a `UNEQUAL` hash key (a known null-safe limitation). "Streams multi-batch" describes the STANDARD/MIXED partial-barrier behavior; BUILD_PROBE already streamed its probe side. Join wrapping chooses the initial `concat_all` flag from the join type; partition / broadcast / mode eligibility remains in `compute_hash_join_partition_strategy`. Runtime partition negotiation strengthens the build CONCAT requirement before publishing input.

#### MARK joins
A MARK join emits every left row plus a `BOOL8` mark column with SQL three-valued semantics: **true** for a match, **false** for a non-match when the probe key is non-NULL and the build side has no NULL key, and **NULL** otherwise (NULL probe key, or no match while a NULL build key exists — the row *might* have matched it). Both build strategies funnel through `resolve_mark_join_result`, which scatters left-row match indices into the mark column and attaches the null mask: when the build side has a NULL key, validity equals the match flags (`cudf::bools_to_mask`); otherwise validity is the probe keys' row validity (`cudf::bitmask_and`). MARK partitions like SEMI (STANDARD at the natural count when either BUILD_PROBE budget is exceeded, each partition's build concat-folded). When every key is null-safe (`IS NOT DISTINCT FROM`, the usual delim-join shape) no comparison is UNKNOWN and every mark is definite, so partitioning needs nothing more. With a null-propagating (`=`) key the NULL rule is global — a NULL build key in *any* partition makes an unmatched mark NULL, and a NULL probe key is NULL whenever the build is non-empty anywhere — so resolution goes through `resolve_mark_null_state`: when the build is one slot or broadcast it uses the task's own build (BUILD_PROBE records it in `_build_has_null`); when the build is hash-partitioned (`_mark_build_partitioned`, set at sizing) it uses join-wide facts that the build PARTITION's `hash_join_partition_observer` records for every build batch before scattering (`observe_partitioned_build_batch`: any row, any NULL key). Those facts are final before any build batch reaches the join, because the build CONCAT folds each partition only after the PARTITION pipeline finishes; STANDARD additionally schedules no MARK pair until the whole build pipeline has finished (`mark_waits_for_whole_build`).

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

Only PRIMARY KEY is considered (not plain UNIQUE) due to NULL handling semantics with `null_equality::UNEQUAL`. For a parquet scan of a pinned table, a column proven distinct at pin time (no nulls, no repeats) also counts, unless `pin_table_prove_unique_keys` is off; see [Pin-Proven Unique Join Keys](optimizations.md#pin-proven-unique-join-keys). IS NOT DISTINCT FROM joins are excluded since they require `null_equality::EQUAL`.

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
**File:** `src/op/sirius_physical_nested_loop_join.hpp`

Fallback for joins not supported by cuDF hash join (pure inequality conditions). Uses `PhysicalNestedLoopJoin::IsSupported()` to validate.

Both CONCAT input ports are PARTIAL barriers. It discovers batches incrementally and runs one task per previously unscheduled pair of left and right batches, using the shared `join_cross_schedule` helpers. If build data is available but no probe batch has arrived, its readiness hint requests the probe producer so demand-driven scheduling can start joining early. The upstream build scan still feeds PARTITION through a FULL barrier. For every join type except INNER, join wrapping (`join_concat_all` in the plan generator) sets `concat_all` on the CONCAT feeding the side that rows are decided against, so each task sees that side whole: the right side for LEFT/SEMI/ANTI/MARK, the left side for RIGHT, and both sides for FULL OUTER. Otherwise a LEFT join would pad a row once per right batch without a match.

Without conditions it runs a cross product with `cudf::cross_join`. When it claims a new batch pair, it reads that pair's row counts and sizes and splits it into tasks over equal row ranges of the left batch (`cross_join_num_slices` in `src/op/cross_join_slicing.hpp`). Each task's output stays near `cross_join_task_bytes` and within the cuDF row limit, with at most one task per left row. A task carries its slice in `cross_join_slice_data` and cross joins a zero-copy `cudf::slice` of the left batch. A batch is popped only after the opposite producer has finished and every pair using it has been scheduled. A pending sliced pair prevents retirement until all of its slices have been issued; outstanding tasks retain their own input references. Before allocating, `execute` computes the output size of its slice and throws when it exceeds the GPU memory space, so the query falls back to CPU instead.

If a producer emits no batches, after both producers finish the scheduler emits one orphan task per surviving batch. It prepares that batch normally (including spill readback), creates the absent side's empty schema on the executor GPU, and uses the existing empty-input semantics: outer joins NULL-pad, ANTI keeps probe rows, MARK marks them FALSE, and INNER/SEMI emit no rows. A real zero-row batch follows the ordinary pair path. Both cases drain their repositories and permit pipeline completion.

Conditional MARK joins produce the same three-valued mark as the hash join, via a two-semi-join scheme in its own `resolve_mark_join_result`: one `conditional_left_semi_join` on the predicate itself yields the *matched* set, and a second on `predicate IS NOT FALSE` — each comparison rewritten as `cᵢ OR IS_NULL(left) OR IS_NULL(right)` with Kleene `NULL_LOGICAL_OR` — yields the *maybe* set. A row's mark is true if matched, NULL if in the maybe set but not matched, false otherwise. Null-safe (`IS [NOT] DISTINCT FROM`) conjuncts skip the `IS_NULL` tainting since they are never NULL-valued; `distinct_from` is lowered as `NOT(NULL_EQUAL(l, r))`.

### `sirius_physical_union` — `UNION`
**File:** `src/op/sirius_physical_union.hpp`

Joins its arms' rows into one stream without inspecting them, so the operator computes nothing
and `execute` is the identity. N-ary: every path loops over `children`.

**How SQL reaches it.** SQL `UNION` and `UNION ALL` both plan a `LOGICAL_UNION`. They differ only
in what DuckDB's binder puts around it. `UNION ALL` is the node alone, and
`a UNION ALL b UNION ALL c` binds to one node with three children. `UNION` and `UNION BY NAME` are
a `DISTINCT` over that same node, which Sirius lowers to a grouped aggregate.
The `EXCEPT ALL` / `INTERSECT ALL` lowering also uses `UNION` to merge its two tagged inputs.
Distinct `EXCEPT` and `INTERSECT` do not use `UNION`: they lower to a null-safe ANTI or SEMI hash
join, then deduplicate its rows with a grouped aggregate (see `plan_except_intersect`).

The plan builder (`plan_union` in `src/planner/sirius_plan_set_operation.cpp`) sends two more
shapes to the CPU. The first is `allow_out_of_order = false`, which requires output in arm order.
The second is an arm that plans a column type other than the declared one, since arms are
forwarded unconverted. A `WITH RECURSIVE` body with no
self-reference binds to `setop_all = false` with no `DISTINCT` above it, so `plan_union` puts the
DISTINCT lowering's grouped aggregate over the union itself, keyed on every column.

- **One port per arm.** `wrap_union` wraps each arm `child -> PASSTHROUGH_SINK`, feeding a distinct
  `"union_{i}"` port. The distinct names are required: `add_port` is last-writer-wins and the
  repository manager keys by `(operator_id, port_id)`, so a shared name would orphan an arm's
  repository and finish the pipeline while that arm still had rows.
- **`PASSTHROUGH_SINK -> UNION` is `PARTIAL`**, not the base's `FULL` default, which
  would hold every arm's output in repositories until all arms finished.
- **Overrides both task-driver methods.** UNION nominates and drains arms sequentially in child
  order. Only the active arm can be nominated by UNION or popped; a finished empty arm is skipped,
  and UNION issues one normal draining nomination per live arm. An arm that already has data when
  it becomes active was started by `start_query`'s seed or, under lookahead, by a one-task
  request; UNION latches it as nominated and, only under lookahead, enqueues a full request for
  it. Popping the final queued batch from a finished non-final arm advances the cursor and
  schedules UNION once so the next arm is activated even when no producer completion can provide
  another wakeup. One arm per task also keeps each batch on the GPU that produced it.
- **Arm pipelines are created in reverse child order.** The converter schedules child
  meta-pipelines last-created-first, so reversing makes arm 0 pipeline #0: the scan `start_query`
  seeds and the top execution priority. Drain order therefore matches execution order.
- **Output order:** `NO_ORDER`; independently ordered arms do not define a global union order.
- **Arm ports are cached.** Both task-driver methods run on every task-creation walk that reaches
  the operator, so the `"union_{i}"` names are resolved to `port*` once on first use.
- **Carriers.** The compressed-schema pass treats `UNION` as a native boundary, which prevents two
  arms presenting different physical carriers for the same logical column.

### `sirius_physical_order` — `ORDER_BY`
**File:** `src/op/sirius_physical_order.hpp`

Local sort: turns its input into sorted runs for SORT_SAMPLE, SORT_PARTITION and MERGE_SORT.

- **GPU execution:** `gpu_order_impl::local_order_by()` per sort unit: a row order, then one
  `cudf::gather()`.
- **Row order:** `cudf::sorted_order()` for a single fixed-width key (cuDF radix-sorts it). String
  keys and multi-column keys use `order::encoded_sorted_order()`
  (`src/op/order/encoded_sort_order.hpp`) when `sort_key_encoding` is true (the default):
  - Each key column becomes bit fields whose unsigned order is the column's order: a null flag,
    the value minus the column minimum (or the maximum minus the value for DESC), and for a
    string its 8-byte big-endian chunks followed by its length. The fields are packed, as one
    bit string, into 64-bit words.
  - The words are applied most-significant first. Each step radix-sorts only the rows still tied
    on the earlier words, by (tie group, word), over only the bits that differ inside a group. A
    word that is constant inside every group (a shared string prefix) costs one pass and no sort.
  - The result is the order of `cudf::stable_sorted_order()`: the same row comparator as the
    merge (`kway_comparator.cuh`), equal keys in input order.
  - It falls back to `cudf::sorted_order()` for batches under 4,096 rows, key types other than
    fixed-width and STRING, strings longer than 256 bytes, and inputs whose encoded sort would
    need more than a quarter of the GPU memory space.
  - Memory: the result and all temporaries come from the task's memory space, at most 32 bytes
    per row at a time (`order::encoded_sort_working_bytes()`; cuDF's comparison sort needs
    about 8, its single-key radix sort about 24), and the temporaries are freed before the
    gather. An allocation failure is not caught: the executor records the peak and retries the
    task with a larger reservation.
- **Key members:** `orders` (sort keys with ASC/DESC and null ordering), `projections` (output columns), `is_index_sort`

It has two modes, chosen by the planner (`order_by_run_limits()` in `wrap_order_by()`):

| Mode | Plan | Sort unit |
|------|------|-----------|
| per batch (`sort_coalesce_runs = false`, a string sort key with `sort_coalesce_string_keys = false` or `sort_key_encoding = false`, a nested column, or an operator built outside the planner) | `… → SORT_SAMPLE → ORDER_BY → input`; ORDER_BY is the sink of its input's pipeline | Each input batch of a task |
| coalescing (default) | `… → SORT_SAMPLE → ORDER_BY → PASSTHROUGH_SINK → input`; ORDER_BY starts its own pipeline and reads the input's batches from a port (PARTIAL barrier) | A group of parked batches of up to `sort_coalesce_memory_fraction × GPU limit / sort_coalesce_concurrent_runs` bytes, concatenated |

Why coalesce: SORT_PARTITION cuts every sorted run by range, so each partition holds one run per
sort unit, however small the partition. MERGE_SORT reads at most `sort_merge_max_fan_in` runs per
task and rewrites a partition once for every extra factor of the fan-in in its run count, while
the sort itself costs the same per row at any run size for fixed-width keys. Coalescing changes
which rows share a run; it changes neither the rows nor how a unit is sorted.

Coalescing mode:

- **Group planning** (`plan_next_run()`, used by both `get_next_task_hint()` and
  `get_next_task_input_data()` under the operator mutex). Parked batches are grouped per GPU
  device, so a run is built from batches already on the GPU that will sort it; batches that are
  not on a GPU (spilled, or being moved) form one more group. A group is complete when the next
  batch of its device would take it past the byte limit or 2^31 - 1 rows, or when it reaches the
  byte limit; a batch larger than the limit is a group of one. With a string column in the
  schema a group also stays below 2^31 string bytes, counting for each batch what it holds
  beyond its fixed-width values and string offsets (its decoded bytes when its row count is
  unknown), which bounds every string column. Until a group is complete the hint is WAITING on
  the producer. When the producer has finished, the remainder is packed in repository order
  regardless of device.
- **Concurrent runs** (`sort_coalesce_concurrent_runs`, `make_sort_run_limits()`). The inputs of
  the runs one GPU sorts at the same time stay within `sort_coalesce_memory_fraction` of its
  limit: a complete group is claimed only if it fits beside the groups already claimed for its
  GPU and not yet sorted, or if nothing is being sorted there. A group that does not fit stays
  parked; the sort task that finishes schedules the operator again. This is what lets a run
  cover several input batches without every executor thread holding one. With the setting at 0
  a run is the fraction divided by the executor thread count and any number are sorted at once.
- **Input backpressure** (`sort_input_backpressure`). Without it a scan runs ahead of the sort:
  its tasks are created as fast as its batches are prepared and are scheduled first, so the
  unsorted batches fill the GPU and are spilled and read back whole. With it,
  - ORDER_BY reports `input_backlogged()` while the parked batches plus the producer's
    unfinished tasks (at the average batch size) reach twice the fraction of every GPU, and the
    task creator creates no further task for the producing pipeline until a sort task finishes;
  - SORT_SAMPLE does the same for the sorted runs waiting to be partitioned, once the
    boundaries exist (not with cross-rank boundaries);
  - the pipelines of ORDER_BY and SORT_SAMPLE are scheduled ahead of their producers
    (`runs_ahead_of_producer()`, see [Task Creator](task-creator.md#producer-backpressure));
  - the unsorted input is spilled after the query's other batches (downgrade priority -50).
- **Task.** The input is a plain `pipelineable_operator_data`: the task creator places it by data
  locality and `prepare_for_processing()` brings spilled batches back to the GPU. `execute()`
  splits the task's batches into sort units again from the real tables (row count, bytes of each
  string column, same memory space, same cuDF column types), so the cuDF limits hold whatever
  the batch metadata said. A unit of one batch is sorted directly; a larger unit is concatenated
  (`gpu_merge_impl::concat()`), sorted, and the concatenation is freed.
- **Memory.** A coalesced task holds its input batches, their concatenation, the sort's scratch
  and the sorted copy. The first task of the pipeline reserves three times its input
  (`no_history_peak_memory_estimate()`); later tasks use the pipeline's memory history. An
  out-of-memory error is retried by the executor like any other task's.
- **Counters and log.** `counters()` / `process_counters()` (`order_by_counters`) count tasks,
  input batches, runs and concatenated bytes. When its pipeline finishes the operator logs one
  line at INFO:

```
[order_by][stats] op=<id> coalesce=<0|1> max_run_bytes=… max_concurrent_bytes=… max_backlog_bytes=… tasks=… batches_in=… runs_out=… concat_bytes=…
```

### `sirius_physical_top_n` — `TOP_N`
**File:** `src/op/sirius_physical_top_n.hpp`

Combined ORDER + LIMIT: selects and sorts the top N rows.

- **GPU execution:** Two-step process: `cudf::top_k_order()` selects top-N row indices, then `cudf::sort_by_key()` sorts the gathered rows to ensure deterministic output ordering- **Key members:** `orders`, `limit`, `offset`, `dynamic_filter`

### `sirius_physical_ungrouped_aggregate` — `UNGROUPED_AGGREGATE`
**File:** `src/op/sirius_physical_ungrouped_aggregate.hpp`

Aggregate without GROUP BY (e.g., `SELECT COUNT(*), SUM(x) FROM t`).

- **GPU execution:** `gpu_aggregate_impl::local_ungrouped_aggregate()` using `cudf::reduce()`
- **Supported:** SUM, MIN, MAX, COUNT (of valid values), COUNT(*), AVG, FIRST, and COUNT(DISTINCT) on a single non-nested column
- **FILTER clause:** Handled entirely at plan time for SUM (incl. `sum_no_overflow`), MIN, MAX, COUNT, COUNT(DISTINCT), COUNT(*), and AVG over non-nested inputs — filtered inputs are mask-wrapped to NULL in the projection below the operator, identical safe generated masks reuse a single projected column, safe nontrivial complete predicates still repeated across distinct masks share a lower BOOLEAN column, and the planner rebinds `count(*) FILTER` to `count(mask)` before generic translation; other DISTINCT aggregates, FIRST, nested inputs (such as the STRUCT of `COUNT(DISTINCT (a, b))`), and any aggregate outside this set fall back to CPU when filtered. See [Aggregate Planning](physical-plan-generation.md).
- **COUNT(DISTINCT) handling:** each batch reduces the column to a one-row LIST with `COLLECT_SET` (NULLs excluded), `MERGE_AGGREGATE` unions the lists with `MERGE_SETS` and counts the elements. Other DISTINCT aggregates fall back to the CPU at plan time.
- **AVG handling:** Decomposed into SUM + COUNT and finalized on-device. `make_avg_column()` divides the single-row merged sum/count columns with `cudf::binary_operation` — DECIMAL output divides directly in fixed point to preserve precision; DOUBLE output goes through `average_as_double()` (`op/aggregate/average.cu`). A DECIMAL sum there is divided as DuckDB divides it, the unscaled integer sum over `count * P`, where P is 10^scale rounded to DOUBLE the way DuckDB rounds it. Rounding the scaled sum, the sum, the count or `count * P` to one FLOAT64 first would make equal averages over different counts differ in the last bit, so the sum and count are each split into two DOUBLEs, `count * P` is kept as two DOUBLEs, and the quotient is formed in double-double and rounded once. Its relative error before that rounding stays below 2^-102 across the DECIMAL128 sum and INT64 count range, so the result is correctly rounded except for quotients that close to a midpoint between two DOUBLEs. DuckDB's own `long double` (or DOUBLE, for DECIMAL inputs of width 4 or less) quotient can still differ in the last bit where it is not correctly rounded. Other sums are cast to FLOAT64 and divided. This keeps AVG off the host `long double` path, avoiding both the device→host sync and the precision loss of decimal round-trips. The denominator is the count of *non-null* values (matching SUM, which skips NULLs), computed with a NULL-excluding COUNT reduction; when the column has no NULLs the row count is used directly.
- **DECIMAL overflow handling:** a DECIMAL32/DECIMAL64 SUM or AVG input is reduced over DECIMAL128 only when the batch could overflow its storage width, i.e. when `rows × max|value|` can exceed 2^31-1 or 2^63-1 (`sums_to_widen`). The magnitude bound comes from base-table statistics resolved at plan time (`planner::resolve_sum_input_bounds`) when the input is a scan column that reaches the aggregate unchanged, otherwise from a per-batch `cudf::minmax` scan (`sums_needing_widening`). A batch that cannot overflow is reduced at the input width and only its one-row partial is widened. Conversely, a DECIMAL128 SUM or AVG input whose batch provably sums within 64 bits (`rows × max|value|` ≤ 2^63-1, `decimal_sums_to_narrow`) is reduced over DECIMAL64 at the same scale and its one-row partial is cast back to DECIMAL128, so the merge sees the same types and combines the partials at 128 bits.
- **Integer SUM/AVG overflow handling:** DuckDB accumulates `sum` and `avg` over INTEGER and BIGINT in 128 bits and returns HUGEINT for SUM (`is_wide_integer_sum`), unless its statistics proved the total fits INT64 and planned `sum_no_overflow`, which keeps the plain INT64 path. On the GPU HUGEINT is carried as DECIMAL128 with scale 0. Each batch sums the INTEGER or BIGINT input in 128 bits in a single pass (`wide_integer_sum`, a CUB reduction over INT32/INT64 that skips NULLs). That needs no magnitude scan and no widening copy, so it reads the input once like the INT64 reduction it replaces. The one-row partial is an exact DECIMAL128 with scale 0 (`wide_integer_sum_type()`), so the merge cannot overflow. After the merge, AVG divides the 128-bit sum with `average_as_double()` as a DECIMAL with scale 0 (P = 1), so it is rounded once from the exact quotient; DuckDB divides in `long double`, so the last bit can still differ where its quotient is not correctly rounded. An aggregate whose result is HUGEINT (`AggregateSlot::returns_hugeint`) is finalized to DECIMAL128 (`to_hugeint_carrier`): an INT64 sum, from `sum_no_overflow`, a narrower input, or a grouped partial that stayed INT64, is widened, and a DECIMAL128 one passes through. So a total past BIGINT, and DuckDB's `SUM(x) / COUNT(x)` rewrite of AVG (ClickBench Q03's `AVG(UserID)`), stay on the GPU with exact results. A SUM over a HUGEINT input takes the decimal SUM path.

### `sirius_physical_grouped_aggregate` — `HASH_GROUP_BY`
**File:** `src/op/sirius_physical_grouped_aggregate.hpp`

Hash-based GROUP BY.

- **GPU execution:** `gpu_aggregate_impl::local_grouped_aggregate()` using `cudf::groupby()`
- **AVG handling:** Decomposed into SUM + COUNT_VALID via `AggregateSlot`; after the merge a DOUBLE AVG is finalized by `average_as_double()`, the same rounding as the ungrouped aggregate
- **STDDEV_SAMP:** Each local group emits a STRUCT of valid count (INT64), mean (DOUBLE), and M2 (DOUBLE, the sum of squared deviations). Local grouping sorts keys and values to select cuDF's centered M2 implementation; its hash implementation uses a cancellation-prone difference of large sums. Merge uses cuDF `MERGE_M2`, then computes `sqrt(M2 / (count - 1))`. NULL inputs are skipped; fewer than two valid values yield NULL. Non-finite finalized results raise an out-of-range error. DISTINCT remains unsupported. The local and PARTITION schemas carry the STRUCT; the merge schema carries the final DOUBLE. Ungrouped aggregation uses the same state and finalization, obtaining local M2 from population variance times count because cuDF has no M2 reduction API.
- **COUNT(DISTINCT):** Implemented via `COLLECT_SET` aggregation with struct column synthesis
- **FILTER clause:** Same plan-time mask-to-null lowering and CPU-fallback rules as UNGROUPED_AGGREGATE above; group keys are never masked, so groups whose every row fails the filter still appear.
- **Grouping sets with STDDEV_SAMP:** Subtotals merge count/mean/M2 states with `MERGE_M2` before the final cross-batch merge and stddev finalization. The partial schema includes the set id and `GROUPING()` columns before the expanded aggregate carriers. An empty grand total carries a valid zero-count state and finalizes to NULL; the set id keeps duplicate grouping sets distinct.
- **DECIMAL overflow handling:** the same gate as the ungrouped aggregate (`sums_to_widen`): a DECIMAL32/DECIMAL64 SUM whose batch could overflow is widened to the next decimal width before `cudf::groupby`; a SUM proven safe runs at the input width and its per-group result is cast afterwards, so the merge sees one type either way. The plan-time bounds arrive in `cudf_aggregate_input_max_abs`. A DECIMAL128 SUM that provably fits 64 bits in the batch is computed over DECIMAL64 and cast back the same way (`decimal_sums_to_narrow`).
- **Integer SUM/AVG overflow handling:** a wide integer SUM over BIGINT (see the ungrouped aggregate; flagged in `cudf_aggregate_is_wide_integer_sum`) is gated the same way. A batch that could overflow INT64 produces a DECIMAL128 per-group partial with scale 0. While the previous batch had at most 64K groups (`split_wide_sum_max_groups`), it sums the high and low 32-bit halves of each value as two INT64 SUMs and combines them (`split_int64_halves`, `combine_int64_halves`); otherwise it groups over DECIMAL128 directly. A batch proven safe stays INT64 and is not widened, so high-cardinality GROUP BYs keep INT64 partials. `gpu_merge_impl::merge_grouped_aggregate()` therefore sees INT64 or DECIMAL128 partials per batch. Once any batch carries DECIMAL128 it widens the rest, and it also widens all-INT64 partials whose total magnitude reaches 2^62 (`partial_sums_may_overflow`). Otherwise it merges in INT64. The planner records DECIMAL128 as the alternate carrier of those columns on HASH_GROUP_BY and PARTITION (`wide_integer_sum_partial_types`, `set_alternate_physical_types`), so the task-level output-type check accepts either width. The merge's post-processing widens INT64 SUM results to HUGEINT's DECIMAL128 carrier (`to_hugeint_carrier`), passes DECIMAL128 through, and finalizes AVG with `average_as_double()`. A SUM or AVG over a HUGEINT input falls back to DuckDB when its total could pass 2^127, which cuDF would wrap: per batch when rows x max |v| could (`decimal_sums_to_narrow`), and at the merge when the partials' bound could (`require_decimal128_partial_sums_in_range`).
- **Label-encoded group keys:** when a COLLECT_SET aggregation is present, the input has at least 1,048,576 rows, the non-nested key is not already a single null-free fixed-width column, and an HLL estimate puts group cardinality below 1% of rows, distinct key rows are sorted lexicographically with nulls last and installed as the build side of a `cudf::distinct_hash_join`. Probing the original rows yields their dense sorted-key indices, so cuDF's `stable_sorted_order` takes its single-column INT32 radix path; representative keys are recovered by a gather at group cardinality. A single nullable or variable-width key can therefore qualify, as can a multi-column key. Non-fatal label construction failures fall back to the original key columns, and an active label path bypasses STRING dictionary encoding.
- **FIRST:** each `FIRST` is a carried column, not a cuDF request. The local batch is `[keys..., partials..., carried...]`: one `MIN` over row positions rides the same `cudf::groupby()` call and every carried column is gathered at its result, so all of them come from one row of each group. With carried columns, at least one key and no cuDF request (every all-`FIRST` list, including `SELECT DISTINCT ON (k)` with carried columns), `local_grouped_aggregate()` runs `cudf::distinct` on `[keys..., carried...]` instead. A `FIRST` over several grouping sets or beside `GROUPING()` is refused by `create_plan(LogicalAggregate&)`
- **Sorted-key path:** a batch whose single fixed-width, null-free key is verified ascending (`keys_ascending()`) is grouped with `cudf::sorted::YES` (run detection and segmented reductions instead of hashing). Unsorted batches keep the hash path. See [Sorted-Key Grouped Aggregation](optimizations.md#sorted-key-grouped-aggregation).
- **Low-cardinality kernel (enabled by default, `enable_low_cardinality_groupby`):** tried before the cuDF path for batches with one or two non-null one-character STRING keys and at most 64 groups, aggregating SUM/COUNT over non-null DECIMAL64 or INT64 inputs. Kernels in `src/cuda/low_cardinality_groupby.cu` map keys to dense group ids and accumulate per thread in registers (up to 8 groups) or per warp in shared memory (9-64 groups), combining into DECIMAL128 sums and INT64 counts. Any other batch returns `nullptr` and takes the cuDF path. See [Low-Cardinality Grouped Aggregation](optimizations.md#low-cardinality-grouped-aggregation).
- **Key members:** `group_idx`, `cudf_aggregates`, `cudf_aggregate_idx`, `cudf_aggregate_input_max_abs`, `cudf_aggregate_is_wide_integer_sum`, `aggregate_slots`, `has_avg`, `has_count_distinct`, `has_first`

### `sirius_physical_dense_count_join` — `DENSE_COUNT_JOIN`
**File:** `src/op/sirius_physical_dense_count_join.hpp`, `src/op/sirius_physical_dense_count_join.cpp`, kernels in `src/cuda/dense_count_join_impl.cu`

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

### Whole-partition window evaluator
**File:** `src/op/window/whole_partition.hpp`, `src/op/window/whole_partition.cpp`

`evaluate_whole_partition_window()` computes `SUM`, `AVG`, `COUNT`, `COUNT(*)`, `MIN` and `MAX` over the whole partition of one table. It runs a `cudf::groupby()` (a reduction when there are no partition columns) and joins the per-partition values back to the input rows with `cudf::distinct_hash_join`, so it does no sort and the results keep the input row order. `SUM` accumulates exactly in `DECIMAL128` and throws on overflow. A floating point `MIN` or `MAX` orders NaN above every other value, as in DuckDB, with the helpers in `src/op/window/float_extremum.hpp`. `AVG` divides that sum by the count with `average_as_double()`, the same rounding as the aggregates.

### Ranking window evaluator
**File:** `src/op/window/ranking.hpp`, `src/op/window/ranking.cpp`

`evaluate_ranking_window()` computes `rank`, `dense_rank` and `row_number` over the window `ORDER BY` of one table. It sorts the partition columns and order keys with `cudf::stable_sorted_order()`, gathers only those key columns, runs a sorted `cudf::groupby` scan with the rank (`MIN` or `DENSE`) or count aggregation, and scatters the `INT64` results back to the input row order. The payload columns are neither sorted nor copied. Each order key has its own direction and NULL placement. With `row_number` only, the order keys are sorted on but not gathered. A window without `PARTITION BY` is one partition: a constant column stands in for the partition key, and the whole input is one bucket on one GPU. The input, the results and the scratch memory must then fit on that GPU, and the input must have fewer than 2^31 rows.

### Cumulative window evaluator
**File:** `src/op/window/cumulative.hpp`, `src/op/window/cumulative.cpp`

`evaluate_cumulative_window()` computes `SUM`, `COUNT`, `COUNT(*)`, `MIN` and `MAX` over the frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` of one table. It sorts like the ranking evaluator, gathers the partition columns and the argument columns, runs a sorted `cudf::groupby` scan with the sum (with overflow detection for `DECIMAL128`), count, min or max aggregation, fills the NULL results of rows with a NULL argument from the closest value before them in the same partition (`groupby::replace_nulls` with `PRECEDING`), and scatters the results back to the input row order. `SUM` accumulates like the whole-partition `SUM` and throws on overflow. `MIN` and `MAX` take integer, decimal, floating point, date and timestamp arguments. Over floating point they order NaN above every other value with the same helpers as the whole-partition evaluator: an argument without NaN is scanned as it is, and an argument with NaN is scanned with NaN replaced by NULL, then NaN is restored from a running NaN flag. A window without `PARTITION BY` is one partition on one GPU. The input, the results and the scratch memory must then fit on that GPU, and the input must have fewer than 2^31 rows.

## Pipeline Breakers (Sirius-Specific)

These operators are injected during pipeline splitting. They don't map to DuckDB logical operators.

### `sirius_physical_partition` — `PARTITION`
**File:** `src/op/sirius_physical_partition.hpp`

Routes rows using explicit key columns and normalization types supplied by the physical planner.
Complex join keys are materialized into columns before this description is constructed.

- **Routing:** hash, single-destination, replication, and resident-device selection use a resolved
  `partition_layout`. Sort range routing retains its separate implementation pending the ordered
  repository work.
- **Order-preserving hash partition:** a `HASH` partition whose single fixed-width, non-floating-point key is ascending uses a stable partition (`src/op/partition/gpu_partition_impl.cpp`), so rows keep their input order within each partition and sorted partials stay sorted for the merge; any other input uses `cudf::hash_partition`. Both assign rows by MurmurHash3 row hash modulo the partition count, so a key lands in one partition whichever path a batch takes. See [Order-Preserving Partition and Sorted Merge](optimizations.md#order-preserving-partition-and-sorted-merge).
- **Fixed layout:** `set_fixed_layout()` supplies the destination count/placement without a sizing
  consumer. Conflicting changes to an installed layout are rejected.
- **Adaptive layout:** explicitly bound input policies resolve count, placement, and consumer
  execution requirements once through `partition_exchange`, before any input claim. PARTITION
  contains no join/groupby type dispatch or sibling-role discovery. A bound input that is not the
  sizing input and sits under a BUILD_FILTER first waits on its own BUILD_FILTER stage, not on the
  sizing input.
- **Publication:** `routed_operator_data` attaches a destination index to each output batch.
  Indexed publication is available on every physical operator; receivers need no sizing API.
- **Placement:** consumer policy validates admitted GPUs and shares the placement with its input
  routers and receivers. Stateful tasks retain the existing device stamps.
- **Projected sizing:** aggregate fanout can latch a projected input size under a `PARTIAL`
  ingress; measurements floor the projection. See
  [data management](data-management.md#runtime-data-size-estimation).
- **Dynamic filters:** a hash-join build-input observer records arrivals, certifies before claims,
  contributes before scatter, publishes after successful scatter, and closes on finalization.

### `sirius_physical_build_filter` — `BUILD_FILTER`
**File:** `src/op/dynamic_filter/sirius_physical_build_filter.hpp`

Publishes an optional equality-key Bloom filter before a RIGHT_SEMI/RIGHT_ANTI join's probe-driven partition sizing. Both joins preserve only build-side rows, so probe rows with no matching build key can be removed safely. The stage requires dynamic filters, `enable_dynamic_filter_multi_partition`, a positive Bloom cap, and admitted INT32 or INT64 keys.

- **Complete input:** a FULL barrier and finished source pipeline let the arrival ledger certify every original batch before any is popped. Certification is independent of downstream partition count; accumulation inside ordinary PARTITION still requires more than one output partition.
- **GPU execution:** scheduled tasks contribute keys to the owning join's publication session and forward the same batches unchanged. The final contribution publishes before stage completion, while partitioning and join strategies remain unchanged.
- **Fallback:** unsupported input or an optional publication failure keeps the data path intact. Consumers never wait on filter channels, and there is no global scan barrier. See [early publication](dynamic-filters.md#early-publication-for-right-semianti-joins).

### `sirius_physical_concat` — `CONCAT`
**File:** `src/op/sirius_physical_concat.hpp`

Groups batches within an existing partition layout and preserves each output partition index.
The caller selects the grouping requirement:

- `concat_all = true`: waits for producer completion, then emits one indexed batch per nonempty input partition, forwarding singletons;
- `concat_all = false`: retains byte-target grouping, including oversized singletons and the final under-target tail.

Group selection also checks available row-count metadata before claiming or transferring input.
Bounded groups stop before exceeding the cuDF row-count limit; a complete partition above the
limit is rejected. Execution checks the actual prepared tables, including representations
without row-count metadata. Input preparation and allocation failures use the standard
partitioned task and executor OOM path.

### `sirius_physical_passthrough_sink` — `PASSTHROUGH_SINK`
**File:** `src/op/sirius_physical_passthrough_sink.hpp`

Terminates one arm of a `UNION`, forwarding every batch unchanged into the downstream `UNION`'s
`"union_{i}"` port. It replaces the join arm's `PARTITION -> CONCAT` chain, which a union needs
neither half of.

- **Emits `pipelineable_operator_data`, not `partitioned_operator_data`.** With no partition and no
  stamped device the task creator selects a device by data locality instead of a partition
  placement, so each batch is consumed on the GPU its scan produced it on.
- **Owns its port name.** `sirius_physical_union::input_port_for` returns a `string_view` into
  `_union_port_label`, which the wiring descriptor and `next_port_info` retain for the life of the
  query.
- Inherits the base task-driver methods: it is single-input, so `UNION`'s fan-in hazards cannot
  arise.

### `sirius_physical_sort_sample` — `SORT_SAMPLE`
**File:** `src/op/sirius_physical_sort_sample.hpp`

Samples input batches to compute P-1 partition boundary rows for range partitioning. Sampling is byte-based: it accumulates batches until `sort_sample_bytes` worth of input is available, rather than a fixed batch count.

Boundary computation follows an explicit `BoundaryState` lifecycle: `NOT_DONE → SCHEDULED → DONE`.
- `get_next_task_hint()` waits in `NOT_DONE` until enough sample bytes are available (or the upstream finishes), then signals READY; it returns `std::nullopt` while `SCHEDULED` so at most one boundary task is in flight.
- `get_next_task_input_data()` (overridden) claims the accumulated sample batches and moves the state to `SCHEDULED`, handing them to `execute()` as a single multi-batch input.
- `execute()` merges the pre-sorted sample batches, computes the boundaries, and moves to `DONE`. If a GPU allocation throws (e.g. OOM), the state stays `SCHEDULED` and the rescheduled task retries with the same input, preventing a duplicate boundary task.
- Once `DONE`, the operator falls back to default scheduling and passes through remaining batches unchanged.

**Publishing the boundaries.** `DONE` makes SORT_PARTITION tasks start on every executor thread of every GPU, so the boundaries must be complete and readable everywhere at that point. `execute()` takes the boundary rows as a device table, from its own sample (`local_boundaries()`) or, in a distributed plan, from the cross-rank hook (`cross_rank_sort_boundaries::make()`, see [distributed-sirius.md](distributed-sirius.md)); `store_boundaries()` copies them to host memory (`cudf::pack` layout) and synchronises the stream before `DONE` is stored. `get_partition_boundaries(space, stream)` then gives each GPU its own device copy, uploaded on first use under a mutex and synchronised before it is returned; `clear_partition_boundaries()` frees each copy with its GPU current. A task never reads a table that lives on another GPU or that another stream may still be writing.

**Partition count.** `compute_sort_partition_sizing()` (same header) decides it from the estimated input size:
- `max_sort_partition_bytes` set: `ceil(data / value)`, nothing else applies.
- `sort_partition_auto_sizing = true` (default), the largest of:
  - memory: `ceil(data / (max_sort_partition_memory_fraction x GPU memory limit))` while the data and its sorted copy fit the GPUs (`2 x data <= limit x GPUs`); beyond that `ceil(data / (0.10 x limit))`, because runs are then spilled before the merge starts and smaller partitions keep more of each merge on the GPU;
  - where partitions take the one-shot merge (`sort_merge_mode = 'one_shot'`, or `auto` with partitions under its one-shot threshold), `ceil(rows / (0.75 x 2^31))`, so that the partition stays a valid cuDF table;
  - with `sort_merge_mode = 'windowed'` on more than one GPU, one partition per executor thread of a GPU (at most 4), as long as partitions stay at or above 2 GiB and the pieces of an input batch at or above 4 MiB;
  - with `sort_merge_mode = 'one_shot'`, `ceil(data / (limit / ((3 + 8 / row width) x 2)))`, because a one-shot merge needs 3 x partition bytes + 8 bytes per row on the GPU.

  In `auto` mode the count is then lowered, if needed, to the largest one whose partitions are at least 1.2 times the one-shot threshold of `merge::choose_route`, unless the one-shot merges of the smaller partitions (one per executor thread) fit the GPU limit together. Several partitions just under the threshold would otherwise each be merged whole at the same time and run out of GPU memory.
- `sort_partition_auto_sizing = false`: the previous rule, `ceil(data / (fraction x GPU memory free at that moment))`.

The decision is logged at INFO as `[sort_sample][sizing] partitions=... rule=...`.

### `sirius_physical_sort_partition` — `SORT_PARTITION`
**File:** `src/op/sirius_physical_sort_partition.hpp`

Range-partitions data according to boundaries computed by SORT_SAMPLE. Links to the sample operator via `_sample_op` and asks it for the boundaries on the GPU of each batch (`get_partition_boundaries(space, stream)`).

Each range slice is published to its original repository partition, retaining indices even when
intermediate ranges are empty. A FULL barrier separates SORT_PARTITION from MERGE_SORT.

In one process SORT_SAMPLE runs in the same pipeline, so the boundaries exist before any batch
arrives. After a distributed plan's range exchange it does not, and the exchange's source holds
its batches until the boundaries are stored (see [distributed-sirius.md](distributed-sirius.md));
a batch that arrives before them is an error, not a pass-through to partition 0.

### `sirius_physical_merge_sort` — `MERGE_SORT`
**Files:** `src/op/sirius_physical_merge_sort.hpp`, `src/op/sirius_physical_merge_sort.cpp`, `src/op/merge/` (`merge_config`, `windowed_merge_input`, `windowed_merge`, `bounded_kway_merge`, `merge_chunk_position`, `detail/`)

Merges the sorted runs of each range partition into the query's output order. Every batch that
SORT_PARTITION publishes to a partition is one sorted run. The operator starts after the
SORT_PARTITION producer finishes (FULL barrier) and skips empty partitions.

#### Routes

A partition takes one of two routes, chosen when its first task is created (`arm_partition()`):

| Route | Task input | Behaviour |
|-------|------------|-----------|
| one-shot | `pipelineable_operator_data` holding every batch of the partition | One task pops the whole partition, uploads it to the GPU and merges it with `gpu_merge_impl::merge_order_by()` (cuDF multi-way merge). Peak memory is the partition plus its merged output. |
| windowed | `merge::windowed_merge_input` holding cursors, not batches | A chain of tasks, one at a time per partition, reads the runs in bounded windows. The runs stay in the input repository on whatever tier they are in. Peak memory is a few windows per run. |

The `sort_merge_mode` setting selects the route (see
[Sort Merge](configuration.md#sort-merge)):

| `sort_merge_mode` | Route |
|-------------------|-------|
| `one_shot` | Always one-shot. Output position is the partition index. |
| `windowed` | Windowed, unless the partition is one of the exceptions below. |
| `auto` (default) | Windowed when any batch of the partition is not on the GPU tier, when the partition's bytes plus its output (the same bytes again) exceed `sort_merge_one_shot_max_fraction` of the device's total GPU memory, or when the one-shot merges of the sort do not fit the device together (below). Otherwise one-shot. |

A partition under the threshold is not merged alone: merge tasks of several partitions run at the
same time, next to the runs of the partitions that wait. `merge::one_shot_peak_bytes()` adds up
what the device then holds: its share of the runs of the whole sort (the bytes of every partition,
divided by the number of GPUs) plus, for the largest partitions under the threshold, one per
merge task the device runs at once, twice the partition's bytes and 8 bytes per row for the merge
temporaries and the output. When that exceeds the device's total GPU memory, the partitions under
the threshold are windowed too. So every partition of a sort larger than the device is windowed,
whether or not its own runs are on the GPU when it is armed, and the route of a partition does
not depend on which runs a downgrade happened to move. `sort_merge_disabled_rules` bit 512
switches this off.

In `auto` and `windowed` mode these partitions take the one-shot route:

- a fetched column (a sort key or an output column) is nested (LIST or STRUCT), or any column is
  nested and the partition has more runs than the fan-in cap, because an intermediate pass rewrites
  every column;
- late materialization is installed on the operator (`has_late_materialization()`);
- no batch exposes its column types (for a compressed batch, its header cannot be parsed), a batch
  has no row count, the partition has zero rows, or a sort key is not a plain column reference.

A one-shot partition in these modes sets `set_materialize_on_gpu(true)` on its input, which
overrides the port's residency policy for that task (see
[Input Residency and Downgrade Priority](memory-management.md#input-residency-and-downgrade-priority)).
An operator that never receives settings through `set_merge_settings()` behaves as `one_shot`;
the plan generator installs the query's settings.

The sizes of a windowed partition (`merge::derived_merge_config`: window rows, output target,
fan-in, one-shot fraction) are derived once at arming from the settings, the device's GPU memory and
the partition's run count and row width. `set_device_budget_provider()` supplies the GPU memory and
task concurrency. The plan generator installs `make_device_budget_provider()`, which reports the
GPU memory space holding most of the partition's bytes (the smallest GPU space when no run is on a
GPU) and the configured GPU executor thread count. The GPU memory it reports is the space's free
bytes plus the bytes reserved by tasks in flight (`get_available_memory() +
get_total_reserved_memory()`): the derivation already divides it among the concurrent tasks, so
partitions of a device get the same sizes whichever task is running when each is armed. Bytes
allocated outside a reservation (resident batches, kept window buffers) stay subtracted. A directly
constructed operator has no
provider: it uses the majority space (or the last one it saw, or 8 GiB with a warning when it has
seen none) and the default GPU executor width.

Two rules apply to that budget at arming:

- **Floor.** The derivation uses at least a quarter of the device's GPU memory
  (`merge::kBudgetFloorFraction`). The runs are spilled first, so on a full device the free bytes
  at arming are whatever a downgrade had freed at that instant; without the floor the same query
  ran with windows several times smaller or larger from one execution to the next.
- **Divisor.** The budget is divided by the partitions that can be open, not by the executor
  width: `clamp(partitions that hold batches or are armed, 1, executor threads)`. The same number
  is the open-partition cap (below), so two partitions under sixteen threads are sized, and
  opened, as two.

#### Continuation tasks

`_partitions` maps each partition to a `partition_merge_state`, guarded by `_drain_mutex`. It holds
one `run_state` per run, the next output chunk index and an `in_flight` flag.

- **Cursors.** A run is a list of batch ids in key order (`merge::merge_run`): one batch, or the
  batches of an intermediate run. A `merge::merge_input_cursor` is the run plus the next unread
  `(batch, row)`. Cursors hold no device state.
- **Peek without pop.** Arming reads batch ids, row counts, sizes, tiers and column types through
  `get_batch_ids()` and `get_data_batch_by_id()`. The batches are not popped and the task does not
  subscribe them, so they stay spillable while tasks are queued and running. Zero-row batches are
  popped at arming.
- **Arming never waits for a batch lock under `_drain_mutex`.** `on_input_batch_pushed()` records
  each run's rows and bytes when it arrives, and arming uses `try_to_read_only()`. A run that is
  locked by a tier move counts as off the GPU. When such a run was never sized, or no readable
  batch exposes the schema, the partition stays unarmed (`arm_deferred`): a committing task makes
  it armable again, and when no task is in flight the task-creation thread waits for that batch
  outside the mutex and arms again.
- **One task per partition at a time.** Each task starts from the cursors the previous one
  committed. Different partitions run in parallel. `get_next_task_hint()` returns READY only when
  some partition has runs and no task in flight, and no hint otherwise.
- **One positioned chunk per task.** A task runs one `merge::windowed_merger::run()`, which
  returns one table of about `output_target_bytes`. A final-pass task fetches the output columns
  plus the keys they lack, drops the key-only columns and publishes one batch.
- **One merger per task chain.** The `windowed_merger` is kept in the partition's state between
  tasks (`partition_merger`: the merger, its cursors and the run ids they belong to), so rows a
  task read into a window buffer but did not emit are not read again by the next task. A task
  moves it out under `_drain_mutex`, runs it without the mutex, and puts it back at commit. A new
  merger is built when the task's GPU memory space or window sizes differ (an OOM shrink), when the
  run set changes (an intermediate pass starts or ends), and after any failed task. A run that
  ends before the others does not force a rebuild. Between tasks the merger holds references to
  the unread batches and its window buffers stay allocated on the GPU
  (see [Windowed Merge Memory](memory-management.md#windowed-merge-memory)).
- **Open partition cap.** A windowed partition is *open* while it has a task in flight or keeps
  its merger. A kept merger's window buffers are in no repository, so they cannot be spilled, and
  the window budget (`derive_merge_config()`) assumes one merger per concurrent task of a device.
  `get_next_task_input_data()` therefore plans a task for a partition that holds no merger (its
  first task, or the first task of a new pass) only while fewer partitions are open than the
  `device_budget::concurrent_tasks` the partition was sized with (`at_open_partition_cap()`);
  otherwise it skips the partition, and `get_next_task_hint()` does not report it. Open partitions
  are always served, so they finish, and the commit that closes one schedules the operator again,
  which opens the next waiting partition. The count is per device, like the budget: an open
  partition counts on the device of its merger or running task, and one whose first task has not
  started counts on every device, because it may land on any. Arming is not delayed (it allocates
  nothing on the GPU), and one-shot partitions are neither counted nor held back: they keep no
  merger and their inputs are covered by their task's reservation.
- **Positions.** The output position is `(partition << 32) | chunk`
  (`merge::make_merge_chunk_position()`), which orders chunks by partition, then by chunk. It
  throws when either part reaches 2^32. In `auto` and `windowed` mode one-shot partitions use the
  same encoding with chunk 0. In `one_shot` mode the position is the partition index. In a
  distributed plan the position travels with the batch through the gather to rank 0
  (see "Order through the final gather" in [distributed-sirius.md](distributed-sirius.md)).
- **Commit → pop → `schedule(this)`.** After the merge returns, the task calls
  `windowed_merger::rebase_consumed()`, which drops the merger's references to the batches read to
  the end and rebases its cursors to the remaining ones. It then commits under `_drain_mutex`: it
  pops those batches (which frees them), copies the cursors to the partition's runs by run id,
  advances the chunk index, returns the merger to the partition and clears `in_flight`. It
  then calls `task_creator::schedule(this)` when some partition can produce a task, because nothing
  else asks an operator for its next task when one of its own tasks finishes (see
  [Self-Rescheduling Operators](task-creator.md#self-rescheduling-operators)).
- **Retry.** The partition's cursors change only at commit. A task that fails (for example on OOM)
  leaves them and `in_flight` unchanged and destroys the merger it held, which frees its window
  buffers. The rescheduled task builds a new merger and replays from the same cursors,
  possibly with smaller windows (see [Shrink Before Retry](memory-management.md#shrink-before-retry)).
  The operator does not check for progress itself: `windowed_merger::run` returns no rows only
  when every cursor is exhausted and throws `internal_exception` on a round that makes no progress.
- **Counters.** `merge_sort_counters` counts the partitions armed on each route, the windowed and
  intermediate tasks, the mergers built and the summed `merge::merge_round_stats`. `counters()`
  reads one operator's; `process_counters()` / `reset_process_counters()` are process-wide, for
  tests and diagnostics (they are not per query).

#### Bounded k-way merge round

`windowed_merger::run()` repeats rounds until the accumulated output reaches
`output_target_bytes` (counted with `cudf::table::alloc_size()`, on the host) or every cursor is
exhausted, then concatenates the rounds into one table.
With `n` = `window_rows`, each round does the following:

1. **Windows.** Every run that is not exhausted gets a window of at most `n` rows of its current
   batch, starting at the cursor. A window never spans two batches of a run.
   - A `gpu_table_representation` batch in the task's memory space is windowed as offset column
     views over a read-locked table. Nothing is copied. The lock is held until the batch is
     finished or `run()` returns.
   - Any other batch is read into a `detail::window_buffer`: two preallocated slots per run, of
     `min(n, largest remaining batch of the run)` rows, with each STRING column's chars sized from
     that column's own bytes per row in the batch metadata. Consuming rows only moves a start
     offset. When fewer rows are left than half of what the last refill held, `consume()` copies
     the unread tail to the front of the other slot and `refill()` appends rows after it with
     `cucascade::read_rows_into`, under a read lock held only for that refill (see
     [Row-Range Readers](memory-management.md#row-range-readers)). A row is copied at most once.
     A row that does not fit a STRING slot grows the slots (`window_row_too_large_error` is
     handled by the merger). The buffer is freed when the run ends or its batch is found on the
     GPU.
   - A window's `null_count()` is an upper bound (exact only when 0): the kernel reads the masks.
2. **Safe bound.** A window is *truncated* when its run has unread rows beyond it. The round emits
   `target = min(n, r_safe, Σ window rows)` rows, where `r_safe` is the smallest, over the
   truncated windows, of the rank of the window's last row in the combined order, plus one
   (`detail::safe_target()`). Every emitted row therefore orders at or before every unread row, so
   the emitted prefix is final.
3. **Co-rank.** `detail::corank_split()` finds, for an output rank `r`, how many rows of each
   window are among the first `r` rows of the merged order. It is evaluated at every tile boundary
   (`detail::bounded_merge_tile_rows()`: 2048 output rows, or 4096 with the dense launch shape
   below) and at `target`. With a specialised comparator and a fan-in of at most 32 one warp per
   rank keeps a lane per input and its bounds in registers; otherwise the general kernel runs.
4. **Tile merge.** `detail::tile_merge()` merges each tile independently and writes an
   `(input, row)` gather map of `target` entries. A tile is split into sub-tiles, one thread
   each with a tournament tree in shared memory; the split is chosen from the tile count, the
   device's SM count and the shared memory the fan-in needs: a block uses at most 40 KiB, and at
   most 32 KiB when there are more blocks than SMs, because above that fewer blocks fit an SM
   (`detail::tile_merge_launch_shape()`). The thread density is a policy
   (`detail::tile_merge_policy_for()`): 256 merge threads per SM in blocks of at most 64 threads
   by default, and 384 per SM in blocks of up to 128 with 4096-row tiles where that was measured
   to be faster (a specialised first key, a fan-in of at most 32, compute capability 10.x). It
   accepts at most `kMaxKwayMergeInputs` (4096) windows; the configured fan-in is clamped to that.
5. **Gather.** `detail::gather_kway()` copies the selected rows of all fetched columns, with null
   masks and STRING data, into the round's output table. The null counts, the STRING sizes and
   the consumed counts come back to the host in one read.
6. **Advance.** The co-ranks at `target` are the rows consumed from each window. Cursors advance by
   them and the window buffers drop the consumed rows.

`bounded_kway_merge()` never materializes the `k × n` window rows, and it breaks ties by window
index and then row, so the result matches a stable merge. Nested key types are not supported.
The key comparator copies the column views of all windows to the device in one asynchronous
copy, so a call waits on the stream twice (the target, then the gather's read-back), three times
with a STRING column, whatever the fan-in.

When the key is a single fixed-width column (integers, BOOL8, floating point, timestamps,
durations, decimals), the three kernels compare it through `kway_fixed_key_device_view<T>`: one small entry
per window (data pointer, null mask, bit offset), a load of the storage type and one unsigned
compare of the normalised value, instead of the column views and the type dispatch. The order is
the generic comparator's, tie order included (nulls by the null order, NaN equal to NaN and after
every value, -0.0 equal to 0.0). Every other key (STRING, several key columns) uses the generic
comparator; `specialized_keys=2` also specialises a fixed-width first column of a multi-column
key, with a tie on it continuing in the generic comparison from the second column on (measured
on INTEGER + BIGINT + VARCHAR keys: 2 to 7% faster at fan-in 4, 8 and 64, 2% slower at 16, so it
is off by default).
`detail::kway_tuning` (environment `SIRIUS_KWAY_TUNING`, for example
`specialized_keys=0;warp_corank=0;baseline_shape=1`) switches the specialised view, the one-warp
co-rank and the launch-shape rule off for A/B measurements; no switch changes a result.

#### Fan-in cap and intermediate runs

One task reads at most `fan_in` runs: the largest count whose windows fit the task budget with `n`
at its 65,536-row floor, capped by `sort_merge_max_fan_in` and never below 2.

- With more runs than the cap, `merge::plan_pass()` selects the smallest runs by bytes for an
  **intermediate pass**: `fan_in` of them, except that with `R` runs and `(R - 1) mod (fan_in - 1)
  != 0` the first merge takes `((R - 2) mod (fan_in - 1)) + 2`. After that one short merge every
  later merge and the final pass read exactly `fan_in` runs (the k-ary Huffman plan, which
  writes the fewest rows; without it the shortfall lands in the final pass, behind a merge that
  rewrote nearly every row). Its tasks fetch every column, add their output batches to the same
  partition of the input repository, and publish nothing downstream.
- The selected runs stay in the pass until they are exhausted. The batches the pass wrote then
  join the partition as one run, and the next pass is planned.
- With at most `fan_in` runs left, tasks are **final-pass** tasks and publish positioned chunks.
- Intermediate runs are ordinary repository batches: they can be spilled like the original runs.

#### Convert-once fallback

A batch whose representation has no row-range reader, or whose read throws
`row_range_unsupported_error`, is converted once to the GPU tier in the task's memory space
(`detail::convert_run_batch_to_gpu()`), under the batch's exclusive lock. It stays in the
repository and is then windowed as a GPU batch. A Simpatico-compressed batch takes this path only
when its reader refuses it (a projection, a filtered view, or a plan that cannot be decoded by
chunk). The task input adds the largest batch without a reader to its memory estimate; a batch
whose reader refuses it only at read time is not in the estimate (see
[Window reservations](memory-management.md#window-reservations)). If a downgrade moves the batch
again before the merger re-locks it, the merger retries, three attempts in all, then throws.

#### Statistics line

Each MERGE_SORT logs one line per query at INFO, built by `merge::format_merge_sort_stats`
(`src/op/merge/merge_sort_stats_log.hpp`):

```text
[merge_sort][stats] op=<id> windowed_partitions=… one_shot_partitions=… windowed_tasks=… intermediate_tasks=… mergers_built=… rounds=… rows_out=… round_rows_min=… round_rows_max=… refill_rows=… refill_bytes=… refill_bytes_host=… refill_bytes_disk=… refill_bytes_compressed=… runs_read_compressed=… window_growths=… compacted_rows=… fallback_conversions=… shrinks=…
```

- It is logged at the commit of the operator's last task. An operator that never gets there (no
  input, or a pipeline that finished early under a LIMIT) logs it from `on_finalize_operator()`
  with the counters so far. A flag under `_drain_mutex` keeps it to one line.
- `refill_bytes_host`, `_disk` and `_compressed` split `refill_bytes` by where a refill read
  from. A GPU-tier batch read through a reader counts in `refill_bytes` only.
- `runs_read_compressed` counts (run, windowed task) pairs, not distinct runs.
- `shrinks` counts windowed inputs that shrank their window for an OOM retry.
- The keys and their order are the contract with `test/sort_performance/bench_metrics.py`
  (`test_bench_metrics_source_formats.py` fails when they drift): append new keys at the end.

### `sirius_physical_grouped_aggregate_merge` — `MERGE_GROUP_BY`
**File:** `src/op/sirius_physical_grouped_aggregate_merge.hpp`

Merges grouped aggregate results from multiple partitions. Drains one partition per task, similar to MERGE_SORT.

- **FIRST:** `merge_grouped_aggregate()` picks one partial row per group with `MIN` over row positions and gathers the carried columns from it, or runs `cudf::distinct` when there is no cuDF request. `partials_are_output()` decides whether the merged partial layout is already the declared output; a list whose `FIRST`s all follow its other aggregates passes through, and any other mix is reordered into the declared order

**Row floor on the partition count.** The merge concatenates its partition's input, and a cuDF column holds fewer than 2^31 rows. Its `get_partition_strategy` raises the byte-based count to at least `ceil(rows / 2^30)` partitions (saturating at `INT_MAX`), using the row count that PARTITION reports next to the byte total (`partition_sizing_input::total_rows`, measured from batch metadata or projected at the received row width). This targets at most 2^30 rows per partition on average; hash skew and row-count estimation can still leave an individual partition over cuDF's limit. The floor is a sizing safeguard for configurations with large `hash_partition_bytes` targets. Before it existed, a 122 GB, 3.4 billion-row merge input at a 1 TiB target failed with "Total number of concatenated rows exceeds the column size limit".

**Sorted merge.** With a single fixed-width, non-floating-point group key, the merge orders its input partials by their first key before concatenating and verifies the result with `keys_ascending()`; when it is sorted, the merge uses cuDF's sorted-keys group-by. Otherwise it keeps the hash merge.

### `sirius_physical_ungrouped_aggregate_merge` — `MERGE_AGGREGATE`
**File:** `src/op/sirius_physical_ungrouped_aggregate_merge.hpp`

Merges ungrouped aggregate results from multiple partitions.

An empty input may contain zero batches or a batch with zero rows. After its producer finishes, the merge schedules exactly one task even if no partial aggregates arrived. With no batches, `input_free_operator_data` captures the task reservation's GPU space and the merge constructs one final row: COUNT variants return zero, and the other supported aggregates return typed NULLs. Normal input still merges all partials in one task. Pending finalization keeps the source non-exhausted until its task descriptor is handed out; ordinary task counters then keep execution and retries in flight. Grouped aggregates do not synthesize a group for empty input.

### `sirius_physical_top_n_merge` — `MERGE_TOP_N`
**File:** `src/op/sirius_physical_top_n_merge.hpp`

Merges local top-N results from multiple partitions. The final output batch is assigned position zero.

## CTE / Delim Join Operators

### `sirius_physical_cte` — `CTE`
**File:** `src/op/sirius_physical_cte.hpp`

Materializes Common Table Expression results into a `ColumnDataCollection` for later scanning by CTE_SCAN operators.

- **Key members:** `working_table`, `cte_scans`, `ctename`, `table_index`

### `sirius_physical_left_delim_join` — `LEFT_DELIM_JOIN`
### `sirius_physical_right_delim_join` — `RIGHT_DELIM_JOIN`
**File:** `src/op/sirius_physical_delim_join.hpp`

Handle correlated subqueries via duplicate elimination. Wrap an inner join (hash or nested loop) and embed a `sirius_physical_grouped_aggregate` for DISTINCT on duplicate-eliminated columns.

- `join` — the actual join operator
- `distinct` — embedded aggregate for duplicate elimination
- `delim_scans` — downstream scan operators that receive the deduplicated data

### `sirius_physical_partition_consumer_operator`
**File:** `src/op/sirius_physical_partition_consumer_operator.hpp`

Base for operators whose tasks need partition placement and the admitted GPU set. Sizing targets
and decisions belong to concrete consumers. Indexed batch publication is inherited from
`sirius_physical_operator`, so generic receivers do not need this base.

## Result Operators

### `sirius_physical_result_collector` / `sirius_physical_materialized_collector` — `RESULT_COLLECTOR`
**File:** `src/op/sirius_physical_result_collector.hpp`

Final sink that materializes query results into a `ColumnDataCollection`. The GPU executor checks for this operator type to determine query completion.

The materialized collector builds local collections in parallel. For positioned input, it combines
them by ascending position in `get_result()`, independently of task completion order.

### `sirius_physical_discard_sink` — `DISCARD_SINK`
**Files:** `src/op/sirius_physical_discard_sink.hpp`, `src/op/discard_copy_function.hpp`

Runs a query without returning or keeping its rows. It exists for benchmarks and validation
(`test/sort_performance/`), and is reached only through the `sirius_discard` COPY format:

```sql
COPY (SELECT ... ORDER BY k) TO '/dev/null' (FORMAT sirius_discard);
COPY (SELECT ... ORDER BY k) TO 'ignored' (FORMAT sirius_discard, check_order true);
```

The path is never opened. The result is COPY's: one BIGINT row, the row count. Every other COPY
format stays on DuckDB, and with `gpu_execution = false` (or after a runtime fallback) DuckDB's
copy function for the format counts and drops the rows on the CPU.

- **Plan.** `query → DISCARD_SINK → UNGROUPED_AGGREGATE(SUM) → MERGE_AGGREGATE → PROJECTION`. The
  sink emits one row per task (the task's row count, at the input's position), so its declared
  output type is one BIGINT, not the schema it drops (`input_types()`).
- **Streaming input.** `input_barrier_for` returns `PIPELINE` for any producer, so each batch is
  dropped as it arrives and the sort's output never accumulates.
- **Counts batches where they are.** `input_residency_policy` returns
  `{.materialize_inputs_on_gpu = false, .downgrade_priority = -100}`: a batch that spilled while
  it waited is counted from its metadata, not read back, and the sink's repository is the last
  one of the query the downgrade sweep visits. The count row is allocated in the task's GPU
  memory space, which the sink's task input records at `prepare_for_processing`.
- **`check_order`** is a validation mode. It is on only when the query's root is an ORDER BY
  whose keys are all plain output columns; otherwise the planner logs the reason at DEBUG and
  leaves it off. With it the input is materialized on the GPU: each batch is checked with
  `cudf::is_sorted`, and the first and last key rows of each task are copied to the host. When
  the pipeline finishes (`on_finalize_operator` → `finish()`), the tasks are walked in position
  order and every seam is compared with a host comparator that follows cudf's row order.
- `finish()` logs `[discard_sink] rows=… bytes=… order_violations=…` at INFO, once.

### `sirius_physical_empty_result` — `EMPTY_RESULT`
**File:** `src/op/sirius_physical_empty_result.hpp`

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
| UNORDERED_STREAMING_LIMIT | Relational | Inline claims, retry-safe slices, early pipeline completion |
| STREAMING_LIMIT | Relational | Serial slice claims after FULL input barrier |
| REPLICATE | Relational | `gpu_replicate_impl::plan_slices()` + `materialize()`; one `cudf::gather` per output batch |
| ORDER_BY | Sort | `gpu_order_impl::local_order_by()` |
| TOP_N | Sort | Order + limit |
| SORT_SAMPLE | Sort | Sample + boundary computation |
| SORT_PARTITION | Sort | Range partition by boundaries |
| MERGE_SORT | Sort | `gpu_merge_impl::merge_order_by()` (one-shot), `merge::windowed_merger` (windowed) |
| UNGROUPED_AGGREGATE | Agg | `gpu_aggregate_impl::local_ungrouped_aggregate()` |
| HASH_GROUP_BY | Agg | `gpu_aggregate_impl::local_grouped_aggregate()` |
| MERGE_AGGREGATE | Agg | Merge ungrouped partitions |
| MERGE_GROUP_BY | Agg | Merge grouped partitions |
| HASH_JOIN | Join | `cudf::{inner,left,right,outer}_join()`, `cudf::distinct_hash_join`, or `cudf::{filtered,mark}_join` (MARK) |
| DENSE_COUNT_JOIN | Join+Agg | Fused dense/sparse COUNT over an outer join |
| NESTED_LOOP_JOIN | Join | Fallback nested loops |
| LEFT_DELIM_JOIN | Join | Correlated subquery wrapper |
| RIGHT_DELIM_JOIN | Join | Correlated subquery wrapper |
| UNION | Set op | N-ary identity fan-in, no data touched; distinct `UNION` is a DISTINCT over it; also merges the tagged inputs of `EXCEPT ALL` / `INTERSECT ALL` |
| PARTITION | Pipeline | Hash/range partitioning |
| BUILD_FILTER | Pipeline | Complete-build Bloom publication; forwards batches unchanged |
| CONCAT | Pipeline | Partition reassembly |
| PASSTHROUGH_SINK | Pipeline | UNION arm terminator; forwards batches unpartitioned |
| MERGE_TOP_N | Pipeline | Merge per-partition top-N |
| CTE | CTE | Materialize to ColumnDataCollection |
| RESULT_COLLECTOR | Result | Final result materialization |
| EMPTY_RESULT | Result | Empty result set |
| STREAMING_SINK | Result | Terminal operator of a streaming fragment |
| DISCARD_SINK | Result | Counts and drops each batch (`FORMAT sirius_discard`); optional `cudf::is_sorted` order check |
