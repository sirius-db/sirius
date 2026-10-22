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

**Current mechanism:** `discard_pending_tasks(query_id)` disposes of queued requests/callables. The runtime owner then waits for work leases before `reset(query_id)` releases creator state and lookahead hints. Other queries remain active. See [Query lifetime and retirement](query-lifecycle.md).

**Code path:** `src/creator/task_creator.cpp` — `discard_pending_tasks()` and `reset()`

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

**Mechanism:** `sirius_physical_sort_sample` overrides `get_next_task_input_data()` so the boundary task receives the full multi-batch sample the hint waited for. Sampling is byte-based: it pulls batches until `sort_sample_bytes` is reached (or upstream finishes), then merges the already-sorted sample batches with `gpu_merge_impl::merge_order_by` and computes boundaries from the merged run — no concatenate-and-full-sort. When the sample contains the complete upstream input, its actual bytes determine the partition count; partial samples still extrapolate from estimated cardinality. Boundary scheduling uses an explicit `NOT_DONE -> SCHEDULED -> DONE` state machine: `get_next_task_input_data()` claims the sample and moves to `SCHEDULED`, `get_next_task_hint()` returns `nullopt` while `SCHEDULED` so no duplicate boundary task is created, `execute()` moves to `DONE`, and OOM resets to `NOT_DONE` for retry. After boundaries are computed the operator falls back to single-batch passthrough.

**Code path:** `src/op/sirius_physical_sort_sample.cpp` — `get_next_task_input_data()`, `get_next_task_hint()`, `execute()`; `src/planner/sirius_physical_plan_generator.cpp` — wiring `sort_sample_bytes` into SORT_SAMPLE

**Config:** `sort_sample_bytes` (default: 512 MB), settable via YAML and the `sort_sample_bytes` SET option

### Merge Pipeline Fusion (PR #1190)

**Motivation:** A `MERGE_GROUP_BY` / `MERGE_TOP_N` operator normally forms its own terminal pipeline whose only downstream is a streaming chain ending at the sink (typically `RESULT_COLLECTOR`). That standalone pipeline costs an extra task launch and a repository round-trip for no additional parallelism.

**Mechanism:** At plan time `mark_fusable_merge_pipelines()` walks parent pointers from each merge to the first downstream sink. When that path is unary and streaming and the sink accepts a fused input, the merge is marked to fuse. The merge's `build_pipelines()` override then adds itself as an intermediate operator to the downstream pipeline and recurses into its child (which still cuts the upstream boundary), instead of opening a new terminal pipeline. Total-input structural sinks (`ORDER_BY`, `TOP_N`, an outer `GROUP BY`) are fusable because they already buffer their full input. Excluded: join/CTE/delim terminals, partition sinks, delim-owned distinct merges, and `MERGE_AGGREGATE` (ungrouped aggregate).

**Code path:**
- `src/planner/sirius_physical_plan_generator.cpp` — `mark_fusable_merge_pipelines()`, `terminal_sink_supports_fusion()`
- `src/op/sirius_physical_grouped_aggregate_merge.cpp`, `src/op/sirius_physical_top_n.cpp` — `build_pipelines()` overrides
- `src/sirius_engine.cpp` — invokes marking after parent pointers are refreshed

**Policy:** Sirius applies eligible merge fusion automatically. See
[physical-plan-generation.md](physical-plan-generation.md) → Merge fusion for pipeline-shape
details.

### Task Creator Look-Ahead (PR #1174)

**Motivation:** With demand-driven (`active`) task creation, a drained task queue leaves GPU workers idle even when not-yet-activated scans could already be producing work.

**Mechanism:** The task creator retains a per-query `lookahead_queue` of candidate operators (built at query start from the plan's scan operators after the first, cleared on query drain/reset). When an engine-controlled policy selects the internal `request_type::lookahead` primitive and the task scheduler finds its task queue empty, `schedule_lookahead(device_hint)` emits one speculative request for the next not-yet-activated operator, warming scans up one task at a time. The manager loop creates a single task per look-ahead request rather than draining the source. See [task-creator.md](task-creator.md).

**Code path:** `src/creator/task_creator.cpp` — `schedule_lookahead()`; `src/pipeline/task_scheduler.cpp` — empty-queue trigger; `src/creator/config.hpp` — `request_type`

**Policy:** internal. The current shipped policy is active and demand-driven;
look-ahead is not a user-selectable YAML setting.

### Two-Sleep-Site Scheduler Loop (PR #1478)

**Motivation:** The management event loop slept only on the task-request channel, but the downgrade executor returns extracted pipeline tasks by pushing directly into the scheduler's task queue — an RAII return that emits no channel event. With every executor parked, runnable tasks could sit next to ready devices forever (deterministic hang at SF1000 TPC-H q18 with host-pinned tables).

**Mechanism:** Each pass of `management_eventloop()` tops up the two things the matcher needs, sleeping at most once per missing ingredient: it blocks on the request channel only when no device is parked (a busy executor will post `device_ready`), and sleeps on the task queue's own `multi_index_priority_queue::wait()` when the queue is empty and a device is parked (a `task_available` wake can carry no ready device, and a per-query drain can empty the queue), issuing `schedule_lookahead` first. The queue's condvar hears every push — `schedule()` or a silent direct return — so any producer that bypasses `schedule()` is tolerated by construction. `wait()` blocks until non-empty without extracting, keeping device-preference selection via `try_pop_from`; `push()` uses `notify_all` because mixed waiter kinds share one condvar. Both sleeps are interrupted by `stop()`.

**Code path:** `src/pipeline/task_scheduler.cpp` — `management_eventloop()`; `src/exec/multi_index_priority_queue.hpp` — `wait()`

**Config:** none

### Single Plan Build per Query (PR #1548)

**Motivation:** `OnFinalizePrepare` must build the full Sirius physical plan to decide whether a query is GPU-supported; discarding that plan and rebuilding an identical one at execution doubles plan-construction work on every transparent query.

**Mechanism:** `OnFinalizePrepare` hands the validated plan to the `PhysicalSiriusExecution` operator together with the pin-registry epoch it was built under. The first `GetData` consumes it (one-shot — the move empties the slot) while the epoch still matches; a pin or unpin landing between the finalize and execution windows discards it, and the plan is rebuilt against the current registry. Re-executions of the same prepared operator rebuild from the retained logical-plan template, or re-plan from the SQL string when the template cannot be copied, so data-drift and rebind semantics are unchanged.

**Code path:** `src/sirius_context.cpp` — `OnFinalizePrepare` plan handoff; `src/transparent/physical_sirius_execution.hpp` — `validated_sirius_plan_`, `validated_plan_pin_epoch_`; `src/transparent/physical_sirius_execution.cpp` — epoch check and one-shot consumption in `GetDataInternal`

**Config:** none

### Cardinality Estimates Preserved in the Plan Copy (PR #1945)

**Motivation:** Transparent execution plans from a copy of DuckDB's optimized plan, and `LogicalOperator::Copy()` is a serialization round trip that carries neither `estimated_cardinality` nor `has_estimated_cardinality`. Every node of the copy recomputes its estimate — scans as whole tables, joins from their largest child — so filtered inputs look like full tables to the Sirius planner, flipping hash-join mode decisions and mis-sizing the PARTITION and CONCAT operators ahead of `DELIM_GET`/`CTE_SCAN` build sides.

**Mechanism:** `copy_logical_plan()` wraps `Copy()` and walks the original and the copy in lockstep with `copy_cardinality_estimates()`, copying both fields onto every node; a subtree whose type or child count differs is left untouched and logged at debug level. Every plan copy on the transparent path goes through it — the optimizer hook's capture, the finalize-time validation copy, and the execute-time rebuild — so DuckDB's estimates survive to each execution. The explicit `CALL gpu_execution(...)` path does not copy the plan and is unaffected.

**Code path:** `src/transparent/sirius_optimizer_extension.cpp` — `copy_logical_plan()`, `copy_cardinality_estimates()`

**Config:** none

## Operator-Level Optimizations

### Multi-Literal LIKE SWAR Fast Path (PR #1610)

**Motivation:** cuDF's thread-per-row, byte-at-a-time LIKE matcher is load-instruction-bound on wide text columns. The TPC-H q13 predicate `%special%requests%` measured about 6x faster in the specialized kernel, reducing the query's LIKE work without changing SQL semantics.

**Mechanism:** Constant patterns of the form `%lit1%lit2%...%litN%` are classified and compiled once per query and pattern value, then shared immutably across task-local evaluators. The CUDA kernel scans aligned 64-bit words, uses SWAR digram masks to find candidates, and verifies complete literals in order. Unsupported pattern shapes and ineligible column layouts fall back to `cudf::strings::like`; NOT LIKE is fused into the output write.

**Code path:**
- `src/expression_evaluator/specializations/function.cpp` — query-cache lookup and dispatch
- `src/cuda/sirius_like_multiliteral.cu` — classifier, query-cache implementation, compiled descriptors, and SWAR kernel
- `src/expression_evaluator/like_multiliteral.hpp` — launcher and input contracts

**Config:** `like_swar_fastpath` (default: `true`, connection-local). The query snapshots this setting at engine initialization. Supported Sirius ingestion must supply valid UTF-8 for DuckDB VARCHAR/cuDF STRING input; the hot path treats that as a precondition and does not add a redundant validation scan.

### Adaptive Join BUILD_PROBE Mode (PR #423)

**Motivation:** For small build-side datasets, building the hash table once and probing many times is more efficient than the standard multi-partition Cartesian product approach.

**Mechanism:** `compute_hash_join_partition_strategy()` selects BUILD_PROBE mode when:
- `num_partitions <= num_gpus` (one hash table per partition, at most one partition per GPU; reduces to a single partition when `num_gpus == 1`)
- per-partition average build side < `max_build_hash_table_bytes`, foldable to a single batch per partition
- the join is neither RIGHT-family nor `MIXED_JOIN`

In BUILD_PROBE mode, each partition's first task builds a `cudf::hash_join` hash table and caches it; subsequent tasks for that partition only probe.

**Broadcast small build tables (multi-GPU):** when the build side is small (`< small_table_bytes`), the PARTITION operator replicates it to every GPU (one partition placed on each GPU, `_broadcast` flag) instead of funneling the build to one GPU, so every GPU builds its own hash table and probes locally. Build-only slots are discarded once the probe side finishes.

**Code path:** `src/op/sirius_physical_hash_join.cpp` — `compute_hash_join_partition_strategy()`, `get_partition_strategy()`; `src/op/sirius_physical_partition.cpp` — broadcast slot routing by the join's partition placement

**Config:** `max_build_hash_table_bytes` (default: 500 MB)

### COUNT DISTINCT Optimization (PR #414)

**Motivation:** Exact COUNT(DISTINCT) requires expensive deduplication.

**Mechanism:** Uses cuDF's `COLLECT_SET` aggregation for distinct value collection, with `MERGE_SETS` in the merge phase. For multi-column DISTINCT, synthesizes struct columns from multiple input columns.

**Code path:**
- `src/op/aggregate/gpu_aggregate_impl.cpp` — `local_grouped_agg()` COLLECT_SET handling

### Label-Encoded Group Keys for COUNT DISTINCT (PR #1375)

**Motivation:** cuDF's sorted groupby finds group boundaries via `stable_sorted_order(keys)`, which takes the radix fast path only for a single key column. A multi-column group key falls to lexicographic merge sort — on TPC-H q16 at SF1000, 308.7 ms for 118.8M rows, 92% of the aggregate. The pre-existing dictionary-encode path narrows the comparators but leaves multiple key columns, so the single-column gate is never reached.

**Mechanism:** the key table collapses into one dense INT32 label; the groupby sorts on that label, and representative keys are recovered with a gather at group cardinality. Sirius first computes distinct keys with equal nulls and all NaNs equal, then sorts those keys lexicographically with nulls last. The sorted distinct table is the unique build side of a `cudf::distinct_hash_join`; probing the original rows returns one build-row index per input row, which is already the required dense label. Gated on: a COLLECT_SET aggregation, at least 1,048,576 input rows, a non-nested key that is not already a single null-free fixed-width column, and an HLL estimate below 1% of rows. A single nullable or variable-width key can qualify. Non-fatal label construction failures fall back to the original key columns; an active label path bypasses STRING dictionary encoding.

**Code path:** `src/op/aggregate/gpu_aggregate_impl.cpp` -- `gpu_aggregate_impl::local_grouped_aggregate()`, label path (`use_label_keys`)

**Config:** none (thresholds are internal). TPC-H SF1000 GB300 measured workload: q16 0.490 s -> 0.298 s.

### Fused Dense Count Join (PR #1606)

**Motivation:** A `COUNT(col | *) GROUP BY key` over a preserved-side outer equi-join on the same key materializes the whole join result only to collapse it again. The count is a pure cardinality product, so the joined rows never have to exist.

**Mechanism:** `sirius_physical_dense_count_join` replaces the join-plus-aggregate fragment with two direct-address histograms over the preserved key domain: one holding each key's preserved-side multiplicity `P`, one holding its counted-side match count `M`. For a key with `V` matches whose COUNT argument is non-NULL, the emitted value is `P * max(M, 1)` for COUNT(*) and `P * V` for COUNT(col) — the rule `sirius::op::dense_count_semantics` states once for both strategies. Preserved-side NULL keys form a single group carrying the same formula at `M == 0`.

The dense path is taken only when `dense_count_layout::plan` succeeds — the domain must be non-empty and not the full 64-bit range, `2 * slots * slot_bytes` must fit `size_t`, and `slots` must fit `int64_t` — and the domain is then dense enough to be worth direct addressing: `total_bytes() <= min(max_bins_bytes, 4 x input bytes)`, `slots <= 8 x non-NULL preserved rows`, and `slots <= 2 x total input rows`. Every term is measured over the one partition a task holds, and the budget is not divided by the partition count: hash partitioning gives every task the full key domain, so tasks replicate a full-width histogram rather than splitting one — a domain worth direct addressing over the whole input is rejected once it would have to be replicated per partition. Otherwise the operator falls back to exact sparse aggregation (per-batch groupby, balanced partial merge, left join, multiply).

Slots are 32-bit unless either side has at least `UINT32_MAX` rows, in which case they widen to 64-bit. Narrow slots halve the per-key footprint and so double the key range admitted under the same `max_bins_bytes` budget.

**Why this is not a group-by strategy:** direct addressing needs the key domain of everything a task will accumulate before the first row lands, which is why both inputs take FULL barriers. Both inputs are also hash-partitioned on the join key, so equal keys — NULL keys included — land in one partition: each task sizes its histogram from its own partition's preserved min/max, and the per-task outputs concatenate with no merge step. `sirius_physical_dense_count_join::execute` asserts that co-location at runtime, rejecting a second partition that arrives carrying NULL preserved keys. `gpu_aggregate_impl::local_grouped_aggregate` sees one batch of one table and has no min/max at all to size a histogram against.

Two fast paths live inside the dense path. When every slot in the domain is occupied, the selected-group list is the identity permutation, so no gather map is built and the histogram reads stream instead of gathering; otherwise the map is sized from the exact group count, which a duplicate-heavy preserved side keeps far below both the domain and the row count. Below a 48 KiB shared-memory budget, and with at least eight rows per slot, accumulation privatizes the histogram per block to keep a low-cardinality domain off a handful of global atomic addresses.

**Code path:**
- `src/op/sirius_physical_dense_count_join.cpp` — strategy gate (`dense_admits`, `max_admitted_histogram_bytes`), sparse fallback
- `src/cuda/dense_count_join_impl.cu` — histogram accumulation and emit kernels
- `src/op/aggregate/dense_count_join_impl.hpp` — `dense_count_layout`, `dense_count_semantics`, `dense_count_bounds`, `dense_count_state`

**Config:** `enable_dense_count_join` (default: `true`, see [Configuration](configuration.md)); the histogram byte budget is engine-internal and not user-tunable. Operator entry: [Operators](operators.md).

### Distinct Hash Join (PR #558)

**Motivation:** `cudf::hash_join` does not exploit build-side uniqueness, performing unnecessary work for 1:1 joins.

**Mechanism:** When build-side keys are proven unique via logical plan analysis, Sirius uses `cudf::distinct_hash_join` instead of `cudf::hash_join`, in both STANDARD and BUILD_PROBE modes. `prove_unique_columns()` walks the DuckDB logical plan subtree and detects uniqueness from:
- PRIMARY KEY constraints on `LogicalGet`
- GROUP BY columns on `LogicalAggregate` (single grouping set)
- Delim scans (`LogicalDelimGet`) — unique by construction over the complete column set (see the next entry)
- Propagation through `LogicalFilter`, `LogicalOrder`, `LogicalLimit`, `LogicalTopN`, `LogicalProjection`, `LogicalComparisonJoin`

Only applies to INNER and LEFT joins with pure equality conditions (excludes IS NOT DISTINCT FROM due to `null_equality::UNEQUAL` semantics).

**Code path:** `src/planner/sirius_plan_comparison_join.cpp` — `prove_unique_columns()`, `src/op/sirius_physical_hash_join.cpp` — distinct hash table construction

### DELIM_GET Build-Side Uniqueness Proof (PR #1550)

**Motivation:** The distinct hash join's plan-time uniqueness proof recognizes PRIMARY KEY constraints and GROUP BY outputs, so on schemas with no declared keys (TPC-H declares none) delim-join build sides — duplicate-eliminated by construction — still took the general two-pass `cudf::hash_join`. Measured at SF1000: q22 -24%, q2 -9%.

**Mechanism:** `prove_unique_columns()` claims a `LOGICAL_DELIM_GET` build side as unique: a delim scan replays the enclosing delim join's duplicate-eliminated chunk, which Sirius materializes with a grouped aggregate over all delim columns (globalized via `PARTITION_DISTINCT → DISTINCT_MERGE`), so the full output row set is unique by construction — under EQUAL and UNEQUAL null semantics alike, since GROUP BY collapses NULL keys into one group. Only the complete column set is claimed as a key; subsets stay refused. Being a plan-time proof, it fires in STANDARD mode as well as BUILD_PROBE, adds no runtime pass or stream synchronization, and has no build-size cap — complementing the runtime distinct-build probe (next entry), which covers BUILD_PROBE builds the planner cannot prove.

**Code path:** `src/planner/sirius_plan_comparison_join.cpp` — `LOGICAL_DELIM_GET` case in `prove_unique_columns()`

**Config:** none

### Runtime Distinct-Build Probe (PR #1466)

**Motivation:** The single-pass `cudf::distinct_hash_join` is gated on a plan-time proof of build-key uniqueness, which (outside delim builds) needs a declared PRIMARY KEY — an ART index with memory and insert/update/delete cost. Builds that are distinct in fact but unprovably so pay `cudf::hash_join`'s two probe passes (count + retrieve) on every probe batch. TPC-H SF1000: 22-query suite -4.8%, q11 -48%, q9 -18%.

**Mechanism:** In BUILD_PROBE mode — where one test amortizes across every streamed probe batch — an INNER/LEFT equality join whose uniqueness the planner could not prove tests build distinctness at runtime: `build_keys_are_distinct()` runs one `cudf::distinct_count` pass over the cached build keys (above 4M rows, a 1M-row prefix sample refutes non-distinct builds cheaply first) and, when the keys are distinct, the partition slot builds a `cudf::distinct_hash_join` instead of `cudf::hash_join`. The test reads a count back to the host (one stream synchronization per cached build), applies only under `null_equality::UNEQUAL`, and is skipped above 128M build rows. Non-distinct builds keep the general two-pass path.

**Code path:** `src/op/sirius_physical_hash_join.cpp` — `build_keys_are_distinct()` and the BUILD_PROBE build-arm gate; `src/planner/sirius_plan_comparison_join.cpp` — wiring `enable_runtime_distinct_build_probe` into the operator

**Config:** engine-internal policy, temporarily disabled by default while `cudf::distinct_count` carries a cuCollections defect (NVIDIA/cuCollections#834, tracked as issue #1600): `DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE = false` in `src/sirius_config.hpp`. Not YAML-configurable (the key is rejected); the `enable_runtime_distinct_build_probe` SET option exists only with `SIRIUS_ENABLE_TEST_OPTIONS=1`.

### Scan Scheduling Tuning (PR #507)

**Motivation:** Eagerly depleting all scan sources at query startup wastes GPU memory on multi-scan plans (e.g., joins with two scanned tables).

**Mechanism:** At query startup, `task_scheduler::start_query()` schedules exactly the first operator in `query.get_scan_operators()` — the remaining scans are not eagerly depleted. Subsequent scan activation is topology-driven through `get_next_task_hint()` and completion-driven downstream scheduling (plus, under the `lookahead` task-creator strategy, the empty-queue look-ahead).

**Code path:** `src/pipeline/task_scheduler.cpp` — `start_query()`; `src/creator/task_creator.cpp` — hint-driven task creation

**Config:** `max_build_hash_table_bytes` (default: 500 MB) — now independent from `concat_batch_bytes`, enabling larger build sides in BUILD_PROBE mode without affecting other joins.

### Zero-Copy Projection Passthrough (PR #991)

**Motivation:** A projection that simply re-references input columns (`SELECT a, c, a`) previously deep-copied every output column on device via the expression evaluator's BOUND_REF path, even though the data already lived on the GPU.

**Mechanism:** `sirius_physical_projection::execute()` classifies each `select_list` entry as a pure passthrough (`sirius::ast::reference`) or an expression to evaluate, then takes one of three paths per batch:
1. **All evaluated:** owned `cudf::table` (unchanged).
2. **All passthrough:** output is a `cudf::table_view` over the input columns, wrapped as a view-backed batch whose owner is the input's `read_only_data_batch` lock — **zero device copies**.
3. **Mixed:** only the non-passthrough entries are evaluated; the output view mixes evaluated columns with input columns, jointly owned by the evaluated table (`shared_ptr<cudf::table>`) and the input lock.

Only the entries needing evaluation are handed to the evaluator (its `std::vector<sirius::ast::node const*>` constructor), so passthrough columns are never materialized.

**Code path:** `src/op/sirius_physical_projection.cpp` — `execute()`; `src/data/data_batch_utils.hpp` — `make_data_batch_from_view()`; `src/expression_evaluator/expression_evaluator.hpp` — subset constructor.

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
- `src/expression/ast/utils.cpp` — `visit_references()`, `substitute_references()`

### Top-K Gather for TopN Sort Fallbacks (PR #1392)

**Motivation:** TopN orderings that `cudf::top_k_order` cannot serve — a nullable key (it takes no `null_order` and treats NULLs as the largest value, so it cannot honor SQL NULLS FIRST/LAST) or a multi-column key (it is single-column only) — fell back to `cudf::sort_by_key` over the whole input, materializing a fully sorted copy of every payload column just to keep the first `offset + limit` rows.

**Mechanism:** `sorted_order_top_k()` sorts only the key columns into an INT32 permutation with `cudf::sorted_order` (which honors null placement), slices the permutation to the first `keep_rows` indices, and gathers just the surviving rows from the full input: the sort materializes an index column rather than sorted payload copies, and the gather touches `keep_rows` rows instead of all of them. Both fallbacks — nullable single-key and multi-key — route through it, in the local TOP_N and in MERGE_TOP_N. The non-nullable single-key path keeps `cudf::top_k_order` followed by a `sort_by_key` over the k gathered rows for deterministic output order.

**Code path:** `src/op/sirius_physical_top_n.cpp` — `sorted_order_top_k()`, `compute_top_n_table()`

**Config:** none

### Decimal SUM Width Selection (PRs #2027, #2073)

**Motivation:** A decimal SUM must not overflow, but the widest type is also the slowest. cuDF sums
64-bit decimals much faster than 128-bit ones, especially in a grouped aggregation with few groups,
where 128-bit accumulation contends on the same groups. DuckDB v2 types some sums wider than their
values need: the TPC-H Q9 profit `l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity`
is `DECIMAL(19,4)`, held as DECIMAL128, although every value fits 64 bits. At SF1000 its group-by
took 3.8 s instead of 28 ms.

**Mechanism:** Per batch, grouped and ungrouped aggregations pick the width of each decimal SUM (and
of AVG's SUM) from one exact
proof: a sum of n values of magnitude at most m is at most n * m. A DECIMAL32 or DECIMAL64 input is
widened to the next type only when rows * max|value| exceeds its storage width. A DECIMAL128 input is
narrowed to DECIMAL64 at the same scale when rows * max|value| is at most 2^63 - 1, and the result is
cast back to DECIMAL128, so no value and no partial sum of any group can leave the 64-bit range, and
the merge combines the per-batch partials at the original 128-bit width. The
bound comes from base-table statistics when the SUM input is a scan column
(`planner::resolve_decimal_sum_input_bounds`); otherwise the batch is measured with `cudf::minmax`,
costing one stream synchronization per batch for all measured columns together.

**Code path:**
- `src/op/aggregate/aggregate_op_util.cpp` — `decimal_sums_to_widen`, `decimal_sums_to_narrow`
- `src/op/aggregate/gpu_aggregate_impl.cpp` — per-batch request rewriting in the grouped aggregation
- `src/op/sirius_physical_ungrouped_aggregate.cpp` — per-batch input casts in the ungrouped reduction
- `src/planner/decimal_sum_input_bounds.cpp` — plan-time bounds from scan statistics

**Config:** `enable_decimal_sum_stats_bound` (default: `true`) lets plan-time bounds skip the
per-batch measurement.

### Per-Task Result Collection (PR #2156)

**Motivation:** The materialized collector receives one batch per final partition, so the last stage of a multi-partition query sinks several batches at once, one per task. Converting each batch into 2,048-row DuckDB chunks and appending every chunk to the one shared `ColumnDataCollection` under the operator mutex serializes the tasks on the lock — about 73,000 acquisitions for a 150M-row result, with tasks spending most of their time waiting for each other, and adding pipeline threads makes it worse.

**Mechanism:** `sirius_physical_materialized_collector::sink()` appends each batch's chunks into a task-local `ColumnDataCollection`, then takes the collector mutex once and merges via `ColumnDataCollection::Combine`, which moves the local segments into the query result without copying rows. At most one local collection per pipeline thread is held before combining, so host memory peak is unchanged.

**Code path:** `src/op/sirius_physical_result_collector.cpp` — `sirius_physical_materialized_collector::sink()`

**Config:** none

## Memory Optimizations

### Memory-Pressure-Driven Downgrade (PR #368)

**Motivation:** GPU memory can be exhausted during complex queries with many concurrent pipelines.

**Mechanism:** Each downgrade executor is bound to one memory space and monitors it for pressure. When `should_downgrade_memory()` triggers (usage above `downgrade_trigger_fraction`), it enqueues a `downgrade_request`; the processing thread dequeues requests sequentially and collects candidates:
1. Partitioned repositories first, sorted by data size descending
2. Non-active partitions before active ones
3. Last-to-first partition iteration

Data is moved from GPU to HOST tier via converter registry.

**Code path:** `src/downgrade/downgrade_executor.cpp` — `monitor_loop()`, `processing_loop()`

**Config:** `downgrade_trigger_fraction` (default: 0.8 for GPU, 0.9 for host), `downgrade_stop_fraction` (default: 0.6 for GPU, 0.8 for host). Configuration requires `0 < stop < trigger <= 1`.

### OOM Retry Mechanism (PR #364)

**Motivation:** Transient GPU OOM can occur when multiple tasks compete for memory.

**Mechanism:** Operators throw `oom_reschedule_exception` carrying intermediate results and resume index. The GPU executor catches this and:
1. Preserves intermediate operator data
2. Creates a rescheduled task that resumes at the operator index the exception carries (an OOM at operator *i* resumes at *i*; a prepare-phase OOM keeps the task's existing start index)
3. Retries up to 100 times with 50 ms backoff

**Code path:**
- `src/pipeline/oom_reschedule_exception.hpp` — exception class
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
- `src/pipeline/pipeline_memory_history.hpp` — history ring buffer and estimation
- `src/pipeline/gpu_pipeline_task.cpp` — `get_estimated_reservation_size_info()`

### Downgrade Request Pattern (PR #579)

**Motivation:** The previous downgrade retry loop over-freed memory and caused contention between concurrent downgrade requests competing for the same batches.

**Mechanism:** `request_downgrade(predicate)` enqueues a `downgrade_request` struct onto an MPMC queue. A single processing thread handles requests sequentially, lazily fetching candidates from data repositories, then task queues, dispatching them to a thread pool one-by-one via `convertible_data::convert()`, and evaluating the caller-supplied `predicate` after each completion. The predicate defines "done" (e.g., "memory reservation succeeded") -- no retry loop, no over-freeing.

**Code path:** `src/downgrade/downgrade_executor.cpp` -- `request_downgrade()`, `processing_loop()`

### Pinned Host Memory Caching (PR #437)

**Motivation:** Standard host memory requires page-locking for GPU transfers, which is expensive.

**Mechanism:** `small_pinned_host_memory_resource` maintains pre-allocated pinned memory pools with NUMA affinity. Used for GPU↔CPU transfers and scan output caching.

**Code path:** cuCascade `cucascade/src/memory/small_pinned_host_memory_resource.cpp`, integrated in `src/sirius_context.hpp`

**Config:** Memory manager settings in `sirius.yaml` (see [Configuration](configuration.md))

## Scan Optimizations

### Compressed Materialization (PR #1260)

**Motivation:** Integer and fixed-point DECIMAL columns often use only a fraction of their declared
range. Carrying their native width through every GPU batch increases memory traffic and cache
pressure even when the SQL type must remain unchanged.

**Mechanism:** A complete physical-type sidecar records narrower signed, unsigned, or same-scale
decimal cuDF carriers without changing the logical schema. `pin_table` computes exact bounds for
each materialized cache chunk and records the chosen carriers, so different chunks may use different
widths. At plan time, a scan derives its targets from that recorded metadata only when the pinned
cache can serve all requested columns; unpinned scans stay native. Exact per-batch bounds guard any
runtime wider-to-narrower cast. Pure-reference payloads and eligible grouped-aggregate keys can
remain narrow, and comparisons against representable constants can execute in the narrow domain.
Arithmetic, hash-join keys, value-sensitive aggregate inputs, ordering, unsupported boundaries, and
result materialization restore native carriers.

**Code path:**
- `src/helper/numeric_narrowing.cpp` — exact range extraction and carrier selection
- `src/planner/sirius_plan_get.cpp` — pinned-residency gate and metadata-derived scan sidecars
- `src/planner/sirius_plan_narrowing_policy.cpp` — tier-aware keep-or-retract policy
- `src/planner/sirius_plan_compressed_schema.cpp` — sidecar propagation, restore projections, and pruning
- `src/op/scan/sirius_gpu_scan_operator.cpp` — runtime verification and schema normalization
- `src/pin_table.cpp` — exact batch-granular pin narrowing
- `src/expression_evaluator/specializations/reference.cpp` and `narrow_domain.cpp` — reference restoration and narrow-domain constants

**Config:** `enable_compressed_materialization` (default: `true`), settable through YAML under
`sirius.operator_params` and the DuckDB SET option. See
[Compressed Materialization](compressed-materialization.md).

### Late Materialization (unreleased, experimental)

**Motivation:** A query that selects wide columns carries them from the scan to whatever finally
reads them — through joins that copy them beside their keys, partitions that write them to a
repository and read them back, and aggregates that group on them. Nothing in that stretch reads
what is IN them. On TPC-H q10 at SF1000 the five wide `customer` columns are 158 B/row and cross
eleven port boundaries before anything needs their values.

**Mechanism:** For a PINNED table, the scan emits a pin-order rowid (UINT32 where the table's rows
fit 32 bits, UINT64 otherwise) in place of the deferred columns plus 1-byte placeholders, so arity
and positions are unchanged and every operator in between is unaffected. A directive on the
consuming operator gathers the values back out of the pinned chunks, matching its batch by whole
schema. Both halves install together or not at all. A plan pass reports how far each column travels
and how many port crossings it survives; a policy with measured floors decides whether the ride
repays the rowid. Where the deferred columns are GROUP BY keys, the ride can continue past the
aggregate to materialize one row per group instead of one per join match, subject to a pin-time
distinctness proof.

**Code path:**
- `src/planner/late_mat_plan_pass.cpp` — column lifetimes, group-by/top-n/join modelling
- `src/late_mat/defer_directive.hpp` — the pair, the substituted schemas, rowid widths
- `src/late_mat/defer_policy.hpp` — value/boundary floors and their measurements
- `src/late_mat/pin_uniqueness.cpp` — the pin-time distinctness proof
- `src/scan_manager/sirius_scan_manager.cpp` — admission, the port hop, riders
- `src/late_mat/port_materialize.cpp`, `materialize.cpp` — putting the values back
- `src/op/scan/sirius_gpu_scan_operator.cpp` — rowid emission, including for filtered scans

**Config:** `SIRIUS_EXP_LATE_MAT=1` gates the feature (off by default, inert when off);
`SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS` selects the columns the uniqueness probe observes, without
which no group-by-rowid ride is admissible. Five further `SIRIUS_EXP_LATE_MAT_*` knobs tune the
floors and the dark count-on-deferred path. Composes with `enable_compressed_materialization`
(default on) — a deferred column riding through a carrier-restore cast is carried past it rather
than suppressing the deferral.

**Measured:** GB300 SF1000, default `enable_compressed_materialization`. See
[Late Materialization](late-materialization.md) for the full results table.

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

### Null-Count Row-Group Pruning (PR #1430)

**Motivation:** Null-test predicates cannot be handed to cuDF's min/max stats filter (it faults on them — see PR #1417), so `WHERE v IS NULL` read every row group even though the parquet footer's `null_count` statistic answers the test directly.

**Mechanism:** A second pruning pass over each footer applies null tests from the `TableFilterSet`: `IS NULL` cannot match a row group with `null_count == 0`; `IS NOT NULL` cannot match one where every row is null (`null_count == num_rows`). An absent `null_count` (it is optional in the spec) keeps the row group. Applies only to scalar leaf columns (nested-column leaf stats describe repeated/leaf-level nullness; legacy top-level `REPEATED` primitives are excluded) and only to conjunctive predicate positions — null tests inside `OR` cannot prune. Only provably non-matching row groups are skipped; the full predicate still runs at read/post-decode time.

**Code path:** `src/op/scan/parquet_gpu_ingestible.cpp` — `collect_null_prune_predicates()` and the null-count pruning loop; `src/op/scan/parquet_gpu_ingestible.hpp` — `null_prune_predicate`

**Config:** none

### Pure-Filter Column Elision (PRs #1019, #1027)

**Motivation:** The scan's post-decode filter path gathered every decoded column and then dropped pure-filter columns (read only for the predicate) during projection — materializing data just to throw it away. The standalone `FILTER` operator had the same waste for columns not needed downstream.

**Mechanism:** `expression_evaluator::select(input, output_indices)` evaluates the predicate over the full input (filter-only columns stay visible to it) but gathers only the requested output columns. The planner records the output boundary before appending filter reads. `scan_plan` resolves it into materialized slot roles and ordered output positions; the reader still places outputs before filter-only columns. A row-count carrier has its own role and is retained for column-less scans. For the FILTER operator, `sirius_plan_filter` translates DuckDB's `LogicalFilter::projection_map` into `output_indices` on `sirius_physical_filter` (empty means keep all). `count(*)`-style filters with no output columns use the all-columns overload, since a 0-column/N-row gather result is unrepresentable. On a GPU-pinned Simpatico-compressed chunk whose filtered decode applied the scan's whole filter, the decode itself omits the filter-only slots (`pushdown_outcome::filter_only_columns_dropped`): they are tested while the selection is built but never decoded, and the batch can be taken over by the scan without a copy when its planned output layout is a leading identity (see [Compressed Pinning](compressed-pinning.md)).

**Code path:** `src/expression_evaluator/expression_evaluator.hpp` — `select()` overloads, `compute_mask()`; `src/op/scan/scan_plan.hpp` — `output_data_positions()`; `src/op/scan/parquet_gpu_ingestible.cpp`, `duckdb_native_gpu_ingestible.cpp`; `src/planner/sirius_plan_filter.cpp`; decode-time variant: `src/compression/compressed_scan.cpp`, `src/compression/simpatico_codegen/src/simpatico_codegen.cpp`, `src/op/scan/scan_plan.cpp`

**Config:** none

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

**Mechanism:** `sirius::io` provides a `cudf::io::datasource` (`sirius_datasource`) backed by io_uring reactors and an optional pinned-memory `prefetching_cache`. The cache converts hits, claimed chunks, and gaps into one logical prepared-slice batch; `templated_ioctx` balances that batch across the two least-busy reactors with one shared coordinator. The worker chooses 256 KiB–16 MiB physical operations, uses `O_DIRECT` when the complete operation is compatible, and holds any multi-block CuCascade staging through its CUDA event. Cache-hit H2D copies likewise hold pins and a completion credit until the stream reaches them. A packed `atomic<uint64_t>` state machine combines lifecycle, reader pins, populated extent, and subscribers so coverage checks and claims are one CAS. Eviction is driven by a tiered LRU score; read-ahead budgets bound work while completion credits make teardown wait for all cache-owned accesses.

**Code path:**
- `src/io/sirius_datasource.cpp` — `cudf::io::datasource` implementation
- `src/io/cache/prefetching_cache.cpp` — chunk cache, worker, evictor, buffer pool
- `src/io/uring/uring_reactor.cpp` — io_uring backend reactor
- `src/io/io_request.hpp` — prepared request grouping and exact completion fan-in

### DuckDB-Native Scan Metadata Walk (PRs #868, #895, #936, #900)

**Motivation:** The DuckDB-native GPU scan's per-row-group metadata walk dominates cold-query runtime at small scale factors. It issues roughly one cold block read per (row group × projected column), and the original walk both read metadata for every column and parsed DuckDB's stringified `GetColumnSegmentInfo` output, which builds and re-parses compression / column-path / `statistics.ToString()` blobs.

**Mechanism:** The walk is structured for minimal, parallel, typed metadata access with statistics pruning:
1. **Projected-column-only, typed walk (#868, #936):** `walk_duckdb_native_row_group_range()` walks the DuckDB segment trees directly for only the projected columns, reading typed `block_id` / compression / row counts / validity-child / max-string-length per segment instead of calling `GetColumnSegmentInfo` and re-parsing strings.
2. **Stats pruning (#900):** `prepare_duckdb_native_walk()` evaluates DuckDB's own `TableFilter::CheckStatistics` against each row group's per-column statistics and drops any row group a pushed-down filter proves `FILTER_ALWAYS_FALSE` before it is staged, copied to the GPU, or decoded; an all-pruned table routes to DuckDB CPU up front.
3. **Parallel range walk + early decode (#895):** `prepare_duckdb_native_walk()` runs as a cheap serial pre-step (partition stats, type-viability gate, row-group count) with no per-segment I/O; the row groups are sliced into fixed internal ranges of eight groups, and the scan-manager pool walks the ranges in parallel so cold segment reads for different ranges overlap. The batch coalescer packs parsed ranges into cap-sized batches that decode while later ranges are still being parsed.

**Code path:**
- `src/op/scan/duckdb_native_metadata.cpp` — `prepare_duckdb_native_walk()`, `walk_duckdb_native_row_group_range()`, `run_fused_stats_pass()`
- `src/op/scan/duckdb_native_gpu_ingestible.cpp` — parse-range slicing, per-range walk thunks, and the `duckdb_native_batch_coalescer`

### Deferred Metadata Walk for Pin-Served Scans (PR #1548)

**Motivation:** Running `prepare_duckdb_native_walk()` at ingestible construction — `GetPartitionStats` plus a statistics pass over every row group — is wasted for a scan the MVCC-pinned cache serves: its splits come from the pinned entry's chunk plan and insert-delta bundles, not from the walk.

**Mechanism:** Every native walk is deferred to execution preparation; construction only seeds a safety value for the range count and keeps the projected-type gate eager (`unsupported_projected_type_reason`, also applied at plan time), so an undecodable projection still refuses where the throw is a clean CPU fallback. `sirius_scan_manager::prepare_for_query` runs the walk through `gpu_ingestible::ensure_metadata_prepared()` on the query thread (`GetPartitionStats` touches `ClientContext`/`LocalStorage`), under the shared checkpoint key it acquires for the scan's database, and only for scans that miss the pinned cache — a `try_match_cached_entry` hit skips the disk-reading provider and the walk entirely. The walk runs under a re-arming `call_once`, so a failure surfaces as a runtime scan error and is retried rather than latched, and a checkpoint-iteration change during preparation throws a checkpoint-revalidation error. `next_split_provider` keeps a warn-and-run backstop off the query thread. When the walk does run, the snapshot/product cache and fused parallel pass make it cheap (see "Cached, Fused, Parallel Native Metadata Prepare Walk").

**Code path:** `src/op/scan/duckdb_native_gpu_ingestible.cpp:306-313` — unconditional deferral at construction; `:318-375` — `run_metadata_walk` with the checkpoint-iteration guard; `:377-402` — `ensure_metadata_prepared` (checkpoint-key check, re-arming `call_once`); `:427-433` — backstop; `src/scan_manager/sirius_scan_manager.cpp:1617-1649` — cache-hit skip, checkpoint-key acquisition, query-thread walk

**Config:** none

### DuckDB-Native Async Coalesced Reads (PR #849)

**Motivation:** The DuckDB-native decoder issues many small segment reads. Synchronous `host_read()` calls bypass the datasource backend in favor of direct `pread()`, serializing I/O and inflating the request count per split.

**Mechanism:** The decoder coalesces file-adjacent segment reads — bridging the small per-block header gaps up to a `coalesce_max_gap` derived from the block header size — into large sequential ranges, then issues them as one batch via `sirius_datasource::host_read_ranges_async()` into pinned host blocks. Each coalesced range maps to a contiguous destination span, and the decoder issues bulk asynchronous H2D memcpy into aligned device memory. This cuts read requests per split several-fold and raises read throughput, especially on warm runs.

**Code path:** `src/op/scan/duckdb_native_decoder.cpp` — range coalescing and `host_read_ranges_async()` dispatch

### Async S3 / REST Reactor Backend (PR #859)

**Motivation:** A per-request serial S3 backend staged each chunk as GET → H2D copy → `cudaStreamSynchronize`, serializing the GPU stream once per chunk and leaving request latency unhidden — costly when the reader issues many small ranged reads over high-RTT links.

**Mechanism:** The remote read path is an asynchronous, concurrent reactor that plugs into the same `templated_ioctx<Reactor>` abstraction as the local io_uring backend, so the backend-agnostic machinery (sync→async bridge, completion aggregation, per-request fan-out, device chunking) is shared rather than reimplemented. A device read issues async ranged GETs into a bounded pinned host staging block, then `cudaMemcpyAsync` H2D, detecting completion by polling a per-chunk `cudaEvent` (`cudaEventQuery`) rather than synchronizing the stream — so the GET window and copy window overlap and up to `max_connections` chunks are in flight while host staging stays O(`max_connections`).

**Code path:**
- `src/io/rest/rest_reactor.cpp`, `src/io/rest/rest_ioctx.cpp` — async REST/S3 reactor over the shared `templated_ioctx` base
- `src/io/templated_ioctx.hpp` — backend-agnostic async machinery shared with the io_uring path

**Config:** `object_store` config (endpoint / region / credentials / signing mode) under `executor.scan_manager`

### Single-Request S3 Parquet Footer Bind (PR #1087)

**Motivation:** A cold S3 parquet bind issued a HEAD for the object size plus separate trailer and footer GETs — three round-trips per file over high-RTT links.

**Mechanism:** Opening with `open_hint::parquet_footer_probe` makes the REST backend issue one suffix-range GET (`Range: bytes=-N`), which returns the object size (from `Content-Range`) and stashes the trailing N bytes on the open object; the reactor then serves cuDF's trailer/footer reads from that per-open stash. `describe_parquet` is metadata-aware: a cold bind probes, while a warm re-bind opens `generic` (one HEAD) and reuses the parsed footer from the metadata store. Unusable suffix responses (a 200 full-body reply, 416, or a missing `Content-Range`) abort the body mid-stream and fall back to a plain HEAD. See the scan doc's S3 backend section for the open-path details.

**Code path:** `src/io/io_context.hpp` — `open_hint`; `src/io/rest/rest_ioctx.cpp`, `src/io/rest/rest_reactor.cpp`; `src/io/cache/metadata_store.cpp`; `src/scan_manager/sirius_scan_manager.cpp`

**Config:** `scan_manager.rest.footer_probe_bytes` (default 512 KiB) — must cover the footer or the probe falls back to a body re-GET

### Zero-Copy from Pinned and Cached Tables (PR #881)

**Motivation:** When scan input is an already-resident pinned/cached host table, copying the data again on consumption is wasted work.

**Mechanism:** The pinned-table scan path moves host-pinned data to the GPU during prepare-for-processing (so the work is attributed correctly and bounded by the per-task reservation) and avoids copying out of cached tables when the batch can be consumed directly. A synchronization point after prepare-for-processing makes the prepare cost observable in logs and Quent.

**Code path:**
- `src/scan_manager/sirius_scan_manager.cpp` — cached-entry matching (`try_match_cached_entry`) and zero-copy resident-batch handoff
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
- `src/io/io_context.cpp` — `ioctx` cache integration and coverage policy
- `src/exec/semi_future.hpp` — async I/O completion primitive

**Config:** the `cache` block under `executor.scan_manager` — `mode: sirius` arms it, `eviction` (`lru` / `idle`) picks what retires an idle chunk, and `eviction_threshold_fraction` / `min_prefetching_budget_fraction` size it

### Load-Balanced Scan Batch Coalescing (PR #997)

**Motivation:** Scan splits arrive sized by metadata-parse completion rather than by the configured batch size, so batches could come out smaller than requested, and multi-GPU runs need scan inputs spread across devices. Opportunistic prefetch hints issued in metadata-completion order also rob the head-of-line pipeline of lead time.

**Mechanism:** The scan manager creates a `load_balancing_scan_batch_coalescer` for each query. Each scan pipeline registers a per-pipeline slot holding a `batch_coalescer`, a `split_connector`, and a `balancing_strategy`. That query's sequencer task drains the slots in registration (execution) order: it coalesces each pipeline's splits to the requested batch size — independent of the configured max batch count — chooses a device via the balancing strategy, issues `fadvise(opportunistic)` / prefetch hints, and pushes the placed split onto the connector, advancing to the next pipeline only after the current one closes. The `balancing_strategy` interface stamps a `preferred_device_id` on each split (which the task creator honors); `round_robin_strategy` hands out GPUs via an atomic cursor and is the default.

**Code path:**
- `src/scan_manager/load_balancing_scan_batch_coalescer.cpp` — per-pipeline slots, sequencer loop, coalesce + place + push
- `src/scan_manager/balancing_strategy.hpp`, `src/scan_manager/round_robin_strategy.cpp` — device-distribution interface and default
- `src/scan_manager/split_connector.hpp` — blocking queue between the coalescer and the scan operator

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

**Config:** zone-map capture and pruning are automatic and enabled by default. The advanced YAML
escape hatch `sirius.operator_params.enable_pinned_zone_map_pruning` gates both capture and
pruning; entries pinned while it is disabled remain statless until re-pinned. The direct DuckDB
session override is test-only. See [Pinned-table zone maps](scan.md#zone-maps) for limitations.

### cuDF Hardware Decompression (PR #1302)

**Motivation:** cuDF's parquet reader can route supported codecs through the GPU's hardware decompression engine, but only when `LIBCUDF_HW_DECOMPRESSION=ON` is exported, and exporting it on a machine with any unsupported GPU is unsafe.

**Mechanism:** The opt-in operator param `use_hw_decompression` gates a one-time probe in `SiriusContext::initialize()`. Topology discovery is asked for runtime attributes (which touch the CUDA driver), and only when the flag is set, at least one GPU exists, and **every** GPU reports `runtime_attributes->hw_decomp` does the context emplace a move-only RAII `sirius::util::env_guard` exporting `LIBCUDF_HW_DECOMPRESSION=ON` for the context's lifetime. An unpopulated attribute is a real "not supported" and skips the export, leaving cuDF on software decompression. The guard restores the variable's prior state (previous value or unset) in `terminate()` and on destruction. The decision is startup-only: `SET use_hw_decompression` to a different value throws and points at the YAML.

**Code path:** `src/util/env_guard.hpp` / `src/util/env_guard.cpp` — the RAII guard; `src/sirius_context.cpp:1046-1060` — all-GPU support check and export; `src/sirius_context.cpp:1198` — reset in terminate

**Config:** `sirius.operator_params.use_hw_decompression` (YAML; default `false`, opt-in). The `use_hw_decompression` extension option exposes the value; changing it at runtime throws.

### Decode-Time Membership Pushdown for DuckDB-Native Pins (PR #1555)

**Motivation:** A dynamic join filter can compact a compressed pinned chunk during decode, but only when the scan's splits carry the dynamic-filter channel and the published probe matches the pinned carrier. A duckdb-native pin's splits carried no channel (the channel resolved from parquet bind data only), and a pin-time-narrowed key declined every probe published at the key's native carrier — either way the chunk abandoned its filtered decode and fell back to a full-width decode.

**Mechanism:** The scan operator and the cached provider resolve the dynamic-filter channel from either format's `ingestible_table_info`, so duckdb-native pinned splits are stamped with it and the decode-time membership snapshot in `scan_operator_input::prepare_for_processing` composes join-published filters into the chunk's decompression pushdown. All three membership implementations probe through a shared key domain: build keys are stored at the domain's rep, and the probe kernel converts each probe element into that rep in place — per element, with no materialized widened copy — using the same carrier-family rules as compressed-materialization narrowing, so a pin-time-narrowed probe answers exactly as the native column would; this half is format-agnostic and also covers parquet pins. Decode-side attaches are gated per chunk rather than per entry: a chunk with no MVCC keep-mask still receives the row-selecting pushdown even when other chunks of the same entry are masked. The native ingestible's `post_filter_and_project` also honors the `ROW_FILTERED` contract, mirroring the parquet path.

**Code path:** `src/planner/sirius_physical_plan_generator.cpp` — `make_dynamic_filter_consumer` built from either format's decode bindings and parked on the operator and the cached provider; `src/scan_manager/sirius_scan_manager.cpp` — cached-provider channel capture and per-chunk gate; `src/op/dynamic_filter/dynamic_filter_key_domain.{hpp,cpp}` — key reps and carrier families; `src/cuda/dynamic_filter_probe.cuh` + `src/cuda/sirius_dynamic_{small_in_list,in_list,bloom}_filter.cu` — in-kernel per-element probe conversion in `compute_mask`; `src/op/scan/duckdb_native_gpu_ingestible.cpp` — `ROW_FILTERED` bypass

**Config:** rides the experimental `SIRIUS_EXP_FUSED_SCAN_FILTER` decompression-pushdown gate and `enable_dynamic_filter`; `SIRIUS_EXP_FUSED_SCAN_DIAG=1` traces per-chunk decisions

### MVCC Keep-Mask Version Cache and Decode-Side Composition (PR #1556)

**Motivation:** A pinned duckdb table with committed deletes pays for its visibility keep-masks on every query — visibility-plan capture, `GetSelVector` fill, pinned carve — even though a table's visible state only changes at commits. And a masked compressed chunk that excludes row selection from its decode loses compaction outright, falling back to a full-width decode plus filter-by-copy.

**Mechanism:** Two parts. (1) Each pinned entry carries a single-version keep-mask cache keyed by `{DuckTransactionManager::GetLastCommit, the querying transaction's start_time, ChangesMade}`. `prepare_for_query` reuses the cached set when `last_commit` is unchanged, the query's snapshot covers it, and the query is writer-free; it publishes a fresh set only when the builder's snapshot covers every existing commit, since a writer sees its own uncommitted deletes. This is exact because DuckDB assigns start times and commit ids from one counter under the lock that publishes `last_commit`. `prepare_mvcc_mask_tasks` skips `masks_ready` requests; a (re-)pin resets the cache. (2) The per-chunk MVCC gate composes instead of excluding: a masked chunk's keep-mask is attached as a `decode_visibility_mask` (device and host tiers) alongside the pushdown scan and ANDed into the decode's selection as one additional combine source, so the decode compacts straight to the visible survivors. `pushdown_outcome::visibility_mask_applied` reports consumption; `prepare_for_processing` clears the split's mask on it (re-enabling the zero-copy steal) and throws on a compacted batch with an unconsumed mask. A mask is never a filter source on its own — with no real filter the request is refused, and every non-applied outcome keeps the full-width decode and the scan-side positional mask.

**Code path:** `src/scan_manager/mvcc_mask_cache.hpp` — snapshot key and reuse/publish predicates; `src/scan_manager/sirius_scan_manager.cpp` — cache probe/publish in `prepare_for_query`, mask attach in the cached provider; `src/compression/compressed_scan.{hpp,cpp}` — `decode_visibility_mask`, `pushdown_outcome::visibility_mask_applied`; `src/op/scan/sirius_gpu_scan_operator_data.cpp` — consumption and steal re-enable

**Config:** `SIRIUS_MVCC_MASK_CACHE` environment variable (default on; `0` disables the version cache). Decode-side composition rides the experimental `SIRIUS_EXP_FUSED_SCAN_FILTER` decompression-pushdown gate.

### Cast-Through Range Extraction for DATE Predicates (PR #1577)

**Motivation:** DuckDB constant-folds qgen-style date arithmetic (`l_shipdate <= DATE '1998-12-01' - INTERVAL '72' DAY`) into `CAST(l_shipdate AS TIMESTAMP) <= TIMESTAMP '1998-09-20 00:00:00'`, which arrives at the scan as an `EXPRESSION_FILTER`. A range extraction that only understands constant comparisons refuses the clause, and the scan loses its in-decode masking: full-width decode, a per-row cast AST filter, and a gather-compact. The shape appears in 9 predicates across TPC-H q1/q4/q5/q6/q10/q12/q14/q15/q20 under spec-official (qgen) parameters.

**Mechanism:** `analyze_scan_filters` recognizes `cmp(CAST(date_col AS TIMESTAMP[_S|_MS|_NS]), ts_const)` in either operand order, recursing through AND conjunctions (`fold_expression_conjunct`); `to_decoded_bound` / `lower_timestamp_to_days` lower the timestamp constant into the stored-day domain by floor/ceil. Midnight of day d is strictly monotonic in d, so the bound is exact for every comparison op, coverage stays full, and the residual filter drops; `= non-midnight` folds to a provably empty range. `±infinity` constants lower exactly to the infinite date's stored days, and a finite date whose midnight overflows the timestamp domain is answered by the instant it denotes (DuckDB raises on such queries; a decode range cannot, and neither does the residual cast it replaces). Refused, keeping the residual filter: `TRY_CAST` (NULL-on-overflow diverges from day math), `TIMESTAMP_TZ` (midnight is session-timezone dependent), `INT64_MIN` ticks, and non-comparison shapes. The extracted ranges take effect where the decompression pushdown decodes a compressed pinned chunk.

**Code path:** `src/op/scan/scan_filter_analysis.cpp` — `fold_expression_conjunct`, `to_decoded_bound`, `lower_timestamp_to_days`, `fold_comparison_bound`; contract in `src/op/scan/scan_filter_analysis.hpp`

**Config:** none (the in-decode masking it feeds rides `SIRIUS_EXP_FUSED_SCAN_FILTER`)

### Cached, Fused, Parallel Native Metadata Prepare Walk (PR #1578)

**Motivation:** For unpinned duckdb-native tables, the prepare walk reads every row group's geometry, the min/max statistics of every prunable filter column, and the max string length of every projected varchar column — from scratch, serially, on the query thread, on every query.

**Mechanism:** Three layers, each removing more of the walk.
1. **Per-table snapshot.** A process-wide `duckdb_native_metadata_cache` captures the row-group geometry plus a copy of each needed column's per-row-group statistics into an immutable `table_walk_snapshot`, in one pass under DuckDB's segment-tree lock; a capture a commit tears mid-pass is detected and retried, never cached. Every acquire revalidates per-row-group identity through `weak_ptr` — a `weak_ptr` only locks back to the exact object it was created from, so a freed-and-reallocated row group (including checkpoint compaction, which does not bump the commit counter) never validates. Transaction-local appends (`LocalStorage`) bypass the cache for that query.
2. **Per-query-shape product.** The pruning decisions and varchar-overflow verdict are additionally cached as a `walk_plan_product`, keyed by a `walk_product_key` (projection signature plus deep copies of the prunable pushed-down filters resolved to storage primary indexes). Each product is tagged with the snapshot generation it was computed from (a cache-wide monotonic clock, never reused) and is dropped when the snapshot rebuilds. A product hit reduces the prepare step to a validity probe.
3. **Fused parallel pass.** When the product misses, `run_fused_stats_pass` checks all filter columns and varchar probes in a single pass over the row groups, parallelized by `parallel_over_row_groups` with deterministic results across worker counts; it runs over cached snapshot statistics on a snapshot hit, or over live `PartitionRowGroup` statistics on the uncached path.

`prepare_duckdb_native_walk` tries the cache first and falls through to the uncached fused walk on a bypass; a refusal is reduced to the same (column, row group) the serial pass would report.

**Code path:** `src/op/scan/duckdb_native_metadata_cache.hpp` / `.cpp` — snapshot, product cache, identity probe, torn-capture retry; `src/op/scan/duckdb_native_metadata.cpp:942-975` — cache-first prepare, `:516-532` — fused stats pass, `:1027-1042` — uncached pass; `src/op/scan/metadata_walk_parallel.hpp` — worker policy

**Config:** env `SIRIUS_METADATA_WALK_THREADS` caps the walk's workers (`1` forces serial); env `SIRIUS_DISABLE_NATIVE_METADATA_CACHE` (set and not `"0"`) bypasses both caches. No YAML or extension options.

### Distributed Byte-Range Parquet Splits (PR #2043)

**Motivation:** A distributed frontend load-balances a parquet scan by cutting files at instance byte boundaries, so one file's row groups are split across compute nodes. A node that can only read whole files refuses such plans.

**Mechanism:** Substrait `ReadRel` `LocalFiles` items carry `start`/`length`; `extract_scan_byte_ranges` collects them per canonical path by protobuf reflection (so reads nested under any rel type are found) and refuses remote schemes, a file named both whole and ranged, offset overflow, and a ranged path appearing in more than one read (ranges could not be attributed to one scan). The FFI parks the result as `scan_byte_ranges_state` on the client context; `attach_byte_ranges` claims each path exactly once onto `parquet_ingestible_table_info::resolved_file_ranges` — one entry per bound file holding all of its ranges — and after plan generation `assert_all_consumed` fails the plan if any range went unclaimed, since an unread range would silently drop rows. The per-file metadata task then keeps only the row groups the file's ranges own, before statistics pruning. Ownership is start-offset containment under the StarRocks reader rule (`row_group_start_offset`: the minimum of the first column chunk's data/index/dictionary page offsets and the row group's own `file_offset`, each counted only when present), so every reader of the same file derives the same owner and an exact tiling of the file reads every row group exactly once. A range ending past EOF throws (the producer sized the split against a different file); a zero-length range owns nothing; a range owning no row group is a valid empty scan served through the all-pruned empty-split fallback. The ranges join the split certificate's identity, so the scan-contract check covers them.

**Code path:** `src/op/scan/parquet_byte_range.hpp` / `.cpp` — ownership rule; `src/planner/substrait_scan_ranges.hpp` / `.cpp` — extraction, claim, consumption check; `src/sirius_ffi.cpp:116-121` — plan-bytes entry point; `src/planner/sirius_physical_plan_generator.cpp:218-237,1817-1820,2271-2273` — attach and verify; `src/op/scan/parquet_gpu_ingestible.hpp:84-133`, `.cpp:1361-1389` — owned-row-group union before pruning

**Config:** none

### Per-File Parquet Footer Reader Reuse (PR #2090)

**Motivation:** Computing a batch's column-chunk byte ranges constructs a cuDF `hybrid_scan_reader`, whose constructor copies the file's entire `FileMetaData`. A file split into N batches copies its footer N times — and since both N and the footer grow with the file, the cost is quadratic in file size. The copies run serially on the coalescer path, delaying the scan's first split (a 14.4 GB single-file table read in ~200 batches waited ~10 s before scanning a row).

**Mechanism:** `column_chunk_range_cache` keeps the `hybrid_scan_reader` built for the last (file metadata, reader options) pair and reuses it for every batch of that file; a hit costs two pointer comparisons, and the reader (with its one footer copy) is rebuilt only when the batch's file or reader options change, destroying the old one first so at most one copy lives at a time. Each `parquet_batch_coalescer` owns one cache, consulted for every emitted split's per-slice ranges — which feed both the split's prefetch `fadvise` entries and the bulk materialize route. The ranges themselves are unchanged, and a file read in one batch still builds exactly one reader.

**Code path:** `src/op/scan/parquet_materialize.hpp:69-86`, `src/op/scan/parquet_materialize.cpp:51-73` — the cache; `src/op/scan/parquet_gpu_ingestible.cpp:618,643-657,576-581` — per-coalescer ownership and per-split use

**Config:** none
