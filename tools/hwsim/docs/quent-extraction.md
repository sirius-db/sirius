# Quent Telemetry Extraction — Inventory, Simulator Mapping, Gaps (WS1)

**Status:** definitive as of 2026-08-04, grounded in three sources:

1. The emitted-schema definitions in `rust/crates/telemetry/model/src/*.rs` (Sirius-specific
   entities) and the pinned quent crate rev `2a5ca834` (`quent-query-engine-model` for
   Engine/Worker/QueryGroup/Query/Plan/Operator/Port, `quent-stdlib` for Memory/Channel) —
   checkout inspected at `~/.cargo/git/checkouts/quent-515d44f958e14372/2a5ca83/`.
2. Every C++ emission site (file:line cited below), all under `src/` (Super Sirius; nothing in
   `src/legacy/` emits quent).
3. A **real 654 MB trace** on this box:
   `/localhome/local-faramburu/repos/sirius/telemetry_data/019fbafc-e2cc-7bb3-b1ba-915fd149e7f5/`
   — 66 queries (a TPC-H run), ~3.0 M events: 1,312,823 batch_placement, 1,057,774 data_batch,
   644,038 task, 3,372 port, 1,725 operator(=pipeline), 264 query, 66 plan lines. Every
   field/shape claim below was verified against this data.

All C++ paths are relative to the repo root. Line numbers are from a `dev`-based
checkout at the time of writing and may drift.

---

## (a) Event / field inventory

### Envelope and file layout

Every ndjson line is one event:

```json
{"id": "<uuid-v7 of the entity instance>", "timestamp": <u64 unix nanoseconds>, "data": ...}
```

Two `data` shapes exist:

- **Plain entities** (declaration/event style): `"data": {"EventName": {…fields}}` or
  `{"EventName": null}`. Entities: `engine`, `worker`, `query_group`, `plan`, `operator`,
  `port`, `gpu_device`, `thread_group`.
- **FSM entities** (state-machine style): `"data": {"seq": N, "state": {"StateName": {…}}}`,
  terminal event `"data": {"seq": N, "state": "Exit"}`. Entities: `query`, `task`,
  `data_batch`, `batch_placement`, and the resource FSMs `task_queue`, `executor_thread`,
  `task_manager_loop_thread`, `memory`, `memory_tier`, `channel` (whose states are always
  `XInitializing` → `XOperating` → `XFinalizing` → `Exit`).

FSM state payloads contain **attributes** (plain fields) and **usages** — a usage is
`{"resource_id": "<uuid of a resource entity>", "capacity": {"capacity_bytes": B} | {"capacity_entries": E} | null}`.
A usage means "while in this state, this entity occupies that amount of that resource";
resource occupancy over time is reconstructed by integrating usages of overlapping states.

Directory layout per telemetry session (one session = one `SiriusContext`/engine lifetime,
i.e. one CLI process):

```
<output_directory>/<session-uuid-v7>/
  model.qmi                       # JSON metadata: quent version+commit, Sirius model source commit
  engine/<stream-uuid>.ndjson     # one subdir per entity type, one file per type in practice
  query/<...>.ndjson  task/<...>.ndjson  operator/<...>.ndjson  port/<...>.ndjson
  plan/  query_group/  worker/  gpu_device/  thread_group/  task_queue/
  executor_thread/  task_manager_loop_thread/  memory/  memory_tier/  channel/
  data_batch/  batch_placement/
```

Config (`docs/super-sirius/quent-telemetry.md`, `struct telemetry_config` in
`src/include/sirius_config.hpp:141`): `enable_quent` (default true), `exporter`
(`ndjson`|`msgpack`|`postcard`), `output_directory` (default `telemetry_data`), `engine_name`
(default `siriusDB`), `enable_batch_events` (default true — gates data_batch/batch_placement
emission, "roughly doubles telemetry volume"; install gate at `src/sirius_context.cpp:572-577`).

### Static / session-scoped entities

| Entity | Events & fields | Emitted at | Cardinality (observed) |
|---|---|---|---|
| `engine` | `Init{implementation{name,version,custom_attributes}, instance_name}`; `Exit` | `src/telemetry/telemetry_context.cpp:71-80`, dtor `:175` | 1/session. `custom_attributes` emitted **empty** |
| `worker` | `Init{parent_engine_id, instance_name="worker-<pid>"}`; `Exit` | `telemetry_context.cpp:82-86`, `:174` | 1/session |
| `query_group` | `Declaration{instance_name="<engine>-session-<pid>", engine_id}` | `telemetry_context.cpp:92-97` | 1/session (all queries share it) |
| `gpu_device` | `Declaration{instance_name="gpu-N", parent_group_id=engine, ordinal}` | `telemetry_context.cpp:117-122` | 1/GPU |
| `thread_group` | `Declaration{instance_name∈{"shared","executor_thread","task_manager_loop_thread"}, parent_group_id}` | `telemetry_context.cpp:105-109` (shared), `:123-132` (per-GPU buckets) | 1 + 2/GPU |
| `memory` (resource) | `MemoryInitializing{instance_name="memory_space(tier=GPU, device_id=0, limit=229053290249)", parent_group_id, resource_type_name="memory"}`; `MemoryOperating{capacity_bytes=<pool max>}`; `MemoryFinalizing`; `Exit` | `src/telemetry/memory_context.cpp:44-51`, dtor `:87-97` | 1 per cucascade memory space (observed: GPU-0 229 GB, HOST 405 GB; **no DISK space** on this box) |
| `channel` (resource) | `ChannelInitializing{instance_name="host-0->gpu-0" etc., parent_group_id, resource_type_name="channel", source_id, target_id (memory uuids)}`; `ChannelOperating{capacity_bytes=u64::MAX}`; finalize/exit | `memory_context.cpp:63-84` | full mesh between memory spaces (2 on a 1-GPU box). **Capacity is a placeholder (u64::MAX), not a measured bandwidth** |
| `memory_tier` (resource) | `MemoryTierInitializing{instance_name∈{"GPU-N","HOST","DISK"}}`; `MemoryTierOperating{capacity_bytes}` (DISK=0); finalize/exit | `src/telemetry/batch_telemetry.cpp:200` (install), `:230` (uninstall) | GPU per device + HOST + DISK |
| `task_queue` (resource) | `TaskQueueInitializing{instance_name, parent_group_id, resource_type_name="task_queue"}`; `TaskQueueOperating{capacity_entries=u64::MAX}`; finalize/exit | scheduler queue `src/pipeline/task_scheduler.cpp:83` ("task-scheduler-gpu-queue", parent=shared); per-GPU executor queue `src/parallel/task_executor.cpp:37-41` ("gpu_pipeline-task-queue", parent=gpu-N) | 1 + 1/GPU. Declared capacity is a placeholder; occupancy derived from task `Queued` usages |
| `executor_thread` (resource) | `ExecutorThreadInitializing{instance_name="gpu_pipeline-gpuN-exec-K"}`; Operating; finalize/exit | thread-local init at `src/pipeline/gpu_pipeline_executor.cpp:74` via wrapper in `src/include/telemetry/telemetry_context.hpp:131-156,215-227` | pool threads per GPU (4 observed). **Only GPU-executor pool threads** — scan-manager, downgrade, and IO reactor threads have no telemetry |
| `task_manager_loop_thread` (resource) | same 4-state resource FSM | `task_scheduler.cpp:319` ("task-scheduler-thread"); `gpu_pipeline_executor.cpp:97` ("gpu-N-exec-manager") | **ephemeral**: the exec-manager wrapper is a local of `manager_loop()`, re-declared each run — 67 instances for 66 queries in the trace. Parsers must not assume stable thread ids across queries |

### Per-query entities

| Entity | Events & fields | Emitted at |
|---|---|---|
| `query` (FSM) | `Init{instance_name=<query label or "unnamed_query">, query_group_id}` → `Planning{}` → `Executing{}` → `Exit` | create `src/sirius_engine.cpp:80`; `planning()` `:128`; `executing()` `:137`; `exit()` in dtor `:90`. Label comes from `sirius_set_query_label` / `gpu_execution(query_label=…)`. The engine's numeric query id ↔ telemetry uuid mapping exists **only in the log line** at `sirius_engine.cpp:148` (`query {} telemetry_query={:016x}{:016x}`), not in the trace |
| `plan` | `Declaration{parent{query_id, plan_id:null}, instance_name="pipeline_plan", edges:[{source:<port uuid>, target:<port uuid>}], worker_id}` | `telemetry_context.cpp:255-265` (`emit_plan_telemetry`, called from `src/planner/query.cpp:32` once per query) |
| `operator` | `Declaration{plan_id, parent_operator_ids:[], instance_name="GPU_SCAN(0) -> PROJECTION(1) -> … -> HASH_GROUP_BY(3)", type_name="Pipeline Id N", custom_attributes:[]}` | `telemetry_context.cpp:209-217`. **One quent Operator = one Sirius pipeline**, not one physical operator. The physical-operator chain is only a display string here; physical operators are identified per task via `Computing.current_operator_id`. A `Statistics{custom_attributes}` event exists in the schema but is **never emitted** |
| `port` | `Declaration{operator_id=<pipeline uuid>, instance_name="<port>_receiver" \| "<port>_sender"}` | receivers (pipeline source ports) `telemetry_context.cpp:223-227`; pseudo-sink senders `:238-242`. `Statistics` never emitted |

Plan `edges` connect a producing pipeline's **sender** port to the consuming pipeline's
**receiver** port — this is the pipeline-level dataflow DAG.

### `task` FSM (the core scheduling record)

Schema: `rust/crates/telemetry/model/src/task.rs`. One FSM instance per
`sirius_pipeline_itask` (in practice all tasks are `gpu_pipeline_task`, including scan-source
pipelines whose first operator is `GPU_SCAN`). Instance name `task-<numeric task id>`.

| State | Attributes | Usages | Emitted at | Semantics |
|---|---|---|---|---|
| `Created` | `pipeline_uuid` (== quent Operator id) | — | task ctor `src/pipeline/sirius_pipeline_itask.cpp:33-42` | task object exists |
| `Queued` | — | `queue{resource_id, capacity_entries:1}` | scheduler queue `src/pipeline/task_scheduler.cpp:132-135`; per-GPU executor queue `src/parallel/task_executor.cpp:51-55` | occurs **twice** per normal task (scheduler queue, then executor queue after Routing) |
| `Routing` | `preferred_device_id` (actually the *dispatched* device) | `manager_thread` | `task_scheduler.cpp:387-391` | scheduler matched task→ready device |
| `Reserving` | `requested_bytes`, `input_basis`, `peak_estimate`, `bytes_to_materialize` | `manager_thread` | `src/pipeline/gpu_pipeline_executor.cpp:145-152` | memory-reservation attempt; values are the **pre-clamp** estimator outputs (comment at `:153-164`) |
| `Downgrading` | `shortfall_bytes`, `partial_bytes` | `manager_thread` | `gpu_pipeline_executor.cpp:200-205` | reservation shortfall → predicate downgrade requested (spill trigger). Zero occurrences in the sample trace (unpressured run) |
| `Preparing` | `origin_tier` (string, e.g. "HOST"), `target_tier` ("GPU"), `input_bytes` | `executor_thread`, `reservation{MemoryTier resource, capacity_bytes=granted reservation}` | `src/pipeline/gpu_pipeline_task.cpp:546-554` | input materialization onto the target device (host→device upgrades, cross-GPU clones). **`stream.synchronize()` at `:559` makes the span accurate H2D wall time**; span ends at first `Computing` |
| `Computing` | `instance_name="OPNAME(op_id)"`, `current_operator_id`, `input_bytes`, `peak_allocated_bytes` (allocator peak on this task's stream *at operator start*) | `executor_thread`, `reservation` | `gpu_pipeline_task.cpp:374-383`, one per physical operator in the pipeline | per-operator execution. `run_one_operator` ends with `stream.synchronize()` (`gpu_pipeline_task.cpp:215`), so each Computing→next span is true wall time including all GPU work for that operator |
| `Finalizing` | `success` (bool) | — | success `gpu_pipeline_executor.cpp:433-439`; OOM-reschedule (task replaced, success=false) `:413-418`; dtor fail-safe `sirius_pipeline_itask.cpp:44-53` | followed immediately by `Exit` |

Allowed transitions (from the `fsm!` block, incl. abnormal paths): created→queued,
queued→routing, routing→queued, routing→reserving, queued→reserving, reserving→downgrading,
reserving→preparing, downgrading→preparing, preparing→computing, computing→computing,
computing→finalizing, and {created,queued,routing,reserving,downgrading,preparing}→finalizing
(cancellation/failure). An OOM'd task is finalized with `success:false` and a **new task id /
new FSM instance** is created for the reschedule (`gpu_pipeline_executor.cpp:398-420`); the
old→new task linkage is not in the trace (only `original_task_id` internally, not emitted).

### `data_batch` FSM (physical batch lifecycle & tier moves)

Schema `model/src/data_batch.rs`; emitted by `quent_data_batch_probe`
(`src/include/telemetry/data_batch_probe.hpp:88-186`), attached at batch creation:
scan-manager batches `src/scan_manager/sirius_scan_manager.cpp:164,200,222`; operator-output
factories `src/include/data/data_batch_utils.hpp:117-189`; partition clones
`src/op/sirius_physical_partition.cpp:235`.

| State | Attributes | Usages |
|---|---|---|
| `Constructed` | `data_batch_id` (process-unique u64), `producer_pipeline_uuid` | — |
| `Stationary` | — | `memory{resource_id=<memory space>, capacity_bytes=<batch size>}` |
| `InTransit` | — | `source_memory`, `dest_memory`, `channel` — all with `capacity_bytes=<batch size>` |
| `Destructed` → `Exit` | — | — |

`InTransit` brackets a cucascade tier conversion (`conversion_started`/`conversion_completed`)
— its span is the **measured H2D/D2H transfer+conversion time for that batch** and its channel
usage identifies direction (host-0→gpu-0 vs gpu-0→host-0). 23,205 InTransit events in the
sample trace. `Stationary→Stationary` self-transitions occur on `data_replaced`.

### `batch_placement` FSM (batch × consumer-port routing, tier residency)

Schema `model/src/batch.rs`; one placement per (physical batch × consuming pipeline's input
port) — fan-out yields one placement per consumer. Registry: `src/telemetry/batch_telemetry.cpp`
(`src/include/telemetry/batch_telemetry.hpp` for the enums).

| State | Attributes | Usages | Driven from |
|---|---|---|---|
| `BatchRegistered` | `batch_id`, `pipeline_uuid` (consumer), `port_uuid` (consumer's receiver), `origin` ∈ {`operator_output`,`partition_output`,`reschedule_intermediate`} | `tier{MemoryTier, capacity_bytes=batch bytes}` | `on_published`: `src/op/sirius_physical_operator.cpp:270-275` (push into consumer repo), `src/op/sirius_physical_partition_consumer_operator.cpp:31-36`; lazy registration at claim |
| `BatchQueued` | — | `tier` | waiting in the consumer port's data repository |
| `BatchPackaged` | `task_uuid` (consuming task) | `tier` | `on_packaged`: `gpu_pipeline_task.cpp:305` (task ctor claims inputs) |
| `BatchProcessing` | `task_uuid` | `tier` | `on_processing[_by_id]`: `gpu_pipeline_task.cpp:598-611` (after prepare) |
| `BatchConsumed` | `reason` ∈ {`processed`,`task_failed`,`query_end`} → `Exit` | — | `on_consumed`: `gpu_pipeline_task.cpp:312-320` (task dtor); query-end drain `src/sirius_context.cpp:318` |

Tier changes re-emit the current state as a self-transition with the new tier usage
(`on_tier_change` callers: `src/include/data/convertible_data_batch.hpp:143`,
`src/include/pipeline/batch_lock_utils.hpp:146,173`). Consumer-port→(pipeline,port) mapping is
registered during plan emission (`telemetry_context.cpp:228`).

### What is **not** instrumented anywhere (verified by exhaustive grep)

- **The entire IO subsystem** (`src/io/`: uring/REST reactors, prefetching cache, admission
  control, metadata store) — zero quent events. Disk reads, S3 GETs, cache hits/misses,
  prefetch lead time: invisible.
- **Scan manager** (`src/scan_manager/`): split providers, the coalescer/sequencer,
  `split_connector` queue depth — invisible (its batches only appear via the data_batch probe
  once constructed). The blocking wait of `GPU_SCAN`'s `get_next_task_input_data()` on the
  split connector happens **before task creation**, so scan starvation appears only as an
  absence of tasks, not as a wait state.
- **Downgrade executor** (`src/downgrade/downgrade_executor.cpp`): no events for its scan/
  select/spill work; spills are only visible indirectly as batch tier changes + InTransit.
- **Allocator level**: no rmm/cucascade per-allocation or pool-watermark events.
- **CUDA kernels/streams**: no kernel-level timing (nsys territory, WS2). NVTX ranges *do*
  exist with pipeline/task ids in the label (`gpu_pipeline_task.cpp:513-515`, `:180-182`) —
  this is the natural quent↔nsys join key.
- `Operator`/`Port` `Statistics` events and `custom_attributes`: schema exists, never emitted.

---

## (b) Mapping: simulator needs → quent events/fields

| # | Simulator need | Status | Where it comes from |
|---|---|---|---|
| 1a | Pipelines, ports, edges (pipeline-level DAG) | **FULLY captured** | `operator` + `port` Declarations, `plan.edges`; `operator.instance_name` gives the physical-op chain per pipeline |
| 1b | Physical operators within a pipeline | **Partial** | Only names/ids via `task.Computing{instance_name, current_operator_id}` and the chain string; no per-physical-op declaration, parameters, or expressions |
| 1c | Task-level dependency/ordering ("what must finish before what starts") | **Partial → derivable, approximate** | No explicit task-DAG edges. Derivation chain: consuming task ← `BatchPackaged.task_uuid` ← placement `{batch_id, pipeline_uuid, port_uuid}` ← `data_batch.producer_pipeline_uuid` + plan edge. The **producing task** of a batch is NOT recorded (gap G2); producer attribution needs (pipeline, time-window) correlation. Pipeline barrier semantics (build-before-probe, partition sink→consumer) are implicit in observed ordering, not declared |
| 2a | Queue identity, enqueue/dequeue per task | **FULLY captured** | `Queued.queue.resource_id` (scheduler vs per-GPU executor queue); span Queued→Routing = scheduler-queue wait; Queued→Reserving = executor-queue wait |
| 2b | Start/end timestamps, per-state timeline | **FULLY captured** | task FSM transitions; per-operator spans via consecutive `Computing`; end = `Finalizing`/`Exit` |
| 2c | Thread/GPU assignment | **FULLY captured** | `Preparing/Computing.executor_thread.resource_id` → thread → `thread_group` → `gpu_device` group; `Routing.preferred_device_id` |
| 2d | Blocking/wait **reasons** | **Partial** | Memory wait = Reserving→Preparing span + `Downgrading`; queue wait spans exist but *cause* (no free thread vs no dispatchable task vs pull-signal backpressure vs scan starvation) is not annotated (gap G5) |
| 3a | H2D/D2H transfer time per task | **FULLY captured** (good) | `Preparing` span (synchronized) + `input_bytes` + `origin_tier→target_tier`; per-batch `InTransit` spans with bytes + directional channel → effective bandwidth is directly measurable from the trace |
| 3b | GPU kernel busy vs CPU time within an operator | **MISSING** | `Computing` spans are synchronized wall time (kernel + launch + CPU + intra-op transfers, undifferentiated). Gap G4 / WS2 |
| 3c | Disk I/O time per scan task | **MISSING** | Parquet read + decode are folded into `GPU_SCAN`'s single `Computing` span. No IO-layer events at all. Gap G1 |
| 4a | Bytes in per task / per operator | **FULLY captured** | `Preparing.input_bytes`, `Computing.input_bytes` (per physical operator) |
| 4b | Bytes per edge / per port | **FULLY captured** | placement `BatchRegistered{pipeline_uuid, port_uuid}` + tier usage bytes; sum per port = edge volume |
| 4c | Bytes out per task | **Partial** | Not stamped; derivable by attributing downstream `BatchRegistered` events to the producing task via (producer_pipeline, timestamp∈task-compute-window) — exact only when one task per pipeline is computing at a time |
| 4d | Rows in/out | **MISSING** | No row counts anywhere in the trace |
| 4e | Column sizes / schema / compressed-vs-uncompressed | **MISSING** | A batch is an opaque byte size (current representation only) |
| 5a | Pool/tier capacities | **FULLY captured** | `memory.MemoryOperating.capacity_bytes`, `memory_tier` capacities |
| 5b | Per-tier residency over time (fill level) | **FULLY captured (derived)** | Integrate `batch_placement`/`data_batch` usages + task `reservation` usages over time (this is exactly what the analyzer UI does) |
| 5c | Alloc/free events, pool high-water marks | **Partial / MISSING** | `Computing.peak_allocated_bytes` = per-stream allocator peak at operator start (task-local); no pool-level watermark or allocation events |
| 5d | Back-pressure stalls | **Partial** | Reserving→Preparing gap + `Downgrading{shortfall,partial}`; scan-side admission-control waits and bounded-pool waits invisible |
| 5e | Downgrade/spill events + triggers | **Partial** | Trigger and magnitude: `Downgrading`; effect: placement tier-change self-transitions + `InTransit` (gpu→host). Which batches were chosen / downgrade-executor time not attributed. Sample trace has **zero** Downgrading events — a memory-pressured capture is still needed for spill modeling |
| 6 | Config context in-trace | **Weak** | Present: memory-space limits (capacity + name string), tier capacities, engine name, thread/queue identities (thread count observable by counting resources). Missing: YAML snapshot (executor thread count as config, scan-manager pool size, IO backend, prefetch-cache size, reservation policy), hardware info (GPU model, SM count, clocks, C2C bandwidth, NUMA), dataset identity. `engine.Init.custom_attributes` is the intended extension point but is emitted empty |

---

## (c) Gap list, ranked by importance to the simulator

### G1 — Disk/scan I/O is completely invisible (blocks the `io_bandwidth` knob)

The read of a split (uring/S3 → host, or direct decode input) and the GPU decode both live
inside `GPU_SCAN`'s single `Computing` span. There are no events for read issue/complete,
bytes-from-storage (≠ decoded bytes), cache hits, or prefetch lead time. Without this, the
simulator cannot scale I/O rate at all, let alone reproduce "faster I/O → earlier memory
fill → scan back-pressure".

**Proposal (two tiers):**
1. *Split-level (cheap, do first):* in `parquet_gpu_ingestible::materialize_metadata_to_table`
   (`src/op/scan/parquet_gpu_ingestible.cpp`) and
   `duckdb_native_gpu_ingestible::materialize_metadata_to_table`, bracket the storage-read
   portion vs the decode portion and emit either (a) two sub-`Computing` self-transitions on
   the owning task (needs plumbing the task handle into the scan operator — it is available in
   `gpu_pipeline_task::compute_task`), or (b) a new small FSM `io_request{Issued{file_count,
   bytes, target_device} → Completed{bytes_read, cache_hit_bytes}}` created per split.
   Include compressed (on-disk) bytes: available from the split's `scan_info`
   (`fadvise_entries`/decoded-buffer estimates already carry byte ranges).
2. *IO-layer (finer):* instrument `sirius_ioctx` read entry points
   (`host_read_async_io`/`device_read_async_io`/`host_read_ranges_async_io` in
   `src/io/templated_ioctx.hpp`) with one event pair per logical request (not per chunk), and
   the `prefetching_cache` hit/miss counters flushed once per split.

Overhead: split-level is O(splits) ≈ hundreds of events/query — negligible. IO-layer at
request granularity ≈ thousands/query — still small vs the current ~45 k events/query.
Risk: per-chunk instrumentation on the uring hot path would be too hot — aggregate.

> **Implemented (WS9, 2026-08-04)** as tier 1 + a datasource-level read accumulator, via option
> (b): a new `io_request` FSM (`Issued{task_uuid, pipeline_uuid, file_count,
> estimated_compressed_bytes, estimated_decoded_bytes}` → `Completed{bytes_read, read_time_ns,
> read_calls, rows}`), one instance per fresh-read split, emitted around
> `_ingestible->materialize_table()` in `src/op/scan/sirius_gpu_scan_operator.cpp` (both
> parquet and duckdb-native; resident splits emit nothing). Deviation from the proposal: the
> read and decode cannot be bracketed sequentially — `cudf::io::read_parquet` fuses them — so
> read time is instead accumulated per call at `sirius_datasource`
> (`src/io/sirius_datasource.{hpp,cpp}`; sync spans + async issue→settle spans, atomics only)
> plus the duckdb-native decoder's direct ranged read
> (`src/op/scan/duckdb_native_decoder.cpp`), and snapshot-diffed per split via
> `scan_info::io_totals()`. `read_time_ns` is a span sum, so concurrent reads can exceed
> critical-path read time; cache hits count at cache-copy speed. `cache_hit_bytes` and the
> tier-2 io-context request events were NOT implemented (not cheap/clean at request grain).
> Sub-`Computing` self-transitions (option a) were rejected to keep the Computing stream's
> semantics unchanged for existing parsers.

### G2 — No producer-task link on batches (task-level DAG is approximate)

`data_batch.Constructed` carries `producer_pipeline_uuid` only; `BatchRegistered` carries the
consumer side. Task→task edges must be inferred by matching batch-construction timestamps to
producer-pipeline task compute windows, which is ambiguous whenever ≥2 tasks of one pipeline
compute concurrently (common: 4 executor threads).

**Proposal:** add `producer_task_uuid: Uuid` (nil-able) to `data_batch::Constructed` (model
`rust/crates/telemetry/model/src/data_batch.rs`) and to `batch::BatchRegistered`, plumbed via
`telemetry::batch_telemetry_info` (`src/include/telemetry/data_batch_probe.hpp:39-42`) — the
factories in `src/include/data/data_batch_utils.hpp:117-189` are called from operator code
where the current task is known; scan-manager staging batches keep nil. One extra field, no
extra events, ~zero overhead. Risk: touching every factory call site (mechanical).

> **Implemented (WS9, 2026-08-04):** `producer_task_uuid` added to both `data_batch.Constructed`
> and `batch_placement.BatchRegistered`. Deviation: the current task is NOT known at the factory
> call sites (operators never see the task handle), so instead of touching every site the task
> uuid flows through a thread-local (`telemetry::current_task_telemetry_uuid`, set RAII-scoped by
> `gpu_pipeline_task::execute`, read by `sirius_physical_operator::batch_telemetry()` and
> `batch_telemetry_registry::on_published`). Nil for scan-manager staging batches, lazy
> (`reschedule_intermediate`) registrations, and any batch created off the executor thread.

### G3 — No rows, no output bytes, no column/schema info

Only *input* bytes per operator are stamped. Output volume, selectivity, and row counts —
needed to rescale operator work under different plans/data and to model per-edge volumes
robustly — must be inferred from downstream placements (ambiguous per G2, and terminal-sink
output is never registered anywhere).

**Proposal:** in `run_one_operator` (`src/pipeline/gpu_pipeline_task.cpp:169-230`) the output
`operator_data` and its size are already computed for logging — extend the *next* `Computing`
/ the `Finalizing` transition, or emit the schema's existing (currently unused)
`Operator::Statistics{custom_attributes}` event, with `{rows_in, rows_out, output_bytes,
op_wall_us}` per (task, operator). Add `num_rows` (+ column count) to
`data_batch::Constructed`. Overhead: one event or a few fields per operator execution —
bounded by existing Computing volume. Risk: none material; `get_estimated_size_in_bytes()` is
already called on this path.

> **Implemented (WS9, 2026-08-04)** as fields, no new events: `task.Computing.input_rows` (op
> i's input == op i-1's output, so per-op output rows fall out; bytes already did via
> `input_bytes`), `task.Finalizing.{output_rows, output_bytes}` for the last operator's output
> (success path only; 0 on failure), and `data_batch.Constructed.{num_rows, num_columns}`
> (filled by the `make_data_batch*` factories and the sites with a table view at hand; 0 =
> unknown, e.g. undecoded host staging batches and a few direct-probe sites like limit/top-n).
> The `Operator::Statistics` event remains unemitted. Column sizes / schema (4e) remain missing.

### G4 — No GPU-busy vs wall decomposition per operator (blocks separating `gpu_compute` from `gpu_mem_bandwidth`)

`Computing` spans are honest wall time (each operator ends with `stream.synchronize()`), but
kernel-busy time, launch gaps, and intra-operator D2D traffic are indistinguishable.

**Proposal:** primary path is **nsys correlation (WS2)** — the NVTX ranges already carry
`Pipeline {} Task {} [...]` labels (`gpu_pipeline_task.cpp:513-515`) and per-operator labels
(`:180-182`); add the quent task UUID (hex of `telemetry_handle().uuid()`) to those labels so
WS6 can join traces exactly. Optional in-quent fallback: two `cudaEvent`s around
`op.execute()` in `run_one_operator` and stamp `gpu_busy_us` on the next transition —
~2–5 µs/operator added, small relative to the existing per-op sync. Risk: cudaEvent timing
measures stream-occupied time, not SM-busy time; fine for a first-order split.

### G5 — Wait/stall *reasons* not annotated (scan starvation, admission control, pool waits, downgrade work)

The simulator must reproduce *why* a resource sat idle. Today: executor idle gaps, scan-task
absence, and manager-loop blocking are all unlabeled dead time.

**Proposal:**
- Declare the per-scan `split_connector` as a `task_queue`-style resource
  (`src/scan_manager/split_connector.cpp`): one usage per queued split → scan starvation
  becomes visible as an empty queue while the pipeline is live.
- Emit a wait-state on the manager loop around `_bounded_pool->reserve()`
  (`gpu_pipeline_executor.cpp:105`) — e.g. reuse the thread's resource FSM with a
  `Waiting` span, or a usage on the pool.
- Give the downgrade executor a Task-like FSM (request{predicate bytes} → scanning →
  spilling{bytes_moved} → done) in `src/downgrade/downgrade_executor.cpp`.
- Prefetch-cache `admission_control` wait spans (`src/exec/admission_control`), aggregated
  per split.
Overhead: O(splits + downgrade requests + pool cycles) — small. Risk: the manager-loop wait
instrumentation sits on the dispatch path; keep it to two events per cycle.

### G6 — Traces are not self-describing (config/hardware snapshot missing)

Only memory-pool limits are in-trace. Thread counts must be *counted*, queue bounds are
placeholders (u64::MAX), and nothing records scan-pool size, IO backend, prefetch-cache
bytes, GPU model/SMs/clocks, C2C link, or dataset identity — all of which the simulator needs
as the baseline the knobs multiply against.

**Proposal:** populate `engine.Init.custom_attributes` (already in the schema,
`quent_query_engine_model::engine::EngineImplementationAttributes.custom_attributes`,
emitted empty at `telemetry_context.cpp:77`) at `telemetry_context` creation with: the
resolved `sirius_config` essentials (memory-space configs, task-creator/executor thread
counts, scan-manager pool + cache config, IO backend), `cudaDeviceProp` per GPU (name,
SM count, memory + SM clocks, memory bus width), and host topology (NUMA nodes, core count).
Zero steady-state overhead; one-time emission. Risk: none.

> **Implemented (WS9, 2026-08-04):** `engine.Init.implementation.custom_attributes` now carries
> the snapshot (`build_engine_custom_attributes` in `src/telemetry/telemetry_context.cpp`;
> `telemetry_context::create` gained an optional `const sirius_config*`, passed from
> `SiriusContext::initialize`). Includes host/GPU hardware (hostname, NUMA nodes, core count,
> per-GPU name/SM count/SM+mem clocks/bus width via `cudaDeviceGetAttribute`), memory-space
> capacities + reservation limits, executor/task-creator/downgrade/scan-manager thread counts,
> IO backend, prefetch-cache settings, `scan_task_batch_size`, `hash_partition_bytes`, and
> `enable_batch_events`. Dataset identity is still not in-trace (unknown at engine init).
> Attribute keys are listed in `tools/hwsim/docs/ws9-new-fields.md`. Unit-test contexts pass no
> config and keep `custom_attributes` empty.

### G7 (minor) — Parser hazards, not instrumentation gaps

- `task_manager_loop_thread`/queue resources are re-declared per query (67 instances/66
  queries) — treat thread identity as (name, span), not uuid.
- Declared queue/channel capacities are `u64::MAX` placeholders; derive occupancy from usages.
- OOM-rescheduled work appears as a *new* task FSM with no link to the failed one
  (`success:false` + a fresh `Created` on the same pipeline); `reschedule_intermediate`
  placement origin marks its lazily-registered inputs.
- The engine numeric `query_id` ↔ telemetry query UUID join lives only in the Sirius log.

---

## (d) How to parse (for WS6)

### Reading a session

1. Pick a session dir `<output_dir>/<session-uuid>/`; `model.qmi` gives quent + Sirius commits
   (schema provenance — bail if the model commit doesn't match what your parser expects).
2. Read every `*/​*.ndjson`; each line → `(entity_type = dirname, id, timestamp, data)`.
3. **Ordering:** lines within a file are *not* globally timestamp-sorted (multi-threaded
   emission). For FSM entities, order events per `id` by `seq` (verified contiguous from 0,
   no gaps, terminal `"state":"Exit"`). Plain entities have no seq (order by timestamp; they
   are declarations, order rarely matters).
4. **Timestamps:** u64 Unix nanoseconds, but derived from one process-wide
   `(Instant, SystemTime)` anchor captured at first use (quent `crates/time/src/lib.rs:67-72`)
   — monotonic within a session, safe to subtract; do not compare across sessions/processes.

### Id graph (all ids are UUIDv7 strings)

```
engine.id ──< worker.Init.parent_engine_id
          ──< query_group.Declaration.engine_id
          ──< gpu_device.Declaration.parent_group_id ──< thread_group.parent_group_id
          ──< memory / channel / memory_tier .parent_group_id     (channels: source_id/target_id → memory)
query.Init.query_group_id → query_group
plan.Declaration.parent.query_id → query ;  plan.edges[].source/target → port ids
operator.Declaration.plan_id → plan          (operator == Sirius pipeline)
port.Declaration.operator_id → operator
task.Created.pipeline_uuid → operator        (task → pipeline → plan → query)
task.Queued.queue.resource_id → task_queue
task.Routing/Reserving/Downgrading.manager_thread.resource_id → task_manager_loop_thread
task.Preparing/Computing.executor_thread.resource_id → executor_thread (→ thread_group → gpu_device = GPU binding)
task.Preparing/Computing.reservation.resource_id → memory_tier
data_batch.Constructed.producer_pipeline_uuid → operator
data_batch.Stationary.memory.resource_id → memory ;  InTransit.channel.resource_id → channel
batch_placement.BatchRegistered.{pipeline_uuid → operator (consumer), port_uuid → port,
                                 batch_id ↔ data_batch.Constructed.data_batch_id (u64, not uuid)}
batch_placement.BatchPackaged/BatchProcessing.task_uuid → task
```

Note the two batch id spaces: `data_batch` entity uuid (per FSM instance) vs the numeric
process-unique `batch_id`/`data_batch_id` (shared by a batch's placements across consumers).

### Recipes

- **Query wall time:** query `Init`→`Exit`; execution phase = `Executing`→`Exit`.
- **Pipeline DAG:** nodes = operators of the query's plan; edges = plan.edges mapped through
  port→operator.
- **Task timeline:** spans between consecutive task transitions; label each span by the
  *from*-state. Per-operator durations: `Computing(op_i)` timestamp → next transition.
  Queue wait = Queued→(Routing|Reserving); reservation wait = Reserving→(Preparing|Downgrading).
- **Task→GPU:** `executor_thread` resource → its `thread_group.parent_group_id` → `gpu_device.ordinal`.
- **Effective H2D bandwidth per task:** `Preparing.input_bytes ÷ (Computing[0].ts − Preparing.ts)`
  (only when `origin_tier != target_tier`); per-batch: bytes ÷ InTransit span, direction from
  channel name.
- **Tier fill level over time:** sum, at each instant, `capacity_bytes` of all usages that
  reference the tier/memory resource from currently-open states (placements' `tier`,
  data_batches' `memory`, tasks' `reservation`). Placement bytes and reservation bytes
  **overlap** (a reservation covers input materialization) — model them as separate series,
  as the analyzer does (`task_working_space` vs batch residency in
  `rust/crates/telemetry/analyzer/src/lib.rs:80-86`).
- **Task-level dependencies (v0 heuristic):** task T consumed batches B (BatchPackaged with
  T's uuid). For each B: producer pipeline P from data_batch; producer task = the P-task whose
  [first Computing, Finalizing] window contains B's Constructed timestamp (unique when P ran
  single-threaded; otherwise pick any/all overlapping — pessimistic edges are safe for
  replay). Scan pipelines' tasks have no upstream tasks (their inputs come from the
  uninstrumented scan manager).
- **Spill/downgrade activity:** `Downgrading` states (trigger) + placement tier-change
  self-transitions & gpu→host `InTransit` (effect).
- **Queue depth:** count overlapping task-`Queued` usages per queue resource (declared
  capacities are placeholders).
- Rust references if needed: FSM reconstruction `analyzer/src/task.rs`, resource handling
  `analyzer/src/model.rs`, tier/data-flow aggregation `analyzer/src/lib.rs` — but the above
  suffices without reading them.

### Volume expectations

~45 k events/query (SF100-class TPC-H): dominated by batch_placement (≈20 k) and data_batch
(≈16 k) — set `enable_batch_events: false` to halve volume at the cost of losing per-edge
volumes and tier residency. Task events ≈10 k/query. ndjson ≈10 MB/query.

---

## Verdict for a v0 simulator

**Yes — current quent output is sufficient to build a v0 discrete-event replay** (pipeline
DAG + per-task state timelines + queue waits + synchronized per-operator durations + measured
H2D transfer spans/bytes + tier capacities & residency), **with two hard caveats**:

1. The `io_bandwidth` knob has no trace input: scan read time is fused into `GPU_SCAN`
   Computing (G1) — v0 can only scale it as part of "scan compute", which breaks the
   flagship back-pressure scenario until G1 lands (or WS2's nsys extraction supplies the
   read/decode split).
2. `gpu_compute` vs `gpu_mem_bandwidth` cannot be separated from quent alone (G4) — v0 should
   scale whole-operator durations and rely on WS2 for the split; add the task-uuid to the
   NVTX labels to make that join exact.

Also: capture at least one memory-pressured trace (smaller pool config) before trusting the
spill path — the existing sample contains zero `Downgrading` events.
