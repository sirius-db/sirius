# Simulator Design — Discrete-Event Replay of Sirius Task Graphs (WS6, v0)

**Status:** implemented and validated against the sample trace, 2026-08-04.
Code: `tools/hwsim/sim/` (pure Python 3 stdlib; CLI `python -m hwsim`).
Inputs: a Quent ndjson telemetry session (parsing per WS1's spec in
[`quent-extraction.md`](quent-extraction.md) §d).
Validation trace: `telemetry_data/019fbafc-e2cc-7bb3-b1ba-915fd149e7f5`
(654 MB, 66 = 22×3 TPC-H queries, 67,818 tasks, ~200k batches, 1 GPU
(GB300), 4 executor threads, GPU pool 229 GB, unpressured — zero
`Downgrading` events).

## 1. What the simulator is

A discrete-event simulation that **replays the traced task graph under a
resource model**, then **re-times and re-flows** it when hardware knobs
change. It is *not* per-phase time scaling: queue waits, memory-admission
blocking and channel contention are never read from the trace — they emerge
from the resource model. Only three things are taken from the trace per task:

1. **work quanta** — the synchronized per-operator `Computing` spans, the
   `Preparing` materialization span + bytes, and the granted reservation size;
2. **structure** — which tasks feed which (via batches), on which device,
   in which traced dispatch order;
3. **fixed scheduling overheads** that v0 does not model mechanistically
   (task-creation lag, `Created→Queued`, `Reserving→Preparing` grant cost,
   `Finalizing→Exit` tail) — replayed as constants per task.

Everything else (when a task is created, queues, is admitted, starts
preparing, computes, finishes; how full the pool is; how busy the channel is)
is computed by the engine.

## 2. Event model

Per-task lifecycle in the sim (mirrors the traced FSM;
`docs/super-sirius/pipeline-execution.md` is the engine-side reference):

```
release ──pre_queue──▶ enqueue ──[emergent: slot + head-of-line + memory]──▶
admit ──grant_ns──▶ prep (channel transfer or fixed span) ──▶
compute (Σ scaled op spans) ──tail──▶ finish
```

- **release**: root tasks (scan-source pipelines; their inputs come from the
  uninstrumented scan manager) release at their traced `Created` offset from
  the query's `Executing` timestamp. Dependent tasks release at
  `max(dep finish) + creation_lag`, where `creation_lag = traced Created −
  max(traced dep Exit)`, clamped ≥ 0. At knobs=1 this reproduces traced
  creation exactly; under knobs the lag ports the task-creator latency.
- **admission** (per GPU, mirroring `gpu_pipeline_executor::manager_loop`):
  a task is admitted when (a) it is at the head of the device queue,
  (b) a thread slot is free, and (c) the memory pool can grant
  `min(reservation_bytes, capacity)`. The head **blocks the queue**
  (head-of-line) exactly as the real manager loop does — it pops one task
  and cannot pop the next until the head's reservation is granted.
- **prep**: if `origin_tier != target_tier`, the materialization is served
  by a fluid transfer channel (§3.3); otherwise the traced span is replayed
  (same-tier prep is lock/allocate work, not a link transfer).
- **compute**: each traced `Computing(op)` span scaled by the knob rules
  (§4); the thread slot is held from admit to finish.
- **finish**: releases the slot + reservation, publishes output batches
  (GPU-resident bytes enter the pool), consumes input batches (bytes leave
  the pool when the *last* consumer finishes), and notifies dependents.

Query wall = `max(task finish) + finish_tail`, where `finish_tail` is the
traced gap between the last task Exit and the query Exit (result
collection); the compared metric is the `Executing→Exit` execution phase.

### 2.1 Dependency inference (gap G2)

The trace has no producer-task field on batches, so task→task edges are
derived exactly as WS1's recipe prescribes:

- consumer side is exact: `BatchPackaged.task_uuid` per placement;
- producer side is inferred: a batch's `Constructed` timestamp is matched to
  the execution windows `[Preparing, Exit]` (±5 ms slack) of the producing
  pipeline's tasks. Multiple candidates ⇒ flagged ambiguous, and the
  **earliest-finishing** candidate is chosen (least-constraining edge).
  On the sample trace **98% of batches are ambiguous** (197,358 / 200,409)
  because 4 executor threads run same-pipeline tasks concurrently — this is
  the single strongest argument for the G2 `producer_task_uuid` field.
- **full-barrier edges**: for each plan edge, if every consumer task was
  created after every producer task finalized, the edge is treated as a FULL
  barrier (build-before-probe, partition→merge) and all producer tasks
  become dependencies of every consumer task. This is observational — it
  only encodes orderings that actually held in the trace.
- **false-dep filter**: any inferred dependency whose producer finalized
  > 1 ms after the consumer was created is physically impossible
  (publish precedes finalize) and is dropped.
- batches with no producer window (scan-manager staging batches constructed
  before any task of their pipeline ran) are treated as externally available;
  they are HOST-resident and do not occupy the GPU pool.

Mis-attribution can in principle produce cycles; the engine breaks a cycle
deterministically and counts it (`dep_cycle_breaks` — zero on all 66 sample
queries).

### 2.2 Queue discipline (a measured decision)

Among *released* tasks, the device queue dispatches in **traced queue-entry
order** (`task.t_queued`), not simulated arrival order. Rationale: the traced
order encodes task-creator/scheduler decisions (hint-chain recursion,
CONCAT byte-threshold batching, build-side priority) that v0 does not model.
With arrival order, a 5 ms upstream drift on `tpch_q09_iter3` flipped the
queue position of 16 tiny PARTITION tasks against a 470-task scan burst
(~3 s of service) and produced a **+24.2% wall error**; with traced order the
same query reproduces at **+0.11%**. Admission *timing* remains fully
emergent — only relative order is anchored. `queue_order="arrival"` is kept
for experiments and unit tests, and the anchoring is an explicit v0
limitation for extreme knob settings (the real creator could reorder).

## 3. Resource model

### 3.1 Executor threads
`N` slots per GPU, counted from `executor_thread` resources (4 on the sample
box). A task holds one slot from admission to finish — including the whole
Preparing transfer, matching the real worker thread blocking on
`stream.synchronize()`.

### 3.2 GPU memory pool (admission ⇒ back-pressure)
Capacity = `memory.MemoryOperating.capacity_bytes` × `gpu_mem_capacity`.
Occupancy = **active reservations** + **published-but-unconsumed
GPU-resident batch bytes** (batch bytes/tier from `BatchRegistered`,
lifetime = producer finish → last consumer finish). Reservations use the
traced *granted* bytes (`Preparing.reservation.capacity_bytes`), clamped to
capacity exactly as the manager loop clamps to `get_max_memory()`. Input
batches materialized from HOST are covered by the reservation
(`bytes_to_materialize`), so they are not double-counted; GPU-resident
inputs are counted as residency, not reservation — matching the estimator's
accounting.

**Progress guarantee:** if the simulation stalls completely (event heap
empty, a queue head memory-blocked, nothing running that could free bytes)
the head is force-admitted; occurrences are counted (`forced_admissions`).
In v0 this was the only stand-in for the engine's `request_downgrade()`
path. **Since the G5 work this is a last-resort fallback only**: the spill
layer (§3.2.1) evicts/downgrades before it triggers.

### 3.2.1 Spill / downgrade layer (G5 — see [spill-model.md](spill-model.md))

The pool now has a downgrade mechanism with three modes (`spill_mode="auto"`
picks per graph/knobs; `off` reproduces v0 byte-identically and is what
unpressured knobs=1 runs resolve to):

- **replay** (trace contains `Downgrading`/OOM-rescheduled tasks): idle
  resident batches are evicted at zero time cost when the head is
  memory-blocked — the traced spans already carry every real cost (downgrade
  waits inside grant_ns, re-upgrades inside retry Preparing spans, recompute
  waste = the traced failed attempts). This fixed the +2743% replay error on
  the E2-mid pressured capture to +0.25…+0.41%.
- **model** (capacity knob on an unpressured trace): mirrors the real
  admission/downgrade policy — reservation-dominated deficits block
  (waiting suffices; the real engine emits no Downgrading there);
  resident-data overshoot triggers LRU downgrade sweeps to the engine's
  `downgrade_stop_fraction`, bounded by the HOST pool, stalling the manager
  at measured rates; still-ungrantable heads dispatch-and-OOM as the real
  engine does — calibrated 50 ms reschedule cycles that bank partial
  progress (`reschedule_intermediate` semantics), free consumed inputs and
  materialize a calibrated fraction of their outputs host-side. Slowdown is
  emergent. Calibrated on the E2 pressure captures: q21@0.25× +38.8%,
  q9@0.15× −20.4% (v0: −82%/−58%, inverted); held-out transfer is
  one-sided — see spill-model.md §5–6 for the full table and caveats.

### 3.3 Transfer channels (fluid model)
One channel per `(origin_tier, target_tier, device)`. Each transferring task
demands its **traced achieved rate** (`prep_bytes / traced prep span`) ×
`c2c_bandwidth`; channel capacity is the **peak aggregate rate observed in
the trace** (line-sweep over overlapping Preparing spans; 709.6 GB/s
HOST→GPU on the sample) × `c2c_bandwidth`. When Σ demand > capacity all
active transfers are throttled proportionally (fluid sharing, re-solved at
every transfer start/finish). Properties:

- at knobs=1 with trace-like overlap, no transfer is throttled and every
  Preparing span reproduces exactly;
- when *other* knobs increase overlap beyond anything the trace exhibited
  (faster compute ⇒ more concurrent scans), the channel saturates — an
  emergent cross-dependency, not a rule;
- the capacity is a *lower bound* on the real link (the trace may never have
  saturated it) — documented, and overridable in code.

### 3.4 What deliberately stays replayed (v0 omissions)
Manager-loop serialization (~10–20 µs/admission), task-creator thread
contention, Routing hop latency, and the scan manager/IO subsystem
(invisible per G1) are not modeled as resources; the first three appear only
inside the replayed per-task constants. Their absence is measurable: the
self-consistency bias is −0.2% median (sim slightly fast, §5).

## 4. Knob semantics and fidelity caveats

Multipliers, default 1.0; >1 = faster/bigger. Every knob whose fidelity is
degraded by a known WS1 gap emits a WARNING when moved.

| knob | mechanics | honest caveat |
|---|---|---|
| `c2c_bandwidth` | transfer demand rates and channel capacities scale by k; durations re-derived from bytes | Preparing includes GPU decompression that is **SM-bound** on GB300 (nsys doc §5.1, memory note); scaling the whole span with the link is optimistic for k>1. SM-driven zero-copy C2C traffic inside kernels is invisible (nsys §5.2). |
| `gpu_mem_capacity` | pool capacity scales; admission waits emerge; below the spill knee the **calibrated downgrade model** runs (§3.2.1, [spill-model.md](spill-model.md)) | calibrated on two SF1000 pressure points (q21@0.25× +39%, q9@0.15× −20%; v0 was −82%/−58% *in the dangerous direction*). Held-out transfer one-sided; treat sub-knee predictions as order-of-magnitude ±40% and expect over-warning at shallow pressure. |
| `gpu_compute` | all `Computing` spans ÷ k | conflates SM throughput, HBM bandwidth, launch overhead and host glue (G4). Launch overhead does not scale with clocks ⇒ optimistic for k>1. |
| `gpu_mem_bandwidth` | placeholder: spans ÷ min(gpu_compute, gpu_mem_bandwidth) | pessimistic roofline stand-in until the nsys physics join supplies a per-kernel compute/HBM split (G4). |
| `io_bandwidth` | `GPU_SCAN` Computing spans ÷ k | **G1: no I/O events exist.** Scan read time is fused with GPU decode in one span; the knob scales both, and the flagship "faster I/O → scan blocks on memory" scenario cannot be reproduced until G1 instrumentation lands. On the sample trace the run was host-cached, so disk I/O is doubly invisible. |
| `cpu_mem_capacity` / `cpu_mem_bandwidth` / `cpu_compute` | accepted, no-op | no host-pool admission or host-bandwidth telemetry in v0. |

Composition: `GPU_SCAN` spans divide by `io_bandwidth × gpu_speed`; all other
ops by `gpu_speed = gpu_compute` (or the min-roofline above).

## 5. Self-consistency (acceptance #1)

All 66 queries, knobs = 1.0, simulated vs traced `Executing→Exit` wall:

**|error|: median 0.17%, p90 0.52%, worst 1.20%** (66/66 within ±1.2%).
Signed bias −0.2% (sim slightly fast — unmodeled µs-scale manager/creator
serialization accumulates on task-storm queries).

| query (iter1/2/3 err %) | | | |
|---|---|---|---|
| q01 −0.42 / −0.33 / −0.35 | q02 −0.04 / −0.14 / −0.14 | q03 −0.43 / −0.23 / −0.15 | q04 **−1.00 / −1.20 / −1.15** |
| q05 −0.31 / −0.33 / −0.24 | q06 −0.16 / −0.43 / −0.29 | q07 −0.13 / −0.08 / −0.11 | q08 −0.31 / −0.35 / −0.37 |
| q09 −0.37 / −0.19 / +0.11 | q10 −0.06 / −0.08 / −0.06 | q11 −0.20 / −0.18 / −0.23 | q12 −0.06 / −0.02 / −0.11 |
| q13 −0.31 / −0.31 / −0.35 | q14 −0.52 / −0.72 / −0.49 | q15 **−0.95 / −0.61 / −0.97** | q16 −0.07 / −0.05 / −0.03 |
| q17 −0.11 / −0.11 / −0.26 | q18 −0.07 / −0.06 / −0.07 | q19 −0.02 / −0.02 / −0.03 | q20 −0.03 / −0.03 / −0.03 |
| q21 −0.11 / −0.07 / −0.08 | q22 −0.13 / −0.18 / −0.28 | | |

(Full per-query CSV: `python -m hwsim selfcheck <trace> --csv ...`.)

**Worst outliers and what they reveal:**

- **q04 (−1.0…−1.2%)**: tasks with ~70 dependencies (COLUMN_DATA_SCAN /
  PARTITION fan-ins) finish up to 10 ms early; the emergent queue waits are
  slightly shorter than traced because per-admission manager cost and
  creator-loop serialization are not modeled. A serial "manager" resource
  with per-admission cost would close this (v1).
- **q15 (−0.6…−1.0%)**: same signature.
- **Historical q09_iter3 (+24.2% before the queue-discipline fix)**: the
  most instructive failure — see §2.2. It demonstrates that *scheduling
  order*, not span accuracy, is the fidelity frontier of a replay simulator,
  and directly motivates G2 (`producer_task_uuid`) plus creator-decision
  telemetry (G5) for v1.

## 6. Sensitivity sanity (acceptance #2)

Three heavyweight queries (largest walls / transfer volumes). All curves are
monotonic and saturating; "binding" is the reported binding constraint.

### 6.1 `c2c_bandwidth` × {0.25, 0.5, 1, 2, 4}

Δ wall vs simulated baseline:

| query | 0.25× | 0.5× | 1× | 2× | 4× | binding (all points) |
|---|---|---|---|---|---|---|
| q21_iter2 (4382 ms, 340 GB H2D) | **+58.9%** | +19.2% | 0 | −3.6% | **−5.4%** | executor_threads |
| q09_iter2 (2340 ms, 284 GB) | **+95.7%** | +31.7% | 0 | −6.4% | **−6.7%** | executor_threads |
| q19_iter2 (1520 ms, 325 GB) | **+139.4%** | +45.4% | 0 | −0.7% | **−1.1%** | executor_threads |

Shape: steep degradation below 1× (transfer time comes to dominate the
thread-held task span) and fast saturation above 1× — the win from a faster
link caps at the compute/dependency floor. The shift is visible in thread
utilization: q19 thread-busy falls 96.7% → 62.2% between 1× and 4× (threads
stop being the bottleneck; the dependency chain is), while at 0.25× busy is
~99% with all of it transfer-inflated. Channel throttling stays 0 in these
sweeps because peak concurrency is capped at 4 by the executor threads —
i.e. the *link* was never the binding constraint on this box; the **thread
slots holding transfers** are. That itself is an emergent, non-obvious
finding.

### 6.2 `gpu_mem_capacity` × {0.02, 0.05, 0.1, 0.25, 0.5, 1, 2}

Δ wall vs simulated baseline (fa = forced admissions ≈ would-be spills):

| query | 0.02× | 0.05× | 0.1× | 0.25× | 0.5× | 1× | 2× | baseline peak pool |
|---|---|---|---|---|---|---|---|---|
| q21_iter2 | +1431% (fa 2620) | +1358% (fa 1842) | +1312% (fa 1605) | **+1113% (fa 598)** | 0 | 0 | 0 | 76.3 GiB = 36% |
| q09_iter2 | +1426% (fa 1618) | +1399% (fa 1596) | +1343% (fa 1475) | **+0.6% (fa 0, mem-blocked 41 ms)** | 0 | 0 | 0 | 55.7 GiB = 26% |
| q19_iter2 | +214% (fa 921) | +32.9% (fa 32) | 0 | 0 | 0 | 0 | 0 | 19.2 GiB = 9% |

Shape: exactly flat above the query's traced working set (a capacity knob
cannot help an unpressured query), a knee at peak-pool ≈ capacity
(q09 at 0.25× is the marginal point: 99.6% pool peak, 41 ms of emergent
memory blocking, +0.6%), then a cliff where the binding constraint flips
`executor_threads → gpu_memory`, thread utilization collapses (99% → 7%)
and admission serializes. The cliff is **honestly pessimistic**: the real
engine would downgrade batches to HOST and keep more parallelism; v0 waits
instead (every `forced_admission` marks a would-be spill). Calibrating that
requires a memory-pressured trace (v1 roadmap).

### 6.3 `gpu_compute` × {0.5, 1, 2, 4, 8} (q09_iter2)

−36% is unreachable: +64.1% at 0.5×, −18.9% at 2×, −28.2% at 4×, **−32.9% at
8×** — saturating against the un-scaled transfer + dependency floor
(thread-busy falls 97% → 61%). Cross-dependency in action with zero
hand-coded rules.

## 7. How back-pressure emerges (acceptance #3, unit-tested)

No knob has a scheduling rule attached. The only mechanisms are: admission
needs slot+memory with head-of-line blocking; resident batches occupy the
pool from producer-finish to last-consumer-finish; transfers share finite
channel capacity; dependents release on producer finish. The unit tests
(`tests/test_engine.py`, 19 tests, all analytic) pin the canonical behaviors:

- `test_capacity1_pool_serializes_two_producers` — the required toy case:
  two producers through a capacity-1 pool run 2T, not T, and report
  `gpu_memory` as binding;
- `test_resident_batches_backpressure_consumer_chain` — a producer's output
  bytes block the *next* producer until the consumer frees them;
- `test_head_of_line_memory_block` — a fitting small task cannot overtake a
  blocked head (manager-loop semantics);
- `test_gpu_compute_saturates_on_channel_floor` — 4× compute yields 1.33×,
  100× compute saturates at the transfer floor (300→225→~201 µs analytic);
- `test_c2c_speedup_throttled_by_memory_admission` — 4× link yields 1.43×
  because pool admission caps transfer concurrency (500→350 µs analytic);
- channel fluid-sharing tests (proportional throttling, staggered arrivals),
  knob scaling tests, traced-vs-arrival queue-order semantics, forced
  admission / deadlock resolution.

## 8. Performance

654 MB session, ~3M events: cold parse+build 17 s (streaming line parse;
substring pre-filters skip ~60% of batch-event `json.loads`), pickled model
cache → warm start ~1 s including one query simulation; a single query
simulates in 30–300 ms; the full 66-query selfcheck runs in ~3 s warm.
Stdlib only.

## 9. Model omissions discovered (ranked by measured impact)

1. **Scheduler/creator decision logic is replayed, not modeled** (queue
   order anchored to trace; creation lags constant). Bounded here by the
   q09_iter3 experiment (+24% if ignored); under aggressive knobs the real
   creator could genuinely reorder. → needs G5-style creator telemetry or a
   task-creator model in v1.
2. **Spill/downgrade** — ~~not priced~~ **addressed 2026-08-05** with the
   G5 spill layer (§3.2.1, [spill-model.md](spill-model.md)), calibrated on
   the WS8 E2 pressure captures. Remaining gap: a single global
   `spin_output_host_fraction` cannot express reality's per-stage split
   (partition stage pegged, downstream clean) — the born-tier telemetry ask
   in spill-model.md §7 would close it.
3. **Manager-loop / creator serialization (~10–20 µs/admission) not
   modeled** → the −0.2% median bias and the −1.2% q04 worst case.
4. **Preparing is treated as pure link transfer** though it contains
   SM-bound decompress (WS2 §5.1) — c2c>1 optimistic; a compute/transfer
   split of Preparing needs the nsys join (G4).
5. **Channel capacity = observed peak aggregate** (709.6 GB/s H2D here) —
   a lower bound on the real link; transfers were thread-capped at 4
   concurrent in the whole trace.
6. **No output-size model (G3)** — batch bytes are replayed, so knobs that
   would change *data volumes* (plan changes, selectivity) are out of scope.
7. **Host pool not modeled** — `cpu_mem_capacity` no-op; scan staging
   batches assumed available at their traced times regardless of knobs (G1).

## 10. v1 roadmap (keyed to WS1's gap list)

| gap | v1 item | what it unlocks |
|---|---|---|
| G1 (no I/O events) | consume split-level read/decode events once instrumented (`io_request` FSM or sub-Computing states); model disk/S3 as a bandwidth resource feeding scan staging batches | a real `io_bandwidth` knob and the flagship "faster I/O → memory back-pressure at the scan" scenario; also unlocks `cpu_mem_capacity` (staging pool admission) |
| G2 (no producer-task id) | switch attribution to the emitted `producer_task_uuid`; drop the time-window heuristic (98% ambiguous today) | exact task DAG; removes the dep-cycle guard; enables per-edge volumes |
| G3 (no rows/output bytes) | use `rows_in/rows_out/output_bytes` per (task, operator) | volume-aware re-timing (operator cost models), plan-change what-ifs |
| G4 (no GPU-busy split) | join the paired nsys capture by structural key (query, pipeline, operator, task ordinal — WS2 §4.4); split spans into kernel-busy / launch / sync / memcpy; per-kernel compute-vs-HBM classification (WS2 §5.1) | separate `gpu_compute` from `gpu_mem_bandwidth`; stop scaling launch overhead; split Preparing into decompress (SM) + copy (link) |
| G5 (wait reasons, downgrade events) | **spill model shipped 2026-08-05** ([spill-model.md](spill-model.md)): downgrade sweeps + OOM-reschedule cycles calibrated on the E2 pressure captures. Still wanted: born-tier batch telemetry, retry linkage, downgrade-executor FSM (spill-model.md §7); manager loop as a serial server; split-connector queue telemetry | replaces forced-admission warnings with calibrated spill predictions; the remaining asks would replace the calibrated β with measured per-stage truth; closes the −0.2% bias; scan starvation becomes visible |
| G6 (config snapshot) | read thread counts / pool config / hardware from `engine.Init.custom_attributes` when populated | drop the count-resources heuristics; portable across boxes |
| — (validation) | capture a **memory-pressured trace** (shrunk pool YAML) and a **cold-I/O trace** (cache disabled); WS7/WS8 then compare simulator predictions against the physically throttled runs (WS3–WS5) | closes the loop on knob fidelity |
