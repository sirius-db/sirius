# Pipeline Execution

This document explains how Sirius executes queries on the GPU through its pipeline execution framework. It covers physical operators, pipeline construction, task creation, and the GPU executor.

> **Note:** This document evolved from `docs/onboarding-docs/pipeline-execution.md` and expands on the original with full coverage of the pipeline executor, GPU executor, task scheduling, OOM handling, and error recovery.

## Overview

Sirius translates a DuckDB physical plan into a graph of **pipelines**. Each pipeline is an ordered list of operators:

```
operators[0] --> operators[1] --> ... --> operators[N-1]
   (source)                                  (sink)
```

- `source` is an alias for `operators[0]` — the first operator in the list
- `sink` is an alias for the last operator — it also has a `sink()` method called after the execute loop
- `operators` contains **all** operators, including source and sink (unlike DuckDB where source/sink are separate from the operators list)

Each operator's `execute()` method is called in sequence by `compute_task()`. After the loop, the sink's `sink()` method is called via `publish_output()` to push results to downstream ports.

Pipelines are connected through **ports** on operators. When a sink pushes output into ports, a **task creator** monitors data availability and creates `gpu_pipeline_task` objects. These tasks are scheduled on the `gpu_pipeline_executor`, which manages GPU memory, CUDA streams, and a thread pool.

```
Pipeline 1                  Pipeline 2        Pipeline 3
[HASH_GROUP_BY]  --repo(FULL)--->  [PARTITION]  --repo(FULL)--->  [MERGE_GROUP_BY]
                   port "default"                 port "default"
                 (data_repository)              (data_repository)
```

For example, in a GROUP BY query after pipeline splitting (see [Physical Plan Generation](physical-plan-generation.md#hash_group_by)):
- Pipeline 1 performs partial aggregation, pushing results into a data repository with `FULL` barrier
- Pipeline 2 partitions the partial aggregates
- Pipeline 3 merges partitioned results into the final output

## Physical Operators

**File:** `src/include/op/sirius_physical_operator.hpp`, `src/op/sirius_physical_operator.cpp`

See [Operators](operators.md) for the complete operator reference.

### Execution Model

After pipeline finalization, `source` and `sink` are simply aliases for the first and last operator in the `operators` list. During execution:

- `execute(input_data, stream)` — called on **every** operator in the pipeline by `compute_task()`
- `sink(output_data, stream)` — called on the **last** operator by `publish_output()` to push results to downstream ports

See [Operators](operators.md) for the complete operator reference.

### Ports

Ports pass data **between pipelines**. Each port is an input buffer on an operator:

```cpp
struct port {
    MemoryBarrierType type;              // PIPELINE, PARTIAL, or FULL
    cucascade::shared_data_repository* repo;
    shared_ptr<sirius_pipeline> src_pipeline;
    shared_ptr<sirius_pipeline> dest_pipeline;
};
```

- **`PIPELINE` barrier** (streaming): downstream consumes batches as they arrive
- **`PARTIAL` barrier**: downstream can consume incrementally but respects pipeline boundaries
- **`FULL` barrier**: downstream waits for upstream to complete entirely

When a sink's `sink()` method produces output, it pushes each batch into downstream ports via `next_port_after_sink`.

## Tasks

### Class Hierarchy

```
parallel::itask                          // base: local_state + global_state + execute(stream)
  └── sirius_pipeline_itask              // adds compute_task() / publish_output() split
        └── gpu_pipeline_task            // concrete: executes a pipeline on GPU
```

### `gpu_pipeline_task`

**File:** `src/include/pipeline/gpu_pipeline_task.hpp`, `src/pipeline/gpu_pipeline_task.cpp`

**State classes:**
- `gpu_pipeline_task_global_state` — holds the `sirius_pipeline` to execute
- `gpu_pipeline_task_local_state` — holds input `data_batch` vector, memory reservation, `_start_operator_index` (for OOM resume), `retry_count`

**`compute_task(stream)`** iterates through **all** operators in the pipeline (source through sink inclusive), calling `execute()` on each:
```cpp
auto operators = pipeline->get_operators();  // includes source and sink
for (size_t i = start_index; i < operators.size(); i++) {
    operator_input_output_data = run_one_operator(operators[i], input, stream, ...);
}
return operator_input_output_data;
```

On OOM at any operator, throws `oom_reschedule_exception` with the current operator index for later resumption.

**`publish_output(batches, stream)`** then calls the sink's `sink()` method to push results to downstream ports:
```cpp
pipeline->get_sink()->sink(output_data, stream);
```

**`execute(stream)`** handles the full flow:
1. Lock each input batch and convert to GPU if needed (`lock_or_prepare_batch`)
2. Call `compute_task()` (iterates all operators' `execute()`)
3. Call `publish_output()` (calls sink's `sink()` to push to downstream ports)
4. Processing handles released automatically on scope exit

The **destructor** calls `pipeline->mark_task_completed()` to update pipeline completion tracking.

**`get_output_consumers()`** returns the first operator of each parent pipeline — these downstream operators are scheduled next by the GPU executor.

## Per-task-device contract under SCHED-RR

This section is the authoritative per-task-device contract every operator MUST honor when reading a memory space from one of its input batches under multi-GPU execution.

### Why this contract exists

**Pre-Phase-14 history.** Before Phase 14 (`feat/sched-rr-distribution`) landed, the task scheduler stored its per-GPU executors in a `std::unordered_map<int, std::unique_ptr<gpu_pipeline_executor>>`. The code path in `task_scheduler::management_eventloop` that picked a default GPU for a preference-less task did so via:

```cpp
int target_device_id = _gpu_executors.begin()->first;
```

That `begin()` is hash-bucket-ordered — but for any single process it returns the *same* GPU on every call. Every preference-less source-pipeline task (metadata scan, parquet scan with no locality hint) piled onto whichever GPU happened to live in the first hash bucket. The implicit-and-undocumented contract was: "default GPU is `_gpu_executors.begin()->first`."

**Phase 14 SCHED-RR change.** Phase 14 replaced the `unordered_map` with a `std::map<int, std::unique_ptr<gpu_pipeline_executor>>` (deterministic ascending-by-`device_id` iteration), added `std::atomic<size_t> _no_pref_rr_counter{0}`, and inserted a round-robin walk in `management_eventloop` that distributes preference-less tasks across all configured GPUs. Source-pipeline tasks now genuinely land on multiple GPUs within a single query — exactly what an N-GPU configuration is supposed to deliver.

**The hazard this exposes.** Several operators read `valid_batches[0]->get_memory_space()` (or an equivalent expression on a single input batch) as the authoritative target memory space, then perform their concat/merge/sort directly on that space. Pre-Phase-14, this was *accidentally* safe — every batch in the input vector was already on the implicit "default GPU" because every upstream task was dispatched to that same default. Under SCHED-RR, that accident is gone. If an operator reads `batches[0]->get_memory_space()` without a guarantee that *all* batches in the input vector are colocated on that space, it can silently produce wrong results, mis-allocate, or skip data on the other GPU.

The fix is not to patch every read site to detect cross-GPU input. The fix is the upstream contract below: every operator's input batches are colocated by the task scheduler **before** the operator's `execute()` runs, so reading `batches[0]->get_memory_space()` is a SAFE alias for the task's reservation device.

### The contract

> **Every operator's input batches MUST arrive on the task's reservation device.** Operators MUST NOT use `batches[0]->get_memory_space()` as the authoritative target memory space; that read is acceptable only as an alias for `target_space` *after* `prepare_for_processing` has run upstream. New operators that read `get_memory_space()` from a batch they did not themselves construct MUST add an `INVARIANT` comment naming the upstream enforcement path (see "For new operator authors" below).

This is a four-layer contract: the scheduler picks `target_space`, the task layer enforces it, the per-batch lock protocol implements it, and the operator layer relies on the postcondition. Each layer is shown below with the source line where it lives.

### How the contract is enforced

**Layer 1 — `gpu_pipeline_task::execute` captures `target_space` from the task's reservation.**

`src/pipeline/gpu_pipeline_task.cpp` (`release_reservation()`):

```cpp
auto reservation = local_state.release_reservation();
if (!reservation) { throw std::runtime_error("GPU pipeline task requires a memory reservation"); }
const auto* requested_memory_space =
  reservation != nullptr ? &reservation->get_memory_space() : nullptr;
```

The reservation was attached by the GPU executor's manager loop (see [GPU Pipeline Executor](#gpu-pipeline-executor) above) on the SCHED-RR-chosen device. `requested_memory_space` is the authoritative target for every input batch this task will touch.

**Layer 2 — `gpu_pipeline_task::execute` calls `prepare_for_processing` on the operator-data input.**

This is the gate. `compute_task(stream)` — which iterates the pipeline's operators and calls each one's `execute()` — does not run until `prepare_for_processing(requested_memory_space, stream)` has returned. Every batch is available read-only on `requested_memory_space` by the time any operator sees it.

**Layer 3 — `pipelineable_operator_data::prepare_for_processing` walks each batch and locks-or-clones it.**

`src/op/sirius_physical_operator.cpp` — the method returns `void` and stores the resulting accessors:

```cpp
void pipelineable_operator_data::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  std::vector<::cucascade::read_only_data_batch> ro_batches;
  for (const auto& batch : _data_batches) {
    auto ro = pipeline::lock_or_prepare_batch(batch, requested_memory_space, stream);
    ...  // throws sirius::internal_exception if a batch cannot be locked
    ro_batches.emplace_back(std::move(*ro));
  }
  _read_only_data_batches = std::move(ro_batches);
  _data_batches = std::nullopt;  // accessors are the source of truth from here on
}
```

Every batch is fed through `lock_or_prepare_batch`. There is no early-exit short-circuit — partial colocation is not possible; a batch that cannot be locked throws `sirius::internal_exception`. After the walk, `_data_batches` is reset to `std::nullopt`: accessor *i* may reference a cross-GPU *clone* rather than the original batch, so the idle batch view is lazily rebuilt from the accessors rather than kept alongside them.

**Layer 4 — `lock_or_prepare_batch` does the actual clone/conversion.**

`src/include/pipeline/batch_lock_utils.hpp`:

```cpp
inline std::optional<cucascade::read_only_data_batch> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream);
```

If the batch is already on `target_space`, it is locked in place under a shared (read) lock. If it lives on a *different GPU*, it is **cloned** into the target space under the same shared lock (`read_accessor.clone_to<...>`) — the source batch is never exclusively locked and never mutated, so concurrent readers on its home GPU proceed unhindered (see [Multi-GPU Architecture](multi-gpu-architecture.md)). Host/disk-resident batches are converted via `mut_accessor.convert_to<...>` (an upgrade, not a clone). The cross-GPU route goes through `cucascade::convert_gpu_to_gpu` (peer-DMA on server hardware, automatic host-staging on consumer hardware whose chipset misreports peer-access support). For HOST-targeted conversions the helper first attempts a caller-owned reservation (`make_reservation_or_null`) and passes it to the reservation-taking `convert_to` overload, falling back with a warning to the no-reservation overload — see [Memory Management](memory-management.md).

**Postcondition.** When `prepare_for_processing` returns, every input accessor references data on `requested_memory_space`. Therefore the per-operator expression `batches[0].get_memory_space() == target_space` holds at every audited read site. Operators that walk every batch and adopt the first non-null batch's space (e.g. `sirius_physical_sort_sample.cpp`, `sirius_physical_merge_sort.cpp`, `sirius_physical_table_scan.cpp`) are safe by the same postcondition.

### The SCHED-RR distribution policy

The contract above is necessary because the scheduler distributes preference-less tasks across multiple GPUs. The distribution policy itself lives in three places.

**Storage: deterministic ordering.** `src/include/pipeline/task_scheduler.hpp:224-228`:

```cpp
/// device_id -> GPU executor. std::map (not unordered_map) so iteration
/// order is deterministic (ascending by device_id) — keeps preference-less
/// task dispatch reproducible across runs.
std::map<int, std::unique_ptr<gpu_pipeline_executor>> _gpu_executors;
std::atomic<size_t> _no_pref_rr_counter{0};
```

`std::map` gives an ascending-by-`device_id` iteration order, which makes `_gpu_executors.begin()` deterministic instead of hash-bucket-dependent and makes `std::advance(it, idx)` walk a fixed sequence.

**Per-query reset.** `src/pipeline/task_scheduler.cpp:156-160`, inside `task_scheduler::prepare_for_query`:

```cpp
// Reset SCHED-RR counter so the round-robin walk is reproducible across
// iterations of the same query (cache=table_gpu warm path keys cache
// entries by device_id; without this reset the second iteration's source
// tasks would assign to a different GPU and miss the cache entries).
_no_pref_rr_counter.store(0, std::memory_order_relaxed);
```

The reset is mandatory for `cache=table_gpu` warm-path correctness. Without it, iteration `N+1` of the same query would dispatch preference-less source tasks to a different starting GPU than iteration `N`, missing the per-device cache populated on iteration `N`. Phase 13 follow-up #17 scale-up test is the regression gate that locks this behavior in.

**Per-task round-robin walk.** `src/pipeline/task_scheduler.cpp:259-265`, inside `management_eventloop`:

```cpp
if (!have_pref && _gpu_executors.size() > 1) {
  auto idx = _no_pref_rr_counter.fetch_add(1, std::memory_order_relaxed) %
             _gpu_executors.size();
  auto it = _gpu_executors.begin();
  std::advance(it, idx);
  target_device_id = it->first;
}
```

The walk is gated on `!have_pref && _gpu_executors.size() > 1` so two configurations stay untouched:

- **1-GPU configurations.** The single executor is always picked by the line just above this block (`int target_device_id = _gpu_executors.begin()->first;`).
- **Preference-bearing tasks.** `SCHED-01/02/04` tasks (data-locality-bearing, e.g. downstream pipeline tasks consuming a specific repository) keep their `preferred_device_id` and skip the round-robin walk entirely. Locality is preserved.

Only *preference-less* source-pipeline tasks (metadata scans, parquet scans with no upstream locality) round-robin across GPUs.

### Migration note (Phase 14)

> **The pre-Phase-14 "default GPU is `_gpu_executors.begin()->first`" behavior is gone.** Any operator that hardcodes single-GPU assumptions, defaults to GPU 0, or uses `batches[0]->get_memory_space()` without going through the lock protocol upstream is now WRONG under SCHED-RR distribution. Phase 15 (cross-GPU operator-colocation audit) verified all 11 known sites; new operators MUST follow the same pattern.

If you are reading older operator code that says "all batches are expected to share the same space in practice" or similar unverified-assumption phrasing, that comment predates the contract and should be replaced with the verified `INVARIANT` comment shown below — the original phrasing is exactly the wording the Phase 15 audit removed from `top_n.cpp` (see [empirical evidence](#empirical-evidence) below).

### Empirical evidence

Three pieces of evidence corroborate that the contract holds for every currently-shipping operator:

- **Phase 14 ship-validation** — `[mgpu]` 12/13 PASS, `[TPC-H][parquet]` 22/22 PASS, `[integration][TPC-H]` 48/48 PASS (71608 assertions). The single `[mgpu]` fail is the Phase-12-territory `physical_order - small sort stays single-GPU` `vector::_M_range_check`, fixed on `fix/order-small-sort-rangecheck` and unrelated to operator colocation.
- **Phase 15 Wave 1 audit** — All 11 operator sites that read `valid_batches[0]->get_memory_space()` (or equivalent) are classified `SAFE` based on upstream-trace through `gpu_pipeline_task::execute -> pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch`.
- **Phase 15 Wave 2 stress test** — `test/cpp/operator/test_mgpu_stress.cpp` exercises five representative `[mgpu]` queries under 100 distinct `_no_pref_rr_counter` starting offsets (500 inner runs, 77053 assertions), each asserting CPU baseline match via `require_gpu_matches_cpu`. Catches hash-bucket-order-dependent bugs and any latent off-by-one that a counter-always-starts-at-0 test would mask.

### For new operator authors

When you write a new `sirius_physical_operator` subclass that calls `get_memory_space()` on any input batch your operator did not itself construct, add an `INVARIANT` comment immediately above the call naming the upstream enforcement path. The audited form (see `src/op/sirius_physical_top_n.cpp`) is:

```cpp
// INVARIANT: all input batches arrive on target_space via
// gpu_pipeline_task::execute -> pipelineable_operator_data::prepare_for_processing
// -> lock_or_prepare_batch.
// See docs/super-sirius/pipeline-execution.md "Per-task-device contract under SCHED-RR".
cucascade::memory::memory_space* space = input_batches[0].get_memory_space();
```

This makes the upstream-protection assumption explicit and reviewable. The comment is mandatory for any code touching `get_memory_space()` on a batch the operator did not itself construct. If your operator constructs an output batch (e.g. by calling `make_data_batch(table, mem_space, writer_stream)`), reads on *that* output are out of scope — the operator chose its own `mem_space` and is the authority for it.

If you cannot satisfy the contract — for example, your operator legitimately needs to consume input batches from multiple GPUs without going through `pipelineable_operator_data` — then you must explicitly call `lock_or_prepare_batch` per batch yourself, or use `cucascade::convert_gpu_to_gpu` to colocate before reading. Do not assume `batches[0]`'s space is authoritative.

## Pipeline Executor

**File:** `src/include/pipeline/task_scheduler.hpp`, `src/pipeline/task_scheduler.cpp`

The `task_scheduler` is the top-level GPU-pipeline orchestrator. It owns the shared pipeline-task
queue, one `gpu_pipeline_executor` per active GPU, and the management thread that matches queued
tasks to ready devices. Scan execution is reached through `task_creator`; there is no scan
sub-executor or scan-priority queue owned here.

### Key Methods

| Method | Purpose |
|--------|---------|
| `start()` | Starts every GPU executor, then launches the management thread |
| `stop()` | Interrupts/closes scheduler channels, joins the management thread, then stops GPU executors |
| `prepare_for_query(query)` | Drains executor leftovers, installs query/completion state, and resets per-query scheduler state |
| `start_query()` | Schedules `query.get_scan_operators().front()` through `task_creator` and returns the completion future |
| `terminate_query(exception)` | Reports error to completion handler |
| `drain_after_error()` | Multi-stage drain for clean shutdown |

### Management Event Loop

`management_eventloop()` is a pull-signal matcher on a dedicated thread. GPU executors publish
`device_ready` when a worker is available; `schedule()` publishes `task_available` after adding a
task. Ready devices remain recorded until a compatible task arrives:

```
while running:
    1. Wait for device_ready or task_available; drain the current event burst
    2. If the task queue is empty, ask the task creator to look ahead
       (schedule_lookahead(first ready device) — speculatively warms up a
       not-yet-activated scan; no-op unless strategy is `lookahead`)
    3. For each ready device, select a compatible queued task:
       a. exact preferred-device match
       b. unpreferred task (or one with a stale preference)
    4. Dispatch the selected task to that device's GPU executor
```

Tasks stay in the top-level queue until a ready device can accept them, preserving visibility to
the downgrade machinery. A live preferred device is binding because the task may reference
device-local data.

### Initial Scan Scheduling

`start_query()` schedules exactly the first operator in `query.get_scan_operators()`. Subsequent
work is exposed by task hints and completion-driven downstream scheduling — plus, under the
`lookahead` task-creator strategy, the empty-queue look-ahead above (see
[task-creator.md](task-creator.md)); there is no `schedule_next_scan_tasks()` or
`_priority_scans` walk.

### Dynamic-filter independence

The scheduler is filter-agnostic: it does not inspect hash joins or reorder queued work to advance
dynamic-filter publication. Immediate probes remain strictly ordered by synchronous build-CONCAT
publication in the join pipeline. A scan reached transitively through an intervening join has no
such edge and samples whatever complete filters are visible at its reader and post-decode
checkpoints.

Issue [#1124](https://github.com/sirius-db/sirius/issues/1124) measured the former build-subtree
preference. It provided no coverage benefit while costing wall time and run-to-run variance, so it
was removed. See
[Transitive scan targets and publication timing](dynamic-filters.md#transitive-scan-targets-and-publication-timing)
for the consumer semantics.

## GPU Pipeline Executor

**File:** `src/include/pipeline/gpu_pipeline_executor.hpp`, `src/pipeline/gpu_pipeline_executor.cpp`

One `gpu_pipeline_executor` exists per GPU device. It manages a thread pool for executing GPU pipeline tasks.

### Executor Class Hierarchy

`gpu_pipeline_executor` inherits from `itask_executor`, which provides shared infrastructure: thread pool, task queue, `_running` flag, and `start/stop/schedule/drain_and_wait` lifecycle methods. Subclasses implement `manager_loop()` (required) and optional hooks `get_per_thread_init`, `on_start`, `on_stop`.

Concurrency is managed via `exec::bounded_thread_pool`, which uses a two-phase `reserve() -> pool.dispatch(slot, fn)` model with RAII slot release.

### Components

| Component | Type | Purpose |
|-----------|------|---------|
| `_thread_pool` | `exec::bounded_thread_pool` | Worker threads (default: 4), each pinned to GPU device, with slot-based concurrency control |
| `_task_queue` | `interruptible_mpmc<itask>` | Thread-safe queue for incoming tasks |
| `_manager_thread` | `std::thread` | Runs `manager_loop()` |
| `_stream_pool` | `exclusive_stream_pool` | Pool of CUDA streams, one per worker |
| `_memory_space` | `memory_space*` | GPU memory for making reservations |
| `_task_request_publisher` | `publisher<task_request>` | Channel to signal pipeline executor |
| `_task_creator` | `task_creator*` | For scheduling downstream consumer tasks |
| `_completion_handler` | `completion_handler*` | For signaling query completion |

### Manager Loop

```
while running:
    1. thread_pool.reserve()              -- block until a worker slot is available (RAII)
    2. task_request_publisher.send()      -- tell pipeline executor we can accept work
    3. task_queue.pop()                   -- block until a task is available
    4. clamp request to get_max_memory()  -- bound the history-based estimate by the space limit
    5. memory_space.make_reservation()    -- reserve GPU memory for the task
    6. task.set_reservation(reservation)  -- attach reservation to task
    7. stream_pool.acquire_stream()       -- get a CUDA stream
    8. thread_pool.dispatch(slot, lambda): -- dispatch to worker (slot released on completion)
         a. task.execute(stream)
         b. On OOM: retry (see below)
         c. On success: check query completion
         d. Schedule downstream consumers via task_creator
         e. Or: completion_handler.mark_completed()
```

The reservation request size comes from the task's memory-history estimate (`peak_memory_estimate + bytes_to_materialize_input`). Before reserving, the manager loop clamps this request to the memory space's reservation limit (`memory_space::get_max_memory()`). The estimate can extrapolate far past GPU capacity — a small input that once drove a near-capacity peak yields a large `peak/estimated` ratio. An unclamped over-limit request would receive only a partial reservation from `make_reservation()`, while the predicate-based downgrade that follows requires reserving the **full** requested size, which the space can never grant — livelocking the task through the OOM-reschedule loop until the retry cap trips. Clamping to `get_max_memory()` loses no reservable memory (`make_reservation()` already caps there) and keeps both the reservation and the downgrade target achievable; per-batch overflow during execution is still handled by the OOM-reschedule + tiering path.

### Downstream Scheduling

After a task completes:

1. Retrieve `output_consumers` — first operators of parent pipelines
2. If query not complete: call `task_creator->schedule(consumer)` for each
3. If pipeline sink is `RESULT_COLLECTOR` and pipeline is finished: `completion_handler->mark_completed()`

The completion check happens **before** scheduling downstream tasks to prevent scheduling tasks that reference already-destroyed operators.

### Task Request Flow

GPU executors communicate with the pipeline executor via `exec::channel<task_request>`:

```
gpu_executor → task_request_publisher.send() → task_scheduler.management_eventloop()
             ← task_queue.push()              ← task_creator.schedule()
```

## Completion Handler

**File:** `src/include/pipeline/completion_handler.hpp`

Thread-safe signaling for query completion using promise/future:

| Method | Behavior |
|--------|----------|
| `mark_completed()` | Atomically sets promise value (first caller wins via CAS) |
| `report_error(exception)` | Atomically sets exception on promise (first caller wins) |
| `get_awaitable()` | Returns the future for blocking |
| `is_completed()` / `has_error()` | Atomic status checks |

All methods are idempotent — subsequent calls after the first are no-ops.

## Reschedule Handling (OOM and CUDA launch failures)

**File:** `src/include/pipeline/oom_reschedule_exception.hpp`

Retryable execution failures are modeled as a small exception hierarchy: `task_reschedule_exception` is the base, with two subclasses —

- `oom_reschedule_exception` — a GPU operator ran out of memory. Carries `intermediate_data` (partial results computed so far) and `_resume_operator_index` (which operator to resume from).
- `cuda_launch_reschedule_exception` — a transient CUDA kernel-launch failure. `gpu_pipeline_task` translates retryable `thrust::system_error` codes (`cudaErrorLaunchOutOfResources` / `cudaErrorInvalidValue`, seen from concurrent PDL launches in `cudf::hash_partition`) into this exception rather than failing the query.

The GPU executor catches the **base** `task_reschedule_exception` and:

1. Checks if the completion handler already has an error (skip if so)
2. Increments `retry_count` (max 100 retries, `MAX_RETRIES`)
3. Logs the retry attempt
4. Marks the original task as rescheduled (skips pipeline completion tracking)
5. Transitions intermediate data from idle to `task_created` state
6. Creates a new rescheduled task via `create_rescheduled_task()` virtual factory
7. Sleeps 50ms for backoff
8. Reschedules the new task back through the manager loop

If max retries are exceeded, the error propagates and terminates the query.

## Error Handling and Draining

**File:** `src/pipeline/task_scheduler.cpp`

`drain_after_error()` performs a multi-stage clean shutdown:

1. **Stop task creator thread pool** — prevents new tasks from being created
2. **Drain the task queue** — clears pending pipeline tasks
3. **Drain GPU executors** — `drain_and_wait()` per device: interrupts the queue, joins the manager thread, waits for all in-flight tasks
4. **Drain the task creator's pending tasks** — `drain_pending_tasks()` (also clears look-ahead state)
5. **Clear the task queue again** — catches tasks enqueued during the drain

This ensures that when `drain_after_error()` returns, no tasks are referencing operators or data repositories that are about to be destroyed.

## Key Files

| File | Purpose |
|------|---------|
| `src/include/pipeline/task_scheduler.hpp` | Top-level executor |
| `src/pipeline/task_scheduler.cpp` | Event loop, query lifecycle |
| `src/include/pipeline/gpu_pipeline_executor.hpp` | Per-GPU executor |
| `src/pipeline/gpu_pipeline_executor.cpp` | Manager loop, OOM handling |
| `src/include/pipeline/gpu_pipeline_task.hpp` | GPU task class |
| `src/pipeline/gpu_pipeline_task.cpp` | Task execution |
| `src/include/pipeline/completion_handler.hpp` | Promise/future completion |
| `src/include/pipeline/oom_reschedule_exception.hpp` | OOM retry mechanism |
| `src/include/pipeline/sirius_pipeline.hpp` | Pipeline structure |
| `src/include/pipeline/sirius_pipeline_itask.hpp` | Task interface |
| `src/include/pipeline/task_request.hpp` | Executor↔pipeline request |
| `src/include/exec/bounded_thread_pool.hpp` | Slot-based thread pool with RAII concurrency control |
| `src/include/parallel/task_executor.hpp` | `itask_executor` base class for all executors |
