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

**Phase 14 SCHED-RR change.** Phase 14 made preference-less dispatch distribute across all configured GPUs instead of piling onto whichever executor happened to sit in the first hash bucket. Source-pipeline tasks now genuinely land on multiple GPUs within a single query — exactly what an N-GPU configuration is supposed to deliver. (The original implementation used a round-robin counter; it has since been replaced by the pull-signal matcher described below.)

**The hazard this exposes.** Several operators read `valid_batches[0]->get_memory_space()` (or an equivalent expression on a single input batch) as the authoritative target memory space, then perform their concat/merge/sort directly on that space. Pre-Phase-14, this was *accidentally* safe — every batch in the input vector was already on the implicit "default GPU" because every upstream task was dispatched to that same default. Under SCHED-RR, that accident is gone. If an operator reads `batches[0]->get_memory_space()` without a guarantee that *all* batches in the input vector are colocated on that space, it can silently produce wrong results, mis-allocate, or skip data on the other GPU.

The fix is not to patch every read site to detect cross-GPU input. The fix is the upstream contract below: every operator's input batches are colocated by the task scheduler **before** the operator's `execute()` runs, so reading `batches[0]->get_memory_space()` is a SAFE alias for the task's reservation device.

### The contract

> **Every operator's input batches MUST arrive on the task's reservation device.** Operators MUST NOT use `batches[0]->get_memory_space()` as the authoritative target memory space; that read is acceptable only as an alias for `target_space` *after* `prepare_for_processing` has run upstream. New operators that read `get_memory_space()` from a batch they did not themselves construct MUST add an `INVARIANT (SCHED-RR contract)` comment naming the upstream enforcement path (see "For new operator authors" below).

This is a four-layer contract: the scheduler picks `target_space`, the task layer enforces it, the per-batch lock protocol implements it, and the operator layer relies on the postcondition. Each layer is shown below with the source line where it lives.

### How the contract is enforced

**Layer 1 — `gpu_pipeline_task::execute` captures `target_space` from the task's reservation.**

`src/pipeline/gpu_pipeline_task.cpp:310-315`:

```cpp
auto reservation         = local_state.release_reservation();
if (!reservation) { throw std::runtime_error("GPU pipeline task requires a memory reservation"); }
auto reservation_bytes = reservation->size();
const auto* requested_memory_space =
  reservation != nullptr ? &reservation->get_memory_space() : nullptr;
```

The reservation was attached by the GPU executor's manager loop (see [GPU Pipeline Executor](#gpu-pipeline-executor) above) on the SCHED-RR-chosen device. `requested_memory_space` is the authoritative target for every input batch this task will touch.

**Layer 2 — `gpu_pipeline_task::execute` calls `prepare_for_processing` on the operator-data input.**

`src/pipeline/gpu_pipeline_task.cpp:329-332`:

```cpp
std::optional<std::vector<cucascade::data_batch_processing_handle>> handles_opt;
try {
  handles_opt =
    local_state._input_data.get()->prepare_for_processing(requested_memory_space, stream);
```

This is the gate. `compute_task(stream)` (line 373) — which iterates the pipeline's operators and calls each one's `execute()` — does not run until `prepare_for_processing` has returned a non-empty `handles_opt`. Every batch in the input vector is colocated on `requested_memory_space` by the time any operator sees it.

**Layer 3 — `pipelineable_operator_data::prepare_for_processing` walks each batch and locks-or-converts it.**

`src/op/sirius_physical_operator.cpp:37-84`:

```cpp
std::optional<std::vector<::cucascade::data_batch_processing_handle>>
pipelineable_operator_data::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  std::vector<::cucascade::data_batch_processing_handle> handles;
  handles.reserve(_data_batches.size());

  for (const auto& batch : _data_batches) {
    ...
    handle = pipeline::lock_or_prepare_batch(batch, requested_memory_space, stream);
    ...
    handles.emplace_back(std::move(*handle));
  }

  return handles;
}
```

Every batch in `_data_batches` is fed through `lock_or_prepare_batch`. There is no early-exit short-circuit — partial colocation is not possible. Either every batch ends up on `requested_memory_space` or the function returns `std::nullopt` and the task is rescheduled (line 351-353 of `gpu_pipeline_task.cpp`).

**Layer 4 — `lock_or_prepare_batch` does the actual conversion.**

`src/include/pipeline/batch_lock_utils.hpp:48-126`:

```cpp
inline std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  ...
  while (!lock_result.success && lock_result.status == status::memory_space_mismatch) {
    ...
    case cucascade::memory::Tier::GPU: {
      ...
      batch->convert_to<cucascade::gpu_table_representation>(registry, target_space, stream);
      ...
    }
    ...
  }
  ...
  return std::move(lock_result.handle);
}
```

If the batch is already on `target_space`, it is locked in place. If it is on a different GPU, `batch->convert_to<gpu_table_representation>(...)` invokes the cucascade converter registry, which routes the GPU↔GPU path through `cucascade::convert_gpu_to_gpu` (peer-DMA on server hardware, automatic host-staging on consumer hardware whose chipset misreports peer-access support).

**Postcondition.** When `prepare_for_processing` returns successfully, every batch in `_input_data->_data_batches` lives on `requested_memory_space`. Therefore the per-operator expression `batches[0]->get_memory_space() == target_space` holds at every audited read site. Operators that walk every batch and adopt the first non-null batch's space (e.g. `sirius_physical_sort_sample.cpp:112`, `sirius_physical_merge_sort.cpp:92`, `sirius_physical_table_scan.cpp:129`) are safe by the same postcondition.

### The SCHED-RR distribution policy

The contract above is necessary because the scheduler distributes preference-less tasks across multiple GPUs.

**Pull-signal matching.** Distribution is demand-driven rather than counter-driven. A
`gpu_pipeline_executor`'s manager loop reserves a worker slot and then publishes a `device_ready`
signal; `task_scheduler::management_eventloop` matches each ready device against the task queue:

```cpp
// Exact preference match first: the highest-priority task preferring this device.
task = _task_queue.try_pop_from(exec::gpu_index{device_id}).value_or(nullptr);
if (!task) {
  // Otherwise any task with no preference.
  task = _task_queue.try_pop_from(exec::gpu_index{exec::no_preferred_device}).value_or(nullptr);
}
```

Preference-less tasks live in the `no_preferred_device` bucket of the
`multi_index_priority_queue`, so the GPU that serves one is simply whichever executor signalled
ready first. That is self-balancing — a GPU that finishes its work sooner asks for more sooner —
and it needs no shared counter, which is what makes it safe when several queries are in flight.

Tasks that *do* carry a `preferred_device_id` (`SCHED-01/02/04`, e.g. downstream pipeline tasks
consuming a specific repository) only ever match their own device, so locality is preserved.
Single-GPU configurations are unaffected: there is one executor, so every ready signal comes from
it.


### Migration note (Phase 14)

> **The pre-Phase-14 "default GPU is `_gpu_executors.begin()->first`" behavior is gone.** Any operator that hardcodes single-GPU assumptions, defaults to GPU 0, or uses `batches[0]->get_memory_space()` without going through the lock protocol upstream is now WRONG under SCHED-RR distribution. Phase 15 (cross-GPU operator-colocation audit) verified all 11 known sites; new operators MUST follow the same pattern.

If you are reading older operator code that says "all batches are expected to share the same space in practice" or similar unverified-assumption phrasing, that comment predates the contract and should be replaced with the verified `INVARIANT (SCHED-RR contract)` comment shown below — the original phrasing is exactly the wording the Phase 15 audit removed from `top_n.cpp` (see [empirical evidence](#empirical-evidence) below).

### Empirical evidence

Three pieces of evidence corroborate that the contract holds for every currently-shipping operator:

- **Phase 14 ship-validation** — `[mgpu]` 12/13 PASS, `[TPC-H][parquet]` 22/22 PASS, `[integration][TPC-H]` 48/48 PASS (71608 assertions). The single `[mgpu]` fail is the Phase-12-territory `physical_order - small sort stays single-GPU` `vector::_M_range_check`, fixed on `fix/order-small-sort-rangecheck` and unrelated to operator colocation.
- **Phase 15 Wave 1 audit** — All 11 operator sites that read `valid_batches[0]->get_memory_space()` (or equivalent) are classified `SAFE` based on upstream-trace through `gpu_pipeline_task::execute -> pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch`. The per-site classification table and justification were recorded in the Phase 15 audit log.
- **Phase 15 Wave 2 stress test** — `test/cpp/operator/test_mgpu_stress.cpp` exercises five representative `[mgpu]` queries over 100 iterations (500 inner runs, 77053 assertions), each asserting CPU baseline match via `require_gpu_matches_cpu`. PASS in 86.6s on `2 × RTX 6000 Ada`. Because the serving GPU is decided by which executor signals ready first, repeating each query this many times samples both assignments and catches an operator that is only correct on one of them.

### For new operator authors

When you write a new `sirius_physical_operator` subclass that calls `get_memory_space()` on any input batch your operator did not itself construct, add an `INVARIANT (SCHED-RR contract)` comment immediately above the call. The audited form (see `src/op/sirius_physical_concat.cpp:193`) is:

```cpp
// INVARIANT (SCHED-RR contract): all input batches arrive on target_space
// via gpu_pipeline_task::execute_pipeline_task_round ->
// pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch.
// See docs/super-sirius/pipeline-execution.md "Per-task-device contract under SCHED-RR".
cucascade::memory::memory_space* space = valid_batches[0]->get_memory_space();
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
| `start_query(query)` | Schedules `query.get_scan_operators().front()` through `task_creator`; completion is signalled through the query's own `completion_handler`, which its `sirius_engine` owns |
| `terminate_query(handler, error)` | Reports the error to that one query's completion handler; other in-flight queries keep running |
| `drain_query_tasks(query_id)` | Drops one query's queued tasks from the scheduler queue and every GPU executor's queue, leaving other queries' work in place |
| `wait_for_completion(query_id)` | Success path: quiesces the query in the lifecycle registry, validates its queues are empty (throws if not), waits out in-flight executor work |
| `drain_after_error(query_id)` | Error path: per-query multi-stage drain (see below) |
| `set_query_lifecycle_registry(registry)` | Binds the per-query enqueue gate, propagated to every GPU executor |

Per-query state is no longer installed here: `task_creator::prepare_for_query(query, handler)`
registers the query's task-creation state (see [Task Creator](task-creator.md)), and the
scheduler is stateless across queries apart from its queue. The per-query lifecycle is
documented in [Concurrency Model](concurrency-model.md).

### Management Event Loop

`management_eventloop()` is a pull-signal matcher on a dedicated thread. GPU executors publish
`device_ready` when a worker is available; `schedule()` publishes `task_available` after adding a
task. Ready devices remain recorded until a compatible task arrives:

```
while running:
    1. Wait for device_ready or task_available; drain the current event burst
    2. If the queue is empty and devices are idle: task_creator.schedule_lookahead()
       (warm up one not-yet-activated scan, rotating across live queries)
    3. For each ready device, select a compatible queued task:
       a. exact preferred-device match
       b. unpreferred task (or one with a stale preference)
    4. Dispatch the selected task to that device's GPU executor
```

Tasks stay in the top-level queue until a ready device can accept them, preserving visibility to
the downgrade machinery. A live preferred device is binding because the task may reference
device-local data.

The queue itself is an `exec::multi_index_priority_queue` whose dispatch pops round-robin across
query bands (register F1) — within one query the strict priority order holds, but across queries
the pop rotates over live query ids so an earlier-admitted query cannot starve a later one. See
[Concurrency Model](concurrency-model.md#scheduling) for the fairness contract.

### Initial Scan Scheduling

`start_query()` schedules exactly the first operator in `query.get_scan_operators()`. Subsequent
work is exposed by task hints and completion-driven downstream scheduling; there is no
`schedule_next_scan_tasks()` or `_priority_scans` walk. When the scheduler queue runs empty while
devices are idle, the event loop calls `task_creator::schedule_lookahead()`, which warms up one
not-yet-activated scan of a live query, rotating round-robin across the queries that still accept
work (see [Task Creator](task-creator.md#lookahead-scan-warm-up)).

### Dynamic-filter independence

The scheduler is filter-agnostic: it does not inspect hash joins or reorder queued work to advance
dynamic-filter publication. Immediate probes remain strictly ordered by synchronous build-CONCAT
publication in the join pipeline. A scan reached transitively through an intervening join has no
such edge and samples whatever complete filters are visible at its reader and post-decode
checkpoints.

Issue [#1124](https://github.com/sirius-db/sirius/issues/1124) measured the former build-subtree
preference at SF300. It provided no coverage benefit; disabling it cut wall time by 9–25% and
substantially reduced run-to-run variance, so it was removed. See
[Transitive scan targets and publication timing](dynamic-filters.md#transitive-scan-targets-and-publication-timing)
for the consumer semantics.

## GPU Pipeline Executor

**File:** `src/include/pipeline/gpu_pipeline_executor.hpp`, `src/pipeline/gpu_pipeline_executor.cpp`

One `gpu_pipeline_executor` exists per GPU device. It manages a thread pool for executing GPU pipeline tasks.

### Executor Class Hierarchy

`gpu_pipeline_executor` inherits from `itask_executor`, which provides shared infrastructure: thread pool, task queue, `_running` flag, and `start/stop/schedule/drain_and_wait` lifecycle methods, plus the per-query variants `drain_query_tasks(query_id)`, `wait_and_validate_empty(query_id)` (success path) and `wait_and_drain_query(query_id)` (error path). Subclasses implement `manager_loop()` (required) and optional hooks `get_per_thread_init`, `on_start`, `on_stop`. `itask_executor::schedule()` consults the query-lifecycle gate and silently refuses work for a quiescing query — the OOM reschedule path re-enters it from a worker thread long after a drain may have passed. A push bounced by a *transient* queue interruption (a peer query's error-path quiesce bracket) is retried via `push_or_bounce` rather than dropped; only a real shutdown drops it, loudly.

Concurrency is managed via `exec::bounded_thread_pool`, which uses a two-phase `reserve() -> pool.dispatch(slot, fn)` model with RAII slot release; slots are attributed to a query (`slot.attach(query_id)`) so per-query drains can wait on exactly that query's in-flight work.

### Components

| Component | Type | Purpose |
|-----------|------|---------|
| `_bounded_pool` | `exec::bounded_thread_pool` | Worker threads (default: 4), each pinned to GPU device, with slot-based concurrency control and per-query slot attribution (`slot.attach(query_id)`) |
| `_task_queue` | `exec::multi_index_priority_queue<itask>` | Priority queue for incoming tasks, indexed by query so one query's staged work can be dropped without touching another's |
| `_manager_thread` | `std::thread` | Runs `manager_loop()` |
| `_stream_pool` | `exclusive_stream_pool` | Pool of CUDA streams, one per worker |
| `_memory_space` | `memory_space*` | GPU memory for making reservations |
| `_task_request_publisher` | `publisher<task_request>` | Channel to signal pipeline executor |
| `_task_creator` | `task_creator*` | For scheduling downstream consumer tasks |
| `_memory_waiter_parked` | `std::atomic<bool>` | The executor's single memory-wait slot (register C4, see below) |

There is no per-executor completion handler: each task carries its own query's
`completion_handler` in its global state (stamped by `task_creator::prepare_for_query`), and the
executor resolves it per task (`gpu_pipeline_task::get_completion_handler()`).

### Manager Loop

The manager thread never blocks on memory (register C4): it only pops, attributes, and
dispatches. The reservation — and any downgrade round trip it triggers — runs inside the
dispatched worker (`prepare_and_execute`), where the pool's per-query slot accounting covers it.

```
manager_loop (manager thread):
    1. bounded_pool.reserve()             -- block until a worker slot is available (RAII)
    2. task_request_publisher.send()      -- publish device_ready to the task_scheduler
    3. task_queue.pop()                   -- block until a task is available (fair pop, F1)
    4. process_task():                    -- resolve the task's completion handler,
                                             slot.attach(query_id), dispatch to a worker

prepare_and_execute (pool worker):
    5. clamp request to get_max_memory()  -- bound the history-based estimate by the space limit
    6. memory_space.make_reservation()    -- reserve GPU memory (may block / trigger downgrade)
    7. task.set_reservation(reservation)  -- attach reservation to task
    8. stream_pool.acquire_stream()       -- get a CUDA stream
    9. task.execute(stream)
         a. On OOM: retry (see below)
         b. On success: check query completion
         c. Schedule downstream consumers via task_creator
         d. Or: completion_handler.mark_completed()
```

At most **one** task per executor may park in a blocking memory wait (`_memory_waiter_parked` —
the historical one-blocking-reservation-per-device arbitration). A task that needs to wait while
the slot is taken is re-queued through the executor's own queue after a 10 ms worker-held backoff
(`executor_metrics::tasks_requeued_on_memory_wait` counts these), so a memory-hungry query cannot
fill every worker slot with parked waits either. Re-queues never consume the OOM retry budget.

The reservation request size comes from the task's memory-history estimate (`peak_memory_estimate + bytes_to_materialize_input`). Before reserving, the worker clamps this request to the memory space's reservation limit (`memory_space::get_max_memory()`). The estimate can extrapolate far past GPU capacity — a small input that once drove a near-capacity peak yields a large `peak/estimated` ratio. An unclamped over-limit request would receive only a partial reservation from `make_reservation()`, while the predicate-based downgrade that follows requires reserving the **full** requested size, which the space can never grant — livelocking the task through the OOM-reschedule loop until the retry cap trips. Clamping to `get_max_memory()` loses no reservable memory (`make_reservation()` already caps there) and keeps both the reservation and the downgrade target achievable; per-batch overflow during execution is still handled by the OOM-reschedule + tiering path.

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

## OOM Handling

**File:** `src/include/pipeline/oom_reschedule_exception.hpp`

When a GPU operator runs out of memory during execution, it throws `oom_reschedule_exception` carrying:

- `intermediate_data` — partial results computed so far
- `_resume_operator_index` — which operator to resume from

The GPU executor catches this and:

1. Checks if the completion handler already has an error (skip if so)
2. Increments `retry_count` and checks it against the per-query retry cap: the pipeline's
   `reservation_max_retries()`, stamped onto `pipeline_build_context` from the admission-time
   config snapshot (`operator_params.gpu_reservation_max_retries`, default 100 —
   `exec::default_gpu_reservation_max_retries`; settable via YAML and
   `SET gpu_reservation_max_retries`)
3. Logs the retry attempt
4. Marks the original task as rescheduled (skips pipeline completion tracking)
5. Transitions intermediate data from idle to `task_created` state
6. Creates a new rescheduled task via `create_rescheduled_task()` virtual factory, carrying the
   per-task device pin forward (a rescheduled partition task must not scatter to another GPU)
7. Sleeps 50 ms for backoff (rides out cross-GPU batch-lock contention as well as true OOM)
8. Reschedules the new task back through the manager loop

If the cap is exceeded, the error is reported to that query's own completion handler and
terminates only that query. C4 memory-wait re-queues do not count against this budget.

## Error Handling and Draining

**File:** `src/pipeline/task_scheduler.cpp`

`drain_after_error(query_id)` performs a per-query multi-stage drain — other in-flight queries
keep producing and executing throughout:

1. **Quiesce the query** — `query_lifecycle_registry::quiesce(query_id)` makes every enqueue
   point refuse this query's work, so a completion callback cannot add work behind a drain that
   already passed (this replaced stopping the shared task creator, which halted every query)
2. **Drain the scheduler queue** — `_task_queue.drain(query_index{...})` drops this query's
   queued pipeline tasks only
3. **Drain GPU executors** — `wait_and_drain_query(query_id)` on each: waits out in-flight pool
   work, then drops this query's staged tasks
4. **Drain task creation** — `task_creator::drain_pending_tasks(query_id)` discards this query's
   pending creation requests (they hold raw operator pointers into the plan about to die)
5. **Final sweep** — one more `_task_queue.drain(query_index{...})`: a task completing during
   the drains can have handed one more task to the queue before the gate refused its successor

This ensures that when `drain_after_error()` returns, no tasks of this query reference operators
or data repositories that are about to be destroyed. The plan itself is destroyed even later, by
`SiriusContext::run_mandatory_cleanup` — see
[Concurrency Model](concurrency-model.md#query-end-the-mandatory-cleanup).

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
| `src/include/exec/bounded_thread_pool.hpp` | Slot-based thread pool with RAII concurrency control and per-query slot attribution |
| `src/include/exec/multi_index_priority_queue.hpp` | Priority queue with query/operator/device indexes and fair (round-robin-across-queries) dispatch pops |
| `src/include/exec/query_lifecycle_registry.hpp` | Per-query enqueue gate (open → quiescing → closed) |
| `src/include/parallel/task_executor.hpp` | `itask_executor` base class for all executors |
