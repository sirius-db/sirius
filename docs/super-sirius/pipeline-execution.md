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

## Pipeline Executor

**File:** `src/include/pipeline/task_scheduler.hpp`, `src/pipeline/task_scheduler.cpp`

The `task_scheduler` is the top-level orchestrator that owns GPU and scan sub-executors.

### Key Methods

| Method | Purpose |
|--------|---------|
| `start()` | Initializes scan executor, GPU executors, launches management thread |
| `stop()` | Stops all sub-executors, joins threads |
| `prepare_for_query(query)` | Drains leftover tasks, prepares scan cache, populates priority scan queue |
| `start_query()` | Creates completion handler, distributes to executors, schedules initial scans, returns future |
| `terminate_query(exception)` | Reports error to completion handler |
| `drain_after_error()` | Multi-stage drain for clean shutdown |

### Management Event Loop

`management_eventloop()` runs on a dedicated thread:

```
while running:
    1. task_request_channel.get()  -- block for GPU executor request
    2. task_queue.pop()            -- dequeue a pipeline task
    3. Route to GPU executor by device_id
```

The event loop bridges task creation (which pushes to `_task_queue`) with GPU executors (which pull via task requests).

### Initial Scan Scheduling

`schedule_next_scan_tasks()` pops scan operators from `_priority_scans` and calls `task_creator->schedule(scan_op)` for each. This kicks off the first wave of scan tasks.

## GPU Pipeline Executor

**File:** `src/include/pipeline/gpu_pipeline_executor.hpp`, `src/pipeline/gpu_pipeline_executor.cpp`

One `gpu_pipeline_executor` exists per GPU device. It manages a thread pool for executing GPU pipeline tasks.

### Executor Class Hierarchy

All executors (`gpu_pipeline_executor`, `downgrade_executor`, `duckdb_scan_executor`) inherit from `itask_executor`, which provides shared infrastructure: thread pool, task queue, `_running` flag, and `start/stop/schedule/drain_and_wait` lifecycle methods. Subclasses implement `manager_loop()` (required) and optional hooks `get_per_thread_init`, `on_start`, `on_stop`.

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
    4. memory_space.make_reservation()    -- reserve GPU memory for the task
    5. task.set_reservation(reservation)  -- attach reservation to task
    6. stream_pool.acquire_stream()       -- get a CUDA stream
    7. thread_pool.dispatch(slot, lambda): -- dispatch to worker (slot released on completion)
         a. task.execute(stream)
         b. On OOM: retry (see below)
         c. On success: check query completion
         d. Schedule downstream consumers via task_creator
         e. Or: completion_handler.mark_completed()
```

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
2. Increments `retry_count` (max 10 retries, `MAX_OOM_RETRIES`)
3. Logs the retry attempt
4. Marks the original task as rescheduled (skips pipeline completion tracking)
5. Transitions intermediate data from idle to `task_created` state
6. Creates a new rescheduled task via `create_rescheduled_task()` virtual factory
7. Sleeps 5ms for backoff
8. Reschedules the new task back through the manager loop

If max retries are exceeded, the error propagates and terminates the query.

## Error Handling and Draining

**File:** `src/pipeline/task_scheduler.cpp`

`drain_after_error()` performs a multi-stage clean shutdown:

1. **Stop task creator threads** — prevents new tasks from being created
2. **Drain task queue** — clears pending pipeline tasks
3. **Drain GPU executors** — `drain_and_wait()` stops kiosk, interrupts queue, joins manager, waits for all in-flight tasks
4. **Drain scan executor** — same pattern
5. **Restart task creator** — prepares for the next query

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
