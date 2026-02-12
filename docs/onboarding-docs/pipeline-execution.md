# Pipeline Execution

This document explains how Sirius executes queries on the GPU through its pipeline execution framework. It covers physical operators, pipeline construction, task creation, and the GPU executor.

## Overview

Sirius translates a DuckDB physical plan into a graph of **pipelines**. Each pipeline is a linear chain of operators:

```
source --> [operator_0] --> [operator_1] --> ... --> sink
```

- The **source** produces data batches (e.g. a table scan).
- **Intermediate operators** transform data batches in sequence (e.g. filter, projection).
- The **sink** consumes the final output (e.g. builds a hash table, collects results).

Pipelines are connected to each other through **ports** on operators. When a sink finishes processing a batch, it pushes the output into the input ports of the next pipeline's first operator. A **task creator** monitors these ports and creates `gpu_pipeline_task` objects when data is available. These tasks are scheduled on the `gpu_pipeline_executor`, which manages GPU memory, CUDA streams, and a thread pool to execute them.

```
Pipeline A                          Pipeline B
[scan] -> [filter] -> [hash_build]  [hash_probe] -> [projection] -> [result_collector]
                           |                |
                           +--- port -------+
                         (data_repository)
```

## Physical Operators

**File:** `src/include/op/sirius_physical_operator.hpp`, `src/op/sirius_physical_operator.cpp`

`sirius_physical_operator` is the base class for every operator in a Sirius execution plan. An operator plays one or more of three roles depending on where it sits in a pipeline:

### Roles

| Role | Key methods | Example operators |
|------|-------------|-------------------|
| **Source** | `is_source()`, `get_global_source_state()` | `DUCKDB_SCAN`, `PARQUET_SCAN`, `COLUMN_DATA_SCAN` |
| **Operator** | `execute(input_batches, stream)` | `FILTER`, `PROJECTION`, `HASH_GROUP_BY` |
| **Sink** | `is_sink()`, `sink(input_batches, stream)` | `ORDER_BY`, `HASH_JOIN` (build side), `RESULT_COLLECTOR` |

An operator knows its role through the virtual `is_source()` and `is_sink()` methods. Intermediate operators override `execute()` — they receive a vector of `data_batch` shared pointers and return a new vector of transformed batches.

### Operator Tree and Children

Operators form a tree via the `children` vector. For most operators, there is exactly one child. Sinks (like joins) may have one child in their main pipeline and create separate child pipelines for other inputs.

### Ports

Ports are the mechanism for passing data **between pipelines**. Each port is a named input buffer on an operator:

```cpp
struct port {
    MemoryBarrierType type;              // PIPELINE, PARTIAL, or FULL
    cucascade::shared_data_repository* repo;  // holds queued data_batch objects
    shared_ptr<sirius_pipeline> src_pipeline;  // the pipeline producing data for this port
    shared_ptr<sirius_pipeline> dest_pipeline; // the pipeline consuming from this port
};
```

- **`PIPELINE` barrier** (streaming): The downstream operator can consume batches as soon as they arrive. No need to wait for the upstream pipeline to finish.
- **`FULL` barrier**: The downstream operator must wait until the upstream pipeline is completely finished before consuming. Used for operators like hash join build side, where the entire hash table must be built before probing begins.

When a sink's `sink()` method is called with output batches, the default implementation pushes each batch into the ports of the operators listed in `next_port_after_sink`:

```cpp
void sirius_physical_operator::sink(const vector<shared_ptr<data_batch>>& output_batches, stream) {
    for (auto& batch : output_batches) {
        for (auto& [next_op, port_id] : next_port_after_sink) {
            next_op->push_data_batch(port_id, batch);
        }
    }
}
```

### Task Creation Hints

The `get_next_task_hint()` method on an operator examines its ports to determine readiness:

1. If any port with a `FULL` barrier has an unfinished source pipeline, return `WAITING_FOR_INPUT_DATA` with a pointer to the producing operator.
2. If all ports have data available (and FULL barriers have finished source pipelines), return `READY`.
3. If any non-FULL port has an unfinished source pipeline but no data yet, return `WAITING_FOR_INPUT_DATA`.
4. Otherwise return `nullopt` — nothing to do.

The task creator uses these hints to decide when to create new `gpu_pipeline_task` instances.

### Pipeline Construction (`build_pipelines`)

Each operator implements `build_pipelines()`, which is called recursively to construct the pipeline graph. The default logic:

- **If the operator is a sink**: It becomes the **source** of the current pipeline (the sink reads from its own built state). A child meta-pipeline is created, and the operator's child subtree is built into that child pipeline.
- **If the operator is a leaf** (no children): It becomes the **source** of the current pipeline.
- **Otherwise**: The operator is added as an intermediate operator to the current pipeline, and recursion continues into its child.

Note: During construction, operators are added in bottom-up order. The `is_ready()` call reverses them to get the correct top-down execution order.

## Pipelines

**File:** `src/include/pipeline/sirius_pipeline.hpp`, `src/pipeline/sirius_pipeline.cpp`

### Structure

A `sirius_pipeline` holds:

| Field | Type | Purpose |
|-------|------|---------|
| `source` | `optional_ptr<sirius_physical_operator>` | The operator that produces data |
| `operators` | `vector<reference<sirius_physical_operator>>` | Intermediate operators, in execution order |
| `sink` | `optional_ptr<sirius_physical_operator>` | The operator that consumes the final output |
| `dependencies` | `vector<shared_ptr<sirius_pipeline>>` | Pipelines that must finish before this one starts |
| `parents` | `vector<weak_ptr<sirius_pipeline>>` | Pipelines that depend on this one finishing |

### Pipeline Build State

`sirius_pipeline_build_state` is a friend class that provides controlled write access to a pipeline's internals during construction. It exposes methods like `set_pipeline_source()`, `set_pipeline_sink()`, `add_pipeline_operator()`, and `create_child_pipeline()`. This keeps the pipeline's fields private while allowing the meta-pipeline builder to assemble the pipeline graph.

### Completion Tracking

Each pipeline tracks how many tasks have been created and completed:

```cpp
std::atomic<size_t> tasks_created = 0;
std::atomic<size_t> tasks_completed = 0;
```

- `mark_task_created()` is called when a task is created for this pipeline.
- `mark_task_completed()` is called from the `gpu_pipeline_task` destructor when a task finishes.

`update_pipeline_status()` checks whether the pipeline is finished:

- For `DUCKDB_SCAN` sources: finished when the scan is exhausted.
- For other sources: finished when the upstream pipeline(s) are done, all input ports are empty, and `tasks_created == tasks_completed`.

### Batch Index Tracking

Pipelines maintain a `multiset<idx_t>` of active batch indexes for order-preserving execution. `register_new_batch_index()` and `update_batch_index()` manage this set under a mutex to track the minimum in-flight batch index.

## Tasks

### Class Hierarchy

```
parallel::itask                          // base: local_state + global_state + execute(stream)
  └── sirius_pipeline_itask              // adds compute_task() / publish_output() split
        └── gpu_pipeline_task            // concrete: executes a pipeline on GPU
```

**File:** `src/include/parallel/task.hpp`

`itask` is the base task interface. It holds:
- `_local_state` (`unique_ptr<itask_local_state>`) — per-task instance data
- `_global_state` (`shared_ptr<itask_global_state>`) — shared across tasks in the same context

### `sirius_pipeline_itask`

**File:** `src/include/pipeline/sirius_pipeline_itask.hpp`

Extends `itask` with a two-phase execution model:

1. **`compute_task(stream)`** — perform the computation, return output batches.
2. **`publish_output(batches, stream)`** — push results to their destination.

The default `execute()` simply calls `compute_task()` then `publish_output()`.

It also defines `get_estimated_reservation_size()` for memory budgeting and `get_output_consumers()` to identify downstream operators that should be scheduled next.

### `sirius_pipeline_itask_local_state`

**File:** `src/include/pipeline/sirius_pipeline_itask_local_state.hpp`

Extends `itask_local_state` with memory reservation management. The executor sets a `cucascade::memory::reservation` on the task before execution via `set_reservation()`, and the task can later `release_reservation()` to use the reserved GPU memory.

### `gpu_pipeline_task`

**File:** `src/include/pipeline/gpu_pipeline_task.hpp`, `src/pipeline/gpu_pipeline_task.cpp`

This is the concrete task that executes a pipeline on the GPU.

**State classes:**
- `gpu_pipeline_task_global_state` — holds a `shared_ptr<sirius_pipeline>` (the pipeline to execute).
- `gpu_pipeline_task_local_state` — holds the input `data_batch` vector and inherits the memory reservation from `sirius_pipeline_itask_local_state`.

**`compute_task(stream)`** iterates through the pipeline's intermediate operators in order, threading output batches forward:

```cpp
auto data_batches = local_state._batches;
for (auto& op : pipeline->get_operators()) {
    data_batches = op.get().execute(data_batches, stream);
}
return data_batches;
```

**`publish_output(batches, stream)`** passes the final batches to the pipeline's sink:

```cpp
pipeline->get_sink()->sink(output_batches, stream);
```

**`execute(stream)`** (the full override) handles the additional concern of memory space management:
1. For each input batch, calls `lock_or_prepare_batch()` which locks the batch for processing and converts it to the target memory space (CPU → GPU) if needed.
2. Calls `compute_task()` + `publish_output()`.
3. Processing handles are released automatically when they go out of scope.

**Destructor** calls `pipeline->mark_task_completed()`, which updates the pipeline's completion tracking.

**`get_output_consumers()`** returns the first operator of each parent pipeline — these are the downstream operators that the executor will ask the task creator to schedule next.

## GPU Pipeline Executor

**File:** `src/include/pipeline/gpu_pipeline_executor.hpp`, `src/pipeline/gpu_pipeline_executor.cpp`

The `gpu_pipeline_executor` is responsible for executing `gpu_pipeline_task` instances on GPU worker threads. It manages GPU resources (memory reservations, CUDA streams) and coordinates with the task creator for downstream scheduling.

### Components

| Component | Type | Purpose |
|-----------|------|---------|
| `_thread_pool` | `exec::thread_pool` | Pool of worker threads, each pinned to the correct CUDA device |
| `_task_queue` | `interruptible_mpmc<unique_ptr<itask>>` | Thread-safe queue for incoming tasks |
| `_manager_thread` | `std::thread` | Runs the `manager_loop()` |
| `_kiosk` | `exec::kiosk` | Ticket-based semaphore limiting concurrency to `num_threads` |
| `_stream_pool` | `exclusive_stream_pool` | Pool of CUDA streams, one per worker |
| `_memory_space` | `memory_space*` | GPU memory space for making reservations |
| `_task_request_publisher` | `publisher<task_request>` | Channel to signal the task creator that the executor is ready for work |
| `_task_creator` | `task_creator*` | Used to schedule downstream consumer tasks after a task completes |
| `_completion_handler` | `completion_handler*` | Signaled when the entire query is finished |

### Manager Loop

The `manager_loop()` runs on its own thread and orchestrates task execution:

```
while running:
    1. kiosk.acquire()          -- block until a worker thread is free
    2. task_request_publisher.send()  -- tell task_creator we can accept work
    3. task_queue.pop()         -- block until a task is available
    4. memory_space.make_reservation(task.estimated_size)  -- reserve GPU memory
    5. task.local_state.set_reservation(reservation)       -- attach reservation to task
    6. stream_pool.acquire_stream()   -- get a CUDA stream
    7. thread_pool.schedule(lambda):  -- dispatch to worker thread
         a. task.execute(stream)      -- run the pipeline
         b. check if query is complete (RESULT_COLLECTOR sink + pipeline finished)
         c. if not complete: schedule downstream consumers via task_creator
         d. if complete: completion_handler.mark_completed()
```

### Lifecycle

- **`start()`**: Creates the thread pool (with `cudaSetDevice` initializers) and launches the manager thread.
- **`stop()`**: Stops the kiosk, interrupts the task queue, joins the manager thread, waits for all workers, and stops the thread pool.
- **`schedule(task)`**: Pushes a task onto the MPMC queue.
- **`drain_leftover_tasks()`**: Clears any remaining tasks from a previous query.

### Downstream Scheduling

After a task completes successfully, the executor checks whether downstream work should be created:

1. It retrieves the task's `output_consumers` — these are the first operators of parent pipelines.
2. If the query is not yet complete, it calls `task_creator->schedule(consumer)` for each consumer, which will check the consumer's ports and potentially create new tasks.
3. If the pipeline's sink is a `RESULT_COLLECTOR` and the pipeline is finished, the query is complete and `completion_handler->mark_completed()` is called instead.

The check for query completion happens **before** scheduling downstream tasks. This prevents scheduling tasks that reference operators which may be destroyed once completion is signaled.
