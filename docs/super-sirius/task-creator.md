# Task Creator

This document covers the task creation subsystem: how the system decides when and what tasks to create based on operator readiness and data availability.

## Overview

**File:** `src/include/creator/task_creator.hpp`, `src/creator/task_creator.cpp`

The `task_creator` is a multi-threaded component that converts operator scheduling requests into concrete scan or GPU pipeline tasks. It maintains global state maps for each operator type and uses a hint-chain recursion to find the deepest ready operator.

## Core Flow

```
schedule(operator*)
    ↓
_task_creation_queue.push(request)
    ↓
manager_loop() picks up request
    ↓
get_operator_for_next_task(operator) — follows hint chain
    ↓
operator->get_next_task_hint() → READY or WAITING_FOR_INPUT_DATA
    ↓
Create task (gpu_pipeline_task)
    ↓
Dispatch to executor (task_scheduler)
```

## Global State Maps

Initialized during `prepare_for_query()`, cleared during `reset()`:

| Map | Key | Value | Purpose |
|-----|-----|-------|---------|
| `_gpu_operator_global_state_map` | operator ID | `gpu_pipeline_task_global_state` | Shared per-operator pipeline state |

All map access is protected by `_global_state_mutex`.

## `TaskCreationHint` Enum

**File:** `src/include/op/sirius_physical_operator.hpp`

```cpp
enum class TaskCreationHint { WAITING_FOR_INPUT_DATA, READY };

struct task_creation_hint {
    TaskCreationHint hint{TaskCreationHint::WAITING_FOR_INPUT_DATA};
    sirius_physical_operator* producer{nullptr};
};
```

- `READY` — operator has sufficient input data, create a task now
- `WAITING_FOR_INPUT_DATA` — follow `producer` pointer to find upstream operator

## `get_operator_for_next_task()` — Recursive Hint Chain

**File:** `src/creator/task_creator.cpp`

```
function get_operator_for_next_task(node):
    hint = node->get_next_task_hint()
    if hint is READY:
        return hint.producer  // create task from this operator
    if hint is WAITING_FOR_INPUT_DATA:
        producer = hint.producer
        return get_operator_for_next_task(producer)  // recurse upstream
    if no hint:
        return nullptr  // nothing to do
```

This recursion ensures data flows from the deepest producers first, respecting pipeline dependencies.

## Base Class `get_next_task_hint()`

**File:** `src/op/sirius_physical_operator.cpp`

Default implementation checks all input ports in order:

1. If any `FULL` barrier port's source pipeline is not finished → return `WAITING_FOR_INPUT_DATA` pointing to that pipeline's source
2. If all ports have data available (and FULL barriers have finished source pipelines) → return `READY`
3. If any `PARTIAL` barrier port's source pipeline is not finished → return `WAITING_FOR_INPUT_DATA`
4. Otherwise → return `nullopt` (nothing to do)

## Base Class `get_next_task_input_data()`

**File:** `src/op/sirius_physical_operator.cpp`

Default implementation pops one batch from each input port:

```
for each port (in order):
    batch = port.repo->pop_data_batch(state=task_created)
    batches.push_back(batch)
return operator_data(batches)
```

Returns `nullptr` if no batches are available.

## Per-Operator Overrides

The core of the task creator's behavior comes from operator-specific overrides:

### HASH_JOIN (BUILD_PROBE mode)

| Method | Behavior |
|--------|----------|
| `get_next_task_hint()` | Tracks build state machine: `NOT_BUILT` → `SCHEDULING` → `SCHEDULED` → `BUILT`. Returns READY when both build and probe data available (NOT_BUILT) or probe data available (BUILT). |
| `get_next_task_input_data()` | `get_next_task_input_data_for_build_probe()`: On SCHEDULING → pop one build + one probe batch. On BUILT → pop one probe batch. |
| Why custom | Build/probe asymmetry: first task needs both sides, subsequent tasks only need probe |

State machine transitions:
```
NOT_BUILT: build_size>0 AND probe_size>0 → SCHEDULING (return READY)
SCHEDULING/SCHEDULED: → WAITING_FOR_INPUT_DATA (probe source)
BUILT: probe_size>0 → READY
```

### HASH_JOIN (STANDARD mode)

Uses base class for both `get_next_task_hint()` and `get_next_task_input_data()`. Input data walks the partition × left × right grid using snapshot batch IDs for Cartesian product iteration.

### PARTITION

| Method | Behavior |
|--------|----------|
| `get_next_task_hint()` | If the non-driving side has no `_num_partitions`, delegates to the sizing sibling's hint. Otherwise: base class. |
| `get_next_task_input_data()` | Mutex-locked with sibling to atomically determine partition count from the designated sizing side on first call. Notifies the hash join when build-side sizing can select an execution mode. Then delegates to base class. |
| Why custom | Sibling pair coordination: build normally sizes both sides; RIGHT-family joins other than `RIGHT_DELIM_JOIN` size both from the retained probe side. |

Deadlock prevention: both this and sibling partition locks are acquired in a fixed order using `std::scoped_lock`.

### CONCAT

| Method | Behavior |
|--------|----------|
| `get_next_task_hint()` | If source finished: READY if data exists. If `_concat_all`: WAITING. Otherwise: checks if any partition's accumulated bytes ≥ `_concat_batch_bytes`. |
| `get_next_task_input_data()` | For each partition: accumulates batches until byte threshold. Returns `partitioned_operator_data` with partition index. |
| Why custom | Byte-threshold batching; `_concat_all` mode for LEFT/ANTI/OUTER joins requires all data before output |

### SORT_SAMPLE

| Method | Behavior |
|--------|----------|
| `get_next_task_hint()` | If boundaries computed: base class. Otherwise: waits for N sample batches OR source finished. |
| `get_next_task_input_data()` | Base class (after boundary computation) |
| Why custom | Two-phase: cannot compute boundaries until N samples are collected |

### MERGE_SORT

| Method | Behavior |
|--------|----------|
| `get_next_task_hint()` | Base class (no override) |
| `get_next_task_input_data()` | Mutex-locked: drains ALL batches from one partition, advances `_current_partition_index`. |
| Why custom | One task per partition — must drain entire partition for multi-way merge |

### GROUPED_AGGREGATE_MERGE

Same pattern as MERGE_SORT: drains all batches from one partition per call, advancing partition index.

### TOP_N_MERGE

Same pattern as MERGE_SORT: drains all batches from one partition per call.

### DELIM_JOIN

| Method | Behavior |
|--------|----------|
| `get_next_task_input_data()` | Delegates to internal `partition_join` operator |
| Why custom | Wrapper pattern — the actual task creation is handled by the embedded join |

## Override Summary Table

| Operator | `get_next_task_hint()` | `get_next_task_input_data()` | Why Custom |
|----------|------------------------|------------------------------|------------|
| HASH_JOIN (BUILD_PROBE) | Build state machine | Build+probe or probe only | Build/probe asymmetry |
| HASH_JOIN (STANDARD) | Base class | Cartesian product walk | Multi-partition iteration |
| PARTITION | Delegates to sizing sibling | Mutex-locked count determination | Sibling coordination |
| CONCAT | Byte-threshold check | Accumulate until threshold | Batching + blocking mode |
| SORT_SAMPLE | Wait for N batches | Base class | Two-phase sampling |
| MERGE_SORT | Base class | Drain all from one partition | Per-partition merge |
| GROUPED_AGGREGATE_MERGE | Base class | Drain all from one partition | Per-partition merge |
| TOP_N_MERGE | Base class | Drain all from one partition | Per-partition merge |
| DELIM_JOIN | N/A | Delegates to partition_join | Wrapper pattern |

## Task Creator Manager Loop

**File:** `src/creator/task_creator.cpp`

### Scan Scheduling Strategy

At query startup, at most 2 scans are scheduled initially. In the manager loop, scan exhaustion (continuous creation to deplete the source) only runs when `_num_scans_in_plan == 1` — to maximize I/O parallelism for single-table scans. For plans with 2+ scans, the `get_next_task_hint()` topology-driven mechanism controls task creation, avoiding excessive memory consumption from eagerly scanning all tables.

```
while running:
    1. thread_pool.reserve()              -- wait for thread availability (bounded_thread_pool slot)
    2. _task_creation_queue.pop()         -- get next scheduling request
    3. node = get_operator_for_next_task(request.node)  -- follow hint chain
    4. if node is nullptr: continue

    5. Schedule work on the thread pool. Every source — a GPU_SCAN scan or any
       GPU operator with buffered input — drives the same loop:
           - Loop while (!node.all_ports_empty()):
             - pipeline.mark_task_created()  // BEFORE popping data
             - data = node.get_next_task_input_data()  // a GPU_SCAN blocks on its split_connector
             - If data: create a gpu_pipeline_task, dispatch to task_scheduler
             - If no data: pipeline.mark_task_completed()
```

The `mark_task_created()` call before data popping prevents a race condition where the pipeline could appear finished between data check and task creation.

## Device Assignment for GPU Tasks

**File:** `src/creator/task_creator.cpp`

When the manager loop builds a `gpu_pipeline_task`, it also chooses the task's `preferred_device_id` (which GPU executor the scheduler should route it to). The choice is resolved in priority order: an upstream scan split's stamped device, then a **partition device pin**, then data-locality by input bytes, then NUMA-affinity. The full locality math lives in [`multi-gpu-architecture.md`](multi-gpu-architecture.md); the partition pin is described here because it is owned by the task creator and is a correctness requirement.

### Partition device pin

Partitioned operators — BUILD_PROBE hash join, `grouped_aggregate_merge`, and the other partition-keyed operators above — build a per-partition cuco hash table that is valid only on the GPU it was built on. Touching it from a stream bound to another device trips `cudaErrorInvalidValue`. So when a task's input is a `partitioned_operator_data`, the task creator pins it to a fixed device:

```
preferred_device_id = _active_gpu_ids[ partition_idx % _active_gpu_ids.size() ]
```

This keeps every task of a given partition (build + all probes) on one GPU while spreading partitions across GPUs.

`_active_gpu_ids` is built once in the `task_creator` constructor from the memory manager's `Tier::GPU` memory spaces — i.e. the device ids that actually have a GPU executor, which is the same set `task_scheduler` keys executors on. The pin indexes this active set rather than the physical hardware topology: when the configured GPU count is smaller than the physical count, a physical-topology modulo could name a device with no executor, which the scheduler treats as "no preference" and round-robins — scattering a partition across GPUs.

The pin is reapplied on the OOM-reschedule path (`gpu_pipeline_executor`): when a task is rebuilt with a fresh local state, its per-task `preferred_device_id` is carried forward so a rescheduled probe doesn't lose its pin and scatter.

## `can_create_more_tasks()` and `has_processed_all_tasks()`

These methods signal task exhaustion:
- `can_create_more_tasks()` — returns false when no more tasks can be created (e.g., scan exhausted, all partitions processed)
- `has_processed_all_tasks()` — returns false when tasks are still in flight

Both throw `not implemented` in the base class and must be overridden by operators that need them.

## `drain_pending_tasks()`

**File:** `src/creator/task_creator.cpp`

Called during `drain_after_error()` to cleanly shut down:
1. `_task_creation_queue.drain()` — clears pending requests
2. `_kiosk.wait_all()` — waits for in-flight task creation lambdas to complete

## Key Files

| File | Purpose |
|------|---------|
| `src/include/creator/task_creator.hpp` | Task creator interface |
| `src/creator/task_creator.cpp` | Manager loop, hint chain, task dispatch |
| `src/include/op/sirius_physical_operator.hpp` | Base `get_next_task_hint()`, `get_next_task_input_data()` |
| `src/op/sirius_physical_operator.cpp` | Base implementations |
| `src/op/sirius_physical_hash_join.cpp` | BUILD_PROBE hint/data overrides |
| `src/op/sirius_physical_partition.cpp` | Sibling sync, adaptive count |
| `src/op/sirius_physical_concat.cpp` | Byte-threshold batching |
| `src/op/sirius_physical_sort_sample.cpp` | N-batch sampling |
| `src/op/sirius_physical_merge_sort.cpp` | Per-partition drain |
