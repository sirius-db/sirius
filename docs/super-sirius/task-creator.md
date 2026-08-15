# Task Creator

This document covers the task creation subsystem: how the system decides when and what tasks to create based on operator readiness and data availability.

## Overview

**File:** `src/include/creator/task_creator.hpp`, `src/creator/task_creator.cpp`

The `task_creator` is a multi-threaded component, shared by every in-flight query, that converts operator scheduling requests into concrete GPU pipeline tasks. It holds an entry of per-query state per in-flight query and uses a hint-chain recursion to find the deepest ready operator.

## Core Flow

```
schedule(operator*)
    ↓  (refused if the query is quiescing — query_lifecycle_registry gate)
_task_creation_queue.push(request)     — a multi_index_priority_queue: ordered like the
    ↓                                    execution queue, indexed by query id
manager_loop() picks up request (fair pop across query bands)
    ↓  resolves the query's state shared_ptr, attaches the pool slot to the query
get_operator_for_next_task(operator) — follows hint chain
    ↓
operator->get_next_task_hint() → READY or WAITING_FOR_INPUT_DATA
    ↓
Create task (gpu_pipeline_task)
    ↓
Dispatch to executor (task_scheduler)
```

Each `task_creation_request` carries its `query_id`, its pipeline's `queue_priority`, and its
operator type — all resolved at `schedule()` time so the queue's key extractor never dereferences
an operator that a racing teardown may have freed.

## Per-Query State

`prepare_for_query(query, handler)` registers one `query_task_global_state` entry for the query
(it does not touch other queries' entries); `reset(query_id)` drops exactly that entry. The
entries live in `_query_task_global_states` (a map keyed by query id, guarded by
`_global_state_mutex`) and are handed out as `shared_ptr`, so a worker's resolved copy stays
alive even if the query is reset mid-flight. Each entry holds:

| Member | Purpose |
|--------|---------|
| `global_states` | Source operator id → that pipeline's `gpu_pipeline_task_global_state`. Written once by `prepare_for_query`, read-only afterwards. Operator ids restart at 0 per query, so the map is only unique *within* an entry |
| `client_context` | The connection running this query (bound by `set_client_context` at window begin) |
| `completion_handler` | This query's completion signal, for the creation-failure path |
| `lookahead_queue` / `lookahead_mutex` | Not-yet-activated scans for the lookahead rotation (see below) |
| `active_gpu_ids` / `full_gpu_count` | The GPU subset THIS query was admitted onto (see Device Assignment) |

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

At query startup, `task_scheduler::start_query()` schedules exactly the query's **first** scan
operator. Further scans are activated by the `get_next_task_hint()` topology-driven mechanism
(avoiding excessive memory consumption from eagerly scanning all tables) and by the lookahead
warm-up below, which fires only when the scheduler queue is empty while devices are idle.

```
while running:
    1. _bounded_pool.reserve()            -- wait for thread availability (bounded_thread_pool slot)
    2. _task_creation_queue.pop()         -- next scheduling request (fair pop across query bands)
    3. state = get_query_task_global_state(request.query_id)
       -- nullptr means the query was reset while the request sat queued: drop it,
          request.node may already dangle
    4. slot.attach(request.query_id)      -- count the slot against this query BEFORE touching
                                             the operator, so drain_pending_tasks(query_id)
                                             cannot return while the walk is in progress
    5. node = get_operator_for_next_task(request.node)  -- follow hint chain
    6. if node is nullptr: re-evaluate visited pipelines' status; continue

    7. Dispatch work to the pool. Every source — a GPU_SCAN scan or any
       GPU operator with buffered input — drives the same loop:
           - Loop while (!node.all_ports_empty()):
             - pipeline.mark_task_created()  // BEFORE popping data
             - data = node.get_next_task_input_data()  // a GPU_SCAN blocks on its split_connector
             - If data: create a gpu_pipeline_task, dispatch to task_scheduler
             - If no data: pipeline.mark_task_completed()
```

The `mark_task_created()` call before data popping prevents a race condition where the pipeline could appear finished between data check and task creation.

### Lookahead scan warm-up

`schedule_lookahead()` warms up one not-yet-activated scan of a live query when the scheduler's
queue runs dry (called from `task_scheduler::management_eventloop`). It rotates round-robin
across the queries that still accept work (register D3 — it used to hard-code the oldest entry,
so every newer query started cold); a query with nothing warmable, or one quiescing per the
lifecycle registry, does not pin the rotation. The walk-and-push for one query runs under that
query's `lookahead_mutex`, which — together with `drain_pending_tasks()` clearing the lookahead
queue under the same mutex *before* its request drain — makes the operator dereferences safe
against the query's teardown.

## Device Assignment for GPU Tasks

**File:** `src/creator/task_creator.cpp`

When the manager loop builds a `gpu_pipeline_task`, it also chooses the task's `preferred_device_id` (which GPU executor the scheduler should route it to). The choice is resolved in priority order: an upstream scan split's stamped device, then a **partition device pin**, then data-locality by input bytes, then NUMA-affinity. The full locality math lives in [`multi-gpu-architecture.md`](multi-gpu-architecture.md); the partition pin is described here because it is owned by the task creator and is a correctness requirement.

### Partition device pin

Partitioned operators — BUILD_PROBE hash join, `grouped_aggregate_merge`, and the other partition-keyed operators above — build a per-partition cuco hash table that is valid only on the GPU it was built on. Touching it from a stream bound to another device trips `cudaErrorInvalidValue`. So when a task's input is a `partitioned_operator_data`, the task creator pins it to a fixed device:

```
preferred_device_id = active_gpu_ids[ partition_idx % active_gpu_ids.size() ]
```

This keeps every task of a given partition (build + all probes) on one GPU while spreading partitions across GPUs.

`active_gpu_ids` is **per query** — a member of the query's `query_task_global_state`, not of the shared creator (on a shared creator, a member would let one query's narrowing clamp another query's tasks onto GPUs it was never admitted to). It is seeded at registration from `_default_gpu_ids` — every device with a GPU executor, i.e. the memory manager's `Tier::GPU` memory spaces, the same set `task_scheduler` keys executors on — and is then narrowed per query: `sirius_engine::initialize_internal` computes the admitted subset and installs it via `set_active_gpu_ids(query_id, ids, full_count)` before any pipeline is built. The pin indexes this set rather than the physical hardware topology, for two reasons. A physical-topology modulo could name a device with no executor, which the scheduler treats as "no preference" and round-robins — scattering a partition across GPUs. And a query admitted onto a subset must not pin work to a device outside it.

Because the same list is used to build `pipeline_build_context`, the partition floor and the broadcast-join device→slot map agree with what the pin routes across. See [configuration.md](configuration.md) for `topology.gpus_per_query`.

The other device preferences the task creator computes — the operator's own hint, GPU-resident byte counts, NUMA locality via `gpus_of()`, and a cached chunk's home device — are all derived from where data already lives and can name a device outside the admitted subset. They are clamped back into the query's `active_gpu_ids` at the point the preference is written to the task's local state.

A task with no preference at all is confined the same way, but only when the query was admitted onto a strict subset. The scheduler hands an unpreferred task to whichever executor asks first, excluded ones included (a GPU_VALUES task carries no device-bound state and so sets no preference), so on a subset those tasks are pinned round-robin over the query's `active_gpu_ids`. Pinning costs the scheduler's freedom to place the task on whatever device frees up first, so it is skipped when nothing was narrowed away and there is nothing to confine.

The pin is reapplied on the OOM-reschedule path (`gpu_pipeline_executor`): when a task is rebuilt with a fresh local state, its per-task `preferred_device_id` is carried forward so a rescheduled probe doesn't lose its pin and scatter.

## `can_create_more_tasks()` and `has_processed_all_tasks()`

These methods signal task exhaustion:
- `can_create_more_tasks()` — returns false when no more tasks can be created (e.g., scan exhausted, all partitions processed)
- `has_processed_all_tasks()` — returns false when tasks are still in flight

Both throw `not implemented` in the base class and must be overridden by operators that need them.

## `drain_pending_tasks(query_id)` and `reset(query_id)`

**File:** `src/creator/task_creator.cpp`

`drain_pending_tasks(query_id)` is per query — other queries' pending requests and in-flight
creation work are untouched. Called from both the success path
(`task_scheduler::wait_for_completion`) and the error path (`drain_after_error`); `reset(query_id)`
runs it and then drops the query's state entry. Order is load-bearing:

1. Clear the query's **lookahead queue** under its `lookahead_mutex` — before the request drain,
   so a racing `schedule_lookahead()` walk either lands its push ahead of the drain (dropped) or
   finds the queue empty (register D3's teardown-safety half)
2. `_task_creation_queue.drain(query_index{query_id})` — drops this query's pending requests
3. `_bounded_pool->drain_and_wait(query_id)` — waits for this query's in-flight creation lambdas
   (the pool tracks in-flight work per query via the slots attached in `manager_loop`)

`reset(query_id)` must complete before the query's `planner::query`/plan is destroyed: queued
requests hold raw operator pointers into the plan. `SiriusContext::run_mandatory_cleanup` calls it
first, while the parked plan is still alive (see
[Concurrency Model](concurrency-model.md#query-end-the-mandatory-cleanup)).

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
