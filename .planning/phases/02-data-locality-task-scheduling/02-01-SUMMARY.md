---
phase: 02-data-locality-task-scheduling
plan: 01
subsystem: pipeline-scheduling
tags: [multi-gpu, data-locality, task-routing, numa]
dependency_graph:
  requires: [01-01, 01-02, 01-03]
  provides: [preferred-device-id-plumbing, locality-aware-routing]
  affects: [pipeline_executor, task_creator, gpu_pipeline_executor]
tech_stack:
  added: []
  patterns: [push-model-task-dispatch, data-locality-scoring]
key_files:
  created: []
  modified:
    - src/include/pipeline/gpu_pipeline_task.hpp
    - src/include/pipeline/sirius_pipeline_task_states.hpp
    - src/include/creator/task_creator.hpp
    - src/creator/task_creator.cpp
    - src/pipeline/pipeline_executor.cpp
    - src/pipeline/gpu_pipeline_executor.cpp
    - src/sirius_context.cpp
decisions:
  - preferred_device_id lives on both local_state (per-task) and global_state (pipeline default) with local_state taking precedence
  - Switched management_eventloop from pull model (wait for task_request) to push model (pop task first, route by preference)
  - NUMA-to-GPU mapping uses first GPU found on each NUMA node as the representative
  - Task waits on preferred GPU when at capacity rather than trying other GPUs (future optimization noted)
metrics:
  duration: 34min
  completed: 2026-04-03T15:57:00Z
  tasks_completed: 2
  tasks_total: 2
  files_modified: 7
requirements: [SCHED-01, SCHED-02, SCHED-03, SCHED-04]
---

# Phase 02 Plan 01: Data Locality Computation and Locality-Aware Routing Summary

Data-locality-aware GPU selection added to the task creation and dispatch flow, routing pipeline tasks to the GPU where their input data resides using a push-model dispatch pattern.

## What Was Done

### Task 1: Add preferred_device_id to task state and compute locality score in task_creator (59bc2848)

Added `preferred_device_id` field with getter/setter to both `gpu_pipeline_task_local_state` (per-task override) and `sirius_pipeline_task_global_state` (pipeline-level default). The `gpu_pipeline_task` class exposes a unified `get_preferred_device_id()` that checks local state first, then falls back to global state.

In `task_creator::manager_loop()`, after creating GPU pipeline task local state, the code now scans input data batches to compute a locality score:
- **SCHED-01**: If any GPU has data loaded, routes to the GPU with the most bytes.
- **SCHED-02**: If data is only on HOST, routes to the GPU on the same NUMA node (using a NUMA-to-GPU mapping built at construction time from `system_topology_info`).

The `task_creator` constructor now accepts an optional `system_topology_info*` pointer, and `SiriusContext` passes `config_.get_hw_topology()` to it.

### Task 2: Change management_eventloop to route tasks by preferred_device_id (dd9264b3)

Replaced the pull-model dispatch (wait for `task_request`, then pop task, dispatch to requesting GPU) with a push-model dispatch (pop task first, read `preferred_device_id`, push to correct GPU executor).

Key changes:
- `pipeline_executor::management_eventloop()` now pops from `_task_queue` first, reads the task's preferred device, and routes to `_gpu_executors.at(target_device_id)`.
- `gpu_pipeline_executor::manager_loop()` no longer sends task_request signals (removed `_task_request_publisher.send()` call). Capacity control still works via `_bounded_pool->reserve()`.
- Scan task flow is unchanged (scan tasks go directly to `_scan_executor` in `schedule()`).
- **SCHED-03**: Tasks wait on their preferred GPU when it's at capacity (the task sits in the GPU executor's internal queue).
- **SCHED-04**: Different tasks from the same query can route to different GPUs based on their individual data locality.

## Decisions Made

1. **Push model over pull model**: The management_eventloop no longer waits for GPU executors to signal readiness. Instead, it immediately routes tasks based on data locality. Capacity control is handled within each GPU executor via bounded_pool.
2. **Local-then-global preference**: Per-task `preferred_device_id` (from locality scoring) takes precedence over pipeline-level default. This allows tasks in the same pipeline to route to different GPUs.
3. **Wait-on-preferred over try-others**: When the preferred GPU is at capacity, the task queues on that GPU rather than trying alternative GPUs. This avoids data movement thrashing. A TODO is documented for future optimization.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functionality is wired end-to-end.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 59bc2848 | Add preferred_device_id to task state and locality scoring |
| 2 | dd9264b3 | Change management_eventloop to push-model locality routing |
