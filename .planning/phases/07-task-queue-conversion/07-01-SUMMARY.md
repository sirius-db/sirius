---
phase: 07-task-queue-conversion
plan: 01
subsystem: data
tags: [inspectable_mpsc, convertible_data, RAII, gpu_pipeline_task, memory_tier]

# Dependency graph
requires:
  - phase: 05-state-machine-interfaces
    provides: convertible_data and convertible_data_provider abstract interfaces
  - phase: 06-batch-conversion
    provides: convertible_data_batch pattern (save/lock/convert/restore)
provides:
  - convertible_gpu_pipeline_task (RAII task wrapper with queue push-back on destruction)
  - convertible_gpu_pipeline_task_provider (queue-based provider using mutable_pop_if)
affects: [07-02-plan, downgrade integration, memory pressure handling]

# Tech tracking
tech-stack:
  added: []
  patterns: [RAII queue ownership for temporary task extraction, dynamic_cast chain for heterogeneous queue inspection]

key-files:
  created:
    - src/include/data/convertible_gpu_pipeline_task.hpp
  modified: []

key-decisions:
  - "get_bytes_in_space returns 0 because inspectable_mpsc lacks const iteration; callers use get_all_convertible + bytes_in_space for exact totals"
  - "Predicate in has_matching_batches only performs dynamic_casts and state checks (no I/O, no allocation) per T-07-01 mitigation"

patterns-established:
  - "RAII queue ownership: extract task via mutable_pop_if, wrap in RAII object, destructor pushes back via queue.push()"
  - "Dynamic_cast chain predicate: itask -> gpu_pipeline_task -> local_state -> gpu_pipeline_task_local_state -> pipelineable_operator_data"

requirements-completed: [TASK-01, TASK-02, TASK-03]

# Metrics
duration: 2min
completed: 2026-04-15
---

# Phase 7 Plan 1: Task Queue Conversion Summary

**RAII convertible_gpu_pipeline_task wrapper and mutable_pop_if-based provider for inspectable_mpsc queue task discovery and memory-tier conversion**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-15T23:10:45Z
- **Completed:** 2026-04-15T23:12:43Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Implemented `convertible_gpu_pipeline_task` with RAII destructor that pushes task back to queue on all code paths (TASK-01)
- Implemented `convertible_gpu_pipeline_task_provider` using `mutable_pop_if` with lightweight dynamic_cast predicate filtering (TASK-02)
- `convert()` follows save/lock/convert/restore pattern with per-batch independence and exception safety (TASK-03)
- Both classes inherit from Phase 5 abstract interfaces (`convertible_data` / `convertible_data_provider`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement convertible_gpu_pipeline_task and convertible_gpu_pipeline_task_provider** - `2707800e` (feat)

## Files Created/Modified
- `src/include/data/convertible_gpu_pipeline_task.hpp` - Header-only file containing both classes: RAII task wrapper and queue-based provider

## Decisions Made
- `get_bytes_in_space()` returns 0 rather than attempting extract-count-reinsert, because `inspectable_mpsc` lacks const iteration and temporary removal is unsafe under concurrent producers. Callers needing exact totals should use `get_all_convertible()` + `bytes_in_space()`.
- Predicate `has_matching_batches()` is a static method performing only dynamic_casts and batch state checks -- no I/O, no allocation, no exceptions -- satisfying the lightweight predicate contract of `inspectable_mpsc` and mitigating T-07-01.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `convertible_gpu_pipeline_task.hpp` is ready for integration testing in plan 07-02
- Both classes follow the same patterns established by `convertible_data_batch.hpp` in Phase 6
- Provider can be wired into downgrade or memory pressure systems alongside the batch provider

---
*Phase: 07-task-queue-conversion*
*Completed: 2026-04-15*
