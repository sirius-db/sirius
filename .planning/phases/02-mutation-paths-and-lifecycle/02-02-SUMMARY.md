---
phase: 02-mutation-paths-and-lifecycle
plan: "02"
subsystem: pipeline
tags: [cucascade, data_batch, read_only_data_batch, clone_to, subscribe, unsubscribe, gpu_pipeline_task, result_collector]

# Dependency graph
requires:
  - phase: 01-pipeline-data-path
    provides: lock_or_prepare_batch returning read_only_data_batch; pipelineable_operator_data using new accessor types
provides:
  - result_collector using read_only_data_batch::clone_to for GPU-to-HOST conversion (one-step)
  - gpu_pipeline_task subscribes to input batches in constructor and unsubscribes in destructor
  - _input_batches member on gpu_pipeline_task for lifecycle tracking
affects: [03-compile-cleanup, gpu_pipeline_executor, downgrade_executor, task_creator]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "clone_to<TargetRepresentation> pattern: one-step deep copy + representation conversion via read_only_data_batch"
    - "subscribe/unsubscribe RAII lifecycle: track input batches in constructor, release in destructor"
    - "to_read_only() + to_idle() bracketing: acquire shared lock to inspect data, release on all exit paths"

key-files:
  created: []
  modified:
    - src/op/sirius_physical_result_collector.cpp
    - src/include/pipeline/gpu_pipeline_task.hpp
    - src/pipeline/gpu_pipeline_task.cpp

key-decisions:
  - "Use one-step ro.clone_to<host_data_representation>() instead of two-step clone()+convert_to() per D-05/CONV-03"
  - "Centralize subscribe/unsubscribe in gpu_pipeline_task constructor/destructor to cover all operators using tasks (D-06)"
  - "Destructor wraps unsubscribe() in try/catch to prevent exception propagation (T-02-02 threat mitigation)"
  - "OOM rescheduling inherits correct subscriber count automatically: old task destructs (unsubscribe), new task constructs (subscribe)"

patterns-established:
  - "clone_to pattern: call ro.clone_to<T>(registry, next_batch_id, &mem_space, stream) on read_only_data_batch to convert representation in one step"
  - "subscribe/unsubscribe pattern: push_back to _input_batches in ctor, iterate and call unsubscribe() in dtor with try/catch"

requirements-completed: [CONV-03, LIFE-01, LIFE-02]

# Metrics
duration: 8min
completed: 2026-04-22
---

# Phase 02 Plan 02: Mutation Paths and Lifecycle (Part 2) Summary

**Result collector rewritten to use read_only_data_batch::clone_to one-step GPU-to-HOST conversion; gpu_pipeline_task subscribe/unsubscribe lifecycle wired via _input_batches member**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-22T14:21:00Z
- **Completed:** 2026-04-22T14:29:16Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Replaced two-step `clone()+convert_to()` pattern in `sirius_physical_materialized_collector::sink()` with one-step `ro.clone_to<cucascade::host_data_representation>()` (D-05/CONV-03)
- All data access in the result collector now goes through RAII accessors (`read_only_data_batch`) with explicit `to_idle()` on all exit paths — no direct `get_data()` on idle batches
- Added `_input_batches` member to `gpu_pipeline_task` and wired `subscribe()`/`unsubscribe()` in constructor/destructor, replacing the removed `batch_state::task_created` lifecycle (LIFE-01/LIFE-02/D-06)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite result collector to use read_only_data_batch::clone_to** - `8082e34c` (feat)
2. **Task 2: Wire subscribe/unsubscribe lifecycle into gpu_pipeline_task** - `a7117968` (feat)

## Files Created/Modified

- `src/op/sirius_physical_result_collector.cpp` - Rewrote sink_single_batch lambda: to_read_only() at entry, clone_to<host_data_representation>() for GPU tier, to_idle() on all exit paths, HOST path also uses ro accessor
- `src/include/pipeline/gpu_pipeline_task.hpp` - Added `_input_batches` private member (vector<shared_ptr<data_batch>>) for lifecycle tracking
- `src/pipeline/gpu_pipeline_task.cpp` - Constructor subscribes to all input batches; destructor unsubscribes with try/catch

## Decisions Made

- Used `ro.clone_to<cucascade::host_data_representation>()` (one-step) instead of the old `clone()` + `convert_to()` two-step per D-05 and CONV-03. The new cucascade API provides this as a first-class operation on `read_only_data_batch`.
- Scoped `using host_table_chunk_reader` declarations inside each branch (GPU and HOST) rather than at the top of the sink function, since different branches use it independently.
- Centralized `subscribe()`/`unsubscribe()` in `gpu_pipeline_task` constructor/destructor (D-06): all operators create tasks through `gpu_pipeline_task`, so this single centralization covers all input batches for every operator.
- Wrapped `unsubscribe()` in try/catch in the destructor to comply with the C++ rule that destructors must not propagate exceptions (T-02-02 mitigation).
- OOM rescheduling correctly handles subscriber counts: when a task is rescheduled, the old task destructs (calls unsubscribe), then the new task constructs (calls subscribe) — net change is zero, no code change needed in `create_rescheduled_task()`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - all data access is wired to real cucascade API calls.

## Next Phase Readiness

- CONV-03, LIFE-01, LIFE-02, D-05, D-06 all satisfied
- Result collector now uses the new 3-class accessor API exclusively for data access
- gpu_pipeline_task lifecycle management replaces the removed `batch_state::task_created` with subscribe/unsubscribe reference counting
- Ready for Phase 03 (compile cleanup) which will address remaining sites still using old API patterns

---
*Phase: 02-mutation-paths-and-lifecycle*
*Completed: 2026-04-22*
