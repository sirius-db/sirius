---
phase: 03-lifecycle-and-pipeline-integration
plan: 01
subsystem: pipeline
tags: [downgrade, memory-management, gpu-pipeline, retry-loop]

# Dependency graph
requires:
  - phase: 02-request-execution-and-api
    provides: request_free_memory_and_wait API on downgrade_executor
provides:
  - downgrade_executor* injection into gpu_pipeline_executor via constructor
  - retry-with-downgrade loop in gpu_pipeline_executor::manager_loop
  - pipeline_executor plumbing of downgrade_executor pointers
  - SiriusContext initialization reordering (downgrade before pipeline)
affects: [03-02-PLAN, monitor-loop, drain-semantics]

# Tech tracking
tech-stack:
  added: []
  patterns: [constructor-injection-with-nullptr-default, retry-with-backpressure]

key-files:
  created: []
  modified:
    - src/include/pipeline/gpu_pipeline_executor.hpp
    - src/pipeline/gpu_pipeline_executor.cpp
    - src/include/pipeline/pipeline_executor.hpp
    - src/pipeline/pipeline_executor.cpp
    - src/sirius_context.cpp

key-decisions:
  - "Retry releases partial reservation before requesting downgrade to avoid pinning memory"
  - "Early break on freed==0 prevents pointless retries when no candidates exist"
  - "Downgrade executors start() deferred until after all objects constructed"

patterns-established:
  - "Constructor injection with nullptr default: backward-compatible dependency injection"
  - "Retry-with-downgrade: release reservation, request downgrade, re-acquire, check"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03]

# Metrics
duration: 88min
completed: 2026-04-06
---

# Phase 03 Plan 01: Pipeline Integration Summary

**Retry-with-downgrade loop in gpu_pipeline_executor calling request_free_memory_and_wait up to 5 times when memory reservation falls short**

## Performance

- **Duration:** 88 min
- **Started:** 2026-04-06T16:51:38Z
- **Completed:** 2026-04-06T18:19:28Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- gpu_pipeline_executor receives downgrade_executor* via constructor injection and retries reservation up to 5 times using request_free_memory_and_wait
- pipeline_executor matches downgrade executors to GPU spaces by space_id and passes them to each gpu_pipeline_executor
- SiriusContext initialization reordered: downgrade executors created before pipeline_executor so pointers are available at construction time
- Backward compatibility preserved: nullptr defaults mean all existing call sites and tests compile without changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Add downgrade_executor to constructors** - `1ec0318d` (feat)
2. **Task 2: Implement retry loop and reorder SiriusContext** - `7ce33282` (feat)
3. **Task 3: Build verification and formatting** - `e720f7ca` (chore)

## Files Created/Modified
- `src/include/pipeline/gpu_pipeline_executor.hpp` - Added forward decl, constructor param, private member
- `src/pipeline/gpu_pipeline_executor.cpp` - Added retry-with-downgrade loop in manager_loop
- `src/include/pipeline/pipeline_executor.hpp` - Added downgrade_executors constructor param
- `src/pipeline/pipeline_executor.cpp` - Space-id matching and downgrade_executor plumbing
- `src/sirius_context.cpp` - Reordered init: downgrade executors before pipeline_executor

## Decisions Made
- Release partial reservation before requesting downgrade -- avoids pinning memory that could be freed
- Early break when freed == 0 -- no point retrying if nothing was freed
- Defer downgrade executor start() until after pipeline_executor and task_creator constructed -- clean initialization order
- Use space->get_id() for matching rather than constructing new memory_space_id -- more direct comparison

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Worktree was on dev branch, not downgrade_request -- merged to get Phase 1/2 changes before executing
- clang-format adjusted include order and line wrapping -- auto-fixed in Task 3

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Pipeline integration complete, gpu_pipeline_executor can now request memory reclamation from downgrade_executor
- Ready for Plan 02: monitor loop preservation and start/stop/drain lifecycle integration

---
*Phase: 03-lifecycle-and-pipeline-integration*
*Completed: 2026-04-06*
