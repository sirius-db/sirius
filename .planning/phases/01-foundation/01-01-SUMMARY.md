---
phase: 01-foundation
plan: 01
subsystem: memory
tags: [downgrade, thread-pool, mpmc, cuda, memory-management]

# Dependency graph
requires: []
provides:
  - "Plain downgrade_task struct with direct batch/res_mgr members"
  - "Standalone downgrade_executor with own bounded_thread_pool and interruptible_mpmc request queue"
  - "downgrade_request struct with target_bytes, predicate, and promise fields"
  - "processing_loop that dequeues requests and dispatches batch downgrades"
  - "collect_all_candidates helper for shared candidate selection"
affects: [01-02, phase-02, phase-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Request-queue pattern: monitor enqueues downgrade_request, processing thread dequeues"
    - "Collect-then-dispatch: candidate selection separated from pool dispatch"

key-files:
  created: []
  modified:
    - "src/include/downgrade/downgrade_task.hpp"
    - "src/downgrade/downgrade_task.cpp"
    - "src/include/downgrade/downgrade_executor.hpp"
    - "src/downgrade/downgrade_executor.cpp"

key-decisions:
  - "Dropped itask_executor inheritance -- queue-of-requests model replaces queue-of-tasks"
  - "Kept run_downgrade_pass synchronous dispatch for backward compatibility with tests"
  - "CUDA stream creation deferred from constructor to start() for cleaner lifecycle"

patterns-established:
  - "downgrade_request as unit of work for memory reclamation requests"
  - "processing_loop consumes requests sequentially, dispatches batches concurrently"

requirements-completed: [EXEC-01, EXEC-02, CAND-01, CAND-02]

# Metrics
duration: 3min
completed: 2026-04-06
---

# Phase 01 Plan 01: Decouple Executor and Simplify Task Summary

**Standalone downgrade_executor with own thread pool/request queue, plain downgrade_task struct without itask hierarchy**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T13:39:34Z
- **Completed:** 2026-04-06T13:42:37Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Rewrote downgrade_task as a plain struct with direct `batch` and `res_mgr` members, eliminating itask inheritance and global/local state classes
- Rewrote downgrade_executor as a standalone class owning its own bounded_thread_pool and interruptible_mpmc<downgrade_request> queue
- Introduced processing_loop that dequeues requests sequentially and dispatches batch downgrades concurrently to the pool
- Updated monitor_loop to enqueue downgrade_request instead of calling run_downgrade_pass directly
- Extracted collect_all_candidates helper to share candidate selection logic between processing_loop and run_downgrade_pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite downgrade_task as plain struct** - `0c90c959` (feat)
2. **Task 2: Rewrite downgrade_executor as standalone class** - `d1159117` (feat)

## Files Created/Modified
- `src/include/downgrade/downgrade_task.hpp` - Plain struct with batch and res_mgr members
- `src/downgrade/downgrade_task.cpp` - Simplified execute() accessing members directly
- `src/include/downgrade/downgrade_executor.hpp` - Standalone class with own pool, request queue, processing/monitor threads
- `src/downgrade/downgrade_executor.cpp` - processing_loop, monitor_loop, start/stop/drain, candidate selection

## Decisions Made
- Dropped itask_executor inheritance: the base class queue-of-tasks model does not fit queue-of-requests; fighting the abstraction adds complexity
- Kept run_downgrade_pass as synchronous pool dispatch for backward compatibility with tests that check return value
- Deferred CUDA stream creation from constructor to start() so resource lifecycle aligns with running state

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
- `downgrade_request::predicate` field is declared but unused in Phase 1 (will be wired in Phase 2 for predicate-based requests)
- `downgrade_request::result` promise field is declared but unused in Phase 1 (will be wired in Phase 2 for async/blocking API)

## Next Phase Readiness
- Structural skeleton complete: own thread pool, request queue, sequential processing
- Ready for Phase 1 Plan 02 (if any remaining foundation work) and Phase 2 (predicate-based API, blocking/async semantics)
- Candidate selection logic preserved verbatim and factored into reusable collect_all_candidates

---
*Phase: 01-foundation*
*Completed: 2026-04-06*
