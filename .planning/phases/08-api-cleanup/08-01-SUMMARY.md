---
phase: 08-api-cleanup
plan: 01
subsystem: downgrade
tags: [downgrade-executor, api-cleanup, predicate-based, inspectable-mpsc]

# Dependency graph
requires:
  - phase: 07-task-queue-conversion
    provides: inspectable_mpsc<itask> as production task queue in itask_executor
provides:
  - Predicate-only request_downgrade API (no target_bytes)
  - Constructor with optional task queue pointers for tiered fallback
  - Post-reserve satisfied check preventing over-dispatch
affects: [08-02, downgrade-executor, processing-loop-rewrite]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Predicate-only downgrade request pattern: caller defines stopping condition via predicate, no target_bytes needed"
    - "Post-reserve satisfied check: re-check predicate after blocking reserve() to prevent dispatching unnecessary work"

key-files:
  created: []
  modified:
    - src/include/downgrade/downgrade_executor.hpp
    - src/downgrade/downgrade_executor.cpp
    - src/pipeline/gpu_pipeline_executor.cpp
    - test/cpp/downgrade/test_downgrade_executor.cpp

key-decisions:
  - "Use std::numeric_limits<size_t>::max() for unlimited candidate collection instead of 0"
  - "Add post-reserve satisfied check to prevent over-dispatching when predicate controls stopping"

patterns-established:
  - "Predicate-controlled downgrade stopping: no target_bytes needed, predicate defines termination"

requirements-completed: [DAPI-01, DAPI-02]

# Metrics
duration: 95min
completed: 2026-04-16
---

# Phase 8 Plan 1: API Cleanup Summary

**Removed target_bytes from downgrade API, added task queue constructor params, fixed over-dispatch race in predicate-controlled candidate processing**

## Performance

- **Duration:** 95 min (mostly build time -- incremental build + submodule init)
- **Started:** 2026-04-16T18:05:04Z
- **Completed:** 2026-04-16T19:40:46Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Stripped `target_bytes` from `downgrade_request` struct and `request_downgrade` API across header, implementation, call site, and tests
- Extended `downgrade_executor` constructor with optional `gpu_task_queue` and `pipeline_task_queue` pointers (preparation for Plan 02 tiered fallback)
- Fixed a race condition where unlimited candidate collection could over-dispatch batches past predicate satisfaction
- All 20 downgrade tests passing (12 executor + 8 lifecycle, 104 assertions total)

## Task Commits

Each task was committed atomically:

1. **Task 1: Strip target_bytes from downgrade API and extend constructor** - `ca97a561` (feat)
2. **Task 2: Update gpu_pipeline_executor call site and all tests** - `bb383033` (feat)

## Files Created/Modified
- `src/include/downgrade/downgrade_executor.hpp` - Removed target_bytes member, updated request_downgrade signature, added task queue member pointers and constructor params
- `src/downgrade/downgrade_executor.cpp` - Updated constructor, request_downgrade, request_free_memory, monitor_loop, and processing_loop to remove all target_bytes usage; added post-reserve satisfied check
- `src/pipeline/gpu_pipeline_executor.cpp` - Removed target_bytes calculation and updated request_downgrade call to predicate-only
- `test/cpp/downgrade/test_downgrade_executor.cpp` - Updated request_downgrade test to use predicate-only API

## Decisions Made
- Used `std::numeric_limits<size_t>::max()` instead of `0` for unlimited candidate collection, since the original `collect_all_candidates` function treats 0 as "collect nothing" (0 >= 0 is true), not "collect everything"
- Added a post-reserve() satisfied check in the processing loop to prevent dispatching batches when the predicate was already satisfied during the blocking reserve() call

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unlimited candidate collection value**
- **Found during:** Task 2 (test verification)
- **Issue:** Plan specified passing `0` to `collect_all_candidates` to mean "collect all", but the function interprets 0 as "collect nothing" because `collected_bytes >= 0` is always true
- **Fix:** Changed to `std::numeric_limits<size_t>::max()` and added `#include <limits>`
- **Files modified:** `src/downgrade/downgrade_executor.cpp`
- **Verification:** All 12 downgrade_executor tests pass
- **Committed in:** bb383033 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed over-dispatch race condition in processing loop**
- **Found during:** Task 2 (test verification)
- **Issue:** With unlimited candidate collection, the processing loop could dispatch a batch even after the predicate was satisfied, because the satisfied check happens before `reserve()` blocks. While blocked in reserve(), the previous batch's callback may set satisfied=true, but the loop doesn't re-check before dispatching.
- **Fix:** Added second `req->satisfied.load()` check after `reserve()` returns
- **Files modified:** `src/downgrade/downgrade_executor.cpp`
- **Verification:** "request_free_memory iterates partitions from last to first" test passes -- only 2 of 4 batches downgraded as expected
- **Committed in:** bb383033 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes were necessary for correctness when switching from target_bytes-limited to unlimited candidate collection. No scope creep.

## Issues Encountered
- cucascade submodule required manual clone from local reference (remote fetch failed for the required commit)
- Build took significant time due to fresh worktree requiring full compilation

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 02 (processing loop rewrite) can proceed -- target_bytes is fully removed, constructor accepts task queue pointers
- `collect_all_candidates` still exists with its scoring/sorting logic -- Plan 02 will replace this with convertible_data_provider-based lazy iteration

---
*Phase: 08-api-cleanup*
*Completed: 2026-04-16*
