---
phase: 08-api-cleanup
plan: 02
subsystem: downgrade
tags: [downgrade-executor, processing-loop, convertible-data, tiered-providers]

# Dependency graph
requires:
  - phase: 08-api-cleanup
    plan: 01
    provides: Predicate-only request_downgrade API, task queue constructor pointers
provides:
  - Rewritten processing_loop with tiered convertible_data providers
  - downgrade_task eliminated from codebase
  - Per-tier breakdown logging (repos/gpu_queue/pipeline_queue)
affects: [downgrade-executor, memory-management-docs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tiered candidate fetching: repos -> gpu queue -> pipeline queue via convertible_data_provider"
    - "Snapshot-then-dispatch: get_all_convertible() per repo to avoid re-dispatching same batch"
    - "Post-reserve satisfied check: prevents over-dispatch when predicate satisfied during blocking reserve()"

key-files:
  created: []
  modified:
    - src/downgrade/downgrade_executor.cpp
    - src/include/downgrade/downgrade_executor.hpp
    - CMakeLists.txt
    - test/cpp/downgrade/test_downgrade_executor.cpp
    - test/cpp/downgrade/test_downgrade_lifecycle.cpp
    - docs/super-sirius/memory-management.md
    - docs/super-sirius/optimizations.md
  deleted:
    - src/include/downgrade/downgrade_task.hpp
    - src/downgrade/downgrade_task.cpp

key-decisions:
  - "Use get_all_convertible() per repo instead of lazy get_next_convertible() loop to avoid re-dispatching same batch before state changes"
  - "Rename 'prioritizes partitioned repos' test since lazy iteration does not guarantee partitioned-first ordering"

patterns-established:
  - "Tiered provider pattern: convertible_data_batch_provider for repos, convertible_gpu_pipeline_task_provider for task queues"

requirements-completed: [LOOP-01, LOOP-02, LOOP-03, LOOP-04, LOOP-05, LOG-01]

# Metrics
duration: 76min
completed: 2026-04-16
---

# Phase 8 Plan 2: Processing Loop Rewrite Summary

**Rewrote downgrade processing loop to use tiered convertible_data providers, eliminated downgrade_task, added per-tier logging**

## Performance

- **Duration:** 76 min (build time dominant -- full rebuild from scratch in worktree)
- **Started:** 2026-04-16T19:44:22Z
- **Completed:** 2026-04-16T21:00:10Z
- **Tasks:** 2
- **Files modified:** 7 (+ 2 deleted)

## Accomplishments

- Rewrote `processing_loop()` to fetch candidates through tiered `convertible_data_provider` chain: data repositories -> gpu_pipeline_executor task queue -> pipeline_executor task queue
- Replaced `downgrade_task::execute()` with `convertible_data::convert()` calls throughout
- Removed all dead code: `collect_all_candidates`, `get_repo_data_size_on_tier`, `is_partition_active`, `collect_candidates_from_partition`, `scored_repo`, `downgrade_repository_info`
- Deleted `downgrade_task.hpp` and `downgrade_task.cpp` from codebase and build system
- Added per-tier breakdown logging: repos/gpu_queue/pipeline_queue batches and bytes per request
- Predicate checked both in dispatch loop (between dispatches) and in workers (after each convert)
- All 19 downgrade tests passing (11 executor + 8 lifecycle, 98 assertions total)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite processing_loop with tiered providers and convert() calls** - `054e170d` (feat)
2. **Task 2: Delete downgrade_task, update CMake, tests, and docs** - `1fa9172b` (feat)

## Files Created/Modified

- `src/downgrade/downgrade_executor.cpp` - Complete processing_loop rewrite with tiered providers, removed all helper function implementations
- `src/include/downgrade/downgrade_executor.hpp` - Removed downgrade_repository_info struct, dead method declarations, downgrade_task.hpp include
- `CMakeLists.txt` - Removed downgrade_task.cpp from EXTENSION_SOURCES
- `test/cpp/downgrade/test_downgrade_executor.cpp` - Removed downgrade_task include, deleted obsolete single-task test, updated partitioned-repos test
- `test/cpp/downgrade/test_downgrade_lifecycle.cpp` - Removed downgrade_task include
- `docs/super-sirius/memory-management.md` - Updated downgrade request pattern and candidate selection strategy sections, removed downgrade_task section
- `docs/super-sirius/optimizations.md` - Updated downgrade request pattern mechanism description

## Files Deleted

- `src/include/downgrade/downgrade_task.hpp` - Obsolete: conversion now handled by convertible_data::convert()
- `src/downgrade/downgrade_task.cpp` - Obsolete: conversion now handled by convertible_data::convert()

## Decisions Made

- Used `get_all_convertible()` per repository instead of `get_next_convertible()` in a loop. The stateless provider re-scans from scratch each call, which could return the same batch before its state changes from idle. Snapshot-then-dispatch avoids this race.
- Renamed "prioritizes partitioned repos" test to "downgrades across multiple repos" since the new lazy iteration processes repos in `for_each_repository` order without the old scored_repo priority sort.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed double-dispatch race in processing loop**
- **Found during:** Task 2 (test verification -- "partial fulfillment returns actual bytes freed")
- **Issue:** `get_next_convertible()` in a loop re-scans the same batch before its state changes from idle (conversion happens asynchronously in worker thread), causing the same batch to be dispatched twice
- **Fix:** Changed to `get_all_convertible()` per repository to snapshot eligible batches once, then iterate the snapshot for dispatch
- **Files modified:** `src/downgrade/downgrade_executor.cpp`
- **Verification:** "partial fulfillment" test passes -- freed matches single batch size
- **Committed in:** 1fa9172b (Task 2 commit)

**2. [Rule 1 - Bug] Fixed missing post-reserve() satisfied check**
- **Found during:** Task 2 (test verification -- "iterates partitions from last to first")
- **Issue:** After `_pool->reserve()` blocks and returns, the previous candidate's worker may have set satisfied=true. Without re-checking, a new candidate is dispatched unnecessarily.
- **Fix:** Added `if (req->satisfied.load()) return true;` after `_pool->reserve()` in dispatch_candidate lambda
- **Files modified:** `src/downgrade/downgrade_executor.cpp`
- **Verification:** "iterates partitions from last to first" test passes -- only 2 of 4 batches downgraded
- **Committed in:** 1fa9172b (Task 2 commit)

**3. [Rule 2 - Behavioral Change] Updated partitioned-repos test**
- **Found during:** Task 2 (test analysis)
- **Issue:** Old test asserted partitioned repos are downgraded before non-partitioned repos. The new code iterates repos in for_each_repository order without priority sorting.
- **Fix:** Updated test to verify lazy iteration across multiple repos without asserting specific ordering
- **Files modified:** `test/cpp/downgrade/test_downgrade_executor.cpp`
- **Committed in:** 1fa9172b (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 behavioral test update)
**Impact on plan:** Bugs 1-2 were race conditions inherent to the lazy iteration approach. Both fixed with minimal changes. Test update reflects expected behavioral difference from old scored_repo ordering.

## Issues Encountered

- cucascade submodule required local clone (remote fetch failed for required commit)
- Full rebuild required from fresh worktree -- dominated execution time

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 8 Plan 2 is complete -- both plans in Phase 8 are done
- downgrade_task eliminated, processing loop uses convertible_data providers
- All downgrade tests passing (19 tests, 98 assertions)

---
*Phase: 08-api-cleanup*
*Completed: 2026-04-16*
