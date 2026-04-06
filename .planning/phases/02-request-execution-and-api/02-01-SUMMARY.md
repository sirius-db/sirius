---
phase: 02-request-execution-and-api
plan: 01
subsystem: memory
tags: [downgrade, predicate, future, async, thread-pool, cuda, memory-management]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "Standalone downgrade_executor with own bounded_thread_pool and request queue"
provides:
  - "Predicate-driven incremental dispatch in processing_loop"
  - "request_free_memory(bytes) async API returning future<size_t>"
  - "request_free_memory_and_wait(bytes) synchronous blocking API"
  - "request_downgrade(predicate) custom predicate API returning future<size_t>"
  - "downgrade_request with atomic bytes_freed and satisfied members"
  - "monitor_loop fire-and-forget requests with predicate"
affects: [02-02, phase-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Predicate-checked incremental dispatch: check satisfied flag before reserving next pool slot"
    - "Atomic per-request counters: bytes_freed.fetch_add after each batch, predicate check after each success"
    - "Promise fulfillment after wait_all: set_value with final bytes_freed count"

key-files:
  created: []
  modified:
    - "src/include/downgrade/downgrade_executor.hpp"
    - "src/downgrade/downgrade_executor.cpp"

key-decisions:
  - "Predicate null-check before calling: monitor fire-and-forget requests always have a predicate, but request_downgrade callers might pass nullptr"
  - "bytes_freed counts batch size before downgrade (estimated), not after -- matches collect_candidates accounting"

patterns-established:
  - "Request API pattern: make_unique<downgrade_request>, set predicate, get_future, push to queue"
  - "Incremental dispatch: for each candidate, check satisfied, reserve slot, dispatch with atomic tracking"

requirements-completed: [RAPI-01, RAPI-02, RAPI-03, RAPI-04, RAPI-05, EXEC-03, EXEC-04, EXEC-05]

# Metrics
duration: 2min
completed: 2026-04-06
---

# Phase 02 Plan 01: Request Execution and API Summary

**Predicate-driven incremental dispatch engine with async/sync/custom-predicate public APIs for GPU memory reclamation**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-06T15:10:53Z
- **Completed:** 2026-04-06T15:12:40Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Evolved downgrade_request with atomic bytes_freed and satisfied members for per-request progress tracking
- Rewrote processing_loop to check satisfied flag before each dispatch and evaluate predicate after each successful batch downgrade
- Implemented three public API methods: request_free_memory (async), request_free_memory_and_wait (blocking), request_downgrade (custom predicate)
- Updated monitor_loop to push fire-and-forget requests with proper byte-threshold predicate
- Removed legacy run_downgrade_pass and run_downgrade_pass_all_repos methods

## Task Commits

Each task was committed atomically:

1. **Task 1: Evolve downgrade_request and add public API declarations** - `0125aa57` (feat)
2. **Task 2: Implement incremental dispatch, public API methods, and update monitor_loop** - `90748ad9` (feat)

## Files Created/Modified
- `src/include/downgrade/downgrade_executor.hpp` - Evolved struct with atomics, three new public method declarations, legacy methods removed
- `src/downgrade/downgrade_executor.cpp` - Incremental dispatch processing_loop, three API implementations, updated monitor_loop, legacy methods removed

## Decisions Made
- Added null-check on predicate before calling (`if (req_ptr->predicate && req_ptr->predicate())`) to handle edge cases where predicate might be empty
- bytes_freed tracks the pre-downgrade batch size (from get_size_in_bytes before execute), consistent with candidate collection accounting

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all API methods are fully wired with predicates, atomics, and promise fulfillment.

## Next Phase Readiness
- Public API surface complete and ready for test updates in Plan 02
- Tests referencing run_downgrade_pass will need updating (handled in 02-02-PLAN.md)
- Compilation verification deferred to Plan 02 per plan design

---
*Phase: 02-request-execution-and-api*
*Completed: 2026-04-06*
