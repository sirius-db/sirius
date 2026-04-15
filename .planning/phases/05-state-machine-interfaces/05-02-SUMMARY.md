---
phase: 05-state-machine-interfaces
plan: 02
subsystem: data
tags: [c++20, abstract-interface, convertible-data, memory-tiers, pure-virtual]

# Dependency graph
requires:
  - phase: 05-01
    provides: "data_batch state machine extension (task_created -> in_transit transition)"
provides:
  - "convertible_data abstract interface with convert() and bytes_in_space()"
  - "convertible_data_provider abstract interface with get_next_convertible(), get_all_convertible(), get_bytes_in_space()"
  - "Compile-verified interface contracts for Phase 6 and Phase 7 implementations"
affects: [06-batch-conversion, 07-task-queue-conversion]

# Tech tracking
tech-stack:
  added: []
  patterns: [forward-declaration-for-heavy-headers, abstract-interface-with-pure-virtuals]

key-files:
  created:
    - src/include/data/convertible_data.hpp
    - test/cpp/data/test_convertible_data.cpp
  modified:
    - CMakeLists.txt

key-decisions:
  - "Both interfaces in single header (provider depends on convertible_data)"
  - "Forward declarations for memory_space and sirius_memory_reservation_manager to minimize header deps"

patterns-established:
  - "Abstract interface pattern: pure virtuals in sirius namespace with forward-declared cucascade types"
  - "Compile-test pattern: stub subclasses proving interface can be implemented"

requirements-completed: [IFACE-01, IFACE-02]

# Metrics
duration: 3min
completed: 2026-04-15
---

# Phase 5 Plan 2: Abstract Interfaces Summary

**convertible_data and convertible_data_provider abstract interfaces with pure virtual convert/inspect contracts for uniform memory-tier conversion**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-15T19:08:56Z
- **Completed:** 2026-04-15T19:11:27Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Defined convertible_data abstract interface with convert() and bytes_in_space() pure virtuals matching IFACE-01 signature
- Defined convertible_data_provider abstract interface with get_next_convertible(), get_all_convertible(), get_bytes_in_space() matching IFACE-02 signature
- Created compile-only Catch2 tests proving both interfaces can be subclassed, instantiated, and called through base pointers

## Task Commits

Each task was committed atomically:

1. **Task 1: Create convertible_data.hpp with both abstract interfaces** - `3522519e` (feat)
2. **Task 2: Add compile test and register in CMakeLists.txt** - `f7a35b92` (test)

## Files Created/Modified
- `src/include/data/convertible_data.hpp` - Abstract interfaces for convertible_data and convertible_data_provider in sirius namespace
- `test/cpp/data/test_convertible_data.cpp` - Compile-only test with stub subclasses verifying interface contracts
- `CMakeLists.txt` - Added test_convertible_data.cpp to TEST_SOURCES

## Decisions Made
None - followed plan as specified

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Abstract interfaces are stable compilation targets for Phase 6 (convertible_data_batch, convertible_data_batch_provider) and Phase 7 (convertible_gpu_pipeline_task, convertible_gpu_pipeline_task_provider)
- No blockers or concerns

## Self-Check: PASSED

All files exist, all commits verified.

---
*Phase: 05-state-machine-interfaces*
*Completed: 2026-04-15*
