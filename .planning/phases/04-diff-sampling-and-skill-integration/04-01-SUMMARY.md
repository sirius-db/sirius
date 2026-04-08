---
phase: 04-diff-sampling-and-skill-integration
plan: 01
subsystem: debug-utilities
tags: [cudf, gather, diff, sampling, mt19937, cuda, debug]

# Dependency graph
requires:
  - phase: 03-full-type-coverage-and-checksums
    provides: debug_head formatting pipeline, all type dispatch helpers, host_column_nulls, copy_null_mask_to_host
provides:
  - debug_diff function for two-batch schema/value comparison with per-column diff reporting
  - debug_sample function for random row selection via std::mt19937 and cudf::gather
  - format_rows_to_output shared helper extracted from debug_head
affects: [04-02 (unit tests for diff/sample), 04-03 (skill integration references)]

# Tech tracking
tech-stack:
  added: [std::mt19937, cudf::gather (in debug_utils)]
  patterns: [shared formatting helper extraction, host-side batch comparison]

key-files:
  created: []
  modified:
    - src/include/debug_utils.hpp
    - src/debug_utils.cpp

key-decisions:
  - "Exact equality for float comparison (D-05) -- no epsilon, debug tool catches every bit flip"
  - "Host-side comparison for debug_diff rather than GPU-side cudf::binaryop -- simpler code, full per-type control"
  - "cudf::column_view wrapping rmm::device_buffer for gather indices -- avoids column factory dependency"

patterns-established:
  - "format_rows_to_output: shared cell extraction + output formatting reused by debug_head and debug_sample"
  - "compare_numeric lambda template: generic host-side typed comparison with null awareness for debug_diff"

requirements-completed: [DIFF-01, DIFF-02, DIFF-03, DIFF-04, DIFF-05, SAMPLE-01, SAMPLE-02, SAMPLE-03]

# Metrics
duration: 7min
completed: 2026-04-08
---

# Phase 04 Plan 01: Debug Diff and Sample Summary

**debug_diff for two-batch comparison with schema/value mismatch reporting, debug_sample for random row selection via cudf::gather, and format_rows_to_output shared helper extracted from debug_head**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-08T22:43:15Z
- **Completed:** 2026-04-08T22:50:14Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added debug_diff and debug_sample declarations to debug_utils.hpp with full Doxygen documentation
- Extracted format_rows_to_output shared helper from debug_head, eliminating ~260 lines of duplication
- Implemented debug_diff with schema mismatch checks (column count, types), row count mismatch, max_rows guard (10M default), and per-column host-side value comparison for all Sirius-supported types
- Implemented debug_sample with std::mt19937 RNG (optional seed for reproducibility), cudf::gather for GPU row extraction, and format_rows_to_output for display
- Project builds successfully with no errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Add debug_diff and debug_sample declarations to debug_utils.hpp** - `28d032da` (feat)
2. **Task 2: Implement debug_diff, debug_sample, and extract shared formatting helper** - `3893ac00` (feat)

## Files Created/Modified
- `src/include/debug_utils.hpp` - Added debug_diff and debug_sample declarations with Doxygen docs, added #include <optional>
- `src/debug_utils.cpp` - Extracted format_rows_to_output helper, refactored debug_head, implemented debug_diff and debug_sample

## Decisions Made
- Used exact equality for float comparison (D-05) -- no epsilon tolerance; debug tool should catch every bit flip
- Used host-side comparison for debug_diff rather than GPU cudf::binaryop -- simpler code, full per-type control
- Constructed cudf::column_view wrapping rmm::device_buffer for gather indices rather than using column factory functions
- Sorted random sample indices before gather for better GPU memory coalescing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- pixi `clang` environment specified in plan verification command does not exist in this workspace; used default environment instead (build succeeded)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- debug_diff and debug_sample are compiled and linked, ready for unit testing in plan 04-02
- format_rows_to_output helper available for any future formatting functions
- Skill integration (plan 04-03) can reference the complete debug utility API

## Self-Check: PASSED

All files verified present, all commits verified in git log.

---
*Phase: 04-diff-sampling-and-skill-integration*
*Completed: 2026-04-08*
