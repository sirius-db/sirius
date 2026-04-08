---
phase: 03-full-type-coverage-and-checksums
plan: 02
subsystem: debug-utils
tags: [cudf, xxhash_64, xor-reduce, checksum, gpu-hashing, fingerprint]

# Dependency graph
requires:
  - phase: 03-full-type-coverage-and-checksums
    plan: 01
    provides: debug_head with full type coverage, debug_stats with GPU reductions
provides:
  - debug_checksum function with per-column xxhash_64 + XOR reduce GPU pipeline
  - Output format "col[N] checksum: 0xHEX nulls=N" for cross-pipeline diff comparison
  - Empty batch and all-NULL column handling with 0x0000000000000000
  - 5 new Catch2 unit tests for checksum coverage
affects: [validate-skill, runtime-errors-skill, 04-diff]

# Tech tracking
tech-stack:
  added: [cudf/hashing.hpp]
  patterns: [per-column xxhash_64 hash + bitwise XOR reduce to single 64-bit fingerprint]

key-files:
  created: []
  modified: [src/include/debug_utils.hpp, src/debug_utils.cpp, test/cpp/debug/test_debug_utils.cpp]

key-decisions:
  - "Seed value 0 for xxhash_64 (standard default, deterministic across runs per D-12)"
  - "Per-column iteration: wrap each column in single-column table_view for xxhash_64"
  - "All-NULL guard before cudf::reduce to avoid invalid scalar access (T-03-05)"
  - "Header line includes batch_id, row count, column count for context (following existing pattern)"

patterns-established:
  - "GPU-only checksum: xxhash_64(single_col_tv) -> reduce(XOR) -> scalar.value(stream)"
  - "All-NULL/empty guard pattern: check nc == col.size() before GPU operations"

requirements-completed: [CHKSUM-01, CHKSUM-02, CHKSUM-03]

# Metrics
duration: 14min
completed: 2026-04-08
---

# Phase 3 Plan 2: Checksum Implementation Summary

**debug_checksum computing per-column xxhash_64 fingerprints with GPU-only XOR reduce -- enabling cross-pipeline data comparison via log diff**

## Performance

- **Duration:** 14 min
- **Started:** 2026-04-08T21:42:02Z
- **Completed:** 2026-04-08T21:56:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Implemented debug_checksum function that computes deterministic per-column 64-bit checksums using cudf::hashing::xxhash_64 (hash all rows) + cudf::reduce with bitwise XOR (collapse to single value)
- Entire computation stays on GPU -- no column data copied to host, only the final 64-bit scalar value
- Output format per D-11: "col[N] checksum: 0xABCD1234EF567890 nulls=2" for easy diff between log files
- Handles edge cases: empty batches print "(empty batch)", all-NULL columns output 0x0000000000000000, null-data batches log warning without crashing
- Added 5 new Catch2 unit tests covering numeric columns, multi-type batches (INT32+STRING+DECIMAL64), empty batches, all-NULL columns, and tier guard

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement debug_checksum function with xxhash_64 + XOR reduce** - `ce299da1` (feat)
2. **Task 2: Add Catch2 unit tests for debug_checksum** - `8b6c55d4` (test)

## Files Created/Modified
- `src/include/debug_utils.hpp` - Added debug_checksum declaration with docstring after debug_stats
- `src/debug_utils.cpp` - Added cudf/hashing.hpp include; implemented debug_checksum with per-column xxhash_64 + XOR reduce pipeline, empty/all-NULL guards, try/catch wrapping
- `test/cpp/debug/test_debug_utils.cpp` - Added 5 new TEST_CASE entries (tests 27-31) for debug_checksum

## Decisions Made
- Seed value 0 for xxhash_64 (standard default, ensures deterministic output per D-12)
- Per-column hashing: each column wrapped in a single-column table_view for xxhash_64 call
- All-NULL column guard (nc == col.size()) prevents calling cudf::reduce on all-null data (T-03-05)
- Header line includes batch_id, rows, cols for context -- consistent with debug_schema/debug_stats/debug_head patterns
- Used cudf::get_current_device_resource_ref() for memory resource -- consistent with GPU-only operation (no host allocation needed)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- debug_checksum is production-ready for cross-pipeline data comparison
- 31 unit tests pass (92 assertions), build clean
- All Phase 1, Phase 2, and Phase 3 Plan 01 tests unaffected
- Ready for debug_diff implementation in Phase 4

## Self-Check: PASSED

- [x] src/include/debug_utils.hpp exists and contains debug_checksum declaration
- [x] src/debug_utils.cpp exists and contains cudf::hashing::xxhash_64 usage
- [x] test/cpp/debug/test_debug_utils.cpp exists with 31 test cases
- [x] 03-02-SUMMARY.md exists
- [x] Commit ce299da1 found (Task 1)
- [x] Commit 8b6c55d4 found (Task 2)

---
*Phase: 03-full-type-coverage-and-checksums*
*Completed: 2026-04-08*
