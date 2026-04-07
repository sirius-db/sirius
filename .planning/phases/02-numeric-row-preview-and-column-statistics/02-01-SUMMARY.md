---
phase: 02-numeric-row-preview-and-column-statistics
plan: 01
subsystem: debug-utils
tags: [cudf, cuda, gpu-to-host, cudf-reduce, cudf-minmax, cudf-slice, debug, logging]

# Dependency graph
requires:
  - phase: 01-infrastructure-and-metadata-inspection
    provides: "debug_utils module with host_column_nulls, copy_null_mask_to_host, debug_schema, debug_nulls, tier guard, output buffering pattern"
provides:
  - "DebugFormat enum class (ALIGNED, CSV)"
  - "debug_head function for first-N-rows preview with numeric type support"
  - "debug_stats function for GPU-side per-column min/max/sum statistics"
  - "is_stats_numeric helper for numeric type classification (excludes BOOL8)"
  - "sum_output_type helper for SUM overflow prevention via type widening"
  - "scalar_to_string helper for type-dispatched cudf scalar formatting"
affects: [03-string-temporal-decimal-types, 04-checksum-and-diff]

# Tech tracking
tech-stack:
  added: [cudf/copying.hpp (cudf::slice), cudf/reduction.hpp (cudf::reduce, cudf::minmax), cudf/scalar/scalar.hpp (numeric_scalar)]
  patterns: [zero-copy row slicing via cudf::slice, GPU-side statistics via cudf::minmax+cudf::reduce, generic lambda type dispatch for numeric extraction]

key-files:
  created: []
  modified:
    - src/include/debug_utils.hpp
    - src/debug_utils.cpp

key-decisions:
  - "Used generic lambda with template operator() for type dispatch in debug_head instead of cudf::type_dispatcher -- simpler for the switch-case pattern with bulk memcpy"
  - "Used cudf::minmax for combined min+max (1 GPU kernel) plus cudf::reduce for SUM (1 kernel) = 2 kernel launches per column instead of 3"
  - "Cast int8_t to int before fmt::format to prevent char-like formatting"

patterns-established:
  - "Bulk column copy: cudaMemcpyAsync for entire sliced column, then format all values on host"
  - "Null offset awareness: col.data<T>() is offset-adjusted, but null_mask() is NOT -- use col.offset() + r for null bit checks"
  - "SUM type widening: INT8/16/32 -> INT64, UINT8/16/32 -> UINT64, FLOAT32 -> FLOAT64"

requirements-completed: [HEAD-01, HEAD-02, HEAD-03, STATS-01, STATS-02, STATS-03]

# Metrics
duration: 48min
completed: 2026-04-07
---

# Phase 02 Plan 01: Numeric Row Preview and Column Statistics Summary

**debug_head with aligned/CSV output for all numeric+bool types via cudf::slice zero-copy, and debug_stats with GPU-side cudf::minmax+cudf::reduce per-column min/max/sum statistics**

## Performance

- **Duration:** 48 min
- **Started:** 2026-04-07T05:37:45Z
- **Completed:** 2026-04-07T06:25:45Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- debug_head prints first N rows in aligned-column or CSV format for INT8-64, UINT8-64, FLOAT32/64, and BOOL8 with NULL display and dynamic column widths
- debug_stats computes per-column min, max, sum entirely on GPU using cudf::minmax (single-pass min+max) and cudf::reduce (SUM with overflow-safe type widening)
- Non-numeric columns display as "(non-numeric, skipped)" in stats and "(unsupported)" in head for types deferred to Phase 3
- All output routed through [SIRIUS_DIAG] prefix via SIRIUS_LOG_DEBUG, zero use of cudaDeviceSynchronize, full try/catch safety wrapping

## Task Commits

Each task was committed atomically:

1. **Task 1: Add DebugFormat enum and function declarations** - `0471c134` (feat)
2. **Task 2: Implement debug_head** - `22197588` (feat)
3. **Task 3: Implement debug_stats** - `aa8b9ced` (feat)

## Files Created/Modified
- `src/include/debug_utils.hpp` - Added DebugFormat enum, debug_head and debug_stats declarations with doxygen documentation
- `src/debug_utils.cpp` - Added debug_head implementation (aligned+CSV output, cudf::slice, all numeric type dispatch, BOOL8, null handling), debug_stats implementation (cudf::minmax, cudf::reduce, is_stats_numeric, sum_output_type, scalar_to_string helpers), plus new includes (cudf/copying.hpp, cudf/reduction.hpp, cudf/scalar/scalar.hpp)

## Decisions Made
- Used generic lambda with `template operator()` for type dispatch in debug_head rather than cudf::type_dispatcher functor -- the switch-case + generic lambda pattern is simpler when doing bulk memcpy followed by host-side formatting
- Used cudf::minmax for combined min+max in a single GPU pass, reducing kernel launches from 3 to 2 per numeric column
- Cast int8_t values to int before fmt::format to prevent char-like formatting (fmt treats int8_t as integer, but defensive cast ensures correctness)
- Per-column stream.synchronize() rather than batching all async copies -- copy_null_mask_to_host already requires per-column sync, so the optimization window is narrow

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- pixi cache lock was read-only in the sandbox environment, preventing `pixi run` builds. Worked around by setting PATH directly to the pixi environment binaries and building with cmake/ninja directly
- Worktree had no initialized submodules; ran `git submodule update --init --recursive` before build
- Both issues resolved; build succeeded (964/964 targets), all 8 existing debug_utils tests pass (29 assertions)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- debug_head and debug_stats are functional for all numeric types plus BOOL8
- Phase 3 (STRING, TIMESTAMP, DATE, DECIMAL type support) can extend the existing switch-case dispatch in debug_head and add type cases to is_stats_numeric/scalar_to_string
- Phase 4 (debug_checksum, debug_diff) can build on the established patterns: tier guard, output buffering, try/catch, [SIRIUS_DIAG] routing
- Blocker carried forward: cudf::strings_column_view API verification needed for Phase 3 STRING support

## Self-Check: PASSED

- [x] src/include/debug_utils.hpp exists
- [x] src/debug_utils.cpp exists
- [x] Commit 0471c134 exists
- [x] Commit 22197588 exists
- [x] Commit aa8b9ced exists
- [x] Build succeeds (964/964 targets)
- [x] All existing tests pass (8 test cases, 29 assertions)

---
*Phase: 02-numeric-row-preview-and-column-statistics*
*Completed: 2026-04-07*
