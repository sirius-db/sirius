---
phase: 02-numeric-row-preview-and-column-statistics
verified: 2026-04-07T07:15:00Z
status: human_needed
score: 4/4 must-haves verified
re_verification: false
human_verification:
  - test: "Run sirius_unittest '[debug_utils]' on GPU hardware and confirm all 19 test cases pass"
    expected: "All 19 test cases pass with exit code 0 — '19 test cases passed'"
    why_human: "Test binary requires a live CUDA GPU (NVML, CUDA driver). Sandbox environment has no GPU; the Summary documents compilation succeeded but runtime execution was deferred to CI/GPU hardware."
  - test: "Confirm debug_head aligned output actually shows correct values (not garbage or default zeroes) for a multi-type batch in a live pipeline"
    expected: "Aligned table shows integer values, float values formatted with {:g}, booleans as true/false, and NULL for null-marked positions"
    why_human: "Functional correctness of formatted values requires executing GPU memory copies and checking log output — not verifiable by static analysis alone"
---

# Phase 2: Numeric Row Preview and Column Statistics Verification Report

**Phase Goal:** Developers can call `debug_head(batch, N, stream)` and see the first N rows in aligned-column and CSV format for all numeric types, and call `debug_stats(batch, stream)` to see GPU-computed min, max, and sum per numeric column — all output routed through `[SIRIUS_DIAG]`
**Verified:** 2026-04-07T07:15:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

All four truths come from ROADMAP.md success criteria. The additional must-haves from PLAN frontmatter are merged in.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `debug_head(batch, 5, stream)` on INT32/INT64/FLOAT/DOUBLE/BOOL batch prints five rows in fixed-width aligned-column format with correct values and NULL for null positions | ✓ VERIFIED | `debug_head` implemented (src/debug_utils.cpp:284). ALIGNED format branch at line 423. All numeric types dispatched via switch at lines 380-398. NULL display at line 349 `cells[c][r] = "NULL"`. BOOL8 displays true/false at line 375. `[SIRIUS_DIAG] head:` prefix on all output lines. Test cases 9 (multi-type ALIGNED), 13 (null positions) present in test file. |
| 2 | `debug_head(batch, 5, stream, format=csv)` prints the same five rows in CSV format | ✓ VERIFIED | CSV branch at src/debug_utils.cpp:406 `if (format == DebugFormat::CSV)`. `DebugFormat` enum declared in header (line 81: `enum class DebugFormat { ALIGNED, CSV }`). Default parameter `DebugFormat format = DebugFormat::ALIGNED` in header (line 101). Test case 10 calls `sirius::DebugFormat::CSV`. |
| 3 | `debug_stats(batch, stream)` prints per-column min, max, and sum for numeric columns; non-numeric columns appear as `(non-numeric, skipped)` | ✓ VERIFIED | `debug_stats` implemented at src/debug_utils.cpp:470. `is_stats_numeric()` at line 87 returns true for INT8-64/UINT8-64/FLOAT32/64, false for all else (BOOL8 excluded). `"(non-numeric, skipped)"` string at line 513. `[SIRIUS_DIAG] stats:` header at line 482. Test cases 15 (numeric cols), 16 (BOOL skip), 17 (all-NULL), 18 (empty batch), 19 (tier guard). |
| 4 | `debug_stats` uses `cudf::reduce` — no full column is copied to host for statistics computation | ✓ VERIFIED | `cudf::minmax(col, stream)` at line 518 (combined min+max in 1 GPU pass). `cudf::reduce(col, *sum_agg, sum_type, stream)` at line 523 (SUM). No `cudaMemcpyAsync` or host vector in debug_stats body — only `scalar_to_string` helper calls `.value(stream)` on the returned scalar (host-side scalar extraction of a single value, not full column copy). `cudaDeviceSynchronize` confirmed absent from entire file. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/include/debug_utils.hpp` | DebugFormat enum, debug_head declaration, debug_stats declaration | ✓ VERIFIED | `enum class DebugFormat { ALIGNED, CSV }` at line 81. `void debug_head(...)` at line 98 with correct signature and default `DebugFormat::ALIGNED`. `void debug_stats(...)` at line 116. Both inside `namespace sirius`. Existing Phase 1 declarations (host_column_nulls, copy_null_mask_to_host, debug_schema, debug_nulls) unchanged. |
| `src/debug_utils.cpp` | debug_head and debug_stats implementations | ✓ VERIFIED | `void debug_head(...)` at line 284. `void debug_stats(...)` at line 470. Both have full implementations with tier guard, stream sync, try/catch, output buffering, `SIRIUS_LOG_DEBUG`. 545 lines total — substantive. |
| `test/cpp/debug/test_debug_utils.cpp` | Catch2 unit tests for debug_head and debug_stats | ✓ VERIFIED | 19 total `TEST_CASE` entries. 6 debug_head tests (cases 9-14). 5 debug_stats tests (cases 15-19). All use `[debug_utils]` tag. Contains `sirius::DebugFormat::ALIGNED` and `sirius::DebugFormat::CSV`. Contains `cudf::mask_state::ALL_NULL` for all-null test. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/debug_utils.cpp` | `cudf::slice` | zero-copy row selection in debug_head | ✓ WIRED | `cudf::slice(tv, {0, keep}, stream)` at line 313. `#include <cudf/copying.hpp>` at line 11 provides the symbol. |
| `src/debug_utils.cpp` | `cudf::reduce` | GPU-side SUM in debug_stats | ✓ WIRED | `cudf::reduce(col, *sum_agg, sum_type, stream)` at line 523. `#include <cudf/reduction.hpp>` at line 13 provides the symbol. |
| `src/debug_utils.cpp` | `cudf::minmax` | combined min+max in single GPU pass | ✓ WIRED | `cudf::minmax(col, stream)` at line 518. `#include <cudf/reduction.hpp>` provides this. |
| `test/cpp/debug/test_debug_utils.cpp` | `src/debug_utils.cpp` | includes debug_utils.hpp and calls debug_head/debug_stats | ✓ WIRED | `#include "debug_utils.hpp"` at line 18. `sirius::debug_head(...)` called at lines 368, 397, 422, 446, 489, 499. `sirius::debug_stats(...)` called at lines 537, 564, 590, 614, 624. |

### Data-Flow Trace (Level 4)

debug_utils functions output via `SIRIUS_LOG_DEBUG` (side effects, not return values). Data flows: GPU device memory → `cudaMemcpyAsync` to host vectors → formatted into `std::string output` → emitted via `SIRIUS_LOG_DEBUG("{}", output)`. For debug_stats: GPU column → `cudf::minmax`/`cudf::reduce` → GPU scalar → `.value(stream)` (single scalar D2H transfer) → `scalar_to_string` → appended to output string. No hollow props or disconnected data sources found. The flow is complete in implementation.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `debug_head` cell extraction | `cells[c][r]` | `cudaMemcpyAsync(host_vals.data(), col.data<T>(), ...)` from sliced GPU column | Yes — copies actual column data from device | ✓ FLOWING |
| `debug_stats` min/max | `min_scalar`, `max_scalar` | `cudf::minmax(col, stream)` — GPU kernel reduction | Yes — GPU reduction result | ✓ FLOWING |
| `debug_stats` sum | `sum_scalar` | `cudf::reduce(col, *sum_agg, sum_type, stream)` — GPU kernel reduction | Yes — GPU reduction result | ✓ FLOWING |

### Behavioral Spot-Checks

Step 7b: SKIPPED — test binary (`sirius_unittest`) requires a live CUDA GPU (NVML initialization, GPU memory allocation). No GPU is available in the sandbox build environment. Runtime execution is deferred to CI/GPU hardware. Compilation was verified clean by the executor (964/964 targets, no errors).

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `debug_head` symbol present in implementation | `grep -c "void debug_head" src/debug_utils.cpp` | 1 | ✓ PASS |
| `debug_stats` symbol present in implementation | `grep -c "void debug_stats" src/debug_utils.cpp` | 1 | ✓ PASS |
| No `cudaDeviceSynchronize` in implementation | `grep -c "cudaDeviceSynchronize" src/debug_utils.cpp` | 0 | ✓ PASS |
| 19 test cases in test file | `grep -c "TEST_CASE" test/cpp/debug/test_debug_utils.cpp` | 19 | ✓ PASS |
| `[SIRIUS_DIAG]` prefix present throughout | `grep -c "\[SIRIUS_DIAG\]" src/debug_utils.cpp` | 32 | ✓ PASS |
| 4 `try {` blocks (one per public function) | count of `try {` in src/debug_utils.cpp | 4 | ✓ PASS |
| Runtime test execution on GPU | `sirius_unittest "[debug_utils]"` | Cannot run — no GPU | ? SKIP |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| HEAD-01 | 02-01-PLAN.md, 02-02-PLAN.md | `debug_head(batch, N)` prints first N rows in aligned-column format | ✓ SATISFIED | ALIGNED format branch in debug_head (line 422). Fixed-width column widths computed dynamically (lines 425-431). Header + separator + data rows formatted with `{:<{}s}` (lines 434-454). |
| HEAD-02 | 02-01-PLAN.md, 02-02-PLAN.md | `debug_head(batch, N, format=csv)` prints first N rows in CSV format | ✓ SATISFIED | CSV branch at line 406 producing comma-separated output. `DebugFormat::CSV` enum value used in test case 10. |
| HEAD-03 | 02-01-PLAN.md, 02-02-PLAN.md | Uses `cudf::slice` for zero-copy row selection before GPU-to-host transfer | ✓ SATISFIED | `cudf::slice(tv, {0, keep}, stream)` at line 313. Slice result used as `sliced_tv` for all subsequent data access. |
| STATS-01 | 02-01-PLAN.md, 02-02-PLAN.md | `debug_stats(batch)` prints per-column min, max, sum for numeric columns only | ✓ SATISFIED | `debug_stats` outputs min, max, sum for columns passing `is_stats_numeric()`. Non-numeric columns use the skip branch. |
| STATS-02 | 02-01-PLAN.md, 02-02-PLAN.md | Non-numeric columns (STRING, BOOL, DATE, TIMESTAMP) skipped with note | ✓ SATISFIED | `is_stats_numeric()` explicitly excludes BOOL8 and all non-numeric types. Skip message `"(non-numeric, skipped)"` at line 513. Test case 16 verifies BOOL skip. |
| STATS-03 | 02-01-PLAN.md, 02-02-PLAN.md | Uses `cudf::reduce` / `cudf::minmax` for GPU-side computation — no full column copy to host | ✓ SATISFIED | Both `cudf::minmax` and `cudf::reduce` used in debug_stats body. Only the resulting scalars (single values) transferred to host via `.value(stream)`. |

No orphaned requirements: REQUIREMENTS.md maps only HEAD-01, HEAD-02, HEAD-03, STATS-01, STATS-02, STATS-03 to Phase 2. All 6 are claimed by both plans (02-01 and 02-02). No Phase 2 requirements are unaccounted for.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/debug_utils.cpp` | 393-397 | `"(unsupported)"` for STRING/DECIMAL/TIMESTAMP/DATE in debug_head switch default | ℹ️ Info | Intentional — comment explicitly notes `// Unsupported types (STRING, DECIMAL, TIMESTAMP, DATE) -- Phase 3`. Phase 3 is the planned successor phase for full type coverage. Not a stub — it is a documented in-progress state that does not affect Phase 2 scope. |

No TODO/FIXME comments. No `cudaDeviceSynchronize`. No empty `return {}` or placeholder returns. No hardcoded empty data passed as props. No `return null` stubs. The `"(unsupported)"` fallback is the correct Phase 2 behavior per PLAN acceptance criteria.

### Human Verification Required

#### 1. Full test suite on GPU hardware

**Test:** Build the project with `pixi run -e default bash -c 'CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make'`, then execute `build/release/extension/sirius/test/cpp/sirius_unittest "[debug_utils]"`
**Expected:** Output ends with "All tests passed (N assertions in 19 test cases)" and exit code 0
**Why human:** Requires a live CUDA GPU with NVML and device memory. The sandbox CI environment has no GPU. The executor confirmed compilation succeeded (964/964 targets) but explicitly documented that runtime execution was deferred to CI/GPU hardware.

#### 2. Functional output correctness for debug_head

**Test:** Call `sirius::debug_head(*batch, 5, stream)` on a batch with known INT32 values `{10, 20, 30, 40, 50}`, then check the log output.
**Expected:** Log contains a row with values `10`, `20`, `30`, `40`, `50` in aligned column format, with the `[SIRIUS_DIAG] head:` prefix and a separator line of dashes
**Why human:** Verifying actual formatted log output requires running on GPU hardware with logging enabled (`SIRIUS_LOG_LEVEL=debug`). Static analysis confirms the code path is implemented; runtime execution confirms values are correct.

### Gaps Summary

No gaps found. All four roadmap success criteria are implemented and verified at the code level. The implementations in `src/debug_utils.cpp` and `src/include/debug_utils.hpp` match the plan specifications exactly. The test file has the correct 19 test cases covering all required behaviors.

The status is `human_needed` because runtime test execution requires a GPU that is not available in the sandbox. This is an environmental constraint, not an implementation gap.

---

_Verified: 2026-04-07T07:15:00Z_
_Verifier: Claude (gsd-verifier)_
