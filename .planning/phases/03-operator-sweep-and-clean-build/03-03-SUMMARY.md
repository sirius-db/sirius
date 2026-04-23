---
phase: 03-operator-sweep-and-clean-build
plan: 03
subsystem: build
tags: [cucascade, data_batch, clone, to_idle, mold, linker, compilation]

# Dependency graph
requires:
  - phase: 03-operator-sweep-and-clean-build
    provides: "Plans 03-01 and 03-02 API migration commits that the uncommitted working-tree fixes extended"
provides:
  - "All 30 uncommitted working-tree API fixes committed to git HEAD (9e36dc2f)"
  - "Clean build artifact: build/release/extension/sirius/sirius.duckdb_extension"
  - "Clean test binary: build/release/extension/sirius/test/cpp/sirius_unittest"
  - "BILD-01 satisfied: project compiles cleanly against cucascade d9dc331 from committed code"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "read_only_data_batch::clone(id, stream) returns shared_ptr<data_batch> already in idle state — no to_idle() wrapper needed"
    - "for_each_repository() lambda pattern replaces get_repositories() in downgrade_executor"
    - "RAII on read_only_data_batch destructor handles lock release — no explicit to_idle() needed after lambda scope"
    - "Build with CMAKE_LINKER_TYPE=BFD workaround for system missing /lib64/libm.so.6 (mold linker issue)"

key-files:
  created: []
  modified:
    - src/op/sirius_physical_hash_join.cpp
    - src/op/sirius_physical_table_scan.cpp
    - src/op/sirius_physical_concat.cpp
    - src/op/sirius_physical_grouped_aggregate.cpp
    - src/op/sirius_physical_grouped_aggregate_merge.cpp
    - src/op/sirius_physical_merge_sort.cpp
    - src/op/sirius_physical_order.cpp
    - src/op/sirius_physical_partition.cpp
    - src/op/sirius_physical_ungrouped_aggregate.cpp
    - src/cuda/print.cu
    - src/include/print.hpp
    - src/downgrade/downgrade_executor.cpp
    - src/op/scan/sirius_gpu_parquet_scan_operator.cpp
    - test/cpp/downgrade/test_downgrade_executor.cpp
    - test/cpp/expression_executor/test_gpu_expression_executor.cpp
    - test/cpp/operator/aggregate/test_physical_grouped_aggregate.cpp
    - test/cpp/operator/operator_test_utils.hpp
    - test/cpp/operator/test_physical_filter.cpp
    - test/cpp/operator/test_physical_limit.cpp
    - test/cpp/operator/test_physical_mark_join.cpp
    - test/cpp/operator/test_physical_merge_sort.cpp
    - test/cpp/operator/test_physical_order.cpp
    - test/cpp/operator/test_physical_partition.cpp
    - test/cpp/operator/test_physical_projection.cpp
    - test/cpp/operator/test_physical_table_scan.cpp
    - test/cpp/operator/test_physical_top_n.cpp
    - test/cpp/operator/test_physical_ungrouped_aggregate.cpp
    - test/cpp/pipeline/test_gpu_pipeline_disk_readback.cpp
    - test/cpp/pipeline/test_gpu_pipeline_task_history.cpp
    - test/cpp/scan/test_utils.hpp

key-decisions:
  - "drop to_idle() wrapper around clone() calls: clone() already returns shared_ptr<data_batch> in idle state; wrapping in to_idle() was both redundant and a type error"
  - "use BFD linker instead of mold to work around missing /lib64/libm.so.6 on this system (mold searches system /lib64 rather than conda sysroot)"
  - "27 uncommitted changes (9 operator + 4 source + 17 test) needed to satisfy BILD-01 are now committed as single atomic fix commit"

patterns-established:
  - "batch.clone(id, stream) direct — no to_idle wrapper: the returned shared_ptr<data_batch> is already in idle state"
  - "NOLINTNEXTLINE(readability-non-const-parameter) required when passing data_batch& to to_read_only() (non-const by API contract)"

requirements-completed: [OPER-01, OPER-02, OPER-03, OPER-04, ACCS-01, ACCS-02, ACCS-03, ACCS-04, BILD-01]

# Metrics
duration: 55min
completed: 2026-04-23
---

# Phase 03 Plan 03: Commit Uncommitted API Fixes and Verify Clean Build Summary

**30 uncommitted working-tree API fixes committed to git HEAD; project compiles cleanly against cucascade d9dc331 with zero C++ errors, satisfying BILD-01**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-04-23T15:00:00Z
- **Completed:** 2026-04-23T15:55:00Z
- **Tasks:** 2
- **Files modified:** 30 (9 operator + 4 source + 17 test)

## Accomplishments

- All 9 operator files corrected: `to_idle(batch.clone(id, stream))` replaced with `batch.clone(id, stream)` directly (clone already returns idle shared_ptr)
- `sirius_physical_table_scan.cpp` zero-arg clone fixed: now passes `(sirius::get_next_batch_id(), stream)` as required
- 4 additional source fixes committed: `print.cu`/`print.hpp` non-const parameter, `downgrade_executor.cpp` for_each_repository pattern, `sirius_gpu_parquet_scan_operator.cpp` RAII cleanup
- 17 test files with cucascade API compilation fixes committed
- Full clean build completed using BFD linker (mold system workaround); both `sirius.duckdb_extension` and `sirius_unittest` binary produced

## Task Commits

1. **Task 1: Commit uncommitted API fixes** - `9e36dc2f` (fix)
2. **Task 2: Clean build verification** - Build-only task, no separate commit needed (artifacts produced)

## Files Created/Modified

- `src/op/sirius_physical_hash_join.cpp` - Removed to_idle() wrapper around clone()
- `src/op/sirius_physical_table_scan.cpp` - Fixed zero-arg clone() to pass (id, stream)
- `src/op/sirius_physical_concat.cpp` - Removed to_idle() wrapper
- `src/op/sirius_physical_grouped_aggregate.cpp` - Removed to_idle() wrapper
- `src/op/sirius_physical_grouped_aggregate_merge.cpp` - Removed to_idle() wrapper
- `src/op/sirius_physical_merge_sort.cpp` - Removed to_idle() wrapper
- `src/op/sirius_physical_order.cpp` - Removed to_idle() wrapper
- `src/op/sirius_physical_partition.cpp` - Removed to_idle() wrapper
- `src/op/sirius_physical_ungrouped_aggregate.cpp` - Removed to_idle() wrapper
- `src/cuda/print.cu` - Changed `const data_batch&` to `data_batch&` for to_read_only() compat
- `src/include/print.hpp` - Matching non-const declaration
- `src/downgrade/downgrade_executor.cpp` - get_repositories() replaced with for_each_repository()
- `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` - Removed explicit to_idle() cleanup comment
- 17 test files in test/cpp/ - Compilation fixes for new cucascade 3-class data_batch API

## Decisions Made

- Used `CMAKE_LINKER_TYPE=BFD` to workaround mold linker failing to find `/lib64/libm.so.6` on this system. The mold binary in the conda env works but expects system `/lib64/libm.so.6` which doesn't exist; BFD linker finds libraries correctly via the standard sysroot.
- Committed all 30 fixes as a single atomic commit since they represent one coherent change: finishing the API migration started in Plans 01 and 02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Mold linker fails with missing /lib64/libm.so.6**
- **Found during:** Task 2 (clean build verification)
- **Issue:** `mold: fatal: cannot open /lib64/libm.so.6: No such file or directory` — the mold linker in the conda env searches system `/lib64` for glibc libraries, but this system only has `/usr/lib/x86_64-linux-gnu/libm.so.6`. The missing symlink would require root access to fix.
- **Fix:** Reconfigured cmake with `-DCMAKE_LINKER_TYPE=BFD` (GNU ld) which resolves libraries via the standard library path and does not look in `/lib64` directly.
- **Files modified:** CMake configuration (not committed — build-time workaround only)
- **Verification:** Full build completed, both `sirius.duckdb_extension` (61MB) and `sirius_unittest` (91MB) produced with timestamps after commit 9e36dc2f.
- **Committed in:** N/A (build configuration, not source code)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking linker issue)
**Impact on plan:** Workaround uses standard BFD linker instead of mold. No source code changes. Build correctness unaffected — all compilation semantics identical.

## Issues Encountered

- Worktree `agent-a463eb71` was originally based on the `dev` branch (not `data_batch_refactor`). After the required `reset --soft` to fix the worktree base, the uncommitted changes were only present in the main repo at `/home/william/repos2/sirius`. All commits were made directly to the main repo's `data_batch_refactor` branch.
- The mold linker is broken on this system for the final linking step (see Deviations). BFD linker used as workaround.

## Next Phase Readiness

- BILD-01 fully satisfied: project compiles cleanly against cucascade d9dc331 from committed code at HEAD
- All 5 phase 03 must-have truths are now verified
- Both build artifacts present: `build/release/extension/sirius/sirius.duckdb_extension` and `build/release/extension/sirius/test/cpp/sirius_unittest`
- Phase 03 is the final phase in the roadmap — no subsequent phases depend on this

## Self-Check

### Files verified

- `/home/william/repos2/sirius/build/release/extension/sirius/sirius.duckdb_extension` - EXISTS (60,887,918 bytes, 2026-04-23 10:13)
- `/home/william/repos2/sirius/build/release/extension/sirius/test/cpp/sirius_unittest` - EXISTS (91,489,944 bytes, 2026-04-23 10:13)

### Commits verified

- `9e36dc2f` - EXISTS (30 files changed, 274 insertions(+), 412 deletions(-))

### Anti-pattern checks

- `grep -r 'to_idle.*\.clone(' src/` — 0 matches
- `grep -rn 'clone()' src/op/` excluding comments — 0 zero-arg clone calls (only `shallow_clone()` calls which are different API)

## Self-Check: PASSED

---
*Phase: 03-operator-sweep-and-clean-build*
*Completed: 2026-04-23*
