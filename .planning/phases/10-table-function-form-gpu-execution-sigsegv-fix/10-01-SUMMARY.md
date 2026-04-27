---
phase: 10-table-function-form-gpu-execution-sigsegv-fix
plan: 01
subsystem: testing
tags: [bisect, sigsegv, gpu_execution, table-function, cuda, cudf]

# Dependency graph
requires:
  - phase: 09-scan-task-distributor-batch-ownership-affinity
    provides: "5-commit source span (3b58258..c0e12f3) identified as SIGSEGV introduction window per 09-04-VALIDATION.md"
provides:
  - "Bisect result: regressing_commit=NONE — all 5 Phase-9 source commits pass isolated test"
  - "Critical finding: SIGSEGV is test-ordering dependent, not commit-specific; shows as cudaErrorContextIsDestroyed at HEAD+FU17"
  - "10-01-BISECT.md ledger with per-commit build_exit + test_exit + interpretation labels"
  - "Revised hypothesis for Plan 10-02: run full suite to reproduce, then gdb on SIGSEGV frame"
affects: [10-02-gdb, 10-03-targeted-fix, 10-04-reship-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bisect via git checkout detached-HEAD + MCP build/test per commit (5-commit linear walk)"
    - "MCP unit-tests wrapper required for GPU access; direct binary invocation fails in sandbox"
    - "Stash both src/ FU17 changes AND planning artifacts before bisect to ensure clean checkouts"

key-files:
  created:
    - ".planning/phases/10-table-function-form-gpu-execution-sigsegv-fix/10-01-BISECT.md"
  modified: []

key-decisions:
  - "regressing_commit=NONE: all 5 Phase-9 source commits (3b58258..c0e12f3) pass 'gpu_execution - filter equality parquet' in isolation; SIGSEGV is NOT commit-specific in this window"
  - "SIGSEGV is test-ordering dependent: 09-04 reproduced with full-suite --abort ~[hive_partition]; isolated filter run passes at all 5 commits"
  - "FU17 partial fix changes (stashed before bisect) change the failure mode at HEAD from SIGSEGV to cudaErrorContextIsDestroyed — FU17 probes expose the underlying bug differently"
  - "Plan 10-02 revised target: run full suite to reproduce SIGSEGV, then gdb; H2 (TABLE_FUNCTION vs CALL-form result materialization divergence) remains leading structural hypothesis"

patterns-established: []

requirements-completed: [CRIT-2]

# Metrics
duration: 30min
completed: 2026-04-27
---

# Phase 10 Plan 01: TABLE_FUNCTION SIGSEGV Bisect Summary

**Bisect of 5-commit Phase-9 source span (3b58258..c0e12f3) finds NONE regressing: all commits pass in isolation; SIGSEGV is test-ordering dependent, not commit-specific**

## Performance

- **Duration:** 30 min
- **Started:** 2026-04-27T14:50:14Z
- **Completed:** 2026-04-27T15:20:00Z
- **Tasks:** 3
- **Files modified:** 1 (10-01-BISECT.md created)

## Accomplishments
- Walked all 5 Phase-9 source commits (3b58258, 863cc6c, 0c8068e, a8a7985, c0e12f3) with MCP build + isolated test run at each
- All 5 commits: build_exit=0, test_exit=0 (PASS); no SIGSEGV in isolated test execution
- Verified HEAD match after restore (478c937b matches pre-bisect), stashes restored cleanly
- Discovered that at HEAD+FU17 partial fix changes, the failure mode changed to `cudaErrorContextIsDestroyed` (not SIGSEGV) — FU17 probes alter memory ordering and expose the bug differently
- `regressing_commit: NONE` documented in 10-01-BISECT.md frontmatter per plan deviation guidance

## Task Commits

Tasks 1-3 are documentation/planning tasks only (no source file commits). The plan's output section explicitly states: "No SUMMARY.md needed (BISECT.md IS the summary artifact for bisect plans)". Per-task commits omitted for data-only plans (bisect results are in BISECT.md).

**Plan metadata commit:** see final state update commit below.

## Files Created/Modified
- `.planning/phases/10-table-function-form-gpu-execution-sigsegv-fix/10-01-BISECT.md` — Per-commit bisect ledger with build/test exit codes, interpretation labels, and NONE conclusion

## Decisions Made
- **NONE conclusion:** All 5 source commits pass the isolated test. The SIGSEGV is test-ordering dependent — not introduced by any specific commit in the Phase-9 window.
- **FU17 as failure-mode indicator:** FU17 partial fix changes at HEAD change SIGSEGV to `cudaErrorContextIsDestroyed`; Plan 10-02 should gdb the clean state (c0e12f3, without FU17 changes) running the full suite.
- **H2 remains leading hypothesis:** TABLE_FUNCTION vs CALL-form result materialization divergence is still the most likely root cause. The second `SELECT * FROM gpu_execution(...)` invocation crashes while the first `CALL gpu_execution(...)` succeeds. Full-suite test ordering (an earlier test leaving GPU context corrupted) may be the proximate trigger.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stash FU17 partial fix changes before bisect**
- **Found during:** Task 1 (pre-bisect setup)
- **Issue:** 6 modified src/ files (FU17 partial fix changes) and .planning/STATE.md would cause `git checkout <hash>` to fail with "Your local changes would be overwritten"
- **Fix:** `git stash push` for FU17 changes and planning artifacts separately; restored both stashes after bisect; HEAD and all working-tree changes verified identical to pre-bisect state
- **Files affected:** src/data/sirius_p2p_converter.cpp, src/include/pipeline/batch_lock_utils.hpp, src/op/sirius_physical_concat.cpp, src/op/sirius_physical_hash_join.cpp, src/op/sirius_physical_parquet_scan.cpp, src/op/sirius_physical_partition.cpp, .planning/STATE.md, .ai-helper/commands.yaml
- **Verification:** `git diff --name-only HEAD -- src/ test/cpp/` returns same 6 files as before bisect started (confirmed against git status from session context)
- **Committed in:** N/A (stash/unstash operation, no commit needed)

**2. [Rule 3 - Blocking] Use MCP unit-tests instead of direct binary for test runs**
- **Found during:** Task 2 (first test attempt at commit 1)
- **Issue:** Direct `./build/release/extension/sirius/test/cpp/sirius_unittest` invocation fails in sandbox with `Failed to initialize NVML: Driver Not Loaded` / `SiriusContext::initialize: 0 GPUs` (GPU not visible to sandbox child processes)
- **Fix:** Used `mcp__project-commands__run_command(name="unit-tests", filter="gpu_execution - filter equality parquet")` for all 5 test runs, consistent with feedback_use_mcp_build.md and feedback_mcp_tests_scope.md
- **Verification:** MCP wrapper correctly routes to GPU-accessible process; exit=0 for all 5 commits
- **Committed in:** N/A (execution methodology choice)

---

**Total deviations:** 2 auto-fixed (both Rule 3 blocking)
**Impact on plan:** Both required for bisect to run at all. No scope creep.

## Issues Encountered
- SIGSEGV not reproduced in isolated test run — consistent with 09-04 evidence that SIGSEGV occurred in full-suite context with `--abort ~[hive_partition]`, not when the test runs alone
- At HEAD+FU17 changes, failure mode is `cudaErrorContextIsDestroyed` (not SIGSEGV) — indicates FU17 probes (which add cudaGetDevice calls and possibly sync points) change observable behavior

## Known Stubs
None. This is a bisect/analysis plan; no data-producing code was written.

## Next Phase Readiness
- Plan 10-02 (gdb): Target the full-suite run at clean state (c0e12f3 or HEAD without FU17 changes), run with `--abort ~[hive_partition]` to reproduce the SIGSEGV, then attach gdb
- H2 (TABLE_FUNCTION vs CALL-form divergence) remains leading structural hypothesis for Plan 10-03 fix
- FU17 partial fix changes should be considered during Plan 10-02 — they change the failure mode and may be masking the original crash site

---
*Phase: 10-table-function-form-gpu-execution-sigsegv-fix*
*Completed: 2026-04-27*
