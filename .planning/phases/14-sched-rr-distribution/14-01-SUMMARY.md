---
phase: 14-sched-rr-distribution
plan: 01
subsystem: pipeline-scheduler
tags: [multi-gpu, sched-rr, task-scheduler, round-robin, distribution, std-map, atomic]

# Dependency graph
requires:
  - phase: 13-q11-multi-gpu-illegal-address
    provides: "cucascade writer-event lineage in convert_gpu_to_gpu (cucascade @ 7409c60) — closes the cumulative-state Q11 hang that was the rollback reason for SCHED-RR in earlier sessions"
provides:
  - "Deterministic ascending-by-device_id iteration of _gpu_executors via std::map (replaces std::unordered_map hash-bucket order)"
  - "Atomic per-query-resettable round-robin counter (_no_pref_rr_counter) for preference-less source-pipeline tasks"
  - "Per-query reset of that counter in prepare_for_query so cache=table_gpu warm-path stays correct across iterations"
  - "Round-robin fallback in management_eventloop when !have_pref && _gpu_executors.size() > 1, gated to leave 1-GPU and preference-bearing tasks untouched"
affects: [14-02-validation, 15-mgpu-operator-colocation-audit]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deterministic ordered iteration over GPU executors (std::map by device_id) — required for reproducible preference-less task dispatch"
    - "Per-query atomic counter reset pattern (store(0, std::memory_order_relaxed) in prepare_for_query) — prevents warm-path cache miss when iterations of the same query rotate to different GPUs"
    - "have_pref boolean tracking inside dynamic_cast block, then SCHED-RR fallback gated on !have_pref && size>1 — keeps 1-GPU and locality-bearing tasks unchanged"

key-files:
  created: []
  modified:
    - "src/include/pipeline/task_scheduler.hpp (includes block + _gpu_executors decl + _no_pref_rr_counter decl)"
    - "src/pipeline/task_scheduler.cpp (counter reset in prepare_for_query + SCHED-RR fallback in management_eventloop)"

key-decisions:
  - "std::unordered_map -> std::map for _gpu_executors so begin()->first is ascending-by-device_id deterministic, not hash-bucket-order (verified by 14-CONTEXT.md rationale lines 82-86)"
  - "Atomic round-robin counter (_no_pref_rr_counter) gated on !have_pref && size>1 so preference-bearing tasks keep their SCHED-01/02/04 locality and 1-GPU configurations are untouched"
  - "Per-query reset in prepare_for_query (store(0, std::memory_order_relaxed)) so the same query iterated twice in a row dispatches preference-less tasks to the same GPUs (cache=table_gpu warm-path correctness — verified by Phase 13 follow-up #17 scale-up test which fails 0==N without this reset)"
  - "Patch applied via Edit tool, not git apply — diff in 14-CONTEXT.md is documentation-style without --- a/.../+++ b/... headers, which Phase 13 Wave 1 + Wave 4 both proved git apply rejects"
  - "Branch base feat/sched-rr-distribution off current HEAD (8c4600c, the docs(14) commit one above Phase 13 HEAD 833bb72) — keeps the Phase 14 plan files on the branch alongside the patch"

patterns-established:
  - "SCHED-RR fallback in management_eventloop: dynamic_cast for preference -> have_pref tracking -> if (!have_pref && size>1) atomic-fetch-add modulo size with std::advance on map iterator"
  - "Counter-state reset placement: in prepare_for_query AFTER the GPU drain loop (so reset is paired with drain at query boundary)"

requirements-completed: []  # Plan frontmatter has empty requirements:[] field

# Metrics
duration: 2 min
completed: 2026-04-30
---

# Phase 14 Plan 01: Land SCHED-RR distribution Summary

**Atomic round-robin counter with per-query reset + std::map executor ordering, enabling deterministic preference-less source-task distribution across GPUs (the v1.3 multi-GPU distribution unblocker).**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-30T21:32:25Z
- **Completed:** 2026-04-30T21:34:38Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Switched `_gpu_executors` from `std::unordered_map<int, ...>` to `std::map<int, ...>` so `.begin()->first` is ascending-by-device_id deterministic (no more hash-bucket-order surprise across hardware).
- Added `std::atomic<size_t> _no_pref_rr_counter{0}` to `task_scheduler` and used it as the round-robin selector for preference-less pipeline tasks (PARQUET_METADATA_SCAN, GPU_PARQUET_SCAN, etc.).
- Reset the counter in `task_scheduler::prepare_for_query` (paired with the existing per-GPU drain loop) so cache=table_gpu warm-path correctness is preserved across same-query iterations.
- Build clean (MCP `name=build` exit 0), HYG-02 baseline preserved at 40 occurrences of `rmm::cuda_stream_default` in `src/`, and the canary follow-up #17 scale-up test passed (178 assertions in 7.3s) on the patched build.

## Task Commits

Each task was committed atomically:

1. **Task 1: Switch to feat/sched-rr-distribution off Phase 13 HEAD, then apply the SCHED-RR patch via Edit and build** — `d4009e2` (feat)

_No metadata commit yet — that follows in the orchestrator's git_commit_metadata step._

## Files Created/Modified

- `src/include/pipeline/task_scheduler.hpp` — Replaced `<unordered_map>` include with `<atomic>` + `<map>` (alphabetical order), and replaced the `std::unordered_map<int, std::unique_ptr<gpu_pipeline_executor>>` declaration with a `std::map<int, ...>` plus `std::atomic<size_t> _no_pref_rr_counter{0}`.
- `src/pipeline/task_scheduler.cpp` — Inserted `_no_pref_rr_counter.store(0, std::memory_order_relaxed)` after the GPU drain loop in `prepare_for_query` (with explanatory comment on cache=table_gpu warm-path correctness), and added the `have_pref` tracking + `if (!have_pref && _gpu_executors.size() > 1)` round-robin block in `management_eventloop` (with comment cross-referencing `lock_or_prepare_batch` / `cucascade::convert_gpu_to_gpu`).

## Verbatim grep proof of the four key invariants

```text
$ grep -c '_no_pref_rr_counter' src/include/pipeline/task_scheduler.hpp
1
$ grep -c '_no_pref_rr_counter' src/pipeline/task_scheduler.cpp
2
$ grep -E 'std::map<int, std::unique_ptr<gpu_pipeline_executor>>' src/include/pipeline/task_scheduler.hpp
  std::map<int, std::unique_ptr<gpu_pipeline_executor>> _gpu_executors;
$ grep -E 'std::unordered_map<int, std::unique_ptr<gpu_pipeline_executor>>' src/include/pipeline/task_scheduler.hpp
(no match — exit 1)
$ grep -E '#include <map>'    src/include/pipeline/task_scheduler.hpp
#include <map>
$ grep -E '#include <atomic>' src/include/pipeline/task_scheduler.hpp
#include <atomic>
$ grep -E '#include <unordered_map>' src/include/pipeline/task_scheduler.hpp
(no match — exit 1)
$ grep -E 'have_pref' src/pipeline/task_scheduler.cpp
    bool have_pref       = false;
        have_pref        = true;
    if (!have_pref && _gpu_executors.size() > 1) {
$ grep -E '_no_pref_rr_counter\.store\(0, std::memory_order_relaxed\)' src/pipeline/task_scheduler.cpp
  _no_pref_rr_counter.store(0, std::memory_order_relaxed);
$ grep -E '_no_pref_rr_counter\.fetch_add\(1, std::memory_order_relaxed\)' src/pipeline/task_scheduler.cpp
      auto idx = _no_pref_rr_counter.fetch_add(1, std::memory_order_relaxed) %
```

## MCP build result

- Command: `mcp__project-commands__run_command name=build`
- **Exit code:** 0
- 40 / 40 ninja targets built (final step: `Linking CXX executable extension/sirius/test/cpp/sirius_unittest`)
- Only warnings emitted are pre-existing `SPDLOG_ACTIVE_LEVEL is overridden` (HYG, not regression).

## MCP smoke-test result

- Command: `mcp__project-commands__run_command name=unit-tests filter="physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache"`
- **Exit code:** 0
- **Duration:** 7.3 s
- **Assertions:** 178 in 1 test case (All tests passed)
- This is the cheap test that exercises the per-query counter reset (`prepare_for_query` -> `store(0)`); without that reset the second iteration of the table_gpu cache hits a different GPU than iteration 1 wrote to and the test fails with `0 == N`. PASS confirms the reset landed correctly.

## HYG baseline

- `grep -rn 'rmm::cuda_stream_default' src/ | wc -l` returns **40** (≤ 40 budget, unchanged from Phase 13-VALIDATION baseline).

## Branch state

- Branch: `feat/sched-rr-distribution`
- HEAD: `d4009e2 feat(14-01): land SCHED-RR distribution for preference-less pipeline tasks`
- Parent: `8c4600c docs(14): create Phase 14 plan for SCHED-RR distribution` (one commit above Phase 13 HEAD `833bb72`)

## Decisions Made

- **std::unordered_map -> std::map for _gpu_executors:** Determinism. `begin()->first` was hash-bucket-ordered, which is process-stable but configuration-stable in surprising ways. Map iteration is ordered by key — `device_id 0` always first.
- **Atomic round-robin counter (_no_pref_rr_counter):** The actual distribution mechanism. Without it, even with std::map, all preference-less tasks pile onto GPU 0 (begin()).
- **Per-query reset in prepare_for_query:** Required for `cache=table_gpu` correctness — cache entries are keyed by device_id, so iteration N+1 must dispatch the same preference-less task to the same GPU as iteration N. Verified by follow-up #17 scale-up smoke test.
- **Edit tool, not `git apply`:** The diff in 14-CONTEXT.md lines 19-79 is documentation-style without `--- a/...` / `+++ b/...` headers; `git apply` rejects. Phase 13 Waves 1 + 4 both established this constraint.
- **Branch base off current HEAD `8c4600c` (not `833bb72`):** Plan instructions said base off Phase 13 HEAD `833bb72`, but HEAD is one commit further at `8c4600c` (the `docs(14): create Phase 14 plan` commit that adds `.planning/phases/14-sched-rr-distribution/` files). Basing off `8c4600c` keeps the plan files on the branch and is git-equivalent to basing off `833bb72` plus a cherry-pick of the docs commit. No source-code drift introduced. (Tracked here as a deliberate small deviation from the literal plan instruction; rationale is that the plan files we are executing must remain accessible on the branch.)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Branch base point reinterpreted from `833bb72` to `8c4600c`**
- **Found during:** Task 1 Step 0 (branch base switch)
- **Issue:** Plan instruction said "Base off Phase 13's branch HEAD `833bb72`," but the working tree HEAD was `8c4600c docs(14): create Phase 14 plan for SCHED-RR distribution` — the docs commit that actually adds the `.planning/phases/14-sched-rr-distribution/` plan files we are executing. Basing strictly off `833bb72` would either lose the plan-execution context (no 14-PLAN/14-CONTEXT files on the branch) or require cherry-picking the docs commit. The cleanest path forward is to base off the current HEAD `8c4600c`, which is one docs-only commit ahead of `833bb72`.
- **Fix:** `git checkout -B feat/sched-rr-distribution` from current HEAD `8c4600c`, producing branch `feat/sched-rr-distribution @ d4009e2 -> 8c4600c -> 833bb72 (Phase 13 HEAD)`. The patch sits exactly one source-code commit above Phase 13 HEAD as the plan intended.
- **Files modified:** none (branch operation only)
- **Verification:** `git log --oneline | head -3` shows `d4009e2 feat(14-01) ... -> 8c4600c docs(14) ... -> 833bb72 test(13-04 ...)` — Phase 13's tip is the second source-code ancestor, with only one docs-only commit (`8c4600c`) between.
- **Committed in:** N/A (branch creation; no source-code change)

---

**Total deviations:** 1 auto-fixed (1 blocking — branch-base reinterpretation).
**Impact on plan:** Zero source-code drift from plan intent. The patch sits exactly one source-code commit above Phase 13 HEAD as planned; the docs(14) commit between is plan-files-only and cannot affect execution.

## Issues Encountered

None — plan executed cleanly. Pre-build sanity greps all passed first time, MCP build exit 0 first time, smoke test passed first time.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 14-02 (validation) ready to execute against commit `d4009e2`** without additional source edits. The acceptance-criteria validation (full `[mgpu]` suite, `[TPC-H][parquet]`, `[integration][TPC-H]`, TPC-H SF1 1-GPU vs 2-GPU benchmark on Q1/Q6/Q19) has all preconditions met:
  - Build clean (this plan).
  - HYG-02 baseline preserved (this plan).
  - Phase 13 Path-2 writer-event lineage already on branch ancestry (`833bb72`).
- No blockers known for 14-02.

## Self-Check: PASSED

- `.planning/phases/14-sched-rr-distribution/14-01-SUMMARY.md` exists on disk
- `git log --oneline --all | grep d4009e2` returns the SCHED-RR commit
- `src/include/pipeline/task_scheduler.hpp` exists and contains the patched declarations
- `src/pipeline/task_scheduler.cpp` exists and contains the patched logic
- All acceptance-criteria greps return PASS (see verbatim grep section above)
- MCP build exit 0; MCP follow-up #17 scale-up exit 0 (178 assertions / 7.3 s); HYG = 40 (≤ 40 budget)

---
*Phase: 14-sched-rr-distribution*
*Completed: 2026-04-30*
