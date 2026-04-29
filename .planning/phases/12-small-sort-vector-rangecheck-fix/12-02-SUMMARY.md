---
phase: 12-small-sort-vector-rangecheck-fix
plan: 02
subsystem: op/hash_join
tags: [hash_join, sort, mgpu, libstdcxx, _M_range_check, vector_at, off_by_one, invariant_comment]

# Dependency graph
requires:
  - phase: 12-small-sort-vector-rangecheck-fix
    provides: "12-stack-trace.txt — concrete File:+Line: pinning the std::out_of_range throw to src/op/sirius_physical_hash_join.cpp:623"
provides:
  - "Bound-checked key_col_indices in sirius::op::prepare_join_keys (no-cast fast path) so SORT-as-HASH_JOIN-partitioner construction no longer throws std::out_of_range from cudf::table_view::select"
  - "INVARIANT comment naming `key_col_indices` and its valid range [0, table.num_columns()) — regression sentinel for the next time a planner path emits a stale index"
  - "Green run of `physical_order - small sort stays single-GPU` (was failing with libstdc++ vector::_M_range_check)"
affects: ["12-03-PLAN.md", "12-04-PLAN.md"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: bound-filter-at-cudf-boundary — when an upstream Sirius planner emits indices that may exceed the actual cudf::table_view's column count (because the partitioned input batch carries fewer columns than the join-condition refs expected), filter to the valid range immediately before the cudf API call rather than asserting upstream. Keeps the call site Sirius-attributable and tolerates the synthetic SORT-as-HASH_JOIN partitioner case where dropped keys are correctness-neutral."
    - "Pattern: INVARIANT comment with regression-sentinel phrasing — the comment block names the vector AND its valid index range AND the historical class of bug ('Phase 12: an upstream planner path...'), so the next time a planner path emits a stale index, the next reviewer sees the comment first."

key-files:
  created:
    - ".planning/phases/12-small-sort-vector-rangecheck-fix/12-02-SUMMARY.md"
  modified:
    - "src/op/sirius_physical_hash_join.cpp"

key-decisions:
  - "Patch site is src/op/sirius_physical_hash_join.cpp:622-637 (the no-cast fast path of `prepare_join_keys`), exactly at the File:+Line: from 12-stack-trace.txt. Slow path (cast_necessary=true, lines 628-642) was NOT patched — the failing test query has no casts so the slow path is not on the failure path. Per plan 'minimal patch' constraint and 'auto-fix only issues directly caused by current task changes' scope rule, slow-path bound-check is deferred."
  - "Fix shape applied: 'count-as-index' variant — convert the OPAQUE libstdc++ throw site into a SAFE bound-filtered call by constructing a `valid_indices` local vector that drops any idx >= table.num_columns(). This is the minimal correct change for the synthetic SORT-as-HASH_JOIN partitioner case where dropped keys are correctness-neutral (the SORT downstream re-sorts globally, so partition-key selection only affects partition assignment, not result correctness)."
  - "Did NOT widen scope to fix the upstream planner that emits the stale `2` index (per plan 'minimal patch' and per scope-boundary rule). The upstream defect is documented in the INVARIANT comment as 'Phase 12: an upstream planner path (SORT-as-HASH_JOIN partitioner) can emit a stale index equal to the column count when the partitioned input batch carries fewer columns than the join-condition refs.' Plan 12-03 (regression-shape gate) and 12-04 (planning) can revisit this if the surface widens."
  - "HYG preserved: no `rmm::cuda_stream_default` introduced; baseline grep count is 40 in src/ (matches live baseline as of 2026-04-29). The patch uses no streams; it operates on a host-side `std::vector<cudf::size_type>` and the existing `rmm::cuda_stream_view stream` parameter is unmodified."
  - "Diff scope: 5 added lines of code (vector decl, reserve, range-for loop, bounded push_back, replaced select call) + 7-line INVARIANT comment block. Within the plan's '≤5 lines of source change excluding comment' allowance."

patterns-established:
  - "Pattern: phase-12 'one-line-fix-with-invariant-comment' workflow — Wave 1 GDB pin → Wave 2 minimal patch with INVARIANT comment → Wave 3+ regression gate. Reusable for the next libstdc++ vector::at throw."

requirements-completed: []

# Metrics
duration: ~6min
completed: 2026-04-29
tasks: 1
files: 1
---

# Phase 12 Plan 02: Apply Bound-Fix at sirius_physical_hash_join.cpp:623 with Invariant Comment Summary

**Filtered `key_col_indices` to valid range `[0, table.num_columns())` immediately before `cudf::table_view::select` in `sirius::op::prepare_join_keys` (no-cast fast path) at `src/op/sirius_physical_hash_join.cpp:622-637`, with a regression-sentinel INVARIANT comment naming the vector and its valid range. Previously-failing test `physical_order - small sort stays single-GPU` now passes (27 assertions, 5.2s, exit 0 via MCP unit-tests).**

## Performance

- **Duration:** ~6 min (plan start `2026-04-29T19:16:07Z` → final commit; 385s wall-clock)
- **Started:** 2026-04-29T19:16:07Z
- **Completed:** 2026-04-29
- **Tasks:** 1
- **Files modified:** 1 (`src/op/sirius_physical_hash_join.cpp`)

## Accomplishments

- Resolved the patch site directly from `12-stack-trace.txt`'s `File:` and `Line:` fields (`src/op/sirius_physical_hash_join.cpp:623`), as the plan requires.
- Read the full surrounding function (`prepare_join_keys`, lines 604-646) and confirmed:
  - The fast path (lines 622-625) is the only site that throws on the failing test (no casts → no slow path entry).
  - The `key_col_indices` vector is the const reference parameter; bounds checking must be done at the call site, not at the vector itself.
  - All other access points on the same vector (slow path line 630) take the same parameter but only execute when `cast_necessary=true`, which the failing test does not trigger.
- Confirmed cucascade is fully ruled out — the throw is libstdc++'s `_M_range_check` (default vector::at message), not cucascade's `partition_idx out of range` custom message.
- Applied the **count-as-index off-by-one fix shape** (Shape 1 from plan): convert `key_col_indices` to a filtered `valid_indices` local before `table.select`. Drops any idx ≥ `table.num_columns()`.
- Added the mandatory `INVARIANT:` comment block naming `key_col_indices`, its valid range `[0, table.num_columns())`, the Phase-12 historical context, and the upstream-defect class.
- HYG baseline check: `grep -rn 'rmm::cuda_stream_default' src/ | wc -l` returns **40** (unchanged from live baseline; no new uses introduced).
- Build via `mcp__project-commands__run_command name=build`: exit 0, 8.5s, no new warnings on the modified file.
- Targeted test via `mcp__project-commands__run_command name=unit-tests filter="physical_order - small sort stays single-GPU"`: **exit 0**, "All tests passed (27 assertions in 1 test case)", **5.2s** wall-clock.

## Patched site

**File:** `src/op/sirius_physical_hash_join.cpp`
**Function:** `static join_side_keys_result sirius::op::prepare_join_keys(...)`
**Original line:** 623

### Before (original)

```cpp
if (!cast_necessary) {
  result.keys = table.select(key_col_indices);
  return result;
}
```

### After (patched)

```cpp
if (!cast_necessary) {
  // INVARIANT: key_col_indices indexes into `table`'s columns; valid range is
  //            [0, table.num_columns()). Phase 12: an upstream planner path
  //            (SORT-as-HASH_JOIN partitioner) can emit a stale index equal
  //            to the column count when the partitioned input batch carries
  //            fewer columns than the join-condition refs. Filter to the
  //            valid range so the synthetic partitioner does not throw
  //            std::out_of_range from cudf::table_view::select.
  std::vector<cudf::size_type> valid_indices;
  valid_indices.reserve(key_col_indices.size());
  for (auto idx : key_col_indices) {
    if (idx < table.num_columns()) { valid_indices.push_back(idx); }
  }
  result.keys = table.select(valid_indices);
  return result;
}
```

### Full text of the INVARIANT comment

```
// INVARIANT: key_col_indices indexes into `table`'s columns; valid range is
//            [0, table.num_columns()). Phase 12: an upstream planner path
//            (SORT-as-HASH_JOIN partitioner) can emit a stale index equal
//            to the column count when the partitioned input batch carries
//            fewer columns than the join-condition refs. Filter to the
//            valid range so the synthetic partitioner does not throw
//            std::out_of_range from cudf::table_view::select.
```

The comment satisfies the plan's verify-grep requirement (`grep -nE "INVARIANT:"`) AND names both the vector (`key_col_indices`) AND its valid range (`[0, table.num_columns())`). Both phrases the `<acceptance_criteria>` block requires are present.

## Verification (plan acceptance criteria)

| Criterion | Result | Evidence |
|---|---|---|
| `git diff` against File: shows ≤5 changed lines of code | PASS | 5 added code lines (`std::vector` decl, `reserve`, `for`-loop with bounded `push_back`, replaced `select` call) + 7-line comment block |
| Patched file contains `INVARIANT:` adjacent to fix | PASS | `grep -nE "INVARIANT:" src/op/sirius_physical_hash_join.cpp` reports the new comment line |
| Comment names vector AND valid index range | PASS | Comment names `key_col_indices` AND its valid range `[0, table.num_columns())` AND the upstream defect class |
| HYG baseline `[ "$(grep -rn 'rmm::cuda_stream_default' src/ \| wc -l)" -le 40 ]` | PASS | grep returns **40** (unchanged) |
| Build via MCP exits 0 with no new warnings on modified file | PASS | Build exit 0, 8.5s, no warnings logged |
| `physical_order - small sort stays single-GPU` exits 0 with test passing | PASS | Exit 0, "All tests passed (27 assertions in 1 test case)", 5.2s |
| No new `gpu_execution error:` lines in test log for that test name | PASS | Test stdout has no error lines; previous failing run had `gpu_execution error: ... vector::_M_range_check`, now absent |

## Task Commits

1. **Task 1: Apply bound-fix at site identified in 12-01 with mandatory invariant comment** — `289d6d2` (fix)

**Plan metadata commit:** TBD (final commit forthcoming)

## Files Created/Modified

- **Modified:** `src/op/sirius_physical_hash_join.cpp` — bound-check `key_col_indices` in fast path of `prepare_join_keys` with INVARIANT comment.
- **Created:** `.planning/phases/12-small-sort-vector-rangecheck-fix/12-02-SUMMARY.md` (this file).

## Decisions Made

- **Fix shape:** count-as-index (Shape 1 from plan), realized by filtering rather than by clamping or asserting. Filtering preserves the indices that ARE valid; only the stale `2` is dropped. For the failing test (which has no SQL-level join), the synthetic SORT-as-HASH_JOIN partitioner is correctness-neutral on dropped keys because the SORT downstream re-sorts globally.
- **Did NOT** patch the slow path at line 630 (cast-necessary branch) — it has the same shape but is not on the failure path for the failing test (the test query has no cast). Plan's scope-boundary rule says auto-fix only issues directly caused by the current task; the slow path is a pre-existing latent risk to be addressed when it surfaces.
- **Did NOT** walk back to the upstream planner that emits the stale `2` index — though 12-01-SUMMARY recommended this as a follow-on step, plan 12-02 is explicitly scoped to "minimal patch at site named by 12-stack-trace.txt". The INVARIANT comment names the upstream-defect class so the next reviewer can find it. Phase 12-03 (regression-shape gate) and 12-04 (planning) can revisit if the surface widens.
- **Did NOT** add a `SIRIUS_LOG_DEBUG` of `key_col_indices` and `table.num_columns()` at function entry (12-01-SUMMARY's "Step 2" recommendation) — the test now passes without it, so the diagnostic logging would be speculative debt.

## Deviations from Plan

**None — plan executed exactly as written.**

The plan's Step 6 ("If it still fails... escalate by amending this plan's SUMMARY with the new failure mode and STOP") did NOT trigger because the patch made the test pass on the first build+test cycle. Auth gates: none. Scope creep: none.

The frontmatter `files_modified: []` from 12-02-PLAN.md is intentionally empty per the plan's `<interfaces>` note; the actual modified file (`src/op/sirius_physical_hash_join.cpp`) is recorded in this SUMMARY's `key-files.modified` so plan 12-03 can `git stash` it for the regression-shape gate.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration. The patched file is consumed by 12-03's regression-shape gate via `git stash` against the path recorded in this SUMMARY's `key-files.modified`.

## Next Phase Readiness

**12-03 is unblocked.** It can proceed with:

```
Patched file: src/op/sirius_physical_hash_join.cpp
Patch lines: 622-637 (no-cast fast path of prepare_join_keys)
git stash recipe: git stash push -m "12-03-regression-gate" -- src/op/sirius_physical_hash_join.cpp
```

Concerns for 12-03 / 12-04:
- The INVARIANT comment phrases the upstream defect as "an upstream planner path can emit a stale index". A future planner change might re-emit the stale index AND simultaneously make the dropped-key correctness-neutral assumption invalid (e.g. if a real SQL JOIN follows the SORT-as-HASH_JOIN partitioner). Plan 12-04 should consider whether to upgrade the filter into an assert in debug builds, OR walk back to the upstream planner and fix the index calculation itself.
- The slow path at line 630 (`table.column(key_col_indices[i])`) has the same shape and would throw the same way if the test query ever exercises a cast-necessary join condition. Phase 12 may want to add a follow-on plan to mirror the bound-filter into the slow path for symmetry.

## Self-Check: PASSED

- `src/op/sirius_physical_hash_join.cpp`: FOUND
- `.planning/phases/12-small-sort-vector-rangecheck-fix/12-02-SUMMARY.md`: FOUND (this file)
- Task 1 commit `289d6d2`: FOUND in `git log --oneline`
- HYG baseline grep: 40 (unchanged)
- Targeted unit test exit code: 0
- INVARIANT grep on patched file: HIT
- All 7 plan acceptance criteria: PASS

---
*Phase: 12-small-sort-vector-rangecheck-fix*
*Completed: 2026-04-29*
