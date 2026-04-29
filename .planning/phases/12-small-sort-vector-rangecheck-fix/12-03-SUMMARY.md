---
phase: 12-small-sort-vector-rangecheck-fix
plan: 03
subsystem: test/operator-mgpu
tags: [regression-test, mgpu, order, hash_join, _M_range_check, stash-roundtrip, libstdcxx]

# Dependency graph
requires:
  - phase: 12-small-sort-vector-rangecheck-fix
    provides: "12-02-SUMMARY.md — bound-checked key_col_indices in prepare_join_keys (no-cast fast path) at src/op/sirius_physical_hash_join.cpp:622-637"
provides:
  - "Regression TEST_CASE 'physical_order - small sort rangecheck regression' in test/cpp/operator/test_physical_order_mgpu.cpp pinning the small-sort _M_range_check off-by-one shut"
  - "Empirical proof (via git checkout-and-stash round-trip on src/op/sirius_physical_hash_join.cpp) that the new test reproduces the EXACT 12-stack-trace.txt failure shape `__n (which is 2) >= this->size() (which is 2)` on a pre-12-02 tree"
  - "Tagged with [regression] so future `--filter [regression]` invocations hit the Phase 12 gate collectively"
affects: ["12-04-PLAN.md (if planned)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: pre-commit shape-sanity gate via git checkout-then-stash round-trip — for a regression test that gates a fixed bug, prove it actually exercises the bug shape by (1) `git checkout <fix-commit>^ -- <fixed-file>` to restore pre-fix state in the working tree, (2) build, (3) run only the new test, (4) inspect stdout for the same libstdc++ message and small-N vector shape, (5) `git checkout HEAD -- <fixed-file>` to restore the fix, (6) rebuild + re-run to confirm post-restore PASS. Validates the test is a real regression gate rather than an incidental correctness test near the bug site."

key-files:
  created:
    - ".planning/phases/12-small-sort-vector-rangecheck-fix/12-03-SUMMARY.md"
  modified:
    - "test/cpp/operator/test_physical_order_mgpu.cpp"

key-decisions:
  - "Tuned constants used (final, no iteration needed): kNumFiles=4, kRowsPerFile=256, hash_partition_bytes=1024. The very first parameter combination from the plan's <action> block reproduced the EXACT verbatim stashed-tree failure message `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` — same shape, same N=2, same vector size as 12-stack-trace.txt's primary catchpoint. No tuning iterations consumed."
  - "Used `git checkout 289d6d2^ -- <file>` followed by `git stash push -- <file>` rather than a direct `git stash` of working-tree-clean state. Reason: the 12-02 patch is committed (commit 289d6d2), so the working tree was already clean and `git stash push` had nothing to save. The checkout-then-stash pattern produces an equivalent stashed pre-fix state that the working tree can hold while the test runs, then `git checkout HEAD -- <file>` restores the fix and `git stash drop` removes the now-redundant stash entry."
  - "Did NOT add any helper to mgpu_test_utils.hpp (verified: `git diff test/cpp/operator/mgpu_test_utils.hpp` is empty). The new TEST_CASE composes scoped_mgpu_env + scoped_log_dir + require_gpu_matches_cpu + generate_parquet_surface + the file-local make_params() / make_tmp_dir() helpers exactly like the 3 existing TEST_CASEs, satisfying the 'no new test util' constraint from the plan."
  - "Diff scope: 47 added lines in test_physical_order_mgpu.cpp (one new TEST_CASE: 4-line comment block + 42-line body). Zero source-code changes outside the test file. HEAD's hash_join.cpp unchanged from post-12-02 state (verified by INVARIANT comment grep)."

patterns-established:
  - "Pattern: phase-12 'one-line-fix-with-invariant-comment' workflow now COMPLETE through Wave 3 — Wave 1 GDB pin (12-stack-trace.txt) → Wave 2 minimal patch with INVARIANT comment (289d6d2) → Wave 3 regression gate with stash-roundtrip empirical proof. The Wave-3 stash-roundtrip step is what differentiates a 'happens-to-be-near-the-bug' test from a real regression gate."

requirements-completed: []

# Metrics
duration: ~3min
completed: 2026-04-29
tasks: 1
files: 1
---

# Phase 12 Plan 03: Add Small-Sort Rangecheck Regression Test Summary

**Added a focused regression TEST_CASE `physical_order - small sort rangecheck regression` (test/cpp/operator/test_physical_order_mgpu.cpp:120-165) that exercises the smallest-data ORDER BY + LIMIT path which previously triggered libstdc++ vector::_M_range_check inside `sirius::op::prepare_join_keys`. Empirically confirmed via git checkout-and-stash round-trip on the 12-02-patched file (`src/op/sirius_physical_hash_join.cpp`) that the new test FAILS on a pre-12-02 tree with the EXACT verbatim 12-stack-trace.txt message `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` — same N=2, same vector size — and PASSES once the patch is restored. Tuned values (final, first try): kNumFiles=4, kRowsPerFile=256, hash_partition_bytes=1024.**

## Performance

- **Duration:** ~3 min wall-clock (plan start `2026-04-29T19:27:08Z` → final commit; ~180s incl. 2 build cycles + 4 MCP test runs)
- **Started:** 2026-04-29T19:27:08Z
- **Completed:** 2026-04-29
- **Tasks:** 1 (one TEST_CASE addition)
- **Files modified:** 1 (`test/cpp/operator/test_physical_order_mgpu.cpp`, +47 lines)
- **Files created:** 1 (this SUMMARY)
- **Stash-roundtrip iterations consumed:** 1 of the 3 allowed (first parameter combination produced the exact target shape)

## Accomplishments

- Added the new TEST_CASE at line 124 (TEST_CASE statement) / lines 120-165 inclusive (comment + body) of `test/cpp/operator/test_physical_order_mgpu.cpp`.
- Test name string is exactly `"physical_order - small sort rangecheck regression"` (verifiable by `grep -F` on the file).
- Tags include all five required: `[mgpu]`, `[operator-mgpu]`, `[order]`, `[gpu_execution]`, `[regression]`.
- Test name is compiled into the unittest binary, verified via `strings build/release/extension/sirius/test/cpp/sirius_unittest | grep -F "small sort rangecheck regression"` (the `--list-test-names-only` form aborts under bare-shell driver isolation per 12-stack-trace.txt's note; `strings` provides the equivalent compile-time evidence).
- Test reuses `scoped_mgpu_env`, `scoped_log_dir`, `require_gpu_matches_cpu`, and `generate_parquet_surface` from `mgpu_test_utils.hpp` — no new helpers added.
- `params.hash_partition_bytes = 1024` (`< 1'000'000` default in `make_params()`) is set inline in the new TEST_CASE, satisfying the plan acceptance-criterion grep requirement.
- New test PASSES on patched (post-12-02) tree via MCP unit-tests: **exit 0, 19 assertions, 5.2s**.
- Original `physical_order - small sort stays single-GPU` test STILL passes via MCP unit-tests: **exit 0, 27 assertions, 5.2s** — 12-02 fix not regressed by the test addition.

## Step 6 — Stash Round-Trip Evidence (mandatory shape-sanity gate)

The plan requires empirical proof that the new test actually exercises the same off-by-one shape as the original failure. Performed the round-trip:

### Round-trip steps and observations

1. **Identified patched file from 12-02-SUMMARY.md `key-files.modified`:** `src/op/sirius_physical_hash_join.cpp`.
2. **Reverted to pre-12-02 state in working tree:** `git checkout 289d6d2^ -- src/op/sirius_physical_hash_join.cpp` (restores the file as it was BEFORE the 12-02 fix commit). Confirmed via `grep -nE 'INVARIANT:'` that the comment block is absent and via `sed -n '617,627p'` that line 623 is the original `result.keys = table.select(key_col_indices);` (unchecked) call.
3. **Stashed the pre-fix state:** `git stash push -m "phase-12-03-shape-check" -- src/op/sirius_physical_hash_join.cpp`. Stash created at `stash@{0}`. Re-applied pre-fix state to working tree afterward (`git checkout 289d6d2^ -- ...` again) so the build sees pre-fix code while the stash holds an equivalent copy for cleanup parity with the plan's recipe.
4. **Built via MCP:** `mcp__project-commands__run_command name=build` — **exit 0, 1.4s**, recompiled `sirius_physical_hash_join.cpp` and relinked the unittest binary.
5. **Ran the new test only via MCP:** `mcp__project-commands__run_command name=unit-tests filter="physical_order - small sort rangecheck regression"` — **exit 1, 5.0s**.

### Verbatim stashed-tree failure message

Captured from MCP stdout:

```
/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/operator/mgpu_test_utils.hpp:343: FAILED:
  REQUIRE_FALSE( gpu_result->HasError() )
with expansion:
  !true
with message:
  gpu_execution error: Invalid Error: SiriusExecuteQuery error: Invalid Error:
  vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)


[1/1] (100%): physical_order - small sort rangecheck regression
===============================================================================
test cases:  1 |  0 passed | 1 failed
assertions: 15 | 14 passed | 1 failed
```

The failure message `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` is BYTE-IDENTICAL to the libstdc++ string captured in `12-stack-trace.txt` at frame #1 (`__s = "vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)"`). N=2, vector size=2 — the same off-by-one shape.

### Restore and re-verify

6. **Restored the 12-02 patch:** `git checkout HEAD -- src/op/sirius_physical_hash_join.cpp`. Confirmed via `grep -nE 'INVARIANT:'` that the comment is back at line 623.
7. **Dropped the redundant stash:** `git stash drop stash@{0}` — clean stash list.
8. **Rebuilt via MCP:** **exit 0, 1.4s**.
9. **Re-ran new test via MCP:** **exit 0, 19 assertions, 5.1s** — post-restore PASS confirmed.
10. **Spot-checked original test still passes via MCP:** `physical_order - small sort stays single-GPU` — **exit 0, 27 assertions, 5.2s** — 12-02 fix intact.

### Final tuned parameters

| Constant                       | Value | Rationale                                                                                                  |
| ------------------------------ | ----- | ---------------------------------------------------------------------------------------------------------- |
| `kNumFiles`                    | 4     | Smaller than the original 8-file failing surface; tested first per plan.                                  |
| `kRowsPerFile`                 | 256   | ~4× smaller than the original 1000-row surface; combined w/ hash_partition_bytes=1024 forces multi-partition. |
| `params.hash_partition_bytes`  | 1024  | Aggressively small partition threshold forces the partitioner to produce multiple partitions on tiny input, exposing the 2-element index shape. |
| Iterations consumed            | 1/3   | First combination produced the exact target shape; no further tuning needed.                              |

## Verification (plan acceptance criteria)

| Criterion | Result | Evidence |
|---|---|---|
| New TEST_CASE name string is exactly `"physical_order - small sort rangecheck regression"` | PASS | `grep -F "physical_order - small sort rangecheck regression" test/cpp/operator/test_physical_order_mgpu.cpp` finds it at line 124 |
| Tags include `[mgpu]`, `[operator-mgpu]`, `[order]`, `[gpu_execution]`, `[regression]` | PASS | All five tags present on line 125 |
| TEST_CASE calls `require_gpu_matches_cpu` (asserts correctness) | PASS | Line 159 |
| `params.hash_partition_bytes` strictly less than default 1'000'000 | PASS | Line 144 sets `params.hash_partition_bytes = 1024` |
| Test name appears in unittest binary | PASS | `strings build/release/extension/sirius/test/cpp/sirius_unittest \| grep -F` returns the literal string (bare-shell `--list-test-names-only` aborts on driver isolation per 12-stack-trace.txt; `strings` is the equivalent compile-time evidence) |
| MCP unit-tests run exits 0 with `assertions: <N> passed` and 0 failures | PASS | Exit 0, 19 assertions, 5.2s |
| Original `physical_order - small sort stays single-GPU` STILL exits 0 | PASS | Exit 0, 27 assertions, 5.2s |
| Stash round-trip performed and documented | PASS | This section above; verbatim message recorded |
| `git diff test/cpp/operator/mgpu_test_utils.hpp` is empty (no new helpers) | PASS | `git diff --stat` reports only `test_physical_order_mgpu.cpp` modified |

## Task Commits

1. **Task 1: Add small-sort rangecheck regression TEST_CASE in test_physical_order_mgpu.cpp** — `163d622` (test)

**Plan metadata commit:** TBD (final commit forthcoming, will include this SUMMARY + STATE.md update).

## Files Created/Modified

- **Modified:** `test/cpp/operator/test_physical_order_mgpu.cpp` — +47 lines, one new TEST_CASE at lines 120-165.
- **Created:** `.planning/phases/12-small-sort-vector-rangecheck-fix/12-03-SUMMARY.md` (this file).

## Decisions Made

- **First-iteration parameter combination accepted (no tuning):** kNumFiles=4 / kRowsPerFile=256 / hash_partition_bytes=1024 from the plan's `<action>` block reproduced the verbatim stashed-tree failure message on iteration 1. The plan permitted up to 3 tuning iterations; only 1 was needed.
- **Used `git checkout` + `git stash` together rather than pure `git stash`:** the 12-02 patch is COMMITTED (commit 289d6d2), not a working-tree change, so a bare `git stash push -- <file>` had nothing to save. Plan recipe assumed the working tree carried the patch. Resolved by restoring pre-fix state via `git checkout 289d6d2^ -- <file>`, which produced the equivalent semantic state. Documented as a deviation note (non-blocking — the recipe outcome is identical).
- **Did NOT touch `mgpu_test_utils.hpp`:** plan acceptance criterion explicitly requires zero diff there. Verified empty diff after Task 1 commit.
- **Did NOT widen the test to cover the slow-path (cast_necessary=true) branch:** the new test, like the patch in 12-02, only exercises the no-cast fast path (the failing test query has no casts). Per the plan's "minimal patch / minimal test" framing and 12-02's scope-boundary note, slow-path bound-check coverage is deferred to a future plan if/when that path surfaces.

## Deviations from Plan

**Recipe-shape deviation (Step 6 stash command):** the plan's literal recipe `git stash push -m "..." -- "$PATCHED_FILE"` was authored assuming the 12-02 patch lived in the working tree (uncommitted). In reality 12-02 is a committed change (commit 289d6d2), so a direct `git stash push` had nothing to stash. The deviation was to substitute `git checkout 289d6d2^ -- src/op/sirius_physical_hash_join.cpp` (which produces the equivalent pre-fix working-tree state) and then either stash + re-checkout, or simply `git checkout HEAD -- ...` to restore. The semantic outcome — pre-fix state for the build/test, then restored fix — is identical to the plan's intent. No iterations were lost; the round-trip succeeded on the first try.

This is a Rule 3 auto-fix (blocking issue: plan recipe assumed wrong git state). Fixed inline; documented here.

No other deviations. The new test's first parameter combination produced the target failure shape; no Rule 4 escalation was needed.

## Issues Encountered

- **Bare-shell `--list-test-names-only` aborts under sandbox:** invoking the unittest binary directly fails with `cucascade::topology_discovery reported 0 GPUs` (NVML driver isolation, same as 12-stack-trace.txt notes for bare-shell GDB). Worked around by using `strings <binary> | grep -F "..."` to confirm the test name compiled into the binary; this is sufficient evidence per the plan's acceptance criterion (the criterion was that the name appears in the binary's listing — both `--list-test-names-only` and `strings` would emit the same literal string from `.rodata`).

## User Setup Required

None — the new test runs under the existing 2-GPU integration env via MCP. No env vars, secrets, or external setup needed.

## Next Phase Readiness

- **Phase 12 ship-criteria, plan-by-plan:**
  - 12-01: pin fix-site → DONE (12-stack-trace.txt).
  - 12-02: apply minimal patch with INVARIANT comment → DONE (commit 289d6d2).
  - 12-03 (this plan): regression gate with shape-sanity proof → DONE.
  - 12-04 (if planned): potential follow-on for slow-path mirror or upstream planner walk-back. Optional per 12-02 scope-boundary decision.
- **CONTEXT.md acceptance criterion 2 ('New regression test passes (smallest-data ORDER BY + LIMIT path, prevents reappearance)') is now satisfied with empirical proof of regression-gate validity.**
- **Phase 14 (SCHED-RR distribution) is unblocked w.r.t. Phase 12** (Phase 12's bug class is closed and gated). Phase 13 (Q11 multi-GPU illegal-address) remains the v1.3 ship blocker.

## Self-Check: PASSED

- `test/cpp/operator/test_physical_order_mgpu.cpp`: FOUND
- `.planning/phases/12-small-sort-vector-rangecheck-fix/12-03-SUMMARY.md`: FOUND (this file)
- Task 1 commit `163d622`: FOUND in `git log --oneline --all`
- Test name `physical_order - small sort rangecheck regression`: present in source (line 124) and in `.rodata` of unittest binary (`strings | grep -F` hit)
- `git diff test/cpp/operator/mgpu_test_utils.hpp`: 0 lines (no helper edits)
- `src/op/sirius_physical_hash_join.cpp` INVARIANT comment at line 623: present (post-Step-6 restore intact)
- All 9 plan acceptance-criteria rows: PASS
- Stash round-trip empirical proof: verbatim `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` failure on pre-fix tree, PASS on post-restore tree

---
*Phase: 12-small-sort-vector-rangecheck-fix*
*Completed: 2026-04-29*
