---
phase: 12-small-sort-vector-rangecheck-fix
plan: 04
subsystem: validation
tags: [validation, mgpu, tpch, integration, ship-gate, mcp-discover]

# Dependency graph
requires:
  - phase: 12-small-sort-vector-rangecheck-fix
    provides: "12-02-SUMMARY.md (bound-fix at sirius_physical_hash_join.cpp:622-637) and 12-03-SUMMARY.md (regression TEST_CASE with stash-roundtrip empirical proof)"
provides:
  - "Phase 12 ship verdict (PASS) with verbatim MCP evidence per CONTEXT.md acceptance criterion"
  - "Catalog of post-12 [mgpu] suite state: 12 pass / 2 fail / 15 total — both failures classified Phase 14 distribution territory by failure shape (gpu0 pipelines=0 gpu1 pipelines=N)"
  - "TPC-H × 2-GPU integration regression baseline: 48/48 cases, 71608 assertions, post-12 unchanged"
affects: ["12-SUMMARY.md (phase-level rollup, if generated)", "ROADMAP.md (Phase 12 → Complete)", "STATE.md (current focus → Phase 13)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: validation-via-MCP-tag-filter — when no dedicated MCP command exists for a criterion's test scope (e.g. tpch-mgpu was hypothesized but not present), fall back to `unit-tests` filtered by Catch2 tag set ([integration][TPC-H], [mgpu-audit][TPC-H], [mgpu]). Discovered via `mcp__project-commands__list_commands` — never guess command names."
    - "Pattern: --abort-aware suite tally — `unit-tests` runs with `--abort` (halts on first failure). Recover full test counts by running aborted-after tests via separate filtered MCP invocations. The new test from a regression-gate plan should be classified by shape: if its name doesn't appear in the abort-truncated output, run it isolated to attribute correctly."

key-files:
  created:
    - ".planning/phases/12-small-sort-vector-rangecheck-fix/12-VALIDATION.md"
    - ".planning/phases/12-small-sort-vector-rangecheck-fix/12-04-SUMMARY.md"
  modified: []

key-decisions:
  - "TPC-H × 2-GPU MCP command name DISCOVERED (not guessed): no dedicated command exists. Used `unit-tests filter='[integration][TPC-H]'` (48 cases, exercises BOTH 1-GPU and 2-GPU variants of all 22 TPC-H queries × {DuckDB, parquet} via the [08-04] RUN_TPCH_MGPU GENERATE macro) PLUS `unit-tests filter='[mgpu-audit][TPC-H]'` (4 cases, explicit per-GPU dispatch audit on Q1 num_gpus=2). The two filters together cover the TPC-H × 2-GPU integration scope."
  - "Criterion 3 [mgpu] tally: 12 pass / 2 fail / 15 total. The `unit-tests --abort` stopped at index [11/15] when `physical_order - large sort distributes across two GPUs` failed; recovered remaining 3 test results via separate filtered MCP runs. Both failures share shape `gpu0 pipelines=0 gpu1 pipelines=N` — Phase 14 distribution territory, NOT Phase 12 regressions."
  - "Overall verdict: PASS — all 4 CONTEXT.md acceptance criteria evaluate PASS with real MCP evidence. No gap-closure plan needed. Recommended next: orchestrator marks Phase 12 complete in ROADMAP.md and advances STATE.md focus to Phase 13."
  - "Did NOT modify ROADMAP.md or STATE.md per Step 7 instruction ('orchestrator owns those'). Did NOT modify any source or test code per plan's <critical_constraints> ('only artifact created is 12-VALIDATION.md')."

patterns-established:
  - "Pattern: phase-12 'one-line-fix-with-invariant-comment' workflow now fully GATED through Wave 4 — Wave 1 GDB pin → Wave 2 minimal patch with INVARIANT comment → Wave 3 regression gate with stash-roundtrip empirical proof → Wave 4 ship-gate validation against verbatim CONTEXT.md acceptance criteria. Reusable phase shape for any focused libstdc++/cudf-API throw class."

requirements-completed: []

# Metrics
duration: ~10min
completed: 2026-04-29
tasks: 1
files: 2
---

# Phase 12 Plan 04: Phase 12 Ship-Gate Validation Summary

**Verified all four CONTEXT.md acceptance criteria PASS via real MCP test runs (no static analysis); emitted `12-VALIDATION.md` with per-criterion verbatim evidence and overall verdict PASS. The TPC-H × 2-GPU MCP command was DISCOVERED (not guessed) via `mcp__project-commands__list_commands` — no dedicated `tpch-mgpu`/`integration-tpch-mgpu` command exists, so the [integration][TPC-H] tag filter on `unit-tests` (48 cases × 71608 assertions) plus the [mgpu-audit][TPC-H] filter (4 cases × 64 assertions) was used together to cover the 2-GPU integration scope. Phase 12 ships clean; orchestrator advances to Phase 13.**

## Overall Verdict: PASS

| # | Criterion | Verdict | Headline Evidence |
|---|-----------|---------|-------------------|
| 1 | `physical_order - small sort stays single-GPU` passes | PASS | exit 0, 27 assertions, 5.2s |
| 2 | New regression test passes | PASS | exit 0, 19 assertions, 5.1s |
| 3 | [mgpu] suite ≥11 passing | PASS | 12/15 pass; both failures are Phase 14 distribution territory (`gpu0 pipelines=0` shape) |
| 4 | No regression in TPC-H × 2-GPU integration | PASS | 48/48 [integration][TPC-H] + 4/4 [mgpu-audit][TPC-H], 71672 total assertions |

## Per-Criterion Summary

- **Criterion 1 — Small-sort single-GPU test:** the explicit ship-gate test from Phase 12's goal. Previously failing with libstdc++ `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)`; now passes 27/27 assertions in 5.2s after the 12-02 bound-filter patch in `prepare_join_keys`.
- **Criterion 2 — Regression test:** `physical_order - small sort rangecheck regression` (added by 12-03 at `test/cpp/operator/test_physical_order_mgpu.cpp:124`) passes 19/19 assertions in 5.1s. Stash-roundtrip evidence in 12-03-SUMMARY proves the test reproduces the EXACT byte-identical pre-fix failure shape, confirming it is a real regression gate.
- **Criterion 3 — [mgpu] suite:** 12 pass / 2 fail / 15 total. Phase 12 advanced this from 10/14 (pre-12) → 12/15 (post-12): +1 from 12-02 fix, +1 from 12-03 new regression test. Both remaining failures (`physical_order - large sort distributes across two GPUs`, `physical_order - order by with limit over large input`) share the failure shape `gpu0 pipelines=0 gpu1 pipelines=N` and are explicitly Phase 14 SCHED-RR distribution territory per ROADMAP.md.
- **Criterion 4 — TPC-H × 2-GPU integration:** 48/48 [integration][TPC-H] cases pass with 71608 assertions (TPC-H Q1–Q22 × {DuckDB attach, parquet} × {num_gpus=1, num_gpus=2} per the [08-04] RUN_TPCH_MGPU macro), plus 4/4 [mgpu-audit][TPC-H] cases pass with 64 assertions. Zero new failures introduced by Phase 12.

## Discovered MCP command name for TPC-H × 2-GPU integration

**No dedicated `tpch-mgpu` / `integration-tpch-mgpu` / `tpch-2gpu` MCP command exists.** Discovered via `mcp__project-commands__list_commands`:

| Available command | Description |
|-------------------|-------------|
| `unit-tests` | Run C++ unit tests (abort on first failure). Pass `filter=` to restrict to a Catch2 test-spec. |
| `tpch-benchmark` | Run TPC-H benchmark and validate results. (Benchmark, not integration test.) |
| `tpch-parquet` | Run TPC-H queries against Parquet files. (Benchmark, not integration test.) |

The TPC-H × 2-GPU **integration** scope is exercised via `unit-tests` with Catch2 tag filters:
- `unit-tests filter="[integration][TPC-H]"` — 48 test cases including all 22 TPC-H queries × {DuckDB, parquet} × {1-GPU, 2-GPU} (parameterized via `RUN_TPCH_MGPU` GENERATE macro from Phase 08).
- `unit-tests filter="[mgpu-audit][TPC-H]"` — 4 test cases for explicit per-GPU dispatch audit.

Both ran to completion in this validation; both PASS.

## Recommended Next Action

**Ship Phase 12 — orchestrator should:**
1. Mark Phase 12 row Complete in ROADMAP.md (via `node $HOME/.claude/get-shit-done/bin/gsd-tools.cjs roadmap update-plan-progress 12`).
2. Update STATE.md current focus → Phase 13 (`13-q11-multi-gpu-illegal-address` — the v1.3 ship blocker per ROADMAP).
3. Open PR for `fix/order-small-sort-rangecheck` branch (4 source/test commits + 4 docs commits + this validation commit = 9 commits since `7b9af88`'s parent).

No gap-closure plan needed.

## Task Commits

1. **Task 1: Run all four CONTEXT.md acceptance criteria via MCP and record per-criterion verdict in 12-VALIDATION.md** — `3e7a3a1` (docs)

**Plan metadata commit:** TBD (final commit forthcoming, will include this SUMMARY + STATE.md update).

## Files Created/Modified

- **Created:** `.planning/phases/12-small-sort-vector-rangecheck-fix/12-VALIDATION.md` (247 lines, 4 criterion sections, verbatim MCP output for each).
- **Created:** `.planning/phases/12-small-sort-vector-rangecheck-fix/12-04-SUMMARY.md` (this file).
- **Source/test code:** None modified (per plan's `<critical_constraints>`).

## Decisions Made

- **TPC-H × 2-GPU command discovery:** Discovered via `mcp__project-commands__list_commands`; no dedicated command exists. Used `unit-tests` with two complementary tag filters (`[integration][TPC-H]` for the full 22-query × 2-flavor × 2-GPU-count matrix, `[mgpu-audit][TPC-H]` for the explicit per-GPU dispatch audit). Both ran cleanly.
- **Criterion 3 abort-recovery:** the `unit-tests` MCP command runs with `--abort`, halting at index [11/15] on the first failure (`physical_order - large sort distributes across two GPUs`). The 3 unrun tests were classified by separate filtered MCP runs: criteria 1 and 2 above already attribute 2 of them; the third (`physical_order - order by with limit over large input`) was run isolated and confirmed FAIL with the same `gpu0 pipelines=0` shape — Phase 14 territory.
- **Both Criterion 3 failures classified Phase 14 territory by SHAPE not by name:** the failure messages (`gpu0 pipelines=0 gpu1 pipelines={16,17}`) demonstrate that the scan-task distributor is placing ALL pipeline tasks on a single GPU rather than splitting cross-GPU. This is the exact bug class Phase 14 (`14-sched-rr-distribution`) was scoped to fix per ROADMAP.md ("Patch tested working in isolation, rolled back from session due to Phase 13 dependency"). Phase 12's bug class (libstdc++ `vector::_M_range_check` from `cudf::table_view::select`) is separate and is closed.
- **Did NOT modify ROADMAP.md or STATE.md** per the plan's Step 7 explicit instruction ("Do NOT modify ROADMAP.md or STATE.md in this plan — those updates belong to the orchestrator"). Those updates happen in the orchestrator's post-phase wrap-up.
- **Did NOT modify any source or test code** per the plan's `<critical_constraints>` ("This plan does NOT modify source or test code. The only artifact created is 12-VALIDATION.md"). Verified by `git diff --stat HEAD~1..HEAD` showing only the validation file added.

## Deviations from Plan

**None — plan executed exactly as written.**

The plan's Step 1 ("discover the TPC-H × 2-GPU MCP command name; do NOT guess") was followed: no dedicated command was found, and the appropriate fallback (Catch2 tag filter on `unit-tests`) was used and clearly documented in 12-VALIDATION.md "Criterion 4" section. The plan's `<acceptance_criteria>` row "Criterion 4's command name is a name discovered via `mcp__project-commands__list_commands` and is recorded verbatim (not 'TBD' or a guess)" is satisfied: 12-VALIDATION.md records the literal `unit-tests filter="[integration][TPC-H]"` and `unit-tests filter="[mgpu-audit][TPC-H]"` invocations, both of which are MCP commands discovered via `list_commands` (the `unit-tests` command + the literal Catch2 tags from the test source).

Note: the plan-supplied `git add` commit step needed `-f` flag because `.planning/` is in `.gitignore`. This is a Rule 3 auto-fix (blocking — plan recipe assumed non-ignored path); resolved inline.

## Issues Encountered

- **MCP `unit-tests` runs with `--abort`:** halts on first failure. To recover full counts for Criterion 3 (which has 2 expected failures), filtered each remaining test individually. Acceptable workaround per Phase 8 [08-05] "MCP daemon caches commands.yaml at session start; hot-reload unsupported. unit-tests cannot be invoked with --abortx 999 or tag filter from this agent" — same constraint applies here. The chosen approach (separate filtered runs for the 3 unrun tests) covers all 15 [mgpu] cases.
- **`.planning/` is gitignored:** required `git add -f` for VALIDATION.md commit. No content concern.

## User Setup Required

None — all MCP commands ran cleanly under the existing 2-GPU integration env (2 × RTX 6000 Ada visible). The only env-gated tests (SF10 Q1/Q6/Q12 2-GPU variants requiring `SIRIUS_TEST_SF10_PATH`) were intentionally skipped per the [08-05] policy and counted as PASS by Catch2 since they exit cleanly via WARN+return.

## Next Phase Readiness

**Phase 12 ships clean. Orchestrator can now:**
- Mark Phase 12 row Complete in ROADMAP.md.
- Update STATE.md current focus → Phase 13 (`13-q11-multi-gpu-illegal-address` — v1.3 ship blocker).
- Open PR on `fix/order-small-sort-rangecheck` branch.

**Phase 14 (SCHED-RR distribution) is the natural follow-on** — the 2 remaining [mgpu] failures (`large sort distributes`, `order by with limit over large input`) are exactly the test set Phase 14 was scoped to address. Phase 14 remains blocked by Phase 13 (Q11 hang in batch) per ROADMAP.

**Phase 13 is the v1.3 ship blocker** and is independent of Phase 12 (per CONTEXT.md "This phase is INDEPENDENT of Phase 13").

## Self-Check: PASSED

- `.planning/phases/12-small-sort-vector-rangecheck-fix/12-VALIDATION.md`: FOUND (247 lines, 4 criterion sections)
- `.planning/phases/12-small-sort-vector-rangecheck-fix/12-04-SUMMARY.md`: FOUND (this file)
- Task 1 commit `3e7a3a1`: FOUND in `git log --oneline`
- All 8 plan-specified verification checks pass:
  - file exists and non-empty: PASS
  - ≥80 lines: PASS (247 lines)
  - exactly 4 `## Criterion ` sections: PASS
  - Overall verdict matches `^Overall: \*\*(PASS|PARTIAL|FAIL)\*\*$`: PASS
  - alternation placeholder absent: PASS
  - ≥4 `Verdict:` lines populated: PASS (exactly 4)
  - small-sort test name present: PASS
  - regression test name present: PASS

---
*Phase: 12-small-sort-vector-rangecheck-fix*
*Completed: 2026-04-29*
