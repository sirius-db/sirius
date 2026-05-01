---
phase: 15-mgpu-operator-colocation-audit
plan: 01
subsystem: testing
tags: [mgpu, sched-rr, audit, operator-colocation, comments-only]

# Dependency graph
requires:
  - phase: 14-sched-rr-distribution
    provides: SCHED-RR preference-less round-robin source-pipeline distribution; deterministic std::map _gpu_executors; per-query reset of _no_pref_rr_counter; empirical PASS evidence ([mgpu] 12/13, [TPC-H][parquet] 22/22, [integration][TPC-H] 48/48) corroborating that operators downstream of prepare_for_processing already honor cross-GPU colocation.
provides:
  - "INVARIANT (SCHED-RR contract) comments at all 11 audited operator sites that read batches[0]->get_memory_space() — anchors the per-task device contract in the source so future maintainers don't re-derive the proof."
  - "15-AUDIT-LOG.md with per-site classification (SAFE x 11, NEEDS-PATCH=0, UNCLEAR=0) and upstream-trace evidence."
  - "Smoking-gun assumption-language comment in top_n.cpp ('all batches are expected to share the same space in practice') replaced by the verified INVARIANT phrasing — string is gone from src/op/."
affects: [phase-15-02, phase-15-03, phase-15-04, future-mgpu-operator-additions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "INVARIANT (SCHED-RR contract) comment pattern: 5-line block above any get_memory_space() read on operator inputs, naming the upstream contract path (gpu_pipeline_task::execute_pipeline_task_round -> pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch) and the audit-log reference."

key-files:
  created:
    - .planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md
    - .planning/phases/15-mgpu-operator-colocation-audit/15-01-SUMMARY.md
  modified:
    - src/op/sirius_physical_concat.cpp
    - src/op/sirius_physical_top_n.cpp
    - src/op/sirius_physical_ungrouped_aggregate.cpp
    - src/op/sirius_physical_sort_sample.cpp
    - src/op/sirius_physical_sort_partition.cpp
    - src/op/sirius_physical_table_scan.cpp
    - src/op/sirius_physical_order.cpp
    - src/op/sirius_physical_merge_sort.cpp
    - src/op/sirius_physical_nested_loop_join.cpp

key-decisions:
  - "Classification verdict for all 11 sites: SAFE. Trace: operator execute() dynamic-casts to pipelineable_operator_data (or derived partitioned_operator_data); the input vector is the same _data_batches that gpu_pipeline_task::execute_pipeline_task_round (gpu_pipeline_task.cpp:332) lock-converted onto target_space via prepare_for_processing -> lock_or_prepare_batch. POSTCONDITION batches[0]->get_memory_space() == target_space holds at every site."
  - "Smoking-gun replacement (top_n.cpp:240) chose REPLACE not AUGMENT: the original 'all batches are expected to share the same space in practice' phrasing is exactly the unverified-assumption-language this audit is removing. The verbatim string no longer appears in src/op/."
  - "Plan Task 3 verify command (! grep -F 'NEEDS-PATCH' on the audit log) is internally inconsistent with the must_have artifact requirement (the summary line itself contains the verbatim tokens 'NEEDS-PATCH=0 UNCLEAR=0'). Followed the orchestrator's success_criteria interpretation: the summary line MUST exist with =0 values, and no per-site row may have a non-SAFE verdict. The audit log accordingly contains those tokens only in (a) the explanatory framing prose and (b) the structured summary line — never as a per-site verdict."
  - "Branch base: HEAD reset to current commit 11d20b9 (Phase 15 docs-creation HEAD); ancestry verified via merge-base --is-ancestor 0ee3166 — descends from Phase 14 HEAD."

patterns-established:
  - "Audit-by-comment pattern: when an upstream invariant is empirically PASS but not encoded in the read site, a 5-line // INVARIANT (...) block plus a per-site audit-log row is sufficient documentation; no behavioral diff."
  - "Comment-only diff verification recipe: git diff src/op/...cpp | grep -E '^[+-]' | grep -v -E '^(\\+\\+\\+|---)' | grep -v -E '^[+-][[:space:]]*//' must emit zero lines."

requirements-completed: []  # Plan 15-01 frontmatter `requirements:` is empty.

# Metrics
duration: 6min
completed: 2026-05-01
---

# Phase 15 Plan 01: Cross-GPU operator-colocation audit Summary

**Comment-only audit of 11 operator-input get_memory_space() sites — all SAFE under SCHED-RR contract; INVARIANT comments + audit log encode the upstream-trace proof, smoking-gun assumption-language string removed from top_n.cpp.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-05-01T01:33:12Z
- **Completed:** 2026-05-01T01:38:43Z
- **Tasks:** 4 (executed; Tasks 1+2 are analysis-only, Tasks 3+4 produce one atomic commit)
- **Files modified:** 9 operator .cpp files + 1 new audit log

## Accomplishments

- All 11 audit sites in `15-CONTEXT.md` lines 26-58 carry an `// INVARIANT (SCHED-RR contract): ...` comment immediately above the `get_memory_space()` call.
- `15-AUDIT-LOG.md` created with: per-site classification table (11 SAFE rows), upstream-trace justification per site, the verbatim per-task device contract recap, an out-of-scope discoveries section documenting `table_scan.cpp:217` (output-side `get_memory_space()`, not input), Phase 14 cross-reference, and the structured-grep-checkable summary line `## Classification: SAFE=11 NEEDS-PATCH=0 UNCLEAR=0`.
- Smoking-gun assumption-language comment in `top_n.cpp` (`"Use the memory space from the first valid batch (all batches are expected to share the same space in practice)."`) REPLACED by the INVARIANT phrasing. Verbatim string is gone from `src/op/` (verified: `grep -rF 'all batches are expected to share the same space in practice' src/op/` exits 1).
- Build: MCP `name=build` exit 0 in 11.2s; the 9 modified files re-compile cleanly into both `sirius_extension` and `sirius_loadable_extension` targets, plus `sirius_unittest` re-links.
- Smoke tests:
  - `[mgpu]`: 12/13 PASS, exit 1, 32.1s — same Phase-12-territory `physical_order - small sort stays single-GPU` failure as Phase 14 baseline (`vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` verbatim). Not in this branch's ancestry.
  - `[TPC-H][parquet]`: 22/22 PASS, exit 0, 80.7s, 36256 assertions. In-noise vs Phase 14's 80.3s.
- HYG-02 baseline preserved: `grep -rn 'rmm::cuda_stream_default' src/ | wc -l` returns 40, exactly equal to Phase 14 baseline.

## Task Commits

Each task was committed atomically:

1. **Tasks 1+2: Branch + line-number reconciliation + classification analysis** (no source changes — analysis-only; line numbers reconciled and verdicts drafted in scratch memory).
2. **Tasks 3+4: Apply INVARIANT comments + write audit log + smoke-test commit** — `b91afc3` (audit)

The plan structure makes Tasks 1+2 explicitly analysis-only with no committable artifacts; the source modifications and audit log are produced in Task 3 and verified in Task 4, then committed once at the end of Task 4 via the gsd-tools-equivalent direct git commit (used direct `git commit --no-verify` per the parallel_execution flag in the executor prompt; gsd-tools `commit` does not natively support multi-paragraph body messages).

## Files Created/Modified

- **Created:** `.planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md` — Per-site classification table, upstream-trace evidence, structured summary line.
- **Modified (9 .cpp files, comment-only):**
  - `src/op/sirius_physical_concat.cpp` — INVARIANT at line 193 (now ~line 198)
  - `src/op/sirius_physical_top_n.cpp` — INVARIANT at line 173 (TopN single-batch) AND line 240 (MERGE_TOP_N, smoking-gun replacement)
  - `src/op/sirius_physical_ungrouped_aggregate.cpp` — INVARIANT at line 339 (per-batch) AND line 505 (multi-batch concat)
  - `src/op/sirius_physical_sort_sample.cpp` — INVARIANT at line 112 (if (!space) site)
  - `src/op/sirius_physical_sort_partition.cpp` — INVARIANT at line 98 (per-batch)
  - `src/op/sirius_physical_table_scan.cpp` — INVARIANT at line 129 (multi-batch coalesce path)
  - `src/op/sirius_physical_order.cpp` — INVARIANT at line 76 (per-batch)
  - `src/op/sirius_physical_merge_sort.cpp` — INVARIANT at line 92 (if (!space) site)
  - `src/op/sirius_physical_nested_loop_join.cpp` — INVARIANT at line 415 (left_batch->get_memory_space() site)

## Per-site verdict table (copied from 15-AUDIT-LOG.md)

| #  | File | Line | Verdict | Input type |
|----|------|------|---------|------------|
| 1  | src/op/sirius_physical_concat.cpp | 193 | SAFE | partitioned_operator_data |
| 2  | src/op/sirius_physical_top_n.cpp | 173 | SAFE | pipelineable_operator_data |
| 3  | src/op/sirius_physical_top_n.cpp | 240 | SAFE | pipelineable_operator_data |
| 4  | src/op/sirius_physical_ungrouped_aggregate.cpp | 339 | SAFE | pipelineable_operator_data |
| 5  | src/op/sirius_physical_ungrouped_aggregate.cpp | 505 | SAFE | pipelineable_operator_data |
| 6  | src/op/sirius_physical_sort_sample.cpp | 112 | SAFE | pipelineable_operator_data |
| 7  | src/op/sirius_physical_sort_partition.cpp | 98 | SAFE | pipelineable_operator_data |
| 8  | src/op/sirius_physical_table_scan.cpp | 129 | SAFE | pipelineable_operator_data |
| 9  | src/op/sirius_physical_order.cpp | 76 | SAFE | pipelineable_operator_data |
| 10 | src/op/sirius_physical_merge_sort.cpp | 92 | SAFE | pipelineable_operator_data |
| 11 | src/op/sirius_physical_nested_loop_join.cpp | 415 | SAFE | pipelineable_operator_data |

**Verdict:** SAFE=11 NEEDS-PATCH=0 UNCLEAR=0.

## Line-number drift catalog (CONTEXT.md vs current tree)

The planner pre-flagged that some sites had drifted between `15-CONTEXT.md` (written before intervening edits) and the current tree. Reconciliation results:

| File | CONTEXT.md line | Current line | Drift |
|------|----------------|---------------|-------|
| concat.cpp | 193 | 193 | none |
| top_n.cpp (TopN single-batch) | 173 | 173 | none |
| top_n.cpp (MERGE_TOP_N) | 230 | 240 | +10 lines (planner-anticipated; smoking-gun comment site) |
| ungrouped_aggregate.cpp (per-batch) | 339 | 339 | none |
| ungrouped_aggregate.cpp (multi-batch concat) | 501 | 505 | +4 lines (planner-anticipated) |
| sort_sample.cpp | 112 | 112 | none |
| sort_partition.cpp | 98 | 98 | none |
| table_scan.cpp | 129 | 129 | none |
| order.cpp | 76 | 76 | none |
| merge_sort.cpp | 92 | 92 | none |
| nested_loop_join.cpp | 415 | 415 | none |

Out-of-scope `table_scan.cpp:217` (`output_batch->get_memory_space()`) was confirmed present in the grep but not touched per CONTEXT.md scope.

## MCP smoke-test verdicts (verbatim tail)

### [mgpu] (Phase 14 baseline equivalence)

```
test cases:   13 |   12 passed | 1 failed
assertions: 1979 | 1978 passed | 1 failed

physical_order - small sort stays single-GPU
.../mgpu_test_utils.hpp:343: FAILED:
  REQUIRE_FALSE( gpu_result->HasError() )
with expansion:
  !true
with message:
  gpu_execution error: Invalid Error: SiriusExecuteQuery error: Invalid Error:
  vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)
```

Exit code: 1, duration: 32.1s. Same Phase-12-territory failure as Phase 14 baseline; same verbatim message; not in this branch's ancestry. **No new failures introduced** (acceptance criterion satisfied).

### [TPC-H][parquet]

```
[0/22]..[22/22] (100%)
===============================================================================
All tests passed (36256 assertions in 22 test cases)
```

Exit code: 0, duration: 80.7s. Matches Phase 14 baseline (80.3s, in-noise).

## HYG-02 confirmation

```
$ grep -rn 'rmm::cuda_stream_default' src/ | wc -l
40
```

Baseline of 40 preserved. The audit is comment-only, so this is the expected outcome.

## Branch / commit pointers

- **Branch:** `audit/mgpu-operator-colocation`
- **Branch HEAD:** `b91afc3 audit(15-01): comment-only INVARIANT (SCHED-RR contract) at 11 operator-colocation sites`
- **Branch base:** `11d20b9 docs(15): create Phase 15 plan for mgpu operator-colocation audit` (current Phase 15 docs HEAD)
- **Ancestry assertion:** `git merge-base --is-ancestor 0ee3166 HEAD` = true (descends from Phase 14 HEAD `0ee3166 docs(14-02): complete validate-SCHED-RR-distribution plan`).

## Phase 14 baseline cross-reference

`14-VALIDATION.md` (commit `76c3342`) Phase 14 ship-gate:
- C1 `[mgpu]`: 12/13 PASS-with-Phase-12-note ← matched here verbatim (same single failure, same message).
- C2 `[TPC-H][parquet]`: 22/22 PASS in 80.3s ← matched here in-noise (80.7s).
- HYG-02 = 40 ← preserved here exactly.

The empirical PASS evidence on Phase 14 is what made the SAFE classification overwhelmingly likely. This audit confirms it via upstream-trace proof + comment-only documentation, with the smoke tests redundantly re-verifying that no regression was introduced.

## Decisions Made

- **Classification verdict for all 11 sites: SAFE** — based on the upstream-trace through `gpu_pipeline_task::execute_pipeline_task_round -> pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch`. POSTCONDITION `batches[0]->get_memory_space() == target_space` holds at every site.
- **Smoking-gun replacement (top_n.cpp:240)** chose REPLACE over AUGMENT: the original phrasing was exactly the unverified-assumption-language this audit is removing.
- **Plan verify-vs-must_have inconsistency resolved per orchestrator's success_criteria** (the summary line MUST contain the literal tokens "NEEDS-PATCH=0 UNCLEAR=0"; the strict `! grep -F` rule cannot be satisfied alongside the must_have).
- **Single-commit pattern for Tasks 3+4**: the plan structure makes Tasks 1+2 analysis-only and Tasks 3+4 produce one atomic comment-only artifact; one commit captures both edits and the audit log.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan Task 3 verify command is internally inconsistent with the must_have artifact requirement**
- **Found during:** Task 3 (writing 15-AUDIT-LOG.md)
- **Issue:** Plan Task 3 `<verify>` requires `! grep -F 'NEEDS-PATCH' .planning/.../15-AUDIT-LOG.md` AND `! grep -F 'UNCLEAR' .planning/.../15-AUDIT-LOG.md`. But the plan must_haves require the audit log to contain a structured summary line `## Classification: SAFE=N NEEDS-PATCH=M UNCLEAR=U` — which by definition contains both tokens. Mutually exclusive constraints.
- **Fix:** Followed the orchestrator's success_criteria interpretation (the summary line MUST exist with =0 values for NEEDS-PATCH and UNCLEAR; no per-site verdict cell may say NEEDS-PATCH or UNCLEAR). The structured-grep `grep -E '^## Classification: SAFE=[0-9]+ NEEDS-PATCH=0 UNCLEAR=0$'` test (which is the more specific verify) PASSES; the strict `! grep -F` test would fail solely because of the summary line, which is the exact mandated artifact.
- **Files modified:** `.planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md`
- **Verification:** Structured-grep classification line passes (1 match). No per-site row contains NEEDS-PATCH or UNCLEAR (only the summary line + framing prose, which are necessary).
- **Committed in:** `b91afc3` (single audit commit)

**2. [Rule 3 - Blocking] `.planning/` is gitignored; needed `git add -f` for the audit log**
- **Found during:** Task 4 (staging files for commit)
- **Issue:** Initial `git add .planning/.../15-AUDIT-LOG.md` failed: "The following paths are ignored by one of your .gitignore files: .planning". The repository's `.gitignore` ignores `.planning/`, but every prior phase's planning artifacts are under that path and have been successfully committed. This means the existing convention is to force-add planning artifacts.
- **Fix:** Used `git add -f .planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md` (force-add). Source files added without `-f`.
- **Files modified:** Staging only (no code change).
- **Verification:** `git diff --cached --stat` shows audit log + 9 operator files staged.
- **Committed in:** `b91afc3`

---

**Total deviations:** 2 auto-fixed (1 plan-internal-inconsistency Rule 1, 1 staging-blocking Rule 3)
**Impact on plan:** Both auto-fixes were necessary to satisfy the orchestrator's success criteria and to land the audit on the correct branch. No scope creep — diff remains comment-only.

## Issues Encountered

- **MCP `unit-tests` filter quoting** — confirmed via grep of recent state that filter must be passed as a string parameter, e.g. `args={"filter": "[mgpu]"}`. No issue arose.
- **Working tree had pre-existing dirty `.planning/` and submodule entries at start of Task 1** — these are baseline state churn from prior phases and were not blockers. The audit branch was created with `git checkout -B` so HEAD remained at `11d20b9` and the dirty entries follow the branch. The actual audit commit (`b91afc3`) only contains the 9 operator files + new audit log; the unrelated dirty entries remain unstaged.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Phase 15 plans 15-02, 15-03, 15-04 are unblocked. Plan 15-01 lands the foundational audit comments and audit log; subsequent plans address (per ROADMAP):
- mgpu_stress_test (CONTEXT.md "Stress-test mode" section)
- docs/super-sirius/ updates documenting the per-task device contract under SCHED-RR

The branch `audit/mgpu-operator-colocation` is ready for additional plans to add commits on top.

## Self-Check: PASSED

- [x] `.planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md` exists (verified: `[ -f ... ] && echo FOUND` → FOUND).
- [x] Commit `b91afc3` exists (verified: `git log --oneline -1` shows `b91afc3 audit(15-01): comment-only INVARIANT (SCHED-RR contract) at 11 operator-colocation sites`).
- [x] All 9 source files modified with comment-only diffs (verified: `git diff --stat HEAD~1 HEAD` shows +146/-2 across 10 files; non-comment-line filter pipeline emits zero output).
- [x] Smoking-gun string `all batches are expected to share the same space in practice` is gone from `src/op/` (verified: `grep -rF` exits 1).
- [x] HYG-02 = 40 (verified).
- [x] Structured classification line present (verified: `grep -E '^## Classification: SAFE=[0-9]+ NEEDS-PATCH=0 UNCLEAR=0$'` returns 1 match).
- [x] Branch `audit/mgpu-operator-colocation` descends from Phase 14 HEAD `0ee3166` (verified: `git merge-base --is-ancestor` → exit 0).
- [x] Build clean, [mgpu] regression-equivalent to Phase 14 baseline, [TPC-H][parquet] 22/22 PASS.

---
*Phase: 15-mgpu-operator-colocation-audit*
*Completed: 2026-05-01*
