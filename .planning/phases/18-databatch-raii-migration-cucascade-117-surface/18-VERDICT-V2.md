---
phase: 18-databatch-raii-migration-cucascade-117-surface
verdict_date: 2026-05-05
status: PASS
supersedes: 18-VERDICT.md
gap_closure_plan: 18-07
---

# Phase 18 Verdict V2

## Summary

**Phase 18 PASS.** This V2 verdict supersedes 18-VERDICT.md (PARTIAL, 2026-05-05). The DB-05 runtime gap identified in V1 has been closed by plan 18-07's Path A architectural fix: `gpu_pipeline_task::execute` no longer holds a `std::vector<cucascade::mutable_data_batch> processing_handles` across `op->execute()`, and `pipelineable_operator_data::prepare_for_processing` now performs eager memory-space conversion under SHORT-scoped exclusive accessors that are released BEFORE the function returns. Operators inside `execute()` acquire their own per-call `to_read_only()` / `to_mutable()` accessors at narrowest scope (already migrated by plans 18-03/18-04). The glibc EDEADLK ("Resource deadlock avoided") signature that fired on every `[mgpu]` test in 18-06 is gone — `[mgpu]` now passes 16/16, `[mgpu_stress]` default-mode passes in 75.5s, and compute-sanitizer racecheck on the `[downgrade_lifecycle]` proxy reports 0 hazards. DB-01..04 invariants are preserved (no static-gate regression).

## Requirement Status

| ID | Description | V1 Status | V2 Status | Evidence |
|----|-------------|-----------|-----------|----------|
| DB-01 | `batch_lock_utils.hpp` rewritten | PASS | PASS | 18-01-SUMMARY commits `850f4e9`, `cc9546f`, `5233ce9`. Plan 18-07 audit dropped `try_acquire_mutable` (zero production callers); helpers retained: `prepare_and_acquire_mutable`, `acquire_read_only`. |
| DB-02 | All call sites migrated | PASS | PASS | Repo-wide grep gates (re-run 2026-05-05 post-18-07): `DELETED_FSM_GREP_HITS=0` (live, non-comment); `FSM_STATE_LITERAL_HITS=0`; `THREE_ARG_POPID_HITS=0`. |
| DB-03 | Operators + tests adapted | PASS | PASS | 18-02..18-05 SUMMARYs + 18-06 prelude. 18 production .cpp + 23 test/cpp/ + 8 inventory-miss src/ (18-05) + 8 inventory-miss test/ (18-06) all migrated. No regressions from 18-07. |
| DB-04 | Compile-clean + HYG-02 ≤ 40 | PASS | PASS | `mcp__project-commands__run_command build` exit 0 (verified post-18-07 Tasks 1+2; 48 targets compiled, sirius_unittest linked); `HYG02_TOTAL=40`, `HYG02_NON_LEGACY=0`. |
| **DB-05** | **[mgpu] 16/16 + [mgpu_stress] + racecheck** | **FAIL** | **PASS** | **[mgpu] 16/16 PASS, 79091 assertions, 103.5s; [mgpu_stress] PASS, 77053 assertions, 75.5s; racecheck on [downgrade_lifecycle] proxy: 0 hazards.** |

## Phase 18 Success Criteria (from ROADMAP)

| # | Criterion | V2 Status | Evidence |
|---|-----------|-----------|----------|
| 1 | Deleted-FSM-symbol grep returns zero live hits | PASS | `DELETED_FSM_GREP_HITS=0` (live, non-comment) — see 18-07-gate-evidence.log static-gates section. |
| 2 | MCP build exits 0 | PASS | Verified post-Task 1 (104s, 48 targets) and post-Task 2 (8.3s incremental). |
| 3 | `grep -c rmm::cuda_stream_default src/` ≤ 40 | PASS | 40 total / 0 non-legacy. |
| 4 | [mgpu] 16/16 | **PASS (was FAIL in V1)** | All 16 sub-tests passed in 103.5s. 79091 assertions. Sub-tests listed in 18-07-gate-evidence.log. |
| 5 | [mgpu_stress] 1-iter exit 0 | **PASS (was NOT RUN in V1)** | 75.5s, 77053 assertions, exit 0. Well under 180s expected runtime. |

## Static Gates (re-run post-18-07)

```
DELETED_FSM_GREP_HITS=0           (live, non-comment)
FSM_STATE_LITERAL_HITS=0
THREE_ARG_POPID_HITS=0
HYG02_TOTAL=40                    (all in src/legacy/)
HYG02_NON_LEGACY=0
PROCESSING_HANDLES_LIVE=0         (Path A drop — no held vector across op->execute())
PROCESSING_HANDLES_ALL=1          (1 archival comment in sirius_physical_grouped_aggregate_merge.cpp documenting the revert)
```

Full evidence in `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-gate-evidence.log`.

## Dynamic Gates (DB-05 closure — primary V2 deliverable)

### [mgpu] filter — DB-05 PRIMARY GATE
- **V1 result:** FAIL — 0/16, glibc EDEADLK on every test.
- **V2 result:** PASS — 16/16, 79091 assertions, 103.5s.
- **MCP invocation:** `mcp__project-commands__run_command unit-tests --filter "[mgpu]"`
- **Sub-tests passed (in order, per Catch2 listing):**
  1. `gpu_execution - table_gpu cache warm cross-GPU hazard (follow-up #17)`
  2. `grouped_aggregate_merge - group by with high cardinality distributes across both GPUs`
  3. `grouped_aggregate_merge - group by with single key forces single-GPU path`
  4. `grouped_aggregate_merge - count(*)-only aggregate across two GPUs`
  5. `physical_hash_join - BUILD_PROBE probe-heavy join across two GPUs`
  6. `physical_hash_join - MIXED_JOIN large-vs-large join distributes partitions`
  7. `physical_hash_join - repeated BUILD_PROBE queries don't wedge on leftover state` (this was a EDEADLK site in V1)
  8. `hash_join bisect 1 - simple JOIN+GROUP BY+ORDER BY, cache=none`
  9. `hash_join bisect 2 - simple JOIN+GROUP BY+ORDER BY, cache=table_gpu`
  10. `hash_join bisect 3 - Q11 shape with HAVING subquery, cache=none`
  11. `physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache`
  12. `physical_order - large sort distributes across two GPUs`
  13. `physical_order - small sort rangecheck regression` (this was a EDEADLK site in V1)
  14. `physical_order - small sort stays single-GPU`
  15. `physical_order - order by with limit over large input`
  16. `mgpu_stress - SCHED-RR counter offset rotation` (16/16 in [mgpu] filter; full [mgpu_stress] re-run captured below)

### [mgpu_stress] default-mode
- **V1 result:** NOT RUN (precondition failed under [mgpu] EDEADLK).
- **V2 result:** PASS — 1 test case, 77053 assertions, 75.5s (well under 180s budget).
- **MCP invocation:** `mcp__project-commands__run_command unit-tests --filter "[mgpu_stress]"`

### compute-sanitizer racecheck
- **V1 result:** 0 hazards on `[downgrade_lifecycle]` proxy (GPU-side clean; CPU EDEADLK was the V1 blocker).
- **V2 result:** 0 hazards on `[downgrade_lifecycle]` proxy (preserved; no GPU-side regression from Path A).
- **Invocation (Bash + timeout 600 per project memory feedback_sanitizer_via_bash_not_mcp):**
  ```bash
  timeout 600 /usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck \
    build/release/extension/sirius/test/cpp/sirius_unittest "[downgrade_lifecycle]"
  ```
- **Result:** `========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`. 8/8 sub-tests passed (53 assertions).
- **Tag rationale:** `[mgpu_foundation]` does not exist in the suite (matches 18-06 finding). `[downgrade_lifecycle]` retained as the closest non-deadlocking proxy. Note: racecheck is GPU-side only; the CPU std::shared_mutex deadlock that was the V1 blocker is not detected by compute-sanitizer (which is why the V1 racecheck was 0 hazards even with [mgpu] EDEADLK).

## Pitfall Compliance Audit

- **P1 (RAII lock scope):** **NOW SATISFIED.** V1 verdict marked this VIOLATED at runtime; V2 closes the violation by dropping the R5 lock-and-hold pattern. `gpu_pipeline_task::execute` no longer stores `vector<mutable_data_batch> processing_handles` across `op->execute()` (commit `0575b0a`); operators inside `execute()` take their own per-call accessors at narrowest scope (migrated by 18-03/18-04). `batch_lock_utils.hpp` audited — `try_acquire_mutable` removed (zero callers); `prepare_and_acquire_mutable` retained with explicit Path-A doc warning; `acquire_read_only` retained with Path-A reminder (commit `99e6765`).
- **P3 (pop_next_data_batch):** PASS (preserved from V1). Compile-time gate clean (zero `pop_data_batch.*task_created|in_transit` hits). [mgpu] runtime gate now serves as the smoke proxy for P3 (which was blocked under V1).
- **P7 (PR #739 × #117):** PASS (preserved from V1). Zero `data_batch_processing_handle` re-introductions in src/ or test/ outside descriptive comments.

## Hand-off to Phase 19

- Cucascade pin still `1c1e648` (defended by Phase 17 D-G6).
- Build clean against post-#117 RAII (with liburing-dev installed via `pixi install`).
- **P1 architectural blocker is RESOLVED.** Phase 19 IO Framework adoption can begin runtime-gate work without inheriting the EDEADLK deadlock from 18-02's R5 design. Phase 21 REG-XX gates can also run unconditionally.
- Open follow-ups:
  - `mark_task_created` Sirius-method renaming (not done; Phase 18 carryover).
  - `readonly_to_mutable` demotion opportunity from RESEARCH.md Open Question 1.
  - `convertible_data_batch` readonly path optimization.
  - Phase 21 REG-02 [TPC-H][parquet] correctness check (deferred from Phase 18 per scope).

## Files Modified — Plan 18-07 (V2 closure)

**src/ production code (4 files):**
- `src/op/sirius_physical_operator.cpp` — `pipelineable_operator_data::prepare_for_processing` rewritten for Path A (eager conversion under `{}`-scoped accessors; returns empty vector). Commit `0575b0a`.
- `src/include/op/sirius_physical_operator.hpp` — doc comment block updated to describe Path A semantics; references 18-VERIFICATION.md and 18-07-SUMMARY.md. Commit `0575b0a`.
- `src/pipeline/gpu_pipeline_task.cpp` — `processing_handles` and `handles_opt` storage dropped; `prepare_for_processing` result consumed for OOM/lock-failure detection only. Commit `0575b0a`.
- `src/op/sirius_physical_grouped_aggregate_merge.cpp` — stale R5 comment updated to describe Path A semantics for the size==1 path. Commit `0575b0a`.

**src/include helpers (1 file):**
- `src/include/pipeline/batch_lock_utils.hpp` — `try_acquire_mutable` removed; `prepare_and_acquire_mutable` and `acquire_read_only` retained with Path-A doc clarifications; file-level doc block references 18-VERIFICATION.md + EDEADLK runtime evidence. Commit `99e6765`.

**Documentation (3 files):**
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT-V2.md` (this file).
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-SUMMARY.md`.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-gate-evidence.log`.

## Plan-by-Plan Status (final)

| Plan | Status | Notes |
|------|--------|-------|
| 18-01 | PASS | DB-01 closed; header-first ripple |
| 18-02 | PASS | Operator base layer + R5 lock-and-hold (P1 risk introduced) |
| 18-03 | PASS | 8 stateful operators; P1 risk documented |
| 18-04 | PASS | Read-only operators + Pitfall 4 closure |
| 18-05 | PASS | DB-03 closure (23 test files migrated) |
| 18-06 | PARTIAL → SUPERSEDED | DB-04 PASS; DB-05 FAIL; surfaced P1 deadlock — superseded by 18-07 |
| 18-07 | PASS | DB-05 closure via Path A architectural fix; Phase 18 verdict flipped to PASS |

## Verdict

**Phase 18 PASS — all DB-01..05 requirements satisfied.** Path A architectural fix landed in plan 18-07 closes the runtime regression that 18-06 surfaced. Phase 19 IO Framework adoption can proceed without the P1 blocker. ROADMAP.md, STATE.md, and REQUIREMENTS.md updated to reflect the closure.
