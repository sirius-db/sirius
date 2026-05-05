---
phase: 18-databatch-raii-migration-cucascade-117-surface
plan: 07
subsystem: pipeline
tags: [cucascade, raii, data_batch, mutable_data_batch, read_only_data_batch, deadlock-fix, path-a, phase-18-closure, db-05]

# Dependency graph
requires:
  - phase: 18-databatch-raii-migration-cucascade-117-surface
    plan: 06
    provides: Phase 18 PARTIAL verdict + 8 inventory-miss test/cpp/ files migrated; DB-04 closure (MCP build exit 0); DB-05 FAIL with P1 deadlock evidence captured
provides:
  - DB-05 closure via Path A architectural fix (drop R5 lock-and-hold from gpu_pipeline_task)
  - pipelineable_operator_data::prepare_for_processing rewritten for eager conversion under short-scoped accessors (returns empty vector)
  - batch_lock_utils.hpp audited; try_acquire_mutable removed (zero callers); Path-A doc warnings on retained helpers
  - Phase 18 verdict flipped from PARTIAL to PASS
  - 18-VERDICT-V2.md superseding 18-VERDICT.md
  - State + roadmap + requirements updates reflecting Phase 18 PASS
affects: [19, 20, 21]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Path A semantics: pipelineable_operator_data::prepare_for_processing performs eager memory-space conversion under SHORT-scoped exclusive accessors (acquired/released inside `{}` block per iteration); returns empty vector on success. Operators inside execute() acquire their own per-call to_read_only / to_mutable accessors at narrowest scope."
    - "EDEADLK fail-fast as runtime gate: glibc std::shared_mutex's same-thread re-lock detection (POSIX EDEADLK, abort with 'Resource deadlock avoided') is the canonical runtime test for any future plan tempted to reintroduce R5-style lock-and-hold. [mgpu] suite serves as the smoke gauntlet for this class of bug."
    - "Helper API hygiene: drop unused non-blocking variants when the only conceptual use case (lock-and-hold) is reverted. Restoring requires a Path-A-compatible call site."

key-files:
  created:
    - .planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT-V2.md
    - .planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-SUMMARY.md
    - .planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-gate-evidence.log
  modified:
    - src/op/sirius_physical_operator.cpp
    - src/include/op/sirius_physical_operator.hpp
    - src/pipeline/gpu_pipeline_task.cpp
    - src/op/sirius_physical_grouped_aggregate_merge.cpp
    - src/include/pipeline/batch_lock_utils.hpp
    - .planning/STATE.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "[18-07] Path A chosen over Path B (per user direction in plan objective): drop R5 lock-and-hold semantics from gpu_pipeline_task::compute_task. Operators inside execute() take their own narrow-scoped accessors. This aligns with CONTEXT.md P1 mitigation guidance ('scope every read_only_data_batch / mutable_data_batch accessor to the narrowest possible block') and is a centralized fix vs Path B's per-operator delta of exposing get_locked_accessors()."
  - "[18-07] try_acquire_mutable DELETED (zero production callers). Only conceptual use case was R5 lock-and-hold-adjacent patterns; with R5 reverted, restoring requires a Path-A-compatible call site. Documented removal in file-level doc block."
  - "[18-07] handles_opt renamed to prepare_result in gpu_pipeline_task::execute and explicitly .reset() before compute_task — preserves OOM/lock-failure detection (the optional remains the failure signal) but makes lifetime explicit in the source. The vector itself is always empty under Path A on success."
  - "[18-07] Stale R5 comment in sirius_physical_grouped_aggregate_merge.cpp (line 211) updated to describe Path A semantics for the size==1 path. The scoped to_mutable in this operator is now the ONLY exclusive lock on the batch for the merge step (no overlap with a R5 vector)."
  - "[18-07] Static gates (HYG-02, deleted-FSM-symbol grep, FSM literal grep, 3-arg pop_data_batch_by_id) all preserved at 0 / ≤ 40 baseline. DB-01..04 invariants did not regress."
  - "[18-07] [mgpu_foundation] tag still does not exist in the test suite (matches 18-06 finding). compute-sanitizer racecheck retained the [downgrade_lifecycle] proxy. Note: racecheck is GPU-side only and would NOT have caught the V1 EDEADLK CPU deadlock — the runtime [mgpu] suite is the canonical gate for that class of bug."

patterns-established:
  - "Pattern: Path A scoped-conversion helper. The prepare_for_processing implementation acquires an exclusive accessor inside a `{}` block, performs the side-effect (memory-space conversion), then drops the accessor at the block boundary before iterating to the next batch. The function returns an empty vector to satisfy the existing `std::optional<vector<>>` contract."
  - "Pattern: empty-vector + optional sentinel return. The optional return value is preserved across Path A even though the vector is always empty on success — std::nullopt remains the OOM / lock-failure signal. Caller treats the vector as a no-op."
  - "Pattern: helper API audit on revert. When an architectural design pattern is reverted, audit dependent helpers and DROP rather than retain those whose only conceptual use was the now-reverted pattern. Re-introducing requires showing a Path-A-compatible call site."

requirements-completed: [DB-05]

# Metrics
duration: 111min
completed: 2026-05-05
---

# Phase 18 Plan 07: DataBatch RAII Path A — DB-05 Closure Summary

**Phase 18 closure via Path A architectural fix: dropped 18-02's R5 lock-and-hold from `gpu_pipeline_task::execute` and `pipelineable_operator_data::prepare_for_processing`. The held `std::vector<cucascade::mutable_data_batch> processing_handles` across `op->execute()` is gone; operators inside `execute()` acquire their own per-call accessors at narrowest scope. The glibc EDEADLK ("Resource deadlock avoided") that 18-06 confirmed on every `[mgpu]` test is gone. `[mgpu]` 16/16 PASS in 103.5s; `[mgpu_stress]` PASS in 75.5s; racecheck 0 hazards. Phase 18 verdict flipped from PARTIAL to PASS.**

## Performance

- **Duration:** ~111min (build + dynamic gates dominated; full [mgpu] takes ~100s, [mgpu_stress] ~75s, racecheck ~30s, plus build cycles).
- **Started:** 2026-05-05T21:39:38Z
- **Completed:** 2026-05-05T23:31:04Z
- **Tasks:** 3 / 3
- **Files modified:** 8 (5 src + 3 docs)

## Accomplishments

### Task 1: Drop R5 lock-and-hold

**`src/op/sirius_physical_operator.cpp` — `pipelineable_operator_data::prepare_for_processing` rewritten:**
- The function now iterates `_data_batches` and, for each batch, opens a `{}` block that acquires `pipeline::prepare_and_acquire_mutable(...)` and performs eager memory-space conversion. The accessor is destroyed at the block boundary BEFORE the next iteration.
- nullptr-detection observability log preserved (Phase 9 FIX-B `[mgpu-probe]` breadcrumb).
- OOM and unknown-exception handlers preserved (with the same SIRIUS_LOG_ERROR signatures).
- Returns `std::vector<::cucascade::mutable_data_batch>{}` (empty) on success — the `std::optional` is retained as the failure signal (caller still uses `prepare_result.has_value()` for OOM detection).
- Old R5 comment block replaced with Path A description referencing 18-VERIFICATION.md.

**`src/include/op/sirius_physical_operator.hpp` — doc comment updated:**
- Replaced the "R5 lock-and-hold (Phase 18 / DB-01)" block on `operator_data::prepare_for_processing` with Path A semantics: returned vector is EMPTY for non-source operator data (eager conversion done under short-scoped accessors); operators inside `execute()` acquire their own per-call accessors at narrowest scope.
- References 18-VERIFICATION.md (gap analysis) and 18-07-SUMMARY.md (closure record).

**`src/pipeline/gpu_pipeline_task.cpp` — `processing_handles` storage removed:**
- `handles_opt` renamed to `prepare_result` (clearer intent under Path A).
- The `prepare_for_processing` call is preserved as-is (still triggers eager conversion + OOM/lock-failure detection).
- The line `std::vector<::cucascade::mutable_data_batch> processing_handles = std::move(*handles_opt);` is DELETED. The optional is `.reset()` explicitly before `compute_task` runs.
- The `// At this point, all input batches are locked for processing.` comment block (lines ~371-372) replaced with a Path A explainer.
- The `// Processing handles are automatically released here when they go out of scope` trailing comment replaced with a Path A explainer.
- The peak-memory bookkeeping loop on output batches (`get_data_batches()` per `pipelineable_output`) and the `get_input_size` loop on input batches both kept untouched. Their scoped `to_read_only()` reads were already audited as SAFE in 18-02-SUMMARY's "P1 Lock-Scope Concerns Surfaced" section; under Path A they are trivially safe (no held exclusive locks).

**`src/op/sirius_physical_grouped_aggregate_merge.cpp` — stale R5 comment updated:**
- Line 211 comment that referred to "gpu_pipeline_task's processing_handles holds a separate mutable accessor" rewritten to reflect Path A: "gpu_pipeline_task no longer holds processing_handles across op->execute(); this scoped to_mutable is the ONLY exclusive lock on the batch."

**Acceptance gates passed:**
- MCP build exit 0 (48 targets compiled, sirius_unittest linked).
- `grep -rn "processing_handles" src/ | grep -v /legacy/ | grep -v "//"` → 0 (only 1 archival comment remains, allowed by acceptance criteria).
- `grep -n "handles_opt" src/pipeline/gpu_pipeline_task.cpp` → 0 (variable renamed and reset).
- HYG-02 baseline preserved at 40 total / 0 non-legacy.
- Deleted-FSM-symbol grep (live, non-comment) → 0.
- "Path A" comment present in operator.cpp (3 occurrences for traceability).
- "R5 lock-and-hold" only in archival comments (2 hits, both archival per acceptance criteria allowance).

### Task 2: Audit batch_lock_utils.hpp

**`src/include/pipeline/batch_lock_utils.hpp`:**
- File-level doc block (lines 17-45) updated to:
  - Drop the bullet for `try_acquire_mutable` (now removed).
  - Reference 18-VERIFICATION.md and the EDEADLK runtime evidence from 18-06.
  - Explicitly note the Phase 18-02 R5 design was reverted in 18-07 after the EDEADLK fail-fast fired on every [mgpu] test.
- `prepare_and_acquire_mutable` doc block updated with a Path-A clarification: the single production caller is `pipelineable_operator_data::prepare_for_processing`, which uses it inside a `{}` block per iteration so the lock releases before return. The "P1 lock-scope warning" reformulated for Path A.
- `try_acquire_mutable` (lines 141-171 originally) DELETED. Replaced with a comment explaining the removal: zero production callers (verified via `grep -rn try_acquire_mutable src/ test/` returning only doc-comment hits in this file). Restoring requires a Path-A-compatible call site.
- `acquire_read_only` doc block: added a 1-line Path-A clarification ("scope the returned accessor to a narrow `{}` block — never hold across a downstream call that re-acquires on the same batch").

**Acceptance gates passed:**
- MCP build exit 0 (incremental, 8.3s).
- `grep -c "Path A" src/include/pipeline/batch_lock_utils.hpp` → 4 (≥ 1 target).
- `try_acquire_mutable` references outside this file: 0 (verified across src/ + test/).
- HYG-02 in this file: 0.
- File-level doc block references 18-VERIFICATION.md + EDEADLK runtime evidence.
- Helper API contract for `prepare_and_acquire_mutable` and `acquire_read_only` preserved — Task 1's `prepare_for_processing` rewrite still compiles.

### Task 3: Run full Phase 18 gauntlet + write VERDICT-V2 + state updates

**Static gates (re-run from 18-06 to confirm preservation):**
- `DELETED_FSM_GREP_HITS=0` (live, non-comment) → PASS
- `HYG02_TOTAL=40` (all in src/legacy/) → PASS
- `HYG02_NON_LEGACY=0` → PASS
- `PROCESSING_HANDLES_LIVE=0` (Path A drop) → PASS
- `FSM_STATE_LITERAL_HITS=0` → PASS
- `THREE_ARG_POPID_HITS=0` → PASS

**Dynamic gates (DB-05 closure — primary V2 deliverable):**

| Gate | V1 (18-06) | V2 (18-07) | Evidence |
|------|------------|------------|----------|
| MCP build exit 0 | 0 | 0 | 48 targets compiled, sirius_unittest linked |
| [mgpu] 16/16 | 0/16 (EDEADLK) | **16/16 PASS** | 79091 assertions, 103.5s |
| [mgpu_stress] default-mode | NOT RUN | **PASS** | 77053 assertions, 75.5s (budget 360s, expected ≤ 180s) |
| compute-sanitizer racecheck on [downgrade_lifecycle] proxy | 0 hazards | **0 hazards** | 8/8 sub-tests, 53 assertions |

`[mgpu_foundation]` tag still does not exist in the test suite (matches 18-06 finding) — `[downgrade_lifecycle]` retained as the closest non-deadlocking proxy. Note: racecheck is GPU-side only (would not detect CPU EDEADLK); the runtime `[mgpu]` suite is the canonical gate for the V1 bug class.

**Documentation updates:**
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT-V2.md` written, status PASS, all 5 DB-XX rows PASS, supersedes V1.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-SUMMARY.md` written (this file).
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-gate-evidence.log` written with full static + dynamic gate evidence.
- `.planning/STATE.md` updated: Phase 18 row → `Complete (7/7 plans, PASS)`; `completed_plans` 15 → 16; "P1 RAII lock-scope" blocker removed from `## Blockers / Concerns`; Performance Metrics row appended for Phase 18 P07.
- `.planning/ROADMAP.md` updated: Phase 18 entry text → `**COMPLETE 2026-05-05** ... 7/7 plans shipped. DB-01..05 PASS. Path A architectural fix landed in 18-07 ...`. Progress table row → `7/7 | Complete | 2026-05-05`.
- `.planning/REQUIREMENTS.md` updated: DB-05 row marked `[x]`; traceability table → `Complete`.

## Task Commits

| Task | Title | Commit |
|------|-------|--------|
| 1 | drop R5 lock-and-hold to close DB-05 P1 deadlock | `0575b0a` |
| 2 | audit batch_lock_utils.hpp for Path A semantics | `99e6765` |
| 3 | docs commit (this summary + VERDICT-V2 + state updates) | (final commit, see below) |

## Decisions Made

(See `key-decisions` in frontmatter for the canonical list.)

## Deviations from Plan

### None

The plan was executed exactly as written. Path A architectural fix landed; static gates preserved; dynamic gates closed DB-05. No Rule 1/2/3 auto-fixes were needed (the build was already compile-clean from 18-06; the Path A drop produced a clean incremental build on first attempt).

## Verification Gates Passed

| Gate | Target | Actual | Pass |
|------|--------|--------|------|
| MCP build exit 0 | 0 | 0 (48 targets linked into sirius_unittest) | yes |
| DELETED_FSM_GREP_HITS (live, non-comment) | 0 | 0 | yes |
| FSM_STATE_LITERAL_HITS | 0 | 0 | yes |
| THREE_ARG_POPID_HITS | 0 | 0 | yes |
| HYG02_TOTAL | ≤ 40 | 40 | yes |
| HYG02_NON_LEGACY | 0 | 0 | yes |
| PROCESSING_HANDLES_LIVE | 0 | 0 | yes |
| Path A comment in operator.cpp | ≥ 1 | 3 | yes |
| Path A count in batch_lock_utils.hpp | ≥ 1 | 4 | yes |
| try_acquire_mutable callers outside batch_lock_utils.hpp | 0 | 0 | yes |
| [mgpu] passed | 16/16 | 16/16 | **yes (DB-05 PRIMARY GATE — was FAIL in V1)** |
| [mgpu] exit | 0 | 0 | yes |
| [mgpu] runtime | ≤ 240s | 103.5s | yes |
| [mgpu] assertions | ≥ 79091 | 79091 | yes |
| [mgpu_stress] exit | 0 | 0 | yes |
| [mgpu_stress] runtime | ≤ 360s (expected ≤ 180s) | 75.5s | yes |
| racecheck hazards | 0 | 0 | yes (on proxy) |
| 18-VERDICT-V2.md exists with status PASS + all 5 DB-XX PASS | yes | yes | yes |
| STATE.md row for Phase 18 → 7/7 PASS | yes | yes | yes |
| STATE.md "P1 RAII lock-scope" blocker removed | yes | yes | yes |
| ROADMAP.md Phase 18 entry → COMPLETE (PASS) | yes | yes | yes |
| ROADMAP.md progress table → 7/7 | yes | yes | yes |
| REQUIREMENTS.md DB-05 → [x] | yes | yes | yes |

## Self-Check: PASSED

All 8 modified/created files exist on disk:
- `src/op/sirius_physical_operator.cpp` — verified contains "Path A" comment and the new `{}`-scoped block.
- `src/include/op/sirius_physical_operator.hpp` — verified contains "Path A semantics (Phase 18-07)".
- `src/pipeline/gpu_pipeline_task.cpp` — verified `processing_handles` and `handles_opt` are both gone (only archival comment in another file).
- `src/op/sirius_physical_grouped_aggregate_merge.cpp` — verified Path A comment present.
- `src/include/pipeline/batch_lock_utils.hpp` — verified `try_acquire_mutable` removed (only 2 doc-comment references documenting removal); 4 "Path A" mentions.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT-V2.md` — verified status PASS in frontmatter.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-SUMMARY.md` — this file.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-gate-evidence.log` — verified contains static + dynamic gate evidence.

Task commits verified in `git log --oneline`:
- `0575b0a fix(18-07): drop R5 lock-and-hold to close DB-05 P1 deadlock`
- `99e6765 refactor(18-07): audit batch_lock_utils.hpp for Path A semantics`

## Hand-off note for Phase 19+

The Phase 18 P1 architectural blocker is RESOLVED. Phase 19 IO Framework adoption can begin runtime-gate work without inheriting the EDEADLK deadlock from 18-02's R5 design. Phase 21 v1.4 ship-gate REG-XX requirements (mgpu, TPC-H parquet, integration TPC-H, SF100 Q1, mgpu_stress 500-iter, HYG-02 ≤ 40) can run unconditionally — the static infrastructure (DB-01..04) is preserved and the runtime regression (DB-05) is closed.

**Carryover follow-ups from Phase 18 verdict (still open, not blocking):**
- `mark_task_created` Sirius-method renaming (Phase 18 carryover).
- `readonly_to_mutable` demotion opportunity from RESEARCH.md Open Question 1.
- `convertible_data_batch` readonly path optimization.
- Phase 21 REG-02 [TPC-H][parquet] correctness check (deferred from Phase 18 per scope; runs as part of full v1.4 ship gate).

## Self-Check: PASSED

All claimed artifacts and commits verified on disk and in `git log`:

- File `src/op/sirius_physical_operator.cpp`: FOUND
- File `src/include/op/sirius_physical_operator.hpp`: FOUND
- File `src/pipeline/gpu_pipeline_task.cpp`: FOUND
- File `src/op/sirius_physical_grouped_aggregate_merge.cpp`: FOUND
- File `src/include/pipeline/batch_lock_utils.hpp`: FOUND
- File `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT-V2.md`: FOUND
- File `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-SUMMARY.md`: FOUND (this file)
- File `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-07-gate-evidence.log`: FOUND
- Commit `0575b0a` (fix(18-07): drop R5 lock-and-hold to close DB-05 P1 deadlock): FOUND in `git log`
- Commit `99e6765` (refactor(18-07): audit batch_lock_utils.hpp for Path A semantics): FOUND in `git log`
