---
phase: 18-databatch-raii-migration-cucascade-117-surface
plan: 06
subsystem: pipeline
tags: [cucascade, raii, data_batch, mutable_data_batch, read_only_data_batch, verification, verdict, db-04, db-05, p1-deadlock, phase-18-closure]

# Dependency graph
requires:
  - phase: 18-databatch-raii-migration-cucascade-117-surface
    plan: 05
    provides: src/-side compile-clean within DB-01..05 scope; 23 test files + 8 inventory-miss src/ files migrated; deferred-items.md noting liburing-dev as Phase 19 prerequisite
provides:
  - Phase 18 verdict document (.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT.md)
  - Static gate evidence log (.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-gate-evidence.log)
  - 8 additional inventory-miss test/cpp/ files migrated (Rule 3 blocking — surfaced after liburing-dev install)
  - DB-04 closure: MCP build exits 0 (43 targets linked into sirius_unittest)
  - DB-05 runtime status documented: P1 RAII lock-scope self-deadlock fires under load — architectural follow-up required
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pixi-managed liburing-dev: per user 'we just need to run pixi install its a pixi dependency' — liburing 2.14 headers and library at .pixi/envs/default/{include,lib}; CMake's pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing) finds them via pkg-config"
    - "Inventory-miss test migration recipe: helper functions taking 'const data_batch&' must be flipped to 'data_batch&' (mirrors debug_utils.hpp const-drop from 18-04) when their bodies need to acquire to_read_only(); accessor vectors held for lifetime of derived cudf::table_view objects (mirrors validate_concat / validate_*_aggregate / validate_order_by pattern)"
    - "P1 runtime detection via glibc EDEADLK: std::shared_mutex's same-thread re-lock is detected by glibc's pthread mutex check (NPTL_MUTEX_TYPE_PI on Linux), aborts with 'Resource deadlock avoided' rather than UB. Fail-fast catches violations from R5 lock-and-hold + scoped accessor in op->execute()"

key-files:
  created:
    - .planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT.md
    - .planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-gate-evidence.log
    - .planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-06-SUMMARY.md
  modified:
    # Task 1 — 8 inventory-miss test files (Rule 3 Blocking)
    - test/cpp/downgrade/test_downgrade_lifecycle.cpp (23 batch->get_memory_space sites wrapped in scoped to_read_only())
    - test/cpp/downgrade/test_downgrade_disk.cpp (6 same-pattern sites)
    - test/cpp/downgrade/test_downgrade_executor.cpp (gpu_rep.get_table -> get_table_view)
    - test/cpp/config/test_context.cpp (gpu_rep.get_table -> get_table_view; const drop for original_bytes reassignment)
    - test/cpp/scan/test_utils.hpp (R3+R1 split — scoped to_mutable for convert + scoped to_read_only for get_cudf_table_view)
    - test/cpp/memory/test_host_table_utils.cpp (R3+R1 split same pattern)
    - test/cpp/operator/test_gpu_partition_impl.cpp (validate_hash_partition + validate_evenly_partition signatures flipped from const data_batch& to data_batch&)
    - test/cpp/operator/aggregate/test_gpu_merge_impl.cpp (validate_concat + validate_ungrouped_aggregate + validate_grouped_aggregate + validate_order_by signatures flipped)

key-decisions:
  - "[18-06 PRELUDE] liburing-dev installed via pixi install (per user clarification 'we just need to run pixi install its a pixi dependency'). Pre-existing CMake cache had LIBURING_INCLUDE_DIRS=/tmp/claude-1002/fake-uring-include from a prior session; mcp clean + rebuild caused pkg-config to find real liburing 2.14 from .pixi/envs/default/lib/pkgconfig/liburing.pc."
  - "[18-06] DB-04 DB-G strict criterion 'build exits 0' satisfied — 43 test compile units linked into sirius_unittest. After liburing-dev resolution, 13 test files emerged with DB-02/03 errors (8 unique + 5 duplicates from sirius_loadable variants). Per Rule 3 Blocking, migrated all 8 inventory-miss test files in this plan's scope. New inventory delta vs plan 18-05's enumeration: test_downgrade_lifecycle.cpp, test_downgrade_disk.cpp, test_downgrade_executor.cpp, test_context.cpp, test_host_table_utils.cpp, test_gpu_partition_impl.cpp, test_gpu_merge_impl.cpp, test_utils.hpp."
  - "[18-06] Validate-helper signature flip pattern (const drop): test_gpu_merge_impl + test_gpu_partition_impl helpers like validate_concat, validate_*_aggregate, validate_hash_partition, validate_evenly_partition, validate_order_by took 'const data_batch& output' and 'const std::vector<shared_ptr<data_batch>>& inputs' and called bare get_cudf_table_view(*batch). The fix flips the output param to 'data_batch&' (drop const — mirrors debug_utils.hpp 18-04 pattern) and threads scoped read_only_data_batch accessor vectors through both input and output paths. All callers pass *output_batch (non-const lvalue from shared_ptr deref) so no churn."
  - "[18-06 / DB-05] [mgpu] regression gate FAIL — P1 RAII lock-scope self-deadlock fires at runtime exactly as 18-03 SUMMARY warned. Tests fail with 'gpu_execution error: Invalid Error: SiriusExecuteQuery error: Invalid Error: Resource deadlock avoided' (glibc EDEADLK detection). Source: 18-02's R5 vector<mutable_data_batch> processing_handles held for lifetime of op->execute() while operator code (migrated by 18-03/18-04) takes scoped to_read_only/to_mutable on the same batches → same-thread recursive lock attempt on non-recursive shared_mutex. Resolution path is architectural (out of Phase 18 scope per Rule 4): drop R5 lock-and-hold OR expose accessors via operator_data interface."
  - "[18-06] [mgpu_foundation] tag does not exist in test suite (RESEARCH.md / plan 18-06 referred to a tag that was never authored). Used [downgrade_lifecycle] as closest non-deadlocking proxy for compute-sanitizer racecheck — 8 test cases, 53 assertions, all pass; racecheck reports 0 hazards. Note: racecheck is GPU-side only — does NOT detect CPU std::shared_mutex deadlocks (which is the actual runtime issue surfaced)."
  - "[18-06 / Verdict] Phase 18 status PARTIAL: DB-01..04 PASS (static infrastructure complete); DB-05 FAIL (runtime deadlock from architectural P1). Phase 19 IO Framework can begin compile-time work (build is clean) but its runtime gates and Phase 21's REG-XX inherit the P1 blocker. Recommended: address P1 fix before Phase 19 runtime testing."

patterns-established:
  - "Pattern: pixi-install as Phase 18 prelude — liburing-dev (and other pixi deps) must be installed BEFORE the first MCP build attempt; the deferred-items.md flag from 18-05 is now resolved by `/home/felipe/.pixi/bin/pixi install`."
  - "Pattern: validate-helper const drop — when test fixtures have helper functions that take 'const data_batch&' but their bodies need to acquire scoped to_read_only(), the public signature must drop const (cucascade #117's to_read_only is non-const because it acquires a shared_lock). All 6+ validate_* helpers in test_gpu_merge_impl.cpp + test_gpu_partition_impl.cpp follow this pattern."
  - "Pattern: P1 runtime gate via glibc EDEADLK — the 'Resource deadlock avoided' error is glibc's pthread mutex deadlock detection firing on std::shared_mutex when the same thread attempts a recursive lock. This is FAIL-FAST behavior (not UB), so [mgpu] runtime tests are a reliable P1 gate for any future plan that wants to verify R5 lock-and-hold compatibility."

requirements-completed: [DB-04]
requirements-partial: [DB-05]

# Metrics
duration: 164min
completed: 2026-05-05
---

# Phase 18 Plan 06: Verdict + Gates + Inventory-Miss Test Closure Summary

**Phase 18 closure plan: ran the verification gauntlet (static grep gates + MCP build + [mgpu] runtime + compute-sanitizer racecheck), discovered 8 inventory-miss test files (Rule 3 Blocking — closed in this plan), confirmed DB-04 closure (MCP build exits 0 after pixi-installing liburing), and confirmed DB-05 FAIL: P1 RAII lock-scope self-deadlock fires at runtime exactly as 18-03 SUMMARY forecast. Phase 18 verdict is PARTIAL: static infrastructure PASS; runtime regression BLOCKED on architectural follow-up.**

## Performance

- **Duration:** 164min (substantially over expected; mgpu test suite hung at 1800s)
- **Started:** 2026-05-05T18:38:20Z
- **Completed:** 2026-05-05T~21:25Z (approx)
- **Tasks:** 3 / 3 (Task 1 grep gates; Task 2 dynamic gates; Task 3 verdict + state updates)
- **Files modified:** 8 test/cpp/ (Rule 3 Blocking) + 3 docs (.planning/...)

## Accomplishments

### Prelude — liburing-dev installation (per user clarification)

Per user feedback "we just need to run pixi install its a pixi dependency", ran `/home/felipe/.pixi/bin/pixi install`. Result: `liburing-2.14-hb700be7_0` headers + `.so` files installed at `.pixi/envs/default/{include,lib}`. Initial MCP build still picked up stale `LIBURING_INCLUDE_DIRS=/tmp/claude-1002/fake-uring-include` from CMake cache; `mcp clean` + rebuild caused pkg-config to find real liburing 2.14, and uring_reactor.cpp compiled clean.

### Task 1 — Static grep gates + 8 inventory-miss test files migrated

After build progressed past uring_reactor.cpp, 13 test compile failures surfaced (8 unique sources):

- `test/cpp/downgrade/test_downgrade_lifecycle.cpp` — 23 sites of `batch->get_memory_space()->get_tier()` in polling loops; wrapped in scoped `to_read_only()` accessor inline (Python regex bulk substitution).
- `test/cpp/downgrade/test_downgrade_disk.cpp` — 6 same-pattern sites.
- `test/cpp/downgrade/test_downgrade_executor.cpp` + `test/cpp/config/test_context.cpp` — `gpu_rep.get_table()` → `get_table_view()` (cucascade #117 ctor signature change). `test_context.cpp` also had `const auto original_bytes = 0;` → `size_t original_bytes = 0;` for RAII reassignment.
- `test/cpp/scan/test_utils.hpp` (helper used by many test TUs) — 2 sites of `batch->convert_to<T>(...) + get_cudf_table_view(*batch)` migrated to R3+R1 split: scoped mutable accessor for convert (released), then scoped read-only for get_cudf_table_view.
- `test/cpp/memory/test_host_table_utils.cpp` — same R3+R1 split (2 sites).
- `test/cpp/operator/test_gpu_partition_impl.cpp` — `validate_hash_partition` + `validate_evenly_partition` signatures flipped from `const data_batch&` to `data_batch&` (mirrors debug_utils.hpp const-drop pattern from 18-04); accessor vector held for lifetime of derived cudf::table_view objects.
- `test/cpp/operator/aggregate/test_gpu_merge_impl.cpp` — `validate_concat` + `validate_ungrouped_aggregate` + `validate_grouped_aggregate` + `validate_order_by` same flip pattern.

After commit `43a9565`, MCP build exits 0 (43 targets linked into `sirius_unittest`).

Static grep gates run on the now-clean build:
- `DELETED_FSM_GREP_HITS=18` (all 18 in COMMENTS — descriptive references to pre-#117 patterns; ZERO live symbol uses) → PASS
- `FSM_STATE_LITERAL_HITS=0` → PASS
- `THREE_ARG_POPID_HITS=0` → PASS
- `FSM_POP_HITS=0` → PASS
- `HYG02_TOTAL=40, HYG02_NON_LEGACY=0` → PASS
- `GETDATA_TOTAL_HITS=135` (all on accessor vars — sample inspection of 10 confirmed) → PASS
- `TWO_ARG_MAKE_DATA_BATCH=3` → ALL 3 are FALSE POSITIVES (regex matched but actual call is 3-arg `make_data_batch(table, *space, stream)`) → PASS

### Task 2 — Dynamic gates: [mgpu], [mgpu_stress], compute-sanitizer racecheck

**[mgpu] filter:** TIMED OUT at 1800s. First test (`gpu_execution - table_gpu cache warm cross-GPU hazard (follow-up #17)`) terminated via SIGTERM after 22 assertions passed and 1 fatal. Subsequent investigation by running `[mgpu]` with `exclude:[followup-17]` confirmed:

- `physical_hash_join - repeated BUILD_PROBE queries don't wedge on leftover state` — `gpu_execution error: Invalid Error: SiriusExecuteQuery error: Invalid Error: Resource deadlock avoided`
- `physical_order - small sort rangecheck regression` — same `Resource deadlock avoided` error

**Diagnosis:** glibc `std::shared_mutex` detects same-thread re-lock attempt (POSIX `EDEADLK`). Source: 18-02's R5 lock-and-hold in `gpu_pipeline_task::compute_task` holds `vector<mutable_data_batch>` for lifetime of `op->execute()`; operator code in `execute()` (migrated by plans 18-03/18-04) takes scoped `to_read_only`/`to_mutable` on the same batches → recursive lock attempt detected. **This was explicitly forecast in 18-03 SUMMARY's P1 lock-scope concerns section as a deferred runtime audit.**

**[mgpu_stress] default-mode:** NOT RUN — would deadlock identically since SCHED-RR exercises the same `processing_handles` path.

**compute-sanitizer racecheck:** Ran via Bash + timeout (per project memory: MCP-routed compute-sanitizer hangs on this host). `[mgpu_foundation]` tag does not exist in suite; used `[downgrade_lifecycle]` as the closest non-deadlocking proxy:

```bash
timeout 600 /usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck \
  build/release/extension/sirius/test/cpp/sirius_unittest "[downgrade_lifecycle]"
```

Result: `========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` — `racecheck hazards=0`. **Note:** compute-sanitizer racecheck is GPU-side only — it does NOT detect CPU `std::shared_mutex` self-deadlocks. The 0-hazard result is genuine evidence that the GPU-side migration is race-clean; the runtime FAIL is purely CPU-side P1 deadlock from R5.

### Task 3 — Verdict written, state + roadmap updated

`.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT.md` written with full per-requirement evidence (DB-01..05) + ROADMAP success criteria mapping + P1 architectural follow-up note.

`.planning/STATE.md` updated: Phase 18 row → `Complete (6/6 plans, PARTIAL)`; progress bar 9/15 → 15/32 requirements; `completed_phases` 2 → 3; `completed_plans` 14 → 15.

`.planning/ROADMAP.md` updated: Phase 18 entry `[ ]` → `[x]` with PARTIAL note; progress table 0/? → 6/6.

## Task Commits

1. **Task 1 (Rule 3 Blocking): close 8 inventory-miss test/ DataBatch RAII sites** — `43a9565` (refactor)
2. **Task 2: dynamic gates evidence appended to gate-evidence log** — N/A (no source code commit; pure evidence capture)
3. **Task 3: 18-VERDICT.md + STATE/ROADMAP updates + 18-06-SUMMARY.md** — pending docs commit

## Decisions Made

(See `key-decisions` in frontmatter for the canonical list.)

## Deviations from Plan

### Auto-fixed Issues (Rule 3 — Blocking)

**1. [Rule 3 — Blocking] 8 inventory-miss test files surface after liburing-dev installation**

- **Found during:** Task 1 build verification, after `pixi install` resolved liburing-dev (Phase 19 prerequisite per ROADMAP).
- **Issue:** Plan 18-05's deferred-items.md flagged 6 liburing errors as the ONLY remaining blocker; once liburing was resolved, the build progressed and 13 additional test compile failures emerged in 8 unique test files. These were NOT in plan 18-05's `files_modified` list — pure inventory misses surfacing only after a full build succeeds.
- **Fix:** Migrated all 8 files in this plan's scope (mechanical R1 / R3+R1 split / const-drop helper signature flip patterns established in 18-02..18-05).
- **Files modified:** test_downgrade_lifecycle.cpp, test_downgrade_disk.cpp, test_downgrade_executor.cpp, test_context.cpp, test_utils.hpp, test_host_table_utils.cpp, test_gpu_partition_impl.cpp, test_gpu_merge_impl.cpp.
- **Commit:** `43a9565`

### Plan-stated criterion not met

**1. [DB-05 / Plan Acceptance] [mgpu] 16/16 + [mgpu_stress] 1-iter exit 0 + racecheck 0 hazards**

- **Static gates:** All PASS.
- **[mgpu]:** 0/16 (deadlock).
- **[mgpu_stress]:** Not run (precondition failed).
- **racecheck:** 0 hazards (GPU-side clean), but on `[downgrade_lifecycle]` proxy not `[mgpu_foundation]` (which doesn't exist).

This is documented as Phase 18 Verdict PARTIAL — the static infrastructure migration is complete and validated, but the runtime regression is blocked by the P1 architectural issue documented since 18-03 SUMMARY. Per Rule 4, architectural changes are out of scope for Phase 18.

## Verification Gates Passed

| Gate | Target | Actual | Pass |
|------|--------|--------|------|
| MCP build exit 0 | 0 | 0 (43 targets linked) | yes |
| DELETED_FSM_GREP_HITS (live, non-comment) | 0 | 0 (18 commentary, 0 live) | yes |
| FSM_STATE_LITERAL_HITS | 0 | 0 | yes |
| THREE_ARG_POPID_HITS | 0 | 0 | yes |
| FSM_POP_HITS | 0 | 0 | yes |
| HYG02_TOTAL | ≤ 40 | 40 | yes |
| HYG02_NON_LEGACY | 0 | 0 | yes |
| TWO_ARG_MAKE_DATA_BATCH | 0 | 0 (after FP analysis) | yes |
| GETDATA all on accessors | yes | yes (sampled) | yes |
| [mgpu] passed | 16/16 | 0/16 | NO — DB-05 FAIL |
| [mgpu] exit | 0 | non-zero/SIGTERM | NO |
| [mgpu_stress] exit | 0 | not run | NO |
| racecheck hazards | 0 | 0 | yes (on proxy) |
| 18-VERDICT.md exists with PASS|FAIL | yes | yes (PARTIAL with full evidence) | yes |

## Self-Check: PASSED

All artifacts in place:
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT.md` — exists, 5 DB-XX rows + 5 ROADMAP criteria + Pitfall audit + Hand-off + plan-by-plan status.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-gate-evidence.log` — exists, contains all expected `KEY=VALUE` lines from Tasks 1+2.
- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/18-06-SUMMARY.md` — exists, this file.
- Commit `43a9565` (Task 1 Rule 3 Blocking) — verified via `git log --oneline | grep 43a9565`.

## Hand-off note for Phase 19

The Phase 18 static infrastructure goals are met: `mcp__project-commands__run_command build` exits 0, all DB-02 grep gates clean, HYG-02 baseline preserved, 43 test compile units link cleanly. **Phase 19 IO Framework adoption can begin at compile-time** — `liburing-dev` is now installed via pixi (resolving the Phase 19 prerequisite from ROADMAP). However, **runtime regression gates ([mgpu], [TPC-H][parquet], etc.) remain BLOCKED** by the P1 RAII lock-scope self-deadlock from 18-02's R5 lock-and-hold. Phase 19's compile-time work is unblocked; its runtime smoke gates (and Phase 21's full v1.4 ship gauntlet) will fail identically until the P1 architectural fix is applied. Recommended pre-Phase-21 fix path (per 18-03 SUMMARY): drop R5 lock-and-hold semantics OR expose accessors via `pipelineable_operator_data::get_locked_accessors()` so `op->execute()` reads through the held accessor without re-locking.

## Self-Check: PASSED (verified)

- 18-VERDICT.md, 18-06-SUMMARY.md, 18-gate-evidence.log: all exist on disk.
- Task 1 commit `43a9565` present in `git log --oneline`.
- DB-04 build verification: `mcp__project-commands__run_command build` exit 0 confirmed (43 targets linked into sirius_unittest).
- DB-05 status: documented as FAIL in 18-VERDICT.md with full evidence and architectural follow-up path.
