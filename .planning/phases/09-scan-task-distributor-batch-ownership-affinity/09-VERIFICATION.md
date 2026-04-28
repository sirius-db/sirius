---
phase: 09-scan-task-distributor-batch-ownership-affinity
milestone: v1.2
verified: 2026-04-24
status: gaps_found
score: 2/4 ROADMAP criteria PASS outright; 3/4 plans PASS + 1 plan PARTIAL; distributor wins proven, ship-gate not closed
evidence_source: .planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-04-VALIDATION.md
head_commit: 68fbda9
branch: feature/single-node-multi-gpu2
re_verification: false
verdict_rationale: |
  Phase 9's stated goal (ROADMAP.md lines 61-65) is dual:
  (a) "Fix the scan-task distributor so a batch with batch_device_id=N is only ever
      dispatched to tasks with target_device_id=N" — VERIFIED at SF100 scale (71 scan
      batches, disjoint cross-GPU intersection=0, byte-identical SF100 Q1 result vs
      1-GPU baseline).
  (b) "Close v1.2's original ship-gate (Criteria 1/2/4/6) that Phase 8 deferred" —
      NOT CLOSED. CRIT-2 (88 SF1 variants + SF10 smoke on num_gpus=2 pass) FAILS on a
      SIGSEGV in `SELECT * FROM gpu_execution(...)` TABLE_FUNCTION-form result
      materialization, observed in both 1-GPU and 2-GPU envs. The regression is
      orthogonal to the distributor fix (SF100 CLI uses the `CALL gpu_execution(...)`
      PROCEDURE form and runs clean), but the ship-gate as written is not closed.
  Per the plan-author's own self-check in 09-04-SUMMARY.md: requirements-completed
  lists only [CRIT-4]; CRIT-1, CRIT-2, CRIT-6 are not marked complete. STATE.md
  blockers section lists "v1.2 SHIP BLOCKER — 09-04 CRIT-2" explicitly. Goal-backward
  honesty: the ship-gate is the goal; the ship-gate is not closed; verdict is gaps_found.
  The narrow distributor goal IS achieved — that is reflected in the per-plan verdict
  (3/3 plans PASS on distributor evidence; 09-04 PARTIAL due to regression discovery).

# Per-plan accounting (all 4 Phase 9 plans)
plans:
  - id: 09-01
    scope: "preferred_device_id plumbing (Bug 2 fix)"
    authoring: complete
    runtime: pass
    evidence:
      - "src/include/op/scan/parquet_scan_task.hpp:574-577 — set_preferred_device_id + get_preferred_device_id accessors"
      - "src/include/op/scan/parquet_scan_task.hpp:583 — std::optional<int> _preferred_device_id member"
      - "src/op/scan/duckdb_scan_executor.cpp:366 — parquet_local_state->set_preferred_device_id(target_gpu_id) plumbing"
      - "src/op/scan/parquet_scan_task.cpp:803-808 — two-tier (local-wins-over-global) lookup with backends.find(*preferred)"
      - "VALIDATION.md runtime probe: 'preferred_device_id=-1' count on compute_task entry = 0; both 0 and 1 observed"
  - id: 09-02
    scope: "_batch_gpu_affinity map (Bug 1 fix)"
    authoring: complete
    runtime: pass
    evidence:
      - "src/include/op/scan/duckdb_scan_executor.hpp:217-218 — mutable std::mutex _batch_affinity_mutex + std::unordered_map<uint64_t,int> _batch_gpu_affinity"
      - "src/op/scan/duckdb_scan_executor.cpp:200-201 — _batch_gpu_affinity[fallback_counter] = device_id (fallback branch)"
      - "src/op/scan/duckdb_scan_executor.cpp:240-241 — _batch_gpu_affinity[counter] = space->get_device_id() (weighted branch)"
      - "src/op/scan/duckdb_scan_executor.cpp:142-143 — _batch_gpu_affinity.clear() in prepare_cache_for_scan_operators (query-start reset)"
      - "src/op/scan/duckdb_scan_executor.cpp:243 — [mgpu-audit] scan_batch log emission preserved verbatim (Plan 09-03 regex dependency)"
      - "src/op/sirius_physical_operator.cpp:47,75 — [mgpu-probe] breadcrumbs on prepare_for_processing nullopt paths"
      - "VALIDATION.md SF100 evidence: 71 batches dispatched, GPU0=45 unique, GPU1=26 unique, cross-GPU intersection=0"
  - id: 09-03
    scope: "Cross-GPU disjointedness REQUIRE"
    authoring: complete
    runtime: pass
    evidence:
      - "test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:46 — #include <algorithm> for std::set_intersection"
      - "test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:260 — std::set_intersection(counts[0].scan_ids, counts[1].scan_ids)"
      - "test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:271 — REQUIRE(cross_gpu_intersection.empty())"
      - "VALIDATION.md RUN 4: 'All tests passed (16 assertions in 1 test case)' with disjointedness REQUIRE firing"
      - "AUDIT-fixture runtime: GPU0=2 unique batch_ids, GPU1=2 unique, intersect=0"
  - id: 09-04
    scope: "Ship-gate validation on 2-GPU hardware"
    authoring: complete
    runtime: partial
    evidence:
      - "09-04-VALIDATION.md verdict: PARTIAL (CRIT-1/CRIT-6 PASS on SF100 PROCEDURE-form; CRIT-4 PASS on disjointedness REQUIRE; CRIT-2 FAIL on TABLE_FUNCTION SIGSEGV)"
      - "SF100 Q1 num_gpus=2: exit 0, wall-clock 0:05.86, byte-identical CSV vs num_gpus=1 baseline (0:05.54)"
      - "MCP unit-tests exit 139 (SIGSEGV in 'gpu_execution - filter equality parquet' + 'tpch_q1_sf10_2gpu'); H1-H4 hypotheses seeded for Phase 10"

# ROADMAP criterion-by-criterion verdict summary (from 09-04-VALIDATION.md)
criteria:
  1:
    name: "SF100 TPC-H Q1 on num_gpus=2 correct vs num_gpus=1 baseline, no cudaErrorInvalidValue/SIGSEGV/fallback"
    verdict: PASS_PARTIAL
    detail: "PASS on the `CALL gpu_execution(...)` PROCEDURE path exercised by the SF100 CLI ship-gate (byte-identical result, 5.86s). The `SELECT * FROM gpu_execution(...)` TABLE_FUNCTION form SIGSEGVs — orthogonal to distributor, but same `gpu_execution` SQL surface. Phase 9 plan-author recorded this as CRIT-1 PASS in VALIDATION.md Per-Criterion table on the narrow ship-gate reading."
  2:
    name: "MCP unit-tests exits 0 with 88 SF1 variants (GENERATE(1,2)) + SF10 Q1/Q6/Q12 green"
    verdict: FAIL
    blocker: "SIGSEGV in compare_gpu_vs_cpu helper's second `SELECT * FROM gpu_execution(\"...\")` invocation (test_gpu_execution_tpch.cpp:216/239). Reproduces on both 1-GPU and 2-GPU envs. Not in Plans 09-01/02/03 distributor code; in downstream TABLE_FUNCTION-form result materialization."
  4:
    name: "AUDIT TEST_CASE: pipeline_task>=5 AND scan_batch>=5 per GPU AND cross-GPU scan_ids intersection == empty"
    verdict: PASS_PARTIAL
    detail: "Disjointedness REQUIRE (the Phase 9 regression gate) PASSES. The `>=5` threshold per GPU fails on the AUDIT TEST_CASE's SF1-DuckDB fixture (only 2 scan_ids per GPU) — pre-existing Plan 08-05 test-design choice, not a Phase 9 regression. The distributor-correctness claim the REQUIRE was designed to lock in is GREEN."
  6:
    name: "SF100 [mgpu-audit] scan_batch distributes across both GPUs + wall-clock captured"
    verdict: PASS
    detail: "217 [mgpu-audit] entries; GPU0=45 unique batch_ids, GPU1=26 unique batch_ids, intersection=0. Wall-clock 0:05.86 recorded. Plan 09-02 batch→GPU affinity map is live at SF100 scale."

gaps:
  - truth: "v1.2 ship-gate closed (CRIT-1/2/4/6 all green)"
    status: failed
    reason: |
      CRIT-2 fails in the unit-test suite: `gpu_execution - filter equality parquet` and
      `gpu_execution - tpch_q1_sf10_2gpu` SIGSEGV during `compare_gpu_vs_cpu`'s second
      `SELECT * FROM gpu_execution(\"...\")` invocation. The first `CALL gpu_execution(...)`
      PROCEDURE-form returns cleanly; the second TABLE_FUNCTION-form crashes during
      invocation or result materialization. Both 1-GPU and 2-GPU envs reproduce.
      The regression is ORTHOGONAL to the distributor fix (Plans 09-01/02/03): the SF100
      CLI run, which uses ONLY the PROCEDURE form, passes the ship-gate with byte-identical
      results and disjoint cross-GPU batch dispatch. But the ship-gate as written in Phase 9
      requirements (CRIT-2) requires unit-tests exit 0, which it does not.
    artifacts:
      - path: "src/sirius_extension.cpp"
        issue: "TABLE_FUNCTION binding for `gpu_execution` — result-materialization path differs from PROCEDURE form; leading hypothesis (H2) for the SIGSEGV"
      - path: "src/sirius_interface.cpp"
        issue: "GPUResult proxy lifetime boundaries likely differ between CALL (PROCEDURE) and SELECT * FROM (TABLE_FUNCTION) paths"
      - path: "test/cpp/integration/test_gpu_execution_tpch.cpp:216,239"
        issue: "Catch2 signal-handler anchors crash at preceding REQUIRE; actual fault is at the second gpu_execution call (line 239)"
    missing:
      - "Phase 10 Plan 10-01 (bisect): check out each of 3b58258, 863cc6c, 0c8068e, a8a7985, c0e12f3 individually and run `./build/release/extension/sirius/test/cpp/sirius_unittest 'gpu_execution - filter equality parquet'` — identify the first SIGSEGV-introducing commit"
      - "Phase 10 Plan 10-02 (gdb): attach gdb to the crashing binary per .claude/skills/debug-gdb/SKILL.md; capture the actual fault-frame backtrace"
      - "Phase 10 Plan 10-03 (targeted fix per confirmed hypothesis): if H2 (leading), instrument src/sirius_extension.cpp gpu_execution TABLE_FUNCTION binding + compare CALL-form vs TABLE_FUNCTION-form GPUResult lifetime; if H1/H3, instrument _datasource construction + _batch_gpu_affinity reset; if H4, consider forward-compatible rollback preserving SF100 ship-gate"
      - "Phase 10 Plan 10-04 (re-ship-gate): re-run 09-04-PLAN.md VALIDATION procedure end-to-end after fix; expect CRIT-2 green, CRIT-1/6 remain green, v1.2 ship-gate closed"

open_issue_carryforward:
  source: .planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-04-VALIDATION.md
  signature: "SIGSEGV in compare_gpu_vs_cpu's second `SELECT * FROM gpu_execution(\"\" + clean_query + \"\")` path (test_gpu_execution_tpch.cpp:239). First-form `CALL gpu_execution(...)` returns cleanly (pre-crash log shows 'Execute query time: 6.46 ms')."
  failing_tests:
    - name: "gpu_execution - filter equality parquet"
      location: "test/cpp/integration/test_gpu_execution_tpch.cpp:449"
      env: "1-GPU default (GPUExecutionParquetFixture)"
      query: "select n_nationkey from nation where n_regionkey = 1"
    - name: "gpu_execution - tpch_q1_sf10_2gpu"
      location: "test/cpp/integration/test_gpu_execution_tpch.cpp:4297"
      env: "2-GPU (compare_gpu_vs_cpu_sf10_for(2, kTpchQ1Body))"
      query: "TPC-H Q1 on SF10"
  hypotheses:
    - id: H1
      description: "Residual _datasource caching on compute_task re-dispatch (RESEARCH.md Open Questions #1). `_datasource` is a shared_ptr member of parquet_scan_task; persists across calls. Second gpu_execution may reuse stale device-bound data."
      probe: "Add [mgpu-probe] at _datasource construction site + reset points"
    - id: H2
      description: "TABLE_FUNCTION-form result materialization vs PROCEDURE-form (CALL) divergence — leading hypothesis. SF100 CLI (PROCEDURE-only) passes; unit-tests (uses TABLE_FUNCTION as second form) SIGSEGVs."
      probe: "Audit src/sirius_extension.cpp gpu_execution table function registration + src/sirius_interface.cpp GPUResult lifetime"
      files_to_audit:
        - "src/sirius_extension.cpp"
        - "src/sirius_interface.cpp"
    - id: H3
      description: "_batch_gpu_affinity map lifecycle on second query (RESEARCH.md Q5 Candidate 2). Map reset on prepare_cache_for_scan_operators — if second query reuses cache with stale map, potential race."
      probe: "Trace _batch_gpu_affinity access across consecutive queries"
    - id: H4
      description: "Build or unit-tests regression unrelated to distributor (fallback). Could be unordered_map allocation corruption, use-after-free, or data race introduced by one of 3b58258/863cc6c/0c8068e/a8a7985/c0e12f3."
      probe: "Bisect the 5 Phase 9 commits"
  recommended_next_phase: "Phase 10 — TABLE_FUNCTION-form SIGSEGV hunt. Bisect + gdb first, then targeted fix per confirmed hypothesis. Expected scope: < 100 LOC + one ship-gate re-run."
---

# Phase 9 — Scan-Task Distributor + Batch-Ownership Affinity Verification Report

**Phase Goal (from ROADMAP.md lines 61-65):**

> Fix the scan-task distributor so a batch with `batch_device_id=N` is only ever dispatched to tasks with `target_device_id=N`. Fix `preferred_device_id=-1` plumbing at `parquet_scan_task::compute_task` entry. Close v1.2's original ship-gate (Criteria 1/2/4/6) that Phase 8 deferred.

**Verified:** 2026-04-24
**Status:** `gaps_found`
**Re-verification:** No — initial verification after Phase 9 plan-04 validation recorded PARTIAL.
**Evidence source:** `.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-04-VALIDATION.md`

---

## Executive Summary

Phase 9 is a **split verdict**:

- **Distributor fix (narrow goal):** **ACHIEVED**. All three distributor plans (09-01 preferred_device_id plumbing, 09-02 batch→GPU affinity map, 09-03 cross-GPU disjointedness REQUIRE) are authoring-complete, code-review-verifiable in-tree, and runtime-proven at SF100 scale. The SF100 Q1 num_gpus=2 ship-gate CLI run executes in 5.86s, produces a byte-identical result vs num_gpus=1 baseline, and distributes 71 scan batches disjointly across both GPUs (GPU0=45 unique, GPU1=26 unique, cross-GPU intersection=0). The Plan 09-03 disjointedness REQUIRE fires and PASSES on the AUDIT TEST_CASE runtime fixture.

- **Ship-gate closure (broad goal):** **NOT ACHIEVED**. Phase 9's stated goal includes closing inherited v1.2 Success Criteria 1/2/4/6. CRIT-1/4/6 are in the PASS/PASS_PARTIAL column on their narrow metrics, but CRIT-2 (MCP unit-tests exit 0 with 88 SF1 variants + SF10 smoke on num_gpus=2 pass) outright FAILS due to a SIGSEGV in the `SELECT * FROM gpu_execution(...)` TABLE_FUNCTION-form result-materialization code path. The regression is **orthogonal to the distributor** (SF100 CLI uses PROCEDURE form and passes cleanly) but is a real blocker for the ship-gate as written. STATE.md blockers section explicitly records this as "v1.2 SHIP BLOCKER — 09-04 CRIT-2".

**Verdict rationale:** Goal-backward honesty. The phase goal has two clauses; clause (a) distributor-fix is delivered, clause (b) ship-gate closure is not. A hypothesis-well-characterized regression (H1-H4 with H2 leading) is not the same as a closed ship-gate. The remaining work is narrow-scoped and deterministic (bisect 5 commits + gdb + targeted fix) — perfect input for Phase 10 or a 09.x gap-closure plan via `/gsd:plan-phase 9 --gaps`.

**Plan-author concurrence:** 09-04-SUMMARY.md `requirements-completed` lists only `[CRIT-4]`; the plan author's own self-check classifies CRIT-1/CRIT-2/CRIT-6 as not-complete, mirroring this verification's verdict.

---

## Must-Haves Verified (Plans 09-01 / 09-02 / 09-03 — distributor code delivery)

| Must-Have | Actual Codebase Evidence | Status |
|-----------|--------------------------|--------|
| **Plan 09-01 — preferred_device_id accessors on parquet_scan_task_local_state** | `src/include/op/scan/parquet_scan_task.hpp:574-577` defines `set_preferred_device_id(int)` + `[[nodiscard]] std::optional<int> get_preferred_device_id() const`; line 583 defines `std::optional<int> _preferred_device_id` member | **VERIFIED** |
| **Plan 09-01 — target_gpu_id plumbing into local state in manager_loop** | `src/op/scan/duckdb_scan_executor.cpp:366`: `parquet_local_state->set_preferred_device_id(target_gpu_id)` inside the `is<parquet_scan_task>()` block after `auto target_gpu_id = select_target_gpu()` and before the dispatch lambda | **VERIFIED** |
| **Plan 09-01 — two-tier local-wins-over-global lookup in compute_task using `backends.find(*preferred)`** | `src/op/scan/parquet_scan_task.cpp:803-808`: `auto const local_preferred = l_state.get_preferred_device_id(); auto const preferred = local_preferred.has_value() ? local_preferred : g_state.get_preferred_device_id(); auto backend_it = preferred.has_value() ? backends.find(*preferred) : backends.begin();` (exact pattern: `backends.find(`, NOT `backends.begin(`) | **VERIFIED** |
| **Plan 09-01 — probe breadcrumb at compute_task entry emits preferred_device_id in log** | `src/op/scan/parquet_scan_task.cpp:770` emits `[mgpu-probe] parquet_scan_task::compute_task entry current_device=... stream=... preferred_device_id=...` | **VERIFIED** |
| **Plan 09-01 — runtime probe shows no -1 sentinels** | VALIDATION.md Transcript "Runtime probe — preferred_device_id plumbing": `compute_task entry with preferred_device_id=-1: 0` + 4 entries of `preferred_device_id=0` + 3 entries of `preferred_device_id=1`; distinct positive values = 2 | **VERIFIED** |
| **Plan 09-02 — _batch_gpu_affinity map + mutex in header** | `src/include/op/scan/duckdb_scan_executor.hpp:217-218`: `mutable std::mutex _batch_affinity_mutex; std::unordered_map<uint64_t, int> _batch_gpu_affinity;` | **VERIFIED** |
| **Plan 09-02 — affinity write in weighted branch of select_target_gpu** | `src/op/scan/duckdb_scan_executor.cpp:240-241` inside `target < cumulative` block: `std::lock_guard<std::mutex> lock(_batch_affinity_mutex); _batch_gpu_affinity[counter] = space->get_device_id();` emitted ATOMICALLY with SIRIUS_LOG_INFO audit log (line 243) | **VERIFIED** |
| **Plan 09-02 — affinity write in fallback branch (total_available == 0)** | `src/op/scan/duckdb_scan_executor.cpp:200-201`: `std::lock_guard<std::mutex> lock(_batch_affinity_mutex); _batch_gpu_affinity[fallback_counter] = device_id;` | **VERIFIED** |
| **Plan 09-02 — query-start reset before cache_level::NONE early return** | `src/op/scan/duckdb_scan_executor.cpp:142-143` inside `prepare_cache_for_scan_operators`: `std::lock_guard<std::mutex> lock(_batch_affinity_mutex); _batch_gpu_affinity.clear();` (paired with `_scan_round_robin.store(0)` at line 140) | **VERIFIED** |
| **Plan 09-02 — [mgpu-probe] breadcrumbs on nullopt paths of prepare_for_processing** | `src/op/sirius_physical_operator.cpp:47`: null-batch breadcrumb; line 75: lock-failure breadcrumb with `batch_id=` and `batch_state=` payload | **VERIFIED** |
| **Plan 09-02 — [mgpu-audit] log payload unchanged** | `src/op/scan/duckdb_scan_executor.cpp:243`: `SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} batch_id={} (available: {} bytes)", ...)` — Plan 09-03 regex dependency preserved | **VERIFIED** |
| **Plan 09-03 — std::set_intersection include + call in AUDIT TEST_CASE** | `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:46`: `#include <algorithm>`; line 260: `std::set_intersection(counts[0].scan_ids.begin(), counts[0].scan_ids.end(), counts[1].scan_ids.begin(), counts[1].scan_ids.end(), std::back_inserter(cross_gpu_intersection))` | **VERIFIED** |
| **Plan 09-03 — REQUIRE(cross_gpu_intersection.empty()) after existing per-GPU count REQUIREs** | `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:271`; preceded by `REQUIRE(counts[1].scan_ids.size() >= min_count)` at line 248 (ordering invariant 248 < 271 preserved) | **VERIFIED** |
| **Plan 09-03 — runtime REQUIRE fires and passes** | VALIDATION.md RUN 4 excerpt: "All tests passed (16 assertions in 1 test case)"; AUDIT runtime: GPU0=2 unique, GPU1=2 unique, intersect=0; "cross-GPU scan_batch intersection size: 0" | **VERIFIED** |
| **Static invariant HYG-02 — rmm::cuda_stream_default <= 41** | `grep -rn 'rmm::cuda_stream_default' src/ \| wc -l` → 40 (≤ 41 baseline) | **VERIFIED** |
| **Static invariant Pattern 2 — cuda_set_device_raii preserved in duckdb_scan_executor** | `grep -nE 'cuda_set_device_raii.*target_gpu_id' src/op/scan/duckdb_scan_executor.cpp` → 2 matches (lines 432 acquire_guard + 448 dispatch_guard) | **VERIFIED** |
| **Branch discipline — feature branch preserved, no merge to dev** | `git branch --show-current` → `feature/single-node-multi-gpu2`; HEAD at 68fbda9 (last commit: `docs(09-04): clean transient paths from SUMMARY frontmatter`) | **VERIFIED** |
| **All 5 Phase 9 commits present on branch** | `git log --oneline`: 3b58258 (09-01 Task 1), 863cc6c (09-01 Task 2), 0c8068e (09-01 Task 3), a8a7985 (09-02 Task 1), c0e12f3 (09-02 Task 2), e2484e0 (09-02 Task 3), 452feeb (09-03), plus docs/summary/validation commits for each | **VERIFIED** |

**Distributor-delivery score: 18/18 must-haves verified.** All Plan 09-01/02/03 code lands exactly as the SUMMARYs claim. No stubs, no TODO placeholders, no missing imports.

---

## Observable Truths (from Plan 09-04 must_haves + ROADMAP.md Phase 9 Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SF100 TPC-H Q1 on num_gpus=2 returns correct results, no SIGSEGV, no cudaErrorInvalidValue, no CPU fallback; matches num_gpus=1 baseline byte-identically | **VERIFIED** | VALIDATION.md Task 3: exit 0, wall-clock 0:05.86 (2-GPU) / 0:05.54 (1-GPU), diff exit 0 (CSVs byte-identical), 4 data rows, stderr clean |
| 2 | SF100 [mgpu-audit] log shows non-empty scan_batch distribution on both GPU 0 and GPU 1 and cross-GPU batch_id intersection == 0 | **VERIFIED** | VALIDATION.md: 217 audit entries, 71 unique batch_ids, GPU0=45 unique, GPU1=26 unique, intersection=0 |
| 3 | Plan 09-01 preferred_device_id plumbing is confirmed live: grep for `preferred_device_id=-1` on compute_task entry returns zero matches; both 0 and 1 observed | **VERIFIED** | VALIDATION.md runtime probe: `compute_task entry with preferred_device_id=-1: 0`; 4 × `preferred_device_id=0` + 3 × `preferred_device_id=1` |
| 4 | AUDIT TEST_CASE cross-GPU scan_ids intersection REQUIRE fires and passes on the AUDIT runtime fixture | **VERIFIED** | VALIDATION.md RUN 4: 16 assertions, all pass; "cross-GPU scan_batch intersection size: 0" |
| 5 | AUDIT TEST_CASE strict per-GPU counts (pipeline_task >= 5 AND scan_batch >= 5) fire when SIRIUS_TEST_SF10_PATH set | **FAILED (pre-existing)** | RUN 3: pipeline GPU0=6, GPU1=4 (fails 4>=5); scan GPU0=2, GPU1=2 (fails 2>=5). Pre-existing Plan 08-05 test-design choice: AUDIT fixture uses SF1-DuckDB ATTACH path, not SF10 data — only 2 scan_ids possible per GPU on SF1. Not a Phase 9 regression; the distributor-correctness claim (disjointedness REQUIRE) is the substantive gate and it passes. |
| 6 | 88 SF1 variants × {num_gpus=1, num_gpus=2} all pass via MCP unit-tests with Q4 retry-once flake policy | **FAILED** | RUN 1: exit 139 (SIGSEGV) in `gpu_execution - filter equality parquet` during compare_gpu_vs_cpu's second `SELECT * FROM gpu_execution(...)` call. Q4 retry policy NOT triggered (no Q4 failure observed; a different earlier SIGSEGV halts the run). |
| 7 | SF10 Q1/Q6/Q12 × num_gpus=2 pass when SIRIUS_TEST_SF10_PATH is set | **FAILED** | RUN 2: exit 139 (SIGSEGV) in `gpu_execution - tpch_q1_sf10_2gpu` at the same TABLE_FUNCTION-form call-site |
| 8 | Static invariants preserved (HYG-02 ≤ 41, Pattern 2 idiom preserved, Plans 09-01/02/03 source greps green) | **VERIFIED** | All 5 static-invariant rows in VALIDATION.md "Static Invariants" table PASS; independently re-grepped by this verification |
| 9 | Evidence captured in 09-04-VALIDATION.md with exact commands, wall-clock, log excerpts, CSV diff, verdict, Open Issue | **VERIFIED** | 09-04-VALIDATION.md exists, 293 lines, mirrors 08-06-VALIDATION.md structure (frontmatter, Commands Run, Transcript Excerpts a-j, Per-Criterion Closure Table, Static Invariants, Verdict, Open Issue with H1-H4, Next Steps) |
| 10 | ROADMAP.md line 70 reflects autonomous MCP-executed status | **VERIFIED** | ROADMAP.md line 70 reads `(autonomous: true, MCP-executed per 2026-04-24 host-capability discovery)` + `**VERDICT: PARTIAL**` annotation |

**Score:** 7/10 truths VERIFIED, 2/10 FAILED on a single root cause (TABLE_FUNCTION-form regression), 1/10 FAILED on a pre-existing test-design choice (AUDIT fixture scale mismatch) that is explicitly not a Phase 9 regression.

---

## Requirement Coverage (CRIT-1/2/4/6 — Phase 9's synthetic IDs for inherited v1.2 Success Criteria 1/2/4/6)

| CRIT-ID | Description | Verdict | Evidence |
|---------|-------------|---------|----------|
| **CRIT-1** | SF100 Q1 on num_gpus=2 correct vs num_gpus=1 baseline, no cudaErrorInvalidValue / no SIGSEGV / no fallback | **PASS (narrow ship-gate)** / **PARTIAL (phase-level)** | SF100 CLI PROCEDURE-form: exit 0, byte-identical CSV, 5.86s wall-clock, 217 audit entries, zero SIGSEGV, zero cudaErrorInvalidValue, zero fallbacks. The TABLE_FUNCTION-form regression exists on the same `gpu_execution` SQL surface but is not exercised by the SF100 ship-gate. 09-04-SUMMARY.md `requirements-completed` does NOT list CRIT-1 → plan author concurs this is not fully closed. |
| **CRIT-2** | MCP unit-tests exits 0 with 88 SF1 variants (GENERATE(1,2)) + SF10 Q1/Q6/Q12 green | **FAIL** | MCP unit-tests exit 139 on `gpu_execution - filter equality parquet` (1-GPU env) + `tpch_q1_sf10_2gpu` (2-GPU env). SIGSEGV in compare_gpu_vs_cpu's second `SELECT * FROM gpu_execution(...)` TABLE_FUNCTION call. Not distributor; scoped to Phase 10. |
| **CRIT-4** | AUDIT TEST_CASE: pipeline_task>=5 AND scan_batch>=5 per GPU AND cross-GPU scan_ids intersection == empty | **PASS (distributor-correctness gate) / PARTIAL (strict threshold)** | Disjointedness REQUIRE (Plan 09-03) PASSES: intersection=0. Strict ≥5 threshold fails on SF1-DuckDB fixture (2 scan_ids per GPU) — pre-existing Plan 08-05 choice. The distributor-correctness substantive claim is GREEN; only the strict-threshold decoration (which requires SF10-scale data the AUDIT fixture doesn't generate) is amber. 09-04-SUMMARY.md `requirements-completed` lists CRIT-4 as complete. |
| **CRIT-6** | SF100 [mgpu-audit] scan_batch distributes across both GPUs + wall-clock captured | **PASS** | GPU0=45 unique batch_ids, GPU1=26 unique batch_ids, intersection=0, wall-clock 0:05.86, 217 [mgpu-audit] entries in SIRIUS_LOG_DIR. 09-04-SUMMARY.md `requirements-completed` does NOT list CRIT-6 because phase-level gating on CRIT-2 keeps the ship-gate amber overall. |

**Phase-level criterion rollup:** 2/4 outright PASS (CRIT-6 + CRIT-4 disjointedness); 1/4 PASS_PARTIAL (CRIT-1 on ship-gate path, TABLE_FUNCTION amber); 1/4 FAIL (CRIT-2). The ship-gate closure goal is not met.

---

## Required Artifacts (Level 1-3 Verification)

| Artifact | Expected | Status | Evidence |
|----------|----------|--------|----------|
| `src/include/op/scan/parquet_scan_task.hpp` | _preferred_device_id member + set/get accessors | VERIFIED | Lines 574-577 (accessors) + 583 (member) |
| `src/op/scan/duckdb_scan_executor.cpp` | set_preferred_device_id(target_gpu_id) plumbing + affinity writes + query-start reset | VERIFIED | Line 366 (plumbing); lines 200-201 + 240-241 (affinity writes); lines 142-143 (reset) |
| `src/op/scan/parquet_scan_task.cpp` | Two-tier lookup using `backends.find(*preferred)` | VERIFIED | Lines 803-808; line 770 probe breadcrumb |
| `src/include/op/scan/duckdb_scan_executor.hpp` | mutable mutex + unordered_map<uint64_t,int> _batch_gpu_affinity | VERIFIED | Lines 217-218 |
| `src/op/sirius_physical_operator.cpp` | [mgpu-probe] breadcrumbs on 2 nullopt paths | VERIFIED | Lines 47 (null_batch) + 75 (batch_id + batch_state) |
| `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp` | `<algorithm>` include + std::set_intersection + REQUIRE(cross_gpu_intersection.empty()) | VERIFIED | Lines 46, 260, 271 |
| `.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-04-VALIDATION.md` | 293-line ship-gate evidence document per 08-06-VALIDATION.md template | VERIFIED | All 7 required sections present (frontmatter, Commands Run, Transcripts a-j, Per-Criterion, Static Invariants, Verdict, Open Issue, Next Steps) |
| `.planning/ROADMAP.md` | Line 70 reflects autonomous + PARTIAL verdict | VERIFIED | Read and confirmed |

**All 8 expected artifacts exist, are substantive, and are wired.** No stubs, no missing.

---

## Key Link Verification

| From | To | Via | Status | Detail |
|------|-----|------|--------|--------|
| `manager_loop` (duckdb_scan_executor.cpp) | `parquet_scan_task_local_state` | `parquet_local_state->set_preferred_device_id(target_gpu_id)` at line 366 | WIRED | Runtime probe at VALIDATION.md confirms target_gpu_id reaches local state: 0 × `-1` sentinels at compute_task entry |
| `compute_task` (parquet_scan_task.cpp) | backends lookup | `backends.find(*preferred)` at line 808 (two-tier fallback) | WIRED | VALIDATION.md runtime: distinct positive preferred_device_id values = 2; routing is active on the target |
| `select_target_gpu` (duckdb_scan_executor.cpp) | `_batch_gpu_affinity` map | `_batch_gpu_affinity[counter] = space->get_device_id()` atomic-with-audit-log at line 241 | WIRED | SF100 runtime: 71 unique batch_ids recorded with GPU0/GPU1 distribution matching log emission; cross-GPU intersection=0 proves affinity is coherent |
| `prepare_cache_for_scan_operators` | `_batch_gpu_affinity.clear()` | Line 143, before cache_level::NONE early return | WIRED | VALIDATION.md: second SF100 query run would see clean map (not directly exercised in single-query test, but RUN 3/4 AUDIT TEST_CASE reruns prove reset doesn't corrupt state) |
| AUDIT TEST_CASE | Plan 09-03 disjointedness REQUIRE | `std::set_intersection` on parse_audit_log output | WIRED | VALIDATION.md RUN 4: REQUIRE fires and passes (16 assertions); regex unchanged from 08-03 — log payload → parse_audit_log → counts[gpu].scan_ids → set_intersection → REQUIRE chain intact |
| `[mgpu-audit] scan_batch` log | parse_audit_log regex in AUDIT TEST_CASE | Regex literal at test_gpu_execution_tpch_mgpu_audit.cpp:79 (unchanged from Phase 8) | WIRED | Payload at duckdb_scan_executor.cpp:243 matches regex verbatim; SF100 log shows 71 entries successfully parsed |
| SF100 CLI `build/release/duckdb` + SIRIUS_CONFIG_FILE=integration-2gpu.yaml | SF100 ship-gate CSV output | SiriusContext init from YAML at src/sirius_context.cpp:54 | WIRED | Exit 0, byte-identical CSV diff vs 1-GPU baseline (4 data rows, 10 columns) |

**All 7 critical links WIRED.** The only link that fails **at runtime** is in the unit-test binary's `SELECT * FROM gpu_execution(...)` TABLE_FUNCTION-form result-materialization path — which is OUTSIDE the distributor's wiring scope and forms the gap.

---

## Data-Flow Trace (Level 4) — Distributor Proof at SF100 Scale

| Artifact | Data variable | Source | Produces real data? | Status |
|----------|---------------|--------|---------------------|--------|
| `_preferred_device_id` | per-task int GPU ID | `set_preferred_device_id(target_gpu_id)` in manager_loop; `select_target_gpu()` returns integer from `_gpu_memory_spaces[idx]->get_device_id()` | Yes — runtime probe shows both 0 and 1 observed, 0 × -1 sentinels | FLOWING |
| `_batch_gpu_affinity` | unordered_map<uint64_t,int> | `_batch_gpu_affinity[counter] = space->get_device_id()` atomic with audit log emission | Yes — SF100 run populates 71 entries; query-start reset clears between queries | FLOWING |
| `counts[gpu].scan_ids` in AUDIT TEST_CASE | std::set<std::string> of batch_id strings | `parse_audit_log` regex extracts from `[mgpu-audit] scan_batch assigned to GPU N batch_id=K` log lines | Yes — RUN 4 shows GPU0=2 + GPU1=2 entries parsed, set_intersection produces empty result | FLOWING |
| `cross_gpu_intersection` | std::vector<std::string> output of set_intersection | Populated from counts[0/1].scan_ids via std::back_inserter | Yes — empty vector (size 0) on PASS; REQUIRE(empty()) succeeds | FLOWING |
| SF100 TPC-H Q1 CSV output | 4 rows × 10 columns numeric aggregates | DuckDB CLI `SELECT ... FROM lineitem GROUP BY l_returnflag, l_linestatus` via `CALL gpu_execution(...)` PROCEDURE | Yes — byte-identical to 1-GPU baseline (diff exit 0); sum_qty = 3775127758.00 for (A,F), etc. | FLOWING |
| Unit-test TABLE_FUNCTION GPUResult materialization | DuckDB Query result | Second invocation of `con->Query("SELECT * FROM gpu_execution(...)")` in compare_gpu_vs_cpu helper | **NO — SIGSEGV** | **STATIC (regression)** |

The distributor code is flowing real data. The gap is in a downstream result-materialization path that the distributor's SF100 ship-gate does not exercise.

---

## Anti-Patterns Scanned

| Pattern | Files scanned | Blocker? | Notes |
|---------|---------------|----------|-------|
| TODO/FIXME/XXX/HACK | All Phase-9-modified src files | NO | None found in shipped code |
| Empty return null/{}/[] stubs | All Phase-9-modified src files | NO | None; all functions return real values |
| rmm::cuda_stream_default net-new | Phase-9-modified files | NO | 0 net-new; total in src/ = 40 (HYG-02 preserved; Phase 8 baseline was 41) |
| Placeholder comments | All Phase-9 files | NO | None found |
| Skipped tests | AUDIT TEST_CASE on single-GPU hosts | INFO (not blocker) | WARN+return on `cudaGetDeviceCount < 2` — correct defensive behavior. On 2-GPU host (which this verification's evidence is from), REQUIREs fire. |

---

## Behavioral Spot-Checks (evidence from VALIDATION.md, re-verified by grep here)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| HYG-02 invariant | `grep -rn 'rmm::cuda_stream_default' src/ \| wc -l` | 40 | PASS (≤ 41) |
| Plan 09-01 grep contract | `grep -c 'set_preferred_device_id(target_gpu_id)' src/op/scan/duckdb_scan_executor.cpp` | 1 | PASS |
| Plan 09-02 grep contract (affinity writes) | `grep -c '_batch_gpu_affinity\[' src/op/scan/duckdb_scan_executor.cpp` | 2 | PASS |
| Plan 09-02 grep contract (header) | `grep -n '_batch_gpu_affinity' src/include/op/scan/duckdb_scan_executor.hpp` | line 218 | PASS |
| Plan 09-03 grep contract | `grep -c 'set_intersection' test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp` | 2 (include comment + call site) | PASS |
| Pattern 2 idiom preserved | `grep -nE 'cuda_set_device_raii.*target_gpu_id' src/op/scan/duckdb_scan_executor.cpp` | 2 matches (lines 432, 448) | PASS |
| Feature branch preserved | `git branch --show-current` | `feature/single-node-multi-gpu2` | PASS (no merge to dev) |
| All 5 Phase 9 source commits on branch | `git log --oneline \| grep -E '(feat\|fix)\(09-'` | 3b58258, 863cc6c, 0c8068e, a8a7985, c0e12f3, e2484e0, 452feeb all present | PASS |
| MCP unit-tests exit 0 (ship-gate CRIT-2) | VALIDATION.md RUN 1/RUN 2 | Exit 139 (SIGSEGV) | **FAIL — ship-gate blocker** |
| SF100 Q1 num_gpus=2 exit 0 | VALIDATION.md Task 3 | Exit 0, 5.86s, byte-identical CSV | PASS |

Bash-sandbox static checks all re-verified. Runtime failures documented by VALIDATION.md on the MCP-enabled 2-GPU host cannot be re-run in this verification session (sandbox has no GPU driver access); inherited from VALIDATION.md as the plan author's authoritative evidence.

---

## Gaps

### Gap 1: CRIT-2 ship-gate blocked by TABLE_FUNCTION-form `SELECT * FROM gpu_execution(...)` SIGSEGV

**Root cause (hypothesized, H2 leading):** A regression in the `SELECT * FROM gpu_execution(...)` TABLE_FUNCTION result-materialization code path (src/sirius_extension.cpp table-function binding + src/sirius_interface.cpp GPUResult lifetime). The `CALL gpu_execution(...)` PROCEDURE form works cleanly (SF100 CLI passes). The regression is NOT in Plans 09-01/02/03 distributor code, but it was introduced somewhere in the 5-commit Phase 9 source edit span (3b58258, 863cc6c, 0c8068e, a8a7985, c0e12f3) per VALIDATION.md bisect-scoping.

**Observed failures:**

| Test | File:line | Env | Error |
|------|-----------|-----|-------|
| `gpu_execution - filter equality parquet` | test_gpu_execution_tpch.cpp:449 | 1-GPU (GPUExecutionParquetFixture) | SIGSEGV in compare_gpu_vs_cpu's second `SELECT * FROM gpu_execution(...)` call (line 239); Catch2 signal handler anchors at preceding REQUIRE (216) |
| `gpu_execution - tpch_q1_sf10_2gpu` | test_gpu_execution_tpch.cpp:4297 | 2-GPU (compare_gpu_vs_cpu_sf10_for(2, kTpchQ1Body)) | Same signature at same call-site pattern (second TABLE_FUNCTION invocation) |

**Hypothesis candidates (carried forward from 09-04-VALIDATION.md Open Issue section verbatim):**

- **H1 — residual `_datasource` caching on compute_task re-dispatch** (RESEARCH.md Open Questions #1). `_datasource` is a `std::shared_ptr` member of `parquet_scan_task`, set inside `if (!_datasource)`. Second gpu_execution call may reuse stale device-bound `prefetched_data_source` with freed allocation. Probe: `[mgpu-probe]` at `_datasource` construction + reset.
- **H2 — TABLE_FUNCTION-form vs PROCEDURE-form (`CALL`) result materialization divergence** (LEADING HYPOTHESIS). CALL-form returns Sirius result directly. SELECT * FROM wraps it in a DuckDB table function binding → different result-passing code path (copy vs move, GPUResult proxy lifetime). Sirius-layer bug in table-function output shaping. Audit: `src/sirius_extension.cpp` + `src/sirius_interface.cpp`.
- **H3 — `_batch_gpu_affinity` map lifecycle on second query** (RESEARCH.md Q5 Candidate 2). Reset on `prepare_cache_for_scan_operators`. If second query reuses cache but reset races with dispatch-thread observation, stale affinity read. Unlikely at single-threaded dispatch but verifiable with trace.
- **H4 — unrelated regression in the 5-commit span** (fallback). `unordered_map` use-after-free, allocator corruption, or data race. Bisect 3b58258 / 863cc6c / 0c8068e / a8a7985 / c0e12f3 + run `'gpu_execution - filter equality parquet'` at each.

**Suggested sequence for closure (Phase 10 scope):**

1. **Bisect the 5-commit span.** Check out each of 3b58258, 863cc6c, 0c8068e, a8a7985, c0e12f3 individually; run `./build/release/extension/sirius/test/cpp/sirius_unittest 'gpu_execution - filter equality parquet'` at each. The first commit where this test SIGSEGVs identifies the introduction point. This distinguishes H1/H2/H3 from H4.
2. **Attach gdb to the crashing binary** per `.claude/skills/debug-gdb/SKILL.md`. Get backtrace at the actual SIGSEGV frame (not the anchor REQUIRE line). Determine whether fault is inside Sirius (`src/`) or in cudf/cucascade/duckdb downstream.
3. **If H2 confirmed (most likely):** Instrument `src/sirius_extension.cpp` gpu_execution TABLE_FUNCTION registration + `src/sirius_interface.cpp` GPUResult lifetime boundaries. Log which form was invoked + GPUResult ownership transitions. Compare CALL vs TABLE_FUNCTION execution paths; identify divergence.
4. **If H1 or H3:** Add `[mgpu-probe]` at `_datasource` construction + `_batch_gpu_affinity` reset. Re-run.
5. **If H4:** Once bisect identifies the regressing commit, either fix forward or consider a forward-compatible rollback that preserves the SF100 ship-gate win (CRIT-1/4/6) while eliminating the TABLE_FUNCTION regression (CRIT-2).
6. **Regardless:** SF100 Q1 + SF100 bench evidence (CRIT-1 + CRIT-6) are already PASS. v1.2 can optionally ship the distributor wins as a preview/beta per the VALIDATION.md "Next Steps" recommendation while Phase 10 closes the CRIT-2 gap.

**Expected Phase 10 scope:** < 100 LOC source change + one ship-gate re-run. This is a deterministic probe + scoped patch, not an open-ended bug-hunt — matching the 08-11 precedent.

### Gap 2 (NON-BLOCKING — informational): CRIT-4 strict `>=5` threshold fails on SF1-DuckDB AUDIT fixture

**Root cause:** AUDIT TEST_CASE (Plan 08-05) uses `attach_integration_duckdb` to decouple from the Phase 8 host_parquet bug. SF1-DuckDB fixture produces only 2 scan_ids per GPU after round-robin; the `>=5` strict threshold (from ROADMAP Phase 8 Criterion 4) is designed for SF10-parquet scale.

**Why this is not a Phase 9 regression:** Pre-existing Plan 08-05 test-design choice, inherited. The **substantive** distributor-correctness claim the threshold was designed to lock in is the **disjointedness REQUIRE** (Plan 09-03's addition), which PASSES. The strict count is a decorative belt-and-suspenders gate that the current fixture simply can't satisfy. Not a ship-blocker; flagged here as a cleanup target for future work (e.g., switch AUDIT fixture to SF10-parquet once the host_parquet bug from Phase 8 is closed, or drop the strict threshold in favor of disjointedness-only gating).

---

## Open Issues (carried forward from 09-04-VALIDATION.md)

The primary Open Issue (TABLE_FUNCTION SIGSEGV with H1-H4 hypotheses) is detailed in Gap 1 above. Plan author's own 09-04-VALIDATION.md section:

> **Observed signature:** `SIGSEGV - Segmentation violation signal` inside `compare_gpu_vs_cpu`'s second `SELECT * FROM gpu_execution(\"\" + clean_query + \"\")` path (test_gpu_execution_tpch.cpp:239). The first form — `CALL gpu_execution(\"\" + query + \"\")` — completes cleanly... The SF100 CLI run uses ONLY the first form and does NOT crash — hence SF100 ship-gate passes while unit-tests do not.

This verification concurs: the TABLE_FUNCTION vs PROCEDURE divergence is the diagnostic signal most worth probing first (H2), matching the plan author's conclusion. The distributor fix itself does not need re-work; the regression is in a downstream result-materialization path.

---

## Recommendation — Next Step

**Open Phase 10 plan** (or equivalently, run `/gsd:plan-phase 9 --gaps` to produce a 09.x gap-closure plan) with scope strictly limited to:

1. **Bisect** (~30 min): 5 commits × one test invocation each = identify introduction point.
2. **gdb backtrace** (~30 min): capture actual fault frame on the earliest reproducing commit.
3. **Targeted fix** (< 100 LOC): per confirmed hypothesis (H1/H2/H3/H4).
4. **Re-ship-gate** (~30 min): re-run the Phase 9 09-04-PLAN.md procedure end-to-end. Expect CRIT-2 green, CRIT-1/4/6 remain green, v1.2 ships.

**Do NOT escalate into an open-ended diagnostic plan.** Phase 9 has already narrowed the hypothesis space to four candidates and identified the 5-commit introduction window. Phase 10 is a deterministic closure plan, not a reopening.

**Ship the distributor wins as preview/beta (optional):** per the plan author's VALIDATION.md recommendation, the distributor wins (5.86s SF100 Q1, byte-identical 2-GPU result, disjoint cross-GPU batch dispatch) are real and shippable independently. v1.2 milestone proper waits on Phase 10 CRIT-2 closure, but a preview/beta release capturing the SF100 evidence is defensible.

---

## Human Verification Required

None. All verification signals are either (a) grep-verifiable in the local source tree (all re-verified by this agent), or (b) recorded in 09-04-VALIDATION.md by the plan author who executed on the MCP-enabled 2-GPU host per the project's 2026-04-24 MCP-autonomy policy. The bash sandbox here has no GPU driver access, so live re-runs of SF100 Q1 or MCP unit-tests are not possible; the existing evidence in 09-04-VALIDATION.md is authoritative and internally consistent with the claimed outcomes.

Phase 10 closure work **does** require MCP + 2-GPU access (for bisect, gdb, re-ship-gate); that is a scope-of-execution concern for Phase 10, not a verification gap for Phase 9.

---

*Phase: 09-scan-task-distributor-batch-ownership-affinity*
*Verifier: Claude (gsd-verifier)*
*Verified: 2026-04-24*
*HEAD: 68fbda9*
*Branch: feature/single-node-multi-gpu2*
