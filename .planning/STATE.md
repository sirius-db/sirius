---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Multi-GPU SQL Pipeline Fix
status: verifying
stopped_at: "Completed 08-07-PLAN.md (gap-closure instrumentation: [mgpu-probe] breadcrumbs landed at host_parquet converter entry+exit + parquet_scan_task::compute_task entry; build exit 0; HYG-02 preserved; ready for 08-08 reproduction)"
last_updated: "2026-04-22T13:45:33.356Z"
last_activity: 2026-04-22
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 10
  completed_plans: 8
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 08 — multi-gpu-sql-pipeline-fix

## Current Position

Phase: 08 (multi-gpu-sql-pipeline-fix) — COMPLETE (ship-blocked)
Plan: 6 of 6
Status: Phase complete — ready for verification
Last activity: 2026-04-22

Progress: [██████████] 100% (6/6 plans complete)
Ship verdict: BLOCKED_ON_RESIDUAL_FIX_SITE — see `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-SUMMARY.md`

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Completed           |
| ----- | ---- | -------- | ----- | ----- | ------------------- |
| 08    | 01   | 6min     | 3     | 3     | 2026-04-22T01:19:13Z |
| Phase 08 P02 | 15min | 3 tasks | 5 files |
| Phase 08 P03 | 6min | 2 tasks | 2 files |
| Phase 08 P04 | 20min | 3 tasks | 5 files |
| Phase 08 P05 | 86min | 3 tasks | 4 files |
| Phase 08 P06 | 21min | 3 tasks | 3 files |
| Phase 08 P07 | 10min | 2 tasks | 2 files |

## Decisions

- **[08-01]** FIX-01: Per-GPU stream pool map in duckdb_scan_executor replaces singular GPU-0-bound pool. Dispatch lambda opens with rmm::cuda_set_device_raii pinned to target_gpu_id. Pattern 2 idiom extended from p2p converter to scan executor.
- **[08-01]** Hoisted select_target_gpu() from parquet-only block to top of manager_loop so the dispatch lambda can capture target_gpu_id. Non-parquet scan tasks (cpu_source_task, duckdb_scan_task) now also route through a well-defined target device.
- **[08-01]** Single-GPU host runtime reproduction deferred to Plan 08-06 ship gate (verification hardware has 2 × RTX 6000 Ada). Static invariants + MCP build gate verified here.
- [Phase 08]: FIX-02 Branch B authored — Sirius-side host_data_representation -> gpu_table_representation converter override with target-bound stream + target-device RAII, using public-API-only column-tree reconstruction. cucascade submodule pin unchanged.
- [Phase 08]: Distinct fix-site discovered during verification: Sirius's own convert_host_parquet_to_gpu_with_prefetched_data_source has same bug shape as cucascade's convert_host_fast_to_gpu but on host_parquet_representation path. Handed off to 08-06 per plan scope; Branch B is canonical template.
- [Phase 08]: [08-03] AUDIT payload extension: append task_id (from gpu_pipeline_task::get_task_id()) and batch_id (from existing _scan_round_robin counter) to the two [mgpu-audit] INFO emissions. No new atomics added; counter reused. Accessor VARIANT A applied; pointer fallback not needed.
- [Phase 08]: [08-03] Grep-stable payload shape locked — emissions end with `GPU N task_id=K` / `GPU N batch_id=K` suffixes. Plan 08-05 grep pattern: grep the prefix, extract the `key=value` with `grep -oE`, then `sort -u | wc -l` for unique-count assertion. Backward-compat preserved with v1.1 verification greps.
- [Phase 08]: [08-04] TEST-01/02: added integration-2gpu.yaml + g_integration_env_2gpu + acquire_integration_env_for() helper + RUN_TPCH_MGPU macro. All 44 TPC-H TEST_CASEs parameterized on num_gpus in {1,2} via Catch2 GENERATE. integration.yaml unchanged (1-GPU default preserved).
- [Phase 08]: [08-04] Chose research-recommended Option A (Catch2 GENERATE inside TEST_CASE body) over Option B (TEMPLATE_TEST_CASE_METHOD) and Option C (duplicated flavors). Lowest-churn: zero TEST_CASE_METHOD header edits, 44 mechanical one-line call-site substitutions.
- [Phase 08]: [08-04] Virtual setup_schema() hook on GPUExecutionFixtureBase re-runs subclass DDL on fresh connection after each bind_env(num_gpus); avoids leaking 1-GPU schema into 2-GPU connection when bind_env(2) reassigns con.
- [Phase 08]: [08-05] AUDIT TEST_CASE authored routing through DuckDB ATTACH path (not parquet) — decouples assertion from open 08-06 host_parquet bug. Pre-verified statically; runtime deferred to 08-06.
- [Phase 08]: [08-05] SF10 Q1/Q6/Q12 TEST_CASEs gated on SIRIUS_TEST_SF10_PATH env var + cudaGetDeviceCount>=2, both WARN+return on miss. Runtime verification deferred to 08-06 verification host.
- [Phase 08]: [08-05] Q4 retry wrapper scoped to tpch_q4 TEST_CASE only (DuckDB + parquet flavors). Other queries keep RUN_TPCH_MGPU so real regressions fail loudly. Per ROADMAP Phase 8 Success Criterion 2 flake policy.
- [Phase 08]: [08-05] Audit TEST_CASE threshold: >=5 per GPU when SIRIUS_TEST_SF10_PATH is set (ROADMAP criterion 4 strict), >=1 per GPU otherwise (SF1 lineitem ~6 total batches). Strict threshold fires on 08-06 verification host.
- [Phase 08]: [08-05] MCP daemon caches commands.yaml at session start; hot-reload unsupported. unit-tests cannot be invoked with --abortx 999 or tag filter from this agent. 08-06 will use a fresh session or close the host_parquet bug so --abort never trips.
- [Phase 08]: [08-06] Applied carryover fix (Pattern 2 idiom) to convert_host_parquet_to_gpu_with_prefetched_data_source per orchestrator directive; mirrors 08-02 Branch B template. Build + HYG clean. Same cudaErrorInvalidValue signature persists on num_gpus=2 parquet TPC-H Q1 — residual fix-site beyond 08-06's scope, handed off with 4 hypothesis candidates.
- [Phase 08]: [08-06] FIX-03 verdict: PASS — grep of rmm::cuda_stream_default in src/ returns 41 matches (unchanged phase-7 baseline); 0 net-new introductions by Phase 8.
- [Phase 08]: [08-06] FIX-04 verdict: PASS — mcp build exit 0 after rm -rf build. ROADMAP criterion 5 (Pattern 2 idiom grep) PASS with 6 code matches across 4 fix sites.
- [Phase 08]: [08-06] Phase 8 ship verdict BLOCKED — criteria 1/2/4/6 DEFERRED because TPC-H Q1 parquet + num_gpus=2 still hits cudaErrorInvalidValue @ cuda_memcpy.cu:42 after carryover fix. SF100 Q1 ship-gate not run because SF1 parquet already reproduces the blocker.
- [Phase 08]: [08-07] Added three [mgpu-probe] INFO breadcrumbs (host_parquet entry+exit, parquet_scan_task::compute_task entry) with grep-stable payload discriminating hypotheses A/B/C/D carried forward from 08-VERIFICATION.md. Plan's <interfaces> claim on log/logging.hpp include wiring was wrong — added it inline per Rule 3.
- [Phase 08]: [08-07] HYG-02 baseline still 41 matches; zero logic changes, zero new RAII, zero new stream acquires, zero yaml edits, zero cucascade edits. Instrumentation-only gap-closure plan unblocks 08-08 reproduction.

## Accumulated Context

### Key Findings from v1.1 Verification

- **Bug site:** `pipelineable_operator_data::prepare_for_processing` → `pipeline::lock_or_prepare_batch` → `cuda_memcpy.cu:42`
- **Error:** `cudaErrorInvalidValue: invalid argument`
- **Trigger:** non-trivial SQL (filter+sort, aggregation, join, TPC-H Q1/Q6/Q12) when `num_gpus >= 2`
- **Not triggered:** trivial SQL (`SELECT count(*) FROM nation`) — pipeline tasks distribute to both GPUs, returns correct result
- **Root cause signal:** cross-device stream-correctness. Same shape as the Sirius-side P2P converter override (`src/data/sirius_p2p_converter.cpp`) — pack on source-device RAII + source stream, copy on target stream.
- **Reproduction:** set `num_gpus: 2` in `test/cpp/integration/integration.yaml`, run any TPC-H parquet/join integration test → fails. Same tests pass on `num_gpus: 1`.
- **Existing observability:** `[mgpu-audit]` info-level dispatch logs in `src/pipeline/pipeline_executor.cpp:247-249` + `src/op/scan/duckdb_scan_executor.cpp:180-184` (from commit `fd24174`).
- **Evidence:** `.planning/milestones/v1.1-E2E-VERIFICATION.md` (full report: 389 lines, reproduction steps, v1.2 recommendations).

### Decisions carried from v1.1

- **Sirius-side converter override is the fix pattern** for cross-device stream-correctness bugs (Pattern 2 from Plan 07 research). Same approach applies to `lock_or_prepare_batch`.
- **Sticky `cudaGetLastError()` consume** is required after any cuda* call that can leave state in the thread-local slot.

### Roadmap Decisions (v1.2)

- **Single phase (Phase 8)** — user explicitly chose single-phase scoping during new-milestone questioning; granularity `coarse` in config.json reinforces this. All 11 requirements form one coherent delivery with internal dependencies (can't meaningfully test without fix; can't audit without both).
- **Phase numbering continues from v1.1** — Phase 8 follows v1.1's Phase 7, keeping the milestone history linear rather than resetting.
- **Integration fixture strategy (per FIX+TEST+AUDIT coupling):** scope the `num_gpus: 2` flip to TPC-H integration parameterization first, rather than flipping every fixture globally. Other fixtures can follow in later milestones if the pattern proves stable (per REQUIREMENTS.md Out of Scope note).

### Pending Todos

- Plan Phase 8 (`/gsd:plan-phase 8`) — decompose into plans covering FIX → TEST → AUDIT.
- Execute Phase 8 plans.
- Validate against all 5 success criteria on real N=2 hardware (2 × RTX 6000 Ada).

### Blockers / Concerns

- **[v1.2 SHIP BLOCKER — 08-06]** Residual `cudaErrorInvalidValue @ cuda_memcpy.cu:42` on num_gpus=2 parquet path. Failing tests: `gpu_execution hive partition - filter on data column` and `gpu_execution - TPC-H Query 1 parquet`. The 08-06 carryover fix at `convert_host_parquet_to_gpu_with_prefetched_data_source` (Pattern 2 idiom mirroring 08-02 Branch B template) was applied and build+HYG pass, but the same bug signature persists — indicating at least one additional fix-site. 4 hypothesis candidates and concrete suggested next actions documented in `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md` "Open Issue — Residual Carryover-Fix Incompleteness" section. Blocks ROADMAP criteria 1 + 2 + 4 + 6 (criteria 3 + 5 pass as static invariants).
- **Integration fixture scope:** TPC-H fixture currently hard-codes `num_gpus: 1` via `setenv` inside the test fixture. Flipping globally may uncover other multi-GPU bugs not exposed by the unit-test suite today. Phase 8 plans should parameterize TPC-H specifically (per TEST-01) rather than flip the default globally — the parameterization approach is what AUDIT-03 requires anyway (2-GPU variant MUST execute in default unit-tests run, but the 1-GPU variant need not be removed).

## Session Continuity

Last session: 2026-04-22T13:45:33.354Z
Stopped at: Completed 08-07-PLAN.md (gap-closure instrumentation: [mgpu-probe] breadcrumbs landed at host_parquet converter entry+exit + parquet_scan_task::compute_task entry; build exit 0; HYG-02 preserved; ready for 08-08 reproduction)
Resume file: None
