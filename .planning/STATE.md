---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Multi-GPU SQL Pipeline Fix
status: verifying
stopped_at: "Completed 10-04: ship-gate validation — PARTIAL verdict"
last_updated: "2026-04-27T20:46:42.775Z"
last_activity: 2026-04-27
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 18
  completed_plans: 16
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 10 — table-function-form-gpu-execution-sigsegv-fix

## Current Position

Phase: 10 (table-function-form-gpu-execution-sigsegv-fix) — EXECUTING
Plan: 4 of 4
Status: Phase complete — ready for verification
Last activity: 2026-04-27

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
| Phase 09 P01 | 25min | 3 tasks | 3 files |
| Phase 09 P02 | 4min | 3 tasks | 3 files |
| Phase 09 P03 | 5min | 1 tasks | 1 files |
| Phase 09 P04 | 2h | 4 tasks | 2 files |
| Phase 10 P01 | 26min | 3 tasks | 1 files |
| Phase 10 P02 | 46min | 3 tasks | 1 files |
| Phase 10-table-function-form-gpu-execution-sigsegv-fix P03 | 55 | 2 tasks | 2 files |
| Phase 10 P04 | 115min | 4 tasks | 3 files |

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
- [Phase 09]: Path B for parquet_scan_task_local_state: sirius_pipeline_task_local_state base does NOT have preferred_device_id accessors (only global state does); accessors added directly to local state class
- [Phase 09]: Two-tier local-wins-over-global preferred_device_id lookup in compute_task mirrors gpu_pipeline_task::get_preferred_device_id (gpu_pipeline_task.hpp:188-194)
- [Phase 09]: Affinity reset placed unconditionally at top of prepare_cache_for_scan_operators (before cache_level::NONE early return) so _scan_round_robin and _batch_gpu_affinity reset together on every query start regardless of caching mode (Pitfall 3 compliance)
- [Phase 09]: [Phase 09-02] Affinity map (_batch_gpu_affinity) is written at dispatch time but not yet consulted at dispatch time — provides data structure for Plan 09-03 disjointness assertion; dispatch-time re-routing deferred to Phase 10+ if 09-04 validation shows residual cross-GPU collisions
- [Phase 09]: [09-03] std::set_intersection on counts[0/1].scan_ids provides the permanent regression gate for Bug 1 (hypothesis E double-dispatch); REQUIRE fires on 2-GPU hosts, silently skipped on 1-GPU hosts via existing device_count < 2 WARN+return guard
- [Phase 09]: [09-04] SF100 Q1 num_gpus=2 ship-gate PASSES — byte-identical to 1-GPU baseline, 71 scan batches dispatched disjointly across GPUs (GPU0=45, GPU1=26, intersect=0), wall-clock 5.86s, zero cudaErrorInvalidValue/SIGSEGV/fallback. Plans 09-01 (preferred_device_id), 09-02 (batch affinity), 09-03 (disjointness REQUIRE) all proven live at runtime.
- [Phase 09]: [09-04] Verdict: PARTIAL. v1.2 ship BLOCKED on new regression (unrelated to distributor): SIGSEGV in 'SELECT * FROM gpu_execution(...)' TABLE-FUNCTION-form result materialization path. CALL-form works (SF100 passes); TABLE_FUNCTION-form crashes. Scoped to Phase 10.
- [Phase 09]: [09-04] MCP unit-tests wrapper does not pass agent shell env to child process; SIRIUS_TEST_SF10_PATH and SIRIUS_LOG_DIR had to be set via direct binary invocation (Rule 3 auto-fix). MCP build gate still used for build verification.
- [Phase 10]: regressing_commit=NONE: all 5 Phase-9 source commits (3b58258..c0e12f3) pass isolated test; SIGSEGV is test-ordering dependent
- [Phase 10]: FU17 partial fix changes at HEAD change SIGSEGV to cudaErrorContextIsDestroyed; Plan 10-02 should gdb clean state (c0e12f3) with full-suite --abort ~[hive_partition] to reproduce original SIGSEGV
- [Phase 10]: H1 confirmed: SIGSEGV is stream-ordered race in sirius_physical_parquet_scan.cpp using rmm::cuda_stream_default for gpu_expression_translator; scalars race with planning_stream in parquet_scan_task.cpp:492
- [Phase 10]: GDB Heisenbug: Catch2 sigsetjmp/siglongjmp signal handler causes SIGSEGV to be swallowed under GDB (test completes normally). Static analysis + FU17 diff developer comment used as primary fault-frame evidence source
- [Phase 10]: H2 (TABLE_FUNCTION vs PROCEDURE divergence) ruled out: both CALL and SELECT * FROM gpu_execution() use same GPUExecutionBind/GPUExecutionFunction; crash is parquet-fixture-specific, not TABLE_FUNCTION-form-specific
- [Phase 10-table-function-form-gpu-execution-sigsegv-fix]: Root cause is use-after-destroy: translation_stream destroyed at for-loop scope exit while scalars retain stale cudaStream_t handle; fix: move stream into translated_expression::owned_stream
- [Phase 10-table-function-form-gpu-execution-sigsegv-fix]: std::optional<rmm::cuda_stream> owned_stream declared BEFORE owned_literals in translated_expression struct — C++ reverse-destruction order ensures stream outlives scalars for cudaFreeAsync
- [Phase 10]: [10-04] Verdict PARTIAL: Phase 10 fix objective (filter equality parquet + tpch_q1_sf10_2gpu GREEN) COMPLETE; pre-existing [mgpu-audit] SIGSEGV prevents All tests passed
- [Phase 10]: [10-04] SF100 Q1 2-GPU ship-gate PASS: exit 0, 5.70s, 4 rows, byte-identical vs 1-GPU baseline, GPU0=42 GPU1=29 intersection=0

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

### Roadmap Evolution

- Phase 10 added (2026-04-24): TABLE_FUNCTION-form gpu_execution SIGSEGV fix — closes v1.2 ship-gate CRIT-1/2/6 after Phase 9 distributor fix proved correct at SF100 but unit-test SIGSEGV in `SELECT * FROM gpu_execution(...)` result-materialization path blocked the ship. Scope: bisect `3b58258..c0e12f3` → gdb → targeted fix → re-run 09-04 ship-gate. Evidence in `.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-VERIFICATION.md`.

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
- **[v1.2 SHIP BLOCKER — 09-04 CRIT-2]** `SELECT * FROM gpu_execution("...")` TABLE_FUNCTION-form materialization path SIGSEGVs in unit tests (both 1-GPU and 2-GPU envs). The `CALL gpu_execution("...")` PROCEDURE-form works fine — SF100 Q1 num_gpus=2 ship-gate PASSES with byte-identical result vs 1-GPU baseline and disjoint cross-GPU batch dispatch. Distributor fixes (Plans 09-01/02/03) are all proven correct at runtime (`preferred_device_id=-1` count=0, cross-GPU intersection=0 at SF100 scale). Regression scoped to Phase 10: bisect across commits `3b58258`/`863cc6c`/`0c8068e`/`a8a7985`/`c0e12f3` + gdb on `gpu_execution - filter equality parquet` test. See `.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-04-VALIDATION.md` Open Issue section H1-H4 (H2 TABLE_FUNCTION vs CALL-form result shaping is the leading hypothesis).

## Session Continuity

Last session: 2026-04-27T20:46:42.773Z
Stopped at: Completed 10-04: ship-gate validation — PARTIAL verdict
Resume file: None
