---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: Rebase After DataBatch Changes
status: executing
stopped_at: Completed 16-01-PLAN.md
last_updated: "2026-05-04T23:12:19.867Z"
last_activity: 2026-05-04
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 5
  completed_plans: 1
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 16 — Cucascade Submodule Rebase + Pin Recovery

## Current Position

Phase: 16 (Cucascade Submodule Rebase + Pin Recovery) — EXECUTING
Plan: 2 of 5
Status: Ready to execute
Last activity: 2026-05-04

```
v1.4 Progress: [                    ] 0/6 phases | 0/32 requirements | 0 plans
```

## Phase Overview (v1.4)

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 16 | Cucascade Submodule Rebase + Pin Recovery | CC-01..04 | Not started |
| 17 | Sirius origin/dev Merge — Base Layer | MERGE-01..05 | Not started |
| 18 | DataBatch RAII Migration | DB-01..05 | Not started |
| 19 | IO Framework Adoption | IO-12..17 | Not started |
| 20 | Scan Manager + Pin Tables Port | SM-01..06 | Not started |
| 21 | v1.4 Ship Gate | REG-01..06 | Not started |

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
| Phase 12 P01 | 10min | 1 tasks | 1 files |
| Phase 12 P02 | 6min | 1 tasks | 1 files |
| Phase 12 P03 | 3min | 1 tasks | 1 files |
| Phase 12 P04 | 10min | 1 tasks | 2 files |
| Phase 13 P01 | 33min | 1 tasks | 1 files |
| Phase 13 P02 | 10min | 1 tasks | 2 files |
| Phase 13 P03 | 30min | 1 tasks | 2 files |
| Phase 13 P04 | 60min | 1 tasks | 9 files |
| Phase 14 P01 | 2 min | 1 tasks | 2 files |
| Phase 14 P02 | 7 min | 1 tasks | 2 files |
| Phase 15 P01 | 6min | 4 tasks | 10 files |
| Phase 15 P02 | 8min | 2 tasks | 5 files |
| Phase 15 P03 | 4min | 2 tasks | 2 files |
| Phase 15 P04 | 39min | 2 tasks | 1 files |
| Phase 16 P01 | 3 | 3 tasks | 1 files |

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
- [Phase 09]: [Phase 09-02] Affinity map (_batch_gpu_affinity) is written at dispatch time but not yet consulted at dispatch time — provides data structure for Plan 09-03 disjointedness assertion; dispatch-time re-routing deferred to Phase 10+ if 09-04 validation shows residual cross-GPU collisions
- [Phase 09]: [09-03] std::set_intersection on counts[0/1].scan_ids provides the permanent regression gate for Bug 1 (hypothesis E double-dispatch); REQUIRE fires on 2-GPU hosts, silently skipped on 1-GPU hosts via existing device_count < 2 WARN+return guard
- [Phase 09]: [09-04] SF100 Q1 num_gpus=2 ship-gate PASSES — byte-identical to 1-GPU baseline, 71 scan batches dispatched disjointly across GPUs (GPU0=45, GPU1=26, intersect=0), wall-clock 5.86s, zero cudaErrorInvalidValue/SIGSEGV/fallback. Plans 09-01 (preferred_device_id), 09-02 (batch affinity), 09-03 (disjointedness REQUIRE) all proven live at runtime.
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
- [Phase 12]: [12-01] Fix-site identified: src/op/sirius_physical_hash_join.cpp:623 — sirius::op::prepare_join_keys -> cudf::table_view::select(key_col_indices) throws std::out_of_range when key_col_indices contains 2 on a 2-column table_view.
- [Phase 12]: [12-02] Bound-checked key_col_indices in sirius::op::prepare_join_keys at src/op/sirius_physical_hash_join.cpp:622-637. HYG baseline preserved at 40.
- [Phase 12]: [12-03] Added regression TEST_CASE 'physical_order - small sort rangecheck regression' (test_physical_order_mgpu.cpp:120-165).
- [Phase 12]: [12-04] Phase 12 ship-gate validation PASS — all 4 CONTEXT.md acceptance criteria verified. TPC-H × 2-GPU integration 48/48 pass (71608 assertions).
- [Phase 13]: [13-02] FIRST stream-ordered race localized to cucascade::convert_gpu_to_gpu at cucascade/src/data/representation_converter.cpp:801; fix shape: cucascade-side gpu_table_representation extension with set_writer_event/get_writer_event accessor + cudaStreamWaitEvent.
- [Phase 13]: [13-04] PARTIAL fix: cucascade pin bumped to 7409c60; Q11-alone PASS (9011 assertions, 7s). Residual ~22 producer sites migrated via Path-2 architectural fix in subsequent work.
- [Phase 14-01]: std::unordered_map -> std::map for _gpu_executors for deterministic iteration; Atomic round-robin counter (_no_pref_rr_counter) gated on !have_pref && size>1; Per-query reset of _no_pref_rr_counter in prepare_for_query.
- [Phase 14]: [14-02] Phase 14 ship-gate PASS via 4 MCP runs. C1 [mgpu] 12/13 PASS-with-Phase-12-note. C2 [TPC-H][parquet] 22/22. [integration][TPC-H] 48/48, 71608 assertions. HYG-02=40.
- [Phase 15-01]: All 11 audited operator sites classified SAFE — SCHED-RR INVARIANT contract holds. SAFE=11 NEEDS-PATCH=0 UNCLEAR=0.
- [Phase 15]: [15-02] [mgpu_stress] test: 100 iterations × 5 representative [mgpu] queries = 500 inner runs; 86.6s, 77053 assertions, exit 0.
- [Phase 15]: [15-04] Phase 15 ship-gate PASS. All gauntlet criteria green. [mgpu] 16/16 post-FU-A merge.
- [FU-A]: Merged fix/order-small-sort-rangecheck into Phase 15 tip. [mgpu] lifted 12/13 → 16/16 (79091 assertions, 120.3s, exit 0).
- **[v1.4 ROADMAP]** 2026-05-04: 6 phases (16-21) created. 32 requirements mapped 100%. Phase 16 is the first plannable phase. Compile-graph dependency chain: 16 → 17 → 18 → 19 → 20 → 21. Verification policy: light gates per phase 16-20 (grep + targeted unit tests + SF1 smoke); full v1.3 gauntlet at Phase 21 only.
- **[v1.4 ROADMAP]** Key ordering constraint: Phase 19 (IO Framework) must precede Phase 20 (Scan Manager) because `parquet_split_provider::run_batch` calls `sirius_datasource` — compile-graph dependency. This overrides FEATURES.md's original Scan Manager-first proposal; ARCHITECTURE.md's ordering is adopted.
- **[v1.4 ROADMAP]** Phase 17 expected build errors: 26+ `batch->get_data() is private` errors + RAII compile errors. These are EXPECTED and do not constitute a phase failure. Documented in MERGE-05.
- [Phase 16-01]: Squash commit reordering: e23f3a2 (Group 1 memory) and eda349a (Group 3 pipeline) reordered in rebase todo to match D-A1 logical grouping; 2-pass rebase (squash then reword) used for clean separation of concerns

## Accumulated Context

### Key Findings from v1.1 Verification

- **Bug site:** `pipelineable_operator_data::prepare_for_processing` → `pipeline::lock_or_prepare_batch` → `cuda_memcpy.cu:42`
- **Error:** `cudaErrorInvalidValue: invalid argument`
- **Trigger:** non-trivial SQL (filter+sort, aggregation, join, TPC-H Q1/Q6/Q12) when `num_gpus >= 2`
- **Root cause signal:** cross-device stream-correctness. Same shape as the Sirius-side P2P converter override.
- **Existing observability:** `[mgpu-audit]` info-level dispatch logs in `src/pipeline/pipeline_executor.cpp:247-249` + `src/op/scan/duckdb_scan_executor.cpp:180-184`.
- **Evidence:** `.planning/milestones/v1.1-E2E-VERIFICATION.md`.

### Decisions carried from v1.1

- **Sirius-side converter override is the fix pattern** for cross-device stream-correctness bugs (Pattern 2 from Plan 07 research).
- **Sticky `cudaGetLastError()` consume** is required after any cuda* call that can leave state in the thread-local slot.

### v1.4 Critical Context

- **Phase 17 expected compile errors:** 26+ `batch->get_data() is private` errors from the new cucascade RAII model. These are expected and document that Phase 18 work is needed. Do not attempt to fix them in Phase 17.
- **PR #739 handling:** Never cherry-pick PR #739 as-is. It targets cucascade `0cd4a6a` (pre-#117). Use it only as a file-list reference during Phase 18 DataBatch migration; the actual API recipe follows the #117 RAII pattern.
- **`sirius_parquet_metadata_scan_operator.hpp` deletion:** The Phase 13 stream-lineage work lives in this file which PR #731 deletes. MERGE-03 requires extracting the attachment points BEFORE accepting the deletion. Re-attachment happens in Phase 20.
- **Phase 19 prerequisite:** Install `liburing-dev` (`sudo apt-get install -y liburing-dev`) BEFORE the first build attempt after Phase 19 source changes land. Runtime library (`liburing2:amd64 2.5`) is already present on the host; headers are not.
- **SF100 Q11 2-GPU:** Must be run explicitly at Phase 20 and Phase 21 — not just `[mgpu]` suite at SF1. The stream-ordering race (P2) only manifests at SF100 data volume.

### Pending Todos

- Plan Phase 16 (`/gsd:plan-phase 16`) — decompose cucascade rebase into plans covering the 6 conflict files.

### Blockers / Concerns

None at roadmap creation. Phase 16 is ready to plan.

## Session Continuity

Last session: 2026-05-04T23:12:19.864Z
Stopped at: Completed 16-01-PLAN.md
Resume file: None
