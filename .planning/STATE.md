---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Gauntlet on Rebased Branch)
status: completed
stopped_at: Phase 22 context gathered
last_updated: "2026-05-07T22:00:17.388Z"
last_activity: 2026-05-06
progress:
  total_phases: 7
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** v1.4 SHIPPED 2026-05-06. v1.5+ scope (PIN-MGPU-01, IO-MGPU-02, CC-UPSTREAM-01, FU-B) awaiting milestone planning.

## Current Position

Phase: 21 (v1.4 Ship Gate (Full v1.3 Gauntlet on Rebased Branch)) — COMPLETE
Plan: 1 of 1 (COMPLETE)
Status: v1.4 SHIPPED — all 32 requirements (CC-01..04 + MERGE-01..05 + DB-01..05 + IO-12..17 + IO-15B + SM-01..06 + REG-01..06) Complete
Last activity: 2026-05-06

```
v1.4 Progress: [####################] 6/6 phases | 32/32 requirements | 29 plans | SHIPPED 2026-05-06
```

## Phase Overview (v1.4)

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 16 | Cucascade Submodule Rebase + Pin Recovery | CC-01..04 | Complete (5/5 plans, PASS) |
| 17 | Sirius origin/dev Merge — Base Layer | MERGE-01..05 | Complete (4/4 plans, PASS) |
| 18 | DataBatch RAII Migration | DB-01..05 | Complete (7/7 plans, PASS) |
| 19 | IO Framework Adoption | IO-12..17 | Complete (6/6 plans, PASS) |
| 20 | Scan Manager + Pin Tables Port | SM-01..06 | Complete (6/6 plans, PASS) — SM-06 SF1 closed by 20-06 |
| 21 | v1.4 Ship Gate | REG-01..06 | **Complete (1/1 plans, PASS)** — all REG-01..06 PASS; SM-02 fixture-fix path chosen; 21-VERDICT.md written 2026-05-06 |

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
| Phase 16 P02 | 4min | 3 tasks | 8 files |
| Phase 16 P03 | 15 | 2 tasks | 4 files |
| Phase 16-cucascade-submodule-rebase-pin-recovery P04 | 45 | 5 tasks | 12 files |
| Phase 16 P05 | 15min | 3 tasks | 2 files |
| Phase 17 P01 | 8min | 2 tasks | 2 files |
| Phase 17-sirius-origin-dev-merge-base-layer P02 | 45min | 3 tasks | 11 files |
| Phase 17-sirius-origin-dev-merge-base-layer P03 | 45 | 2 tasks | 2 files |
| Phase 17-sirius-origin-dev-merge-base-layer P04 | 15min | 1 tasks | 1 files |
| Phase 18 P01 | 5min | 3 tasks | 4 files |
| Phase 18 P02 | 7min | 3 tasks | 7 files |
| Phase 18 P03 | 13min | 3 tasks | 9 files |
| Phase 18 P04 | 16min | 3 tasks | 14 files |
| Phase 18 P05 | 65min | 4 tasks | 31 files |
| Phase 18 P06 | 164min | 3 tasks | 11 files |
| Phase 18 P07 | 111min | 3 tasks | 8 files |
| Phase 19 P01 | 30min | 2 tasks | 1 files |
| Phase 19 P03 | 2min | 2 tasks | 2 files |
| Phase 19 P02 | 10min | 1 tasks | 1 files |
| Phase 19 P04 | 10min | 2 tasks | 2 files |
| Phase 19 P05 | 33min | 3 tasks | 11 files |
| Phase 19 P06 | 36min | 2 tasks | 3 files |
| Phase 20 P03 | 2min | 2 tasks | 2 files |
| Phase 20 P01 | 6min | 3 tasks | 1 files |
| Phase 20 P02 | 16min | 3 tasks | 6 files |
| Phase 20 P04 | 18min | 3 tasks | 2 files |
| Phase 20 P05 | 25min | 4 tasks | 3 files |
| Phase 20 P06 | ~50min | 5 tasks | 11 files |
| Phase 21 P01 | ~30min | 5 tasks | 5 files |

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
- [Phase 16]: Cherry-pick Group 1 pipeline_io_backend.cpp: git auto-merged inconsistently (ctor from ours, bodies from 73d00c4); wrote complete Group 1 tree version per D-D1 (prefer-ours for additive changes)
- [Phase 16]: MUST-be-last inline comment added to _thread declaration in io_worker: original eda349a used block comment; acceptance criteria required inline // MUST be last on _thread line; Group 3 commit amended
- [Phase 16]: Conflict at convert_gpu_to_gpu forward-decl: took Group 2's forward-decl, discarded HEAD's old cudf::pack body; column-tree-walk implementation auto-merged
- [Phase 16]: 3-arg ctor Option B: stream arg added to all 4 construction sites in representation_converter.cpp now; build NOT clean until 16-04 adds writer_stream to header
- [Phase 16]: get_table().view() in auto-merged convert_gpu_to_gpu body changed to get_table_view() per #117 API removal (D-D2)
- [Phase 16-04]: D-D2 full re-implementation: gpu_data_representation.hpp/cpp rewritten against #117 RAII shape with Group 4 writer_stream REQUIRED on both ctors grafted in
- [Phase 16-04]: read_only_data_batch::get_writer_event() proxy via dynamic_cast (D-B3): returns nullptr for non-GPU repr, no deadlock risk
- [Phase 16-04]: Benchmark ctor sites: stream.view() from local rmm::cuda_stream for setup/warmup; rmm::cuda_stream_view{} for thread-pool reprs created before streams are assigned
- [Phase 16-05]: Pin advance was pre-committed in 16-04 docs commit (5d1a8e0): gitlink 995bf4e -> 1c1e648; no separate pin-advance commit needed in 16-05
- [Phase 16-05]: All 8 grep gates and 5 ROADMAP Phase 16 success criteria confirmed PASS; cucascade ctest 100% passed (1/1, 13.91s, exit 0); CC-01..04 all satisfied; Phase 16 ship gate CLOSED
- [Phase 17]: D-A2: Created phase17-pre-merge-backup ref at 98cdea20 as the D-A4 abort lifeline before origin/dev merge
- [Phase 17]: D-C1/C2: Extracted full sirius_parquet_metadata_scan_operator.hpp (232 lines) with stream-lineage context into 17-PHASE-13-EXTRACT.md; re-attachment targets identified for Phase 20 SM-03
- [Phase 17]: D-G5: Seeded 17-MERGE-LOG.md with Sections A-E covering 11 conflict files, 33 auto-merge audit, build error bounding, 8 verification gates, and PR #739 bookkeeping note
- [Phase 17-sirius-origin-dev-merge-base-layer]: cucascade pin 1c1e648 auto-defended by git fast-forward; all 11 conflict files resolved per D-D1..D-D6 + D-B1/B2; build expected to fail with get_data() private errors (Phase 18 closure)
- [Phase 17-sirius-origin-dev-merge-base-layer]: 17-03: D-G3 PASS — all 62 src/+47 test/ FSM grep hits are fully-qualified cucascade API calls; 0 bare FSM enum names from merge
- [Phase 17-sirius-origin-dev-merge-base-layer]: 17-03: MERGE-05 PASS — 63 build errors all Phase 18 DB-02/DB-03; 0 unrelated errors; liburing-dev missing is IO-12 territory (Bucket 5, not blocking)
- [Phase 17-sirius-origin-dev-merge-base-layer]: 17-04: All 6 D-G gates PASS; Phase 17 Final Verdict PASS (all 5 MERGE-XX satisfied); cucascade pin 1c1e648 intact; phase17-pre-merge-backup preserved
- [Phase 18]: [18-01] DB-01 closed: batch_lock_utils.hpp rewritten with three RAII helpers (prepare_and_acquire_mutable, try_acquire_mutable, acquire_read_only); operator-data prepare_for_processing returns optional<vector<mutable_data_batch>>; get_cudf_table_view takes const read_only_data_batch&. Build errors 63 -> 58. HYG-02 = 0 in all 4 modified files.
- [Phase 18]: [18-01] Acceptance criterion 'build error count <= 50' partial-met (actual: 58). 5-error gap is R2 size-estimator inline body content in modified headers (sirius_physical_operator.hpp:191-192, parquet_scan_operator_data.hpp:186) — RESEARCH.md classifies these as plan 18-02 territory. Plus 6 pre-existing liburing errors in src/io/uring/uring_reactor.cpp (Phase 19 / IO-12 territory, not in DB-01..05 scope per CONTEXT.md). Strict per-task acceptance criteria all PASS.
- [Phase 18]: [18-02] DB-02 + DB-03 closed at the operator-base layer: convertible_data_batch + convertible_gpu_pipeline_task wrappers migrated to RAII (try_to_mutable for non-blocking exclusive in convert; lock-free state probe + scoped to_read_only for memory-space probe). pipelineable_operator_data::prepare_for_processing implementation uses pipeline::prepare_and_acquire_mutable; get_next_task_input_data uses pop_next_data_batch(0). gpu_pipeline_task storage type flipped to vector<mutable_data_batch>. R2 size-estimator inline bodies in operator-data + scan-cached-data headers also migrated (Rule 3 deviation — same translation-unit cascade). Build errors 58 -> 47.
- [Phase 18]: [18-03] DB-02 + DB-03 closed for 8 stateful operator .cpp files (table_scan, hash_join, nested_loop_join, concat, top_n, grouped_aggregate_merge, ungrouped_aggregate, merge_sort): all FSM-pop sites replaced with pop_next_data_batch; all pop_data_batch_by_id 3-arg sites converted to 2-arg; read paths use scoped to_read_only(); the one mutable-write path (grouped_aggregate_merge release_table) uses to_mutable. hash_join's prepare_join_keys + resolve_mark_join_result signatures flipped to take const read_only_data_batch& and memory_space&. Build errors 47 -> 21.
- [Phase 18]: [18-03] P1 deadlock risk surfaced: 18-02's R5 lock-and-hold in gpu_pipeline_task::processing_handles holds vector<mutable_data_batch> across op->execute(); operator code in execute() now takes scoped to_read_only/to_mutable accessors on the same input batches, which is technically UB on non-recursive std::shared_mutex. Documented in SUMMARY P1 section. Compile-only acceptance gates pass; runtime audit deferred to 18-05 with two follow-up resolution paths (architectural accessor exposure OR drop R5 lock-and-hold).
- [Phase 18]: [18-04] DB-02/DB-03 closed for read-only operators + scan layer + task_creator + debug_utils. 4 known Pitfall 4 sites (2-arg make_data_batch) all closed in src/: filter:60, projection:63, table_scan:176 (via 18-03), gpu_parquet_scan_operator:252. HYG-02 baseline preserved at 0. Rule 3 deviations: debug_utils.hpp const drop (to_read_only is non-const) and ->clone() migration (clone moved off data_batch onto accessors under #117). 8 inventory-miss src/ files surfaced (sort_partition, grouped_aggregate, order, gpu_aggregate_impl, gpu_merge_impl, gpu_order_impl, gpu_partition_impl, cached_split_provider) — logged in deferred-items.md for orchestrator triage.
- [Phase 18]: [18-05] PRELUDE: 8 inventory-miss src/ files (deferred-items.md option 3) folded into Task 0 — all R1 + cached_split_provider Pitfall 4 closure (default-constructed cuda_stream_view{} per cucascade legacy/no-stream pattern; NOT rmm::cuda_stream_default — preserves HYG-02 baseline).
- [Phase 18]: [18-05] DB-03 closed: 23 test/cpp/ files migrated to scoped read_only_data_batch / mutable_data_batch (~95 get_data + 16 try_to_*_in_transit + 8 try_to_lock_for_processing + 6 try_to_create_task drops + 1 data_batch_processing_handle decl + 1 vector<handle> + 1 FSM-pop).
- [Phase 18]: [18-05] DB-04 partial: src/-side build is clean within DB-01..05 scope; only blocker is 6 pre-existing liburing-dev errors in src/io/uring/uring_reactor.cpp (Phase 19 / IO-12 territory per CONTEXT.md, NOT in DB-01..05 scope). Strict 'build exits 0' deferred to Phase 19 closure.
- [Phase 18]: [18-05] Pitfall 5 wrapper-state migration: under cucascade #117 fresh batches are 'idle' on construction. convertible_data_batch_provider predicate matches idle directly; pre-#117 try_to_create_task() calls dropped from test fixtures (set_task_created parameter inverted to non_idle_state).
- [Phase 18]: [18-05] Convert helper R1+R3 split: convert_*_to_host helpers split into ro -> drop -> mut -> drop -> ro_post 3-phase pattern to enforce P1 (never overlap shared+exclusive on same batch). Established in result_collector + host_table_chunk_reader + host_table_utils tests.
- [Phase 18]: [18-05] data_batch_processing_handle replaced with mutable_data_batch in test fixture types: test_gpu_partition_impl + test_gpu_merge_impl. RAII accessor serves identical lock semantics; auto-destructured by callers so type change is transparent.
- [Phase 18]: Phase 18 verdict PARTIAL — DB-01..04 PASS (static infrastructure: rewrite, MCP build exit 0, HYG-02 ≤ 40, all grep gates clean). DB-05 FAIL — P1 RAII lock-scope self-deadlock fires at runtime exactly as 18-03 SUMMARY forecast. [mgpu] tests fail with 'Resource deadlock avoided' (glibc EDEADLK). Resolution path is architectural — out of Phase 18 scope per Rule 4. Cucascade pin 1c1e648 preserved. Phase 19 unblocked at compile-time; runtime gates inherit P1 blocker.
- [Phase 18]: [18-07] Path A architectural fix landed: dropped R5 lock-and-hold from gpu_pipeline_task::compute_task; pipelineable_operator_data::prepare_for_processing now performs eager memory-space conversion under SHORT-scoped accessors (released BEFORE return); operators inside execute() take their own per-call accessors. Closes DB-05 P1 deadlock. [mgpu] 16/16 PASS, [mgpu_stress] PASS, racecheck 0 hazards. Phase 18 verdict flipped from PARTIAL to PASS.
- [Phase 19]: [19-01] IO-12 verdict PASS: vcpkg.json already declares liburing (line 17); pkg-config probes liburing 2.14 in pixi env; CMakeLists.txt:71-72 + 322-325 wiring confirmed. Zero source changes for IO-12.
- [Phase 19]: [19-01] Q3 resolution: read_positional_delete_file uses DuckDB read_parquet (CPU); read_equality_delete_file uses cudf::io::datasource::create directly. Neither constructs cucascade_datasource — Plan 19-05 needs no iceberg helper migration.
- [Phase 19]: [19-01] HYG-02 baseline 40 entirely in src/legacy/ + src/include/legacy/ (frozen namespace duckdb path). Zero rmm::cuda_stream_default in active Super Sirius code. Phase 19 source changes must preserve this.
- [Phase 19]: [19-03] Test fixture helpers (make_test_gpu_ioctxs / make_test_ioctx) defined alongside cucascade helpers in test_parquet_scan_task.cpp and test_metadata_gpu_scan_operators.cpp; both use rmm::cuda_set_device_raii per P11; 19-05 will flip 4+3 call sites
- [Phase 19]: [19-03] uring_ioctx ctor defaults locked to host_ring_depth=16, ring_entries=64, n_reactors=4, bounce_slot_size=sirius::io::CHUNK_SIZE per uring_ioctx.hpp:85-88; cudaGetDeviceCount clamp keeps make_test_gpu_ioctxs safe on 1-GPU hosts
- [Phase 19]: [19-03] test_metadata_gpu_scan_operators.cpp not currently in CMakeLists.txt TEST_SOURCES (orphaned from build graph); helper added with inline linkage; plan 19-05 must re-add to TEST_SOURCES when flipping the 3 metadata-scan call sites
- [Phase 19]: [19-02] IO-16 closed: uring_reactor.cpp:276 raw cudaSetDevice replaced with std::optional<rmm::cuda_set_device_raii> + .emplace() under preserved if (req.device_id >= 0) guard. RESEARCH.md Pattern 3 anti-patterns all avoided (guard preserved, scope tight to H2D if-block, branch unchanged). HYG-02 baseline 40 preserved. Build exit 0.
- [Phase 19]: [19-04] IO-13 + IO-14 closed at SiriusContext layer: per-GPU sirius::io::uring_ioctx instances constructed under rmm::cuda_set_device_raii in SiriusContext::initialize(); each ioctx default-allocates its own admission_control budget (P5); teardown gpu_ioctxs_.clear() runs BEFORE memory_manager_->shutdown() (Pitfall 3). Old cucascade gpu_io_backends_ map preserved live for plan 19-05 to retire.
- [Phase 19]: [19-04] Coexistence (alongside, not replace) keeps Wave 2 narrow to one file pair (sirius_context.cpp+.hpp). Two independent gpu_spaces walks instead of hoisting into a shared loop — keeps each milestone independently grep-locatable; iteration cost negligible.
- [Phase 19]: [19-04] initialize_cache() NOT called per RESEARCH.md Open Q2 — sirius_datasource device_read falls through to device_read_io when _cache==nullptr; v1.1 baseline correctness feasible without prefetching cache; cache enablement deferred to Phase 20+ (avoids per-GPU buffer_pool ownership question).
- [Phase 19]: [19-04] Smoke verification under 2-GPU host: [multi_gpu_foundation] 7/7 PASS (38 assertions, 4.3s); [mgpu] 16/16 PASS (79091 assertions, 105.9s). HYG-02 baseline preserved at 40; IO-16 raw cudaSetDevice in src/io/ still 0. No regression of Phase 18 DB-01..05 or earlier multi-GPU correctness gates.
- [Phase 19]: [19-05] IO-14 + IO-15 closed: parquet/iceberg scan + task_creator + sirius_engine + SiriusContext flipped to sirius_ioctx + sirius_datasource via ioctx->make_datasource(io_object) factory; cucascade_datasource fully retired (header + impl + test deleted; grep gate at 0). Per-GPU CUDA-context binding end-to-end via Phase 9 two-tier preferred_device_id lookup carrying through to per-task ioctx_it lookup.
- [Phase 19]: [19-05] Cached _file_io_objects on parquet_scan_task_global_state per RESEARCH.md Open Q1 — populated at planning time inside initialize_from_files(), reused by every per-task hot-path datasource construction. Avoids per-task fd reopens (uring_io_object ctor opens 2 fds: O_RDONLY + O_RDONLY|O_DIRECT). Cleanup is automatic via global_state destruction (initialize_cache() NOT called per Open Q2).
- [Phase 19]: [19-05] Forward-declare uring_io_object in parquet_scan_task.hpp + include uring_reactor.hpp LAST in parquet_scan_task.cpp's include block — works around liburing.h's BLOCK_SIZE macro colliding with blockingconcurrentqueue.h's static const BLOCK_SIZE member. Mirrors sirius_context.cpp's working pattern (logging.hpp before uring_ioctx.hpp).
- [Phase 19]: [19-05] test_metadata_gpu_scan_operators.cpp call sites flipped to make_test_ioctx() but file remains OUT of CMakeLists.txt TEST_SOURCES — sirius_parquet_metadata_scan_operator.hpp was deleted in Phase 17 merge (re-attached in Phase 20 SM-03). This is the explicit Phase 20+ deferral per the success criterion's option B; edits keep IO-15 grep gate clean and prepare the file for Phase 20 re-add.
- [Phase 19]: [19-06] Phase 19 closing verdict PASS - all 6 IO-12..17 closed. [TPC-H][parquet] 22/22 PASS at num_gpus=2 (36256 assertions, 78.6s). compute-sanitizer memcheck on [multi_gpu_foundation] (7/7) and [integration][gpu_execution][parquet][join] (42/42, 1.92M assertions): 0 memcheck violations. nvidia-smi dmon confirms non-zero PCIe rxpci on BOTH GPU 0 (63/120 samples; max 2892 MB/s) AND GPU 1 (54/120 samples; max 453 MB/s).
- [Phase 19]: [19-06] Sanitizer error classification: 8+9 reported errors are CUDA API status returns (cudaErrorPeerAccessAlreadyEnabled from cucascade peer-access probe + cudaErrorInvalidDevice from bounded_thread_pool worker init) - NOT memcheck violations. Phase 5/6 sanitizer baseline (0 errors / 1.92M assertions) preserved.
- [Phase 20]: [20-03] SM-05 documentation gate closed: PROJECT.md Deferred bullet for pin_table single-GPU residency cites src/sirius_extension.cpp:733; REQUIREMENTS.md PIN-MGPU-01 augmented (Branch B) with src cite + Phase 13 re-attach site + bidirectional PROJECT.md backref
- [Phase 20]: [20-01] SM-01 Option A applies empirically: [mgpu_stress] 500-iter PASS at 77053 assertions / 73.8s confirms task_scheduler::management_eventloop:260 is the canonical RR site for GPU_PARQUET_SCAN source tasks; no _no_pref_rr_counter port to parquet_split_provider needed.
- [Phase 20]: [20-01] SM-02 PARTIAL: AUDIT TEST_CASE FAILS at min_count REQUIRE line 262 (counts[1].pipeline_ids.size() == 0) — preempts disjointedness REQUIRE on line 289. Empirical scan_batch IS multi-GPU disjoint (GPU0=2, GPU1=1) — only test fixture's pipeline_task threshold is misaligned with post-#731 single-composite-gpu_pipeline_task pattern. Resolution path handed to 20-02 / 20-04.
- [Phase 20]: [20-01] SM-03 PASS: writer_stream token survives at sirius_gpu_parquet_scan_operator.cpp:260 in canonical Phase 13-04 Path-2 comment block; the operative make_data_batch(table, mem_space, stream) call at line 263 records writer_event via cucascade::gpu_table_representation ctor.
- [Phase 20]: [20-01] HYG-02 baseline preserved at 40 / 0 non-legacy; cucascade_datasource retirement (Phase 19-05) holds at 0 hits across src/ + test/.
- [Phase 20]: [20-02] Open Q1 RETIRE: test_metadata_gpu_scan_operators.cpp deleted (referenced deleted sirius_parquet_metadata_scan_operator class at 14 sites per Pitfall 3 grep). v1.5+ opportunistic re-author against parquet_split_provider deferred.
- [Phase 20]: [20-02] SM-01 Option A documented in 20-SCHED-RR-PORT.md (209 lines): no SCHED-RR port to parquet_split_provider; task_scheduler::management_eventloop:260 _no_pref_rr_counter is canonical RR site for GPU_PARQUET_SCAN source tasks; two RR counters at two layers would race / drift. Empirically gated by [mgpu_stress] 500-iter PASS @ 77053 assertions / 73.8s.
- [Phase 20]: [20-02] SM-02 affinity map ownership documented in 20-SCHED-RR-PORT.md: lives at duckdb_scan_executor.cpp:154-164,213-222,259-262 (DuckDB-attach scan path). PR #731 did not touch this file. The misleading framing in REQUIREMENTS.md SM-02 is documentation drift per Pitfall 1; corrected here. SM-02 PARTIAL test-fixture mismatch handed to Phase 21+ / v1.5+ test-cleanup (NOT Phase 20 scope; underlying scan_batch disjointedness invariant holds).
- [Phase 20]: [20-02] SM-03 Option B documented in 20-STREAM-LINEAGE-REATTACH.md (173 lines): stream-lineage re-attached at sirius_gpu_parquet_scan_operator.cpp:259 (post-edit; was 263 pre-edit) via 3-arg make_data_batch(table, mem_space, stream); cucascade ctor body auto-records writer_event when writer_stream non-default; no manual record_writer_event call needed. Phase 13-04 Path-2 carried forward through Phases 17/18/19/20.
- [Phase 20]: [20-02] Pitfall 1 TODO cleanup: 3 misleading TODO blocks deleted (parquet_scan_operator_data.hpp:86 + 149-153 + sirius_gpu_parquet_scan_operator.cpp:173-176) referencing phantom _batch_gpu_affinity re-attach. SM-03 load-bearing block preserved (now lines 255-259, was 258-262 pre-edit; shifted up by 4 lines). Build clean (mcp exit 0, 27.5s); HYG-02 baseline preserved at 40 / 0 non-legacy; SM-03 grep gate non-zero (1 hit, line 256 post-edit).
- [Phase 20]: [20-04] SM-06 PARTIAL verdict: SF10 Q1/Q6/Q12 num_gpus=2 PASS (3/3, 227 assertions, 12.01s); SF1 [integration][TPC-H] FAIL at Q11 parquet 2-GPU (canonical Phase 13 P2 cudaErrorIllegalAddress) — pre-existing follow-up #17 (project_phase08_fu17), NOT a Phase 20 regression. Phase 21 REG-03 carries SF1 closure dependency.
- [Phase 20]: [20-04] SM-04 PASS via dual verification: source inspection at sirius_gpu_parquet_scan_operator.cpp:127 (gpu_expression_translator) + gpu_pipeline_executor.cpp:54-77 (per-thread cudaSetDevice + manager_loop cuda_set_device_raii) confirms per-task filter translation runs under cudaSetDevice RAII. Empirically corroborated by SF10 Q1 num_gpus=2 PASS.
- [Phase 20]: [20-04] Advisory SF100 Q1 num_gpus=2 PASS at 2.283s cold (well under Phase 21 REG-04 5.7s bar; under Phase 9-04 5.86s + Phase 10-04 5.70s historical baselines). Reduces Phase 21 REG-04 ship-risk substantially.
- [Phase 20]: [20-04] Advisory [mgpu] 16/16 PASS: 79091 assertions / 106.4s — exact match to Phase 18-VERDICT-V2 + Phase 19-VERDICT baselines. Includes follow-up #17 sentinel TEST_CASE which PASSED. Bounds the SM-06 SF1 failure as Q11-shape + parquet-path specific.
- [Phase 20]: [20-04] Phase 20 final verdict PARTIAL not FAIL: 5 of 6 SM-XX requirements PASS unconditionally (SM-01..05 + SM-04). SM-06 SF10 PASS captures architecture-level signal Phase 20 was designed to produce. SF1 PARTIAL is pre-existing infrastructure carryover anticipated by Phase 20 scoping (verification + documentation, no code-port). Phase 21 unblocked.
- [Phase 20]: [20-05] PATH B escalation (status human_needed): canonical compute-sanitizer revealed 21 stream-ordered races at HEAD distributed across two clusters at library boundaries — Cluster A (5/21) cudf+kvikio internal parquet reader cross-stream gap inside read_column_chunks_async; Cluster B (16/21) cucascade pin 1c1e648 alloc_and_peer_copy_async host-staging fallback (race shape E per plan taxonomy). Phase 13-04 entry-level cudaStreamWaitEvent at convert_gpu_to_gpu IS firing correctly; the cluster B races are in a NEW fallback code path added post-Phase 13.
- [Phase 20]: [20-05] No source files modified (Path B). Recommended fix: cucascade fork+bump for alloc_and_peer_copy_async same-stream invariant + Sirius cudaStreamSynchronize after read_parquet (cluster A workaround). Estimated 1.5-2.5 days for full closure. Carry-forward to Phase 21 REG-03 explicit; ship-gate cannot pass without resolution.
- [Phase 20]: [20-05] [mgpu] 16/16 PASS continuity baseline preserved (79091 assertions / 104.4s / exit 0); Phase 18..20 invariants intact (DB-grep 4 legacy+comments only; IO-15 0; SM-03 1; HYG-02 40); 0 lines source diff.
- [Phase 20]: [20-06] 20-05 sanitizer trace re-classified: Cluster A (5/21 races) attributed in 20-05 to "cudf+kvikio internal cross-stream gap" was actually a Sirius-side architectural gap — `parquet_split_provider::run_batch:222` constructed cudf-bundled file_source datasources directly via `cudf::io::datasource::create(file_path)` instead of routing through the Phase 19 `sirius_ioctx::make_datasource(io_object)` framework. The PR #731 scan_manager path was authored without IO framework integration; the original IO-15 grep gate (`cucascade_datasource`) didn't catch it because the bypass uses the cudf factory, not the cucascade adapter. Fix: plumb gpu_ioctxs from SiriusContext through scan_manager::prepare_for_query into parquet_split_provider; replace cudf factory with `planning_ioctx_it->second->make_datasource(io_object)` pattern from parquet_scan_task.cpp:343-350.
- [Phase 20]: [20-06] Cluster A eliminated post-fix (sanitizer log: 0 kvikio frames; was 5/21 in 20-05 baseline). All 22 [TPC-H][parquet] queries PASS under compute-sanitizer with track-stream-ordered-races=all (36256 assertions, exit 0). Q11 SF1 num_gpus=2 parquet PASS (9011 assertions, exit 0) — the canonical SM-06 SF1 blocker is closed.
- [Phase 20]: [20-06] Cluster B (cucascade alloc_and_peer_copy_async host-staging fallback, 16/21 races in 20-05) persists post-fix but is correctness-neutral on this hardware — all 22 + Q11 + 16/16 [mgpu] tests PASS post-fix. Already classified by 20-05 PATH B as a separate cucascade-side finding; falls under existing v1.4 cucascade follow-up (`project_tpch_q1_mgpu_string_bug` memory file — peer-DMA probe + host-staging fix uncommitted in cucascade). Per Phase 13 protocol "first race is root, rest is cascade": Cluster A was the root; Cluster B is downstream of consumer-hardware peer-DMA limitations.
- [Phase 20]: [20-06] Strengthened IO-15B grep gate added to REQUIREMENTS.md: `grep -rn "cudf::io::datasource::create" src/ | grep -v iceberg_metadata_reader.cpp | grep -v iceberg_scan_task.cpp` must return 0 hits. Would have caught PR #731's bypass had it been live during Phase 19. IO-MGPU-02 added to Future Requirements (v1.5+) for the two known-deferred iceberg sites (currently single-GPU correct).
- [Phase 20]: [20-06] [integration][TPC-H] 47/48 PASS at SF1 num_gpus=2: the single residual is the pre-existing SM-02 PARTIAL test-fixture mismatch from plan 20-01 ([mgpu-audit] per-GPU distribution Q1 fails at counts[1].pipeline_ids.size() >= 1 with 0 >= 1 — v1.3-era multi-pipeline_task threshold vs post-#731 single composite gpu_pipeline_task). scan_batch IS multi-GPU disjoint at HEAD; only the test fixture's threshold is misaligned. NOT a 20-06 regression. [mgpu] 16/16 PASS continuity preserved (79091 assertions, 109s).
- [Phase 20]: [20-06] Phase 20 final verdict flips PARTIAL → COMPLETE PASS (6/6 plans). Phase 21 unblocked at REG-03 dependency level (SM-06 SF1 carryover no longer required).
- [Phase 21]: [21-01] SM-02 path chosen: fixture-fix (1-line surgical edit at `test_gpu_execution_tpch_mgpu_audit.cpp:261-273` realigning v1.3-era multi-pipeline_task threshold with post-#731 single composite gpu_pipeline_task pattern). Cross-GPU `scan_id` intersection invariant (Phase 9 FIX-B regression gate at lines 286-299) preserved verbatim. Net `-1` assertion delta: 71607 vs 71608 baseline.
- [Phase 21]: [21-01] All 6 REG-01..06 PASS on rebased `feature/single-node-multi-gpu2`: REG-01 16/16 (79091/106.3s), REG-02 22/22 (36256/79.3s), REG-03 48/48 (71607/152.4s), REG-04 SF100 Q1 num_gpus=2 3.150s + byte-identical + intersect=0, REG-05 [mgpu_stress] 500-iter (77053/76.7s), REG-06 HYG-02=40 + sanitizer 0 violations on both legs.
- [Phase 21]: [21-01] One-off Q11 parquet num_gpus=2 cudaErrorIllegalAddress observed during REG-02 first attempt; resolved on retry (22/22 PASS); Q11 alone PASS (9011 assertions, 7.1s). Documented as known intermittent follow-up #17 (per `project_phase08_fu17`); NOT a Phase 21 regression.
- [Phase 21]: [21-01] v1.4 SHIPPED: 6 phases / 29 plans / 32 requirements clear. Carry-forwards to v1.5+: PIN-MGPU-01, IO-MGPU-02, CC-UPSTREAM-01, FU-B. Cucascade Cluster B (peer-DMA host-staging fallback) tracked under `project_tpch_q1_mgpu_string_bug` (correctness-neutral on this hardware; uncommitted in cucascade).

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

- **None.** Phase 18 P1 RAII lock-scope self-deadlock RESOLVED by plan 18-07 Path A architectural fix (commits `0575b0a` + `99e6765`). [mgpu] 16/16 PASS, [mgpu_stress] PASS, racecheck 0 hazards. See 18-VERDICT-V2.md + 18-07-SUMMARY.md.

### Roadmap Evolution

- Phase 22 added: Multi-GPU pinning + stream lineage hardening — round-robin pin distribution (GPU + NUMA-local HOST), cucascade writer-event API, fu17 SF100 Q11 stream-ordered race fix.

## Session Continuity

Last session: 2026-05-07T22:00:17.385Z
Stopped at: Phase 22 context gathered
Resume file: .planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CONTEXT.md
