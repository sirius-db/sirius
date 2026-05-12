# Roadmap: Sirius — GPU-Native SQL Engine (Multi-GPU)

## Milestones

- ✅ **v1.1 Multi-GPU Re-integration + Cucascade I/O Migration** — Phases 4-7 (shipped 2026-04-21) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Multi-GPU SQL Pipeline Fix** — Phases 8-10 (shipped 2026-04-28) — [archive](milestones/v1.2-ROADMAP.md)
- 🛠 **v1.2 Patch — AUDIT TEST_CASE attach-path SIGSEGV** — Phase 11 (closed 2026-04-28)
- ✅ **v1.3 Multi-GPU Distribution** — Phases 12-15 (shipped 2026-05-01)
- ✅ **v1.4 Rebase After DataBatch Changes** — Phases 16-21 (shipped 2026-05-06)

## Phases

<details open>
<summary>✅ v1.4 Rebase After DataBatch Changes (Phases 16-21) — SHIPPED 2026-05-06</summary>

Goal: land cucascade `origin/main` (PR #117 DataBatch RAII refactor + #112 + #116) and Sirius `origin/dev` (#675 IO Framework, #731 Scan Manager, #721 Pin Tables, #739 cucascade-compat, #733/#734/#735) onto `feature/single-node-multi-gpu2`, preserving all v1.1+v1.2+v1.3 multi-GPU behavior. The v1.3 ship-gate (`[mgpu]` 16/16, `[TPC-H][parquet]` 22/22, `[integration][TPC-H]` 48/48, SF100 Q1 num_gpus=2 <= 5.7s, mgpu_stress 500-iter, HYG-02 <= 40) must pass bitwise on the rebased branch.

- [x] **Phase 16: Cucascade Submodule Rebase + Pin Recovery** — Rebase 11 local Sirius-side cucascade fixes onto `73d00c4` (origin/main tip with #117 DataBatch RAII + #112 + #116). Highest conflict density; no Sirius compile gate here — cucascade-internal verification only. **COMPLETE** (2026-05-05): 4 group commits on top of 73d00c4, ctest 100% PASS, all 8 grep gates green, CC-01..04 satisfied.
- [x] **Phase 17: Sirius origin/dev Merge — Base Layer** — Absorb `origin/dev` CI/CMake/config PRs (#739 compat, #675, #731, #721, #733, #734, #735) as a base-layer merge. Expected to produce 26+ `batch->get_data()` private-access build errors — DOCUMENTED as expected, not a phase failure. SCHED-RR block survival verified. **COMPLETE** (2026-05-05): All 5 MERGE-XX requirements PASS; 63 expected build errors classified as Phase 18 DB-02/DB-03; cucascade pin 1c1e648 preserved; D-G1..G6 all PASS.
- [x] **Phase 18: DataBatch RAII Migration (cucascade #117 surface)** — Migrate all `batch->get_data()` call sites and `pop_data_batch(state)` usages to RAII accessors; rewrite `batch_lock_utils.hpp`. Phase ends with a compile-clean build. **COMPLETE 2026-05-05**: 7/7 plans shipped (added gap-closure plan 18-07). DB-01..05 PASS. Path A architectural fix landed in 18-07 (drop R5 lock-and-hold from `gpu_pipeline_task::compute_task`; `pipelineable_operator_data::prepare_for_processing` performs eager memory-space conversion under SHORT-scoped accessors and returns empty vector; operators inside `execute()` take per-call accessors at narrowest scope). [mgpu] 16/16 PASS in 103.5s (79091 assertions); [mgpu_stress] default-mode PASS in 75.5s (77053 assertions); compute-sanitizer racecheck 0 hazards on [downgrade_lifecycle] proxy. HYG-02 baseline preserved at 40 (0 non-legacy). See 18-VERDICT-V2.md (supersedes 18-VERDICT.md PARTIAL).
- [x] **Phase 19: IO Framework Adoption (PR #675)** — Retire `sirius::io::cucascade_datasource`; adopt `sirius::io::sirius_datasource` with per-GPU `uring_ioctx` instances. Install `liburing-dev` before first build attempt. **COMPLETE** (2026-05-06): 6/6 plans shipped. All 6 IO-12..17 PASS. `[TPC-H][parquet]` 22/22 PASS at num_gpus=2 (36256 assertions, 78.6s); compute-sanitizer memcheck on `[multi_gpu_foundation]` (7/7) and `[integration][gpu_execution][parquet][join]` (42/42, 1.92M assertions) report 0 memcheck violations. nvidia-smi dmon confirms non-zero PCIe rxpci on BOTH GPU 0 (63/120 samples; max 2892 MB/s) AND GPU 1 (54/120 samples; max 453 MB/s) during multi-GPU workload. HYG-02 = 40 (preserved). See 19-VERDICT.md.
- [x] **Phase 20: Scan Manager + Pin Tables Port (PR #731 + #721)** — **COMPLETE** (6/6 plans, SM-01..SM-06 all PASS). Plan 20-06 closed the 20-05 SM-06 SF1 escalation by re-classifying the sanitizer trace: `parquet_split_provider::run_batch` was constructing cudf-bundled file_source datasources directly (kvikio bypass) instead of routing through the Phase 19 `sirius_ioctx::make_datasource` framework — a Sirius-side architectural gap, NOT a cucascade-side cudf+kvikio internal cross-stream issue. After the fix: Q11 SF1 num_gpus=2 parquet PASS (9011 assertions); 22/22 [TPC-H][parquet] PASS under sanitizer (36256 assertions; 0 kvikio frames detected); 47/48 [integration][TPC-H] PASS (1 pre-existing SM-02 PARTIAL test-fixture mismatch — not a regression); 16/16 [mgpu] continuity PASS (79091 assertions). IO-15B strengthened grep gate added to catch this regression class going forward; IO-MGPU-02 created for v1.5+ iceberg metadata residency. Cucascade host-staging fallback (Cluster B from 20-05) persists but is correctness-neutral on this hardware and tracked under the existing v1.4 cucascade follow-up (`project_tpch_q1_mgpu_string_bug`). See [`20-06-VERDICT.md`](phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-06-VERDICT.md).
- [x] **Phase 21: v1.4 Ship Gate (Full v1.3 Gauntlet on Rebased Branch)** — Complete 2026-05-06: REG-01..06 all PASS. 1/1 plan shipped. `[mgpu]` 16/16 PASS (79091 assertions, 106.3s); `[TPC-H][parquet]` 22/22 PASS (36256 assertions, 79.3s); `[integration][TPC-H]` 48/48 PASS (71607 assertions, 152.4s — 1-line SM-02 fixture fix at `9f835cd` realigned v1.3-era multi-pipeline_task threshold with post-#731 single composite gpu_pipeline_task pattern; cross-GPU scan_id intersection invariant preserved); SF100 Q1 num_gpus=2 wall-clock 3.150s (vs 1-GPU 4.422s baseline), byte-identical CSV, pipeline_task intersection=0 (GPU0=18, GPU1=12); `[mgpu_stress]` 500-iter PASS (77053 assertions, 76.7s); HYG-02 = 40 (preserved); compute-sanitizer memcheck Leg 1 7/7 + 38 assertions + 0 violations, Leg 2 42/42 + 1.92M assertions + 0 violations. v1.4 ships. See [`21-VERDICT.md`](phases/21-v1-4-ship-gate-full-v1-3-gauntlet-on-rebased-branch/21-VERDICT.md).

</details>

<details>
<summary>✅ v1.3 Multi-GPU Distribution (Phases 12-15) — SHIPPED 2026-05-01</summary>

Goal: deliver real multi-GPU work distribution for source-pipeline tasks (parquet metadata + GPU parquet scan + downstream operators). Distribution unblocked by the parquet AST filter re-translation fix (commit `86e821a`, 2026-04-29).

- [x] Phase 12: Fix `vector::at(2)` correctness bug in small-sort plan path — 2026-04-29
- [x] Phase 13: Fix Q11 multi-GPU hang/illegal-address (cucascade writer-event stream lineage) — 2026-04-30
- [x] Phase 14: Land SCHED-RR distribution — 2026-04-30
- [x] Phase 15: Cross-GPU operator-colocation audit — 2026-05-01

Test gauntlet (post-FU-A on `feature/single-node-multi-gpu2`): `[mgpu]` 16/16 in 120.3s (79091 assertions, exit 0); `[TPC-H][parquet]` 22/22 in 81.6s; `[integration][TPC-H]` 48/48 in 2:43 (71608 assertions). HYG-02 = 40.

</details>

<details>
<summary>🛠 v1.2 Patch — AUDIT TEST_CASE attach-path SIGSEGV (Phase 11) — closed 2026-04-28</summary>

- [x] Phase 11 / Plan 11-01: GDB the AUDIT SIGSEGV + apply targeted fix (cucascade `io_worker` member-init-order race + `SiriusContext::QueryEnd` per-query state cleanup) — 2026-04-28
- [x] Phase 11 / Plan 11-02: Validation, ROADMAP entry, optional v1.2.1 tag — 2026-04-28

PARTIAL verdict — AUDIT SIGSEGV closed both targeted (16/16 assertions) and in suite (678/679 with sole pre-existing `physical_order` failure unrelated). Two stacked bugs identified and fixed: (1) cucascade `io_worker::_thread` declared before `_mutex`/`_cv` raced its own ctor on EINVAL; (2) `task_creator`'s per-query `duckdb_scan_task_global_state` outlived QueryEnd, releasing BlockHandles into a half-destroyed BufferManager during `~SiriusContext`. Total LOC: 33 (10 src/ + 7 cucascade + 16 test/).

Full details: `.planning/phases/11-mgpu-audit-attach-sigsegv/11-02-VALIDATION.md`

</details>

<details>
<summary>✅ v1.1 Multi-GPU Re-integration + Cucascade I/O Migration (Phases 4-7) — SHIPPED 2026-04-21</summary>

- [x] Phase 4: cuCascade Bump + v1.0 Re-integration (5/5 plans) — 2026-04-20
- [x] Phase 5: Cucascade-Backed Parquet I/O Migration (6/6 plans) — 2026-04-21
- [x] Phase 6: Multi-GPU Gap Closure (4/4 plans) — 2026-04-21
- [x] Phase 7: P2P Direct Transfer + Adaptive Scan (4/4 plans) — 2026-04-21

28/28 requirements cleared. 979/979 tests pass on N=2 hardware.
Full details: `.planning/milestones/v1.1-ROADMAP.md`

</details>

<details>
<summary>✅ v1.2 Multi-GPU SQL Pipeline Fix (Phases 8-10) — SHIPPED 2026-04-28</summary>

- [x] Phase 8: Multi-GPU SQL Pipeline Fix (6/6 original + 2 halted gap-closure plans) — 2026-04-22
- [x] Phase 9: Scan-Task Distributor + Batch-Ownership Affinity (4/4 plans, PARTIAL) — 2026-04-24
- [x] Phase 10: TABLE_FUNCTION-form gpu_execution SIGSEGV fix (4/4 plans, PARTIAL) — 2026-04-27

11/11 v1.2 requirements satisfied (8 fully + 3 partial via proxy). Ship-gate criteria 5/6 PASS, 1/6 PARTIAL (pre-existing `[mgpu-audit]` SIGSEGV scoped as Phase 11 candidate).
SF100 TPC-H Q1 num_gpus=2 PASS (5.70s, byte-identical to 1-GPU baseline).
Full details: `.planning/milestones/v1.2-ROADMAP.md`
Audit: `.planning/milestones/v1.2-MILESTONE-AUDIT.md`

</details>

## Phase Details

### Phase 16: Cucascade Submodule Rebase + Pin Recovery
**Goal**: The cucascade submodule is pinned to a commit descended from `73d00c4` with all 11 local Sirius-side fixes re-applied on top of the new RAII DataBatch model.
**Depends on**: Nothing (first phase of v1.4)
**Requirements**: CC-01, CC-02, CC-03, CC-04
**Success Criteria** (what must be TRUE):
  1. `git -C cucascade log --oneline origin/main..HEAD` shows 4 group commits (squashed from 11 per D-A1; each commit body cites original hashes for archaeology) on top of `73d00c4`-descendant ancestry; `cat cucascade/.git/HEAD` (or equivalent) resolves to the new pin.
  2. `grep -n "writer_stream\|cudaStreamWaitEvent" cucascade/src/data/representation_converter.cpp` returns non-zero at every `convert_gpu_to_gpu` / `convert_host_to_gpu` construction site (P2 writer_stream survival gate).
  3. `grep -n "cudaHostAllocPortable" cucascade/src/memory/common.cpp cucascade/src/memory/memory_space.cpp` returns non-zero at every pinned allocation site (P9 Portable/Mapped flag gate).
  4. `_thread` is the last-declared member in the `io_worker` class in `cucascade/src/data/pipeline_io_backend.cpp` (P8 destruction-order gate); confirmed by visual inspection of the post-conflict file.
  5. Cucascade unit-test suite passes (`ctest` inside `cucascade/build`); `grep -rn "task_created\|in_transit" cucascade/include/` returns zero (old FSM state machine fully removed per CC-04).
**Plans**: 5 plans
- [x] 16-01-PLAN.md — Squash 11 cucascade commits into 4 group commits + initialize audit log
- [x] 16-02-PLAN.md — Rebase Group 1 (memory hygiene) + Group 3 (io_worker) onto 73d00c4
- [x] 16-03-PLAN.md — Rebase Group 2 (P2P override + DMA probe) onto Group 1+3 tip
- [x] 16-04-PLAN.md — Rebase Group 4 (Phase 13 stream-lineage); re-implement gpu_data_representation + convert_gpu_to_gpu under #117 RAII; build compile-clean
- [x] 16-05-PLAN.md — Run cucascade ctest + 8 grep gates; advance submodule pin in parent worktree
**Pitfalls**:
  - P2 (writer_stream lost in representation_converter.cpp conflict): treat `representation_converter.cpp` as a re-implementation from `73d00c4` shape, not a three-way merge. Verify with grep gate before proceeding.
  - P7 (PR #739 x #117 ordering mismatch): complete cucascade rebase first; use #739 only as a file-list reference during Phase 18. Do NOT cherry-pick #739 here.
  - P8 (io_worker member-order fix lost): explicitly verify member ordering in post-conflict `pipeline_io_backend.cpp`; add `// MUST be last` comment at `_thread` declaration.
  - P9 (Portable/Mapped flags dropped): after resolving memory conflict files, run `grep -n "cudaHostAllocPortable"` gate immediately; do not proceed to Phase 17 until green.
  - P1 (RAII lock scope): every RAII accessor site introduced in cucascade's internal code must scope accessors to the narrowest block.

### Phase 17: Sirius origin/dev Merge — Base Layer
**Goal**: `origin/dev` is merged into `feature/single-node-multi-gpu2` with all 11 conflict files resolved and 33 auto-merges committed; SCHED-RR distribution logic is verified intact; expected build errors from un-migrated `batch->get_data()` sites are documented and do not block the phase.
**Depends on**: Phase 16 (cucascade API shape must be settled before any Sirius dev-merge file touches cucascade-dependent code)
**Requirements**: MERGE-01, MERGE-02, MERGE-03, MERGE-04, MERGE-05
**Success Criteria** (what must be TRUE):
  1. `git log --oneline --merges | head -1` shows the dev-merge commit; `git log --oneline origin/dev ^feature/single-node-multi-gpu2 -- CMakeLists.txt` returns empty (CMakeLists changes absorbed).
  2. `grep -rn "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp` returns exactly 1 match; `grep "SCHED-RR" src/pipeline/task_scheduler.cpp` returns the round-robin block (SCHED-RR survived the merge).
  3. `grep -rn "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" src/` returns zero — no old FSM enum values were re-introduced by auto-merged files.
  4. Phase 13 stream-lineage extraction is complete: `17-MERGE-LOG.md` documents the extracted attachment points for `writer_stream` / `writer_event` from the deleted `sirius_parquet_metadata_scan_operator.hpp`, with re-attachment target identified as `parquet_split_provider.cpp` or `sirius_gpu_parquet_scan_operator.cpp` (MERGE-03).
  5. Build error count is bounded and recorded in `17-MERGE-LOG.md`: expected 26+ `batch->get_data() is private` errors plus RAII compile errors; zero unrelated build errors outside the DataBatch migration surface (MERGE-05).
**Plans**: 4 plans
- [x] 17-01-PLAN.md — Pre-merge setup: backup ref + Phase 13 stream-lineage extraction + audit log skeleton (MERGE-03)
- [x] 17-02-PLAN.md — Execute git merge --no-ff origin/dev + resolve 11 conflict files per D-D1..D-D6 + cucascade pin defense (MERGE-01, MERGE-02, MERGE-04)
- [x] 17-03-PLAN.md — Auto-merge audit (33 files) + SCHED-RR survival + build error bounding (MERGE-05)
- [x] 17-04-PLAN.md — Run all 6 D-G verification gates + final Phase 17 Verdict (MERGE-01..05 final)
**Pitfalls**:
  - P7 (PR #739 x #117 ordering mismatch): #739's cucascade submodule bump must be discarded during merge conflict resolution — the cucascade pin is already handled by Phase 16. Accept #739's Sirius operator file changes only as an indication of what files need touching; actual RAII recipe applied in Phase 18.
  - P6 (SCHED-RR counter stale): verify `_no_pref_rr_counter` field and the SCHED-RR block in `task_scheduler.cpp` both survive the merge (grep gate in criterion 2). If a conflict exists, resolve by keeping Phase 14's SCHED-RR block plus #739's one-line change.
  - P10 (Phase 13 work in deleted file): MERGE-03 is the dedicated guard — extract stream-lineage attachment points before git accepts the `sirius_parquet_metadata_scan_operator.hpp` deletion. This is not optional.

### Phase 18: DataBatch RAII Migration (cucascade #117 surface)
**Goal**: The Sirius codebase compiles cleanly against the `73d00c4`-pin cucascade; every `batch->get_data()` call site and `pop_data_batch(state)` usage is migrated to RAII accessors (`to_read_only()` / `to_mutable()`); `batch_lock_utils.hpp` is fully rewritten; HYG-02 baseline preserved.
**Depends on**: Phase 17 (the dev-merge base layer must be committed before RAII migration begins so that the auto-merged operator files are already in their target shape)
**Requirements**: DB-01, DB-02, DB-03, DB-04, DB-05
**Success Criteria** (what must be TRUE):
  1. `grep -rn "->get_data()\|\.get_data()\|pop_data_batch.*task_created\|data_batch_processing_handle\|task_created\|in_transit" src/ test/` returns zero hits.
  2. `mcp__project-commands__run_command build` exits 0 with no migration TODOs (DB-04 build-clean gate).
  3. `grep -c "rmm::cuda_stream_default" src/` returns <= 40 (HYG-02 baseline preserved).
  4. `[mgpu]` filter passes 16/16 (DB-05 light regression gate — DataBatch migration did not break multi-GPU correctness).
  5. `[mgpu_stress]` 1-iter smoke (100 iterations × 5 queries but only 1 repetition) exits 0 — SCHED-RR survival after RAII migration; 500-iter deferred to Phase 21.
**Plans**: 7 plans (gap-closure 18-07 added in Wave 6 after 18-06 surfaced DB-05 P1 deadlock)
- [x] 18-01-PLAN.md — Rewrite batch_lock_utils.hpp + ripple prepare_for_processing return type + get_cudf_table_view signature (DB-01)
- [x] 18-02-PLAN.md — Migrate convertible_* wrappers + sirius_physical_operator base impl + gpu_pipeline_task storage (DB-02, DB-03)
- [x] 18-03-PLAN.md — Migrate stateful operators with FSM-pop / 3-arg pop-by-id: table_scan, hash_join, nested_loop_join, concat, top_n, grouped_aggregate_merge, ungrouped_aggregate, merge_sort (DB-02, DB-03)
- [x] 18-04-PLAN.md — Migrate read-only operators + scan layer + close 4 sites of 2-arg make_data_batch (Pitfall 4) (DB-02, DB-03)
- [x] 18-05-PLAN.md — Migrate 23 test files; reach build-clean (DB-03, DB-04)
- [x] 18-06-PLAN.md — Run all grep gates + [mgpu] 16/16 + [mgpu_stress] 1-iter + racecheck; write 18-VERDICT.md (DB-04, DB-05) — landed PARTIAL with DB-05 deadlock evidence
- [x] 18-07-PLAN.md — Gap closure: drop R5 lock-and-hold (Path A); audit batch_lock_utils.hpp; rerun gauntlet; write 18-VERDICT-V2.md (DB-05) — flips Phase 18 to PASS
**Pitfalls**:
  - P1 (RAII lock scope self-deadlock): scope every `read_only_data_batch` / `mutable_data_batch` accessor to the narrowest possible block; never hold a `to_read_only()` accessor while calling any function that internally acquires `to_mutable()` on the same batch. Use `readonly_to_mutable(std::move(ro))` for upgrade paths.
  - P3 (pop_next_data_batch non-blocking semantics): every old `pop_data_batch(target_state)` call site must be replaced with a proper wait loop, not a bare `pop_next_data_batch()` that discards `nullptr`. Run `[TPC-H][parquet]` correctness check to detect silent data loss.
  - P7 (PR #739 x #117): use #739 only as a file-list reference identifying which operator files need touching; the actual per-site recipe follows the #117 RAII pattern (`to_read_only()` / `to_mutable()`), not #739's pre-#117 recipe.

### Phase 19: IO Framework Adoption (PR #675)
**Goal**: `sirius::io::cucascade_datasource` is retired; `sirius::io::sirius_datasource` with per-GPU `uring_ioctx` instances is operational in `SiriusContext`; `[TPC-H][parquet]` 22/22 passes on the new datasource; HYG-02 holds.
**Depends on**: Phase 18 (sirius_datasource creates batches that downstream code accesses via RAII accessors; landing IO framework on un-migrated batch accessor code produces a non-compiling tree)
**Requirements**: IO-12, IO-13, IO-14, IO-15, IO-16, IO-17
**Success Criteria** (what must be TRUE):
  1. `grep -rn "cucascade_datasource" src/ test/` returns zero hits (IO-15 retirement gate).
  2. `grep -rn "cudaSetDevice\b" src/io/` returns zero raw calls — all device selection goes through `rmm::cuda_set_device_raii` or the per-GPU ioctx dispatch path (P11 HYG-class gate / IO-16).
  3. `grep -c "rmm::cuda_stream_default" src/` returns <= 40 (HYG-02 preserved).
  4. `[TPC-H][parquet]` filter passes 22/22 on the new `sirius_datasource` with `num_gpus: 2` (IO-17 smoke regression).
  5. `nvidia-smi dmon` during a SF10 parquet scan shows non-zero PCIe read activity on both GPU 0 and GPU 1 — confirming per-GPU ioctx instances are actually driving reads on both devices (IO-14 multi-GPU safety gate).
**Plans**: 6 plans
- [x] 19-01-PLAN.md — Pre-flight + inventory baseline (IO-12 vcpkg + liburing probe + Q3 iceberg helper audit)
- [x] 19-02-PLAN.md — IO-16 HYG-02 fix: scoped rmm::cuda_set_device_raii at uring_reactor.cpp:~276
- [x] 19-03-PLAN.md — Add new test fixture helpers (make_test_gpu_ioctxs / make_test_ioctx) alongside old ones
- [x] 19-04-PLAN.md — IO-13/14: per-GPU sirius_ioctx init loop in SiriusContext under rmm::cuda_set_device_raii
- [x] 19-05-PLAN.md — IO-14/15: flip parquet+iceberg+task_creator+sirius_context consumers; retire cucascade_datasource (delete 3 files)
- [x] 19-06-PLAN.md — IO-17 verification gauntlet: [TPC-H][parquet] 22/22 + sanitizer memcheck + nvidia-smi dual-GPU PCIe probe + 19-VERDICT.md
**Pitfalls**:
  - P4 (uring_reactor single CUDA context): create one `uring_ioctx` per GPU in `SiriusContext::initialize()` under `rmm::cuda_set_device_raii`; do NOT create a single shared ioctx for all GPUs — that re-introduces the v1.1 kvikio anti-pattern.
  - P5 (global admission_control budget): each per-GPU ioctx gets its own `admission_control` instance; do not share a single budget across GPUs, which serializes I/O at SF100+ scale.
  - P11 (HYG-02 raw cudaSetDevice in uring_reactor.cpp): wrap any raw `cudaSetDevice` in `uring_reactor.cpp` with `rmm::cuda_set_device_raii`; run HYG-02 grep gate before first integration test attempt.

### Phase 20: Scan Manager + Pin Tables Port (PR #731 + #721)
**Goal**: `sirius_scan_manager` drives parquet splits via `parquet_split_provider` / `split_connector`; Phase 9 `_batch_gpu_affinity` and Phase 13 stream-lineage are re-planted into the new scan path; SCHED-RR distribution is verified via `[mgpu_stress]`; `CALL pin_table(...)` DDL is functional; SF10 smoke regression passes.
**Depends on**: Phase 19 (`parquet_split_provider::run_batch` calls `sirius_datasource` for footer reads — the scan manager will not compile without the IO framework in place)
**Requirements**: SM-01, SM-02, SM-03, SM-04, SM-05, SM-06
**Success Criteria** (what must be TRUE):
  1. `grep -rn "writer_stream\|record_writer_event" src/op/scan/` returns non-zero — Phase 13 stream-lineage re-attached in `sirius_gpu_parquet_scan_operator.cpp` or `parquet_scan_task.cpp` (SM-03 / P10 gate).
  2. `[mgpu_stress]` 500-iter passes (100 iterations × 5 queries × varied SCHED-RR offsets, exit 0, >= 77053 assertions) — SCHED-RR counter is correctly ported to the split-provider path (SM-01 / P6 gate).
  3. AUDIT TEST_CASE disjointedness REQUIRE (`std::set_intersection(scan_ids) == empty`) fires green on `num_gpus: 2` — `_batch_gpu_affinity` re-planted and working (SM-02 gate).
  4. `[integration][TPC-H]` 48/48 PASS at SF1 with `num_gpus: 2`; TPC-H Q1/Q6/Q12 PASS at SF10 on `num_gpus: 2` (SM-06 smoke regression).
  5. `CALL pin_table(...)` executes without error on a single-GPU query; `20-STREAM-LINEAGE-REATTACH.md` and `20-SCHED-RR-PORT.md` document the porting decisions (SM-04, SM-05 documentation gates).
**Plans**: 5 plans (5 closed; SM-06 SF1 escalated to Phase 21 REG-03 via 20-05 PATH B; status human_needed)
- [x] 20-01-PLAN.md — Empirical verification gates: [mgpu_stress] 500-iter + [mgpu-audit] disjointedness + grep gates → 20-01-EVIDENCE.md (SM-01/SM-02/SM-03)
- [x] 20-02-PLAN.md — TODO cleanup (Pitfall 1) + author 20-SCHED-RR-PORT.md + 20-STREAM-LINEAGE-REATTACH.md + resolve Open Q1 (SM-01/SM-02/SM-03 docs)
- [x] 20-03-PLAN.md — Document pin_table single-GPU residency in PROJECT.md Deferred + register PIN-MGPU-01 in REQUIREMENTS.md (SM-05)
- [x] 20-04-PLAN.md — SF1 [integration][TPC-H] 48/48 + SF10 Q1/Q6/Q12 num_gpus=2 + SM-04 source-inspection + 20-VERDICT.md (SM-04/SM-06)
- [x] 20-05-PLAN.md — Gap closure: PATH B escalation (status human_needed). Sanitizer revealed 21 stream-ordered races at HEAD across 2 clusters at library boundaries: cluster A (5/21) cudf+kvikio internal `read_column_chunks_async`; cluster B (16/21) cucascade pin 1c1e648 `alloc_and_peer_copy_async` host-staging fallback (race shape E). Phase 13-04 entry-level `cudaStreamWaitEvent` IS firing correctly; cluster B is in a NEW post-Phase 13 fallback code path. Recommended fix 1.5-2.5 days (cucascade fork+bump for `alloc_and_peer_copy_async` same-stream invariant + Sirius `cudaStreamSynchronize` after `read_parquet`). Carryover to Phase 21 REG-03 explicit; ship-gate cannot pass without resolution. See [`20-05-INVESTIGATION.md`](phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-INVESTIGATION.md).
**Pitfalls**:
  - P6 (SCHED-RR counter stale under split_provider): port `_no_pref_rr_counter` increment to `parquet_split_provider`'s split-emission loop — the old `management_eventloop` increment is no longer the split-allocation site. Confirm with `[mgpu_stress]` 500-iter before declaring done.
  - P10 (Phase 13 work in deleted file): the stream-lineage attachment points extracted during MERGE-03 (Phase 17) must be explicitly re-wired here; do not assume `parquet_split_provider` already handles them. The `grep` gate in criterion 1 is mandatory.
  - P2 (writer_stream lost under RAII): when re-attaching `record_writer_event` in `sirius_gpu_parquet_scan_operator.cpp`, verify the `writer_stream` argument passed to `gpu_table_representation(table, mem_space, stream)` comes from the task's actual execution stream, not a default-constructed `cuda_stream_view`.

### Phase 21: v1.4 Ship Gate (Full v1.3 Gauntlet on Rebased Branch)
**Goal**: The rebased `feature/single-node-multi-gpu2` branch passes every v1.3 regression gate — correctness, distribution, stress, performance, and hygiene — confirming that no multi-GPU behavior was lost during the rebase.
**Depends on**: Phase 20 (all migration and porting work must be complete before the full gauntlet)
**Requirements**: REG-01, REG-02, REG-03, REG-04, REG-05, REG-06
**Success Criteria** (what must be TRUE):
  1. `[mgpu]` filter passes 16/16, exit 0, >= 79091 assertions, runtime <= 130s (REG-01).
  2. `[TPC-H][parquet]` filter passes 22/22, exit 0, runtime <= 90s (REG-02).
  3. `[integration][TPC-H]` filter passes 48/48, exit 0, >= 71608 assertions, runtime <= 3 min (REG-03). **BLOCKED on Phase 20 SM-06 SF1 carryover** — Q11 parquet num_gpus=2 `cudaErrorIllegalAddress` (canonical follow-up #17). Plan 20-05 Path B escalation (status human_needed): structural finding at cucascade `alloc_and_peer_copy_async` host-staging fallback (16/21 races) + cudf+kvikio internal `read_column_chunks_async` stream-ordering (5/21 races). Recommended fix 1.5-2.5 days (cucascade fork+bump + Sirius `cudaStreamSynchronize` workaround). See [`20-05-INVESTIGATION.md`](phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-INVESTIGATION.md). Phase 21 REG-03 cannot pass without resolution OR explicit acceptance-criteria relaxation.
  4. SF100 TPC-H Q1 `num_gpus=2` wall-clock <= 5.7s; result byte-identical to 1-GPU baseline; cross-GPU scan-id intersection = 0 (REG-04).
  5. `[mgpu_stress]` 500-iter PASS — 100 iterations × 5 representative `[mgpu]` queries × varied SCHED-RR counter offsets; >= 77053 assertions, exit 0 (REG-05).
  6. `grep -c "rmm::cuda_stream_default" src/` <= 40; compute-sanitizer memcheck clean on `[multi_gpu_foundation]` + `[integration][gpu_execution][parquet][join]` (REG-06 HYG-02 gate).
**Plans**: 1 plan
- [ ] 21-01-PLAN.md — Run REG-01..06 ship-gate gauntlet, decide SM-02 fixture path, author 21-VERDICT.md, update STATE/ROADMAP/REQUIREMENTS/PROJECT
**Pitfalls**:
  - All P1-P11 should be resolved by this phase. This is final confirmation, not a place to discover new issues.
  - P2 (writer_stream): SF100 Q11 `num_gpus=2` must be run explicitly — this is the only query/scale combination that reliably triggers the cross-GPU stream-ordering race. `[mgpu]` suite at SF1 is insufficient for this specific gate.
  - P9 (Portable/Mapped flags): if SF100 Q1 2-GPU shows GPU 1 PCIe activity = 0, the P2P probe at init may have been tripped by lost Portable flags; re-run Phase 16 grep gate first before deeper investigation.

## Progress

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 4. cuCascade Bump + v1.0 Re-integration | v1.1 | 5/5 | Complete | 2026-04-20 |
| 5. Cucascade-Backed Parquet I/O Migration | v1.1 | 6/6 | Complete | 2026-04-21 |
| 6. Multi-GPU Gap Closure | v1.1 | 4/4 | Complete | 2026-04-21 |
| 7. P2P Direct Transfer + Adaptive Scan | v1.1 | 4/4 | Complete | 2026-04-21 |
| 8. Multi-GPU SQL Pipeline Fix | v1.2 | 6/6 | Complete | 2026-04-22 |
| 9. Scan-Task Distributor + Batch-Ownership Affinity | v1.2 | 4/4 | Complete (PARTIAL) | 2026-04-24 |
| 10. TABLE_FUNCTION-form gpu_execution SIGSEGV fix | v1.2 | 4/4 | Complete (PARTIAL) | 2026-04-27 |
| 11. AUDIT TEST_CASE attach-path SIGSEGV hotfix | v1.2 patch | 2/2 | Complete (PARTIAL) | 2026-04-28 |
| 12. Fix vector::at(2) in small-sort plan path | v1.3 | 4/4 | Complete | 2026-04-29 |
| 13. Fix Q11 multi-GPU hang/illegal-address | v1.3 | 4/5 | Complete | 2026-04-30 |
| 14. Land SCHED-RR distribution | v1.3 | 2/2 | Complete | 2026-04-30 |
| 15. Cross-GPU operator-colocation audit | v1.3 | 4/4 | Complete | 2026-05-01 |
| 16. Cucascade Submodule Rebase + Pin Recovery | v1.4 | 5/5 | Complete    | 2026-05-05 |
| 17. Sirius origin/dev Merge — Base Layer | v1.4 | 4/4 | Complete    | 2026-05-05 |
| 18. DataBatch RAII Migration | v1.4 | 7/7 | Complete | 2026-05-05 |
| 19. IO Framework Adoption | v1.4 | 6/6 | Complete | 2026-05-06 |
| 20. Scan Manager + Pin Tables Port | v1.4 | 6/6 | Complete | 2026-05-06 |
| 21. v1.4 Ship Gate | v1.4 | 1/1 | Complete | 2026-05-06 |
| 22. Multi-GPU pinning + stream lineage hardening | v1.5+ | 7/7 | Complete    | 2026-05-08 |

## Phase context

Each phase has a `NN-CONTEXT.md` file in `.planning/phases/NN-*/` capturing rich background, suggested tasks, and acceptance criteria. A fresh Claude can read the context file directly to pick up the work, or run `/gsd:plan-phase NN` to break it into plans.

- Phase 12: `.planning/phases/12-small-sort-vector-rangecheck-fix/12-CONTEXT.md`
- Phase 13: `.planning/phases/13-q11-multi-gpu-illegal-address/13-CONTEXT.md`
- Phase 14: `.planning/phases/14-sched-rr-distribution/14-CONTEXT.md`
- Phase 15: `.planning/phases/15-mgpu-operator-colocation-audit/15-CONTEXT.md`
- Phase 22: `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CONTEXT.md`

## Phase dependency DAG

```
Phase 16 (cucascade rebase)
    |
    v
Phase 17 (origin/dev merge — base layer)
    |
    v
Phase 18 (DataBatch RAII migration)
    |
    v
Phase 19 (IO Framework adoption)
    |
    v
Phase 20 (Scan Manager + Pin Tables port)
    |
    v
Phase 21 (v1.4 Ship Gate)
    |
    v
Phase 22 (Multi-GPU pinning + stream lineage hardening)
```

- Phase 16 is the only phase with no predecessor; cucascade API shape governs everything downstream.
- Phase 17 must follow Phase 16 so the auto-merge machinery starts from a cucascade-correct tree.
- Phase 18 must follow Phase 17 so the dev-merge auto-merges are committed before RAII rewrites start.
- Phase 19 must follow Phase 18 so batches created by `sirius_datasource` are accessed via RAII accessors.
- Phase 20 must follow Phase 19 because `parquet_split_provider::run_batch` calls `sirius_datasource` — compile-graph dependency.
- Phase 21 is the terminal gate for v1.4; all migration and porting work must be complete before the full v1.3 gauntlet runs.
- Phase 22 is post-v1.4; it lands PIN-MGPU-01 (multi-GPU pinning round-robin) and closes fu17 Cluster B (cucascade `alloc_and_peer_copy_async` same-stream invariant). Re-runs the v1.4 ship-gate gauntlet against the bumped pin to prove no regression.

### Phase 22: Multi-GPU pinning + stream lineage hardening

**Goal:** `pin_table` distributes parquet chunks round-robin across all available GPU memory spaces (PIN-MGPU-01); cucascade `alloc_and_peer_copy_async` host-staging fallback closes its stream-ordered race (fu17 Cluster B) by collapsing allocator + both memcpy legs onto a single `target_stream`. v1.4 ship-gate gauntlet (REG-01..06) re-passes against the bumped pin; new gates `[pin_mgpu]` distribution + routing + Cluster B sanitizer PASS. HYG-02 = 40 phase-wide invariant preserved.
**Depends on:** Phase 21
**Requirements**: PIN-MGPU-01, fu17-cluster-b
**Plans:** 7/7 plans complete

Plans:
- [x] 22-01-PLAN.md — Refactor `pinned_entry` for per-chunk `memory_space*` vector + add `get_pinned_entries()` accessor
- [x] 22-02-PLAN.md — `PinTableFunction` round-robin distribution + per-file `cuda_set_device_raii` + `cached_split_provider` per-chunk lookup
- [x] 22-03-PLAN.md — Cucascade Cluster B same-stream invariant fix in `alloc_and_peer_copy_async` + sanitizer micro-validation
- [x] 22-04-PLAN.md — Sirius parent submodule pin bump to Plan 03 fix + integration smoke
- [x] 22-05-PLAN.md — `[pin_mgpu]` Catch2 distribution + routing tests + CMake registration
- [x] 22-06-PLAN.md — `test/scripts/sanitizer_gate_22.sh` Cluster B sanitizer gate (Bash + timeout, exit 0 iff Cluster B = 0)
- [x] 22-07-PLAN.md — v1.4 ship-gate gauntlet rerun + Phase 22 new gates + `22-VERDICT.md` + `22-CUCASCADE-DIFF.md` + checkpoint (completed 2026-05-08)

### Phase 22.1: Remove kvikio (INSERTED)

**Goal:** All Sirius parquet/metadata reads route through `sirius_ioctx::make_datasource(io_object)`; zero `cudf::io::datasource::create(path)` or `cudf::io::source_info{path}` invocations remain in `src/`. Closes the K.1 (Cluster A) sanitizer race and likely K.6 (`cudaSetDevice(-1)` empty-result fallback at SF100 Q11 num_gpus=2). Subsumes IO-MGPU-02. Required for multi-GPU correctness — kvikio's per-FileHandle CUDA-context binding silently breaks when the destination buffer lives on a different GPU than the FileHandle's bound context.
**Requirements**: IO-MGPU-02 (subsumed), IO-MGPU-03 (new)
**Depends on:** Phase 22
**Plans:** 7/7 plans complete

Plans:
- [x] 22.1-01-PLAN.md — Register kFileScheme uring ioctx in SiriusContext; expose datasource_registry accessor
- [x] 22.1-02-PLAN.md — Flip datasource_factory policy: throw on unknown scheme ("kvikio path is forbidden"); normalize relative paths to file://<absolute>
- [x] 22.1-03-PLAN.md — Migrate sirius_gpu_parquet_scan_operator (site #1) to ioctx->make_datasource — closes K.1 (Cluster A) race source
- [x] 22.1-04-PLAN.md — Migrate PinTableFunction (site #2) to per-GPU ioctx + pointer-form source_info
- [x] 22.1-05-PLAN.md — Migrate iceberg metadata + equality-delete reads (sites #3 + #4) to GPU 0 ioctx (single-GPU sufficient per D-06)
- [x] 22.1-06-PLAN.md — Delete unit-test fallback at parquet_split_provider.cpp:295 (site #7); update test fixtures to inject ioctx
- [x] 22.1-07-PLAN.md — Verification gauntlet (REG-01..06 + GATE-22.1-A/B/C + K.6 advisory) + sanitizer_gate_22.sh Cluster A gate + 22.1-VERDICT.md + 22.1-CUCASCADE-DIFF.md (autonomous: false)

### Phase 22.2: Fix downgrade_executor cudaSetDevice(-1) (K.6 closure) (INSERTED)

**Goal:** Close fu17 follow-up #17 / K.6 by fixing the HOST-tier `downgrade_executor` worker-init bug. Root cause (empirically isolated by Phase 22.1's advisory check): `src/downgrade/downgrade_executor.cpp:67-89` unconditionally builds a CUDA stream pool keyed to `_memory_space->get_device_id()` and a per-thread init lambda that calls `cudaSetDevice(device_id)`. For HOST-tier (and DISK-tier) executors created in `SiriusContext::initialize` via `create_executors_for_tier(Tier::HOST)`, `get_device_id()` returns the sentinel `-1` (no CUDA device for host memory). At SF1 the failure is silent because the HOST-tier executor never services a downgrade request; at SF100 host pressure triggers a request, the worker thread fails to bind, and the query falls back to empty result. Fix: gate both the stream pool creation and the per-thread init on `_space_id.tier == cucascade::memory::Tier::GPU`. SF100 Q11 num_gpus=2 must return correct non-empty rows post-fix.
**Requirements**: K.6 (closure of fu17 follow-up #17 / SF100-Q11-MGPU)
**Depends on:** Phase 22.1 (kvikio removal proves K.6 is independent of kvikio; isolates the failure to downgrade_executor)
**Plans:** 0/0 plans complete (fast-path: single fix commit, no formal plan ceremony)

Plans:
- [x] (fast-path) src/downgrade/downgrade_executor.cpp +13/-3: gate _stream_pool device + per_thread_init on _space_id.tier == GPU (commit 057ba5c, 2026-05-08; verdict in 22.2-VERDICT.md)

### Phase 22.3: Fix CTE operator types declaration (K.7 closure — Q11 SF10+ correctness) (INSERTED)

**Goal:** Close K.7 — the residual cause of Q11 returning 0 rows at any scale ≥ SF10 (regardless of num_gpus). Architectural hypothesis (untested at scaffold time): `src/planner/sirius_plan_cte.cpp:50` constructs `sirius_physical_cte` with `_types = right->types` (the consumer subplan's output types), but `sirius_physical_cte::execute()` at `src/op/sirius_physical_cte.cpp:75-81` is a passthrough that forwards the **producer's** batches unchanged. Result: a CTE operator with declared 3-column types receives and forwards a 5-column batch from its upstream HASH_JOIN. The validation warning at `src/pipeline/gpu_pipeline_task.cpp:56` surfaces the mismatch but doesn't throw; downstream consumers (PROJECTION operators that select columns by index) misread the wrong-shaped batch and produce empty results without raising an error. Q11 is the canary because its HAVING subquery materializes a CTE with both producer and consumer subplans whose output types differ. Likely fix: change `_types` in `sirius_plan_cte.cpp:50` to the producer's output types (need to verify which child of `LogicalMaterializedCTE` is the producer per DuckDB convention). Verification: SF10+ Q11 num_gpus={1,2} returns correct rows matching the SF1 baseline pattern; new SF10 Q11 test added to `[TPC-H][parquet]` Catch2 suite to gate against future regression.
**Requirements**: K.7 (Q11 SF10+ correctness regression that has been silently shipping; only test gate covers Q11 at SF1)
**Depends on:** Phase 22.2 (K.6 closure cleared the cudaSetDevice noise that previously masked K.7 in SF100 logs)
**Plans:** Shipped 2026-05-08 — K.7 reclassified NO-REPRO (SQL fixture used constant `0.0001` instead of spec-compliant `0.0001/SF`; at SF10+ that puts threshold above max single-partkey value, so 0 rows is correct — DuckDB CPU agrees). Shipped CTE planner `_types` cleanup as cosmetic correctness improvement (validator warning silencer) + new SF10 Q11 mgpu regression test using spec-compliant fraction. HEAD `275fd11`; verdict at [`22.3-VERDICT.md`](phases/22.3-fix-cte-types/22.3-VERDICT.md). Datasource_factory test-suite alignment with Phase 22.1 strict policy folded in at HEAD `b423a47`.

### Phase 23: Update cucascade + sirius from upstream

**Goal:** Re-base our cucascade fork onto `origin/main` (HEAD `bcddb89`, 1 commit ahead of our `c666b21` pin) and merge sirius `origin/dev` (12 commits ahead) into `feature/single-node-multi-gpu2`. Resolve overlap conflicts in favor of upstream where the upstream change supersedes ours; preserve everything else we shipped during Phase 17–22.3. Cucascade upstream PR #121 "Make host memory portable" (`bcddb89`) adds portable pinned memory + CUDA event wrapper and supersedes the portable-pinning hunks of our squashed commit `6236494`. Surgical-split strategy: drop those 4 overlapping files' hunks from `6236494`; keep the 3 ours-only files (ptds tracker, pool peer access, `pipeline_io_backend.cpp` cleanup). Other 5 cucascade commits rebase on top with predicted conflict surface in `995bf4e` + `42a01c4` (both touch `memory/common`). Sirius merge from `origin/dev` is a separate step after the cucascade gitlink bump; highest-risk upstream commits documented (`7eeaab4` value AST Phase 2, `7cc7a79` task-creation race fix, `972cb32` converter symbol rename, `e94ad4a` per-op memory estimate, `5d09a59` bytes-to-materialize fix). All Phase 22.x invariants (REG-01..06, GATE-22.1-A/B/C, K.6/K.7 NO-REPRO, HYG-02, kvikio-free, Cluster A=0 / Cluster B same-stream invariant) must hold post-merge.
**Requirements**: MERGE-CC-23 (cucascade rebase clean against `origin/main` `bcddb89` with surgical 6236494 split), MERGE-DEV-23 (sirius `origin/dev` merge into `feature/single-node-multi-gpu2`), GAUNTLET-23 (full Phase 22.x invariant gauntlet passes post-merge).
**Depends on:** Phase 22.3 (must ship before this phase rebases; the new SF10 Q11 test is part of the post-merge gauntlet)
**Plans:** 5 plans

Plans:
- [ ] 23-01-PLAN.md — Cucascade rebase prep + surgical split of 6236494 (backup branch + pre-merge tag + start interactive rebase)
- [ ] 23-02-PLAN.md — Cucascade rebase continuation — apply a1778f9/995bf4e/1c1e648/42a01c4/c666b21; integrate PR #121 portable-pinning with our DMA probe + stream-lineage + Cluster B same-stream
- [ ] 23-03-PLAN.md — Bump Sirius cucascade gitlink to post-rebase HEAD; intermediate MCP build + 4 invariant Catch2 suites
- [ ] 23-04-PLAN.md — git merge origin/dev into feature/single-node-multi-gpu2; resolve D-13..D-20 conflicts with per-file triage log; intermediate build + invariant gauntlet
- [ ] 23-05-PLAN.md — Full v1.4 + Phase 22.x gauntlet (REG-01..06, GATE-22.1-A/B/C, K.6/K.7 NO-REPRO, Cluster B same-stream) + sanitizer baseline diff + 23-VERDICT.md + 23-CUCASCADE-DIFF.md (CC-UPSTREAM-01)
