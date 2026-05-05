# Roadmap: Sirius — GPU-Native SQL Engine (Multi-GPU)

## Milestones

- ✅ **v1.1 Multi-GPU Re-integration + Cucascade I/O Migration** — Phases 4-7 (shipped 2026-04-21) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Multi-GPU SQL Pipeline Fix** — Phases 8-10 (shipped 2026-04-28) — [archive](milestones/v1.2-ROADMAP.md)
- 🛠 **v1.2 Patch — AUDIT TEST_CASE attach-path SIGSEGV** — Phase 11 (closed 2026-04-28)
- ✅ **v1.3 Multi-GPU Distribution** — Phases 12-15 (shipped 2026-05-01)
- 🚧 **v1.4 Rebase After DataBatch Changes** — Phases 16-21 (in progress, started 2026-05-04)

## Phases

<details open>
<summary>🚧 v1.4 Rebase After DataBatch Changes (Phases 16-21) — in progress</summary>

Goal: land cucascade `origin/main` (PR #117 DataBatch RAII refactor + #112 + #116) and Sirius `origin/dev` (#675 IO Framework, #731 Scan Manager, #721 Pin Tables, #739 cucascade-compat, #733/#734/#735) onto `feature/single-node-multi-gpu2`, preserving all v1.1+v1.2+v1.3 multi-GPU behavior. The v1.3 ship-gate (`[mgpu]` 16/16, `[TPC-H][parquet]` 22/22, `[integration][TPC-H]` 48/48, SF100 Q1 num_gpus=2 <= 5.7s, mgpu_stress 500-iter, HYG-02 <= 40) must pass bitwise on the rebased branch.

- [ ] **Phase 16: Cucascade Submodule Rebase + Pin Recovery** — Rebase 11 local Sirius-side cucascade fixes onto `73d00c4` (origin/main tip with #117 DataBatch RAII + #112 + #116). Highest conflict density; no Sirius compile gate here — cucascade-internal verification only.
- [ ] **Phase 17: Sirius origin/dev Merge — Base Layer** — Absorb `origin/dev` CI/CMake/config PRs (#739 compat, #675, #731, #721, #733, #734, #735) as a base-layer merge. Expected to produce 26+ `batch->get_data()` private-access build errors — DOCUMENTED as expected, not a phase failure. SCHED-RR block survival verified.
- [ ] **Phase 18: DataBatch RAII Migration (cucascade #117 surface)** — Migrate all `batch->get_data()` call sites and `pop_data_batch(state)` usages to RAII accessors; rewrite `batch_lock_utils.hpp`. Phase ends with a compile-clean build.
- [ ] **Phase 19: IO Framework Adoption (PR #675)** — Retire `sirius::io::cucascade_datasource`; adopt `sirius::io::sirius_datasource` with per-GPU `uring_ioctx` instances. Install `liburing-dev` before first build attempt.
- [ ] **Phase 20: Scan Manager + Pin Tables Port (PR #731 + #721)** — Integrate `parquet_split_provider` / `sirius_scan_manager` / `split_connector`; re-plant `_batch_gpu_affinity` (Phase 9) and stream-lineage hooks (Phase 13); port SCHED-RR counter to `parquet_split_provider`.
- [ ] **Phase 21: v1.4 Ship Gate (Full v1.3 Gauntlet on Rebased Branch)** — Full regression: `[mgpu]` 16/16, `[TPC-H][parquet]` 22/22, `[integration][TPC-H]` 48/48, SF100 Q1 num_gpus=2 <= 5.7s, mgpu_stress 500-iter, HYG-02 <= 40.

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
- [ ] 16-05-PLAN.md — Run cucascade ctest + 8 grep gates; advance submodule pin in parent worktree
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
**Plans**: TBD
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
**Plans**: TBD
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
**Plans**: TBD
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
**Plans**: TBD
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
  3. `[integration][TPC-H]` filter passes 48/48, exit 0, >= 71608 assertions, runtime <= 3 min (REG-03).
  4. SF100 TPC-H Q1 `num_gpus=2` wall-clock <= 5.7s; result byte-identical to 1-GPU baseline; cross-GPU scan-id intersection = 0 (REG-04).
  5. `[mgpu_stress]` 500-iter PASS — 100 iterations × 5 representative `[mgpu]` queries × varied SCHED-RR counter offsets; >= 77053 assertions, exit 0 (REG-05).
  6. `grep -c "rmm::cuda_stream_default" src/` <= 40; compute-sanitizer memcheck clean on `[multi_gpu_foundation]` + `[integration][gpu_execution][parquet][join]` (REG-06 HYG-02 gate).
**Plans**: TBD
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
| 16. Cucascade Submodule Rebase + Pin Recovery | v1.4 | 4/5 | In Progress|  |
| 17. Sirius origin/dev Merge — Base Layer | v1.4 | 0/? | Not started | - |
| 18. DataBatch RAII Migration | v1.4 | 0/? | Not started | - |
| 19. IO Framework Adoption | v1.4 | 0/? | Not started | - |
| 20. Scan Manager + Pin Tables Port | v1.4 | 0/? | Not started | - |
| 21. v1.4 Ship Gate | v1.4 | 0/? | Not started | - |

## Phase context

Each phase has a `NN-CONTEXT.md` file in `.planning/phases/NN-*/` capturing rich background, suggested tasks, and acceptance criteria. A fresh Claude can read the context file directly to pick up the work, or run `/gsd:plan-phase NN` to break it into plans.

- Phase 12: `.planning/phases/12-small-sort-vector-rangecheck-fix/12-CONTEXT.md`
- Phase 13: `.planning/phases/13-q11-multi-gpu-illegal-address/13-CONTEXT.md`
- Phase 14: `.planning/phases/14-sched-rr-distribution/14-CONTEXT.md`
- Phase 15: `.planning/phases/15-mgpu-operator-colocation-audit/15-CONTEXT.md`

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
```

- Phase 16 is the only phase with no predecessor; cucascade API shape governs everything downstream.
- Phase 17 must follow Phase 16 so the auto-merge machinery starts from a cucascade-correct tree.
- Phase 18 must follow Phase 17 so the dev-merge auto-merges are committed before RAII rewrites start.
- Phase 19 must follow Phase 18 so batches created by `sirius_datasource` are accessed via RAII accessors.
- Phase 20 must follow Phase 19 because `parquet_split_provider::run_batch` calls `sirius_datasource` — compile-graph dependency.
- Phase 21 is the terminal gate; all migration and porting work must be complete before the full v1.3 gauntlet runs.
