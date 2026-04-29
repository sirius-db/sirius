# Roadmap: Sirius — GPU-Native SQL Engine (Multi-GPU)

## Milestones

- ✅ **v1.1 Multi-GPU Re-integration + Cucascade I/O Migration** — Phases 4-7 (shipped 2026-04-21) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Multi-GPU SQL Pipeline Fix** — Phases 8-10 (shipped 2026-04-28) — [archive](milestones/v1.2-ROADMAP.md)
- 🛠 **v1.2 Patch — AUDIT TEST_CASE attach-path SIGSEGV** — Phase 11 (closed 2026-04-28)
- 🚧 **v1.3 Multi-GPU Distribution** — Phases 12-15 (in progress, started 2026-04-29)

## Phases

<details open>
<summary>🚧 v1.3 Multi-GPU Distribution (Phases 12-15) — in progress</summary>

Goal: deliver real multi-GPU work distribution for source-pipeline tasks (parquet metadata + GPU parquet scan + downstream operators). Today every preference-less task piles onto `_gpu_executors.begin()->first` because no SCHED policy in `task_creator.cpp` produces a `preferred_device_id` for source-pipeline operator data types. Distribution is unblocked by the parquet AST filter re-translation fix (commit `86e821a`, 2026-04-29) which lets `GPU_PARQUET_SCAN` run on a different GPU than its `metadata_scan` parent.

- [ ] Phase 12: Fix `vector::at(2)` correctness bug in small-sort plan path — independent of Phases 13/14, can run in parallel with Phase 13.
- [ ] Phase 13: Fix Q11 multi-GPU hang/illegal-address — blocks Phase 14. Cross-references project memory `project_phase08_fu17.md`.
- [ ] Phase 14: Land SCHED-RR distribution — depends on Phase 13. Patch already prepared (rolled back from session 2026-04-29 due to Phase 13 dependency).
- [ ] Phase 15: Cross-GPU operator-colocation audit — depends on Phase 14.

Pre-requisite (already shipped on `feature/single-node-multi-gpu2`):
- `86e821a fix(parquet scan): re-translate AST filter on the scan task's current device` — cudf::ast scalars are device-resident; metadata-scan-built filter would silently prune all row groups when read on a different GPU.
- `e2cf105 fix(mgpu test): assert pipeline_ids for hash_join cross-GPU audits` — switched audit assertions to the new GPU_PARQUET_SCAN pipeline architecture.
- `ce8b426 fix(mgpu): bump cucascade for peer-DMA probe; client-side sort in mgpu test wrapper` — empirical peer-DMA probe at init; consumer Intel chipset host-staging in `convert_gpu_to_gpu`.

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
| 12. Fix vector::at(2) in small-sort plan path | v1.3 | 1/4 | In Progress|  |
| 13. Fix Q11 multi-GPU hang/illegal-address | v1.3 | 0 | Not planned | — |
| 14. Land SCHED-RR distribution | v1.3 | 0 | Not planned | — |
| 15. Cross-GPU operator-colocation audit | v1.3 | 0 | Not planned | — |

## Phase context

Each phase has a `NN-CONTEXT.md` file in `.planning/phases/NN-*/` capturing rich background, suggested tasks, and acceptance criteria. A fresh Claude can read the context file directly to pick up the work, or run `/gsd:plan-phase NN` to break it into plans.

- Phase 12: `.planning/phases/12-small-sort-vector-rangecheck-fix/12-CONTEXT.md`
- Phase 13: `.planning/phases/13-q11-multi-gpu-illegal-address/13-CONTEXT.md`
- Phase 14: `.planning/phases/14-sched-rr-distribution/14-CONTEXT.md`
- Phase 15: `.planning/phases/15-mgpu-operator-colocation-audit/15-CONTEXT.md`

## Phase dependency DAG

```
Phase 12 ──┐
           ├──> Phase 15
Phase 13 ──┴──> Phase 14 ──┘
```

- Phase 12 and Phase 13 are **independent** and can run in parallel.
- Phase 14 is **blocked by Phase 13** (SCHED-RR distribution surfaces the Q11 hang).
- Phase 15 is **blocked by Phase 14** (audit only meaningful with distribution active).
