# Roadmap: Sirius — GPU-Native SQL Engine (Multi-GPU)

## Milestones

- ✅ **v1.1 Multi-GPU Re-integration + Cucascade I/O Migration** — Phases 4-7 (shipped 2026-04-21) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Multi-GPU SQL Pipeline Fix** — Phases 8-10 (shipped 2026-04-28) — [archive](milestones/v1.2-ROADMAP.md)

## Phases

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

### 📋 Next Milestone — TBD

Run `/gsd:new-milestone` to scope v1.3.

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
