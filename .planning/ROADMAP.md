# Roadmap: Sirius — GPU-Native SQL Engine (Multi-GPU)

## Milestones

- ✅ **v1.1 Multi-GPU Re-integration + Cucascade I/O Migration** — Phases 4-7 (shipped 2026-04-21) — [archive](milestones/v1.1-ROADMAP.md)
- 🚧 **v1.2 Multi-GPU SQL Pipeline Fix** — Phase 8 (active, started 2026-04-21)

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

### 🚧 v1.2 — Active

- [ ] **Phase 8: Multi-GPU SQL Pipeline Fix** — Cross-device stream-correctness fix in `lock_or_prepare_batch` + TPC-H 2-GPU test parameterization + `[mgpu-audit]` acceptance gate

## Phase Details

### Phase 8: Multi-GPU SQL Pipeline Fix
**Goal**: TPC-H SQL queries execute correctly end-to-end on `num_gpus: 2` with pipeline tasks distributed across both GPUs, and the integration test suite catches multi-GPU regressions by default.
**Depends on**: v1.1 (Phases 4-7 shipped — topology, P2P, io_backend, converter registry, adaptive scan all in place)
**Requirements**: FIX-01, FIX-02, FIX-03, FIX-04, TEST-01, TEST-02, TEST-03, TEST-04, AUDIT-01, AUDIT-02, AUDIT-03
**Success Criteria** (what must be TRUE):
  1. **SF100 TPC-H Q1 on `num_gpus: 2` returns correct results** — using `/datasets/tpch_parquet_sf100/lineitem.parquet` (22.8 GB, ~600M rows). This query scans the full lineitem table and is guaranteed to produce many pipeline batches, forcing real cross-GPU distribution. Ship criterion: no `cudaErrorInvalidValue` from `cuda_memcpy.cu:42`, result matches the `num_gpus: 1` baseline. A smaller SF10 smoke-test variant (also Q1 on `/datasets/tpch_parquet_sf10/` if available, else SF1) PASSES first as a prerequisite.
  2. `mcp__project-commands__run_command unit-tests` exits 0 with TPC-H integration tests executing on `num_gpus: 2` (not silently skipped via `setenv` override) — `grep 'num_gpus.*2' test/cpp/integration/` shows the parameterization, all 22 SF1 queries run green on the 2-GPU variant, plus SF10 smoke subset (Q1, Q6, Q12) on 2-GPU. Q4 parquet flake policy: retry once per v1.1 precedent, not treated as regression.
  3. `grep -rn 'rmm::cuda_stream_default' src/` shows zero net-new matches introduced by Phase 8 (HYG discipline from v1.1 preserved).
  4. At least one Catch2 TEST_CASE runs a multi-batch TPC-H query (Q1 at SF1 or larger, lineitem-scanning) on `num_gpus: 2` with `[mgpu-audit]` logging enabled and asserts **`pipeline_task` count ≥ 5 on BOTH GPU 0 and GPU 1** AND **`scan_batch` count ≥ 5 on BOTH GPUs** — higher than the `> 0` floor so edge cases where one task happens to land on each GPU don't mask regressions. Regressions to single-GPU-only distribution break the default `unit-tests` build.
  5. A code-verifiable pattern match proves the fix shape: `grep -rnE 'cuda_set_device_raii.*source|pack.*source_stream|copy.*target_stream' src/pipeline/` or equivalent returns the Pattern 2 idiom in whatever file `lock_or_prepare_batch` lives in — mirrors the `src/data/sirius_p2p_converter.cpp` structure from Plan 07-02.
  6. **Bench evidence on N=2 hardware**: a recorded TPC-H SF100 Q1 run (`.planning/phases/08-*/*-VALIDATION.md` or similar) captures the full `[mgpu-audit]` log showing `scan_batch` distribution across both GPUs (batch count per GPU listed) and wall-clock. No specific regression threshold vs SF10 baseline required — just "it runs and completes correctly". SF300 if it completes cleanly is icing, not required.
**Plans**: 6 plans
  - [x] 08-01-PLAN.md — FIX-01 duckdb_scan_executor per-GPU stream pool (root cause fix)
  - [x] 08-02-PLAN.md — FIX-02 probe + conditional Sirius-side host→gpu converter override
  - [x] 08-03-PLAN.md — AUDIT-01/02/03 [mgpu-audit] log payload extension (task_id/batch_id)
  - [ ] 08-04-PLAN.md — TEST-01/02 integration-2gpu.yaml fixture + GENERATE(1,2) parameterization
  - [ ] 08-05-PLAN.md — TEST-03/04 + AUDIT TEST_CASE (SF1 full + SF10 Q1/Q6/Q12 + log-grep assertions)
  - [ ] 08-06-PLAN.md — FIX-03/04 HYG+build sweep + SF100 Q1 VALIDATION on N=2 hardware

## Progress

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 4. cuCascade Bump + v1.0 Re-integration | v1.1 | 5/5 | Complete | 2026-04-20 |
| 5. Cucascade-Backed Parquet I/O Migration | v1.1 | 6/6 | Complete | 2026-04-21 |
| 6. Multi-GPU Gap Closure | v1.1 | 4/4 | Complete | 2026-04-21 |
| 7. P2P Direct Transfer + Adaptive Scan | v1.1 | 4/4 | Complete | 2026-04-21 |
| 8. Multi-GPU SQL Pipeline Fix | v1.2 | 0/6 | Not started | - |
