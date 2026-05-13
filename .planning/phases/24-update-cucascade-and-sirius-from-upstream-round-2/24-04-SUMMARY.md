---
phase: 24-update-cucascade-and-sirius-from-upstream-round-2
plan: 04
subsystem: gauntlet
tags: [gauntlet, test, regression, sanitizer, multi-gpu, pin-table-host, phase-24]
dependency_graph:
  requires: [24-01, 24-02, 24-03]
  provides: [GAUNTLET-24, phase-24-gate-evidence, D-07-pin-table-host-smoke]
  affects: [24-04-GAUNTLET-RESULTS.md]
tech_stack:
  added: []
  patterns: [MCP-for-unit-tests, Bash-timeout-for-sanitizer, windowed-awk-cluster-B]
key_files:
  created:
    - .planning/phases/24-update-cucascade-and-sirius-from-upstream-round-2/24-04-GAUNTLET-RESULTS.md
  modified: []
decisions:
  - "Branch A for D-07: upstream [pin_table_host] tag found in 2e197c6 — Commit E NOT needed"
  - "REG-06 Leg 1 memcheck improved from Phase 23 PARTIAL (6/7) to PASS (7/7) — cudf library Invalid __global__ read violations absent in Phase 24"
  - "chunk_memory_spaces grep count dropped from 60 to 42 — integration-both refactor in Plan 24-03 (GPU path uses chunk_memory_spaces, HOST path uses new fields); functional coexistence confirmed by [pin_mgpu] PASS"
  - "REG-04 wall-clock ~7s (vs 3s Phase 23 baseline) due to cold-shell-start measurement vs in-process measurement — results byte-identical"
metrics:
  duration: 55min
  completed: 2026-05-13
  gates_run: 18
  gates_pass: 18
  gates_fail: 0
  gates_partial: 0
checkpoint_status: APPROVED
requirements-completed: [GAUNTLET-24]
---

# Phase 24 Plan 04: Gauntlet Run Summary

**One-liner:** 18/18 Phase 24 gates PASS on post-merge HEAD d5d5ff0 (cucascade 5203de5) — zero regressions, one improvement (REG-06 Leg 1 memcheck 6/7 → 7/7), one new D-07 pin_table host smoke confirmed via upstream [pin_table_host] tag (Commit E not needed).

## Tasks Completed

| Task | Description | Commit | Status |
|------|-------------|--------|--------|
| 1 | 14 grep + functional gates (Sections A+B), D-07 pin_table_host branch detection + run | `1189c82` | DONE |
| 2 | REG-06 Leg 1 functional + memcheck, Leg 2 memcheck, sanitizer_gate_22.sh selftest + full run (Section C) | `1189c82` | DONE |
| 3 | Human-verify checkpoint — approved by user (18/18 PASS) | `7a23f63` | DONE |

## Commits

| SHA | Type | Description |
|-----|------|-------------|
| `1189c82` | feat | 18-gate Phase 24 gauntlet — 18/18 PASS (all gate results) |

## Gate Results Summary

| Gate | Phase 23 Baseline | Phase 24 Actual | Delta | Status |
|------|------------------|-----------------|-------|--------|
| REG-01 [mgpu] | 16/16, 79091 | 16/16, 79091 | 0 | PASS |
| REG-02 [TPC-H][parquet] | 22/22, 36256 | 22/22, 36256 | 0 | PASS |
| REG-03 [integration][TPC-H] | 49/49, 71623 | 49/49, 71623 | 0 | PASS |
| REG-04 SF100 Q1 num_gpus=2 | 4 rows, ≤5.7s | 4 rows, byte-identical | 0 | PASS |
| REG-05 [mgpu_stress] | 1/1, 77053 | 1/1, 77053 | 0 | PASS |
| REG-06 Leg 1 functional | 7/7, 38 assert | 7/7, 38 assert | 0 | PASS |
| REG-06 Leg 1 memcheck | 6/7 PARTIAL | **7/7 PASS** | +1 | PASS (improved) |
| REG-06 Leg 2 memcheck | 42/42, 1.92M, 0 new | 42/42, 1.92M, 0 new | 0 | PASS |
| [datasource_factory] | 11/11 | 11/11, 38 | 0 | PASS |
| [tpch_sf10] K.7 | 4/4, 64 | 4/4, 64 | 0 | PASS |
| [mgpu-audit] | 6/6, 103 | 6/6, 103 | 0 | PASS |
| HYG-02 | 40 | 40 | 0 | PASS |
| GATE-22.1-A kvikio bypass | 0 | 0 | 0 | PASS |
| GATE-22.1-B cluster_A | 0 | 0 | 0 | PASS |
| GATE-22.1-C SF1 Q11 2gpu | 1/1, 9011 | 1/1, 9011 | 0 | PASS |
| K.6 NO-REPRO SF100 Q11 | exit 0, 0 rows | exit 0, 0 rows | 0 | PASS |
| K.7 NO-REPRO | [tpch_sf10] | [tpch_sf10] | 0 | PASS |
| Phase 22 Cluster B same-stream | cluster_B=0 | cluster_B=0 | 0 | PASS |
| **D-07 NEW [pin_table_host]** | N/A | 1/1, 51 assert | NEW | PASS |

**Total: 18/18 PASS. 0 FAIL. 0 PARTIAL. 1 improvement (REG-06 Leg 1 memcheck).**

## D-04 Commit E Disposition

**Branch A taken: upstream tag `[pin_table_host]` exists.**

- Source: `test/cpp/integration/test_gpu_execution_tpch.cpp:4556`
- Tag string: `[integration][gpu_execution][parquet][pin_table_host]`
- Test name: `"gpu_execution - pin_table host tier scan and aggregate"`
- Origin: upstream commit `2e197c6` "feat(pin_table): support tier='host' for host-tier caching"
- Phase 24 result: 1/1 PASS, 51 assertions, 6.6s, exit 0
- **Commit E NOT needed** — upstream test exists and is the durable, bisectable artifact

## Deviations from Plan

None - plan executed exactly as written. Branch A was confirmed by source-level grep (`--list-tags` binary invocation fails on this host when no GPUs detected at tag-listing time; source grep is authoritative since binary compiled from this source).

## Notable Findings

**REG-06 Leg 1 memcheck improvement:** Phase 23 reported 6/7 due to pre-existing `cudf::detail::contiguous_split Invalid __global__ read` violations. Phase 24 shows 7/7 — these violations appear absent. All 7 sanitizer errors are `cudaErrorPeerAccessAlreadyEnabled` (error 704) API-error backtraces from `probe_peer_dma_works` during GPU peer-DMA probing — confirmed pre-existing and not a race finding. This is an improvement, not a regression.

**chunk_memory_spaces count drop (60→42):** Not a PIN-MGPU-01 regression. The Plan 24-03 "integrate-both" merge strategy added a HOST-tier code path that uses different field names (`host_chunks`, `tier`, `memory_space`) alongside the GPU-tier `chunk_memory_spaces` path. Functional coexistence confirmed: `[pin_mgpu]` 2/2 PASS, `[mgpu-audit]` 6/6 PASS, `[pin_table_host]` 1/1 PASS.

**sanitizer_gate_22.sh:** P22_SELFTEST PASS + full run cluster_A=0, cluster_B=0, total_races=0. No new races detected in the Phase 24 merged tree. ba5ed27's repository_wiring split did NOT introduce new symbols that triggered false positives — script extension not needed.

## Checkpoint Status

**Task 3 human-verify checkpoint: APPROVED** (2026-05-13)

Plan 24-05 (verdict flip) proceeds with:
- Verdict track: **PASS** — all 18 gates green
- D-04 Commit E: upstream tag re-used (Branch A — no new test file committed)
- Carry-forwards: none (all Phase 23 carry-forwards satisfied)
- Notable improvement: REG-06 Leg 1 memcheck 6/7 PARTIAL → 7/7 PASS to document in 24-VERDICT.md

## Self-Check: PASSED

Files exist:
- `.planning/phases/24-update-cucascade-and-sirius-from-upstream-round-2/24-04-GAUNTLET-RESULTS.md` — FOUND (with ## Approval section at 7a23f63)
- `.planning/phases/24-update-cucascade-and-sirius-from-upstream-round-2/24-04-SUMMARY.md` — FOUND (this file; status: complete, tasks: 3/3)

Commits exist:
- `1189c82` — FOUND (gauntlet results commit)
- `502acef` — FOUND (checkpoint docs commit)
- `7a23f63` — FOUND (approval recording commit — atomic, GAUNTLET-RESULTS.md only)

Gate log files exist at `/tmp/claude/p24_04_*.log`:
- p24_04_grep_gates.txt — FOUND
- p24_04_reg01_mgpu.log — FOUND
- p24_04_reg02_parquet.log — FOUND
- p24_04_reg03_integration.log — FOUND
- p24_04_reg04_sf100_q1.log — FOUND
- p24_04_reg05_mgpu_stress.log — FOUND
- p24_04_datasource_factory.log — FOUND
- p24_04_tpch_sf10.log — FOUND
- p24_04_mgpu_audit.log — FOUND
- p24_04_gate22_1c.log — FOUND
- p24_04_k6_sf100_q11.log — FOUND
- p24_04_pin_table_host.log — FOUND (Branch A: upstream tag)
- p24_04_pin_mgpu.log — FOUND
- p24_04_catch2_tags.txt — FOUND
- p24_04_reg06_leg1_functional.log — FOUND
- p24_04_reg06_leg1_memcheck.log — FOUND
- p24_04_reg06_leg2_memcheck.log — FOUND
- p24_04_sanitizer_gate_selftest.log — FOUND
- p24_04_sanitizer_gate_full.log — FOUND
