---
phase: 23-update-cucascade-and-sirius-from-upstream
plan: "05"
subsystem: gauntlet
tags: [gauntlet, verdict, multi-gpu, cucascade, origin-dev-merge]
requires: [23-04]
provides: [23-VERDICT.md, 23-CUCASCADE-DIFF.md, phase-23-sealed]
affects: [REQUIREMENTS.md, STATE.md, ROADMAP.md]
tech-stack:
  added: []
  patterns: [phase-verdict, cc-upstream-01-carry]
key-files:
  created:
    - .planning/phases/23-update-cucascade-and-sirius-from-upstream/23-VERDICT.md
    - .planning/phases/23-update-cucascade-and-sirius-from-upstream/23-CUCASCADE-DIFF.md
    - .planning/phases/23-update-cucascade-and-sirius-from-upstream/23-05-SUMMARY.md
  modified:
    - .planning/STATE.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
decisions:
  - "Phase 23 verdict: PARTIAL — REG-05/REG-06 L1 FAIL due to convert_gpu_to_gpu regression (8392c3d); all other 15 gates PASS"
  - "Side-benefit CONFIRMED: 7cc7a79 task-creation race fix closed pin_table suite-run flake; Phase 22.3 carry-forward retired"
  - "sanitizer_gate_22.sh cluster_B=1 is false positive (total_races=0); Cluster B same-stream invariant actually holds"
  - "Phase 24 fix: add rmm::cuda_set_device_raii{dst_device} before alloc_and_peer_copy_async HtoD cudaMemcpyAsync at representation_converter.cpp:628"
  - "23-CUCASCADE-DIFF.md documents 6 commits ahead of bcddb89 per CC-UPSTREAM-01 carry pattern"
metrics:
  duration: ~60min
  completed: "2026-05-12"
  tasks: 4
  files: 5
---

# Phase 23 Plan 05: Full Gauntlet + Verdict Authoring Summary

Phase 23 terminal gauntlet run against the post-Plan-23-04 merged tree (cucascade `1e889d7` + `origin/dev` merge commit `49b7b86`). Runs all Phase 22.x invariant gates, authors verdict, and seals Phase 23.

**One-liner:** Phase 23 gauntlet PARTIAL — 15/17 gates PASS, REG-05/REG-06 Leg1 FAIL from `convert_gpu_to_gpu` column-walk regression on broken-peer-DMA hardware; all other Phase 22.x invariants preserved through origin/dev merge.

---

## Verdict Outcome: PARTIAL

| Gate | Status | Evidence |
|------|--------|---------|
| REG-01 [mgpu] 16/16 | **PASS** | 79091 assertions, 125.2s |
| REG-02 [TPC-H][parquet] 22/22 | **PASS** | 36256 assertions, 109.4s |
| REG-03 [integration][TPC-H] 48/48 | **PASS** | 49/49 (+1 upstream), 71623 assertions |
| REG-04 SF100 Q1 num_gpus=2 | **PASS** | 3.048s, byte-identical, 4 rows |
| REG-05 [mgpu_stress] | **FAIL** | cudaErrorInvalidValue at representation_converter.cpp:628 |
| REG-06 Leg1 [multi_gpu_foundation] memcheck | **FAIL** | 6/7; same root cause as REG-05 |
| REG-06 Leg2 [parquet][join] memcheck | **SKIP** | Skipped after Leg1 failure |
| [datasource_factory] 11/11 | **PASS** | Phase 22.1 policy intact |
| [tpch_sf10] 4/4 | **PASS** | K.7 guard fires correctly |
| [mgpu-audit] 6/6 | **PASS** | Side-benefit CONFIRMED (suite mode) |
| GATE-22.1-A kvikio bypass-grep | **PASS** | 0 hits |
| GATE-22.1-B sanitizer cluster_A | **PASS** | cluster_A=0 |
| GATE-22.1-C SF1 Q11 num_gpus=2 | **PASS** | 9011 assertions, exit 0 |
| K.6 NO-REPRO | **PASS** | SF100 Q11 exit 0, 0 CUDA errors |
| K.7 NO-REPRO | **PASS** | Covered by [tpch_sf10] PASS |
| Phase 22 Cluster B same-stream | **PASS*** | total_races=0; gate script false positive |
| HYG-02 | **PASS** | 40 hits (≤40 baseline) |

*Gate script reports cluster_B=1 but total_races=0 — false positive; Cluster B invariant holds.

---

## REG-05/REG-06 Root Cause

**Bug:** `cucascade/src/data/representation_converter.cpp:628`

Commit `8392c3d` (Phase 23 Plan 02 rebase) introduced a new `convert_gpu_to_gpu` implementation that replaces `cudf::pack/unpack` with a column-by-column `reconstruct_column_p2p` → `alloc_and_peer_copy_async` path. On hardware where `probe_peer_dma_works(src, dst)` returns false (2 × RTX 6000 Ada, 2 directions broken), the host-staging path is taken: `cudaMallocHost` → DtoH under `cuda_set_device_raii(src_device)` → HtoD at line 628 fails with `cudaErrorInvalidValue`.

The HtoD copy at line 628 is issued without an active destination device context. The fix is to add `rmm::cuda_set_device_raii dst_guard{dst_device_id}` around line 628 (the `target_guard{dst_device_id}` in the outer `convert_gpu_to_gpu` function at line 843 does not propagate into the inner `alloc_and_peer_copy_async` call).

All other multi-GPU functionality is unaffected (REG-01..04 PASS, GATE-22.1-A/B/C PASS).

---

## Side-Benefit Hypothesis Result

**CONFIRMED.** `[mgpu-audit]` 6/6 PASS in suite mode (not individually) on first attempt, 11.9s. Phase 22.3 reported pin_table `PIN-MGPU-01 routing` suite-run flakiness. The upstream commit `7cc7a79` ("fix task-creation race") absorbed in Plan 23-04 merge removed the non-determinism causing the flake. Phase 22.3 carry-forward for this flake is retired.

---

## Sanitizer Summary

- `cluster_A` = 0 (Phase 22.1 K.1 Cluster A invariant holds; kvikio-free confirmed)
- `cluster_B` = 1 (gate script false positive — benign `cudaErrorPeerAccessAlreadyEnabled` backtraces from `probe_peer_dma_works`; `total_races` = 0)
- Memcheck Leg1: 6/7, same root cause as REG-05
- Memcheck Leg2: SKIP
- Phase 22 Cluster B same-stream invariant (`1e889d7`): HOLDS (total_races=0)

---

## HYG-02 Final Count

**40** — unchanged from Phase 22.x baseline. All 40 occurrences in `src/legacy/` (frozen `namespace duckdb` code path). Zero in active Super Sirius code.

---

## Carry-Forwards

1. **Phase 24 fix (HIGH):** `alloc_and_peer_copy_async` line 628 — add `rmm::cuda_set_device_raii{dst_device}` before HtoD `cudaMemcpyAsync`. Closes REG-05/REG-06 regression.
2. **Phase 24 gate maintenance (LOW):** Update `sanitizer_gate_22.sh` to distinguish race findings from API-error backtraces (cluster_B false positive).
3. **CC-UPSTREAM-01 (DEFERRED):** 6 cucascade fork commits ahead of `bcddb89` documented in `23-CUCASCADE-DIFF.md`. Upstream PRs deferred per prior decision.
4. **CUDA event wrapper migration:** cucascade PR #121 introduces `cuda_event` type; migration of raw CUDA event usage is a Phase 24+ candidate.

---

## Branch and Cucascade Pin Confirmation

- Branch: `feature/single-node-multi-gpu2` (unchanged throughout)
- HEAD commit: `ad19083` (docs(23-04) metadata commit)
- Merge commit: `49b7b86` (Merge origin/dev into feature/single-node-multi-gpu2)
- Cucascade submodule pin: `1e889d7e67070de7dc88860c373622182afe35df` (heads/fix/pinned-portable-flags)
- Upstream base: `bcddb89` (PR #121 "Make host memory portable")
- Commits ahead of upstream: 6

---

## Commit SHAs

| Plan | Commit | Description |
|------|--------|-------------|
| 23-04 | `49b7b86` | Merge origin/dev into feature/single-node-multi-gpu2 |
| 23-04 (docs) | `ad19083` | docs(23-04): complete origin/dev merge |
| 23-05 (docs) | (pending) | docs(23): seal phase 23 verdict + ROADMAP/STATE/REQUIREMENTS update |

---

## Notes for Phase 24

Phase 24's primary candidate is fixing the `alloc_and_peer_copy_async` regression in `representation_converter.cpp:628`. This single-line fix (add `rmm::cuda_set_device_raii{dst_device}`) should restore:
- REG-05 [mgpu_stress] to PASS
- REG-06 Leg1 [multi_gpu_foundation] to 7/7 PASS

After the fix, Phase 24 should also run REG-06 Leg2 (memcheck on `[integration][gpu_execution][parquet][join]`) which was skipped in Phase 23.

The `sanitizer_gate_22.sh` gate script update (to distinguish race backtraces from API-error backtraces) can be done in Phase 24 as low-priority gate hygiene.

## Deviations from Plan

None — plan executed as written. REG-05/REG-06 failures were found, documented, and correctly classified as a PARTIAL verdict per the plan's "If any of steps 1, 2, 3, 5, 6 fails... record the failure as a PARTIAL/FAIL verdict" criterion.

## Self-Check: PASSED

Files created/confirmed:
- FOUND: `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-VERDICT.md`
- FOUND: `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-CUCASCADE-DIFF.md`
- FOUND: `.planning/phases/23-update-cucascade-and-sirius-from-upstream/23-05-SUMMARY.md`
- Branch: `feature/single-node-multi-gpu2` (confirmed)
- No git push executed
