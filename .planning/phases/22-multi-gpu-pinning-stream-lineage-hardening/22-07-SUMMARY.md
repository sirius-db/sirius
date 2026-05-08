---
phase: 22-multi-gpu-pinning-stream-lineage-hardening
plan: 07
subsystem: phase-verdict + ship-gauntlet
tags: [verdict, ship-gate, fu17, cluster-b, pin-mgpu-01, cucascade, sanitizer, gauntlet, terminal]
status: PASS
requirements:
  - PIN-MGPU-01
  - fu17-cluster-b
dependency_graph:
  requires:
    - 22-01 pinned_entry::chunk_memory_spaces vector + get_pinned_entries() accessor
    - 22-02 PinTableFunction round-robin distribution + cached_split_provider per-chunk lookup
    - 22-03 cucascade Cluster B same-stream invariant fix (commit c666b21)
    - 22-04 Sirius parent gitlink bump 42a01c4 -> c666b21 + sanitizer micro-validation
    - 22-05 [pin_mgpu] distribution + routing Catch2 tests
    - 22-06 sanitizer_gate_22.sh Cluster B gate script
    - Phase 21 v1.4 ship-gate baseline (REG-01..06)
  provides:
    - 22-VERDICT.md (terminal phase artifact, 604 lines, sections A-L) — final ship verdict PASS
    - 22-CUCASCADE-DIFF.md (cucascade fork-side diff for CC-UPSTREAM-01 carry pattern)
    - 22-07-SUMMARY.md (this file — plan-level summary referencing the verdict)
    - Phase 22 closure with PIN-MGPU-01 + fu17 Cluster B both shipped
  affects:
    - v1.4 milestone status: PIN-MGPU-01 promoted Future -> Validated; fu17 Cluster B closed
    - v1.6+ milestone: CC-UPSTREAM-01 carry-forward updated (cucascade fork holds c666b21 fix; upstream PR deferred per D-08)
    - Project memory project_phase08_fu17 (#17): SF100 Q11 num_gpus=2 query-level fallback remains open as separate concern
tech-stack:
  added: []
  patterns:
    - "v1.4 ship-gate gauntlet re-run pattern (mirroring 21-VERDICT.md Sections A-F) on bumped pin"
    - "Phase 22 new gates (GATE-07/08/09) layered onto existing v1.4 baseline"
    - "Cucascade fork-side diff capture for future upstreaming (CC-UPSTREAM-01)"
    - "Advisory-only sanitizer recording for SF100 Q11 num_gpus=2 (D-13) — not gating"
key-files:
  created:
    - .planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md
    - .planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CUCASCADE-DIFF.md
    - .planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-07-SUMMARY.md
  modified: []
decisions:
  - "Final verdict PASS — all 6 v1.4 ship-gate gates (REG-01..06) re-passed against the bumped cucascade pin (c666b21) with no regression vs Phase 21 baseline; all 3 Phase 22 new gates (GATE-07/08/09) PASS; advisory SF100 Q11 num_gpus=2 sanitizer recorded; HYG-02 = 40 invariant preserved phase-wide."
  - "Cluster A (cudf+kvikio internal cross-stream race) remains advisory-only carry-forward per D-09 — out of Sirius scope without unwinding Phase 19's IO framework adoption. Documented in 22-VERDICT.md Section K.1."
  - "CC-UPSTREAM-01 carry pattern reaffirmed — local cucascade pin c666b21 only; upstream PR deferred to v1.6+ per D-08. 22-CUCASCADE-DIFF.md captures the readable diff for future review."
  - "SF100 Q11 num_gpus=2 query-level fallback (downgrade_executor cudaSetDevice(-1) error -> empty result drain) is independent from Cluster B (which is 0 even at SF100 scale) and remains a separate carry-forward to v1.6+ as follow-up #17. Cluster B closure does NOT solve the Q11 SF100 query-level path."
metrics:
  duration: ~50min
  completed: 2026-05-07T21:10Z
  tasks_completed: 3
  tasks_deferred: 0
  files_created: 3
  files_modified: 0
  parent_commits: 2
  cucascade_commits: 0
  sanitizer_runs: 2
  sanitizer_runtime_total_s: ~1820
---

# Phase 22 Plan 07: v1.4 ship-gate gauntlet rerun + Phase 22 new gates + terminal verdict Summary

The Phase 22 terminal artifact `22-VERDICT.md` is authored, the cucascade fork-side diff is captured at `22-CUCASCADE-DIFF.md` for future CC-UPSTREAM-01 review, and the Phase 22 ship verdict is **PASS**. The bumped cucascade pin `c666b21` (Plan 22-03 same-stream invariant fix in `alloc_and_peer_copy_async`) introduces zero regression vs the Phase 21 v1.4 ship baseline; PIN-MGPU-01 round-robin pin distribution + routing both PASS on the 2-GPU host; fu17 Cluster B is empirically closed under sanitizer (cluster_B=0 at both SF1 and SF100 scale). User accepted the verdict 2026-05-07.

## Status: PASS

All three tasks complete. Final ship verdict at the top of `22-VERDICT.md` reads PASS. User checkpoint signal: **"approved"** (resume-signal accepted; PASS path).

---

## Pointers to phase artifacts

- **Terminal verdict artifact:** `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md` (604 lines, sections A-L; final ship verdict at top reads PASS)
- **Cucascade fork-side diff:** `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CUCASCADE-DIFF.md` (204 lines; embedded `git diff 1c1e648..c666b21 -- src/data/representation_converter.cpp`; upstreaming notes for CC-UPSTREAM-01)
- **Cucascade fix commit:** `c666b21926dec70b26a1febd509435635bea8deb` on cucascade fork branch `fix/pinned-portable-flags`
- **Sirius parent commit chain:** `4df5c33` (22-02) -> `af4266a` (22-02 docs) -> `1211a02` (22-04 pin bump) -> `45885f0` (22-05 tests) -> `daea6db` (22-04 docs) -> `18bdbe6` (22-05 test fix) -> `c446305` (22-05 docs) -> `0b6134d` (22-06 script) -> `8f13eb7` (22-06 docs) -> `e865e8c` (22-06 SUMMARY + artifact) -> `80039f1` (22-07 verdict + cucascade diff) -> [this commit] (22-07 SUMMARY)

## Final ship verdict

**PASS — all 6 v1.4 ship-gate gates (REG-01..06) re-passed against the bumped pin; all 3 Phase 22 new gates (GATE-07/08/09) PASS; advisory SF100 Q11 num_gpus=2 sanitizer recorded; HYG-02 = 40 invariant preserved phase-wide.**

User checkpoint signal: **"approved"** (verbatim resume-signal accepted in Plan 22-07 Task 3 checkpoint:human-verify).

## Gauntlet results table (mirror of 22-VERDICT.md Section L)

| Req | Verdict | Evidence | Reference baseline |
|---|---|---|---|
| REG-01 [mgpu] 16/16 | **PASS** | Section A — 16/16, 79091 assertions, 110.6s, exit 0 | 21-VERDICT (106.3s) — within 4.3s drift |
| REG-02 [TPC-H][parquet] 22/22 | **PASS** | Section B — 22/22, 36256 assertions, 85.2s, exit 0 (no Q11 intermittency this gauntlet) | 21-VERDICT (79.3s with one-off Q11 retry) |
| REG-03 [integration][TPC-H] 48/48 | **PASS** | Section C — 48/48, 71607 assertions, 162.4s, exit 0 | 21-VERDICT (152.4s, 71607 assertions) |
| REG-04 SF100 Q1 num_gpus=2 | **PASS** | Section D — 2.807s wall-clock, byte-identical CSV vs 1-GPU baseline (4.938s), pipeline_task intersect=0 (GPU0=18, GPU1=12) | 21-VERDICT (3.150s, GPU0=18/GPU1=12) — faster |
| REG-05 [mgpu_stress] 500-iter | **PASS** | Section E — 1/1, 77053 assertions, 80.5s, exit 0 | 21-VERDICT (76.7s, 77053) — within 4s drift |
| REG-06 HYG-02 + sanitizer | **PASS** | Section F — HYG-02=40 (≤40); Leg 1 7/7 + 38 assertions + 0 memcheck violations; Leg 2 42/42 + 1922202 assertions + 0 memcheck violations | 21-VERDICT (40 / 7/7 / 42/42) — exact match |
| GATE-07 PIN-MGPU-01 distribution | **PASS** | Section G — `[pin_mgpu]` distribution TEST_CASE PASS, 46 assertions (combined with GATE-08), 7.3s | Plan 22-05 SUMMARY (2/2 PASS, 46, 6.9s) |
| GATE-08 PIN-MGPU-01 routing | **PASS** | Section H — `[pin_mgpu][mgpu-audit]` routing TEST_CASE PASS, GPU0{pipeline=6} GPU1{pipeline=4} ≥1 each | Plan 22-05 SUMMARY (same emission shape) |
| GATE-09 fu17 Cluster B sanitizer | **PASS** | Section I — `bash test/scripts/sanitizer_gate_22.sh` exit 0; cluster_B=0; cluster_A=16 advisory; total_races=6; runtime ~9s | Plan 22-06 SUMMARY self-test (cluster_B=0, cluster_A=14, total_races=5) |
| ADVISORY SF100 Q11 num_gpus=2 sanitizer | **RECORDED (NOT GATING)** | Section J — cluster_B=0, cluster_A=6, total_races=2, ERROR SUMMARY=4, Q11 query-level fallback to empty result (follow-up #17) | none — first-time SF100 Q11 num_gpus=2 sanitizer recording |

**Total wall-clock for gauntlet:** ~50 min (build cached → REG-01 110.6s → REG-02 85.2s → REG-03 162.4s → REG-05 80.5s → REG-04 ~10s → REG-06b ~30 min sanitizer leg suite → Phase 22 new gates ~12s → ADVISORY SF100 Q11 sanitizer ~6s).

## Accepted carry-forwards (from 22-VERDICT.md Section K)

5 known carry-forwards explicitly accepted as NOT gating Phase 22:

1. **K.1 Cluster A (cudf+kvikio internal cross-stream race)** — 6 race blocks at SF1 Q11; 2 at SF100 Q11. Out of Sirius scope without unwinding Phase 19's IO framework adoption. Per D-09 advisory only. Tracked for upstream cudf+kvikio fix or future local IO framework workaround.
2. **K.2 CC-UPSTREAM-01 (cucascade upstream PR)** — Plan 22-03 commit `c666b21` is local pin only; cucascade fork has NOT been pushed to NVIDIA/cuCascade per D-08. `22-CUCASCADE-DIFF.md` (this phase) captures the readable diff for v1.6+ upstreaming.
3. **K.3 HOST-tier `pin_table` path** — currently rejected; HOST-tier pinning with NUMA-local round-robin (reusing SCHED-02 `_numa_to_gpu_rr`) deferred to v1.6+ per D-06.
4. **K.4 PIN-MGPU-02 adaptive (free-memory-proportional) GPU pin distribution** — Phase 22 ships simple `idx % N` per D-01; adaptive variant deferred to v1.6+ contingent on observed skew at SF100 multi-table workloads.
5. **K.5 OOM retry budget restoration (100 → 10 in `gpu_pipeline_executor.cpp:262`)** — stretch goal not pursued in Phase 22; existing 100-iteration budget preserved (REG-05 PASS at 77053 assertions/80.5s).
6. **K.6 SF100 Q11 num_gpus=2 query-level fallback (follow-up #17 / `project_phase08_fu17`)** — independent from Cluster B (which is 0); the trigger is `cudaSetDevice(-1)` in `downgrade_executor` per-thread init. Tracked for v1.6+ separate phase.

## Cross-plan deviations (across all 7 plans)

Compiled from per-plan SUMMARYs:

| Plan | Rule | Description |
|------|------|-------------|
| 22-01 | (none — clean) | PIN-MGPU-01 prerequisite landed; build intentionally broken at hand-off sites for Plan 22-02 |
| 22-02 | Rule 3 (scope expansion) | files_modified declared 2 files but build break Plan 22-01 left also spanned `src/scan_manager/sirius_scan_manager.cpp:107,176` (create_provider_for); all 3 hand-off sites closed |
| 22-03 | (none — clean) | Cucascade local fork commit `c666b21` (D-07 same-stream invariant fix); Task 2 sanitizer micro-validation deferred to Plan 22-04 due to parallel-wave Plan 22-01 transient build state |
| 22-04 | Rule 3 (scope expansion) | Picked up Plan 22-03's deferred Task 2 sanitizer micro-validation (Cluster B = 0 PASS) |
| 22-04 | Rule 1 (doc drift) | Plan body claimed pre-bump pin = `1c1e648`; reality was `42a01c4` (cucascade pre-commit cleanup commit). Both ancestors of `c666b21` per `merge-base --is-ancestor` |
| 22-04 | Rule 3 (scope clarification) | HYG-02 invariant scope clarification (`src/`-only = 40 PASS = canonical; combined `src/`+`cucascade/` = 59 informational) |
| 22-05 | Rule 1 (emission shape) | Routing assertion uses `pipeline_ids` (load-bearing, from `task_scheduler.cpp:275`), NOT `scan_ids` — cached-parquet pin path drives `sirius_gpu_parquet_scan_operator` + `pipeline_task`, NOT `duckdb_scan_executor`'s scan_batch path. Combined `pipeline_ids+scan_ids ≥ 1` preserved for forward-compat |
| 22-05 | Rule 3 (blocking) | `scoped_mgpu_env` held in `std::unique_ptr` + `spdlog::default_logger()->flush()` before `env.reset()` — `Config::LOG_FLUSH_SECONDS=3s` but SF1 query is ~600ms; without explicit flush the `[mgpu-audit]` emissions stay in spdlog's 8192-byte file_sink buffer |
| 22-06 | (none — clean) | Sanitizer gate script authored at `test/scripts/sanitizer_gate_22.sh`; live self-test PASS; negative-test confirms reactive |
| 22-07 | (none — clean) | Verdict + cucascade-diff authored; checkpoint reached → user "approved" → SUMMARY + state advancement |

No Rule 4 (architectural) deviations across the entire phase.

## Links to all artifacts

### Per-plan SUMMARYs (chronological order)

- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-01-SUMMARY.md` — pinned_entry::chunk_memory_spaces refactor
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-02-SUMMARY.md` — PinTableFunction round-robin + cached_split_provider per-chunk lookup
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-03-SUMMARY.md` — cucascade Cluster B same-stream invariant fix (c666b21)
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-04-SUMMARY.md` — Sirius parent pin bump 42a01c4 → c666b21 + sanitizer micro-validation
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-05-SUMMARY.md` — [pin_mgpu] distribution + routing Catch2 tests
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-06-SUMMARY.md` — sanitizer_gate_22.sh Cluster B gate script
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-07-SUMMARY.md` — this file

### Phase-level artifacts

- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md` — terminal phase artifact (sections A-L, final verdict PASS)
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CUCASCADE-DIFF.md` — cucascade fork-side diff for CC-UPSTREAM-01

### Phase-level evidence (sanitizer logs + gauntlet output)

- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/artifacts/22-04/sanitizer_microvalidation.log` — Plan 22-04 SF1 Q11 sanitizer micro-validation log (Cluster B = 0 baseline)
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/artifacts/22-04/sanitizer_stdout.log` — Plan 22-04 stdout
- `.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/artifacts/22-06/sanitizer_gate_selftest.log` — Plan 22-06 sanitizer gate live self-test (cluster_B=0 reactive baseline)
- `/tmp/claude-1002/p22_07/sanitizer_mgf.log` — REG-06b Leg 1 [multi_gpu_foundation] sanitizer
- `/tmp/claude-1002/p22_07/sanitizer_join.log` — REG-06b Leg 2 [integration][gpu_execution][parquet][join] sanitizer
- `/tmp/claude-1002/p22_07/sanitizer_gate_22_q11.log` — GATE-09 sanitizer_gate_22.sh live run
- `/tmp/claude-1002/p22_07/sf100_q11_sanitizer.log` — ADVISORY SF100 Q11 num_gpus=2 sanitizer
- `/tmp/claude-1002/p22_07/p22_sf100_*.csv` — SF100 Q1 byte-identical proofs

### Source code surface (for cross-reference)

- Cucascade fix file: `cucascade/src/data/representation_converter.cpp` (`alloc_and_peer_copy_async` lines 611-633)
- PIN-MGPU-01 source: `src/sirius_extension.cpp` (PinTableFunction round-robin), `src/include/scan_manager/sirius_scan_manager.hpp` (pinned_entry + get_pinned_entries), `src/scan_manager/sirius_scan_manager.cpp` (create_provider_for)
- PIN-MGPU-01 cached split: `src/scan_manager/cached_split_provider.cpp` (per-chunk memory_space lookup)
- PIN-MGPU-01 tests: `test/cpp/scan_manager/test_pin_table_multi_gpu.cpp`
- Sanitizer gate script: `test/scripts/sanitizer_gate_22.sh`

## Recommended next step

After this plan completes, the orchestrator (`/gsd:complete-phase 22` follow-up) should:

1. **Promote PIN-MGPU-01:** Move PIN-MGPU-01 from Future Requirements → Validated in REQUIREMENTS.md + PROJECT.md `Validated` section. Add note "Validated in Phase 22: Multi-GPU pinning + stream lineage hardening".
2. **Close fu17 Cluster B:** Mark Cluster B closed in PROJECT.md / project memory; preserve Cluster A as carry-forward to v1.6+ per D-09. Update `project_tpch_q1_mgpu_string_bug` memory file noting the cucascade host-staging fallback now has the same-stream invariant fix at `c666b21`.
3. **Carry to v1.6+:** Open carry-forwards (CC-UPSTREAM-01, HOST-tier pin path, PIN-MGPU-02 adaptive distribution, OOM retry budget, follow-up #17 SF100 Q11 query-level fallback). These should be tracked in PROJECT.md Deferred section.
4. **Branch hygiene:** `feature/single-node-multi-gpu2` is local-only. No `git push`; no merge to `dev`. Phase 22 commit chain stays on the worktree.
5. **HYG-02 invariant:** preserved at 40 phase-wide; future phases must continue the discipline (no `rmm::cuda_stream_default` introductions in `src/`).

## Cucascade pin advance summary

| Stage | Cucascade pin | Sirius parent commit |
|---|---|---|
| Phase 21 ship | `1c1e648` | (Phase 21 HEAD) |
| Pre-Plan-22-03 (intermediate cleanup) | `42a01c4` | (no parent commit; cucascade work-tree advanced via Plan 22-03 prep) |
| Plan 22-03 + Plan 22-04 bump | `c666b21926dec70b26a1febd509435635bea8deb` | `1211a02` (Plan 22-04 pin bump commit) |

**Net cucascade work shipped this milestone:** 1 logic-bearing commit (`c666b21` parent `42a01c4`) modifying `cucascade/src/data/representation_converter.cpp` lines 611-633 (`alloc_and_peer_copy_async` host-staging fallback same-stream invariant). Hunks 2-8 are clang-format adjustments from intermediate `42a01c4`; load-bearing change is hunk 1 only. See `22-CUCASCADE-DIFF.md`.

## HYG-02 invariant final state

`grep -rn "rmm::cuda_stream_default" src/ | wc -l` → **40** (entirely in `src/legacy/` + `src/include/legacy/` — frozen `namespace duckdb` path). Zero `rmm::cuda_stream_default` introduced by Phase 22 (Plans 22-01..07). Baseline preserved across Phases 8-21 + Phase 22.

## Self-Check: PASSED

- File `/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md` exists (FOUND, 35960 bytes).
- File `/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CUCASCADE-DIFF.md` exists (FOUND, 17593 bytes).
- File `/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-07-SUMMARY.md` exists (this file; FOUND post-Write).
- Sirius parent commit `80039f1` exists (FOUND via `git log --oneline -1` for verdict + cucascade-diff atomic commit).
- HYG-02 invariant verified at 40 (FOUND via `grep -rn "rmm::cuda_stream_default" src/ | wc -l`).
- Cucascade gitlink at `c666b21926dec70b26a1febd509435635bea8deb` (FOUND via `git ls-tree HEAD cucascade`).
- Final ship verdict at top of 22-VERDICT.md reads PASS (FOUND, line 20).
- All 11 sections (A-L) present in 22-VERDICT.md (FOUND via grep).
- 22-CUCASCADE-DIFF.md has embedded diff with `alloc_and_peer_copy_async` reference (FOUND via grep).
- Branch `feature/single-node-multi-gpu2` (FOUND via `git rev-parse --abbrev-ref HEAD`).

---

*Phase: 22-multi-gpu-pinning-stream-lineage-hardening*
*Plan: 07 (terminal verdict)*
*Authored: 2026-05-07*
*Final verdict: PASS — Phase 22 ships clean*
