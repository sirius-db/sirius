---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Re-integration
status: executing
stopped_at: Completed 06-02-PLAN.md (MGPU-03 device-guard enforcement in 2 noexcept callbacks); Wave 1 parallel with 06-01 + 06-03
last_updated: "2026-04-21T14:12:12.714Z"
last_activity: 2026-04-21
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 15
  completed_plans: 13
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-20)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 06 — multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter

## Current Position

Phase: 06 (multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter) — EXECUTING
Plan: 2 of 4
Status: Ready to execute
Last activity: 2026-04-21

Progress: [██████████] 100% (Phase 4 + Phase 5 complete; 11 of 11 scoped plans done across the two shipped phases)

### Phase 5 Shipped State

All 13 Phase 5 requirements closed:

- **IO-01..03** (cucascade_datasource adapter + pinned-host staging + async) — Plans 05-01 + 05-02
- **IO-04, IO-11** (per-GPU backend cache + multi-GPU validation) — Plans 05-03 + 05-06
- **IO-05** (3 datasource::create call sites migrated) — Plans 05-04 + 05-05
- **IO-06** (iceberg delete-file reads via source_info{&ds}) — Plan 05-05
- **IO-07** (prefetched_data_source fallback is cucascade-backed) — Plan 05-04 transitive
- **IO-08** (global grep gate) — Plan 05-06 Task 1: 0 hits project-wide
- **IO-09** (SF1 correctness preserved) — Plan 05-06: Tier-A failure-mode match + full unit-tests 973/973 PASS on real N=2 hardware
- **IO-10** (SF10 wall-clock) — Plan 05-06: absolute Phase-5 numbers captured on real N=2 hardware; Phase-4 regression comparison deferred per user directive 2026-04-21
- **HYG-01** (parquet_scan_task:468 explicit stream) — Plan 05-04 Task 1
- **HYG-02** (sweep across Phase-5 modified files) — Plan 05-06 Task 1: 15/15 files clean

Phase SUMMARY at `.planning/phases/05-cucascade-backed-parquet-i-o-migration/05-SUMMARY.md` (written 2026-04-21 after Task 2b `approved` checkpoint).

### Resume Pointer

- **Next action:** `/gsd:plan-phase 6` to decompose Multi-GPU Gap Closure (MGPU-01..05) into plans.
- Phase 6 foundation is already in place from Phases 4 + 5: per-GPU executor + memory-space plumbing (Plan 04-02) + per-GPU idisk_io_backend cache on SiriusContext (Plan 05-03) + IO-11 cudaGetDevice audit pattern + Phase 4 hidden-test regression anchors (`test_downgrade_executor.cpp` TODO markers).

## Performance Metrics

**Velocity:**

- Total plans completed (v1.1): 11 (5 in Phase 4 + 6 in Phase 5)
- Average duration: Phase 4 ~66 min/plan (5h30min total), Phase 5 ~11 min/plan (65min total)
- Total execution time: ~6h35min across both phases

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 4 | 5 | 5h30min | 66 min |
| 5 | 6 | 65min | 11 min |
| 6 | TBD | — | — |
| 7 | TBD | — | — |

**Recent Trend:**

- Last 5 plans: 05-02 (6min), 05-03 (9min), 05-04 (9min), 05-05 (20min), 05-06 (spread across two host visits, ~35min aggregate)
- Trend: Phase 5 plans averaged ~2× faster than Phase 4 plans — smaller per-plan scope + Wave 2/3 parallelism paid off

| Phase 04 P02 | 2h | 6 tasks | 13 files |
| Phase 04 P03 | 25min | 6 tasks | 8 files |
| Phase 04-cucascade-bump-v1-0-re-integration P04 | 8min | 2 tasks | 10 files |
| Phase 04-cucascade-bump-v1-0-re-integration P05 | 35min | 4 tasks | 5 files (2 summaries + STATE/ROADMAP/REQUIREMENTS) |
| Phase 05-cucascade-backed-parquet-i-o-migration P01 | 5.5min | 3 tasks | 5 files |
| Phase 05 P02 | 6min | 2 tasks | 2 files |
| Phase 05 P03 | 9 min | 2 tasks | 2 files |
| Phase 05 P04 | ~9 min | 2 tasks | 3 files |
| Phase 05 P05 | 20min | 2 tasks | 7 files |
| Phase 05 P06 | ~35min (spread; Task 1 + 2a-first + 2a-re-run + 2b + Task 3) | 3 tasks | 4 files (VALIDATION + MULTIGPU-VALIDATION + SUMMARY + state) |
| Phase 06 P02 | 2m 34s | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table. Carried from v1.0 (unmerged) as validated patterns:

- **[Phase 02-01 v1.0]** Push-model task dispatch by `preferred_device_id` (pop task first, route by preference). NOT pull-model.
- **[Phase 02-01 v1.0]** `preferred_device_id` lives on both `local_state` (per-task) and `global_state` (pipeline default); local_state wins.
- **[Phase 02-01 v1.0]** NUMA→GPU mapping uses first GPU found on each NUMA node as representative.
- **[Phase 02-01 v1.0]** At-capacity task waits on preferred GPU rather than falling back to another GPU.
- **[Phase 03-01 v1.0]** Verify reservation ordering via cucascade `strategy.get_candidates()` rather than memory-exhaustion tests.
- **[Phase 01-03 v1.0]** Catch2 v2 disabled tests use `WARN+return`, not `SKIP` (compatibility).

New for v1.1 (from research synthesis):

- **[Roadmap v1.1]** cuDF 26.04 has no env-var or public-API escape from kvikio — only viable approach is a Sirius-owned `cudf::io::datasource` subclass. See `.planning/research/CUDF-DATASOURCE.md` §"Migration options".
- **[Roadmap v1.1]** `cucascade_datasource` will report `supports_device_read() == false` so cuDF host-stages via `host_read` + pinned memory; `cuda_memcpy_async` on the caller's explicit stream remains truly async. See `.planning/research/CUCASCADE-IO.md` §"Recommended architecture".
- **[Roadmap v1.1]** Per-GPU `idisk_io_backend` instances are required (one per device, constructed under `rmm::cuda_set_device_raii`) because `pipeline_io_backend` pins `cudaStream_t` + pinned buffers to the CUDA context current at construction. See `.planning/research/CUCASCADE-IO.md` §"Per-GPU backend ownership".
- **[Roadmap v1.1]** Cucascade has only one built-in backend (`"pipeline"`); S3/HTTP/remote URIs are out of scope (covered in PROJECT.md Out of Scope).
- **[Roadmap v1.1]** Bump + port combined in Phase 4 because the port cannot compile without PR #96 headers from the bumped submodule — splitting would create an unproductive intermediate state.
- **[Roadmap v1.1]** HYG-01/02 (cuda_stream_default removal) fold into Phase 5 because the adjacent line (`parquet_scan_task.cpp:468`) is touched by the I/O migration anyway.
- [Phase 04]: Wait-on-preferred-device invariant confirmed in Task 3b: management_eventloop routes to preferred GPU executor whose manager_loop handles capacity via bounded_pool->reserve(); NO fallback re-dispatch to different GPU.
- [Phase 04]: Test adaptation over re-authoring: test_gpu_pipeline_executor.cpp and test_oom_reschedule.cpp adapted to push-model by scheduling tasks directly on executor (removing request_channel.get() wait loops).
- [Phase 04]: Plan 04-03: NUMA-aware downgrade re-authored onto dev PR #579 shape (not cherry-picked); POD-extension Strategy A chosen over executor-internal Strategy B (preferred_numa_node on downgrade_task POD preserves v1.0 per-task override semantics)
- [Phase 04-cucascade-bump-v1-0-re-integration]: Plan 04-04: PORT-03 confirmed as no-op (grep -rn 'libconfig' src/ test/ = 0 hits); Task 1.5 conditional remediation skipped. All v1.0 multi-GPU settings reachable through dev's YAML config reader (PR #565). Pre-commit fixups committed as f5afde1 (pure formatting across 10 files). Build verification blocked by executor-subagent sandbox — deferred to orchestrator/04-05.
- [Phase 04-cucascade-bump-v1-0-re-integration]: Plan 04-05: Full unit-tests PASS (966 test cases, ~78.8M assertions); all 4 PORT-05 visible tags explicitly invoked and verified to actually run (no silent filtering); 3 of 5 hidden multi-GPU tags PASS on N=2 verification host; 2 hidden tags fail on GPU1->GPU0 converter return leg — deferred to Phase 6 (MGPU-03 device guards) + Phase 7 (MGPU-06 P2P direct). Task 3 checkpoint auto-approved by orchestrator in autonomous full-run mode with "approved — ship with deferral note". All structural grep gates PASS (PORT-02 0 hits LogicalType::*, PORT-03 0 hits libconfig, dead-v1.0-shape 0 hits in live code, BUMP-01 pin f47de0b exact match, PORT-01 26 commits dev..HEAD, PORT-04 7/7 symbol greps hit). Phase 4 SHIPPED.
- [Phase 05-cucascade-backed-parquet-i-o-migration]: Plan 05-01: supports_device_read() locked to false in cucascade_datasource header (IO-02 multi-GPU safety); copy/move deleted so shared_ptr<idisk_io_backend> cannot cross CUDA contexts; stub .cpp includes header to verify standalone-compileability; Tier-A baseline (this host) + Tier-B baseline (2+ GPU validation host from plan 04-05) together form the authoritative pre-migration correctness snapshot for IO-09
- [Phase 05-cucascade-backed-parquet-i-o-migration]: Plan 05-02: cucascade_datasource uses cudaMallocHost + RAII pinned_host_buffer instead of fixed_size_host_memory_resource — adapter stays context-independent (no SiriusContext coupling), preserving unit testability. std::launch::async (not deferred) for host_read_async per CONTEXT lock.
- [Phase 05]: Plan 05-03: SiriusContext now owns cucascade io_backend_registry + per-GPU idisk_io_backend cache; both accessors (get_io_backend_for point-lookup + get_gpu_io_backends map-view) declared here so Plans 04+05 are pure consumers (sirius_context.hpp sealed for Phase 5)
- [Phase 05]: Plan 05-03: Per-GPU backend construction under rmm::cuda_set_device_raii with IO-11 audit log (device_id + cudaGetDevice readback); teardown clears gpu_io_backends_ + io_backend_registry_ BEFORE memory_manager_->shutdown() to avoid cudaErrorInvalidResourceHandle at extension unload
- [Phase 05]: Plan 05-04: Approach C plumbing — task_creator seeds parquet_scan_task_global_state with SiriusContext::get_gpu_io_backends() map. Pure-consumer invariant on sirius_context.hpp upheld (Plan 03 sole owner).
- [Phase 05]: Plan 05-04: parquet_scan_task inherits from sirius_pipeline_itask (not gpu_pipeline_task) so there is no get_preferred_device_id() helper on the task. Hot-path backend selection uses g_state.get_preferred_device_id() with first-backend fallback — mirrors pipeline_executor's default routing for non-gpu_pipeline_task instances.
- [Phase 05]: Plan 05-05: Approach A (locked) for iceberg delete-file helpers — helper signatures gain std::shared_ptr<cucascade::idisk_io_backend> backend parameter; callers resolve via inherited get_gpu_io_backends(). Completes Plan 05-04's declared iceberg handoff (iceberg_scan_task_global_state ctor forwards gpu_io_backends to base + task_creator iceberg branch seeds map). Pure-consumer invariant on sirius_context.hpp upheld.
- [Phase 05]: Plan 05-06: All 13 Phase-5 requirements closed on real N=2 hardware (2 × RTX 6000 Ada, driver 595.58.03, CUDA 13.2). compute-sanitizer memcheck 0 errors across 57 test cases / 1.92M assertions; per-backend cudaGetDevice readback matches target (GPU 0→0, GPU 1→1); SF10 wall-clock captured on both 1-GPU and 2-GPU configs with correct results. IO-10 Phase-4 regression comparison explicitly deferred to future optimization work per user directive 2026-04-21 ("we don't need to run any comparisons, let's just make sure everything is working, we can optimize later"). Phase 5 SHIPPED.
- [Phase 06-02]: MGPU-03 device-guard teeth: both Super Sirius noexcept per-thread init callbacks (gpu_pipeline_executor + downgrade_executor) now check cudaSetDevice return and log spdlog::error on failure. No RAII conversion — per-thread pinning is lifetime-scoped (documented rationale in MGPU-03 comment blocks).

### Pending Todos

- `/gsd:plan-phase 6` — decompose Multi-GPU Gap Closure (MGPU-01..05: topology discovery, single-GPU no-regression, device-guard enforcement, GPU↔GPU converter registration, per-NUMA host memory spaces).
- After Phase 6: `/gsd:plan-phase 7` for MGPU-06 P2P direct + MGPU-07 adaptive scan.

### Blockers / Concerns

- **Phase 5 sign-off is CLEAR** (prior blocker resolved on 2026-04-21: Task 2b `approved` after N=2 real-hardware re-run — see Phase 5 SUMMARY §"Phase 5 Outcome").
- **Dev drift:** 47 dev commits since multi-gpu branch diverged touched sirius-native types (#643), YAML config (#565), DuckDB vocabulary removal (#564/#626/#628). Addressed in Phase 4; no further drift compensation needed for v1.1.
- **Multi-GPU hardware gating:** MGPU-03 + MGPU-06 + MGPU-07 (Phase 6/7) require an N>1 GPU machine. The N=2 verification host (`6f7e4c9-lcedt`, 2 × RTX 6000 Ada) used in Plans 04-05 + 05-06 remains available for Phase 6 validation.
- **TPC-H SF10 Phase-4 regression comparison deferred to future optimization work** per user directive on 2026-04-21. Absolute Phase-5 SF10 numbers are recorded in `05-06-MULTIGPU-VALIDATION.md` as the starting reference point.
- **Per-file `open`/`close` in `pipeline_io_backend`:** Research pitfall P1 — no file-handle cache. Not measured in Phase 5 (deferred with the regression comparison). If it dominates later profiles, file upstream issue.
- **Cross-GPU converter return-leg fails on 2-GPU HW — scoped to Phase 6 (MGPU-03) / Phase 7 (MGPU-06).** Surfaced in Plan 04-05 Task 2: `[.][multi_gpu_transfer]` and `[.][mem_04_p2p_transfer]` hidden tests PASS on GPU0→GPU1 forward leg but FAIL on GPU1→GPU0 return leg via cucascade converter. Phase 5 confirmed these are pre-existing (not Phase-5 regressions) — compute-sanitizer on N=2 reported 0 errors across 57 test cases / 1.92M assertions, so Phase 5 code is clean of its own multi-GPU bugs. MGPU-03 (Phase 6) likely closes the device-guard root cause; MGPU-06 (Phase 7) replaces the host-staged path with `cudaMemcpyPeerAsync`. Regression gate seeded: `test/cpp/downgrade/test_downgrade_executor.cpp:813` with `TODO(MGPU-06)` marker.
- **TPC-H Q4 parquet flake (pre-existing).** Not observed during Phase 5 runs; remains a pre-existing deferral for future observation.

## Session Continuity

Last session: 2026-04-21T14:12:12.712Z
Stopped at: Completed 06-02-PLAN.md (MGPU-03 device-guard enforcement in 2 noexcept callbacks); Wave 1 parallel with 06-01 + 06-03
Resume file: None
