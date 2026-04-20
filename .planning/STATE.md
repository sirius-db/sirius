---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Re-integration
status: executing
stopped_at: Phase 4 COMPLETE — all 5 plans landed; PORT-01..05 + BUMP-01..03 cleared; 2 hidden-test failures deferred to Phase 6 (MGPU-03) + Phase 7 (MGPU-06) per roadmap scope. Ready for /gsd:transition to Phase 5.
last_updated: "2026-04-20T23:15:00.000Z"
last_activity: 2026-04-20
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-20)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 04 — cucascade-bump-v1-0-re-integration

## Current Position

Phase: 04 (cucascade-bump-v1-0-re-integration) — COMPLETE
Plan: 5 of 5
Status: Ready for /gsd:transition to Phase 5 (Cucascade-Backed Parquet I/O Migration)
Last activity: 2026-04-20

Progress: [██████████] 100% (phase-scoped)

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 4 | — | — | — |
| 5 | — | — | — |
| 6 | — | — | — |
| 7 | — | — | — |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

| Phase 04 P02 | 2h | 6 tasks | 13 files |
| Phase 04 P03 | 25min | 6 tasks | 8 files |
| Phase 04-cucascade-bump-v1-0-re-integration P04 | 8min | 2 tasks | 10 files |
| Phase 04-cucascade-bump-v1-0-re-integration P05 | 35min | 4 tasks | 5 files (2 summaries + STATE/ROADMAP/REQUIREMENTS) |

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

### Pending Todos

- `/gsd:plan-phase 4` — decompose cuCascade bump + v1.0 re-integration into plans.
- After Phase 4 lands: `/gsd:plan-phase 5` for the parquet I/O migration.
- After Phase 5: `/gsd:plan-phase 6` for topology + device safety + converter + NUMA host allocator.
- After Phase 6: `/gsd:plan-phase 7` for P2P direct + adaptive scan.

### Blockers / Concerns

- **Dev drift:** 47 dev commits since multi-gpu branch diverged touched sirius-native types (#643), YAML config (#565), DuckDB vocabulary removal (#564/#626/#628). Phase 4 must adapt all 23 porting commits to these APIs.
- **Multi-GPU hardware gating:** Several v1.0 validation tests (and the new IO-11 / MGPU-03 / MGPU-06 / MGPU-07 criteria) require an N>1 GPU machine. Single-GPU dev boxes use the Catch2-v2 `WARN+return` convention.
- **TPC-H SF10 scan regression risk:** Cucascade's `pipeline_io_backend` always stages through pinned host (no GDS) — IO-10 budgets ≤30% regression. If exceeded, escalate upstream (cucascade issue) rather than gate the milestone.
- **Per-file `open`/`close` in `pipeline_io_backend`:** Research pitfall P1 — no file-handle cache. Profile during Phase 5; if it dominates, file upstream issue.
- **Cross-GPU converter return-leg fails on 2-GPU HW — scoped to Phase 6 (MGPU-03) / Phase 7 (MGPU-06).** Surfaced in Plan 04-05 Task 2: `[.][multi_gpu_transfer]` and `[.][mem_04_p2p_transfer]` hidden tests PASS on GPU0→GPU1 forward leg but FAIL on GPU1→GPU0 return leg via cucascade converter. Not a Phase 4 regression — exactly at the documented Phase 6/7 scope boundary. MGPU-03 (Phase 6) likely closes the device-guard root cause; MGPU-06 (Phase 7) replaces the host-staged path with `cudaMemcpyPeerAsync`. Regression gate seeded: `test/cpp/downgrade/test_downgrade_executor.cpp:813` with `TODO(MGPU-06)` marker.
- **TPC-H Q4 parquet flake (pre-existing).** Recurred once in Plans 01/02/05 (retry green). Outside Phase 4 scope. Root-cause investigation scoped to Phase 5 (parquet I/O migration touches the responsible code paths).

## Session Continuity

Last session: 2026-04-20T23:15:00.000Z
Stopped at: Phase 4 COMPLETE — all 5 plans landed; PORT-01..05 + BUMP-01..03 cleared; 2 hidden-test failures deferred to Phase 6/7. Ready for /gsd:transition to Phase 5.
Resume file: None
