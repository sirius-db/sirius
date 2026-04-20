---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Re-integration
status: executing
stopped_at: Completed 04-03-PLAN.md (NUMA-aware downgrade re-authored); Plan 04-04 next
last_updated: "2026-04-20T22:42:24.045Z"
last_activity: 2026-04-20
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 5
  completed_plans: 3
  percent: 60
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-20)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 04 — cucascade-bump-v1-0-re-integration

## Current Position

Phase: 04 (cucascade-bump-v1-0-re-integration) — EXECUTING
Plan: 3 of 5
Status: Ready to execute
Last activity: 2026-04-20

Progress: [██████░░░░] 60%

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

## Session Continuity

Last session: 2026-04-20T22:42:02.264Z
Stopped at: Completed 04-03-PLAN.md (NUMA-aware downgrade re-authored); Plan 04-04 next
Resume file: None
