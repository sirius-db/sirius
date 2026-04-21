---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Re-integration
status: blocked
stopped_at: Plan 05-06 Task 2b HALTED — human reviewer REJECTED sign-off checkpoint; Phase 5 cannot ship until IO-10 SF10 wall-clock + IO-11 compute-sanitizer memcheck are run on the N=2 GPU verification host used in Plan 04-05, and 05-06-MULTIGPU-VALIDATION.md is rewritten with real evidence (not deferred). Task 3 (05-SUMMARY.md) NOT RUN.
last_updated: "2026-04-21T02:55:14Z"
last_activity: 2026-04-21
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 11
  completed_plans: 10
  percent: 91
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-20)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 05 — cucascade-backed-parquet-i-o-migration

## Current Position

Phase: 05 (cucascade-backed-parquet-i-o-migration) — **BLOCKED AT PLAN 05-06 TASK 2B**
Plan: 6 of 6 (in-flight; partially executed)
Status: **BLOCKED — pending N=2 GPU validation re-run; human rejected sign-off checkpoint**
Last activity: 2026-04-21

Progress: [█████████·] 91% (phase-scoped — 10 of 11 plans complete; Plan 05-06 awaiting Task 2b re-run)

### Plan 05-06 Execution State

**Completed (committed):**
- Plan 05-06 Task 1 — 05-06-VALIDATION.md written (IO-08 grep gate PASS, HYG-02 sweep 15/15 files clean, SF1 Tier-A failure-mode match, adapter unit tests 7/7 PASS, full unit-tests 973/973 PASS). Commit: `a2c2166`
- Plan 05-06 Task 2a — 05-06-MULTIGPU-VALIDATION.md written on Tier-A (GPU-less) host with IO-10 + IO-11 evidence marked DEFERRED. Commit: `fa640f4`

**Blocked (NOT run; reject loop):**
- Plan 05-06 Task 2b — human sign-off checkpoint **REJECTED**. Reviewer requires Tier-B evidence (real compute-sanitizer log + real SF10 wall-clock numbers) before Phase 5 can ship.
- Plan 05-06 Task 3 — 05-SUMMARY.md NOT WRITTEN. Cannot be written until Task 2b resolves with an `approved` signal.

### Unblock Procedure

The following MUST be completed on the N=2 GPU verification host previously used in Plan 04-05:

1. **Re-run IO-11 compute-sanitizer memcheck** (per plan Task 2a Step 1):
   ```bash
   compute-sanitizer --tool memcheck --require-cuda-init \
     build/release/test/unittest --test-dir . test/sql/tpch-sirius.test \
     > /tmp/phase5-sanitizer.log 2>&1
   ```
   Classify each `invalid device` / `context mismatch` line as `pre-existing` (matches Phase 4 baseline shape per `04-SUMMARY.md §"Hidden-tag explicit invocation on N=2 GPU verification host"`) or `NEW` (blocker). Capture the per-backend `SiriusContext: io_backend created for GPU {device_id} (cudaGetDevice readback={n})` log lines — one per GPU, each readback must equal target.

2. **Re-run IO-10 SF10 wall-clock** (per plan Task 2a Step 2):
   - Build Phase-4 HEAD (`13e4322`) in a clean worktree, run `python3 test/tpch_performance/generate_test_data.py 10` then `python3 test/tpch_performance/performance_test.py 10`, capture Q1/Q3/Q6 wall-clock.
   - Switch to Phase-5 HEAD (current `fa640f4` or a later head if additional migration fixes land), re-run the SF10 perf test, capture wall-clock.
   - Compute aggregate `regression_pct`; apply the decision matrix (≤30% PASS / 30–50% PASS+escalate / >50% STOP).

3. **Rewrite `05-06-MULTIGPU-VALIDATION.md`** replacing every `DEFERRED` / `UNKNOWN` / `— (not run)` cell with actual measurements. Include:
   - Real sanitizer log last-100-lines excerpt
   - Per-backend `cudaGetDevice` readback rows for N=2 (device 0 + device 1)
   - Error classification table with every error line marked `pre-existing` or `NEW`
   - Real Q1/Q3/Q6 baseline + post-migration ms values + regression_pct
   - Updated recommendation section (must read `approve` / `approve with note` / `reject` based on the actual numbers)

4. **Re-invoke Plan 05-06 Task 2b** (sign-off checkpoint). If reviewer responds `approved` (possibly with a documented note about an upstream cucascade issue if SF10 regression is 30–50%), then Task 3 (05-SUMMARY.md) unblocks and Phase 5 can ship.

### Resume Pointer

- **Next action:** Plan 05-06 Task 2a re-run on N=2 verification host → update 05-06-MULTIGPU-VALIDATION.md with real evidence → re-invoke Task 2b checkpoint.
- **Do NOT** write 05-SUMMARY.md, advance phase counters, update ROADMAP plan progress, or mark IO-08..11 / HYG-02 requirements complete until Task 2b returns `approved`.
- **Preserve** Plan 05-06 work-in-progress: commits `a2c2166` (Task 1) and `fa640f4` (Task 2a Tier-A artifact) remain in history as the starting point; the re-run of Task 2a will overwrite `05-06-MULTIGPU-VALIDATION.md` with Tier-B evidence but should keep Task 1's VALIDATION.md intact (it passed autonomously and is host-independent).

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
| Phase 05-cucascade-backed-parquet-i-o-migration P01 | 5.5min | 3 tasks | 5 files |
| Phase 05 P02 | 6min | 2 tasks | 2 files |
| Phase 05 P03 | 9 min | 2 tasks | 2 files |
| Phase 05 P04 | ~9 min | 2 tasks | 3 files |
| Phase 05 P05 | 20min | 2 tasks | 7 files |

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

### Pending Todos

- `/gsd:plan-phase 4` — decompose cuCascade bump + v1.0 re-integration into plans.
- After Phase 4 lands: `/gsd:plan-phase 5` for the parquet I/O migration.
- After Phase 5: `/gsd:plan-phase 6` for topology + device safety + converter + NUMA host allocator.
- After Phase 6: `/gsd:plan-phase 7` for P2P direct + adaptive scan.

### Blockers / Concerns

- **[ACTIVE BLOCKER] Phase 5 sign-off requires N=2 GPU validation re-run.** Plan 05-06 Task 2b was REJECTED on 2026-04-21 because Task 2a was run on a Tier-A (GPU-less) planning/CI host and its 05-06-MULTIGPU-VALIDATION.md artifact documents IO-10 + IO-11 as `DEFERRED` rather than measured. Per reviewer: "Phase 5 cannot ship until compute-sanitizer memcheck (IO-11) and SF10 wall-clock measurement (IO-10) are run on the N=2 verification host used in Phase 4." The must_haves frontmatter in 05-06-PLAN.md explicitly requires Tier-B evidence (items 4 and 5). Unblock procedure documented under "Current Position → Unblock Procedure" above.
- **Dev drift:** 47 dev commits since multi-gpu branch diverged touched sirius-native types (#643), YAML config (#565), DuckDB vocabulary removal (#564/#626/#628). Phase 4 must adapt all 23 porting commits to these APIs.
- **Multi-GPU hardware gating:** Several v1.0 validation tests (and the new IO-11 / MGPU-03 / MGPU-06 / MGPU-07 criteria) require an N>1 GPU machine. Single-GPU dev boxes use the Catch2-v2 `WARN+return` convention.
- **TPC-H SF10 scan regression risk:** Cucascade's `pipeline_io_backend` always stages through pinned host (no GDS) — IO-10 budgets ≤30% regression. If exceeded, escalate upstream (cucascade issue) rather than gate the milestone.
- **Per-file `open`/`close` in `pipeline_io_backend`:** Research pitfall P1 — no file-handle cache. Profile during Phase 5; if it dominates, file upstream issue.
- **Cross-GPU converter return-leg fails on 2-GPU HW — scoped to Phase 6 (MGPU-03) / Phase 7 (MGPU-06).** Surfaced in Plan 04-05 Task 2: `[.][multi_gpu_transfer]` and `[.][mem_04_p2p_transfer]` hidden tests PASS on GPU0→GPU1 forward leg but FAIL on GPU1→GPU0 return leg via cucascade converter. Not a Phase 4 regression — exactly at the documented Phase 6/7 scope boundary. MGPU-03 (Phase 6) likely closes the device-guard root cause; MGPU-06 (Phase 7) replaces the host-staged path with `cudaMemcpyPeerAsync`. Regression gate seeded: `test/cpp/downgrade/test_downgrade_executor.cpp:813` with `TODO(MGPU-06)` marker.
- **TPC-H Q4 parquet flake (pre-existing).** Recurred once in Plans 01/02/05 (retry green). Outside Phase 4 scope. Root-cause investigation scoped to Phase 5 (parquet I/O migration touches the responsible code paths).

## Session Continuity

Last session: 2026-04-21T02:55:14Z
Stopped at: Plan 05-06 Task 2b HALTED — human reviewer rejected sign-off checkpoint. Task 2a produced a Tier-A-only artifact (05-06-MULTIGPU-VALIDATION.md) with IO-10 + IO-11 marked DEFERRED; reviewer requires Tier-B evidence from the N=2 GPU verification host before Phase 5 can ship. Task 3 (05-SUMMARY.md) NOT RUN. Plan 05-06 work-in-progress preserved: Task 1 VALIDATION.md (commit a2c2166) + Task 2a Tier-A MULTIGPU-VALIDATION.md (commit fa640f4) both intact in history.
Resume file: .planning/phases/05-cucascade-backed-parquet-i-o-migration/05-06-PLAN.md Task 2a (re-run on N=2 host) → Task 2b (re-invoke checkpoint)
