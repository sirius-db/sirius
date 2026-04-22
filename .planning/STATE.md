---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Multi-GPU SQL Pipeline Fix
status: executing
stopped_at: Completed 08-01-PLAN.md (build-gated; N=2 runtime reproduction deferred to 08-06)
last_updated: "2026-04-22T01:21:01.187Z"
last_activity: 2026-04-22
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 6
  completed_plans: 1
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Phase 08 — multi-gpu-sql-pipeline-fix

## Current Position

Phase: 08 (multi-gpu-sql-pipeline-fix) — EXECUTING
Plan: 2 of 6
Status: 08-01 complete (FIX-01 build-gated); 08-02 (FIX-02 probe) next
Last activity: 2026-04-22 — 08-01 landed

Progress: [██░░░░░░░░] 17% (1/6 plans complete)

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Completed           |
| ----- | ---- | -------- | ----- | ----- | ------------------- |
| 08    | 01   | 6min     | 3     | 3     | 2026-04-22T01:19:13Z |

## Decisions

- **[08-01]** FIX-01: Per-GPU stream pool map in duckdb_scan_executor replaces singular GPU-0-bound pool. Dispatch lambda opens with rmm::cuda_set_device_raii pinned to target_gpu_id. Pattern 2 idiom extended from p2p converter to scan executor.
- **[08-01]** Hoisted select_target_gpu() from parquet-only block to top of manager_loop so the dispatch lambda can capture target_gpu_id. Non-parquet scan tasks (cpu_source_task, duckdb_scan_task) now also route through a well-defined target device.
- **[08-01]** Single-GPU host runtime reproduction deferred to Plan 08-06 ship gate (verification hardware has 2 × RTX 6000 Ada). Static invariants + MCP build gate verified here.

## Accumulated Context

### Key Findings from v1.1 Verification

- **Bug site:** `pipelineable_operator_data::prepare_for_processing` → `pipeline::lock_or_prepare_batch` → `cuda_memcpy.cu:42`
- **Error:** `cudaErrorInvalidValue: invalid argument`
- **Trigger:** non-trivial SQL (filter+sort, aggregation, join, TPC-H Q1/Q6/Q12) when `num_gpus >= 2`
- **Not triggered:** trivial SQL (`SELECT count(*) FROM nation`) — pipeline tasks distribute to both GPUs, returns correct result
- **Root cause signal:** cross-device stream-correctness. Same shape as the Sirius-side P2P converter override (`src/data/sirius_p2p_converter.cpp`) — pack on source-device RAII + source stream, copy on target stream.
- **Reproduction:** set `num_gpus: 2` in `test/cpp/integration/integration.yaml`, run any TPC-H parquet/join integration test → fails. Same tests pass on `num_gpus: 1`.
- **Existing observability:** `[mgpu-audit]` info-level dispatch logs in `src/pipeline/pipeline_executor.cpp:247-249` + `src/op/scan/duckdb_scan_executor.cpp:180-184` (from commit `fd24174`).
- **Evidence:** `.planning/milestones/v1.1-E2E-VERIFICATION.md` (full report: 389 lines, reproduction steps, v1.2 recommendations).

### Decisions carried from v1.1

- **Sirius-side converter override is the fix pattern** for cross-device stream-correctness bugs (Pattern 2 from Plan 07 research). Same approach applies to `lock_or_prepare_batch`.
- **Sticky `cudaGetLastError()` consume** is required after any cuda* call that can leave state in the thread-local slot.

### Roadmap Decisions (v1.2)

- **Single phase (Phase 8)** — user explicitly chose single-phase scoping during new-milestone questioning; granularity `coarse` in config.json reinforces this. All 11 requirements form one coherent delivery with internal dependencies (can't meaningfully test without fix; can't audit without both).
- **Phase numbering continues from v1.1** — Phase 8 follows v1.1's Phase 7, keeping the milestone history linear rather than resetting.
- **Integration fixture strategy (per FIX+TEST+AUDIT coupling):** scope the `num_gpus: 2` flip to TPC-H integration parameterization first, rather than flipping every fixture globally. Other fixtures can follow in later milestones if the pattern proves stable (per REQUIREMENTS.md Out of Scope note).

### Pending Todos

- Plan Phase 8 (`/gsd:plan-phase 8`) — decompose into plans covering FIX → TEST → AUDIT.
- Execute Phase 8 plans.
- Validate against all 5 success criteria on real N=2 hardware (2 × RTX 6000 Ada).

### Blockers / Concerns

- **Integration fixture scope:** TPC-H fixture currently hard-codes `num_gpus: 1` via `setenv` inside the test fixture. Flipping globally may uncover other multi-GPU bugs not exposed by the unit-test suite today. Phase 8 plans should parameterize TPC-H specifically (per TEST-01) rather than flip the default globally — the parameterization approach is what AUDIT-03 requires anyway (2-GPU variant MUST execute in default unit-tests run, but the 1-GPU variant need not be removed).

## Session Continuity

Last session: 2026-04-22T01:21:01.185Z
Stopped at: Completed 08-01-PLAN.md (build-gated; N=2 runtime reproduction deferred to 08-06)
Resume file: None
