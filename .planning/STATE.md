---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: multi-gpu-sql-pipeline-fix
status: defining-requirements
stopped_at: Milestone v1.2 initialized
last_updated: "2026-04-21T23:30:00.000Z"
last_activity: 2026-04-21
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.
**Current focus:** Milestone v1.2 — defining requirements for multi-GPU SQL pipeline fix.

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-04-21 — Milestone v1.2 started

Progress: [░░░░░░░░░░] 0%

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

### Pending Todos

- Define requirements (`/gsd:new-milestone` continuation — in progress).
- Plan Phase 8 (single phase: fix + test parameterization + acceptance gate).
- Execute Phase 8.

### Blockers / Concerns

- **Integration fixture scope:** the integration test fixture currently hard-codes `num_gpus: 1`. Flipping it globally may uncover other multi-GPU bugs not exposed by the unit-test suite. Phase planning should consider whether to flip globally OR parameterize TPC-H specifically.

## Session Continuity

Last session: 2026-04-21 — v1.1 shipped; v1.2 initialized
Stopped at: v1.2 scope confirmed; requirements phase next
Resume file: .planning/PROJECT.md
