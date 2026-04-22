---
status: in-progress
phase: 08-multi-gpu-sql-pipeline-fix
purpose: Resume the Phase 8 bug hunt after /clear — captures everything learned + exact next steps
created: 2026-04-22
---

# Phase 8 Bug Hunt — Resume Plan

## The bug we're chasing

**Test:** `gpu_execution hive partition - filter on data column`
**File:** `test/cpp/integration/test_gpu_execution_multi_format.cpp:815`
**Trigger:** `integration.yaml` set to `num_gpus: 2` (normally `num_gpus: 1`)
**Error:**
```
CUDA error encountered at: .../cpp/src/utilities/cuda_memcpy.cu:42:
  1 cudaErrorInvalidValue invalid argument
```
cudf's `cuda_memcpy.cu:42` is inside `copy_pinned()` — pinned host → device memory copy via `cudaMemcpyBatchAsync` (CUDA 13+ batch API introduced by cudf PR [#20800](https://github.com/rapidsai/cudf/pull/20800) merged 2026-03-09).

Also fails the same way: `gpu_execution - TPC-H Query 1 parquet` (test 3368).

## Hypotheses RULED OUT (with evidence)

| # | Hypothesis | Disproved by |
|---|-----------|--------------|
| 1 | cudf host buffer lifetime race (PR #20800 use-after-free pattern, issues #21680 / #21920) | `compute-sanitizer memcheck` on only the failing test: **0 memory violations** — a use-after-free or freed-buffer read would be flagged |
| 2 | Target stream from wrong GPU | Replaced `target_memory_space->acquire_stream()` with a FRESH `rmm::cuda_stream{}` created under `rmm::cuda_set_device_raii target_device_raii(target_device_id)`. Still failed identically. |
| 3 | Upstream wrong device context at our converter entry | `[mgpu-probe] host_parquet_to_gpu entry current_device=1 target_device_id=1` — all aligned at converter entry on successful calls |
| 4 | cudf's `host_worker_pool` doesn't propagate device context | `host_worker_pool::submit_task()` calls `cudaGetDevice` at submit + `cudaSetDevice(device_id)` in the worker lambda before executing (verified in cudf source at `cpp/include/cudf/detail/utilities/host_worker_pool.hpp:95-101`) |
| 5 | `mr_ref` captured on wrong device | `mr_ref = target_memory_space->get_default_allocator()` — device-specific regardless of caller's current device |
| 6 | Our converter is the failure site | **Our `[mgpu-probe]` in the converter never fires** on the failing run. Neither does `prefetched_data_source::host_read` / `device_read`. The failure is upstream of these. |

## What the data actually shows

Run log excerpt (`build/release/extension/sirius/test/cpp/log/sirius_2026-04-22.log`) from `num_gpus: 2` hive-partition test:

```
SiriusContext: topology summary — 2 GPU(s), 1 NUMA node(s)
SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
SiriusContext: io_backend created for GPU 1 (cudaGetDevice readback=1)
SiriusContext: P2P enabled 0 -> 1 (MGPU-06)
SiriusContext: P2P enabled 1 -> 0 (MGPU-06)
SiriusContext: cuDF pinned memory resource configured (max slab 8192 B)
...
QueryBegin: CALL gpu_execution("SELECT * FROM read_parquet(...) WHERE id >= 2 ORDER BY id")
[mgpu-audit] scan_batch assigned to GPU 0 batch_id=0 (available: 25433701888 bytes)
[mgpu-audit] scan_batch assigned to GPU 0 batch_id=1 ...
<crash — cudaErrorInvalidValue>
```

**Key observations:**
- Both scan batches went to **GPU 0 only** — not round-robin to GPU 1 for this test. (May be because `num_partitions = 2` and the hive scan splits by partition into device-0-only tasks? Unclear.)
- Our converter never entered — probe would have logged `host_parquet_to_gpu entry` and didn't.
- The error fires **between** scan-batch assignment and converter entry, i.e., during `compute_task` OR earlier.
- `compute-sanitizer` says no memory violation — so it's an API-level arg rejection, not a bad pointer or stream.

## What's different on num_gpus=2 vs num_gpus=1

Based on `src/sirius_context.cpp:256-315`:

- **Peer access enabled** between devices 0↔1 (`cudaDeviceEnablePeerAccess`)
- **Two `io_backend` instances** created (one per GPU) via `rmm::cuda_set_device_raii`
- **Two GPU memory spaces** with separate stream pools
- **Same cudf pinned MR** set globally (`cudf::set_pinned_memory_resource(*small_pinned_allocator_)`) — one instance used by both devices

Anything else that would differ at runtime is in code we haven't instrumented yet (the code path BEFORE the converter runs).

## Resume plan — where to probe next

The failure is in the "pre-converter" layers. Add `[mgpu-probe]` `cudaGetDevice()` breadcrumbs at these specific sites, rebuild, re-run, and inspect `build/release/extension/sirius/test/cpp/log/` after the fail.

### Probe site 1 — `parquet_scan_task::compute_task`
**File:** `src/op/scan/parquet_scan_task.cpp`
Already has one breadcrumb at line 745 (`[mgpu-probe] parquet_scan_task::compute_task entry ...`). Add:
- Breadcrumb at function EXIT (before return) showing `cudaGetDevice()` and whether `cudaGetLastError()` returns success
- Breadcrumb just BEFORE any `cudf::io::read_parquet` or `cudf::concatenate` or cudf allocation calls within `compute_task`
- Log sizes of any pinned buffers allocated

Look for calls to `cudf::io::read_parquet` INSIDE `compute_task` — this is the likely true failure site (not our host_parquet converter, which is a DIFFERENT path).

### Probe site 2 — `duckdb_scan_executor::get_scan_output`
**File:** `src/op/scan/duckdb_scan_executor.cpp`
At line ~216 (`return task->compute_task(stream);`) add:
- Breadcrumb before calling `compute_task` showing `cudaGetDevice()`, stream handle, task type (is_duckdb_scan vs is_parquet_scan)
- Breadcrumb after it returns, same info

### Probe site 3 — `cached_host_parquet_representation`
**File:** `src/data/cached_data_representation.cpp` or wherever `cached_host_parquet_representation` lives.
- Add breadcrumb at any `convert_to` / `clone` / `lock_or_prepare_batch` methods — these may call cudf internally.

### Probe site 4 — Sirius intercept layer (SiriusExecuteQuery)
**File:** `src/sirius_interface.cpp` or similar.
- Breadcrumb at `SiriusExecuteQuery` entry: `cudaGetDevice()`. Confirms what device the call arrives on from DuckDB.

### Run command (with sandbox bypass — GPU access needed)
```bash
# 1. Flip yaml (commit reverted after)
# Edit test/cpp/integration/integration.yaml: num_gpus: 2
# 2. Build
mcp__project-commands__run_command build
# 3. Run single failing test directly (with sandbox bypass)
build/release/extension/sirius/test/cpp/sirius_unittest \
  "gpu_execution hive partition - filter on data column"
# 4. Inspect probes
grep '\[mgpu-probe\]' build/release/extension/sirius/test/cpp/log/sirius_$(date +%Y-%m-%d).log
# 5. REVERT yaml to num_gpus: 1 before any commit
```

MCP doesn't pass through test filters (tested — `unit-tests` runs all 983 regardless of commands.yaml args), so direct binary invocation via Bash + `dangerouslyDisableSandbox: true` is the only way to run just this test.

### Invariants to preserve
- **No `rmm::cuda_stream_default`** anywhere (`grep -rn 'rmm::cuda_stream_default' src/` must stay at 41)
- **No edits to cucascade submodule**
- **integration.yaml must be reverted** to `num_gpus: 1` before any commit
- All probes prefixed with `[mgpu-probe]` (not `[mgpu-audit]` — that's owned by the AUDIT TEST_CASE regex at `test_gpu_execution_tpch_mgpu_audit.cpp:78-79`)
- Build via `mcp__project-commands__run_command build` (never `pixi run make` directly)

## Other paths that might be worth trying

| Approach | Payoff | Cost |
|---|---|---|
| **A.** Probe the pre-converter path (above plan) | Should pinpoint which layer trips `cudaMemcpyBatchAsync` | 30–60 min |
| **B.** `LD_PRELOAD` a shim around `cudaMemcpyBatchAsync` that logs `dsts[0]`, `srcs[0]`, `sizes[0]`, `attrs`, stream, current device BEFORE calling the real function | Absolute ground truth on args | 45 min to write + debug |
| **C.** Force-disable cudf's `cudaMemcpyBatchAsync` path by setting `CUDART_VERSION < 13000` at compile time in a one-file cudf copy, to confirm the bug is in the new batched API specifically | Proves/disproves CUDA-13-bug hypothesis | Heavy — requires patched cudf build |
| **D.** File an upstream cudf issue with the reproducer + this HANDOFF.md's evidence and move on | Unblocks Phase 8 ship | Low, but bug stays open |

**Recommendation:** Start with A. If A doesn't localize the failure in 1 hour, jump to B or D.

## Files changed in this branch (none from this session — all probes reverted)
```
git diff → empty (working tree matches committed state)
git log → only Phase 8 plan/execution commits through 5780e5b
```

## Task state (in GSD task tracker)
- #22 Deep investigation: probe cudf read_parquet failure site — completed (found it's NOT read_parquet)
- #23 Option A: Run num_gpus=2 without --abort — completed (couldn't skip abort; limited data)
- #24 cuda-gdb inspection — completed (pivoted to direct probe; user was right it was something else)
- #19 Gap closure: execute Phase 8 — still in_progress (this hunt is gap closure work)
- #20 Re-verify Phase 8 after gap closure — pending (needs fix first)
- #21 Milestone v1.2 lifecycle — pending (blocked on Phase 8 close)

## Key files to re-read on resume
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-VERIFICATION.md` — the official gaps_found report
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md` — original Open Issue section with 4 hypotheses (A/B/C/D all now disproved, so E — something we haven't considered — is the actual answer)
- This file (08-HUNT-HANDOFF.md)
