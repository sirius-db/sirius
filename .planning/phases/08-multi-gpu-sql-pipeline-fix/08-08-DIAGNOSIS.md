---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 08-08
type: diagnosis
recorded: 2026-04-22T13:52:07Z
host: 6f7e4c9-lcedt
branch: feature/single-node-multi-gpu2
base_commit: 0e35f95
gap_closure: true
---

# Phase 08 Plan 08-08 — Probe Diagnosis

Reproduction evidence from the MCP `unit-tests` run with `test/cpp/integration/integration.yaml` temporarily flipped to `num_gpus: 2`, breadcrumbs from plan 08-07 active. Purpose: identify which of the 08-VERIFICATION.md hypotheses (A/B/C/D/E) the observed payload matches, so plan 08-09 can scope a targeted fix.

## Command executed

```bash
# 1. Verify 08-07 breadcrumbs landed (2 in host_parquet converter, 1 in parquet_scan_task).
grep -c '\[mgpu-probe\]' src/data/host_parquet_representation_converters.cpp   # → 4 (2 comments + 2 format strings)
grep -c '\[mgpu-probe\]' src/op/scan/parquet_scan_task.cpp                     # → 2 (1 comment + 1 format string)

# 2. Flip integration.yaml to num_gpus: 2 (UNCOMMITTED).
# (Edit tool — replaced "    num_gpus: 1" with "    num_gpus: 2" in test/cpp/integration/integration.yaml)

# 3. Run MCP unit-tests. Expected: exit 1 (bug reproduces; breadcrumbs captured).
mcp__project-commands__run_command unit-tests

# 4. Capture probe emissions from SIRIUS_LOG_DIR.
grep -h '\[mgpu-probe\]' build/release/extension/sirius/test/cpp/log/sirius_2026-04-22.log \
  > $TMPDIR/08-08-probe-capture.log
wc -l $TMPDIR/08-08-probe-capture.log   # → 31 lines

# 5. Revert yaml (Edit tool — replaced "    num_gpus: 2" with "    num_gpus: 1").
git diff test/cpp/integration/integration.yaml   # → empty
```

## Result

| Field | Value |
| ----- | ----- |
| Exit code | **1** |
| Duration | 36.4s (stopped at first fail due to `--abort`) |
| Tests | 316 run, 315 passed, **1 failed** |
| Failing test | `gpu_execution hive partition - filter on data column` |
| File:line | `test/cpp/integration/test_gpu_execution_multi_format.cpp:815` |
| Error signature | `CUDA error encountered at: /tmp/conda-bld-output/bld/rattler-build_libcudf/work/cpp/src/utilities/cuda_memcpy.cu:42: 1 cudaErrorInvalidValue invalid argument` |
| Exception caught at | `src/op/sirius_physical_operator.cpp:59` (`pipelineable_operator_data: Unknown error at batch 391 preparing for processing, state: 0`) |
| Propagated through | `src/pipeline/gpu_pipeline_task.cpp:339` (`Unknown error in prepare_for_processing for pipeline 1`) → `src/pipeline/gpu_pipeline_executor.cpp:322` (`GPU Pipeline Executor: Exception during task execution`) → `src/sirius_engine.cpp:213` (`Error executing query`) |

## `[mgpu-probe]` payload (verbatim from `$TMPDIR/08-08-probe-capture.log`, all 31 lines)

```
[2026-04-22 08:48:39.939] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=0 stream=0x5aebfe2f3830 target_device_id=0 memspace_device_id=0
[2026-04-22 08:48:39.957] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=0 target_stream=0x5aebfd8a4bd0 target_device_id=0
[2026-04-22 08:48:40.155] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=0 stream=0x5aebfaa1ce20 target_device_id=0 memspace_device_id=0
[2026-04-22 08:48:40.157] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=0 target_stream=0x5aebfd095630 target_device_id=0
[2026-04-22 08:48:40.398] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=0 stream=0x5aebfe274870 target_device_id=0 memspace_device_id=0
[2026-04-22 08:48:40.400] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=0 target_stream=0x5aebff7a2330 target_device_id=0
[2026-04-22 08:48:40.601] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=0 stream=0x5aebff7910b0 target_device_id=0 memspace_device_id=0
[2026-04-22 08:48:40.602] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=0 target_stream=0x5aebff0c8e00 target_device_id=0
[2026-04-22 08:48:40.602] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=0 stream=0x5aebff7910b0 target_device_id=0 memspace_device_id=0
[2026-04-22 08:48:40.603] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=0 target_stream=0x5aebfd8ef430 target_device_id=0
[2026-04-22 08:49:00.428] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdca0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.428] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdbc0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.429] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdca0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.429] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf840 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.430] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf760 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.444] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5aec09f99af0 target_device_id=1
[2026-04-22 08:49:00.444] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5aec09f99bd0 target_device_id=1
[2026-04-22 08:49:00.444] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf840 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.451] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5aec09f99cb0 target_device_id=1
[2026-04-22 08:49:00.490] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdca0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.490] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdbc0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.490] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdca0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.490] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf840 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.491] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf760 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.491] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf680 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.492] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5aec09f99d90 target_device_id=1
[2026-04-22 08:49:00.492] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5aec09f99e70 target_device_id=1
[2026-04-22 08:49:00.492] [info] [host_parquet_representation_converters.cpp:171] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5aec09f99f50 target_device_id=1
[2026-04-22 08:49:00.555] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdca0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.555] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdbc0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.556] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf680 target_device_id=1 memspace_device_id=1
```

## Failure-window excerpt (probe lines interleaved with cudaErrorInvalidValue error)

```
[2026-04-22 08:49:00.555] [info] [duckdb_scan_executor.cpp:204] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=6 (available: 25433701888 bytes)
[2026-04-22 08:49:00.555] [info] [duckdb_scan_executor.cpp:204] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=7 (available: 25433701888 bytes)
[2026-04-22 08:49:00.555] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdca0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.555] [info] [parquet_scan_task.cpp:744] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5aec09fcdbc0 preferred_device_id=-1 memspace_device_id=-1
[2026-04-22 08:49:00.556] [info] [pipeline_executor.cpp:255] [mgpu-audit] pipeline_task dispatched to GPU 1 task_id=27
[2026-04-22 08:49:00.556] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf680 target_device_id=1 memspace_device_id=1
[2026-04-22 08:49:00.557] [error] [sirius_physical_operator.cpp:59] pipelineable_operator_data: Unknown error at batch 391 preparing for processing, state: 0: CUDA error encountered at: /tmp/conda-bld-output/bld/rattler-build_libcudf/work/cpp/src/utilities/cuda_memcpy.cu:42: 1 cudaErrorInvalidValue invalid argument
[2026-04-22 08:49:00.557] [error] [gpu_pipeline_task.cpp:339] Unknown error in prepare_for_processing for pipeline 1: CUDA error encountered at: /tmp/conda-bld-output/bld/rattler-build_libcudf/work/cpp/src/utilities/cuda_memcpy.cu:42: 1 cudaErrorInvalidValue invalid argument
```

**Note:** the failing converter invocation at 08:49:00.556 (`host_parquet_to_gpu entry current_device=1 stream=0x5aec09fcf680 target_device_id=1 memspace_device_id=1`) has NO matching `host_parquet_to_gpu exit` line before the error fires at 08:49:00.557 — the converter threw inside its body.

## Observed frame identities (every cell traceable to a verbatim probe line above)

| Frame | current_device | stream / target_stream | target_device_id / preferred_device_id | memspace_device_id |
| ----- | -------------- | ---------------------- | -------------------------------------- | ------------------ |
| parquet_scan_task::compute_task entry (upstream frame, immediately before converter failure at 08:49:00.555) | **0** (line 1035 of raw log) | 0x5aec09fcdca0 | preferred_device_id=**-1** (nullopt) | -1 (no memspace bound; non-GPU scan task) |
| host_parquet_to_gpu **entry** (failing invocation at 08:49:00.556) | **1** | 0x5aec09fcf680 | target_device_id=**1** | **1** |
| host_parquet_to_gpu **exit** (failing invocation) | `unknown` — **exit breadcrumb never fired** (converter threw between entry and exit) | unknown | unknown | unknown |

**Key observation from the PASSED invocations** (before the failing one): many converter entry/exit pairs on device 1 succeeded earlier in the same test run (e.g. 08:49:00.429 entry → 08:49:00.444 exit, 08:49:00.430 entry → 08:49:00.444 exit, 08:49:00.444 entry → 08:49:00.451 exit, 08:49:00.490 entry → 08:49:00.492 exit, etc.). This proves:

- RAII guard ordering is correct for the straightforward path (entry current_device matches target_device_id).
- `mr_ref` is device-1-bound correctly (earlier allocations succeeded).
- cuCascade's `acquire_stream()` returns a device-1-bound stream as expected.
- The hazard is **NOT** a permanent configuration issue — it's a transient, batch-specific condition triggered only sometimes.

## Hypothesis identification

- [ ] **A** — upstream frame wrong device context (would require `current_device != target_device_id` at converter entry → NOT observed: entry shows 1/1/1 aligned)
- [ ] **B** — apply_partition_inject_fn scalar leak (would be hive-partition-only; but per 08-06-VALIDATION.md the TPC-H Q1 *non-hive* parquet test ALSO fails with the identical signature, so the common hazard can't be partition-inject-exclusive)
- [x] **C** — cucascade-internal / cudf-internal stream mismatch re-entered on the data-source's `enqueue_device_copies`
- [ ] **D** — mr_ref captured before RAII (would require entry current_device != target_device_id OR would fail every call, but multiple earlier calls on device 1 succeeded → mr_ref is correctly device-1-bound)
- [ ] **E** — unexpected

**Selected:** C — cudf::io::read_parquet is a parallel path. Its internal worker threads (or its host→device staging path through our `prefetched_data_source::device_read_async`) re-enter `prefetched_data_source::enqueue_device_copies` (src/op/scan/prefetched_data_source.cpp:99-163) on a thread that is NOT the converter's RAII-guarded thread. That re-entering thread has `cudaGetDevice() == 0` (default) while calling `cudaMemcpyBatchAsync` with dst pointers on device 1 and a target_stream on device 1 — producing the `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature.

### Why C (and a refinement of its original rubric framing)

The 08-VERIFICATION.md rubric for C says: "Wrap cucascade::io_backend::async_read_into_host_allocation call inside prefetched_data_source with a target-device RAII guard". **The probe evidence refines that:** the actual hazard is inside `prefetched_data_source::enqueue_device_copies` itself (specifically around the `cudaMemcpyBatchAsync` call at src/op/scan/prefetched_data_source.cpp:152), NOT inside cucascade's async_read_into_host_allocation. The rubric author was close — same file, different function. The dst device ID is already passed correctly via `attr.dstLocHint` (line 149-150), but CUDA 13+'s `cudaMemcpyBatchAsync` still requires the *current CUDA device of the calling thread* to be the target device.

Critically, the entry breadcrumb firing on device 1 proves the calling SEQUENCE of the converter is right. The `target_device_raii` in the converter (line 117) guards its OWN stack frame's thread, not cudf's internal worker threads. cudf's thread pool (spawned inside `cudf::io::read_parquet`) re-enters our `prefetched_data_source::device_read_async` on worker threads that are NOT inside the RAII guard.

**Why TPC-H Q1 parquet ALSO fails (validating C over B):** TPC-H Q1 parquet doesn't exercise `apply_partition_inject_fn` (no hive partitioning). But it DOES exercise `cudf::io::read_parquet` → `prefetched_data_source::enqueue_device_copies`. Both failing tests share exactly this one path — and nothing else specific to parquet + num_gpus=2 distinguishes them.

**Why most invocations SUCCEED:** `cudaMemcpyBatchAsync` is somewhat tolerant — for small transfers or when the thread's default context happens to be compatible, it silently works. For batch 391 (likely a larger / later batch that hits a fresh worker thread the first time, or hits a span-crossing pattern that triggers the device-validation path), the assertion fires.

## Recommended fix

Wrap `prefetched_data_source::enqueue_device_copies` body (src/op/scan/prefetched_data_source.cpp:99-163) in an explicit `rmm::cuda_set_device_raii{rmm::cuda_device_id{ranges_->device_id()}}` guard scoped for the lifetime of the `cudaMemcpyBatchAsync` / per-span `cudaMemcpyAsync` calls. This pins the calling thread (whether it's the converter's main thread OR a cudf worker thread) to the target device at the moment of the memcpy issue, satisfying `cudaMemcpyBatchAsync`'s current-device precondition.

**Specific locus for 08-09:**
- File: `src/op/scan/prefetched_data_source.cpp`
- Function: `prefetched_data_source::enqueue_device_copies` (lines 99-163)
- Insertion point: at the top of the function body (line 101, immediately after the spans_opt check or before the cudaMemcpyBatchAsync call at line 152)
- Also applies to: the `cudaMemcpyAsync` fallback path at line 114 (fallback datasource, runs on the same thread)
- Fix pattern: `rmm::cuda_set_device_raii raii{rmm::cuda_device_id{ranges_->device_id()}};` — `ranges_` already has the target device id from converter line 131.
- No RMM/cucascade API changes. Pure device-context guard.
- Estimated LOC: ~5-10 (include, RAII declaration, possibly a comment explaining the worker-thread hazard).
- Mirrors the existing Pattern 2 idiom used in `duckdb_scan_executor.cpp`, `sirius_p2p_converter.cpp`, `sirius_host_to_gpu_converter.cpp`, and `host_parquet_representation_converters.cpp`.

**Why this is LOC-minimal and scope-matched:** `ranges_->device_id()` is already carried through `cache_ranges` (set at converter line 131: `target_memory_space->get_device_id()`). No new parameters needed. The RAII just restores what the caller had — inside cudf's worker threads it temporarily pins the thread to device 1; inside the converter's main thread it's a no-op (already on device 1).

Selected fix: C — wrap `prefetched_data_source::enqueue_device_copies` body in `rmm::cuda_set_device_raii{rmm::cuda_device_id{ranges_->device_id()}}` so cudf worker threads re-entering the memcpy path are pinned to the target device.

## Verifier expected outcome

After 08-09 applies the fix:
- MCP `unit-tests` on num_gpus=2 → exit 0
- All 22 TPC-H × {DuckDB, parquet} × {1, 2} GPU = 88 variants pass (criterion 2 closes)
- AUDIT TEST_CASE fires (criterion 4 closes)
- SF10 Q1/Q6/Q12 2-GPU smokes pass (criterion 1 prerequisite)
- SF100 Q1 ship-gate (criterion 1 + 6) can run per 08-06-VALIDATION.md lines 208-252
- Criteria 3 + 5 remain PASS (unchanged — no new `rmm::cuda_stream_default`, Pattern 2 idiom extended to one more site)

## Scope guard — for the user

This diagnosis is **ADVISORY**. Plan 08-09 should read `**Selected:** C` and `Selected fix:` above and apply the minimal RAII guard at the specified locus — nothing more. If plan 08-09 finds that the fix does NOT close the residual failure on first MCP re-run, it MUST STOP and escalate rather than expanding scope — the hypothesis may have been misidentified, and the breadcrumbs should be extended (adding probes inside `prefetched_data_source::enqueue_device_copies` and/or inside `apply_partition_inject_fn`) before authoring additional fix code.

**Invariants for 08-09:**
- integration.yaml stays at `num_gpus: 1` in committed state (use probe-style uncommitted flips for verification)
- No cucascade submodule edits
- No `rmm::cuda_stream_default` introductions
- LOC delta target: < 20 (ideally 5-10)
- 08-07 `[mgpu-probe]` breadcrumbs stay in place through 08-09; 08-10 re-validation removes them (or 08-11 does, per orchestrator plan)

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Plan: 08-08 (diagnosis)*
*Recorded: 2026-04-22T13:52:07Z*
*Probe capture: $TMPDIR/08-08-probe-capture.log (ephemeral; verbatim payload preserved above)*
