---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 08-08
type: diagnosis
recorded: 2026-04-24T03:55:00Z
host: local workstation (2x RTX 6000 Ada, driver 595.58.03, CUDA 13.2)
branch: feature/single-node-multi-gpu2
base_commit: 6d73680
gap_closure: true
---

# Phase 08 Plan 08-08 — Probe Diagnosis

Reproduction evidence from two MCP `unit-tests` runs with `test/cpp/integration/integration.yaml` temporarily flipped to `num_gpus: 2`, with plan 08-07 `[mgpu-probe]` breadcrumbs active. The yaml was reverted before the diagnosis commit.

## Command executed

```bash
# 1. Flip integration.yaml to num_gpus=2 (TEMPORARY, reverted before commit)
sed -i 's/    num_gpus: 1/    num_gpus: 2/' test/cpp/integration/integration.yaml

# 2. Run via MCP (twice — first TPC-H Q1 parquet, then hive partition)
mcp__project-commands__run_command unit-tests filter='"gpu_execution - TPC-H Query 1 parquet"'
mcp__project-commands__run_command unit-tests filter='"gpu_execution hive partition - filter on data column"'

# 3. Extract probes from build log
LOG=build/release/extension/sirius/test/cpp/log/sirius_2026-04-24.log
grep '[mgpu-probe]' $LOG > /tmp/claude/08-08-probe-capture.log

# 4. REVERT yaml flip (probe style; NEVER commit the flip)
git checkout -- test/cpp/integration/integration.yaml
```

## Result

| Field | Value |
| ----- | ----- |
| Exit code (Q1 parquet) | -1 (SIGSEGV) |
| Exit code (hive partition) | -1 (SIGSEGV) |
| Failing test 1 | `gpu_execution - TPC-H Query 1 parquet` @ test_gpu_execution_tpch.cpp:3368 |
| Failing test 2 | `gpu_execution hive partition - filter on data column` @ test_gpu_execution_multi_format.cpp:815 |
| Error signature | **SIGSEGV - Segmentation violation signal** (NOT `cudaErrorInvalidValue @ cuda_memcpy.cu:42` as documented in 08-06-VALIDATION.md) |
| Failure location (Q1) | test_gpu_execution_tpch.cpp:207 (REQUIRE_FALSE(gpu_result->HasError()) region) |
| Failure location (hive) | test_gpu_execution_multi_format.cpp:100 |

## [mgpu-probe] payload

Total captured: 20 probe lines. Hive-partition test produced ~15 of them under num_gpus=2 (03:54:17 timestamps). Verbatim capture from `/tmp/claude/08-08-probe-capture.log`:

```
[2026-04-24 03:54:17.296] [info] [parquet_scan_task.cpp:763] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5c2313dca670 preferred_device_id=-1 memspace_device_id=-1
[2026-04-24 03:54:17.296] [info] [parquet_scan_task.cpp:763] [mgpu-probe] parquet_scan_task::compute_task entry current_device=0 stream=0x5c2313dca590 preferred_device_id=-1 memspace_device_id=-1
[2026-04-24 03:54:17.298] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=0 batch_state=1 current_device=0 stream=0x5c2313dcb5a0 batch_device_id=-1 target_device_id=0 lock_status=3 success=false
[2026-04-24 03:54:17.298] [info] [batch_lock_utils.hpp:95] [mgpu-probe] lock_or_prepare_batch memspace_mismatch batch_id=0 batch_state=1 current_device=0 stream=0x5c2313dcb5a0 batch_device_id=-1 target_device_id=0 target_tier=0
[2026-04-24 03:54:17.298] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=0 stream=0x5c2313dcb5a0 target_device_id=0 memspace_device_id=0
[2026-04-24 03:54:17.298] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=1 batch_state=1 current_device=1 stream=0x5c2313dcc210 batch_device_id=-1 target_device_id=1 lock_status=3 success=false
[2026-04-24 03:54:17.298] [info] [batch_lock_utils.hpp:95] [mgpu-probe] lock_or_prepare_batch memspace_mismatch batch_id=1 batch_state=1 current_device=1 stream=0x5c2313dcc210 batch_device_id=-1 target_device_id=1 target_tier=0
[2026-04-24 03:54:17.298] [info] [host_parquet_representation_converters.cpp:98] [mgpu-probe] host_parquet_to_gpu entry current_device=1 stream=0x5c2313dcc210 target_device_id=1 memspace_device_id=1
[2026-04-24 03:54:17.329] [info] [host_parquet_representation_converters.cpp:183] [mgpu-probe] host_parquet_to_gpu exit current_device=0 target_stream=0x5c231178d4e0 target_device_id=0
[2026-04-24 03:54:17.329] [info] [host_parquet_representation_converters.cpp:183] [mgpu-probe] host_parquet_to_gpu exit current_device=1 target_stream=0x5c2313c0e1a0 target_device_id=1
[2026-04-24 03:54:17.341] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=2 batch_state=2 current_device=1 stream=0x5c2313dcc210 batch_device_id=1 target_device_id=1 lock_status=0 success=true
[2026-04-24 03:54:17.341] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=3 batch_state=2 current_device=0 stream=0x5c2313dcb5a0 batch_device_id=0 target_device_id=0 lock_status=0 success=true
[2026-04-24 03:54:17.341] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=3 batch_state=2 current_device=0 stream=0x5c2313dcb5a0 batch_device_id=0 target_device_id=0 lock_status=0 success=true
[2026-04-24 03:54:17.342] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=2 batch_state=2 current_device=1 stream=0x5c2313dcc210 batch_device_id=1 target_device_id=1 lock_status=0 success=true
[2026-04-24 03:54:17.342] [info] [batch_lock_utils.hpp:69] [mgpu-probe] lock_or_prepare_batch entry batch_id=3 batch_state=1 current_device=1 stream=0x5c2313dcc210 batch_device_id=0 target_device_id=1 lock_status=3 success=false
```

## Observed frame identities

| Frame | current_device | stream / target_stream | target/preferred_device_id | memspace/batch_device_id |
| ----- | -------------- | ---------------------- | -------------------------- | ------------------------ |
| compute_task entry (task 1) | 0 | 0x5c2313dca670 | preferred_device_id=**-1** (UNSET) | memspace_device_id=**-1** (UNSET) |
| compute_task entry (task 2) | 0 | 0x5c2313dca590 | preferred_device_id=**-1** (UNSET) | memspace_device_id=**-1** (UNSET) |
| host_parquet_to_gpu entry (batch 0 on GPU 0) | 0 | 0x5c2313dcb5a0 | target_device_id=0 | memspace_device_id=0 |
| host_parquet_to_gpu entry (batch 1 on GPU 1) | **1** | 0x5c2313dcc210 | target_device_id=**1** | memspace_device_id=**1** |
| host_parquet_to_gpu exit (batch 0) | 0 | 0x5c231178d4e0 | target_device_id=0 | — |
| host_parquet_to_gpu exit (batch 1) | **1** | 0x5c2313c0e1a0 | target_device_id=**1** | — |
| lock_or_prepare_batch (batch_id=3 on GPU 0) | 0 | 0x5c2313dcb5a0 | target_device_id=0, lock_status=0, success=true | batch_device_id=0 |
| lock_or_prepare_batch (batch_id=3 on GPU 1) | **1** | 0x5c2313dcc210 | target_device_id=**1**, lock_status=3 (memspace_mismatch), success=**false** | batch_device_id=**0** (collision!) |

## Hypothesis identification

- [ ] A — upstream frame wrong device context (rejected: `host_parquet_to_gpu entry current_device` MATCHES `target_device_id` on both GPU 0 and GPU 1 paths — the converter DOES enter the correct device context)
- [ ] B — `apply_partition_inject_fn` scalar leak (rejected: entry+exit `current_device` = `target_device_id` on both paths; converter is clean per 08-07 Pattern 2 fix)
- [ ] C — cucascade-internal stream mismatch (rejected: converter enters AND exits cleanly on both GPUs; cudf::read_parquet completes on the correct target_stream)
- [ ] D — `mr_ref` captured before RAII (rejected: entry `memspace_device_id` = `target_device_id` — mr_ref IS resolving to the correct per-device allocator)
- [x] **E — unexpected** (NOT A/B/C/D; novel pattern — see "Evidence" below)

**Selected:** E — The host_parquet converter itself is healthy (entry/exit both device-correct on both GPUs). But `lock_or_prepare_batch` reveals **the same batch_id=3 (with batch_device_id=0) being dispatched to BOTH GPU 0 tasks (lock_status=0 success=true) AND a GPU 1 task (lock_status=3 memspace_mismatch, success=false)**. The GPU 1 task's failed lock leaves the pipeline in an inconsistent state, producing SIGSEGV downstream (different failure shape than the cudaErrorInvalidValue @ cuda_memcpy.cu:42 documented in 08-06-VALIDATION.md).

## Evidence of novel pattern

**Finding 1 — compute_task preferred_device_id=-1:** Both compute_task entries show `preferred_device_id=-1, memspace_device_id=-1`. The preference field is unset at scan-task dispatch time. Upstream pipeline dispatcher (`pipeline_executor` + `duckdb_scan_executor`) is not propagating target GPU preference into `parquet_scan_task` before it lands — 08-07 probe captures the field as sentinel -1.

**Finding 2 — batch_id=3 double-dispatch:** `batch_id=3 batch_device_id=0` is concurrently locked by tasks running on `current_device=0` (success=true, batch_state=2=locked) AND a task running on `current_device=1` (success=false, batch_state=1 reverted, lock_status=3=memspace_mismatch). The lock-attempt on GPU 1 would trigger `convert_host_parquet_to_gpu_with_prefetched_data_source` a second time (as a re-import) on an already-resident batch, racing with the GPU 0 task.

**Finding 3 — failure mode changed from v1.1:** 08-06-VALIDATION.md recorded `cudaErrorInvalidValue @ cuda_memcpy.cu:42`. This run (post-08-07) produces **SIGSEGV**. Either the 08-07 breadcrumbs altered timing sufficiently to reshape the race, OR an intermediate commit (`1f80c2a fix(08): multi-GPU task distribution + partition pinning`, or `6d73680 fix(08): bump OOM retry budget for cross-GPU batch-lock contention`) changed the crash path. The SIGSEGV is likely a null-deref or use-after-free downstream of the failed lock, since the task on GPU 1 apparently proceeds past the failed lock-or-prepare without bailing cleanly.

## Recommended fix

**STOP. Do not author 08-09 against any of hypotheses A-D.** The observed pattern does not match. Instead, the following investigation sequence is required before a fix can be scoped:

1. **Audit the scan-task distributor** — `duckdb_scan_executor::manager_loop` and the pipeline task queue:
   - Why is the same `batch_id` being assigned to tasks on both GPUs?
   - Is the task-to-GPU assignment respecting `batch_device_id` for already-resident batches, or is it purely round-robin / first-available?
   - Does `08-01`'s per-GPU stream pool change (`_gpu_stream_pools` + `select_target_gpu()`) correctly steer a re-dispatched batch back to its owning GPU, or does it blindly route to available-capacity GPU?
2. **Extend instrumentation at the dispatcher level** — add `[mgpu-probe]` emissions at:
   - `pipeline_executor.cpp:255` (already emits `[mgpu-audit]` — extend to include `batch_id`)
   - `duckdb_scan_executor.cpp:204` (already emits `[mgpu-audit] scan_batch` — extend with already-assigned-to metadata)
   - `batch_lock_utils.hpp:lock_or_prepare_batch` — log the CALLER's task_id so double-dispatches can be correlated back to pipeline_task assignment.
3. **Fix candidates (to be scoped AFTER investigation):**
   - **Candidate 1:** When `lock_or_prepare_batch` returns success=false + lock_status=3 on a batch with `batch_device_id >= 0`, the caller task should yield/reassign to the owning GPU rather than forcing a cross-device re-import via host_parquet_to_gpu.
   - **Candidate 2:** Scan-task distributor should use `batch_device_id` as a sticky-assignment hint — already-resident batches never route to a non-owning GPU task.
   - **Candidate 3:** `parquet_scan_task::compute_task` should not see `preferred_device_id=-1`. The pipeline must set preference before dispatch; currently the plumbing is broken.

Selected fix: STOP. Hypothesis E means gap-closure cannot proceed within the current LOC budget. Open a new investigation plan (v1.2.2 or first plan of Phase 9) scoped to scan-task distribution + batch-ownership affinity.

## Verifier expected outcome

This diagnosis is NOT a closure path for v1.2 as originally scoped. After the scan-task-distribution investigation recommended above lands, the verification matrix is:

- MCP `unit-tests` on num_gpus=2: exit 0 (all 22 SF1 TPC-H × {DuckDB, parquet} × {1,2} = 88 variants pass)
- AUDIT TEST_CASE fires (`pipeline_task ≥ 5` AND `scan_batch ≥ 5` per GPU)
- Add a new AUDIT REQUIRE: `counts[0].batch_ids ∩ counts[1].batch_ids == ∅` (no batch appears on both GPUs concurrently)
- SF100 Q1 ship-gate per 08-06-VALIDATION.md lines 208-252

## Scope guard — for the user

**This diagnosis is ADVISORY and the gap-closure stream is now HALTED.** Per 08-08-PLAN.md's hypothesis-E corner-case instruction: *"STOP. Do not author 08-09. Surface the novel pattern as a decision checkpoint back to the user."*

The following decisions are now the user's to make:

1. **Defer v1.2.1 ship and open a new investigation phase** (v1.2.2 or Phase 9) scoped to scan-task distribution + batch-ownership affinity. Estimated LOC: unknown (investigation first); the 50-LOC hypothesis budget is insufficient.
2. **OR: patch blindly with Candidate 2 above** (batch_device_id sticky-assignment) and hope it closes both SIGSEGV and cudaErrorInvalidValue paths. Higher risk.
3. **OR: revert the 08-07 breadcrumbs** to restore the original cudaErrorInvalidValue failure mode, and investigate from the known signature. The SIGSEGV is novel.

**Recommended:** option 1. The probe evidence is sufficient to scope the investigation precisely — the bug is **not** in the host_parquet converter (08-07 Pattern 2 is correct), it is **upstream in the scan-task distributor** allowing cross-GPU batch double-dispatch.

---
*Phase: 08-multi-gpu-sql-pipeline-fix*
*Recorded: 2026-04-24*
*Verdict: HALT — hypothesis E (unexpected pattern)*
