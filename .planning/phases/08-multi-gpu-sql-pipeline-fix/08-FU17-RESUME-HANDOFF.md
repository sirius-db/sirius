---
phase: 08-multi-gpu-sql-pipeline-fix
type: resume-handoff
recorded: 2026-04-23T09:15:00Z
branch: feature/single-node-multi-gpu2
base_commit: HEAD (post OOM-retry bump + MGPU tests)
purpose: Resume follow-up #17 after the partial fix landed — the cross-GPU illegal-address hazard still needs a real fix
---

# Follow-up #17 — resume handoff after partial fix

## What is already resolved in this branch

1. **Batch-lock retry exhaustion on SF100 Q11 cold.** The original crash signature
   `gpu_pipeline_task: failed to lock batch for processing — exceeded 10 OOM
   retries` was the symptom of cross-GPU BUILD_PROBE contention timing out
   the OOM-retry budget before the conflicting task's processing handle
   released. Bumping `MAX_OOM_RETRIES` from 10 → 100 and the per-retry
   backoff from 5 ms → 50 ms (≈ 5 s total patience) in
   `src/pipeline/gpu_pipeline_executor.cpp` gets past this contention
   layer. Q1 – Q11 cold now all complete at SF100.

2. **MGPU regression test infrastructure.** `test/cpp/operator/mgpu_test_utils.hpp`
   plus four `*_mgpu.cpp` tests (hash_join, grouped_aggregate_merge, order,
   table_gpu cache warm) give us deterministic per-operator coverage on
   num_gpus=2 without needing a full SF100 run. 12 / 13 MGPU cases pass at
   HEAD — see the "Known pre-existing failure" section below for the one
   that doesn't.

3. **Two unrelated bugs discovered along the way.** Called out below so they
   don't get lost.

## What is still broken

`SF100 2-GPU parquet with cache=table_gpu` still fails — just at a different
stage and with a different signature. After the retry bump, Q11 now crashes
with the **real** symptom the original resume note described:

```
[ERROR] pipelineable_operator_data: Unknown error at batch 29564 preparing
         for processing, state: 0:
         CUDA error at: .../rmm/cuda_stream_view.cpp:45:
         cudaErrorIllegalAddress an illegal memory access was encountered

[ERROR] Unknown error in prepare_for_processing for pipeline 12:  (same)
[ERROR] Fatal CUDA error encountered at: .../cudf/utilities/cuda_memcpy.cu:42:
        700 cudaErrorIllegalAddress
```

The illegal address was produced by a kernel launched **earlier** on the
stream — the next `stream.synchronize()` is where it surfaces. This is the
cross-GPU stale-pointer / peer-access hazard the original follow-up #17
hypothesized about. The retry bump only removed an earlier failure mode
that was masking it.

Evidence this is a cross-GPU memory access bug (not a contention bug):

- No batch_lock_utils errors anywhere in the log.
- The `[mgpu-probe]` breadcrumb immediately before the error shows
  `host_parquet_to_gpu entry current_device=1`, but the faulting
  `cudaMemcpy` is launched from the pipeline on `current_device=?`. The
  device at launch is not captured at the faulting site — adding a
  breadcrumb there is the first concrete action below.
- The "invalid hash table build state 2" warning fires BEFORE the error
  (benign, unchanged — from the over-eager task_creator loop on BUILD_PROBE).

## Investigation plan to pick up

### Step 1 — localize the faulting kernel with synchronous launches

```bash
# Run from the repo root.
mkdir -p /tmp/fu17-resume && \
env CUDA_LAUNCH_BLOCKING=1 \
    SIRIUS_LOG_LEVEL=debug \
    SIRIUS_LOG_DIR=/tmp/fu17-resume \
    SIRIUS_CONFIG_FILE=/tmp/claude/bench-cfg/sirius-2gpu-sf100.yaml \
    OUTPUT_DIR=/tmp/fu17-resume \
    ./test/tpch_performance/run_tpch_parquet.sh \
      --parquet-dir /home/felipe/sirius/test_datasets/tpch_parquet_sf100 \
      --iterations 2 sirius 100 11
```

`CUDA_LAUNCH_BLOCKING=1` makes every kernel synchronous, so the backtrace
lands at the real bad launch instead of the next `stream.synchronize()`.
The expected artifacts are:

1. An earlier `cudaErrorIllegalAddress` coming out of a specific operator's
   `execute()` call — name + operator id in the log line.
2. The `[mgpu-probe]` breadcrumb immediately before that error, telling us
   `current_device` vs `target_device_id` at the time the kernel launched.

### Step 2 — tighten the `[mgpu-probe]` coverage on the suspect frames

The existing probes are at `parquet_scan_task::compute_task` entry,
`host_parquet_to_gpu` entry + exit. They don't cover downstream operators
(hash_join `execute`, partition / concat, merge_sort). Add entry probes to
the usual suspects so the last breadcrumb before the crash names the
operator that launched the bad kernel:

- `src/op/sirius_physical_hash_join.cpp::execute` — already has a large
  body; add `[mgpu-probe]` at the SCHEDULED branch and at the probe branch.
  Focus on BUILD_PROBE because that's what Q11 uses.
- `src/op/sirius_physical_concat.cpp::execute`.
- `src/op/sirius_physical_partition.cpp::execute`.

Each probe should log `current_device`, `stream.value()`, and
`input_batch->get_memory_space()->get_device_id()` so the mismatch
is obvious when it appears.

### Step 3 — classify the hazard shape

Three candidate root causes, each with a distinct fingerprint:

**A. Stale pointer from cache=table_gpu.** Cached batches share
`shared_ptr<data_batch>` across warm iterations. If a cached batch was
mutated to `gpu_table_representation` on GPU 0 during cold, and warm's
downstream gets pinned to GPU 1, the batch's underlying `cudf::table`
holds GPU-0 pointers. The P2P converter at
`src/data/sirius_p2p_converter.cpp` should kick in via `lock_or_prepare_batch`
on the memspace mismatch path. Possible failure: the converter's
`cudaMemcpyPeerAsync` is launched on target_stream but the source
pointer's home device isn't actually enabled for peer access from the
current context. Verify with
`nvidia-smi topo -p2p -r` and `cudaDeviceCanAccessPeer` logging at
SiriusContext init time.

**B. BUILD_PROBE `_build_table` held as `shared_ptr<data_batch>` on the
build-task's GPU.** The probe tasks pin to the same GPU via
`operator_id % num_gpus` (see SCHED-00 in `src/creator/task_creator.cpp`).
If the build-task's execute() sets `_hash_table` on GPU X but a probe task
lands on GPU Y (e.g. the partition-pinning math rounds differently across
num_partitions > num_gpus), the probe's cuco lookup accesses GPU-X memory
from a GPU-Y context.

**C. `parquet_scan_task`'s `_datasource` cached across batches.** `_datasource`
is a `shared_ptr<cucascade::io::cucascade_datasource>` bound to the
GPU-specific backend picked on first call
(`src/op/scan/parquet_scan_task.cpp:804`). If two scan tasks for the same
rowgroup-partition get scheduled on different GPUs, they share the same
datasource pinned to the first GPU — the second task's `host_read_async`
goes to the wrong backend.

Use the Step-2 probes to distinguish these by noting which operator fired
the first bad launch.

### Step 4 — fix + regression guard

Once the exact kernel + frame is known, the fix will be one of:

- P2P path hardening in `sirius_p2p_converter.cpp`.
- Per-GPU datasource materialization in `parquet_scan_task`.
- Partition-pinning math correction in `task_creator.cpp` SCHED-00.

Add a TEST_CASE to `test/cpp/operator/test_physical_hash_join_mgpu.cpp`
that covers the exact operator/shape. The `bisect-*` TEST_CASEs that
landed in this branch are a template: pick the minimal data size that
reliably reproduces, keep it `[stress]`-tagged if the repro needs
100+ MiB, and assert GPU matches CPU so a regression fails the test.

## Tools the branch leaves in place for the next session

- `test/cpp/operator/mgpu_test_utils.hpp` — `scoped_mgpu_env`,
  `scoped_log_dir`, `write_mgpu_yaml`, `generate_parquet_surface`,
  `parse_audit_log`, `require_gpu_matches_cpu`. Header-only. Reuse in the
  Step-4 regression guard.
- `test/cpp/operator/test_physical_hash_join_mgpu.cpp` — `[followup-17]`
  tagged scale-up TEST_CASE runs a Q11-shaped BUILD_PROBE at ~512 MiB
  with `cache=table_gpu`. Currently passes because 512 MiB isn't enough
  to reliably trigger the cross-GPU illegal-address race. Bump the data
  size or add `CUDA_LAUNCH_BLOCKING` via `GTEST_MGPU_LAUNCH_BLOCKING`
  env var if you want to force it into unit-test scope.

## Known pre-existing failures unrelated to follow-up #17

1. **HASH_JOIN SEMI column-index out-of-bounds.**
   `test_physical_order_mgpu - small sort stays single-GPU` fails with
   `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)`
   in `prepare_join_keys` at `sirius_physical_hash_join.cpp:637`.
   - Not MGPU-specific (same plan shape fails on num_gpus=1).
   - DuckDB compiles small ORDER BY + LIMIT into a TOP_N → SEMI-JOIN plan
     that Sirius's plan converter carries a schema for where every
     operator expects 3/4 columns but the batches carry 1/2 — the
     "bobbi (todo): delim join will return this warning for now" at
     `gpu_pipeline_task.cpp:54` already flags this region as shaky.
   - Marked as a known-failing regression guard; the MGPU tests
     otherwise stay green.

2. **Q22 OOM across all configurations** — follow-up #2 in
   `08-FOLLOWUPS.md`. Unchanged.

## Branch state

```
feature/single-node-multi-gpu2
  HEAD
  ├── test(08): add per-operator MGPU regression tests + shared utils
  │   test/cpp/operator/mgpu_test_utils.hpp                                    [NEW]
  │   test/cpp/operator/test_physical_hash_join_mgpu.cpp                       [NEW]
  │   test/cpp/operator/test_physical_grouped_aggregate_merge_mgpu.cpp         [NEW]
  │   test/cpp/operator/test_physical_order_mgpu.cpp                           [NEW]
  │   test/cpp/integration/test_table_gpu_cache_warm_mgpu.cpp                  [NEW]
  │   CMakeLists.txt                                                           [TEST_SOURCES]
  │
  └── fix(08): bump OOM retry budget for cross-GPU batch-lock contention
      src/pipeline/gpu_pipeline_executor.cpp                                   [constants]
```

`.ai-helper/commands.yaml` with the new `tpch-parquet` + diagnostics
commands is a separate (user-authored) commit outside these two.

## How to resume

```
Pick up follow-up #17 from
.planning/phases/08-multi-gpu-sql-pipeline-fix/08-FU17-RESUME-HANDOFF.md.
Start with Step 1 — localize the faulting kernel with CUDA_LAUNCH_BLOCKING.
```

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Authored: 2026-04-23T09:15:00Z*
*Covers: follow-up #17 continuation after the OOM-retry bump*
