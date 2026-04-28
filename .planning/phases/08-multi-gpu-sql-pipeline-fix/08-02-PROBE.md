# FIX-02 Probe Record

**Plan:** 08-02 (FIX-02 audit + conditional host→gpu converter override)
**Recorded:** 2026-04-21
**Host:** `.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272` (Felipe's dev worktree)
**Branch:** `feature/single-node-multi-gpu2`
**Base commits under test:**

- `2ff8a3c` — feat(08-01): add `_gpu_stream_pools` per-GPU map to `duckdb_scan_executor`
- `2150777` — fix(08-01): dispatch scan on target-GPU stream with device guard
- `0d03ca1` — docs(08-01): reproduction deferral

## Probe Verdict

**Probe FAIL** — re-run on 2026-04-21 by the orchestrator against real GPU
hardware via MCP project-commands (the worktree bash sandbox has no GPU, but
MCP does). One integration test fails on `num_gpus: 2` with the exact v1.1
bug signature: `cudaErrorInvalidValue` at `cuda_memcpy.cu:42`. FIX-01 closed
the scan-dispatch path (Site A) but NOT the hive-partition / host-to-gpu
converter path (Site C: `convert_host_fast_to_gpu`).

**Branch Decision: Branch B** — author a Sirius-side `host_data_representation
→ gpu_table_representation` converter override mirroring Pattern 2 shape with
a target-bound stream + target-device RAII guard.

See "Probe re-run on 2026-04-21 (orchestrator)" section below for exact
command, exit code, and failing-test details. The original "DEFERRED" note
(preserved below for history) reflected an earlier sandboxed attempt that
had no GPU access.

## Probe re-run on 2026-04-21 (orchestrator)

The initial probe (section "Hardware Availability" below) returned DEFERRED
because the worktree's shell sandbox has no GPU access. The orchestrator
re-ran the probe using `mcp__project-commands__run_command unit-tests`, which
DOES have GPU access on this host. The 2-GPU configuration was applied by
temporarily editing `test/cpp/integration/integration.yaml` to
`num_gpus: 2` and reverting after the run (verified via `git diff` empty on
the yaml file before this record).

### Command

```bash
# Temporarily flipped test/cpp/integration/integration.yaml num_gpus: 1 -> 2
mcp__project-commands__run_command unit-tests
# Then reverted the yaml back to num_gpus: 1
```

### Result

- **Exit code:** 1
- **Tests:** 316 run, 315 passed, **1 FAILED**
- **Failing test:** `gpu_execution hive partition - filter on data column`
  - **File:** `test/cpp/integration/test_gpu_execution_multi_format.cpp:815`
  - **Assertion:** `REQUIRE_FALSE` at line 100 (of the fixture) reported the
    diff between GPU and CPU result was non-zero (i.e., the GPU path raised
    an exception and failed to produce a result).
- **Error verbatim:**

```
CUDA error encountered at:
/tmp/conda-bld-output/work/cpp/src/copying/cuda_memcpy.cu:42:
1 cudaErrorInvalidValue invalid argument
```

### Interpretation

This is EXACTLY the v1.1 bug signature. The hive-partition read path hits
`host_data_representation → gpu_table_representation` via cucascade's
`convert_host_fast_to_gpu` (Site C in the audit below). FIX-01 closed the
scan-executor dispatch path (Site A), but it does NOT bind a target-bound
stream across the host→gpu converter frame — cucascade's body at
`representation_converter.cpp:849` calls `batch.flush(stream, ...)` with the
caller's stream, which may not be bound to the target device.

### Branch Decision

**Branch B selected** — author Sirius-side host→gpu converter override
mirroring Pattern 2 with `cudaMemcpyHostToDevice` under `rmm::cuda_set_device_raii`
for the target GPU, and consume a target-bound stream acquired from
`target_memory_space->acquire_stream()` (not the caller's stream).

Task 3 of 08-02-PLAN.md is UNBLOCKED by this decision.

## Post-Override Re-Probe on 2026-04-21

After landing Branch B (commit `96481df`), the orchestrator flipped
`test/cpp/integration/integration.yaml` back to `num_gpus: 2` and re-ran
`mcp__project-commands__run_command unit-tests`.

### Result

- **Exit code:** 1
- **Tests:** 316 run, 315 passed, **1 FAILED** (SAME test, SAME signature)
- **Failing test:** `gpu_execution hive partition - filter on data column`
- **Error verbatim:**

```
CUDA error encountered at:
/tmp/conda-bld-output/.../cuda_memcpy.cu:42:
1 cudaErrorInvalidValue invalid argument
```

### Interpretation — Branch B Did Not Close the Hive-Partition Failure

Branch B was designed to close the `host_data_representation →
gpu_table_representation` converter path (Site C from the static audit,
cucascade's `convert_host_fast_to_gpu`). The hive-partition failing test
does NOT go through that path — it uses `host_parquet_representation` and
routes through Sirius's OWN converter
`convert_host_parquet_to_gpu_with_prefetched_data_source` at
`src/data/host_parquet_representation_converters.cpp:55-119`.

That Sirius-owned converter has the SAME bug shape as cucascade's
`convert_host_fast_to_gpu`:

- Line 66 opens `rmm::cuda_set_device_raii{ target_device_id }` (correct).
- Line 92 calls `cudf::io::read_parquet(opts, stream, mr_ref)` with the
  CALLER's stream. Under `num_gpus == 2` the caller's stream may be bound
  to a different device than `target_device_id`, raising
  `cudaErrorInvalidValue` inside cudf's internal H2D path.

### This is a Distinct Fix-Site — Not Recharged Here

Per 08-02-PLAN's `<resume_instructions>` step 3: "If a different test fails
instead, document but do NOT chase it here — that's 08-06 ship-gate scope."
Interpretation for this probe: the TEST is the same, but the FIX-SITE is
different (host_parquet_representation, not host_data_representation).
Branch B is scoped to `host_data_representation → gpu_table_representation`
per the plan's `files_modified` and acceptance criteria; it cannot close
a separate-representation converter. The plan explicitly called out only
Site C (cucascade's convert_host_fast_to_gpu) as the Branch B target.

**Handoff for 08-06 (or a future FIX-02-extension plan):** Apply the same
"acquire target-bound stream + RAII on target device" pattern to
`convert_host_parquet_to_gpu_with_prefetched_data_source` in
`src/data/host_parquet_representation_converters.cpp`. This fix is
structurally identical to Branch B: acquire the target stream from
`target_memory_space->acquire_stream()`, replace the caller's `stream`
with it for the `cudf::io::read_parquet` call and all subsequent
`apply_post_convert` / `apply_partition_inject` / `stream.synchronize()`
calls, and add a final sync. Branch B's
`src/data/sirius_host_to_gpu_converter.cpp` is the canonical template.

### What Branch B DID Accomplish

- Closed the `host_data_representation → gpu_table_representation` path
  structurally (the failure site that cucascade's body exposed at L849).
- Established the Pattern 2 idiom for any Sirius-owned host→gpu conversion,
  including the public-API-only column-tree reconstruction approach used
  by `sirius_host_fast_to_gpu_factory`.
- Registered cleanly at converter_registry::initialize() after the MGPU-06
  block; MCP build exits 0; num_gpus=1 unit-tests full-suite passes (979
  test cases, 78789857 assertions — no regression).
- HYG-02 baseline preserved (41 `rmm::cuda_stream_default` matches across
  `src/`, 0 introduced by new file).

### HYG-02 Recheck

`grep -rn 'rmm::cuda_stream_default' src/` → 41 matches across 12 files
(unchanged from baseline). `src/data/sirius_host_to_gpu_converter.{hpp,cpp}`
contains 0 matches.

## Hardware Availability

```
$ nvidia-smi -L
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

Runtime probe attempt on this host (recorded in
`/tmp/claude-1002/fix02-probe-log/probe-attempt-q1.log`):

```
$ SIRIUS_CONFIG_FILE=/tmp/claude-1002/sirius-fix02-probe.yaml \
  build/release/extension/sirius/test/cpp/sirius_unittest \
    "[integration][gpu_execution][TPC-H][Q1]"
Failed to initialize NVML: Driver Not Loaded
Failed to initialize NVML: Driver Not Loaded
terminate called after throwing an instance of 'std::runtime_error'
  what():  SiriusContext::initialize: cucascade::topology_discovery reported 0 GPUs
           — refusing to initialize on stub topology (MGPU-01 fail-hard).
```

The binary loads; cucascade's topology discovery reports 0 GPUs;
`SiriusContext::initialize` correctly fail-hards per MGPU-01. This is the
expected outcome on a driver-less host and matches 08-01 REPRODUCTION.md.

## Probe Commands (for re-execution on verification host)

The 2-GPU config lives at `/tmp/claude-1002/sirius-fix02-probe.yaml` (copy of
`test/cpp/integration/integration.yaml` with `num_gpus: 2`). A verification
host with 2 GPUs re-runs the probe as follows (mirrors the exact command
matrix Plan 08-02 prescribes):

```bash
mkdir -p "$TMPDIR/fix02-probe-log"
cp test/cpp/integration/integration.yaml "$TMPDIR/sirius-fix02-probe.yaml"
sed -i 's/num_gpus: 1/num_gpus: 2/' "$TMPDIR/sirius-fix02-probe.yaml"

# Core 4-query probe: DuckDB-fixture variants first, then parquet-fixture
# variants (which exercise the materialized_columns / cached-scan path that
# routes through cucascade's convert_host_fast_to_gpu).
for tag in \
  "[integration][gpu_execution][TPC-H][Q1]" \
  "[integration][gpu_execution][parquet][TPC-H][Q1]" \
  "[integration][gpu_execution][TPC-H][Q3]" \
  "[integration][gpu_execution][parquet][TPC-H][Q3]" \
  "[integration][gpu_execution][TPC-H][Q6]" \
  "[integration][gpu_execution][parquet][TPC-H][Q6]" \
  "[integration][gpu_execution][TPC-H][Q12]" \
  "[integration][gpu_execution][parquet][TPC-H][Q12]" ; do
  SIRIUS_LOG_LEVEL=info \
  SIRIUS_LOG_DIR="$TMPDIR/fix02-probe-log" \
  SIRIUS_CONFIG_FILE="$TMPDIR/sirius-fix02-probe.yaml" \
    build/release/extension/sirius/test/cpp/sirius_unittest "$tag"
done

# Cache-hit variant: run the same parquet Q1 twice back-to-back so the
# second invocation materializes from the cached_host_data_representation
# path — this is the convert_host_fast_to_gpu path specifically.
for i in 1 2 ; do
  SIRIUS_LOG_LEVEL=info \
  SIRIUS_LOG_DIR="$TMPDIR/fix02-probe-log" \
  SIRIUS_CONFIG_FILE="$TMPDIR/sirius-fix02-probe.yaml" \
    build/release/extension/sirius/test/cpp/sirius_unittest \
      "[integration][gpu_execution][parquet][TPC-H][Q1]"
done
```

Then analyze:

```bash
grep -cE 'cudaErrorInvalidValue' "$TMPDIR"/fix02-probe-log/*.log
grep -cE 'cuda_memcpy\.cu'       "$TMPDIR"/fix02-probe-log/*.log   # file match only (Pitfall 2)
grep -cE '\[mgpu-audit\] scan_batch assigned to GPU 1' "$TMPDIR"/fix02-probe-log/*.log
grep -cE '\[mgpu-audit\] scan_batch assigned to GPU 0' "$TMPDIR"/fix02-probe-log/*.log
```

A clean probe shows: (a) zero matches on the first two greps, (b) non-zero
matches on both of the `[mgpu-audit]` greps (dispatch actually hit both GPUs),
(c) result-comparison delta vs `num_gpus: 1` baseline is byte-identical for
every listed TEST_CASE.

## Per-Query Results

Runtime results **pending on verification host**. The table below is seeded
with the expected shape so 08-06 can fill it in.

| Query   | Fixture      | Status  | cudaErrorInvalidValue | cuda_memcpy.cu | GPU-0 batches | GPU-1 batches | Result delta vs 1-GPU |
| ------- | ------------ | ------- | --------------------- | -------------- | ------------- | ------------- | --------------------- |
| Q1      | DuckDB       | PENDING | —                     | —              | —             | —             | —                     |
| Q1      | parquet      | PENDING | —                     | —              | —             | —             | —                     |
| Q1-x2   | parquet (cache hit) | PENDING | —              | —              | —             | —             | —                     |
| Q3      | DuckDB       | PENDING | —                     | —              | —             | —             | —                     |
| Q3      | parquet      | PENDING | —                     | —              | —             | —             | —                     |
| Q6      | DuckDB       | PENDING | —                     | —              | —             | —             | —                     |
| Q6      | parquet      | PENDING | —                     | —              | —             | —             | —                     |
| Q12     | DuckDB       | PENDING | —                     | —              | —             | —             | —                     |
| Q12     | parquet      | PENDING | —                     | —              | —             | —             | —                     |

## Audit Sites Covered

Static audit of every cross-device cudaMemcpy* call-site in `src/pipeline/`
and `src/op/` — classified against the v1.1 root-cause bug shape
(stream-device mismatch with allocation-device when `num_gpus >= 2`).

### Site A — `src/op/scan/duckdb_scan_executor.cpp` (scan dispatch)

- **Status:** CLOSED by FIX-01 (Plan 08-01).
- **Shape:** Was the `v1.1` bug site. `exc_stream` was acquired from a GPU-0-bound pool even when `target_gpu_id == 1`. Post-FIX-01, the executor holds a per-GPU pool map `_gpu_stream_pools[target_gpu_id]`; the dispatch lambda opens with `rmm::cuda_set_device_raii dispatch_guard{rmm::cuda_device_id{target_gpu_id}}`.
- **Evidence:** `src/include/op/scan/duckdb_scan_executor.hpp:197` (member); `src/op/scan/duckdb_scan_executor.cpp:70, 350-351, 382` (populate + lookup + dispatch guard). `grep -c '_gpu_stream_pools' src/include/op/scan/duckdb_scan_executor.hpp` == 1 (FIX-01 landed).
- **FIX-02 action:** None — already closed upstream.

### Site B — `src/include/pipeline/batch_lock_utils.hpp:48-125` `lock_or_prepare_batch`

- **Status:** SAFE (no bug shape present on this frame).
- **Shape:** The `stream` parameter is supplied by `pipelineable_operator_data::prepare_for_processing` (`src/op/sirius_physical_operator.cpp:51`), which received it from `gpu_pipeline_task::execute` (`src/pipeline/gpu_pipeline_task.cpp:280`). The upstream stream is bound to the pipeline-executor's single GPU (`src/pipeline/gpu_pipeline_executor.cpp:45`: `_stream_pool` is per-executor, one executor per GPU). So the `stream` reaching `lock_or_prepare_batch` is already target-bound when `target_space->get_tier() == GPU`.
- **Caveat:** `batch->convert_to<gpu_table_representation>(registry, target_space, stream)` at `:87` consumes this caller stream and hands it to cucascade's converter. The converter (`cucascade::convert_host_fast_to_gpu` at `cucascade/src/data/representation_converter.cpp:820-856`) also consumes the caller stream for `reconstruct_column` + `batch.flush` + cudf::table ctor under a target `device_guard`. **The ONLY residual risk** is that cucascade's body does not acquire its OWN target-bound stream — so if the CALLER ever supplies a stream bound to a non-target device, a cross-device hazard exists. Site A is the known such caller; FIX-01 closed it. See Site C.
- **FIX-02 action:** No Sirius-side fix on this file. Risk only materializes via a mis-bound caller stream, and all known callers are now correct post-FIX-01.

### Site C — cucascade `convert_host_fast_to_gpu` (FIX-02 probe target)

- **Location:** `cucascade/src/data/representation_converter.cpp:820-856` (pinned at `f47de0b`, not modifiable by v1.2 per user constraint).
- **Shape:** Sets `rmm::cuda_set_device_raii{target_device_id}` at L837 (correct); but performs `reconstruct_column` + `batch.flush` + cudf::table ctor on the CALLER's stream (L843-851). If the caller stream is bound to a non-target device, the H2D copies inside `batch.flush(caller_stream, ...)` can race against the target device_guard — this IS the v1.1 bug shape, just in a different frame.
- **Why it matters for FIX-02:** this is the frame FIX-02's probe is designed to expose. Post-FIX-01, the `cached_host_data_representation` path (`parquet_scan_task.cpp:855-860` → returns a cached host batch → later converted on a pipeline-executor stream) uses this converter. The pipeline-executor stream IS target-bound, so the bug is structurally avoided for that caller — but only the probe can confirm no residual caller is mis-bound.
- **FIX-02 action (probe PASS):** Document only; FIX-01 closed the known caller.
- **FIX-02 action (probe FAIL):** Author `src/data/sirius_host_to_gpu_converter.{hpp,cpp}` mirroring Pattern 2 — acquire target-bound stream, sync caller stream, perform reconstruct + flush on target_stream. Register via `sirius_converter_registry` unregister+register after the MGPU-06 P2P block.
- **Feasibility caveat for Branch B:** `BatchCopyAccumulator` and `reconstruct_column` are NOT in cucascade's public headers (`cucascade/include/cucascade/data/*.hpp`) — they live inside `cucascade/src/data/representation_converter.cpp:396, 717`. Verified via `grep -rn 'BatchCopyAccumulator|reconstruct_column' cucascade/include/` → 0 matches. A Sirius-side override therefore cannot call `cucascade::reconstruct_column` directly. Two options: (B-i) fallback pack/unpack shape mirroring `src/data/sirius_p2p_converter.cpp` L64-125 with `cudaMemcpyHostToDevice` instead of `cudaMemcpyPeerAsync`, or (B-ii) expose the symbols in cucascade (requires submodule change, forbidden in v1.2). Branch B MUST take path (B-i) if exercised.

### Site D — `src/op/scan/duckdb_scan_task.cpp:444-467` `make_data_batch`

- **Status:** SAFE (host-only).
- **Shape:** Constructs a `host_data_representation` from a `host_table_allocation`; no cudaMemcpy* call. The H2D step happens later, when `convert_to<gpu_table_representation>` runs in `lock_or_prepare_batch` (Site B/C).
- **FIX-02 action:** None.

### Site E — `src/op/scan/cpu_source_task.cpp:86-304`

- **Status:** SAFE (host-only).
- **Shape:** Flattens DuckDB chunks into a host allocation, wraps as `host_data_representation`. No cudaMemcpy*. Same as Site D: H2D deferred to the converter.
- **FIX-02 action:** None.

### Site F — `src/op/scan/parquet_scan_task.cpp:855-862` (materialized_columns path)

- **Status:** CONDITIONALLY safe.
- **Shape:** When `_materialized_columns == true`, the scan task calls `registry.convert<gpu_table_representation>(*parquet_representation, _gpu_memory_space, stream)` at L851 with a stream it received from `duckdb_scan_executor`. Post-FIX-01, that stream IS target-bound — fine. Then at L855 it converts back down to `host_data_representation` for caching. The cached batch is later re-converted to GPU via `lock_or_prepare_batch` → Site B/C, which is when the convert_host_fast_to_gpu frame executes.
- **FIX-02 action:** Covered by Sites A (caller) + C (converter). No new fix sites here.

### Site G — `src/op/scan/prefetched_data_source.cpp:100-163`

- **Status:** SAFE on scan-dispatch path.
- **Shape:** Issues `cudaMemcpyAsync(dst, host_buf, cudaMemcpyHostToDevice, stream.value())` to H2D bytes into a device buffer. The `stream` is the scan-task stream that FIX-01 now binds to `target_gpu_id`. `cudaMemcpyBatchAsync` branch uses `attr.dstLocHint.id = ranges_->device_id()` which is the target-device id (correct).
- **FIX-02 action:** None. This site was already correct because the caller stream was already correctly target-bound — FIX-01 just closed the OUTER scope where the scan executor itself was picking a wrong-GPU stream to pass DOWN to this frame.

### Site H — `src/op/scan/positional_delete_filter.cpp:69-73`

- **Status:** SAFE on scan-dispatch path.
- **Shape:** `cudaMemcpyAsync(bool_col->mutable_view().data, keep.data(), N, cudaMemcpyHostToDevice, stream.value())` to upload a delete-mask. Same analysis as Site G — caller stream is target-bound post-FIX-01; no cross-device hazard.
- **FIX-02 action:** None.

### Site I — `src/op/scan/iceberg_scan_task.cpp:96-121`

- **Status:** SAFE (D2H, not cross-device in the bug sense).
- **Shape:** Three `cudaMemcpy(...DeviceToHost)` calls to read positional-delete vectors back to host for processing. D2H is single-device; the hazard is stream-device mismatch, and the calls don't use an async stream at all (synchronous `cudaMemcpy`). Still correct as long as the current device at call time matches the device that holds `pos_col`. FIX-01 pins the thread via `rmm::cuda_set_device_raii{target_gpu_id}` at dispatch-lambda entry, so the current device here is `target_gpu_id` == the device that holds the column. Safe.
- **FIX-02 action:** None. (A future hygiene pass could convert these to async on a target-bound stream, but that's out of scope for FIX-02.)

### Site J — `src/op/sirius_physical_sort_partition.cpp:124-128`, `src/op/sirius_physical_sort_sample.cpp:236-240`

- **Status:** SAFE (post-scan operators run under gpu_pipeline_executor).
- **Shape:** `cudaMemcpyAsync` for D2H (partition) and H2D (sample indices). These execute inside gpu_pipeline_task bodies where the worker thread is already pinned to the executor's GPU (`src/pipeline/gpu_pipeline_executor.cpp:54-72`), and the stream comes from the executor's own pool. Single-device, not cross-device.
- **FIX-02 action:** None.

### Site K — `src/pipeline/*` — cross-device memcpy audit

- **Status:** CLEAN.
- **Evidence:** `grep -rn 'cudaMemcpy' src/pipeline/` returns 0 matches. All pipeline-level cross-device work is mediated by `lock_or_prepare_batch` → converter registry → Site C, already covered above.
- **FIX-02 action:** None.

### Site L — `src/op/sirius_physical_result_collector.cpp:158`

- **Status:** SAFE (GPU→host conversion at result boundary).
- **Shape:** `clone_batch->convert_to<cucascade::host_data_representation>(registry, &mem_space, stream)` is the OPPOSITE direction from FIX-02's concern (GPU→host, driven by the CPU-tier target). Caller stream is the final-pipeline-stage stream; no cross-device memcpy hazard in this direction because all GPU state is flushed before cucascade writes to host.
- **FIX-02 action:** None.

## Cross-Device Memcpy Hygiene — Summary

| Source directory          | cross-device memcpy sites classified | covered-by-FIX-01 | still-to-fix |
| ------------------------- | ------------------------------------ | ----------------- | ------------ |
| `src/pipeline/`           | 0                                    | N/A               | 0            |
| `src/op/scan/` (dispatch) | 1 (Site A)                           | Yes               | 0            |
| `src/op/scan/` (H2D H-ops) | 4 (Sites G, H, I, part of D/E/F)    | Transitively (via caller-stream fix) | 0 |
| `src/op/` (other)         | 2 (Site J — single-device D2H/H2D)   | N/A (single-GPU frame) | 0      |
| cucascade (Site C)        | 1 (convert_host_fast_to_gpu)         | No                | **Probe needed (deferred)** |

HYG-02 baseline: `grep -rn 'rmm::cuda_stream_default' src/` == 41 (unchanged
from 08-01 SUMMARY.md). No net-new uses introduced by Task 1 (no source
edits).

## Branch Decision (updated 2026-04-21)

**ACTUAL (post-re-run): Branch B selected.** The MCP-driven re-probe above
produced a concrete FAIL on the hive-partition filter test with
`cudaErrorInvalidValue` at `cuda_memcpy.cu`, confirming Site C (cucascade's
`convert_host_fast_to_gpu`) is not discharged by FIX-01. Task 3 authors
the host→gpu converter override per Pattern 2 (pack/unpack fallback with
`cudaMemcpyHostToDevice`) since `BatchCopyAccumulator` and
`reconstruct_column` are not in cucascade's public headers.

The original Branch-C recommendation (preserved below) applied under the
constraint that no GPU was available; MCP's runtime access has now made the
probe deterministic.

## Branch Decision (original, superseded)

**Recommended: Branch C (DEFERRED — re-run probe on verification host during 08-06)**

Rationale:

1. **Host has no GPU driver.** `nvidia-smi -L` fails and the Sirius unittest
   binary fail-hards in `SiriusContext::initialize` before any query runs.
   Runtime cudaErrorInvalidValue / cuda_memcpy.cu signal cannot be produced.
2. **Static audit is complete.** All 12 candidate cross-device memcpy sites
   in `src/pipeline/` + `src/op/` are classified against the v1.1 bug shape
   and documented above. The audit found no new fix-site beyond Site C
   (cucascade's `convert_host_fast_to_gpu`), and Site C can only be
   discriminated (closed by FIX-01 vs. needs-override) with runtime signal.
3. **Branch B feasibility risk if forced.** If the verification-host probe
   FAILs on Site C, Branch B must author the override. But
   `BatchCopyAccumulator` / `reconstruct_column` are NOT cucascade public
   symbols (verified via grep in `cucascade/include/`). Branch B would have
   to use the pack/unpack fallback shape (`cudaMemcpyHostToDevice` via the
   `sirius_p2p_converter.cpp` L64-125 idiom). Doable, but non-trivial —
   authoring it speculatively without probe signal violates Pitfall 5
   ("don't add preemptive overrides").
4. **Consistency with 08-01's pattern.** Plan 08-01's reproduction was also
   deferred to 08-06 under the same host-constraint — this is the approved
   pattern for the Felipe dev worktree.

Branches not chosen:

- **Branch A (Probe PASS, doc-only).** Cannot be claimed without runtime
  signal — doing so would be a false-positive.
- **Branch B (Author host→gpu override).** Speculative without probe
  failure; adds ~150 lines of code with a feasibility caveat (public-symbol
  issue forces pack/unpack fallback). Violates Pitfall 5.

**Decision Checkpoint (for orchestrator/user):** confirm Branch C selection,
OR override to A or B with explicit rationale. If B is selected, Task 3 must
use the pack/unpack fallback shape per the feasibility caveat documented in
Site C above.

## Handoff

- **Plan 08-06 (ship gate):** execute the Probe Commands block above on the
  N=2 verification host. Fill in the Per-Query Results table. If the verdict
  is PASS, flip this file's verdict line to `Probe PASS` and leave the
  Branch C recommendation intact (FIX-02 discharged by FIX-01 scope). If the
  verdict is FAIL, return to Plan 08-02 with Branch B selected; the Site C
  feasibility caveat (pack/unpack fallback) is the required implementation
  shape.

## Cleanup

- `/tmp/claude-1002/sirius-fix02-probe.yaml` — 2-GPU yaml config (left in
  place for 08-06 reuse; not committed to the repo).
- `/tmp/claude-1002/fix02-probe-log/` — probe-attempt log directory
  (contains the NVML/0-GPUs error transcript). Not committed.
