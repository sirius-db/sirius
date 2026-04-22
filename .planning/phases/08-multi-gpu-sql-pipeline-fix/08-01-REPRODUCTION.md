# FIX-01 Reproduction Record

**Plan:** 08-01 (FIX-01: per-GPU stream pool in duckdb_scan_executor)
**Recorded:** 2026-04-22
**Host:** `.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272` (Felipe's dev worktree)
**Branch:** `feature/single-node-multi-gpu2`
**Commits under test:**

- `2ff8a3c` — feat(08-01): add `_gpu_stream_pools` per-GPU map to `duckdb_scan_executor`
- `2150777` — fix(08-01): dispatch scan on target-GPU stream with device guard

## Hardware Availability

**Single-GPU host — no GPU available.** Probed via:

```
$ nvidia-smi -L
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

The execution host for this plan (`ws-9aa781df-6d8c-4395-9329-737a67e8e272`)
runs on a development machine that has NO NVIDIA driver active at the time
of plan execution. Accordingly, the 2-GPU TPC-H reproduction that the plan
prescribes (Step 3: `SIRIUS_CONFIG_FILE=$TMPDIR/sirius-fix01-reproduction.yaml
build/release/extension/sirius/test/cpp/sirius_unittest [tpch_q1]`) cannot
be exercised here.

## Build-Gate Outcome (still exercised)

```
$ mcp__project-commands__run_command build
Exit code: 0
[1/12] Updating .cache/clangd (release)
[2/12] Building CXX object .../duckdb_scan_executor.cpp.o
[3/12] Building CXX object .../duckdb_scan_executor.cpp.o (loadable)
...
[12/12] Linking CXX shared library src/libduckdb.so
```

The FIX-01 code changes compile cleanly under the project's standard
release build. This is the maximum validation possible on a single-GPU
host. Runtime correctness must be proven on N=2 hardware.

## Static Invariants Verified

All grep-based acceptance criteria from 08-01-PLAN.md pass:

| Invariant                                                         | Location(s)                                   | Result                                                |
| ----------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| `_gpu_stream_pools` declared in header                            | `src/include/op/scan/duckdb_scan_executor.hpp` | 1 match                                               |
| `_gpu_stream_pools` used in cpp                                   | `src/op/scan/duckdb_scan_executor.cpp`        | 3 matches (constructor populate + `find` at dispatch) |
| Old `_stream_pool` singular member removed                        | both files                                    | 0 matches                                             |
| Old `_stream_pool->acquire_stream(...)` call removed              | cpp                                           | 0 matches                                             |
| `rmm::cuda_set_device_raii` wrapping acquire + dispatch           | cpp                                           | 2 matches (acquire guard + dispatch guard)            |
| `rmm::cuda_stream_default` net-new uses in modified files         | both files                                    | 0 matches (HYG-02 preserved)                          |
| `rmm::cuda_stream_default` baseline across `src/` (context check) | all of `src/`                                 | 41 (unchanged from pre-plan baseline)                 |

## Expected Runtime Behavior (deferred to 08-06 ship gate)

On N=2 hardware, when a TPC-H parquet query (e.g., Q1 SF1) runs with
`num_gpus: 2`:

1. `select_target_gpu()` emits `[mgpu-audit] scan_batch assigned to GPU 0`
   AND `[mgpu-audit] scan_batch assigned to GPU 1` across the stream
   of batches (already proven on 1-GPU scheduling by v1.1 E2E logs from
   commit `fd24174`).

2. For each batch routed to GPU 1, the dispatch lambda:
   - Acquires `exc_stream` from `_gpu_stream_pools[1]` (not `[0]`).
   - Pins the worker thread to GPU 1 via
     `rmm::cuda_set_device_raii dispatch_guard{rmm::cuda_device_id{1}}`.
   - Calls `get_scan_output(scan_task, stream)` where `stream` is now
     bound to GPU 1 AND the current device is GPU 1 — matching the pair.

3. No `cudaErrorInvalidValue` is reported from `cuda_memcpy.cu` (the
   v1.1 E2E failure signature — see
   `.planning/milestones/v1.1-E2E-VERIFICATION.md`).

4. Query results match the 1-GPU baseline (validation via
   `test/cpp/integration/tpch_queries/compare_gpu_vs_cpu`).

Confirmation pending: **single-GPU host — FIX-01 build-gate passed;
2-GPU runtime reproduction pending execution on verification host with
N=2 GPUs.** The full 2-GPU validation occurs at Plan 08-06 (SF100 ship
gate) on the verification host (2 × RTX 6000 Ada, per
`.planning/STATE.md` "v1.1 Verification").

## Handoff

- **Plan 08-02 (FIX-02 probe):** proceed normally. If the 2-GPU run at
  08-06 reveals `cudaErrorInvalidValue` still fires on a different
  materialized_columns / host→gpu code path, 08-02's override branch
  (Pattern 2 applied to `host_data_representation → gpu_table_representation`)
  must be exercised. If the 08-06 run is clean, 08-02 documents
  probe-pass only — no additional override needed.

- **Plan 08-06 (ship gate):** is the canonical 2-GPU runtime
  confirmation for FIX-01. If the scan-path still fails there, treat it
  as a FIX-01 regression rather than a fresh bug — the fix landed here
  but wasn't sufficient. (No indication from static analysis that this
  is likely; the code shape matches the proven Pattern 2 idiom from
  `src/data/sirius_p2p_converter.cpp`.)

## Cleanup

No throwaway files were created; no cleanup required.
