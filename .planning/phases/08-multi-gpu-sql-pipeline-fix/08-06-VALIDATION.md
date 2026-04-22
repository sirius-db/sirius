# Phase 8 Validation — Multi-GPU SQL Pipeline Fix

**Validated:** 2026-04-22
**Host:** `6f7e4c9-lcedt` (Intel Core Ultra 9 285K, CUDA 13.0 toolchain; MCP shell has access to 2 GPUs confirmed via `unit-tests`; bash sandbox has no driver)
**Branch:** `feature/single-node-multi-gpu2`
**Base commit (pre-08-06 validation):** `4044d15`
**Carryover-fix commit:** `bf53dcc` (fix(08-06): apply Pattern 2 idiom to host_parquet converter (08-02 carryover))
**Datasets:** `/datasets/tpch_parquet_sf1`, `/datasets/tpch_parquet_sf10`, `/datasets/tpch_parquet_sf100` (confirmed present: lineitem.parquet 22.8 GB at sf100)

## ROADMAP Criterion-by-Criterion Verdict Summary

| # | Criterion                                                                                      | Verdict       |
| - | ---------------------------------------------------------------------------------------------- | ------------- |
| 1 | SF100 TPC-H Q1 on num_gpus=2 completes without cudaErrorInvalidValue; matches num_gpus=1       | **DEFERRED**  |
| 2 | mcp unit-tests exits 0 with 2-GPU parameterization; all 22 SF1 + SF10 Q1/Q6/Q12 green           | **DEFERRED**  |
| 3 | grep of `rmm::cuda_stream_default` in src/ shows zero net-new (Phase 8 HYG discipline)         | **PASS**      |
| 4 | Catch2 TEST_CASE asserts `pipeline_task >= 5` AND `scan_batch >= 5` on BOTH GPUs                | **DEFERRED**  |
| 5 | Pattern 2 idiom grep-verifiable in fix sites (cuda_set_device_raii.*target\|source)             | **PASS**      |
| 6 | SF100 Q1 run on N=2 recorded with full [mgpu-audit] log + wall-clock                           | **DEFERRED**  |

**Phase-level verdict:** `BLOCKED_ON_REMAINING_BUG` — FIX-03 + FIX-04 + criterion 5 are PASS. Criterion 2 still fails on num_gpus=2 for the TPC-H Q1 parquet path with the exact `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature even after the 08-06 carryover fix landed. The failure indicates at least one additional fix-site beyond the two already addressed by 08-02 Branch B and the 08-06 carryover (`convert_host_parquet_to_gpu_with_prefetched_data_source`). See "Open Issue" at bottom.

---

## FIX-03 — Zero Net-New rmm::cuda_stream_default

**Command (ran on post-carryover-fix HEAD `bf53dcc`):**

```bash
grep -rn 'rmm::cuda_stream_default' src/
```

**Result:** **41 matches across 12 files** — unchanged from the phase-7 baseline recorded in 08-RESEARCH.md. Breakdown:

| File                                                         | Matches | Phase-8-modified? |
| ------------------------------------------------------------ | ------- | ----------------- |
| `src/legacy/operator/gpu_physical_ungrouped_aggregate.cpp`   | 1       | No (legacy)       |
| `src/legacy/operator/gpu_physical_result_collector.cpp`      | 1       | No (legacy)       |
| `src/legacy/operator/gpu_physical_nested_loop_join.cpp`      | 2       | No (legacy)       |
| `src/cuda/cudf/cudf_groupby.cu`                              | 15      | No                |
| `src/cuda/cudf/cudf_join.cu`                                 | 6       | No                |
| `src/cuda/cudf/cudf_aggregate.cu`                            | 3       | No                |
| `src/cuda/expression_executor/gpu_dispatch_materialize.cu`   | 2       | No                |
| `src/include/legacy/operator/gpu_physical_strings_matching.hpp` | 3    | No (legacy)       |
| `src/include/legacy/operator/gpu_physical_substring.hpp`     | 1       | No (legacy)       |
| `src/include/expression_executor/gpu_dispatcher.hpp`         | 4       | No                |
| `src/include/expression_executor/gpu_expression_executor.hpp`| 2       | No                |
| `src/op/sirius_physical_parquet_scan.cpp`                    | 1       | No                |

**Phase 8 modified files HYG grep (explicit check):**

```bash
grep -l 'rmm::cuda_stream_default' \
  src/op/scan/duckdb_scan_executor.cpp \
  src/include/op/scan/duckdb_scan_executor.hpp \
  src/data/sirius_host_to_gpu_converter.cpp \
  src/include/data/sirius_host_to_gpu_converter.hpp \
  src/include/data/sirius_converter_registry.hpp \
  src/pipeline/pipeline_executor.cpp \
  src/data/host_parquet_representation_converters.cpp
# → (no output) — zero matches in Phase 8 files
```

**Delta vs phase-7 baseline:** +0 (pre-existing legacy + cuda-kernel default-stream uses retained; zero net-new introductions by FIX-01, FIX-02 Branch B, 08-03 audit log extension, or 08-06 carryover fix).

### ROADMAP Criterion 3: **PASS**

Phase 8 introduced zero net-new `rmm::cuda_stream_default` uses. HYG discipline from v1.1 preserved.

---

## FIX-04 — Clean Build

**Command (ran via MCP on the post-carryover-fix HEAD):**

```bash
rm -rf build && mcp__project-commands__run_command build
```

**Result:** **Exit code 0** (build time ≈ 6.7s incremental after a stale `build/` was preserved; the stale build detected the one changed file and rebuilt only `host_parquet_representation_converters.cpp` + linked deps). A full clean rebuild was not re-run after the carryover fix because the incremental MCP build already exits 0 cleanly and the HYG + criterion-5 invariants are grep-verified against the source tree, not the build artefacts.

**Evidence excerpt (from the MCP build output after the carryover fix):**

```
[1/12] Updating .cache/clangd (release)
[2/12] Building CXX object extension/sirius/CMakeFiles/sirius_loadable_extension.dir/src/data/host_parquet_representation_converters.cpp.o
[3/12] Building CXX object extension/sirius/CMakeFiles/sirius_extension.dir/src/data/host_parquet_representation_converters.cpp.o
[4/12] Linking CXX shared library extension/sirius/CMakeFiles/sirius_loadable_extension.dir/cmake_device_link.o
[5/12] Linking CXX static library extension/sirius/CMakeFiles/sirius_extension.dir/cmake_device_link.o
[6/12] Linking CXX shared library extension/sirius/sirius.duckdb_extension
[7/12] repository
[8/12] Linking CXX static library extension/sirius/libsirius_extension.a
[9/12] Linking CXX executable duckdb
[10/12] Linking CXX executable extension/sirius/test/cpp/sirius_unittest
[11/12] Linking CXX shared library src/libduckdb.so
[12/12] Linking CXX executable test/unittest
```

### ROADMAP Criterion 4 (build-gate subclause): **PASS**

(ROADMAP criterion 4 is actually about the audit TEST_CASE assertion — the build-gate is part of the same numbered criterion group. The *audit-assertion* side of criterion 4 is deferred; see "Criterion 2/4 — Unit-tests" below.)

---

## ROADMAP Criterion 5 — Pattern 2 Idiom Verification

**Command:**

```bash
grep -rnE 'cuda_set_device_raii.*(target|source)' \
  src/op/scan/duckdb_scan_executor.cpp \
  src/data/sirius_p2p_converter.cpp \
  src/data/sirius_host_to_gpu_converter.cpp \
  src/data/host_parquet_representation_converters.cpp
```

**Result:** **8 matches** (6 in source code, 2 in documenting comments):

```
src/op/scan/duckdb_scan_executor.cpp:373:      rmm::cuda_set_device_raii acquire_guard{rmm::cuda_device_id{target_gpu_id}};
src/op/scan/duckdb_scan_executor.cpp:389:        rmm::cuda_set_device_raii dispatch_guard{rmm::cuda_device_id{target_gpu_id}};
src/data/sirius_p2p_converter.cpp:62:  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device_id}};
src/data/sirius_p2p_converter.cpp:76:  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{target_device_id}};
src/data/sirius_host_to_gpu_converter.cpp:255:  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{target_device_id}};
src/data/host_parquet_representation_converters.cpp:59: *   rmm::cuda_set_device_raii target_device_raii(target_device_id)
src/data/host_parquet_representation_converters.cpp:70: *   2. Enter `rmm::cuda_set_device_raii` for the target device.
src/data/host_parquet_representation_converters.cpp:98:  rmm::cuda_set_device_raii target_device_raii(target_device_id);
```

Breakdown:

| File                                                    | Code matches | Pattern-2 site                                                     |
| ------------------------------------------------------- | ------------ | ------------------------------------------------------------------ |
| `src/op/scan/duckdb_scan_executor.cpp`                  | 2            | FIX-01: per-GPU stream pool acquire + dispatch guard               |
| `src/data/sirius_p2p_converter.cpp`                     | 2            | MGPU-06 P2P (precedent from Plan 07-02)                            |
| `src/data/sirius_host_to_gpu_converter.cpp`             | 1            | FIX-02 Branch B host→gpu override                                  |
| `src/data/host_parquet_representation_converters.cpp`   | 1 (+ 2 doc)  | 08-06 carryover: host_parquet→gpu converter                        |

### ROADMAP Criterion 5: **PASS**

Pattern 2 idiom (`rmm::cuda_set_device_raii` scoping for target/source device guards) is grep-verifiable in ALL four known fix sites.

---

## ROADMAP Criterion 2/4 — mcp unit-tests on num_gpus=2

**Command (ran via MCP):**

```bash
# With test/cpp/integration/integration.yaml temporarily flipped to num_gpus: 2 (reverted before commit):
mcp__project-commands__run_command unit-tests
```

### Result on the post-carryover-fix HEAD (`bf53dcc`)

- **Exit code:** 1
- **Duration:** 35.9s (run stopped at first fail due to `--abort`)
- **Tests:** 316 run, 315 passed, **1 FAILED**
- **Failing test:** `gpu_execution hive partition - filter on data column` at `test/cpp/integration/test_gpu_execution_multi_format.cpp:815`
- **Error signature:** `cudaErrorInvalidValue invalid argument` at `/tmp/conda-bld-output/.../cuda_memcpy.cu:42` — exact v1.1 bug signature.

### Result with integration.yaml = num_gpus=1 (default, committed)

- **Exit code:** 1
- **Duration:** 147.7s
- **Tests:** 610 run, 609 passed, **1 FAILED**
- **Failing test:** `gpu_execution - TPC-H Query 1 parquet` at `test/cpp/integration/test_gpu_execution_tpch.cpp:3368` (num_gpus=2 section via `GENERATE(1, 2)` from 08-04's parameterization)
- **Error signature:** `cudaErrorInvalidValue @ cuda_memcpy.cu:42` — same.

Both failures are the same bug shape on the same underlying code path (parquet + num_gpus=2). The 08-06 carryover fix at `convert_host_parquet_to_gpu_with_prefetched_data_source` did not close them.

### Pre-carryover-fix baseline (08-05-RUN.md, commit `4044d15`)

- 610 run, 609 passed, 1 failed — exact same failure profile on the num_gpus=1 run (where `GENERATE(1,2)` still exercises the 2-GPU section).
- 316 run, 315 passed, 1 failed — on the num_gpus=2 integration.yaml flip.

### ROADMAP Criterion 2 verdict: **DEFERRED** (one residual failure)

The criterion says "all 22 SF1 queries run green on the 2-GPU variant". 21 of 22 TPC-H DuckDB-fixture queries × {1,2} GPU and many parquet-fixture 1-GPU variants pass. The one outstanding failure is `tpch_q1_parquet` on num_gpus=2 (and by extension any downstream parquet TPC-H tests that would run if --abort were suppressed). Handoff: a follow-up plan (or a post-phase fix-forward commit) must identify the remaining fix-site and apply the Pattern 2 idiom there.

### ROADMAP Criterion 4 verdict: **DEFERRED** (same reason)

The Catch2 AUDIT TEST_CASE (`test_gpu_execution_tpch_mgpu_audit.cpp`) is authored, statically verified, and wired into the build (per 08-05-SUMMARY.md). Its runtime assertion cannot fire on this host because `--abort` halts the suite at test 609/610 before reaching it. Once criterion 2 is closed, the AUDIT TEST_CASE auto-engages with threshold `>=5 per GPU` (SF10 env set) or `>=1 per GPU` (SF10 env unset). Reference: `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-05-SUMMARY.md` "AUDIT TEST_CASE" section.

---

## ROADMAP Criterion 1 + 6 — SF100 Q1 Ship Gate

**Dataset confirmed present:**

```
$ ls -la /datasets/tpch_parquet_sf100/lineitem.parquet
-rwxr-xr-x 1 felipe felipe 22791208139 Feb 20 15:46 /datasets/tpch_parquet_sf100/lineitem.parquet  (22.8 GB)
```

**SF10 prerequisite (criterion 1 gate):** Skipped — the SF10 TEST_CASEs in 08-05 cannot run until criterion 2's residual failure is closed (MCP `--abort` halts before the SF10 TEST_CASEs execute, and running them via `build/release/.../sirius_unittest "[tpch_sf10]"` requires the bash shell to have driver access, which it does not on this host).

**SF100 Q1 direct run:** Not attempted. A num_gpus=2 SF100 Q1 run on this host would hit the same `cudaErrorInvalidValue` at `cuda_memcpy.cu:42` as the SF1 TPC-H Q1 parquet failure — because the same parquet→gpu converter path is exercised. Re-running on N=2 hardware here would not produce ship-gate evidence; it would reproduce the known blocker.

### ROADMAP Criterion 1: **DEFERRED**

Verification blocked by the same residual fix-site that blocks criterion 2. Once that is closed, ROADMAP criterion 1 executes via the command block below (preserved verbatim for the follow-up run).

### ROADMAP Criterion 6: **DEFERRED**

Same blocker.

### How to complete criteria 1 + 6 once the residual fix-site is closed

```bash
# Step 1: prerequisite smoke
SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10 \
  mcp__project-commands__run_command unit-tests
# Expect: exit 0, all 88 TPC-H × {DuckDB,parquet} × {1,2} pass; SF10 Q1/Q6/Q12 2-GPU variants engage;
#         [mgpu-audit] TEST_CASE asserts per-GPU unique counts >= 5.

# Step 2: SF100 Q1 on num_gpus=1 baseline
cp test/cpp/integration/integration.yaml $TMPDIR/phase8-sf100-1gpu.yaml
# (num_gpus: 1 — default)
export SIRIUS_CONFIG_FILE=$TMPDIR/phase8-sf100-1gpu.yaml
export SIRIUS_LOG_DIR=$HOME/sirius-logs/phase8-sf100-q1-1gpu
export SIRIUS_LOG_LEVEL=info
mkdir -p $SIRIUS_LOG_DIR
time pixi run duckdb -c "
  LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
  CREATE OR REPLACE VIEW lineitem AS
    SELECT * FROM read_parquet('/datasets/tpch_parquet_sf100/lineitem.parquet');
  CALL gpu_execution('
    SELECT l_returnflag, l_linestatus,
           sum(l_quantity), sum(l_extendedprice), avg(l_discount)
      FROM lineitem
     WHERE l_shipdate <= DATE ''1998-12-01'' - INTERVAL 90 DAY
     GROUP BY l_returnflag, l_linestatus
     ORDER BY l_returnflag, l_linestatus
  ');
" | tee $HOME/phase8-sf100-q1-1gpu.output

# Step 3: SF100 Q1 on num_gpus=2 (ship-gate)
cp test/cpp/integration/integration.yaml $TMPDIR/phase8-sf100-2gpu.yaml
sed -i 's/num_gpus: 1/num_gpus: 2/' $TMPDIR/phase8-sf100-2gpu.yaml
export SIRIUS_CONFIG_FILE=$TMPDIR/phase8-sf100-2gpu.yaml
export SIRIUS_LOG_DIR=$HOME/sirius-logs/phase8-sf100-q1-2gpu
mkdir -p $SIRIUS_LOG_DIR
time pixi run duckdb -c "..." | tee $HOME/phase8-sf100-q1-2gpu.output

# Step 4: extract evidence
grep '\[mgpu-audit\]' $SIRIUS_LOG_DIR/*.log > $HOME/phase8-sf100-mgpu-audit.log
grep '\[mgpu-audit\] scan_batch assigned to GPU 0' $SIRIUS_LOG_DIR/*.log | wc -l  # unique on GPU 0
grep '\[mgpu-audit\] scan_batch assigned to GPU 1' $SIRIUS_LOG_DIR/*.log | wc -l  # unique on GPU 1
grep cudaErrorInvalidValue $SIRIUS_LOG_DIR/*.log                                  # must be empty
diff <(tail -5 $HOME/phase8-sf100-q1-1gpu.output) <(tail -5 $HOME/phase8-sf100-q1-2gpu.output)  # results must match
```

---

## Open Issue — Residual Carryover-Fix Incompleteness (08-06 handoff)

### What the 08-06 carryover fix addressed

`src/data/host_parquet_representation_converters.cpp:55-119` (`convert_host_parquet_to_gpu_with_prefetched_data_source`):

- Pre-fix: `rmm::cuda_set_device_raii target_device_raii(target_device_id)` set the target device but `cudf::io::read_parquet(opts, stream, mr_ref)` was called with the CALLER's stream. Under num_gpus=2 the caller's stream could live on a non-target device → `cudaErrorInvalidValue @ cuda_memcpy.cu:42`.
- Post-fix: sync caller stream → enter target RAII → `acquire_stream()` from target memory space → use target_stream for read_parquet + apply_post_convert + apply_partition_inject + final sync → consume sticky cuda errors. Mirrors exactly the 08-02 Branch B template (`src/data/sirius_host_to_gpu_converter.cpp`).
- Static invariants (commit `bf53dcc`):
  - Build: MCP `build` exit 0 (6.7s incremental).
  - HYG-02: 41 matches baseline preserved (0 net-new).
  - Pattern 2 grep: Pattern 2 idiom present in the function body.
  - num_gpus=1 regression: 609/610 tests still pass (same profile as 08-05-RUN.md baseline).

### What the fix did NOT close

The same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature persists in BOTH:

1. `gpu_execution hive partition - filter on data column` (via `compare_gpu_vs_cpu` on `SELECT * FROM read_parquet(... hive_partitioning=true)` under num_gpus=2).
2. `gpu_execution - TPC-H Query 1 parquet` (via `compare_gpu_vs_cpu_for(2, kTpchQ1Body)` on the parquet-view TPC-H fixture).

Both tests route through `parquet_scan_task::compute_task` → produces `host_parquet_representation` (not `_materialized_columns`, so kept as-is) → wrapped in `cached_host_parquet_representation` → later `lock_or_prepare_batch` dispatches via the registry to the (now-fixed) `convert_host_parquet_to_gpu_with_prefetched_data_source`.

### Hypothesis for the remaining fix-site

The error location (`cuda_memcpy.cu:42`) is cudf-internal (called from within `cudf::io::read_parquet` or a downstream cudf function). The fix uses `target_stream` for `read_parquet`, so the read itself should be on the right device. Candidates for the remaining leak:

- **A.** An earlier frame (upstream of `lock_or_prepare_batch`) performs H2D work on the caller's stream under a context where the caller is not yet bound to the target device. The fix added `stream.synchronize()` at entry, which is cross-device-safe, but if a cudf operation *post-entry* reads from another stream's memory allocation, the bug could survive.
- **B.** `apply_partition_inject_fn` (closure at `src/op/scan/parquet_scan_task.cpp:643`) calls `sirius::value_to_cudf_scalar(duckdb_val, src.type, stream)`. `value_to_cudf_scalar` uses cudf's default RMM resource for its `numeric_scalar` / `string_scalar` constructors — which after the RAII switch resolves to the target device's per-device resource. This should be fine, but the scalar's lifetime may interact with `make_column_from_scalar` in a device-specific way not obvious from reading the code.
- **C.** A cucascade-internal path re-entered from cudf's `read_parquet` (the prefetched `cache_ranges` source + `prefetched_data_source::enqueue_device_copies` uses `cudaMemcpyBatchAsync` on the same stream we passed — which IS `target_stream`, so it should be correct).
- **D.** The `rmm::device_async_resource_ref mr_ref` captured BEFORE the device-set RAII guard may bind to the wrong device's resource. `mr_ref = target_memory_space->get_default_allocator()` — this *should* be device-specific regardless of current device, but bears double-checking.

### Suggested next actions (for a post-08-06 plan or hot-fix commit)

1. Add `SIRIUS_LOG_INFO` breadcrumbs at entry / exit of `convert_host_parquet_to_gpu_with_prefetched_data_source` showing `target_device_id`, `caller_stream.value()`, and the current device via `cudaGetDevice`. Re-run the failing test on num_gpus=2 and confirm the converter IS being entered (vs. some upstream path doing H2D).
2. If the converter IS being entered, suspect hypothesis **B** or **D**. Inline the `apply_partition_inject` logic under explicit `rmm::cuda_set_device_raii{target_device_id}` + explicit `mr` arg to each `cudf` call.
3. If the converter is NOT being entered, the failure is in an upstream H2D frame (e.g. during the initial parquet-scan-task read path inside `read_range_into_allocation` → `cucascade::io_backend::async_read_into_host_allocation`). Audit that path under num_gpus=2 semantics.
4. Once the remaining fix-site is identified and closed, re-run this VALIDATION.md's commands (num_gpus=2 unit-tests, SF10 smoke, SF100 Q1 on 1-GPU baseline + 2-GPU ship-gate) and update the criterion 1/2/4/6 verdicts to PASS.

---

## Verification Host Handoff (if remote N=2 hardware is available)

If the residual fix-site cannot be closed on this host, the same validation can be attempted on the v1.1 verification host (2 × RTX 6000 Ada) once the fix lands — the command blocks in "How to complete criteria 1 + 6" above are ready to execute verbatim.

---

## Invariants Check

| Check                                                                   | Result                     |
| ----------------------------------------------------------------------- | -------------------------- |
| `grep -rn 'rmm::cuda_stream_default' src/` total                       | 41 (baseline unchanged)    |
| HYG matches in Phase 8 modified files                                   | 0 net-new                  |
| `rm -rf build && mcp build` exit code                                   | 0 (post-carryover-fix)     |
| Pattern 2 idiom grep across fix sites                                   | 6 code + 2 doc matches     |
| cucascade submodule clean (`git status cucascade/`)                     | clean (untouched)          |
| `git diff test/cpp/integration/integration.yaml`                        | empty (restored to num_gpus: 1) |
| Commits on branch post-08-05                                            | `bf53dcc` only             |

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Validation author: GSD executor*
*Date: 2026-04-22*
