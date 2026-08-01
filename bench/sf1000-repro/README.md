# TPC-H SF1000 reproduction — 9.139 s on GB300

Reproduces a **9.139 s** TPC-H SF1000 suite (22 queries, best-of-3, 22/22 byte-identical),
against a **15.99 s** starting point — **−42.8%**.

Measured 2026-08-01 on `pmgb300ws-0163`: GB300, 152 SMs, 256 GB HBM, 72-core Grace aarch64,
driver 595.58.03, CUDA 13.2.

This branch is for reproduction only. It is not proposed for merge, and several changes here
are deliberately not PR-ready (see *Status* at the bottom).

---

## Quick start

```bash
# 1. Sirius
git clone https://github.com/felipeblazing/sirius.git && cd sirius
git checkout repro/sf1000-9.14s
git submodule update --init --recursive     # NOT automatic; required
pixi run make

# 2. Patched libcudf (clones felipeblazing/cudf @ perf/sirius-sf1000-repro)
pixi run bash bench/sf1000-repro/build-libcudf.sh

# 3. Run
DATA=/path/to/tpch_parquet_sf1000 pixi run bash bench/sf1000-repro/run.sh
```

---

## What produces the number

Seven changes, each measured in isolation on this machine.

| # | Change | Effect | Where |
|---|---|---|---|
| 1 | `expression_evaluator_strategy = 'ast_jit'` | **−4.17% suite** (q6 −49.7%, q12 −21.9%, q14/q15 −24%, q1 −8.8%) | **config only** — `src/config.cpp:27` ships the slow `AST_INTERPRET` |
| 2 | cuDF `strings::like` backtrack skip | **q13 −36.5%** | `felipeblazing/cudf` `4a345cc` |
| 3 | q17 `LOGICAL_DELIM_GET` in `build_side_is_derived` | −6.7% suite | `sirius_plan_comparison_join.cpp` |
| 4 | `interruptible_mpmc` wake-up sentinels | −5.7% suite | `src/include/exec/interruptible_mpmc.hpp` |
| 5 | q16 count-distinct → radix-sortable label | **q16 −39.0%** | `gpu_aggregate_impl.cpp` |
| 6 | q19 OR-branch derivation + dictionary predicate pushdown | −3.0% suite, q19 −21.4% | optimizer hook + scan |
| 7 | `scan_task_batch_size` 5GB → 8GB | −1.85% (q4 −25%, q12 −18%) | config only |
| | cuDF memcpy 2 MiB threshold | q9 −5.8% | `felipeblazing/cudf` `9af88b0` |
| | cuDF groupby shmem replication | q1 ~−5% | `felipeblazing/cudf` `7375a46` |

### The single most valuable line

`SET expression_evaluator_strategy = 'ast_jit'` is **−4.17% for zero code**. Sirius already
dispatches to `cudf::compute_column_jit` (`expression_evaluator.cpp:222-226`) but defaults to the
interpreted AST walker. Nothing in the repo recorded it ever being measured.

The JIT kernel cache persists to disk (`$HOME/.cudf/$VERSION/$ARCH`, override with
`LIBCUDF_KERNEL_CACHE_PATH`), so the ~19 s of first-run NVRTC compilation is one-time per
`(expression, sm arch, nvrtc version)` and survives process restart — measured: suite-wide
first-iteration cost fell 28.99 s → 10.50 s on a warm cache.

---

## Prerequisites

- **SF1000 TPC-H parquet**, one directory per table (`lineitem/*.parquet`, …). Generate with
  `test/tpch_performance/generate_test_data.py`.
- **~256 GB GPU.** Peak usage is **253.9 GB of 256 GB**. There is under 2 GB of headroom; this
  will not run on a smaller card without lowering `scan_task_batch_size` and re-tuning.
- **~470 GB host RAM** for the host memory pool (`capacity_bytes` in the YAML).
- The pixi environment (`pixi run …`). Do not use a bare `build/release/duckdb` — see *Gotchas*.

---

## Gotchas that cost us real time

**`LD_PRELOAD`, never `LD_LIBRARY_PATH`.** The Sirius extension `.so` carries `DT_RPATH`, which
the loader searches *before* `LD_LIBRARY_PATH`. Using the latter silently loads the pixi libcudf
and your patches do nothing. Verify with `LD_DEBUG=libs` that only the custom lib initialises.

**Never pass a bare `-DCMAKE_CXX_FLAGS=` to the cuDF build.** It clobbers conda's `$CXXFLAGS`,
dropping `-isystem $CONDA_PREFIX/include`; CMake then strips rmm's include dir as "redundant"
because it is still cached in `CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES`, and 17 CXX files fail with
`rmm/cuda_stream_view.hpp: No such file`. Always append (`"$CXXFLAGS -isystem …"`).
`build-libcudf.sh` does this correctly.

**`l_shipinstruct` must stay `dictionary` in `plans/lineitem.txt`.** The decode-time predicate
pushdown resolves q19's equality against the dictionary keys. If that column is switched to
`identity`, the pushdown has nothing to answer from and silently no-ops — costing q19 ~21% and
~78 GB of GPU peak.

**A GPU-vs-CPU check on an *in-memory* DuckDB compares CPU to CPU.** Sirius rejects the scan
("requires a single-file block manager") and both arms fall back, so the outputs match trivially.
Use a file-backed DB *and* assert positively that the GPU path ran.

**Custom vs shipped libcudf are not SASS-identical.** `CMAKE_CUDA_ARCHITECTURES=NATIVE` on GB300
yields `sm_103a` only; the conda package ships sm_75/80/86/90a/**100**/120/120a with no sm_103 and
no PTX, so stock Sirius runs sm_100 SASS on this device. Measured difference is inside the ±1.2%
noise band, but **A/B any cuDF patch against a control built from the unmodified fork**, never
against conda-libcudf numbers. Use `-DCMAKE_CUDA_ARCHITECTURES=100-real` to match the prebuilt.

**Measurement noise.** Suite-level is ~±1.2%. Per query, q1/q6/q9/q13/q18/q21 hold to ±1.7%
across runs, but **q7/q8/q10/q19 swing 13–28%** even at best-of-3 — the variance is between runs
(pinning order, allocator layout), not between iterations, so more iterations will not help.
Do not read a small-query delta under ~20% as signal.

---

## Knobs that were swept and measured inert

Do not spend time retuning these on this machine:

| knob | tested | result |
|---|---|---|
| `pipeline.num_threads` | 6 / 8 / 12 | all within 0.55% |
| `hash_partition_bytes` | 16GB vs 32GB | −0.06% |
| `scan_manager.num_threads` | 18 vs 24 | −0.33% |
| `scan_task_batch_size` | 2GB | +0.10% |
| `scan_task_batch_size` | 10GB | +0.19% vs 8GB, and costs q9 2.6% |

**GPU-busy is 91–97% of wall.** Scheduling and parallelism knobs cannot help; only removing work
or raising achieved bandwidth moves the clock. That also explains why a lookahead scheduler, a
CONCAT-barrier overlap, decompression prefetching, and a BUILD_PROBE probe-side split all measured
neutral and were dropped.

---

## Validation

- **22/22 byte-identical** across every run, against a saved reference.
- **Independent oracle**: GPU vs DuckDB CPU on SF10 parquet, same binary, compared as sorted
  multisets. 22/22 — q1 differs only in printed precision (max relative difference 1.388e-16,
  below double epsilon).

Note the byte-identical check is *self-referential* — it proves the stack did not change behaviour,
not that the behaviour is correct. The DuckDB-CPU comparison is the real oracle.

---

## Status

Reproduction branch, not merge-ready. Specifically:

- The three cuDF patches live on a fork and are not proposed upstream.
- `ast_jit` is set per-run rather than as a default; two queries (q19, q22) regressed under it in
  isolation, though both recovered in combination — that interaction is unresolved.
- The project's own unit and SQLLogic suites have **not** been run against this stack.
- Nothing in the repo's tests crosses the 1M-row gate that the q16 change depends on, so that path
  is covered only by an ad-hoc GPU-vs-CPU harness.
