# GPU microbench (`sirius_gpu_microbench`)

libcudf-focused Google Benchmark binaries live next to the extension tests. They time hash join, group-by sum, sort keys, boolean filter, and an optional **full-table** Parquet read with **cold** vs **warm** OS page cache.

**Timing:** all benchmarks use **wall-clock (real) time** in **milliseconds** (`UseRealTime`, `Unit(kMillisecond)`), not CPU time.

## Build

From the repo root (Pixi environment):

```bash
pixi run build
# or: pixi run make
```

Incremental rebuild of only the microbench target:

```bash
pixi run build-microbench
```

The binary is written to:

`build/release/extension/sirius/test/cpp/sirius_gpu_microbench`

Enable/disable is controlled by CMake (`SIRIUS_BUILD_GPU_MICROBENCH`); the release preset turns it on.

## Run locally

```bash
pixi run microbench -- --benchmark_format=json
pixi run microbench -- --benchmark_filter='BM_HashJoin|BM_GroupBySum'
```

Or invoke the binary directly:

```bash
./build/release/extension/sirius/test/cpp/sirius_gpu_microbench --benchmark_format=json
```

### Sweep profiles (JSON)

`tools/microbench/microbench_sweep.json` defines `daily`, `weekly`, and `full` profiles (filter regex + extra Google Benchmark flags). Run (Pixi):

```bash
pixi run microbench-daily
pixi run microbench-sweep weekly
pixi run microbench-sweep full
```

Or run the scripts directly (ensure they are executable):

```bash
./tools/microbench/run_microbench_daily.sh
./tools/microbench/run_microbench_sweep.sh weekly
```

Outputs default to `runs/microbench/<UTC-timestamp>_<profile>/benchmark.json`. Override with `SIRIUS_MICROBENCH_OUT`, `SIRIUS_MICROBENCH_RUN_DIR`, or `SIRIUS_GPU_MICROBENCH_BIN`.

YAML is not parsed by the scripts; if you prefer YAML for hand-editing, mirror the same structure as the JSON `profiles` map.

### Benchmark parameters (P0)

| Benchmark | Args / meaning |
|-----------|----------------|
| `BM_HashJoin` | `build_rows`, `probe_rows` |
| `BM_GroupBySum` | `rows`, `num_groups` (NDV) |
| `BM_SortKeys` | `rows` |
| `BM_FilterMask` | `rows`, `permille_true` (approximate selectivity ×1000) |
| `BM_ParquetReadTable_Cold` | none — full file, all columns; **fadvise** excluded from timed section |
| `BM_ParquetReadTable_Warm` | none — full file, all columns; no cache drop |

### Optional Parquet table read

Prepare TPC-H SF1 Parquet (same as CI / `test.yml`):

```bash
./setup_test_datasets.sh
mkdir -p test_datasets/tpch_parquet_sf1
./build/release/duckdb -f scripts/tpch_to_parquet.sql
export SIRIUS_MICROBENCH_PARQUET_FILE=$PWD/test_datasets/tpch_parquet_sf1/lineitem.parquet
```

- **Cold:** before each timed read, the benchmark calls `posix_fadvise(..., POSIX_FADV_DONTNEED)` on the file **outside** the timed section (`PauseTiming` / `ResumeTiming`) to encourage dropping clean page-cache pages (Linux only; best-effort).
- **Warm:** no fadvise; measures reads with whatever pages the kernel keeps cached.

The `full` sweep profile runs all benchmarks, including the Parquet pair, when `SIRIUS_MICROBENCH_PARQUET_FILE` is set (CI sets it to `lineitem.parquet`).

## CI

Workflow: `.github/workflows/gpu-microbench.yml`

- **Data (full profile only):** `setup_test_datasets.sh` + `scripts/tpch_to_parquet.sql` → `test_datasets/tpch_parquet_sf1/lineitem.parquet`.
- **Env:** for `full`, `SIRIUS_MICROBENCH_PARQUET_FILE` is set to that `lineitem.parquet` path before the sweep script runs (`daily` / `weekly` skip dataset prep and omit the env var).
- **Schedule:** Mon–Sat `daily` profile; Sunday `weekly` profile (UTC).
- **Manual:** `workflow_dispatch` with profile choice.
- **Artifacts:** JSON under `runs/microbench/ci_<run_id>_<profile>/` (30-day retention).
- **Summary:** Markdown table of benchmark **real** time written to the job summary.

## Interpreting artifacts

Download the artifact zip and open `benchmark.json`. With `UseRealTime` and millisecond units, `real_time` and `time_unit` are the primary wall-clock metrics. `cpu_time` may still appear in the schema; prefer **`real_time`** for these benchmarks.
