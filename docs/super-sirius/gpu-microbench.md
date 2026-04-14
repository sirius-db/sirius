# GPU microbench (`sirius_gpu_microbench`)

libcudf-focused Google Benchmark binaries live next to the extension tests. They time hash join, group-by sum, sort keys, boolean filter, and an optional Parquet column read.

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

| Benchmark            | Args / meaning |
|----------------------|----------------|
| `BM_HashJoin`        | `build_rows`, `probe_rows` |
| `BM_GroupBySum`      | `rows`, `num_groups` (NDV) |
| `BM_SortKeys`        | `rows` |
| `BM_FilterMask`      | `rows`, `permille_true` (approximate selectivity ×1000) |
| `BM_ParquetReadColumn` | `max_rows` |

### Optional Parquet read

```bash
export SIRIUS_MICROBENCH_PARQUET_FILE=/path/to/file.parquet
export SIRIUS_MICROBENCH_PARQUET_COLUMN=column_name
```

The `full` profile runs all benchmarks (`--benchmark_filter=.*`), including `BM_ParquetReadColumn`; without the env vars that case may skip or error in the report.

## CI

Workflow: `.github/workflows/gpu-microbench.yml`

- **Schedule:** Mon–Sat `daily` profile; Sunday `weekly` profile (UTC).
- **Manual:** `workflow_dispatch` with profile choice.
- **Artifacts:** JSON under `runs/microbench/ci_<run_id>_<profile>/` (30-day retention).
- **Summary:** Markdown table of benchmark real time written to the job summary.

## Interpreting artifacts

Download the artifact zip and open `benchmark.json`. Google Benchmark's schema lists one object per row in `benchmarks` with `name`, `real_time`, `cpu_time`, `iterations`, and `time_unit`. Compare runs by matching `name` strings (they include argument packs).
