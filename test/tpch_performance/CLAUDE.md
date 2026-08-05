# TPC-H Performance Testing

This directory contains benchmarking, profiling, and performance testing tools for comparing DuckDB (CPU) vs Sirius (GPU) on TPC-H queries at various scale factors.

## Prerequisites

- Sirius must be built: `pixi run make -j12` (from project root)
- Binary: `build/release/duckdb` with Sirius extension at `build/release/extension/sirius/sirius.duckdb_extension`
- Sirius config: `test/cpp/integration/integration.yaml` (set `SIRIUS_CONFIG_FILE` env var)
- Parquet data must exist in `test_datasets/tpch_parquet_sf<N>/` (auto-generated if missing)

## Generating Test Data

### Using generate_tpch_data.sh (recommended)

Clones and builds `sirius-db/tpchgen-rs` from source with native CPU optimizations, then generates partitioned parquet files with optimized row groups. Run this once per scale factor before invoking `performance_test.py`.

```bash
cd test/tpch_performance

# Generate SF100 dataset (auto-detects output dir)
pixi run bash generate_tpch_data.sh 100

# Specify custom output directory and parallelism
pixi run bash generate_tpch_data.sh 100 /data/tpch_sf100 16
```

### Clustered (sorted) DuckDB datasets for zone-map pruning

`--cluster` (duckdb format only) physically sorts tables at load so each row group's
per-column min/max becomes selective. This is what makes Sirius's native-scan row-group
pruning (`mark_row_groups_pruned_by_filter_stats`) actually skip work on date-filtered
queries — on an unsorted dbgen dataset every row group spans the full date range and
nothing prunes. It writes a fresh, compact `tpch_sf<SF>_sorted.duckdb` (dbgen → staging →
sorted copy, so there is no dead-block bloat).

```bash
# Sort lineitem by l_shipdate and orders by o_orderdate (defaults)
pixi run bash generate_tpch_data.sh 10 --format duckdb --cluster
# → test_datasets/tpch_sf10_sorted.duckdb

# Override the sort keys (implies --cluster); "table:col,table:col,..."
pixi run bash generate_tpch_data.sh 10 --format duckdb \
    --cluster-keys "lineitem:l_shipdate,orders:o_orderdate"
```

Benchmark it through the unchanged native runner and validate against DuckDB CPU:

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml
./test/tpch_performance/benchmark_and_validate.sh \
    --data-source duckdb --duckdb-file ./test_datasets/tpch_sf10_sorted.duckdb 10
```

Confirm pruning fired (looking for `stats-pruned N row groups` / a reduced `decoded split`):

```bash
SIRIUS_NATIVE_SCAN_VERIFY=1 OUTPUT_DIR=/tmp/cluster_verify \
  ./test/tpch_performance/run_tpch_duckdb.sh \
    --duckdb-file ./test_datasets/tpch_sf10_sorted.duckdb sirius 10 1 6
grep -oE 'stats-pruned [0-9]+ row groups' /tmp/cluster_verify/sirius_*.log
```

> **Why duckdb-only.** Sorting works for parquet too in principle, but the default
> tpchgen-rs (Arrow) writes `DECIMAL` as `FIXED_LEN_BYTE_ARRAY`, which trips Sirius's
> `skip_pushdown_due_to_flba` and disables row-group pruning for the *whole* parquet file —
> so a sorted parquet wouldn't prune. The duckdb path stores decimals as `INT64` and prunes
> correctly. `--cluster` therefore errors if combined with `--format parquet`.

### From DuckDB's built-in TPC-H generator

```bash
# From project root - generates parquet files with DuckDB's default row groups (122K rows)
./build/release/duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=100); EXPORT DATABASE 'test_datasets/tpch_parquet_sf100' (FORMAT PARQUET);"
```

### Rewriting parquet with GPU-optimized settings

The `rewrite_parquet.py` script reads existing parquet files and rewrites them with larger row groups, snappy compression, V2 page headers, dictionary encoding, and configurable max file size (large tables are split into numbered files). Uses cudf (GPU) if available, otherwise falls back to pyarrow (CPU-only). Requires the pixi environment in this directory (`pixi install`).

```bash
cd test/tpch_performance

# Rewrite with 10M-row row groups (recommended for GPU workloads)
pixi run python rewrite_parquet.py ../../test_datasets/tpch_parquet_sf100 ../../test_datasets/tpch_parquet_sf100_optimized 10000000

# Rewrite with 2M-row row groups, 20 GB max file size
pixi run python rewrite_parquet.py ../../test_datasets/tpch_parquet_sf100 ../../test_datasets/tpch_parquet_sf100_rg2m 2000000 20
```

### From tpchgen-rs Python wrapper (alternative, supports partitioned output)

```bash
cd test/tpch_performance
pixi run python generate_test_data_tpchgen-rs.py <SF> <partitions> <format>
```

## Running Benchmarks

All commands run from the **project root** directory.

### TPC-H benchmark with performance_test.py (primary runner)

`performance_test.py` is the canonical TPC-H runner shared by the `benchmark` and `profile-analyzer` skills. With `--data-source parquet` (default) it registers TPC-H tables as parquet views over a single in-memory DuckDB connection per engine; with `--data-source duckdb` it opens a `.duckdb` file directly (read-only) and queries its native tables (GPU-native `seq_scan`). Either way it runs each query for N iterations and produces a structured per-query benchmark directory, and pinning works for both sources.

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml

# Both engines, 2 iterations, hot-cache (grouped) mode
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf100 \
    --engine both --iterations 2

# Sirius vs DuckDB result validation, queries 1/3/6 only
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --engine both --iterations 1 --validation --queries 1,3,6

# Per-query GPU-tier pinning
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf100 \
    --engine gpu --iterations 3 --pin gpu

# DuckDB native source from disk (both engines, validate). --input is a .duckdb FILE.
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_sf1.duckdb --data-source duckdb \
    --engine both --iterations 1 --validation --queries 1,3,6

# DuckDB native source, pinned into the GPU cache
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_sf100.duckdb --data-source duckdb \
    --engine gpu --iterations 3 --pin gpu

# Cold-start measurement (drops OS cache between runs; requires passwordless sudo)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf10 \
    --engine gpu --iterations 2 --mode isolated

# nsys-profile mode (one .nsys-rep + .sqlite per query under <bench>/sirius/q<N>/)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --engine gpu --iterations 2 --mode nsys-profile --queries 1,3,6
```

Key flags:
- `--data-source parquet|duckdb` — input source/format (default `parquet`). `parquet`: `--input` is a directory of TPC-H parquet files (scanned via `read_parquet` → `GPU_PARQUET_SCAN`). `duckdb`: `--input` is a single `.duckdb` file whose native tables are scanned via the GPU-native `seq_scan` → `GPU_DUCKDB_NATIVE_SCAN`. Works in all modes (incl. `nsys-profile`), and `--pin` works for both. (This is the harness's own 2-value flag — see the disambiguation note below, distinct from the legacy shell `--data-source`.)
- `--engine gpu|cpu|both` — which engine to benchmark.
- `--iterations N` — per-query iteration count.
- `--mode grouped|sequential|isolated|nsys-profile` — `grouped` (default, hot cache), `sequential` (round-robin), `isolated` (fresh connection + drop_os_cache per run; requires passwordless sudo), `nsys-profile` (see below).
- `--queries 1,3,6-10` — subset selection.
- `--pin gpu|host|none` — Sirius cache pre-load tier. Both `gpu` and `host` are supported; `host` converts the pinned table into NUMA-local pinned host memory. Any other tier throws `NotImplementedException` at bind time (`src/sirius_extension.cpp:811-813`).
- `--validation` — byte-compare GPU vs CPU `result.txt` after timing (with `abs_tol=1e-10` on float columns). Requires `--engine both`.
- `--mode nsys-profile` — wrap each query in `nsys profile` (one DuckDB CLI subprocess per query; the cudaProfilerApi capture range covers the cold + hot iterations). Requires `--engine gpu`; incompatible with `--validation` and `--duckdb-profiling`.
- `--query-timeout N` — per-query subprocess timeout in nsys-profile mode (default 90s).
- `--name <NAME>` — override the auto-timestamped benchmark subdirectory name.
- `--config <yaml>` — override `$SIRIUS_CONFIG_FILE` for this run.

#### `--data-source parquet | duckdb | duckdb-native` (shell runners — scan path)

> **Two flags, same name.** This `--data-source` belongs to the **legacy shell**
> orchestrator `benchmark_and_validate.sh` and is 3-value (`parquet`, `duckdb`,
> `duckdb-native` — the last a redundant alias of `duckdb`). The Python harness
> `performance_test.py` has its **own** 2-value `--data-source` (`parquet`,
> `duckdb`), documented above. They are independent flags on independent tools.

`benchmark_and_validate.sh --data-source` selects which engine scan path is exercised:

| Value | Runner | Input | Sirius scan path |
|-------|--------|-------|------------------|
| `parquet` (default) | `run_tpch_parquet.sh` | `test_datasets/tpch_parquet_sf<SF>/` or `--parquet-dir` | `read_parquet` → `GPU_PARQUET_SCAN` |
| `duckdb` | `run_tpch_duckdb.sh` | `performance_test.duckdb` or `--duckdb-file` | `seq_scan` → `GPU_DUCKDB_NATIVE_SCAN` (the engine default) |
| `duckdb-native` | `run_tpch_duckdb.sh` | same `.duckdb` file as `duckdb` | alias of `duckdb` — kept for compatibility |

The **GPU-native DuckDB scan is the only `seq_scan` path** in the engine, so the `duckdb` data source routes `seq_scan` to `GPU_DUCKDB_NATIVE_SCAN` via `build_duckdb_native_table_info` (`src/planner/sirius_physical_plan_generator.cpp`). `duckdb-native` and the `--gpu-native-scan` flag are redundant no-op aliases. The `duckdb` engine remains the unchanged DuckDB CPU baseline (it runs with `SIRIUS_DISABLE=1`), so `validation.csv` validates GPU-native-scan output against DuckDB CPU.

```bash
# Generate the native .duckdb tables
pixi run bash test/tpch_performance/generate_tpch_data.sh 1 --format duckdb   # → test_datasets/tpch_sf1.duckdb

# Benchmark + validate the GPU-native scan (now the default duckdb path)
./test/tpch_performance/benchmark_and_validate.sh --data-source duckdb --duckdb-file ./test_datasets/tpch_sf1.duckdb 1
```

**Verifying the native operator actually ran.** A passing result alone does not show which scan path ran. The native-scan log markers are `SIRIUS_LOG_DEBUG`, and timed runs stay at the default `info` level to keep query timings clean. To expose the markers, set `SIRIUS_NATIVE_SCAN_VERIFY=1` (it emits `SET sirius_log_level='debug'`). Run this as a separate, untimed check:

```bash
SIRIUS_NATIVE_SCAN_VERIFY=1 OUTPUT_DIR=/tmp/native_verify \
  ./test/tpch_performance/run_tpch_duckdb.sh \
    --duckdb-file ./test_datasets/tpch_sf1.duckdb sirius 1 1 6
# decoded-split count >= 1 with refused = 0 proves the native path ran:
grep -c 'duckdb_native_gpu_ingestible::materialize_table] decoded split' /tmp/native_verify/sirius_*.log
grep -c 'duckdb_native_metadata] refused'                       /tmp/native_verify/sirius_*.log   # expect 0
```

> **Note:** `--pinning-mode per-query` and `--pinning-mode pinned-hot` are parquet-only — `run_tpch_duckdb.sh` does not accept `--pinning-mode`, so do not combine either with `--data-source duckdb` or `duckdb-native`. This limitation is specific to the **shell runners**. The Python harness `performance_test.py` **does** support the duckdb native scan via its own `--data-source duckdb`, including pinning with `--pin gpu|host` (which emits `CALL pin_table(format='duckdb', name=<table>, cols=[...])`). To confirm a Python-harness duckdb run hit the pinned cache, grep the per-query log `<bench>/sirius/q<N>/sirius.log` for `using cached_split_provider` (cache hit) vs `not all the columns are pinned` (fell through to disk).

#### `--pinning-mode per-query | pinned-hot` (PR #721 pin_table)

When passed `--pinning-mode per-query`, the Sirius engine wraps each query block with `CALL pin_table(<glob>, tier='gpu', name=<table>, cols=[...])` for every table the query reads, runs the query's remaining iterations, then `CALL unpin_table(<table>)` for each pinned table. This isolates per-query pinning cost from query execution: the query-iteration timings written to `timings.csv` reflect query-only time on the pinned-cache scan path.

`--pinning-mode pinned-hot` instead pins the union of every query's referenced columns once up front (before the first query) and keeps it pinned for the whole run, unpinning only at the end. It requires the default single-session mode (rejected with `--multi-session`, since fresh DuckDB processes can't preserve the cross-query cache). **Be careful at large scale factors**: pinning everything up front means the full working set must fit in the target tier's memory at once — this is what OOM'd the SF500/SF1000 nightly runs on 2026-06-30 (union-pinned to GPU memory). The `sirius-ci` nightly benchmarks no longer use `pinned-hot` for this reason; they always use `per-query` (see below).

The per-query column-set is sourced from `tpch_pin_columns.py` (must be a superset of every column the query references, otherwise the scan falls through to disk). The pin path is a glob whose `FileSystem::GlobFiles` expansion must equal the file list of the corresponding `CREATE VIEW … read_parquet([…])` — otherwise `sirius_scan_manager::create_provider_for` will not match and the cache is silently bypassed.

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop_caches
```

Output layout (under `--output` root, default `test/tpch_performance/output/`):

`--pin-after-iteration N` (per-query mode only; ignored with pinned-hot) leaves each query's first `N` iterations (e.g. cold + warm) unpinned, then pins for the remaining ("hot") iterations before unpinning again. Default is `0` (pin from the first iteration — the original per-query behavior).

```bash
./test/tpch_performance/benchmark_and_validate.sh --pinning-mode per-query --pin-after-iteration 2 --iterations 5 100
./test/tpch_performance/benchmark_and_validate.sh --pinning-mode pinned-hot --iterations 5 100
```

To verify a query actually hit the cache, grep `runs/.../sirius/q<N>/sirius.log` for `using cached_split_provider`; the matching-fallback log line is `not all the columns are pinned for this query`.

Tier override: the helper defaults to `tier='gpu'`. Both `gpu` and `host` are supported in `src/sirius_extension.cpp`; set `SIRIUS_PIN_TIER=host` to pin into NUMA-local pinned host memory instead. Any other tier throws `NotImplementedException` at bind time (`src/sirius_extension.cpp:811-813`).

### Unified query runner

`run_tpch_parquet.sh` is the core runner used by all benchmarks. It runs all queries in a single DuckDB session with 2 iterations each (cold + warm, back-to-back) and auto-generates missing datasets.

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml

# Run Sirius on queries 1-22
./test/tpch_performance/run_tpch_parquet.sh sirius 100 $(seq 1 22)

# Run DuckDB baseline
./test/tpch_performance/run_tpch_parquet.sh duckdb 100 $(seq 1 22)

# Use custom parquet directory
./test/tpch_performance/run_tpch_parquet.sh --parquet-dir /data/tpch sirius 100 1 3 6
```
<bench>/                              # tpch_<ts>_<mode>_<engine>_iter<N>[_nsys] or --name override
  metadata.json                       # commit, branch, date, mode, iterations, engine, data_source, queries, pin, nsys_profile
  csv/runtimes.csv                    # engine,query,iteration,runtime_s
  log_dir/sirius_<YYYY-MM-DD>.log     # combined Sirius spdlog (non-profile mode)
  <engine>/q<N>/result.txt            # fetched rows, one repr(row) per line (last iter wins)
  sirius/q<N>/sirius.log              # per-query log split (non-profile mode)
  sirius/q<N>/{nsys.nsys-rep,         # nsys-profile mode only
               nsys.sqlite,
               nsys.sql, timings.csv,
               log_dir/}
```

### Thread configuration sweep

`sweep_threads.sh` runs Sirius-only across multiple thread configurations (pipeline, scan, task_creator threads) to find optimal settings. It writes each configuration to a temporary file exported through `SIRIUS_CONFIG_FILE`; the tracked `integration.yaml` is never modified.

```bash
bash test/tpch_performance/sweep_threads.sh
```

Results are saved as one CSV per configuration under a unique timestamped directory in `benchmark_results_thread_sweep/`. Set `SWEEP_OUTPUT_DIR` to use an explicit run directory.

### Legacy shell runners

The shell runners (`benchmark_and_validate.sh`, `run_tpch_parquet.sh`, `run_tpch_parquet_duckdb.sh`, `run_tpch_legacy.sh`, `profile_tpch_nsys.sh`) remain in the tree for backward compatibility with CI (`.github/workflows/test.yml`) and `.ai-helper/commands.yaml`, but are superseded by `performance_test.py`. New work — and the `benchmark` / `profile-analyzer` / `optimization-advisor` skills — should use the Python runner.

## Profiling with Nsight Systems

A suite of scripts for GPU performance profiling and analysis using NVIDIA Nsight Systems (nsys).

### Profiling queries

The primary entry point is `performance_test.py --mode nsys-profile` (subprocess-per-query, one `.nsys-rep` + `.sqlite` per query under the standard `<bench>/sirius/q<N>/` layout):

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml

pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --engine gpu --iterations 2 --mode nsys-profile --queries 1,3,6
```

For end-to-end profiling + analysis packaging, use `nsys_report.sh` (orchestrator below) — it delegates to `performance_test.py --mode nsys-profile` under the hood and flattens the per-query outputs into the report's `profiles/` directory for the analyze/compare tools.

### Analyzing profiles

`nsys_analyze.sh` extracts GPU kernel, memory transfer, NVTX operator, and I/O data from nsys-exported SQLite files.

```bash
# Analyze a single query
./test/tpch_performance/nsys_analyze.sh /path/to/q1.sqlite

# Analyze all queries in a directory
./test/tpch_performance/nsys_analyze.sh /path/to/nsys_profiles/sf300/

# Analyze specific queries
./test/tpch_performance/nsys_analyze.sh /path/to/nsys_profiles/sf300/ 1 3 6
```

### Identifying optimization targets

`nsys_hotspots.sh` maps GPU hotspots back to source code functions, detects efficiency bottlenecks, sync overhead, memory issues, and parallelism opportunities.

```bash
./test/tpch_performance/nsys_hotspots.sh /path/to/profiles/ 1 3 6
```

### Comparing runs

`nsys_compare.sh` compares per-query timings and aggregate metrics between a baseline and current report, flagging regressions and improvements.

```bash
./test/tpch_performance/nsys_compare.sh reports/baseline/ reports/current/ --threshold 5
```

### Full report generation

`nsys_report.sh` orchestrates profiling + analysis + report packaging into a self-contained report directory with human-readable markdown, machine-readable JSON, and all raw artifacts.

```bash
# Profile and generate report (parquet source, default)
./test/tpch_performance/nsys_report.sh --sf 300_rg2m
./test/tpch_performance/nsys_report.sh --sf 100 --iterations 4 1 3 6 10

# DuckDB-native source: --data-source duckdb (defaults to test_datasets/tpch_sf<SF>.duckdb,
# or pass --duckdb-file). Forwards --data-source to performance_test.py --mode nsys-profile.
./test/tpch_performance/nsys_report.sh --sf 10 --data-source duckdb 1 3 6
./test/tpch_performance/nsys_report.sh --data-source duckdb --duckdb-file ./test_datasets/tpch_sf10.duckdb --sf 10

# Report from existing profiles
./test/tpch_performance/nsys_report.sh --profile-dir /path/to/nsys_profiles/sf300/

# Report with baseline comparison
./test/tpch_performance/nsys_report.sh --profile-dir ./profiles/ --compare reports/baseline/
```

Output: `reports/<label>_<YYYYMMDD_HHMMSS>/` containing `report.md`, `summary.json`, `metadata.json`, and `profiles/`.

## Query Files

- `tpch_queries/orig/q*.sql` — Plain SQL queries used by both Sirius and DuckDB runners

## Key Files

### Primary runner and supporting modules

| File | Purpose |
|------|---------|
| `benchmark_and_validate.sh` | Full DuckDB vs Sirius benchmark with validation and timestamped runs |
| `run_tpch_parquet.sh` | Unified query runner for both engines (sirius/duckdb), single-session with cold+warm |
| `run_tpch_duckdb.sh` | Query runner over a `.duckdb` file (native tables); `--duckdb-file`; `--gpu-native-scan` is a no-op alias |
| `run_tpch_parquet_duckdb.sh` | DuckDB-only baseline runner |
| `generate_tpch_data.sh` | Generate TPC-H parquet or duckdb data via tpchgen-rs / dbgen (`--format duckdb`) |
| `sweep_threads.sh` | Thread configuration sweep (Sirius-only) |
| `profile_tpch_nsys.sh` | Profile queries with nsys, producing .nsys-rep and .sqlite per query |
| `nsys_analyze.sh` | Analyze nsys SQLite profiles (kernels, memory, NVTX, I/O) |
| `nsys_compare.sh` | Compare two nsys reports and flag regressions |
| `nsys_hotspots.sh` | Map GPU hotspots to source functions, detect bottlenecks |
| `nsys_report.sh` | Orchestrate profiling + analysis into a self-contained report |
| `rewrite_parquet.py` | Rewrite parquet with GPU-optimized row groups (cudf or pyarrow fallback) |
| `performance_test.py` | Python-based benchmark with result verification |
| `queries.py` | TPC-H query definitions (base SQL) |
| `tpch_pin_columns.py` | Per-query and union column → table mapping for `--pinning-mode per-query` / `pinned-hot` (union helpers also used by `performance_test.py --mode sequential`); emits `CALL pin_table(...)` / `CALL unpin_table(...)` SQL |
| `generate_test_data.py` | Generate test data via dbgen |
| `generate_test_data_tpchgen-rs.py` | Generate test data via tpchgen-rs Python wrapper + query files |
| `pixi.toml` | Python environment with cudf, pyarrow, rust for tooling |

## Sirius Configuration

The Sirius config file (`test/cpp/integration/integration.yaml`) controls:
- **GPU memory**: `usage_limit_fraction`, `reservation_limit_fraction`
- **Host memory**: `capacity_bytes`, `initial_number_pools`, `pool_size`, `block_size`
  - Initial allocation = `initial_number_pools * pool_size * block_size`
- **Thread pools**: `pipeline`, `task_creator`, `downgrade` thread counts
- **Cold-run benchmarking**: pass `--mode isolated` to `performance_test.py` to renew the DuckDB connection and drop OS filesystem cache before every run. Requires one-time passwordless sudo setup:
  ```bash
  echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop_caches
  ```

## Parquet Format Notes

- DuckDB's default export creates 122,880-row row groups (its internal vector size)
- For GPU workloads, 2M-10M row groups perform significantly better
- The `rewrite_parquet.py` script preserves the original schema (date32, decimal128) to avoid type mismatch issues with Sirius
- cudf internally promotes date32 to timestamp; the rewriter casts back before writing
- Large tables are split into multiple numbered files when exceeding the max file size limit
