# TPC-H Performance Testing

This directory contains benchmarking, profiling, and performance testing tools for comparing DuckDB (CPU) vs Sirius (GPU) on TPC-H queries at various scale factors.

## Prerequisites

- Sirius must be built: `pixi run make -j12` (from project root)
- Binary: `build/release/duckdb` with Sirius extension at `build/release/extension/sirius/sirius.duckdb_extension`
- Sirius config: `test/cpp/integration/integration.yaml` (set `SIRIUS_CONFIG_FILE` env var)
- Parquet data must exist in `test_datasets/tpch_parquet_sf<N>/` (auto-generated if missing)

## Generating Test Data

### Using generate_tpch_data.sh (recommended)

Clones and builds `sirius-db/tpchgen-rs` from source with native CPU optimizations, then generates partitioned parquet files with optimized row groups. Called automatically by `run_tpch_parquet.sh` when data is missing.

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

### Full DuckDB vs Sirius benchmark with validation (recommended)

`benchmark_and_validate.sh` runs all 22 TPC-H queries for both Sirius and DuckDB, compares results for correctness, and produces a timestamped run directory with comprehensive output.

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml

./test/tpch_performance/benchmark_and_validate.sh <scale_factor>
# Example:
./test/tpch_performance/benchmark_and_validate.sh 100
```

Each run creates a directory under `runs/<timestamp>_sf<SF>_2iter/` containing:
- `run_info.txt` — git branch/revision, tree clean/dirty, build freshness, hostname, memory, CPUs, GPUs, filesystem read benchmark, pinning_mode setting
- `run_info.patch` — full git diff when tree is dirty
- `sirius_config.yaml` — copy of the Sirius config used
- `sirius/` and `duckdb/` — per-engine logs, per-query results and timings
- `validation.csv` — per-query match/error status
- `comparison.txt` — cold/warm timing table with speedup ratios
- `timings.csv` — long-format iteration runtimes (engine,query,iteration,runtime_s)

#### `--data-source parquet | duckdb | duckdb-native` (scan path)

`--data-source` selects which engine scan path is exercised:

| Value | Runner | Input | Sirius scan path |
|-------|--------|-------|------------------|
| `parquet` (default) | `run_tpch_parquet.sh` | `test_datasets/tpch_parquet_sf<SF>/` or `--parquet-dir` | `read_parquet` → `GPU_PARQUET_SCAN` |
| `duckdb` | `run_tpch_duckdb.sh` | `performance_test.duckdb` or `--duckdb-file` | `seq_scan` → `GPU_DUCKDB_NATIVE_SCAN` (the engine default) |
| `duckdb-native` | `run_tpch_duckdb.sh` | same `.duckdb` file as `duckdb` | alias of `duckdb` — kept for compatibility |

The **GPU-native DuckDB scan is the only `seq_scan` path** in the engine, so the `duckdb` data source routes `seq_scan` to `GPU_DUCKDB_NATIVE_SCAN` via `insert_duckdb_native_scan_operator` (`src/pipeline/sirius_pipeline_converter.cpp`). `duckdb-native` and the `--gpu-native-scan` flag are redundant no-op aliases. The `duckdb` engine remains the unchanged DuckDB CPU baseline (it runs with `SIRIUS_DISABLE=1`), so `validation.csv` validates GPU-native-scan output against DuckDB CPU.

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

> **Note:** `--pinning-mode per-query` is parquet-only — `run_tpch_duckdb.sh` does not accept `--pinning-mode`, so do not combine it with `--data-source duckdb` or `duckdb-native`. The Python harness (`performance_test.py`) is also parquet-only and does not support the native scan.

#### `--pinning-mode per-query` (PR #721 pin_table)

When passed `--pinning-mode per-query`, the Sirius engine wraps each query block with `CALL pin_table(<glob>, tier='gpu', name=<table>, cols=[...])` for every table the query reads, runs the query for `--iterations` runs back-to-back, then `CALL unpin_table(<table>)` for each pinned table. This isolates per-query pinning cost from query execution: the query-iteration timings written to `timings.csv` reflect query-only time on the pinned-cache scan path.

The per-query column-set is sourced from `tpch_pin_columns.py` (must be a superset of every column the query references, otherwise the scan falls through to disk). The pin path is a glob whose `FileSystem::GlobFiles` expansion must equal the file list of the corresponding `CREATE VIEW … read_parquet([…])` — otherwise `sirius_scan_manager::create_provider_for` will not match and the cache is silently bypassed.

```bash
./test/tpch_performance/benchmark_and_validate.sh --pinning-mode per-query 100
```

The DuckDB engine ignores the flag (pinning is Sirius-only) and runs as the unchanged baseline. Pinning time is **not** measured — markers (`__TPCH_PIN_BEGIN__`, `__TPCH_UNPIN_BEGIN__`, …) keep pin/unpin `Run Time` lines outside the iteration-time parser window. In single-session mode, every pinned table is unpinned at the end of its query block before the next query's pin calls run — no carry-over even when consecutive queries reference the same table. In multi-session mode the unpin still runs before each per-query process exits, defensively releasing GPU memory back to the allocator.

To verify a query actually hit the cache, grep `runs/.../sirius/q<N>/sirius.log` for `using cached_split_provider`; the matching-fallback log line is `not all the columns are pinned for this query`.

Tier override: the helper defaults to `tier='gpu'` — the only tier currently implemented in `src/sirius_extension.cpp`. **`tier='host'` is not supported right now and will be added later**: setting `SIRIUS_PIN_TIER=host` today makes the emitted `CALL pin_table` throw `NotImplementedException` at bind time (`src/sirius_extension.cpp:681-683`), and queries fall through to disk reads. Once host-tier support lands, flip via `SIRIUS_PIN_TIER=host`.

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

Environment variables:
- `SIRIUS_CONFIG_FILE` — path to Sirius config (required for sirius engine)
- `TIMING_CSV` — path to write per-query timing CSV (optional)
- `OUTPUT_DIR` — directory for structured output (set by `benchmark_and_validate.sh`)

### DuckDB-only baseline

```bash
./test/tpch_performance/run_tpch_parquet_duckdb.sh <scale_factor> <query_numbers...>
./test/tpch_performance/run_tpch_parquet_duckdb.sh --parquet-dir /data/tpch 100 1 3 6
```

### Thread configuration sweep

Runs Sirius-only across multiple thread configurations (pipeline, scan, task_creator threads) to find optimal settings. Modifies `integration.yaml` during the run and restores baseline when done.

```bash
bash test/tpch_performance/sweep_threads.sh
```

Results are saved to `benchmark_results_thread_sweep/` as CSV files per configuration.

### Python-based performance test (in-memory database)

Loads data into a DuckDB database, runs all 22 queries with both CPU and GPU, verifies results match:

```bash
pixi run python test/tpch_performance/performance_test.py <scale_factor>
```

## Profiling with Nsight Systems

A suite of scripts for GPU performance profiling and analysis using NVIDIA Nsight Systems (nsys).

### Profiling queries

`profile_tpch_nsys.sh` runs each query in its own DuckDB process wrapped by nsys, producing per-query `.nsys-rep` and `.sqlite` files.

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml

# Profile all queries at SF300 with 2M row groups
./test/tpch_performance/profile_tpch_nsys.sh 300_rg2m

# Profile specific queries with custom timeout
QUERY_TIMEOUT=120 ./test/tpch_performance/profile_tpch_nsys.sh 100 1 3 6 9
```

Output is saved to `nsys_profiles/sf<SF>/` with per-query `.nsys-rep`, `.sqlite`, result, and timing files.

Environment variables: `DUCKDB`, `PARQUET_DIR`, `QUERY_DIR`, `OUTPUT_DIR`, `QUERY_TIMEOUT`, `ITERATIONS`.

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
# Profile and generate report
./test/tpch_performance/nsys_report.sh --sf 300_rg2m
./test/tpch_performance/nsys_report.sh --sf 100 --iterations 4 1 3 6 10

# Report from existing profiles
./test/tpch_performance/nsys_report.sh --profile-dir /path/to/nsys_profiles/sf300/

# Report with baseline comparison
./test/tpch_performance/nsys_report.sh --profile-dir ./profiles/ --compare reports/baseline/
```

Output: `reports/<label>_<YYYYMMDD_HHMMSS>/` containing `report.md`, `summary.json`, `metadata.json`, and `profiles/`.

## Query Files

- `tpch_queries/orig/q*.sql` — Plain SQL queries used by both Sirius and DuckDB runners

## Key Files

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
| `tpch_pin_columns.py` | Per-query column → table mapping for `--pinning-mode per-query`; emits `CALL pin_table(...)` / `CALL unpin_table(...)` SQL |
| `generate_test_data.py` | Generate test data via dbgen |
| `generate_test_data_tpchgen-rs.py` | Generate test data via tpchgen-rs Python wrapper + query files |
| `pixi.toml` | Python environment with cudf, pyarrow, rust for tooling |

## Sirius Configuration

The Sirius config file (`test/cpp/integration/integration.yaml`) controls:
- **GPU memory**: `usage_limit_fraction`, `reservation_limit_fraction`
- **Host memory**: `capacity_bytes`, `initial_number_pools`, `pool_size`, `block_size`
  - Initial allocation = `initial_number_pools * pool_size * block_size`
- **Thread pools**: `pipeline`, `duckdb_scan`, `task_creator`, `downgrade` thread counts
- **Scan cache**: `duckdb_scan.cache` controls cache level (default: `none`, valid: `none`, `parquet`, `table_host`, `table_gpu`)
  - In single-session benchmarks, the config YAML controls the cache level directly
  - In multi-session benchmarks, per-query overrides can be set in `scan_cache_levels.yaml`
- **Cold-run benchmarking**: Use `--multi-session --drop-os-cache` to drop OS filesystem cache between queries. Requires one-time passwordless sudo setup:
  ```bash
  echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop_caches
  ```

## Parquet Format Notes

- DuckDB's default export creates 122,880-row row groups (its internal vector size)
- For GPU workloads, 2M-10M row groups perform significantly better
- The `rewrite_parquet.py` script preserves the original schema (date32, decimal128) to avoid type mismatch issues with Sirius
- cudf internally promotes date32 to timestamp; the rewriter casts back before writing
- Large tables are split into multiple numbered files when exceeding the max file size limit
