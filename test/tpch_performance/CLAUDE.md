# TPC-H Performance Testing

This directory contains benchmarking, profiling, and performance testing tools for comparing DuckDB (CPU) vs Sirius (GPU) on TPC-H queries at various scale factors.

## Prerequisites

- Sirius must be built: `pixi run make -j12` (from project root)
- Binary: `build/release/duckdb` with Sirius extension at `build/release/extension/sirius/sirius.duckdb_extension`
- Sirius config: `test/cpp/integration/integration.cfg` (set `SIRIUS_CONFIG_FILE` env var)
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
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg

./test/tpch_performance/benchmark_and_validate.sh <scale_factor> <iterations>
# Example:
./test/tpch_performance/benchmark_and_validate.sh 100 3
```

Each run creates a directory under `runs/<timestamp>_sf<SF>_<N>iter/` containing:
- `run_info.txt` — git branch/revision, tree clean/dirty, build freshness, hostname, memory, CPUs, GPUs, filesystem read benchmark
- `run_info.patch` — full git diff when tree is dirty
- `sirius_config.cfg` — copy of the Sirius config used
- `sirius/` and `duckdb/` — per-engine logs, per-query results and timings
- `validation.csv` — per-query match/error status
- `comparison.txt` — cold/warm timing table with speedup ratios
- `timings.csv` — long-format iteration runtimes (engine,query,iteration,runtime_s)

### Unified query runner

`run_tpch_parquet.sh` is the core runner used by all benchmarks. It supports both engines, configurable iterations, and auto-generates missing datasets.

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg

# Run Sirius with 3 iterations on queries 1-22
./test/tpch_performance/run_tpch_parquet.sh sirius 100 3 $(seq 1 22)

# Run DuckDB baseline
./test/tpch_performance/run_tpch_parquet.sh duckdb 100 3 $(seq 1 22)

# Use custom parquet directory
./test/tpch_performance/run_tpch_parquet.sh --parquet-dir /data/tpch sirius 100 3 1 3 6
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

Runs Sirius-only across multiple thread configurations (pipeline, scan, task_creator threads) to find optimal settings. Modifies `integration.cfg` during the run and restores baseline when done.

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
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg

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

- `tpch_queries/orig/q*.sql` — Plain SQL queries
- `tpch_queries/gpu/q*.sql` — Queries wrapped in `call gpu_execution('...');` for Sirius

## Key Files

| File | Purpose |
|------|---------|
| `benchmark_and_validate.sh` | Full DuckDB vs Sirius benchmark with validation and timestamped runs |
| `run_tpch_parquet.sh` | Unified query runner for both engines (sirius/duckdb) with iterations |
| `run_tpch_parquet_duckdb.sh` | DuckDB-only baseline runner |
| `generate_tpch_data.sh` | Generate TPC-H parquet data via tpchgen-rs (auto-builds from source) |
| `sweep_threads.sh` | Thread configuration sweep (Sirius-only) |
| `profile_tpch_nsys.sh` | Profile queries with nsys, producing .nsys-rep and .sqlite per query |
| `nsys_analyze.sh` | Analyze nsys SQLite profiles (kernels, memory, NVTX, I/O) |
| `nsys_compare.sh` | Compare two nsys reports and flag regressions |
| `nsys_hotspots.sh` | Map GPU hotspots to source functions, detect bottlenecks |
| `nsys_report.sh` | Orchestrate profiling + analysis into a self-contained report |
| `rewrite_parquet.py` | Rewrite parquet with GPU-optimized row groups (cudf or pyarrow fallback) |
| `performance_test.py` | Python-based benchmark with result verification |
| `queries.py` | TPC-H query definitions (base SQL) |
| `generate_test_data.py` | Generate test data via dbgen |
| `generate_test_data_tpchgen-rs.py` | Generate test data via tpchgen-rs Python wrapper + query files |
| `pixi.toml` | Python environment with cudf, pyarrow, rust for tooling |

## Sirius Configuration

The Sirius config file (`test/cpp/integration/integration.cfg`) controls:
- **GPU memory**: `usage_limit_fraction`, `reservation_limit_fraction`
- **Host memory**: `capacity_bytes`, `initial_number_pools`, `pool_size`, `block_size`
  - Initial allocation = `initial_number_pools * pool_size * block_size`
- **Thread pools**: `pipeline`, `duckdb_scan`, `task_creator`, `downgrade` thread counts
- **Scan cache**: `duckdb_scan.cache = true` enables caching

## Parquet Format Notes

- DuckDB's default export creates 122,880-row row groups (its internal vector size)
- For GPU workloads, 2M-10M row groups perform significantly better
- The `rewrite_parquet.py` script preserves the original schema (date32, decimal128) to avoid type mismatch issues with Sirius
- cudf internally promotes date32 to timestamp; the rewriter casts back before writing
- Large tables are split into multiple numbered files when exceeding the max file size limit
