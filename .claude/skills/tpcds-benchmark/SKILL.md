---
name: tpcds-benchmark
description: Run TPC-DS benchmarks on Legacy Sirius, Super Sirius, or DuckDB CPU baseline — generate data, execute queries, and compare results across engines.
---

# TPC-DS Benchmark Runner

You are running TPC-DS benchmarks for Sirius, a GPU-accelerated SQL query engine built on DuckDB. This skill manages data generation, benchmark execution across three engines (Legacy Sirius, Super Sirius, DuckDB CPU), and result comparison.

## Three Engines

| Engine | Script | Entry Point | Config |
|--------|--------|-------------|--------|
| **Legacy Sirius** | `run_tpcds_legacy.sh` / `run_tpcds_legacy_gpu.sh` | `gpu_buffer_init` + `gpu_processing("...")` | Unsets `SIRIUS_CONFIG_FILE` |
| **Super Sirius** | `run_tpcds_super.sh` / `run_tpcds_super_gpu.sh` | `gpu_execution("...")` | **Requires** `SIRIUS_CONFIG_FILE` |
| **DuckDB CPU** | `run_tpcds_duckdb.sh` | Raw SQL (no GPU wrapping) | Unsets `SIRIUS_CONFIG_FILE` |

**Config file behavior:**
- Legacy Sirius: `SIRIUS_CONFIG_FILE` is automatically unset — it interferes with `gpu_buffer_init`
- Super Sirius: `SIRIUS_CONFIG_FILE` must be set and point to a valid file — the script errors if not
- DuckDB CPU: `SIRIUS_CONFIG_FILE` is automatically unset — pure CPU baseline

## Working Directory

All commands run from the **project root**. Scripts are in `test/tpcds_performance/`.

## TPC-DS Tables

All 24 TPC-DS tables: `call_center`, `catalog_page`, `catalog_returns`, `catalog_sales`, `customer`, `customer_address`, `customer_demographics`, `date_dim`, `household_demographics`, `income_band`, `inventory`, `item`, `promotion`, `reason`, `ship_mode`, `store`, `store_returns`, `store_sales`, `time_dim`, `warehouse`, `web_page`, `web_returns`, `web_sales`, `web_site`.

## Identifying GPU Fallback and Errors

When a query cannot run on GPU, each engine produces distinct error messages (see `src/sirius_extension.cpp`). A query is considered **failing** if any of these conditions are true:
1. The log contains a fallback or error message (query ran on CPU, not GPU)
2. The timer output is missing (`-1` in timings.csv / `NO_TIMER` status) — indicates the query crashed or hung
3. The timings.csv shows `FAILED` status

**Legacy Sirius (`gpu_processing`) messages:**
- `"Error in GPUGeneratePhysicalPlan: <message>"` — logged when plan generation fails; sets `plan_error = true`, always falls back
- `"Error in GPUExecuteQuery, fallback to DuckDB"` — printed to stdout when:
  - `GPUBufferManager` not initialized (missing `gpu_buffer_init`)
  - Plan generation failed (`plan_error = true`)
  - Execution error (`GPUExecuteQuery` returned an error)
- Legacy **always** falls back silently to DuckDB CPU — it never throws to the caller

**Super Sirius (`gpu_execution`) messages:**
- `"Error in SiriusGeneratePhysicalPlan: <message>"` — logged when plan generation fails. If `ENABLE_DUCKDB_FALLBACK` is true, sets `plan_error = true` and falls back. If false, throws `std::runtime_error`.
- `"Error in SiriusExecuteQuery, fallback to DuckDB"` — printed to stdout when plan or execution error occurs AND fallback is enabled
- `"SiriusExecuteQuery error: <message>"` — thrown as `std::runtime_error` when `ENABLE_DUCKDB_FALLBACK` is false

**How to detect failures in benchmark logs:**
```bash
# Check which queries fell back to DuckDB (ran on CPU, not GPU)
grep -l "fallback to DuckDB" <output_dir>/log_q*.txt

# Check for plan generation errors
grep -l "Error in GPUGeneratePhysicalPlan" <output_dir>/log_q*.txt      # Legacy
grep -l "Error in SiriusGeneratePhysicalPlan" <output_dir>/log_q*.txt   # Super

# Check for execution errors
grep -l "Error in GPUExecuteQuery" <output_dir>/log_q*.txt              # Legacy
grep -l "Error in SiriusExecuteQuery" <output_dir>/log_q*.txt           # Super

# Check for queries that failed to produce timer output (crash/hang)
grep "NO_TIMER\|FAILED" <output_dir>/timings.csv

# Find queries that ran successfully on GPU (no fallback, no errors, timer present)
for log in <output_dir>/log_q*.txt; do
    if ! grep -q "fallback to DuckDB\|Error in.*PhysicalPlan\|Error in.*ExecuteQuery" "$log"; then
        echo "$log"
    fi
done
```

A query is **GPU-clean** only if its log contains NO fallback/error messages AND its timings.csv entry shows `OK` status for both runs.

## Workflows

### Workflow A: Generate TPC-DS Data

```bash
bash test/tpcds_performance/generate_tpcds_data.sh <scale_factor> [--format parquet|duckdb]
```

- Default format is `duckdb` (creates a `.duckdb` database file)
- Use `--format parquet` for parquet files (needed by Super Sirius and DuckDB parquet mode)
- Output location: `test_datasets/tpcds_sf<SF>.duckdb` or `test_datasets/tpcds_parquet_sf<SF>/`
- Also generates query files in `test/tpcds_performance/queries/q1.sql` through `q99.sql`

Examples:
```bash
# Generate SF1 DuckDB database
bash test/tpcds_performance/generate_tpcds_data.sh 1

# Generate SF10 parquet files
bash test/tpcds_performance/generate_tpcds_data.sh 10 --format parquet
```

### Workflow B: Run Legacy Sirius Benchmark

Legacy Sirius uses `gpu_buffer_init` + `gpu_processing`. Config file is automatically unset.

**All 99 queries:**
```bash
bash test/tpcds_performance/run_tpcds_legacy.sh "<gpu_caching_size>" "<gpu_processing_size>" [options]
```

**GPU-only queries (22 queries):**
```bash
bash test/tpcds_performance/run_tpcds_legacy_gpu.sh "<gpu_caching_size>" "<gpu_processing_size>" [options]
```

Options:
- `--sf <N>` — Scale factor, used to locate database (default: 1)
- `--db <path>` — Override path to DuckDB database file
- `--queries <N...>` — Specific query numbers
- `--output-dir <path>` — Results directory (default: `tpcds_results_sf<SF>/`)

Examples:
```bash
# Run all queries at SF1
bash test/tpcds_performance/run_tpcds_legacy.sh "1 GB" "2 GB"

# Run GPU-only queries at SF10
bash test/tpcds_performance/run_tpcds_legacy_gpu.sh "10 GB" "20 GB" --sf 10

# Run specific queries
bash test/tpcds_performance/run_tpcds_legacy.sh "1 GB" "2 GB" --queries 3 7 42
```

### Workflow C: Run Super Sirius Benchmark

Super Sirius uses `gpu_execution`. **Requires `SIRIUS_CONFIG_FILE` to be set.**

**All 99 queries:**
```bash
export SIRIUS_CONFIG_FILE=/path/to/config.cfg
bash test/tpcds_performance/run_tpcds_super.sh <parquet_dir> [options]
```

**GPU-only queries (15 queries):**
```bash
export SIRIUS_CONFIG_FILE=/path/to/config.cfg
bash test/tpcds_performance/run_tpcds_super_gpu.sh <parquet_dir> [options]
```

Options:
- `--queries <N...>` — Specific query numbers
- `--output-dir <path>` — Results directory (default: `tpcds_super_results/`)

Examples:
```bash
# Run all queries
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
bash test/tpcds_performance/run_tpcds_super.sh test_datasets/tpcds_parquet_sf1

# Run GPU-only queries
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
bash test/tpcds_performance/run_tpcds_super_gpu.sh test_datasets/tpcds_parquet_sf10

# Run specific queries with custom output
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
bash test/tpcds_performance/run_tpcds_super.sh /data/tpcds_parquet_sf100 --queries 3 7 --output-dir /results/super_sf100
```

### Workflow D: Run DuckDB CPU Baseline

Pure DuckDB execution — no Sirius, no GPU. Supports both parquet and DuckDB database sources.

**From parquet files:**
```bash
bash test/tpcds_performance/run_tpcds_duckdb.sh --parquet-dir <path> [options]
```

**From DuckDB database:**
```bash
bash test/tpcds_performance/run_tpcds_duckdb.sh --db <path> [options]
```

Options:
- `--queries <N...>` — Specific query numbers (default: all 1-99)
- `--output-dir <path>` — Results directory (default: `tpcds_duckdb_results/`)

Examples:
```bash
# Run all queries from parquet
bash test/tpcds_performance/run_tpcds_duckdb.sh --parquet-dir test_datasets/tpcds_parquet_sf1

# Run specific queries from database
bash test/tpcds_performance/run_tpcds_duckdb.sh --db test_datasets/tpcds_sf1.duckdb --queries 3 7 42

# Custom output directory
bash test/tpcds_performance/run_tpcds_duckdb.sh --parquet-dir /data/tpcds_parquet_sf10 --output-dir /results/duckdb_sf10
```

### Workflow E: Compare Results

Compare timing results across engines. Each script produces a `timings.csv` with the same format:
```
query,run1_time,run1_status,run2_time,run2_status
```

**To compare, read the timings.csv files from each engine's output directory and build a comparison table:**

```bash
# Example: compare Super Sirius vs DuckDB CPU
# 1. Run both benchmarks (same queries, same data)
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
bash test/tpcds_performance/run_tpcds_super.sh test_datasets/tpcds_parquet_sf1 --queries 3 7 42 --output-dir results_super/
bash test/tpcds_performance/run_tpcds_duckdb.sh --parquet-dir test_datasets/tpcds_parquet_sf1 --queries 3 7 42 --output-dir results_duckdb/

# 2. Compare timings.csv files
# Claude will read both files and produce a comparison table with speedup ratios
```

**Comparison output format:**
| Query | DuckDB Cold | DuckDB Warm | Sirius Cold | Sirius Warm | Speedup (Warm) |
|-------|-------------|-------------|-------------|-------------|----------------|

Use warm (Run 2) timings for speedup calculation since cold runs include I/O and caching overhead.

### Workflow F: Full Suite + Report

Run all three engines on the same data and produce a comprehensive report.

**Steps:**
1. Generate data (if not already present)
2. Run DuckDB CPU baseline
3. Run Legacy Sirius (if applicable)
4. Run Super Sirius (if applicable)
5. Compare all results and produce a summary report

```bash
# 1. Generate SF1 parquet data
bash test/tpcds_performance/generate_tpcds_data.sh 1 --format parquet

# 2. DuckDB baseline
bash test/tpcds_performance/run_tpcds_duckdb.sh --parquet-dir test_datasets/tpcds_parquet_sf1 --output-dir results/duckdb_sf1/

# 3. Super Sirius (GPU-compatible queries)
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
bash test/tpcds_performance/run_tpcds_super_gpu.sh test_datasets/tpcds_parquet_sf1 --output-dir results/super_sf1/

# 4. Compare
# Claude reads results/duckdb_sf1/timings.csv and results/super_sf1/timings.csv
# and produces a comparison table
```

## Output Format

All scripts produce the same output structure:
- `timings.csv` — Per-query timing data (cold + warm runs)
- `result_q<N>.txt` — Query output for each query
- `log_q<N>.txt` — Full DuckDB output including errors for each query

Each query runs twice:
- **Run 1 (cold)**: First execution — includes I/O, JIT compilation, caching overhead
- **Run 2 (warm)**: Second execution — benefits from cached data

## Before Running

- **Ask the user** for any paths you don't know. Do NOT assume paths.
- Ensure the DuckDB binary is built: `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`
- For Super Sirius: ensure `SIRIUS_CONFIG_FILE` is set
- For Legacy Sirius: the config is automatically unset, no action needed
- Query files must exist in `test/tpcds_performance/queries/` — run `generate_tpcds_data.sh` first

## GPU-Compatible Query Lists

Not all TPC-DS queries can run on GPU. The GPU-only wrapper scripts select known-good queries:

- **Legacy Sirius GPU** (`run_tpcds_legacy_gpu.sh`, 22 queries): 3, 7, 17, 25, 26, 29, 37, 42, 43, 46, 50, 52, 55, 62, 68, 69, 79, 82, 85, 91, 92, 96
- **Super Sirius GPU** (`run_tpcds_super_gpu.sh`, 15 queries): 3, 7, 22, 26, 32, 37, 42, 52, 55, 62, 82, 85, 92, 93, 97

These lists are maintained in the respective `*_gpu.sh` wrapper scripts. To update them:
1. Run all 99 queries with the full script (e.g., `run_tpcds_super.sh`)
2. Check which queries fell back: `grep -l "fallback to DuckDB" <output_dir>/log_q*.txt`
3. Check for plan generation errors: `grep -l "Error in SiriusGeneratePhysicalPlan" <output_dir>/log_q*.txt`
4. Check for missing timer output: `grep "NO_TIMER\|FAILED" <output_dir>/timings.csv`
5. Queries whose logs contain NO fallback/error messages AND show `OK` status in timings.csv ran purely on GPU
6. Update the `GPU_QUERIES` array in the wrapper script
