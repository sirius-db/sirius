---
name: benchmark
description: >
  Run TPC-H or TPC-DS benchmarks on Super Sirius or DuckDB CPU baseline — generate data, execute
  queries, validate results, and compare timings. Trigger when the user mentions benchmarking,
  TPC-DS, TPC-H, performance testing, query runtimes, or wants to compare Sirius vs DuckDB speed.
disable-model-invocation: true
---

# Benchmark Runner

You are running SQL benchmarks for Sirius, a GPU-accelerated SQL query engine built on DuckDB. This skill manages data generation, benchmark execution across two engines (Super Sirius, DuckDB CPU), and result comparison. Each benchmark suite is documented in its own section below — to add a new benchmark in the future, add a new section following the same pattern (tables, query files, data generation, workflows).

## Supported Benchmarks

| Benchmark | Queries | Tables | Scripts Directory | Best For |
|-----------|---------|--------|-------------------|----------|
| **TPC-H** | 22 | 8 | `test/tpch_performance/` | I/O-heavy joins, aggregations, standard analytics |
| **TPC-DS** | 99 | 24 | `test/tpcds_performance/` | Broad SQL coverage (joins, aggregates, subqueries, window functions) |

**Ask the user which benchmark to run if not specified.**

## Two Engines

| Engine | Entry Point | Config | TPC-H Scripts | TPC-DS Scripts |
|--------|-------------|--------|---------------|----------------|
| **Super Sirius** | `gpu_execution("...")` | **Requires** `SIRIUS_CONFIG_FILE` | `benchmark_and_validate.sh` / `run_tpch_parquet.sh` | `benchmark_and_validate.sh` / `run_tpcds_super.sh` |
| **DuckDB CPU** | Raw SQL (no GPU wrapping) | Unsets `SIRIUS_CONFIG_FILE` | `benchmark_and_validate.sh` / `run_tpch_parquet.sh` | `benchmark_and_validate.sh` / `run_tpcds_duckdb.sh` |

**Config file behavior:**
- Super Sirius: `SIRIUS_CONFIG_FILE` must be set and point to a valid file — the scripts error if not
- DuckDB CPU: `SIRIUS_CONFIG_FILE` is automatically unset — pure CPU baseline

## Working Directory

All commands run from the **project root**. TPC-H scripts are in `test/tpch_performance/`. TPC-DS scripts are in `test/tpcds_performance/`.

## GPU-Compatible Queries

A query is **GPU-compatible** when it meets both criteria:
1. **Executes completely on Super Sirius** — every operator in the query plan runs on the GPU without falling back to DuckDB's CPU engine
2. **Produces correct results** — the GPU output matches DuckDB CPU output exactly

- **TPC-H:** All 22 queries (1-22) are GPU-compatible
- **TPC-DS:** 15 queries are GPU-compatible: 3, 7, 22, 26, 32, 37, 42, 52, 55, 62, 82, 85, 92, 93, 97

The TPC-DS GPU-compatible list grows as more queries gain full GPU support.

## Identifying GPU Fallback and Errors

When a query cannot run on GPU, Super Sirius produces distinct error messages (see `src/sirius_extension.cpp`). A query is considered **failing** if any of these conditions are true:
1. The log contains a fallback or error message (query ran on CPU, not GPU)
2. The timer output is missing (`-1` in timings.csv / `NO_TIMER` status) — indicates the query crashed or hung
3. The timings.csv shows `FAILED` status

**Super Sirius (`gpu_execution`) messages:**
- `"Error in SiriusGeneratePhysicalPlan: <message>"` — plan generation failed; falls back if `ENABLE_DUCKDB_FALLBACK` is true
- `"Error in SiriusExecuteQuery, fallback to DuckDB"` — plan or execution error with fallback enabled
- `"SiriusExecuteQuery error: <message>"` — thrown when fallback is disabled

**How to detect failures in benchmark logs:**
```bash
# Queries that fell back to DuckDB CPU
grep -l "fallback to DuckDB" <output_dir>/log_q*.txt

# Plan generation errors
grep -l "Error in SiriusGeneratePhysicalPlan" <output_dir>/log_q*.txt

# Execution errors
grep -l "Error in SiriusExecuteQuery" <output_dir>/log_q*.txt

# Queries that crashed or hung (no timer output)
grep "NO_TIMER\|FAILED" <output_dir>/timings.csv

# Queries that ran successfully on GPU
for log in <output_dir>/log_q*.txt; do
    if ! grep -q "fallback to DuckDB\|Error in.*PhysicalPlan\|Error in.*ExecuteQuery" "$log"; then
        echo "$log"
    fi
done
```

---

# TPC-H Benchmarks

Scripts are in `test/tpch_performance/`.

## TPC-H Tables

8 tables: `customer`, `lineitem`, `nation`, `orders`, `part`, `partsupp`, `region`, `supplier`.

## TPC-H Query Files

- **Sirius (GPU):** `test/tpch_performance/tpch_queries/gpu/q*.sql` — wrapped with `call gpu_execution('...');`
- **DuckDB (CPU):** `test/tpch_performance/tpch_queries/orig/q*.sql` — plain SQL

## Workflow H-A: Generate TPC-H Data

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh <scale_factor> [output_dir] [jobs]
```

- Builds `sirius-db/tpchgen-rs` from source, generates partitioned parquet files to `test_datasets/tpch_parquet_sf<SF>/`
- **Auto-called** by `run_tpch_parquet.sh` if data is missing — no need to run separately in most cases

## Workflow H-B: Full Benchmark with Validation (Recommended)

Runs all 22 queries for both Sirius and DuckDB, compares results for correctness, and produces a timestamped run directory.

```bash
export SIRIUS_CONFIG_FILE=/path/to/config.cfg
./test/tpch_performance/benchmark_and_validate.sh <scale_factor>
```

**Options:**
- `--config <file>` — Override Sirius config file
- `--parquet-dir <path>` — Custom parquet data location (default: `test_datasets/tpch_parquet_sf<SF>/`)
- `--engines <engines>` — Space-separated list (default: `"sirius duckdb"`)
- `--iterations <N>` — Iterations per query (default: 2 = 1 cold + 1 warm)
- `--timeout <seconds>` — Session timeout (default: 1200)
- `--multi-session` — Run each query in its own DuckDB process (fresh state per query, useful for DuckDB baselines)
- `--duckdb-results <run_dir>` — Reuse previously stored DuckDB results (skip re-running DuckDB)
- `--report <run_dir>` — Regenerate reports from existing run directory (no benchmarks run)

**Examples:**
```bash
# Basic run at SF100
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
./test/tpch_performance/benchmark_and_validate.sh 100

# 3 iterations, 5-minute timeout
./test/tpch_performance/benchmark_and_validate.sh --config ~/my.cfg --iterations 3 --timeout 300 100

# Reuse prior DuckDB results
./test/tpch_performance/benchmark_and_validate.sh --duckdb-results runs/2026-03-10_12-00-00_sf100_2iter 100

# Regenerate report from existing run
./test/tpch_performance/benchmark_and_validate.sh --report runs/2026-03-10_12-00-00_sf100_2iter
```

**Output:** `runs/<timestamp>_sf<SF>_<N>iter/`
- `run_info.txt` — git, hardware, build info
- `sirius/` and `duckdb/` — per-engine: `q<N>/result.txt`, `q<N>/timings.csv`, `run.log`
- `validation.csv` — per-query match/error status
- `comparison.txt` — cold/warm timing table with speedup ratios
- `timings.csv` — long-format: `engine,query,iteration,runtime_s`

## Workflow H-C: Run Individual Queries

For running specific queries or a single engine without the full orchestrator.

```bash
export SIRIUS_CONFIG_FILE=/path/to/config.cfg
./test/tpch_performance/run_tpch_parquet.sh [options] <engine> <scale_factor> <query_numbers...>
```

**Options:** `--parquet-dir`, `--iterations`, `--timeout`, `--cache-level`, `--multi-session`

**Examples:**
```bash
./test/tpch_performance/run_tpch_parquet.sh sirius 100 $(seq 1 22)
./test/tpch_performance/run_tpch_parquet.sh duckdb 100 1 3 6
./test/tpch_performance/run_tpch_parquet.sh --multi-session duckdb 100 $(seq 1 22)
./test/tpch_performance/run_tpch_parquet.sh --iterations 5 --parquet-dir /data/tpch sirius 100 $(seq 1 22)
```

---

# TPC-DS Benchmarks

Scripts are in `test/tpcds_performance/`.

## TPC-DS Tables

All 24 tables: `call_center`, `catalog_page`, `catalog_returns`, `catalog_sales`, `customer`, `customer_address`, `customer_demographics`, `date_dim`, `household_demographics`, `income_band`, `inventory`, `item`, `promotion`, `reason`, `ship_mode`, `store`, `store_returns`, `store_sales`, `time_dim`, `warehouse`, `web_page`, `web_returns`, `web_sales`, `web_site`.

## TPC-DS Query Files

Located at `test/tpcds_performance/queries/q1.sql` through `q99.sql`. Generated automatically by `generate_tpcds_data.sh` using DuckDB's `tpcds_queries()` function.

## Workflow DS-A: Generate TPC-DS Data

```bash
bash test/tpcds_performance/generate_tpcds_data.sh <scale_factor> [--format parquet|duckdb] [--output <path>]
```

- Default format is `duckdb`; use `--format parquet` for Super Sirius
- Output: `test_datasets/tpcds_sf<SF>.duckdb` or `test_datasets/tpcds_parquet_sf<SF>/`
- Also generates query files `q1.sql` through `q99.sql`

**Examples:**
```bash
bash test/tpcds_performance/generate_tpcds_data.sh 1
bash test/tpcds_performance/generate_tpcds_data.sh 10 --format parquet
```

## Workflow DS-B: Full Benchmark with Validation (Recommended)

Runs TPC-DS queries for both Super Sirius and DuckDB, compares results for correctness, and produces a timestamped run directory.

**Ask the user which query set to run:**

| Option | Flag | Queries | Description |
|--------|------|---------|-------------|
| **GPU-compatible only** | `--gpu-only` | 15 queries | Only queries that execute completely on Super Sirius and produce correct results matching DuckDB CPU. All should succeed. This list grows as more queries gain GPU support. |
| **All 99 queries** | *(default)* | 99 queries | Runs every TPC-DS query. Some will fall back to DuckDB CPU or error — useful for discovering which queries work and measuring coverage. |

```bash
export SIRIUS_CONFIG_FILE=/path/to/config.cfg

# GPU-compatible queries only (recommended for benchmarking)
./test/tpcds_performance/benchmark_and_validate.sh --gpu-only <scale_factor>

# All 99 queries (some may fall back to CPU or error)
./test/tpcds_performance/benchmark_and_validate.sh <scale_factor>
```

**Options:**
- `--config <file>` — Override Sirius config file
- `--parquet-dir <path>` — Custom parquet data location (default: `test_datasets/tpcds_parquet_sf<SF>/`)
- `--engines <engines>` — Space-separated list (default: `"sirius duckdb"`)
- `--gpu-only` — Run only the GPU-compatible query subset
- `--queries <N...> --` — Specific query numbers (use `--` before scale_factor)
- `--duckdb-results <run_dir>` — Reuse previously stored DuckDB results
- `--report <run_dir>` — Regenerate reports from existing run directory

**Examples:**
```bash
# GPU-compatible queries at SF1
export SIRIUS_CONFIG_FILE=~/sirius_config.cfg
./test/tpcds_performance/benchmark_and_validate.sh --gpu-only 1

# All 99 queries at SF10
./test/tpcds_performance/benchmark_and_validate.sh 10

# Specific queries
./test/tpcds_performance/benchmark_and_validate.sh --queries 3 7 42 -- 1

# Reuse prior DuckDB results
./test/tpcds_performance/benchmark_and_validate.sh --duckdb-results runs/tpcds_2026-04-05_sf1 --gpu-only 1

# Regenerate report from existing run
./test/tpcds_performance/benchmark_and_validate.sh --report runs/tpcds_2026-04-05_sf1
```

**Output:** `runs/tpcds_<timestamp>_sf<SF>/`
- `run_info.txt` — git, hardware, build info
- `sirius/` and `duckdb/` — per-engine: `result_q<N>.txt`, `log_q<N>.txt`, `timings.csv`, `run.log`
- `validation.csv` — per-query match/error status (`success` = results match, `validation` = diff found, `error` = query failed)
- `comparison.txt` — cold/warm timing table with speedup ratios
- `timings.csv` — long-format: `engine,query,iteration,runtime_s`

## Workflow DS-C: Run Individual Engine

For running a single engine without the full orchestrator.

**Super Sirius:**
```bash
export SIRIUS_CONFIG_FILE=/path/to/config.cfg
bash test/tpcds_performance/run_tpcds_super.sh <parquet_dir> [--queries N...] [--output-dir path]
bash test/tpcds_performance/run_tpcds_super_gpu.sh <parquet_dir> [--output-dir path]  # GPU-compatible only
```

**DuckDB CPU:**
```bash
bash test/tpcds_performance/run_tpcds_duckdb.sh --parquet-dir <path> [--queries N...] [--output-dir path]
bash test/tpcds_performance/run_tpcds_duckdb.sh --db <path> [--queries N...] [--output-dir path]
```

## Updating the GPU-Compatible Query List

To discover which TPC-DS queries are GPU-compatible:
1. Run all 99 queries: `./test/tpcds_performance/benchmark_and_validate.sh <SF>`
2. Check `validation.csv` — queries with `success` status executed on GPU with correct results
3. Cross-check logs for fallback: `grep -l "fallback to DuckDB" <run_dir>/sirius/log_q*.txt`
4. Queries that show `success` in validation AND have no fallback in logs are GPU-compatible
5. Update the `GPU_QUERIES` array in both `run_tpcds_super_gpu.sh` and `benchmark_and_validate.sh`

---

# Before Running

- **Ask the user** for any paths you don't know. Do NOT assume paths.
- **Ask which benchmark** (TPC-H or TPC-DS) if not specified.
- **For TPC-DS**, ask which query set: GPU-compatible only (queries that fully execute on GPU with correct results) or all 99 queries (some may fall back to CPU or error).
- **Ask about data**: Does the dataset already exist? If yes, ask for the parquet directory path. If no, confirm the user wants to generate it before proceeding — data generation can take significant time and disk space at large scale factors. Do NOT auto-generate without asking.
- Ensure the DuckDB binary is built: `pixi run -e clang make release`
- For Super Sirius: ensure `SIRIUS_CONFIG_FILE` is set
