---
name: benchmark
description: >
  Run TPC-H benchmarks on Super Sirius or DuckDB CPU baseline — generate data, execute
  queries, validate results, and compare timings. Trigger when the user mentions benchmarking,
  TPC-H, performance testing, query runtimes, or wants to compare Sirius vs DuckDB speed.
disable-model-invocation: true
---

# Benchmark Runner

You are running SQL benchmarks for Sirius, a GPU-accelerated SQL query engine built on DuckDB. This skill manages data generation, benchmark execution across two engines (Super Sirius, DuckDB CPU), and result comparison.

Currently this skill covers **TPC-H only**. TPC-DS support has been removed from this skill for now and will be added back later.

## Two Engines

| Engine | Entry Point | Config | TPC-H Runner |
|--------|-------------|--------|--------------|
| **Super Sirius** | `gpu_execution("...")` | **Requires** `SIRIUS_CONFIG_FILE` | `performance_test.py --engine gpu` (or `both`) |
| **DuckDB CPU** | Raw SQL (no GPU wrapping) | Unsets `SIRIUS_CONFIG_FILE` | `performance_test.py --engine cpu` (or `both`) |

**Config file behavior:**
- Super Sirius: `SIRIUS_CONFIG_FILE` must be set and point to a valid file — the script errors if not
- DuckDB CPU: `SIRIUS_CONFIG_FILE` is automatically unset — pure CPU baseline

## Working Directory

All commands run from the **project root**. TPC-H scripts live in `test/tpch_performance/`.

## GPU-Compatible Queries

A query is **GPU-compatible** when it meets both criteria:
1. **Executes completely on Super Sirius** — every operator in the query plan runs on the GPU without falling back to DuckDB's CPU engine
2. **Produces correct results** — the GPU output matches DuckDB CPU output exactly

All 22 TPC-H queries (1-22) are GPU-compatible.

## Identifying GPU Fallback and Errors

When a query cannot run on GPU, Super Sirius produces distinct error messages (see `src/sirius_extension.cpp`):

- `"Error in SiriusGeneratePhysicalPlan: <message>"` — plan generation failed; falls back if `ENABLE_DUCKDB_FALLBACK` is true
- `"Error in SiriusExecuteQuery, fallback to DuckDB"` — plan or execution error with fallback enabled
- `"SiriusExecuteQuery error: <message>"` — thrown when fallback is disabled

Underlying these you'll often see a CUDA / RMM runtime error from the cuDF kernels. These strings are produced by the CUDA runtime, cuDF, RMM, or Thrust and propagate up through the Sirius error wrappers above:

- `"an illegal memory access was encountered"` / `cudaErrorIllegalAddress` — out-of-bounds device pointer, usually a kernel bug or stale buffer
- `"out of memory"` / `"rmm::bad_alloc"` / `"RMM failure"` / `cudaErrorMemoryAllocation` — GPU pool exhausted (often size-driven; try lower `usage_limit_fraction` or pin host-tier)
- `"misaligned address"` / `cudaErrorMisalignedAddress` — alignment bug in a kernel
- `"unspecified launch failure"` / `cudaErrorLaunchFailure` — kernel crashed (often follows an earlier illegal-memory access)
- `"invalid device pointer"` / `cudaErrorInvalidDevicePointer` — host pointer passed where a device pointer was expected
- `"thrust::system_error"` / `"cudaError"` / `"CUDA error:"` — generic catch-all prefixes
- `"Assertion `...` failed"` — cuDF/libcudf assertion (often a precondition violation)

These messages land in the post-split per-query Sirius logs at `<benchmark_dir>/sirius/q<N>/sirius.log`:

```bash
# Queries that fell back to DuckDB CPU
grep -l "fallback to DuckDB" <benchmark_dir>/sirius/q*/sirius.log

# Sirius-level plan / execution errors
grep -l "Error in SiriusGeneratePhysicalPlan" <benchmark_dir>/sirius/q*/sirius.log
grep -l "Error in SiriusExecuteQuery"        <benchmark_dir>/sirius/q*/sirius.log

# CUDA / RMM runtime errors (catches the common categories in one pass)
grep -lE "illegal memory access|out of memory|rmm::bad_alloc|RMM failure|misaligned address|unspecified launch failure|invalid device pointer|thrust::system_error|cudaError|CUDA error:|Assertion .* failed" \
    <benchmark_dir>/sirius/q*/sirius.log
```

A query that **falls back internally** (Sirius fallback enabled — the default) still returns rows to the Python client, so `csv/runtimes.csv` is complete and `<engine>/q<N>/result.txt` is populated; the fallback only shows up in the per-query Sirius log. A query that **errors outright** (fallback disabled, or a non-recoverable CUDA error like illegal memory access) raises a Python exception in `con.execute(...).fetchall()` and aborts the run — `csv/runtimes.csv` is truncated at the failing iteration and the per-query Sirius log captures the underlying error.

**Note on illegal memory access**: once CUDA hits `cudaErrorIllegalAddress`, the device context is poisoned and **every subsequent kernel launch fails** with the same error until the process exits. If you see "illegal memory access" repeating across queries, look at the **first** occurrence — the rest are downstream noise.

---

# TPC-H Benchmarks

Scripts are in `test/tpch_performance/`.

## TPC-H Tables

8 tables: `customer`, `lineitem`, `nation`, `orders`, `part`, `partsupp`, `region`, `supplier`.

## TPC-H Query Files

- **Sirius and DuckDB runners:** queries are defined in `test/tpch_performance/queries.py` (the `QUERIES` dict, keyed `q1`..`q22`) — imported by `performance_test.py`.
- Plain SQL files at `test/tpch_performance/tpch_queries/orig/q*.sql` are kept for reference but are no longer wired into the recommended Python runner.

## Workflow H-A: Generate TPC-H Data

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh <scale_factor> [output_dir] [jobs]
```

- Builds `sirius-db/tpchgen-rs` from source, generates partitioned parquet files to `test_datasets/tpch_parquet_sf<SF>/`.
- `performance_test.py` only accepts parquet directories (`.duckdb` files are rejected), so generate (or pre-stage) the parquet first.

## Workflow H-B: Python Performance Test (only supported TPC-H runner)

`performance_test.py` runs the 22 TPC-H queries against a parquet dataset for either or both engines, writes per-query timings/results/logs into a structured benchmark directory, and supports pinning tables into the Sirius cache.

```bash
export SIRIUS_CONFIG_FILE=/path/to/config.yaml
pixi run python test/tpch_performance/performance_test.py \
    --input <parquet_dir> [options]
```

**Required:**
- `--input <path>` — Parquet directory containing TPC-H tables (one `.parquet` file or sub-directory per table).

**Common options:**
- `--engine {gpu, cpu, both}` — default `both`. `gpu`/`both` loads the Sirius extension; `cpu` does not.
- `--iterations <N>` — default `1`.
- `--queries <spec>` — comma list with ranges, e.g. `1,3,6-10` (default: all 22).
- `--config <yaml>` — overrides `$SIRIUS_CONFIG_FILE` for the run; copied into `<benchmark_dir>/config.yml`.
- `--output <dir>` — root directory for benchmark output (default: `test/tpch_performance/output/`). A timestamped subdir is always created underneath.
- `--name <NAME>` — override the auto-generated benchmark subdirectory name. When set, output goes to `<output>/<NAME>/` instead of `<output>/tpch_<ts>_<mode>_<engine>_iter<N>/`. Useful for labeling runs (e.g. `--name baseline`, `--name with-fix`). Omit for the default timestamped name.
- `--validation` — after timing, compare the saved `<engine>/q<N>/result.txt` files for CPU and GPU. Byte-exact match first, then `abs_tol=1e-10` on float columns (strict equality on Decimal/int/string/date). Requires `--engine both`; rejected otherwise. No query re-execution (so no second Sirius extension load and no risk of GPU pool re-init OOM).

**Iteration ordering (`--mode`):**
- `grouped` (default) — per query: run all N iterations back-to-back; one connection per engine. Best for hot-cache measurement.
- `sequential` — round-robin: iter 0 runs q1, q2, …; iter 1 runs q1, q2, … Single connection per engine.
- `isolated` — renew the DuckDB connection per (query, iteration) and `drop_os_cache` before every run. True cold-start every execution. Requires the passwordless sudo setup below.

**Pin-table (Sirius cache pre-load):**
- `--pin {none, gpu, host}` — default `none`. `gpu` or `host` selects the Sirius cache tier; `none` disables pinning. Pin is per-query in `grouped`/`isolated` mode (pinned/unpinned around each query's iteration block) and a single union-pin (all referenced columns per table) at session start in `sequential` mode.
- `--pin gpu`/`--pin host` is rejected when combined with `--engine cpu` (pin is Sirius-only).
- Column lists per query live in `test/tpch_performance/tpch_pin_columns.py` (`QUERY_COLUMNS`). The runner imports `emit_pin` / `emit_unpin` / `emit_pin_all` / `emit_unpin_all` from that module.

**Examples:**
```bash
# SF1, hot cache, both engines, 2 iterations
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --iterations 2 --mode grouped --engine both

# Per-query GPU-tier pinning, 3 iterations, queries 1/3/6 only
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf100 \
    --queries 1,3,6 --iterations 3 --mode grouped --engine gpu \
    --pin gpu

# Sequential round-robin with a single host-tier union-pin
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf10 \
    --iterations 2 --mode sequential --engine gpu \
    --pin host

# Cold-start measurement (renew connection + drop OS cache per run)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf10 \
    --iterations 2 --mode isolated --engine gpu

# Validate GPU vs CPU after timing
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --iterations 1 --engine both --validation
```

### Cold-Run Benchmarking (`--mode isolated`)

`drop_os_cache()` writes `3` to `/proc/sys/vm/drop_caches` via passwordless `sudo`. Before running `--mode isolated`, ask the user to run this one-time setup (do not proceed until the user confirms it is done):

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop_caches
```

### Output Layout

`<output_dir>/tpch_<YYYYMMDD_HHMMSS>_<mode>_<engine>_iter<N>/`

- `config.yml` — copy of the Sirius config used (only when `--config` is provided).
- `metadata.json` — fields: `commit`, `branch_name`, `date`, `mode`, `iterations`, `engine`, `queries`, `pin`, `runtime_file`.
- `csv/runtimes.csv` — long-format header `engine, query, iteration, runtime_s`.
- `log_dir/sirius_YYYY-MM-DD.log` — combined Sirius spdlog daily-sink output (`SIRIUS_LOG_DIR` target).
- `<engine>/q<N>/result.txt` — fetched rows for that query, one `repr(row)` per line. Overwritten on each iteration (last iter wins).
- `sirius/q<N>/sirius.log` — per-query Sirius log split, post-processed from the combined log after the run completes (Sirius engine runs only).

---

# Before Running

- **Ask the user** for any paths you don't know. Do NOT assume paths.
- **Ask about data**: Does the parquet dataset already exist? If yes, ask for the directory path. If no, confirm the user wants to generate it before proceeding — data generation can take significant time and disk space at large scale factors. Do NOT auto-generate without asking.
- **Optionally ask for `--name`**: offer the user the option to label the output subdirectory (e.g. `baseline`, `with-fix`, `sf1000-host-pin`). If they decline or don't provide one, omit `--name` and let the default `tpch_<ts>_<mode>_<engine>_iter<N>` name be used. Do not invent a name unprompted.
- Ensure the Sirius extension is built: `pixi run -e clang make release` (the runner loads `build/release/extension/sirius/sirius.duckdb_extension` for any GPU engine).
- For Super Sirius: ensure `SIRIUS_CONFIG_FILE` is set, or pass `--config <yaml>` to `performance_test.py`.
