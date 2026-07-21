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

**Generation gate — ALWAYS ask before generating.** Before generating anything (via `/dataset-manager` or the script below), ask the user whether the dataset already exists and where. A missing path is not proof the dataset doesn't exist — datasets are typically shared across worktrees (e.g. the main checkout's `test_datasets/`, not the current worktree's) or kept elsewhere on disk, so the user may already have it at a location you haven't checked. Generate only after the user explicitly confirms no existing dataset is available.

Data generation is owned by the **`/dataset-manager`** skill — the single source of truth for both formats. Prefer delegating to it rather than calling the script directly. It produces:

- **parquet** → `test_datasets/tpch_parquet_sf<SF>/` (a directory; built via `tpchgen-rs`)
- **duckdb** → `test_datasets/tpch_sf<SF>.duckdb` (a single file; built via DuckDB `dbgen()`), or `tpch_sf<SF>_sorted.duckdb` when generated with `--cluster` (tables physically sorted so native-scan row-group pruning fires on date-filtered queries)

If `/dataset-manager` is unavailable, the underlying script is:

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh <scale_factor> --format parquet|duckdb [--cluster] [--output <path>]
```

`performance_test.py` consumes either source: pass the parquet **directory** with `--data-source parquet` (default) or the `.duckdb` **file** with `--data-source duckdb`.

## Workflow H-B: Python Performance Test (only supported TPC-H runner)

`performance_test.py` runs the 22 TPC-H queries against a parquet **or** duckdb dataset (`--data-source`) for either or both engines, writes per-query timings/results/logs into a structured benchmark directory, and supports pinning tables into the Sirius cache.

```bash
export SIRIUS_CONFIG_FILE=/path/to/config.yaml
pixi run python test/tpch_performance/performance_test.py \
    --input <parquet_dir> [options]
```

**Required:**
- `--input <path>` — the dataset. A **parquet directory** (with `--data-source parquet`, default) or a single **`.duckdb` file** (with `--data-source duckdb`).

**Most-used flags** — the complete reference lives in `test/tpch_performance/CLAUDE.md` and `performance_test.py --help`; consult it rather than re-deriving flags here:
- `--data-source {parquet,duckdb}` (default `parquet`) — input source/format. `parquet`: `--input` is a parquet directory (`read_parquet` scan). `duckdb`: `--input` is a single `.duckdb` file (GPU-native `seq_scan`). Works in all modes; `--pin` works for both.
- `--engine {gpu,cpu,both}` (default `both`) — `gpu`/`both` load the Sirius extension; `cpu` does not.
- `--iterations <N>` (default `1`) and `--queries 1,3,6-10` (default: all 22).
- `--mode {grouped,sequential,isolated,nsys-profile}` (default `grouped`) — `grouped` is hot-cache; `isolated` is true cold-start (needs the sudo setup below); `nsys-profile` is GPU-only and owned by the `profile-analyzer` skill.
- `--pin {none,gpu,host}` (default `none`) — Sirius cache pre-load tier; rejected with `--engine cpu`.
- `--validation` — byte-compare GPU vs CPU results after timing (`abs_tol=1e-10` on floats); requires `--engine both`.
- `--name <NAME>` — label the output subdir instead of the default `tpch_<ts>_<mode>_<engine>_iter<N>`.

The CLAUDE.md is the source of truth for the rest of the surface (`--config`, `--output`, `--duckdb-profiling`, `--query-timeout`), the per-query pin column lists in `tpch_pin_columns.py`, and the shell-runner-only `--pinning-mode` path. (Note: `performance_test.py --data-source` is the harness's own 2-value flag, distinct from the legacy shell `benchmark_and_validate.sh --data-source`.)

**Examples:**
```bash
# SF1, hot cache, both engines, 2 iterations (parquet directory)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --iterations 2 --mode grouped --engine both

# Validate GPU vs CPU after timing (queries 1/3/6)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --engine both --iterations 1 --validation --queries 1,3,6

# DuckDB source (a .duckdb FILE) from disk, both engines, validate
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_sf1.duckdb --data-source duckdb \
    --engine both --iterations 1 --validation --queries 1,3,6

# DuckDB source pinned into the GPU cache
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_sf100.duckdb --data-source duckdb \
    --engine gpu --iterations 3 --pin gpu
```

### Data Source: parquet vs duckdb (disk vs pinned)

- **parquet** (`--data-source parquet`, default): `--input` is a **directory** of TPC-H `.parquet` files (`read_parquet` → `GPU_PARQUET_SCAN`).
- **duckdb** (`--data-source duckdb`): `--input` is a single **`.duckdb` file** whose native TPC-H tables are scanned via the GPU-native `seq_scan` → `GPU_DUCKDB_NATIVE_SCAN`. Works in all modes (incl. `nsys-profile`).
- **Disk vs pinned** is controlled by `--pin`, independently of the source:
  - *from disk*: omit `--pin` (or `--pin none`) — data is read from the parquet/`.duckdb` file each scan.
  - *pinned*: `--pin gpu` or `--pin host` pre-loads the referenced columns into the Sirius cache.
- To confirm a pinned run actually hit the cache, grep `<benchmark_dir>/sirius/q<N>/sirius.log` for `using cached_split_provider` (cache hit) vs `not all the columns are pinned for this query` (fell through to disk).

### Cold-Run Benchmarking (`--mode isolated`)

`drop_os_cache()` writes `3` to `/proc/sys/vm/drop_caches` via passwordless `sudo`. Before running `--mode isolated`, ask the user to run this one-time setup (do not proceed until the user confirms it is done):

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop_caches
```

### Output Layout

A timestamped benchmark dir `<output>/tpch_<ts>_<mode>_<engine>_iter<N>/` (or `<output>/<NAME>/`) containing `metadata.json`, `csv/runtimes.csv` (`engine,query,iteration,runtime_s`), per-query `<engine>/q<N>/result.txt`, and per-query `sirius/q<N>/sirius.log`. See `test/tpch_performance/CLAUDE.md` for the full layout.

---

# Before Running — ALWAYS Confirm Every Parameter First

**This is mandatory and must not be skipped.** Before invoking `performance_test.py`, you MUST confirm **every** parameter with the user via the `AskUserQuestion` tool — never silently assume defaults, and never skip this gate even when the request looks complete. Pre-fill any value the user already gave (present it as the recommended option) and ask for the rest, so a `/benchmark` run never starts with an unconfirmed configuration.

Ask in ~3 grouped rounds (`AskUserQuestion` allows up to 4 questions per call). Mark sensible defaults as "(Recommended)".

**Round 1 — data & engine**
1. **Data source** (`--data-source`) — `parquet` or `duckdb`.
2. **Dataset** (`--input`) — the parquet **directory** or `.duckdb` **file** path. ALWAYS ask the user where the dataset lives — never assume a path, and never treat a missing path as "needs generating": the dataset may already exist at a different location (see the generation gate in Workflow H-A). If the given path doesn't exist, report that and ask for the correct location; offer generation via `/dataset-manager` only after the user confirms no existing dataset is available (generation can take significant time and disk at large scale factors — **never auto-generate**). For duckdb, clarify plain vs `--cluster` (sorted) if generating.
3. **Config** (`--config` / `SIRIUS_CONFIG_FILE`) — which Sirius config YAML to use (required for any GPU engine). Do not guess the path; confirm it.
4. **Engine** (`--engine`) — `gpu`, `cpu`, or `both` (default `both`).

**Round 2 — execution shape**
1. **Mode** (`--mode`) — `grouped` (default, hot cache), `sequential` (round-robin), `isolated` (true cold-start; needs the sudo setup below), or `nsys-profile` (GPU-only; owned by the `profile-analyzer` skill).
2. **Pin** (`--pin`) — `none` (from disk), `gpu`, or `host` (pinned cache tier; rejected with `--engine cpu`). This is how "from disk vs pinned" is chosen, for either source.
3. **Iterations** (`--iterations`) — per-query iteration count (default `1`).
4. **Queries** (`--queries`) — all 22 (default) or a subset like `1,3,6-10`.

**Round 3 — output**
1. **Validation** (`--validation`) — byte-compare GPU vs CPU after timing (requires `--engine both`).
2. **Name** (`--name`) — optional label for the output subdir (e.g. `baseline`, `sf1000-host-pin`); if declined, omit it and let the default `tpch_<ts>_<mode>_<engine>_iter<N>` be used. Do not invent a name unprompted.

After confirming, echo the final `performance_test.py` command back to the user before running it.

**Environment prerequisites** (verify these yourself; they are not user questions):
- Sirius extension built: `pixi run -e clang make release` (the runner loads `build/release/extension/sirius/sirius.duckdb_extension` for any GPU engine).
- For Super Sirius: `SIRIUS_CONFIG_FILE` set, or `--config <yaml>` passed (confirmed in Round 1).
