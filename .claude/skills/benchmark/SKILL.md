---
name: benchmark
description: >
  Run TPC-H benchmarks on Super Sirius or DuckDB CPU baseline — either the regular per-query
  timing/validation run, or the TPC-H official power run + throughput run (RF1/RF2 refresh
  functions reporting Power@Size / Throughput@Size / QphH@Size). Generate data, execute queries,
  validate results, and compare timings. Trigger when the user mentions benchmarking, TPC-H,
  performance testing, query runtimes, power run, throughput run, QphH, refresh functions, or
  wants to compare Sirius vs DuckDB speed.
disable-model-invocation: true
---

# Benchmark Runner

You are running SQL benchmarks for Sirius, a GPU-accelerated SQL query engine built on DuckDB. This skill manages data generation, benchmark execution across two engines (Super Sirius, DuckDB CPU), and result comparison.

Currently this skill covers **TPC-H only**. TPC-DS support has been removed from this skill for now and will be added back later.

## Choose the Benchmark Type

Use the request to select the workflow. Ask only if the choice remains ambiguous:

- **Regular benchmark**: per-query timing and GPU-vs-CPU validation with
  `performance_test.py` (Workflow H-B).
- **TPC-H official (power + throughput run)**: refresh functions and Power@Size /
  Throughput@Size / QphH@Size with `tpch_power_throughput.py` (Workflow H-C).

The engine, GPU compatibility, and fallback sections below apply to both.

## Two Engines

| Engine | Entry Point | Config | TPC-H Runner |
|--------|-------------|--------|--------------|
| **Super Sirius** | Normal SQL with `gpu_execution=true` | Optional for regular runs | `performance_test.py --engine gpu` (or `both`) |
| **DuckDB CPU** | Raw SQL (no GPU wrapping) | No explicit `LOAD` | `performance_test.py --engine cpu` (or `both`) |

**Config file behavior:**
- Regular runner: uses `SIRIUS_CONFIG_FILE` when supplied, otherwise Sirius defaults.
- Official power/throughput runner: requires `SIRIUS_CONFIG_FILE` pointing to a valid file.
- DuckDB CPU: the runner does not issue an explicit extension `LOAD`.

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
rg -l "Transparent execution fallback|fallback to DuckDB|falling back to DuckDB" <benchmark_dir>/sirius/q*/sirius.log

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

- **Sirius and DuckDB runners:** queries are defined in `test/tpch_performance/queries.py`; `performance_test.py` renders the scale-dependent Q11 fraction from `--scale-factor`.
- Plain SQL files at `test/tpch_performance/tpch_queries/orig/q*.sql` are kept for reference but are no longer wired into the recommended Python runner.

## Workflow H-A: Generate TPC-H Data

**Dataset generation:** Check the supplied path and plausible shared locations, including the
main checkout's `test_datasets/`, before concluding data is missing. A missing path alone does not
mean a dataset must be generated. Generation can consume substantial time and disk, especially at
large scale factors: use explicit authorization to generate it, or ask if generation is needed and
not already authorized. For DuckDB generation, resolve whether a plain or clustered dataset is
needed.

Data generation is owned by the **`/dataset-manager`** skill — the single source of truth for both formats. Prefer delegating to it rather than calling the script directly. It produces:

- **parquet** → `test_datasets/tpch_parquet_sf<SF>/` (a directory; built via `tpchgen-rs`)
- **duckdb** → `test_datasets/tpch_sf<SF>.duckdb` (a single file; built via DuckDB `dbgen()`), or `tpch_sf<SF>_sorted.duckdb` when generated with `--cluster` (tables physically sorted so native-scan row-group pruning fires on date-filtered queries)

If `/dataset-manager` is unavailable, the underlying script is:

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh <scale_factor> --format parquet|duckdb [--cluster] [--output <path>]
```

`performance_test.py` consumes either source: pass the parquet **directory** with `--data-source parquet` (default) or the `.duckdb` **file** with `--data-source duckdb`.

## Workflow H-B: Python Performance Test (regular per-query runner)

`performance_test.py` runs the 22 TPC-H queries against a parquet **or** duckdb dataset (`--data-source`) for either or both engines, writes per-query timings/results/logs into a structured benchmark directory, and supports pinning tables into the Sirius cache.

```bash
export SIRIUS_CONFIG_FILE=/path/to/config.yaml
pixi run python test/tpch_performance/performance_test.py \
    --input <parquet_dir> --scale-factor <SF> [options]
```

**Required:**
- `--input <path>` — the dataset. A **parquet directory** (with `--data-source parquet`, default) or a single **`.duckdb` file** (with `--data-source duckdb`).
- `--scale-factor <SF>` — the dataset scale used to render Q11 as `0.0001 / SF`. Always pass it explicitly; omission defaults to SF1 with a warning only for rollout compatibility.

For the available flags and defaults, see `performance_test.py --help` and
`test/tpch_performance/README.md`. The Python harness's `--data-source` differs from the legacy
shell `benchmark_and_validate.sh --data-source`; the README also describes the shell-only
`--pinning-mode` and per-query pin column lists.

**Examples:**
```bash
# SF1, both engines, 2 iterations (parquet directory)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --scale-factor 1 \
    --iterations 2 --engine both

# Validate GPU vs CPU after timing (queries 1/3/6)
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_parquet_sf1 \
    --scale-factor 1 \
    --engine both --iterations 1 --validation --queries 1,3,6

# DuckDB source (a .duckdb FILE) from disk, both engines, validate
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_sf1.duckdb --data-source duckdb \
    --scale-factor 1 \
    --engine both --iterations 1 --validation --queries 1,3,6

# DuckDB source pinned into the GPU cache
pixi run python test/tpch_performance/performance_test.py \
    --input ~/sirius/test_datasets/tpch_sf100.duckdb --data-source duckdb \
    --scale-factor 100 \
    --engine gpu --iterations 3 --pin gpu
```

### Data Source: parquet vs duckdb (disk vs pinned)

- **parquet** (`--data-source parquet`, default): `--input` is a **directory** of TPC-H `.parquet` files (`read_parquet` → `GPU_PARQUET_SCAN`).
- **duckdb** (`--data-source duckdb`): `--input` is a single **`.duckdb` file** whose native TPC-H tables are scanned via the GPU-native `seq_scan` → `GPU_DUCKDB_NATIVE_SCAN`. Works with every runner.
- **Disk vs pinned** is controlled by `--pin`, independently of the source:
  - *from disk*: omit `--pin` (or `--pin none`) — data is read from the parquet/`.duckdb` file each scan.
  - *pinned*: `--pin gpu` or `--pin host` pre-loads the referenced columns into the Sirius cache.
- To confirm a pinned run actually hit the cache, grep `<benchmark_dir>/sirius/q<N>/sirius.log` for `using cached_split_provider` (cache hit) vs `not all the columns are pinned for this query` (fell through to disk).

### Cold-Run Benchmarking (`--profile cold`)

`performance_test.py` advises the OS to evict the dataset's local files with
`posix_fadvise(DONTNEED)` before every run on either engine. It also resets the Sirius cache
before each GPU run. Neither requires sudo.
This is a filesystem hint, so filesystems may ignore it. Pinned table data remains resident;
S3 input does not use this local-file eviction. Check the run logs and report these limits
when interpreting cold timings.

### Output Layout

A timestamped benchmark dir `<output>/tpch_<ts>_<profile>_<engine>_iter<N>[_<NAME>]/` (`default` profile when unset) containing `metadata.json` (including `scale_factor`), effective `queries/q<N>.sql`, `csv/runtimes.csv` (`engine,query,iteration,runtime_s`), per-query `<engine>/q<N>/result.txt`, and per-query `sirius/q<N>/sirius.log`. See `test/tpch_performance/README.md` for the full layout.

---

# TPC-H Official — Power & Throughput Run

## Workflow H-C: `tpch_power_throughput.py`

The official run adds the TPC-H refresh functions (RF1 insert, RF2 delete) to the query workload
and reports the spec composite metrics. Use it whenever the user asks for a power run, throughput
run, QphH, refresh functions, or a "TPC-H official"/"spec" benchmark.

- **Power run** (single stream, update set 1): optional clean pass → RF1 → 22 queries in spec
  stream-0 order (timed; feeds Power@Size) → RF2 → timed post-RF2 pass (delete mask active).
- **Throughput run** (update sets 2..N+1): N concurrent query streams (spec permutations 1..N) +
  one refresh stream running N RF1/RF2 pairs.
- `Power@Size = 3600·SF / geomean(22 query times + T_RF1 + T_RF2)`,
  `Throughput@Size = N·22·3600 / interval · SF`, `QphH@Size = sqrt(Power·Throughput)`.

**Hard requirements:**
- Data source is a file-backed `.duckdb` with native TPC-H tables, never parquet. The MVCC
  insert-delta/delete-mask path that makes RF1/RF2 visible on the GPU only works on pinned
  duckdb-native tables. `--input` is that `.duckdb` file; the runner copies it and mutates the
  copy, never the original.
- Always pinned. The runner pins all 8 tables once up front, each with only the union of the
  columns the 22 queries reference (not whole tables — `ps_comment`/`l_comment`/`o_clerk`/
  `p_comment` are never read, and pinning them does not fit at SF1000). The user only chooses the
  tier (`gpu` or `host`). An unpinned run would not show the refreshes on the GPU.
- Refresh sets come from `generate_tpch_refresh.sh` (classic dbgen `-U`): `orders.tbl.u*`,
  `lineitem.tbl.u*`, `delete.*` under `test_datasets/tpch_refresh_sf<SF>/`. Need at least
  `streams + 1` sets (set 1 = power run; sets 2..N+1 = throughput streams).
- With varied predicates, per-stream query sets come from `generate_tpch_queries.sh` (`qgen`):
  `stream<N>.sql` under `test_datasets/tpch_queries_sf<SF>/`, streams 0..N (0 = power run).

```bash
# One-time per SF: generate refresh sets (num_sets >= streams + 1)
./test/tpch_performance/generate_tpch_refresh.sh <SF> <num_sets>   # -> test_datasets/tpch_refresh_sf<SF>/

# Required for an official/spec run: per-stream query sets from qgen (streams 0..N)
./test/tpch_performance/generate_tpch_queries.sh <SF> <streams>    # -> test_datasets/tpch_queries_sf<SF>/

export SIRIUS_CONFIG_FILE=/path/to/config.yaml
pixi run python test/tpch_performance/tpch_power_throughput.py \
    --sf <SF> --input <base.duckdb> --refresh-dir <refresh_dir> \
    --mode both --pin gpu|host [--streams N] --vary-predicates --query-dir <dir>
```

**Flags** (full reference in `test/tpch_performance/README.md`): `--mode power|throughput|both`
(default `both`), `--streams N` (default: spec minimum for SF — SF1→2, SF10→3, SF30→4, SF100→5),
`--pin gpu|host` (always one of these), `--pin-compression/--no-pin-compression`
(Simpatico-compressed pins; plan dir defaults to the shipped TPC-H plans),
`--compression-plan-dir <dir>`, `--vary-predicates/--no-vary-predicates` (default fixed;
varied runs per-stream qgen parameters and rejects `--validation`), `--query-dir <dir>`,
`--validation/--no-validation` (fixed predicates only, default on; the power run's post-RF1 and
post-RF2 GPU rows are diffed against pure DuckDB in a child process after every pinned phase
completes, with the refresh functions replayed on a fresh copy — it cannot share the benchmark
process, since the host pool does not return memory to the OS on unpin. In `--mode both` the
throughput streams are validated too: the child re-runs the N throughput RF1/RF2 pairs and
snapshots the CPU results after the baseline and each commit, and every stream's GPU rows must
match one of those 2N+1 snapshots — which committed refresh state a concurrent query observed is
a scheduling accident, but it must be one of them),
`--baseline-pass/--no-baseline-pass`
(default on; adds the clean pre-refresh timing pass so delta/mask overhead is attributable),
`--query-timeout`, `--output`, `--keep-scratch-db`.

**Output** (`test/tpch_performance/output/tpch_power_<ts>_sf<SF>_s<N>/`): `metrics.json`,
`summary.txt` (per-query clean/post-RF1/post-RF2 timings + delta/mask overhead + the three
metrics), `timings.csv`, per-query `power/q<N>/result_*.txt`, `log_dir/`. A **non-zero exit** means
a validation mismatch or an unexpected row-count after a refresh — treat it as a failure.

---

# Before Running the Regular Benchmark (`performance_test.py`)

Use parameters supplied by the user and applicable runner defaults. Check that the dataset path,
format, and scale factor match; the scale factor controls Q11's threshold. For GPU runs, use the
provided config or a valid `SIRIUS_CONFIG_FILE`. Ask only for information that cannot be determined
and materially affects the run. If an input is missing, search for an existing dataset before
considering generation (Workflow H-A).

Choose the requested execution profile, pin tier, compression plan, iteration count, queries,
validation, and output location. Preserve the runner's constraints: CPU-only runs cannot pin;
compression requires a GPU or host pin; validation requires both engines or supplied `--duckdb-results`; nsys and gdb external
runners reject `--profile`. Verify that the extension is built for GPU runs. For comparisons, match the dataset, scale factor, queries, config, cache and
pin settings, and execution shape unless the difference is the subject of the comparison.
Report the exact invocation and the effective config path (or the source config path if no
derived config was written) with the results.

Inspect per-query Sirius logs for fallback and errors before describing timings as GPU results;
check the cache-hit messages when pinning matters (see the sections above). A populated results
file alone does not establish GPU execution.

---

# Before Running the Official Run (`tpch_power_throughput.py`)

For an explicit official/spec run, use `--vary-predicates` with qgen-generated query streams;
the runner's fixed-predicate default is not spec-compliant. If streams are missing, resolve
generation authorization before running. For diagnostic runs, honor the requested predicate mode.

Use supplied parameters and applicable runner defaults. Verify the base input is a file-backed
DuckDB dataset at the requested scale factor, and use a valid supplied config or
`SIRIUS_CONFIG_FILE`. Check existing shared locations for the base dataset, refresh sets, and,
when varying predicates, query streams. Generation can be expensive; use explicit authorization
or ask before generating missing data. Ensure refresh data has at least `streams + 1` sets and
varied query data has streams 0 through N.

Use the requested mode, pin tier, compression, streams, predicates, validation, and output
settings. The run always pins; host pinning may be needed at large scale to avoid GPU OOM.
Varied predicates require qgen query streams and cannot be validated by this runner; report that
such a run is timing-only. Fixed-predicate validation adds a base database copy and an untimed
CPU pass. Ask when an unresolved choice materially changes cost or the intended measurement.
For comparisons, match the dataset, scale factor, mode, streams, predicates, pinning, config,
and other execution settings except the variable under test. Report the exact invocation and
effective config path with the results.
In `--mode both`, validation also captures throughput results and compares them with the
post-power baseline and each throughput RF1/RF2 committed state (2N+1 CPU snapshots).
This checks membership in a collected state, not visibility during the query's transaction.
Throughput-only mode skips this validation with a warning.


Verify that the Sirius extension is built. Refresh and query generation also need `tpch-dbgen`;
the generation scripts can build it from `test_datasets/tpch-dbgen.zip`. Inspect Sirius logs for
fallback and errors before presenting results as GPU execution.
