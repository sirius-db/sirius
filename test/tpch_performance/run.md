# TPC-H Performance Testing

Benchmarking tools for comparing DuckDB (CPU) vs Sirius (GPU) on TPC-H queries at various scale factors.

## Prerequisites

1. Build the project:
   ```bash
   pixi run make -j12
   ```

2. Ensure a Sirius config file exists. The binary looks for config in this order:
   1. `SIRIUS_CONFIG_FILE` environment variable (explicit path)
   2. `./sirius.yaml` in the current working directory
   3. `~/.sirius/sirius.yaml` in the user's home directory

3. Ensure parquet data exists (auto-generated if missing via `generate_tpch_data.sh`).
   - On the GB300 machine, the SF1000 dataset is at `/home/nvidia/tpch_parquet_sf1000`.

For `generate_tpch_data.sh --format duckdb`, keys use `INTEGER` until their full
dataset range requires `BIGINT`, matching tpchgen-rs's Parquet defaults. Each key
family widens across all tables together: order keys at integer SF358, part keys
at SF10,738, customer keys at SF14,317, and supplier keys at SF214,749. Nation and
region keys stay `INTEGER`. Clustered DuckDB datasets use the same widths.
Regenerate existing datasets to apply the new types.

## Generating local Iceberg tables

Generate the reference TPC-H Parquet data first, then write an Iceberg V2 warehouse:

```bash
pixi run bash test/tpch_performance/generate_tpch_data.sh 1 --format parquet
pixi run python test/tpch_performance/generate_tpch_iceberg.py \
  test_datasets/tpch_parquet_sf1 test_datasets/tpch_iceberg_sf1
```

An existing Parquet dataset can be passed directly to the second command. All eight
tables are required. Supported layouts are `table.parquet`, `table/*.parquet`,
`table_*.parquet`, or `table.N.parquet`; mixing layouts for one table is rejected.
The output must be a new directory. An interrupted run leaves partial output for
inspection; use a fresh directory when retrying.

The writer uses the existing Python and PyArrow in the Pixi development
environment. PyIceberg's `sql-sqlite` extra supplies SQLAlchemy for the local
SQLite catalog. Writing the tables needs no GPU or Sirius build.

Each table gets its own `data/` and `metadata/` directories. PyIceberg rewrites
the input into Parquet with Iceberg field IDs, preserving the source column types
and values, including dates, decimals and adaptive key widths. The warehouse owns
the new files. Source Parquet files are unchanged and are no longer needed to read
the Iceberg tables. `--batch-rows` (default 1,000,000) controls rows per write batch;
each batch is appended as a snapshot. The final snapshot includes all batches.

`dataset.json` records the source files, row counts, writer versions, metadata
paths and final snapshot IDs. `create_views.sql` loads the DuckDB Iceberg extension
and creates the eight TPC-H views with exact metadata paths and pinned snapshots:

```sql
.read test_datasets/tpch_iceberg_sf1/create_views.sql
SELECT sum(l_extendedprice * l_discount) FROM lineitem;
```

Use this in the DuckDB shell after loading Sirius; `iceberg` must already be installed
for that DuckDB build. Generated paths are absolute: regenerate if the warehouse moves.
Use the Iceberg benchmark below to compare results and timings.

## Running Benchmarks

All commands run from the **project root** directory.

### Full benchmark with validation (recommended)

`benchmark_and_validate.sh` runs all 22 TPC-H queries, compares Sirius vs DuckDB results for correctness, and produces a timestamped run directory.

```bash
# Basic usage (uses ~/.sirius/sirius.yaml, both engines, dataset from test_datasets/)
./test/tpch_performance/benchmark_and_validate.sh 100

# With explicit options
./test/tpch_performance/benchmark_and_validate.sh \
  --config ~/.sirius/sirius.yaml \
  --parquet-dir /path/to/tpch_parquet_sf1000 \
  --engines "sirius duckdb" \
  1000

# Sirius only (skip DuckDB baseline)
./test/tpch_performance/benchmark_and_validate.sh \
  --config ~/.sirius/sirius.yaml \
  --parquet-dir /path/to/tpch_parquet_sf1000 \
  --engines sirius \
  1000

# Regenerate report from an existing run
./test/tpch_performance/benchmark_and_validate.sh --report runs/<run_dir>
```

Run the same benchmark on local Iceberg tables with:

```bash
pixi run bash test/tpch_performance/benchmark_and_validate.sh --data-source iceberg 1
```

This uses `test_datasets/tpch_iceberg_sf1` by default. Pass `--iceberg-dir` to reuse
a different generated warehouse. If the warehouse does not exist, the benchmark
prepares the source Parquet and generates Iceberg once before running either engine.
`--parquet-dir` selects the source Parquet for generation. Use a fresh output path
if a previous generation left an incomplete warehouse.

Both engines use the same saved `create_views.sql` with fixed snapshot IDs. As with
Parquet, each engine uses one session by default and Sirius allows CPU fallback.
The DuckDB Iceberg extension must already be installed for that build.

Options:
- `--config <path>` — Sirius config file (default: `~/.sirius/sirius.yaml`)
- `--data-source <source>` — `parquet` (default), `iceberg`, `duckdb`, or `duckdb-native`
- `--parquet-dir <path>` — parquet dataset directory (default: `test_datasets/tpch_parquet_sf<SF>`)
- `--iceberg-dir <path>` — Iceberg warehouse directory (default: `test_datasets/tpch_iceberg_sf<SF>`)
- `--engines <list>` — space-separated engine list (default: `"sirius duckdb"`)
- `--pinning-mode none|per-query|pinned-hot` — Sirius-only parquet pinning mode. `per-query` pins each query's referenced columns around that query block; `pinned-hot` pins the union of referenced columns once before the single-session run and unpins after all queries.

Each run creates a directory under `runs/<timestamp>_sf<SF>_2iter/` containing:
- `run_info.txt` — git branch/revision, tree clean/dirty, build freshness, hostname, memory, CPUs, GPUs, filesystem read benchmark
- `run_info.patch` — full git diff when tree is dirty
- `sirius_config.yaml` — copy of the Sirius config used
- `sirius/` and `duckdb/` — per-engine logs, per-query results and timings
- `validation.csv` — per-query match/error status
- `comparison.txt` — cold/warm timing table with speedup ratios
- `timings.csv` — long-format iteration runtimes (engine,query,iteration,runtime_s)

**Note:** Both engines use `sirius-duckdb/build/release/duckdb`. The Sirius runner loads the extension explicitly; the DuckDB baseline leaves it unloaded and runs on CPU.

### Running individual queries

`run_tpch_parquet.sh` is the core runner used by all benchmarks. It runs queries in a single DuckDB session with 2 iterations each (cold + warm, back-to-back) and auto-generates missing datasets.

```bash
# Run Sirius on queries 1-22
SIRIUS_CONFIG_FILE=~/.sirius/sirius.yaml \
  ./test/tpch_performance/run_tpch_parquet.sh sirius 100 $(seq 1 22)

# Run DuckDB baseline (no config needed)
./test/tpch_performance/run_tpch_parquet.sh duckdb 100 $(seq 1 22)

# Run specific queries with custom parquet directory
SIRIUS_CONFIG_FILE=~/.sirius/sirius.yaml \
  ./test/tpch_performance/run_tpch_parquet.sh --parquet-dir /data/tpch sirius 100 1 3 6

# Run Sirius with union-pinned hot cache across the query stream
SIRIUS_CONFIG_FILE=~/.sirius/sirius.yaml \
  ./test/tpch_performance/run_tpch_parquet.sh --pinning-mode pinned-hot --iterations 5 sirius 100 $(seq 1 22)
```

Environment variables:
- `SIRIUS_CONFIG_FILE` — path to Sirius config (required for sirius engine; unset automatically for duckdb engine)
- `TIMING_CSV` — path to write per-query timing CSV (optional)
- `OUTPUT_DIR` — directory for structured output (set by `benchmark_and_validate.sh`)

### Generating telemetry

Telemetry is controlled by the Sirius YAML config used for the run. Enable Quent
export and choose the output directory. NVTX capture is off by default; set
`enable_nvtx: true` to include NVTX ranges in the Quent output:

```yaml
sirius:
  telemetry:
    enable_quent: true
    enable_nvtx: true
    output_directory: telemetry_data
    engine_name: siriusDB
```

`run_tpch_parquet_and_generate_telemetry.sh` runs TPC-H queries in Sirius,
labels each `(query, iteration)` pair with `sirius_set_query_label`, and writes
Quent NDJSON files to `sirius.telemetry.output_directory`.

```bash
pixi run -- ./test/tpch_performance/run_tpch_parquet_and_generate_telemetry.sh \
  --iterations 1 \
  --parquet-dir /data/tpch/sf100/p16/zstd-8/ \
  100
```

The final `100` is the TPC-H scale factor. If no query numbers are provided,
all 22 queries are run; append query numbers to limit the run, for example
`100 1 6 9`.

The script uses `test/tpch_performance/tpch_telemetry_sirius.yaml` by default.
That config only enables telemetry, so pass `--config <path>` when the workload
also needs custom memory, executor, scan-cache, or operator settings:

```bash
pixi run -- ./test/tpch_performance/run_tpch_parquet_and_generate_telemetry.sh \
  --config ~/.sirius/sirius.yaml \
  --iterations 1 \
  --parquet-dir /data/tpch/sf100/p16/zstd-8/ \
  100 1 6 9
```

The custom config is used as-is, so it must include
`sirius.telemetry.enable_quent: true`.

Query labels are optional but make the Quent UI easier to navigate. They can be
set in either of two ways:

```sql
-- Applies to the next Sirius query, including transparent plain-SQL execution.
CALL sirius_set_query_label('tpch_q1_iter1');
SELECT *
FROM lineitem
WHERE l_orderkey < 100;

-- Inline label for an explicit gpu_execution call.
CALL gpu_execution(
  'SELECT * FROM lineitem WHERE l_orderkey < 100',
  query_label = 'tpch_q1_iter1'
);
```

The telemetry helper script uses `sirius_set_query_label` so plain SQL queries
keep the same execution path as the normal TPC-H runner.

Start the Quent analyzer server over the telemetry directory:

```bash
pixi run quent
```

The `quent` Pixi task defaults to `telemetry_data` and runs the telemetry server
with the UI enabled. If the config writes telemetry somewhere else, pass that
path as the task argument:

```bash
pixi run quent /path/to/telemetry_data
```

Open `http://localhost:8080` and select the captured Sirius engine/query.

## Query Files

- `tpch_queries/orig/q*.sql` — Plain SQL queries used by both Sirius and DuckDB runners

## Sirius Configuration

The Sirius config file (e.g. `~/.sirius/sirius.yaml`) controls:
- **GPU memory**: `usage_limit_fraction`, `reservation_limit_fraction`, `downgrade_trigger_fraction`, `downgrade_stop_fraction`
- **Host memory**: `capacity_bytes`, `initial_number_pools`, `pool_size`, `block_size`
  - Initial allocation = `initial_number_pools * pool_size * block_size`
- **Thread pools**: `pipeline`, `task_creator`, `downgrade` thread counts
- **Operator params**: `scan_task_batch_size`, `hash_partition_bytes`, `concat_batch_bytes`
- **Telemetry**: `telemetry.enable_quent`, `telemetry.output_directory`, `telemetry.engine_name`

### Example config (GB300, SF1000)

```yaml
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.9
      reservation_limit_fraction: 1.0
      downgrade_trigger_fraction: 0.8
      downgrade_stop_fraction: 0.6
    host:
      capacity_bytes: 471200000000       # ~471 GB
      initial_number_pools: 785
      pool_size: 512
      block_size: 1048576                # 1 MB
  executor:
    pipeline:
      num_threads: 4
    downgrade:
      num_threads: 1
    task_creator:
      num_threads: 6
  operator_params:
    scan_task_batch_size: 5368709120       # 5 GB
    max_sort_partition_bytes: 0            # 0 = auto (33% GPU memory)
    hash_partition_bytes: 5368709120       # 5 GB
    concat_batch_bytes: 5368709120         # 5 GB
  telemetry:
    enable_quent: true
    output_directory: telemetry_data
    engine_name: siriusDB
```
