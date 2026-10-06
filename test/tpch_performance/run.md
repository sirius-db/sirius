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

## Running Benchmarks

All commands run from the **project root** directory.

### Read-only concurrent queries

`performance_test.py --concurrency N` runs N client streams in **one shared database and
Sirius runtime**, without refresh operations. Each stream has its own connection and at most
one outstanding query. It consumes the result before submitting the next query. The default
`--concurrency 1` retains the existing serial runner and output format.

Client concurrency and Sirius admission are independent. For example, this configuration
admits two GPU queries at a time:

```yaml
sirius:
  max_concurrent_queries: 2
```

Run four clients against it, with validation:

```bash
pixi run python test/tpch_performance/performance_test.py \
  --input test_datasets/tpch_sf10.duckdb --data-source duckdb \
  --config /path/to/sirius.yaml \
  --engine both --queries 1,6 --iterations 3 \
  --concurrency 4 --stream-order permuted --validation
```

The benchmark never changes `max_concurrent_queries`. Each engine performs
`concurrency * selected_queries * iterations` executions: 24 per engine in this example.
CPU and GPU phases run separately, at the same client concurrency. This is a read-only
throughput experiment, not an official TPC-H Power/Throughput score.

For concurrency above one, **only `--stream-order` controls query ordering**:

- `permuted` (default): deterministic TPC-H stream permutations filtered to selected queries.
  The recorded stream orders are authoritative; permutations wrap after the available table.
- `same`: every stream follows the order supplied in `--queries`.

Each stream completes a full pass before its next iteration. There are no barriers between
queries or iterations. A common start barrier follows connection setup and pinning. Up to N
requests are outstanding; the number falls as streams finish. Increasing concurrency also
increases total work, so compare queries/second and latency, not just total elapsed time.

| Option | Concurrent behavior |
|---|---|
| `--mode isolated/sequential/grouped` | Rejected when explicitly supplied. Use `--stream-order`. |
| `--profile cold` | Rejected: per-query cache reset conflicts with overlap. |
| `--profile hot/lukewarm` | Cache settings apply; their serial query ordering does not. Both retain a shared LRU cache, and neither guarantees residency under contention. |
| `--pin gpu/host/parquet` | One union pin of the selected queries' columns before timing; unpin after all streams finish. Per-query pin/unpin is disallowed. The union can need more memory than serial per-query pinning. |
| `--pin-after-iteration` | Nonzero values rejected. No maintenance transitions during the measured workload. |
| `--pin-compression` | Supported for GPU/HOST pins during untimed preparation. |
| `--precmd nsys/gdb`, `--nsys-profile` | Rejected: their per-query subprocess model does not share a runtime. |
| `--duckdb-profiling` | Unique per-stream/query/iteration profiles. Adds instrumentation overhead. |
| `--query-timeout` | Seconds from request submission through result fetch, including admission waiting. Default 90; 0 disables. |
| `--run-timeout` | Whole child-process deadline including setup, all engine phases, validation and cleanup. Default 3600 seconds; 0 disables. The supervisor kills the complete runtime on expiry. |
| `--validation-memory-limit-mb` | Measured-result capture budget in MiB, default 1024. Exceeding it fails the run instead of silently dropping checks. |

All GPU worker connections disable CPU fallback so failed GPU work cannot silently become a
successful CPU measurement. Ordinary errors are recorded and other requests continue; failures,
timeouts, incomplete streams and validation mismatches make the final command exit nonzero.
The whole-run watchdog also bounds native hangs that connection interruption cannot resolve.

#### Validation and results

Without `--validation` or `--duckdb-results`, results are consumed in bounded fetch chunks and
discarded. No result files or accumulated result rows are retained. With validation enabled,
**every measured execution** is captured in memory, then written and compared after timing;
queries are not re-executed for validation. A bad intermediate iteration cannot be overwritten
by a later good result. Capture accounting includes Python row/value sizes and conservative
container overhead; it is not a process-RSS limit and does not bound native buffers or comparison
workspace. Capturing rows adds client overhead, so compare runs with the same validation setting.

`--validation` requires `--engine both`. Alternatively, `--engine gpu --duckdb-results <run>`
validates against existing CPU results and automatically enables capture. References produced
by a validated concurrent run include canonical `duckdb/q<N>/result.txt` files for reuse.
The comparator checks row multisets, duplicate counts, exact non-float values and the existing
absolute float tolerance; it does not check ORDER BY semantics.

New references carry SQL hashes and input file size/mtime fingerprints; incompatible references
are rejected. Legacy references without provenance are accepted with an explicit notice.
Fingerprints are not content hashes. Keep inputs unchanged; remote S3 object contents cannot be
verified by this local manifest. S3 retains the existing GPU-only/no-pinning restrictions and
needs externally supplied matching references for validation.

CPU runs first under `--engine both` and can warm the OS cache. The initial best-effort cache
drop occurs once before the phases, with no resets during overlap. Use a GPU-only run with
reused references when you want to avoid running a CPU baseline immediately before measurement.

#### Concurrent output (schema version 2)

- `metadata.json`: client concurrency, configured admission limit, exact stream orders,
  effective config path, input/query fingerprints and final process status.
- `concurrent_config.yml`: frozen effective Sirius configuration, when a config was found.
- `csv/concurrent_runtimes.csv`: one row per attempted request, including stream, query,
  iteration, sequence, relative start/end, latency, row count, status and error.
- `concurrent_summary.json`: expected/submitted/successful/failed/not-submitted counts,
  timeouts, workload elapsed time, successful queries/second, latency median/p95 and sample
  counts, per-stream completion times and validation status. Failed runs are explicitly marked.
- `validation.csv`: one verdict per captured execution, when requested.
- `<engine>/stream<S>/q<Q>/result_iter<I>.txt`: measured results, only with validation.
- `duckdb/stream<S>/q<Q>/profile_iter<I>.json`: optional CPU profiles.
- `sirius/execution_logs/readonly_s<S>_q<Q>_i<I>.log` and `log_index.json`: lines attributed by
  instance/connection/query/window identity. Original shared logs retain unattributed lines.

Latency includes admission waiting and result consumption. Workload elapsed time runs from the
common barrier release through the last result/error, so throughput includes client scheduling,
logging and between-request overhead. It is **not** the sum of individual latencies. Legacy serial
summary/log-splitting tools are not used for these artifacts. Partial request CSVs survive a
watchdog timeout; metadata and the summary mark the run failed.

Harness regression tests:

```bash
pixi run python -m unittest discover -s test/tpch_performance -p test_concurrent_benchmark.py
```

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

Options:
- `--config <path>` — Sirius config file (default: `~/.sirius/sirius.yaml`)
- `--parquet-dir <path>` — parquet dataset directory (default: `test_datasets/tpch_parquet_sf<SF>`)
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

**Note:** The DuckDB baseline uses the same Sirius-built binary (`build/release/duckdb`) but with `SIRIUS_CONFIG_FILE` unset so the Sirius extension does not initialize. This means DuckDB runs on CPU using all available cores.

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
Quent postcard files to `sirius.telemetry.output_directory`.

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
