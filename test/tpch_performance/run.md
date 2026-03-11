# TPC-H Performance Testing

Benchmarking tools for comparing DuckDB (CPU) vs Sirius (GPU) on TPC-H queries at various scale factors.

## Prerequisites

1. Build the project:
   ```bash
   pixi run make -j12
   ```

2. Ensure a Sirius config file exists. The binary looks for config in this order:
   - `SIRIUS_CONFIG_FILE` environment variable
   - `~/.sirius/sirius.cfg` (default)

3. Ensure parquet data exists (auto-generated if missing via `generate_tpch_data.sh`).
   - On the GB300 machine, the SF1000 dataset is at `/home/nvidia/tpch_parquet_sf1000`.

## Running Benchmarks

All commands run from the **project root** directory.

### Full benchmark with validation (recommended)

`benchmark_and_validate.sh` runs all 22 TPC-H queries, compares Sirius vs DuckDB results for correctness, and produces a timestamped run directory.

```bash
# Basic usage (uses ~/.sirius/sirius.cfg, both engines, dataset from test_datasets/)
./test/tpch_performance/benchmark_and_validate.sh 100

# With explicit options
./test/tpch_performance/benchmark_and_validate.sh \
  --config ~/.sirius/sirius.cfg \
  --parquet-dir /path/to/tpch_parquet_sf1000 \
  --engines "sirius duckdb" \
  1000

# Sirius only (skip DuckDB baseline)
./test/tpch_performance/benchmark_and_validate.sh \
  --config ~/.sirius/sirius.cfg \
  --parquet-dir /path/to/tpch_parquet_sf1000 \
  --engines sirius \
  1000

# Regenerate report from an existing run
./test/tpch_performance/benchmark_and_validate.sh --report runs/<run_dir>
```

Options:
- `--config <path>` — Sirius config file (default: `~/.sirius/sirius.cfg`)
- `--parquet-dir <path>` — parquet dataset directory (default: `test_datasets/tpch_parquet_sf<SF>`)
- `--engines <list>` — space-separated engine list (default: `"sirius duckdb"`)

Each run creates a directory under `runs/<timestamp>_sf<SF>_2iter/` containing:
- `run_info.txt` — git branch/revision, tree clean/dirty, build freshness, hostname, memory, CPUs, GPUs, filesystem read benchmark
- `run_info.patch` — full git diff when tree is dirty
- `sirius_config.cfg` — copy of the Sirius config used
- `sirius/` and `duckdb/` — per-engine logs, per-query results and timings
- `validation.csv` — per-query match/error status
- `comparison.txt` — cold/warm timing table with speedup ratios
- `timings.csv` — long-format iteration runtimes (engine,query,iteration,runtime_s)

**Note:** The DuckDB baseline uses the same Sirius-built binary (`build/release/duckdb`) but with `SIRIUS_CONFIG_FILE` unset so the Sirius extension does not initialize. This means DuckDB runs on CPU using all available cores.

### Running individual queries

`run_tpch_parquet.sh` is the core runner used by all benchmarks. It runs queries in a single DuckDB session with 2 iterations each (cold + warm, back-to-back) and auto-generates missing datasets.

```bash
# Run Sirius on queries 1-22
SIRIUS_CONFIG_FILE=~/.sirius/sirius.cfg \
  ./test/tpch_performance/run_tpch_parquet.sh sirius 100 $(seq 1 22)

# Run DuckDB baseline (no config needed)
./test/tpch_performance/run_tpch_parquet.sh duckdb 100 $(seq 1 22)

# Run specific queries with custom parquet directory
SIRIUS_CONFIG_FILE=~/.sirius/sirius.cfg \
  ./test/tpch_performance/run_tpch_parquet.sh --parquet-dir /data/tpch sirius 100 1 3 6
```

Environment variables:
- `SIRIUS_CONFIG_FILE` — path to Sirius config (required for sirius engine; unset automatically for duckdb engine)
- `TIMING_CSV` — path to write per-query timing CSV (optional)
- `OUTPUT_DIR` — directory for structured output (set by `benchmark_and_validate.sh`)

## Query Files

- `tpch_queries/orig/q*.sql` — Plain SQL queries (used by duckdb engine)
- `tpch_queries/gpu/q*.sql` — Queries wrapped in `call gpu_execution('...');` (used by sirius engine)

## Sirius Configuration

The Sirius config file (e.g. `~/.sirius/sirius.cfg`) controls:
- **GPU memory**: `usage_limit_fraction`, `reservation_limit_fraction`, `downgrade_trigger_fraction`, `downgrade_stop_fraction`
- **Host memory**: `capacity_bytes`, `initial_number_pools`, `pool_size`, `block_size`
  - Initial allocation = `initial_number_pools * pool_size * block_size`
- **Thread pools**: `pipeline`, `duckdb_scan`, `task_creator`, `downgrade` thread counts
- **Scan cache**: `duckdb_scan.cache`, `cache_decoded_table`, `cache_in_gpu`
- **Operator params**: `scan_task_batch_size`, `hash_partition_bytes`, `concat_batch_bytes`

### Example config (GB300, SF1000)

```
sirius = {
    topology = {
        num_gpus = 1;
    };
    memory = {
        gpu = {
            usage_limit_fraction = 0.9;
            reservation_limit_fraction = 1.0;
            downgrade_trigger_fraction = 0.8;
            downgrade_stop_fraction = 0.6;
        };
        host = {
            capacity_bytes = 471200000000;       // ~471 GB
            initial_number_pools = 785;
            pool_size = 512;
            block_size = 1048576;                // 1 MB
        };
    };
    executor = {
        pipeline = {
            num_threads = 4;
        };
        downgrade = {
            num_threads = 1;
        };
        duckdb_scan = {
            num_threads = 4;
            cache = true;
            cache_decoded_table = true;
            cache_in_gpu = false;
        };
        task_creator = {
            num_threads = 6;
        };
    };
    operator_params = {
        scan_task_batch_size = 5368709120;       // 5 GB
        default_scan_task_varchar_size = 256;
        max_sort_partition_bytes = 0;            // 0 = auto (33% GPU memory)
        hash_partition_bytes = 5368709120;       // 5 GB
        concat_batch_bytes = 5368709120;         // 5 GB
    };
};
```
