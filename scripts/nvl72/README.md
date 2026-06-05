# NVL72 (GB200) multi-GPU TPC-H benchmark harness

Reproducible TPC-H SF1000 benchmarking for Super Sirius on a GB200 NVL72 node
(2 Grace + 4 Blackwell in the configuration these configs target). Each query runs
in its own DuckDB process with the referenced columns pinned in the host tier across
the configured GPU count, then is timed over cold+warm iterations.

## Contents

- `configs/sirius_{1,2,4}gpu.yaml` — Sirius configs differing only in `topology.num_gpus`.
  Tuned for SF1000 on GB200: `scan_task_batch_size`/`concat_batch_bytes = 5 GiB`,
  host tier `capacity_bytes = 430 GiB` (~90% of one Grace LPDDR node), GPU usage 0.9.
- `run_benchmarks.sh` — driver: runs the 22 queries per scenario, each in an isolated
  process, with per-query timeout and status classification (ok / cuda_error / timeout /
  fallback); writes a timestamped dir under `results/` (CSV + summary).
- `profile_query.sh` — nsys-profile a single pinned query (capture scoped to execution
  via `profiler_start()`/`profiler_stop()`).
- `gen_query_sql.py` — generates per-query SQL (views + `SET gpu_execution` + per-query
  column pinning), reusing `queries.py` / `tpch_pin_columns.py` from `test/tpch_performance/`.

## Usage

From the repo root, with Sirius built (`pixi run make -j$(nproc)`):

```bash
# all 7 scenarios (disk/host {1,2,4} GPU + gpu 4-GPU), all 22 queries
./scripts/nvl72/run_benchmarks.sh

# subset
SCENARIOS="host_4gpu" QUERIES="1,9,21" ./scripts/nvl72/run_benchmarks.sh

# point at a different dataset
DATA=/path/to/sf1000 ./scripts/nvl72/run_benchmarks.sh
```

Env knobs: `SCENARIOS`, `QUERIES`, `DATA`, `DUCKDB`, `QUERY_TIMEOUT`, `ITERATIONS`, `OUT`.

> Reading SF1000 from a GPFS mount (e.g. `/scratch`) currently requires the temporary
> io_uring workaround in `src/io/uring/uring_reactor.cpp` (see the TODO there).
