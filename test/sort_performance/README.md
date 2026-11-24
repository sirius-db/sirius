# Sort merge benchmark

`sort_merge_bench.py` compares the two merge paths of `MERGE_SORT` (`sort_merge_mode`:
`one_shot` and `windowed`) with spill compression off and on, over five generated datasets, each
sized as a multiple (default 10×) of a **configured** GPU memory limit.

Each run is one statement in its own process:

```sql
COPY (SELECT * FROM sirius_generate(...) ORDER BY <keys>) TO '/dev/null' (FORMAT sirius_discard)
```

The data is generated on the GPU, the whole sort runs, and the discard sink drops the sorted
output as it arrives, so only the spilled inputs take host memory and disk. The statement returns
the row count, which the harness compares with `num_batches × rows_per_batch`.

`bench_metrics.py` holds the log parsers and the GPU memory sampler the harness uses.

A summary of the measured results and the `one_shot` deprecation recommendation is in
[RESULTS.md](RESULTS.md). Result directories are not committed: write them under the default
`output/` or under `results/`, both git-ignored.

## How to run it

Run it from the repository root on a machine with an NVIDIA GPU, after `pixi run make` has built
the extension. Nothing else should use that GPU during a run, and the spill directory needs a
fast disk with room for the data (see [Disk](#disk)). `$BENCH` below stands for a directory on
that disk.

```bash
# Print the matrix, each run's config and SQL; runs nothing (needs no GPU)
pixi run python -B test/sort_performance/sort_merge_bench.py --gpu-limit-bytes 4294967296 --dry-run

# Smoke run: the whole matrix, validated, at 3x a 4 GiB limit (about 12 GiB per dataset)
pixi run python -B test/sort_performance/sort_merge_bench.py \
  --gpu-limit-bytes 4294967296 --scale 3 --device 0 --validate \
  --out-dir $BENCH/smoke

# One run of the whole-GPU matrix: 5x a 94 GiB limit with a 96 GiB host cap (see Sizing)
pixi run python -B test/sort_performance/sort_merge_bench.py \
  --gpu-limit-bytes 100931731456 --scale 5 --device 0 \
  --base-config $BENCH/whole-gpu-5x/base_config.yml \
  --datasets uniform-1key --compression false --modes windowed --timeout-s 10800 \
  --spill-dir $BENCH/spill \
  --out-dir $BENCH/whole-gpu-5x/cells/uniform-1key__comp-false__windowed

# Data that fits the GPU (half the limit), all three modes, three repeats per cell
pixi run python -B test/sort_performance/sort_merge_bench.py \
  --gpu-limit-bytes <limit> --scale 0.5 --modes one_shot,windowed,auto --repeat 3 \
  --out-dir $BENCH/fits-gpu
```

- **Output.** Without `--out-dir` the results go to `test/sort_performance/output/<timestamp>`
  (git-ignored). If you run the benchmark on another machine through a wrapper of your own that
  syncs the checkout there, pass an `--out-dir` outside the synced tree so that the next sync
  does not delete the results.
- **One GPU.** `--device` is an `nvidia-smi` index. Each run gets `CUDA_VISIBLE_DEVICES=<device>`
  with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the GPU the run uses is the one whose memory is sampled
  (CUDA's default order is fastest first, which need not be `nvidia-smi`'s). A full matrix
  outlasts one shell call: start it detached and poll its output.
- **Extension.** `--extension` defaults to `build/release/extension/sirius/sirius.duckdb_extension`;
  when that is missing the harness uses
  `sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension`, where `pixi run make`
  puts it.

## Sizing

`--gpu-limit-bytes` is written to `sirius.memory.gpu.usage_limit_bytes`. Each dataset has
`num_batches × rows_per_batch × row_width ≈ --scale × limit` bytes, with `rows_per_batch`
8,000,000 (`--rows-per-batch`) and the row width taken from the column types: 8 bytes per BIGINT
or DOUBLE, 4 per INTEGER, mean length + 4 per VARCHAR. Always quote the GPU limit and the data
size with a result; `results.md` states both.

`--scale` defaults to 10 and may be below 1 for data that fits the GPU (`--scale 0.5` is half
the limit). The data is a whole number of batches and never fewer than one. When that puts a
dataset more than 10 % away from `--scale × limit`, the harness prints a `WARNING` with the
matrix; lower `--rows-per-batch` to get closer.

`--modes` takes any of `one_shot`, `windowed` and `auto` (default `one_shot,windowed`).
`--repeat N` runs every cell N times, one CSV row each, numbered in the `repeat` column.

### The whole-GPU matrix

The A/B in [RESULTS.md](RESULTS.md) uses the **whole GPU with data at 5× the limit**:

- **GPU limit 94 GiB** (`--gpu-limit-bytes 100931731456`). It is the largest limit found to
  start on a 97,887 MiB card: 95 GiB and 96 GiB are rejected with
  `sirius.memory.gpu.usage_limit_bytes exceeds the capacity of GPU 0`. The limit must not exceed
  the capacity Sirius computes for the device, which is below `nvidia-smi`'s total.
- **`--scale 5`**: about 470 GiB of data per run.
- **A 96 GiB host cap**, passed with `--base-config`:

  ```yaml
  sirius:
    memory:
      host:
        capacity_bytes: 103079215104
  ```

  Without it the host capacity defaults to 0.9 of RAM. A run with that default filled the
  125 GiB of RAM of the machine it was measured on (the process grew to 113 GiB), the machine
  swapped, and the run was stopped unfinished.
- **One cell at a time, as a pair**: the `windowed` run, then the `one_shot` run of the same
  dataset and compression setting, each its own harness invocation.

### Disk

Before each run the harness checks the free space under `--spill-dir`. It needs about
`1.05 × (data bytes − host tier)`, where the host tier is the host capacity of the run's config
(default 0.9 of RAM) times its `downgrade_stop_fraction` (default 0.8). A run that cannot fit is
recorded as `refused` and not started; `--skip-disk-check` overrides that.

That estimate covers the spilled inputs only. Intermediate merge passes spill their output
again: an uncompressed windowed run of the whole-GPU matrix moved up to 977 GB to disk over its
duration (how much the disk held at one time was not sampled). Check the free space of the
spill disk (`df -h`) before each run. Each run spills under
`<spill-dir>/sort_bench_<timestamp>/<run>` and that directory is
deleted when the run ends. A script that runs several invocations should also remove leftover
`sort_bench_*` directories before each run.

The smoke command (`--gpu-limit-bytes 4294967296 --scale 3`) never spills to disk, because the
default host tier holds all of its data. To exercise the disk tier at that size, add a
`--base-config` with a small host capacity:

```yaml
sirius:
  memory:
    host:
      capacity_bytes: 4294967296
```

## Duration

Measured on the machine described in [RESULTS.md](RESULTS.md). A run's time is dominated by spill I/O.

- **Whole-GPU matrix** (470 GiB per run): a `windowed` run took 404–999 s, and the 20 runs took
  5 h 32 min.
- **`one_shot` timeout rule** used there: `--timeout-s` = 4× the windowed wall time of the same
  cell, at least 1800 s. The windowed runs had `--timeout-s 10800` and never reached it.
- **Failing `one_shot` cells are costly before they fail, so budget for them.** The four that
  were killed at their timeout had by then each written 1.6–3.6 TB to disk (cumulative).

`--timeout-s` defaults to 7200 (`0` disables it) and bounds each run.

## Datasets

The column specs are the `DATASETS` table in `sort_merge_bench.py`; `seed` is 42.

| Dataset | Columns | `ORDER BY` | Bytes/row |
|---|---|---|---|
| `uniform-1key` | `k` BIGINT uniform [0, 1e15], `v` DOUBLE uniform | `k` | 16 |
| `zipf-1key` | `k` BIGINT zipf s = 1.1, cardinality 1e9, `v` DOUBLE uniform | `k` | 16 |
| `normal-multikey` | `k1` INTEGER normal(0, 1e6), `k2` BIGINT uniform [0, 1e15], `s` VARCHAR length 8–16 | `k1, k2, s` | 28 |
| `few_distinct-strings` | `s` VARCHAR few_distinct, 18 distinct values of length 18 | `s` | 22 |
| `sequential` | `k` BIGINT sequential from 0 step 1, `v` DOUBLE uniform | `k` | 16 |

For a VARCHAR column `sirius_generate` uses `p1` and `p2` both as the distribution's parameters
and as the length range, so a `few_distinct` string column has as many distinct values as its
minimum length. `few_distinct-strings` therefore uses 18 distinct 18-character strings (every row
ties with one in 18 of the others), not 1000 distinct values of length 12–24.

## What a run does

For each cell of dataset × mode × compression × repeat the harness:

1. Writes `per_run/<run>/config.yaml`: the `--base-config` file (if any) plus the GPU limit, the
   run's spill directory (`sirius.memory.disk.downgrade_root_dirs`) and, **in the compressed arm
   only**, `sirius.compression.spill_min_batch_bytes` lowered to 1 MiB
   (`--spill-min-batch-bytes`). With the 64 MiB default the merge's inputs, which are slices of a
   batch, would not be compressed at all.
2. Starts a fresh Python process with `SIRIUS_CONFIG_FILE`, `SIRIUS_LOG_DIR`,
   `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER=PCI_BUS_ID` set. It opens one DuckDB
   connection, loads the extension, runs `SET gpu_execution = true`, `SET sort_merge_mode`,
   `SET spill_compression`, and times the `COPY`.
3. Samples the GPU's used memory once a second, with one `nvidia-smi --loop-ms` process that
   lives as long as the run.
4. Reads the run's `sirius.log` for the merge statistics, the spill totals, the OOM retries and
   the discard sink's line, then deletes the spill directory.

`--validate` adds `check_order true` to the `COPY`, and requires the discard sink's row count to
match and its order violations to be zero. Use it on small scales.

## Output

In `--out-dir`:

- `results.csv`: one row per run. `results.md`: one table per dataset. Both are rewritten after
  every run, so an interrupted matrix keeps what it finished.
- `run_info.json`: the command line, host, extension path and the dataset specs with their sizes.
- `base_config.yml`: a copy of `--base-config`, when given.
- `per_run/<run>/`: `config.yaml`, `query.sql`, `job.json`, `result.json`, `row.json`,
  `child_stdout.txt` (what the run's process printed) and `sirius_log_tail.txt` (the last 200
  lines of the log, `--log-tail-lines`). The whole log is deleted unless `--keep-logs` is given,
  which keeps it under `log/`.
- `per_run/<run>/sirius_log_first_errors.txt`, **only for a run that did not end `ok`**: the
  first 200 log lines at level `warning`, `error` or `critical`, in log order. The tail of a long
  failing run holds only its last retries; this file holds the first out-of-memory and error
  messages. A message that repeats with only its numbers changing (a warning logged for every
  batch, for example) is kept 5 times, so it cannot fill the file. The file is empty when the
  log has no such line.

Result directories are not committed; `output/` and `results/` under `test/sort_performance/`
are git-ignored.

### Columns of `results.csv`

| Column | Meaning |
|---|---|
| `dataset`, `mode`, `compression`, `repeat` | The matrix cell |
| `gpu_limit_bytes` | The configured GPU limit |
| `num_batches`, `rows_per_batch`, `rows` | Generated input; `rows` is the expected row count |
| `bytes` | `rows × row width`, the nominal data size |
| `status` | See below |
| `wall_s` | Wall-clock seconds of the `COPY` statement alone |
| `load_s` | Seconds to load the extension and apply the settings |
| `rows_counted` | The row count the `COPY` returned |
| `peak_gpu_mib` | Highest `memory.used` that `nvidia-smi` reported for the whole device during the run. Not Sirius's own count, and not comparable with the limit: see below |
| `spill_bytes_host`, `spill_bytes_disk` | Bytes the downgrade executor moved to the host tier and to disk |
| `spill_batches_host`, `spill_batches_disk` | The same, in batches |
| `downgrade_requests` | Downgrade requests completed |
| `oom_retries` | Pipeline tasks rescheduled after running out of GPU memory |
| `discard_rows`, `discard_bytes`, `order_violations` | The discard sink's own counts; `order_violations` is only meaningful with `--validate` |
| `merge_sort_operators` | `[merge_sort][stats]` lines found; the columns below are summed over them. 0, with the columns below empty, for a run that failed or timed out before the operator finished |
| `windowed_partitions`, `one_shot_partitions` | Partitions merged by each path |
| `windowed_tasks`, `intermediate_tasks` | Windowed merge tasks, and those that were intermediate passes (fan-in above the cap) |
| `mergers_built`, `rounds`, `rows_out` | Windowed mergers created, merge rounds run, rows they emitted |
| `round_rows_min`, `round_rows_max` | Fewest and most rows emitted by one round |
| `refill_rows`, `refill_bytes` | Rows and bytes read into the windows |
| `refill_bytes_host`, `refill_bytes_disk`, `refill_bytes_compressed` | Of `refill_bytes`: from uncompressed host runs, uncompressed disk runs, and compressed runs |
| `runs_read_compressed` | Runs whose window was filled from a compressed batch at least once: how many of the merge's inputs the compressed arm really compressed |
| `window_growths`, `compacted_rows`, `fallback_conversions` | Window growths, rows compacted, runs converted by the fallback path |
| `shrinks` | Windowed inputs that shrank their window for an OOM retry |
| `error` | The error text of a run that did not end `ok` |

Keys a newer build appends to the `[merge_sort][stats]` line appear as extra columns before
`error`.

**`peak_gpu_mib` must not be read as "peak ≤ limit".** Sirius reserves its pool when it starts,
so the column starts at about the limit plus 0.6 GiB: a process with a 4,096 MiB limit and
nothing running showed 4,682 MiB. The column says little about how much the sort used
([RESULTS.md](RESULTS.md), "What Peak GPU measures").

**A timed-out run** keeps what its log held when it was killed: the spill totals (`spill_*`,
`downgrade_requests`) and `oom_retries`. It cannot have merge statistics or `discard_rows`: the
operator logs its `[merge_sort][stats]` line and the sink its `[discard_sink]` line when they
finish, and the process is killed before that.

### Status values and `one_shot` failures

| Status | Meaning |
|---|---|
| `ok` | The `COPY` returned the expected row count (and passed `--validate`) |
| `oom` | The statement failed with an out-of-memory error |
| `error` | The statement failed with any other error |
| `timeout` | The run exceeded `--timeout-s` and was killed |
| `crashed` | The process died without reporting (a signal, an abort, the kernel's OOM killer) |
| `row_mismatch` | The row count returned or seen by the discard sink is not the expected one |
| `order_violation` | `--validate` found sorted output out of order |
| `refused` | Not started: the spill directory's disk is too small |

Running out of memory under `one_shot` at a multiple of the GPU limit is the result the
comparison is after, not a harness problem. A `one_shot` run that ends in `oom`, `error`,
`timeout` or `crashed` is recorded with its status, its error text, and whatever metrics its
log holds, and
the harness still exits 0. The exit code is 1 when a `windowed` (or `auto`) run does not end
`ok`, or when any run ends in `row_mismatch`, `order_violation` or `refused`.

## Tests

No GPU and no Sirius extension are needed. From the repository root:

```bash
for t in test/sort_performance/test_*.py; do pixi run python -B "$t"; done
```

- `test_sort_merge_bench.py` tests the datasets, the sizing, the generated SQL, the result
  checks, the log scan and a `--dry-run` of the default matrix; `test_bench_metrics.py` tests
  the log parsers. `test_bench_metrics_source_formats.py` checks the `[merge_sort][stats]` tag
  and counters against the format string in the C++ sources.
- Running a matrix (child processes, timeouts, signals, the samplers) has no unit tests: check
  it with the smoke run in [How to run it](#how-to-run-it).
- `-B` keeps Python from writing `test/sort_performance/__pycache__/` (git-ignored) into the checkout.
- pytest is not in the pixi environment; the files run under `unittest`. Run each file as a
  script: `python -m unittest test.sort_performance...` does not work, because the standard
  library's `test` package shadows the repository's `test/` directory.
