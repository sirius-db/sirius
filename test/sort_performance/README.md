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

## Where to run it

Run it on the workstation, never on the development machine: no number taken there is reported.
From the worktree root, `.ai-helper/workstation.sh` syncs the code, builds it and runs the command
inside the workstation's pixi environment while holding the workstation lock, so nothing else
uses the GPUs during the benchmark.

```bash
# Print the matrix, each run's config and SQL; runs nothing (works anywhere)
pixi run python test/sort_performance/sort_merge_bench.py --gpu-limit-bytes 4294967296 --dry-run

# The full matrix at 10x an 8 GiB limit, on GPU 0 of the workstation
.ai-helper/workstation.sh python test/sort_performance/sort_merge_bench.py \
  --gpu-limit-bytes 8589934592 --device 0 \
  --spill-dir /mnt/disk_2/sirius_spill \
  --out-dir /mnt/disk_2/sort_bench_results/8g

# A small validated run of one dataset
.ai-helper/workstation.sh python test/sort_performance/sort_merge_bench.py \
  --gpu-limit-bytes 1073741824 --scale 4 --datasets uniform-1key --validate \
  --out-dir /mnt/disk_2/sort_bench_results/smoke
```

- **Pass `--out-dir` outside the mirror.** The default is `test/sort_performance/output/<timestamp>`
  inside the checkout, and the next `workstation.sh` sync deletes what the worktree does not have.
- **One GPU.** `--device` sets `CUDA_VISIBLE_DEVICES` for each run and is the index whose memory
  is sampled. A full matrix outlasts one shell call: start it detached and poll its output.
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

Before each run the harness checks the free space under `--spill-dir`. It needs about
`1.05 × (data bytes − host tier)`, where the host tier is the host capacity of the run's config
(default 0.9 of RAM) times its `downgrade_stop_fraction` (default 0.8). A run that cannot fit is
recorded as `refused` and not started; `--skip-disk-check` overrides that.

On the workstation (125 GiB RAM, `/mnt/disk_2`), ten times a whole 96 GiB GPU is about 960 GB of
input, of which roughly 850 GiB must fit on `/mnt/disk_2`. Check `df -h /mnt/disk_2` first, and
prefer a smaller limit when it answers the question. Each run spills under
`<spill-dir>/sort_bench_<timestamp>/<run>` and that directory is deleted when the run ends.

## Duration

No duration has been measured yet. A run's time is dominated by spill I/O: every byte beyond the
GPU limit is written to host memory, and beyond the host tier to disk, and read back at least
once. `--timeout-s` (default 7200, `0` disables it) bounds each run; the full matrix is 20 runs.
Raise the timeout for limits of tens of GiB.

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

1. Writes `runs/<run>/config.yaml`: the `--base-config` file (if any) plus the GPU limit, the
   run's spill directory (`sirius.memory.disk.downgrade_root_dirs`) and, **in the compressed arm
   only**, `sirius.compression.spill_min_batch_bytes` lowered to 1 MiB
   (`--spill-min-batch-bytes`). With the 64 MiB default the merge's inputs, which are slices of a
   batch, would not be compressed at all.
2. Starts a fresh Python process with `SIRIUS_CONFIG_FILE`, `SIRIUS_LOG_DIR` and
   `CUDA_VISIBLE_DEVICES` set. It opens one DuckDB connection, loads the extension, runs
   `SET gpu_execution = true`, `SET sort_merge_mode`, `SET spill_compression`, and times the
   `COPY`.
3. Samples the GPU's used memory with `nvidia-smi` while the process runs.
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
- `runs/<run>/`: `config.yaml`, `query.sql`, `sirius_log_tail.txt` (the last 200 lines of the
  log; `--keep-logs` keeps the whole log under `log/`), `child_output.txt`, `result.json`,
  `row.json`.

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
| `peak_gpu_mib` | Highest `memory.used` of the GPU during the run, for the whole device, so it includes anything else on it and the CUDA context |
| `spill_bytes_host`, `spill_bytes_disk` | Bytes the downgrade executor moved to the host tier and to disk |
| `spill_batches_host`, `spill_batches_disk` | The same, in batches |
| `downgrade_requests` | Downgrade requests completed |
| `oom_retries` | Pipeline tasks rescheduled after running out of GPU memory |
| `discard_rows`, `discard_bytes`, `order_violations` | The discard sink's own counts; `order_violations` is only meaningful with `--validate` |
| `merge_sort_operators` | `[merge_sort][stats]` lines found; the columns below are summed over them |
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

Running out of memory under `one_shot` at 10× the GPU limit is the result the comparison is
after, not a harness problem. A `one_shot` run that ends in `oom`, `error`, `timeout` or
`crashed` is recorded with its status, its error text, and whatever metrics its log holds, and
the harness still exits 0. The exit code is 1 when a `windowed` (or `auto`) run does not end
`ok`, or when any run ends in `row_mismatch`, `order_violation` or `refused`.
