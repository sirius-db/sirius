# Reproducing Sirius benchmarks

Hardware-specific configs and selected results live together. Planning and AWS
storage checks are shared; each hardware has one documented reproduction path.
The M9gd directory additionally preserves the standalone DuckDB CPU baseline.

```text
bench/
  replay.sh                  dispatches to the hardware's reproducer
  common/                    shared planning, AWS/storage checks, GB300 runner
  gb300/                     GB300 configs, results, studies and figures
  m9gd/                      DuckDB 2.0.0 same-region S3 CPU benchmarks
  g7e/
    replay.sh                G7e guide reproducer entry point
    guide-26b5caba/           guide presets and measured worker protocol
    results/guide-ktls-20261123/  validated SF100–SF10000 timings and provenance
```

See [G7e](g7e/README.md), [GB300](gb300/README.md) and [M9gd](m9gd/README.md) for requirements, selected
results and limitations. These are read-only query experiments, not official
TPC-H benchmark submissions. No reproducer provisions or stops instances.

## G7e: the guide-derived recipe

The single G7e reproducer targets the saved `26b5caba` London measurements.
It chooses the CPU-budget guide preset for SF100/SF300/SF1000 and the SF10000
balanced guide preset for SF3000/SF10000. Hot changes eviction to LRU. It has no
alternative custom preset or `--config` override.

```bash
bash bench/replay.sh g7e 3000 cold \
  --source-dir /path/to/built/sirius-26b5caba \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf3000_regenerated/ \
  --iterations 1 --dry-run

bash bench/replay.sh g7e 3000 hot \
  --source-dir /path/to/built/sirius-26b5caba \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf3000_regenerated/ \
  --iterations 2 --dry-run
```

`bench/g7e/replay.sh <SF> <profile>` is an equivalent entry to this same code.
Dry runs need only standard-library Python and do not contact AWS, read data,
load Sirius or create output. Actual runs use the specified engine's locked
Pixi environment. See the [complete instructions](g7e/guide-26b5caba/README.md)
for builds, host pinning, CPU references, outputs and exact checks.

Cold starts a fresh process and clears OS/Sirius caches before every query
measurement. Hot/pinned restart per query and retain the process/cache between
its consecutive measurements. The harness receives `--mode isolated/grouped`
and no `--profile` override. There is no extra warmup or YAML rewriting; only
Quent settings are added to effective copies. Same-region S3, generation
manifests, native kTLS and enabled local NVMe spilling are verified.

## GB300: the existing hardware replay

GB300 continues to use its saved configs, selected timings and studies:

```bash
bash bench/replay.sh gb300 1000 cold --dry-run
bash bench/gb300/replay.sh 1000 cold --input /mnt/datasets/tpch_parquet_sf1000
```

Actual runs use this checkout's Pixi environment and build by default. To use a
different engine, pass `--source-dir /path/to/built/sirius`; when its environment
differs, run `bench/common/replay.py` with that engine's Pixi Python. `PIXI_BIN`
can select Pixi. Inputs default to `$DATASET_ROOT/tpch_parquet_sf<SF>` with
`DATASET_ROOT=/mnt/datasets`. No data is generated or copied across regions.

Cold/hot default to two iterations and pinned to one. GB300 retains its existing
`--iterations`, `--queries`, explicit `--config` and separate SF10000 Q10 config
behavior. Cold starts a fresh process per query iteration, with the harness's
cold profile clearing caches. Hot/pinned preserve their documented session
protocol; selected pinned runs use host pins and compressed lineitem/orders.

GB300 outputs default to `test_datasets/gb300-replay-output/`, or a new `--output`
directory. Reports contain iteration sums, coverage and clearly labeled sums
of per-query minima. Config hashes, effective YAML, source/binary provenance,
worker logs and Quent traces are retained. Timeouts and missing measurements
remain unavailable. See [GB300](gb300/README.md) for the full protocol and studies.

## CPU-only runner checks

The [M9gd CPU recipe](m9gd/duckdb-2.0-s3/README.md) has its own
`bench/m9gd/duckdb-2.0-s3/reproduce.sh` entry point. Its November 24 results save
both iterations of SF1000, SF3000 and SF10000, with external-file and
Parquet-metadata caches enabled, fresh processes/cache clears between query
pairs, and verified local NVMe spill. It does not load Sirius or use the GPU
replay's `--profile` controls.

The [selected November 25 CPU results](m9gd/duckdb-2.0-s3/results/SELECTED.md)
extend from SF100 through SF10000 using a native build and a 64 KiB page kernel.
SF10000 combines retries and uses 256 GiB only for Q12, with 296.7 GiB for
other queries; that selection is not a uniform-memory full-suite run.

```bash
pixi run python -m unittest discover -s bench/common -p 'test_*.py' -v
pixi run python -m unittest discover -s bench/g7e/guide-26b5caba -p 'test_*.py' -v
```
