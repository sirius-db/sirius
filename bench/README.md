# Reproducing Sirius benchmarks

Hardware-specific configs and selected results live together. Execution and
reporting code is shared; each hardware directory describes its run protocol.

```text
bench/
  replay.sh                  shared Pixi entry point
  common/                    planning, preflight checks, timing reports, tests
  gb300/
    configs/                 cold, hot, pinned, and SF10000 Q10 overrides
    results/                 selected timings, manifests, concurrency, Quent summaries
    replay.sh                GB300 entry point
    ablation/                GB300-specific study and its results
    spill-sweep/             host-cap study and results
    device-spill-sweep/      GPU-cap study and results
    *-figures/               plotting code and saved figures
  g7e/
    configs/                 current G7e cold, hot, and pinned presets
    guide-26b5caba/           exact guide presets and versioned reproduction adapter
    results/guide-ktls-20261123/  selected London SF100–SF10000 sums and provenance
    results/baseline-20261111/  frozen older timings and exact measured configs
    replay.sh                G7e entry point
```

See [GB300](gb300/README.md) and [G7e](g7e/README.md) for hardware requirements,
selected results, config history, and limitations. These are read-only query
experiments, not official TPC-H benchmark submissions.

The [G7e guide reproduction adapter](g7e/guide-26b5caba/README.md) targets the
saved `26b5caba` London measurements. It preserves their per-scale guide YAML,
session settings, `--mode` behavior without harness profile overrides, and
per-query process lifetimes. It reuses the shared planner and AWS/storage checks;
the replay commands below retain their existing hardware-preset behavior.

## Plan and run

Dry runs require only Python's standard library. They print a JSON plan without
loading Sirius, contacting AWS, reading datasets, creating output directories,
or checking hardware:

```bash
bash bench/replay.sh gb300 1000 cold --dry-run
bash bench/replay.sh g7e 1000 hot --queries 3,6,9,18 --dry-run
bash bench/g7e/replay.sh 3000 pinned --pin-tier host --dry-run
```

For an actual run, build this checkout with `pixi run make`, provide complete
TPC-H Parquet data, and remove `--dry-run`. For example:

```bash
bash bench/gb300/replay.sh 1000 cold --input /mnt/datasets/tpch_parquet_sf1000
bash bench/g7e/replay.sh 1000 hot --input s3://YOUR_SAME_REGION_BUCKET/tpch_sf1000/
```

The shared entry point runs Python inside this checkout's Pixi environment.
To measure a different engine revision without editing either checkout, pass
`--source-dir /path/to/built/sirius`. Run with the engine checkout's Pixi Python
when its locked environment differs, for example `pixi run --manifest-path
/path/to/built/sirius/pixi.toml python /path/to/runner/bench/common/replay.py ...`.
Provenance records the engine commit/binaries separately from the runner commit.
`PIXI_BIN` can select Pixi. `DATASET_ROOT` defaults to `/mnt/datasets`, with
`tpch_parquet_sf<SF>` appended when `--input` is omitted. Inputs are never
generated or copied across regions by these scripts.

Cold/hot default to two iterations; pinned defaults to one. Override with
`--iterations N`, or select queries using `--queries 3,6,9,18`. `pin` remains an
alias for `pinned`. Pinning selects host memory by default; G7e also accepts
`--pin-tier gpu` and optional `--pin-compression`. GB300's selected pinned
protocol always uses host pinning and compressed `lineitem`/`orders` plans.
Neither the profile nor pinning changes the saved YAML cache policy.
Use `--config /path/to/custom.yaml` to measure a run-specific config without
editing the hardware presets. This explicit file applies to every selected
query, including GB300 SF10000 Q10; implicit preset overrides do not replace it.

On both hardware types, cold starts a fresh Sirius process for every query
iteration and uses `performance_test.py --profile cold` to drop the local
dataset-file OS cache and reset Sirius's cache before the measurement. Order is
Q1(1), ..., Q22(1), Q1(2), ..., Q22(2), with each worker exiting before the next
starts. Engine startup and cache clearing are outside query timing. The EC2
instance remains running. This cold protocol applies to new runs; stored
measurements retain their original process lifetimes.

G7e checks EC2 identity and the actual spill mount before running. Every S3
input must be in the instance's actual region, verified using AWS bucket
metadata. Each table must contain Parquet objects; verify the generation
manifest/expected partition counts separately when choosing a dataset. A table
presence check alone cannot prove dataset completeness. Missing or unverifiable
inputs fail the run. The runner never provisions, stops, or terminates instances.

## Outputs and sums

Runs create a new directory under `test_datasets/<hardware>-replay-output/`.
Set `--output /new/unique/path` for an explicit directory; existing directories
are refused. `GB300_REPLAY_OUTPUT` and `G7E_REPLAY_OUTPUT` select parent roots.
Each worker is bounded by `--worker-timeout` (3600 seconds by default).

The output contains source and binary provenance, a command plan, worker logs,
the exact source YAML, a separate effective YAML, raw Quent traces, per-query
iteration times, and `summary.json`. The effective copy changes only telemetry
enablement and its output directory. Saved configs are hashed and left intact;
the supplied YAML is also checked after execution. CPU fallback is disabled,
and successful timings require matching GPU plan evidence in Sirius logs.

Every report includes each iteration's query-time **sum**, the **sum of per-query
best times**, and coverage. Missing, timed-out, and failed measurements remain
unavailable, rather than zero. An incomplete worker stops the campaign and
returns failure while retaining earlier results. Setup and process startup are
excluded from query-time sums; these sums are not campaign wall time.

Replays measure the checkout that you build. They do not reproduce old binaries
or automatically replace selected snapshots. Import a replacement only after
reviewing its source revision, dataset, config, protocol, and coverage; retain
the previous snapshot and its provenance.

## Runner tests

```bash
pixi run python -m unittest discover -s bench/common -p 'test_*.py' -v
```

These use temporary files and fake measurements. They do not execute GPU
queries, contact S3/EC2, or require a Sirius build.
