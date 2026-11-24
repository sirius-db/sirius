# G7e guide runs on `26b5caba`

These are the configurations and worker protocol used for the
[November 23 London results](../results/guide-ktls-20261123/README.md).
The portable adapter preserves the measured timing calls, per-iteration answer
export, native GPU evidence, Quent capture and process lifetimes. It replaces
machine-specific paths and instance IDs with explicit arguments. Original
machine scripts remain in the retained run archives; their hashes are in
[original-runner-sha256.json](original-runner-sha256.json).

The guide-derived family is the only G7e configuration published by this PR.
The reproducer fixes the guide YAML and the process protocol below; it exposes
no custom-config override.

## Recipes

| Scale | Cold YAML | Hot YAML | Session threads |
| --- | --- | --- | ---: |
| SF100 / SF300 | `cpu-budget-cold.yaml` | `cpu-budget-hot.yaml` | Engine default |
| SF1000 | `cpu-budget-cold.yaml` | `cpu-budget-hot.yaml` | 24 |
| SF3000 / SF10000 | `balanced-cold.yaml` | `balanced-hot.yaml` | 24 |

The YAML files in [configs](configs/) are byte-for-byte copies from the measured
runs. Do not fill in omitted options: the CPU-budget preset deliberately uses
the pinned engine's defaults, including disabled disk spilling. Its omitted
eviction setting is also preserved. Hot explicitly uses LRU. For SF3000 and
SF10000, hot differs from cold only by `eviction: lru`; both enable compressed
spilling to `/opt/dlami/nvme/spill`, which resolved to the local NVMe spill mount.
The CPU-budget pinned YAML is identical to its hot YAML.

Cold starts a new process for every query iteration in suite order. Hot and
host-pinned start a new process per query and reuse it for that query's two
measurements. Linux `drop_caches=3` runs before each process, and the harness's
`--mode isolated` or `--mode grouped` controls Sirius cache clearing. The wrapper
passes no harness `--profile`, so it does not derive cache overrides. Only
Quent settings are added to an effective YAML copy; source YAML is hashed and
checked after every worker. There is no extra warmup or retired `SIRIUS_EXP_*`
environment flag. CPU fallback is disabled for all GPU measurements.

## Run cold and hot

Keep this runner checkout separate from an engine checkout built at
`26b5caba3e2c96290fdc63d4cf40ac42572eaab8`, with its recursive submodules and locked
Pixi environment. Build the engine with `pixi run make` from that checkout.
The launcher uses that engine's Pixi Python, CUDA 13.5 override and library path.
It refuses a different source revision or tracked engine changes.

On the EC2 instance, load kTLS before running:

```bash
sudo modprobe tls
```

From this runner checkout:

```bash
bash bench/g7e/replay.sh 3000 cold \
  --source-dir /path/to/built/sirius-26b5caba \
  --iterations 1 \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf3000_regenerated/ \
  --output /path/to/new-sf3000-cold-run

bash bench/g7e/replay.sh 3000 hot \
  --source-dir /path/to/built/sirius-26b5caba \
  --iterations 2 \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf3000_regenerated/ \
  --output /path/to/new-sf3000-hot-run
```

Add `--dry-run` to preview without a build, AWS access, dataset reads or output
creation. Select a subset with `--queries 1` or `--queries 3,6,9,18`. Cold defaults
to one iteration at SF3000 and two at other scales, matching the saved campaign;
hot and pinned default to two. Every run requires a new output directory.

The runner checks the actual EC2 type and bucket region, all eight tables, and
the complete Parquet inventory against `manifest.json`, including the requested
scale factor. It requires writable local NVMe instance storage with enough free
space for enabled spilling, follows symlinks and checks disk models. It never
formats storage, falls back to EBS, generates data, launches or stops instances.
kTLS is verified from new native TLS connections and native fallback logs for
each direct-S3 worker. Execution timeouts are 180 seconds at SF100/SF300,
300 seconds at SF1000, and 600 seconds at larger scales. Setup has a separate
1800-second timeout; host-pin setup is capped at 600 seconds at SF1000 and
300 seconds otherwise. Interruptions clean up the child process.

## Host pinning and answer checks

At this commit, native `pin_table` cannot read an S3 URI. For the selected
SF100/SF300/SF1000 host-pin runs, byte-identical Parquet files were staged on the
same instance's NVMe before timing. Supply that existing mirror:

```bash
bash bench/g7e/replay.sh 1000 pinned \
  --source-dir /path/to/built/sirius-26b5caba \
  --iterations 2 \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf1000_regenerated/ \
  --host-input /mnt/datasets/pinned-inputs/sf1000 \
  --output /path/to/new-sf1000-pinned-run
```

Before pinning, every local file is checked against the S3 generation manifest's
full-file SHA256 and byte count. The worker pins only that query's selected
columns on host, measures twice, unpins, then exits. Native logs must show every
expected table assigned from its pin. Staging and pin setup are outside query
timing. No data is staged automatically or read across AWS regions. No complete
SF3000 or SF10000 host-pin result is selected in this snapshot.

Use `--references /path/to/cpu-references` to check both iterations against the
independent CPU answers after all GPU measurements. The directory contains
`q01/result.txt` through `q22/result.txt` and each query's `provenance.json` with
`independent_cpu`, `sha256` and `matching_sql_sha256`. Q3 and Q18 also require
`expanded-ties.txt` containing the CPU LIMIT-boundary candidates. These answer
artifacts are retained in the source archives, rather than committed as large
files. Without references, the new report explicitly says `cpu_validated: false`;
GPU execution alone is not labeled answer validation. The checked-in results
all passed independent CPU comparison.

Outputs include both iteration sums and coverage, per-query timings, durable
per-iteration answers, exact configs, command/environment records, native logs,
kTLS proofs and raw Quent traces. Incomplete iterations have no full-suite sum.
Startup, cache clearing, result export and answer checking are outside query
timing. A best run is one complete correct iteration, never mixed query minima.

## Validation of this adapter

The adapter is a portable version of the measured EC2 worker, not another GPU
measurement of the selected results. Its CPU-only checks cover recipe selection,
process scheduling, immutable configs, incomplete sums and dry-run behavior:

```bash
pixi run python -m unittest discover -s bench/g7e/guide-26b5caba -p 'test_*.py' -v
pixi run python -m unittest discover -s bench/common -p 'test_*.py' -v
```
