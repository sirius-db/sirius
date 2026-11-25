# DuckDB 2.0.0 on M9gd, reading S3

The [selected results](results/SELECTED.md) now use the
[November 25 native / 64 KiB measurements](results/20261125-native64k/RESULTS.md)
for SF1000, SF3000 and SF10000. The latter is assembled with retries and **Q12 at
256 GiB**, while other queries use 296.7 GiB. Exact source phases, memory
budgets, kernel/native build evidence, executed scripts and validation proofs
are saved with that snapshot. The historical replay instructions below target
the original November 24 wheel/default-memory baseline.

This recipe reproduces the [November 24, 2026 results](results/20261124/RESULTS.md)
for SF1000, SF3000 and SF10000 on `m9gd.24xlarge`, Amazon Linux 2023 ARM64, in
`us-east-1a`. It uses 96 DuckDB threads and the engine's default `296.7 GiB`
memory limit. The measured instance had three 1900 GB local NVMe disks in an XFS
RAID0 mounted at `/mnt/nvme`; its EBS root disk was excluded.

## Measured protocol

- Read Parquet directly from same-region S3 views. No dataset is staged locally.
- Set `enable_external_file_cache=true` and `parquet_metadata_cache=true`.
- Run `Q1 Q1`, close DuckDB and exit its process, sync/drop Linux page, dentry
  and inode caches, then `Q2 Q2`, and so on through `Q22 Q22`.
- Start 22 distinct query processes per scale, with exactly two timed queries
  in each. Clear OS caches before Q1 too; do not clear between a query's two
  iterations. No extra warmup is executed.
- Time `connection.execute()` only, using the preserved harness helper. Setup,
  S3 view binding, fetching/exporting Python rows, cache inspection and process
  startup are outside the timer. Preflight footer counts finish before the
  first cache clear.
- Spill only to the verified local NVMe volume. Both caches stay enabled, but
  large SF10000 working sets can evict cached data and spill; iteration 2 does
  not imply the complete input is resident in RAM.
- Q11 uses the corresponding scale factor's predicate. All other query SQL is
  identical across scales. The SQL and harness were saved from Sirius main
  `26b5caba3e2c96290fdc63d4cf40ac42572eaab8`.

Sirius is not loaded. The wrapper calls the pinned harness's Python timing and
answer-export helpers, not its command-line entry point: neither `--profile`
nor `--mode` is passed. There is no Sirius YAML or GPU configuration for this run.
Exact effective DuckDB settings for all 66 processes are in the dated per-scale
`config.json` files.

## Prepare the instance

Use an already-running `m9gd.24xlarge` in `us-east-1`, with passwordless sudo for
Linux cache clearing and a writable local NVMe spill mount. The replay requires
at least 4 TiB free on that mount. It never formats disks, substitutes EBS, changes
instance tags, or starts, stops or terminates an instance.

The credential chain must allow reading/listing the measured Virginia bucket.
The actual bucket region is checked with GetBucketLocation, or the region in
AWS's HEAD response when that permission is absent. No query reads S3 across
regions. The datasets must match their saved generation-manifest fingerprints:

| Scale | Prefix in `sae-dataproc-aws-dev-sirius-scratch-us-east-1` | Parquet files |
| --- | --- | ---: |
| SF1000 | `datasets/tpch_sf1000/` | 90 |
| SF3000 | `datasets/tpch_sf3000_regenerated/` | 263 |
| SF10000 | `datasets/tpch_sf10000_regenerated/` | 868 |

Install the saved Python 3.11 environment and DuckDB's matching extensions on
the instance, from this checkout:

```bash
python3.11 -m venv /home/ec2-user/duckdb200-venv
/home/ec2-user/duckdb200-venv/bin/python -m pip install \
  -r bench/m9gd/duckdb-2.0-s3/requirements.txt
/home/ec2-user/duckdb200-venv/bin/python -c \
  "import duckdb; duckdb.connect().execute('INSTALL httpfs; INSTALL aws;')"
```

## Replay

Preview needs only standard-library Python; it does not import DuckDB, contact
AWS, clear caches, or create output:

```bash
bash bench/m9gd/duckdb-2.0-s3/reproduce.sh 1000 \
  --output /home/ec2-user/new-duckdb-sf1000 --dry-run
```

Run inside a detached terminal session so an SSH disconnect does not interrupt
the benchmark. Set `PYTHON_BIN` to the saved environment's Python:

```bash
export PYTHON_BIN=/home/ec2-user/duckdb200-venv/bin/python
bash bench/m9gd/duckdb-2.0-s3/reproduce.sh 1000 \
  --output /home/ec2-user/new-duckdb-sf1000
bash bench/m9gd/duckdb-2.0-s3/reproduce.sh 3000 \
  --output /home/ec2-user/new-duckdb-sf3000
bash bench/m9gd/duckdb-2.0-s3/reproduce.sh 10000 \
  --output /home/ec2-user/new-duckdb-sf10000
```

Each invocation requires a new output directory. `--spill-root` may select
another existing, verified local NVMe directory. Preflight checks EC2 identity,
S3 region, the complete manifest inventory, all eight Parquet footer row counts,
DuckDB version/default memory, and physical spill-disk models before Q1.

The wrapper stages the original measured controller and worker plus four shared
Python modules. Its only worker edit is a unique spill-directory prefix, recorded
alongside both source hashes; query SQL, timing calls, iteration ordering, cache
clears and timeouts remain unchanged. It rechecks inventory and staged sources
after the benchmark. The original controller limits SF1000 pairs to 1800 seconds;
SF3000/SF10000 pairs use 3600 seconds and each scale has a 14400-second cap. These
limits affect benchmark processes, never the instance. Incomplete suites keep
partial measurements and have no full-suite sum.

Outputs include `campaign.log`, pre/postflight evidence, a replay plan with source
hashes, and `runner/results/` with native logs, actual settings, SQL, answers,
cache/lifecycle journals, per-query CSV and both complete iteration sums.
The timing replay does not bundle the large independent CPU reference answers
or automatically certify new answers. The saved measurements were independently
validated before publication; their proof records are retained with the results.

## Exact sources and saved proofs

`measured/shared/` contains the four byte-identical helper modules shared by all
three measurements. `measured/sf<SF>/` contains the original controller, worker
and host-specific launch/check/archive scripts, with SHA256s in
`measured/source-sha256.json`. Use the portable wrapper above rather than directly
executing those historical `run.sh` files, which retain their original paths.
The original SF1000 GetBucketLocation preflight failure occurred before timing;
the archived corrected launch used an independently AWS-verified region receipt.
The replay uses the existing shared live-region checks instead of that receipt.

The [dated results](results/20261124/RESULTS.md) preserve all 132 timings and
independent validation records, all 66 process lifetimes, settings, source/SQL
fingerprints, cache evidence and full-archive checksums. Full native logs and
answer files remain in the locally retained, verified archives; large raw
archives are not committed to Git. No benchmark is rerun by the offline audit:

```bash
pixi run python bench/m9gd/duckdb-2.0-s3/verify_saved.py
```
