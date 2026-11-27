# Core-extension RC blog: cold-S3 TPC-H settings

Exact settings used for the November 2026 Sirius and DuckDB comparison at
SF100, SF1K, SF3K, and SF10K. These are measured recipes for the hardware below;
they do not change the engine defaults or establish that each setting is
independently necessary.

## Versions, hardware, and input

- Sirius: `e6a43ebcde9d04524a80aa059e06727c03da578d`, CUDA 12 Pixi environment,
  NVIDIA driver `595.91.07`, Ubuntu 24.04, kernel `7.0.0-1014-aws`.
- DuckDB: `cb5673685153159d22bf1042ab444273c9ad268a`, reporting `v2.0.0-dev1`.
  This is development source, not a tagged final 2.0.1 release.
- Sirius uses g7e.8xlarge (one 96-GB RTX PRO 6000 Blackwell Server Edition GPU,
  256 GiB host RAM) or g7e.12xlarge (two GPUs, 512 GiB host RAM) at every scale.
- DuckDB uses m9gd.8xlarge / 16xlarge / 24xlarge / 48xlarge at
  SF100 / SF1K / SF3K / SF10K, respectively, in both regimes.
- All input is direct S3 Parquet in `us-east-2`, with one directory per table
  (`lineitem/*.parquet`, `orders/*.parquet`, etc.). Local NVMe is scratch/spill,
  not a copy of the input. Both engines use the same objects at each scale.
  Exact performance also depends on schema and Parquet file/row-group layout;
  newly generated TPC-H data need not be performance-equivalent. The measured
  input objects are not distributed here.

## Sirius settings

Apply the YAML at startup and the SQL after loading Sirius. Paths are relative
to this directory. Defaults omit both the YAML and the extra SQL.

| Instance | Scale | Startup YAML | Session SQL |
| --- | --- | --- | --- |
| g7e.8xlarge | SF100, SF3K | [tune8-cpu-budget.yaml](configs/tune8-cpu-budget.yaml) | None |
| g7e.8xlarge | SF1K | [tune8-cpu-budget.yaml](configs/tune8-cpu-budget.yaml) | [sirius-8xl-sf1k.sql](configs/sirius-8xl-sf1k.sql) |
| g7e.8xlarge | SF10K | [tune8-sf10k-balanced-large-scans.yaml](configs/tune8-sf10k-balanced-large-scans.yaml) | [sirius-8xl-sf10k.sql](configs/sirius-8xl-sf10k.sql) |
| g7e.12xlarge | SF100, SF1K, SF3K | [tune12-proven-default-operators.yaml](configs/tune12-proven-default-operators.yaml) | [sirius-12xl.sql](configs/sirius-12xl.sql) |
| g7e.12xlarge | SF10K | [tune12-sf10k-host432-cache-watermarks.yaml](configs/tune12-sf10k-host432-cache-watermarks.yaml) | [sirius-12xl.sql](configs/sirius-12xl.sql) |

The presets adjust concurrency, host/GPU memory budgets, reader/cache behavior,
and spill compression. The two-GPU SF10K preset reserves 432 GiB of host memory;
use it only on a machine with sufficient RAM. Verify free NVMe space for the YAML
spill budgets. Both regimes use all accessible GPUs and full instance RAM,
unlimited locked memory, and 1,048,576 file handles.

Build the pinned revision in a separate checkout, then copy this recipe directory
into it. Pixi and build prerequisites must already be installed:

```bash
git clone https://github.com/sirius-db/sirius.git sirius-repro
cd sirius-repro
git checkout --detach e6a43ebcde9d04524a80aa059e06727c03da578d
git submodule update --init --recursive --depth 1 --jobs 5 \
  duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run -e cuda12 env CUDAARCHS=120-real make TEST_BUILD_TARGET= -j24
```

Example: tuned two-GPU SF10K, from that checkout. Configure AWS credentials for
read access to your input, create a writable spill directory, and ensure the
shell's hard limits permit the limits below. Set `CPU_RESULTS` to the timestamped
CPU output directory produced by the first invocation:

```bash
REPRO_DIR="$PWD/bench/core-extension-rc-2026-11"
TPCH_INPUT='s3://YOUR-BUCKET/tpch_sf10000/'
mkdir -p /opt/dlami/nvme/spill
ulimit -l unlimited
ulimit -n 1048576
unset SIRIUS_CONFIG_FILE SIRIUS_PRE_SQL SIRIUS_PIN_TIER CUDA_VISIBLE_DEVICES
pixi run -e cuda12 python test/performance/performance_test.py \
  --input "$TPCH_INPUT" --scale-factor 10000 --engine cpu \
  --iterations 1 --mode isolated --output results/cpu-sf10000
CPU_RESULTS='/absolute/path/to/results/cpu-sf10000/tpch_TIMESTAMP_RUN'
SIRIUS_PRE_SQL="$(cat "$REPRO_DIR/configs/sirius-12xl.sql")" \
  pixi run -e cuda12 python test/performance/performance_test.py \
  --input "$TPCH_INPUT" --scale-factor 10000 --engine gpu \
  --iterations 1 --mode isolated \
  --config "$REPRO_DIR/configs/tune12-sf10k-host432-cache-watermarks.yaml" \
  --duckdb-results "$CPU_RESULTS" --output results/tuned-12xl-sf10000
```

For defaults, clear the same tuning variables and omit `SIRIUS_PRE_SQL` and
`--config`. The harness enables GPU execution in both regimes. The CPU invocation
above captures reference answers; it is not the separately built CPU comparison.

## DuckDB settings

Stock defaults use a generic Release build and a 4-KiB-page kernel, with no
performance overrides. The best tested CPU setup retains defaults at SF1K,
where defaults beat the optional tuning recipe.

| Scale | Best tested CPU build / page size | SQL |
| --- | --- | --- |
| SF100 | Native Release / 64 KiB | [duckdb-sf100.sql](configs/duckdb-sf100.sql) |
| SF1K | Generic Release / 4 KiB | Defaults, no performance overrides |
| SF3K | Native Release / 64 KiB | [duckdb-sf3000.sql](configs/duckdb-sf3000.sql) |
| SF10K | Native Release / 64 KiB | [duckdb-sf10000.sql](configs/duckdb-sf10000.sql) |

The native build adds `-DNATIVE_ARCH=ON` to this Release configuration. Build on
Graviton5; omit that flag for the generic build:

```bash
git clone https://github.com/duckdb/duckdb.git duckdb-repro
cd duckdb-repro
git checkout --detach cb5673685153159d22bf1042ab444273c9ad268a
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_UNITTESTS=OFF -DBUILD_SHELL=ON -DNATIVE_ARCH=ON \
  -DLINK_CORE_EXTENSIONS=ON -DBUILD_EXTENSIONS='parquet;json;tpch;httpfs' \
  -DDUCKDB_EXTENSION_CONFIGS="$PWD/.github/config/extensions/httpfs.cmake"
cmake --build build/native --target duckdb --parallel 48
```

Tuned hosts boot `7.0.0-1014-aws-64k` instead of `7.0.0-1014-aws`.
Verify `getconf PAGESIZE` returns `65536` for native recipes and `4096` for
stock defaults. This kernel change requires a reboot; SQL alone does not
reproduce the tuned environment. No NIC/IRQ/TCP tuning was applied.

For each CPU query, start a fresh CLI process, load `httpfs` and `parquet`,
configure S3 credentials, create table views over the same input objects, and
apply the selected SQL before enabling `.timer on`. For example:

```sql
LOAD httpfs;
LOAD parquet;
CREATE VIEW lineitem AS SELECT * FROM read_parquet('s3://YOUR-BUCKET/tpch_sf10000/lineitem/*.parquet');
-- Create the other seven TPC-H table views in the same way.
.read /absolute/path/to/configs/duckdb-sf10000.sql
SET temp_directory = '/absolute/path/on/local-nvme/spill';
-- Set max_temp_directory_size to 85% of the currently free scratch bytes.
.mode csv
.headers on
.nullvalue ""
.timer on
-- Run the selected TPC-H query and save its CSV output for validation.
```

Set the same scratch directory/free-space budget for defaults and tuned CPU
runs. Raise the file-handle limit to 1,048,576. Before each query, the campaign
ran `sync` and wrote `3` to `/proc/sys/vm/drop_caches` on a dedicated host.
Run one benchmark at a time on each host.

## Measurement

- Each time is the sum of all 22 queries from one correct complete suite.
  Report the best whole suite; never combine individual query minima.
  Run three independent suites. The recorded tuned one-GPU SF10K result has
  only one correct pass; stock DuckDB SF3K also has one.
- Sirius clears input/prefetch caches per query, retaining process/JIT/kernel
  state and REST connections; an SF100 warmup precedes measurement. CPU uses a
  fresh process and dropped OS cache per query. No GPU table pinning is used.
- Sirius times execute/materialization and excludes fetch/export. CPU CLI
  timing includes CSV output. Both exclude startup and correctness checking.
  Compare every query with independent CPU answers, including legal
  LIMIT-boundary ties in Q3/Q18; the pinned harness alone does not implement
  the expanded tie check used by the campaign.
- Per-query limits are 180/300/600/600 seconds at SF100/SF1K/SF3K/SF10K,
  enforced by an external supervisor. The Sirius runner's `--query-timeout`
  applies only to profiler/debugger subprocess modes. DNF has no numeric suite
  runtime/cost: at SF10K, defaults failed for DuckDB and both Sirius sizes.
- Query-time compute cost is whole-suite seconds multiplied by instance USD/hour
  divided by 3,600. The recorded rates were $5.26824/$8.28608 for g7e.8xlarge/12xlarge
  and $2.0112/$4.0224/$6.0336/$12.0672 for m9gd.8xlarge/16xlarge/24xlarge/48xlarge.
  These exclude startup, validation, storage, S3, idle time, tax, and discounts.
