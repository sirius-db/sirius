# StarRocks and Sirius on NVIDIA GB10

These helpers adapt the transferred RTX notes to Ubuntu 24.04 aarch64, one
NVIDIA GB10 (SM121), and 119 GiB shared CPU/GPU memory. The current build uses
`all22/integration` at `95bec853b684e1510c7ddb3d9becc9b73374e983`.

## Build

Run from the repository root:

```bash
bash scripts/local-gb10/build-engine.sh
bash scripts/local-gb10/build-transport.sh
pixi run --frozen bash experimental/starrocks/scripts/apply-starrocks-patches.sh
bash scripts/local-gb10/build-cn2.sh
```

The engine uses the frozen Pixi environment, CUDA 13.2, and `CUDAARCHS=121-real`.
The transport helper installs CUDA-enabled UCX 1.21.0 and NIXL 1.3.2 under
`build/local-gb10/transport/`. The CN helper links the matching engine and real
NIXL libraries with the system C++ linker; it refuses the NIXL stub fallback.
These paths keep generated dependencies local to this checkout.

The previously built StarRocks FE package is reusable because its source commit
is unchanged. To rebuild it:

```bash
cd experimental/starrocks
MAVEN_ARGS="--settings $(pwd)/../../scripts/local-gb10/maven-settings.xml" \
  pixi run --frozen -e fe fe-build
cd ../..
```

The Maven settings use Google's Maven Central mirror. The FE was built with
JDK 17.0.18 and is packaged at `experimental/starrocks/starrocks/output/fe`.

## SF100 data

```bash
bash scripts/local-gb10/generate-sf100.sh
```

The dataset is `test_datasets/tpch_parquet_sf100/`: 14 Parquet shards, 25.779 GiB,
including exactly 600,037,902 lineitems. `generation-manifest.json` records the
pinned generator (`cdcf74d`), dependency lock, options, schemas, counts, and file
hashes. Verification decodes every column and row. The helper validates existing
output and does not silently accept a partial dataset.

## Start one FE and two CNs on the same GPU

```bash
export TOOLS_DIR="$PWD/build/local-gb10/transport"
source experimental/starrocks/scripts/cn-env.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TOOLS_DIR/toolenv/.pixi/envs/default/lib:/usr/lib/aarch64-linux-gnu"
export RUST_LOG=info
bash scripts/local-gb10/stack.sh --cn-count 2 \
  --sirius-config scripts/local-gb10/sirius-sf100-2cn.yaml \
  --run-dir build/starrocks-sf100-2cn --timeout 600
```

Both CNs use GPU 0, with separate working directories and port blocks
9100–9104 and 9110–9114. Each has a 24 GiB GPU pool, up to 8 GiB host memory,
and a separate 2 GiB exchange staging arena. The FE heap is 2 GiB. The launcher
waits for exactly two alive CNs, no alive backends, and NIXL sessions in both
directions. MySQL listens on port 9030. Ctrl-C stops the owned processes;
`launcher.pid` identifies this launcher while it is running.

CN listeners bind to loopback. This FE advertises loopback but its HTTP, MySQL,
and Thrift listeners bind all interfaces. The launcher checks ports before
starting and uses an isolated FE metadata directory.

## Run all 22 queries

From another terminal at the repository root:

```bash
pixi run --frozen python scripts/local-gb10/tpch-starrocks.py \
  --data-dir test_datasets/tpch_parquet_sf100 \
  --run-dir build/tpch-starrocks-sf100-2cn \
  --expected-cns 2 --scale-factor 100 \
  --timeout 600 --oracle-timeout 600 \
  --oracle-memory-limit 8GB --oracle-threads 4 \
  --set cbo_cte_reuse_rate=0 --reuse-oracle --stop-on-error \
  --cn-binary experimental/starrocks/target/release/sirius-starrocks-cn
```

Use `--prepare-only` to prepare CPU references without connecting to StarRocks,
or `--queries q6` to select a query. Inputs use `FILES()` CTEs over local Parquet;
no StarRocks database or external storage volume is required. The report records
Q8/Q9's documented join ordering, Q11's SF-scaled threshold, and Q22's equivalent
comma-form `substring`. Every adjustment is applied to both engines.

Results are compared as multisets, retaining duplicates. Numeric tolerances
are explicit in the report; integers, text, and NULLs must match exactly. Output
ordering is not verified. Elapsed times include MySQL client launch and result
transfer; this is a correctness run, not a formal TPC-H performance benchmark.

`--stop-on-error` stops after a SQL error, timeout, or unhealthy topology so the
owned stack can be restarted before subsequent queries. It continues after
numeric mismatches. Engine cancellation cannot interrupt a fragment already
running; a client deadline alone does not restore a wedged CN.

## SF100 validation

On 2026-09-05, all 22 queries passed the DuckDB comparison with two CNs on one
GB10: 138,812 result rows checked. Q2–Q22 matched exactly after numeric
normalization; Q1's largest relative difference was `8.68e-8`, within `1e-6`.
The measured query times totaled 64.665 seconds. See the
[run summary](../../build/tpch-starrocks-sf100-summary.md) and
[per-query results](../../build/tpch-starrocks-sf100-2cn/report.md).

## Earlier validation

The earlier `dev` build and single-CN demo remain documented in
`build/tpch-starrocks-1cn-summary.md`. The separate demo checkout at
`/home/aocsa/git/sirius-worktrees/starrocks-1cn` remains available. It predates
the decimal and aggregate fixes in the current integration branch.

## SF500: standalone Sirius and two CNs

Generate and fully verify the larger dataset, then prepare shared CPU references:

```bash
pixi run --frozen bash scripts/local-gb10/generate-tpch.sh 500
pixi run --frozen python scripts/local-gb10/tpch-starrocks.py \
  --data-dir test_datasets/tpch_parquet_sf500 \
  --run-dir build/tpch-starrocks-sf500-2cn \
  --expected-cns 2 --scale-factor 500 --prepare-only \
  --timeout 1800 --oracle-timeout 1800 \
  --oracle-memory-limit 8GB --oracle-threads 4 \
  --set cbo_cte_reuse_rate=0
```

Run standalone while the StarRocks CNs are stopped, so it has the GPU to itself:

```bash
pixi run --frozen python scripts/local-gb10/tpch-standalone.py \
  --reference-dir build/tpch-starrocks-sf500-2cn \
  --run-dir build/tpch-sirius-sf500-standalone \
  --config scripts/local-gb10/sirius-standalone.yaml --timeout 1800
```

The standalone helper uses the canonical performance runner's connection and
timing functions with exactly the prepared SQL and input fingerprints. It
disables whole-query DuckDB fallback and checks GPU execution start/completion
logs for every query. The single process has a 48 GiB GPU budget, a 16 GiB host
budget and a 1 TiB disk spill limit. Output directories must be fresh.

After standalone exits, start the two CN stack in one terminal:

```bash
export TOOLS_DIR="$PWD/build/local-gb10/transport"
source experimental/starrocks/scripts/cn-env.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TOOLS_DIR/toolenv/.pixi/envs/default/lib:/usr/lib/aarch64-linux-gnu"
export RUST_LOG=info SIRIUS_QUERY_WATCHDOG_SECS=1800 SIRIUS_CN_RPC_TIMEOUT_SECS=1800
bash scripts/local-gb10/stack.sh --cn-count 2 \
  --sirius-config scripts/local-gb10/sirius-sf500-2cn.yaml \
  --run-dir build/starrocks-sf500-2cn --timeout 600
```

Each CN has a 24 GiB GPU budget, 8 GiB host budget, 2 GiB exchange staging,
and 512 GiB disk spill limit. The launcher gives each CN its own working
directory and creates its relative `spill` directory before startup.

Run the StarRocks queries in another terminal, reusing the CPU references:

```bash
pixi run --frozen python scripts/local-gb10/tpch-starrocks.py \
  --data-dir test_datasets/tpch_parquet_sf500 \
  --run-dir build/tpch-starrocks-sf500-2cn \
  --expected-cns 2 --scale-factor 500 \
  --timeout 1800 --oracle-timeout 1800 \
  --oracle-memory-limit 8GB --oracle-threads 4 \
  --set cbo_cte_reuse_rate=0 --reuse-oracle --stop-on-error \
  --cn-binary experimental/starrocks/target/release/sirius-starrocks-cn
pixi run --frozen python scripts/local-gb10/cn-activity.py --dir build/starrocks-sf500-2cn
```

Both reports use duplicate-preserving result comparison with the same numeric
tolerances. Standalone timings cover SQL execution and fetching; StarRocks
timings also include MySQL client startup and transfer. Neither run resets the
OS cache or constitutes a formal TPC-H benchmark.

The SF500 run on 2026-09-05 passed all 22 queries in standalone Sirius
(244.319 seconds; all 567,682 result rows exact after numeric normalization).
StarRocks with two CNs passed 15 of 22. Q5, Q7, Q8, Q9, Q17, Q18 and Q21
exhausted the 24 GiB per-CN GPU pools: retained exchange data could not spill.
Every query was attempted once, with a clean cluster restart after each failure.
See [the SF500 results](../../build/tpch-sf500-summary.md), including per-query
timings, failure details, and the attempt ledger. `--queries` can select the
remaining queries when continuing after a failure; keep the original results.
