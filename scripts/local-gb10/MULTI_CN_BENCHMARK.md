# Reproduce the SF500 multi-CN comparison on GB10

This is a local Ubuntu 24.04 aarch64 adaptation for one NVIDIA GB10 (SM121)
with about 119 GiB of shared CPU/GPU memory. It runs one StarRocks FE and two
Sirius CN processes on physical GPU 0. It measures this topology; it cannot
establish scaling across GPUs or a remote fabric. Current implementation and validation
status are in [STATUS.md](../../STATUS.md); this guide makes no timing claim.

The comparison uses baseline source
`95bec853b684e1510c7ddb3d9becc9b73374e983` and the optimized implementation.
`SIRIUS_EXCHANGE_OPTIMIZED` is off by default; only the exact value `1` enables
it. The harness sets `0` for baseline and `1` for optimized. The new protocol
requires homogeneous participating CNs. It retains the receiver EOS barrier,
waits for ingress copying before publication acknowledgment, and synchronizes
the independent packing stream before transfer. Oversized frames fail
explicitly. Peer epoch changes have no in-place recovery: stop and restart
both CNs. See [CN tunables](../../experimental/starrocks/docs/TUNABLES.md).

Asynchronous admission of sender-only fragments is a separate, existing switch.
The harness leaves it off unless `--async-sender-dispatch` is supplied to
`benchmark-multi-cn.py`. Enabling optimized exchange alone does not change
that setting.

## Prepare the build environment

Run commands from the optimized repository root unless a block changes
directory. Stop other GPU queries and build jobs first. Required host tools
include Pixi, Git, system GCC/G++, CUDA with SM121 support, NVIDIA driver tools,
curl, tar, and binutils. The engine helper targets `CUDAARCHS=121-real` and uses
the repository's frozen Pixi environment. The transport helper defaults to
`/usr/local/cuda-13.0`; set `CUDA_HOME` to an installed compatible toolkit when
that path differs. Its UCX/NIXL builds use the system compiler.

```bash
export BENCHMARK_REPO="$PWD"
unset CUDA_VISIBLE_DEVICES
git submodule update --init --recursive
pixi install --frozen
pixi install --frozen --manifest-path experimental/starrocks/pixi.toml
pixi install --frozen --manifest-path experimental/starrocks/pixi.toml -e fe
bash scripts/local-gb10/build-transport.sh
```

`build-transport.sh` installs pinned UCX 1.21.0 and NIXL 1.3.2 under
`build/local-gb10/transport`. It verifies source identities and checks native
CUDA transport dependencies. It creates a separate tools environment whose
initial package resolution is not pinned by the repository lock. Preserve its
generated `toolenv/pixi.toml` and `toolenv/pixi.lock` with the run provenance.
Do not update that environment, the repository environments, CUDA, or the
transport installations between comparison arms. Frozen CN/engine files still
use these shared dynamic libraries; they are not a self-contained deployment.

Build the common FE with the checked-in patches and Maven mirror settings:

```bash
pixi run --frozen bash experimental/starrocks/scripts/apply-starrocks-patches.sh
(
  cd experimental/starrocks
  MAVEN_ARGS="--settings $BENCHMARK_REPO/scripts/local-gb10/maven-settings.xml" \
    pixi run --frozen -e fe fe-build
)
```

The launcher expects the package at
`experimental/starrocks/starrocks/output/fe`, the MySQL client in that project's
default Pixi environment, and Java in its `fe` environment. `MYSQL_BIN` and
`GB10_JAVA_HOME` can override the client and Java paths.

## Freeze baseline before building optimized

Capture the baseline **before** an optimized build replaces the normal build
outputs. If verified baseline artifacts already exist, preserve them and skip
their rebuild. Do not overwrite an existing `baseline-bin` or `optimized-bin`;
keep each earlier artifact set and its results together.

For a fresh reproduction, build baseline in an isolated checkout. The old
commit does not contain these local helpers, so copy only the build wrappers
into that checkout. Both builds use the same native transport installation.

```bash
export BASELINE_REPO="$BENCHMARK_REPO/../sirius-multi-cn-baseline"
git worktree add --detach "$BASELINE_REPO" \
  95bec853b684e1510c7ddb3d9becc9b73374e983
git -C "$BASELINE_REPO" submodule update --init --recursive
mkdir -p "$BASELINE_REPO/scripts/local-gb10"
cp scripts/local-gb10/build-engine.sh scripts/local-gb10/build-cn2.sh \
  "$BASELINE_REPO/scripts/local-gb10/"
(
  cd "$BASELINE_REPO"
  pixi install --frozen
  pixi install --frozen --manifest-path experimental/starrocks/pixi.toml
  pixi run --frozen bash experimental/starrocks/scripts/apply-starrocks-patches.sh
  bash scripts/local-gb10/build-engine.sh
  TRANSPORT_ROOT="$BENCHMARK_REPO/build/local-gb10/transport" \
    bash scripts/local-gb10/build-cn2.sh
)
mkdir -p build/multi-cn-throughput
mkdir build/multi-cn-throughput/baseline-bin
cp -L "$BASELINE_REPO/build/release/extension/sirius/libsirius.so" \
  build/multi-cn-throughput/baseline-bin/libsirius.so
cp "$BASELINE_REPO/experimental/starrocks/target/release/sirius-starrocks-cn" \
  build/multi-cn-throughput/baseline-bin/
ln -s libsirius.so build/multi-cn-throughput/baseline-bin/sirius.duckdb_extension
```

Now build and freeze the optimized checkout:

```bash
bash scripts/local-gb10/build-engine.sh
bash scripts/local-gb10/build-cn2.sh
mkdir build/multi-cn-throughput/optimized-bin
cp -L build/release/extension/sirius/libsirius.so \
  build/multi-cn-throughput/optimized-bin/libsirius.so
cp experimental/starrocks/target/release/sirius-starrocks-cn \
  build/multi-cn-throughput/optimized-bin/
ln -s libsirius.so build/multi-cn-throughput/optimized-bin/sirius.duckdb_extension
for arm in baseline optimized; do
  (
    cd "build/multi-cn-throughput/$arm-bin"
    sha256sum libsirius.so sirius-starrocks-cn > SHA256SUMS
    readelf -d libsirius.so | rg SONAME
  )
done
```

The `sirius.duckdb_extension -> libsirius.so` alias is required: the engine's
ELF SONAME is `sirius.duckdb_extension`. Copying `libsirius.so` alone can load
the engine from another search path or fail startup. The harness prepends the
selected artifact directory and checks each running CN's `/proc/PID/maps` for
the intended engine. It saves executable paths, library maps, and binary hashes.
Keep baseline and optimized source identities, the repository lockfiles,
transport tool lockfiles, FE package identity, and compiler/toolkit versions
with the artifacts. Do not rebuild or replace artifacts during a sweep.

## Generate SF500 and prepare one CPU oracle

The generator is a separate checkout, not a Sirius submodule. On a fresh
machine, create it at the following path and pinned commit:

```bash
mkdir -p test_datasets
git clone https://github.com/sirius-db/tpchgen-rs.git test_datasets/tpchgen-rs
git -C test_datasets/tpchgen-rs checkout --detach \
  cdcf74def0072f94bf1886667e8d2ac51feb8721
pixi run --frozen env RUSTFLAGS='-C target-cpu=native' \
  cargo build --manifest-path test_datasets/tpchgen-rs/Cargo.toml \
  --release --locked -p tpchgen-cli -j 4
pixi run --frozen bash scripts/local-gb10/generate-tpch.sh 500
```

For an existing pinned checkout, omit clone/checkout. Generation requires a
fresh output directory or a completed generation manifest. The verifier reads
every Parquet column/page, checks schemas and exact counts, hashes all files,
and preserves the generator lockfile. Its independent lineitem counter needs
`verify-sf100.py` and `tpch-lineitem-count.rs` despite the SF100 verifier name.
Allow enough disk for the dataset, per-CN spill, binaries, and retained logs.

Prepare references while no benchmark stack is running:

```bash
pixi run --frozen python scripts/local-gb10/tpch-starrocks.py \
  --data-dir test_datasets/tpch_parquet_sf500 \
  --run-dir build/tpch-starrocks-sf500-2cn \
  --expected-cns 2 --scale-factor 500 --prepare-only \
  --timeout 1800 --oracle-timeout 1800 \
  --oracle-memory-limit 8GB --oracle-threads 4 \
  --set cbo_cte_reuse_rate=0 --reuse-oracle
```

Both arms use this exact reference directory and dataset location. The runner
reuses an oracle only when its SQL/input fingerprint matches. SQL comes from
`test/tpch_performance/queries.py`, with the same documented Q8/Q9 join-order,
Q11 scale, and Q22 syntax adjustments in both engines. Comparison preserves
duplicates; numeric tolerances are `1e-6` relative and `1e-8` absolute. Integers,
text, and NULLs must match. Result ordering is not checked.

## Run alternating A/B blocks

Initialize the native transport runtime in the shell that launches the sweep:

```bash
export TOOLS_DIR="$BENCHMARK_REPO/build/local-gb10/transport"
source experimental/starrocks/scripts/cn-env.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TOOLS_DIR/toolenv/.pixi/envs/default/lib:/usr/lib/aarch64-linux-gnu"
unset CUDA_VISIBLE_DEVICES
export SIRIUS_CN_NIXL_WARMUP=on
pixi run --frozen python scripts/local-gb10/sweep-multi-cn.py \
  --output results/multi-cn-throughput-ab --repetitions 3
```

Use a fresh output path for each new experiment. `--queries q01 q06 q05` selects
a smaller sweep. A block runs one query on one fresh cluster: the first sample
is application-cold, followed by two warm samples. Arm order alternates by
query: baseline/optimized for Q1, optimized/baseline for Q2, and so on. Transport
warmup precedes timing; the OS page cache is uncontrolled. A failed sample or
an execution made ineligible by retry placement is retained, its remaining
warm slots are marked skipped, and the cluster is stopped before the next
block. There is no automatic resume into an existing output directory.

The harness requires exactly one physical GPU, takes an exclusive advisory
lock at `/tmp/sirius-benchmark-GPU_UUID.lock`, and refuses an existing GPU
compute process. The lock coordinates cooperating harnesses; avoid other
GPU workloads, builds, and telemetry analysis during timed runs. The launcher
checks its ports and verifies exactly two live CNs and zero live backends.
It also checks topology before and after each timed query. Both CNs use GPU 0;
the launcher intentionally removes `CUDA_VISIBLE_DEVICES`.

Registered-node availability does not establish actual query placement. An FE
deployment timeout can blacklist a still-live CN and retry on the other CN;
the result can match the oracle while failing the intended distributed
comparison. The shared `execution_validation.py` helper therefore reads FE
retry/query-ID transitions, blacklist history, and per-CN fragment completion
records. It follows the final FE execution rather than pooling fragments from
an unsuccessful initial execution with the successful retry. Administrative
queries and engine-internal query UUIDs are not substitutes for the timed
query's FE identity.

The harness records this audit as `execution_validation`, separately from the
raw SQL status, oracle comparison, and elapsed time. A proven distributed
execution that retries and completes on fewer CNs is `INELIGIBLE`, with failure
class `DEGRADED_RETRY_TOPOLOGY`. An observed retry whose final placement cannot
be resolved is `UNKNOWN` / `EXECUTION_TOPOLOGY_UNKNOWN` and is also excluded
from benchmark aggregates. The harness marks such samples
`benchmark_eligible=false` and stops that cluster even if raw correctness is
`PASS`. The analyzer applies the same helper to archived attempts, preserving
their original results while excluding ineligible samples from completion and
timing comparisons.

A plan that legitimately uses one CN without a retry is not automatically
rejected. Missing execution logs without an observed retry remain explicitly
`UNKNOWN`; the current helper preserves their historical eligibility, so that
case must not be described as verified participation by both CNs. Review the
recorded initial/final CN sets, retry chains, and missing-evidence fields along
with the topology checks.

| Resource or setting | Value |
|---|---|
| GPU allocation per CN | 24 GiB |
| Host allocation per CN | 8 GiB |
| Exchange arena per CN | 2 GiB |
| Disk spill limit per CN | 512 GiB, separate relative `spill` directory |
| FE heap | 2 GiB |
| FE session settings | `pipeline_dop=1`, `cbo_cte_reuse_rate=0` |
| Optimized transfer window | 2 active frames per peer |
| Asynchronous sender dispatch | Off in the controlled A/B sweep |
| Engine watchdog | 300 seconds |
| Peer control/retry deadline | 900 seconds |
| SQL client deadline | 1200 seconds; FE query timeout is 1198 seconds |

The watchdog is deliberately shorter than the client deadline. Classify
watchdog, control, client, and WRITE timeouts separately. A client timeout does
not interrupt an already-running engine fragment; the harness stops its owned
cluster after failure and preserves the failure evidence. GPU/host/staging
budgets share the GB10's physical memory. See the copied per-block
`sirius-config.yaml` and runtime logs for actual configuration.

The SQL timing includes MySQL client startup and result transfer. It excludes
cluster startup, EXPLAIN COSTS, CPU comparison, and restart time. The manifest
retains every planned sample, raw status, SQL/input identities, source snapshot,
configuration, runtime binary identity, and per-CN logs. Failed query durations
never count as successful throughput.

## Run a separate asynchronous sender-dispatch experiment

The FE fragment deployment RPC has its own deadline, independent of the SQL
client and engine watchdog. A long sender that executes before acknowledging
deployment can reach that deadline even when exchange memory is available.
The existing asynchronous sender-dispatch switch acknowledges after enqueueing
sender-only work. Test it separately using the same frozen binaries and budgets:

```bash
pixi run --frozen python scripts/local-gb10/benchmark-multi-cn.py \
  --arm optimized --output results/multi-cn-throughput-q21-async-dispatch \
  --queries q21 --repetitions 3 --transfer-window 2 --async-sender-dispatch
pixi run --frozen python scripts/local-gb10/benchmark-multi-cn.py \
  --arm optimized --output results/multi-cn-throughput-q09-async-dispatch \
  --queries q09 --repetitions 3 --transfer-window 2 --async-sender-dispatch
```

The flag defaults to false and explicitly sets
`SIRIUS_CN_ASYNC_SENDER_DISPATCH=1` only for this experiment; without the flag,
the harness explicitly sets `0`, even if the parent shell exported `1`.
The manifest records `async_sender_dispatch`. `sweep-multi-cn.py` does not pass
the flag and keeps the original controlled protocol. This is a run setting,
not a change to the optimized feature's default semantics.

Use a fresh diagnostic directory for each block, keep the original A/B
manifest and every Q09/Q21 attempt, and do not replace either frozen artifact
set. The original controlled optimized sweep has **20/22 eligible queries**:
its raw correctness count is 21/22, but Q09's returned answers came from
single-CN execution after an FE blacklist/retry. Raw correctness success does
not make those samples valid evidence for the distributed comparison.

Successful Q09 or Q21 diagnostics under asynchronous sender dispatch remain
separate results under a changed setting. They cannot be combined with the
original sweep to claim one uniformly configured all-22 success. Validate the
final query's CN participation and retries in each diagnostic, including warm
samples that may inherit blacklist state. These commands specify experiments,
not asserted outcomes or a complete suite result.

## Collect a separate credit profile and analyze after timing

Run verbose credit diagnostics in a separate output after the main sweep:

```bash
pixi run --frozen python scripts/local-gb10/benchmark-multi-cn.py \
  --arm optimized --output results/multi-cn-throughput-credit-profile \
  --queries q05 --repetitions 1 --transfer-window 2 \
  --log-filter 'info,sirius_starrocks_cn::exchange_protocol=debug,sirius_starrocks_cn::compute_node_service=debug'
```

DEBUG adds per-frame grant, copy completion, and credit-return records. INFO
already contains per-query retirement accounting. Keep this profile out of the
timing comparison because log volume changes its workload. Cold lazy peer
setup can be checked separately by setting `SIRIUS_CN_NIXL_WARMUP=off` for an
optimized block with a new output path; restore `on` for the documented A/B
protocol.

The default configuration emits NDJSON telemetry and flushes it at engine
shutdown. The harness retains `engine/cn*/telemetry_data` and creates a
`telemetry` alias for the distribution analyzer. Crashes or forced kills can
leave incomplete telemetry; missing records do not prove zero work. After all
timed blocks and GPU processes have stopped, decode the saved records into
distribution reports sequentially:

```bash
for cluster in results/multi-cn-throughput-ab/{baseline,optimized}/q*/cluster-*; do
  test -d "$cluster/engine" || continue
  pixi run --frozen python experimental/starrocks/scripts/cn-distribution.py \
    --dir "$cluster/engine" --prefix cn --all-runs --json \
    > "$cluster/telemetry-distribution.json"
done
pixi run --frozen python scripts/local-gb10/analyze-multi-cn.py \
  --baseline results/multi-cn-throughput-ab/baseline \
  --optimized results/multi-cn-throughput-ab/optimized \
  --reference build/tpch-starrocks-sf500-2cn \
  --output results/multi-cn-throughput-ab/analysis --repetitions 3
```

These captures are already NDJSON; the distribution step parses them, rather
than converting a binary exporter. `benchmark-multi-cn.py --analyze-telemetry`
can run that step after each stopped cluster, but the default defers it until
after the timed sweep. `cn-activity.py` is invoked automatically for an INFO-log
summary. `analyze-multi-cn.py` writes derived `RESULTS.md`, `analysis.json`,
`samples.csv`, and per-query profiles without rewriting raw query evidence.

Report raw correctness and eligible completion rates separately from timings
on the common eligible query subset. Warm aggregates are sums of per-query
medians, not elapsed suite wall time. Missing, failed, skipped, mismatched, and
degraded-placement samples remain visible. Engine query UUIDs and FE query
UUIDs are distinct; the analyzer does not infer their identity from event
order. CN task/fragment counts establish participation, not equal scan bytes,
utilization, or the cause of a speedup.

The runtime helper dependency set is `benchmark-multi-cn.py`,
`sweep-multi-cn.py`, `analyze-multi-cn.py`, `stack.py`, `tpch-starrocks.py`,
`cn-activity.py`, `execution_validation.py`, and `sirius-sf500-2cn.yaml`.
Preparation additionally uses the three build helpers, `maven-settings.xml`,
`generate-tpch.sh`, `verify-sf100.py`, and `tpch-lineitem-count.rs`. The older standalone/SF100
helpers and `ALL22_AGENT_PROMPT.md` are not prerequisites for this workflow.
