---
name: tpch-bench
description: Operate the Sirius-as-StarRocks-CN demo and its TPC-H benchmark on the 4x GB200 box. Use when bringing the multi-GPU (N-CN, one CN per GPU) cluster up or down, running a query or the 22-query sweep, verifying the nixl/NVLink transport, triaging a failing/hanging/slow query, checking results against DuckDB, or reproducing the A-vs-B comparison against stock StarRocks.
---

Runs from the repo root `/home/prestouser/aocsa/sirius` unless noted. This is the *operational* skill (StarRocks CN cluster + A-vs-B); the sibling `benchmark` skill is Sirius-vs-DuckDB in-process and does not apply.

**Current refs**: `OPEN-ISSUES.md` (root) — the 2026-08-08 4x GB200/SF100 audit + work queue, read first. `QUERY-TIMEOUT-ANALYSIS.md` (root) — per-defect history. `experimental/starrocks/benchmarks/8GPU-NVLINK-RUNBOOK.md` — multi-GPU bring-up, §5 data generation, §8 transport verification.
**STALE — do not copy numbers or commands from**: `benchmarks/tpch/REPRODUCE.md`, `benchmarks/tpch/results/`, and the pixi `cluster2`/`cluster` tasks are all SF1 on one 23 GiB L4 with 2 CNs sharing a GPU. `QUERY-TIMEOUT-ANALYSIS.md`'s per-query verdicts are SF1/2-CN and partly superseded (see Baseline).

## Box facts

- 4x NVIDIA GB200, 185.03 GiB HBM each (189471 MiB), cc 10.0, NV18 all-to-all full mesh. CUDA 13.0, driver 580.105.08. aarch64 Grace, 144 cores.
- NUMA: node0 = CPU 0-71, node1 = CPU 72-143; `numactl --hardware` reports 489960 + 489823 "MB" which are really **MiB** — ~478 GiB per node, **~957 GiB total CPU DRAM**. `free -g` says ~1692 GiB because it counts HBM. `SwapTotal 0`. **Nodes 2/10/18/26 are cpuless GPU HBM (184 GiB each) — NEVER membind them.** NVML affinity: GPU0,GPU1 -> node0; GPU2,GPU3 -> node1.
- **No docker.** JDK `/usr/lib/jvm/java-21-openjdk-arm64`. Home is NFS, clock ~0.2 s ahead of local — breaks meson clock-skew checks, so out-of-tree build dirs must live on local disk.
- Data `/home/prestouser/aocsa/tpch_parquet_sf100` (SF100, 26 GB, `<table>/*.parquet`). `lineitem` is 6 files `part.0..5.parquet` — not a multiple of 4, so byte-range splitting across 4 CNs is uneven. `lineitem count(*) = 600,037,902` (correct for SF100). To regenerate or make another scale: runbook §5.
- CN binary `sirius-starrocks-cn` (a pkill pattern naming anything else kills nothing); FE java class `StarRocksFE`; FE MySQL port 9030.
- **`mysql` exists only in the starrocks pixi env** (`$SR/.pixi/envs/default/bin/mysql`). Either wrap it — `pixi run --manifest-path /home/prestouser/aocsa/sirius/experimental/starrocks/pixi.toml bash -c "mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;'"` — or, for scripts that call bare `mysql` (`bench.sh`, `setup-engine-b.sh`), `export PATH=$SR/.pixi/envs/default/bin:$PATH` first. The binary runs standalone from that path.

## Traps (these silently corrupt results)

- **There is NO correctness gate anywhere in the harness.** `bench.sh:73-75` scores `pass` on exit-code + non-empty file + no `ERROR` on line 1. `analyze.py` reads only `status` and `ms` — the `rows` column is written and never compared between A and B. A query returning 1 row instead of 100000 is recorded as a fast WIN. Any number you quote must be backed by a DuckDB-oracle check of that query's result.
- **Unset `CUDA_VISIBLE_DEVICES` before launching.** `engine.rs:207-216`: an already-exported value **wins over `--gpu-device` and is only `warn!`ed about**, collapsing all 4 CNs onto one GPU — a cluster that still answers queries, so the harness records numbers. `cluster8.sh` does not clear it. The CN's own `ensure_gpu_unclaimed` preflight (`main.rs:265-270`) short-circuits whenever `--gpu-memory-limit` is set, which every command here does, so nothing else catches it. After launch, `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` must list 4 **distinct** GPUs.
- **`grep -c true` does not count alive nodes.** `SHOW COMPUTE NODES` also emits `SystemDecommissioned`, `ClusterDecommissioned`, `HasStoragePath`, so a booting node with `Alive=false, HasStoragePath=true` matches. `bench.sh:46-47` and `run-comparison.sh:24` both carry this. `Alive` is column 9 (`ComputeNodeProcDir.java:48-53`: ComputeNodeId, IP, HeartbeatPort, BePort, HttpPort, BrpcPort, LastStartTime, LastHeartbeat, Alive; shared-data columns append at the end, so the index is stable). Count it with `awk -F'\t' '$9=="true"' | wc -l`.
- **`MIN_BACKENDS=4` does not repair `wait_alive`.** It raises the threshold, but `bench.sh:46-47` still miscounts (above) *and* sums `SHOW COMPUTE NODES` with `SHOW BACKENDS`, so 4 booting CNs satisfy 4. You cannot fix the in-sweep restart path without editing `bench.sh:46`. Mitigate by making `RESTART_CMD` itself block on a real Alive-column check instead of a `sleep` (done below) — post-restart phantom wedges otherwise cascade through the rest of the sweep.
- **`QUERY_TIMEOUT` defaults to 30 s** (`bench.sh:37`). No SF100 pass exceeds it (slowest median q21 14342 ms), but at 30 s every failure collapses into an indistinguishable `wedge` — the 60 s `REPLY_TIMEOUT` refusals and the real hangs both read as `30004`. Set 180 so the failure class is readable. `MIN_BACKENDS` defaults to 2 (`bench.sh:39`); set it too.
- **`RESTART_CMD` is mandatory for engine A and MUST carry `NUM_CNS=4` + the memory env.** `cluster8.sh:24` defaults to 8 CNs and `:48` hard-fails "asked for 8 CNs but only 4 GPUs are visible" -> `wait_alive` fails -> `bench.sh:80/85` exits and the whole sweep aborts. Append (`>>`), never truncate (`>`), the cluster log — a restart is fired *by* a failure whose only evidence is in that log.
- **`SIRIUS_QUERY_WATCHDOG_SECS` is unset by `cluster8.sh`, and the Baseline below was measured with it unset.** Unset, `sirius_engine.cpp:110-113` blocks on `future.get()` and a wedged statement poisons the CN for every later query. Setting it is a **recommendation, not a measured fix**: it is a *no-scheduling-progress* watchdog (`:89-95`, polls every 250 ms), so the value must be **strictly below `QUERY_TIMEOUT`** to fire at all (300 with a 180 s client timeout can never fire) and well above any single kernel's real runtime or it fails healthy queries. `configs/gb200-4gpu/engine-a.env:172-177` recommends **120** for unattended SF100 sweeps (~13x margin over the slowest passing query, q07 at 8.9 s). Nothing has been validated at SF100 — if you set it, record the value with the run, and note the numbers are then no longer comparable to the Baseline.
- **`bench.sh` arg shifting** (`:31` `shift $(( $# >= 2 ? 2 : 1 ))`): a query subset must be preceded by an explicit runs count — `bench.sh out.csv q05` sets `RUNS=q05` and sweeps all 22.
- **`bench.sh` breaks the run loop on the first refusal/wedge**, so a run-0 failure yields zero timed rows; a refusal at r=0 *is* written to the CSV (`:78`) while a pass at r=0 is not (`:75`).
- **Engines A and B share port 9030 and the host CPUs — NEVER run them simultaneously.** `run-comparison.sh`'s `engines_down()` pkills `sirius-starrocks-cn`, `starrocks_be` and `StarRocksFE`; do not run it while a cluster you care about is up.
- **`analyze.py` exits non-zero here**: it prints and writes the markdown table (`:71-72`), then dies at `:75` with `ModuleNotFoundError: matplotlib` (matplotlib/numpy are in neither pixi env). The table is valid, the PNG is not produced — do not read the exit code as failure.
- **`setup-engine-b.sh:42-63` rewrites `mem_limit` into both `be.conf`s on EVERY run**, not just first setup — it silently reverts tuning.

## One-time bring-up

`TOOLS_DIR=/home/prestouser/aocsa/tools` holds `ucx-install/` (UCX 1.21.0) and `nvda_nixl/`, both built from source for aarch64. nixl installs to `lib/aarch64-linux-gnu/` with `plugins/libplugin_UCX.so`; `cn-env.sh` globs `lib/*-linux-gnu`, so aarch64 resolves automatically. The nixl **meson build dir must live on local disk (`/tmp`), not NFS** (clock skew). nixl's python-wheel step needs a working `uv` — `~/.local/bin/uv` is an **x86-64** binary here and dies with `Exec format error`.

**Three build shims are required for `cargo build -p sirius-starrocks-cn`**, and therefore for anything engine-linked:

| Shim | Why |
|---|---|
| `g++` -> `/usr/bin/g++` | vendored `nixl-sys` 1.3.2 `build.rs` hardcodes `cc::Build::compiler("g++")`, overriding `cn-env.sh`'s `CXX`. Under pixi, bare `g++` is conda's, which mixes the conda sysroot with `-I/usr/include` and dies on `bits/timesize.h`. |
| `ld` -> `/usr/bin/ld` | otherwise pixi's `ld` links conda-sysroot libpthread/libdl and fails on `GLIBC_PRIVATE` undefined refs. |
| `libnvidia-ml.so` -> `/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1`, fed via `RUSTFLAGS -L/-l` | the engine `.so` has 14 undefined nvml symbols and the driver ships no `.so` symlink. Link-time only — the `.so` already has `NEEDED libnvidia-ml.so.1`. |

```bash
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
SHIMS=/home/prestouser/aocsa/tools/toolchain-shims     # NOT /tmp -- must survive a reboot
mkdir -p $SHIMS
ln -sf /usr/bin/g++ $SHIMS/g++
ln -sf /usr/bin/ld  $SHIMS/ld
ln -sf /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 $SHIMS/libnvidia-ml.so
# The PATH prefix must go INSIDE pixi. An outer `PATH=$SHIMS:$PATH pixi run ...` is DEFEATED --
# pixi prepends its own env bin to the inherited PATH (measured: g++ still resolves to
# .pixi/envs/default/bin/g++). `pixi run cn-build` cannot carry it, so call cargo directly.
pixi run --manifest-path $SR/pixi.toml bash -c "
  export PATH=$SHIMS:\$PATH
  export RUSTFLAGS=\"-C link-arg=-L$SHIMS -C link-arg=-lnvidia-ml\"
  cd $SR && source scripts/cn-env.sh && cargo build --release -p sirius-starrocks-cn"
```

`cn-env.sh:49` already sets `CARGO_TARGET_AARCH64_..._LINKER=/usr/bin/gcc`, so it fixes neither the hardcoded `g++` nor `ld`. Permanent fixes worth landing: `cuda-nvml-dev` in `$SR/pixi.toml [feature.engine.dependencies]`; `$SR/.cargo/config.toml` with `[target.aarch64-unknown-linux-gnu] linker="/usr/bin/gcc"`; a checked-in shim dir.

**StarRocks submodule**: the recorded gitlink `04cd3136` **does not exist upstream** (local-only devbox commit), so `git submodule update --init experimental/starrocks/starrocks` fails. Recovery: check out base commit `14b7e3fa` (`[BugFix] Re-record per-partition coordinator claim…`, the submodule's current HEAD), run `pixi run --manifest-path $SR/pixi.toml apply-starrocks-patches` (`nixl-exchange-proto.patch` applies cleanly), then `fe-build` (Maven, ~40 min). Run `git submodule update --init --recursive` at the repo root too — `substrait`, `vcpkg` and the root `duckdb/` are not auto-populated and the engine cmake fails with `No SOURCES given to target: sirius_extension`. Then `engine-build` + the cargo build above.

## Cluster up/down (engine A, 4 CNs)

`cluster8.sh` only *asserts* the binaries exist (`:36-37`) — it does not build — and has no JAVA_HOME fallback. It also has **no port preflight and no GPU-claim preflight**: launching over a live cluster does not fail to bind, it corrupts the FE registry (node identity = `advertise_host` + `heartbeat_port`) and the nixl agent registry (name = `advertise_host:brpc_port`).

```bash
# shell 1 -- PREFLIGHT, then a launch that BLOCKS FOREVER (cluster8.sh:84 ends on `wait -n`).
# Give it its own terminal or background task; never chain it behind `&` inside another command
# (cluster8.sh:15-16 -- "the cluster dies with that shell", and its EXIT/INT trap tears the
# cluster down on Ctrl-C). All three preflight commands must print NOTHING.
pgrep -fa '[s]irius-starrocks-cn|[S]tarRocksFE'
ss -ltn | grep -E ':(8030|9010|9020|9030|91[0-4][0-9])\b'
nvidia-smi --query-compute-apps=pid --format=csv,noheader
cd /home/prestouser/aocsa/sirius/experimental/starrocks
unset CUDA_VISIBLE_DEVICES
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64 TOOLS_DIR=/home/prestouser/aocsa/tools \
NUM_CNS=4 GPU_MEM=140GiB HOST_MEM=160GiB STAGING=16GiB \
./benchmarks/cluster8.sh 2>&1 | tee -a /tmp/cluster4.log
```

```bash
# shell 2 -- wait for all 4. Count the Alive column (col 9); `grep -c true` overcounts.
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
until [ "$(pixi run --manifest-path $SR/pixi.toml \
    bash -c "mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;'" 2>/dev/null \
    | awk -F'\t' '$9=="true"' | wc -l)" -ge 4 ]; do sleep 5; done
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv   # must be 4 DISTINCT gpu_uuids

# shell 2, teardown -- when the run is done
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # THE check: empty. NOT memory.used -- its idle floor here is 30-33 MiB, never 0
```

**Logs.** The CN's Rust/transport output (nixl canary, prpc, exchange) goes to **stdout only** (`main.rs:676-683`, `tracing_subscriber::fmt()`); `cluster8.sh` redirects nothing per-CN, so the FE and all 4 CNs interleave into whatever you teed — `/tmp/cluster4.log` above. The canary line carries a `peer=<host:brpc_port>` field. Separately: C++ engine log `$SR/.cn<i>/log/sirius_<YYYY-MM-DD>.log`, per-query telemetry `$SR/.cn<i>/telemetry/<query-uuid>/`, FE `$SR/starrocks/output/fe/log/fe.log` (+ `fe.audit.log`).

Ports: base 9100, stride 10 -> CN i gets heartbeat `9100+10i`, thrift +1, brpc +2, http +3, starlet +4; FE uses 8030/9010/9020/9030. Sizing: the staging arena sits **outside** `--gpu-memory-limit`, so a CN occupies `GPU_MEM + STAGING + CUDA context`; 140-160 GiB of 185 fits. `STAGING`'s 8GiB default (`:31`) was never exercised at SF100 — 16GiB is the known-working value, not a proven floor. 4x `HOST_MEM` must fit ~957 GiB at `SwapTotal 0`: 160GiB above is `engine-a.env:141`'s residual-budget figure; **the Baseline below was measured at 200GiB, so its medians are not strictly comparable** — pick one and record it.

Other launchers: `configs/gb200-4gpu/cluster4-numa.sh` (+ `engine-a.env`) is `cluster8.sh` plus per-CN `numactl --physcpubind/--membind`, an assertion that every membind target has CPUs (the HBM interlock), a port preflight, a GPU-claim preflight, and a `CUDA_VISIBLE_DEVICES` unset. It is **untracked and has never been run or benchmarked** — the Baseline is unpinned. Treat it as a reviewed candidate, not the supported path, and commit it before relying on it (`OPEN-ISSUES.md` M1.2 still reads "zero implementation"). `benchmarks/nixl-nvlink/script-box.sh` derives `NUM_CNS` from `nvidia-smi` and sets JAVA_HOME from the pixi env, but its memory defaults are H100-sized (`GPU_MEM=40GiB`).

## Verify NVLink before trusting any number

The failure mode is silent: a misconfigured transport delivers **correct bytes** at a fraction of the bandwidth. Full procedure in `8GPU-NVLINK-RUNBOOK.md` §8.

```bash
nvidia-smi topo -m                              # every pair must read NV# (here NV18), not PIX/PHB/SYS
grep 'nixl bandwidth canary' /tmp/cluster4.log  # per-peer, logged on first contact; 12 directed pairs
nvidia-smi nvlink -gt d > /tmp/nvlink.before    # ... run a cross-CN query ...
nvidia-smi nvlink -gt d > /tmp/nvlink.after; diff /tmp/nvlink.before /tmp/nvlink.after
```

Reference points: **322-399 GB/s across all 12 directed pairs on this box**; ~85-90 GB/s healthy `cuda_ipc` and ~0.4 GB/s degraded staged-copy on the L4 (`nixl_transport.rs:180-182`). The transport self-gates at `CANARY_FLOOR_GBPS = 2.0` and refuses the tier below it — that catches only catastrophic links, so a reading one or two orders below 322 is suspect *and will be silently accepted*. NVLink Tx/Rx counter deltas must sum to roughly the volume moved; ~0 means the traffic never touched NVLink. A `rdma_create_event_channel failed` DIAG line is benign (no InfiniBand). Runbook §8 Check 4 points at `benchmarks/nvlink/run.sh`, which **does not exist**; the real dir is `benchmarks/nixl-nvlink/` (only `script-box.sh` + `notes-setup.md`).

## One query, with the DuckDB oracle

Queries in `benchmarks/tpch/queries/` carry a `__TPCH_DATA__` placeholder that `bench.sh:65` substitutes; sed it yourself for manual runs. Passing `-e "$Q"` through nested shells mangles the `FILES()` quotes — pipe the file into mysql instead.

```bash
export TPCH_DATA=/home/prestouser/aocsa/tpch_parquet_sf100
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
sed "s|__TPCH_DATA__|$TPCH_DATA|g" $SR/benchmarks/tpch/queries/q05.sql > /tmp/q.sql
pixi run --manifest-path $SR/pixi.toml bash -c \
  'timeout 180 mysql -h127.0.0.1 -P9030 -uroot --batch < /tmp/q.sql'

# oracle: same SQL, parquet read directly
pixi run python3 - <<'PY'
import duckdb, re
sql = open("/tmp/q.sql").read()
sql = re.sub(r'FILES\("path"="file://([^"]+)","format"="parquet"\)', r"read_parquet('\1')", sql)
print(duckdb.sql(sql).df())
PY
```

At SF100 the oracle is a multi-minute, tens-of-GB host-RAM job (seconds at SF1) — and it is the **only** correctness check in this entire workflow. Comparison bar: **row counts**, keys and ordering must be exact. On `x*(1-l_discount)` expressions, values drifted low by up to 0.39 % **at SF1** against a 0.25 % tolerance (the open decimal-lowering item, `QUERY-TIMEOUT-ANALYSIS.md` #24; q03/q10 were its out-of-band `pass*` rows in the SF1 record — q10 does not even reach `pass` at SF100). The drift at 100x the rows has never been measured: establish the band from the oracle before accepting any value mismatch.

## The sweep — engine A

```bash
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
export PATH=$SR/.pixi/envs/default/bin:$PATH   # bench.sh calls bare `mysql` (bench.sh:40)
cat > /tmp/restart-A.sh <<'EOF'
#!/usr/bin/env bash
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'; sleep 10
unset CUDA_VISIBLE_DEVICES
(cd "$SR" && JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64 TOOLS_DIR=/home/prestouser/aocsa/tools \
   NUM_CNS=4 GPU_MEM=140GiB HOST_MEM=160GiB STAGING=16GiB \
   nohup ./benchmarks/cluster8.sh >>/tmp/cluster4.log 2>&1 &)
# block on a REAL Alive count -- bench.sh's own wait_alive miscounts (see Traps)
for _ in $(seq 1 90); do
  n=$(mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;' 2>/dev/null \
      | awk -F'\t' '$9=="true"' | wc -l)
  [ "${n:-0}" -ge 4 ] && exit 0
  sleep 5
done
exit 1
EOF
chmod +x /tmp/restart-A.sh

TPCH_DATA=/home/prestouser/aocsa/tpch_parquet_sf100 QUERY_TIMEOUT=180 MIN_BACKENDS=4 \
RESTART_CMD=/tmp/restart-A.sh \
  $SR/benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 3
```

1 discarded warm-up + 3 timed runs per query; the first refusal/wedge breaks that query's loop and fires `RESTART_CMD`. **Copy the CSV somewhere durable** — nothing SF100 is committed under `results/`, so `/tmp/bench/A/timings.csv` is the only copy.

## The sweep — engine B (stock StarRocks 3.5.20)

`run-comparison.sh` is **NOT usable unmodified**: `:43-45` hardcode `pixi run cluster2` (2 CNs on GPU 0, 8GiB each) and `alive 2`, it sets no `QUERY_TIMEOUT`/`MIN_BACKENDS`, passes no `RESTART_CMD`, and its engine-B arm calls the docker path and starts BEs with no `numactl`. Run each arm by hand.

Staging engine B without docker: fetch `https://releases.starrocks.io/starrocks/StarRocks-3.5.20-ubuntu-arm64.tar.gz` — that name returns 200, so `setup-engine-b.sh:2-3`'s blanket "the release tarball URLs 403" comment does not hold for aarch64 (the x86 name does 403, which is what the comment was recording). Extract and place `fe/` and `be/` at `$HOME/starrocks-bench/{fe,be}` so the `if [ ! -d $B/fe ]` guard at `:15` skips the docker path.

```bash
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
B=$HOME/starrocks-bench
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
export PATH=$SR/.pixi/envs/default/bin:$PATH

# 1. ONCE. The tarball gives you fe/ and be/ ONLY -- setup-engine-b.sh:38-63 is what creates
#    be1/be2 and writes fe.conf + both be.confs, all OUTSIDE the docker guard.
$SR/benchmarks/tpch/setup-engine-b.sh
# 2. THEN set mem_limit in $B/be{1,2}/conf/be.conf -- step 1 just stamped 16G over it.
# 3. Launch. --membind=0,1 is MANDATORY.
$B/fe/bin/start_fe.sh --daemon
numactl --membind=0,1 $B/be1/bin/start_be.sh --daemon
numactl --membind=0,1 $B/be2/bin/start_be.sh --daemon
mysql -h127.0.0.1 -P9030 -uroot -e 'ALTER SYSTEM ADD BACKEND "127.0.0.1:9050"; ALTER SYSTEM ADD BACKEND "127.0.0.1:9052";'

TPCH_DATA=/home/prestouser/aocsa/tpch_parquet_sf100 QUERY_TIMEOUT=180 MIN_BACKENDS=2 \
  $SR/benchmarks/tpch/bench.sh /tmp/bench/B/timings.csv 3
```

Without `--membind=0,1`, BE anonymous pages fall back onto the cpuless GPU-HBM nodes (2/10/18/26, ZONE_MOVABLE in `N_MEMORY` at distance 80) — correct bytes at far-NUMA bandwidth, silently. StarRocks also sizes `mem_limit` against `/proc/meminfo MemTotal`, which **includes** that HBM. **`mem_limit` is unsettled**: `setup-engine-b.sh` rewrites `16G` on every run, the live `be1/be2` confs currently read `64G` (hand-edited, provenance unrecorded), `configs/gb200-4gpu/engine-b/be1.conf:68` argues for `240G`, and **no value has been validated** for a fair 2-BE SF100 run — choose deliberately and record it. `be3`/`be4` also exist on disk and `configs/gb200-4gpu/engine-b/sensitivity-4be/` ships a 4-BE arm, for which `MIN_BACKENDS=2` would under-wait. StarRocks spill is OFF by default (`SessionVariable enableSpill=false`) and no TPC-H query enables it, so an undersized `mem_limit` yields loud refusals, not slow passes.

## The comparison

```bash
pixi run python3 $SR/benchmarks/tpch/analyze.py \
  /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv /tmp/bench/results.md /tmp/bench/ab.png
```

Positional, in that order (`analyze.py:11-14`); `out.md`/`out.png` default beside the A csv. It medians the `pass` rows per query and geometric-means `mb/ma` over queries both engines passed. It prints the table and then dies on missing matplotlib — expected (see Traps). **Diff A's `rows` against B's yourself; analyze.py never does.**

## Baseline (2026-08-08 — timings only, NOT correctness-validated)

SF100, 4 CNs, `GPU_MEM=140GiB HOST_MEM=200GiB STAGING=16GiB`, watchdog unset, 1 warmup + 3 timed, `QUERY_TIMEOUT=180`: **16/22 recorded as `pass` by bench.sh** — i.e. rc=0 + non-empty + no `ERROR` on line 1. No SF100 result has ever been diffed against the DuckDB oracle; this is a timing/liveness reference, not a correctness reference. Medians (ms):

| q01 | q02 | q03 | q04 | q06 | q07 | q12 | q13 | q14 | q16 | q17 | q18 | q19 | q20 | q21 | q22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6093 | 2395 | 6267 | 3700 | 5285 | 8873 | 4943 | 6496 | 7158 | 878 | 11529 | 4470 | 7787 | 9058 | 14342 | 2041 |

Failing: `q05` wedge 180004 | `q08` refused 60758 | `q09` wedge 180004 | `q10` refused 121126 | `q11` wedge 4076 (EMPTY result, not a timeout) | `q15` wedge 15203 (EMPTY result).

**q02 PASSES here at 2395 ms** despite being `QUERY-TIMEOUT-ANALYSIS.md`'s flagship hard wedge — that was fixed in `59ce6662`; do not triage it as a hang. **q15 was recorded fixed (`312e4535`) and has regressed to EMPTY at this scale**, and **q11 returns EMPTY** having never wedged before. Re-measure before trusting any per-query claim. For reference only, **SF1 on a single-GPU 2-CN box** was 22/22, warm medians ~0.3-1.3 s, geomean 0.48x (B faster) — not achievable-here numbers.

## Triage a non-pass row

| Symptom | Meaning | Move |
|---|---|---|
| `wedge`, ms ~= `QUERY_TIMEOUT`*1000 (180004) | real hang; with the watchdog unset the engine blocks on `future.get()` forever. q05/q09 today | restart, rerun solo; if it reproduces, capture the cluster log + all `.cn*/log/` and `gdb -p <cn-pid> -batch -ex 'thread apply all bt'` — pick the right pid, there are 4 |
| `wedge`, ms FAR below the timeout | NOT a hang: rc=0 with an EMPTY result set; `bench.sh:73`'s `[ -s "$f" ]` records empty as wedge. q11 (4076), q15 (15203) today | the #29 empty-result class — diff against the DuckDB oracle, do **not** gdb |
| `refused` at ~60 s (60758 ms) | `prpc_client.rs:25` `REPLY_TIMEOUT` is a hardcoded 60 s; a peer's `request_staging_lease` queued behind its engine thread and the caller declared it wedged. q08 today | not this query's plan — read the **peer** CN's stdout in the cluster log (match `peer=<host:brpc_port>`); `OPEN-ISSUES.md` M1.4 wants the constant env-configurable |
| `refused` + "staging lease ... arena exhausted" | the 2026-08-07 cumulative leak was fixed and endurance-proven **at SF1 / 2 CNs / 1280MiB arena** (`QUERY-TIMEOUT-ANALYSIS.md` §"The arena leak"; the SHA recorded there does not resolve in this repo, so cite the doc). **Never re-proven at SF100 with a 16GiB arena.** A fresh one is either undersized `STAGING` (one packed-table lease can exceed the whole arena) or a NEW leak | confirm `STAGING`, rerun solo; if it reproduces at 16GiB treat it as a new defect — do not dismiss it as the known one |
| `refused` + "declared X but the source sink produces Y" | translator/engine schema disagreement on a fragment hop | real defect: capture EXPLAIN + exact error; check `QUERY-TIMEOUT-ANALYSIS.md` for the class first |
| `refused` + other error text | loud, query-local; the text names the layer | read it literally — the loud-failure net is trustworthy |
| `pass` but wrong values | check the decimal-drift band first (SF1-derived, unmeasured at SF100) | outside the band -> correctness bug: bisect with the oracle per sub-expression |
| `pass` with a plausible row count | **verified by nothing** | diff row count + values against the oracle before quoting it |

After ANY wedge or refusal, restart before trusting the next row: a wedged statement blocks the next one on the same CN and there is still no `cancel_plan_fragment`.

## Verifying a fix

The three build shims are a prerequisite for everything engine-linked. Suites in cost order:

```bash
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
pixi run --manifest-path $SR/pixi.toml bash -c \
  "cd $SR/crates/starrocks-plan-translator && cargo test"     # translator
pixi run --manifest-path $SR/pixi.toml cn-test-no-engine      # what CI runs, no GPU
pixi run --manifest-path $SR/pixi.toml cn-test                # engine-linked, incl. wire-type parity
pixi run make && pixi run make test                           # only if src/** C++ changed
pixi run bash -c 'export LD_LIBRARY_PATH=$PWD/build/release/extension/sirius:$LD_LIBRARY_PATH; \
  cargo test --manifest-path rust/Cargo.toml -p sirius --lib -- --test-threads=1'
```

Then the live gate: the affected query solo vs the DuckDB oracle (**values AND row counts** — the harness checks neither), its regressions, and a full sweep compared against the Baseline above (timings and pass/fail only; the Baseline carries no correctness claim). For anything touching exchange or the staging arena the established bar is an endurance shape: 2-3 consecutive sweeps, zero restarts, same CN PIDs. A pre-commit-cleanliness hook denies Bash calls while the working tree is dirty — clean the tree before a long run.
