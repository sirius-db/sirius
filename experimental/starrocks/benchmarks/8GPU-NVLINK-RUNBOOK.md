# Runbook — Sirius TPC-H on an 8×A100 NVLink box

Step-by-step bring-up of the Sirius GPU compute-node cluster on a fresh 8-GPU machine, and
how to run the TPC-H A-vs-B benchmark on it.

Target box assumed throughout:

| | |
|---|---|
| GPU | 8× A100 80 GB SXM4 (640 GB total) |
| Compute capability | 8.0 (Ampere GA100) |
| vCPU / RAM | 96 / 1900 GB |
| Disk | 10 TB |
| Interconnect | NVLink 3 (SXM4) — 12 links/GPU, 300 GB/s per direction |

**Provenance note.** Everything below is derived from the working 2-CN configuration
(`pixi run cluster2`) that produced the committed 20/22 TPC-H result on a single 23 GiB L4.
The 8-GPU scale-out (§6) and the A100 memory sizing (§12) are extrapolations from that
configuration — they follow the same code paths but have not themselves been run on an
8×A100 box. The NVLink verification procedure in §8 is the one used to gate the nixl
transport originally (`tools/transport_probe`), which measured 85–90 GB/s cuda_ipc against a
0.48 GB/s no-IPC control on the L4.

---

## 1. Check the box

```bash
# GPUs present and their memory
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# NVLink topology — on SXM4 every pair should read NV# (not PIX/PHB/SYS)
nvidia-smi topo -m

# NVLink links up
nvidia-smi nvlink --status

# Peer access between every pair must be OK, or cuda_ipc cannot use NVLink
nvidia-smi topo -p2p r
```

What you want to see in `topo -m`: `NV12` (or `NV8`/`NV4` depending on the board) between
every GPU pair. If you see `SYS` or `PHB` between pairs, those pairs talk over PCIe/host and
NVLink is not available to them — the exchange tier will still work, just far slower.

`nvidia-smi topo -p2p r` must print `OK` for every pair. `CNS` (chipset not supported) means
peer-to-peer is off and the whole point of §8 is moot.

Also confirm the driver is recent enough for the CUDA version the engine builds against
(CUDA 13.2 per `pixi.toml`):

```bash
nvidia-smi --query-gpu=driver_version --format=csv
```

---

## 2. Clone the repo

```bash
git clone <repo-url> sirius-db
cd sirius-db
git submodule update --init --recursive
```

Worktrees do **not** auto-initialize submodules; if you are on a worktree rather than a fresh
clone, run the `git submodule update` line explicitly from inside it.

Install `pixi`, which drives every build and test in this repo:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL -l
pixi --version
```

Work from the integration tree for the rest of this runbook:

```bash
cd sirius-worktrees/integration        # adjust if you cloned a plain checkout
export REPO=$PWD
```

---

## 3. Install NIXL and UCX

The compute node's cross-node exchange rides **nixl** (NVIDIA Inference Xfer Library) over a
**UCX** backend. On a same-host multi-GPU box, UCX selects its `cuda_ipc` transport for
GPU→GPU transfers, and on SXM4 hardware `cuda_ipc` moves bytes over NVLink. That chain —
nixl → UCX → cuda_ipc → NVLink — is what §8 verifies.

Both libraries live under `sirius-worktrees/tools/`, outside the source tree, and are
referenced by absolute path from `pixi.toml`. Two ways to get them:

**Option A — copy a known-good install** (fastest, if you have one):

```bash
# brev shell sirius-multicn
rsync -av <devbox>:/home/ubuntu/sirius/tools/nvda_nixl/   $REPO/../tools/nvda_nixl/
rsync -av <devbox>:/home/ubuntu/sirius/tools/ucx-install/ $REPO/../tools/ucx-install/
```

**Option B — build from source:**

```bash
mkdir -p $REPO/../tools && cd $REPO/../tools

# UCX first — nixl's UCX plugin links it.
wget https://github.com/openucx/ucx/releases/download/v1.21.0/ucx-1.21.0.tar.gz
tar xf ucx-1.21.0.tar.gz && cd ucx-1.21.0
./configure --prefix=$PWD/../ucx-install --with-cuda=$CUDA_HOME --enable-mt
make -j$(nproc) install
cd ..

# nixl against that UCX
git clone https://github.com/ai-dynamo/nixl nixl-src && cd nixl-src
meson setup build --prefix=$PWD/../nvda_nixl \
      -Ducx_path=$PWD/../ucx-install
ninja -C build install
```

`--enable-mt` on UCX is not optional: the transport touches the agent from a dedicated thread.

### 3.1 The environment file

Every build and every run needs the same four variables. Write them once:

```bash
cat > $REPO/../tools/nvda_nixl/ENV.sh <<'EOF'
# Environment for building/running against the local nixl + UCX install.
# Source this before `cargo build` on anything using nixl-sys, and at runtime.
export NIXL_PREFIX=<abs-path>/sirius-worktrees/tools/nvda_nixl
# nixl looks in NIXL_PLUGIN_DIR first, else <dir of libnixl.so>/plugins.
export NIXL_PLUGIN_DIR="$NIXL_PREFIX/lib/x86_64-linux-gnu/plugins"
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib/x86_64-linux-gnu:<abs-path>/sirius-worktrees/tools/ucx-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Build-time only: nixl-sys uses bindgen; the system libclang has no builtin headers
# installed, so point it at the pixi env's clang resource headers. Harmless at runtime.
export BINDGEN_EXTRA_CLANG_ARGS="-isystem <abs-path>/sirius-worktrees/integration/.pixi/envs/default/lib/clang/21/include"
EOF
```

Substitute the absolute paths. Then verify the plugin actually exists — a missing UCX plugin
is the single most common bring-up failure:

```bash
source $REPO/../tools/nvda_nixl/ENV.sh
ls $NIXL_PLUGIN_DIR/          # must contain libplugin_UCX.so (or similar)
```

---

## 4. Build the engine and the compute node

```bash
cd $REPO
pixi run make                 # builds libsirius (long: CUDA + cudf)
```

Then the CN, which links both the engine and nixl:

```bash
cd $REPO/experimental/starrocks
pixi run cn-build
```

`cn-build` sets, among others:

- `NIXL_PREFIX` — where to find libnixl
- **`NIXL_NO_STUBS_FALLBACK=1`** — mandatory. Without it, a broken nixl link does not fail the
  build; `nixl-sys` silently compiles a dlopen stub and you discover the problem at runtime as
  an agent-creation error, or worse, as a mysteriously slow transport.
- `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc` — the CN crate needs the system
  linker, not the pixi one.

Confirm the binary exists and is nixl-linked:

```bash
ls -la target/release/sirius-starrocks-cn
ldd target/release/sirius-starrocks-cn | grep -E 'nixl|ucp|ucs'
```

If `ldd` shows no nixl/UCX libraries, the stub path was taken — re-check `NIXL_NO_STUBS_FALLBACK`
and rebuild.

The front end is shipped pre-packaged, so you do not need the multi-hour Maven build:

```bash
pixi run fe-check             # asserts starrocks/output/fe/bin/start_fe.sh exists
```

If it fails, `git submodule update --init --recursive experimental/starrocks/starrocks` then
`pixi run fe-build`.

---

## 5. Generate the TPC-H data

The benchmark reads external parquet through `FILES()` CTEs — there is no load step, and both
engines read the same files. Lay it out as `<dir>/<table>/*.parquet`:

```bash
DATA=/data/tpch_sf100          # 10 TB of disk affords SF100 comfortably
mkdir -p $DATA

duckdb <<EOF
INSTALL tpch; LOAD tpch;
CALL dbgen(sf=100);
EOF
```

then export each table to its own directory. Multiple parquet files per table is preferable to
one giant file: the FE byte-range-splits large files across backends, but a per-file split is
cheaper and parallelizes scan setup across all 8 CNs.

Start with SF10 for a first end-to-end pass; move to SF100 once the cluster is known good.
SF1 is too small to say anything about an 8-GPU box — per-query fixed overheads dominate.

---

## 6. Launch the cluster: 1 FE + 8 CNs, one per GPU

### 6.1 The port plan

The `cluster2` task offsets its second CN's ports by `+2`. That stride does not scale to 8
CNs: the heartbeat range (9050, 9052, …) would collide with the thrift range (9060, 9062, …).
Give each CN a contiguous block of 10 ports instead, based at 9100 — clear of the FE's ports
(8030 http, 9010 edit-log, 9020 rpc, 9030 query) and of the CN defaults (9050 heartbeat,
9060 thrift, 8040 http, 8060 brpc, 9070 starlet):

| CN | GPU | base | heartbeat | thrift | brpc | http | starlet |
|----|-----|------|-----------|--------|------|------|---------|
| 0 | 0 | 9100 | 9100 | 9101 | 9102 | 9103 | 9104 |
| 1 | 1 | 9110 | 9110 | 9111 | 9112 | 9113 | 9114 |
| 2 | 2 | 9120 | 9120 | 9121 | 9122 | 9123 | 9124 |
| 3 | 3 | 9130 | 9130 | 9131 | 9132 | 9133 | 9134 |
| 4 | 4 | 9140 | 9140 | 9141 | 9142 | 9143 | 9144 |
| 5 | 5 | 9150 | 9150 | 9151 | 9152 | 9153 | 9154 |
| 6 | 6 | 9160 | 9160 | 9161 | 9162 | 9163 | 9164 |
| 7 | 7 | 9170 | 9170 | 9171 | 9172 | 9173 | 9174 |

Two identities matter:

- The **FE** identifies a CN by `(advertise_host, heartbeat_port)`. Those must be unique.
- The **nixl agent** is named `{advertise_host}:{brpc_port}`. Those must be unique too, or two
  CNs will collide when exchanging agent metadata.

Both hold under this plan.

### 6.2 The launch script

Use `cluster8.sh` in this directory (it is the `cluster2` pixi task generalized to a loop):

```bash
cd $REPO/experimental/starrocks
./benchmarks/cluster8.sh                    # all 8 GPUs
NUM_CNS=4 ./benchmarks/cluster8.sh          # first 4 GPUs
GPU_MEM=48GiB ./benchmarks/cluster8.sh      # smaller carve-out
```

Run it in its own terminal, or as its own background task. **Never chain it behind `&` inside
another shell command** — the cluster dies with that shell.

The CN **registers itself** with the FE at startup (`ALTER SYSTEM ADD COMPUTE NODE`, retried up
to `--registration-max-attempts`, default 120). You do not run `ALTER SYSTEM` by hand.

### 6.3 What the flags mean

```
--gpu-device <i>            CUDA ordinal; exported as CUDA_VISIBLE_DEVICES before engine
                            bring-up (an already-exported value wins, so don't set both)
--gpu-memory-limit 64GiB    engine memory carve-out on that device
--host-memory-limit 128GiB  engine host-memory capacity
--engine-dir .cn<i>         derived config, logs, telemetry — must be unique per CN
--heartbeat-port / --thrift-port / --brpc-port / --http-port / --starlet-port
```

Environment, set once for all CNs:

```
NIXL_PLUGIN_DIR             from ENV.sh
LD_LIBRARY_PATH             engine .so + nixl + UCX
UCX_TLS=cuda_copy,cuda_ipc,tcp,self
SIRIUS_EXCHANGE_STAGING_BYTES=8GiB
```

`UCX_TLS` is load-bearing in both halves:

- **`cuda_copy`** — without it, UCX cannot detect that a pointer is VRAM and nixl memory
  registration fails outright with `NIXL_ERR_BACKEND`.
- **`cuda_ipc`** — the fast same-host GPU→GPU path. Without it the transfer still *succeeds
  with correct bytes*, just ~200× slower through a host bounce. See §8.

`SIRIUS_EXCHANGE_STAGING_BYTES` allocates each CN's `cudaMalloc` exchange staging arena. It
sits **outside** `--gpu-memory-limit`, so a CN really occupies `gpu-memory-limit + staging +
CUDA context`. If the variable is unset there is no arena and the CN refuses cross-fragment
exchange.

---

## 7. Verify the cluster

```bash
mysql --host 127.0.0.1 --port 9030 --user root -e "SHOW COMPUTE NODES\G" \
  | grep -E 'IP|HeartbeatPort|Alive'
```

All 8 must show `Alive: true`. A CN that registered but is not alive usually means the
heartbeat port is wrong or the process died during engine bring-up — check `.cn<i>/` logs.

Smoke test a single-fragment query, then a multi-fragment one (the second exercises the nixl
exchange path across CNs, which is what NVLink accelerates):

```bash
mysql --host 127.0.0.1 --port 9030 --user root <<EOF
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file://$DATA/lineitem/*.parquet","format"="parquet"))
SELECT count(*) FROM lineitem;

WITH lineitem AS (SELECT * FROM FILES(
  "path"="file://$DATA/lineitem/*.parquet","format"="parquet"))
SELECT l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*)
FROM lineitem WHERE l_shipdate <= date '1998-09-02'
GROUP BY 1,2 ORDER BY 1,2;
EOF
```

The second query is the Q1 shape: a partial aggregation per CN, a hash fan-out over the
grouping keys, and a merge — i.e. it moves real data between CNs.

Stale registrations survive FE restarts. If `SHOW COMPUTE NODES` lists nodes that no longer
exist, drop them:

```sql
ALTER SYSTEM DROP COMPUTE NODE "127.0.0.1:9130";
```

---

## 8. Verify NIXL is actually using NVLink

This is worth doing *before* trusting any benchmark number, because the failure mode is
silent: a misconfigured transport delivers **correct bytes** at a fraction of the bandwidth.
Nothing in nixl or UCX raises an error. Three independent checks:

### Check 1 — which lane did UCX select?

Run any cross-CN query with UCX logging on and read the transport configuration lines:

```bash
UCX_LOG_LEVEL=info ./benchmarks/cluster8.sh 2>&1 | grep -E 'cfg#|UCX_TLS'
```

You want `device(cuda_ipc/cuda)` in the `ucp_context_0 intra-node cfg#N` line:

```
ucp_context_0 intra-node cfg#2 rma_am(tcp/lo) amo_am(tcp/lo) device(cuda_ipc/cuda) ...
```

If `device(...)` names `tcp` or is absent, cuda_ipc was not selected and every GPU→GPU
transfer is going through host memory.

A `rdma_create_event_channel failed: No such device` DIAG line is benign on a box with no
InfiniBand — UCX probes rdmacm, fails, and falls through to cuda_ipc.

### Check 2 — the built-in bandwidth canary

The transport already gates itself. On first contact with each peer it WRITEs a 16 MiB
lease→lease probe (after a 1 MiB warm-up to settle wireup), logs the measured rate, and
**refuses the transport tier** below a 2.0 GB/s floor:

```bash
grep 'nixl bandwidth canary' <cn-log>
```

The floor exists precisely to catch the silent degradation. On the L4 reference box this
probe measured 85–90 GB/s over cuda_ipc; on A100 SXM4 with NVLink it should be
substantially higher. A canary reading in the single digits means you are on the host-bounce
path even though the query will still return correct results.

### Check 3 — NVLink byte counters

The definitive proof that bytes crossed NVLink rather than PCIe. Sample the hardware counters
around a transfer:

```bash
nvidia-smi nvlink -gt d > /tmp/nvlink.before
# ... run the multi-fragment query, or the probe in benchmarks/nvlink/ ...
nvidia-smi nvlink -gt d > /tmp/nvlink.after
diff /tmp/nvlink.before /tmp/nvlink.after
```

The per-link Tx/Rx deltas should sum to roughly the volume you moved. A delta of ~0 means the
traffic did not touch NVLink.

### Check 4 — the TCP-vs-NIXL micro-benchmark

For a direct, quantified answer rather than an inference, `benchmarks/nvlink/` holds a
standalone two-process probe that transfers the same GPU buffer between two GPUs both ways —
once over a plain TCP socket with host staging (`cudaMemcpy` D2H → socket → H2D), once over a
nixl GPU→GPU WRITE — verifies the received bytes, and reports both throughputs plus the
NVLink counter delta. See its README; run it as:

```bash
./benchmarks/nvlink/run.sh --gpu-a 0 --gpu-b 1
```

Expect a large ratio. For calibration, the original gating probe on a single L4 measured
85–90 GB/s over cuda_ipc against 0.48 GB/s with cuda_ipc disabled — a 177× spread — and found
that `cudaMallocAsync`-backed memory silently degraded to 0.38 GB/s while still delivering
correct bytes. That is why the exchange arena must be plain `cudaMalloc` (an rmm
`pool_memory_resource<cuda_memory_resource>`, never `cuda_async`).

---

## 9. Run the benchmark — engine A (Sirius)

```bash
cd $REPO/experimental/starrocks/benchmarks/tpch

TPCH_DATA=$DATA \
QUERY_TIMEOUT=120 \
MIN_BACKENDS=8 \
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 10;
  (cd '"$REPO"'/experimental/starrocks && nohup ./benchmarks/cluster8.sh >/tmp/cluster8.log 2>&1 &)' \
  ./bench.sh /tmp/bench/A/timings.csv 3
```

Arguments and environment:

| | |
|---|---|
| `bench.sh <out_csv> [runs] [qNN…]` | 1 discarded warm-up + `runs` timed repetitions |
| `TPCH_DATA` | directory holding `<table>/*.parquet`; substituted into the `FILES()` paths |
| `QUERY_TIMEOUT` | per-run client timeout, seconds. 30 suits SF1; raise for SF100 |
| `MIN_BACKENDS` | alive backends required before the sweep starts — **set to 8**, or a sweep begun while the cluster is still booting records phantom wedges |
| `RESTART_CMD` | full cluster restart after a wedge |
| `FE_PORT` | default 9030 |

`RESTART_CMD` is **mandatory for engine A.** The CN does not implement `cancel_plan_fragment`,
so a hung or mid-execution-failed query strands its fragments; the stranded fragments starve
the CNs and the FE then answers "No available backends" for everything after. Without the
restart, every measurement following the first failure is invalid.

Note the `[s]irius-starrocks-cn` bracket pattern — it stops `pkill` matching its own command
line and killing your shell. The CN binary is `sirius-starrocks-cn`, not
`starrocks-compute-node`.

To sweep a subset:

```bash
TPCH_DATA=$DATA ./bench.sh /tmp/bench/A/timings.csv 3 q01 q06 q14
```

### Expected result quality

The last committed sweep (2-CN L4, SF1) was **20/22 passing**, 17 of them within 0.25 % of the
DuckDB oracle. Known open items you may still hit:

- **q02** hangs hard — an engine-thread wedge with no abort path.
- **q15** returns an empty result on roughly 1 run in 4.
- Up to −0.40 % arithmetic deficit on q03/q10 rows in the multi-fragment
  `sum(x*(1-l_discount))` path; passes are counted within a 0.5 % band.

---

## 10. Run the baseline — engine B (stock StarRocks)

```bash
JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto ./setup-engine-b.sh
# the script prints the exact commands to start the FE + BEs and register them
TPCH_DATA=$DATA ./bench.sh /tmp/bench/B/timings.csv 3
```

Stock StarRocks is laid out as a shared-nothing FE plus **BEs** (`start_be.sh` +
`ALTER SYSTEM ADD BACKEND`), not CNs — CN mode fails `FILES()` with "No alive backends".
Engine B needs no `RESTART_CMD`; it cleans up after itself.

**Run one engine at a time.** A and B share the FE port 9030, the backend port ranges, and the
host CPUs. Take A fully down before measuring B and vice versa, or both sets of numbers are
meaningless.

---

## 11. Compare

```bash
./analyze.py /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv results.md tpch_a_vs_b.png
```

Emits a markdown table (median ms per query, geometric-mean speedup over the set both engines
completed) and a log-scale bar plot. Paste the table into `BENCHMARK-A-VS-B.md`.

---

## 12. Tuning for A100-80GB

The committed configuration targets a 23 GiB L4 shared by two CNs. On an 80 GB A100 with one
CN per GPU, the constraint is different — start here:

| Knob | L4 (2 CNs/GPU) | A100 80 GB (1 CN/GPU) | Why |
|---|---|---|---|
| `--gpu-memory-limit` | 8GiB | 64GiB | leave headroom for the staging arena + CUDA context; the arena is *not* counted inside this limit |
| `SIRIUS_EXCHANGE_STAGING_BYTES` | 1280MiB | 8GiB | 512 MiB starved TPC-H q09 — a single packed-table lease of 648 MB exceeded the whole arena. Scale it with the scale factor |
| `--host-memory-limit` | 12GiB | 128GiB | 1900 GB / 8 CNs leaves generous room |

Budget per GPU: `64 GiB limit + 8 GiB arena + ~2–3 GiB CUDA context/cudf overhead ≈ 75 GiB` of
80. If a CN dies at bring-up with an allocation failure, lower `--gpu-memory-limit` first.

`--gpu-memory-fraction <f>` is available as an alternative to the absolute limit; it is a
fraction of **total** device memory, not free memory.

If q09-style queries fail with a staging-arena error, raise
`SIRIUS_EXCHANGE_STAGING_BYTES` — that is the arena, not the engine limit.

---

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Agent creation fails at CN startup | libnixl not found or the stub was linked. Source `ENV.sh`; rebuild with `NIXL_NO_STUBS_FALLBACK=1`; check `NIXL_PLUGIN_DIR` really contains the UCX plugin |
| nixl registration fails, `NIXL_ERR_BACKEND` | `UCX_TLS` is missing `cuda_copy` — UCX cannot detect VRAM pointers |
| Everything works but is ~200× slow | `UCX_TLS` is missing `cuda_ipc`, or the arena is not `cudaMalloc`-backed. The bandwidth canary should have refused the tier; see §8 |
| "No available backends" for every query | A wedged query stranded its fragments. Restart the cluster — this is why `RESTART_CMD` exists |
| `SHOW COMPUTE NODES` lists dead nodes | FE metadata persists across restarts: `ALTER SYSTEM DROP COMPUTE NODE "host:port"` |
| Sweep records wedges from the first query on | The sweep started before the cluster was up. Set `MIN_BACKENDS=8` |
| `pkill` killed your shell | Use the bracket pattern: `pkill -f '[s]irius-starrocks-cn'` |
| CN exits immediately, no log | Port collision. Check the §6.1 plan against `ss -ltnp` |
| `cargo fmt --check` fails on untouched files | Pre-existing CN files fail fmt. Format only the crates you touched; never run it workspace-wide |

---

## 14. Known limitations of engine A

- No `cancel_plan_fragment` — hung queries strand fragments; `RESTART_CMD` is mandatory.
- No cancel/GC path generally.
- `DISTINCT` aggregation is refused.
- q02 hangs (engine-thread wedge, needs an engine-side abort/watchdog).
- q15 flakes to an empty result ~1 run in 4.
- Merge functions that are not per-column reduces (stddev, median, distinct) are out of reach
  of the current two-phase aggregation design; the stream ABI also cannot spell LIST or STRUCT
  types.

---

## 15. Fairness caveats

Engine A executes on GPUs; engine B is a mature vectorized CPU engine. At small scale factors
fixed per-query overheads — fragment dispatch, first-touch allocation, plan translation —
matter as much as scan throughput, which is why SF10 or SF100 says far more than SF1 about an
8-GPU box. Both engines read the same parquet files through `FILES()` with no load step, and
the FE byte-splits large files across backends in both cases. Results are indicative, not a
TPC-compliant benchmark.
