# Replicating the two-machine Sirius CN experiment

Step-by-step, from a fresh shell, to reproduce the 2026-08-11/12 result on **this** pair:
**StarRocks scheduling plan fragments across two physical machines with NIXL moving
GPU-resident data between them over cross-host NVLink.**

Everything here targets `presto-gb200-gcn-09` + `presto-gb200-gcn-18`. Commands are marked
**[09]**, **[18]** or **[both]**. Nothing needs root.

The 2026-08-12 numbers in §7 were measured on **gcn-17 + gcn-18**, not this pair. Do not quote
them as a 09+18 result. Do **not** copy timings from `notes/2026-08-09-gb200-sf100/`.

Companion docs:

| Need | Doc |
|---|---|
| aarch64 build + 1-CN smoke | [`../../bench/gb200-4gpu/BUILD-AND-SMOKE.md`](../../bench/gb200-4gpu/BUILD-AND-SMOKE.md) |
| Memory knobs, NUMA, SCALE_FACTOR | [`../../bench/gb200-4gpu/SIRIUS-TUNING-RUNBOOK.md`](../../bench/gb200-4gpu/SIRIUS-TUNING-RUNBOOK.md) |
| Box NUMA / HBM nodes | [`../../bench/gb200-4gpu/HARDWARE.md`](../../bench/gb200-4gpu/HARDWARE.md) |
| Transport env vars | [`../docs/TUNABLES.md`](../docs/TUNABLES.md) |

Do **not** launch with `benchmarks/cluster8.sh` or `configs/gb200-4gpu/cluster4-numa.sh` here —
those are single-box 4-CN. This experiment is `benchmarks/cn-2host.sh`.

> **You need a shell on each host.** Two terminals: `prestouser@presto-gb200-gcn-09` and
> `prestouser@presto-gb200-gcn-18`.

---

## 0. What this reproduces, and the one thing that makes it work

| | |
|---|---|
| Shape | 1 FE (**gcn-18**) + 1 Sirius CN per host, GPU 0 each |
| Control LAN | `bond0` `10.87.140.32/27` — gcn-09 = **10.87.140.44**, gcn-18 = **10.87.140.53** |
| Transport | nixl/UCX `cuda_ipc` over the **MNNVL fabric** |
| Cross-host canary (17+18, 2026-08-12) | **98.0 GB/s** (0.41 GB/s before the fabric-arena fix) |

The load-bearing change is `SIRIUS_EXCHANGE_STAGING_ARENA=fabric`. `cudaMalloc`'s IPC handle is
node-local **by construction**, so a peer on another host can never map the staging arena; UCX
silently falls back to a host bounce at ~0.4 GB/s, which is below the transport's hard 2.0 GB/s
admission floor, so the peer is refused and **no distributed query can run at all**.

GPUDirect RDMA is *not* the answer on these hosts: `nvidia_peermem` is not loaded, and even with
dma-buf forced the mlx5 memory domain never advertises the `cuda` memory type. Do not put
`rc_mlx5` in `UCX_TLS` — the CN will fail to start. `configs/gb200-4gpu/engine-a-2host.env`
defaults `UCX_TLS` to include `rc_mlx5`; the launch lines below **override** it.

---

## 1. Prerequisites (verify, don't assume)

**[both]**

```bash
uname -m                                   # aarch64
nvidia-smi -L                              # 4x NVIDIA GB200
# bond0: 10.87.140.44 on gcn-09, 10.87.140.53 on gcn-18, /27
python3 -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
nvidia-smi -q | grep -A3 Fabric            # State: Completed / Status: Success
```

The **ClusterUUID must match on both hosts** — that is what makes MNNVL a single fabric domain.
The 17+18 pair was `3482beb4-a3cd-48a4-9b6c-a6ba43bc59a4`. **Re-check on 09 and 18**; do not
assume the same UUID.

```bash
# Same absolute path on BOTH hosts (IBM Scale). Do not copy datasets onto the box.
ls -d /scratch/sirius/datasets/tpch_sf100
```

`$HOME` is NFS (`master:/home`) and therefore **shared**: the repo, the built engine and the CN
binary are the same files on both hosts. `/scratch` is GPFS (Scale) and is **also** the same
namespace. `/raid` is node-local NVMe: writable on gcn-18 historically, **root-owned / not
writable on gcn-09**. Do not put TPC-H or FE meta on `$HOME`. Do not point `FILES()` at a path
that exists on only one host (`/opt/sirius-ci/datasets`, `/raid/prestouser/aocsa/tpch_parquet_*`
on 18 only).

Nightly CI takes all 4 GPUs **~02:00–03:50 UTC** on this fleet. Do not bring up CNs in that
window (`nvidia-smi --query-compute-apps=pid` must be empty first).

---

## 2. Build

Build once (NFS + Scale share the tree). Follow
[`BUILD-AND-SMOKE.md`](../../bench/gb200-4gpu/BUILD-AND-SMOKE.md) — aarch64 shims, no
`pixi run cn-build`, caches under `/scratch/prestouser/aocsa`.

**[either host]**

```bash
source /scratch/prestouser/aocsa/env.sh          # BIG, DATASETS, TOOLS_DIR, CUDA 13, pixi/cargo
cd /home/prestouser/aocsa/sirius
pixi run make

cd /home/prestouser/aocsa/sirius/experimental/starrocks
pixi run -e fe fe-build
test -x starrocks/output/fe/bin/start_fe.sh

SHIMS=$TOOLS_DIR/toolchain-shims
pixi run --manifest-path "$PWD/pixi.toml" bash -c "
  set -euo pipefail
  export PATH=$SHIMS:\$PATH
  export RUSTFLAGS=\"-C link-arg=-L$SHIMS -C link-arg=-lnvidia-ml\"
  source scripts/cn-env.sh
  cargo build --release -p sirius-starrocks-cn
"
readelf -d target/release/sirius-starrocks-cn | grep -Ei 'nixl|sirius'
# NEEDED: libnixl.so, libnixl_build.so, sirius.duckdb_extension
```

`pixi run cn-build` **cannot** carry those shims: pixi prepends conda `g++`/`ld`. PATH must be
set *inside* `pixi run bash -c`.

**Verify the binaries are newer than any running process.** ninja writes a *new inode*, so a CN
started before a rebuild keeps the old `.so` mapped forever while `ls` shows the new one — two
hosts then silently run different engines:

```bash
stat -c '%i %n' /scratch/prestouser/aocsa/build/release/extension/sirius/sirius.duckdb_extension
grep -m1 sirius.duckdb_extension /proc/$(pgrep -f '[s]irius-starrocks-cn')/maps   # inodes must match
```

---

## 3. Config the FE for two hosts, then start the cluster

Packaged `starrocks/output/fe/conf/fe.conf` is a **single-host** file: `priority_networks` is
unset and `meta_dir` defaults to NFS `output/fe/meta`. Two-host needs both fixed **[18]**
before the first FE start.

```bash
# [18] — packaged conf (fe-build copies experimental/starrocks/conf/fe.conf here)
FE_CONF=/home/prestouser/aocsa/sirius/experimental/starrocks/starrocks/output/fe/conf/fe.conf
# Advertise bond0, not loopback. /27 covers .44 (09) and .53 (18).
grep -q 'priority_networks = 10.87.140.32/27' "$FE_CONF" || \
  printf '\npriority_networks = 10.87.140.32/27\n' >> "$FE_CONF"

# BDB JE must not live on NFS. Prefer gcn-18 local NVMe if writable; else Scale.
META=/raid/prestouser/sr-eng-a-2node/fe/meta
if ! mkdir -p "$META" 2>/dev/null; then
  META=/scratch/prestouser/aocsa/fe/meta
  mkdir -p "$META"
fi
grep -q '^meta_dir' "$FE_CONF" && sed -i "s|^meta_dir.*|meta_dir = $META|" "$FE_CONF" \
  || printf '\nmeta_dir = %s\n' "$META" >> "$FE_CONF"
rm -rf "$META"/* && mkdir -p "$META"
```

Required. A metadata dir bootstrapped under a different address — or left half-written by an
interrupted start — makes the FE exit with `current node is not added to the cluster, will exit`.
Engine A creates no persistent tables (everything is `FILES()` over parquet), so nothing is lost.

### Memory (path A — `cn-2host.sh` always passes `--gpu-memory-limit` / `--host-memory-limit`)

Usable HBM is **184.00 GiB** per GPU (not nameplate). Staging is a bare `cudaMalloc` **outside**
the RMM pool. Occupancy = `GPU_MEM + STAGING + 0.76 GiB`. Units: `GiB` = 1024³; `GB` in YAML is
1000³. See the tuning runbook §1–§2.

`cn-2host.sh` defaults `GPU_MEM=140GiB` `HOST_MEM=160GiB`. `engine-a-2host.env` sets
`SIRIUS_EXCHANGE_STAGING_BYTES=16GiB` (unset = no arena, every remote exchange fails).

Those defaults match the **SF100** 4-CN single-box preset. This experiment is **1 CN per host**,
so `HOST_MEM=160GiB` is conservative (one CN on ~957 GiB LPDDR). Do **not** copy the 4-CN
`HOST_MEM=112GiB` SF1000 split here — that number exists to leave page cache for *four* CNs on
*one* box.

`--membind` is **0 or 1 only**. Nodes 2 / 10 / 18 / 26 **are** GPU HBM. `cn-2host.sh` refuses any
other node. Never `--interleave=all`.

`CN_CPUS="0-71"` gives the single CN GPU 0's whole socket. The launcher default is the 4-CN
disjoint split (`0-35 36-71 …`).

| SCALE_FACTOR | GPU_MEM | STAGING | HOST_MEM (1 CN/host) | watchdog | RPC timeout |
|---|---|---|---|---|---|
| 100 (this recipe) | 140 GiB | 16 GiB | 160 GiB | 0 (or 120 unattended) | 60 s |
| 500 | 132 GiB | 24 GiB | 160 GiB | 180 | 180 s |
| 1000 | 128 GiB | 32 GiB | 160 GiB | 300 | 300 s |

Raise `STAGING` and lower `GPU_MEM` by the same amount. `SIRIUS_CN_RPC_TIMEOUT_SECS` is an env
var (1–3600, fail-closed at bring-up), not a rebuild. `OOM at operator HASH_JOIN` after 100
retries is **not** the same bug as arena exhaustion.

### 3b. FE + CN **[18]**

```bash
source /scratch/prestouser/aocsa/env.sh
unset CUDA_VISIBLE_DEVICES
cd /home/prestouser/aocsa/sirius/experimental/starrocks

SIRIUS_EXCHANGE_STAGING_ARENA=fabric \
UCX_TLS=cuda_copy,cuda_ipc,tcp,self \
GPU_MEM=140GiB HOST_MEM=160GiB \
NUM_CNS_PER_HOST=1 CN_NODE="0" CN_CPUS="0-71" \
  ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.53
```

### 3c. CN only **[09]**

Wait until gcn-18's CN is alive first (§4), then:

```bash
source /scratch/prestouser/aocsa/env.sh
unset CUDA_VISIBLE_DEVICES
cd /home/prestouser/aocsa/sirius/experimental/starrocks

SIRIUS_EXCHANGE_STAGING_ARENA=fabric \
UCX_TLS=cuda_copy,cuda_ipc,tcp,self \
GPU_MEM=140GiB HOST_MEM=160GiB \
NUM_CNS_PER_HOST=1 CN_NODE="0" CN_CPUS="0-71" \
  ./benchmarks/cn-2host.sh 10.87.140.44 10.87.140.53 --no-fe
```

**`SIRIUS_EXCHANGE_STAGING_ARENA=fabric` and `UCX_TLS=...` must be set on BOTH hosts.** A fabric
arena on one side and `cudaMalloc` on the other cannot map each other. If `engine-a-2host.env`
wins on TLS, the CN fails to register the staging arena with nixl (`ibv_reg_mr ... Bad address`).

**Leave both in the foreground.** The launcher `wait`s on its child and its cleanup trap tears the
cluster down when the script exits; Ctrl-C kills the FE too.

Logs: `/tmp/fe.log` **[18]**, `/tmp/cn-53-0.log` **[18]**, `/tmp/cn-44-0.log` **[09]**.

---

## 4. Verify

**[either]** — `mysql` lives only in the pixi env:

```bash
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
mysql -h 10.87.140.53 -P 9030 -uroot --vertical -e "SHOW COMPUTE NODES"
```

Expect **exactly two** rows, `IP: 10.87.140.44` and `IP: 10.87.140.53`, both `Alive: true` with
**recent** `LastHeartbeat`. Count with `awk -F'\t' '$9=="true"'` — `grep -c true` overcounts.

Two *rows* is not two *nodes* — the FE persists a registration after the process dies, showing
`Alive: false`, `StatusCode: DISCONNECTED`, `ErrMsg: java.net.ConnectException: Connection refused`.
Wipe `meta_dir` (§3) rather than measuring a ghost CN.

`LastStartTime` renders in a different timezone than the FE's own `now()` (8 h off here) — do not
correlate it against file mtimes.

Use `--vertical`; this `mysql` rejects `\G` with `-e`.

Confirm GPU 0 only, one uuid per host:

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv   # [both]
```

### The canary — the gate on everything

```bash
# [18]
grep -a 'nixl bandwidth canary' /tmp/cn-53-0.log | tail -2
```

Healthy (17+18 measured 98.0 GB/s; 09+18 will print gcn-09's advertise address):

```
nixl bandwidth canary peer=10.87.140.44:9102 gbps="…" bytes=16777216
```

**Anything near 0.4 GB/s means the fabric arena did not take**, and you will see
`below the 2 GB/s floor — Refusing the transport tier`. Check that
`SIRIUS_EXCHANGE_STAGING_ARENA=fabric` reached *both* CNs.

---

## 5. Prove distribution

Dataset is Scale, same path on both hosts. This measures GPFS as well as the engine; it is not
a like-for-like vs the 17+18 `/raid` NVMe numbers.

```bash
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
cat > /tmp/q.sql <<'EOF'
SET new_planner_agg_stage = 2;
WITH lineitem AS (SELECT * FROM FILES('path'='file:///scratch/sirius/datasets/tpch_sf100/lineitem/*.parquet','format'='parquet'))
SELECT l_suppkey, count(*) AS n FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 5;
EOF
mysql -h 10.87.140.53 -P 9030 -uroot < /tmp/q.sql
```

Expect ~600 rows per supplier (600M lineitems / 1M suppliers) in well under a second.

> Use a **plain column** as the shuffle key. `l_orderkey % 4096` fails with
> `Unsupported expression in projection (falling back to CPU): mod(...)`.

**FE placement** — zero fragments deployed:

```bash
{ echo "SET new_planner_agg_stage = 2;"; echo "EXPLAIN SCHEDULER"; tail -n +2 /tmp/q.sql; } \
  | mysql -h 10.87.140.53 -P 9030 -uroot | grep -E 'PLAN FRAGMENT|INSTANCE\(|BE: '
```

The scan fragment must list **two distinct** `BE:` ids — map them via `ComputeNodeId`.

**NIXL transfer proof:**

```bash
# [18]
grep -aE 'transmitted batches via nixl|received remote batches' /tmp/cn-53-0.log | tail -4
```

A `transmitted ... dest=10.87.140.44:9102` line matched by a `received remote batches` line on
**[09]** `/tmp/cn-44-0.log` is, by construction, work crossing the machine boundary.
`relayed native batches across a fragment boundary` is the same-process short circuit — if that
is all you see, nothing crossed.

Oracle (plain DuckDB — not `$REPO/build/release/duckdb`, which auto-loads Sirius):

```bash
source /scratch/prestouser/aocsa/env.sh
cd /scratch/prestouser/aocsa/oracle
pixi run python -c "import duckdb; print(duckdb.sql(\"SELECT count(*) FROM read_parquet('/scratch/sirius/datasets/tpch_sf100/lineitem/*.parquet')\").fetchall())"
```

---

## 6. TPC-H sweep

`run-abc.sh` launches `cluster4-numa.sh` (4 CNs, one box). **Do not use it here.** `bench.sh`
talks to `127.0.0.1`, so run the sweep **on gcn-18**.

`QUERY_TIMEOUT` default is **30 s** — unusable above SF1. Scale like the tuning runbook:
warm `max(90, 1.8×SF)`, cold `max(300, 6×SF)`.

```bash
# [18]
source /scratch/prestouser/aocsa/env.sh
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
mysql -h 127.0.0.1 -P 9030 -uroot -e \
  "SET GLOBAL enable_pipeline_engine=true; SET GLOBAL pipeline_dop=36; SET GLOBAL query_timeout=1800;"

cd /home/prestouser/aocsa/sirius/experimental/starrocks
# q11 TPC-H clause 2.11.2 is 0.0001/SF. At SF100 that is 0.000001. Revert after.
sed -i 's/0\.0001000000/0.000001000000/' benchmarks/tpch/queries/q11.sql

TPCH_DATA=/scratch/sirius/datasets/tpch_sf100 \
FE_PORT=9030 QUERY_TIMEOUT=180 COLD_TIMEOUT=600 MIN_BACKENDS=2 \
  ./benchmarks/tpch/bench.sh --cold /scratch/prestouser/aocsa/bench-2node-A/timings.csv 3

git checkout HEAD -- benchmarks/tpch/queries/q11.sql
```

`pipeline_dop=36` because `--physcpubind=0-71` makes each CN report 72 cores. Pin it, or an A/B has
a free variable. `runs` is positional and mandatory before any query subset —
`bench.sh out.csv q05` sets `RUNS=q05`.

Warm NVRTC (`$HOME/.cudf/$VERSION/$ARCH`) and `SET expression_evaluator_strategy = 'ast_jit'`
before timing if you care about a few percent on the suite (tuning runbook §4).

---

## 7. Results obtained on the *previous* pair — not 09+18

Warm medians, SF100, 2 CNs across **gcn-17 + gcn-18** (2026-08-12), node-local `/raid` parquet:

| q01 | q02 | q03 | q04 | q05 | q06 | q07 | q11 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1687 | 505 | 576 | 472 | 1430 | 473 | 662 | 630 |

**8 measured passes, 1 real failure, 13 unmeasured — not "8/22".** Read §8 before quoting anything.
A 09+18 sweep on `/scratch` is a different storage path; do not treat a delta as "MNNVL got faster".

Engine B (stock StarRocks 3.5.20, 1 BE per host, same files) for comparison: 21/22, geomean
2762 ms. See `2NODE-ENGINE-B-RESULTS.md`.

---

## 8. Known failures — read before interpreting a sweep

**The harness has no correctness gate.** `bench.sh` scores `pass` on exit code + non-empty output +
no `ERROR` on line 1. Row counts are recorded and **never compared**. A query returning 1 row
instead of 100,000 registers as a fast win. Oracle every quoted query against DuckDB.

**The staging-lease leak cascades — this is the big one.** A failed query strands its staging
leases (there is no `cancel_plan_fragment`), and the arena never recovers:

```
q08:  OOM at operator HASH_JOIN (index 0)                    <- real, 165.8 s, 100 retries
q09:  exchange staging arena exhausted: 2033670144 free of 17179869184
q22:  exchange staging arena exhausted: 2659840 free ... with 35 leases outstanding
```

16 GiB → 2.6 MB, monotonically, across 14 queries. **Every query after the first failure is
collateral, not a verdict** — their sub-second "refused" times are the tell. Restart both CNs after
any failure before trusting another row. Watchdog (`SIRIUS_QUERY_WATCHDOG_SECS=120`) converts a
wedge into a clean error so it stops poisoning the sweep; it does not fix the query.

`RESTART_CMD` normally automates this, but it cannot: restarting requires a shell on **both**
hosts.

**q08 is a genuine OOM** at this shape — with 2 CNs each holds ~half the data, so the build side per
GPU is larger than in the recorded 4-CN single-host baseline (where q08 failed differently, on a
60 s RPC timeout). More `GPU_MEM` did not fix HASH_JOIN OOM on RTX; the lever is more CNs or
engine work.

**Do not compare against the 4-CN baseline** in `.claude/skills/tpch-bench/SKILL.md` or against
this pair's `/scratch` Scale tree vs 17+18 `/raid` NVMe. Storage and encoding (`tpch_sf100` vs
`tpch_parquet_sf100_f64`) move scan-bound queries on their own.

---

## 9. Teardown

**[both]** — this host only; run it on 09 and on 18:

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks
./benchmarks/stop-cn-2host.sh
```

It SIGTERMs then SIGKILLs by `/proc/<pid>/exe` (CN binary, `java`+`StarRocksFE`, `start_fe.sh`,
`cn-2host.sh`). It never `pkill -f`. `nvidia-smi --query-compute-apps=pid` must be empty after.
Idle `memory.used` is ~30–33 MiB, never 0.

---

## 10. Open work

* **Staging-lease leak** — one failed query permanently degrades the cluster. Blocks any unattended
  sweep. Not root-caused.
* **q08 `HASH_JOIN` OOM** at SF100 with 2 CNs. Does the downgrade/spill path apply here?
* **The 100-retry loop** burns 165 s before failing an OOM that retrying cannot fix.
* **No two-host restart script** — `stop-cn-2host.sh` is per-host; relaunch still needs both shells.
* **Correctness** — no SF100 result on this pair has been diffed against the DuckDB oracle.
* **09+18 ClusterUUID / canary GB/s** — not yet measured; re-run §1 and §4 before claiming MNNVL.
