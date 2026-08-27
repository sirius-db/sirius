# Two-machine Sirius CN bring-up — phased implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to work this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prove that StarRocks schedules plan fragments across two physical machines
(`presto-gb200-gcn-17`, `presto-gb200-gcn-18`) and that NIXL moves GPU-resident data between those
fragments over the best available cross-host path — first at the minimal topology (Phase 1: 2 CNs,
one GPU each), then at full density (Phase 2: 8 CNs, 4 GPUs per box).

**Architecture:** One StarRocks FE on gcn-17 coordinates N Sirius Rust compute nodes spread over
both hosts. The FE keys each CN by `(advertise_host, heartbeat_port)` and dispatches fragments to
`advertise_host:brpc_port`; nothing in that path is host-local. Control plane (heartbeat, thrift,
brpc dispatch, registration SQL) rides `bond0`. The GPU data plane is separate: NIXL/UCX picks its
own interface via `UCX_NET_DEVICES`, so exchange payloads ride the 4×400G RoCE fabric while control
traffic stays on `bond0`.

**Tech stack:** StarRocks FE (patched, from `starrocks/output/fe`), `sirius-starrocks-cn` (Rust,
`target/release/`), NIXL + UCX 1.21.0, CUDA/RMM on GB200, `numactl`, `pixi`.

**Companion documents:**
- [2NODE-RUNBOOK.md](2NODE-RUNBOOK.md) — engine A reference; §2 has the measured transport table.
- [2NODE-RUNBOOK-GAPS.md](2NODE-RUNBOOK-GAPS.md) — corrections to that runbook. **Read part A before
  using the runbook**; several of its prescribed commands are wrong.
- [2NODE-ENGINE-B-TUTORIAL.md](2NODE-ENGINE-B-TUTORIAL.md) — the stock-StarRocks baseline (Phase 3).

---

## Global Constraints

Every task inherits these. Violating any one produces a cluster that still answers queries while
silently measuring the wrong thing.

| # | Constraint | Why |
|---|---|---|
| G1 | **Engine A and engine B must never run simultaneously**, on either host. | Both FEs bind `9030`, and both would claim all 144 cores. Check `pgrep -af 'sirius-starrocks-cn\|starrocks_be\|StarRocksFE'` on both hosts before every bring-up. |
| G2 | **Do not measure between 02:00 and ~03:50 UTC.** | A nightly CI job takes all 4 GPUs on this box. Timings taken in that window are garbage. |
| G3 | **Datasets must exist at the same absolute path on both hosts, on node-local storage.** Satisfied today by `/raid/prestouser/aocsa/tpch_parquet_sf{100,500}`; verify with Task 0.2 before every run. | The FE assigns byte ranges without host awareness, and sends `FILES()` schema inference to a *randomly shuffled* alive backend — so a one-host dataset fails intermittently, not deterministically. Never `$HOME` (NFS `master:/home`); never `/scratch` (GPFS — that measures GPFS); never `/opt/sirius-ci/datasets` (gcn-18 only). |
| G4 | **Export `TOOLS_DIR` and `UCX_TLS` *before* `source scripts/cn-env.sh`.** | `cn-env.sh:35` fills them in only when unset, so sourcing first silently discards the cross-host TLS choice. |
| G5 | **`unset CUDA_VISIBLE_DEVICES` before launching any CN.** | An inherited value *overrides* `--gpu-device` and is only warned about — every CN collapses onto one GPU. Confirmed in `--help`: "exported as `CUDA_VISIBLE_DEVICES` before engine bring-up (an already-exported value wins)". |
| G6 | **`--membind` may only ever be `0` or `1`.** | `numactl -H` reports 34 nodes; nodes 2/10/18/26 are 188,416 MB of GPU HBM with **zero CPUs**. Binding host pages there eats the HBM of a GPU a CN is computing on. |
| G7 | **Pin `SIRIUS_CN_USE_SIRIUS_DATASOURCE` explicitly in every run.** | It appears in no committed config — the historical SF100 anchor set it via an uncommitted shell export. It is worth ~20× on scan-bound queries. Unpinned, the arm is unreproducible. |
| G8 | **Pin `pipeline_dop` identically across every arm you intend to compare.** | It derives from the CPU count the CN reports (`src/lib.rs:407` `available_parallelism()`), so a cpubound CN reports fewer cores and resolves to a different DOP than an unpinned one. |
| G9 | **Record an `INVOCATION-<arm>.txt` next to every CSV**, containing the literal shell you ran including every `export`. | This is the only thing that makes a number reproducible. |

**Verified environment facts** (measured on gcn-18 this session unless noted):

| Fact | Value |
|---|---|
| Hosts | gcn-17 = `10.87.140.52`, gcn-18 = `10.87.140.53`, both on `bond0`, CIDR `10.87.140.32/27` |
| NUMA ↔ GPU | GPU0/GPU1 → node 0, CPUs 0-71; GPU2/GPU3 → node 1, CPUs 72-143 (`engine-a.env:376-380`) |
| UCX | 1.21.0 at `/home/prestouser/aocsa/tools/ucx-install`; `UCX_MAX_RNDV_RAILS` default **2** (so `=4` is meaningful) |
| IMEX | `/etc/nvidia-imex/nodes_config.cfg` and `.pending` are **byte-identical** — no pending node-map mismatch |
| CN binary | `target/release/sirius-starrocks-cn`, built 2026-08-09 |
| CN flag defaults | `--fe-host 127.0.0.1`, `--advertise-host 127.0.0.1`, `--bind-host 0.0.0.0`, `--heartbeat-port 9050`, `--thrift-port 9060`, `--http-port 8040`, `--brpc-port 8060`, `--starlet-port 9070`, `--registration-max-attempts 120` (all confirmed via `--help`) |
| Datasets | **Already on BOTH hosts at the identical path**: `/raid/prestouser/aocsa/tpch_parquet_sf100` (26 GB) and `tpch_parquet_sf500` (132 GB). **No SF1** — see Task 0.2. gcn-18 additionally has `/opt/sirius-ci/datasets/tpch_sf{1,10,30,50,100,500,1000}`, which is **not** replicated and must not be used for a two-host run. |
| SF100 `lineitem` | 17,187,602,838 B across 6 files (`part.0..5.parquet`); `orders` 5,051,383,146 B; `nation` 2,250 B (single file — legitimately lands on one host) |
| Filesystems | `/home` is **NFS** (`master:/home`, addr `10.87.140.8`) — one shared tree, so Tasks 1.1/1.2/2.1 are done **once, not per host**, and every relative per-CN path (`--engine-dir`) must be host-suffixed. `/raid` is local `ext4` on `/dev/md0` (hence G3). |
| FE metadata | `starrocks/output/fe/meta/image/ROLE` currently reads `name=127.0.0.1_9010_1786215315408` — bootstrapped on loopback. **Task 1.1 step 3b must migrate it** or the FE `System.exit(-1)`s. |

---

## File Structure

| Path | Responsibility | Phase |
|---|---|---|
| `benchmarks/2NODE-BRINGUP-PLAN.md` | this plan | — |
| `benchmarks/cn-2host.sh` | **new.** Two-host CN launcher: takes a host role, launches N CNs with correct `--advertise-host`/`--fe-host`, NUMA pinning, and per-CN GPU/port assignment. | 2 |
| `configs/gb200-4gpu/engine-a-2host.env` | **new.** Sourced overlay: the cross-host `UCX_*` block and the `SIRIUS_CN_NIXL_WARMUP_*` values, parameterised by `NUM_CNS`. | 1 |
| `conf/fe.conf` | **modify.** `priority_networks` loopback → `10.87.140.32/27`. | 1 |
| `benchmarks/tpch/bench.sh` | unchanged — run it *on gcn-17* where its hardcoded `127.0.0.1` is correct. | 1, 2 |

---

# Phase 0 — Preconditions

Nothing in Phase 1 or 2 can be trusted until these pass. Phase 0 is shared by both phases and by
engine B.

### Task 0.1: Resolve the gcn-17 unknowns

Everything in the companion docs was measured on gcn-18. `ssh` to gcn-17 was blocked by the tooling
that wrote them, so **every gcn-17 fact is currently inference.** Resolve before anything else.

**Files:** none (observation only)

- [ ] **Step 1: Collect the facts on both hosts**

`benchmarks/collect-host-facts.sh` gathers every value this table needs in one read-only pass and
writes `benchmarks/host-facts-<hostname>.txt`. The repo lives on NFS at the identical path on both
machines, so the output of a run on either host is readable from the other — no ssh required to
*read* it.

```bash
# on EACH host
cd ~/aocsa/sirius/experimental/starrocks && bash benchmarks/collect-host-facts.sh
```

gcn-18's baseline is already captured at `benchmarks/host-facts-presto-gb200-gcn-18.txt`
(2026-08-11). Only gcn-17 remains.

- [ ] **Step 2: Check each against the gcn-18 baseline**

Expected, and each is a **stop** if it differs:

| Check | Required value | If it differs |
|---|---|---|
| `uname -m` | `aarch64` | stop — the whole plan assumes it |
| `bond0` | `10.87.140.52/27` | stop — every command in this plan hardcodes it |
| NUMA node 0 / node 1 | CPUs `0-71` / `72-143` | stop — recompute every `--physcpubind` |
| `/raid` | `/dev/md0 ext4`, writable, and holding `aocsa/tpch_parquet_sf{100,500}` | stop — pick a different node-local path and change G3 everywhere |
| `mlx5_0/1/4/5` | present and `ACTIVE / 400 Gb/sec / Ethernet`, mapped to `enp3s0np0`, `enP2p3s0np0`, `enP16p3s0np0`, `enP18p3s0np0` | stop — recompute `UCX_NET_DEVICES` |
| `nvidia-smi -L` | 4 × GB200, idle | stop for Phase 2; Phase 1 needs only GPU 0 |
| IMEX `ClusterUUID` | **`3482beb4-a3cd-48a4-9b6c-a6ba43bc59a4`** — must match gcn-18 exactly | not a stop, but the MNNVL row of the transport table is void: the two boxes are not in one fabric domain |
| GPU↔CPU affinity | GPU0/1 → `0-71`, GPU2/3 → `72-143` | stop — `CN_NODE="0 0 1 1"` and every `--physcpubind` is wrong |

- [ ] **Step 3: Confirm `/raid/prestouser` is writable**

The script's sections 5, 6 and 7 cover the RoCE links, the MNNVL fabric and `/raid` writability, so
there is nothing to run separately. What matters is the comparison:

```bash
cd ~/aocsa/sirius/experimental/starrocks/benchmarks
diff <(sed -n '/5. ROCE/,/6. MNNVL/p' host-facts-presto-gb200-gcn-17.txt) \
     <(sed -n '/5. ROCE/,/6. MNNVL/p' host-facts-presto-gb200-gcn-18.txt)
grep -h ClusterUUID host-facts-presto-gb200-gcn-1*.txt
```

Expected: the HCA blocks differ only in nothing (same four devices, same netdev names, all
`ACTIVE / 400 Gb/sec (4X NDR) / Ethernet`), and **both ClusterUUIDs read
`3482beb4-a3cd-48a4-9b6c-a6ba43bc59a4`**.

Any RoCE port `DOWN` removes a rail and invalidates the multi-rail throughput figure. A differing
ClusterUUID means the two boxes are not in one IMEX domain, so the 765 GB/s MNNVL row of the
transport table does not exist between them — which does not block Phase 1 (the CN cannot reach
that path anyway) but does void a documented ceiling.

Already resolved on gcn-18, and not worth re-deriving: the IMEX node-map pending-reload hazard
(`nodes_config.cfg` and `.pending` are byte-identical) and `UCX_MAX_RNDV_RAILS` (default 2 in the
installed UCX 1.21.0, so `4` is a real change).

### Task 0.2: Verify the datasets are identical on both hosts

**No replication is needed.** SF100 and SF500 already exist at the identical absolute path on both
machines, on node-local ext4:

```
/raid/prestouser/aocsa/tpch_parquet_sf100     26 GB
/raid/prestouser/aocsa/tpch_parquet_sf500    132 GB
```

This task is now a *check*, not a copy — but it is not optional. A silent divergence between the
two trees produces intermittent failures (the FE sends `FILES()` schema inference to a randomly
shuffled backend), which is the hardest failure mode in this whole plan to diagnose.

> **Do not use `/opt/sirius-ci/datasets/`.** That tree exists on gcn-18 only. Pointing a two-host
> run at it means gcn-17 is handed byte ranges for files it cannot open.

**Files:** none (verification)

- [ ] **Step 1: Confirm identical inventory on both hosts**

Run on each host — from your own terminal on gcn-17 if ssh is unavailable to the agent:

```bash
find /raid/prestouser/aocsa/tpch_parquet_sf100 -type f -printf '%P %s\n' | sort | md5sum
```
Expected: **the same md5sum on both hosts.** This hashes relative path + size for every file, so
it catches a missing file, a truncated file, or a half-finished copy. If they differ, stop and
reconcile before anything else — every later result would be unreproducible.

- [ ] **Step 2: Confirm the storage is node-local, not NFS or GPFS**

```bash
df -PT /raid/prestouser/aocsa/tpch_parquet_sf100 | tail -1
```
Expected: `/dev/md0  ext4` (measured on gcn-18). Anything reporting `nfs4` or `gpfs` means the scan
becomes the benchmark.

- [ ] **Step 3: Record the `lineitem` size — the split arithmetic depends on it**

```bash
du -sb /raid/prestouser/aocsa/tpch_parquet_sf100/lineitem
ls -l  /raid/prestouser/aocsa/tpch_parquet_sf100/lineitem
```
Expected (measured on gcn-18): **17,187,602,838 B** across 6 files, `part.0.parquet` …
`part.5.parquet`. The split rule is
`numInstances = clamp(totalBytes / min_bytes_per_broker_scanner, 1, nodes × parallelInstanceNum)`
with `min_bytes_per_broker_scanner = 67108864` (64 MiB), so 17.2 GB → 256 uncapped, capped by the
node count. At any CN count in this plan that is far more than enough to reach every host.

> **There is no SF1 on these hosts.** The original Phase 1 test used SF1 `lineitem` (162 MB → exactly
> 2 instances, one per host) as the minimal honest split. That is not available, so Phase 1 uses
> SF100 instead. The consequence is only that the test is *less* minimal — SF100 splits far more
> widely, so both hosts are guaranteed ranges. Small tables still do not split: `nation` is a single
> 2,250 B file and will land on exactly one host. That is correct behaviour, not a failure.

### Task 0.3: Prove the cross-host data plane standalone, before any cluster

This isolates the transport from StarRocks entirely. If it fails here, no amount of FE/CN
configuration will help.

**Files:** none (uses `scripts/nixl-echo-2node.sh`)

- [ ] **Step 1: Run the echo harness on the default (failing) config, to see the floor**

```bash
cd ~/aocsa/sirius/experimental/starrocks
./scripts/nixl-echo-2node.sh 2>&1 | tail -20
```
Expected: ~**0.37 GB/s**, host-staged TCP. This is the documented default-config failure and
confirms the harness is wired up. It is *below* the 2.0 GB/s admission floor.

- [ ] **Step 2: Re-run on the multi-rail GPUDirect RDMA path**

```bash
export TOOLS_DIR=/home/prestouser/aocsa/tools
export UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0
export UCX_MAX_RNDV_RAILS=4
./scripts/nixl-echo-2node.sh 2>&1 | tail -20
```
Expected: **tens of GB/s**, byte-verified both legs. The runbook's figure is 97 GB/s, but that was
measured *without* `cuda_ipc` in `UCX_TLS`; this line retains it (see GAPS §A), so treat the number
you get here as the new reference and record it.

- [ ] **Step 3: Confirm the MNNVL path still works, for the record**

```bash
NIXL_ECHO_ARENA=fabric ./scripts/nixl-echo-2node.sh 2>&1 | tail -20
```
Expected: ~**765 GB/s** (`leg1 763.07 / leg2 767.80` was the 2026-08-11 measurement),
`verified true`. This path is **not reachable from the CN** — `exchange_staging_arena.cpp:42-48` is
an unconditional `cudaMalloc`. It is recorded here as the ceiling the CN could reach if the arena
gained a VMM/fabric allocation path.

- [ ] **Step 4: Write down which path Phase 1 will use**

Append the three measured numbers to `benchmarks/2NODE-RUNBOOK.md` §2 with today's date, replacing
the unsourced 48.7/97 rows. **Best available today = multi-rail GPUDirect RDMA**, because the CN's
arena is `cudaMalloc` and cannot export a fabric handle.

---

# Phase 1 — Two CNs, one GPU each

The minimal two-machine topology: exactly one CN per box, each owning GPU 0. **This shape is chosen
first precisely because it has a single peer pair.** Every exchange in every query is cross-host, so
there is no same-host `cuda_ipc` pair that could mask a broken fabric, and any failure is
unambiguous.

```
gcn-17 (10.87.140.52)  CN0  GPU0  NUMA node 0, cpus 0-71   ports 9100-9104   + FE
gcn-18 (10.87.140.53)  CN0  GPU0  NUMA node 0, cpus 0-71   ports 9100-9104
```

Ports `9100-9104` are deliberately **not** the CN defaults (`9050/9060/8060/8040`), which collide
with stock BE ports — this keeps engine A and engine B configs from overlapping even though G1
forbids running them together.

### Task 1.1: Point the FE at the control LAN

**Files:**
- Modify: `experimental/starrocks/conf/fe.conf`
- Propagate to: `experimental/starrocks/starrocks/output/fe/conf/fe.conf`

- [ ] **Step 1: Confirm the current value is loopback**

```bash
cd ~/aocsa/sirius/experimental/starrocks
grep -n 'priority_networks\|run_mode' conf/fe.conf
```
Expected: `priority_networks = 127.0.0.1/32` and `run_mode = shared_data`.

- [ ] **Step 2: Change `priority_networks` only**

```bash
sed -i 's|^priority_networks *= *127\.0\.0\.1/32|priority_networks = 10.87.140.32/27|' conf/fe.conf
grep -n priority_networks conf/fe.conf
```
Expected: `priority_networks = 10.87.140.32/27`. The `/27` covers `.32-.63` and therefore both
`.52` and `.53`; the FE breaks on the first local address inside the CIDR.

**Leave `run_mode = shared_data` alone.** The runbook's stated reason for it is wrong (the
fragment scheduler is not BE-only), but the directive is right — see GAPS §A row 2.

- [ ] **Step 3: Propagate to the tree the FE actually reads**

This is the step the runbook omits, and without it nothing changes.

```bash
cp conf/fe.conf starrocks/output/fe/conf/fe.conf
grep -n priority_networks starrocks/output/fe/conf/fe.conf
```
Expected: the new value. `start_fe.sh:90` reads `$STARROCKS_HOME/conf/fe.conf` — i.e.
`output/fe/conf/`, not the tracked `conf/`.

- [ ] **Step 3b: Migrate the FE's persisted identity — MANDATORY, or step 4 exits and hangs**

Changing `priority_networks` changes `selfNode`, but **not** the identity already written into the
FE metadata. Verified on the current tree:

```bash
cat starrocks/output/fe/meta/image/ROLE
```
Today this prints `name=127.0.0.1_9010_1786215315408`, `hostType=IP` — the FE was bootstrapped on
loopback, and `meta_dir` is unset (`conf/fe.conf:38` is commented) so the metadata is
`starrocks/output/fe/meta`.

On the next start `NodeMgr.getClusterIdAndRole` sees `ROLE`+`VERSION` present, so `isFirstTimeStartUp
= false` and the persisted `Frontend` host stays the literal string `127.0.0.1`, while `selfNode`
becomes `10.87.140.52`. After journal replay `GlobalStateMgr:1321` → `nodeMgr.checkCurrentNodeExist()`
(`NodeMgr.java:675-683`) calls `unprotectCheckFeExist` (`NodeMgr.java:977-981`), which matches with
`NetUtils.isSameIP` — pure string equality, **no loopback aliasing** — so it returns `null`, the FE
logs `current node is not added to the cluster, will exit` and calls `System.exit(-1)`.

`ALTER SYSTEM MODIFY FRONTEND HOST` is **not** an option here: `NodeMgr.java:827-829` throws
`can not modify current master node` on a single-FE cluster.

Pick one, on gcn-17, **before** step 4:

```bash
# (a) one-shot reset -- keeps the catalog. checkCurrentNodeExist returns early (NodeMgr.java:676),
#     the node name is regenerated (NodeMgr.java:334-341) and resetFrontends() re-registers self
#     at the new IP (GlobalStateMgr.java:1337).
echo 'bdbje_reset_election_group = true' >> starrocks/output/fe/conf/fe.conf
#     ...start the FE once, confirm step 4 passes, then REMOVE that line and restart.
#     Leaving it set permanently disables the cluster-membership check.

# (b) clean bootstrap -- simpler, and cheap here: the image is ~30 KB and holds no TPC-H catalog
#     (the benchmark uses FILES(), not tables). Reversible: it is a move, not a delete.
mv starrocks/output/fe/meta starrocks/output/fe/meta.bak-127001
```
Either way, re-run `SET GLOBAL enable_pipeline_engine/pipeline_dop` (Task 1.6 step 1) afterwards —
route (b) discards FE-persisted globals.

- [ ] **Step 4: Start the FE on gcn-17 and confirm it advertises a real IP**

The wait loop is **bounded and pid-checked**: an unbounded `until mysql ...` loop turns the
`System.exit(-1)` above into an infinite hang with no diagnostic.

```bash
starrocks/output/fe/bin/start_fe.sh --logconsole > /tmp/fe.log 2>&1 &
FE_PID=$!
for _ in $(seq 90); do
    kill -0 "$FE_PID" 2>/dev/null || { echo "FE EXITED -- see /tmp/fe.log:"; tail -30 /tmp/fe.log; break; }
    mysql -h 10.87.140.52 -P 9030 -uroot -e 'SELECT 1' >/dev/null 2>&1 && break
    sleep 2
done
mysql -h 10.87.140.52 -P 9030 -uroot --vertical -e "SHOW PROC '/frontends'" | grep -E 'IP|Alive'
```
Expected: `IP: 10.87.140.52`, `Alive: true`. **`127.0.0.1` means `priority_networks` did not take**
— re-check step 3. `current node is not added to the cluster, will exit` in `/tmp/fe.log` means
step 3b was skipped.

### Task 1.2: Create the shared cross-host environment overlay

Both hosts need an identical env block. Putting it in one sourced file removes the single largest
source of A/B error: a variable set on one host and not the other.

**Files:**
- Create: `experimental/starrocks/configs/gb200-4gpu/engine-a-2host.env`

- [ ] **Step 1: Write the overlay**

```bash
cat > configs/gb200-4gpu/engine-a-2host.env <<'EOF'
# Cross-host overlay for engine A. Source this INSTEAD OF relying on cn-env.sh defaults, and
# source it BEFORE scripts/cn-env.sh -- cn-env.sh only fills these in when unset (cn-env.sh:35),
# so sourcing it first would silently discard the cross-host TLS choice.
#
# NUM_CNS must be set by the caller BEFORE sourcing: it decides the warmup peer expectation.

: "${NUM_CNS:?set NUM_CNS (total CNs across BOTH hosts) before sourcing}"

export TOOLS_DIR=${TOOLS_DIR:-/home/prestouser/aocsa/tools}

# Multi-rail GPUDirect RDMA -- the best path the CN can actually reach today. cuda_ipc is RETAINED
# (the runbook's line drops it, which would cost same-host peers the 85-90 GB/s NVLink path at
# 4 CNs/host). MNNVL/765 GB/s needs a fabric-handle arena the CN does not have.
export UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self

# NOT optional. Left to itself UCX picks the DPU interface (100.127.x), which is not routable
# between these hosts, and every connection stalls until the TCP timeout.
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0

# Installed UCX 1.21.0 defaults this to 2; 4 engages all four rails.
export UCX_MAX_RNDV_RAILS=4

# GATE, not just a size: unset means no arena, the nixl tier is disabled, and every remote
# destination fails. The arena fails HARD on exhaustion -- it never degrades.
export SIRIUS_EXCHANGE_STAGING_BYTES=16GiB

# The warmup loop breaks on established.len() >= expect. Topology-specific: NUM_CNS - 1.
export SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS=$((NUM_CNS - 1))
# Default 180 assumes a single-box launch loop; two hosts started by hand boot further apart.
export SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=300

# PIN IT (G7). Set by no committed config; worth ~20x on scan-bound queries. The recorded SF100
# anchor ran with =false (kvikio).
export SIRIUS_CN_USE_SIRIUS_DATASOURCE=false

export RUST_LOG=sirius_starrocks_cn=info,info
EOF
```

- [ ] **Step 2: Verify it computes the right peer count for Phase 1**

```bash
( NUM_CNS=2 . configs/gb200-4gpu/engine-a-2host.env && echo "peers=$SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS tls=$UCX_TLS" )
```
Expected: `peers=1 tls=cuda_copy,cuda_ipc,rc_mlx5,tcp,self`

- [ ] **Step 3: Verify it refuses without `NUM_CNS`**

```bash
( . configs/gb200-4gpu/engine-a-2host.env ) ; echo "exit=$?"
```
Expected: an error naming `NUM_CNS`, non-zero exit. A silent default here would produce a wrong
peer expectation and a warmup that never completes.

- [ ] **Step 4: Commit**

```bash
git add configs/gb200-4gpu/engine-a-2host.env
# Pathspec-scoped: this branch routinely carries other in-flight staged work, and a bare
# `git commit` would sweep all of it in under this subject line.
git commit -m "feat(bench): cross-host env overlay for two-machine engine A" \
  -- configs/gb200-4gpu/engine-a-2host.env
```

### Task 1.3: Launch CN0 on gcn-17

**Files:** none (runtime)

- [ ] **Step 1: Launch**

```bash
cd ~/aocsa/sirius/experimental/starrocks
export NUM_CNS=2
. configs/gb200-4gpu/engine-a-2host.env
source scripts/cn-env.sh
unset CUDA_VISIBLE_DEVICES              # G5: it would override --gpu-device

numactl --physcpubind=0-71 --membind=0 -- target/release/sirius-starrocks-cn \
    --fe-host           10.87.140.52 \
    --advertise-host    10.87.140.52 \
    --bind-host         0.0.0.0 \
    --gpu-device        0 \
    --heartbeat-port    9100 \
    --thrift-port       9101 \
    --brpc-port         9102 \
    --http-port         9103 \
    --starlet-port      9104 \
    --gpu-memory-limit  140GiB \
    --host-memory-limit 160GiB \
    --engine-dir        .cn0-52 \
    > /tmp/cn-gcn17.log 2>&1 &
```

`--engine-dir` is **host-suffixed on purpose.** It is a relative path resolved against this
checkout, and `/home` is NFS (`master:/home`, see /proc/mounts) — the same physical directory on
both hosts. Plain `.cn0` on both would make the two CNs race on one
`derived-sirius-config.yaml` and share one `log/` + `telemetry/` tree. `/tmp` is host-local, so the
`/tmp/cn-*.log` paths need no suffix.

`--physcpubind=0-71 --membind=0` is GPU 0's socket. `--gpu-memory-limit` is **per-GPU** and does not
scale with CN count. The staging arena is a bare `cudaMalloc` *outside* the RMM pool, so the real
footprint is `140 + 16 GiB + ~779 MiB` of CUDA context against 188,417 MiB usable.

- [ ] **Step 2: Verify the NUMA pin took**

```bash
P=$(pgrep -f 'sirius-starrocks-cn')
awk '{for(i=2;i<=NF;i++) if($i ~ /^(bind|prefer|interleave|default)/){print $i; break}}' \
    /proc/$P/numa_maps | sort | uniq -c
grep -o 'N\(2\|10\|18\|26\)=[0-9]*' /proc/$P/numa_maps | awk -F= '{s+=$2} END{print (s?s:0)" pages on HBM"}'
```
Expected: a single `bind:0` line covering every mapping, and `0 pages on HBM`. Any `default`
mapping means the membind did not take (G6).

> **Do not use `Mems_allowed_list` for this** — it reports *cpuset*-allowed nodes and stays
> `0-2,10,18,26` even when `--membind` is working. MEASURED on gcn-18 2026-08-11: a process under
> `numactl --membind=0,1` showed `Mems_allowed_list: 0-2,10,18,26` while all 4,474 of its mappings
> were `bind:0-1` with zero HBM pages. Earlier revisions of this plan had that check wrong.

- [ ] **Step 3: Verify the transport came up**

```bash
grep -E 'nixl transport ready; staging arena registered' /tmp/cn-gcn17.log
```
Expected: one line. Absent means `SIRIUS_EXCHANGE_STAGING_BYTES` did not reach the process.

### Task 1.4: Launch CN0 on gcn-18

**Files:** none (runtime)

- [ ] **Step 1: Launch — identical except `--advertise-host`**

Ports may stay the same across hosts: FE node identity is `(advertise_host, heartbeat_port)`, so
the differing IP already separates the two nodes.

```bash
cd ~/aocsa/sirius/experimental/starrocks
export NUM_CNS=2
. configs/gb200-4gpu/engine-a-2host.env
source scripts/cn-env.sh
unset CUDA_VISIBLE_DEVICES

numactl --physcpubind=0-71 --membind=0 -- target/release/sirius-starrocks-cn \
    --fe-host           10.87.140.52 \
    --advertise-host    10.87.140.53 \
    --bind-host         0.0.0.0 \
    --gpu-device        0 \
    --heartbeat-port    9100 \
    --thrift-port       9101 \
    --brpc-port         9102 \
    --http-port         9103 \
    --starlet-port      9104 \
    --gpu-memory-limit  140GiB \
    --host-memory-limit 160GiB \
    --engine-dir        .cn0-53 \
    > /tmp/cn-gcn18.log 2>&1 &
```

- [ ] **Step 2: Verify the pin and transport, same as Task 1.3 steps 2-3**

```bash
P=$(pgrep -f 'sirius-starrocks-cn')
awk '{for(i=2;i<=NF;i++) if($i ~ /^(bind|prefer|interleave|default)/){print $i; break}}' \
    /proc/$P/numa_maps | sort | uniq -c
grep -E 'nixl transport ready; staging arena registered' /tmp/cn-gcn18.log
```
Expected: a single `bind:0` line covering every mapping, and one transport-ready line.
(Use `numa_maps`, not `Mems_allowed_list` — see Task 1.3 Step 2.)

### Task 1.5: Confirm registration and the data plane

**Files:** none (verification)

- [ ] **Step 1: Both CNs registered with routable IPs**

Registration is automatic — each CN dials the FE over MySQL and runs `ALTER SYSTEM ADD COMPUTE
NODE`, retrying with exponential backoff up to `--registration-max-attempts` (default 120, ~57 min),
after which the process **exits**. There is no manual `ALTER SYSTEM` step.

```bash
mysql -h 10.87.140.52 -P 9030 -uroot --vertical -e "SHOW COMPUTE NODES" | grep -E 'ComputeNodeId|IP|Alive|HeartbeatPort'
```
Expected: exactly **two** rows, `IP: 10.87.140.52` and `IP: 10.87.140.53`, both `Alive: true`.
**Loopback in that output means a CN did not get `--advertise-host`** — and two CNs both claiming
`127.0.0.1:9100` are the *same node* to the FE, so they silently collide.

- [ ] **Step 2: The bandwidth canary cleared the floor — the gate on trusting any timing**

```bash
grep 'nixl bandwidth canary' /tmp/cn-gcn17.log /tmp/cn-gcn18.log
grep 'below the 2 GB/s floor' /tmp/cn-gcn17.log /tmp/cn-gcn18.log
```
Expected: canary lines reading **tens of GB/s**; the second grep **empty**.

A `below the 2 GB/s floor` refusal means `UCX_TLS`/`UCX_NET_DEVICES` did not take effect in that
CN's environment. The floor is a compile-time `const` (`src/nixl_transport.rs:318`) with **no env
override** — do not look for one. And there is **no fallback tier**: with one CN per host every
exchange is remote, so a refused peer means the cluster cannot run a single distributed query.

- [ ] **Step 3: Warmup found its one peer**

```bash
grep -E 'pre-established a nixl peer session|nixl session warmup complete|peers left cold' \
     /tmp/cn-gcn17.log /tmp/cn-gcn18.log
```
Expected: warmup complete with 1 peer on each host. `peers left cold` is not fatal — a cold peer
still gets a session on first use — but it means the cold-start deadlock window is open.

### Task 1.6: Prove fragments landed on both hosts

**This is what the whole exercise is for.** `SHOW COMPUTE NODES` proves the CNs *registered*; it
does not prove a query *ran* on both. Two independent proofs are required.

**Files:** none (verification)

- [ ] **Step 1: Pin `pipeline_dop` so the arm is reproducible (G8)**

`--physcpubind=0-71` makes each CN report 72 cores, resolving `pipeline_dop` to `min(64, 72/2) = 36`.
Unpinned it would be 64. Pin it explicitly rather than inheriting the derivation.

```bash
mysql -h 10.87.140.52 -P 9030 -uroot -e \
  "SET GLOBAL enable_pipeline_engine = true; SET GLOBAL pipeline_dop = 36;
   SHOW GLOBAL VARIABLES LIKE 'pipeline_dop';"
```
Expected: `pipeline_dop  36`. (`enable_pipeline_engine` persists in FE metadata across runs.)

- [ ] **Step 2: Pre-flight the placement without deploying anything**

`EXPLAIN SCHEDULER` runs full scheduling and deploys zero fragments, so it is safe to run
repeatedly.

```bash
Q='WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet"))
SELECT l_orderkey % 4096 AS bucket, count(*) AS n, sum(l_quantity) AS q
FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 20'

mysql -h 10.87.140.52 -P 9030 -uroot -e "EXPLAIN SCHEDULER $Q;" \
  | grep -E 'PLAN FRAGMENT|INSTANCE\(|BE: '
```
Expected: **two distinct numeric ids** after `BE:` on the scan fragment. Map them through the
`ComputeNodeId` column of `SHOW COMPUTE NODES`. Two ids = the FE intends to use both machines.
One id = the scan did not split; re-check Task 0.2 step 4.

- [ ] **Step 3: Run the validation query**

`new_planner_agg_stage = 2` (TWO_STAGE) forces a partial-agg → `HASH_PARTITIONED` exchange →
merge-agg instead of letting the optimizer collapse to one stage. `l_orderkey % 4096` is
high-cardinality on purpose so the shuffle carries real bytes.

```sql
-- mysql -h 10.87.140.52 -P 9030 -uroot
SET enable_profile = true;        -- NOT `EXPLAIN ANALYZE`: it forces enable_async_profile=false
                                  -- and blocks for the full 10 s profile_timeout on engine A
SET new_planner_agg_stage = 2;

WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet"))
SELECT l_orderkey % 4096 AS bucket, count(*) AS n, sum(l_quantity) AS q
FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 20;

SELECT last_query_id();
```
Expected: 20 rows, and a query id.

- [ ] **Step 4: Proof 1 — FE placement says two hosts**

```bash
QID=<the id from last_query_id()>
mysql -h 10.87.140.52 -P 9030 -uroot --raw -N \
  -e "SELECT get_query_profile('$QID')" > /tmp/profile.txt
grep -n ' - BackendAddresses:' /tmp/profile.txt
grep -nE ' - (BackendNum|InstanceNum|MissingInstanceIds):' /tmp/profile.txt
```
Expected:
```
   - BackendAddresses: 10.87.140.52:9101,10.87.140.53:9101
   - BackendNum: 2
   - MissingInstanceIds: 019fe1c9-...-c971,019fe1c9-...-c972      <- expected on engine A
```
**Two distinct IPs is the proof.** The port is the CN's *thrift* port (`--thrift-port`, 9101), not
brpc — `ComputeNode.getAddress()` returns `(host, bePort)`.

Engine A produces no BE-side profile: the CN never calls `reportExecStatus` and hardcodes
`query_statistics: None`. So every operator reads `TotalTime: 0ns` / `OutputRows: ?` and every
instance appears under `MissingInstanceIds`. **That is expected and does not invalidate
`BackendAddresses`**, which the FE writes from its own assignment. `ANALYZE PROFILE` prints
`BackendNum` but never `BackendAddresses`, so it cannot answer "which two hosts".

- [ ] **Step 5: Proof 2 — the NIXL log says bytes crossed the machine boundary**

```bash
grep 'transmitted batches via nixl'                     /tmp/cn-gcn17.log   # sender
grep 'received remote batches'                          /tmp/cn-gcn18.log   # receiver
grep 'relayed native batches across a fragment boundary' /tmp/cn-gcn1*.log  # NOT a hop
```
Expected: a line on gcn-17 carrying `dest=10.87.140.53:9102` with non-zero `batches`/`bytes`,
matched by a `received remote batches` line on gcn-18 — **and the reverse pair too**, since a
hash shuffle is bidirectional.

`transmitted batches via nixl` is emitted **only** under `DestinationRoute::Remote`, so its presence
is by construction proof that work crossed the machine boundary. If all you see is `relayed native
batches across a fragment boundary`, that is the same-process short circuit and **nothing crossed**.

There is no receiver-side byte counter in CN tracing — read received bytes off the peer's sender
line.

- [ ] **Step 6: Proof 3 — engine-agnostic corroboration**

```bash
for d in mlx5_0 mlx5_1 mlx5_4 mlx5_5; do
  echo -n "$d "; cat /sys/class/infiniband/$d/ports/1/counters/{port_xmit_data,port_rcv_data} \
    | tr '\n' ' '; echo
done
# ...re-run the query, then repeat and diff
```
Expected: the counters climb by an amount within an order of magnitude of the logged `bytes`.
Near-zero on all four rails while the CN log claims a remote transfer means traffic fell back to
TCP on `bond0` — re-check `UCX_NET_DEVICES`.

- [ ] **Step 7: Record the result**

Write `/raid/prestouser/bench-2node-A-phase1/INVOCATION-engineA.txt` containing the literal shell
from Tasks 1.3 and 1.4 including every `export`, plus the canary GB/s and the `BackendAddresses`
line. This is the Phase 1 acceptance record.

### Task 1.7: TPC-H sweep at the Phase 1 shape

**Files:** none (measurement)

- [ ] **Step 1: Run the sweep on gcn-17**

`bench.sh` hardcodes `--host 127.0.0.1` and there is no `FE_HOST` variable — so run it **on
gcn-17, where the FE is**, and `127.0.0.1` is literally correct.

```bash
cd ~/aocsa/sirius/experimental/starrocks
export PATH=$PWD/.pixi/envs/default/bin:$PATH     # bench.sh calls bare `mysql`

TPCH_DATA=/raid/prestouser/aocsa/tpch_parquet_sf100 \
FE_PORT=9030 \
QUERY_TIMEOUT=180 \
COLD_TIMEOUT=600 \
MIN_BACKENDS=2 \
  ./benchmarks/tpch/bench.sh --cold \
     /raid/prestouser/bench-2node-A-phase1/timings.csv 3
```

Notes, each load-bearing:
- `MIN_BACKENDS=2` — `bench.sh` **aborts if MORE than `MIN_BACKENDS` are alive** unless
  `ALLOW_EXTRA_BACKENDS=1`. Two CNs → 2.
- **`runs` is positional and mandatory before a query subset.** `bench.sh out.csv q05` sets
  `RUNS=q05`. Always write `bench.sh out.csv 3 q05`.
- `QUERY_TIMEOUT=180 / COLD_TIMEOUT=600` are the SF100-scaled values; the defaults are SF1 numbers.

- [ ] **Step 2: Patch q11 for SF100, then revert**

`queries/q11.sql:26` hardcodes the `FRACTION` literal `0.0001000000`, correct only at SF1. At SF100
q11 returns 0 rows and is recorded as `wedge`.

```bash
sed -i 's/0\.0001000000/0.000001000000/' benchmarks/tpch/queries/q11.sql   # 0.0001 / 100
# ...run the sweep...
git checkout -- benchmarks/tpch/queries/q11.sql
```

- [ ] **Step 3: Confirm the sweep is honest**

```bash
awk -F, 'NR>1 && $3=="warm" {n[$4]++} END {for (s in n) print s, n[s]}' \
    /raid/prestouser/bench-2node-A-phase1/timings.csv
```
Expected: a `pass` count of 22 × 3. Any `wedge`/`fail` rows must be reproduced solo and
characterised before the CSV is quoted anywhere.

**Phase 1 exit criteria — all four must hold:**
1. `SHOW COMPUTE NODES` shows two rows with routable IPs, both alive.
2. The canary reads tens of GB/s on the one peer pair, with no floor refusal.
3. `BackendAddresses` spans `10.87.140.52` and `10.87.140.53` for one query.
4. Matched `transmitted batches via nixl` / `received remote batches` pairs exist in **both**
   directions.

---

# Phase 2 — Eight CNs, four per box

Scale out from a known-good 2-CN baseline. **Do not start Phase 2 until Phase 1's exit criteria all
hold** — otherwise a failure at 8 CNs has 28 candidate peer pairs to blame instead of 1.

```
gcn-17   CN0 GPU0 node0   CN1 GPU1 node0   CN2 GPU2 node1   CN3 GPU3 node1
gcn-18   CN0 GPU0 node0   CN1 GPU1 node0   CN2 GPU2 node1   CN3 GPU3 node1
ports    9100-9104        9110-9114        9120-9124        9130-9134
```

**What is genuinely new at this shape** — these are the things Phase 1 cannot have caught:

| New surface | Consequence |
|---|---|
| Same-host peers exist | `cuda_ipc` (85-90 GB/s NVLink) and cross-host RDMA now coexist in one cluster. Dropping `cuda_ipc` from `UCX_TLS` would cost 12 of the 56 directed pairs their fast path — which is exactly the runbook's error (GAPS §A row 3). |
| 56 directed peer pairs, not 2 | Warmup expectation becomes 7, and every CN's arena holds `cudaIpcOpenMemHandle` page tables for 3 same-host peers. |
| Two CNs share a socket | `CN_CPUS` must split each socket, or two CNs contend for cpus 0-71. |
| 4 CNs contend for one host's LPDDR | `--host-memory-limit` is a per-CN total; 4 × 160 GiB against ~956 GiB of CPU-addressable LPDDR. |

### Task 2.1: Add two-host support to the launcher

The three committed launchers (`benchmarks/cluster8.sh`, `benchmarks/cluster8-numa.sh`,
`configs/gb200-4gpu/cluster4-numa.sh`) are all single-host: none passes `--advertise-host` or
`--fe-host`. Launching 8 CNs by hand is where transcription errors live, so this is worth a script.

**Files:**
- Create: `experimental/starrocks/benchmarks/cn-2host.sh`

**Interfaces:**
- Consumes: `configs/gb200-4gpu/engine-a-2host.env` (Task 1.2), `scripts/cn-env.sh`
- Produces: `cn-2host.sh <advertise-host> <fe-host> [--no-fe]` launching `NUM_CNS_PER_HOST` CNs

- [ ] **Step 1: Write the launcher**

```bash
cat > benchmarks/cn-2host.sh <<'SCRIPT'
#!/usr/bin/env bash
# Launch this host's share of a two-machine engine A cluster.
#
#   ./benchmarks/cn-2host.sh 10.87.140.52 10.87.140.52          # gcn-17, also starts the FE
#   ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe  # gcn-18, CNs only
#
# Refuses to start rather than starting degraded: a half-configured cluster still answers
# queries, and the benchmark silently measures it.
#
# --engine-dir is relative and resolved against this checkout, which lives on NFS (`master:/home`,
# see /proc/mounts) and is therefore THE SAME DIRECTORY on both hosts. So the per-CN engine dir is
# suffixed with the advertise host's last octet (.cn0-52 / .cn0-53): unsuffixed, both hosts' CN0
# would race on the same derived-sirius-config.yaml and the same log/ + telemetry/ trees.
set -euo pipefail

ADVERTISE=${1:?usage: cn-2host.sh <advertise-host> <fe-host> [--no-fe]}
FE_HOST=${2:?usage: cn-2host.sh <advertise-host> <fe-host> [--no-fe]}
START_FE=1; [ "${3:-}" = "--no-fe" ] && START_FE=0

SR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SR_DIR"

PER_HOST=${NUM_CNS_PER_HOST:-4}
export NUM_CNS=${NUM_CNS:-$((PER_HOST * 2))}       # total across BOTH hosts -- drives warmup
GPU_MEM=${GPU_MEM:-140GiB}
HOST_MEM=${HOST_MEM:-160GiB}

# Index-aligned with GPU ordinal. ONLY 0 AND 1 ARE EVER VALID for --membind: nodes 2/10/18/26
# are GPU HBM with zero CPUs, and binding host pages there eats the HBM of a GPU a CN is using.
read -r -a NODES <<< "${CN_NODE:-0 0 1 1}"
read -r -a CPUS  <<< "${CN_CPUS:-0-35 36-71 72-107 108-143}"

[ "${#NODES[@]}" -ge "$PER_HOST" ] || { echo "CN_NODE needs $PER_HOST entries" >&2; exit 1; }
[ "${#CPUS[@]}"  -ge "$PER_HOST" ] || { echo "CN_CPUS needs $PER_HOST entries" >&2; exit 1; }

for i in $(seq 0 $((PER_HOST - 1))); do
    case "${NODES[$i]}" in 0|1) ;; *)
        echo "CN$i: --membind ${NODES[$i]} is not a CPU-bearing node (HBM interlock)" >&2
        exit 1 ;;
    esac
done

. configs/gb200-4gpu/engine-a-2host.env            # UCX_*, staging, warmup, datasource pin
source scripts/cn-env.sh                            # LD_LIBRARY_PATH, nixl plugins

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "cn-2host: unsetting inherited CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" \
         "(it would override --gpu-device and collapse all CNs onto one GPU)" >&2
    unset CUDA_VISIBLE_DEVICES
fi

# --- preflight ---------------------------------------------------------------------------------
# The header promises "refuses to start rather than starting degraded". That is only true if we
# actually check; every failure below otherwise produces a HALF cluster, which still answers
# queries and which the benchmark silently measures.
CN_BIN=target/release/sirius-starrocks-cn
FE_BIN=starrocks/output/fe/bin/start_fe.sh
[ -x "$CN_BIN" ] || { echo "cn-2host: no CN binary at $SR_DIR/$CN_BIN" >&2; exit 1; }
if [ "$START_FE" = 1 ]; then
    [ -x "$FE_BIN" ] || { echo "cn-2host: no packaged FE at $SR_DIR/$FE_BIN" >&2; exit 1; }
fi
command -v numactl >/dev/null 2>&1 || { echo "cn-2host: numactl not found" >&2; exit 1; }

# FE node identity is (advertise_host, heartbeat_port) and the nixl agent name is
# {advertise_host}:{brpc_port}, so an overlapping port block is an IDENTITY collision that corrupts
# both registries -- not a clean bind failure. Read /proc/net/tcp{,6} directly (st 0A == TCP_LISTEN):
# no iproute2 dependency, and it sees listeners owned by other users too. This is a pure read.
declare -A BOUND=()
for f in /proc/net/tcp /proc/net/tcp6; do
    [ -r "$f" ] || continue
    while read -r _sl laddr _rem st _rest; do
        [ "$st" = "0A" ] || continue
        hex=${laddr##*:}
        [[ $hex =~ ^[0-9A-Fa-f]+$ ]] || continue
        BOUND[$((16#$hex))]=1
    done < <(tail -n +2 "$f")
done

want=()
if [ "$START_FE" = 1 ]; then want+=(6090 8030 9010 9020 9030); fi   # 6090 = shared_data StarMgr
for i in $(seq 0 $((PER_HOST - 1))); do
    base=$((9100 + i * 10))
    for off in 0 1 2 3 4; do want+=("$((base + off))"); done
done
busy=()
for p in "${want[@]}"; do
    if [ -n "${BOUND[$p]:-}" ]; then busy+=("$p"); fi
done
if [ "${#busy[@]}" -gt 0 ]; then
    echo "cn-2host: required ports already bound: ${busy[*]}" >&2
    echo "  A cluster is very likely already running (G1). Shut it down first -- do NOT launch a" >&2
    echo "  second one on top of it." >&2
    exit 1
fi

# The CN's own ensure_gpu_unclaimed preflight is SKIPPED whenever --gpu-memory-limit is set, which
# this script always does -- so it cannot protect us. The RMM pool is reserved in full at startup,
# so a second CN on a claimed GPU is an allocation failure or a zero-headroom cluster, never just
# a slowdown.
claimed=()
for i in $(seq 0 $((PER_HOST - 1))); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null |
            tr -d ' ' | tr '\n' ',' | sed 's/,$//') || procs="<nvidia-smi query failed>"
    if [ -n "$procs" ]; then claimed+=("gpu$i(pids: $procs)"); fi
done
if [ "${#claimed[@]}" -gt 0 ] && [ "${ALLOW_SHARED_GPUS:-0}" != 1 ]; then
    echo "cn-2host: these GPUs already have compute processes: ${claimed[*]}" >&2
    echo "  Set ALLOW_SHARED_GPUS=1 to override." >&2
    exit 1
fi

# --- launch ------------------------------------------------------------------------------------
# The trap is armed BEFORE the first fork: an interrupt during the launch window would otherwise
# orphan the CNs already started, leaving them holding GPUs, ports 9100-9134 and FE registry
# entries -- exactly the half-cluster the preflight above exists to prevent.
pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if [ "$START_FE" = 1 ]; then
    # Membound to every CPU-bearing node, DERIVED from the hardware rather than hardcoded, so the
    # ~10-20 GiB JVM cannot allocate into GPU HBM -- the one exposure the CN membind exists to
    # close. Deliberately NOT cpubound: the FE's cross-socket float is what absorbs error once all
    # CNs are hard-pinned (cluster4-numa.sh uses PIN_FE=1 to opt out of that float).
    FE_NODES=$(numactl --hardware |
        awk '/^node [0-9]+ cpus:/ && NF > 3 { n = (n == "" ? $2 : n "," $2) } END { print n }')
    [ -n "$FE_NODES" ] ||
        { echo "cn-2host: no NUMA node reports CPUs -- refusing to membind the FE" >&2; exit 1; }
    numactl --membind="$FE_NODES" -- "$FE_BIN" --logconsole > /tmp/fe.log 2>&1 &
    pids+=($!)
    echo "FE started (membind=$FE_NODES, no cpubind) -> /tmp/fe.log"
fi

for i in $(seq 0 $((PER_HOST - 1))); do
    base=$((9100 + i * 10))
    numactl --physcpubind="${CPUS[$i]}" --membind="${NODES[$i]}" -- \
        "$CN_BIN" \
            --fe-host           "$FE_HOST" \
            --advertise-host    "$ADVERTISE" \
            --bind-host         0.0.0.0 \
            --gpu-device        "$i" \
            --heartbeat-port    "$base" \
            --thrift-port       "$((base + 1))" \
            --brpc-port         "$((base + 2))" \
            --http-port         "$((base + 3))" \
            --starlet-port      "$((base + 4))" \
            --gpu-memory-limit  "$GPU_MEM" \
            --host-memory-limit "$HOST_MEM" \
            --engine-dir        ".cn$i-${ADVERTISE##*.}" \
            > "/tmp/cn-${ADVERTISE##*.}-$i.log" 2>&1 &
    pids+=($!)
    echo "CN$i gpu=$i node=${NODES[$i]} cpus=${CPUS[$i]} ports=$base-$((base+4))" \
         "engine-dir=.cn$i-${ADVERTISE##*.} -> /tmp/cn-${ADVERTISE##*.}-$i.log"
done

wait -n "${pids[@]}"
SCRIPT
chmod +x benchmarks/cn-2host.sh
```

- [ ] **Step 2: Verify it refuses an HBM membind**

```bash
CN_NODE="0 0 2 1" ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe; echo "exit=$?"
```
Expected: `CN2: --membind 2 is not a CPU-bearing node (HBM interlock)`, non-zero exit, **no CN
processes started**. This is the single most damaging misconfiguration the script exists to prevent.

- [ ] **Step 3: Verify it refuses a short `CN_CPUS`**

```bash
CN_CPUS="0-35 36-71" ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe; echo "exit=$?"
```
Expected: `CN_CPUS needs 4 entries`, non-zero exit.

- [ ] **Step 4: Verify the derived warmup peer count is 7**

```bash
( NUM_CNS_PER_HOST=4 NUM_CNS=8 . configs/gb200-4gpu/engine-a-2host.env \
  && echo "peers=$SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS" )
```
Expected: `peers=7`

- [ ] **Step 4b: Verify the preflight guards without launching a cluster**

The binary / port / GPU-claim guards sit *after* the env sourcing, so they cannot be reached by a
guard-failure like steps 2-3. Exercise them against a scratch copy whose
`target/release/sirius-starrocks-cn` is a symlink to `/bin/true` — then even a totally broken guard
can only fork `/bin/true`.

```bash
T=$(mktemp -d)/sr; mkdir -p "$T/benchmarks" "$T/target/release" "$T/starrocks/output/fe/bin"
cp benchmarks/cn-2host.sh "$T/benchmarks/"; ln -s "$PWD/configs" "$T/configs"; ln -s "$PWD/scripts" "$T/scripts"
ln -sfn "$PWD/../../.pixi" "$T/../../.pixi"    # cn-env.sh resolves REPO_ROOT as $SR_DIR/../..

( cd "$T" && ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe )   # -> no CN binary
ln -sf /bin/true "$T/target/release/sirius-starrocks-cn"
( cd "$T" && ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 )           # -> no packaged FE
python3 -c 'import socket,time; s=socket.socket(); s.bind(("127.0.0.1",9100)); s.listen(1); time.sleep(20)' &
sleep 2; ( cd "$T" && ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe )  # -> port bound
```
Expected, in order: `cn-2host: no CN binary at .../target/release/sirius-starrocks-cn`,
`cn-2host: no packaged FE at .../start_fe.sh`, `cn-2host: required ports already bound: 9100` —
each with a non-zero exit and **no process launched**.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cn-2host.sh
git commit -m "feat(bench): two-host CN launcher with HBM-membind interlock" \
  -- benchmarks/cn-2host.sh
```

### Task 2.2: Bring up all eight CNs

**Files:** none (runtime)

- [ ] **Step 1: Confirm the box is clear (G1, G2)**

```bash
for h in presto-gb200-gcn-17 presto-gb200-gcn-18; do
  ssh $h "pgrep -af 'sirius-starrocks-cn|starrocks_be|StarRocksFE' || echo '$h clear'"; done
date -u '+%H:%M UTC'
```
Expected: both clear, and the time is outside 02:00-03:50 UTC.

- [ ] **Step 2: Launch gcn-17 (FE + 4 CNs)**

```bash
# on gcn-17
cd ~/aocsa/sirius/experimental/starrocks
NUM_CNS_PER_HOST=4 ./benchmarks/cn-2host.sh 10.87.140.52 10.87.140.52
```

- [ ] **Step 3: Launch gcn-18 (4 CNs, no FE)**

```bash
# on gcn-18
cd ~/aocsa/sirius/experimental/starrocks
NUM_CNS_PER_HOST=4 ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe
```

- [ ] **Step 4: Verify all eight registered, four per host**

```bash
mysql -h 10.87.140.52 -P 9030 -uroot --vertical -e "SHOW COMPUTE NODES" \
  | grep -E '^\s+IP:' | sort | uniq -c
```
Expected:
```
      4        IP: 10.87.140.52
      4        IP: 10.87.140.53
```
Any count other than 4/4 means a heartbeat-port collision or a CN that exited after exhausting
`--registration-max-attempts`.

- [ ] **Step 5: Verify every NUMA pin took**

```bash
for p in $(pgrep -f sirius-starrocks-cn); do
  printf '%s %s\n' "$p" "$(awk '{for(i=2;i<=NF;i++) if($i ~ /^(bind|default)/){print $i; break}}' \
      /proc/$p/numa_maps | sort -u | tr '\n' ' ')"; done
```
Expected: four processes per host, each printing exactly one policy — `bind:0` (CN0/CN1) or
`bind:1` (CN2/CN3). Any `default` means that CN's membind did not take.

### Task 2.3: Verify both transport paths coexist

This is the check Phase 1 structurally could not perform.

**Files:** none (verification)

- [ ] **Step 1: Every peer pair cleared the floor**

```bash
grep -h 'nixl bandwidth canary' /tmp/cn-5*.log | sort
grep -l 'below the 2 GB/s floor' /tmp/cn-5*.log
```
Expected: canary lines for **7 peers per CN**; the second grep prints nothing.

- [ ] **Step 2: Same-host pairs took `cuda_ipc`, cross-host pairs took RDMA**

Same-host peers are the three CNs sharing an `advertise_host`; cross-host peers are the four with
the other IP.

```bash
grep 'nixl bandwidth canary' /tmp/cn-52-0.log
```
Expected, from CN0 on gcn-17: three peers at `10.87.140.52:91x2` reading **85-90 GB/s**
(`cuda_ipc` over NVLink) and four peers at `10.87.140.53:91x2` reading **tens of GB/s** (RDMA).

**If the same-host peers also read tens of GB/s rather than ~85-90, `cuda_ipc` was dropped from
`UCX_TLS`** — that is the runbook's documented error, and it costs 12 of the 56 directed pairs their
fast path without failing anything.

- [ ] **Step 3: Warmup found seven peers**

```bash
grep -E 'nixl session warmup complete|peers left cold' /tmp/cn-5*.log
```
Expected: warmup complete with 7 peers on each of the eight CNs.

### Task 2.4: Prove distribution at eight CNs

**Files:** none (verification)

- [ ] **Step 1: Re-pin `pipeline_dop` for the new CPU split**

`CN_CPUS="0-35 36-71 72-107 108-143"` makes each CN report 36 cores, resolving `pipeline_dop` to
`min(64, 36/2) = 18`. That is a different value from Phase 1's 36 — **the two phases are therefore
not directly comparable unless you pin both to the same number.**

```bash
mysql -h 10.87.140.52 -P 9030 -uroot -e \
  "SET GLOBAL pipeline_dop = 18; SHOW GLOBAL VARIABLES LIKE 'pipeline_dop';"
```
Expected: `pipeline_dop  18`. If you intend a Phase-1-vs-Phase-2 comparison, pin **both** to the
same value and record which, or the comparison has a free variable.

- [ ] **Step 2: Run the same validation query as Task 1.6 step 3, at SF100**

```sql
SET enable_profile = true;
SET new_planner_agg_stage = 2;
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet"))
SELECT l_orderkey % 4096 AS bucket, count(*) AS n, sum(l_quantity) AS q
FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 20;
SELECT last_query_id();
```
SF100 `lineitem` is 17,187,602,838 B → 256 instances uncapped, capped at `8 nodes ×
parallelInstanceNum`. This is large enough that every CN should receive ranges.

- [ ] **Step 3: Confirm all eight hosts appear**

```bash
QID=<last_query_id()>
mysql -h 10.87.140.52 -P 9030 -uroot --raw -N \
  -e "SELECT get_query_profile('$QID')" > /tmp/profile8.txt
grep -o '10\.87\.140\.5[23]:[0-9]*' /tmp/profile8.txt | sort -u
```
Expected: **eight** distinct `host:thrift_port` entries — four on `.52`, four on `.53`.

**`BackendNum: 8` alone is not proof of two machines here** — unlike Phase 1, it could in principle
be eight co-located CNs. Only the IPs prove it.

- [ ] **Step 4: Confirm cross-host transfers, not just same-host**

```bash
grep -h 'transmitted batches via nixl' /tmp/cn-52-*.log \
  | grep -o 'dest=10\.87\.140\.5[23]:[0-9]*' | sort | uniq -c
```
Expected: `dest=10.87.140.53:*` lines present in meaningful volume from gcn-17's CNs. **If every
`dest` is `10.87.140.52`, the plan was scheduled entirely within one host** and the cross-host path
was never exercised despite eight registered CNs.

- [ ] **Step 5: Scan fan-out caveat**

A scan fragment gets one instance per worker that *actually received ranges*. At SF100 `lineitem` is
6 parquet files and `orders` 2 — so small tables legitimately land on a subset of CNs, and `nation`
(a single file) lands on exactly one. Do not read that as a failure; check `lineitem` fan-out.

### Task 2.5: TPC-H sweep at eight CNs, and compare

**Files:** none (measurement)

- [ ] **Step 1: Sweep**

```bash
# on gcn-17
cd ~/aocsa/sirius/experimental/starrocks
export PATH=$PWD/.pixi/envs/default/bin:$PATH

TPCH_DATA=/raid/prestouser/aocsa/tpch_parquet_sf100 \
FE_PORT=9030 QUERY_TIMEOUT=180 COLD_TIMEOUT=600 MIN_BACKENDS=8 \
  ./benchmarks/tpch/bench.sh --cold \
     /raid/prestouser/bench-2node-A-phase2/timings.csv 3
```
`MIN_BACKENDS=8` — the harness aborts if more than that many are alive.

- [ ] **Step 2: Record provenance (G9)**

```bash
cat > /raid/prestouser/bench-2node-A-phase2/INVOCATION-engineA.txt <<'EOF'
<the literal shell from Task 2.2 steps 2-3, including every export,
 plus the pipeline_dop value from Task 2.4 step 1 and the canary GB/s
 for one same-host pair and one cross-host pair>
EOF
```

- [ ] **Step 3: Compare Phase 1 against Phase 2**

```bash
python3 benchmarks/tpch/analyze.py \
  /raid/prestouser/bench-2node-A-phase1/timings.csv \
  /raid/prestouser/bench-2node-A-phase2/timings.csv \
  /raid/prestouser/bench-2node/phase1-vs-phase2.md \
  /raid/prestouser/bench-2node/phase1-vs-phase2.png
```
`analyze.py` medians `phase=warm,status=pass` rows and geomeans the ratio. It **exits 1 if row
counts disagree on any query** — that is a shape check, not a correctness check. Neither harness
compares answer *values*; a real correctness claim needs an out-of-band DuckDB oracle diff.

**Expect losses as well as wins.** Join-heavy queries regress when work is spread wider —
SF200 measured q12 0.81×, q03 0.86×, q10 0.87×; partitioned operators pin partitions per GPU, so
coordination and cross-GPU movement eat the gain. A sweep that shows uniform improvement is more
likely a measurement error than a result.

**Phase 2 exit criteria:**
1. `SHOW COMPUTE NODES` shows 4 rows per host, all alive.
2. Every CN's canary shows 7 peers, none refused, with same-host pairs visibly faster
   (~85-90 GB/s) than cross-host.
3. One query's profile spans eight distinct `host:port` entries across both IPs.
4. `transmitted batches via nixl` shows cross-host `dest=` in volume.

---

# Phase 3 — Engine B reference

The stock-StarRocks CPU baseline, so the Phase 1 and Phase 2 numbers have something to be measured
against. Full procedure: [2NODE-ENGINE-B-TUTORIAL.md](2NODE-ENGINE-B-TUTORIAL.md).

- [ ] **Step 1: Stop engine A completely on both hosts (G1)**

```bash
for h in presto-gb200-gcn-17 presto-gb200-gcn-18; do
  ssh $h "pkill -f sirius-starrocks-cn; pkill -f StarRocksFE; sleep 5;
          pgrep -af 'sirius-starrocks-cn|StarRocksFE' || echo '$h clear'"; done
```
Expected: both clear. Both engines bind `9030`; they cannot coexist.

- [ ] **Step 2: Follow the engine-B tutorial §2 through §6**

Shape: 1 FE + **1 unpinned BE per host**, `run_mode` unset (shared-nothing), stock aarch64 3.5.20
binaries already at `/home/prestouser/starrocks-bench` (`$HOME` is NFS, so the tree is already on
both hosts).

- [ ] **Step 3: Validate distribution the engine-B way**

Engine B *does* produce a full BE profile, so it has proofs engine A lacks:

```bash
grep -n ' - BackendAddresses:' /tmp/profile.txt
awk '/EXCHANGE_SOURCE \(plan_node_id=/{p=1}
     p && / - (BytesReceived|BytesPassThrough):/{print}
     /^[[:space:]]*$/{p=0}' /tmp/profile.txt
```
Expected: two distinct IPs, and `BytesReceived > BytesPassThrough`. **`BytesPassThrough` is the real
cross-host proof** — a channel short-circuits to pass-through only on same-IP *and* same-brpc-port,
so the difference is exactly the wire traffic. `BytesPassThrough == BytesReceived` means the shuffle
never crossed.

- [ ] **Step 4: Sweep and compare all three arms**

```bash
python3 benchmarks/tpch/analyze.py \
  /raid/prestouser/bench-2node-A-phase2/timings.csv \
  /raid/prestouser/bench-2node-B/timings.csv \
  /raid/prestouser/bench-2node/results.md \
  /raid/prestouser/bench-2node/tpch_a_vs_b.png
```

**State the asymmetry in any writeup.** Engine B rides `bond0` TCP at 400 Gb/s and has no way to
reach the RoCE fabric — brpc has no interface-selection knob, and `priority_networks` also governs
heartbeats and FE→BE dispatch. Engine A's NIXL path uses hardware engine B structurally cannot.
That is a real difference in what the two engines exploit, not a misconfiguration, and burying it
would make the comparison dishonest.

---

## Open risks

| # | Risk | Mitigation / status |
|---|---|---|
| 1 | Every gcn-17 fact is still inference — `ssh` is blocked for the agent, so gcn-18 is the only host observed directly | Run `benchmarks/collect-host-facts.sh` on gcn-17 (Task 0.1); output lands in the NFS-shared repo and is readable from either host. **Do not skip.** The one value with no fallback is the IMEX `ClusterUUID`. |
| 1a | ~~Datasets exist on gcn-18 only and need replicating~~ | **Resolved — the premise was wrong.** SF100 and SF500 are already at `/raid/prestouser/aocsa/tpch_parquet_sf{100,500}` on both hosts. Task 0.2 is now a verification, not a copy. |
| 1b | No SF1 on either host, so Phase 1 cannot use the minimal 2-split test | Phase 1 uses SF100 `lineitem` (17.2 GB, 6 files) instead. Strictly easier to split, so the test is weaker as a *minimality* argument but stronger as a *coverage* one. |
| 2 | The corrected `UCX_TLS` (with `cuda_ipc` retained) has never been measured cross-host | Task 0.3 step 2 measures it and replaces the runbook's unsourced 97 GB/s |
| 3 | `SIRIUS_EXCHANGE_STAGING_BYTES=16GiB` is validated at 4 CNs/host, not 2 — fewer CNs means each carries more fan-out | Watch for "exchange staging arena exhausted" in Phase 1; the footprint-neutral rebalance is `128GiB + 28GiB` (derived, never measured at 2 CNs) |
| 4 | `--host-memory-limit 160GiB` × 4 CNs against ~956 GiB LPDDR is tight once the FE is co-resident on gcn-17 | Phase 2 only; watch `be.INFO`-equivalent CN logs for host-memory pressure |
| 5 | `pipeline_dop` differs between Phase 1 (36) and Phase 2 (18) by construction | Task 2.4 step 1 — pin both arms to one value before comparing |
| 6 | No CN-side receiver byte counter exists | Read received bytes off the peer's sender line; finer per-frame logging is DEBUG-only |
| 7 | MNNVL (765 GB/s) is unreachable from the CN | Open work: `exchange_staging_arena` needs a `cuMemCreate` + `CU_MEM_HANDLE_TYPE_FABRIC` path; `two_node_harness::cuda_vmm` is a working reference |
