# Handoff: SF10000 8-CN (gcn-09 + gcn-18)

For a **new Codex / Cursor session** that continues **SF10000 only**. Knob log:
[`bench/gb200-8gpu/sf10000/TUNING-DISCOVERY.md`](../bench/gb200-8gpu/sf10000/TUNING-DISCOVERY.md).
Do not re-time queries that already have passing runs unless the topology changed (more hosts).

2026-08-28. Repo `/home/prestouser/aocsa/sirius`, branch `dev`. Dataset
`/scratch/sirius/datasets/tpch_sf10000` (3.7 T, GPFS, both hosts).

---

## Result: 14/22

| Closed (times) | Knobs | Empty window |
|---|---|---|
| q01, q02, q04, q06, q11–q16, q19, q20, q22 | 112/64/16, **dop=9** (q01/q02 also at dop=12) | |
| q07 | **96/80**, dop=9 | |
| | | q03 (arena fills 64 and 80 GiB) |
| | | q05, q08, q17 (pool `bad_alloc`; q08 even at dop=6; q05 still at 144/32) |
| | | q10 (pool OOM at 112/64; arena fills 48 and 56 GiB, leases grow) |
| | | q18 (64 GiB arena full; 96/80 pool OOM) |
| | | q09, q21 (**not re-run** at SF10000; empty at SF3000 — do not raise STAGING) |

Warm bests, dirs, and arm-by-arm lease lines: discovery “Result: 14/22” table.

Caveats: q11 **9365098 rows** — FRACTION `0.000000010000` too loose, not correctness. q15:
two passing times only (mysql empty `rc=0` ~50s). q02: three warms; r3 was a 600s watchdog
before 1800 was forwarded.

Physical HBM **189471 MiB ≈ 185 GiB**. Occupancy 112+64+0.76 = 176.8 / 185 ≈ 96 %.

---

## Cluster

| | |
|---|---|
| FE | gcn-18 `10.87.140.53` |
| CNs | 4 on 18 + 4 on gcn-09 `10.87.140.44` = 8 |
| Fabric | `SIRIUS_EXCHANGE_STAGING_ARENA=fabric` |
| `UCX_TLS` | `cuda_copy,cuda_ipc,tcp,self` — no `rc_mlx5` |
| Scan | uring (`SIRIUS_CN_USE_SIRIUS_DATASOURCE=true`) |
| CI | GPUs ~02:00–03:50 UTC |
| Last seen | gcn-18 stopped ~11:10 UTC 2026-08-28. **Confirm gcn-09** (`stop-cn-2host.sh`); Cursor often cannot SSH. |

**Do not** use `cluster4-numa.sh`, `run-abc.sh`, `benchmarks/cluster8.sh`.

```bash
# [18] bring up 8 CN — always pass memory + DOP (shell exports override env.sh)
cd /home/prestouser/aocsa/sirius/experimental/starrocks
SCALE_FACTOR=10000 GPU_MEM=112GiB STAGING=64GiB SIRIUS_EXCHANGE_STAGING_BYTES=64GiB \
  PIPELINE_DOP=9 ./configs/gb200-8gpu/relaunch.sh

# sweep (cluster up). From repo root on 18.
GPU_MEM=112GiB STAGING=64GiB PIPELINE_DOP=9 \
  OUT=/scratch/prestouser/aocsa/bench-results/sf10000-8gpu-<tag>-$(date -u +%Y%m%dT%H%M%SZ) \
  ./bench/gb200-8gpu/sweep.sh 10000 q22

# q07 only
SCALE_FACTOR=10000 GPU_MEM=96GiB STAGING=80GiB SIRIUS_EXCHANGE_STAGING_BYTES=80GiB \
  PIPELINE_DOP=9 ./configs/gb200-8gpu/relaunch.sh
GPU_MEM=96GiB STAGING=80GiB PIPELINE_DOP=9 ./bench/gb200-8gpu/sweep.sh 10000 q07
```

**Stop:** `relaunch.sh --stop-only` without `--local-only` **relaunches**. Use
`./benchmarks/stop-cn-2host.sh` on **both** hosts.

`relaunch.sh` must forward GPU_MEM / STAGING / DOP / watchdog / RPC / warmup to 09.
Watchdog 1800, RPC 3600, warmup 900, cold 60000, FE `query_timeout` ≥ cold.

After any refuse, later queries on that cluster are invalid. Restart both hosts.
Client `requested N` is chunk count. Read CN log `/tmp/cn-<octet>-<gpu>.log`:

```
N leases outstanding holding X of CAPACITY (largest free F < request R)
```

One heavy query per clean cluster (or accept 2 warm runs). Restore
`experimental/starrocks/benchmarks/tpch/queries/q11.sql` if you kill a sweep.

---

## If adding CNs — still SF10000

`cn-2host.sh` sets `NUM_CNS = NUM_CNS_PER_HOST * 2`. A third host needs a new launcher
(`MIN_BACKENDS`, warmup peers, SSH list, same fabric ClusterUUID). NUMA:
`CN_NODE=0 0 1 1`, `CN_CPUS=0-35 36-71 72-107 108-143`. Never `--membind` HBM nodes.

First experiment: **16 GPU / 4 hosts**, re-run **q05, q10, q17** only.

| 8-CN failure | More GPUs? |
|---|---|
| q05, q08, q17 pool OOM | Maybe if hash build partitions ~1/N |
| q10, q18 pool + arena squeeze | Maybe (more N can cut per-CN shuffle, free HBM for pool) |
| q03, q09, q21 fill any STAGING | Unlikely; needs copy-out-on-arrival ([OPEN.md](OPEN.md) PLAN-01) |

Do not retune q03/q09/q21 by raising STAGING. Lease count grew 405→588 (q03, 64→80 GiB).

To match SF3000-on-8 per-CN join budget for q05-class at SF10000: `N ≈ 8 × (10000/3000) ≈ 27`
GPUs (~7 hosts). Try **16 GPU** first; if leases/pool do not drop, stop adding boxes for those
queries.

---

## Files

| Path | Role |
|---|---|
| `bench/gb200-8gpu/sf10000/env.sh` | default 112/64 dop=9; pass explicit anyway |
| `bench/gb200-8gpu/sf10000/{README,TUNING-DISCOVERY}.md` | knobs + arms |
| `bench/gb200-8gpu/sweep.sh` | `sweep.sh 10000 qNN` |
| `experimental/starrocks/configs/gb200-8gpu/{launch,relaunch}.sh` | 2-host |
| `/scratch/prestouser/aocsa/bench-results/sf10000-8gpu-*` | `timings.csv` |
