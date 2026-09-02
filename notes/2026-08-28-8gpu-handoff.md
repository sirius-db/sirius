# Handoff: SF3000, two nodes (gcn-09 + gcn-18)

For a new Codex / Cursor session that continues **SF3000 only**. Knob log:
[`bench/gb200-8gpu/sf3000/TUNING-DISCOVERY.md`](../bench/gb200-8gpu/sf3000/TUNING-DISCOVERY.md).
Do not re-time queries that already have passing runs. Do not start SF10000.

2026-08-28. Repo `/home/prestouser/aocsa/sirius`, branch `dev`. Dataset
`/scratch/sirius/datasets/tpch_sf3000` (1.2 T, GPFS, both hosts).

**Fleet is two GB200 nodes, 4 GPUs each, 8 CNs.** That is all there is. No third host,
no 16-GPU experiment, no `N ≈ 27` join-budget scaling. q09 and q21 do not close by
adding boxes even if boxes existed.

---

## Result: 20/22

| Closed (times) | Knobs | Empty window |
|---|---|---|
| q01–q07, q10–q17, q19, q20, q22 | 112/64/16, **dop=12** | |
| q08, q18 | 112/64/16, **dop=9** | q08 r3 pool-OOM on a dirty cluster; r0–r2 ~9.1s |
| | | q09 (64, 72, and 80 GiB all full; dop=9 and dop=6 same ~217 leases) |
| | | q21 (fresh cluster, 301–315 leases / ~63.7 of 64 GiB) |

Warm bests: discovery "Common-arm warm best" plus q08 ~9.1s, q18 14.2s.
q11 **2,806,768 rows**. FRACTION `0.000000033333` is loose; not a correctness result.

Physical HBM **189471 MiB ≈ 185 GiB**. Occupancy 112+64+0.76 = 176.8 / 185 ≈ 96 %.

One knob set does not close the suite. dop=18 filled 64 GiB on q08 then pool-OOM'd.
Growing only STAGING while shrinking the pool (100/64) made q08 worse.

---

## Cluster

| | |
|---|---|
| FE | gcn-18 `10.87.140.53` |
| CNs | 4 on 18 + 4 on gcn-09 `10.87.140.44` = 8 |
| Fabric | `SIRIUS_EXCHANGE_STAGING_ARENA=fabric` |
| `UCX_TLS` | `cuda_copy,cuda_ipc,tcp,self`. No `rc_mlx5` |
| Scan | uring (`SIRIUS_CN_USE_SIRIUS_DATASOURCE=true`) |
| CI | GPUs ~02:00–03:50 UTC |
| Last seen | gcn-18 stopped ~11:10 UTC 2026-08-28. Confirm gcn-09 (`stop-cn-2host.sh`). Cursor often cannot SSH. |

**Do not** use `cluster4-numa.sh`, `run-abc.sh`, `benchmarks/cluster8.sh`.

```bash
# [18] bring up 8 CN. Always pass memory + DOP (shell exports override env.sh)
cd /home/prestouser/aocsa/sirius/experimental/starrocks
SCALE_FACTOR=3000 GPU_MEM=112GiB STAGING=64GiB SIRIUS_EXCHANGE_STAGING_BYTES=64GiB \
  PIPELINE_DOP=12 ./configs/gb200-8gpu/relaunch.sh

# common (cluster up). From repo root on 18.
GPU_MEM=112GiB STAGING=64GiB PIPELINE_DOP=12 \
  OUT=/scratch/prestouser/aocsa/bench-results/sf3000-8gpu-<tag>-$(date -u +%Y%m%dT%H%M%SZ) \
  ./bench/gb200-8gpu/sweep.sh 3000 $(cat bench/gb200-8gpu/sf3000/queries-common.txt)

# heavy. SET GLOBAL is enough if the common arm left the cluster clean.
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL pipeline_dop=9;"
PIPELINE_DOP=9 GPU_MEM=112GiB STAGING=64GiB \
  ./bench/gb200-8gpu/sweep.sh 3000 $(cat bench/gb200-8gpu/sf3000/queries-heavy.txt)
```

**Stop:** `relaunch.sh --stop-only` without `--local-only` relaunches. Use
`./benchmarks/stop-cn-2host.sh` on **both** hosts.

`relaunch.sh` must forward GPU_MEM / STAGING / DOP / watchdog / RPC / warmup to 09.
Watchdog 600, RPC 900, warmup 600 in `sf3000/env.sh`. After a refuse, raise watchdog
before retrying a heavy. FE `query_timeout` ≥ cold (`18000`).

After any refuse, later queries on that cluster are invalid. Restart both hosts.
Client `requested N` is chunk count. Read CN log `/tmp/cn-<octet>-<gpu>.log`:

```
N leases outstanding holding X of CAPACITY (largest free F < request R)
```

One heavy query per clean cluster (or accept 2 warm runs). Restore
`experimental/starrocks/benchmarks/tpch/queries/q11.sql` if you kill a sweep.

---

## What is left on this fleet

q09 and q21 fill every STAGING we gave them. Lease count grew 218 → 245 → 298 as
the arena went 64 → 72 → 80 GiB. DOP is not the lever (q09 dop=9 and dop=6 both
~217 leases at 64 GiB). Do not raise STAGING again. Do not cut GPU_MEM to make
room for a bigger arena.

That is copy-out-on-arrival ([OPEN.md](OPEN.md) PLAN-01), not another split and
not another host.

q08 concurrent leases *do* scale with DOP. Keep dop=9. A fourth warm on a dirty
cluster pool-OOM'd; relaunch before quoting a 4/4.

---

## Files

| Path | Role |
|---|---|
| `bench/gb200-8gpu/sf3000/env.sh` | default 112/64 dop=12; pass explicit anyway |
| `bench/gb200-8gpu/sf3000/{README,TUNING-DISCOVERY}.md` | knobs + arms |
| `bench/gb200-8gpu/sweep.sh` | `sweep.sh 3000 qNN` |
| `experimental/starrocks/configs/gb200-8gpu/{launch,relaunch}.sh` | 2-host |
| `/scratch/prestouser/aocsa/bench-results/sf3000-8gpu-*` | `timings.csv` |
