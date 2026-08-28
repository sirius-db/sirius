# Tuning Sirius on 8× GB200 (gcn-09 + gcn-18)

4 CNs per host, 8 GPUs total. Same cards and NUMA map as
[`../gb200-4gpu/SIRIUS-TUNING-RUNBOOK.md`](../gb200-4gpu/SIRIUS-TUNING-RUNBOOK.md); this file only
covers what changes when you add a second box.

Companion: [`../../experimental/starrocks/benchmarks/2NODE-REPLICATE.md`](../../experimental/starrocks/benchmarks/2NODE-REPLICATE.md)
for fabric-arena bring-up. Launcher: `experimental/starrocks/configs/gb200-8gpu/launch.sh`.

---

## 0. The first 8-GPU SF1000 arm was not a 2× scale-out

Same parquet: `/scratch/sirius/datasets/tpch_sf1000` (GPFS, 264.5 GiB decimal).

| | 4-GPU 1-host | 8-GPU 2-host (first arm) |
|---|---|---|
| Results | `abc-sf1000-gpfs-20260828T013342Z` | `sf1000-8gpu-2host-20260828T030344Z` |
| Launcher | `cluster4-numa.sh` | `cn-2host.sh` |
| GPU_MEM / STAGING / HOST_MEM | 128 / 32 / 112 GiB | **copied 128 / 32 / 112** |
| `pipeline_dop` | FE default `avgCores/2` = **18** | **36** (1-CN / 72-core recipe) |
| `SIRIUS_CN_USE_SIRIUS_DATASOURCE` | unset → **true** (uring) | **false** (kvikio) |
| Transport | same-host `cuda_ipc` | fabric arena + cross-host MNNVL |
| Warm medians | 19/22 (q08/q09/q21 refused) | 22/22 |
| Geomean 8/4 on 19 overlap | | **1.21× slower** |

q01 did get faster (7271 → 4361 ms). Shuffle-heavy queries got worse: q04 2.17×, q06 1.67×,
q03 1.63×. That is the signature of too many fragment instances plus a cross-host exchange,
not of "8 GPUs are slower cards."

---

## 1. `pipeline_dop` is the first-order FE knob

`engine-a.env` already records the formula:

```
pipeline_dop = min(max_pipeline_dop=64, avgNumCoresOfBe / 2)
```

Each CN is `--physcpubind` to 36 cores, so `SHOW COMPUTE NODES` reports `CpuCores: 36`.
Auto DOP is **18**.

`2NODE-REPLICATE.md` pinned `pipeline_dop=36` because that recipe's 1 CN per host used
`--physcpubind=0-71` (72 cores → auto 36). The 8-GPU launch cargo-culted 36 onto 36-core CNs.

Over-reporting cores, in the same file: more fragment instances → smaller batches → more
exchange RPCs, and each CN has one blocking transport thread. CPU is not the scan bottleneck
on this path.

**SF1000 8-CN: `SET GLOBAL pipeline_dop=18`.** Do not use 36.

---

## 2. Staging shrinks when N grows

Measured on 8× A100 (`bench/a100x8/TUNING.md`): fewer GPUs need a **bigger** arena. SF500
closed at 12 GiB staging / 8 CNs vs 24 GiB / 4 CNs. The GB200 4-CN SF1000 preset (32 GiB)
is the 4-CN number. Copying it to 8 CNs wastes HBM that could be RMM pool.

Occupancy still `GPU_MEM + STAGING + 0.76 GiB ≤ ~184 GiB`.

| N | Starting split (SF1000, GB200) | occupancy |
|---|---|---|
| 4 (one box) | 128 / 32 / 112 | 160.8 / 184 = 87 % |
| **8 (two boxes)** | **128 / 32 / 112** | 160.8 / 184 = 87 % |

16 GiB staging exhausted at q05/q07 on the 2026-08-28T034231Z arm. Keep 32 GiB until a
high-water log line shows unused headroom. `pipeline_dop=18` still applies.

`HOST_MEM=112GiB` stays: 4×112 = 448 GiB committed per box, leftover ~509 GiB page cache vs a
264 GiB tree. Lowering it only if GPFS cache is the next measurement.

---

## 3. Scan path: match the 4-GPU GPFS arm

`SIRIUS_CN_USE_SIRIUS_DATASOURCE=false` selects kvikio/cudf. The 2-host overlay pins that
because a standalone Q06/SF100 on NVMe was ~20× faster than uring.

The 4-GPU GPFS sweep left it unset, so the engine defaulted to **true** (uring). kvikio on
IBM Scale is a different I/O stack than uring on the same files. The first 8-GPU arm mixed
that change with topology, so q06 moving 1222 → 2038 ms is not "8 GPUs lost."

**Fair vs 4-GPU GPFS: `SIRIUS_CN_USE_SIRIUS_DATASOURCE=true`.** A later arm can flip it false
on this pair and measure kvikio on Scale alone.

---

## 4. What you cannot config away

Cross-host shuffle still goes over the fabric arena. Same-host peers stay on NVLink
`cuda_ipc` (keep it in `UCX_TLS`). A query whose exchange is mostly off-box will not match
4-GPU NV18 all-to-all even with DOP and staging fixed.

`nvidia_peermem` is still unloaded. Do not put `rc_mlx5` in `UCX_TLS`.

Nightly CI still owns the GPUs ~02:00–03:50 UTC.

---

## 5. Launch (SF1000, 8 CNs)

Wipe FE meta on gcn-18 first (`2NODE-REPLICATE.md` §3). Then:

**[18]** FE + 4 CNs, leave in the foreground:

```bash
source /scratch/prestouser/aocsa/env.sh
cd /home/prestouser/aocsa/sirius/experimental/starrocks
./configs/gb200-8gpu/launch.sh 10.87.140.53 10.87.140.53
```

**[09]** 4 CNs only, as soon as 18's CNs are up (warmup budget 300 s):

```bash
source /scratch/prestouser/aocsa/env.sh
cd /home/prestouser/aocsa/sirius/experimental/starrocks
./configs/gb200-8gpu/launch.sh 10.87.140.44 10.87.140.53 --no-fe
```

Wait for **exactly 8** `Alive=true`. Then on **[18]**:

```bash
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
mysql -h 127.0.0.1 -P 9030 -uroot -e \
  "SET GLOBAL enable_pipeline_engine=true; SET GLOBAL pipeline_dop=18; SET GLOBAL query_timeout=1800;"

# q11: 0.0001/SF = 0.000000100000. Stage a copy; do not leave the repo file dirty.
OUT=/scratch/prestouser/aocsa/bench-results/sf1000-8gpu-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
Q11=benchmarks/tpch/queries/q11.sql
cp "$Q11" "$OUT/q11.sql.orig"
sed -i 's/0\.0001000000/0.000000100000/' "$Q11"

TPCH_DATA=/scratch/sirius/datasets/tpch_sf1000 \
FE_PORT=9030 QUERY_TIMEOUT=1800 COLD_TIMEOUT=6000 MIN_BACKENDS=8 \
  ./benchmarks/tpch/bench.sh --cold "$OUT/timings.csv" 3 \
  | tee "$OUT/bench.log"

git checkout HEAD -- "$Q11"
```

`run-abc.sh` launches `cluster4-numa.sh`. Do not use it here.

Teardown **both** hosts: `./benchmarks/stop-cn-2host.sh`.

---

## 6. One-change fallback

If the combined arm is still slower than 4-GPU on the overlap, the next arm is
**`pipeline_dop=18` only** on the old 128/32/112 + kvikio split, so DOP is isolated from
staging and the scan path.

---

## 7. SF3000: one split does not close all 22

Full log: [`sf3000/TUNING-DISCOVERY.md`](sf3000/TUNING-DISCOVERY.md).

36 GiB and 44 GiB staging both die at q05 with ~44 GiB of outstanding leases.
56 GiB clears q05–q07; q08 then fills 56–64 GiB **and** pool-OOMs at `pipeline_dop=18`.
The lever for q08 is **fewer concurrent fragments** (`pipeline_dop=12` for the 18
easy queries, `9` for q08/q09/q18/q21), not another STAGING bump at 96 % occupancy.

Do not run all 22 in one `bench.sh` process. A q08 refuse poisons q10+.
Use `queries-common.txt` then `queries-heavy.txt`.

## 8. SF10000: 14/22, two splits, eight empty windows

Full log: [`sf10000/TUNING-DISCOVERY.md`](sf10000/TUNING-DISCOVERY.md).

112/64 dop=9 closes q01, q02, q04, q06, q11–q16, q19, q20, q22. q07 needs 96/80.
q03/q09/q21 fill any arena (q09/q21 not re-run; SF3000 empty). q05/q08/q17
pool-OOM even with more GPU_MEM or dop=6. q10 needs both more pool than 112 and
more staging than 56. q18 fills 64 GiB then pool-OOMs at 96/80.
