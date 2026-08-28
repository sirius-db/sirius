# SF3000 8-CN tuning log (gcn-09 + gcn-18)

Measured 2026-08-28. **20/22 with times.** q09 and q21 fill any staging arena
we gave them (empty window). One TPC-H knob set does not close all 22.

Physical HBM is 189471 MiB ≈ 185 GiB, not the 184 GiB round number in older notes.

---

## What actually failed

`exchange staging arena exhausted` on a **hundred-megabyte** lease is not "the
query wants 400 MiB." The CN log line is the measurement:

```
N leases outstanding holding X bytes of CAPACITY
(largest free F < requested R)
```

If X ≈ CAPACITY, the arena is full of live shuffle. Raising STAGING helps only
until the next query's outstanding leases catch up. A starved RMM pool
(`std::bad_alloc: not enough capacity`) then **ratchets** the arena: `push_packed`
deep-copies into the pool before releasing the lease, so a pool OOM leaves
leases parked at capacity.

After any refuse, later queries on the same cluster are invalid. Restart both
hosts. Do not keep sweeping.

---

## Arms (in order)

| Arm | GPU_MEM / STAGING / HOST_MEM | dop | Result |
|---|---|---|---|
| A | 120 / 36 / 16 | 18 | q01–q04, q06 pass. q05 138-style fill: ~377 MiB lease vs 36 GiB full. FE died at q13 (`:9030` gone). Dir `sf3000-8gpu-2host-20260828T041608Z` |
| B | 120 / 36 / 16 (fresh) | 18 | Same q05 refuse on a **clean** cluster. Not leftover poison. Stopped. `...T042752Z` |
| C | 112 / 44 / 16 | 18 | q05 still dies. CN0: **138 leases / 43.7 GiB of 44 GiB**, largest free 317 MiB < 369 MiB request. That is q05's own shuffle. `...T043638Z` |
| D | 100 / 56 / 16 | 18 | **q05–q07 pass.** q08/q09: **236 leases / 55.8 GiB of 56 GiB**, largest free 47 MiB < 196 MiB. `...T044358Z` |
| E | 100 / 64 / 16 | 18 | q01–q07 pass. q08: **pool `bad_alloc` first** (04:57:32), then 324 leases / 63.8 GiB of 64. Growing arena without pool just delayed the OOM. `...T045230Z` |
| F | 112 / 64 / 16 | 18 | q08 still pool-OOM + 278 leases / 63.8 GiB of 64. More pool at dop=18 is not enough. `...T050145Z` |
| G | 112 / 64 / 16 | **12** | Common list **18/18 pass** (no q08/q09/q18/q21). `sf3000-8gpu-common-20260828T051410Z` |
| H | 112 / 64 / 16 | **9** | q08 r0–r2 pass (~9.1s). r3: pool OOM + 287 leases / 63.4 GiB of 64. `sf3000-8gpu-heavy-20260828T052305Z` |
| I | 112 / 64 / 16 | **9** | q09 fresh cluster fills 64 GiB: 218 leases / 63.9 GiB. `...q09-20260828T052806Z` |
| J | 112 / 64 / 16 | **6** | q09 still 217 leases / 63.7 GiB. DOP is not the q09 lever. `...q09-dop6-20260828T053044Z` |
| K | mixed 64+72 | **9** | gcn-09 stayed on 64 GiB. 72 GiB CNs: 245 leases / 71.7 GiB of 72. `...q09-72gi-20260828T053313Z` |
| L | 96 / 80 / 16 | **9** | Uniform 80 GiB. q09 still fills it: **298 leases / 79.95 GiB**. Empty window. `...q09-80gi-20260828T053618Z` |
| M | 112 / 64 / 16 | **9** | **q18 4/4 pass** (best warm 14.2s). `sf3000-8gpu-q18-20260828T053918Z` |
| N | 112 / 64 / 16 | **9** | q21 refused after q18 (298 leases). `...q21-20260828T054052Z` |
| O | 112 / 64 / 16 | **9** | **q21 on a fresh cluster** still fills 64 GiB: 301–315 leases / ~63.7 GiB. Same empty window as q09. Do not raise STAGING. `...q21-fresh-20260828T054315Z` |

Occupancy: 100+56 = 156.8 / 185 ≈ 85 %. 112+64 = 176.8 / 185 ≈ 96 %.

---

## Split: common vs heavy

`queries-common.txt`: q01–q07, q10–q17, q19, q20, q22.

`queries-heavy.txt`: q08, q09, q18, q21.

`pipeline_dop` is an FE `SET GLOBAL`. Memory knobs need a CN relaunch.

```bash
# common (cluster already 112/64)
./bench/gb200-8gpu/sweep.sh 3000 $(cat bench/gb200-8gpu/sf3000/queries-common.txt)

# heavy — SET GLOBAL is enough if the common arm left the cluster clean
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL pipeline_dop=9;"
PIPELINE_DOP=9 ./bench/gb200-8gpu/sweep.sh 3000 $(cat bench/gb200-8gpu/sf3000/queries-heavy.txt)
```

Do not run an all-22 `bench.sh` on one cluster: a q08 refuse poisons q10+.

---

## Common-arm warm best (dop=12, 112/64)

From `sf3000-8gpu-common-20260828T051410Z/timings.csv`. Best of 3 warm runs.

| q | ms | rows |
|---|---|---|
| q01 | 11961 | 4 |
| q02 | 4041 | 100 |
| q03 | 6543 | 10 |
| q04 | 4607 | 5 |
| q05 | 7602 | 5 |
| q06 | 6001 | 1 |
| q07 | 7069 | 4 |
| q10 | 7304 | 20 |
| q11 | 3800 | 2806768 |
| q12 | 5220 | 2 |
| q13 | 3576 | 28 |
| q14 | 4645 | 1 |
| q15 | 7139 | 1 |
| q16 | 2468 | 27840 |
| q17 | 10756 | 1 |
| q19 | 5657 | 1 |
| q20 | 6534 | 224354 |
| q22 | 2571 | 7 |

q11 row count is the fraction `0.000000033333` as applied; sanity-check against
the oracle before quoting it as TPC-H correctness.

dop=12 vs dop=18 on the overlap (q01–q07, 56 GiB arm D vs this): q01 13.4s → 12.0s,
q02 4.1s → 4.0s, q05 9.7s → 7.6s. Lower DOP did not lose the easy queries.

---

## Rules that survived the night

1. **Read the lease line**, not the client `requested 3` snippet. `requested N` is
   chunk count, not bytes.
2. **Do not cut STAGING below the 4-CN SF number** until high-water says so
   (SF1000 16 GiB died; SF3000 36 and 44 GiB died at q05).
3. **Pool OOM + full arena together** → grow GPU_MEM *or* cut `pipeline_dop`.
   Growing only STAGING while shrinking the pool (arm E 100/64) makes q08 worse.
4. **q08 concurrent leases scale with DOP.** dop=18 held ~280–324 / ~64 GiB;
   dop=9 closed three runs. **q09 does not:** dop=9 and dop=6 both ~217 leases
   at 64 GiB. Raising STAGING 64→72→80 grew leases 218→245→298 and still filled
   the arena. Empty window; needs copy-out-on-arrival, not another split.
5. **`pipeline_dop=18` is `avgCores/2` for a 36-core CN**, not a scale-factor
   constant. 36 was the 72-core 1-CN recipe (SF1000 first 8-GPU arm).
6. Fabric arena, `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` (no `rc_mlx5`), uring on
   GPFS. Unchanged from the SF1000 8-CN close.
7. **`relaunch.sh` must forward GPU_MEM/STAGING/PIPELINE_DOP to gcn-09.**
   SCALE_FACTOR-only left 09 on the env.sh default and mixed 64+72 GiB arenas.

Relaunch: `SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh` on gcn-18.
