# SF10000 8-CN tuning log (gcn-09 + gcn-18)

Started 2026-08-28 from the SF3000 measured split (112/64, dop=12 common /
dop=9 heavy). Dataset `/scratch/sirius/datasets/tpch_sf10000` is 3.7 T.

## Arms

| Arm | Split / dop | Result |
|---|---|---|
| A | 112/64/16 dop=12 | q01 4/4 pass (best warm 43.4s). q02 r0–r2 pass (best 15.0s); r3 600s watchdog. `sf10000-8gpu-common-20260828T054629Z` |
| B | 112/64 dop=12 + watchdog 1800 | q03: **405 leases / 63.9 GiB of 64**. `sf10000-8gpu-common-cont-20260828T060324Z` |
| C | 112/64 dop=9 | q03 still **406 leases / 63.9 GiB**. DOP is not the lever. `sf10000-8gpu-common-dop9-20260828T060632Z` |
| D | 96/80 dop=9 | q03 fills 80 GiB: **588 leases / 79.9 GiB**. Skip q03. `...q03-80gi-20260828T061121Z` |
| E | 112/64 dop=9 | **q04 4/4 pass** (best warm 27.5s). `sf10000-8gpu-q04-20260828T061414Z` |
| F | 112/64 dop=9 | q05 after q04: pool `bad_alloc`. `...q05-20260828T061630Z` |
| G | 112/64 dop=9 | **q05 on a fresh cluster** still pool-OOMs. Need more GPU_MEM. `...q05-fresh-20260828T061902Z` |
| H | 128/48 dop=9 | **q05 still pool-OOMs on cold** (fresh 128 GiB pool). `...q05-12848-20260828T062122Z` |
| I | 144/32 dop=6 | **q05 still pool-OOMs on cold.** Skip q05. `...q05-14432-20260828T062438Z` |
| J | 112/64 dop=9 | **q06 4/4 pass** (best warm 42.6s). `sf10000-8gpu-q06-20260828T062731Z` |
| K | 112/64 dop=9 | q07 after 4× q06: arena **487 leases / 63.6 GiB of 64**. May be leftover. Retry on a fresh cluster. `sf10000-8gpu-q07-20260828T063045Z` |
| L | 112/64 dop=9 | **q07 on a fresh cluster** still fills 64 GiB: **492 leases / 63.9 GiB**. Own shuffle, not leftover. `...q07-fresh-20260828T063346Z` |
| M | 96/80 dop=9 | **q07 4/4 pass** (best warm 48.1s). Needs 80 GiB staging; 64 GiB is not enough. `sf10000-8gpu-q07-80gi-20260828T063637Z` |
| N | 112/64 dop=9 | **q10 pool `bad_alloc` on a fresh cluster** (same class as q05). `sf10000-8gpu-q10-20260828T064216Z` |
| O | 128/48 dop=9 | q10 arena **485 leases / 47.9 GiB of 48**. Pool-OK at 128, staging too small. `...q10-12848-*` |
| P | 120/56 dop=9 | q10 fills 56 GiB: **600 leases / 55.8 GiB**. Skip q10. `...q10-12056-*` |
| Q | 112/64 dop=9 | **q11 4/4 pass** (best warm 15.9s, **9365098 rows** — FRACTION too loose, not a correctness result). `sf10000-8gpu-q11-20260828T065107Z` |
| R | 112/64 dop=9 | **q12 4/4 pass** (best warm 26.8s). `sf10000-8gpu-q12-20260828T065235Z` |
| S | 112/64 dop=9 | **q13 4/4 pass** (best warm 17.0s). `sf10000-8gpu-q13-20260828T065717Z` |
| T | 112/64 dop=9 | **q14 4/4 pass** (best warm 29.1s). `sf10000-8gpu-q14-20260828T065849Z` |
| U | 112/64 dop=9 | q15 r0 cold pass 47.9s; r1 empty-output **wedge** ~38s (not a 18000s cut). Retry on a fresh cluster. `sf10000-8gpu-q15-20260828T070116Z` |
| V | 112/64 dop=9 | q15 r0 empty-output wedge ~52s; **r1 pass 55.9s**; r2 wedge ~51s. Two times with first-cluster cold 47.9s. Client empty `rc=0` after ~50s, not arena/pool. `...q15-fresh-20260828T070526Z` |
| W | 112/64 dop=9 | **q16 4/4 pass** (best warm 9.3s). `sf10000-8gpu-q16-20260828T070853Z` |
| X | 112/64 dop=9 | q17 after q16: pool `bad_alloc`. Retry on a fresh cluster. `sf10000-8gpu-q17-20260828T071003Z` |
| Y | 112/64 dop=9 | **q17 pool-OOMs on a fresh cluster.** Own demand. `...q17-fresh-20260828T071319Z` |
| Z | 128/48 dop=9 | **q17 still pool-OOMs on cold.** Skip q17. `...q17-12848-*` |
| AA | 112/64 dop=9 | **q19 4/4 pass** (best warm 36.5s). `sf10000-8gpu-q19-20260828T072018Z` |
| AB | 112/64 dop=9 | **q20 4/4 pass** (best warm 34.0s). `sf10000-8gpu-q20-20260828T072310Z` |
| AC | 112/64 dop=9 | **q22 4/4 pass** (best warm 10.7s). Common list done. `sf10000-8gpu-q22-20260828T072606Z` |
| AD | 112/64 dop=9 | **q08 pool-OOMs on a fresh cluster.** DOP was the SF3000 lever. `...q08-*` |
| AE | 112/64 dop=6 | **q08 still pool-OOMs.** Skip q08. `...q08-dop6-*` |
| AF | 112/64 dop=9 | **q18 arena exhaust on a fresh cluster** (~351 MiB lease). Try 96/80 like q07. `...q18-*` |
| AG | 96/80 dop=9 | **q18 pool-OOMs** (80 GiB staging stole the pool). Skip q18. `...q18-80gi-*` |

q02 already has three passing times; do not wait on r3.

**Skip q03** (arena fills any STAGING), **q05 / q08 / q17** (pool `bad_alloc`; q08 even at dop=6), **q10** (pool OOM at 112/64; arena fills 48 and 56 GiB), **q18** (64 GiB arena full, 80 GiB pool OOM). **q07** needs **96/80**. **q09 / q21** not re-run: empty window at SF3000. Default common is 112/64 dop=9.

---

## Result: 14/22 with times

Physical HBM 189471 MiB ≈ 185 GiB. Occupancy 112+64+0.76 = 176.8 / 185 ≈ 96 %.
One knob set does not close the suite. q11 row count is not a correctness result.

| Queries | Knobs | Best warm |
|---|---|---|
| q01 | 112/64 dop=12 | 43.4s |
| q02 | 112/64 dop=12 | 15.0s (r0–r2; r3 600s watchdog) |
| q04, q06, q11–q16, q19, q20, q22 | 112/64 dop=9 | see below |
| q07 | 96/80 dop=9 | 48.1s |
| q03, q05, q08, q09, q10, q17, q18, q21 | empty window | — |

Warm bests at 112/64 dop=9 unless noted:

| q | ms | rows | dir |
|---|---|---|---|
| q01 | 43400 | — | `sf10000-8gpu-common-20260828T054629Z` (dop=12) |
| q02 | 15000 | — | same (dop=12) |
| q04 | 27500 | — | `sf10000-8gpu-q04-20260828T061414Z` |
| q06 | 42614 | 1 | `sf10000-8gpu-q06-20260828T062731Z` |
| q07 | 48070 | 4 | `sf10000-8gpu-q07-80gi-20260828T063637Z` (96/80) |
| q11 | 15852 | 9365098 | `sf10000-8gpu-q11-20260828T065107Z` (FRACTION too loose) |
| q12 | 26805 | 2 | `sf10000-8gpu-q12-20260828T065235Z` |
| q13 | 16991 | 10 | `sf10000-8gpu-q13-20260828T065717Z` |
| q14 | 29069 | 1 | `sf10000-8gpu-q14-20260828T065849Z` |
| q15 | 47909 / 55866 | 1 | two times; other runs empty mysql `rc=0` |
| q16 | 9320 | 27840 | `sf10000-8gpu-q16-20260828T070853Z` |
| q19 | 36455 | 1 | `sf10000-8gpu-q19-20260828T072018Z` |
| q20 | 34028 | 379709 | `sf10000-8gpu-q20-20260828T072310Z` |
| q22 | 10705 | 7 | `sf10000-8gpu-q22-20260828T072606Z` |

q01/q02/q04 ms are from the arm log (best warm), not re-read from csv here.

```bash
# common default
SCALE_FACTOR=10000 GPU_MEM=112GiB STAGING=64GiB PIPELINE_DOP=9 ./configs/gb200-8gpu/relaunch.sh
GPU_MEM=112GiB STAGING=64GiB PIPELINE_DOP=9 ./bench/gb200-8gpu/sweep.sh 10000 q04 q06 q12 ...

# q07 only
SCALE_FACTOR=10000 GPU_MEM=96GiB STAGING=80GiB PIPELINE_DOP=9 ./configs/gb200-8gpu/relaunch.sh
GPU_MEM=96GiB STAGING=80GiB PIPELINE_DOP=9 ./bench/gb200-8gpu/sweep.sh 10000 q07
```

Pass explicit `GPU_MEM` / `STAGING` / `PIPELINE_DOP` on relaunch and sweep. Persistent
shell exports override `env.sh` `${VAR:-default}`.

