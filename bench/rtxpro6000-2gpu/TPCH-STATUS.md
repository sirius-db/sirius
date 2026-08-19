# TPC-H status — Sirius + StarRocks, this box

**Box**: 2× RTX PRO 6000 Blackwell (97887 MiB each, cc 12.0), driver 580.126.09, GPUs linked
`PIX` (PCIe, not NVLink). Xeon 8559C 48 vCPU, 499 GB RAM. **Data**: SF100,
`/home/ubuntu/tpch_parquet_sf100` (26 GB). **Topology**: 2 CNs, one per GPU.
**Measured** 2026-08-19. Raw CSVs: `results/`.

Two memory arms, everything else identical:

- **A** — `GPU_MEM=40GiB STAGING=16GiB` (56 GiB = 59 % of card) → **19/22 pass**
- **B** — `GPU_MEM=60GiB STAGING=32GiB` (92.6 GiB = 96.8 % of card) → **18/22 pass**

Both: `NUM_CNS=2 HOST_MEM=128GiB SIRIUS_QUERY_WATCHDOG_SECS=90`, `QUERY_TIMEOUT=180`,
1 cold + 3 warm runs. Correctness checked against a DuckDB CPU oracle over the same parquet —
**the benchmark harness itself checks nothing**.

## Per-query

| Query | A (ms) | B (ms) | vs DuckDB | Status |
|---|---|---|---|---|
| q01 | 3998 | 4093 | 0.096 % low | ok, drift |
| q02 | 512 | 573 | exact | **ok** |
| q03 | 1064 | 1058 | **0.336 % low, reorders top-10** | **wrong ranking** |
| q04 | 728 | 691 | exact | **ok** |
| q05 | 1961 | 1949 | 0.096 % low | ok, drift |
| q06 | 791 | 793 | exact | **ok** |
| q07 | 1415 | 1420 | 0.097 % low | ok, drift |
| q08 | refused | refused | — | **FAIL** |
| q09 | wedge 180 s | wedge 180 s | — | **FAIL** |
| q10 | 1271 | 1231 | **0.243 % low, reorders top-20** | **wrong ranking** |
| q11 | empty | empty | **agrees (0 rows)** | **ok** — see below |
| q12 | 808 | 791 | exact | **ok** |
| q13 | 842 | 833 | exact | **ok** |
| q14 | 1052 | 1043 | 0.0006 % low | ok, drift |
| q15 | 1949 | empty | 0.103 % low | **flaky** (1/3 warm in A, 0/3 in B) |
| q16 | 281 | 260 | exact | **ok** |
| q17 | 2249 | 2264 | exact | **ok** |
| q18 | 1605 | 1550 | exact | **ok** |
| q19 | 1046 | 1012 | 0.098 % low | ok, drift |
| q20 | 1464 | 1459 | exact | **ok** |
| q21 | 2710 | 2691 | exact | **ok** |
| q22 | 398 | 394 | 1.7e-16 (IEEE) | **ok** |

Warm medians. 18 common queries total **24195 ms (A)** vs **24105 ms (B)** — **−0.4 %**, i.e. a
50 % larger pool bought no speed. Correctness was **bit-identical** across both arms.

## Summary

- **11 queries byte-exact**, 6 more correct within 0.103 %.
- **2 queries wrong in a way that matters** — q03/q10 drift low enough to permute an
  `ORDER BY revenue DESC LIMIT N`.
- **2 queries fail outright** — q08, q09.
- **1 query flaky** — q15.

## Open defects

| # | Query | Symptom | Notes |
|---|---|---|---|
| 1 | q03, q10 | decimal drift, always **low**, only on `sum(l_extendedprice*(1-l_discount))` | worst 0.336 %; matches the documented ≤0.39 % band from SF1 — SF100 does not widen it. Keys/counts/dates exact. Deterministic, identical in both arms, so not memory-related |
| 2 | q09 | `exceeded 100 retries … OOM at operator HASH_JOIN`, both CNs | **not a sizing problem** — 50 % more pool changed nothing. Widest join in TPC-H; at 2 CNs the per-node build side is too large. Needs more CNs or engine work |
| 3 | q08 | A: `arena exhausted, 14 leases` → B: `no parked sender output to export for SenderSlot` | **two stacked defects.** The arena one *was* sizing and is fixed by `STAGING=32GiB`; that exposed an exchange head-of-line deadlock underneath |
| 4 | q15 | intermittently returns 0 rows | flaky in both arms; not config-sensitive. Known flake |

## q11 is not a defect

Recorded `wedge`, but **DuckDB also returns 0 rows** and the arithmetic confirms it: the query
hardcodes the SF1 threshold `0.0001` where TPC-H scales it `0.0001/SF`. At SF100 the bar is
**801,681,490** against a largest single part value of **23,649,655** — 34× too high, so nothing
qualifies. Sirius and DuckDB agree exactly.

It reads as a failure only because `mysql --batch` prints nothing — not even a header — for a
zero-row result, so `bench.sh`'s `[ -s "$f" ]` test files it as a wedge. Fixing the query would
mean scaling the threshold by SF; fixing the report means not trusting an empty file.

## Caveat on any number here

`bench.sh` scores `pass` on exit-code + non-empty file + no `ERROR` on line 1. It never compares
a value. Every correctness claim above comes from the separate DuckDB oracle
(`tools/oracle.py` + `tools/compare.py` + `tools/drift.py`), not from the harness.
