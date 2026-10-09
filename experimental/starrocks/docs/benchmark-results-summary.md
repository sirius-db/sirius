# TPC-H join + GROUP BY results on 4× GB200, and where each one belongs

**Setup:**
- 4× GB200, one Sirius compute node (CN) per GPU, each with a 156 GiB slab pool;
- StarRocks FE, FILES() over parquet;
- queries q14 q05 q07 q08 q09 q12 q19;
- every result checked against DuckDB on the same parquet, with a leak check after every query.

All runs before 2026-10-06 ~18:30 used the original datasets on `/scratch/sirius/datasets`. These are mostly single runs, and the spread between runs is large: SF3000 q05 ranged 29–41 s.

## Which PR each result supports

| result | PR | why there |
|---|---|---|
| R1. Staging-arena baseline vs direct NIXL | sirius-db/sirius#2040 (NIXL transport) | the reason for a direct transport |
| R2. SF3000 sequential 2/7 → 4/7 | sirius-db/sirius#2041 (failed-query cleanup) | the fix that made those queries stop failing as fallout |
| R3. SF1000 whole files: unpinned vs pinned | sirius-db/sirius#2042 (pinned tables) | what pinning buys |
| R4. SF1000 7/7 on a stock FE | sirius-db/sirius#2043 (byte ranges) | the first run without the FE patch |
| R5. SF3000 q05/q08 pass; streaming on vs off | sirius-db/sirius#2016 (spill + streaming) | already in its description |
| R6. Join-hinted SQL at SF3000 | the tools repo README (`tests/tpch-hinted/`); one line in #2016 as the q09 workaround | no Sirius code |
| R7. Runtime filters at SF1000 and SF3000 | the runtime-filter follow-up PR | its own change |

sirius-db/sirius#2037–#2039 (translator, single-CN exchange, engine direct exchange) have no end-to-end number of their own. Their unit tests are their evidence. Every multi-CN run needs layers 1–4 together.

## SF1000

**R1. Exchange path: staging arena (`feat/pin-table-cn`, unpinned) vs direct NIXL.** The NIXL column is from the byte-range build, R4.

| query | staging arena | direct NIXL |
|---|---|---|
| q14 | 1.36 s | 5.6 s |
| q05 | 7.02 s | 2.0 s |
| q07 | 3.09 s | 2.1 s |
| q08 | 3.96 s | 2.5 s |
| q09 | 9.24 s | 5.7 s |
| q12 | 1.68 s | 1.3 s |
| q19 | 5.86 s | 1.6 s |

The staging-arena run used a 90 GiB pool plus a 72 GiB staging area, and its page cache was lukewarm (iteration 0 of a sweep). So it isn't a cold, like-for-like comparison.

**R3. Whole files per CN (FE patch): unpinned vs pinned.** Pinned means `lineitem` + `orders` on the host tier, compressed; the pin took 45 s.

| query | unpinned | pinned |
|---|---|---|
| q14 | 6.25 s | **2.69 s** |
| q05 | 2.09 s | **0.96 s** |
| q07 | 2.40 s | 4.61 s; 3.16 s like-for-like (the 4.61 s was a warm-up artifact) |
| q08 | 2.80 s | **1.11 s** |
| q12 | 1.36 s | 1.07 s (`l_shipmode` not pinned) |
| q19 | 1.96 s | 1.80 s (`l_shipmode` not pinned) |
| q09 | out of GPU memory | out of GPU memory |

**R4. Byte ranges on a stock FE.** 7/7:

| query | time |
|---|---|
| q14 | 5.6 s |
| q05 | 2.0 s |
| q07 | 2.1 s |
| q08 | 2.5 s |
| q09 | 5.7 s |
| q12 | 1.3 s |
| q19 | 1.6 s |

FE ranges split `lineitem`, `orders`, `customer` and `partsupp` files across CNs, and each range was read once.

**R5 / R7. The later fixes.** All 7 pass in every column.

| query | spill + streaming (#2016) | + runtime filters |
|---|---|---|
| q14 | 6.04 s | 5.17 s |
| q05 | 2.54 s | 1.87 s |
| q07 | 2.44 s | 1.99 s |
| q08 | 2.71 s | 2.05 s |
| q09 | 5.00 s | **2.40 s** |
| q12 | 1.55 s | 1.67 s |
| q19 | 1.64 s | 1.64 s |

**Not comparable:** after the 2026-10-06 reboot, `/scratch` (RAID `md127`) is inactive. A rerun on `/opt/sirius-ci/datasets/tpch_sf1000` passed 7/7 at 6.6–15.4 s per query, with a warm cache too. That copy has a different layout and was modified the same day by another user.

## SF3000

**R1 / R2. Before the memory fixes,** sequential, whole files:

| query | staging arena (cold, fresh cluster per query) | direct NIXL (`d798509d`) |
|---|---|---|
| q14 | 24.46 s | 20 s |
| q05 | out of memory or staging exhausted at every pool/staging split tried | out of GPU memory |
| q07 | 40.44 s | 24 s |
| q08 | not run | out of GPU memory |
| q09 | not run | out of GPU memory |
| q12 | 27.88 s | 8 s |
| q19 | 32.05 s | 20 s |

R2: the direct-NIXL column was 2/7 before the failed-query cleanup. q07 and q12 had failed only as fallout from earlier failed queries, which brought it to 4/7.

Pinning at SF3000 didn't complete: 3 of 4 CNs died during the pin, most likely from host memory exhaustion.

**R5 / R6 / R7. q05, q08, q09 across the fixes:**

| configuration | q05 | q08 | q09 |
|---|---|---|---|
| before | out of memory | out of memory | out of memory |
| spill to host | 39.5 s | 34.9 s | out of memory inside its join |
| spill + join-hinted SQL (R6) | 56.3 s | 33.4 s | **28.3 s** |
| spill + streaming (#2016) | 30.9 s | 22.0 s | out of memory inside its join |
| + runtime filters (R7) | 28.0 s | 27.5 s | **37.8 s alone**; fails in the full sequence |

**R5. Streaming on vs off,** same build, median of 2 fresh clusters, q09 excluded, 24/24 pass:

| query | streaming | ship after run |
|---|---|---|
| q05 | 30.9 s | 32.8 s |
| q07 | 28.6 s | 35.7 s |
| q08 | 22.0 s | 27.8 s |
| q19 | 17.9 s | 21.7 s |
| q14 | 28.2 s | 27.5 s |
| q12 | 20.9 s | 21.1 s |

**All 7 in sequence on one cluster:**

| query | spill + streaming (#2016) | + runtime filters |
|---|---|---|
| q14 | 26.3 s | 36.5 s |
| q05 | 31.4 s | 28.0 s |
| q07 | 33.6 s | 34.1 s |
| q08 | 26.8 s | 27.5 s |
| q09 | out of memory inside its join | out of memory on one CN |
| q12 | fails, fallout from q09 | fails, fallout from q09 |
| q19 | 34.3 s | 20.1 s |

q12 fails only because q09's fragments keep running, and keep their GPU memory, after the FE cancels q09. It passed in every run without q09.

## Open items

- **Restore `/scratch`** (reassemble `md127`, needs root). Then rerun SF1000 and SF3000 on the original data.
- **q09 in the full SF3000 sequence with runtime filters:** one CN runs out of GPU memory while its filtered `lineitem` scan runs. Engine logs (`ENGINE_LOGS=1`) are ready for the rerun.
- **Cancellation:** a cancelled query's running fragments aren't interrupted. That's why q12 fails after q09.
