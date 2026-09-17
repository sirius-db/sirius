# Handover: measuring this branch against a `dev` baseline

**DONE, 2026-09-17 — the measurement this document asks for has been taken. Results are in
`CHUNK_SKIPPING_PLAN.md` §6.21; the runner is `run-dev-baseline.sh` and the raw reports are in
`results-dev-baseline/`. Headline: pinned suite −11.2% (parquet + `cluster_by`) and −11.7%
(`.hpln` + `cluster_by`) against `dev`, with the unclustered parquet arm within 0.1% of `dev`, i.e.
no regression on the untouched path. The pin is 9.95 s against `dev`'s 18.69 s. Cold is
−19.9% (median of 5) but with a 13.7% run-to-run spread that this project had not noticed — read
§6.21 before quoting any cold number.** The rest of this document is kept as the recipe, and its
configuration notes all held up.

Written 2026-09-17, after merging `upstream/dev` (`82003512`) and fixing the API drift it
brought (`04a55c7a`). Nothing below had been measured at the time of writing — the merge landed first, deliberately,
so that any number taken afterwards is against current upstream rather than a three-week-old fork
point.

## What the question is

"What did the zonemap project buy?" `dev` has **no `.hpln` at all** — `read_simpatico` and
`pin_table(format='simpatico')` are this branch's. So the comparison has three arms, not two:

| arm | build | what it isolates |
|---|---|---|
| `dev` parquet | upstream/dev | the starting point |
| ours, parquet + `cluster_by` | this branch | what the zone-map work gave **parquet** |
| ours, `.hpln` + `cluster_by` | this branch | what the new format adds on top |

The middle arm matters: most of the pruning machinery (pin-time clustering, the group index,
chunk-subset fetch) applies to parquet pins and is usable without adopting a new format.

## Build both arms against the SAME cucascade

The merge bumped `cucascade` to upstream's `e9929fff` (ours was `1b0e7b6c`; their source needs
`clone(::cuda::stream_ref)`). `upstream/dev` already points there, so a `dev` worktree gets it
naturally — but **verify** with `git submodule status cucascade` before trusting a number.
Every measurement recorded in `CHUNK_SKIPPING_PLAN.md` before this merge was taken against the
OLDER cucascade, so do not compare across it. Re-baseline both arms on the merged build.

Build `dev` in a separate worktree so this one's build survives:

```bash
git worktree add ../sirius-dev upstream/dev
cd ../sirius-dev && git submodule update --init --recursive   # NOT automatic in a worktree
pixi run make
```

Budget ~40 min per build. If a build fails on `testcontainers_native`'s FetchContent patch step,
`pixi run make clean` first — clearing the `_deps` dirs by hand does not work, the populated
state is recorded elsewhere.

## Configuration — use these, they are measured, not assumed

- **`scan_task_batch_size: 8GB`** for host pins. §0's "host-tier should use 2 GB" is WRONG post-
  clustering: 2/4/8 GB measured 9.830 / 9.073 / **8.958** s (parquet + `cluster_by`). 2 GB is the
  worst by 9.7%.
- **`SIRIUS_EXP_FUSED_SCAN_FILTER=1`** and `SIRIUS_EXP_LATE_MAT=1` +
  `SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS=c_custkey,n_name,n_nationkey`. The fused-scan gate is
  **asymmetric**: it gates `.hpln`'s decode-time filtering entirely while parquet's reader filter
  is untouched, so a run without it understates `.hpln` on every join-heavy query.
  `hpln-suite.py` now sets these itself; `hpln-pin-bench.py` needs them in the environment.
- **`ast_interpret`, NOT `ast_jit`.** ast_jit is −4.17% on the GPU-pinned suite but a COLD
  regression (`.hpln` 52.521 → 59.833 s, +13.9%) and it hurts both *clustered* arms
  (8.853 → 8.985, 8.721 → 8.968). Only plain parquet likes it.
- **Late-mat is inert on host pins** — 1512 declines, all "the pinned entry is not
  device-resident", parquet included. Set it for parity, do not expect it to do anything.

## Datasets

| path | what | volume |
|---|---|---|
| `/datasets/tpch_sf1000` | parquet, natural order — **parquet's best arm** | root |
| `/datasets/tpch_sf1000_hpln_cluster` | `.hpln`, `cluster_by` lineitem+orders, chunks sized for an 8 GB batch | root |
| `/datasets/tpch_sf1000_sorted` | parquet, globally shipdate-sorted — parquet's WORST | root |
| `/scratch/tpch_sf1000_hpln_unsorted_4gb` | `.hpln` natural order | scratch |
| `/scratch/tpch_sf1000_hpln_sorted` | `.hpln` globally sorted | scratch |

**`/scratch` is 1.85x slower than `/` for parallel reads** — 11.73 vs 21.76 GB/s at 16 threads
(single-threaded they are identical; `/` is RAID1 over two NVMe and serves reads from both
members). Anything cold-compared against parquet MUST live on `/`. Moving a dataset between them
invalidates cold comparisons.

Root had ~190 GB free at handover. A full SF1000 `.hpln` is ~400 GB, so a new one means moving or
deleting an old one first.

## Corrections found while executing this

- **The `dev` worktree at `../sirius-dev` was stale** (`ea1c2783`, cucascade `1b0e7b6c`, built
  09-11). Fast-forwarded to `c5a6454c` and re-submoduled to `e9929fff`. The incremental build then
  hit exactly the `testcontainers_native` patch failure predicted below; `pixi run make clean`
  fixed it, and the rebuild took ~4 min, not 40 (ccache is warm across worktrees).
- **Trap #4's "lineitem yes, orders no" is about the SF100/§6.12 datasets, not this one.**
  `/datasets/tpch_sf1000_hpln_cluster` was written by `mk-hpln.py --sort cluster` over the default
  table set, whose `SORT_KEYS` covers lineitem AND orders. The matched parquet-clustered arm
  therefore uses the harness's default `CLUSTER_KEYS` (both), NOT
  `SIRIUS_BENCH_CLUSTER_TABLES=lineitem`. Its `mk-hpln.json` lists only the five small tables —
  the file was overwritten by a later pass — so it cannot be used to check this.
- **Cold `.hpln` runs vary by 13.7% run to run** while reading byte-identical data (§6.21). The
  "numbers to reproduce" below are single runs; the cold one is not reproducible to better than
  ±7%, and the first repetition taken here landed 2.9 s off it and looked exactly like a cucascade
  regression. Take a median of 5.

## Numbers to reproduce (this branch, pre-merge build, 8 GB, ast_interpret)

| | pin | host-pinned | cold | cold bytes |
|---|---|---|---|---|
| parquet | 21.75 s | 9.859 s | 58.824 s | 949.8 GB |
| parquet + `cluster_by` | 25.32 s | 8.853 s | — | — |
| `.hpln` + `cluster_by` | 9.92 s | 8.721 s | 49.873 s | 795.4 GB |

If the merged build moves these materially, suspect the cucascade bump before suspecting the
merge resolution.

## Traps this project has already paid for

1. **Check `hpln_source::stats().transport` before believing any I/O sweep.** Three sweeps
   (`fadvise_entries`, `uring_n_reactors`, `max_bytes_in_flight`) measured nothing because the
   scan was silently reading through `std::ifstream` — a coalescer dropped the `io_ctx`.
2. **A harness that cannot reproduce a known result cannot establish a new one.** Both bench
   harnesses have shipped configurations the project does not run.
3. **SF100 is not a proxy for the ordering decision.** It said a global sort beats per-chunk
   `cluster_by`; at SF1000 the opposite is true (−7.3% vs −14.7%).
4. **`cluster_by` is per TABLE.** lineitem yes, orders no — orders is the build side of most joins
   and its natural order is the join key. Clustering it costs ~4 s cold while reading FEWER bytes.
5. **Parquet's page index is unused** (`setup_page_index`, dictionary-page and bloom-filter
   pruning: zero call sites), and the datasets carry none. A clustered parquet dataset would
   therefore measure nothing today, and an estimated ~83% of q6's pages would prune if it were
   wired up — close to what `.hpln` achieves. **A material part of the cold advantage is a reader
   gap, not a format gap.** Say so when quoting it.
