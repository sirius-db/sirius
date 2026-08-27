# TPC-H Benchmark — One-Page Brief

**Full plan:** [`TPCH-BENCHMARK-PLAN.md`](TPCH-BENCHMARK-PLAN.md) · **Authored** 2026-08-12

## What we are measuring

Three studies of Sirius on TPC-H, at **SF500 and SF1000**, against a GPU peer and a CPU peer.

| # | Study | Question | Engines | Regime |
|---|---|---|---|---|
| **1** | **Scale-Out** | Does Sirius scale 2 → 4 → 8 GPUs? | Sirius only | pinned/hot |
| **2** | **GPU Shootout** | Sirius vs cudf-polars, same box, no bias | Sirius, cudf-polars | **cold** |
| **3** | **Cost Efficiency** | $ per run vs a CPU engine | Sirius, StarRocks | warm |

## Hardware

| Role | Instance | Spec | $/hr |
|---|---|---|---|
| Sirius + cudf-polars | `massedcompute_A100_sxm4_80Gx8` | 8× A100 80 GB SXM4, 640 GB VRAM, 1500 GB host | **$13.25** |
| StarRocks — **primary** | `m8gd.48xlarge` | 192 vCPU Graviton4, 768 GB, local NVMe | **$8.83** |
| StarRocks — x86 ref | `m8i.48xlarge` | 192 vCPU Intel, 768 GB, **no local NVMe** | **$12.19** |

8 GPUs in **one node** — no cross-host work needed. Requires an **x86_64 rebuild** of Sirius.

## Data — both present on `/raid`, no generation needed

| Scale | Path | Size | Pinning on 8× 80 GB |
|---|---|---|---|
| SF500 | `/raid/prestouser/aocsa/tpch_parquet_sf500` | **132 GB** | 16.5 GB/GPU — comfortable |
| SF1000 | `/raid/tpch-sf1000` | **283 GB** | 35.4 GB/GPU — fits at 8 GPUs; **fails at ≤4** |

## Query set — 17 of 22

Engine A fails five queries at SF100, and every mechanism worsens with scale:
**q05, q08, q09, q10, q18** — excluded. (q08 sizes a 142 GiB request off a 12.5 MB input;
q09 asks for 1.13 TiB; q05/q10/q18 wedge from a cross-run leak, not query difficulty.)

| Tier | Queries | Use |
|---|---|---|
| 1 — anchor | `q04 q06` | only queries value-verified at SF500 today |
| 2 — expansion | `q02 q11 q12 q16 q20 q22` | **headline aggregate** (with Tier 1 = 8 queries) |
| 3 — timing-only | `q01 q03 q14 q15 q19` | values wrong (`1-l_discount` defect) — never in an aggregate |
| 4 — probe | `q13 q17 q21 q07` | named SF500 failure projections; run, report, exclude |

Charts must say **"17 of 22 · aggregates over 8"**, never "all 22 complete".
(`ENGINE-CONFIGS-AND-EQUIVALENCE.md:290` says 18/22 — it is wrong, q18's 4th run wedged.)

## The five things that decide whether this is credible

1. **The harness never checks values.** `status=pass` means *exit 0 and ≥1 row*. Needs a DuckDB
   oracle diff at **relative** tolerance 1e-12 — q06 varies ±1 ULP and exact equality false-fails it.
2. **The `(1 - l_discount)` decimal defect is NOT fixed at HEAD** and costs 7 queries. Root cause is
   **one function**: `expr_translator.rs:459-481` (`translate_arithmetic`) casts both decimal
   operands to FP64 and declares FP64 output, so the expression is FP64 before SUM sees it. The
   crate is untouched since the audit; `OPEN-ISSUES.md` #24 is still 🔴 OPEN and warns that fixing
   the SUM/AVG lowering instead **changes nothing**. Localized to the CN path — standalone Sirius is
   exact on the same files. **Fixing it takes the headline from 8 queries to 14.**
3. **Cold mode has a JIT trap.** `ast_jit` pays ~19 s of NVRTC compile on a cold cache (A100 is a
   new arch). "Cold" = page cache dropped, **not** compiler cache dropped.
4. **Never pin Sirius against a re-reading cudf-polars.** Study 1 pinned (Sirius vs itself, fair);
   Study 2 cold (both engines read parquet, fair). Mixing them is the worst bias available.
5. **80 GB/GPU is 43% of a GB200.** The 16 GiB exchange staging arena is **20% of an A100** and
   must shrink before anything runs.

## Cost — decided by speed, not price

Prices are close, so state the **break-even** on the chart:

> Sirius must beat **m8i by 1.09×**, **m8gd by 1.50×**, to break even per run.

At SF100 Sirius beat StarRocks on **all 17** passing queries, several by 3–4×. If that holds it
clears both bars easily. On Study 1 the box is priced whole, so **scaling efficiency converts
directly to cost efficiency** — 8 GPUs 3× faster than 2 means 3× cheaper per run.

## What we already have for free

`bench/sf1000-repro/` is **merged here** (PR #1371): six Sirius-side wins are in the build already
— q16 −39%, q21 −15.2%, q19, q17, `cuco` bucket, mpmc sentinels. Plus `ast_jit`, **−4.17% for zero
code** (`src/config.cpp:27` still ships the slow default). Only three cuDF patches are external;
recommend running **without** them so results reproduce from merged code alone.

**Reference target:** that campaign hit **8.180 s** for SF1000 on *one* GB300, pinned. Eight A100s
have 2.5× the aggregate VRAM but 31% the per-GPU HBM — beating it is a real result, and the
comparison is already apples-to-apples on data and query set.

## Order of work

**Study 2 at SF500** (one box, no new infra) → **Study 1 at SF500** (add topologies) →
**both at SF1000** (SF1000 pins only at 8 GPUs) → **Study 3** (needs AWS, runs independently).

Blocking first: x86_64 rebuild · shrink the staging arena · stand up the oracle diff ·
arm `SIRIUS_QUERY_WATCHDOG_SECS` (currently 0).
