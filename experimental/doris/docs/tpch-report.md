# Apache Doris + Sirius on the GPU: TPC-H Correctness and Performance Report

**Scope:** single node, one GPU · TPC-H SF1 / SF10 / SF100 · every run on 2026-09-19 on one AWS `g6e.8xlarge` (NVIDIA L40S 48 GB, 32 vCPU)

**Version under test.** Sirius upstream `dev` at `e0080cd6` (2026-09-18) plus the three engine changes and this backend as submitted in sirius-db/sirius #1840, #1841 and #1842 — the tree of the `experimental-doris` branch of the `morningman/sirius` fork at `75606ff5` (the two Doris + Sirius scenarios) and `f5116c6c` (the other scenarios; the engine code those scenarios execute is identical at both commits, see Appendix E). Submodules as pinned by the repository at those commits: `duckdb` `3ff87f1e` (v1.5.5), `cucascade` `e9929fff`, `substrait` `a7e045be`; CUDA 13 toolchain and `libcudf` 26.08.01 as pinned by the repository's `pixi.lock`. Apache Doris **4.1.4** (`doris-4.1.4-rc04-ad35a140c7f`), the official FE and BE binaries. NVIDIA driver 580.178.04.

**Test date.** 2026-09-19; every run of every scale and scenario started between 04:42 and 13:01 UTC that day, on the same instance. Exact commits, configuration hashes and session variables are in Appendix E.

## Summary

**What was tested.** A Rust process that joins an **unmodified, official Apache Doris 4.1.4 FE** as a backend node — called *the Sirius backend* below. The FE plans each query exactly as it would for a native BE and sends the resulting plan fragments to it; the Sirius backend translates the fragments of a query into **one Substrait plan** and runs that plan on the embedded Sirius engine on the GPU. Doris itself is not modified: the FE is the official release binary and the integration is configuration only (an FE config file, a set of global session variables, views over the `local()` table function). In the form tested here every query executes as a single fused plan, on one node, one GPU, one query at a time.

**Two questions, two answers.**

- *Correctness.* All 22 TPC-H queries at SF1, SF10 and SF100 return results that match the DuckDB baseline row by row — in every scenario and every round, 2,024 query executions in total, zero mismatches.
- *Performance.* At SF100, Doris reading parquet with its native BE versus Doris handing the same parquet to Sirius: **geometric-mean speedup 3.01×, 22-query total 3.80× (154.4 s → 40.6 s), all 22 queries faster.** The advantage grows with scale: 1.63× at SF1, 1.77× at SF10, 3.01× at SF100. At SF100 the GPU path is on par with DuckDB running on all 32 cores (40.6 s vs 38.9 s).

**The most important qualifier.** The comparison is Doris's *external-table* path (parquet through `local()`) against that same path with the execution engine swapped for Sirius. It is not Doris on its own internal storage, which remains 2.3× faster than the GPU path at SF100. Both sides ran on the same GPU instance, so no cost-normalised (same-price CPU machine) figure is given.

---

## 1. Results at a glance

### 1.1 SF100 headline — hot runs, 22-query total

Parquet 38 GB, `lineitem` a single 25.75 GB file. Hot = median of three warm rounds; seconds.

| # | Scenario | What runs | 22-query total (s) | Engine time (s) | Relative to #1 | CPU cores busy |
|---|---|---|---|---|---|---|
| 1 | **Doris, external table, default configuration** | Official BE 4.1.4, `local()` over parquet, 4.1.4 default session variables | **154.4** | — | 1× | ≈21 |
| 2 | Doris, external table, FE-side file split | Same as #1, file splitting moved to the FE (`file_split_size_on_be = 0`) | 205.4 | — | 0.75× | ≈26 |
| 3 | Doris, internal table | Data loaded into Doris storage; official TPC-H DDL (96 buckets, colocate group, `ANALYZE`) | **18.0** | — | 8.6× | ≈22 |
| 4 | Doris + Sirius, direct I/O | Sirius backend; Sirius's default `O_DIRECT` — every query reads its columns from the NVMe | 91.7 | 88.5 | 1.68× | ≈0.8 |
| 5 | **Doris + Sirius, page cache** | Same as #4, parquet read through the OS page cache | **40.6** | 37.5 | **3.80× (geometric mean 3.01×)** | ≈2.0 |
| 6 | DuckDB, CPU, 32 threads | DuckDB 1.5.5, one process, the same 22 statements over `read_parquet` views | 38.9 | — | 3.97× | ≈23 |
| 7 | DuckDB + Sirius extension, GPU | The same engine without Doris: DuckDB parses and optimises, Sirius executes; direct I/O | 94.6 | — | 1.63× | ≈1.2 |
| 7′ | Supplementary: #7 through the page cache | Two rounds only | 46.5 | — | 3.32× | — |

"Engine time" is the Sirius backend's own measurement of the `execute_substrait` call (Substrait lowering, planning, GPU execution, result conversion); the rest of the wall time is FE planning, RPC and result transfer. Cold-run (round 1) totals: #1 163.6 · #2 206.8 · #3 21.3 · #4 92.9 · #5 55.5 · #6 52.0 · #7 94.8.

### 1.2 Speedup by scale — hot runs, 22-query total, seconds

The Doris baseline at each scale is whichever of the two external-table configurations is faster at that scale (#2 at SF1 and SF10, #1 at SF100); both are shown.

| Scale | Parquet | Doris external table (baseline) | Doris internal table | Doris + Sirius, direct I/O | **Doris + Sirius, page cache** (engine) | DuckDB CPU | DuckDB + Sirius | DuckDB + Sirius, tables pinned in GPU memory | **Speedup, geomean · total** | Queries faster |
|---|---|---|---|---|---|---|---|---|---|---|
| SF1 | 0.25 GB | **5.2** (FE-side split; default 8.0) | 1.8 | 3.1 | **3.1** (1.1) | 1.2 | 1.1 | 0.6 | **1.63× · 1.70×** | 18 / 22 |
| SF10 | 3.6 GB | **15.0** (FE-side split; default 28.4) | 2.9 | 7.2 | **7.1** (4.9) | 4.1 | 7.9 | 2.0 | **1.77× · 2.13×** | 20 / 22 |
| SF100 | 38 GB | **154.4** (default; FE-side split 205.4) | 18.0 | 91.7 | **40.6** (37.5) | 38.9 | 94.6 | not run (dataset exceeds the 41 GB pool) | **3.01× · 3.80×** | 22 / 22 |

The queries that are slower on the GPU path at small scale are Q2, Q11, Q15 and Q22 at SF1 and Q11 and Q22 at SF10 — all sub-second queries whose time is dominated by FE planning and dispatch rather than execution (§3).

### 1.3 SF100 per query — hot median, milliseconds

| Query | Doris external table (#1) | Doris + Sirius, page cache (#5) | **Speedup** | Sirius engine time | DuckDB CPU (#6) | Doris internal table (#3) |
|---|---|---|---|---|---|---|
| Q1 | 5,339 | 2,260 | **2.36×** | 2,168 | 2,125 | 2,644 |
| Q2 | 1,814 | 704 | **2.58×** | 502 | 407 | 128 |
| Q3 | 20,270 | 1,822 | **11.13×** | 1,692 | 1,734 | 384 |
| Q4 | 3,359 | 1,162 | **2.89×** | 1,060 | 1,279 | 172 |
| Q5 | 8,721 | 2,012 | **4.33×** | 1,842 | 1,812 | 511 |
| Q6 | 2,046 | 1,445 | **1.42×** | 1,363 | 785 | 90 |
| Q7 | 4,176 | 2,560 | **1.63×** | 2,380 | 1,548 | 395 |
| Q8 | 11,059 | 2,572 | **4.30×** | 2,342 | 1,678 | 481 |
| Q9 | 14,681 | 3,096 | **4.74×** | 2,908 | 4,272 | 2,551 |
| Q10 | 11,537 | 1,966 | **5.87×** | 1,836 | 1,967 | 756 |
| Q11 | 1,153 | 565 | **2.04×** | 441 | 249 | 380 |
| Q12 | 2,781 | 1,576 | **1.76×** | 1,458 | 1,264 | 429 |
| Q13 | 3,763 | 1,051 | **3.58×** | 998 | 2,531 | 1,893 |
| Q14 | 2,434 | 1,793 | **1.36×** | 1,683 | 1,260 | 153 |
| Q15 | 4,521 | 1,860 | **2.43×** | 1,688 | 1,305 | 292 |
| Q16 | 1,355 | 318 | **4.26×** | 230 | 463 | 450 |
| Q17 | 8,049 | 2,246 | **3.58×** | 2,065 | 1,592 | 358 |
| Q18 | 29,349 | 2,127 | **13.80×** | 1,942 | 3,523 | 2,890 |
| Q19 | 3,162 | 2,090 | **1.51×** | 1,987 | 1,684 | 495 |
| Q20 | 4,123 | 1,762 | **2.34×** | 1,586 | 1,333 | 522 |
| Q21 | 9,330 | 5,066 | **1.84×** | 4,829 | 5,231 | 1,661 |
| Q22 | 1,349 | 541 | **2.49×** | 462 | 820 | 338 |
| **Total** | **154,371** | **40,594** | **geomean 3.01× · total 3.80×** | 37,463 | 38,862 | 17,973 |

Largest gains: Q18 13.8×, Q3 11.1×, Q10 5.9×, Q9 4.7×. Smallest: Q14 1.36×, Q6 1.42×, Q19 1.51×. The complete per-scenario tables for all three scales are in Appendix A.

### 1.4 Correctness

Every result of every round of every scenario was compared against the DuckDB baseline computed from the same parquet files (method in §5.1).

| Scale | Scenarios | Rounds × queries per scenario | Query executions validated | Result |
|---|---|---|---|---|
| SF100 | 7 | 4 × 22 | 616 | all match |
| SF10 | 8 (the seven above plus tables pinned in GPU memory) | 4 × 22 | 704 | all match |
| SF1 | 8 | 4 × 22 | 704 | all match |
| **Total** | | | **2,024** | **0 mismatches** |

In addition, before any GPU run, the 22 fused Substrait plans produced by the Sirius backend were executed on the CPU by DuckDB with its Substrait extension (the same consumer the Sirius engine uses) and compared with the original SQL at SF1: 22 of 22 matched. No query had to be excluded from any performance table.

### 1.5 Resources — scenario 5 at SF100, hot rounds

- GPU busy (share of time a kernel was running, averaged over each query's window): median 48 % across the 22 queries; Q13 83 %, Q21 74 %, Q9 65 %; Q16 13 %.
- GPU memory: high-water mark 35.6 GiB (Q9) inside the 41.4 GB pool; the engine's telemetry placed every batch in GPU memory — nothing spilled to the host or disk tier, and the spill directory stayed empty.
- CPU: ≈2 cores busy for the Sirius backend versus ≈21 for the native BE; resident memory 12 GiB versus 60 GB.
- Engine time as a share of end-to-end wall time: 92 % at SF100, 69 % at SF10, 35 % at SF1.

---

## 2. The seven scenarios: what each measures and which one is the headline

### 2.1 The scenarios

| # | Scenario | What it measures | Question it answers |
|---|---|---|---|
| 1 | Doris, external table, default configuration | Doris on the CPU as a user gets it out of the box, reading parquet through `local()` with every session variable at its 4.1.4 default | How fast is untuned Doris on parquet? The CPU baseline. |
| 2 | Doris, external table, FE-side file split | Same, with exactly one variable changed: `file_split_size_on_be = 0`, i.e. the FE splits files into 32/64 MB ranges (the pre-4.1 behaviour) instead of the BE splitting them. On a single-file table the default leaves the scan under-parallelised at small scale. | Removes a scan-parallelism artefact of the default on single-file tables, so that the Doris baseline at each scale is Doris's faster external-table configuration (this one at SF1 and SF10; the default at SF100). Both are always reported. |
| 3 | Doris, internal table | Doris on its home ground: data loaded into its own storage with the official TPC-H DDL (bucketing, a colocate group for `lineitem`/`orders`, `ANALYZE`); no parquet decoding, zone maps, colocated joins | How far is the GPU path from Doris at its best? |
| 4 | Doris + Sirius, direct I/O | The Sirius backend with Sirius's default I/O assumption: every query reads the columns it needs from the NVMe with `O_DIRECT`, bypassing the page cache | What a default deployment gets; how much of the time is disk |
| **5** | **Doris + Sirius, page cache** | **The execution engines themselves**: both sides read the same parquet from memory | **The headline comparison** (§2.2) |
| 6 | DuckDB, CPU, 32 threads | A strong single-node columnar CPU engine on the same SQL and the same files | Is the Doris external-table baseline "too slow"? What does the GPU buy over the best CPU engine on this machine? |
| 7 | DuckDB + Sirius extension, GPU | The same engine with the frontend swapped: DuckDB plans, Sirius executes, no Doris FE, no backend protocol, DuckDB's plan shape rather than the FE's | The difference between #4 and #7 (and between #5 and #7′) is the cost of going through Doris: FE planning, the backend protocol, and the FE's plan shape |

Supplementary scenarios run at some scales only: **7′**, scenario 7 through the page cache (SF100, two rounds) — the engine-level counterpart of scenario 5; and **tables pinned in GPU memory** (SF10 and SF1): scenario 7 after pinning all eight tables into GPU memory with `pin_table`, i.e. the engine with neither parquet decoding nor host-to-GPU transfer on the query path — an upper bound for this engine on this GPU.

### 2.2 The headline comparison and why it matters

**The headline number is scenario 5 against Doris's faster external-table configuration at each scale** — scenario 1 at SF100, scenario 2 at SF1 and SF10.

Why this pair:

- Same FE process (never restarted between scenarios), same SQL, same views, same parquet files, same plan shape — the FE chooses the join order for both sides — and both sides reading from memory. **The only variable is the execution engine.** This is the cleanest engine-versus-engine comparison available without modifying Doris.
- *Why not scenario 4 (direct I/O)?* With `O_DIRECT` Sirius re-reads every column it needs from disk on every query (≈180 GiB per 22-query round at SF100, about 50 s per round on this NVMe), while Doris keeps the files in the page cache and does not touch the disk in warm rounds. The difference is an I/O path, not an engine. Scenario 4 is reported in full (1.68× at SF100) because it is what a default deployment measures.
- *Why not scenario 3 (internal table)?* The internal table changes the storage format: no parquet decoding, bucketed and colocated data, zone maps. It is not the same data on the same path. It answers a different — and legitimate — question, "how far is this from Doris at its best", and is reported in full: 18.0 s at SF100, 2.3× faster than the headline scenario.

What the headline comparison means:

1. **For Doris users:** with data in parquet (a lakehouse or external-table setup), the optimiser unchanged and only the execution engine swapped, the GPU path delivers 3× at SF100 with every query faster; the advantage grows with data size.
2. **For Sirius:** this is the first end-to-end number for Sirius behind a frontend other than DuckDB, driven through Substrait, with the whole of TPC-H passing correctness at three scales.
3. **The Doris path costs little.** Scenario 7′ shows that the FE's plans, translated by the Sirius backend, execute at least as fast in the engine as DuckDB's own plans (engine time 37.5 s vs 46.5 s at SF100; per-query geometric mean 0.84× — the FE's plans are better on Q9, Q4 and Q18 and worse on Q13 and Q22), and the fixed overhead outside the engine is 50–240 ms per query (FE planning 19–148 ms of it). The 3× is therefore an engine number, not a frontend number.

---

## 3. Reading the numbers

1. **Where the GPU wins most.** Large scans feeding large aggregations and joins lead: Q18 13.8×, Q3 11.1×, Q10 5.9×, Q9 4.7×, Q5 / Q8 / Q16 4.3×. The smallest gains are narrow scans of `lineitem` with little computation — Q14 1.36×, Q6 1.42×, Q19 1.51×, Q7 1.63× — where Doris spreads parquet decoding across 32 cores while Sirius moves every column it touches through page cache → pinned host memory → PCIe → GPU decoding.
2. **The advantage grows with scale** (1.63× → 1.77× → 3.01×). At SF1 and SF10 individual queries take tens to hundreds of milliseconds and the fixed cost of FE planning plus the two-phase fragment dispatch dominates: at SF1 only 35 % of the GPU path's wall time is engine time. Queries with many fragments pay the most (Q2 has twelve), which is why Q2, Q11, Q15 and Q22 are slower on the GPU path at SF1.
3. **The GPU is busy about half the time.** GPU busy is 48 % at the median for scenario 5 at SF100. The pinned-table reference at SF10 runs the 22 queries in 2.0 s against 7.9 s for the same engine reading parquet and 4.9 s of engine time in scenario 5: with the data resident in GPU memory a further ≈2.4× is available. The remaining time is parquet decoding and host-to-GPU transfer, so the lever is data residency (pinned tables, a prefetch cache), not a faster GPU.
4. **Direct I/O costs about 50 s per round on this disk** — scenario 4 91.7 s versus scenario 5 40.6 s, and likewise scenario 7 94.6 s versus 7′ 46.5 s — even though the RAID0 NVMe sustains 2–3 GB/s. Warm comparisons must use the page-cache scenario.
5. **On par with 32-thread DuckDB at SF100** (40.6 s vs 38.9 s); at SF10 DuckDB is still 1.7× faster (4.1 s vs 7.1 s) because of the fixed overhead. DuckDB is 4× faster than Doris's external-table path at SF100, which is the honest scale of the Doris baseline.
6. **Doris on internal tables is 2.3× faster than the GPU path** at SF100 (18.0 s; Q3 0.38 s, Q6 0.09 s, Q18 2.9 s) and 8.6× faster than its own external-table path — bucketing, colocated joins, zone maps and column storage read directly, with no parquet decoding. For the GPU path to compete on Doris's home ground, data has to be resident near the GPU rather than decoded from parquet per query.
7. **SF100 fits in the L40S.** The GPU pool's high-water mark was 35.6 GiB (Q9; Q4 32.1, Q1 29.4) in a 41.4 GB pool, and no batch was placed in the host or disk tier. The "dataset larger than GPU memory" regime was therefore not exercised; by linear extrapolation it starts around SF300 on this GPU.
8. **Resources.** Sirius backend: 1–2 CPU cores, 12 GiB resident. Doris BE: 21 cores / 60 GB (default) or 26 cores / 85 GB (FE-side split). DuckDB: 23 cores / 16 GB. Doris internal table: 22 cores / 50 GB.

---

## 4. Test environment

| Item | Value |
|---|---|
| Instance | AWS `g6e.8xlarge`, us-east-1: NVIDIA **L40S 48 GB** (PCIe 4.0 ×16, 864 GB/s), **32 vCPU** AMD EPYC 7R13, 248 GiB RAM, no swap; on-demand $4.529/h |
| Storage | Two 450 GB instance NVMe drives in RAID0 (`mdadm`, ext4, `noatime`): 2.0–3.2 GB/s `O_DIRECT` sequential read on real files. Datasets, the Doris storage root, the Sirius spill directory and the SF100 baseline live there. Root volume: 300 GB gp3 (binaries, build trees, environments) |
| OS / driver | Ubuntu 24.04, kernel 7.0.0-1012-aws, NVIDIA driver 580.178.04 (CUDA 13.0) |
| Memory layout | Sirius: GPU pool 90 % of the L40S = 41.4 GB, pinned host tier 160 GiB (leaving room for the 38 GB dataset in the page cache), spill to the NVMe. Doris BE: `mem_limit` = 70 % of RAM. FE: 4 GB heap. The two backends never run at the same time |
| Software | Apache Doris **4.1.4** (`doris-4.1.4-rc04-ad35a140c7f`) — the official FE binary and the official BE binary from the same tarball; DuckDB **v1.5.5** (the shell built in the Sirius tree); Sirius engine and Sirius backend (`sirius-doris-be` 0.1.0) at the commits listed in Appendix E |
| Datasets | Generated with `tpchgen` as parquet, **one file per table, generator-default row groups**: SF1 0.25 GB, SF10 3.6 GB, SF100 38 GB (`lineitem` 25.75 GB in one file). Exposed to Doris as views over `local()` with `shared_storage = true`, to DuckDB as views over `read_parquet` |
| Queries | The 22 TPC-H queries as shipped in Doris's TPC-H tooling for 4.1.4, unchanged, run in order Q1…Q22, one stream. (Q11's threshold in that SQL is `0.000002` rather than the specification's `0.0001 / SF`.) |

---

## 5. Method

### 5.1 Correctness

- **Baseline.** DuckDB 1.5.5 executes the same 22 statements over the same parquet files and writes each result as TSV. The SF1 and SF10 baselines are checked into the repository; the SF100 baseline is generated on the machine.
- **Comparison.** Column names are compared case-insensitively. If the query has a top-level `ORDER BY`, rows are compared in order, with rows that tie on the sort keys allowed to appear in either order; otherwise both sides are sorted before comparison. Numeric columns are compared with a small relative tolerance. The validator has negative self-tests: a changed value, a reversed row order, a deleted row and a renamed column are each reported as a mismatch.
- **Two-level differential.** (1) *CPU:* the fused Substrait plan of each query, exactly as the Sirius backend produces it from the FE's fragments, is executed on the CPU by DuckDB with its Substrait extension — the consumer the Sirius engine itself uses — and compared with the original SQL. This isolates translation errors from engine errors. (2) *GPU:* the same plans executed by Sirius, in every round of every scenario at all three scales. Any difference at this level can only come from the engine.
- **Coverage.** Every round's `result.tsv` of every scenario is validated — the Doris-native scenarios and the DuckDB references included — and a query that failed validation in any round would be excluded from every performance table for that scenario. The 22 queries exercise: file scans; hash joins (inner, left semi, right semi, right anti, right outer, null-aware left anti); two-phase aggregations (fused into one), including `count(distinct)`; sort and top-N; cross joins; exchanges (removed by fusing); slot references, string / integer / decimal / date literals, comparison and compound predicates, arithmetic, casts, `IN` lists, `like`, `if`, `year`, `substring`; `sum`, `count`, `avg`, `min`, `max`.

### 5.2 Performance

- **Fairness rules.** One FE process for all scenarios, never restarted. The same SQL, the same views, the same parquet files. One backend at a time, each on its own ports. While a native scenario runs, the Sirius backend is dropped from the FE (`ALTER SYSTEM DROPP BACKEND`) and re-registers itself when it restarts: a registered backend that never reports its core count drags the FE's automatic parallelism down to one instance per fragment. The Doris scenarios use Doris 4.1.4's default session variables with no tuning; scenario 2 changes exactly one variable. The Sirius backend scenarios set the variables its current form requires — one instance per fragment, no local shuffle, no runtime filters, a single result stream, whole-file scan ranges (Appendix E) — which are prerequisites of the single-plan form, not tuning. The only FE configuration change is a longer fragment RPC timeout, because the Sirius backend executes the whole query inside the first execution RPC.
- **Cold and hot rounds** (following ClickBench and the Sirius paper). Cold = the backend process freshly started, every parquet file evicted with `posix_fadvise(DONTNEED)` and the page cache dropped. Each scenario runs 1 cold + 3 hot rounds of the 22 queries; **hot = median of the three**, the minimum is also recorded (Appendix A).
- **Metrics.** *Wall time* (primary): the MySQL client's round trip per query — FE planning, dispatch, execution, result fetch. *Engine time* (Sirius backend only): the `execute_substrait` call. *FE audit log*: FE end-to-end, planning time and first execution RPC per query, joined to the client's window (Appendix C). *Sampler* at 0.5 s: resident memory, bytes read from disk, CPU time of the backend process, GPU memory in use, GPU and memory-controller utilisation (Appendix D). *Engine telemetry*: per-batch placement tier and spill.
- **Aggregation.** Per-query speedup = Doris wall time / Doris + Sirius wall time (hot medians). Summary = geometric mean over the 22 queries, plus the ratio of the 22-query totals. Cost-normalised speedup = the same number, because both sides ran on the same instance at the same price.
- **Baseline choice.** Per scale, whichever of the two Doris external-table configurations is faster; both are reported.
- **Why the page-cache scenario is the headline.** Sirius's default direct I/O re-reads the columns of every query from disk; Doris's warm rounds read from the page cache. Scenario 5 puts both sides on the same I/O path so that the comparison is between engines (§2.2).
- **Reference points.** DuckDB on the CPU (#6); the same engine with DuckDB as the frontend (#7, #7′) — the gap to #4/#5 is the cost of the Doris path; tables pinned in GPU memory (SF10, SF1) — the engine without parquet decoding or transfer; Doris on internal tables (#3).
- **Reproducibility.** One command runs every scenario at a scale (Appendix F). Every run records the repository commit, instance type, kernel, driver, the hashes of every configuration file, the backend list and the effective global session variables (Appendix E).

---

## 6. Why the numbers can be trusted

1. **Same FE, same plan shape.** Both sides of the headline comparison get their plans from the same optimiser in the same process; the comparison is of execution engines, not of query optimisers.
2. **Every result validated.** All 2,024 query executions behind the performance tables — Doris-native and DuckDB references included — were compared against the baseline; nothing in the tables is a wrong answer.
3. **Two-level differential.** Plan-translation errors and engine errors are separated: the fused plans were proven correct on the CPU before they were executed on the GPU.
4. **Standard measurement conventions.** Default configuration and no tuning; cold and hot rounds reported separately; median of three hot rounds; geometric mean across queries alongside the total.
5. **Four reference points prevent cherry-picking.** A strong CPU engine (DuckDB), the same GPU engine with a different frontend, the engine with data resident in GPU memory, and Doris on internal tables — and the last of these beating the GPU path by 2.3× is reported as prominently as the headline.
6. **The Doris baseline is Doris's faster configuration at each scale**, and both configurations are always shown.
7. **Full environment capture and one-command reproduction**: commit, machine, driver, configuration hashes and session variables are recorded per run.

---

## 7. Limitations and caveats

Ordered by how much they affect the headline number.

1. **The comparison target is Doris's external-table path, not Doris at its best.** Reading parquet through `local()` means no statistics, no indexes, no colocation and per-query parquet decoding. Doris on internal tables is 2.3× faster than the GPU path at SF100. The headline is "parquet vs parquet".
2. **No cost-normalised figure.** Both sides ran on the same GPU instance: Doris used all 32 cores but also "paid" for the GPU. The Sirius paper's convention — Doris on a CPU instance of the same hourly price (a `c7i.24xlarge` at $4.28/h would match) — was not run, so no per-dollar speedup is given. A rough expectation is 1.5× or thereabouts; it has to be measured.
3. **The headline scenario turns off Sirius's default direct I/O.** With the default (`O_DIRECT`) the SF100 speedup is 1.68×. Using the page cache is the configuration that favours Sirius; the justification is parity with Doris's I/O path.
4. **Doris-side configuration choices.** Choosing the faster of two Doris configurations per scale is a judgement call (both are reported). The session variables the Sirius backend requires — one instance per fragment, no local shuffle, no runtime filters, whole-file scan ranges — make the FE's plans for the GPU path different from what a production Doris would produce for its own BE.
5. **Execution form.** One fused plan per query is an *upper bound* for the next form, in which each FE fragment executes as its own unit. One GPU, one node, one query at a time; a single-stream run only — no concurrency or throughput numbers.
6. **Scale.** SF100 fits entirely in the L40S's memory (35.6 GiB high-water mark), so the behaviour with a dataset larger than GPU memory was not exercised; SF1000 and multi-node were not tested. At SF1 and SF10 the speedup is diluted by the FE's fixed per-query cost.
7. **Query set.** 22 TPC-H queries in Doris's variant of the SQL, single stream, no refresh functions, no audited TPC-H run.
8. **Data layout.** One file per table with the generator's default row groups is optimal for neither side: Doris parallelises by file split, and Sirius recommends large row groups and multiple files.
9. **Measurement precision.** Three hot rounds per scenario. Resource sampling at 0.5 s makes the CPU and GPU figures of sub-second queries approximate; GPU busy is `nvidia-smi`'s coarse utilisation. The native BE has no engine-time counter — only the FE audit split. Cold rounds are not fully equivalent across sides (the GPU pool is pre-reserved, the JIT is warm) and are reference only.
10. **Shared machine.** While a Doris scenario runs, the FE and the BE share the CPU; this slightly disadvantages Doris. The GPU scenarios leave the CPU to the FE.
11. **Versions.** The engine is a fork build that includes three changes not yet merged upstream (Appendix E); Doris was tested at 4.1.4 only.

---

## 8. Conclusions and next steps

**Conclusions**

1. Sirius runs the full TPC-H behind an unmodified Doris FE through Substrait, with every result matching the baseline at SF1, SF10 and SF100.
2. On Doris's external-table (parquet) path at SF100, swapping the native BE for the Sirius backend yields a 3.0× geometric-mean and 3.8× total speedup with all 22 queries faster; the speedup grows with scale.
3. Going through Doris costs little: 50–240 ms of fixed overhead per query, and the FE's plans execute at least as fast in the engine as DuckDB's own plans.
4. The GPU is busy about half the time; parquet decoding and host-to-GPU transfer take the rest. Data resident in GPU memory is worth a further ≈2.4× on this engine (SF10 measurement).
5. Doris on internal tables remains 2.3× faster than the GPU path at SF100. The next lever for the GPU path is data residency, not GPU compute.

**Next steps**

- Run the Doris scenarios on a CPU instance of the same hourly price for a cost-normalised speedup.
- Measure data residency at SF100: tables pinned in host or GPU memory and the engine's prefetch cache, on the Doris path.
- Merge the three engine changes upstream.
- Execute the FE's fragments as fragments (store-and-forward within the node) instead of one fused plan, which is the path to multi-node execution; the numbers in this report are the upper bound that form is measured against.

---

## Appendix A. Per-query hot runs — all scenarios, all scales

Median of the three hot rounds in ms, minimum in parentheses. Speedup and engine time refer to the headline pair at that scale.

### A.1 SF100

| Query | Doris ext. default (#1) | Doris ext. FE split (#2) | Doris+Sirius direct I/O (#4) | Doris+Sirius page cache (#5) | DuckDB CPU (#6) | DuckDB+Sirius (#7) | Doris internal (#3) | Speedup #1/#5 | #5 engine |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | 5,339 (5,327) | 7,955 (7,882) | 2,678 (2,640) | 2,260 (2,257) | 2,125 (2,104) | 3,379 (2,994) | 2,644 (2,624) | 2.36× | 2,168 |
| Q2 | 1,814 (1,811) | 1,102 (1,097) | 1,024 (1,024) | 704 (693) | 407 (406) | 1,144 (1,121) | 128 (119) | 2.58× | 502 |
| Q3 | 20,270 (20,092) | 24,030 (23,979) | 4,523 (4,514) | 1,822 (1,776) | 1,734 (1,712) | 4,636 (4,631) | 384 (382) | 11.13× | 1,692 |
| Q4 | 3,359 (3,295) | 5,875 (5,843) | 2,383 (2,356) | 1,162 (1,113) | 1,279 (1,269) | 2,412 (2,345) | 172 (161) | 2.89× | 1,060 |
| Q5 | 8,721 (8,684) | 12,147 (12,069) | 5,651 (5,632) | 2,012 (1,952) | 1,812 (1,806) | 5,865 (5,854) | 511 (504) | 4.33× | 1,842 |
| Q6 | 2,046 (1,983) | 4,125 (4,124) | 2,832 (2,830) | 1,445 (1,367) | 785 (779) | 2,959 (2,927) | 90 (89) | 1.42× | 1,363 |
| Q7 | 4,176 (4,172) | 6,390 (6,367) | 6,353 (6,338) | 2,560 (2,555) | 1,548 (1,543) | 6,233 (6,221) | 395 (383) | 1.63× | 2,380 |
| Q8 | 11,059 (10,871) | 13,033 (13,027) | 7,459 (7,439) | 2,572 (2,529) | 1,678 (1,677) | 7,749 (7,747) | 481 (472) | 4.30× | 2,342 |
| Q9 | 14,681 (14,522) | 17,040 (17,010) | 8,383 (8,365) | 3,096 (3,089) | 4,272 (4,255) | 8,588 (8,566) | 2,551 (2,512) | 4.74× | 2,908 |
| Q10 | 11,537 (11,434) | 14,435 (13,716) | 4,346 (4,345) | 1,966 (1,923) | 1,967 (1,956) | 4,594 (4,590) | 756 (748) | 5.87× | 1,836 |
| Q11 | 1,153 (1,141) | 1,409 (1,293) | 1,048 (995) | 565 (562) | 249 (247) | 1,109 (1,099) | 380 (370) | 2.04× | 441 |
| Q12 | 2,781 (2,706) | 5,073 (5,037) | 2,724 (2,722) | 1,576 (1,572) | 1,264 (1,246) | 2,839 (2,838) | 429 (425) | 1.76× | 1,458 |
| Q13 | 3,763 (3,745) | 3,839 (3,836) | 2,672 (2,668) | 1,051 (1,034) | 2,531 (2,527) | 2,427 (2,423) | 1,893 (1,873) | 3.58× | 998 |
| Q14 | 2,434 (2,417) | 4,271 (4,260) | 4,270 (4,231) | 1,793 (1,730) | 1,260 (1,259) | 4,651 (4,595) | 153 (147) | 1.36× | 1,683 |
| Q15 | 4,521 (4,477) | 8,458 (8,395) | 4,355 (4,315) | 1,860 (1,860) | 1,305 (1,294) | 4,527 (4,493) | 292 (287) | 2.43× | 1,688 |
| Q16 | 1,355 (1,331) | 870 (834) | 386 (356) | 318 (315) | 463 (460) | 412 (410) | 450 (441) | 4.26× | 230 |
| Q17 | 8,049 (8,022) | 11,776 (11,722) | 6,048 (6,024) | 2,246 (2,191) | 1,592 (1,582) | 6,380 (6,341) | 358 (309) | 3.58× | 2,065 |
| Q18 | 29,349 (28,630) | 35,550 (33,442) | 3,613 (3,595) | 2,127 (2,107) | 3,523 (3,508) | 3,902 (3,884) | 2,890 (2,791) | 13.80× | 1,942 |
| Q19 | 3,162 (3,004) | 4,805 (4,721) | 4,471 (4,465) | 2,090 (2,088) | 1,684 (1,589) | 4,557 (4,525) | 495 (371) | 1.51× | 1,987 |
| Q20 | 4,123 (4,109) | 6,157 (6,078) | 4,818 (4,812) | 1,762 (1,751) | 1,333 (1,274) | 4,922 (4,900) | 522 (503) | 2.34× | 1,586 |
| Q21 | 9,330 (9,253) | 15,817 (15,741) | 11,043 (11,013) | 5,066 (5,063) | 5,231 (5,094) | 10,728 (10,705) | 1,661 (1,459) | 1.84× | 4,829 |
| Q22 | 1,349 (1,344) | 1,249 (1,204) | 594 (594) | 541 (540) | 820 (802) | 631 (630) | 338 (301) | 2.49× | 462 |
| **Total** | **154,371** | **205,406** | **91,674** | **40,594** | **38,862** | **94,644** | **17,973** | **geomean 3.01×, total 3.80×** | 37,463 |

### A.2 SF10

| Query | Doris ext. default (#1) | Doris ext. FE split (#2) | Doris+Sirius direct I/O (#4) | Doris+Sirius page cache (#5) | DuckDB CPU (#6) | DuckDB+Sirius (#7) | DuckDB+Sirius, pinned | Doris internal (#3) | Speedup #2/#5 | #5 engine |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 1,667 (1,645) | 659 (658) | 321 (313) | 333 (320) | 231 (230) | 402 (401) | 196 (195) | 280 (279) | 1.98× | 274 |
| Q2 | 1,171 (1,149) | 352 (349) | 330 (327) | 329 (325) | 58 (57) | 165 (163) | 84 (82) | 96 (95) | 1.07× | 136 |
| Q3 | 1,927 (1,901) | 1,821 (1,769) | 295 (293) | 298 (297) | 176 (175) | 278 (276) | 91 (90) | 65 (65) | 6.11× | 225 |
| Q4 | 1,260 (1,230) | 425 (408) | 189 (187) | 192 (192) | 138 (130) | 201 (198) | 99 (94) | 93 (88) | 2.21× | 143 |
| Q5 | 1,574 (1,572) | 1,079 (1,072) | 373 (371) | 380 (374) | 180 (179) | 297 (297) | 95 (82) | 181 (180) | 2.84× | 258 |
| Q6 | 796 (781) | 235 (234) | 213 (212) | 198 (196) | 83 (82) | 224 (224) | 62 (61) | 30 (30) | 1.19× | 161 |
| Q7 | 1,488 (1,476) | 462 (459) | 420 (420) | 425 (420) | 163 (155) | 311 (310) | 79 (79) | 148 (139) | 1.09× | 292 |
| Q8 | 1,775 (1,761) | 585 (580) | 506 (505) | 498 (497) | 208 (206) | 484 (477) | 93 (91) | 207 (196) | 1.17× | 323 |
| Q9 | 1,436 (1,426) | 1,277 (1,253) | 491 (490) | 491 (491) | 392 (385) | 765 (762) | 95 (92) | 259 (251) | 2.60× | 353 |
| Q10 | 1,551 (1,531) | 1,061 (988) | 356 (352) | 361 (360) | 216 (213) | 470 (467) | 84 (84) | 162 (161) | 2.94× | 268 |
| Q11 | 674 (669) | 263 (255) | 318 (307) | 316 (307) | 134 (119) | 179 (176) | 119 (117) | 141 (136) | 0.83× | 106 |
| Q12 | 1,080 (1,065) | 272 (271) | 232 (228) | 232 (229) | 131 (129) | 221 (219) | 65 (65) | 74 (74) | 1.17× | 173 |
| Q13 | 2,434 (2,433) | 336 (329) | 180 (178) | 170 (168) | 245 (241) | 225 (224) | 16 (16) | 147 (144) | 1.98× | 127 |
| Q14 | 612 (605) | 284 (282) | 253 (252) | 260 (255) | 127 (127) | 443 (439) | 50 (49) | 48 (48) | 1.09× | 204 |
| Q15 | 758 (727) | 625 (604) | 300 (290) | 305 (299) | 126 (125) | 415 (413) | 83 (81) | 66 (66) | 2.05× | 217 |
| Q16 | 412 (407) | 333 (330) | 159 (158) | 158 (156) | 75 (74) | 112 (112) | 78 (74) | 108 (108) | 2.11× | 72 |
| Q17 | 992 (954) | 806 (798) | 350 (339) | 336 (334) | 153 (150) | 516 (511) | 106 (103) | 56 (54) | 2.40× | 264 |
| Q18 | 2,501 (2,461) | 2,340 (2,308) | 330 (328) | 332 (332) | 302 (300) | 373 (371) | 156 (139) | 276 (275) | 7.05× | 231 |
| Q19 | 942 (937) | 368 (368) | 293 (293) | 307 (307) | 162 (156) | 474 (473) | 67 (61) | 63 (62) | 1.20× | 233 |
| Q20 | 869 (856) | 502 (501) | 355 (349) | 348 (344) | 128 (126) | 425 (424) | 70 (69) | 114 (111) | 1.44× | 232 |
| Q21 | 1,905 (1,892) | 782 (772) | 793 (732) | 620 (614) | 549 (533) | 871 (871) | 200 (200) | 205 (189) | 1.26× | 479 |
| Q22 | 527 (526) | 162 (159) | 167 (167) | 169 (168) | 77 (73) | 88 (88) | 31 (31) | 74 (72) | 0.96× | 98 |
| **Total** | **28,351** | **15,029** | **7,224** | **7,058** | **4,054** | **7,939** | **2,019** | **2,893** | **geomean 1.77×, total 2.13×** | 4,870 |

### A.3 SF1

| Query | Doris ext. default (#1) | Doris ext. FE split (#2) | Doris+Sirius direct I/O (#4) | Doris+Sirius page cache (#5) | DuckDB CPU (#6) | DuckDB+Sirius (#7) | DuckDB+Sirius, pinned | Doris internal (#3) | Speedup #2/#5 | #5 engine |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 685 (681) | 306 (304) | 90 (89) | 96 (94) | 64 (63) | 130 (128) | 69 (69) | 62 (61) | 3.19× | 40 |
| Q2 | 228 (223) | 226 (223) | 275 (273) | 272 (272) | 33 (33) | 104 (101) | 71 (69) | 48 (48) | 0.83× | 80 |
| Q3 | 499 (478) | 304 (300) | 107 (105) | 109 (108) | 42 (41) | 33 (33) | 19 (19) | 149 (148) | 2.79× | 40 |
| Q4 | 353 (346) | 187 (185) | 72 (70) | 72 (71) | 41 (35) | 58 (58) | 43 (41) | 66 (63) | 2.60× | 28 |
| Q5 | 512 (498) | 308 (302) | 169 (167) | 169 (168) | 42 (40) | 51 (50) | 29 (27) | 191 (190) | 1.82× | 58 |
| Q6 | 187 (185) | 85 (85) | 48 (46) | 49 (49) | 15 (14) | 25 (25) | 16 (16) | 22 (22) | 1.73× | 21 |
| Q7 | 378 (376) | 228 (226) | 191 (187) | 195 (191) | 42 (42) | 53 (52) | 26 (25) | 100 (98) | 1.17× | 63 |
| Q8 | 495 (478) | 314 (304) | 249 (247) | 248 (247) | 43 (42) | 65 (65) | 31 (29) | 133 (133) | 1.27× | 79 |
| Q9 | 437 (414) | 226 (224) | 201 (199) | 202 (201) | 123 (109) | 65 (64) | 29 (28) | 128 (125) | 1.12× | 73 |
| Q10 | 466 (463) | 301 (296) | 149 (149) | 147 (145) | 77 (77) | 56 (56) | 26 (26) | 77 (76) | 2.05× | 59 |
| Q11 | 143 (141) | 136 (131) | 194 (193) | 190 (189) | 33 (32) | 53 (53) | 38 (38) | 58 (55) | 0.72× | 59 |
| Q12 | 341 (326) | 203 (199) | 84 (84) | 85 (84) | 34 (33) | 25 (24) | 9 (9) | 66 (57) | 2.39× | 33 |
| Q13 | 708 (687) | 698 (693) | 72 (70) | 74 (72) | 141 (136) | 30 (30) | 6 (6) | 46 (46) | 9.43× | 32 |
| Q14 | 164 (161) | 91 (90) | 81 (79) | 82 (82) | 28 (28) | 21 (21) | 8 (8) | 32 (32) | 1.11× | 32 |
| Q15 | 196 (192) | 113 (105) | 116 (115) | 119 (117) | 24 (23) | 37 (37) | 24 (22) | 54 (54) | 0.95× | 40 |
| Q16 | 286 (285) | 285 (282) | 117 (113) | 113 (113) | 52 (48) | 80 (78) | 68 (66) | 87 (84) | 2.52× | 37 |
| Q17 | 217 (213) | 122 (121) | 103 (103) | 106 (106) | 22 (21) | 34 (34) | 13 (13) | 32 (32) | 1.15× | 44 |
| Q18 | 249 (238) | 236 (219) | 143 (143) | 143 (140) | 81 (80) | 38 (38) | 17 (16) | 72 (67) | 1.65× | 48 |
| Q19 | 293 (271) | 144 (139) | 109 (108) | 108 (107) | 54 (47) | 30 (29) | 11 (11) | 35 (34) | 1.33× | 41 |
| Q20 | 308 (302) | 196 (193) | 168 (166) | 166 (162) | 42 (42) | 45 (44) | 18 (18) | 54 (53) | 1.18× | 57 |
| Q21 | 785 (784) | 402 (394) | 213 (208) | 210 (206) | 138 (126) | 68 (68) | 30 (30) | 155 (148) | 1.91× | 84 |
| Q22 | 102 (102) | 98 (97) | 111 (107) | 106 (105) | 38 (38) | 35 (34) | 19 (19) | 92 (87) | 0.92× | 37 |
| **Total** | **8,032** | **5,209** | **3,062** | **3,061** | **1,209** | **1,136** | **620** | **1,759** | **geomean 1.63×, total 1.70×** | 1,083 |

## Appendix B. Cold runs (round 1, wall ms)

### B.1 SF100

| Query | #1 Doris ext. default | #2 Doris ext. FE split | #4 Doris+Sirius direct I/O | #5 Doris+Sirius page cache | #6 DuckDB CPU | #7 DuckDB+Sirius | #3 Doris internal |
|---|---|---|---|---|---|---|---|
| Q1 | 5,699 | 8,297 | 3,454 | 6,704 | 10,272 | 3,470 | 4,072 |
| Q2 | 2,027 | 1,240 | 975 | 1,219 | 2,798 | 1,152 | 394 |
| Q3 | 20,615 | 24,543 | 4,528 | 3,244 | 4,368 | 4,618 | 825 |
| Q4 | 4,397 | 6,148 | 2,456 | 1,264 | 1,287 | 2,433 | 512 |
| Q5 | 8,647 | 12,040 | 5,560 | 3,019 | 1,987 | 5,823 | 1,012 |
| Q6 | 2,217 | 4,113 | 2,818 | 1,485 | 785 | 2,906 | 108 |
| Q7 | 4,318 | 6,340 | 6,376 | 2,994 | 1,508 | 6,279 | 426 |
| Q8 | 11,023 | 13,051 | 7,467 | 3,614 | 1,651 | 7,790 | 672 |
| Q9 | 14,445 | 16,906 | 8,324 | 3,670 | 4,227 | 8,596 | 2,744 |
| Q10 | 12,392 | 13,881 | 4,456 | 2,395 | 1,971 | 4,604 | 868 |
| Q11 | 1,122 | 1,412 | 1,005 | 856 | 251 | 1,108 | 389 |
| Q12 | 2,667 | 5,042 | 2,721 | 1,501 | 1,257 | 2,828 | 444 |
| Q13 | 3,880 | 3,933 | 2,647 | 1,878 | 2,541 | 2,435 | 1,917 |
| Q14 | 2,466 | 4,267 | 4,279 | 1,933 | 1,241 | 4,631 | 149 |
| Q15 | 4,576 | 8,405 | 4,372 | 1,995 | 1,280 | 4,563 | 275 |
| Q16 | 1,406 | 894 | 433 | 411 | 476 | 455 | 447 |
| Q17 | 8,029 | 11,722 | 5,948 | 2,386 | 1,595 | 6,374 | 318 |
| Q18 | 35,997 | 36,503 | 3,636 | 2,180 | 3,682 | 3,901 | 2,928 |
| Q19 | 3,025 | 4,685 | 4,459 | 2,205 | 1,745 | 4,530 | 369 |
| Q20 | 4,163 | 6,239 | 4,828 | 1,765 | 1,283 | 4,924 | 506 |
| Q21 | 9,069 | 15,879 | 11,519 | 8,184 | 5,031 | 10,735 | 1,626 |
| Q22 | 1,375 | 1,249 | 629 | 555 | 744 | 638 | 305 |
| **Total** | 163,555 | 206,789 | 92,890 | 55,457 | 51,980 | 94,793 | 21,306 |

### B.2 SF10

| Query | #1 | #2 | #4 | #5 | #6 | #7 | #7 pinned | #3 |
|---|---|---|---|---|---|---|---|---|
| Q1 | 1,982 | 845 | 948 | 1,319 | 623 | 805 | 362 | 616 |
| Q2 | 1,658 | 423 | 423 | 452 | 341 | 1,645 | 177 | 334 |
| Q3 | 2,171 | 1,751 | 319 | 471 | 258 | 286 | 107 | 144 |
| Q4 | 1,359 | 469 | 286 | 286 | 136 | 248 | 139 | 144 |
| Q5 | 1,574 | 1,098 | 390 | 473 | 204 | 301 | 92 | 239 |
| Q6 | 789 | 232 | 246 | 233 | 82 | 237 | 83 | 33 |
| Q7 | 1,498 | 459 | 438 | 433 | 161 | 325 | 90 | 167 |
| Q8 | 1,860 | 695 | 546 | 644 | 208 | 375 | 99 | 233 |
| Q9 | 1,413 | 1,309 | 498 | 497 | 387 | 692 | 100 | 299 |
| Q10 | 1,571 | 1,076 | 374 | 413 | 229 | 481 | 105 | 212 |
| Q11 | 695 | 271 | 336 | 351 | 123 | 819 | 146 | 138 |
| Q12 | 1,079 | 285 | 271 | 269 | 134 | 220 | 74 | 91 |
| Q13 | 3,279 | 382 | 193 | 269 | 250 | 593 | 16 | 252 |
| Q14 | 616 | 289 | 262 | 262 | 127 | 243 | 52 | 49 |
| Q15 | 757 | 603 | 325 | 326 | 128 | 251 | 92 | 74 |
| Q16 | 419 | 339 | 236 | 230 | 77 | 178 | 139 | 128 |
| Q17 | 992 | 814 | 359 | 327 | 153 | 362 | 108 | 64 |
| Q18 | 2,320 | 2,414 | 348 | 350 | 295 | 307 | 137 | 313 |
| Q19 | 947 | 378 | 294 | 314 | 158 | 242 | 60 | 92 |
| Q20 | 888 | 502 | 360 | 347 | 126 | 265 | 72 | 123 |
| Q21 | 1,960 | 805 | 656 | 644 | 524 | 764 | 210 | 237 |
| Q22 | 540 | 158 | 189 | 193 | 71 | 102 | 52 | 77 |
| **Total** | 30,367 | 15,597 | 8,297 | 9,103 | 4,795 | 9,741 | 2,512 | 4,059 |

### B.3 SF1

| Query | #1 | #2 | #4 | #5 | #6 | #7 | #7 pinned | #3 |
|---|---|---|---|---|---|---|---|---|
| Q1 | 842 | 470 | 684 | 793 | 123 | 482 | 199 | 252 |
| Q2 | 397 | 263 | 359 | 374 | 52 | 223 | 163 | 122 |
| Q3 | 502 | 326 | 111 | 118 | 60 | 38 | 33 | 165 |
| Q4 | 378 | 201 | 170 | 181 | 44 | 123 | 89 | 75 |
| Q5 | 529 | 317 | 194 | 202 | 51 | 66 | 37 | 195 |
| Q6 | 189 | 93 | 84 | 89 | 20 | 45 | 36 | 24 |
| Q7 | 409 | 235 | 211 | 216 | 44 | 67 | 35 | 108 |
| Q8 | 488 | 318 | 293 | 301 | 56 | 84 | 39 | 142 |
| Q9 | 441 | 233 | 220 | 225 | 113 | 75 | 36 | 114 |
| Q10 | 490 | 303 | 172 | 175 | 93 | 73 | 38 | 85 |
| Q11 | 146 | 140 | 215 | 215 | 38 | 90 | 64 | 58 |
| Q12 | 330 | 199 | 131 | 126 | 47 | 26 | 11 | 66 |
| Q13 | 763 | 757 | 91 | 99 | 146 | 47 | 6 | 92 |
| Q14 | 160 | 93 | 82 | 86 | 35 | 22 | 8 | 34 |
| Q15 | 208 | 108 | 142 | 146 | 24 | 52 | 38 | 55 |
| Q16 | 289 | 288 | 213 | 212 | 46 | 160 | 145 | 100 |
| Q17 | 234 | 127 | 106 | 109 | 31 | 36 | 15 | 38 |
| Q18 | 234 | 268 | 155 | 158 | 77 | 39 | 16 | 91 |
| Q19 | 278 | 143 | 108 | 115 | 52 | 28 | 12 | 38 |
| Q20 | 305 | 200 | 163 | 169 | 51 | 46 | 19 | 60 |
| Q21 | 794 | 407 | 249 | 254 | 136 | 79 | 42 | 166 |
| Q22 | 101 | 101 | 132 | 134 | 40 | 51 | 38 | 102 |
| **Total** | 8,507 | 5,590 | 4,285 | 4,497 | 1,379 | 1,952 | 1,119 | 2,182 |

## Appendix C. SF100 FE audit split (hot medians, ms)

From the FE's audit log, joined to each query's client window: `fe` = FE end-to-end, `plan` = planning, `rpc1` = the first fragment-execution RPC. For the Sirius backend `rpc1` contains the whole execution (the query runs inside that RPC); for the native BE it is only the dispatch.

| Query | #1 Doris ext. default fe / plan / rpc1 | #2 Doris ext. FE split | #4 Doris+Sirius direct I/O | #5 Doris+Sirius page cache | #3 Doris internal |
|---|---|---|---|---|---|
| Q1 | 5,326 / 102 / 5 | 7,942 / 171 / 8 | 2,665 / 40 / 2,585 | 2,247 / 40 / 2,203 | 2,632 / 2 / 8 |
| Q2 | 1,801 / 65 / 19 | 1,090 / 73 / 23 | 1,012 / 47 / 968 | 691 / 46 / 642 | 115 / 10 / 16 |
| Q3 | 20,257 / 141 / 10 | 24,017 / 158 / 18 | 4,511 / 70 / 4,439 | 1,809 / 69 / 1,732 | 372 / 5 / 13 |
| Q4 | 3,347 / 195 / 8 | 5,862 / 136 / 14 | 2,371 / 66 / 2,305 | 1,151 / 67 / 1,082 | 160 / 3 / 13 |
| Q5 | 8,709 / 159 / 15 | 12,135 / 163 / 22 | 5,639 / 89 / 5,552 | 2,001 / 83 / 1,916 | 498 / 18 / 20 |
| Q6 | 2,034 / 156 / 4 | 4,112 / 154 / 6 | 2,821 / 57 / 2,758 | 1,433 / 57 / 1,373 | 79 / 3 / 4 |
| Q7 | 4,164 / 157 / 16 | 6,377 / 159 / 21 | 6,340 / 91 / 6,247 | 2,548 / 80 / 2,467 | 382 / 19 / 24 |
| Q8 | 11,046 / 166 / 18 | 13,019 / 170 / 29 | 7,447 / 92 / 7,362 | 2,560 / 95 / 2,464 | 469 / 37 / 34 |
| Q9 | 14,664 / 161 / 15 | 17,028 / 183 / 27 | 8,372 / 84 / 8,289 | 3,083 / 90 / 2,996 | 2,539 / 19 / 20 |
| Q10 | 11,524 / 147 / 10 | 14,423 / 174 / 18 | 4,333 / 71 / 4,260 | 1,954 / 61 / 1,891 | 743 / 7 / 15 |
| Q11 | 1,140 / 50 / 16 | 1,396 / 52 / 21 | 1,036 / 33 / 999 | 554 / 32 / 518 | 367 / 7 / 19 |
| Q12 | 2,768 / 176 / 8 | 5,059 / 174 / 14 | 2,712 / 68 / 2,641 | 1,563 / 73 / 1,487 | 416 / 4 / 13 |
| Q13 | 3,750 / 31 / 8 | 3,827 / 35 / 13 | 2,660 / 17 / 2,639 | 1,039 / 20 / 1,017 | 1,881 / 4 / 15 |
| Q14 | 2,416 / 143 / 6 | 4,259 / 124 / 11 | 4,258 / 65 / 4,187 | 1,781 / 68 / 1,711 | 141 / 4 / 9 |
| Q15 | 4,509 / 251 / 9 | 8,445 / 230 / 16 | 4,339 / 104 / 4,233 | 1,848 / 114 / 1,736 | 280 / 7 / 17 |
| Q16 | 1,331 / 29 / 10 | 845 / 30 / 16 | 362 / 20 / 336 | 296 / 19 / 270 | 428 / 5 / 18 |
| Q17 | 8,036 / 222 / 9 | 11,763 / 246 / 19 | 6,036 / 107 / 5,916 | 2,233 / 131 / 2,100 | 346 / 5 / 12 |
| Q18 | 29,336 / 242 / 12 | 35,538 / 294 / 26 | 3,601 / 96 / 3,505 | 2,114 / 110 / 2,002 | 2,877 / 8 / 17 |
| Q19 | 3,148 / 178 / 7 | 4,790 / 192 / 12 | 4,459 / 67 / 4,387 | 2,078 / 61 / 2,028 | 482 / 8 / 9 |
| Q20 | 4,103 / 139 / 15 | 6,137 / 166 / 25 | 4,797 / 79 / 4,721 | 1,738 / 72 / 1,661 | 503 / 8 / 22 |
| Q21 | 9,317 / 356 / 16 | 15,803 / 378 / 31 | 11,031 / 143 / 10,860 | 5,054 / 148 / 4,911 | 1,649 / 23 / 21 |
| Q22 | 1,336 / 42 / 13 | 1,236 / 45 / 19 | 582 / 27 / 552 | 528 / 27 / 501 | 325 / 8 / 20 |

## Appendix D. SF100 resources (hot medians)

Per query: peak resident memory of the backend process (MiB) / bytes read from disk during the query (MiB; 0 on a page-cache hit, direct I/O always counts) / CPU cores busy / GPU memory in use (MiB; the pool reserved up front, not a peak) / GPU busy % / GPU memory-controller busy %. Sampled every 0.5 s, so sub-second queries are approximate; `-` where the query was shorter than one sample.

| Query | #1 Doris ext. default | #2 Doris ext. FE split | #4 Doris+Sirius direct I/O | #5 Doris+Sirius page cache | #6 DuckDB CPU | #7 DuckDB+Sirius | #3 Doris internal |
|---|---|---|---|---|---|---|---|
| Q1 | 7,936 / 0 / 26.3 / - / - / - | 28,680 / 0 / 27.7 / - / - / - | 11,816 / 6,234 / 1.7 / 41,421 / 47 / 24 | 11,848 / 0 / 2.2 / 41,421 / 62 / 26 | 2,377 / 0 / 22.0 / - / - / - | 1,956 / 7,296 / 2.6 / 41,383 / 32 / 5 | 42,819 / 0 / 26.2 / - / - / - |
| Q2 | 7,931 / 0 / 10.4 / - / - / - | 27,765 / 0 / 20.1 / - / - / - | 11,816 / 2,603 / 0.9 / 41,421 / 8 / 0 | 11,869 / 0 / 1.7 / 41,421 / 24 / 4 | 2,377 / 0 / 17.6 / - / - / - | 1,956 / 2,431 / 1.1 / 41,391 / 30 / 0 | 42,819 / 0 / 16.2 / - / - / - |
| Q3 | 21,517 / 0 / 6.0 / - / - / - | 38,595 / 0 / 8.6 / - / - / - | 11,874 / 9,234 / 0.8 / 41,421 / 17 / 1 | 11,918 / 0 / 2.0 / 41,421 / 61 / 10 | 2,909 / 0 / 24.1 / - / - / - | 2,104 / 9,162 / 1.8 / 41,391 / 21 / 1 | 43,019 / 0 / 18.5 / - / - / - |
| Q4 | 22,039 / 0 / 21.3 / - / - / - | 41,147 / 0 / 26.7 / - / - / - | 11,874 / 5,025 / 1.1 / 41,421 / 18 / 6 | 11,918 / 0 / 2.2 / 41,421 / 34 / 20 | 3,350 / 0 / 24.5 / - / - / - | 2,095 / 4,977 / 1.9 / 41,393 / 23 / 1 | 43,440 / 0 / 21.5 / - / - / - |
| Q5 | 22,039 / 0 / 10.6 / - / - / - | 41,147 / 0 / 14.7 / - / - / - | 12,136 / 11,374 / 0.7 / 41,421 / 24 / 2 | 11,963 / 0 / 2.0 / 41,421 / 59 / 11 | 3,482 / 0 / 25.4 / - / - / - | 2,101 / 11,952 / 1.6 / 41,393 / 19 / 2 | 43,440 / 0 / 22.1 / - / - / - |
| Q6 | 12,244 / 0 / 22.3 / - / - / - | 28,070 / 0 / 26.8 / - / - / - | 12,136 / 5,756 / 1.1 / 41,421 / 13 / 0 | 11,963 / 0 / 2.1 / 41,421 / 41 / 10 | 3,482 / 0 / 21.0 / - / - / - | 2,101 / 5,756 / 1.9 / 41,393 / 24 / 6 | 43,440 / 0 / 22.4 / - / - / - |
| Q7 | 11,037 / 0 / 22.9 / - / - / - | 27,772 / 0 / 26.3 / - / - / - | 11,895 / 12,277 / 0.9 / 41,421 / 18 / 2 | 11,986 / 0 / 2.3 / 41,421 / 58 / 15 | 4,336 / 0 / 25.1 / - / - / - | 2,368 / 12,329 / 0.9 / 41,393 / 16 / 3 | 43,458 / 0 / 21.8 / - / - / - |
| Q8 | 47,055 / 0 / 23.1 / - / - / - | 70,826 / 0 / 25.9 / - / - / - | 12,034 / 15,469 / 0.7 / 41,421 / 26 / 4 | 11,931 / 0 / 1.8 / 41,421 / 57 / 11 | 4,336 / 0 / 25.7 / - / - / - | 2,198 / 15,230 / 0.7 / 41,393 / 18 / 0 | 43,618 / 0 / 23.3 / - / - / - |
| Q9 | 61,654 / 0 / 23.4 / - / - / - | 87,241 / 0 / 24.8 / - / - / - | 12,064 / 16,701 / 0.8 / 41,421 / 25 / 4 | 12,013 / 0 / 2.1 / 41,421 / 65 / 14 | 16,062 / 0 / 25.9 / - / - / - | 5,629 / 15,699 / 1.8 / 41,401 / 23 / 4 | 48,379 / 0 / 23.4 / - / - / - |
| Q10 | 46,750 / 0 / 9.4 / - / - / - | 64,457 / 0 / 14.2 / - / - / - | 12,064 / 10,691 / 1.0 / 41,421 / 25 / 5 | 12,009 / 0 / 2.2 / 41,421 / 42 / 12 | 16,052 / 0 / 20.9 / - / - / - | 5,663 / 10,169 / 1.9 / 41,403 / 21 / 1 | 47,904 / 0 / 20.8 / - / - / - |
| Q11 | 11,531 / 0 / 10.4 / - / - / - | 14,523 / 0 / 19.9 / - / - / - | 11,810 / 2,346 / 0.5 / 41,421 / 34 / 9 | 11,848 / 0 / 1.1 / 41,421 / 50 / 1 | 5,418 / 0 / 16.6 / - / - / - | 5,663 / 2,600 / 1.4 / 41,405 / 22 / 0 | 47,520 / 0 / 22.9 / - / - / - |
| Q12 | 11,073 / 0 / 21.3 / - / - / - | 30,396 / 0 / 28.3 / - / - / - | 11,814 / 5,466 / 1.4 / 41,421 / 14 / 2 | 11,907 / 0 / 2.1 / 41,421 / 39 / 10 | 5,418 / 0 / 25.7 / - / - / - | 5,683 / 6,534 / 1.4 / 41,405 / 19 / 4 | 46,910 / 0 / 25.6 / - / - / - |
| Q13 | 7,497 / 0 / 21.7 / - / - / - | 27,906 / 0 / 26.3 / - / - / - | 11,814 / 4,702 / 0.6 / 41,421 / 15 / 0 | 11,907 / 0 / 1.6 / 41,421 / 83 / 17 | 8,227 / 0 / 29.1 / - / - / - | 5,683 / 5,171 / 1.3 / 41,405 / 26 / 4 | 46,442 / 0 / 25.9 / - / - / - |
| Q14 | 8,738 / 0 / 22.4 / - / - / - | 24,705 / 0 / 26.1 / - / - / - | 11,941 / 9,082 / 0.8 / 41,421 / 25 / 4 | 11,914 / 0 / 2.0 / 41,421 / 44 / 8 | 8,227 / 0 / 21.1 / - / - / - | 5,644 / 9,709 / 1.6 / 41,405 / 18 / 3 | 44,476 / 0 / 18.3 / - / - / - |
| Q15 | 12,252 / 0 / 25.6 / - / - / - | 41,641 / 0 / 29.0 / - / - / - | 11,941 / 9,823 / 0.8 / 41,421 / 23 / 2 | 12,023 / 0 / 1.7 / 41,421 / 41 / 7 | 5,837 / 0 / 22.2 / - / - / - | 5,948 / 9,551 / 1.5 / 41,409 / 22 / 4 | 43,760 / 0 / 19.5 / - / - / - |
| Q16 | 11,758 / 0 / 7.4 / - / - / - | 41,462 / 0 / 18.1 / - / - / - | 11,840 / 1,216 / 0.8 / 41,421 / 57 / 10 | 11,890 / 0 / 1.0 / 41,421 / 13 / 1 | 5,332 / 0 / 24.1 / - / - / - | 5,642 / 843 / 1.0 / 41,409 / 43 / 12 | 43,760 / 0 / 20.6 / - / - / - |
| Q17 | 13,856 / 0 / 25.5 / - / - / - | 42,309 / 0 / 28.4 / - / - / - | 11,957 / 12,209 / 0.7 / 41,421 / 21 / 2 | 12,080 / 0 / 1.9 / 41,421 / 46 / 10 | 5,332 / 0 / 21.5 / - / - / - | 6,204 / 12,897 / 1.6 / 41,409 / 17 / 3 | 44,284 / 0 / 22.5 / - / - / - |
| Q18 | 37,433 / 0 / 8.0 / - / - / - | 75,932 / 0 / 10.9 / - / - / - | 11,957 / 7,342 / 1.4 / 41,421 / 36 / 15 | 12,064 / 0 / 2.1 / 41,421 / 60 / 18 | 11,133 / 0 / 25.8 / - / - / - | 5,985 / 7,242 / 2.1 / 41,409 / 27 / 9 | 51,588 / 0 / 27.9 / - / - / - |
| Q19 | 24,012 / 0 / 18.4 / - / - / - | 42,016 / 0 / 27.9 / - / - / - | 11,830 / 8,836 / 1.4 / 41,421 / 28 / 9 | 12,064 / 0 / 2.4 / 41,421 / 45 / 9 | 5,680 / 0 / 20.7 / - / - / - | 5,923 / 9,651 / 1.4 / 41,409 / 25 / 4 | 51,588 / 0 / 20.5 / - / - / - |
| Q20 | 21,352 / 0 / 25.2 / - / - / - | 42,271 / 0 / 27.5 / - / - / - | 11,876 / 9,451 / 0.7 / 41,421 / 23 / 3 | 11,917 / 0 / 1.7 / 41,421 / 59 / 6 | 5,015 / 0 / 18.5 / - / - / - | 5,776 / 10,350 / 1.6 / 41,409 / 22 / 4 | 50,674 / 0 / 21.0 / - / - / - |
| Q21 | 17,790 / 0 / 25.2 / - / - / - | 61,396 / 0 / 28.9 / - / - / - | 12,071 / 20,514 / 1.1 / 41,421 / 31 / 7 | 12,472 / 0 / 2.6 / 41,421 / 74 / 21 | 7,548 / 0 / 24.4 / - / - / - | 7,024 / 20,093 / 1.6 / 41,409 / 21 / 4 | 49,651 / 0 / 24.7 / - / - / - |
| Q22 | 14,149 / 0 / 11.6 / - / - / - | 55,820 / 0 / 21.4 / - / - / - | 11,872 / 1,546 / 0.7 / 41,421 / 56 / 9 | 11,908 / 0 / 1.6 / 41,421 / 0 / 0 | 5,381 / 0 / 19.1 / - / - / - | 6,112 / 1,962 / 1.2 / 41,411 / 4 / 0 | 46,090 / 0 / 16.6 / - / - / - |

## Appendix E. Environment snapshot and versions

Recorded by the harness at the start of every run (`env.txt`, `variables.txt`).

| Field | Value |
|---|---|
| Host | `ip-172-31-26-244`, `instance_type: g6e.8xlarge`, kernel `7.0.0-1012-aws` |
| CPU / memory | AMD EPYC 7R13 × 32, `mem_total_kib: 260437876` |
| GPU | NVIDIA L40S, 46,068 MiB, driver 580.178.04 |
| Data | `/mnt/nvme/tpch_parquet_sf{1,10,100}` on `/dev/md0` (ext4); SF100 baseline `/mnt/nvme/expected-sf100`; SF1/SF10 baselines in the repository under `experimental/doris/tests/expected/` |
| Rounds | 4 per scenario, page cache evicted before round 1 |
| Doris | `doris-4.1.4-rc04-ad35a140c7f` (FE and native BE) |
| DuckDB shell | `build/release/duckdb`, `v1.5.5 (Variegata) 3ff87f1e` |
| Sirius backend | `sirius-doris-be/0.1.0`, registered on heartbeat port 9050; the native BE on 9150 |
| Repository commit | Scenarios 4 and 5 (Doris + Sirius): `75606ff5` on the `experimental-doris` branch of the `morningman/sirius` fork — upstream `dev` `e0080cd6` plus the changes of sirius-db/sirius #1840, #1841 and #1842. Scenarios 1, 2, 3, 6, 7: `f5116c6c` on the same branch. Between the two, the engine source changed in one file only — the FFI entry point (`src/sirius_ffi.cpp`), which those scenarios do not execute; the other commits in between are harness and documentation changes |
| Submodule pins | `duckdb` `3ff87f1e` (v1.5.5), `cucascade` `e9929fff`, `substrait` `a7e045be`, as recorded by the repository at both commits |
| Configuration hashes (SHA-256) | `conf/be.conf` `ef763a6b…`, `conf/sirius-bench.yaml` `d7dad097…`, `conf/fe.conf` `da36e00e…`, `sql/session.sql` `ee2c4c64…`, `sql/session-native.sql` `eb618adf…`, `sql/tpch-views.sql` `6ab7149b…`; the effective `sirius.yaml` of each GPU run is stored with the run. The hashes are of the files at the commits above; the header comments of `conf/be.conf`, `conf/sirius-bench.yaml` and `sql/session-native.sql` were edited afterwards (references to a planning workspace replaced by references to this report) without changing any setting |

**Engine version note.** The engine binary is upstream Sirius `dev` (at the fork point, `e0080cd6`) plus three changes submitted upstream as sirius-db/sirius #1840 (items 1 and 2) and #1841 (item 3); they are required to reproduce these results:

1. *Hash join:* a mark join (used for null-aware anti joins, e.g. the `NOT IN` subquery of Q16) that is polled by the scheduler before its partitions are sized now reports that it is waiting for its build side instead of raising an exception that terminates the process.
2. *Comparison join:* in a mixed join (equality plus inequality conditions), inequality operands that the GPU expression evaluator cannot evaluate inline — decimal casts and decimal arithmetic, as in the correlated subqueries of Q17 and Q20 — are materialised as projected columns before the join.
3. *FFI entry point:* DuckDB's `parquet_metadata_cache` is enabled database-wide when the engine is brought up, so that lowering a Substrait plan does not re-parse parquet footers at every relation level (the SF100 `lineitem` file has 5,232 row groups). The DuckDB-side scenarios (6, 7) do not go through this entry point.

**Global session variables — Doris scenarios (4.1.4 defaults):** `parallel_pipeline_task_num 0`, `enable_local_shuffle true`, `enable_parallel_result_sink true`, `runtime_filter_mode GLOBAL`, `enable_cte_materialize true`, `topn_lazy_materialization_threshold 1024`, `file_split_size 0`, `max_file_split_size 64 MB`, `max_initial_file_split_size 32 MB`, `file_split_size_on_be 64 MB` (scenario 2: `0`), `file_split_size_on_fe 512 MB`, `max_file_scanners_concurrency 16`, `enable_file_scanner_v2 true`, `enable_fold_constant_by_be false`, `enable_profile false`, `query_timeout 3600`.

**Global session variables — Doris + Sirius scenarios:** `parallel_pipeline_task_num 1`, `enable_local_shuffle false`, `enable_parallel_result_sink false`, `runtime_filter_mode OFF`, `enable_cte_materialize false`, `topn_lazy_materialization_threshold -1`, `file_split_size 1 TB`, `file_split_size_on_be 0`, `file_split_size_on_fe 1 TB`; the remaining variables as above.

In every scenario the FE's SQL result cache is disabled (`enable_sql_cache = false`, set by each session file), so no warm round is served from a cached result.

**Backend configuration.** Native BE: `mem_limit` 70 % of RAM, `user_files_secure_path = /`, storage root on the NVMe, ports 9150/9160/8140/8160. Sirius backend: GPU pool `usage_limit_fraction 0.9`, host tier `capacity_bytes 160Gi`, disk tier on the NVMe, `use_odirect true` (scenario 4) / `false` (scenario 5), telemetry on. FE: `remote_fragment_exec_timeout_ms 600000`.

## Appendix F. Reproduction

Everything lives under `experimental/doris/` in the repository; one harness runs all scenarios of a scale and writes the tables in this report.

```bash
cd experimental/doris
pixi run -e fe fe-fetch && pixi run -e fe fe-start         # official FE 4.1.4
pixi run bash scripts/fetch-be.sh                           # official BE 4.1.4 → .doris-be/be
# engine: pixi run make at the repository root (CUDA 13 + RAPIDS from the root pixi env)
# data:   tpchgen-cli -s <SF> --format=parquet --parts=1 → /mnt/nvme/tpch_parquet_sf<SF>/<table>/part.0.parquet

# DuckDB baseline (SF1/SF10 are in the repository; SF100 is generated onto the NVMe)
pixi run -e check python scripts/validate_tpch_results.py expected \
    --data /mnt/nvme/tpch_parquet_sf100 --out /mnt/nvme/expected-sf100

# all scenarios, 1 cold + 3 hot rounds, every round validated, report written at the end
pixi run bash scripts/bench-all.sh --data /mnt/nvme/tpch_parquet_sf100 --expected /mnt/nvme/expected-sf100 \
    --rounds 4 --host-capacity 160Gi --baseline native --load-olap \
    --price native=4.529 --price sirius-buffered=4.529
pixi run bash scripts/bench-all.sh --data /mnt/nvme/tpch_parquet_sf10 --rounds 4 --load-olap \
    --price native-split=4.529 --price sirius-buffered=4.529
```

Scenario names accepted by the harness: `native` (#1), `native-split` (#2), `native-olap` (#3), `sirius` (#4), `sirius-buffered` (#5), `duckdb` (#6), `duckdb-gpu` (#7), `duckdb-gpu-pinned` (tables pinned in GPU memory). Each run directory holds `env.txt`, `variables.txt`, the effective `sirius.yaml`, per-round `result.tsv` / `explain.txt` / `timings.csv` / `summary.csv` / `samples.csv`, the engine telemetry, and `rounds.csv`; `scripts/bench-report.py report` regenerates the tables from any set of run directories. The run directories behind this report are kept on the test machine and are not part of the repository.
