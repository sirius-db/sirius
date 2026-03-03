# ClickBench Optimization Analysis — Sirius GPU Engine

> Author context: GH200 (sm_90a, HBM3), DuckDB v1.2.1, libcudf 25.12, 100M row hits table
> Dev machine: RTX 6000 (sm_75, 24GB GDDR6), benchmarking on 10M rows
> Branch: `cudf-25.12-optimization` (forked from `cudf-25.12`)
> This document is an iterative engineering ledger. Update with benchmark numbers as experiments land.

---

## 1. Benchmark Overview

**Total benchmark time (GH200, 100M rows): 1.415s** across 43 queries (Q0–Q42).

### Full Query Breakdown (ranked by runtime)

```
Rank  Q#   Time(s)  %Total  Cumul%  Query Pattern
───── ──── ──────── ─────── ─────── ──────────────────────────────────────────────
  1   Q28   0.267   18.9%   18.9%   REGEXP_REPLACE(Referer) GROUP BY domain    [CPU FALLBACK]
  2   Q27   0.166   11.7%   30.6%   AVG(STRLEN(URL)) GROUP BY CounterID HAVING
  3   Q13   0.122    8.6%   39.2%   COUNT(DISTINCT UserID) GROUP BY SearchPhrase
  4   Q9    0.089    6.3%   45.5%   COUNT(DISTINCT UserID) GROUP BY RegionID +multi-agg
  5   Q8    0.084    5.9%   51.4%   COUNT(DISTINCT UserID) GROUP BY RegionID
  6   Q23   0.066    4.7%   56.1%   SELECT * WHERE URL LIKE '%google%' ORDER BY LIMIT 10
  7   Q32   0.063    4.5%   60.6%   GROUP BY WatchID,ClientIP ORDER BY COUNT LIMIT 10
  8   Q34   0.057    4.0%   64.6%   GROUP BY 1,URL ORDER BY COUNT LIMIT 10
  9   Q33   0.054    3.8%   68.4%   GROUP BY URL ORDER BY COUNT LIMIT 10
 10   Q18   0.045    3.2%   71.6%   GROUP BY UserID,minute(EventTime),SearchPhrase
 11   Q20   0.041    2.9%   74.5%   COUNT(*) WHERE URL LIKE '%google%'
 12   Q29   0.033    2.3%   76.8%   90× SUM(ResolutionWidth + k)
 13   Q22   0.031    2.2%   79.0%   GROUP BY SearchPhrase WHERE Title LIKE '%Google%' + COUNT DISTINCT
 14   Q16   0.028    2.0%   81.0%   GROUP BY UserID,SearchPhrase ORDER BY COUNT LIMIT 10
 15   Q17   0.025    1.8%   82.8%   GROUP BY UserID,SearchPhrase LIMIT 10 (no ORDER BY)
 16   Q35   0.020    1.4%   84.2%   GROUP BY ClientIP variants
 17   Q10   0.016    1.1%   85.3%   COUNT(DISTINCT UserID) GROUP BY MobilePhoneModel
 18   Q11   0.016    1.1%   86.4%   COUNT(DISTINCT UserID) GROUP BY MobilePhone,MobilePhoneModel
 19   Q14   0.016    1.1%   87.6%   COUNT(*) GROUP BY SearchEngineID,SearchPhrase
 20   Q12   0.015    1.1%   88.6%   COUNT(*) GROUP BY SearchPhrase (no ORDER BY)
 21   Q31   0.015    1.1%   89.7%   GROUP BY WatchID,ClientIP WHERE SearchPhrase<>''
─── below this line: each query < 1% of total (< 0.013s), 33 queries total 0.148s ───
 22+  (22 queries each ≤ 0.013s)              100.0%
```

**Key insight**: Top 5 queries = **0.728s = 51.4%** of total. Top 11 queries = **1.054s = 74.5%**.

### Time by Operation Category

```
Category                           Time(s)  %Total   Queries
──────────────────────────────── ──────── ─────── ──────────────────────
REGEXP_REPLACE (CPU fallback)      0.267   18.9%   Q28
COUNT DISTINCT (grouped)           0.358   25.3%   Q8,Q9,Q10,Q11,Q13,Q22
STRLEN(URL) + GroupBy              0.166   11.7%   Q27
String GROUP BY                    0.240   17.0%   Q12,Q14,Q16,Q17,Q18,Q33,Q34
GROUP BY unique key (WatchID)      0.078    5.5%   Q31,Q32
LIKE substring scan                0.120    8.5%   Q20,Q21,Q23 (Q23 overlap)
SELECT * wide row + LIKE           0.066    4.7%   Q23
Integer GROUP BY + simple aggs     0.109    7.7%   Q7,Q15,Q29,Q35,Q36-Q42
Simple scan/filter/agg             0.020    1.4%   Q0-Q3,Q6,Q19
ORDER BY + LIMIT (fast)            0.040    2.8%   Q24-Q26,Q30
```

---

## 2. Transferability Analysis: RTX 6000 (10M rows) → GH200 (100M rows)

### Can we verify optimizations on RTX 6000 with 10M rows?

**YES** — with caveats. Here is the per-optimization analysis:

| Optimization | Transferable? | Why |
|---|---|---|
| P1: COUNT DISTINCT two-phase | **YES** | Algorithmic change — same code path runs on both. 10M rows has same column types and query structure. The 5.7× dedup reduction (100M→17.6M for Q8) will be proportionally similar at 10M (~1.76M distinct pairs). Speedup ratio transfers directly. |
| P2: GPU REGEXP_REPLACE | **YES** | Currently 100% CPU fallback. Moving to GPU is a binary capability change — works or doesn't. Any GPU > CPU improvement at 10M will only be larger at 100M (GPU regex scales better with data size due to parallelism). |
| P3: STRLEN offset arith. | **YES** | Pure memory access pattern change (10.5 GB char buffer → 400 MB offset array). At 10M rows: ~1 GB chars → ~40 MB offsets. Same 25× memory reduction ratio. Effect is proportional. |
| P4: String GROUP BY fingerprint | **YES** | Algorithmic change (string hash → int groupby). Same code path at any scale. |
| P5: Top-K malloc elim. | **YES** | Fixed overhead elimination (~2ms). At 10M rows queries are faster overall, so the 2ms is a larger fraction — easier to measure. |
| P6: Late materialization | **YES** | Same pipeline restructuring. Selectivity (0.016%) is a property of the data, not the scale. |

**All optimizations are algorithmic or pipeline changes, not hardware-specific tuning.** They modify
the GPU code path (which kernels run, what data they read). These changes are identical on RTX 6000
and GH200 — the GPU just happens to be slower per operation.

### What won't transfer?

| Concern | Impact |
|---|---|
| Absolute times won't match | RTX 6000 has ~1/3 bandwidth of GH200 HBM3, so everything is ~3× slower. But **ratios** (before/after) transfer. |
| 10M vs 100M data distribution | Profiled both. Key differences at scale (CounterID 7→6506) only matter for Q27's HAVING filter. For the optimizations we're implementing, the code path is the same regardless of cardinality. |
| VRAM pressure at 100M rows | At 10M rows we have headroom; at 100M rows string columns alone fill 24GB. This is a separate concern from code optimization — it's a memory management issue on GH200's 96GB. Our optimizations don't change memory footprint. |
| GPU kernel occupancy / SM count | RTX 6000 has 36 SMs vs GH200's 132 SMs. Kernel-level parallelism is lower on RTX 6000, which means latency-bound operations (like the Top-K fixed overhead) are proportionally the same, while throughput-bound operations (like the regex scan) take longer. Speedup ratios should be within 0.8×–1.2× of GH200's ratios. |

### Verification methodology for RTX 6000

**For each optimization, measure before/after using runner.py methodology on 10M rows:**

```bash
# Load 10M rows into clickbench.duckdb
# Run: .timer on, 3 runs (skip first 2), take min of warm runs
# Report: speedup ratio = old_time / new_time for each affected query
```

A speedup ratio of 2× on RTX 6000 at 10M rows should translate to ~1.6–2.4× on GH200 at 100M rows.
The ratio won't be exact, but the direction and order-of-magnitude will be correct.

**What to watch for**: If an optimization shows <1.1× speedup at 10M rows, it may be that the
operation is too fast at 10M to measure (e.g., already <1ms). In that case, the optimization
is still valid for 100M but we can't verify the magnitude on RTX 6000. This is unlikely for
our P1–P3 targets since even at 10M, COUNT DISTINCT and REGEXP_REPLACE should be measurable.

---

## 3. Data Characteristics (Profiled at actual 100M rows)

| Column | Distinct (100M) | Total Data | Non-Empty % | Notes |
|--------|-----------------|------------|-------------|-------|
| URL | 18.3M | 10.5 GB | ~100% | avg ~105 bytes |
| Referer | ~18M | 8.6 GB | 81% (81M rows) | avg ~106 bytes |
| SearchPhrase | **6.0M** | ~2.5 GB | **13.2% (13.2M rows)** | avg ~190 bytes |
| CounterID | **6,506** | 400 MB | 100% | **100 groups pass HAVING >100000** |
| RegionID | **9,040** | 400 MB | 100% | medium cardinality |
| UserID | **17.6M** | 800 MB | 100% | |
| WatchID | ~100M | 800 MB | 100% | **~100% unique, confirmed** |
| ClientIP | ~30M | 400 MB | 100% | estimated |

**Critical corrections from 1M→100M scale**:
- CounterID: 7 → **6,506** distinct (100 groups survive HAVING)
- SearchPhrase: 18K → **6M** distinct, 7% → **13.2%** non-empty
- URL LIKE '%google%': 0.0095% → **0.016%** (15,911 rows)
- REGEXP_REPLACE domain groups HAVING cnt>10000: **331 domains**

---

## 4. Prioritized Optimization Roadmap

Priority scoring: **benchmark_impact × generality × (1/effort)**

---

### PRIORITY 1: COUNT DISTINCT two-phase rewrite

| | |
|---|---|
| **Queries** | Q8 (0.084s), Q9 (0.089s), Q13 (0.122s), Q10 (0.016s), Q11 (0.016s), Q22 (0.031s) |
| **Total time** | 0.358s (25.3% of benchmark) |
| **Expected savings** | 0.20–0.25s |
| **Generality** | **Extremely high** — COUNT DISTINCT is the #1 most common advanced aggregation in analytics |
| **Effort** | Medium — modify `gpu_physical_grouped_aggregate.cpp` |

**Why #1 priority**: COUNT DISTINCT affects **6 queries totaling 25.3%** of the benchmark. It is also
the single most-requested aggregation in real-world analytics (every dashboard with "unique users",
"unique sessions", "unique events" uses it). Optimizing this once makes Sirius competitive for all
analytics workloads, not just ClickBench.

**Approach**: Two-phase rewrite in the GPU aggregate executor:
```
Phase 1: cudf::distinct([group_key, value_key])  →  deduped table
Phase 2: cudf::groupby(group_key, COUNT_STAR)    →  final result
```

**Data reality at 100M rows**:
- **Q13**: `cudf::distinct(["SearchPhrase", "UserID"])` on 13.2M rows (after `WHERE SearchPhrase <> ''`).
  6M distinct SearchPhrases × 17.6M UserIDs → very few (SearchPhrase, UserID) duplicates exist →
  dedup output ≈ 13.2M rows → then 6M-group COUNT(*). The dedup on 13.2M rows (vs current per-group
  sort+unique on 100M) should be much faster.
- **Q8/Q9**: `cudf::distinct(["RegionID", "UserID"])` on 100M rows → 9,040 regions × ~1,946 avg
  unique UserIDs = ~17.6M distinct pairs → dedup reduces 100M → 17.6M rows. Then 9,040-group groupby.
  The 5.7× row reduction (100M → 17.6M) before groupby gives substantial savings.
- **Q10/Q11**: Similar pattern with MobilePhoneModel/MobilePhone keys (filtered to non-empty rows).

**Implementation in `gpu_physical_grouped_aggregate.cpp`**:
1. Detect `COUNT_DISTINCT` in the aggregate list
2. Build a combined column list: [group_keys] + [distinct_value_col]
3. Call `cudf::distinct(combined_table, combined_key_indices)` → deduped table
4. Build new groupby request with `COUNT_ALL` on deduped table using only group_key columns
5. If the query has other aggregates (Q9: SUM, AVG), compute those separately via normal groupby
   and join results by group key

---

### ~~PRIORITY 2: GPU REGEXP_REPLACE~~ — ALREADY IMPLEMENTED

| | |
|---|---|
| **Status** | **DONE** — commit `036a6b4 Implement q28 regex using cudf jit transform` |
| **Queries** | Q28 (0.267s on GH200 — this IS the GPU time, not CPU fallback) |

GPU regex for Q28 is already implemented on this branch via cudf JIT transform.
The fallback checker is disabled (`9e3d3e9`). Verified running on GPU at 10M rows (0.123s hot run).
The 0.267s GH200 time reflects the GPU regex cost at 100M rows (81M Referer strings, 8.6 GB).

**No further optimization needed** unless we want to optimize the JIT regex itself (unlikely to
yield significant gains — regex NFA is bandwidth-bound at 8.6 GB).

---

### PRIORITY 3: STRLEN offset arithmetic optimization

| | |
|---|---|
| **Queries** | Q27 (0.166s) |
| **Total time** | 0.166s (11.7% of benchmark) |
| **Expected savings** | 0.12–0.15s |
| **Generality** | **High** — exposes libcudf string column offset API reusable for other string ops |
| **Effort** | **Low** — single function replacement |

**Why #3 priority**: Best effort-to-impact ratio. STRLEN(URL) on 10.5 GB is 53× slower than
the bandwidth floor (166ms vs 3.1ms). The root cause is almost certainly that STRLEN traverses
the character buffer rather than computing `offsets[i+1] - offsets[i]` from the 400 MB offset array.
This is a 25× memory reduction.

**Implementation**: In the STRLEN handler:
1. Get the string column's offsets child column: `col.child(0)` (int32 offsets)
2. Compute `offsets[i+1] - offsets[i]` via `cudf::binary_operation(offsets_shifted, offsets, SUB)`
   or a simple custom kernel
3. Return the resulting int32 column as the STRLEN result

**Data at 100M rows**: CounterID has 6,506 distinct values, 100 groups pass HAVING COUNT>100000.
After STRLEN optimization, the groupby on 6,506 groups with integer aggregates (SUM, COUNT) should
be fast (~5–10ms for 100M rows on integer columns). Total expected: ~15–40ms.

---

### PRIORITY 4: String GROUP BY acceleration

| | |
|---|---|
| **Queries** | Q33 (0.054s), Q34 (0.057s), Q18 (0.045s), Q16 (0.028s), Q17 (0.025s), Q12 (0.015s), Q14 (0.016s) |
| **Total time** | 0.240s (17.0% of benchmark) |
| **Expected savings** | 0.10–0.15s |
| **Generality** | **Very high** — string GROUP BY is ubiquitous in analytics |
| **Effort** | Medium-High |

**Approach**: Dictionary-encode or hash-fingerprint string keys before groupby:
- Compute 64-bit MurmurHash per string row
- GROUP BY integer hash (800 MB int64 vs 10.5 GB strings)
- For top-k results, verify exact string equality (collision probability negligible for k=10)

---

### PRIORITY 5: Top-K pre-allocated scratch buffers

| | |
|---|---|
| **Queries** | All ORDER BY + LIMIT ≤ 32 queries (~15 queries) |
| **Total time saved** | ~0.030s cumulative |
| **Generality** | **Very high** — ORDER BY LIMIT is ubiquitous |
| **Effort** | **Very low** — pre-allocate 16 MB in GPUBufferManager |

---

### PRIORITY 6: Late materialization for SELECT * with selective filters (Q23)

| | |
|---|---|
| **Queries** | Q23 (0.066s) |
| **Expected savings** | 0.05s |
| **Generality** | **High** — applies to any `SELECT *` with selective WHERE |
| **Effort** | Medium |

---

### NOT PRIORITIZED: Degenerate GROUP BY (Q32)

Saves 4.5% but is a benchmark-specific hack with zero real-world transfer value. Deprioritized
in favor of optimizations that benefit all analytics workloads.

---

## 5. Projected Impact

### Conservative estimate (P1–P3)

```
Optimization              Current(s)  Target(s)   Saved(s)
───────────────────────── ────────── ────────── ────────
P1: COUNT DISTINCT        0.358       0.12        0.24
P2: GPU REGEXP_REPLACE    0.267       0.06        0.21
P3: STRLEN offset arith.  0.166       0.03        0.14
───────────────────────── ────────── ────────── ────────
Subtotal P1-P3                                    0.59
New benchmark total:      1.415       0.83        (41% reduction)
```

### All P1–P6

```
P1: COUNT DISTINCT        0.358       0.12        0.24
P2: GPU REGEXP_REPLACE    0.267       0.06        0.21
P3: STRLEN offset arith.  0.166       0.03        0.14
P4: String GROUP BY       0.240       0.12        0.12
P5: Top-K malloc elim.    (spread)    -0.03       0.03
P6: Late materialization  0.066       0.01        0.06
───────────────────────── ────────── ────────── ────────
Total                                             0.80
New benchmark total:      1.415       0.62        (56% reduction)
```

---

## 6. Validation Protocol

**After each optimization, BEFORE committing:**

1. **Full 43-query correctness check**: Run all queries (Q0–Q42) on 10M rows and compare
   outputs against baseline. Every query must produce identical results.
2. **No regressions**: Every query's hot-run time must be ≤ baseline (within noise margin ~5%).
   Flag any query that regresses >10%.
3. **Target query speedup**: Measure the optimized query(s) using runner.py methodology
   (3 runs, min of warm runs, `.timer on`). Record before/after in Section 8.
4. **Commit**: Only commit to `cudf-25.12-optimization` after steps 1–3 pass. Include
   before/after numbers in the commit message.
5. **Iterate**: Move to the next optimization only after the current one is validated.

**Baseline output capture**: Before starting any optimization, save the full 43-query output
(query results, not just timings) as the golden reference for correctness checks.

---

## 7. Data Profiling — 100M Row Results (Complete)

| Metric | Value | Used By |
|--------|-------|---------|
| CounterID distinct | **6,506** (100 pass HAVING>100000) | Q27 |
| REGEXP_REPLACE domains (HAVING cnt>10000) | **331** | Q28 |
| WatchID unique ratio | **~100%** (confirmed) | Q32 |
| SearchPhrase non-empty rows | **13.2M** (6M distinct) | Q13, Q12-Q18 |
| URL total data | **10.5 GB** | Q27, Q33, Q34, Q20-Q23 |
| Referer total data | **8.6 GB** (81M non-empty) | Q28 |
| URL LIKE '%google%' matches | **15,911** (0.016%) | Q20-Q23 |
| UserID distinct | **17.6M** | Q8, Q9, Q13 |
| RegionID distinct | **9,040** | Q8, Q9 |
| URL distinct | **18.3M** | Q33, Q34 |

---

## 7. What We Know Is NOT the Bottleneck

- **Q27/Q32 custom Top-K kernel**: 0.09-0.13ms. Not the bottleneck.
- **DuckDB 1.4 query plan change**: Identical plans for Q27/Q32 between DuckDB 1.2.1 and 1.4.0.
- **Q32 sort after groupby**: All groups have COUNT=1; sort is trivial. Bottleneck = 100M-entry hash groupby.
- **Q27 CounterID groupby**: 6,506 groups fits in L2 cache (208 KB). Bottleneck = STRLEN character traversal.
- **Q27 Top-K routing**: LIMIT 25 ≤ 32 → uses Heap Sort Engine A correctly.

---

## 8. Experiment Log

Track before/after results here. All measurements use runner.py methodology
(3 runs, min of warm runs, .timer on).

### Baseline (RTX 6000, 10M rows, cudf-25.12-optimization branch, pre-optimization)

```
Query  Cold(s)  Warm1(s)  Warm2(s)  HOT(s)   GH200(s)  Ratio
Q8     0.292    0.080     0.079     0.079    0.084     0.9×
Q9     0.210    0.080     0.080     0.080    0.089     0.9×
Q13    0.656    0.232     0.236     0.232    0.122     1.9×
Q27    1.071    0.021     0.021     0.021    0.166     0.1×
Q28    3.107    0.127     0.123     0.123    0.267     0.5×  (already GPU, JIT regex)
Q33    0.030    0.027     0.027     0.027    0.054     0.5×
Q32    0.225    0.052     0.052     0.052    0.063     0.8×
```

**NOTE**: Q28 REGEXP_REPLACE is already implemented on GPU via JIT transform on this branch
(commit 036a6b4). P2 optimization is already done. The 0.267s on GH200 is GPU time, not CPU fallback.

### Debug Profiling — Q27 (STRLEN) Breakdown (10M rows, hot run = 19.38ms)

```
Table Scan (filter + late mat URL,CounterID):  0.44 ms
Projection (STRLEN):                          14.26 ms  ← 73% of total
  └─ Materialize String Kernel:               10.13 ms  (copying URL chars to CUDA column)
  └─ STRLEN computation + cudf conversion:     4.13 ms
GroupBy (CounterID, 112 groups):               3.28 ms
HAVING Filter:                                 0.11 ms
Top-K (Heap Sort, LIMIT 25):                   1.00 ms
Result Collector:                              0.15 ms
```

**Key insight**: 10.13ms is spent materializing URL string data JUST to compute string length.
The offset-subtraction approach eliminates this entirely — STRLEN = offsets[i+1] - offsets[i],
reading only the 40MB offset array instead of ~1GB char buffer. Expected: 14ms → 1-2ms.

### Debug Profiling — Q13 (COUNT DISTINCT) Breakdown (10M rows, hot run = 233.85ms)

```
Table Scan (filter SearchPhrase<>''):           0.34 ms
Materialize String (SearchPhrase):              2.34 ms
Materialize (UserID):                           0.11 ms
CUDF GroupBy (COUNT_DISTINCT):                229.22 ms  ← 98% of total!
  └─ Input: 1,374,133 rows (after WHERE filter)
  └─ Output: 835,092 groups
Top-K (Heap Sort):                              1.52 ms
```

**Key insight**: cudf::groupby with COUNT_DISTINCT takes 229ms on only 1.37M rows. This is the
internal per-group sort+unique that libcudf performs. Two-phase (distinct → count_star) should
bypass this entirely. Expected: 229ms → 10-30ms.

### P3: STRLEN offset arithmetic — DONE (commit 67a12c8)

```
RTX 6000, 10M rows, gpu_processing(), per-query gpu_buffer_init
Q27: 0.021s → 0.007s (3x speedup) — STRLEN(URL) GROUP BY
Q28: 0.125s → 0.109s (13% faster) — AVG(STRLEN(Referer)) with REGEXP_REPLACE
Correctness: PASS (all 43 queries match CPU golden)
No regressions on other queries
```

### P5: Top-K malloc elimination

```
TODO: measure affected queries before/after
```

### P1: COUNT DISTINCT two-phase

```
TODO: measure Q8, Q9, Q13 before/after
```

---

## Future Proposal: Dictionary Encoding for String Columns

**Idea**: During cold-run data loading (gpu_buffer_init), compute cardinality
of each string column. If `distinct_count / row_count < threshold` (e.g., 0.5),
dictionary-encode the column: store a dictionary of unique strings + integer
codes per row.

**Impact on GPU operations**:
- GROUP BY on strings → GROUP BY on integers (fixes VARCHAR GROUP BY fallback)
- JOIN on string keys → integer JOIN
- WHERE col = 'value' → single dict lookup + integer filter
- ORDER BY strings → integer sort
- These cover majority of real-world analytics string operations

**When it helps**: Columns with < ~1M distinct values (categories, status codes,
country names, browser names, search engines). ClickBench examples: CounterID
labels, MobilePhoneModel (~500 distinct), BrowserLanguage, SocialNetwork.

**When it doesn't help**: Near-unique columns (URL with 6M/100M distinct,
WatchID, UserID). LIKE/REGEXP still need to decode back to strings.

**Implementation path**: cudf already has `cudf::dictionary_column_view` with
dict-aware GROUP BY, sort, etc. Main work is integrating dict encoding into
Sirius's GPUColumn and data loading path, and routing operators through cudf's
dictionary APIs.

**Estimated scope**: Medium-term (not a weekend project). High impact across
real workloads beyond ClickBench.
