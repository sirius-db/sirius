# GPU Native Scan — Performance Results & Analysis

**Date:** 2026-04-16
**Branch:** `feature/gpu-scan-duckdb-api`
**Machine:** GH200 480GB (132 SMs, 96GB HBM3e, NVLink-C2C 368 GB/s, 64-core ARM)
**Config:** `bench_config_optimized.yaml` (batch=1GB, 4 scan threads)

## TPC-H SF100 — GPU vs CPU (warm, single-session, batch=1GB)

| Query | GPU (s) | CPU (s) | Speedup | Category |
|-------|---------|---------|---------|----------|
| Q1 | 0.302 | 0.153 | 0.51x | Exec limited — cudf hash_agg with 4 groups |
| Q2 | 0.081 | 0.111 | **1.37x** | GPU wins |
| Q3 | 0.231 | 0.273 | **1.18x** | GPU wins |
| Q4 | 0.151 | 0.229 | **1.52x** | GPU wins |
| Q5 | 0.282 | 0.281 | **1.00x** | GPU wins (tie) |
| Q6 | 0.121 | 0.086 | 0.71x | Scan limited — bitpacking decode |
| Q7 | 0.262 | 0.293 | **1.12x** | GPU wins |
| Q8 | 0.221 | 0.251 | **1.14x** | GPU wins |
| Q9 | 0.534 | 0.949 | **1.78x** | GPU wins |
| Q10 | 0.715 | 0.528 | 0.74x | Scan limited — bitpacking decode |
| Q11 | 0.040 | 0.043 | **1.07x** | GPU wins |
| Q12 | 0.181 | 0.157 | 0.87x | Scan limited — bitpacking decode |
| Q13 | 0.313 | 0.871 | **2.78x** | GPU wins |
| Q14 | 0.130 | 0.186 | **1.43x** | GPU wins |
| Q15 | 0.141 | 0.112 | 0.79x | Scan limited |
| Q16 | 0.131 | 0.160 | **1.22x** | GPU wins |
| Q17 | 0.241 | 0.232 | 0.96x | Scan limited |
| Q18 | 1.058 | 0.766 | 0.72x | Exec limited — cudf binary_op |
| Q19 | 0.231 | 0.292 | **1.26x** | GPU wins |
| Q20 | 0.181 | 0.220 | **1.22x** | GPU wins |
| Q21 | 0.634 | 0.891 | **1.41x** | GPU wins |
| Q22 | 0.080 | 0.140 | **1.75x** | GPU wins |
| **TOTAL** | **6.261** | **7.224** | **1.15x** | **GPU wins overall** |

**GPU wins on 14/22 queries.**

## ClickBench 100shard (100M rows, 24GB)

| Query | GPU (s) | CPU (s) | Speedup | Category |
|-------|---------|---------|---------|----------|
| Q6 | 0.010 | 0.018 | **1.80x** | GPU wins |
| Q15 | 0.060 | 0.071 | **1.18x** | GPU wins |
| Q16 | 0.121 | 0.151 | **1.25x** | GPU wins |
| Q17 | 0.121 | 0.147 | **1.21x** | GPU wins |
| Q18 | 0.181 | 0.327 | **1.81x** | GPU wins |
| Q32 | 0.312 | 0.395 | **1.27x** | GPU wins |
| Q35 | 0.061 | 0.082 | **1.34x** | GPU wins |
| Q36 | 0.020 | 0.022 | **1.10x** | GPU wins |
| Q39 | 0.021 | 0.032 | **1.52x** | GPU wins |
| **TOTAL** | **3.014** | **2.537** | **0.84x** | |

GPU wins 9/29. Scan-limited: 9 queries. Exec-limited: 11 queries.

## Bottleneck Classification

### TPC-H SF100 (from nsys profiling at SF10)
- **GPU wins (14):** Q2,3,4,5,7,8,9,11,13,14,16,19,20,21,22
- **Scan-limited (6):** Q6,10,12,15,17 — bitpacking decode dominates
- **Exec-limited (2):** Q1, Q18 — cudf execution slower than DuckDB vectorized CPU

### ClickBench 100shard
- **GPU wins (9):** Q6,15,16,17,18,32,35,36,39
- **Scan-limited (9):** Q12,14,20,21,30,31,33,34,38
- **Exec-limited (11):** Q1,2,3,7,19,25,26,37,40,41,42

### Key insight: string decode is NOT the bottleneck
String decode dominates only Q13 (81% of scan) and Q19 (78%), both already beat CPU.
The scan-limited queries are all limited by **bitpacking decode** on numeric columns.

## Config Tuning Results

### Batch size (scan_task_batch_size)

Measured on Q1/Q9 SF100 warm:

| Batch Size | Q1 (s) | Q9 (s) | Notes |
|------------|--------|--------|-------|
| 50MB | 0.796 | 1.551 | Per-batch overhead dominates |
| 100MB (default) | 0.443 | 1.258 | |
| 200MB | 0.302 | 0.785 | |
| 500MB | 0.302 | 0.594 | Q1 saturates |
| **1000MB** | **0.292** | **0.534** | **Recommended** |
| 2000MB | 0.312 | 0.413 | Diminishing returns, OOM risk |

Per-batch overhead measured at ~0.95ms (cudaMallocAsync + H2D enqueue + sync).
Overhead comes from: allocating staging buffer, N x cudaMemcpyAsync for each DuckDB
block, decode kernel launches, then stream.synchronize + cudaFreeAsync.

Practical rule: pick largest batch_size where `4 threads × 3 × batch_size < GPU memory pool`.

### Scan threads (duckdb_scan.num_threads)

| Threads | Q1 (s) | Q3 (s) | Q9 (s) | Notes |
|---------|--------|--------|--------|-------|
| 2 | 0.443 | 0.423 | 1.359 | |
| **4** | **0.433** | **0.362** | **1.259** | Sweet spot |
| 8 | 0.423 | 0.362 | 1.168 | OOM risk with large batches |

With 1GB batches, only 13-26 batches per table — more threads don't help.

## Scan Optimization Opportunities (next steps)

### A. Pre-allocate staging buffer (reuse across batches)
**Impact:** Eliminates cudaMallocAsync/cudaFreeAsync per batch (~200us savings each).
**Implementation:** In the `compute_task` loop, allocate staging once before the loop,
reuse across all batch iterations, free once after. Currently each batch iteration
does `cudaMallocAsync` (line 363) and `cudaFreeAsync` (line 384) in
`gpu_native_scan_task.cpp`.

### B. Batch contiguous H2D copies (HIGHEST IMPACT)
**Impact:** nsys shows 2,102 individual cudaMemcpyAsync calls (253KB each = one DuckDB
block). Each call has ~150us CPU-side enqueue overhead → 315ms of just API overhead
for 545MB of data. The actual NVLink transfer at 368 GB/s takes only 1.5ms.
Decode kernels take only 8.3ms (2% of scan time). H2D API overhead is 97%.
**Implementation:** DuckDB blocks are stored sequentially in the .duckdb file. Since
we already mmap the file, contiguous blocks within a column can be copied as one large
region. Instead of 2,102 × 253KB copies, group into ~20 × 27MB copies.
Expected: 20 × 150us + 1.5ms = ~5ms total (vs 352ms now = **70x faster**).
**File:** `src/op/scan/gpu_native_scan_task.cpp` lines 364-376 (the per-block copy loop).
Need to sort blocks by file offset, merge contiguous ranges, issue fewer copies.
**Note:** H2D/decode overlap (previously attempted, commit `1e7c6c6` reverted `e3e2719`)
is NOT helpful — decode is only 8ms, nothing to overlap with. The bottleneck is API
call count, not transfer bandwidth or decode compute.

### C. Reduce stream.synchronize() count
**Impact:** Currently 2-4 syncs per table decode in `gpu_native_decode.cu` (lines 438,
510, 749, 820). Each sync costs ~20us but forces CPU to wait for GPU.
**Implementation:** Defer null count computation to execution phase. Currently each
column's validity decode syncs to read the null count. Could pass a device pointer to
the null count and let the consumer read it later.
**Status:** Null count syncs already consolidated from per-column to per-table (commit
`b7ce106`). Further consolidation requires changing the cudf column construction API.

### D. Bitpacking decode improvements
**Impact:** The scan-limited queries (Q6,10,12,15) are dominated by bitpacking decode.
**Current state:** Each 2048-row metadata group gets one CTA (256 threads). Values per
thread = 8 (VPT). The batched kernel handles all groups in one launch.
**Possible improvements:**
- Increase VPT from 8 to 16 (more work per thread, fewer threads idle)
- Use warp-shuffle for DELTA_FOR prefix sum instead of CUB BlockScan
- Skip decode for filtered-out row groups (predicate pushdown into decode)

## Correctness Status

### TPC-H
- **SF1:** 21/22 pass (Q1: FP rounding at 15th digit — inherent double precision)
- **SF10:** 20/22 pass (Q1: FP rounding, Q21: ORDER BY tie-breaking)
- **SF100:** Exact match on varchar distributions and numeric sums

### ClickBench
- **20/29 pass** on 10shard
- Remaining 9: tie-breaking (7), non-deterministic (1), cudf int64 overflow (1)
- Timestamp decode bug fixed (was 0.56% corrupted due to 3-word unpack issue)
