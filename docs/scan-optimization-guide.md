# GPU Native Scan Optimization Guide

## Current State (2026-04-15, branch: feature/gpu-native-scan-task)

### Commits Applied
1. `6de87fb` — **Batch bitpacking decode**: 48K kernel launches → 1K per query. SF100 suite 13% faster.
2. `a272bba` — **Remove per-operator pipeline sync**: Eliminates ~1.4K syncs/query from pipeline loop. Minimal wall-clock impact on GH200 but may help PCIe GPUs.

### Baseline vs Current (TPC-H SF100, GH200 480GB, all 22 queries)

| Query | Baseline | Batched+SyncFix | Speedup |
|-------|----------|-----------------|---------|
| Q1    | 2.232s   | 2.009s          | 1.11x   |
| Q2    | 0.238s   | 0.198s          | 1.20x   |
| Q3    | 1.567s   | 1.375s          | 1.14x   |
| Q4    | 1.305s   | 1.204s          | 1.08x   |
| Q5    | 1.667s   | 1.476s          | 1.13x   |
| Q6    | 1.687s   | 1.506s          | 1.12x   |
| Q7    | 1.707s   | 1.516s          | 1.13x   |
| Q8    | 2.997s   | 2.613s          | 1.15x   |
| Q9    | 3.097s   | 2.674s          | 1.16x   |
| Q10   | 2.412s   | 2.111s          | 1.14x   |
| Q11   | 0.218s   | 0.177s          | 1.23x   |
| Q12   | 1.919s   | 1.737s          | 1.10x   |
| Q13   | 1.335s   | 1.335s          | 1.00x   |
| Q14   | 1.999s   | 1.757s          | 1.14x   |
| Q15   | 2.160s   | 1.909s          | 1.13x   |
| Q16   | 0.308s   | 0.268s          | 1.15x   |
| Q17   | 2.080s   | 1.727s          | 1.20x   |
| Q18   | 1.477s   | 1.375s          | 1.07x   |
| Q19   | 3.418s   | 3.056s          | 1.12x   |
| Q20   | 1.476s   | 1.285s          | 1.15x   |
| Q21   | 2.352s   | 2.059s          | 1.14x   |
| Q22   | 0.218s   | 0.197s          | 1.11x   |
| **TOTAL** | **37.87s** | **33.56s** | **1.13x** |

---

## Time Breakdown (Q1 SF100, post-optimization)

### GPU Kernel Time: 608ms
```
Groupby agg (single_pass_shmem_aggs):   184ms  30.4%  ← cudf internal
Dict string gather (kernel_gather_dict): 127ms  20.9%  ← OUR CODE
Dict string lengths (kernel_compute_lengths_dict): 124ms  20.4%  ← OUR CODE
Hash mapping (mapping_indices_kernel):    37ms   6.0%  ← cudf internal
Bitpacking decode (batched):              40ms   6.5%  ← already optimized
Binary ops (Mul, Add, Sub, LessEqual):    29ms   4.8%  ← cudf internal
CUB scan/select/memcpy:                   37ms   6.1%  ← cudf internal
Other:                                    30ms   4.9%
```

### Sync Overhead: 624ms
```
Heavy syncs (>100us, real GPU wait):   499ms  1,046 calls  ← cudf/rmm internal
Fast syncs (<100us, GPU already done): 125ms  16,935 calls ← cudf/rmm internal
```

### H2D Transfers: 82ms
```
24,792 calls, 5,780 MB transferred at ~70 GB/s
```

### Wall Clock: ~2.0s = 608ms kernel + 624ms sync + 82ms H2D + ~686ms CPU/API overhead

### GPU Utilization: 31.3% (CRITICAL)
```
Total kernel span:  1940ms (first kernel to last kernel)
Actual compute:      607ms (kernels running)
GPU IDLE:           1339ms (69% of the time!)

Idle gap distribution:
  <1us:       94 gaps,    0.1ms  (negligible)
  1-10us:  12387 gaps,   49.7ms  (inter-kernel dispatch)
  10-100us: 6261 gaps,  195.5ms  (sync overhead, API calls)
  100us-1ms: 1165 gaps, 342.6ms  (stream sync blocking)
  >1ms:      296 gaps,  751.0ms  (CPU-side work between tasks)

SM utilization by kernel grid size:
  1-10 CTAs:   9889 launches,  12.7ms   2.1%  ← tiny kernels (validity fills, etc.)
  11-50 CTAs:  1722 launches, 250.5ms  41.2%  ← STRING DECODE (17 CTAs, underutilized!)
  51-132 CTAs:  580 launches,   0.8ms   0.1%
  133-500 CTAs: 1731 launches, 235.2ms 38.7%  ← groupby, hash ops (good)
  500+ CTAs:   6327 launches, 108.2ms  17.8%  ← bitpacking, CUB (good)
```

The 11-50 CTA bucket (string decode at 13% SM utilization) accounts for 41.2% of kernel
compute time. Increasing its parallelism to 500+ CTAs would reduce its time by ~8x.

The >1ms idle gaps (751ms total, 296 events) are CPU-side work between pipeline tasks:
building cudf tables, submitting to task executors, etc. These are not fixable with
GPU optimizations alone — they require CPU-side pipeline improvements or task batching.

---

## Remaining Optimization Opportunities (Priority Order)

### 0. Increase String Decode GPU Utilization (CRITICAL FINDING)

**Problem**: String decode kernels launch with **grid = 5-17 CTAs** on a GPU with 132 SMs.
Only 13% of SMs are active during the 251ms of dictionary string decode. 115 SMs sit idle.

```
Kernel             Grid  SMs Used  SM Util  Time
─────────────────  ────  ────────  ───────  ────
dict_gather          17   17/132     13%    127ms  ← PROBLEM
dict_lengths         17   17/132     13%    124ms  ← PROBLEM
bitpacking_batched 1020  132/132    100%     40ms  ← already fixed
groupby_agg         264  132/132    100%    184ms  ← cudf, fine
```

This is the same problem we fixed for bitpacking. The string decode is already "batched" (one CTA
per segment in one launch), but each task only has 5-17 string segments per column.

**Fix**: Use more CTAs per segment. Instead of 1 CTA per segment (12K rows, grid-stride loop),
use 1 CTA per 256-row chunk within a segment:

```
Current:  17 CTAs (1 per segment × 17 segments), each processes 12K rows
Proposed: 17 × 47 = ~800 CTAs (1 per 256-row chunk), each processes 256 rows
```

This is exactly what we did for bitpacking (one CTA per 2048-row metadata group).
For string decode, pass 2 (gather) already has precomputed output offsets from the CUB
ExclusiveSum in pass 1, so each sub-CTA knows exactly where to write.

**Expected**: dict_gather 127ms → ~16ms, dict_lengths 124ms → ~16ms. Combined **~219ms saved** on Q1 SF100.

**Implementation sketch**:
```cuda
// Current: one CTA per segment, grid-stride loop
for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) { ... }

// New: one CTA per chunk, each handles 256 rows
struct batched_string_chunk_desc {
    const uint8_t* d_block;
    uint32_t block_offset;
    uint32_t chunk_start;       // first row in this chunk (within segment)
    uint32_t chunk_row_count;   // rows in this chunk (≤256)
    uint32_t global_row_start;  // global row offset for output indexing
    // ... segment-specific fields (bp_width, dict_end, etc.)
};

// Each CTA processes rows [chunk_start, chunk_start + chunk_row_count)
for (uint32_t i = threadIdx.x; i < desc.chunk_row_count; i += blockDim.x) {
    uint32_t row = desc.chunk_start + i;
    // ... same decode logic but with row offset
}
```

**Files**: `src/cuda/scan/gpu_decode_batched_string.cu` (all 6 kernels), caller in the same file

**GH200 vs RTX6000**: Benefits both equally. RTX6000 has fewer SMs (76-84) so the utilization
gap is smaller, but still significant with grid=17. On RTX6000: 17/76 = 22% utilization.

### 1. Vectorized String Writes (HIGHEST IMPACT, EASIEST)

**Problem**: `kernel_gather_dict` and `kernel_gather_fsst` copy strings byte-by-byte:
```cuda
for (uint32_t b = 0; b < str_len; b++) {
    d_chars[out_pos + b] = src[b];  // uncoalesced byte writes
}
```

**Fix**: Replace with `memcpy()` which the CUDA compiler optimizes to vector loads/stores:
```cuda
memcpy(d_chars + out_pos, src, str_len);
```

For FSST symbol writes specifically, use a single 8-byte store (symbols are 1-8 bytes packed in a uint64):
```cuda
// Replace inner loop:
//   for (uint8_t j = 0; j < sym_len; j++) {
//       d_chars[out_pos + j] = static_cast<uint8_t>(sym);
//       sym >>= 8;
//   }
// With:
memcpy(d_chars + out_pos, &sym, sizeof(sym));  // writes 8 bytes, only sym_len valid
out_pos += sym_len;
```
Requires 7 bytes of padding at end of output buffer. This is the technique from the original CPU FSST paper.

**Expected**: 1.5-2x speedup on dict/FSST gather kernels. Q1: save ~60-90ms. ClickBench: save ~500-1000ms.

**Files**: `src/cuda/scan/gpu_decode_batched_string.cu` (kernel_gather_dict ~line 322, kernel_gather_fsst ~line 418-431)

**GH200 vs RTX6000**: Benefits both equally — this is a compute/memory efficiency optimization.

### 2. DICT_FSST Column Optimization: Decode Dictionary Once

**Problem**: For DICT_FSST columns (e.g., ClickBench), the FSST decompression runs per-row. But dictionary columns have a small set of unique strings (often <1000).

**Fix**: 
1. FSST-decode just the dictionary entries (small, fast)
2. Build a decoded string table on GPU
3. For each row, just `memcpy` from the decoded dictionary (no FSST decode per row)

**Expected**: Eliminates FSST decode entirely for DICT_FSST columns. ClickBench Q21: ~2300ms → ~300ms (est.)

**Files**: `src/cuda/scan/gpu_decode_batched_string.cu` — add a new path for DICT_FSST segments

**GH200 vs RTX6000**: Benefits both equally.

### 3. ~~GH200 Zero-Copy Scan Path~~ (REJECTED)

**Tested and rejected.** Zero-copy is a NET REGRESSION for in-memory datasets:
- Kernels read from host at 615 GB/s vs HBM at 3,400 GB/s (5.5x slower reads)
- H2D copy savings (87ms) < kernel slowdown (125ms+)
- Q1 SF100: 1.82s → 2.09s (15% slower)
- dict_gather: 6.5ms → 112ms (17x slower from host reads!)

Zero-copy would only help for out-of-core datasets that don't fit in GPU memory.
For in-memory datasets, copy-then-decode-from-HBM is always faster.

### 4. Shared Memory Staging for String Writes

**Problem**: String gather kernels write decoded bytes to scattered global memory positions. Adjacent threads write to non-adjacent output positions → uncoalesced writes.

**Fix** (from CODAG research): Decode into a per-CTA shared memory buffer. When buffer fills (~4KB), all 256 threads cooperatively flush to global memory as coalesced 4-byte stores. Converts O(N) random byte writes into O(N/128) coalesced cache-line writes.

**Expected**: Additional 2-3x on gather kernels after vectorized writes.

**Files**: `src/cuda/scan/gpu_decode_batched_string.cu`

**GH200 vs RTX6000**: Benefits RTX6000 more (PCIe has lower global memory bandwidth ceiling).

### 5. Reduce cudf Internal Syncs

**Problem**: ~17K cudaStreamSynchronize calls from cudf/rmm internals: hash table operations, memory pool management, CUB temporary allocation.

**Approaches**:
- Pre-allocate CUB temporary storage (avoid per-call size query sync)
- Use CUDA graphs to capture operator pipelines (eliminates all API overhead on replay)
- Increase cuCascade reservation sizes to avoid RMM fallback allocations
- Contribute upstream fixes to cudf for sync-heavy paths

**Expected**: Hard to quantify without cudf changes. Potentially 100-200ms.

**GH200 vs RTX6000**: RTX6000 may see larger benefit (PCIe sync round-trip is slower).

### 6. Operator Fusion: Scan + Filter + Aggregate

**Problem**: Each operator materializes a full cudf table, then the next operator reads it. For Q1 (scan → filter → groupby), this means:
- Scan: decode 600M rows to GPU memory
- Filter: read all, write filtered subset
- GroupBy: read filtered, compute aggregates

**Fix**: Fuse scan + filter into one kernel that only decodes rows passing the filter. Or fuse the entire pipeline: decode → filter → aggregate in one pass without materializing intermediate tables.

**Expected**: 2-5x overall speedup for scan-heavy queries. This is the architecture used by GPU databases like HeavyDB.

**Files**: Major architectural change across `src/op/`, `src/pipeline/`, `src/cuda/scan/`

**GH200 vs RTX6000**: Benefits both, but RTX6000 benefits more (less memory bandwidth = bigger win from avoiding intermediate materialization).

---

## GH200 vs RTX6000 PCIe Compatibility Matrix

| Optimization | GH200 Impact | RTX6000 Impact | Conflict? |
|-------------|-------------|----------------|-----------|
| Batch bitpacking (done) | +13% | Should help (fewer launches = less CPU-GPU latency) | No |
| Pipeline sync removal (done) | ~0% | May help more (PCIe sync round-trip slower) | No |
| Vectorized string writes | +5-10% | +5-10% | No |
| DICT_FSST decode-once | +20-50% on ClickBench | Same | No |
| GH200 zero-copy | +5% on Q1 | N/A (PCIe too slow) | **Yes** — must be behind runtime flag |
| Shared memory staging | +10-15% on string queries | +15-25% (more bandwidth-sensitive) | No |
| CUDA graphs | Unknown | Larger benefit (more API overhead) | No |
| Operator fusion | +50-100% est. | +100-200% est. | No |

### Runtime Detection and Adaptive Heuristics

**Principle**: Never hardcode GPU-specific values. Query `cudaDeviceProp` at startup and compute
optimal parameters from hardware capabilities.

```cpp
// Query once at startup (e.g., in config.cpp or gpu_native_scan_task constructor)
struct gpu_scan_config {
    int sm_count;                    // multiProcessorCount
    int max_threads_per_sm;          // maxThreadsPerMultiProcessor
    int max_blocks_per_sm;           // maxBlocksPerMultiProcessor  
    int warp_size;                   // warpSize
    bool use_zero_copy;              // GH200 unified memory detection
    uint32_t target_ctas_per_launch; // computed: enough to fill the GPU
    uint32_t chunk_size;             // computed: rows per CTA for string decode
};

gpu_scan_config detect_gpu_config() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    gpu_scan_config cfg;
    cfg.sm_count = prop.multiProcessorCount;          // GH200=132, RTX6000=76
    cfg.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;  // 2048
    cfg.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;    // 32 (Hopper), 16 (Ada)
    cfg.warp_size = prop.warpSize;                    // 32
    
    // GH200 zero-copy detection
    cfg.use_zero_copy = prop.pageableMemoryAccessUsesHostPageTables 
                     && prop.concurrentManagedAccess;
    
    // Target: fill all SMs with at least 2 waves of work.
    // This ensures the GPU scheduler has enough CTAs to hide latency.
    // Rule: target_ctas = sm_count × occupancy_blocks × 2 (waves)
    // For 256-thread blocks with low shmem: occupancy ~= 8 blocks/SM
    int occupancy_blocks = cfg.max_threads_per_sm / 256;  // 8 for Hopper/Ada
    cfg.target_ctas_per_launch = cfg.sm_count * occupancy_blocks * 2;
    // GH200: 132 × 8 × 2 = 2112
    // RTX6000: 76 × 8 × 2 = 1216
    
    // Chunk size: rows per CTA for string/fixed-width decode.
    // Trade-off: smaller chunks = more CTAs = better SM utilization,
    //            but also more descriptors and potentially less per-CTA efficiency.
    // Rule: chunk_size = total_rows / target_ctas, clamped to [64, segment_rows]
    // We compute this per-column, not globally. See below.
    cfg.chunk_size = 256;  // default; overridden per-column
    
    return cfg;
}
```

**Per-column chunk size computation**:
```cpp
// Given: total_rows across all segments, num_segments, target_ctas
// Goal: choose chunk_size such that total CTAs ≈ target_ctas

uint32_t compute_chunk_size(uint32_t total_rows, 
                            uint32_t num_segments,
                            uint32_t target_ctas) {
    if (num_segments >= target_ctas) {
        // Already enough segments — use 1 CTA per segment (current behavior)
        return total_rows / num_segments;  // ≈ segment_rows
    }
    
    // Need to split segments into chunks
    uint32_t chunk_size = total_rows / target_ctas;
    
    // Clamp: at least 64 rows (below this, CTA overhead dominates)
    // at most segment_rows (no point splitting tiny segments)
    chunk_size = max(chunk_size, 64u);
    chunk_size = min(chunk_size, total_rows / num_segments);
    
    // Round to warp size for coalescing
    chunk_size = (chunk_size / 32) * 32;
    if (chunk_size == 0) chunk_size = 32;
    
    return chunk_size;
}

// Example for Q1 SF100 dict columns:
// total_rows = 12000 × 17 segments = 204K rows per launch
// target_ctas = 2112 (GH200)
// chunk_size = 204K / 2112 = 96 rows → round to 96 (3 warps)
// CTAs = 204K / 96 = 2125 → fills all 132 SMs with 2 waves
```

**Why this heuristic works across GPUs**:
- On GH200 (132 SMs): chunk_size ≈ 96 → 2125 CTAs → 2 full waves
- On RTX6000 Ada (76 SMs): chunk_size ≈ 168 → 1214 CTAs → 2 full waves  
- On RTX4090 (128 SMs): chunk_size ≈ 100 → 2040 CTAs → 2 full waves
- On A100 (108 SMs): chunk_size ≈ 118 → 1728 CTAs → 2 full waves

The GPU never sits idle during decode regardless of hardware.

**For bitpacking**: chunk_size is already determined by `BP_META_GROUP_SIZE = 2048`
(DuckDB's metadata group size). This gives good utilization because segments have
many groups. No heuristic needed — the format dictates the parallelism.

**For string decode**: The format has no inherent chunk size. We choose it at runtime
based on the GPU. The kernel processes rows `[chunk_start, chunk_start + chunk_size)`
and uses the precomputed output offsets from pass 1 to know where to write.

---

## nsys Profiling Quick Reference

### Profile a query
```bash
NSYS=/usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-sbsa-armv8/nsys
# On x86: NSYS=nsys

echo ".timer on" > /tmp/q.sql
cat test/tpch_performance/tpch_queries/gpu/q1.sql >> /tmp/q.sql

SIRIUS_CONFIG_FILE=test_datasets/bench_config.yaml \
  $NSYS profile --trace cuda,nvtx --cuda-memory-usage=true \
  --force-overwrite=true --output=/tmp/profile_q1 \
  build/release/duckdb -unsigned -readonly -noheader \
  test_datasets/tpch_sf100.duckdb < /tmp/q.sql
```

### Key analysis queries (on exported .sqlite)
```sql
-- Top kernels by time
SELECT s.value, count(*), printf('%.1f ms', sum(k.end-k.start)/1e6)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value ORDER BY sum(k.end-k.start) DESC LIMIT 15;

-- Sync overhead
SELECT printf('%d calls, %.1f ms, avg %.1f us', count(*), sum(end-start)/1e6, avg(end-start)/1e3)
FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;

-- H2D transfers
SELECT printf('%d calls, %.1f MB, %.1f ms', count(*), sum(bytes)/1e6, sum(end-start)/1e6)
FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1;

-- Decode kernels specifically
SELECT s.value, count(*), printf('%.1f ms', sum(k.end-k.start)/1e6)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%decode%' OR s.value LIKE '%gather%' OR s.value LIKE '%lengths%'
   OR s.value LIKE '%bitpack%' OR s.value LIKE '%parse_and_decode%'
GROUP BY s.value ORDER BY sum(k.end-k.start) DESC;
```

---

## ClickBench Profile (baseline, heaviest string queries)

### Q21 (string-dominated, 5.0s wall clock)
```
kernel_gather_fsst:           1462ms  45.1%  ← FSST decompress+write
kernel_compute_lengths_fsst:   842ms  26.0%  ← FSST length compute  
kernel_gather_dict:            785ms  24.2%  ← dict string copy
contains_warp_parallel:         19ms   0.6%
Other:                         133ms   4.1%
```

### Q33/Q34 (sync-dominated, ~5s wall clock)
```
Sync calls: 200K+, 4 seconds of sync overhead
Kernel time: ~2 seconds
```

These ClickBench queries would benefit most from:
1. Vectorized string writes (immediate)
2. DICT_FSST decode-once optimization (large win)
3. Shared memory staging (additional)

---

## References

- GSST paper (GPU FSST decode, 191 GB/s): https://repository.tudelft.nl/record/uuid:71c1dddf-3b7d-4c12-a079-d716fad501b2
- CODAG framework (GPU decompression optimization): https://arxiv.org/html/2307.03760
- GPU FSST compressor: https://github.com/timanema/fsst-gpu
- cudf string optimization blog: https://developer.nvidia.com/blog/mastering-string-transformations-in-rapids-libcudf/
- nvcomp docs: https://docs.nvidia.com/cuda/nvcomp/index.html
- Original FSST paper (CWI): https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
