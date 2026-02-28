# Sirius Optimization Analysis — ClickBench

## Guiding Principles
1. **Systematic, not benchmark-specific** — optimizations must be generalizable
2. **Portable to new Sirius** — changes should work on dev-fused-kernel (DuckDB 1.4, cudf 26.04)
3. **Profile before implementing** — use SIRIUS_LOG_LEVEL=debug + nsys to understand where time goes

## GH200 Baseline (cudf-25.12, hot runs)

| Query | Time (s) | Description | Notes |
|-------|----------|-------------|-------|
| Q0 | 0.001 | COUNT(*) | Trivial |
| Q8 | 0.084 | COUNT(DISTINCT), GROUP BY, ORDER BY | |
| Q9 | 0.089 | COUNT(DISTINCT), GROUP BY, ORDER BY | |
| Q13 | 0.122 | SUM, COUNT(DISTINCT), GROUP BY | **8x slower than Q12 (0.015s) — only difference is COUNT(DISTINCT)** |
| Q16 | 0.028 | GROUP BY URL, ORDER BY LIMIT 10 | TopK helps |
| Q18 | 0.045 | GROUP BY URL, ORDER BY LIMIT 10 | TopK helps |
| Q23 | 0.060 | GROUP BY URL, SUM, ORDER BY LIMIT 10 | |
| Q27 | 0.166 | GROUP BY, STRLEN, AVG, ORDER BY LIMIT 10 | TopK helps; high-cardinality string GROUP BY |
| Q28 | 0.267 | REGEXP_REPLACE, GROUP BY, ORDER BY LIMIT 20 | **Slowest query; JIT exists but hacky** |
| Q32 | 0.063 | GROUP BY URL, COUNT(DISTINCT), ORDER BY LIMIT 10 | TopK helps |
| Q33 | 0.054 | LIKE filter, GROUP BY, ORDER BY LIMIT 10 | TopK helps |
| Q34 | 0.057 | GROUP BY URL, SUM, ORDER BY LIMIT 10 | TopK helps |

## Optimization Opportunities

### 1. Q28 REGEXP_REPLACE — JIT Compilation (0.267s, highest ROI)

**Current state:** A hand-written CUDA kernel exists in `src/expression_executor/regex/regex_playground.cpp` that JIT-compiles via `cudf::transform()`. However, it's dispatched by **exact string matching** against the ClickBench Q28 pattern (`^https?://(?:www\.)?([^/]+)/.*` with replacement `\1`).

**Code path:** `gpu_execute_function.cpp:594` — hardcoded `if (pattern_str == R"(...)"`

**Problem:** This is benchmark-specific, not generalizable. The JIT only triggers for this one regex.

**Potential improvements:**
- Generalize the JIT to handle common regex patterns (URL extraction, domain extraction)
- Build a small regex → CUDA transpiler for simple patterns (character classes, anchors, greedy quantifiers)
- The `cudf::strings::replace_with_backrefs` fallback is significantly slower

**Portability:** Medium — the JIT approach is sound, but the exact API (`cudf::transform`) may differ between cudf versions.

### 2. COUNT(DISTINCT) Optimization (Q13: 0.122s, Q8/Q9: ~0.085s)

**Current state:** COUNT(DISTINCT) maps to `cudf::make_nunique_aggregation()` inside `cudf_groupby.cu`. This uses cudf's standard hash-based groupby with cuco GPU hash maps.

**Q13 vs Q12 comparison:** Adding a single COUNT(DISTINCT) column causes 8x slowdown (0.015s → 0.122s), suggesting nunique aggregation is the bottleneck.

**Potential approaches:**
- **Pre-filter with approximate cardinality** (HyperLogLog) to decide strategy — but HLL itself has GPU cost and traditional databases don't typically use this approach
- **Sort-based COUNT(DISTINCT)** — sort + adjacent-difference. However, cudf engineers likely already benchmarked sort vs hash; their cuco hash maps are highly optimized for GPU (O(n) vs O(n log n))
- **Separate nunique from other aggregations** — run the regular aggregations (SUM, COUNT) in one pass, then compute nunique separately. DuckDB controls query planning so we can't rearrange at the SQL level, but we could split internally in the Sirius operator

**Verdict:** Needs profiling first. The 8x slowdown suggests there may be an implementation issue rather than an algorithmic one. Profile with nsys to see if it's hash table contention, memory allocation, or something else.

### 3. Skewed/High-Cardinality GROUP BY (Q27: 0.166s, Q28: 0.267s)

**Current state:** `cudf_groupby.cu` uses standard `cudf::groupby::groupby` with hash-based approach. No skew detection or handling.

**Background:** In traditional databases, skewed GROUP BY (where some groups are much larger than others) can be optimized with:
- Pre-aggregation / partial aggregation before shuffle
- Adaptive strategies based on cardinality estimation
- Sort-based aggregation when data is already sorted or nearly sorted

**cudf availability:** cudf's groupby does NOT expose sort-based or skew-aware variants through its public API. Any implementation would need to be custom.

**Existing unused code:** `src/cuda/operator/unused/optimized_grouped_aggregate.cu` contains a sort-based string GROUP BY implementation. Its history is unknown — unclear if it was tried and found slower, or never integrated.

**Verdict:** Lower priority. The hash-based approach in cudf is generally efficient for GPU workloads. Skew handling is more relevant for distributed systems; on a single GPU, the hash table approach handles skew reasonably well. Focus on Q28 JIT and COUNT(DISTINCT) first.

### 4. TopK Improvements

**Current state:** Implemented in `cudf_orderby.cu` with routing logic:
- `limit <= 32`: Custom heap sort (multi-column supported)
- `limit > 32, single column`: Custom radix sort
- `limit > 32, multi-column` OR unsupported types: Falls back to full cudf sort

**Queries benefiting:** Q16, Q18, Q26, Q27, Q32, Q33, Q34 (all have ORDER BY ... LIMIT ≤10)

**Q28 specifically:** Has LIMIT 20 with REGEXP_REPLACE — TopK should activate (limit ≤ 32) but the bottleneck is the regex, not the sort.

**Potential improvements:**
- Extend radix sort to multi-column cases
- Support more types (currently limited to INT_64, INT_32, DOUBLE, STRING, INT_128, INT_16, FLOAT_32)
- But ROI is low since most ClickBench queries use LIMIT 10 (handled by heap sort)

**Verdict:** Already well-implemented. Low priority for further optimization.

## Profiling Plan

### Step 1: Baseline Benchmark on L40S
Run all 43 queries with `SIRIUS_LOG_LEVEL=debug` to get per-operator timing.
Config: 20GB caching, 20GB processing, 40GB pinned.

### Step 2: nsys Profiling on Slow Queries
Target Q28, Q27, Q13, Q8, Q9 individually:
```bash
nsys profile --stats=true -o profile_q28 \
  ./build/release/duckdb hits.duckdb \
  -c "call gpu_buffer_init('20 GB', '20 GB', pinned_memory_size = '40 GB');" \
  -c "call gpu_processing('...');"
```

### Step 3: Analyze Kernel-Level Breakdown
- Which CUDA kernels dominate each slow query?
- Is the bottleneck compute-bound or memory-bound?
- For COUNT(DISTINCT): is it hash table construction, or the nunique reduction?
- For Q28: how much time in regex vs groupby vs sort?

### Step 4: Implement Based on Evidence
Priority order (subject to profiling results):
1. Q28 regex generalization (if regex is the dominant cost)
2. COUNT(DISTINCT) optimization (if nunique is clearly the bottleneck in Q13)
3. Skewed aggregation (only if profiling reveals hash table contention)
