---
name: profile-analyzer
description: Analyze Sirius GPU performance from nsys profiles — runs benchmarks, generates reports with kernel occupancy, memory bandwidth, operator attribution, and compares runs for regression detection.
---

# Sirius nsys Profile Analyzer

You are analyzing GPU performance profiles for Sirius, a GPU-accelerated SQL query engine built on DuckDB. The profiles come from NVIDIA Nsight Systems (nsys) and are stored as SQLite databases.

## Profiling Overhead Warning

**nsys profiling adds measurable overhead to query execution times.** Timings captured during a profiled run (cold/hot in `summary.json`) are inflated and should NOT be used to determine whether an optimization actually improved performance. Instead:

1. **Profiled runs** → Use for GPU analysis (kernels, operators, occupancy, memory, bottlenecks)
2. **Non-profiled runs** → Use for accurate performance timing (cold/hot comparisons, regression detection)

Always run both when comparing performance across code changes.

## Workflows

There are four workflows — choose based on what the user wants:

### Workflow A: Full Performance Analysis (recommended)

This is the complete workflow: profile for GPU analysis, then run without profiling for accurate timings. Both runs should use the same queries, scale factor, and iteration count.

**Step 1: Profiled run** (for GPU analysis data)
```bash
bash test/tpch_performance/nsys_report.sh --sf <scale_factor> [query_numbers...]
```

**Step 2: Non-profiled timing run** (for accurate cold/hot times)
```bash
# Option A: Sirius-only timing
export SIRIUS_CONFIG_FILE=<path_to_config>
bash test/tpch_performance/run_tpch_parquet.sh sirius <scale_factor> <iterations> <query_numbers...>

# Option B: Full DuckDB vs Sirius benchmark with validation
export SIRIUS_CONFIG_FILE=<path_to_config>
bash test/tpch_performance/benchmark_and_validate.sh <scale_factor> <iterations>
```

The non-profiled run produces per-query `timings.csv` files with accurate cold/hot timings. `benchmark_and_validate.sh` also validates GPU results against CPU and produces a comparison table with speedup ratios.

**When comparing across runs:**
- Use the non-profiled timings to determine if performance actually improved or regressed
- Use the profiled data to understand *why* performance changed (kernel times, operator attribution, occupancy, memory patterns)
- The profiled `summary.json` timings are useful for relative comparisons within the same profiled run (e.g., which query is slowest) but not across runs with different code

**Full options for the profiled run:**
```bash
bash test/tpch_performance/nsys_report.sh \
    --sf <scale_factor> \
    --output-dir ./reports \
    --label <custom_name> \
    --iterations 4 \
    --query-timeout 120 \
    --compare <baseline_report_dir> \
    [query_numbers...]
```

**Output directory structure (profiled):**
```
reports/<label>_<YYYYMMDD_HHMMSS>/
  report.md        - Human-readable analysis with all metrics
  summary.json     - Machine-readable per-query metrics (profiled timings — use for analysis, not perf comparison)
  metadata.json    - Hardware, git commit, config, driver version
  comparison.md    - (if --compare used) Regression/improvement analysis
  profiles/        - All raw artifacts
    q1.sqlite, q1.nsys-rep, q1_timings.csv, q1_result.txt, ...
    summary.txt
```

**Output from non-profiled run:**
```
# run_tpch_parquet.sh:
timings_sirius_sf<SF>_q<N>.csv   - Per-query cold/hot timings (accurate, no profiling overhead)

# benchmark_and_validate.sh:
runs/<timestamp>_sf<SF>_<N>iter/
  comparison.txt   - DuckDB vs Sirius timing table with speedup ratios
  timings.csv      - Combined long-format timings (engine, query, iteration, runtime_s)
  validation.csv   - Per-query result match status (success/validation/error)
  sirius/q<N>/timings.csv  - Per-query Sirius timings
  duckdb/q<N>/timings.csv  - Per-query DuckDB timings
```

### Workflow B: Generate Report from Existing Profiles

```bash
bash test/tpch_performance/nsys_report.sh --profile-dir <path_to_profiles>
```

### Workflow C: Quick Analysis (no archival)

Use `nsys_analyze.sh` directly for quick, one-off analysis without creating a report directory.

```bash
bash test/tpch_performance/nsys_analyze.sh <path_to_profiles_or_file> [query_numbers...]
```

### Workflow D: Compare Two Existing Reports

```bash
bash test/tpch_performance/nsys_compare.sh <baseline_report_dir> <current_report_dir> [--threshold PCT]
```

Default threshold is 10%. Values beyond the threshold are flagged:
- **REGRESSION**: current is >threshold% slower
- **IMPROVED**: current is >threshold% faster
- **FIXED**: query failed in baseline but passes now
- **BROKEN**: query passed in baseline but fails now

**Important**: The timings in `summary.json` (used by `nsys_compare.sh`) are from profiled runs and include nsys overhead. These comparisons are useful for spotting large changes but should be validated with non-profiled timing runs before concluding a real regression or improvement exists. For definitive performance comparison, compare the non-profiled `timings.csv` files from `run_tpch_parquet.sh` or `benchmark_and_validate.sh`.

## Before Running

- **Ask the user** for any paths you don't know. Do NOT assume paths.
- For profiling (Workflow A without `--profile-dir`), the Sirius config must be set:
  ```bash
  export SIRIUS_CONFIG_FILE=<path_to_config>
  ```
- All paths in `profile_tpch_nsys.sh` are configurable via env vars (`DUCKDB`, `PARQUET_DIR`, `QUERY_DIR`, `OUTPUT_DIR`, `ITERATIONS`, `QUERY_TIMEOUT`).

## Analysis Sections

The report contains these sections per query:

All analysis is **scoped to the query execution window** — the time span from the first Sirius operator start to the last operator end. Init overhead (CUDA context creation, cudaHostAlloc, cuFile init) and cleanup (cudaFreeHost, pool destruction) are excluded from the main metrics and shown separately.

| Section | What it Shows |
|---------|---------------|
| **Execution Time Breakdown** | Trace duration vs query execution time vs init vs cleanup |
| **GPU Hardware** | GPU model, SM count, VRAM, compute capability |
| **NVTX Domain Summary** | Time breakdown across software layers (Sirius, libkvikio, libcudf, CCCL, cuFile) |
| **Sirius Physical Operators** | Per-operator call counts, wall times, percentages |
| **Top GPU Kernels** | Hottest kernels by total GPU time |
| **Kernel Occupancy Estimation** | Theoretical SM occupancy per kernel, limiting factor (registers/shared_mem/warps) |
| **Register Spill Analysis** | Kernels using local memory (register spilling to slow memory) |
| **GPU Kernel Time Summary** | Aggregate kernel stats, stream count, device count |
| **GPU Utilization Overview** | Kernel time as % of query execution time (not total trace) |
| **Memory Transfer Breakdown** | H2D/D2H/D2D with Pageable vs Pinned src/dst, bandwidth in GB/s |
| **CUDA Runtime API Hotspots** | Slowest CUDA API calls *during query execution only* |
| **Host Memory Allocation During Query** | Only alloc calls during runtime (init allocs excluded) |
| **Init/Cleanup Overhead** | What was excluded — cudaHostAlloc, cudaFreeHost, context creation, etc. |
| **GPU Kernel Attribution** | Maps GPU kernel time back to Sirius operators via correlation chain |
| **Top Kernels per Operator** | Which specific kernels each operator launches |
| **GPU Stream Utilization** | Per-stream busy% = kernel_time / stream_active_span |
| **Synchronization Analysis** | GPU sync wait times by type |
| **NVTX Operations by Domain** | Top operations per software layer |
| **Cross-Query Comparison** | (multi-file only) Side-by-side query overview |

## Architecture Context

### Sirius Execution Model
Sirius intercepts DuckDB query plans and offloads them to GPU via cuDF:
1. **DuckDB** parses SQL and creates a logical plan
2. **Sirius** converts it to a physical GPU plan with operators like `sirius_physical_table_scan`, `sirius_physical_hash_join`, etc.
3. **cuDF** (libcudf) provides GPU-accelerated DataFrame primitives
4. **CCCL** (CUB/Thrust) provides GPU algorithm primitives underneath cuDF
5. **CUDA kernels** execute on the GPU

### NVTX Domain Hierarchy
- **Domain 0 (Sirius)**: Physical operator execution (`sirius_physical_*::execute/sink`)
- **Domain 1 (libkvikio)**: GPU-Direct Storage I/O operations (`posix_host_read`, `task`)
- **Domain 2 (cuFile)**: cuFile handle management
- **Domain 3 (libcudf)**: cuDF operations (`aggregate`, `binary_operation`, `materialize_all_columns`, etc.)
- **Domain 4 (CCCL)**: CUB/Thrust primitives (`DeviceFor::ForEachN`, `thrust::transform`, etc.)

### Key Relationships
- **Operator -> Kernel attribution**: Correlates via `CUPTI_ACTIVITY_KIND_RUNTIME.correlationId` (links runtime API calls to kernels; runtime call timestamps fall within NVTX operator ranges)
- **Cold vs Hot**: First iteration is "cold" (I/O, JIT). Subsequent iterations are "hot" (cached).
- **Multi-stream**: Sirius uses 30-40 CUDA streams for parallelism

### Sirius Physical Operators
| Operator | Purpose |
|----------|---------|
| `table_scan` | Read parquet files via cuDF |
| `projection` | Evaluate column expressions |
| `filter` | Row filtering |
| `hash_join` | Hash-based join |
| `grouped_aggregate` | Group-by aggregation |
| `grouped_aggregate_merge` | Merge partial aggregates across partitions |
| `partition` | Data partitioning |
| `order` | ORDER BY |
| `merge_sort` | Merge sorted partitions |
| `sort_partition` / `sort_sample` | Sort-based repartitioning |
| `top_n` / `top_n_merge` | LIMIT processing |
| `concat` | Concatenation |
| `materialized_collector` / `result_collector` | Final result materialization |

## Interpretation Guide

### Occupancy
- **100%**: Maximum warps per SM. Block size and register usage fit perfectly.
- **50-100%**: Generally acceptable. Check if the limiter can be relaxed.
- **Below 50%**: Potential bottleneck. Check the `limiter` column:
  - `registers`: Kernel uses too many registers per thread. Compiler flag `--maxrregcount` could help, or algorithmic changes to reduce register pressure.
  - `shared_mem`: Shared memory per block limits active blocks. The `shmem_b` column shows the *driver-allocated* amount (aligned to SM partition granularity, often 16KB/32KB/64KB/102KB). Reducing shared memory usage or using dynamic allocation may help.
  - `warps`: Block is too large relative to SM warp capacity.
  - `hw_limit`: Hit the max blocks per SM limit (24 on Ada Lovelace).
- **Note**: Low occupancy doesn't always mean poor performance — compute-bound kernels can achieve peak throughput at low occupancy if they have high ILP (instruction-level parallelism).

### Bandwidth
- **Pinned H2D**: Expect 12-14 GB/s on PCIe 4.0 x16. Below 8 GB/s suggests contention or small transfers.
- **Pageable H2D**: Much slower (~0.05 GB/s). Indicates un-pinned host allocations — a major performance issue if significant data goes through this path.
- **D2D**: Internal GPU memory shuffles. Should achieve 100-400 GB/s depending on transfer patterns.
- **D2H**: Usually small volumes (query results). Bandwidth similar to H2D.

### GPU Utilization
- `kernel_pct_of_query`: Kernel time / query execution span (excludes init/cleanup). Values of 40-60% are typical — the remainder is CPU orchestration, sync waits, and memcpy. Below 30% suggests the GPU is starved.
- `kernel_pct_of_ops`: Kernel time / Sirius operator time. Values >100% indicate GPU kernels overlap with CPU operator orchestration (normal with async execution). Low values suggest CPU-side bottlenecks within operators.

### Host Memory Overhead
- The analysis separates init-time allocations from query-runtime allocations.
- `cudaHostAlloc` during init (10+ seconds) is a one-time cost — it's shown in the Init/Cleanup Overhead section.
- If `cudaHostAlloc` appears in the "During Query Execution" section, that's a performance issue — synchronous allocation during active queries stalls the pipeline.
- `cudaStreamSynchronize` is typically the dominant cost during query execution — it represents time the CPU waits for GPU work to complete.

### Register Spill
- `local_bytes_per_thread > 0` means the kernel exceeded the register file and is spilling to local memory (which actually resides in global/L2 memory — much slower).
- This is a red flag for performance-critical kernels. Solutions: reduce register usage, simplify kernel logic, or use `__launch_bounds__` to guide the compiler.

## What to Look For

1. **Where is query time going?** Compare operator time vs query execution span. Large gaps = sync waits, memcpy, or CPU orchestration overhead.
2. **GPU utilization**: Low kernel_pct_of_query (<30%) = GPU underutilized during query execution (CPU-bound, sync-bound, or I/O-bound).
3. **Occupancy hotspots**: Kernels with <50% occupancy that consume significant GPU time.
4. **Memory bandwidth**: Pageable transfers are orders of magnitude slower than pinned.
5. **Host allocation cost**: cudaHostAlloc often dominates CUDA API time.
6. **Cold vs Hot delta**: Large gap = I/O/JIT overhead. Small gap = compute-dominated. Use non-profiled timings for accurate cold/hot comparison.
7. **Stream utilization**: Low busy% with many streams = fine-grained parallelism. High busy% on few streams = load imbalance.
8. **Register spill**: Any kernel with local_bytes_per_thread > 0 is spilling.
9. **Operator attribution**: Which operators consume the most GPU time? Focus optimization here.
10. **OOM failures**: Queries failing with `std::bad_alloc` need memory optimization.
11. **Profiling vs actual performance**: Always validate profiled timing changes with non-profiled runs. nsys overhead can mask or exaggerate real performance differences.

## Output Format

Always present findings in a structured way:
- Start with a high-level summary (pass/fail, total times, report location)
- Identify the top 3-5 bottlenecks
- For each bottleneck, explain what it means and potential causes
- Compare cold vs hot when relevant — clearly label whether timings are from profiled or non-profiled runs
- When comparing reports, highlight regressions and improvements but note that profiled timings include nsys overhead — recommend validating with non-profiled runs if not already done
- When analyzing a single query deeply, walk through the execution timeline: I/O -> operators -> kernels -> output
- When presenting performance conclusions, always distinguish between profiled timings (for analysis) and non-profiled timings (for actual performance measurement)
