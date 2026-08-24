---
name: optimization-advisor
description: >
  Use this skill to find exactly which source code to optimize for better GPU performance. Maps nsys
  profile hotspots to specific Sirius source files and functions, classifies bottlenecks as GPU-bound,
  CPU-bound, or sync-bound, and recommends actionable code changes. Trigger when the user wants to
  know what to optimize, where to focus coding effort, or wants source-level optimization guidance.
  This skill focuses on actionable source code targets — for generating performance reports and
  measurements, use profile-analyzer instead.
---

# Sirius Code Optimization Advisor

You are identifying optimization targets in Sirius, a GPU-accelerated SQL query engine built on DuckDB. You analyze nsys profile data and map hotspots back to specific source code functions so developers know exactly where to focus.

## Profiling Overhead Warning

**nsys profiling adds measurable overhead to query execution times.** When the user wants to validate that an optimization actually improved performance, always recommend running queries both WITH and WITHOUT profiling:

1. **Profiled run** (`nsys_report.sh`, which delegates to `performance_test.py --nsys-profile`) → Provides GPU analysis data (kernels, operators, occupancy, memory) to understand *why* performance changed
2. **Non-profiled run** (`performance_test.py`) → Provides accurate cold/hot timings to confirm the optimization actually helped

Never claim an optimization improved or regressed performance based solely on profiled timings. The profiled data tells you what changed internally; the non-profiled timings tell you whether it actually got faster.

## STEP 1: Confirm Profiles with the User (MANDATORY)

**Before running any analysis, you MUST confirm with the user which profiles to analyze.** Do NOT proceed to STEP 2 until the user has explicitly confirmed.

1. **List available profiles.** Run this to show the user what's available:
   ```bash
   ls -d reports/*/
   ```

2. **Present the options to the user.** Show them the available report directories and ask:
   - Which report directory (or profile path) do they want to analyze?
   - Do they want all queries, or specific ones (e.g., `1 3 10`)?
   - Or do they want to generate new profiles first?

3. **Wait for explicit confirmation.** Do NOT run `nsys_hotspots.sh` until the user has told you which profiles and which queries to use. Even if there is only one report directory, confirm it with the user first.

## STEP 2: Run the Analysis

Once the user has told you what to analyze, run `nsys_hotspots.sh`:

```bash
# Analyze all queries in a profile directory
bash test/tpch_performance/nsys_hotspots.sh <path_to_profiles>

# Analyze specific queries
bash test/tpch_performance/nsys_hotspots.sh <path_to_profiles> 1 3 10

# Analyze a single query
bash test/tpch_performance/nsys_hotspots.sh <path_to_q1.sqlite>

# If the user gave a report directory, use its profiles/ subdirectory
bash test/tpch_performance/nsys_hotspots.sh reports/<label>/profiles/

# Save the report
bash test/tpch_performance/nsys_hotspots.sh <path_to_profiles> > optimization_guide.md
```

### Generating New Profiles (if requested)

If the user wants fresh profiles instead of existing ones:

```bash
# Step 1: Profile (requires SIRIUS_CONFIG_FILE)
bash test/tpch_performance/nsys_report.sh --sf <scale_factor>

# Step 2: Analyze the generated profiles
bash test/tpch_performance/nsys_hotspots.sh reports/<generated_dir>/profiles/
```

### Validating Optimizations (non-profiled timing run)

After identifying and implementing an optimization, run queries WITHOUT profiling to get accurate timings — use the same `performance_test.py` runner the `benchmark` skill uses:

```bash
export SIRIUS_CONFIG_FILE=<path_to_config>

# Sirius-only timing (accurate cold/hot without nsys overhead)
pixi run python test/tpch_performance/performance_test.py \
    --input <parquet_dir> --engine gpu --iterations <N> [--queries 1,3,6-10]

# Full DuckDB vs Sirius benchmark + result validation
pixi run python test/tpch_performance/performance_test.py \
    --input <parquet_dir> --engine both --iterations <N> --validation
```

Compare the resulting `<bench>/csv/runtimes.csv` against a previous non-profiled baseline to confirm the optimization actually improved wall-clock performance. Then run a new profiled analysis (`nsys_report.sh`) to understand what changed internally.

**Data source (parquet or duckdb).** `--input` is a parquet directory (`--data-source parquet`, the default) or a single `.duckdb` file (`--data-source duckdb`); `--pin gpu|host` works for both. When a scan is the hotspot, note which path is exercised: parquet `read_parquet` → the parquet GPU scan, vs duckdb `seq_scan` → the GPU-native scan (`src/op/scan/duckdb_native_gpu_ingestible.cpp`). They are different source files, so the optimization target differs by source.

## Analysis Sections

The report contains these sections per query:

| Section | What It Answers |
|---------|-----------------|
| **Source File Mapping** | Maps every NVTX operator name to its source file path |
| **Hottest Operators (wall time)** | Which functions consume the most elapsed time? |
| **Hottest Operators (GPU time)** | Which functions consume the most GPU kernel time? |
| **Operator Efficiency** | Wall time vs GPU time — identifies CPU-bound vs GPU-bound bottlenecks |
| **Top Kernels per Operator** | Which GPU kernels dominate each operator? |
| **Occupancy Bottlenecks** | Kernels with <50% occupancy, grouped by operator with limiter info |
| **Sync & Wait Overhead** | Synchronization time attributed to operators |
| **Memory Transfer Hotspots** | Data movement volume and bandwidth per operator |
| **Sequential Execution Chains** | Back-to-back operators on the same thread — parallelism opportunities |
| **Stream Utilization** | Stream count, per-stream busy%, GPU utilization during query |
| **Operator Concurrency** | Thread-level parallelism — how many threads execute operators |

For multi-query analysis, the report includes:
- **Cross-Query Optimization Priority Matrix** — operators ranked by total wall time with source files and efficiency
- **Per-Query Performance Summary** — exec time, GPU time, utilization, streams, sync
- **Top Optimization Targets** — top 5 operators with actionable recommendations

## Architecture Context

### Operator-to-Source Mapping

All Sirius physical operators follow a deterministic naming pattern:
- NVTX event name: `sirius_physical_<name>::execute` or `sirius_physical_<name>::sink`
- Source file: `src/op/sirius_physical_<name>.cpp`
- Header file: `src/include/op/sirius_physical_<name>.hpp`

Special cases:
- `sirius_physical_materialized_collector` -> `src/op/sirius_physical_result_collector.cpp`
- `sirius_physical_left_delim_join` / `sirius_physical_right_delim_join` -> `src/op/sirius_physical_delim_join.cpp`
- `sirius_physical_streaming_limit` -> `src/op/sirius_physical_limit.cpp`

### Execution Model

Sirius executes query plans as pipelines:
1. **Pipeline executor** has a thread pool and stream pool (`src/pipeline/gpu_pipeline_executor.cpp`)
2. **Within each task**, operators execute sequentially with `stream.synchronize()` after each operator (`src/pipeline/gpu_pipeline_task.cpp:218`)
3. **Multiple tasks** run concurrently across threads, each with its own CUDA stream
4. **Within each operator**, cuDF may use multiple streams internally

This means:
- **Inter-task parallelism**: Multiple pipeline tasks run on different threads/streams
- **Intra-operator parallelism**: cuDF operations may launch many kernels across streams
- **Inter-operator serialization**: `stream.synchronize()` between operators within a task

### Optimization Dimensions

| Dimension | What to Look For | Threshold |
|-----------|-----------------|-----------|
| **Wall time** | Operators consuming >20% of query time | Top 3 operators |
| **GPU efficiency** | Wall time vs GPU time ratio | <20% = CPU-bound |
| **Occupancy** | Theoretical SM occupancy | <50% with significant GPU time |
| **Sync overhead** | cudaStreamSynchronize / cudaDeviceSynchronize time | >30% of operator wall time |
| **Memory** | Pageable transfers, high D2D volume | Pageable = major issue |
| **Parallelism** | Sequential chains, low stream count | Gap <1ms = serialized |

## Interpretation Guide

### Bottleneck Classification

- **GPU-BOUND** (efficiency >= 50%): The operator spends most time in GPU kernels. Optimize kernel occupancy, memory access patterns, or algorithmic efficiency.
- **MIXED** (efficiency 20-50%): Both GPU compute and CPU/sync overhead are significant. Check for unnecessary synchronization, small kernel launches, or memory allocation during execution.
- **CPU-BOUND** (efficiency < 20%): The operator spends most time on CPU orchestration, sync waits, or memory operations. Look for `cudaStreamSynchronize`, `cudaHostAlloc`, or host-side data preparation.

### What to Do After Identifying Targets

1. **Read the source file** listed in the Source File Mapping
2. **Look at the `execute()` method** — find NVTX-instrumented code section
3. **Check Top Kernels** — identify which cuDF/CCCL calls generate the dominant kernels
4. **For CPU-bound operators** — search for sync calls, memory allocation, or complex host-side logic
5. **For GPU-bound operators** — check occupancy limiters and consider kernel fusion or algorithmic changes
6. **For sequential chains** — consider whether adjacent operators could share streams or be fused

## Output Format

Always present findings in a structured way:
- Start with the Cross-Query Priority Matrix (the most actionable view)
- Focus on the top 3-5 optimization targets
- For each target, explain what makes it slow and what to investigate
- Link operators to source files so the developer can navigate directly
- When comparing across queries, highlight operators that consistently dominate
- When presenting optimization recommendations, remind the user to validate improvements with a non-profiled `performance_test.py` run — profiled timings include nsys overhead and do not reflect actual query performance
