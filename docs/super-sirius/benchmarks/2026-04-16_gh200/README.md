# GH200 Benchmark Artifacts — 2026-04-16

Full data behind [../gh200_sf100_clickbench_20260416.md](../gh200_sf100_clickbench_20260416.md).

Branch: `feature/gpu-scan-duckdb-api` at commit with full self-continuation +
mmap auto-detect. Binary: `build/release/duckdb`. Config:
`test_datasets/bench_config.yaml` (4 scan threads, 8 pipeline threads,
80% GPU memory limit, 100MB scan batch target).

Machine: NVIDIA GH200 480GB (96GB GPU + 480GB host unified memory),
Grace 72-core ARM CPU, Ubuntu. CUDA 12.8.

## Directory layout

```
raw/
  final/              - Main report (GPU vs CPU, cold + warm, SF100 + CB)
    tpch_cold.csv     - TPC-H SF100 cold timings
    tpch_warm.csv     - TPC-H SF100 warm iter 2 timings
    cb_cold.csv       - ClickBench 100shard cold timings
    cb_warm.csv       - ClickBench 100shard warm iter 2 timings
    tpch_cold_<mode>_q<n>.txt  - Individual query outputs for correctness verification
    tpch_warm_<mode>.txt       - Full session output (44 queries + iter 2 extraction)
    cb_cold_<mode>_q<n>.txt    - Individual ClickBench query outputs
    cb_warm_<mode>.txt         - Full ClickBench session output

  cold_3variants/     - Cold performance: mmap + prefault vs lazy vs no-mmap
    <A|B|C>_{tpch,cb}_q<n>.txt
      A = mmap + MADV_POPULATE_READ  (PCIe path, SIRIUS_MMAP_PREFAULT=1)
      B = mmap + lazy faults         (GH200 path, SIRIUS_MMAP_PREFAULT=0)
      C = no mmap, DuckDB Pin()      (SIRIUS_DISABLE_MMAP=1)

  warm_3variants/     - Warm performance for the same 3 variants
    <tpch|cb>_warm_v<A|B|C>.txt

nsys/                 - nsys profile of Q1 SF100 warm × 2 iterations
  q1_sf100_warm_x2iter.nsys-rep    - Open in Nsight Systems GUI
  q1_sf100_warm_x2iter.sqlite      - SQL-queryable events (17 MB)
  summary.txt                      - H2D / D2H / kernel / sync totals
  top_kernels.txt                  - Top 25 GPU kernels by total time
  sirius_kernels.txt               - Our kernels only
```

## Headline numbers

### Warm (single-session, iter 2)

| Workload | GPU | CPU | GPU/CPU |
|----------|----:|----:|--------:|
| TPC-H SF100           | **5.33s** | 7.08s | **1.33x** (GPU wins, 15/22 queries) |
| ClickBench 100shard   | 5.00s | 2.54s | 0.51x (CPU wins — expected; wide-table scans) |

### Cold (per-process, page cache dropped before each query — averaged)

Tested across 3 mmap/prefault variants:

| Variant | TPC-H cold (avg/q) | ClickBench cold (avg/q) |
|---------|-------------------:|------------------------:|
| A: mmap + MADV_POPULATE_READ  (current PCIe path) | 16.4s | 19.7s |
| B: mmap + lazy faults          (**GH200 default** with auto-detect) | 2.4s | 3.3s |
| C: no mmap, BufferManager::Pin | 1.1s | 1.0s |

**Speedup of B vs A on GH200: 6.9x TPC-H, 6x ClickBench** — no warm regression.
**Projected variant D** (`BlockManager::GetDirectBlockPointer()` API, not yet
upstreamed in DuckDB): variant-C cold speed + variant-B warm speed.

## Correctness

All 22 TPC-H queries produce matching results against CPU reference (modulo
floating-point rounding in the last digit of doubles, which is expected for
GPU vs CPU aggregations). See `raw/final/tpch_cold_{gpu,cpu}_q*.txt` for
per-query outputs.

## nsys analysis of Q1 SF100 warm × 2 iterations

Totals across both iterations:

| Category | Time | Count |
|----------|-----:|------:|
| GPU kernels (total) | 1451 ms | 17,744 launches |
| H2D transfers | 76.9 ms | 11,854 calls @ 151 GB/s (11.6 GB total) |
| D2H transfers | 1.6 ms | 440 calls |
| cudaStreamSynchronize | 4313 ms | 16,895 calls @ 255 us avg |

Top 5 kernels (full list in `nsys/top_kernels.txt`):

| Kernel | Time | % of GPU kernel total | Owner |
|--------|-----:|---------------------:|:------|
| cudf hash_group_by single_pass_shmem_aggregate | 812 ms | 56% | cuDF |
| cudf hash_group_by mapping_indices_kernel | 174 ms | 12% | cuDF |
| sirius kernel_decode_bitpacking_batched<long> | 97 ms | 7% | **us** |
| cub BatchMemcpyKernel | 57 ms | 4% | cuDF internal |
| cub scan DeviceScanKernel | 31 ms | 2% | cuDF internal |

**Our code is ~11% of total GPU kernel time.** The remaining 89% is cuDF
internals (hash_group_by is the big one for Q1 specifically) and CUB.

`sync_total / kernel_total = 4313 / 1451 = 3.0x` — **sync overhead is 3x
larger than actual compute.** Most syncs are inside cuDF/RMM, not our
pipeline code. See `nsys/summary.txt`.

## How to reproduce

```bash
# Build
cd /home/ubuntu/sirius
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pixi run --environment default make release

# Warm TPC-H SF100 (iter 2 = warm numbers)
> /tmp/run.sql
echo ".timer on" >> /tmp/run.sql
for iter in 1 2; do
  for q in $(seq 1 22); do
    cat test/tpch_performance/tpch_queries/gpu/q${q}.sql >> /tmp/run.sql
  done
done
SIRIUS_CONFIG_FILE=test_datasets/bench_config.yaml \
  build/release/duckdb -unsigned -readonly -noheader \
  test_datasets/tpch_sf100.duckdb < /tmp/run.sql

# Cold TPC-H SF100 (drop page cache before each)
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
SIRIUS_CONFIG_FILE=test_datasets/bench_config.yaml \
  build/release/duckdb -unsigned -readonly -noheader \
  test_datasets/tpch_sf100.duckdb -c \
  "CALL gpu_execution('...')"

# Force PCIe-style prefault (variant A):      SIRIUS_MMAP_PREFAULT=1
# Force lazy faults (variant B, default GH200): SIRIUS_MMAP_PREFAULT=0
# Force DuckDB Pin path (variant C):           SIRIUS_DISABLE_MMAP=1
```

## Pull to local machine

```bash
git fetch origin feature/gpu-scan-duckdb-api
git checkout origin/feature/gpu-scan-duckdb-api -- \
  docs/super-sirius/benchmarks/2026-04-16_gh200/
```
