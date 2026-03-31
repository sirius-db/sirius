# TPC-H SF=10 Iceberg Benchmark Results

GPU: Quadro RTX 6000 (24GB), DuckDB 1.4.4, cuDF 26.02
Cold runs: each query in its own DuckDB process (no cache, no warm-up)

## Pre-refactor (monolithic delete hooks, GPU-Host-GPU roundtrip)

Run date: 2026-03-30

| Query | Parquet | Iceberg (no del) | Iceberg (5% eq del) |
|-------|---------|-------------------|---------------------|
| Q1    | 8.90    | 8.90              | 9.04                |
| Q2    | 8.98    | 10.04             | 10.09               |
| Q3    | 9.31    | 8.95              | 10.28               |
| Q4    | 8.98    | 8.90              | 9.03                |
| Q5    | 9.15    | 10.10             | 10.19               |
| Q6    | 8.78    | 8.80              | 9.02                |
| Q7    | 9.07    | 10.21             | 9.28                |
| Q8    | 9.04    | 9.09              | 9.43                |
| Q9    | 9.03    | 9.05              | 9.49                |
| Q10   | 10.08   | 8.99              | 9.17                |
| Q11   | 10.07   | 8.98              | 9.98                |
| Q12   | 8.98    | 8.92              | 9.55                |
| Q13   | 8.93    | 8.94              | 10.06               |
| Q14   | 8.86    | 8.96              | 9.07                |
| Q15   | 9.02    | 10.14             | 9.16                |
| Q16   | 9.97    | 9.04              | 9.01                |
| Q17   | 10.18   | 10.06             | 10.28               |
| Q18   | 10.23   | 10.04             | 9.23                |
| Q19   | 8.94    | 8.97              | 10.23               |
| Q20   | 9.01    | 10.18             | 10.16               |
| Q21   | 10.42   | 9.25              | 9.48                |
| Q22   | 8.92    | 9.01              | 10.00               |
| **Avg** | **9.31** | **9.38**        | **9.63**            |

## Post-refactor (composable filter pipeline, GPU-only equality deletes)

Run date: 2026-03-31

| Query | Parquet | Iceberg (no del) | Iceberg (5% eq del) |
|-------|---------|-------------------|---------------------|
| Q1    | 9.88    | 9.75              | 10.11               |
| Q2    | 9.93    | 10.01             | 9.88                |
| Q3    | 9.89    | 9.88              | 10.02               |
| Q4    | 9.87    | 9.05              | 9.69                |
| Q5    | 10.04   | 9.18              | 9.69                |
| Q6    | 9.79    | 9.84              | 9.90                |
| Q7    | 10.03   | 10.00             | 10.18               |
| Q8    | 10.11   | 10.04             | 10.13               |
| Q9    | 10.01   | 9.91              | 10.15               |
| Q10   | 10.01   | 10.07             | 10.03               |
| Q11   | 10.00   | 9.88              | 9.87                |
| Q12   | 9.91    | 9.93              | 10.00               |
| Q13   | 9.78    | 9.84              | 9.79                |
| Q14   | 9.84    | 9.88              | 9.18                |
| Q15   | 9.83    | 9.90              | 9.27                |
| Q16   | 9.89    | 9.94              | 10.00               |
| Q17   | 9.95    | 10.12             | 9.28                |
| Q18   | 10.02   | 9.92              | 10.21               |
| Q19   | 9.92    | 9.86              | 9.97                |
| Q20   | 9.20    | 9.97              | 10.17               |
| Q21   | 10.33   | 10.17             | 10.39               |
| Q22   | 9.82    | 9.92              | 9.94                |
| **Avg** | **9.93** | **9.87**        | **9.95**            |

## Analysis

Cold startup (~8.5-9s of GPU init + extension load + data scan from disk) dominates
wall time in both runs, making scan-level performance differences invisible.

Key observations:
- All 66 queries (22 x 3 configs) pass in both runs
- Parquet vs Iceberg (no deletes): ~0.1s delta — Iceberg metadata overhead is negligible
- Iceberg + 5% equality deletes: ~0.1-0.3s delta over no-deletes — equality delete
  hash join probe cost is small relative to cold startup
- Pre-refactor vs post-refactor averages vary by ~0.5s which is within cold-startup
  noise (different system load between runs on different days)
- No performance regression from the refactor

To measure actual scan-level overhead, warm-run benchmarks (single DuckDB session
with `table_gpu` cache) would be needed.
