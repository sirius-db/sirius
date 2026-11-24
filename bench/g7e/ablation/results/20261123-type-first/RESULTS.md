# G7e SF1000 host-pin type-first ablation — November 23, 2026

Result date: 2026-11-23, America/Chicago. Same London g7e.8xlarge, Sirius main 26b5caba, guide SF1000 host-pinned preset, SET threads=24 and byte-verified SF1000 dataset as the earlier BUILD_PROBE-first study. This follows corrected GB300 stages 15–20.

One fresh engine per stage, one execution per query in Q1–Q22 order. Each query pins its selected columns on host, executes, then unpins. Clear caches once at stage startup; no inter-query resets, warmup, harness profile/mode override or YAML rewriting. Timing excludes startup, pin/unpin, export and independent CPU validation. Every pin explicitly disables compression in stages 0–3; stages 4–5 compress only lineitem/orders with the exact GB300 plans.

| Stage | Optimization added | Completed | Q1–Q22 sum (s) | Partial sum (s) |
|---|---|---:|---:|---:|
| 0 | All off | 0/22 | DNF | — |
| 1 | + Type narrowing | 8/22 | DNF | 24.924423 |
| 2 | + BUILD_PROBE | 8/22 | DNF | 22.314597 |
| 3 | + Dynamic filters | 18/22 | DNF | 40.405679 |
| 4 | + Compressed pinning | 22/22 | 30.713166 | — |
| 5 | + Fused decode filtering | 22/22 | 29.425129 | — |

All 78 completed timed answers matched independent CPU references with native GPU, pin and config evidence. Failed stages have no full-suite total.

- Stage 0 stopped at Q1 during host pinning: `_duckdb.Error: std::bad_alloc: out_of_memory: Not enough free blocks available in fixed_size_host_memory_resource. total_bytes: 2495610880, upstream_tracked_bytes: 2495610880, only 62 free blocks available from 2380 requested blocks.`
- Stage 1 stopped at Q9 during query execution: `_duckdb.Error: Sirius GPU execution failed: GPU pipeline task exceeded maximum retry limit (100) for original task 2086: OOM while preparing batches for processing: std::bad_alloc: out_of_memory: not enough capacity to allocate memory`
- Stage 2 stopped at Q9 during query execution: `_duckdb.Error: Sirius GPU execution failed: GPU pipeline task exceeded maximum retry limit (100) for original task 2079: OOM while preparing batches for processing: std::bad_alloc: out_of_memory: not enough capacity to allocate memory`
- Stage 3 stopped at Q19 during host pinning: `_duckdb.Error: std::bad_alloc: out_of_memory: Not enough free blocks available in fixed_size_host_memory_resource. total_bytes: 1929379840, upstream_tracked_bytes: 1929379840, only 113 free blocks available from 1840 requested blocks.`

With every other feature enabled, fused filtering changed the full 22-query sum by -4.19% (30.713166 → 29.425129 s). This is one pass, not an independent main effect or a statistical comparison.

## Common-query subset

The following sums use the same 8 queries in every populated stage: Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8. They are not full 22-query suite sums.

| Stage | Subset sum (s) | Change from previous subset stage |
|---|---:|---:|
| + Type narrowing | 24.924423 | — |
| + BUILD_PROBE | 22.314597 | -10.47% |
| + Dynamic filters | 18.283847 | -18.06% |
| + Compressed pinning | 9.408622 | -48.54% |
| + Fused decode filtering | 8.686649 | -7.67% |

## Identical end states across orders

| Configuration | Earlier BUILD_PROBE-first sum (s) | New type-first sum (s) |
|---|---:|---:|
| compressed_pin | 30.634477 | 30.713166 |
| fused_decode | 31.678285 | 29.425129 |

Both end-state YAMLs match the earlier order byte for byte. One pass per order cannot establish an advantage from the order itself. Intermediate increments depend on which features are enabled.

Untimed native diagnostics confirmed 2 BUILD_PROBE selections and 2 dynamic-filter publications in Q3, and 68 applied fused filtered decodes in Q6. Their query times are excluded from all sums.

The live S3 bucket and instance regions both matched London. Every staged local NVMe file hash matched the generation manifest before timing; S3 inventory and manifest were unchanged afterward. Pin setup reads the local mirror because this revision cannot pin S3 directly; measured queries use resident host data. The unchanged guide preset leaves disk spilling disabled. No EBS fallback or source rebuild was used.

All source/effective configs, feature/pin controls, independent CPU/GPU answers, full native logs and Quent traces are retained in the locally downloaded archive. The archive and every payload file were hash-verified. Earlier results are preserved separately. The instance remains running.

## Saved reproduction records

- [Protocol and reproducer](../../README.md); use `--order type-first`
- [All stage sums](summary.csv), [per-query comparison](per-query.csv), and [all validated timings](timings.csv)
- [Identical end-state per-query comparison across orders](order-comparison.csv)
- [Exact stage configs](configs/), [session SQL](session.sql), [feature/pin plan](stage-plan.json), and [manifest](manifest.json)
- [Full proofs](summary.json), [validation and failures](validation.json), [native path checks](diagnostics.json)
- [Earlier BUILD_PROBE-first results](../20261123/RESULTS.md), preserved unchanged

Large logs, answers and Quent traces remain in the local archive identified by SHA256 in the manifest.
