# G7e SF1000: all on except compressed pinning — November 23, 2026

London g7e.8xlarge, Sirius main `26b5caba`, guide host-pinned preset and `SET threads=24`. Type narrowing, BUILD_PROBE, dynamic filtering and fused scan/filter are enabled. Every table is explicitly pinned with `compression=false`. The effective YAML is byte-identical to the previous all-on compressed case; per-call pin compression is the sole changed control. This corresponds to GB300 stage 7.

One fresh engine for the stage, Q1–Q22 once, pin/unpin per query and cache clear at startup only. No warmup or harness profile/mode override; YAML is preserved. Timings exclude pin/unpin and independent CPU validation.

Completed **18/22** queries with sum **40.386595 s**. **The full-suite result is DNF; this is a partial sum.**

Stopped at **Q19 during host pinning**:

`_duckdb.Error: std::bad_alloc: out_of_memory: Not enough free blocks available in fixed_size_host_memory_resource. total_bytes: 1929379840, upstream_tracked_bytes: 1929379840, only 113 free blocks available from 1840 requested blocks.`

## Comparison on matching queries

These sums include only Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18. Earlier comparison runs are one pass each, so differences include run-to-run variation.

| Case | Matching-query sum (s) | Full Q1–Q22 sum (s) |
|---|---:|---:|
| New: all on, uncompressed pins | 40.386595 | DNF |
| Previous: all on, compressed pins | 23.195274 | 29.425129 |
| Previous: raw pins, fused switch off | 40.405679 | DNF |

The fused scan/filter switch is on in the new case, but uncompressed pinned inputs give fused decode no compressed data to operate on. Zero compressed chunks were observed. Host pinning fails before query execution when the host pool cannot fit a query's raw selected columns; enabling the fused switch does not reduce that footprint. Allocation error byte counts describe the failed request, not total pool capacity.

All 18 completed answers passed independent CPU, GPU-execution, pin and config checks. The single Quent context contains 18 timed query executions. There were no extra diagnostic runs. The dataset hashes and live S3/EC2 region matched the earlier study; the pinned engine and saved preset were unchanged. Pin setup uses the byte-identical same-region dataset on local NVMe. The guide preset leaves disk spilling disabled; no EBS fallback was added.

[Config](config.yaml), [session SQL](session.sql), [feature plan](stage-plan.json), [explicit pin audit](pin-audit.json), [per-query timings](per-query.csv), [sums](summary.csv), [validation](validation.json), [full proofs](summary.json) and [provenance](manifest.json) are saved here. Full native logs, answers and Quent traces are in the locally retained archive identified by SHA256 in the manifest. All payload hashes were verified after download. The instance remains running.
