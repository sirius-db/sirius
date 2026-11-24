# G7e SF1000 host-pin cumulative ablation — November 23, 2026

Result date: 2026-11-23, America/Chicago. London g7e.8xlarge, Sirius main 26b5caba, unchanged guide SF1000 host-pinned preset and SET threads=24. This follows the corrected GB300 BUILD_PROBE-first order.

One fresh engine per stage; Q1–Q22 once; pin/unpin the query's columns on host before/after execution. Clear caches at stage startup only. No warmup and no saved-YAML rewriting. Query times exclude pin/unpin/startup/export/CPU checks. Compression is explicitly false for every table in stages 0–3, then true only for lineitem/orders in stages 4–5 using the exact GB300 plans.

| Stage | Optimization added | Completed queries | Q1–Q22 sum (s) | Partial sum (s) |
|---|---|---:|---:|---:|
| 0 | All off | 0/22 | DNF | — |
| 1 | + BUILD_PROBE | 0/22 | DNF | — |
| 2 | + Dynamic filters | 0/22 | DNF | — |
| 3 | + Type narrowing | 18/22 | DNF | 40.348261 |
| 4 | + Compressed pinning | 22/22 | 30.634477 | — |
| 5 | + Fused decode filtering | 22/22 | 31.678285 | — |

Enabling fused decode filtering after compressed pinning changes the complete suite sum by +3.41% (30.634477 → 31.678285 s). This is one pass per stage and an order-dependent increment, not an independent main effect.

Stages 0–2 failed while pinning Q1. Stage 3 completed Q1–Q18, then failed while pinning Q19. All four failed with fixed_size_host_memory_resource allocation errors. These are pin-capacity failures before the affected timed query, not zero-second queries. No complete all-off baseline exists, so an overall five-feature speedup cannot be calculated. All 62 completed timed answers independently matched CPU references and have native GPU/pin/config evidence. The incomplete stage is not compared against complete suite sums.

For the common Q1–Q18 subset only: type narrowing 40.348261 s; + compressed pins 24.082372 s; + fused filtering 25.446269 s. These are 18-query sums, not full TPC-H suite sums.

Inputs are the regenerated London SF1000 dataset: staged local NVMe copies were checked against every source-manifest file hash before timing. The live S3 bucket and instance regions matched, and inventory/manifest stayed unchanged. Pinned queries read resident host data. The unchanged guide preset leaves disk spilling disabled; no EBS spill fallback was added.

Separate untimed debug runs use Q3 with stage 3 settings for BUILD_PROBE/dynamic filtering, and Q6 with stage 5 settings for fused decode. Both answers and GPU execution were checked; their timings are excluded from every sum above.

The native compression coverage log uses Parquet paths rather than logical table aliases. The first collector stopped after stage 4 on that label mismatch. Its normalization was fixed; stages 0–4 were recollected from saved output, and only stage 5 was newly executed. No timed query was repeated. The original collector/package and collection error remain archived.

Full CPU/GPU answers, configs, pin audits, native logs, Quent profiles, and both collector versions are preserved in a local archive; archive and every payload hash were verified after download. Configs, per-query comparisons, all stage sums and provenance are saved with this report. The instance remains running.

Untimed native path checks: 2 BUILD_PROBE selections and 2 dynamic-filter publications in Q3; 68 applied fused filtered decodes in Q6. These confirm engagement despite the single-pass suite time increasing.

## Saved reproduction records

- [Run protocol and reproducer](../../README.md)
- [Stage sums](summary.csv), [per-query comparison](per-query.csv), and [all validated measurements](timings.csv)
- [Exact stage configs](configs/), [session SQL](session.sql), [stage feature/pinning plan](stage-plan.json), and [run manifest](manifest.json)
- [Full validation proofs](summary.json), [failure details](validation.json), and [untimed native path evidence](diagnostics.json)

Large logs, result answers and Quent payloads remain in the locally retained, hash-verified archive identified in the manifest.
