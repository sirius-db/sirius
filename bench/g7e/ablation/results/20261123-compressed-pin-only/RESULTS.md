# G7e SF1000: compressed pinning only - November 23, 2026

London g7e.8xlarge, Sirius main `26b5caba`, guide host-pinned preset and `SET threads=24`. Only compressed host pinning is enabled among the five ablation controls. Type narrowing, BUILD_PROBE, dynamic filtering and fused scan/filter are explicitly disabled. Both BUILD_PROBE byte limits are zero. Effective YAML matches the earlier all-off stage byte for byte. Only `lineitem` and `orders` use `compression=true`, with the checked Simpatico plans; every other table pin explicitly uses `compression=false`. This corresponds to GB300 stage 6.

One fresh engine for the stage, Q1-Q22 once, host pin/unpin per query and cache clear at stage startup only. No warmup or harness profile/mode override; YAML is preserved. Timings exclude pin/unpin and independent CPU validation. Other engine settings remain at the pinned guide recipe: this isolates the five ablation controls, not every optimization in the engine.

Completed **22/22** queries. Full Q1-Q22 sum: **51.898669 s**.

## Comparison

Both sums include every query, Q1-Q22. These are single passes; run-to-run variation remains.

| Case | Full Q1-Q22 sum (s) |
|---|---:|
| New: compressed pinning only | 51.898669 |
| Previous: all five on | 29.425129 |

The all-off baseline failed while pinning Q1, so no full-suite speedup over all-off can be calculated. Any incomplete case has no full-suite total.

All 22 completed answers passed independent CPU, GPU, host-pin and config checks. The single Quent context retains all 22 successful query executions. Native logs contain 27 pin events with compressed chunks; compression is restricted to lineitem/orders. No extra diagnostic queries were run. The live bucket and EC2 regions match London. Every staged local NVMe input hash matched the same SF1000 manifest; inventory and manifest were unchanged afterward. The engine was not rebuilt. The guide preset leaves disk spilling disabled; no EBS fallback was added.

[Config](config.yaml), [session SQL](session.sql), [feature plan](stage-plan.json), [pin audit](pin-audit.json), [native compressed chunks](compressed-chunks.json), [per-query timings](per-query.csv), [sums](summary.csv), [validation](validation.json), [full proofs](summary.json) and [provenance](manifest.json) are saved here. Full native logs, answers and Quent traces are retained in the locally downloaded archive identified by SHA256 in the manifest. Every payload hash was verified. The instance remains running.
