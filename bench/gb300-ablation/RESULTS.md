# GB300 SF1000 host-pin cumulative ablation results

Measured on November 20, 2026 with one GB300 GPU in CDMM driver mode, Sirius
source `c082bf10014cc005867ec64e048477b0df848279`, and the SF1000 Parquet
dataset at `/mnt/datasets/tpch_parquet_sf1000`. The rebuilt extension checksum,
base config and Simpatico plan checksums, and exact generated stage-config
checksums are in [manifest.json](results/manifest.json).

Each stage ran Q1–Q22 once, in a fresh Sirius process, with per-query host
pinning and the saved [GB300 pin config](../gb300-pin/config.yaml). The benchmark
used `--profile hot`; it discarded dataset pages once at the start of each
stage, then pinned and unpinned the columns needed by each query. Only
`lineitem` and `orders` used explicit Simpatico compression in stages 2–5.
CPU fallback was disabled. These totals sum query execution time and exclude
pinning and unpinning.

| Stage | Optimization added | 22-query total | Change from prior stage |
| ---: | :--- | ---: | ---: |
| 0 | All five off | 12.381 s | — |
| 1 | Type narrowing | 11.481 s | −0.899 s (−7.3%) |
| 2 | Compressed pinning | 11.152 s | −0.330 s (−2.9%) |
| 3 | Fused filtering during compressed decode | 9.654 s | −1.498 s (−13.4%) |
| 4 | BUILD_PROBE admission | 8.476 s | −1.178 s (−12.2%) |
| 5 | Dynamic filters | 7.031 s | −1.445 s (−17.0%) |

The all-on suite was **5.350 s (43.2%) lower** than the all-off suite, or
**1.76×** as many completed suite runs per unit of query time. These are
cumulative, order-dependent increments; they are not independent main effects.
The [exact totals](results/summary.csv) and [per-query times](results/per-query.csv)
are saved with the [reproduction script](run_sf1000_host.sh).

All six stages exited successfully. The result text matched exactly for 21
queries in every stage. Q11 orders only by value, so equal-value rows can appear
in either order; its sorted result rows matched across all stages. In a separate
untimed diagnostic, Q6 logged 32 applied fused filtered decodes, Q3 selected
BUILD_PROBE twice, and Q3 published two dynamic filters. The diagnostic commands
are in [diagnose_sf1000_host.sh](diagnose_sf1000_host.sh); their timings are not
included above.

This is one pass per stage. The 2.9% compressed-pinning increment in particular
needs repeated measurements before treating it as stable. The Parquet dataset,
historical binary, and full benchmark logs remain on the GB300 machine rather
than in this PR.

## Simpatico pinning alone

A separate paired run kept type narrowing, fused filtering, BUILD_PROBE admission,
and dynamic filters off in both stages. The generated YAML files were identical;
only `--pin-compression` differed. The same SF1000 host-pinned setup ran Q1–Q22
once per stage in fresh Sirius processes, with compression explicitly limited to
`lineitem` and `orders`.

| Configuration | 22-query total | Change from all off |
| :--- | ---: | ---: |
| All five off | 12.400 s | — |
| Simpatico pinning only | 11.849 s | −0.551 s (−4.44%) |

The [paired totals](results/simpatico-only-summary.csv),
[per-query times](results/simpatico-only-per-query.csv), and
[run manifest](results/simpatico-only-manifest.json) are saved here. Both stages
exited successfully. The result text matched exactly for 21 queries; Q11 had
the same rows in a different tied-row order. The baseline logged no pin
compression, while the Simpatico-only stage logged 29 compression events for
`lineitem` and `orders`. Pinning and unpinning time is excluded from these
totals. This is one paired pass, so the 4.44% gain should be repeated to assess
its stability.
