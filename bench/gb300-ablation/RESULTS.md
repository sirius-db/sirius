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

## Compressed pinning with the other four switches on

A fresh paired run enabled type narrowing, fused decode filtering, BUILD_PROBE
admission, and dynamic filters in both stages. The generated YAML files were
identical. The first stage pinned raw data; the second used the same explicit
Simpatico plans for `lineitem` and `orders`. Each ran Q1–Q22 once in a fresh
Sirius process with per-query host pinning.

| Configuration | 22-query total | Change from uncompressed |
| :--- | ---: | ---: |
| All other switches on, compressed pinning off | 7.246 s | — |
| All switches on, compressed pinning on | 6.930 s | −0.316 s (−4.36%) |

The [paired totals](results/all-on-vs-simpatico-summary.csv),
[per-query times](results/all-on-vs-simpatico-per-query.csv), and
[run manifest](results/all-on-vs-simpatico-manifest.json) are saved here. Both
stages completed, and all query result rows matched. Q11 only reordered rows
with tied sort keys. The raw-pin stage logged no pin compression; the
compressed-pin stage logged 29 compression events for `lineitem` and `orders`.
Query time excludes pinning and unpinning. The fused-decode switch remains on
in both YAMLs, but its compressed-input path cannot run with raw pins, so this
pair measures the combined execution effect of compressed pinning and the
enabled consumers of compressed input. This is one pass per condition.

## BUILD_PROBE-first cumulative order

A separate six-stage SF1000 host-pin run started with all five switches off,
then enabled BUILD_PROBE admission, dynamic filters, type narrowing, Simpatico
pinning, and fused filtering during decode in that order. The GB300 pin YAML,
explicit `lineitem` and `orders` plans, hot profile, and 22-query workload were
the same as the first cumulative study. Each stage used one fresh Sirius
process and one iteration per query.

| Stage | Optimization added | 22-query total | Change from prior stage |
| ---: | :--- | ---: | ---: |
| 9 | All five off | 12.343 s | — |
| 10 | BUILD_PROBE admission | 11.203 s | −1.140 s (−9.2%) |
| 11 | Dynamic filters | 9.107 s | −2.096 s (−18.7%) |
| 12 | Type narrowing | 8.950 s | −0.157 s (−1.7%) |
| 13 | Compressed pinning | 8.597 s | −0.353 s (−3.9%) |
| 14 | Fused filtering during decode | 6.945 s | −1.652 s (−19.2%) |

All six stages completed. Results matched exactly for 21 queries; Q11 had the
same rows in a different tied-row order. Stages 9–12 logged no pin compression,
while stages 13–14 each logged 29 compression events for `lineitem` and
`orders`. Query totals exclude pinning and unpinning. The
[exact totals](results/join-first-summary.csv) and
[per-query times](results/join-first-per-query.csv) are saved here, along with
the [run manifest](results/join-first-manifest.json). Untimed diagnostics
confirmed two BUILD_PROBE selections and two dynamic-filter publications for
Q3, plus 32 filtered decodes for Q6. These diagnostic runs are excluded from
the timing table.

This is one pass per stage. The increments depend on their order and should
not be treated as independent effects. In particular, the final 19.2% step
measures enabling fused filtering on top of compressed pinning, BUILD_PROBE,
dynamic filters, and type narrowing. It does not describe Simpatico compression
alone.
