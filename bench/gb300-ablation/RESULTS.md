# GB300 SF1000 host-pin cumulative ablation results

**Correction (November 20, 2026):** The historical runs below used
`SIRIUS_PIN_COMPRESSION_TABLES=lineitem,orders` even when the stage passed
`--no-pin-compression`. That allowlist emits explicit `compression=true` pin
calls, overriding the session's false setting. The purported uncompressed
stages therefore compressed these two tables automatically. Their compression
on/off increments and all-off labels are invalid; retain the first four tables
only as records of what ran. The runner now uses an empty allowlist for
uncompressed stages. Corrected measurements follow below.

Measured on November 20, 2026 with one GB300 GPU in CDMM driver mode, Sirius
source `c082bf10014cc005867ec64e048477b0df848279`, and the SF1000 Parquet
dataset at `/mnt/datasets/tpch_parquet_sf1000`. The rebuilt extension checksum,
base config and Simpatico plan checksums, and exact generated stage-config
checksums are in [manifest.json](results/manifest.json).

Each stage ran Q1–Q22 once, in a fresh Sirius process, with per-query host
pinning and the saved [GB300 pin config](../gb300-pin/config.yaml). The benchmark
used `--profile hot`; it discarded dataset pages once at the start of each
stage, then pinned and unpinned the columns needed by each query. Only
`lineitem` and `orders` used explicit Simpatico plans in stages 2–5; stages
0–1 also compressed them through automatically selected plans.
CPU fallback was disabled. These totals sum query execution time and exclude
pinning and unpinning.

| Stage | Optimization added | 22-query total | Change from prior stage |
| ---: | :--- | ---: | ---: |
| 0 | Nominal all-off (pins compressed) | 12.381 s | — |
| 1 | Type narrowing | 11.481 s | −0.899 s (−7.3%) |
| 2 | Explicit Simpatico plans | 11.152 s | −0.330 s (−2.9%) |
| 3 | Fused filtering during compressed decode | 9.654 s | −1.498 s (−13.4%) |
| 4 | BUILD_PROBE admission | 8.476 s | −1.178 s (−12.2%) |
| 5 | Dynamic filters | 7.031 s | −1.445 s (−17.0%) |

The all-on suite was **5.350 s (43.2%) lower** than the nominal all-off suite,
but this is not the five-feature effect because stage 0 compressed pinned data.
These are cumulative, order-dependent increments, not independent main effects.
The [exact totals](results/summary.csv) and [per-query times](results/per-query.csv)
are saved with the [reproduction script](run_sf1000_host.sh).

All six stages exited successfully. The result text matched exactly for 21
queries in every stage. Q11 orders only by value, so equal-value rows can appear
in either order; its sorted result rows matched across all stages. In a separate
untimed diagnostic, Q6 logged 32 applied fused filtered decodes, Q3 selected
BUILD_PROBE twice, and Q3 published two dynamic filters. The diagnostic commands
are in [diagnose_sf1000_host.sh](diagnose_sf1000_host.sh); their timings are not
included above.

This is one pass per stage. The 2.9% increment does not measure compression
on/off because both adjacent stages compressed input. The Parquet dataset,
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
| Nominal all-off (pins compressed) | 12.400 s | — |
| Explicit Simpatico plans | 11.849 s | −0.551 s (−4.44%) |

The [paired totals](results/simpatico-only-summary.csv),
[per-query times](results/simpatico-only-per-query.csv), and
[run manifest](results/simpatico-only-manifest.json) are saved here. Both stages
exited successfully. The result text matched exactly for 21 queries; Q11 had
the same rows in a different tied-row order. Re-audit found that the nominal
baseline also compressed `lineitem` and `orders`; the 4.44% difference cannot
be attributed to turning compression on. Pinning and unpinning time is excluded
from these totals.

## Compressed pinning with the other four switches on

A fresh paired run enabled type narrowing, fused decode filtering, BUILD_PROBE
admission, and dynamic filters in both stages. The generated YAML files were
identical. The first stage was intended to pin raw data, but re-audit found it
also compressed `lineitem` and `orders`. The second used explicit Simpatico
plans for them. Each ran Q1–Q22 once in a fresh
Sirius process with per-query host pinning.

| Configuration | 22-query total | Change from first stage |
| :--- | ---: | ---: |
| Nominal compression off (pins compressed) | 7.246 s | — |
| Explicit Simpatico plans | 6.930 s | −0.316 s (−4.36%) |

The [paired totals](results/all-on-vs-simpatico-summary.csv),
[per-query times](results/all-on-vs-simpatico-per-query.csv), and
[run manifest](results/all-on-vs-simpatico-manifest.json) are saved here. Both
stages completed, and all query result rows matched. Q11 only reordered rows
with tied sort keys. Both stages compressed pins despite the nominal
uncompressed label. Query time excludes pinning and unpinning. This pair does
not measure compressed versus raw pinning.

## BUILD_PROBE-first cumulative order

A separate six-stage SF1000 host-pin run started with all five switches off,
then enabled BUILD_PROBE admission, dynamic filters, type narrowing, Simpatico
pinning, and fused filtering during decode in that order. The GB300 pin YAML,
explicit `lineitem` and `orders` plans, hot profile, and 22-query workload were
the same as the first cumulative study. Each stage used one fresh Sirius
process and one iteration per query.

| Stage | Optimization added | 22-query total | Change from prior stage |
| ---: | :--- | ---: | ---: |
| 9 | Nominal all-off (pins compressed) | 12.343 s | — |
| 10 | BUILD_PROBE admission | 11.203 s | −1.140 s (−9.2%) |
| 11 | Dynamic filters | 9.107 s | −2.096 s (−18.7%) |
| 12 | Type narrowing | 8.950 s | −0.157 s (−1.7%) |
| 13 | Explicit Simpatico plans | 8.597 s | −0.353 s (−3.9%) |
| 14 | Fused filtering during decode | 6.945 s | −1.652 s (−19.2%) |

All six stages completed. Results matched exactly for 21 queries; Q11 had the
same rows in a different tied-row order. Re-audit found that stages 9–12
compressed `lineitem` and `orders` despite their nominal uncompressed labels.
Query totals exclude pinning and unpinning. The
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

## Corrected type-narrowing-first cumulative order

The corrected six-stage SF1000 host-pin run started with all five optimizations
off, then enabled type narrowing, BUILD_PROBE admission, dynamic filters,
Simpatico pinning, and fused filtering during decode. Each stage ran Q1–Q22 once
in a fresh Sirius process with the GB300 pin config and hot profile. The first
four stages explicitly set `compression=false` on every pin. Stages 19–20 set
`compression=true` only for `lineitem` and `orders`.

| Stage | Optimization added | 22-query total | Change from prior stage |
| ---: | :--- | ---: | ---: |
| 15 | All five off | 22.141 s | — |
| 16 | Type narrowing | 14.171 s | −7.970 s (−36.0%) |
| 17 | BUILD_PROBE admission | 12.978 s | −1.193 s (−8.4%) |
| 18 | Dynamic filters | 11.662 s | −1.316 s (−10.1%) |
| 19 | Compressed pinning | 8.569 s | −3.093 s (−26.5%) |
| 20 | Fused filtering during decode | 7.005 s | −1.564 s (−18.3%) |

All six stages exited successfully. Results matched exactly for 21 queries;
Q11 had the same rows in a different tied-row order. The first four stages
logged no compressed pin events; stages 19–20 each made 29 explicit
`compression=true` pin calls for `lineitem` and `orders`, with 26 pins actually
compressing at least one chunk. The [exact totals](results/narrow-first-summary.csv)
and [per-query times](results/narrow-first-per-query.csv) are saved here with
the [run manifest](results/narrow-first-manifest.json).
Query totals exclude pinning and unpinning. This is one pass per stage, and
the incremental effects depend on the order.

## Corrected BUILD_PROBE-first comparison

To compare the requested order against the previous one, the BUILD_PROBE-first
six-stage run was repeated with the same corrected pin control, binary,
dataset, base config, and hot profile. Its first four stages explicitly pinned
every table uncompressed; stages 13–14 compressed only `lineitem` and `orders`.

| Stage | Optimization added | 22-query total | Change from prior stage |
| ---: | :--- | ---: | ---: |
| 9 | All five off | 22.152 s | — |
| 10 | BUILD_PROBE admission | 21.170 s | −0.982 s (−4.4%) |
| 11 | Dynamic filters | 14.653 s | −6.518 s (−30.8%) |
| 12 | Type narrowing | 11.716 s | −2.937 s (−20.0%) |
| 13 | Compressed pinning | 8.569 s | −3.147 s (−26.9%) |
| 14 | Fused filtering during decode | 7.033 s | −1.536 s (−17.9%) |

The type-first order measured 7.005 s with all switches on, versus 7.033 s
for BUILD_PROBE-first. The 0.028 s difference is too small to interpret from
one pass per order. The different incremental percentages show that feature
effects depend on the enabled context; they do not establish a benefit from
the order of enabling switches itself. All 22 query result sets matched across
stages, with only tied Q11 rows reordered. The [exact totals](results/join-first-corrected-summary.csv)
and [per-query times](results/join-first-corrected-per-query.csv) are saved here
with the [run manifest](results/join-first-corrected-manifest.json). Untimed
diagnostics for each corrected order found two BUILD_PROBE selections and two
dynamic-filter publications for Q3, plus 32 filtered decodes for Q6. These
diagnostic runs do not contribute to the timing tables.
