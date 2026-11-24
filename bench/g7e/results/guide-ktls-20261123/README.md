# London G7e guide results, November 23, 2026

Result date: **2026-11-23, America/Chicago**. UTC timestamps in the retained
artifacts may fall on November 24.

Sirius main `26b5caba3e2c96290fdc63d4cf40ac42572eaab8` on one London
`g7e.8xlarge` running Amazon Linux 2023, with an RTX PRO 6000 Blackwell GPU and
CUDA 13.5. All populated cells are complete, independently CPU-validated Q1–Q22
**sums in seconds** from one iteration. We select the best complete iteration,
never a sum of mixed per-query minima.

| Scale factor | Cold sum (s) | Hot sum (s) | Host-pinned sum (s) |
| --- | ---: | ---: | ---: |
| SF100 | 37.473543 | 8.718906 | 2.691681 |
| SF300 | 53.731252 | 16.496861 | 7.642228 |
| SF1000 | 118.017201 | 45.809267 | 26.244293 |
| SF3000 | 362.445844 | 273.535501 | — |
| SF10000 | 1906.310348 | 2048.194464 | — |

— means no complete validated suite is selected. Older incomplete SF10000 hot
measurements and Q1-only tests are excluded. Pinning setup time is excluded.

## Configs and reproduction

The [versioned recipes and runner](../../guide-26b5caba/README.md) preserve the
measured settings and process boundaries. SF100/SF300/SF1000 use the guide's
CPU-budget preset; SF1000 adds `SET threads=24`. SF3000 and SF10000 use the
SF10000 balanced-large-scans preset plus `SET threads=24`. Large-scale hot changes
only eviction from idle to LRU. No retired experimental environment flags are
set. Quent is enabled in separate effective config copies.

Cold uses a fresh engine and cleared OS/Sirius caches for every measurement.
Hot uses one fresh engine per query and retains it for two consecutive
measurements, with no cache clear or extra warmup between them. Host-pinned
prepares only that query's selected columns on host before the pair, then
unpins and exits. At this engine revision, host-pin setup uses byte-identical
same-region S3 files staged on local NVMe because native pinning rejects S3
URIs. Cold and hot read S3 directly. kTLS was verified for each direct-S3
worker. Enabled spilling was verified on EC2 local NVMe instance storage.

## All iteration sums

Each number below sums all 22 queries from that iteration. Hot and pinned
iterations are paired per query, rather than running Q1–Q22 as two separate
passes. Both timing columns remain saved even when the second is slower.

| Scale factor | Profile | Iteration 1 sum (s) | Iteration 2 sum (s) |
| --- | --- | ---: | ---: |
| SF100 | Cold | 38.364677 | 37.473543 |
| SF100 | Hot | 37.379796 | 8.718906 |
| SF100 | Host-pinned | 33.634397 | 2.691681 |
| SF300 | Cold | 57.164075 | 53.731252 |
| SF300 | Hot | 53.281879 | 16.496861 |
| SF300 | Host-pinned | 14.178257 | 7.642228 |
| SF1000 | Cold | 119.584266 | 118.017201 |
| SF1000 | Hot | 140.075371 | 45.809267 |
| SF1000 | Host-pinned | 38.024536 | 26.244293 |
| SF3000 | Cold | 362.445844 | — |
| SF3000 | Hot | 371.765623 | 273.535501 |
| SF10000 | Cold | 1915.140082 | 1906.310348 |
| SF10000 | Hot | **2048.194464** | **2121.381237** |

SF3000 cold completed one iteration. The two SF10000 hot columns sum to
**4169.575701 s** across all 44 measurements. All 44 CPU answers, GPU execution
boundaries, config checks and kTLS checks passed. The 22 fresh query processes
produced 22 Quent contexts containing 44 query executions.

## Saved artifacts

- [summary.csv](summary.csv): best complete iteration sum for each populated cell.
- [iteration-sums.csv](iteration-sums.csv): all 25 complete iteration sums,
  with the selected columns marked explicitly.
- [per-query.csv](per-query.csv): all 550 validated timings from the contributing
  cases, including both iterations and explicit selected-iteration markers.
- [manifest.json](manifest.json): engine/generator revisions, selected iteration
  IDs, exact config hashes, source-summary hashes and full archive checksums.
- [sf10000-hot/](sf10000-hot/): the complete new hot summary with all 44 CPU/GPU
  proofs, validation results, source/data/storage fingerprints and collector-fix
  provenance. The existing versioned balanced hot YAML is unchanged.

The full logs, CPU and GPU answers, configs, environment records, validation
proofs and Quent payloads remain in the locally retained `.aws-exp` archives.
Their archive IDs, sizes and SHA256 values are recorded in the manifest;
large trace bundles and raw datasets are not committed to Git.

The earlier 506 timings came from the original EC2 campaigns, whose script
hashes and raw output are retained. The new 44 SF10000 hot timings use the
published portable adapter. Its first worker completed Q1 before collection
failed because `collect.py` lacked `import json`. The one-import fix in
`e58e21f7` was tested and published; Q1's measurements were retained, then
Q2–Q22 continued with the same engine and config. No query was repeated or
skipped. Both runner versions and the original collection error are preserved
in the archive. All 955 payload files were hash-verified after download.

Earlier custom G7e snapshots remain retained locally outside this guide-only
publication. Existing GB300 results keep their original dates and provenance.
