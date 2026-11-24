# London G7e guide results, November 23

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
| SF10000 | 1906.310348 | — | — |

— means no complete validated suite is selected. Older incomplete SF10000 hot
measurements and Q1-only tests are excluded. Pinning setup time is excluded.

## Configs and reproduction

The [versioned recipes and runner](../../guide-26b5caba/README.md) preserve the
measured settings and process boundaries. SF100/SF300/SF1000 use the guide's
CPU-budget preset; SF1000 adds `SET threads=24`. SF3000 and SF10000 use the
SF10000 balanced-large-scans preset plus `SET threads=24`. SF3000 hot changes
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

SF3000 cold completed one iteration. SF3000 hot sums are **371.765623 s** and
**273.535501 s**; the sum across all 44 measurements is **645.301124 s**.
SF10000 cold sums are **1915.140082 s** and **1906.310348 s**. The older SF3000
custom-config sum of 980.376540 s and CPU-budget-guide sum of 1041.159627 s
remain in the local archives; they are not the selected SF3000 result here.

## Saved artifacts

- [summary.csv](summary.csv): best complete iteration sum for each populated cell.
- [per-query.csv](per-query.csv): all 506 validated timings from the contributing
  cases, including both iterations and explicit selected-iteration markers.
- [manifest.json](manifest.json): engine/generator revisions, selected iteration
  IDs, exact config hashes, source-summary hashes and full archive checksums.

The full logs, CPU and GPU answers, configs, environment records, validation
proofs and Quent payloads remain in the locally retained `.aws-exp` archives.
Their archive IDs, sizes and SHA256 values are recorded in the manifest;
large trace bundles and raw datasets are not committed to Git.

The portable adapter was checked locally with CPU-only tests and dry runs.
These selected timings came from the original EC2 campaigns, whose script
hashes and raw output are retained; they are not a GPU rerun of the portable
adapter. Earlier checked-in G7e snapshots and all GB300 results are unchanged.
