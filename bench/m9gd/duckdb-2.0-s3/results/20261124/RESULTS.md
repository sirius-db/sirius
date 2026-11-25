# DuckDB 2.0.0 / M9gd results — November 24, 2026

Date: America/Chicago. The SF10000 job finished after midnight UTC on November 25, still November 24 in Chicago. SF1000 ran first, followed by a sequential SF3000 then SF10000 campaign on the same retained `m9gd.24xlarge` instance in `us-east-1a`.

Amazon Linux 2023 ARM64, 96 CPU threads, default DuckDB memory limit **296.7 GiB**. Parquet is read directly from matching Virginia S3 datasets. External-file and Parquet-metadata caches are explicitly enabled; spill uses the three-disk local NVMe RAID0, with EBS root excluded.

Each scale uses **Q1 Q1 → close DuckDB/process, sync/drop OS caches → Q2 Q2 → … Q22 Q22**. There are 22 fresh processes and 22 OS cache clears per scale, including a clear before Q1. Each process executes its query exactly twice, without a warmup or intervening cache reset. Timings measure `execute()` only; fetch/export, startup, binding, preflight footer counts and cache inspection are excluded. Q11 uses the correct scale factor.

## Complete iteration sums

| Scale | Iteration 1 sum (s) | Iteration 2 sum (s) | Validated measurements |
| --- | ---: | ---: | ---: |
| SF1000 | 400.439935 | 66.544754 | 44 / 44 |
| SF3000 | 1318.310405 | 383.664585 | 44 / 44 |
| SF10000 | 6171.235734 | 4437.306026 | 44 / 44 |

Every displayed sum is one complete Q1–Q22 iteration. No per-query minima are mixed. All **132 timed answers independently passed CPU validation**, including legal Q3/Q18 LIMIT ties. All 66 process/cache lifecycles and effective cache settings were checked.

## Per-query times

All times are seconds. Iteration 2 retains each query's caches; it is not guaranteed to fit in RAM at SF10000.

| Query | SF1k iter 1 | SF1k iter 2 | SF3k iter 1 | SF3k iter 2 | SF10k iter 1 | SF10k iter 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | 14.194554 | 1.567631 | 41.384387 | 4.322500 | 156.867925 | 41.109288 |
| Q2 | 4.237449 | 0.543241 | 10.238960 | 1.073319 | 30.237262 | 2.873095 |
| Q3 | 28.059528 | 1.877007 | 73.411370 | 5.639576 | 354.532175 | 349.000397 |
| Q4 | 16.455639 | 1.280906 | 50.168207 | 3.996534 | 165.470730 | 15.717431 |
| Q5 | 24.186424 | 2.004206 | 72.047162 | 5.509634 | 255.770564 | 116.905242 |
| Q6 | 14.935765 | 0.591098 | 44.203743 | 1.462658 | 143.644106 | 16.821482 |
| Q7 | 27.567176 | 2.115428 | 83.350983 | 6.084566 | 283.872012 | 155.406793 |
| Q8 | 22.818014 | 2.210562 | 62.877845 | 6.055997 | 268.070854 | 163.906380 |
| Q9 | 25.515061 | 7.876207 | 114.618830 | 124.960948 | 796.487253 | 813.532047 |
| Q10 | 24.092294 | 4.843854 | 73.614852 | 11.613511 | 347.516669 | 363.600469 |
| Q11 | 2.520437 | 0.545349 | 5.906706 | 1.140987 | 19.089395 | 4.889023 |
| Q12 | 21.816086 | 1.134802 | 67.955476 | 3.478181 | 235.773683 | 35.672742 |
| Q13 | 8.995541 | 7.603165 | 30.983591 | 20.711166 | 124.198538 | 104.094844 |
| Q14 | 21.229499 | 1.637762 | 67.494942 | 5.085683 | 278.258730 | 197.476654 |
| Q15 | 20.424986 | 1.113974 | 66.579278 | 4.033915 | 236.009667 | 139.098483 |
| Q16 | 2.550688 | 0.877175 | 5.664773 | 4.644719 | 20.205841 | 14.151957 |
| Q17 | 17.402488 | 1.229126 | 49.146641 | 3.701665 | 264.224216 | 112.971201 |
| Q18 | 21.851199 | 11.194882 | 63.470592 | 40.614780 | 808.746093 | 792.437587 |
| Q19 | 21.838397 | 1.206595 | 69.989440 | 3.086505 | 290.805444 | 98.243119 |
| Q20 | 22.769278 | 1.233629 | 72.552349 | 3.849248 | 255.604820 | 136.100042 |
| Q21 | 33.096078 | 12.704580 | 183.417019 | 119.382130 | 809.574874 | 753.290584 |
| Q22 | 3.883354 | 1.153575 | 9.233260 | 3.216365 | 26.274884 | 10.007168 |
| **SUM** | **400.439935** | **66.544754** | **1318.310405** | **383.664585** | **6171.235734** | **4437.306026** |

## Reproduction and provenance

The [replay instructions](../../README.md) and [portable launcher](../../reproduce.sh) stage the [exact measured scripts](../../measured/) with a unique spill-directory prefix. The original scripts are preserved byte for byte and checked against source hashes. The launcher was added after measurement; it has not rerun these saved results.

Each per-scale folder ([SF1000](sf1000/), [SF3000](sf3000/), [SF10000](sf10000/)) includes:

- `per-query.csv` and `summary.json`: both timings and sums, result row counts and answer SHA256s.
- `config.json`: exact protocol and effective settings from every query process.
- `queries.sql` and `sql-sha256.json`: all executed SQL, including the scale-specific Q11 fraction.
- `final-verification.json`, `answer-validation.json` and `lifecycle.json`: independent answer and process/cache-clear proofs.
- `cache-evidence.json`: native external-file cache bytes before and after each query iteration.
- `provenance.json` and `environment.txt`: instance, actual bucket region, manifest/inventory/row counts, physical NVMe/RAID evidence, package versions and postflight checks.

The datasets and measured sources remained unchanged before/after each run. S3 Parquet inventories contain 90, 263 and 868 files at SF1000, SF3000 and SF10000. No input was copied across regions or staged on NVMe.

At SF10000, Q9, Q18 and Q21 each take roughly 13 minutes per execution. Native cache snapshots show that cache residency depends on the working set: for example, Q9 had zero resident external-file bytes after iteration 1, while Q11 retained all its external-file bytes. Enabling a cache does not pin the full input in RAM. Local spill-space observations during Q9 reached approximately 1.6 TB; these samples are not a measured peak.

The [archive receipts](full-archives.json) identify the locally retained full logs/answers/source/proof bundles by filename, size and SHA256. All 236 SF1000 payload hashes and all 228 payload hashes in each larger-scale archive were verified after download. Large raw archives and independent CPU reference files are retained locally rather than committed to Git. The checked-in [manifest](manifest.json) fingerprints the selected records; `verify_saved.py` audits them without running queries.

The instance was left running for further experiments as requested. These are read-only query experiments, not official TPC-H benchmark submissions.
