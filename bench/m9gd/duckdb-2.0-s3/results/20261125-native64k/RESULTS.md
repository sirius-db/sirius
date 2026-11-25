# Native DuckDB measurements — November 25, 2026

DuckDB 2.0.0 (`d54cbbe3df17e8e5ce68312f483b4155836112aa`) on the same retained `m9gd.24xlarge` in North Virginia, reading same-region S3 Parquet. SF1000/SF3000/SF10000 reuse the [November 24 baseline](../20261124/RESULTS.md) objects; SF100/SF300 use the existing Virginia prefixes documented below. Amazon Linux 2023 kernel `6.12.115-native64k-exp-20261125` uses **64 KiB (65536-byte) pages**. The Release source build has **`-DNATIVE_ARCH=ON`** and verified **`-mcpu=native`** compilation flags. The saved build and kernel evidence records the exact compiler, commits, module SHA and config changes.

The query pattern remains **Q1 Q1 → close process, sync/drop OS caches → Q2 Q2 → … Q22 Q22**, including a cache clear before Q1, with no extra warmup and no reset inside a pair. Both external-file and Parquet-metadata caches are enabled, 96 threads are used, and spill is on the verified three-disk local NVMe RAID0. Times measure execute only.

| Scale | Iteration 1 sum (s) | Iteration 2 sum (s) | Memory and provenance |
| --- | ---: | ---: | --- |
| SF100 | 33.992618 | 8.177413 | 296.7 GiB throughout; one successful paired suite |
| SF300 | 67.280689 | 16.110636 | 296.7 GiB throughout; one successful paired suite |
| SF1000 | 375.285785 | 50.114462 | 296.7 GiB throughout; one successful paired suite |
| SF3000 | 1241.309344 | 325.166412 | 296.7 GiB throughout; one successful paired suite |
| SF10000 | 5496.954199 | 3788.048400 | Assembled from original timings and retries; **Q12 uses 256 GiB**, other queries 296.7 GiB |

All 220 selected native-build answers in the dated snapshot passed independent CPU validation, including legal LIMIT ties. The SF3000 selection improves on the prior sums of 1318.310405 / 383.664585 s. The SF10000 selection improves on the prior 6171.235734 / 4437.306026 s, but **is not a successful full suite under one memory configuration**. It must not be presented as a uniform-budget benchmark or as an isolated page-size/native-build effect. No per-query minima are selected.

## SF10000 retries and memory exception

The original 296.7 GiB run was killed by Linux global OOM during Q8 iteration 2 and Q12 iteration 1, leaving no Q12 iteration 2. An unchanged-settings retry recovered Q8 iteration 2 in **155.454695 s**; its first execution was required to establish the cache state, and its new first timing was not substituted. Q12 again OOMed at 296.7 GiB, with about 326 GiB anonymous process RSS. DuckDB's managed-memory budget does not bound all resident process/kernel memory.

With the explicitly approved **256 GiB (274877906944-byte)** Q12 limit, both executions succeeded in **231.059908 / 57.270933 s**. All other selected measurements retain the exact original **318617491865-byte (296.7 GiB)** limit. The 64 KiB kernel would otherwise raise the detected default budget to 301.6 GiB, so these limits are set explicitly. The original successful values are preserved and only missing values are filled; the SF10000 totals combine three sessions. The earlier failed attempts and the original dated baseline remain available.

## Saved evidence

[Per-query times](per-query.csv), [SF100](sf100.json), [SF300](sf300.json), [SF1000](sf1000.json), selected [SF3000](sf3000.json) and [SF10000](sf10000.json) records name each source phase and memory budget. `phases/` retains settings, query/process/cache journals, source/data/region/storage fingerprints and independent answer-validation records for both original campaigns and both retry phases. [Build evidence](build.json), [kernel evidence](kernel.json), exact [executed adapter/build scripts](executed-scripts.json), [failures](failures.json) and [archive receipts](full-archives.json) are retained. Raw logs/answers and the exact wheel remain in verified local archives, outside Git. These snapshots describe the measured setup; the historical scripts retain host-specific paths and are not instance-management commands.

The original 4 KiB kernel remains the default for future reboots; the measured 64 KiB kernel is active for the current boot. Native building and kernel/page size changed together, so this experiment does not isolate their individual contributions.

Run the offline selection audit (no AWS access or query execution):

```bash
pixi run python bench/m9gd/duckdb-2.0-s3/verify_selected.py
```

## SF1000 native / 64 KiB follow-up

The same native module and 64 KiB kernel completed all 44 measurements, with one fresh process and OS cache clear per query pair. Both iteration sums are **375.285785 / 50.114462 s**, versus the 4 KiB wheel baseline's **400.439935 / 66.544754 s**. All 44 new answers independently passed CPU validation. The 296.7 GiB budget, cache flags, same-region dataset and timing scope were held fixed. Both new sums improve, so the selected result index now uses this native paired run. No per-query minima are combined.

## SF100 and SF300 additions

Both scales completed all 22 queries twice, with 22 fresh processes and OS cache clears per scale. All **88 new answers independently passed retained CPU references**, including expanded Q3/Q18 LIMIT ties. These are new measurements with no prior 4 KiB CPU runs at these scales; no improvement ratio is claimed. Their 296.7 GiB memory budget and remaining native-build protocol match SF1000.

Inputs are the existing **Virginia** prefixes `datasets/tpch_sf100_plus2201_20261020/` (14 files, 27,679,630,327 bytes) and `datasets/tpch_sf300_plus2201_20261020/` (31 files, 84,134,567,274 bytes), generated by `cdcf74def0072f94bf1886667e8d2ac51feb8721`. All eight table counts were checked against live Parquet footers before timing. Complete per-file generation records, object sizes/ETags and the actual S3 bucket region were verified, then source/data fingerprints were rechecked after timing. The datasets were neither regenerated nor copied across regions. [Input specifications and generation manifests](small-scale-inputs.json) retain the physical-input details; results from other hardware may use different Parquet layouts.

The native adapter reuses the exact measured SF1000 controller/worker and shared SQL/timing modules, changing only the scale-specific input, Q11 rendering parameter and controller scale label (in addition to the existing native memory/page-size/spill adapter). The measured native module, 96 threads, cache settings and query/process ordering are unchanged. The [executed script snapshot](executed-scripts.json) and per-phase replay plans record these edits and source hashes.
