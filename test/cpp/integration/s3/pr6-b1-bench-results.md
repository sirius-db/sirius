# B1 Phase 2 — bench results

**Branch / SHA:** feature-S3datasource-sql-surface @ acb84209 + B1 bench harness working tree
**CI host:** NVIDIA GeForce RTX 3060, 595.71.05
**Date:** 2026-05-21
**SF10 lineitem object size:** 2223320375 bytes (~2.07 GiB)

## Raw iteration data

| query | prewarm | iteration | wall_clock_ms | bytes_read_total | fsmr_borrows_total | hit_count_total | hit_after_wait_total | partial_miss_count_total | full_miss_count_total | range_miss_count_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|count|ON|1|2556.58|2224196819|0|8|1|0|0|27|
|count|ON|2|2498.48|2224196819|0|8|1|0|0|27|
|count|ON|3|2507.38|2224196819|0|8|1|0|0|27|
|count|ON|4|2516.45|2224196819|0|8|1|0|0|27|
|count|ON|5|2500.89|2224196819|0|8|1|0|0|27|
|q1|ON|1|2865.36|2224196819|0|8|1|0|0|27|
|q1|ON|2|3028.87|2224196819|0|8|1|0|0|27|
|q1|ON|3|2780.84|2224196819|0|8|1|0|0|27|
|q1|ON|4|2781.99|2224196819|0|8|1|0|0|27|
|q1|ON|5|3255.13|2224196819|0|8|1|0|0|27|
|join|ON|1|2543.27|2229525614|0|9|2|0|3|51|
|join|ON|2|2547.34|2229525614|0|9|2|0|3|51|
|join|ON|3|2542.92|2229525614|0|9|2|0|3|51|
|join|ON|4|2542.61|2229525614|0|9|2|0|3|51|
|join|ON|5|2533.46|2229525614|0|9|2|0|3|51|
|count|OFF|1|4301.90|2223406958|8|0|0|0|0|35|
|count|OFF|2|4291.99|2223406958|8|0|0|0|0|35|
|count|OFF|3|4291.20|2223406958|8|0|0|0|0|35|
|count|OFF|4|4313.53|2223406958|8|0|0|0|0|35|
|count|OFF|5|4319.85|2223406958|8|0|0|0|0|35|
|q1|OFF|1|4786.41|2223406958|8|0|0|0|0|35|
|q1|OFF|2|4544.50|2223406958|8|0|0|0|0|35|
|q1|OFF|3|4472.39|2223406958|8|0|0|0|0|35|
|q1|OFF|4|4565.07|2223406958|8|0|0|0|0|35|
|q1|OFF|5|4625.18|2223406958|8|0|0|0|0|35|
|join|OFF|1|4321.36|2228733345|9|0|0|0|3|60|
|join|OFF|2|4316.89|2228733345|9|0|0|0|3|60|
|join|OFF|3|4341.89|2228733345|9|0|0|0|3|60|
|join|OFF|4|4454.48|2228733345|9|0|0|0|3|60|
|join|OFF|5|4308.23|2228733345|9|0|0|0|3|60|

## Per-query results

### count(*)

| metric | prewarm ON median [min, max] | prewarm OFF median [min, max] | OFF / ON ratio |
|---|---:|---:|---:|
|wall_clock_ms|2507.38 [2498.48, 2556.58]|4301.90 [4291.20, 4319.85]|1.716|
|bytes_read_total|2224196819 [2224196819, 2224196819]|2223406958 [2223406958, 2223406958]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|0 [0, 0]|8 [8, 8]|n/a|
|hit_count_total|8 [8, 8]|0 [0, 0]|0.000|
|hit_after_wait_total|1 [1, 1]|0 [0, 0]|0.000|
|partial_miss_count_total|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|0 [0, 0]|0 [0, 0]|n/a|
|range_miss_count_total|27 [27, 27]|35 [35, 35]|1.296|

### q1 — TPC-H Q1 shape (narrowed-date filter)

| metric | prewarm ON median [min, max] | prewarm OFF median [min, max] | OFF / ON ratio |
|---|---:|---:|---:|
|wall_clock_ms|2865.36 [2780.84, 3255.13]|4565.07 [4472.39, 4786.41]|1.593|
|bytes_read_total|2224196819 [2224196819, 2224196819]|2223406958 [2223406958, 2223406958]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|0 [0, 0]|8 [8, 8]|n/a|
|hit_count_total|8 [8, 8]|0 [0, 0]|0.000|
|hit_after_wait_total|1 [1, 1]|0 [0, 0]|0.000|
|partial_miss_count_total|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|0 [0, 0]|0 [0, 0]|n/a|
|range_miss_count_total|27 [27, 27]|35 [35, 35]|1.296|

### join — lineitem×orders

| metric | prewarm ON median [min, max] | prewarm OFF median [min, max] | OFF / ON ratio |
|---|---:|---:|---:|
|wall_clock_ms|2542.92 [2533.46, 2547.34]|4321.36 [4308.23, 4454.48]|1.699|
|bytes_read_total|2229525614 [2229525614, 2229525614]|2228733345 [2228733345, 2228733345]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|0 [0, 0]|9 [9, 9]|n/a|
|hit_count_total|9 [9, 9]|0 [0, 0]|0.000|
|hit_after_wait_total|2 [2, 2]|0 [0, 0]|0.000|
|partial_miss_count_total|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|3 [3, 3]|3 [3, 3]|1.000|
|range_miss_count_total|51 [51, 51]|60 [60, 60]|1.176|

## Observations

- count: bytes_read_total OFF/ON=1.000, wall_clock_ms OFF/ON=1.716, prewarm-ON cache hit rate=22.86%, dominant miss ON=range_miss=135, dominant miss OFF=range_miss=175, FSMR borrow count nonzero on both branches=no.
- q1: bytes_read_total OFF/ON=1.000, wall_clock_ms OFF/ON=1.593, prewarm-ON cache hit rate=22.86%, dominant miss ON=range_miss=135, dominant miss OFF=range_miss=175, FSMR borrow count nonzero on both branches=no.
- join: bytes_read_total OFF/ON=1.000, wall_clock_ms OFF/ON=1.699, prewarm-ON cache hit rate=14.29%, dominant miss ON=range_miss=255, dominant miss OFF=range_miss=300, FSMR borrow count nonzero on both branches=no.
- `fsmr_borrows_total=0` on prewarm ON means the scan is served through the cache path and bypasses `s3_ioctx::device_read_io`; prewarm OFF uses the FSMR-staged S3 path (8-9 borrows/run).

## Phase 3 decision trigger

- If prewarm ON saves >= 10% wall-clock on at least one query AND hit rate > 50%: Phase 3 = direction A (fix cache hit alignment).
- If prewarm ON saves < 10% OR hit rate ~= 0%: Phase 3 = direction C (flip default to OFF, tighten byte-budget guard).
- Trigger evaluation from this run: mixed signal: prewarm ON saves wall-clock, but cache hit rate stays below 50%; Phase 3 should review before choosing A vs C.
