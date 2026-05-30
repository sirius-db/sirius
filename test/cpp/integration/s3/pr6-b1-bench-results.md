# B1 Phase 3a — bench results

**Branch / SHA:** feature-S3datasource-sql-surface @ a5530dd2
**CI host:** NVIDIA GeForce RTX 3060, 595.71.05
**Date:** 2026-05-21
**SF10 lineitem object size:** 2223320375 bytes (~2.07 GiB)

## Raw iteration data

| query | config | iteration | wall_clock_ms | bytes_read_total | fsmr_borrows_total | hit_count_total | hit_after_wait_total | partial_miss_count_total | full_miss_count_total | range_miss_count_total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|count_star|cache_off|1|4360.16|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|count_star|cache_off|2|4260.00|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|count_star|cache_off|3|4210.82|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|count_l_orderkey|cache_off|1|4321.03|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|count_l_orderkey|cache_off|2|4302.82|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|count_l_orderkey|cache_off|3|4265.88|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|q1|cache_off|1|4612.56|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|q1|cache_off|2|4583.05|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|q1|cache_off|3|4582.70|2224196819|8|n/a|n/a|n/a|n/a|n/a|
|join|cache_off|1|4390.99|2229525614|9|n/a|n/a|n/a|n/a|n/a|
|join|cache_off|2|4841.90|2229525614|9|n/a|n/a|n/a|n/a|n/a|
|join|cache_off|3|4421.25|2229525614|9|n/a|n/a|n/a|n/a|n/a|
|count_star|cache_on_prewarm_on|1|2484.17|2224196819|0|8|1|0|0|27|
|count_star|cache_on_prewarm_on|2|2473.24|2224196819|0|8|1|0|0|27|
|count_star|cache_on_prewarm_on|3|2487.58|2224196819|0|8|1|0|0|27|
|count_l_orderkey|cache_on_prewarm_on|1|2504.14|2224196819|0|8|1|0|0|27|
|count_l_orderkey|cache_on_prewarm_on|2|2504.22|2224196819|0|8|1|0|0|27|
|count_l_orderkey|cache_on_prewarm_on|3|2497.62|2224196819|0|8|1|0|0|27|
|q1|cache_on_prewarm_on|1|2797.62|2224196819|0|8|1|0|0|27|
|q1|cache_on_prewarm_on|2|2800.06|2224196819|0|8|1|0|0|27|
|q1|cache_on_prewarm_on|3|2812.46|2224196819|0|8|1|0|0|27|
|join|cache_on_prewarm_on|1|2549.11|2229525614|0|9|2|0|3|51|
|join|cache_on_prewarm_on|2|2527.93|2229525614|0|9|2|0|3|51|
|join|cache_on_prewarm_on|3|2540.48|2229525614|0|9|2|0|3|51|
|count_star|cache_on_prewarm_off|1|4203.74|2223406958|8|0|0|0|0|35|
|count_star|cache_on_prewarm_off|2|4273.60|2223406958|8|0|0|0|0|35|
|count_star|cache_on_prewarm_off|3|4251.39|2223406958|8|0|0|0|0|35|
|count_l_orderkey|cache_on_prewarm_off|1|4201.55|2223406958|8|0|0|0|0|35|
|count_l_orderkey|cache_on_prewarm_off|2|4182.70|2223406958|8|0|0|0|0|35|
|count_l_orderkey|cache_on_prewarm_off|3|4283.25|2223406958|8|0|0|0|0|35|
|q1|cache_on_prewarm_off|1|4573.75|2223406958|8|0|0|0|0|35|
|q1|cache_on_prewarm_off|2|4566.04|2223406958|8|0|0|0|0|35|
|q1|cache_on_prewarm_off|3|4495.71|2223406958|8|0|0|0|0|35|
|join|cache_on_prewarm_off|1|4332.54|2228733345|9|0|0|0|3|60|
|join|cache_on_prewarm_off|2|4321.33|2228733345|9|0|0|0|3|60|
|join|cache_on_prewarm_off|3|4393.83|2228733345|9|0|0|0|3|60|

## Cache OFF (production default)

| query | wall_clock_ms median [min, max] | bytes_read / object_size | fsmr_borrows_total median [min, max] |
|---|---:|---:|---:|
|count(*)|4260.00 [4210.82, 4360.16]|1.00 [1.00, 1.00]|8 [8, 8]|
|count(l_orderkey)|4302.82 [4265.88, 4321.03]|1.00 [1.00, 1.00]|8 [8, 8]|
|q1 — TPC-H Q1 shape (narrowed-date filter)|4583.05 [4582.70, 4612.56]|1.00 [1.00, 1.00]|8 [8, 8]|
|join — lineitem×orders|4421.25 [4390.99, 4841.90]|1.00 [1.00, 1.00]|9 [9, 9]|

## Per-query results

### count(*)

| metric | cache OFF median [min, max] | cache ON + prewarm ON median [min, max] | cache ON + prewarm OFF median [min, max] | cache ON prewarm OFF/ON ratio |
|---|---:|---:|---:|---:|
|wall_clock_ms|4260.00 [4210.82, 4360.16]|2484.17 [2473.24, 2487.58]|4251.39 [4203.74, 4273.60]|1.711|
|bytes_read_total|2224196819 [2224196819, 2224196819]|2224196819 [2224196819, 2224196819]|2223406958 [2223406958, 2223406958]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|8 [8, 8]|0 [0, 0]|8 [8, 8]|n/a|
|hit_count_total|n/a|8 [8, 8]|0 [0, 0]|0.000|
|hit_after_wait_total|n/a|1 [1, 1]|0 [0, 0]|0.000|
|partial_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|range_miss_count_total|n/a|27 [27, 27]|35 [35, 35]|1.296|

### count(l_orderkey)

| metric | cache OFF median [min, max] | cache ON + prewarm ON median [min, max] | cache ON + prewarm OFF median [min, max] | cache ON prewarm OFF/ON ratio |
|---|---:|---:|---:|---:|
|wall_clock_ms|4302.82 [4265.88, 4321.03]|2504.14 [2497.62, 2504.22]|4201.55 [4182.70, 4283.25]|1.678|
|bytes_read_total|2224196819 [2224196819, 2224196819]|2224196819 [2224196819, 2224196819]|2223406958 [2223406958, 2223406958]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|8 [8, 8]|0 [0, 0]|8 [8, 8]|n/a|
|hit_count_total|n/a|8 [8, 8]|0 [0, 0]|0.000|
|hit_after_wait_total|n/a|1 [1, 1]|0 [0, 0]|0.000|
|partial_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|range_miss_count_total|n/a|27 [27, 27]|35 [35, 35]|1.296|

### q1 — TPC-H Q1 shape (narrowed-date filter)

| metric | cache OFF median [min, max] | cache ON + prewarm ON median [min, max] | cache ON + prewarm OFF median [min, max] | cache ON prewarm OFF/ON ratio |
|---|---:|---:|---:|---:|
|wall_clock_ms|4583.05 [4582.70, 4612.56]|2800.06 [2797.62, 2812.46]|4566.04 [4495.71, 4573.75]|1.631|
|bytes_read_total|2224196819 [2224196819, 2224196819]|2224196819 [2224196819, 2224196819]|2223406958 [2223406958, 2223406958]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|8 [8, 8]|0 [0, 0]|8 [8, 8]|n/a|
|hit_count_total|n/a|8 [8, 8]|0 [0, 0]|0.000|
|hit_after_wait_total|n/a|1 [1, 1]|0 [0, 0]|0.000|
|partial_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|range_miss_count_total|n/a|27 [27, 27]|35 [35, 35]|1.296|

### join — lineitem×orders

| metric | cache OFF median [min, max] | cache ON + prewarm ON median [min, max] | cache ON + prewarm OFF median [min, max] | cache ON prewarm OFF/ON ratio |
|---|---:|---:|---:|---:|
|wall_clock_ms|4421.25 [4390.99, 4841.90]|2540.48 [2527.93, 2549.11]|4332.54 [4321.33, 4393.83]|1.705|
|bytes_read_total|2229525614 [2229525614, 2229525614]|2229525614 [2229525614, 2229525614]|2228733345 [2228733345, 2228733345]|1.000|
|bytes_read / object_size|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.000|
|fsmr_borrows_total|9 [9, 9]|0 [0, 0]|9 [9, 9]|n/a|
|hit_count_total|n/a|9 [9, 9]|0 [0, 0]|0.000|
|hit_after_wait_total|n/a|2 [2, 2]|0 [0, 0]|0.000|
|partial_miss_count_total|n/a|0 [0, 0]|0 [0, 0]|n/a|
|full_miss_count_total|n/a|3 [3, 3]|3 [3, 3]|1.000|
|range_miss_count_total|n/a|51 [51, 51]|60 [60, 60]|1.176|

## count(*) vs count(l_orderkey)

| config | count(*) bytes/object | count(l_orderkey) bytes/object | count(*) / count(l_orderkey) bytes | count(*) wall ms | count(l_orderkey) wall ms | count(*) / count(l_orderkey) wall |
|---|---:|---:|---:|---:|---:|---:|
|cache OFF (production default)|1.00|1.00|1.000|4260.00|4302.82|0.990|
|cache ON + prewarm ON|1.00|1.00|1.000|2484.17|2504.14|0.992|
|cache ON + prewarm OFF|1.00|1.00|1.000|4251.39|4201.55|1.012|

## Observations

- count(*): cache-OFF bytes/object=1.00, cache-OFF wall/cache-ON+prewarm-ON wall=1.715, cache-ON prewarm OFF/ON bytes=1.000, cache-ON prewarm OFF/ON wall=1.711, prewarm-ON cache hit rate=22.86%, dominant miss ON=range_miss=81, dominant miss OFF=range_miss=105, FSMR borrow count nonzero on cache-OFF and cache-ON+prewarm-OFF=yes, FSMR borrow count zero on cache-ON+prewarm-ON=yes.
- count(l_orderkey): cache-OFF bytes/object=1.00, cache-OFF wall/cache-ON+prewarm-ON wall=1.718, cache-ON prewarm OFF/ON bytes=1.000, cache-ON prewarm OFF/ON wall=1.678, prewarm-ON cache hit rate=22.86%, dominant miss ON=range_miss=81, dominant miss OFF=range_miss=105, FSMR borrow count nonzero on cache-OFF and cache-ON+prewarm-OFF=yes, FSMR borrow count zero on cache-ON+prewarm-ON=yes.
- q1 — TPC-H Q1 shape (narrowed-date filter): cache-OFF bytes/object=1.00, cache-OFF wall/cache-ON+prewarm-ON wall=1.637, cache-ON prewarm OFF/ON bytes=1.000, cache-ON prewarm OFF/ON wall=1.631, prewarm-ON cache hit rate=22.86%, dominant miss ON=range_miss=81, dominant miss OFF=range_miss=105, FSMR borrow count nonzero on cache-OFF and cache-ON+prewarm-OFF=yes, FSMR borrow count zero on cache-ON+prewarm-ON=yes.
- join — lineitem×orders: cache-OFF bytes/object=1.00, cache-OFF wall/cache-ON+prewarm-ON wall=1.740, cache-ON prewarm OFF/ON bytes=1.000, cache-ON prewarm OFF/ON wall=1.705, prewarm-ON cache hit rate=14.29%, dominant miss ON=range_miss=153, dominant miss OFF=range_miss=180, FSMR borrow count nonzero on cache-OFF and cache-ON+prewarm-OFF=yes, FSMR borrow count zero on cache-ON+prewarm-ON=yes.

## Phase 3 decision trigger

- If cache-OFF count(*) bytes/object > 1.5 while count(l_orderkey) stays <= 1.2: file cache-OFF empty-projection redundancy as a separate backlog item.
- If cache-OFF count(*) bytes/object is already ~= 1.0: treat §26's earlier 1.87x as stale or non-reproduced on current code.
- If cache-ON prewarm ON saves >= 10% wall-clock and hit rate > 50%: Phase 3 can consider cache hit alignment; otherwise prefer the measured default/cache policy follow-up.
- Trigger evaluation from this run: cache-OFF count(*) redundancy did not reproduce on current code; §26's 1.87x measurement looks stale or configuration-specific.
- Cache-ON prewarm trigger evaluation: mixed signal: prewarm ON saves wall-clock, but cache hit rate stays below 50%; Phase 3 should review before choosing A vs C.
