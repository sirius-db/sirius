# GPU-native DuckDB scan vs. parquet scan — SF30 performance findings

**Date:** 2026-06-04
**Machine:** NVIDIA GB10 Grace Blackwell (aarch64), 20-core Grace CPU, ~119 GiB **unified** LPDDR5X
(CPU+GPU share one physical pool). 1 GPU.
**Build:** `dev` branch, CUDA 13, driver 580. Config: `test/cpp/integration/integration-gb10.yaml`
(GPU 48 GB absolute cap, host 24 GB, 8 pipeline / 4 duckdb_scan threads, 512 MB batches).
**Data:** TPC-H SF30 (lineitem 179,998,372 rows; orders 45,000,000), generated in both parquet and
DuckDB-native formats from the same spec (validated identical, all 22 queries byte/tolerance-exact).

## Headline (warm, all 22 queries, seconds)

| Mode | Warm total | vs DuckDB CPU |
|---|---|---|
| DuckDB CPU (native tables) | 4.74 | 1.0× baseline |
| Parquet GPU scan | 12.65 | 0.37× (2.7× slower) |
| **GPU-native DuckDB scan** | **16.72** | **0.28× (3.5× slower)** |

At SF30 DuckDB CPU wins outright (small data, fits in cache on a 20-core Grace; GPU fixed per-query
overhead dominates). Among the GPU paths, **parquet beats native by ~32% overall**. GPU advantage is
expected to appear at SF100+. This doc explains *why the GPU-native scan trails parquet*, which is the
actionable gap.

## Where the gap is (warm seconds)

| Query | native | parquet | ratio | shape |
|---|---|---|---|---|
| Q1  | 2.79 | 1.75 | 1.6× | scan + GROUP BY, ~no filter, **no join** |
| Q6  | 0.28 | 0.23 | 1.2× | scan + SUM, selective filter, **no join** |
| Q10 | 0.87 | 0.70 | 1.2× | 3-way join, selective filters |
| Q19 | 1.60 | 0.54 | 3.0× | join + complex disjunctive filter |

## Root cause: the native scan has no predicate pushdown into decode

The GPU-native scan operator
([`src/op/scan/sirius_gpu_duckdb_native_scan_operator.cpp:80-137`](../../src/op/scan/sirius_gpu_duckdb_native_scan_operator.cpp))
**decodes 100% of every row group to the GPU first, then applies `table_filters` as a post-decode GPU
`select`** (`exec.select(table->view())`). It has no in-decode predicate evaluation and no
statistics-based row-group skipping.

The parquet scan instead **pushes the filter into the cuDF parquet reader**
(`Translated filter expression for parquet reader filter pushdown`) — the predicate is applied *during*
decode and row groups can be skipped via min/max statistics.

### Two compounding factors

**1. Row volume (dominant on join queries).** Because the native scan never reduces rows at the scan,
~2× the rows flow through the downstream `PARTITION → CONCAT → HASH_JOIN` stages. Measured rows pushed
through the pipeline (1 iteration):

| Query | native decodes | parquet reads | ratio |
|---|---|---|---|
| Q19 | 192.0M | 102.0M | 1.9× |
| Q10 | 229.5M | 101.3M | 2.3× |

Hash-join build side confirms it (Q19): native **228 MB** vs parquet **183 MB**
(`sirius_physical_partition.cpp:339`). For Q19 there is an additional plan-shape effect: DuckDB's
optimizer fuses the selective lineitem filter *right after the scan* in the read_parquet plan
(`PARQUET_SCAN(19)→FILTER(11)→PROJECTION(12)`), but in the seq_scan plan the lineitem scan is bare and
the predicate is applied *after* the hash join (`FILTER id=14`) — so the join probes the full 180M-row
lineitem. That is why Q19's gap (3.0×) is the largest.

**2. Raw decode throughput (~1.6×).** Q1 isolates this: a GROUP BY over lineitem with an almost
non-selective filter and **no join**, so both paths process ~the same 180M rows — yet native is 2.79s
vs parquet 1.75s (1.6×). Same rows, no downstream blow-up → the DuckDB-native segment→cuDF decode is
slower per row than cuDF's parquet reader (uncompressed native segments; no dictionary/RLE fast paths).
Q6 is the control proving factor 1 needs a join to bite: selective filter, *no join* → only 1.2×, because
the un-pushed rows get summed away immediately.

## Recommended fixes (to close the gap), highest leverage first

1. **Predicate / row-group pushdown in the native scan.** Use DuckDB's per-segment min/max zonemaps to
   skip row groups, and apply `table_filters` *during* decode (or prune before decode) instead of
   decoding everything and GPU-filtering after. Attacks factor 1 (the 2× row volume). Natural homes:
   the row-group walk in [`src/op/scan/duckdb_native_metadata.cpp`](../../src/op/scan/duckdb_native_metadata.cpp)
   (add stats-based skipping) and `decode_duckdb_native_split` /
   [`src/op/scan/duckdb_native_decoder.cpp`](../../src/op/scan/duckdb_native_decoder.cpp).
2. **Surface single-table conjuncts to the seq_scan plan** so the optimizer places the lineitem filter
   before the join (the Q19 plan difference), matching the read_parquet plan shape.
3. **Improve segment→cuDF decode throughput** to address the residual ~1.6× (factor 2).

## How to reproduce

```bash
# Data (note SIRIUS_DISABLE=1 — else the auto-loaded extension reserves the whole unified pool and OOMs)
SIRIUS_DISABLE=1 pixi run bash test/tpch_performance/generate_tpch_data.sh 30 --format duckdb
pixi run bash test/tpch_performance/generate_tpch_data.sh 30 --format parquet

export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration-gb10.yaml
# native (now the default duckdb path) vs parquet, both validated against DuckDB CPU
./test/tpch_performance/benchmark_and_validate.sh --data-source duckdb  --duckdb-file ./test_datasets/tpch_sf30.duckdb 30
./test/tpch_performance/benchmark_and_validate.sh --data-source parquet --parquet-dir  ./test_datasets/tpch_parquet_sf30 30

# To see scan behavior (rows decoded, pipeline shape, filter pushdown):
SIRIUS_LOG_LEVEL=debug OUTPUT_DIR=/tmp/dbg ./test/tpch_performance/run_tpch_duckdb.sh \
    --duckdb-file ./test_datasets/tpch_sf30.duckdb sirius 30 19 10
grep "decoded split" /tmp/dbg/sirius_*.log     # native: full-row-group decodes
```
