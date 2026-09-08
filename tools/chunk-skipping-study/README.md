# Chunk-skipping opportunity study

Offline measurements backing `CHUNK_SKIPPING_PLAN.md`. No Sirius build required —
these use the `pixi` environment's DuckDB + numpy only.

```bash
export SCRATCH=/tmp/sirius-chunk-skipping && mkdir -p $SCRATCH
cd <repo root>
pixi run python tools/chunk-skipping-study/explain.py     # per-query scan filters @ SF1000
pixi run python tools/chunk-skipping-study/dumpstats.py   # parquet footer stats, all 8 tables
pixi run python tools/chunk-skipping-study/prune.py       # prune rate + addressable volume
pixi run python tools/chunk-skipping-study/mkcols.py      # materialize SF10 lineitem columns
pixi run python tools/chunk-skipping-study/gran.py        #      -> numpy arrays
pixi run python tools/chunk-skipping-study/sweep.py       #      granularity x sort-key sweep
pixi run python tools/chunk-skipping-study/proj.py        #      project onto SF1000 byte volumes
pixi run python tools/chunk-skipping-study/coarse.py      # pin-chunk-scale granularity
pixi run python tools/chunk-skipping-study/synth.py       # clustering-window vs chunk-size
pixi run python tools/chunk-skipping-study/idxsize.py     # index size
pixi run python tools/chunk-skipping-study/idx2.py        # index size vs compressed footprint
```

Requires `/datasets/tpch_sf1000` and `/datasets/tpch_sf10`.

**Methodology note:** `explain.py` must build its views over **SF1000**, not SF1. DuckDB derives
join filters from table statistics, so SF1 views emit `c_custkey<=149999`, which evaluated against
SF1000 footers produces a spurious 99% prune on `customer`.

## Clustering (2026-09-07)

```bash
pixi run python tools/chunk-skipping-study/mksorted.py           # build /datasets/tpch_sf100_sorted
pixi run python tools/chunk-skipping-study/verify_clustering.py  # confirm row-group spans collapsed
DATASET=/datasets/tpch_sf100_sorted pixi run python tools/chunk-skipping-study/explain.py
DATASET=/datasets/tpch_sf100_sorted pixi run python tools/chunk-skipping-study/dumpstats.py
DATASET=/datasets/tpch_sf100_sorted pixi run python tools/chunk-skipping-study/prune.py
pixi run python tools/chunk-skipping-study/cluster_modes.py      # clustering-strategy tradeoff
pixi run python tools/chunk-skipping-study/cpusort.py            # CPU sort throughput
# GPU sort throughput — see the build line in gpusort/sortbench.cu
/home/nvidia/joost/bench-lock.sh ./sortbench 189000000 1

# explorer: does clustering change the compression plan?
pixi run python tools/chunk-skipping-study/explore_cols.py       # extract single-column parquet
/home/nvidia/joost/bench-lock.sh <build>/simpatico_codegen/simpatico explore --input <col>.parquet --col 0
```

**Trap:** DuckDB's `FILE_SIZE_BYTES` rotation writes in parallel and does NOT preserve a global
`ORDER BY` across files. `mksorted.py` writes one file then splits by `LIMIT`/`OFFSET`; verify with
`verify_clustering.py` before trusting any pruning number.
