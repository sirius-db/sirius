# Chunk-skipping opportunity study

Offline measurements backing `CHUNK_SKIPPING_PLAN.md` §3–§4. No Sirius build required —
these use the `pixi` environment's DuckDB + numpy only.

```bash
export SCRATCH=/tmp/sirius-chunk-skipping && mkdir -p $SCRATCH
cd <repo root>
pixi run python tools/chunk-skipping-study/explain.py     # §3.1 per-query scan filters @ SF1000
pixi run python tools/chunk-skipping-study/dumpstats.py   # parquet footer stats, all 8 tables
pixi run python tools/chunk-skipping-study/prune.py       # §3.1/§3.2 prune rate + addressable volume
pixi run python tools/chunk-skipping-study/mkcols.py      # §3.3 materialize SF10 lineitem columns
pixi run python tools/chunk-skipping-study/gran.py        #      -> numpy arrays
pixi run python tools/chunk-skipping-study/sweep.py       #      granularity x sort-key sweep
pixi run python tools/chunk-skipping-study/proj.py        #      project onto SF1000 byte volumes
pixi run python tools/chunk-skipping-study/synth.py       # §4.1 clustering-window vs chunk-size
pixi run python tools/chunk-skipping-study/idxsize.py     # §4.2 index size
pixi run python tools/chunk-skipping-study/idx2.py        # §4.2 index size vs compressed footprint
```

Requires `/datasets/tpch_sf1000` and `/datasets/tpch_sf10`.

**Methodology note:** `explain.py` must build its views over **SF1000**, not SF1. DuckDB derives
join filters from table statistics, so SF1 views emit `c_custkey<=149999`, which evaluated against
SF1000 footers produces a spurious 99% prune on `customer`.
